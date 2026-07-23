const { default: makeWASocket, DisconnectReason, fetchLatestWaWebVersion } = require('@whiskeysockets/baileys');
const { MongoClient } = require('mongodb');
const { useMongoDBAuthState } = require('./mongoAuthState');
const qrcodeTerminal = require('qrcode-terminal');
const qrcodeImage = require('qrcode');
const axios = require('axios');
const pino = require('pino');
const http = require('http');

const BACKEND_URL = process.env.BACKEND_URL || "https://neura-ai-df6q.onrender.com/api/chat";
const PORT = process.env.PORT || 10000;
const MONGO_URI = process.env.MONGO_URI;

if (!MONGO_URI) {
    console.error("FATAL ERROR: MONGO_URI environment variable is missing!");
    process.exit(1);
}

let currentQR = "";
let isConnected = false;

// HTTP Web Server serving a visual QR Code webpage & Health Check
const server = http.createServer(async (req, res) => {
    if (isConnected) {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(`
            <html>
                <head><title>NEURA AI WhatsApp Gateway</title></head>
                <body style="font-family: sans-serif; text-align: center; padding-top: 50px; background-color: #0f172a; color: white;">
                    <h1 style="color: #22c55e;">🎉 NEURA AI WhatsApp Gateway is Connected & Active!</h1>
                    <p>Students can now message NEURA AI directly on WhatsApp.</p>
                </body>
            </html>
        `);
        return;
    }

    if (currentQR) {
        try {
            const qrDataUrl = await qrcodeImage.toDataURL(currentQR);
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(`
                <html>
                    <head>
                        <title>Scan QR Code - NEURA AI</title>
                        <meta http-equiv="refresh" content="10">
                    </head>
                    <body style="font-family: sans-serif; text-align: center; padding-top: 40px; background-color: #0f172a; color: white;">
                        <h1>📱 Pair NEURA AI WhatsApp Gateway</h1>
                        <p>1. Open WhatsApp on your phone ➔ <b>Linked Devices</b> ➔ <b>Link a Device</b></p>
                        <p>2. Scan the QR code below:</p>
                        <div style="background: white; display: inline-block; padding: 20px; border-radius: 12px; margin-top: 10px;">
                            <img src="${qrDataUrl}" width="300" height="300" />
                        </div>
                        <p style="color: #94a3b8; font-size: 14px; margin-top: 15px;">Page auto-refreshes every 10 seconds</p>
                    </body>
                </html>
            `);
            return;
        } catch (e) {
            console.error("Error generating QR web image:", e);
        }
    }

    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(`
        <html>
            <body style="font-family: sans-serif; text-align: center; padding-top: 50px; background-color: #0f172a; color: white;">
                <h2>⏳ Generating WhatsApp QR Code... Please refresh in 5 seconds.</h2>
            </body>
        </html>
    `);
});

server.listen(PORT, () => {
    console.log(`HTTP Web server running on port ${PORT}`);
});

async function connectToWhatsApp() {
    console.log("Starting NEURA AI WhatsApp Gateway...");
    
    // Connect to MongoDB and set up the auth collection
    const mongoClient = new MongoClient(MONGO_URI);
    await mongoClient.connect();
    const collection = mongoClient.db('neura_db').collection('whatsapp_auth');
    const { state, saveCreds } = await useMongoDBAuthState(collection);

    // Dynamically fetch the latest WhatsApp Web client version to fix 405 Connection Errors
    let version = [2, 3000, 1015901307];
    try {
        const fetchedVersion = await fetchLatestWaWebVersion({});
        version = fetchedVersion.version;
        console.log(`Fetched latest WhatsApp Web version: v${version.join('.')}`);
    } catch (err) {
        console.log("Using fallback WhatsApp Web version...");
    }

    const socket = makeWASocket({
        version,
        logger: pino({ level: 'silent' }),
        auth: state,
        browser: ['NEURA AI', 'Chrome', '1.0.0'],
        syncFullHistory: false,
        markOnlineOnConnect: true,
        keepAliveIntervalMs: 30000,
        connectTimeoutMs: 60000,
        defaultQueryTimeoutMs: 60000
    });

    socket.ev.on('creds.update', saveCreds);

    socket.ev.on('connection.update', (update) => {
        const { connection, lastDisconnect, qr } = update;
        
        if (qr) {
            currentQR = qr;
            console.log("\n==================================================");
            console.log("NEW QR CODE GENERATED! Open https://neura-whatsapp-gateway.onrender.com to scan!");
            console.log("==================================================\n");
            qrcodeTerminal.generate(qr, { small: true });
        }

        if (connection === 'close') {
            isConnected = false;
            const statusCode = lastDisconnect?.error?.output?.statusCode;
            const shouldReconnect = statusCode !== DisconnectReason.loggedOut;
            console.log(`Connection closed (Code ${statusCode}). Reconnecting: ${shouldReconnect}...`);
            
            if (statusCode === 401) {
                console.log("Credentials invalid or logged out. Clearing MongoDB auth state to generate new QR...");
                collection.drop().catch(() => {}).finally(() => {
                    setTimeout(connectToWhatsApp, 2000);
                });
            } else if (shouldReconnect) {
                setTimeout(connectToWhatsApp, 3000);
            }
        } else if (connection === 'open') {
            isConnected = true;
            currentQR = "";
            console.log('\n🎉 SUCCESS: NEURA AI WhatsApp Gateway is Connected & Live on WhatsApp!\n');
        }
    });

    socket.ev.on('messages.upsert', async (m) => {
        if (m.type !== 'notify') return;

        for (const msg of m.messages) {
            if (!msg.message || msg.key.fromMe) continue;

            const replyToJid = msg.key.remoteJid;
            
            // Ignore status updates
            if (replyToJid === 'status@broadcast') continue;

            // Use senderPn for database user memory if available, otherwise fall back to replyToJid
            const userId = msg.key.senderPn || replyToJid;

            const messageContent = msg.message.conversation || 
                                   msg.message.extendedTextMessage?.text || 
                                   msg.message.imageMessage?.caption || "";

            if (!messageContent.trim()) continue;

            console.log(`📩 Received message from ${replyToJid} (${userId}): "${messageContent}"`);
            console.log(`[DEBUG] Full Message Key:`, JSON.stringify(msg.key));

            try {
                // Send read receipt (blue ticks)
                await socket.readMessages([msg.key]);
                await socket.sendPresenceUpdate('composing', replyToJid);

                const response = await axios.post(BACKEND_URL, {
                    user_id: userId,
                    message: messageContent
                });

                const aiReply = response.data.response;
                await socket.sendPresenceUpdate('paused', replyToJid);
                
                // Construct a normalized quoted message matching the chat JID
                const quotedMsg = {
                    key: {
                        ...msg.key,
                        remoteJid: replyToJid
                    },
                    message: msg.message
                };

                await socket.sendMessage(replyToJid, { text: aiReply }, { quoted: quotedMsg });
                console.log(`✅ Sent reply to ${replyToJid}`);

            } catch (error) {
                console.error("Error communicating with Backend API:", error.message);
                const quotedMsg = {
                    key: {
                        ...msg.key,
                        remoteJid: replyToJid
                    },
                    message: msg.message
                };
                await socket.sendPresenceUpdate('paused', replyToJid);
                await socket.sendMessage(replyToJid, {
                    text: "Sorry, NEURA AI experienced a temporary connection delay. Please try asking your medical question again!"
                }, { quoted: quotedMsg });
            }
        }
    });
}

connectToWhatsApp();
