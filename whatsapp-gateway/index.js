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

// =======================================================================
// FIX 1: In-memory message store for Signal Protocol message retries.
//
// WHY MESSAGES WERE NEVER RECEIVED:
// When the bot sends to a @lid JID, the recipient's device often can't
// decrypt the message (new/stale Signal session). The device asks WhatsApp
// to request a re-send. Baileys calls getMessage() to retrieve the
// original message, re-encrypts it with fresh keys, and sends again.
// Without this store, getMessage() returned undefined, the re-send failed,
// and the message was silently lost. The user saw "typing..." but no reply.
// =======================================================================
const msgRetryStore = new Map();

// Evict old messages every 5 minutes to prevent memory leaks
setInterval(() => {
    const fiveMinutesAgo = Date.now() - 5 * 60 * 1000;
    for (const [id, entry] of msgRetryStore) {
        if (entry.timestamp < fiveMinutesAgo) {
            msgRetryStore.delete(id);
        }
    }
}, 5 * 60 * 1000);

// =======================================================================
// FIX 2: Message retry counter cache.
//
// Baileys tracks how many times each message has been retried.
// Without this cache, Baileys can't properly manage the retry flow,
// leading to infinite retries or giving up too early.
// =======================================================================
class RetryCounterCache {
    constructor() {
        this.cache = new Map();
    }
    get(key) {
        const entry = this.cache.get(key);
        if (!entry) return undefined;
        // Auto-expire after 10 minutes
        if (Date.now() - entry.timestamp > 10 * 60 * 1000) {
            this.cache.delete(key);
            return undefined;
        }
        return entry.value;
    }
    set(key, value) {
        this.cache.set(key, { value, timestamp: Date.now() });
    }
    del(key) {
        this.cache.delete(key);
    }
    flushAll() {
        this.cache.clear();
    }
}

const msgRetryCounterCache = new RetryCounterCache();

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

// Single persistent MongoDB client (never recreated on reconnect)
const mongoClient = new MongoClient(MONGO_URI, {
    maxPoolSize: 5,
    serverSelectionTimeoutMS: 10000,
});

let collection = null;

async function initMongo() {
    await mongoClient.connect();
    collection = mongoClient.db('neura_db').collection('whatsapp_auth_v3');
    console.log("✅ MongoDB connected successfully.");
}

async function connectToWhatsApp() {
    console.log("Starting NEURA AI WhatsApp Gateway...");

    const { state, saveCreds } = await useMongoDBAuthState(collection);

    // Dynamically fetch the latest WhatsApp Web client version
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
        logger: pino({ level: 'warn' }),  // Changed from 'silent' to 'warn' to catch protocol errors
        auth: state,
        browser: ['Ubuntu', 'Chrome', '24.0.10'],
        syncFullHistory: false,
        markOnlineOnConnect: true,
        keepAliveIntervalMs: 15000,
        connectTimeoutMs: 60000,
        defaultQueryTimeoutMs: 0,
        generateHighQualityLinkPreview: false,
        retryRequestDelayMs: 200,  // Quick retry responses for prekey renegotiation

        // FIX 2: Retry counter cache for tracking message retry attempts
        msgRetryCounterCache,

        // FIX 1: getMessage callback — THE critical fix for message delivery.
        // When a recipient's device can't decrypt a message, it requests a re-send.
        // Baileys calls this to get the original message for re-encryption.
        // Without this, messages are silently lost even though logs say "delivered".
        getMessage: async (key) => {
            const entry = msgRetryStore.get(key.id);
            if (entry) {
                console.log(`[RETRY] ✅ getMessage found message ${key.id} in store — re-encrypting for delivery`);
                return entry.message;
            }
            console.log(`[RETRY] ❌ getMessage could NOT find message ${key.id} — delivery will fail`);
            return undefined;
        }
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

    // =======================================================================
    // FIX 3: Track message delivery status updates.
    // This tells us whether WhatsApp actually delivered the message to the
    // user's device, vs just accepting the packet at the server level.
    // =======================================================================
    socket.ev.on('messages.update', (updates) => {
        for (const update of updates) {
            const status = update.update?.status;
            if (status !== undefined) {
                // Status codes: 1=pending, 2=server_ack, 3=delivery_ack, 4=read, 5=played
                const statusNames = { 1: 'PENDING', 2: 'SERVER_ACK', 3: 'DELIVERED_TO_DEVICE', 4: 'READ', 5: 'PLAYED' };
                const statusName = statusNames[status] || `UNKNOWN(${status})`;
                console.log(`[STATUS] Message ${update.key?.id} to ${update.key?.remoteJid}: ${statusName}`);
            }
        }
    });

    // Store ALL incoming messages in the retry store (both sent and received)
    socket.ev.on('messages.upsert', async (m) => {
        // Store every message for potential retry resolution
        for (const msg of m.messages) {
            if (msg.key?.id && msg.message) {
                msgRetryStore.set(msg.key.id, {
                    message: msg.message,
                    timestamp: Date.now()
                });
            }
        }

        if (m.type !== 'notify') return;

        for (const msg of m.messages) {
            if (!msg.message || msg.key.fromMe) continue;

            const replyToJid = msg.key.remoteJid;

            // Ignore status updates
            if (replyToJid === 'status@broadcast') continue;

            const userId = msg.key.participant || msg.key.remoteJid;

            const messageContent = msg.message.conversation ||
                                   msg.message.extendedTextMessage?.text ||
                                   msg.message.imageMessage?.caption || "";

            if (!messageContent.trim()) continue;

            console.log(`📩 Received from [${replyToJid}] user [${userId}]: "${messageContent}"`);
            console.log(`[DEBUG] Full msg.key: ${JSON.stringify(msg.key)}`);

            try {
                // Send read receipt — wrapped in try-catch so failure doesn't block reply
                try {
                    await socket.readMessages([msg.key]);
                } catch (readErr) {
                    console.log(`[WARN] readMessages failed (non-fatal): ${readErr.message}`);
                }

                await socket.sendPresenceUpdate('composing', replyToJid);

                // Axios with explicit timeout to prevent socket death during cold starts
                const response = await axios.post(BACKEND_URL, {
                    user_id: userId,
                    message: messageContent
                }, {
                    timeout: 55000
                });

                const aiReply = response.data.response;

                await socket.sendPresenceUpdate('paused', replyToJid);

                // Send WITHOUT quoted to avoid LID device-list resolution issues
                const sentMsg = await socket.sendMessage(replyToJid, { text: aiReply });

                // Store the SENT message in retry store so retries can re-encrypt it
                if (sentMsg?.key?.id && sentMsg?.message) {
                    msgRetryStore.set(sentMsg.key.id, {
                        message: sentMsg.message,
                        timestamp: Date.now()
                    });
                }

                console.log(`✅ Reply sent to ${replyToJid} (msgId: ${sentMsg?.key?.id}, status: ${sentMsg?.status})`);
                console.log(`[DEBUG] sentMsg.key: ${JSON.stringify(sentMsg?.key)}`);

            } catch (error) {
                console.error("Error processing message:", error.message);
                try {
                    await socket.sendPresenceUpdate('paused', replyToJid);
                    const errMsg = await socket.sendMessage(replyToJid, {
                        text: "Sorry, NEURA AI experienced a temporary connection delay. Please try asking your medical question again!"
                    });
                    if (errMsg?.key?.id && errMsg?.message) {
                        msgRetryStore.set(errMsg.key.id, {
                            message: errMsg.message,
                            timestamp: Date.now()
                        });
                    }
                } catch (sendErr) {
                    console.error("Failed to send error message:", sendErr.message);
                }
            }
        }
    });
}

// Start: Connect MongoDB ONCE, then start WhatsApp
initMongo()
    .then(() => connectToWhatsApp())
    .catch(err => {
        console.error("FATAL: Could not connect to MongoDB:", err);
        process.exit(1);
    });
