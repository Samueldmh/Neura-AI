const { default: makeWASocket, useMultiFileAuthState, DisconnectReason } = require('@whiskeysockets/baileys');
const qrcode = require('qrcode-terminal');
const axios = require('axios');
const pino = require('pino');
const http = require('http');

const BACKEND_URL = process.env.BACKEND_URL || "https://neura-ai-df6q.onrender.com/api/chat";
const PORT = process.env.PORT || 10000;

// Simple health check server for cloud hosting (Render requires binding to $PORT)
http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: "online", service: "NEURA AI WhatsApp Gateway" }));
}).listen(PORT, () => {
    console.log(`HTTP Health check server running on port ${PORT}`);
});

async function connectToWhatsApp() {
    console.log("Starting NEURA AI WhatsApp Gateway...");
    
    // Save authentication state in 'auth_info_baileys' directory
    const { state, saveCreds } = await useMultiFileAuthState('auth_info_baileys');

    const socket = makeWASocket({
        logger: pino({ level: 'silent' }),
        printQRInTerminal: true,
        auth: state
    });

    socket.ev.on('creds.update', saveCreds);

    socket.ev.on('connection.update', (update) => {
        const { connection, lastDisconnect, qr } = update;
        
        if (qr) {
            console.log("\n==================================================");
            console.log("SCAN THIS QR CODE WITH YOUR WHATSAPP APP TO PAIR:");
            console.log("==================================================\n");
            qrcode.generate(qr, { small: true });
        }

        if (connection === 'close') {
            const shouldReconnect = (lastDisconnect?.error?.output?.statusCode !== DisconnectReason.loggedOut);
            console.log('Connection closed due to error:', lastDisconnect?.error, ', reconnecting:', shouldReconnect);
            if (shouldReconnect) {
                setTimeout(connectToWhatsApp, 5000);
            }
        } else if (connection === 'open') {
            console.log('\n🎉 SUCCESS: NEURA AI WhatsApp Gateway is Connected & Live on WhatsApp!\n');
        }
    });

    // Listen for incoming WhatsApp messages
    socket.ev.on('messages.upsert', async (m) => {
        if (m.type !== 'notify') return;

        for (const msg of m.messages) {
            if (!msg.message || msg.key.fromMe) continue; // Ignore messages sent by bot itself

            const senderJid = msg.key.remoteJid;
            // Extract text message content
            const messageContent = msg.message.conversation || 
                                   msg.message.extendedTextMessage?.text || 
                                   msg.message.imageMessage?.caption || "";

            if (!messageContent.trim()) continue;

            console.log(`📩 Received message from ${senderJid}: "${messageContent}"`);

            try {
                // Indicate typing status in WhatsApp
                await socket.sendPresenceUpdate('composing', senderJid);

                // Send message to FastAPI Render Backend
                const response = await axios.post(BACKEND_URL, {
                    user_id: senderJid,
                    message: messageContent
                });

                const aiReply = response.data.response;

                // Stop typing status
                await socket.sendPresenceUpdate('paused', senderJid);

                // Reply to student on WhatsApp
                await socket.sendMessage(senderJid, { text: aiReply }, { quoted: msg });
                console.log(`✅ Sent reply to ${senderJid}`);

            } catch (error) {
                console.error("Error communicating with Backend API:", error.message);
                await socket.sendPresenceUpdate('paused', senderJid);
                await socket.sendMessage(senderJid, {
                    text: "Sorry, NEURA AI experienced a temporary connection delay. Please try asking your medical question again!"
                }, { quoted: msg });
            }
        }
    });
}

// Start WhatsApp Gateway
connectToWhatsApp();
