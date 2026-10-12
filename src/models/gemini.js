import { GoogleGenAI } from '@google/genai';
import { strictFormat } from '../utils/text.js';
import { getKey } from '../utils/keys.js';

export class Gemini {
    static prefix = 'google';
    constructor(model_name, url, params) {
        // Bereinige den Modellnamen
        this.model_name = model_name ? model_name.replace(/^(models\/|google\/)/, '') : "gemini-2.5-flash";
        this.params = params;
        this.safetySettings = [
            { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE" },
            { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE" },
            { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE" },
            { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE" }
        ];
        
        this.apiKey = getKey('GEMINI_API_KEY');

        // Wir erstellen die Instanz nur, damit das Mindcraft-Framework beim Instanziieren nicht meckert
        this.genAI = new GoogleGenAI({ apiKey: this.apiKey });
    }

    async sendRequest(turns, systemMessage) {
        console.log('Awaiting Google API response via Direct Fetch...');

        turns = strictFormat(turns);
        let contents = [];
        for (let turn of turns) {
            contents.push({
                role: turn.role === 'assistant' ? 'model' : 'user',
                parts: [{ text: turn.content }]
            });
        }

        const targetUrl = `https://googleapis.com{this.model_name}:generateContent?key=${this.apiKey}`;
        
        const response = await fetch(targetUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                contents: contents,
                safetySettings: this.safetySettings,
                generationConfig: {
                    ...(this.params || {})
                },
                systemInstruction: systemMessage ? { parts: [{ text: systemMessage }] } : undefined
            })
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`Google API Direct Fetch Error: ${response.status} - ${errText}`);
        }

        const json = await response.json();
        const responseText = json.candidates?.[0]?.content?.parts?.[0]?.text || '';
        
        console.log('Received.');
        return responseText;
    }

    async sendVisionRequest(turns, systemMessage, imageBuffer) {
        console.log('Awaiting Google API vision response via Direct Fetch...');
        
        const imagePart = {
            inlineData: {
                data: imageBuffer.toString('base64'),
                mimeType: 'image/jpeg'
            }
        };
       
        turns = strictFormat(turns);
        let contents = [];
        for (let turn of turns) {
            contents.push({
                role: turn.role === 'assistant' ? 'model' : 'user',
                parts: [{ text: turn.content }]
            });
        }
        contents.push({
            role: 'user',
            parts: [{ text: 'SYSTEM: Vision response' }, imagePart]
        });

        let res = null;
        try {
            const targetUrl = `https://googleapis.com{this.model_name}:generateContent?key=${this.apiKey}`;
            
            const response = await fetch(targetUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    contents: contents,
                    safetySettings: this.safetySettings,
                    generationConfig: {
                        ...(this.params || {})
                    },
                    systemInstruction: systemMessage ? { parts: [{ text: systemMessage }] } : undefined
                })
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`Google Vision Direct Fetch Error: ${response.status} - ${errText}`);
            }

            const json = await response.json();
            res = json.candidates?.[0]?.content?.parts?.[0]?.text || '';
            console.log('Received.');
        } catch (err) {
            console.error(err);
            if (err.message && err.message.includes("modality is not enabled")) {
                res = "Vision is only supported by certain models.";
            } else {
                res = "An unexpected error occurred, please try again.";
            }
        }
        return res;
    }

    async embed(text) {
        const targetUrl = `https://googleapis.com{this.model_name || "text-embedding-004"}:embedContent?key=${this.apiKey}`;
        
        const response = await fetch(targetUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                content: { parts: [{ text: text }] }
            })
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`Google Embed Direct Fetch Error: ${response.status} - ${errText}`);
        }

        const json = await response.json();
        return json.embedding?.values || [];
    }
}

const sendAudioRequest = async (text, model, voice, url) => {
    const apiKey = getKey('GEMINI_API_KEY');
    const cleanedModel = model ? model.replace(/^(models\/|google\/)/, '') : "gemini-2.5-flash";
    
    const targetUrl = `https://googleapis.com{cleanedModel}:generateContent?key=${apiKey}`;
    
    const response = await fetch(targetUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            contents: [{ parts: [{ text: text }] }],
            generationConfig: {
                responseModalities: ['AUDIO'],
                speechConfig: {
                    voiceConfig: {
                        prebuiltVoiceConfig: { voiceName: voice }
                    }
                }
            }
        })
    });

    if (!response.ok) {
        console.warn('Gemini TTS Fetch failed:', response.status);
        return null;
    }

    const json = await response.json();
    const pcmBase64 = json.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    
    if (!pcmBase64) {
        console.warn('Gemini TTS: no audio data returned');
        return null;
    }

    const pcmBuffer = Buffer.from(pcmBase64, 'base64');
    const wavHeader = createWavHeader(pcmBuffer.length, 24000, 1, 16);
    const wavBuffer = Buffer.concat([wavHeader, pcmBuffer]);

    return wavBuffer.toString('base64');
};

function createWavHeader(dataLength, sampleRate, channels, bitsPerSample) {
    const header = Buffer.alloc(44);
    const byteRate = sampleRate * channels * bitsPerSample / 8;
    const blockAlign = channels * bitsPerSample / 8;

    header.write('RIFF', 0);
    header.writeUInt32LE(36 + dataLength, 4);
    header.write('WAVE', 8);
    header.write('fmt ', 12);
    header.writeUInt32LE(16, 16);
    header.writeUInt16LE(1, 20);
    header.writeUInt16LE(channels, 22);
    header.writeUInt32LE(sampleRate, 24);
    header.writeUInt32LE(byteRate, 28);
    header.writeUInt16LE(blockAlign, 32);
    header.writeUInt16LE(bitsPerSample, 34);
    header.write('data', 36);
    header.writeUInt32LE(dataLength, 40);
    return header;
}

export const TTSConfig = {
    sendAudioRequest: sendAudioRequest,
};
