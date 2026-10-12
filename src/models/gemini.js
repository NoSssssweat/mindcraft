import { GoogleGenAI } from '@google/genai';
import { strictFormat } from '../utils/text.js';
import { getKey } from '../utils/keys.js';

export class Gemini {
    static prefix = 'google';
    constructor(model_name, url, params) {
        // Bereinige den Modellnamen: Entferne "models/" oder "google/", falls das Framework dies mitsendet
        this.model_name = model_name ? model_name.replace(/^(models\/|google\/)/, '') : "gemini-2.5-flash";
        this.params = params;
        this.safetySettings = [
            { "category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE" },
            { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE" },
            { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE" },
            { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE" },
            { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE" },
        ];
        
        const apiKey = getKey('GEMINI_API_KEY');

        // Bereinige kollidierende Umgebungsvariablen im Docker-Container
        delete process.env.GOOGLE_APPLICATION_CREDENTIALS;
        delete process.env.GOOGLE_GCLOUD_ADC_JSON;
        delete process.env.GOOGLE_CLOUD_PROJECT;
        delete process.env.GCLOUD_PROJECT;

        // WICHTIG: Übergebe den apiKey direkt UND setze baseUrl explizit auf die Developer-API
        this.genAI = new GoogleGenAI({
            apiKey: apiKey,
            baseUrl: 'https://generativelanguage.googleapis.com',
            platform: 'googleai'
        });
    }

    async sendRequest(turns, systemMessage) {
        console.log('Awaiting Google API response...');

        turns = strictFormat(turns);
        let contents = [];
        for (let turn of turns) {
            contents.push({
                role: turn.role === 'assistant' ? 'model' : 'user',
                parts: [{ text: turn.content }]
            });
        }

        // Das neue SDK erwartet System-Anweisungen und Parameter flach in der 'config'
        const result = await this.genAI.models.generateContent({
            model: this.model_name,
            contents: contents,
            config: {
                systemInstruction: systemMessage || undefined,
                safetySettings: this.safetySettings,
                ...(this.params || {})
            }
        });
        
        const response = result.text;
        console.log('Received.');
        return response;
    }

    async sendVisionRequest(turns, systemMessage, imageBuffer) {
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
            console.log('Awaiting Google API vision response...');
            const result = await this.genAI.models.generateContent({
                model: this.model_name,
                contents: contents,
                config: {
                    systemInstruction: systemMessage || undefined,
                    safetySettings: this.safetySettings,
                    ...(this.params || {})
                }
            });
            res = result.text;
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
        const result = await this.genAI.models.embedContent({
            model: this.model_name || "gemini-embedding-001",
            contents: text,
        });

        return result.embeddings;
    }
}

const sendAudioRequest = async (text, model, voice, url) => {
    const cleanedModel = model ? model.replace(/^(models\/|google\/)/, '') : "gemini-2.5-flash";
    
    delete process.env.GOOGLE_APPLICATION_CREDENTIALS;
    delete process.env.GOOGLE_GCLOUD_ADC_JSON;
    delete process.env.GOOGLE_CLOUD_PROJECT;
    delete process.env.GCLOUD_PROJECT;

    // Auch hier die explizite Konfiguration erzwingen
    const ai = new GoogleGenAI({
        apiKey: getKey('GEMINI_API_KEY'),
        baseUrl: 'https://generativelanguage.googleapis.com',
        platform: 'googleai'
    });

    const response = await ai.models.generateContent({
        model: cleanedModel,
        contents: [{ parts: [{text: text}] }],
        config: {
            responseModalities: ['AUDIO'],
            speechConfig: {
                voiceConfig: {
                    prebuiltVoiceConfig: { voiceName: voice },
                },
            },
        },
    });

    const pcmBase64 = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
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
