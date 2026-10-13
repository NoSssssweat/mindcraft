import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { logoutAgent } from '../mindcraft/mindserver.js';

const GEMINI_REQUEST_INTERVAL_MS = 4100;
const geminiRequestQueues = new Map();

function reserveGeminiRequestSlot(modelName) {
    const state = geminiRequestQueues.get(modelName) || {
        queue: Promise.resolve(),
        nextRequestAt: 0
    };
    const scheduledRequest = state.queue.then(async () => {
        const delay = Math.max(0, state.nextRequestAt - Date.now());
        if (delay > 0) {
            await new Promise(resolve => setTimeout(resolve, delay));
        }
        state.nextRequestAt = Date.now() + GEMINI_REQUEST_INTERVAL_MS;
    });
    state.queue = scheduledRequest.catch(() => { });
    geminiRequestQueues.set(modelName, state);
    return scheduledRequest;
}

const init_agent_path = fileURLToPath(new URL('./init_agent.js', import.meta.url));

export class AgentProcess {
    constructor(name, port) {
        this.name = name;
        this.port = port;
    }

    start(load_memory=false, init_message=null, count_id=0) {
        this.count_id = count_id;
        this.running = true;

        let args = [init_agent_path, this.name];
        args.push('-n', this.name);
        args.push('-c', count_id);
        if (load_memory)
            args.push('-l', load_memory);
        if (init_message)
            args.push('-m', init_message);
        args.push('-p', this.port);

        const agentProcess = spawn(process.execPath, args, {
            stdio: ['inherit', 'inherit', 'inherit', 'ipc'],
            stderr: 'inherit',
        });

        agentProcess.on('message', message => {
            if (message?.type !== 'gemini-rate-limit-request') return;
            reserveGeminiRequestSlot(message.modelName).then(() => {
                if (!agentProcess.connected) return;
                agentProcess.send({
                    type: 'gemini-rate-limit-granted',
                    requestId: message.requestId
                });
            }).catch(error => {
                if (!agentProcess.connected) return;
                agentProcess.send({
                    type: 'gemini-rate-limit-granted',
                    requestId: message.requestId,
                    error: error.message
                });
            });
        });
        
        let last_restart = Date.now();
        agentProcess.on('exit', (code, signal) => {
            console.log(`Agent process exited with code ${code} and signal ${signal}`);
            this.running = false;
            logoutAgent(this.name);
            
            if (code > 1) {
                console.log(`Ending task`);
                process.exit(code);
            }

            if (code !== 0 && signal !== 'SIGINT') {
                // agent must run for at least 10 seconds before restarting
                if (Date.now() - last_restart < 10000) {
                    console.error(`Agent process exited too quickly and will not be restarted.`);
                    return;
                }
                console.log('Restarting agent...');
                this.start(true, 'Agent process restarted.', count_id, this.port);
                last_restart = Date.now();
            }
        });
    
        agentProcess.on('error', (err) => {
            console.error('Agent process error:', err);
        });

        this.process = agentProcess;
    }

    stop() {
        if (!this.running) return;
        this.process.kill('SIGINT');
    }

    forceRestart() {
        if (this.running && this.process && !this.process.killed) {
            console.log(`Agent process for ${this.name} is still running. Attempting to force restart.`);
            
            const restartTimeout = setTimeout(() => {
                console.warn(`Agent ${this.name} did not stop in time. It might be stuck.`);
            }, 5000); // 5 seconds to exit

            this.process.once('exit', () => {
                 clearTimeout(restartTimeout);
                 console.log(`Stopped hanging agent ${this.name}. Now restarting.`);
                 this.start(true, 'Agent process restarted.', this.count_id);
            });
            this.stop(); // sends SIGINT
        } else {
             this.start(true, 'Agent process restarted.', this.count_id);
        }
    }
}