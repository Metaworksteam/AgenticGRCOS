
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, Type } from "@google/genai";
import { virtualAgents } from '../data/virtualAgents';
import type { OrganizationSize, VirtualAgent, Risk, PolicyDocument, AssessmentItem, AuditAction } from '../types';
import { UserGroupIcon, ShieldCheckIcon, SparklesIcon, MicrophoneIcon, ChatBotIcon, UploadIcon, PaperClipIcon, CloseIcon, DocumentTextIcon, EyeIcon } from './Icons';

interface VirtualDepartmentPageProps {
    onDelegateTask: (agentName: string, task: string) => void;
    onConsultAgent: (agent: VirtualAgent) => void;
    risks?: Risk[];
    documents?: PolicyDocument[];
    eccAssessment?: AssessmentItem[];
    pdplAssessment?: AssessmentItem[];
    onAddDocument?: (doc: PolicyDocument) => void;
    onAddRisk?: (risk: Risk) => void;
    onAddAuditLog?: (action: AuditAction, details: string) => void;
}

let _ai: GoogleGenAI | null = null;
const getAI = (): GoogleGenAI => {
    if (!_ai) _ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    return _ai;
};

// Define the simulated dialogue entry
interface DialogueEntry {
    speaker: string;
    message_en: string;
    message_ar: string;
    action?: string; 
    timestamp: number;
    attachment?: { name: string, type: string };
    canRetry?: boolean;
}

// Browser Speech Recognition Types
declare global {
    interface Window {
        webkitSpeechRecognition: any;
        SpeechRecognition: any;
    }
}

export const VirtualDepartmentPage: React.FC<VirtualDepartmentPageProps> = ({ 
    onDelegateTask, 
    onConsultAgent,
    risks = [],
    documents = [],
    eccAssessment = [],
    pdplAssessment = [],
    onAddDocument,
    onAddRisk,
    onAddAuditLog
}) => {
    const [orgSize, setOrgSize] = useState<OrganizationSize>('Mid-Market');
    const [selectedAgent, setSelectedAgent] = useState<VirtualAgent | null>(null);
    const [agentTaskInput, setAgentTaskInput] = useState('');
    
    // Live Collaboration State
    const [isLiveMode, setIsLiveMode] = useState(false);
    const [meetingLog, setMeetingLog] = useState<DialogueEntry[]>([]);
    const [isThinking, setIsThinking] = useState(false);
    const [streamingContent, setStreamingContent] = useState<string>(''); // live typing indicator
    
    // Voice & Input State
    const [isMicActive, setIsMicActive] = useState(false);
    const [userSpeechInput, setUserSpeechInput] = useState('');
    const [userTextInput, setUserTextInput] = useState('');
    const [activeLanguage, setActiveLanguage] = useState<string>('en-US');
    const recognitionRef = useRef<any>(null);
    const bestMaleVoiceRef = useRef<SpeechSynthesisVoice | null>(null);
    const autonomousRunningRef = useRef<boolean>(false);
    const autonomousTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const sessionActionCountRef = useRef<number>(0);
    const SESSION_ACTION_LIMIT = 8;
    // Call-ID: each AI invocation gets a unique ID; stale responses from interrupted calls are discarded
    const currentCallIdRef = useRef<number>(0);
    const isThinkingRef = useRef<boolean>(false); // ref mirror for use inside closures
    const abortControllerRef = useRef<AbortController | null>(null); // cancels in-flight Gemini HTTP request
    const sessionLangRef = useRef<string | null>(null); // explicit language override set by "speak Arabic" etc.
    const lastUserMessageRef = useRef<string>(''); // stored for one-click retry on transient errors

    // Persistent meeting state — survives across turns so agents remember context
    const meetingStateRef = useRef<{
        activeFramework: string | null;
        assessmentPhase: string | null;
        assessmentProgress: number;
        lastTopic: string | null;
        pendingQuestions: string[];
        findings: string[];
    }>({
        activeFramework: null,
        assessmentPhase: null,
        assessmentProgress: 0,
        lastTopic: null,
        pendingQuestions: [],
        findings: [],
    });

    // Consult session state — one-on-one live voice session with a specific agent
    const [consultMode, setConsultMode] = useState<VirtualAgent | null>(null);
    const [consultLog, setConsultLog] = useState<DialogueEntry[]>([]);
    const [interimTranscript, setInterimTranscript] = useState<string>('');
    const [isMicMuted, setIsMicMuted] = useState(false);
    const isMicMutedRef = useRef(false);          // ref so closures read latest value
    const isAgentSpeakingRef = useRef(false);     // suppresses recognition while TTS plays
    const liveListeningRef = useRef(false);       // true = continuous mic should be active
    const consultModeRef = useRef<VirtualAgent | null>(null); // ref mirror of consultMode
    
    // Document Upload for Meeting
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [uploadedFile, setUploadedFile] = useState<{name: string, data: string, type: string} | null>(null);
    const [isAnalyzingDoc, setIsAnalyzingDoc] = useState(false);

    const chatContainerRef = useRef<HTMLDivElement>(null);

    // Scroll to bottom
    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [meetingLog, isThinking]);

    // Initialize Speech Recognition — continuous mode (ChatGPT voice style)
    useEffect(() => {
        if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) return;
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognitionRef.current = new SpeechRecognition();
        recognitionRef.current.continuous = true;   // keep session alive between utterances
        recognitionRef.current.interimResults = true; // show live partial transcript
        recognitionRef.current.lang = activeLanguage;

        recognitionRef.current.onresult = (event: any) => {
            let interim = '';
            let finalText = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                if (event.results[i].isFinal) {
                    finalText += event.results[i][0].transcript;
                } else {
                    interim += event.results[i][0].transcript;
                }
            }
            // User has speaking priority — the moment ANY speech is detected, kill TTS immediately
            if (interim || finalText) {
                window.speechSynthesis.cancel();
                isAgentSpeakingRef.current = false;
            }
            if (interim) setInterimTranscript(interim);
            if (finalText.trim()) {
                setInterimTranscript('');
                setUserSpeechInput(finalText.trim());
                handleUserSpeak(finalText.trim());
            }
        };

        recognitionRef.current.onerror = (event: any) => {
            // 'no-speech' is normal in continuous mode — not an error worth logging
            if (event.error === 'no-speech' || event.error === 'aborted') return;
            console.error("Speech recognition error", event.error);
        };

        // Auto-restart when recognition session ends (browser caps sessions at ~60 s)
        recognitionRef.current.onend = () => {
            setInterimTranscript('');
            if (liveListeningRef.current && !isMicMutedRef.current && !isAgentSpeakingRef.current) {
                try { recognitionRef.current?.start(); } catch (_) {}
            } else if (!liveListeningRef.current) {
                setIsMicActive(false);
            }
        };
    }, [activeLanguage]);

    // Pick the highest-quality male voice available in the browser, once voices load.
    useEffect(() => {
        if (!('speechSynthesis' in window)) return;

        const pickBest = () => {
            const voices = window.speechSynthesis.getVoices();
            if (!voices.length) return;

            const malePreferences = [
                'Google UK English Male',
                'Microsoft Guy Online',
                'Microsoft Davis Online',
                'Microsoft David',
                'Microsoft Mark',
                'Microsoft Aaron',
                'Daniel',
                'Alex',
                'Fred',
                'Rishi',
            ];

            let chosen: SpeechSynthesisVoice | undefined;
            for (const name of malePreferences) {
                chosen = voices.find(v => v.name.includes(name));
                if (chosen) break;
            }
            // Fallback: any voice whose name hints at male
            if (!chosen) {
                chosen = voices.find(v => /male|guy|david|mark|daniel|alex/i.test(v.name) && !/female/i.test(v.name));
            }
            // Final fallback: first English voice
            if (!chosen) {
                chosen = voices.find(v => v.lang.startsWith('en'));
            }
            if (chosen) bestMaleVoiceRef.current = chosen;
        };

        pickBest();
        window.speechSynthesis.onvoiceschanged = pickBest;
        return () => { window.speechSynthesis.onvoiceschanged = null; };
    }, []);

    // -------------------------------------------------------------
    // AUTONOMOUS EXECUTION ENGINE
    // Agents take real actions when gaps are detected — no chatter.
    // Each action: real system update → audit log → past-tense status entry.
    // -------------------------------------------------------------
    const appendStatus = (speaker: string, message_en: string, actionLabel?: string) => {
        const langSnapshot = activeLanguage;
        const entry: DialogueEntry = {
            speaker,
            message_en,
            message_ar: "...",
            timestamp: Date.now(),
            action: actionLabel,
        };
        setMeetingLog(prev => [...prev, entry]);
        // Speak completed-action announcement so the user hears the progress
        speakLine(message_en, speaker, langSnapshot);
    };

    // Discovery scan: reads gaps and REPORTS them — no system writes.
    // User must explicitly direct the team before any action is taken.
    const runDiscoveryScan = async () => {
        if (autonomousRunningRef.current) return;
        autonomousRunningRef.current = true;
        sessionActionCountRef.current = 0; // reset session action counter on new session
        try {
            const ecc = eccAssessment || [];
            const pdpl = pdplAssessment || [];
            const allAssessments = [...ecc, ...pdpl];
            const existingControlIds = new Set((documents || []).map(d => d.controlId));

            const gaps = allAssessments
                .filter(item => item.controlStatus !== 'Implemented')
                .filter(item => !existingControlIds.has(item.controlCode));

            const unscored = (risks || []).filter(r => !r.residualScore).length;
            const evidenceGaps = allAssessments.filter(i => i.controlStatus === 'Implemented' && !i.evidence).length;

            if (gaps.length === 0 && unscored === 0 && evidenceGaps === 0) {
                appendStatus(
                    "Asaad AI",
                    `Scan complete. All controls implemented. No open evidence gaps detected. Awaiting instruction.`,
                    "Scan Complete"
                );
                return;
            }

            // Report findings in a single message — no writes yet
            const parts: string[] = [];
            if (gaps.length > 0) parts.push(`${gaps.length} unaddressed control gap${gaps.length > 1 ? 's' : ''} (${gaps.slice(0,3).map(g => g.controlCode).join(', ')}${gaps.length > 3 ? '…' : ''})`);
            if (unscored > 0) parts.push(`${unscored} risk${unscored > 1 ? 's' : ''} without a residual score`);
            if (evidenceGaps > 0) parts.push(`${evidenceGaps} implemented control${evidenceGaps > 1 ? 's' : ''} missing evidence`);

            appendStatus(
                "Asaad AI",
                `Scan complete. Findings: ${parts.join('; ')}. Awaiting instruction.`,
                `Scan: ${gaps.length} gaps, ${unscored} risks, ${evidenceGaps} evidence gaps`
            );
        } finally {
            autonomousRunningRef.current = false;
        }
    };

    // Lightweight language detection — returns BCP-47 lang for synthesis/recognition.
    const detectLang = (text: string): string => {
        if (/[\u0600-\u06FF]/.test(text)) return 'ar-SA';   // Arabic
        if (/[\u4E00-\u9FFF]/.test(text)) return 'zh-CN';   // Chinese
        if (/[\u0900-\u097F]/.test(text)) return 'hi-IN';   // Hindi
        if (/[\u3040-\u30FF]/.test(text)) return 'ja-JP';   // Japanese
        if (/[\uAC00-\uD7AF]/.test(text)) return 'ko-KR';   // Korean
        if (/\b(hola|buenos|gracias|por favor|qué|cómo)\b/i.test(text)) return 'es-ES';
        if (/\b(bonjour|merci|comment|s'il vous plaît|pourquoi)\b/i.test(text)) return 'fr-FR';
        if (/\b(hallo|danke|bitte|warum|wie)\b/i.test(text)) return 'de-DE';
        return 'en-US';
    };

    // ── Continuous voice control ──────────────────────────────────────────────
    const startLiveListening = () => {
        liveListeningRef.current = true;
        isMicMutedRef.current = false;
        setIsMicMuted(false);
        setIsMicActive(true);
        try { recognitionRef.current?.start(); } catch (_) {}
    };

    const stopLiveListening = () => {
        liveListeningRef.current = false;
        setIsMicActive(false);
        setInterimTranscript('');
        window.speechSynthesis.cancel();
        isAgentSpeakingRef.current = false;
        try { recognitionRef.current?.stop(); } catch (_) {}
    };

    const toggleMute = () => {
        const nowMuted = !isMicMutedRef.current;
        isMicMutedRef.current = nowMuted;
        setIsMicMuted(nowMuted);
        if (nowMuted) {
            setIsMicActive(false);
            setInterimTranscript('');
            try { recognitionRef.current?.stop(); } catch (_) {}
        } else {
            if (liveListeningRef.current && !isAgentSpeakingRef.current) {
                try { recognitionRef.current?.start(); setIsMicActive(true); } catch (_) {}
            }
        }
    };

    // Push-to-talk toggle for when NOT in live/consult mode
    const toggleMic = () => {
        if (isMicActive) {
            try { recognitionRef.current?.stop(); } catch (_) {}
            setIsMicActive(false);
        } else {
            try { recognitionRef.current?.start(); } catch (_) {}
            setIsMicActive(true);
        }
    };

    // Start a one-on-one voice consult session with a specific agent
    const startConsultSession = (agent: VirtualAgent) => {
        consultModeRef.current = agent;
        setConsultMode(agent);
        setConsultLog([]);
        setSelectedAgent(null);
        setIsLiveMode(false);
        stopLiveListening(); // clear any existing live session
        window.speechSynthesis.cancel();
        // Initial greeting from the agent
        const greeting: DialogueEntry = {
            speaker: agent.name,
            message_en: `This is ${agent.name}, your ${agent.title}. I'm ready. Go ahead.`,
            message_ar: `هذا ${agent.name}، ${agent.title} الخاص بك. أنا مستعد، تفضل.`,
            timestamp: Date.now()
        };
        setConsultLog([greeting]);
        setTimeout(() => {
            speakLine(greeting.message_en, agent.name, 'en-US');
            // Start continuous listening after greeting ends (handled by utterance.onend)
            liveListeningRef.current = true;
        }, 200);
    };

    const endConsultSession = () => {
        consultModeRef.current = null;
        setConsultMode(null);
        setConsultLog([]);
        setInterimTranscript('');
        stopLiveListening();
    };

    // Text-to-Speech Helper — natural, human-like male voice with subtle per-agent variation
    const speakLine = (text: string, speaker: string, langHint?: string) => {
        if (!('speechSynthesis' in window)) return;

        // Cancel previous speech to avoid overlap (only one agent speaks at a time)
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        const voices = window.speechSynthesis.getVoices();
        const lang = langHint || detectLang(text);

        // For non-English, prefer a native-language voice (browser TTS quality varies)
        let voice: SpeechSynthesisVoice | undefined;
        if (!lang.startsWith('en')) {
            voice = voices.find(v => v.lang.toLowerCase().startsWith(lang.slice(0, 2)) && /male|guy|david/i.test(v.name) && !/female/i.test(v.name))
                 || voices.find(v => v.lang.toLowerCase().startsWith(lang.slice(0, 2)));
        }
        // English: use the pre-picked best male voice
        if (!voice) voice = bestMaleVoiceRef.current || undefined;

        if (voice) utterance.voice = voice;
        utterance.lang = lang;

        // Natural consulting tone — not robotic, not overly excited
        utterance.rate = 0.96;
        utterance.pitch = 0.92;
        utterance.volume = 1.0;

        // Subtle per-agent personality (all male, but distinguishable)
        if (speaker.includes("Ahmed"))         { utterance.pitch = 0.85; utterance.rate = 0.93; } // Authoritative CISO
        else if (speaker.includes("Fahad"))    { utterance.pitch = 0.95; utterance.rate = 1.00; } // Tech CTO
        else if (speaker.includes("Mohammed")) { utterance.pitch = 0.90; utterance.rate = 0.96; } // Diplomatic CIO
        else if (speaker.includes("Rashid"))   { utterance.pitch = 0.88; utterance.rate = 0.94; } // Methodical risk
        else if (speaker.includes("Ibrahim"))  { utterance.pitch = 0.92; utterance.rate = 0.98; } // Operational
        else if (speaker.includes("Asaad"))    { utterance.pitch = 0.94; utterance.rate = 0.95; } // Compliance
        else if (speaker.includes("Abdullah")) { utterance.pitch = 0.90; utterance.rate = 0.97; } // Auditor

        // Pause recognition while agent speaks — prevents TTS from being transcribed as user input
        isAgentSpeakingRef.current = true;
        if (liveListeningRef.current) {
            try { recognitionRef.current?.stop(); } catch (_) {}
        }
        utterance.onend = () => {
            isAgentSpeakingRef.current = false;
            // Restart continuous listening after agent finishes speaking
            if (liveListeningRef.current && !isMicMutedRef.current) {
                setTimeout(() => {
                    try { recognitionRef.current?.start(); setIsMicActive(true); } catch (_) {}
                }, 350); // brief pause so user doesn't have to interrupt mid-echo
            }
        };
        window.speechSynthesis.speak(utterance);
    };

    // Explicit language switch commands — detected before any other processing
    const LANG_COMMANDS: Array<{ pattern: RegExp; lang: string; name: string }> = [
        { pattern: /speak arabic|respond arabic|switch arabic|تحدث بالعربية|بالعربي/i,   lang: 'ar-SA', name: 'Arabic' },
        { pattern: /speak english|respond english|switch english|in english/i,            lang: 'en-US', name: 'English' },
        { pattern: /speak french|respond french|en français|parle français/i,             lang: 'fr-FR', name: 'French' },
        { pattern: /speak spanish|respond spanish|en español|habla español/i,             lang: 'es-ES', name: 'Spanish' },
        { pattern: /speak german|respond german|auf deutsch|sprich deutsch/i,             lang: 'de-DE', name: 'German' },
        { pattern: /speak chinese|respond chinese|用中文|说中文/i,                         lang: 'zh-CN', name: 'Chinese' },
        { pattern: /speak hindi|respond hindi|हिंदी में बोलो/i,                          lang: 'hi-IN', name: 'Hindi' },
        { pattern: /speak urdu|respond urdu|اردو میں بولو/i,                             lang: 'ur-PK', name: 'Urdu' },
    ];

    const handleUserSpeak = async (text: string) => {
        if (!text.trim()) return;

        // Store for retry
        lastUserMessageRef.current = text;

        // 1. Stop TTS immediately
        window.speechSynthesis.cancel();
        isAgentSpeakingRef.current = false;

        // 2. Abort the in-flight HTTP request to Gemini (not just ignore the response)
        abortControllerRef.current?.abort();
        abortControllerRef.current = null;

        // 3. Bump call-ID so any response that returns after this is discarded
        currentCallIdRef.current += 1;

        // 4. Release the thinking guard — the new call needs to acquire it
        isThinkingRef.current = false;
        setIsThinking(false);
        setStreamingContent('');

        // 5. Explicit language command — "speak Arabic", "respond in French", etc.
        let detected = detectLang(text);
        const langCmd = LANG_COMMANDS.find(lc => lc.pattern.test(text));
        if (langCmd) {
            sessionLangRef.current = langCmd.lang;
            detected = langCmd.lang;
            // Update speech recognition to the new language immediately
            if (recognitionRef.current) {
                try { recognitionRef.current.stop(); } catch (_) {}
                recognitionRef.current.lang = langCmd.lang;
                setTimeout(() => { if (liveListeningRef.current) try { recognitionRef.current?.start(); } catch (_) {} }, 300);
            }
        } else {
            // Auto-detect — but respect a session-level explicit override for Latin-script input
            // (Arabic/CJK scripts always override because character set is unambiguous)
            const scriptDetected = /[\u0600-\u06FF\u4E00-\u9FFF\u0900-\u097F\u3040-\u30FF\uAC00-\uD7AF]/.test(text);
            if (!scriptDetected && sessionLangRef.current) {
                detected = sessionLangRef.current; // honour session override for Latin text
            } else if (scriptDetected) {
                sessionLangRef.current = detected; // script-detected lang becomes new session lang
            }
        }

        if (detected !== activeLanguage) setActiveLanguage(detected);

        const userEntry: DialogueEntry = {
            speaker: "You",
            message_en: text,
            message_ar: detected === 'ar-SA' ? text : "...",
            timestamp: Date.now()
        };

        const lockedAgent = consultModeRef.current?.name;
        if (consultModeRef.current) {
            setConsultLog(prev => [...prev, userEntry]);
        } else {
            setMeetingLog(prev => [...prev, userEntry]);
        }

        await runSimulationTurn(text, undefined, detected, lockedAgent);
    };

    // --- Simulation Logic ---

    const runSimulationTurn = async (userContext?: string, analysisContext?: string, langHint?: string, lockedAgent?: string) => {
        // Re-entrance guard (ref-based so it's visible inside async closures)
        if (isThinkingRef.current) return;
        isThinkingRef.current = true;
        setIsThinking(true);

        // Stamp this call; if the user interrupts, the stamp will be bumped and we bail
        const myCallId = currentCallIdRef.current;

        const addToLog = (entry: DialogueEntry) => {
            if (lockedAgent) {
                setConsultLog(prev => [...prev, entry]);
            } else {
                setMeetingLog(prev => [...prev, entry]);
            }
        };

        // Helper: bail if this call was superseded by user interrupting
        const stale = () => currentCallIdRef.current !== myCallId;

        // Create an AbortController for this call so the HTTP request is truly cancelled on interrupt
        const controller = new AbortController();
        abortControllerRef.current = controller;

        // ── Streaming call with smart retry + model fallback ────────────────
        // Primary: gemini-2.5-flash (3 attempts)
        // Fallback: gemini-2.0-flash (2 attempts) — only tried if primary exhausted with retryable errors
        // 429: 5 s → 15 s → 30 s    503/5xx: 1 s → 2 s → 4 s
        // Streams tokens into streamingContent for a live typing effect.
        const MODELS = ['gemini-2.5-flash', 'gemini-2.0-flash'];
        const callGeminiStream = async (sysInstruction: string, msg: string): Promise<string> => {
            let lastError: any;
            for (let mi = 0; mi < MODELS.length; mi++) {
                const model = MODELS[mi];
                const maxAttempts = mi === 0 ? 3 : 2;
                for (let attempt = 0; attempt < maxAttempts; attempt++) {
                    if (stale() || controller.signal.aborted) throw Object.assign(new Error('aborted'), { name: 'AbortError' });
                    try {
                        const streamIter = getAI().models.generateContentStream({
                            model,
                            contents: msg,
                            config: { systemInstruction: sysInstruction }
                        });
                        let accumulated = '';
                        for await (const chunk of await streamIter) {
                            if (stale() || controller.signal.aborted) throw Object.assign(new Error('aborted'), { name: 'AbortError' });
                            accumulated += (chunk as any).text || '';
                            if (!stale()) setStreamingContent(accumulated);
                        }
                        if (!stale()) setStreamingContent('');
                        return accumulated;
                    } catch (err: any) {
                        if (err?.name === 'AbortError' || controller.signal.aborted) throw err;
                        if (stale()) throw err;
                        lastError = err;
                        setStreamingContent('');
                        const status = err?.status || 0;
                        const is429 = status === 429;
                        const isRetryable = is429 || status === 503 || status === 500 || status === 502 || status === 504 || !status;
                        if (!isRetryable) break; // skip remaining attempts for this model
                        const isLastAttempt = attempt === maxAttempts - 1;
                        if (!isLastAttempt) {
                            // Network/unknown errors (status=0): short 1s delay, no exponential back-off
                            const delay = is429 ? [5000, 15000, 30000][attempt] : (!status ? 1000 : 1000 * Math.pow(2, attempt));
                            await new Promise(r => setTimeout(r, delay));
                        }
                    }
                }
            }
            throw lastError || new Error('All attempts failed');
        };

        try {
            if (!userContext && !analysisContext) return;

            // ── Live dashboard context (compact — minimise token burn) ────
            const eccCompliance = Math.round((eccAssessment.filter(i => i.controlStatus === 'Implemented').length / (eccAssessment.length || 1)) * 100);
            const eccGaps = eccAssessment.filter(i => i.controlStatus !== 'Implemented').length;
            const criticalRisks = risks.filter(r => (r.residualScore || 0) > 15).length;
            // Session-level explicit language override takes priority (set by "speak Arabic" commands)
            const userLang = sessionLangRef.current || langHint || activeLanguage;
            const langName = ({'en-US':'English','ar-SA':'Arabic','es-ES':'Spanish','fr-FR':'French','de-DE':'German','zh-CN':'Chinese','hi-IN':'Hindi','ja-JP':'Japanese','ko-KR':'Korean','ur-PK':'Urdu'} as Record<string,string>)[userLang] || 'English';

            // ── Conversation history — last 6 entries, capped at 180 chars each ─
            const activeLog = lockedAgent ? consultLog : meetingLog;
            const recentHistory = activeLog.slice(-6)
                .map(e => `[${e.speaker}]: ${e.message_en.slice(0, 180)}`)
                .join('\n');

            // ── Persistent meeting state ─────────────────────────────────
            const ms = meetingStateRef.current;
            // Auto-detect if user is starting an assessment
            if (userContext) {
                const u = userContext.toLowerCase();
                if (/\b(nca|ecc)\b/.test(u)) ms.activeFramework = 'NCA ECC';
                else if (/\b(pdpl|personal data)\b/.test(u)) ms.activeFramework = 'PDPL';
                else if (/\b(sama|csf)\b/.test(u)) ms.activeFramework = 'SAMA CSF';
                else if (/\b(iso\s*27001)\b/.test(u)) ms.activeFramework = 'ISO 27001';
                if (/start|begin|initiat|conduct|assess|audit/.test(u)) ms.assessmentPhase = 'discovery';
                if (/continu|next|proceed|go ahead/.test(u) && ms.assessmentPhase) ms.assessmentProgress += 1;
                ms.lastTopic = userContext.slice(0, 100);
            }

            const sessionCtx = [
                `NCA ECC: ${eccCompliance}% (${eccGaps} gaps) | Critical risks: ${criticalRisks}`,
                `Docs: ${documents.length} | Risks: ${risks.length}`,
                ms.activeFramework ? `Active framework: ${ms.activeFramework}` : '',
                ms.assessmentPhase ? `Assessment phase: ${ms.assessmentPhase} (step ${ms.assessmentProgress + 1})` : '',
                ms.findings.length ? `Prior findings: ${ms.findings.slice(-3).join('; ')}` : '',
            ].filter(Boolean).join(' | ');

            // ── Per-turn message ─────────────────────────────────────────
            let userMessage = '';
            if (lockedAgent && userContext) {
                userMessage = `CONSULT MODE — only ${lockedAgent} speaks.\nUser: "${userContext}"`;
            } else if (userContext) {
                userMessage = `User: "${userContext}"`;
            } else if (analysisContext) {
                userMessage = `Document analysis:\n${analysisContext}`;
            }
            if (recentHistory) userMessage += `\n\nRecent conversation:\n${recentHistory}`;

            // ── Compact system prompt (~60% smaller than previous) ───────
            const systemInstruction = `You are an AI GRC boardroom — senior advisors from PwC/Deloitte responding live to a client.

TEAM (select only the 1–2 most relevant):
  Fahad AI — CTO | tech, infra, cloud, networks
  Mohammed AI — CIO | IT strategy, data governance, digital transformation
  Ahmed AI — CISO | cybersecurity, threats, controls, incident response
  Rashid AI — Risk Manager | risk register, ISO 31000, treatment plans
  Ibrahim AI — Operations Dir | continuity, process, implementation
  Asaad AI — Compliance Mgr | NCA ECC, PDPL, SAMA CSF, ISO 27001, NIST
  Abdullah AI — Auditor | gap analysis, evidence, control testing

SESSION: ${sessionCtx}
${lockedAgent ? `CONSULT MODE: ONLY ${lockedAgent} speaks.` : 'BOARDROOM MODE: 1–2 agents max.'}

RULES:
- Jump straight to the answer. No "Hello I am X". No filler.
- Perform work when asked — produce the actual deliverable, not a description.
- For assessments: ask one focused question per turn, capture answers, build findings progressively.
- Use session data above before asking the user for information already known.
- Speak in ${langName}. message_ar always in Arabic.

OUTPUT — strict JSON array, 1–2 objects max:
[{"speaker":"Agent AI","message_en":"response in ${langName}","message_ar":"Arabic","deliverable":{"type":"policy|assessment|gap_analysis|risk_register|remediation_plan|audit_report","title":"...","content":"full markdown doc"},"action":{"type":"create_doc|assess_risk","title":"...","category":"..."}}]
Omit deliverable/action for conversational replies. Include deliverable when producing a formal document.`;

            // ── Stream from Gemini — live typing indicator + retry/fallback ─
            const rawText = await callGeminiStream(systemInstruction, userMessage);

            // Bail if user interrupted while we were waiting for the stream
            if (stale()) return;

            // ── Parse with fallback ───────────────────────────────────────
            let rawScript: any[] = [];
            try {
                rawScript = JSON.parse(rawText);
            } catch {
                const match = rawText.match(/\[[\s\S]*\]/);
                if (match) { try { rawScript = JSON.parse(match[0]); } catch { /* empty */ } }
            }
            if (!Array.isArray(rawScript)) rawScript = rawScript ? [rawScript] : [];

            const maxAgents = lockedAgent ? 1 : 2;
            const script = rawScript
                .filter((line: any) => line && typeof line.message_en === 'string' && line.speaker)
                .slice(0, maxAgents);

            if (script.length === 0) {
                if (!stale()) addToLog({ speaker: 'System', message_en: 'The response could not be parsed. Please try again.', message_ar: 'تعذّر تحليل الاستجابة.', timestamp: Date.now() });
                return;
            }

            // ── Deliver each agent response ───────────────────────────────
            for (let i = 0; i < script.length; i++) {
                if (stale()) break; // user interrupted between agents

                const line = script[i];
                const entry: DialogueEntry = {
                    speaker: line.speaker,
                    message_en: line.message_en,
                    message_ar: line.message_ar || '...',
                    timestamp: Date.now()
                };

                // ── Save deliverable document (enterprise-grade) ──────────
                if (line.deliverable && onAddDocument && sessionActionCountRef.current < SESSION_ACTION_LIMIT) {
                    const d = line.deliverable;
                    const fullContent = d.content || line.message_en;
                    const newDoc: PolicyDocument = {
                        id: `doc-boardroom-${Date.now()}`,
                        controlId: `BR-${Date.now().toString().slice(-5)}`,
                        domainName: 'Boardroom Session',
                        subdomainTitle: d.type ? d.type.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase()) : 'Deliverable',
                        controlDescription: d.title || line.message_en.slice(0, 120),
                        status: 'Pending CISO Approval',
                        content: { policy: `# ${d.title || 'Boardroom Deliverable'}\n\n${fullContent}`, procedure: "See document content.", guideline: "Generated by AI consulting team." },
                        approvalHistory: [],
                        createdAt: Date.now(),
                        updatedAt: Date.now(),
                        generatedBy: 'AI Agent'
                    };
                    onAddDocument(newDoc);
                    onAddAuditLog?.('DOCUMENT_GENERATED', `${line.speaker} produced "${d.title}" during boardroom session.`);
                    entry.action = `📄 ${d.title} saved to Documents`;
                    sessionActionCountRef.current += 1;
                }

                // ── Register risk if requested ────────────────────────────
                if (!entry.action && line.action?.type === 'assess_risk' && onAddRisk && sessionActionCountRef.current < SESSION_ACTION_LIMIT) {
                    const newRisk: Risk = {
                        id: `risk-boardroom-${Date.now()}`,
                        title: line.action.title,
                        description: line.message_en,
                        category: line.action.category || 'Operational',
                        owner: line.speaker,
                        inherentLikelihood: 3, inherentImpact: 4, inherentScore: 12,
                        existingControl: 'Under review', controlEffectiveness: 'Needs Improvement',
                        residualLikelihood: 2, residualImpact: 3, residualScore: 6,
                        likelihood: 3, impact: 4,
                        treatmentOption: 'Mitigate', mitigation: 'To be defined.', responsibility: 'IT Security',
                        dueDate: '', acceptanceCriteria: '', approvedBy: '', remarks: ''
                    };
                    onAddRisk(newRisk);
                    onAddAuditLog?.('RISK_ASSESSMENT_INITIATED', `${line.speaker} registered risk "${line.action.title}".`);
                    entry.action = `⚠️ Risk registered: ${line.action.title}`;
                    sessionActionCountRef.current += 1;
                }

                // ── Also handle create_doc action (without deliverable) ───
                if (!entry.action && line.action?.type === 'create_doc' && onAddDocument && sessionActionCountRef.current < SESSION_ACTION_LIMIT) {
                    const newDoc: PolicyDocument = {
                        id: `doc-boardroom-${Date.now()}`,
                        controlId: `BR-${Date.now().toString().slice(-5)}`,
                        domainName: 'Boardroom Session',
                        subdomainTitle: 'Live Collaboration',
                        controlDescription: line.action.title,
                        status: 'Pending CISO Approval',
                        content: { policy: `# ${line.action.title}\n\n${line.message_en}`, procedure: "TBD", guideline: "TBD" },
                        approvalHistory: [],
                        createdAt: Date.now(),
                        updatedAt: Date.now(),
                        generatedBy: line.speaker
                    };
                    onAddDocument(newDoc);
                    onAddAuditLog?.('DOCUMENT_GENERATED', `${line.speaker} drafted "${line.action.title}".`);
                    entry.action = `📄 Draft saved: ${line.action.title}`;
                    sessionActionCountRef.current += 1;
                }

                // Show in chat immediately
                if (!stale()) addToLog(entry);

                // Speak
                if (!stale()) speakLine(line.message_en, line.speaker, userLang);

                // Gap between multiple agents
                if (i < script.length - 1 && !stale()) {
                    await new Promise(r => setTimeout(r, Math.max(1200, line.message_en.length * 45)));
                }
            }

        } catch (e: any) {
            // Silent discard if user already interrupted this call
            const isAbort = e?.name === 'AbortError' || controller.signal.aborted;
            if (isAbort || stale()) return;

            const errMsg = e?.message || e?.toString() || '';
            const status = e?.status || e?.code || 0;
            console.error(`[Boardroom] Simulation failed — status=${status} msg="${errMsg}"`, e);

            // 403 or SDK "API key" warning → key misconfiguration
            const isApiKeyErr = status === 403 || /api.?key/i.test(errMsg);
            // 429 → quota exhausted
            const isQuota = status === 429;

            if (isApiKeyErr) {
                addToLog({ speaker: 'System', message_en: 'API key issue detected. Please verify your GEMINI_API_KEY secret is set correctly in the Replit Secrets panel, then restart the server.', message_ar: 'تم اكتشاف مشكلة في مفتاح API. يرجى التحقق من إعداد GEMINI_API_KEY في لوحة الأسرار.', timestamp: Date.now() });
            } else if (isQuota) {
                addToLog({ speaker: 'System', message_en: 'Quota limit reached. Please wait a moment and try again.', message_ar: 'تم الوصول إلى حد الحصة. يرجى الانتظار لحظة والمحاولة مجدداً.', timestamp: Date.now(), canRetry: true });
            } else {
                addToLog({ speaker: 'System', message_en: 'A transient error occurred. Click Retry to resend your message.', message_ar: 'حدث خطأ مؤقت. اضغط إعادة المحاولة لإعادة إرسال رسالتك.', timestamp: Date.now(), canRetry: true });
            }
        } finally {
            // Only reset if WE still own the thinking slot — a user interrupt bumps the callId
            // and the new call has already acquired the slot; stomping it would break that call.
            if (!stale()) {
                isThinkingRef.current = false;
                setIsThinking(false);
                setStreamingContent('');
            }
            // Clean up our abort controller reference if it's still ours
            if (abortControllerRef.current === controller) {
                abortControllerRef.current = null;
            }
        }
    };

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (loadEvent) => {
                const base64 = loadEvent.target?.result as string;
                setUploadedFile({
                    name: file.name,
                    data: base64.split(',')[1], // Strip prefix
                    type: file.type
                });
                // Auto-trigger analysis
                analyzeDocument(base64.split(',')[1], file.type);
            };
            reader.readAsDataURL(file);
        }
    };

    const analyzeDocument = async (base64Data: string, mimeType: string) => {
        setIsAnalyzingDoc(true);
        
        // Add "Upload" entry to chat
        const uploadEntry: DialogueEntry = {
            speaker: "You (User)",
            message_en: "I have uploaded a document for review.",
            message_ar: "لقد قمت بتحميل مستند للمراجعة.",
            timestamp: Date.now(),
            attachment: { name: fileInputRef.current?.files?.[0]?.name || "Document", type: "file" }
        };
        setMeetingLog(prev => [...prev, uploadEntry]);

        try {
            // Use Gemini Vision to analyze
            const prompt = `
            You are the collective intelligence of a GRC Department. 
            Analyze this uploaded document image/pdf.
            
            1. Identify what type of document this is (e.g., New Regulation, Audit Evidence, Policy Draft).
            2. If it is a New Regulation/Framework: Identify key requirements and gaps.
            3. If it is Evidence: Validate if it meets typical security controls (firewall rules, logs, etc.).
            
            Provide a concise summary of the analysis to be fed into the meeting simulation.
            `;

            const response = await getAI().models.generateContent({
                model: 'gemini-2.5-flash',
                contents: {
                    parts: [
                        { inlineData: { mimeType: mimeType || 'image/png', data: base64Data } },
                        { text: prompt }
                    ]
                }
            });

            const analysisResult = response.text || "Analysis complete.";
            
            // Feed this context into the simulation loop (use the active language)
            await runSimulationTurn(undefined, analysisResult, activeLanguage);

        } catch (err) {
            console.error("Vision analysis failed", err);
            const errorEntry: DialogueEntry = {
                speaker: "System",
                message_en: "Failed to analyze document. Please ensure it is a valid image or PDF.",
                message_ar: "فشل تحليل المستند.",
                timestamp: Date.now()
            };
            setMeetingLog(prev => [...prev, errorEntry]);
        } finally {
            setIsAnalyzingDoc(false);
            setUploadedFile(null);
        }
    };

    const generateMOM = async () => {
        if (meetingLog.length === 0) return;
        
        const logText = meetingLog.map(entry => `${entry.speaker}: ${entry.message_en} ${entry.action ? `[Action: ${entry.action}]` : ''}`).join('\n');
        
        try {
            const response = await getAI().models.generateContent({
                model: 'gemini-2.5-flash',
                contents: `Generate a formal Minutes of Meeting (MOM) document based on this transcript:\n\n${logText}\n\nInclude: Date, Attendees (Agents & User), Key Discussion Points, Decisions Made, and Action Items. Format as Markdown.`,
            });
            
            const momContent = response.text || "MOM Generation Failed";
            
            if (onAddDocument) {
                const momDoc: PolicyDocument = {
                    id: `mom-${Date.now()}`,
                    controlId: `MOM-${new Date().toISOString().slice(0,10)}`,
                    domainName: 'Governance',
                    subdomainTitle: 'Meeting Records',
                    controlDescription: 'Minutes of Meeting - Live Collaboration Session',
                    status: 'Approved',
                    content: {
                        policy: momContent,
                        procedure: "N/A",
                        guideline: "Distributed to Management"
                    },
                    approvalHistory: [],
                    createdAt: Date.now(),
                    updatedAt: Date.now(),
                    generatedBy: 'AI Agent'
                };
                onAddDocument(momDoc);
                alert("Minutes of Meeting generated and saved to Documents.");
            }
            
            setIsLiveMode(false); // End meeting
            
        } catch (e) {
            console.error("MOM Gen failed", e);
        }
    };

    const handleDelegate = (e: React.FormEvent) => {
        e.preventDefault();
        if (selectedAgent && agentTaskInput.trim()) {
            onDelegateTask(selectedAgent.name, agentTaskInput);
            setAgentTaskInput('');
            alert(`Task delegated to ${selectedAgent.name}. Check Live Assistant/Noora for updates.`);
        }
    };

    return (
        <div className="space-y-8">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                    <h1 className="text-4xl font-bold text-gray-900 dark:text-white flex items-center gap-3">
                        <UserGroupIcon className="w-10 h-10 text-teal-600" />
                        Virtual GRC & Cybersecurity Department
                    </h1>
                    <p className="mt-2 text-lg text-gray-600 dark:text-gray-400">
                        Your dedicated AI-powered security team, orchestrated by Noora.
                    </p>
                </div>
                
                <div className="flex items-center gap-2 bg-white dark:bg-gray-800 p-2 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                    <span className="text-sm font-medium text-gray-600 dark:text-gray-300">Org Size:</span>
                    <select 
                        value={orgSize} 
                        onChange={(e) => setOrgSize(e.target.value as OrganizationSize)}
                        className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-md focus:ring-teal-500 focus:border-teal-500 block p-1.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white"
                    >
                        <option value="Startup">Startup (10-50)</option>
                        <option value="Mid-Market">Mid-Market (100-500)</option>
                        <option value="Enterprise">Enterprise (1000+)</option>
                    </select>
                </div>
            </div>

            {/* LIVE COLLABORATION TOGGLE */}
            <div className="flex justify-end">
                <button
                    onClick={() => {
                        if (isLiveMode) {
                            stopLiveListening(); // always stop mic when ending live mode
                            if (confirm("End meeting and generate Minutes of Meeting?")) {
                                generateMOM();
                            } else {
                                setIsLiveMode(false);
                            }
                        } else {
                            setIsLiveMode(true);
                            setMeetingLog([]);
                            sessionActionCountRef.current = 0;
                            window.speechSynthesis?.cancel();
                            if (autonomousTimerRef.current) clearTimeout(autonomousTimerRef.current);
                            // Start continuous mic — agents wait for user to speak first
                            setTimeout(() => startLiveListening(), 400);
                        }
                    }}
                    className={`flex items-center gap-2 px-6 py-3 rounded-full font-bold shadow-lg transition-all ${
                        isLiveMode 
                        ? 'bg-red-500 hover:bg-red-600 text-white animate-pulse' 
                        : 'bg-green-600 hover:bg-green-700 text-white'
                    }`}
                >
                    {isLiveMode ? (
                        <>
                            <span className="h-3 w-3 bg-white rounded-full animate-ping"></span>
                            End & Generate MOM
                        </>
                    ) : (
                        <>
                            <ChatBotIcon className="w-5 h-5" />
                            Start Live Collaboration
                        </>
                    )}
                </button>
            </div>

            {/* LIVE MEETING ROOM VIEW */}
            {isLiveMode && (
                <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 animate-fade-in">
                    {/* Discussion Log */}
                    <div className="lg:col-span-3 bg-gray-900 rounded-xl shadow-2xl border border-gray-700 flex flex-col h-[600px]">
                        <div className="p-4 border-b border-gray-700 flex justify-between items-center bg-gray-800 rounded-t-xl">
                            <h3 className="text-white font-bold flex items-center gap-2">
                                <span className="h-2 w-2 bg-green-500 rounded-full"></span>
                                Strategic Alignment Meeting - Live
                            </h3>
                            <div className="flex items-center gap-3">
                                {isAnalyzingDoc && <span className="text-xs text-purple-400 animate-pulse flex items-center"><EyeIcon className="w-3 h-3 mr-1"/> CNN Analysis Active...</span>}
                                {isThinking && !streamingContent && <span className="text-xs text-gray-400 animate-pulse">Agents are thinking...</span>}
                                {streamingContent && <span className="text-xs text-teal-400 animate-pulse">● Responding...</span>}
                            </div>
                        </div>
                        
                        <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-6 space-y-4">
                            {meetingLog.length === 0 && (
                                <div className="text-center text-gray-400 mt-20 px-6">
                                    <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-gray-800 border border-gray-700 mb-4">
                                        <UserGroupIcon className="w-7 h-7 text-teal-400" />
                                    </div>
                                    <p className="text-base font-medium text-gray-200">The boardroom is in session.</p>
                                    <p className="text-sm text-gray-500 mt-2">Your AI consultants are ready and waiting. Speak using the microphone or type a message below — they will only respond when you address them.</p>
                                    <p className="text-xs text-gray-600 mt-3">You can speak in any language; they will reply in the same one.</p>
                                </div>
                            )}
                            {meetingLog.map((entry, idx) => {
                                const agent = virtualAgents.find(a => a.name === entry.speaker);
                                const isUser = entry.speaker.startsWith("You");
                                return (
                                    <div key={idx} className={`flex items-start gap-4 animate-fade-in ${isUser ? 'flex-row-reverse' : ''}`}>
                                        <div className="flex-shrink-0">
                                            {agent ? (
                                                <img src={agent.avatarUrl} className="w-10 h-10 rounded-full border border-gray-600" alt={entry.speaker} />
                                            ) : isUser ? (
                                                <div className="w-10 h-10 bg-teal-600 rounded-full flex items-center justify-center text-white font-bold">U</div>
                                            ) : (
                                                <div className="w-10 h-10 bg-gray-700 rounded-full"></div>
                                            )}
                                        </div>
                                        <div className={`flex-grow max-w-[80%] ${isUser ? 'text-right' : ''}`}>
                                            <div className={`flex items-baseline justify-between ${isUser ? 'flex-row-reverse' : ''}`}>
                                                <span className="font-bold text-teal-400 text-sm">{entry.speaker}</span>
                                                <span className="text-xs text-gray-500 mx-2">{new Date(entry.timestamp).toLocaleTimeString()}</span>
                                            </div>
                                            <div className={`mt-1 p-3 rounded-lg ${isUser ? 'bg-teal-900/50 border border-teal-700' : 'bg-gray-800 border border-gray-700'}`}>
                                                <p className="text-gray-200 text-sm">{entry.message_en}</p>
                                                {entry.message_ar && (
                                                    <p className="text-gray-400 text-xs mt-2 text-right font-arabic border-t border-gray-700 pt-1" dir="rtl">{entry.message_ar}</p>
                                                )}
                                            </div>
                                            {entry.attachment && (
                                                <div className="mt-1 inline-flex items-center gap-2 px-3 py-1 bg-purple-900/30 border border-purple-800 rounded text-xs text-purple-300">
                                                    <PaperClipIcon className="w-3 h-3" />
                                                    Attached: {entry.attachment.name}
                                                </div>
                                            )}
                                            {entry.action && (
                                                <div className="mt-1 inline-flex items-center gap-2 px-3 py-1 bg-green-900/30 border border-green-800 rounded text-xs text-green-300">
                                                    <SparklesIcon className="w-3 h-3" />
                                                    {entry.action}
                                                </div>
                                            )}
                                            {entry.canRetry && lastUserMessageRef.current && (
                                                <div className="mt-2">
                                                    <button
                                                        onClick={() => handleUserSpeak(lastUserMessageRef.current)}
                                                        disabled={isThinking}
                                                        className="inline-flex items-center gap-1.5 px-3 py-1 bg-teal-700 hover:bg-teal-600 disabled:opacity-40 text-white text-xs rounded-md transition-colors"
                                                    >
                                                        ↻ Retry
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                        
                        {/* Meeting Controls Footer */}
                        <div className="p-4 border-t border-gray-700 bg-gray-800 rounded-b-xl flex justify-between items-center gap-4">
                            <div className="flex items-center gap-2">
                                {/* Mic status — always-on in live mode, click to mute/unmute */}
                                <button 
                                    onClick={toggleMute}
                                    className={`p-3 rounded-full transition-all duration-300 flex items-center gap-1.5 ${
                                        isMicMuted
                                        ? 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                                        : isMicActive
                                            ? 'bg-green-600 text-white ring-2 ring-green-400 ring-opacity-60'
                                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                                    }`}
                                    title={isMicMuted ? "Unmute — click to listen again" : "Mute microphone"}
                                >
                                    <MicrophoneIcon className="w-5 h-5" />
                                    <span className="text-xs font-bold">{isMicMuted ? 'MUTED' : isMicActive ? '● LIVE' : ''}</span>
                                </button>

                                {/* Language selector — controls speech recognition locale */}
                                {(['en-US', 'ar-SA', 'auto'] as const).map(lang => {
                                    const label = lang === 'en-US' ? '🇬🇧 EN' : lang === 'ar-SA' ? '🇸🇦 AR' : '🌐';
                                    const isActive = lang === 'auto' ? activeLanguage !== 'en-US' && activeLanguage !== 'ar-SA' : activeLanguage === lang;
                                    return (
                                        <button
                                            key={lang}
                                            title={lang === 'auto' ? 'Auto-detect language' : lang === 'ar-SA' ? 'Arabic — تحدث بالعربية' : 'English'}
                                            onClick={() => {
                                                const newLang = lang === 'auto' ? 'en-US' : lang;
                                                sessionLangRef.current = lang === 'auto' ? null : lang;
                                                setActiveLanguage(newLang);
                                                if (recognitionRef.current) {
                                                    try { recognitionRef.current.stop(); } catch (_) {}
                                                    recognitionRef.current.lang = newLang;
                                                    setTimeout(() => {
                                                        if (liveListeningRef.current && !isMicMutedRef.current) {
                                                            try { recognitionRef.current?.start(); setIsMicActive(true); } catch (_) {}
                                                        }
                                                    }, 200);
                                                }
                                            }}
                                            className={`px-2 py-1 rounded text-xs font-bold transition-colors ${isActive ? 'bg-teal-600 text-white' : 'bg-gray-700 text-gray-400 hover:bg-gray-600 hover:text-white'}`}
                                        >
                                            {label}
                                        </button>
                                    );
                                })}

                                {/* Document Upload */}
                                <button 
                                    onClick={() => fileInputRef.current?.click()}
                                    className="p-3 rounded-full bg-gray-700 text-gray-300 hover:bg-gray-600 transition-colors"
                                    title="Upload Document for Analysis"
                                    disabled={isAnalyzingDoc}
                                >
                                    <UploadIcon className="w-5 h-5" />
                                </button>
                                <input 
                                    type="file" 
                                    ref={fileInputRef} 
                                    className="hidden" 
                                    accept="image/*,application/pdf"
                                    onChange={handleFileUpload}
                                />
                            </div>

                            <form
                                className="flex-grow flex items-center gap-2"
                                onSubmit={(e) => {
                                    e.preventDefault();
                                    const text = userTextInput.trim();
                                    if (!text || isThinking) return;
                                    setUserTextInput('');
                                    handleUserSpeak(text);
                                }}
                            >
                                <div className="flex-grow relative">
                                    <input
                                        type="text"
                                        value={interimTranscript && !userTextInput ? interimTranscript : userTextInput}
                                        onChange={(e) => { if (!interimTranscript) setUserTextInput(e.target.value); }}
                                        placeholder={isMicActive && !isMicMuted ? "🔴 Listening — speak or type…" : isMicMuted ? "Mic muted — type your message" : "Type a message in any language…"}
                                        disabled={isThinking}
                                        className={`w-full bg-gray-900 border border-gray-700 text-sm rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-teal-600 placeholder-gray-500 disabled:opacity-50 ${interimTranscript && !userTextInput ? 'text-green-400 italic' : 'text-gray-100'}`}
                                    />
                                </div>
                                <button
                                    type="submit"
                                    disabled={isThinking || !userTextInput.trim()}
                                    className="px-4 py-2 bg-teal-700 hover:bg-teal-600 text-white rounded text-sm disabled:opacity-50"
                                >
                                    Send
                                </button>
                            </form>
                        </div>
                    </div>

                    {/* Active Agents Status Sidebar */}
                    <div className="lg:col-span-1 space-y-3 h-[600px] overflow-y-auto pr-2">
                        {virtualAgents.map(agent => {
                            const isSpeaking = meetingLog.length > 0 && meetingLog[meetingLog.length - 1].speaker === agent.name;
                            return (
                                <div key={agent.id} className={`p-3 rounded-lg border transition-all duration-300 ${isSpeaking ? 'bg-teal-900/40 border-teal-500 scale-105 shadow-lg' : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 opacity-80'}`}>
                                    <div className="flex items-center gap-3">
                                        <img src={agent.avatarUrl} className="w-8 h-8 rounded-full" alt={agent.name} />
                                        <div>
                                            <p className={`text-xs font-bold ${isSpeaking ? 'text-teal-300' : 'text-gray-700 dark:text-gray-300'}`}>{agent.name}</p>
                                            <p className="text-[10px] text-gray-500">{agent.role}</p>
                                        </div>
                                        {isSpeaking && (
                                            <div className="ml-auto flex gap-0.5">
                                                <div className="w-1 h-3 bg-teal-500 animate-bounce"></div>
                                                <div className="w-1 h-3 bg-teal-500 animate-bounce delay-100"></div>
                                                <div className="w-1 h-3 bg-teal-500 animate-bounce delay-200"></div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                        
                        {/* Upload Status Card */}
                        {uploadedFile && (
                            <div className="p-3 rounded-lg border border-purple-500 bg-purple-900/20 mt-4">
                                <p className="text-xs font-bold text-purple-300 flex items-center gap-2">
                                    <DocumentTextIcon className="w-3 h-3"/>
                                    Analyzing File
                                </p>
                                <p className="text-[10px] text-gray-400 mt-1 truncate">{uploadedFile.name}</p>
                                <div className="w-full bg-gray-700 h-1 mt-2 rounded overflow-hidden">
                                    <div className="h-full bg-purple-500 animate-progress"></div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* ─── CONSULT SESSION PANEL ─── */}
            {consultMode && (
                <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/60 backdrop-blur-sm">
                    <div className="w-full max-w-2xl bg-gray-900 border-t border-teal-700 rounded-t-2xl shadow-2xl flex flex-col" style={{height: '60vh'}}>
                        {/* Header */}
                        <div className="flex items-center gap-4 px-5 py-4 border-b border-gray-700 bg-gray-800 rounded-t-2xl shrink-0">
                            <img src={consultMode.avatarUrl} className="w-11 h-11 rounded-full border-2 border-teal-500 object-cover" alt={consultMode.name} />
                            <div className="flex-1 min-w-0">
                                <p className="text-white font-bold text-base leading-tight">{consultMode.name}</p>
                                <p className="text-teal-400 text-xs font-medium">{consultMode.title}</p>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold ${
                                    isMicMuted ? 'bg-gray-700 text-gray-400' : isMicActive ? 'bg-green-600 text-white animate-pulse' : 'bg-gray-700 text-gray-400'
                                }`}>
                                    <MicrophoneIcon className="w-3.5 h-3.5" />
                                    {isMicMuted ? 'MUTED' : isMicActive ? '● LIVE' : 'STANDBY'}
                                </div>
                                <button
                                    onClick={toggleMute}
                                    className="px-3 py-1.5 rounded-full text-xs font-bold bg-gray-700 hover:bg-gray-600 text-gray-300 border border-gray-600"
                                >
                                    {isMicMuted ? 'Unmute' : 'Mute'}
                                </button>
                                <button
                                    onClick={endConsultSession}
                                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold bg-red-700 hover:bg-red-600 text-white"
                                >
                                    End Session
                                </button>
                            </div>
                        </div>

                        {/* Chat log */}
                        <div className="flex-1 overflow-y-auto p-5 space-y-3">
                            {consultLog.length === 0 && (
                                <div className="text-center text-gray-500 mt-12 px-6">
                                    <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-gray-800 border border-gray-700 mb-3">
                                        <MicrophoneIcon className="w-6 h-6 text-teal-400" />
                                    </div>
                                    <p className="text-sm text-gray-300 font-medium">One-on-one with {consultMode.name}</p>
                                    <p className="text-xs text-gray-500 mt-1">Speak or type — only {consultMode.name.split(' ')[0]} will respond.</p>
                                </div>
                            )}
                            {consultLog.map((entry, i) => {
                                const isUser = entry.speaker === 'You';
                                return (
                                    <div key={i} className={`flex items-end gap-2 ${isUser ? 'justify-end' : 'justify-start'}`}>
                                        {!isUser && (
                                            <img src={consultMode.avatarUrl} className="w-7 h-7 rounded-full border border-gray-600 shrink-0 object-cover" alt="" />
                                        )}
                                        <div className={`max-w-[80%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed ${
                                            isUser
                                            ? 'bg-teal-700 text-white rounded-br-sm'
                                            : 'bg-gray-800 text-gray-100 rounded-bl-sm border border-gray-700'
                                        }`}>
                                            {!isUser && <p className="text-[10px] font-bold text-teal-400 mb-0.5">{entry.speaker}</p>}
                                            <p>{entry.message_en}</p>
                                        </div>
                                    </div>
                                );
                            })}
                            {isThinking && !streamingContent && (
                                <div className="flex items-end gap-2 justify-start">
                                    <img src={consultMode.avatarUrl} className="w-7 h-7 rounded-full border border-gray-600 shrink-0 object-cover" alt="" />
                                    <div className="bg-gray-800 border border-gray-700 px-4 py-3 rounded-2xl rounded-bl-sm flex gap-1">
                                        <div className="w-1.5 h-1.5 bg-teal-400 rounded-full animate-bounce"></div>
                                        <div className="w-1.5 h-1.5 bg-teal-400 rounded-full animate-bounce delay-100"></div>
                                        <div className="w-1.5 h-1.5 bg-teal-400 rounded-full animate-bounce delay-200"></div>
                                    </div>
                                </div>
                            )}
                            {streamingContent && (
                                <div className="flex items-end gap-2 justify-start">
                                    <img src={consultMode.avatarUrl} className="w-7 h-7 rounded-full border border-gray-600 shrink-0 object-cover" alt="" />
                                    <div className="bg-gray-800 border border-gray-700 px-4 py-3 rounded-2xl rounded-bl-sm max-w-xs">
                                        <p className="text-xs text-teal-300 animate-pulse">● typing…</p>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Input */}
                        <div className="px-4 py-3 border-t border-gray-700 bg-gray-850 shrink-0">
                            <form className="flex items-center gap-2" onSubmit={(e) => {
                                e.preventDefault();
                                const text = userTextInput.trim();
                                if (!text || isThinking) return;
                                setUserTextInput('');
                                handleUserSpeak(text);
                            }}>
                                <input
                                    type="text"
                                    value={interimTranscript && !userTextInput ? interimTranscript : userTextInput}
                                    onChange={(e) => { if (!interimTranscript) setUserTextInput(e.target.value); }}
                                    placeholder={isMicActive && !isMicMuted ? "🔴 Listening — or type here…" : "Type a question…"}
                                    disabled={isThinking}
                                    className={`flex-grow bg-gray-800 border border-gray-700 text-sm rounded-xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-teal-600 placeholder-gray-500 ${interimTranscript && !userTextInput ? 'text-green-400 italic' : 'text-gray-100'}`}
                                />
                                <button
                                    type="submit"
                                    disabled={isThinking || !userTextInput.trim()}
                                    className="px-4 py-2.5 bg-teal-700 hover:bg-teal-600 text-white rounded-xl text-sm font-medium disabled:opacity-40"
                                >
                                    Send
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
            )}

            {/* Standard Grid View (Visible when NOT in live mode) */}
            {!isLiveMode && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {virtualAgents.map(agent => (
                        <div 
                            key={agent.id} 
                            className={`bg-white dark:bg-gray-800 rounded-xl shadow-lg border-2 transition-all cursor-pointer overflow-hidden flex flex-col h-full ${selectedAgent?.id === agent.id ? 'border-teal-500 ring-2 ring-teal-200 dark:ring-teal-900' : 'border-transparent hover:border-gray-200 dark:hover:border-gray-700'}`}
                            onClick={() => setSelectedAgent(agent)}
                        >
                            <div className="p-6 flex-grow">
                                <div className="flex items-center gap-4 mb-4">
                                    <div className="relative">
                                        <img src={agent.avatarUrl} alt={agent.name} className="w-16 h-16 rounded-full object-cover border-2 border-white dark:border-gray-700 shadow-md" />
                                        {agent.id === 'agent-abdullah' && (
                                            <div className="absolute -bottom-1 -right-1 bg-blue-600 text-white text-[9px] font-bold px-1.5 py-0.5 rounded-full animate-pulse border border-white">CNN ACTIVE</div>
                                        )}
                                    </div>
                                    <div>
                                        <h3 className="font-bold text-lg text-gray-900 dark:text-white">{agent.name}</h3>
                                        <p className="text-sm font-medium text-teal-600 dark:text-teal-400">{agent.title}</p>
                                    </div>
                                </div>
                                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 line-clamp-3">{agent.description}</p>
                                
                                <div className="space-y-2">
                                    {agent.capabilities.slice(0, 3).map((cap, i) => (
                                        <div key={i} className="flex items-center text-xs text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-700/50 px-2 py-1 rounded">
                                            <ShieldCheckIcon className="w-3 h-3 mr-1.5 text-teal-500" />
                                            {cap}
                                        </div>
                                    ))}
                                </div>
                            </div>
                            <div className="p-4 border-t border-gray-100 dark:border-gray-700 flex justify-between items-center bg-gray-50 dark:bg-gray-900/50">
                                <span className="text-xs text-gray-500">Reports to: <span className="font-semibold">{agent.reportingLine}</span></span>
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        startConsultSession(agent);
                                    }}
                                    className="flex items-center gap-1 text-xs font-bold text-purple-600 dark:text-purple-400 hover:text-purple-800 dark:hover:text-purple-300 transition-colors"
                                >
                                    <MicrophoneIcon className="w-3 h-3" />
                                    Consult
                                </button>
                            </div>
                            {agent.currentTask && (
                                <div className="bg-yellow-50 dark:bg-yellow-900/30 px-6 py-2 border-t border-yellow-100 dark:border-yellow-900/50">
                                    <p className="text-xs text-yellow-700 dark:text-yellow-400 font-medium truncate">
                                        <span className="animate-pulse mr-2">●</span>
                                        Working on: {agent.currentTask}
                                    </p>
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}

            {selectedAgent && (
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6 animate-fade-in">
                    <div className="flex justify-between items-start mb-6">
                        <div className="flex items-center gap-4">
                             <img src={selectedAgent.avatarUrl} alt={selectedAgent.name} className="w-20 h-20 rounded-full object-cover border-2 border-teal-500 shadow-md" />
                             <div>
                                <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                                    {selectedAgent.name}
                                </h2>
                                <p className="text-teal-600 dark:text-teal-400 font-medium">{selectedAgent.title}</p>
                                <div className="flex gap-2 mt-2">
                                    {selectedAgent.jobAttributes.map((attr, i) => (
                                        <span key={i} className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded-full text-gray-600 dark:text-gray-300 border border-gray-200 dark:border-gray-600">
                                            {attr}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        </div>
                        <div className="flex gap-3">
                             <button
                                onClick={() => startConsultSession(selectedAgent)}
                                className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium shadow-sm transition-colors"
                             >
                                 <MicrophoneIcon className="w-4 h-4" />
                                 Start Voice Session
                             </button>
                             <button onClick={() => setSelectedAgent(null)} className="text-gray-400 hover:text-gray-600 p-2">Close</button>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                        <div className="md:col-span-2 space-y-6">
                            <div>
                                <h3 className="text-sm font-bold text-gray-900 dark:text-gray-100 uppercase tracking-wide mb-2">Professional Bio</h3>
                                <p className="text-sm text-gray-600 dark:text-gray-300 leading-relaxed">{selectedAgent.fullBio}</p>
                            </div>
                            
                            <div>
                                <h3 className="text-sm font-bold text-gray-900 dark:text-gray-100 uppercase tracking-wide mb-2">Key Responsibilities</h3>
                                <ul className="space-y-2">
                                    {selectedAgent.responsibilities.map((resp, i) => (
                                        <li key={i} className="text-sm text-gray-600 dark:text-gray-300 flex items-start">
                                            <span className="mr-2 text-teal-500">•</span>
                                            {resp}
                                        </li>
                                    ))}
                                </ul>
                            </div>

                            <div className="pt-4 border-t border-gray-100 dark:border-gray-700">
                                <h3 className="text-sm font-bold text-gray-900 dark:text-gray-100 uppercase tracking-wide mb-3">Delegate Task</h3>
                                <form onSubmit={handleDelegate} className="space-y-4">
                                    <div>
                                        <textarea 
                                            className="w-full h-24 p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-teal-500 focus:border-teal-500 text-sm"
                                            placeholder={`Instruct ${selectedAgent.role} to perform a task within their domain...`}
                                            value={agentTaskInput}
                                            onChange={(e) => setAgentTaskInput(e.target.value)}
                                        ></textarea>
                                    </div>
                                    <div className="flex justify-end">
                                        <button 
                                            type="submit" 
                                            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-teal-600 hover:bg-teal-700 focus:outline-none"
                                            disabled={!agentTaskInput.trim()}
                                        >
                                            <SparklesIcon className="w-4 h-4 mr-2" />
                                            Delegate Task
                                        </button>
                                    </div>
                                </form>
                            </div>
                        </div>
                        
                        <div className="md:col-span-1 space-y-4">
                            <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                                <h3 className="text-sm font-bold text-gray-700 dark:text-gray-200 mb-3">Capabilities</h3>
                                <ul className="space-y-2">
                                    {selectedAgent.capabilities.map((cap, i) => (
                                        <li key={i} className="text-xs text-gray-600 dark:text-gray-400 flex items-start">
                                            <ShieldCheckIcon className="w-3 h-3 mr-2 text-green-500" />
                                            {cap}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                            
                            {selectedAgent.id === 'agent-abdullah' && (
                                <div className="p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded text-xs">
                                    <strong className="block text-blue-700 dark:text-blue-300 mb-1">CNN Feature Embedding</strong>
                                    Analyzing compliance artifacts with 98.5% accuracy for automated categorization.
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
            <style>{`
                @keyframes progress {
                    0% { width: 0%; }
                    50% { width: 70%; }
                    100% { width: 100%; }
                }
                .animate-progress {
                    animation: progress 2s ease-in-out infinite;
                }
            `}</style>
        </div>
    );
};
