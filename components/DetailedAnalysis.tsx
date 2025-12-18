"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ChevronDown, ChevronUp, Play, Pause, AlertCircle } from "lucide-react";
import { Analysis, RecordingAnalysis } from "./SpeechFeedback";

interface DetailedAnalysisProps {
    analysis: Analysis | null;
    recordingAnalysis: RecordingAnalysis | null;
    transcription: string | null;
}

export default function DetailedAnalysis({ analysis, recordingAnalysis, transcription }: DetailedAnalysisProps) {
    const [isTranscriptionOpen, setIsTranscriptionOpen] = useState(false);

    // Dynamic feedback text based onmetrics
    const getFillerFeedback = () => {
        if (!analysis) return "No data available.";
        const fillerPct = analysis.filler_percentage;
        if (fillerPct < 1) return "Wow, you're crushing it! Barely any filler words!";
        if (fillerPct < 3) return "Good job keeping filler words under control.";
        return "Try to pause instead of using filler words like 'um' or 'like'.";
    };

    const getVocabFeedback = () => {
        if (!analysis) return "none";
        const level = analysis.ttr_analysis.diversity_level;
        if (level === "very high" || level === "high") return "Excellent vocabulary variety!";
        return "Consider broadening your vocabulary to make your speech more engaging.";
    };

    const renderHighlightedTranscript = () => {
        if (!transcription) return "No speech detected.";
        if (!analysis?.found_fillers || analysis.found_fillers.length === 0) return transcription;

        // Create a case-insensitive regex for fillers
        // Escape special chars in fillers just in case
        const escapeRegExp = (string: string) => string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const fillers = analysis.found_fillers.map(f => escapeRegExp(f.toLowerCase()));

        // Split text by words but keep delimiters to reconstruct
        // Actually, easiest way is to split by space and check generic lowercasing
        const words = transcription.split(/(\s+)/);

        return words.map((part, i) => {
            const cleanPart = part.toLowerCase().replace(/[.,!?]/g, '');
            // Check if this part is a known filler for this session
            // found_fillers contains all detected fillers, even repeats. 
            // We just check purely if it is a filler word type.
            // But found_fillers is a list of ALL instances found. 
            // Better strategy: Check if the word is in the global FILLER_WORDS list or just highlight the ones detected.
            // Since we don't have the global list here, we rely on 'found_fillers' which lists specific instances found.
            // We can just check if 'cleanPart' is in the set of unique found fillers.

            const isFiller = fillers.includes(cleanPart);

            if (isFiller) {
                return <span key={i} className="text-red-400 font-bold underline decoration-wavy underline-offset-2">{part}</span>;
            }
            return part;
        });
    };

    return (
        <div className="w-full max-w-2xl mx-auto space-y-8 text-white">

            {/* 1. Filler Words Feedback */}
            <div className="text-center space-y-2">
                <h3 className="text-gray-400 font-medium uppercase tracking-wider text-sm">Filler Words</h3>
                <p className="text-xl font-medium text-blue-400">
                    {getFillerFeedback()}
                </p>
            </div>

            {/* 2. Vocabulary Feedback */}
            <div className="text-center space-y-2">
                <h3 className="text-gray-400 font-medium uppercase tracking-wider text-sm">Vocabulary Diversity</h3>
                <p className="text-lg text-white font-semibold">
                    {analysis?.ttr_analysis.diversity_level || "none"}
                </p>
                <p className="text-sm text-gray-500">
                    {getVocabFeedback()}
                </p>
            </div>

            {/* 3. Stats Grid (3 Cols: Words, Fillers, Pauses) */}
            <div className="grid grid-cols-3 gap-8 border-y border-gray-800 py-8">
                <div className="text-center">
                    <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Total Words</div>
                    <div className="text-4xl font-bold">{analysis?.speaking_metrics?.word_count ?? analysis?.total_words ?? 0}</div>
                </div>
                <div className="text-center">
                    <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Fillers</div>
                    <div className="text-4xl font-bold">{analysis?.filler_percentage ?? 0}%</div>
                </div>
                <div className="text-center">
                    <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Pauses</div>
                    <div className="text-4xl font-bold">{analysis?.speaking_metrics?.estimated_pauses ?? 0}</div>
                </div>
            </div>

            {/* 4. Emotion & Gaze Analysis Header */}
            <div className="text-center">
                <h3 className="text-lg font-semibold mb-6">Emotion & Gaze Analysis</h3>
            </div>

            {/* 5. Emotions List */}
            <div className="space-y-6">
                <div className="flex items-center gap-3 mb-2">
                    <span className="font-medium text-gray-300">Top Emotions</span>
                    <span className="text-2xl">
                        {recordingAnalysis?.emotions[0]?.emotion === 'happy' ? '😊' :
                            recordingAnalysis?.emotions[0]?.emotion === 'sad' ? '😔' :
                                recordingAnalysis?.emotions[0]?.emotion === 'fear' ? '😨' : '😐'}
                    </span>
                </div>

                {recordingAnalysis?.emotions.map((item, idx) => (
                    <div key={idx} className="flex items-center gap-4 text-sm">
                        <div className="w-20 capitalize text-gray-400 font-medium">{item.emotion}</div>
                        <div className="flex-1 bg-gray-800 rounded-full h-2 overflow-hidden">
                            <div
                                className="h-full bg-white rounded-full transition-all duration-1000 ease-out"
                                style={{ width: `${item.percentage}%` }}
                            />
                        </div>
                        <div className="w-12 text-right font-mono text-gray-300">{item.percentage}%</div>
                    </div>
                ))}

                {!recordingAnalysis && (
                    <div className="text-center text-gray-500 italic py-2">No emotion data available</div>
                )}
            </div>

            {/* 6. Gaze Direction */}
            <div className="space-y-3 pt-4">
                <div className="flex items-center gap-3 mb-2">
                    <span className="font-medium text-gray-300">Gaze Direction</span>
                    <span className="text-2xl">👀</span>
                </div>

                <div className="bg-gray-900 rounded-xl p-6 flex items-center justify-between border border-gray-800">
                    <div>
                        <span className="text-xl font-bold capitalize text-white">
                            {recordingAnalysis?.gaze[0]?.direction || "Center"}
                        </span>
                        <div className="text-gray-500 text-sm mt-1">Primary focus direction</div>
                    </div>
                </div>
            </div>

            {/* 7. Transcription Accordion */}
            <div className="pt-6">
                <button
                    onClick={() => setIsTranscriptionOpen(!isTranscriptionOpen)}
                    className="w-full flex items-center justify-between bg-gray-900 p-5 rounded-lg border border-gray-800 hover:bg-gray-800 transition-colors"
                >
                    <h3 className="font-bold text-lg">View Transcription</h3>
                    {isTranscriptionOpen ? <ChevronUp className="text-gray-400" /> : <ChevronDown className="text-gray-400" />}
                </button>

                {isTranscriptionOpen && (
                    <div className="mt-2 bg-black/40 p-6 rounded-lg border border-gray-800 text-gray-300 font-mono text-base leading-loose whitespace-pre-wrap">
                        {renderHighlightedTranscript()}
                    </div>
                )}
            </div>

        </div>
    );
}
