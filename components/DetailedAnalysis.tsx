"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ChevronDown, ChevronUp, Play, Pause, AlertCircle, Sparkles, TrendingUp, Target, Brain, Eye } from "lucide-react";
import { Analysis, RecordingAnalysis } from "./SpeechFeedback";
import dynamic from "next/dynamic";

// Dynamic import for FloatingPointsCanvas (loads asynchronously)
const FloatingPointsCanvas = dynamic(
    () => import("@/components/ui/FloatingPoints"),
    { ssr: false }
);

interface DetailedAnalysisProps {
    analysis: Analysis | null;
    recordingAnalysis: RecordingAnalysis | null;
    transcription: string | null;
}

interface Suggestion {
    severity: 'good' | 'moderate' | 'needs_work';
    tip: string;
    actionable: string;
    icon: string;
}

interface AISuggestions {
    fillers?: Suggestion;
    pauses?: Suggestion;
    emotions?: Suggestion;
    gaze?: Suggestion;
    overall: {
        score: number;
        summary: string;
        nextSteps: string[];
    };
}

export default function DetailedAnalysis({ analysis, recordingAnalysis, transcription }: DetailedAnalysisProps) {
    const [isTranscriptionOpen, setIsTranscriptionOpen] = useState(false);
    const [suggestions, setSuggestions] = useState<AISuggestions | null>(null);
    const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(false);

    useEffect(() => {
        if (analysis && recordingAnalysis) {
            fetchSuggestions();
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [analysis, recordingAnalysis]);

    const fetchSuggestions = async () => {
        if (!analysis || !recordingAnalysis) return;

        setIsLoadingSuggestions(true);
        try {
            const response = await fetch('http://localhost:5328/api/generate-suggestions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    analysis,
                    recordingAnalysis,
                    transcription,
                    practiceMode: 'general'
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            if (data.success && data.suggestions) {
                setSuggestions(data.suggestions);
            } else {
                console.warn('No suggestions returned from API');
            }
        } catch (error) {
            console.error('Failed to fetch suggestions:', error);
            // Optionally set a fallback or show error to user
        } finally {
            setIsLoadingSuggestions(false);
        }
    };

    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case 'good': return 'border-green-400/40 bg-green-400/10';
            case 'moderate': return 'border-yellow-400/40 bg-yellow-400/10';
            case 'needs_work': return 'border-orange-400/40 bg-orange-400/10';
            default: return 'border-gray-400/40 bg-gray-400/10';
        }
    };

    const getSeverityTextColor = (severity: string) => {
        switch (severity) {
            case 'good': return 'text-green-400';
            case 'moderate': return 'text-yellow-400';
            case 'needs_work': return 'text-orange-400';
            default: return 'text-gray-400';
        }
    };

    const renderHighlightedTranscript = () => {
        if (!transcription) return "No speech detected.";
        if (!analysis?.found_fillers || analysis.found_fillers.length === 0) return transcription;

        const escapeRegExp = (string: string) => string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const fillers = analysis.found_fillers.map(f => escapeRegExp(f.toLowerCase()));
        const words = transcription.split(/(\s+)/);

        return words.map((part, i) => {
            const cleanPart = part.toLowerCase().replace(/[.,!?]/g, '');
            const isFiller = fillers.includes(cleanPart);

            if (isFiller) {
                return <span key={i} className="text-red-400 font-bold underline decoration-wavy underline-offset-2">{part}</span>;
            }
            return part;
        });
    };

    // Get score color based on value
    const getScoreColor = (score: number) => {
        if (score >= 80) return 'from-green-500 to-emerald-500';
        if (score >= 60) return 'from-blue-500 to-cyan-500';
        if (score >= 40) return 'from-yellow-500 to-orange-500';
        return 'from-orange-500 to-red-500';
    };

    return (
        <div className="relative min-h-screen overflow-hidden">
            {/* Three.js Floating Points Background */}
            <div className="absolute inset-0 -z-10">
                <FloatingPointsCanvas />
            </div>

            {/* Main Content */}
            <div className="relative z-10 w-full max-w-6xl mx-auto space-y-8 text-white p-6">


                {/* Hero Section - Performance Score */}
                <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-purple-900/40 via-blue-900/40 to-indigo-900/40 border border-purple-500/30 p-8">
                    <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-10"></div>
                    <div className="relative z-10">
                        <div className="flex items-center justify-between mb-6">
                            <div>
                                <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent mb-2">
                                    Performance Analysis
                                </h1>
                                <p className="text-gray-400">Your speaking session breakdown</p>
                            </div>
                            {suggestions && (
                                <div className="text-right">
                                    <div className={`text-7xl font-black bg-gradient-to-r ${getScoreColor(suggestions.overall.score)} bg-clip-text text-transparent`}>
                                        {suggestions.overall.score}
                                    </div>
                                    <div className="text-gray-400 text-sm mt-1">Overall Score</div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Metrics Dashboard - 3 Column Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Total Words Card */}
                    <div className="group relative overflow-hidden rounded-2xl bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border border-blue-500/30 p-6 hover:scale-105 transition-transform duration-300">
                        <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/20 rounded-full blur-3xl"></div>
                        <div className="relative z-10">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="p-3 bg-blue-500/20 rounded-xl">
                                    <TrendingUp className="w-6 h-6 text-blue-400" />
                                </div>
                                <div className="text-sm text-gray-400 uppercase tracking-wider">Total Words</div>
                            </div>
                            <div className="text-5xl font-black text-white mb-2">
                                {analysis?.speaking_metrics?.word_count ?? analysis?.total_words ?? 0}
                            </div>
                            <div className="text-sm text-blue-300">
                                {analysis?.speaking_metrics?.word_count > 50 ? 'Great length!' : 'Try speaking more'}
                            </div>
                        </div>
                    </div>

                    {/* Filler Words Card */}
                    <div className="group relative overflow-hidden rounded-2xl bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/30 p-6 hover:scale-105 transition-transform duration-300">
                        <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/20 rounded-full blur-3xl"></div>
                        <div className="relative z-10">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="p-3 bg-purple-500/20 rounded-xl">
                                    <Target className="w-6 h-6 text-purple-400" />
                                </div>
                                <div className="text-sm text-gray-400 uppercase tracking-wider">Filler Words</div>
                            </div>
                            <div className="text-5xl font-black text-white mb-2">
                                {analysis?.filler_percentage ?? 0}%
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="flex-1 bg-gray-800 rounded-full h-2 overflow-hidden">
                                    <div
                                        className={`h-full rounded-full transition-all duration-1000 ${(analysis?.filler_percentage ?? 0) < 3 ? 'bg-green-500' :
                                            (analysis?.filler_percentage ?? 0) < 7 ? 'bg-yellow-500' : 'bg-red-500'
                                            }`}
                                        style={{ width: `${Math.min((analysis?.filler_percentage ?? 0) * 10, 100)}%` }}
                                    />
                                </div>
                                <span className="text-xs text-gray-400">{analysis?.filler_count ?? 0} total</span>
                            </div>
                        </div>
                    </div>

                    {/* Pauses Card */}
                    <div className="group relative overflow-hidden rounded-2xl bg-gradient-to-br from-emerald-500/10 to-teal-500/10 border border-emerald-500/30 p-6 hover:scale-105 transition-transform duration-300">
                        <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/20 rounded-full blur-3xl"></div>
                        <div className="relative z-10">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="p-3 bg-emerald-500/20 rounded-xl">
                                    <Pause className="w-6 h-6 text-emerald-400" />
                                </div>
                                <div className="text-sm text-gray-400 uppercase tracking-wider">Pauses</div>
                            </div>
                            <div className="text-5xl font-black text-white mb-2">
                                {analysis?.speaking_metrics?.estimated_pauses ?? 0}
                            </div>
                            <div className="text-sm text-emerald-300">
                                {(analysis?.speaking_metrics?.estimated_pauses ?? 0) > 8 ? 'Excellent pacing!' : 'Add more pauses'}
                            </div>
                        </div>
                    </div>
                </div>

                {/* AI Suggestions Section */}
                <div className="space-y-6">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg">
                            <Sparkles className="h-6 w-6 text-white" />
                        </div>
                        <h2 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                            AI-Powered Insights
                        </h2>
                    </div>

                    {isLoadingSuggestions ? (
                        <div className="flex flex-col items-center justify-center py-16 space-y-6">
                            <div className="relative">
                                <div className="animate-spin text-7xl">🤖</div>
                                <div className="absolute inset-0 bg-purple-500/20 rounded-full blur-2xl animate-pulse"></div>
                            </div>
                            <p className="text-xl text-gray-300 animate-pulse">Analyzing your performance...</p>
                        </div>
                    ) : suggestions ? (
                        <>
                            {/* Overall Summary - Full Width */}
                            <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-indigo-500/20 via-purple-500/20 to-pink-500/20 border border-indigo-400/40 p-8 backdrop-blur-sm">
                                <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-5"></div>
                                <div className="relative z-10">
                                    <div className="flex items-start justify-between mb-6">
                                        <div className="flex-1">
                                            <h3 className="text-2xl font-bold text-white mb-3 flex items-center gap-2">
                                                <Brain className="w-7 h-7 text-purple-400" />
                                                Overall Assessment
                                            </h3>
                                            <p className="text-lg text-gray-100 leading-relaxed">{suggestions.overall.summary}</p>
                                        </div>
                                    </div>

                                    <div className="bg-black/30 rounded-xl p-6 border border-white/10">
                                        <p className="text-sm text-purple-300 font-semibold mb-4 flex items-center gap-2">
                                            <span className="text-2xl">🎯</span> Your Next Steps
                                        </p>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                            {suggestions.overall.nextSteps.map((step, idx) => (
                                                <div key={idx} className="flex items-start gap-3 bg-purple-500/10 rounded-lg p-4 border border-purple-500/20">
                                                    <span className="text-purple-400 font-bold text-lg">{idx + 1}</span>
                                                    <span className="text-gray-200 flex-1">{step}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Detailed Insights Grid - 2x2 */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                {suggestions.fillers && (
                                    <InsightCard
                                        title="Filler Words"
                                        suggestion={suggestions.fillers}
                                        getSeverityColor={getSeverityColor}
                                        getSeverityTextColor={getSeverityTextColor}
                                    />
                                )}
                                {suggestions.pauses && (
                                    <InsightCard
                                        title="Pacing & Pauses"
                                        suggestion={suggestions.pauses}
                                        getSeverityColor={getSeverityColor}
                                        getSeverityTextColor={getSeverityTextColor}
                                    />
                                )}
                                {suggestions.emotions && (
                                    <InsightCard
                                        title="Emotional Expression"
                                        suggestion={suggestions.emotions}
                                        getSeverityColor={getSeverityColor}
                                        getSeverityTextColor={getSeverityTextColor}
                                    />
                                )}
                                {suggestions.gaze && (
                                    <InsightCard
                                        title="Eye Contact"
                                        suggestion={suggestions.gaze}
                                        getSeverityColor={getSeverityColor}
                                        getSeverityTextColor={getSeverityTextColor}
                                    />
                                )}
                            </div>
                        </>
                    ) : null}
                </div>

                {/* Emotion & Gaze Analysis */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Emotions */}
                    <div className="rounded-2xl bg-gradient-to-br from-pink-500/10 to-rose-500/10 border border-pink-500/30 p-6">
                        <div className="flex items-center gap-3 mb-6">
                            <span className="text-3xl">😊</span>
                            <h3 className="text-xl font-bold text-white">Emotion Detection</h3>
                        </div>

                        <div className="space-y-4">
                            {recordingAnalysis?.emotions.map((item, idx) => (
                                <div key={idx} className="space-y-2">
                                    <div className="flex items-center justify-between text-sm">
                                        <span className="capitalize text-gray-300 font-medium">{item.emotion}</span>
                                        <span className="text-pink-400 font-bold">{item.percentage}%</span>
                                    </div>
                                    <div className="bg-gray-800/50 rounded-full h-3 overflow-hidden">
                                        <div
                                            className="h-full bg-gradient-to-r from-pink-500 to-rose-500 rounded-full transition-all duration-1000 ease-out"
                                            style={{ width: `${item.percentage}%` }}
                                        />
                                    </div>
                                </div>
                            ))}
                            {!recordingAnalysis?.emotions && (
                                <div className="text-center text-gray-500 italic py-4">No emotion data available</div>
                            )}
                        </div>
                    </div>

                    {/* Gaze */}
                    <div className="rounded-2xl bg-gradient-to-br from-cyan-500/10 to-blue-500/10 border border-cyan-500/30 p-6">
                        <div className="flex items-center gap-3 mb-6">
                            <Eye className="w-7 h-7 text-cyan-400" />
                            <h3 className="text-xl font-bold text-white">Gaze Analysis</h3>
                        </div>

                        <div className="bg-black/30 rounded-xl p-6 border border-cyan-500/20">
                            <div className="text-center">
                                <div className="text-5xl font-black capitalize bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent mb-2">
                                    {recordingAnalysis?.gaze[0]?.direction || "Center"}
                                </div>
                                <div className="text-gray-400 text-sm mb-4">Primary gaze direction</div>
                                <div className="flex items-center justify-center gap-2">
                                    <div className="flex-1 bg-gray-800 rounded-full h-2 overflow-hidden max-w-xs">
                                        <div
                                            className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-1000"
                                            style={{ width: `${recordingAnalysis?.gaze[0]?.percentage || 0}%` }}
                                        />
                                    </div>
                                    <span className="text-cyan-400 font-bold">{recordingAnalysis?.gaze[0]?.percentage || 0}%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Transcription Section */}
                <div className="rounded-2xl bg-gradient-to-br from-gray-800/50 to-gray-900/50 border border-gray-700/50 overflow-hidden">
                    <button
                        onClick={() => setIsTranscriptionOpen(!isTranscriptionOpen)}
                        className="w-full flex items-center justify-between p-6 hover:bg-gray-800/30 transition-colors"
                    >
                        <div className="flex items-center gap-3">
                            <div className="p-2 bg-gray-700 rounded-lg">
                                <Play className="w-5 h-5 text-gray-300" />
                            </div>
                            <h3 className="font-bold text-xl text-white">Full Transcription</h3>
                        </div>
                        {isTranscriptionOpen ?
                            <ChevronUp className="text-gray-400 w-6 h-6" /> :
                            <ChevronDown className="text-gray-400 w-6 h-6" />
                        }
                    </button>

                    {isTranscriptionOpen && (
                        <div className="px-6 pb-6">
                            <div className="bg-black/40 p-6 rounded-xl border border-gray-700 text-gray-300 font-mono text-base leading-loose whitespace-pre-wrap">
                                {renderHighlightedTranscript()}
                            </div>
                            <p className="text-xs text-gray-500 mt-3 text-center">
                                <span className="text-red-400">■</span> Highlighted words are detected fillers
                            </p>
                        </div>
                    )}
                </div>

            </div>
        </div>
    );
}

// Enhanced Insight Card Component
function InsightCard({
    title,
    suggestion,
    getSeverityColor,
    getSeverityTextColor
}: {
    title: string;
    suggestion: Suggestion;
    getSeverityColor: (s: string) => string;
    getSeverityTextColor: (s: string) => string;
}) {
    return (
        <div className={`relative overflow-hidden border rounded-2xl p-6 ${getSeverityColor(suggestion.severity)} transition-all hover:scale-[1.02] hover:shadow-2xl group`}>
            <div className="absolute top-0 right-0 w-24 h-24 bg-white/5 rounded-full blur-2xl group-hover:bg-white/10 transition-all"></div>
            <div className="relative z-10">
                <div className="flex items-center gap-3 mb-4">
                    <span className="text-4xl">{suggestion.icon}</span>
                    <div className="flex-1">
                        <h4 className="font-bold text-white text-lg">{title}</h4>
                        <span className={`text-xs uppercase tracking-wider font-semibold ${getSeverityTextColor(suggestion.severity)}`}>
                            {suggestion.severity.replace('_', ' ')}
                        </span>
                    </div>
                </div>

                <p className="text-sm text-gray-200 mb-5 leading-relaxed">{suggestion.tip}</p>

                <div className="bg-black/40 rounded-xl p-4 border-l-4 border-white/50">
                    <p className="text-xs text-gray-400 mb-2 flex items-center gap-2 font-semibold">
                        <span className="text-yellow-400">💡</span> Action Step
                    </p>
                    <p className="text-sm font-medium text-white leading-relaxed">{suggestion.actionable}</p>
                </div>
            </div>
        </div>
    );
}

