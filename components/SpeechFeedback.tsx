"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ChevronDown, ChevronUp } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export interface TTRAnalysis {
    ttr: number;
    unique_words: number;
    diversity_level: string;
    emoji: string;
}



export interface EmotionAnalysis {
    dominant_emotion: string;
    emotional_stability: number;
    engagement_score: number;
    emotion_distribution: { [key: string]: number };
    mood_consistency: number;
}

export interface SpeakingMetrics {
    word_count: number;
    unique_words: number;
    vocabulary_richness: number;
    avg_word_length: number;
    estimated_pauses: number;
    confidence_score: number;
    confidence_level: string;
    confidence_emoji: string;
}

export interface Analysis {
    total_words: number;
    filler_count: number;
    filler_percentage: number;
    found_fillers: string[];
    filler_emoji: string;
    ttr_analysis: TTRAnalysis;
    emotion_analysis?: EmotionAnalysis;
    speaking_metrics?: SpeakingMetrics;
}

export interface RecordingEmotion {
    emotion: string;
    percentage: string;
}

export interface RecordingGaze {
    direction: string;
    percentage: string;
}

export interface RecordingAnalysis {
    emotions: RecordingEmotion[];
    gaze: RecordingGaze[];
    duration: number;
}

interface FeedbackStep {
    id: string;
    title: string;
    content: string;
    emoji: string;
    score?: number;
}

interface SpeechFeedbackProps {
    analysis: Analysis | null;
    recordingAnalysis: RecordingAnalysis | null;
    audioUrl: string;
    filename?: string; // Optional, for local storage lookups if needed
}

export default function SpeechFeedback({
    analysis,
    recordingAnalysis,
    audioUrl,
    filename
}: SpeechFeedbackProps) {
    const [currentStep, setCurrentStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [audioError, setAudioError] = useState<string | null>(null);
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const [practiceCategory, setPracticeCategory] = useState<string | null>(null);

    useEffect(() => {
        // Check if the audio URL is a Blob URL (which means it's an enhanced audio) or if filename is provided
        if ((audioUrl && audioUrl.startsWith("blob:")) || filename) {
            // We'll rely on the parent to pass the category usually, but if we need to fetch it:
            if (filename) {
                try {
                    // First try metadata
                    const metadataKey = `recording_metadata_${filename}`;
                    const metadata = localStorage.getItem(metadataKey);
                    if (metadata) {
                        const parsedMetadata = JSON.parse(metadata);
                        if (parsedMetadata.category) {
                            setPracticeCategory(parsedMetadata.category);
                        }
                    } else {
                        // Try analysis data
                        const analysisKey = `recording_analysis_${filename}`;
                        const analysisData = localStorage.getItem(analysisKey);
                        if (analysisData) {
                            const parsedData = JSON.parse(analysisData);
                            if (parsedData.category) {
                                setPracticeCategory(parsedData.category);
                            }
                        }
                    }
                } catch (e) {
                    console.warn("Error extracting practice category:", e);
                }
            }
        }
    }, [audioUrl, filename]);

    const getFeedbackSteps = (): FeedbackStep[] => {
        if (!analysis || !recordingAnalysis) return [];

        return [
            {
                id: "stats",
                title: "Speech Statistics",
                content: `You used ${analysis.total_words} total words, with ${analysis.ttr_analysis.unique_words
                    } being unique.`,
                emoji: "📊",
            },
            {
                id: "emotions",
                title: "Emotional Tone",
                content: `Your speech mostly sounded ${recordingAnalysis.emotions[0]?.emotion.toLowerCase() || "neutral"
                    }, with some moments of ${recordingAnalysis.emotions[1]?.emotion.toLowerCase() || "variation"
                    }.`,
                emoji: "🎭",
            },
            {
                id: "gaze",
                title: "Eye Direction",
                content: `You were mostly looking ${recordingAnalysis.gaze[0]?.direction.toLowerCase() || "forward"
                    } during your speech.`,
                emoji: "👀",
            },
        ];
    };

    const steps = getFeedbackSteps();

    const startPlayback = async () => {
        if (!audioRef.current) return;

        try {
            setAudioError(null);
            setIsPlaying(true);
            await audioRef.current.play();

            // Progress through steps every 4 seconds
            const interval = setInterval(() => {
                setCurrentStep((prev) => {
                    if (prev >= steps.length - 1) {
                        clearInterval(interval);
                        return prev;
                    }
                    return prev + 1;
                });
            }, 4000);

            audioRef.current.onended = () => {
                setIsPlaying(false);
                clearInterval(interval);
                setCurrentStep(steps.length - 1);
            };
        } catch (error) {
            console.error("Error playing audio:", error);
            setAudioError("Could not play enhanced audio. Please try again.");
            setIsPlaying(false);
        }
    };

    // Calculate a score based on analysis data for visual display
    const getOverallScore = (): number => {
        if (!analysis) return 70; // Default score

        // Calculate score based on filler words, vocabulary diversity, and logical flow
        const fillerScore = 100 - Math.min(100, analysis.filler_percentage * 4);

        // Convert string diversity level to score
        const diversityScore =
            analysis.ttr_analysis.diversity_level === "very high"
                ? 95
                : analysis.ttr_analysis.diversity_level === "high"
                    ? 85
                    : analysis.ttr_analysis.diversity_level === "average"
                        ? 70
                        : analysis.ttr_analysis.diversity_level === "low"
                            ? 50
                            : 30;

        // Logical flow is already a percentage
        // Weighted average (excl logic)
        return Math.round(
            fillerScore * 0.5 + diversityScore * 0.5
        );
    };

    const overallScore = getOverallScore();

    return (
        <Card className="w-full max-w-2xl border-2">
            <CardHeader className="pb-6">
                <div className="flex flex-col items-center space-y-4">
                    <CardTitle className="flex items-center gap-2 text-2xl">
                        <span className="font-bold">AI-Enhanced Speech Analysis</span>
                        {practiceCategory && (
                            <Badge className="ml-2" variant="outline">
                                {practiceCategory.charAt(0).toUpperCase() +
                                    practiceCategory.slice(1).replace("-", " ")}
                            </Badge>
                        )}
                    </CardTitle>

                    {/* Score display */}
                    <div className="mt-6 flex items-center justify-center">
                        <div className="relative w-28 h-28">
                            <svg className="w-28 h-28" viewBox="0 0 100 100">
                                <circle
                                    className="text-muted stroke-current"
                                    strokeWidth="8"
                                    cx="50"
                                    cy="50"
                                    r="40"
                                    fill="transparent"
                                />
                                <circle
                                    className="text-primary stroke-current"
                                    strokeWidth="8"
                                    strokeLinecap="round"
                                    cx="50"
                                    cy="50"
                                    r="40"
                                    fill="transparent"
                                    strokeDasharray={`${2 * Math.PI * 40}`}
                                    strokeDashoffset={`${2 * Math.PI * 40 * (1 - overallScore / 100)
                                        }`}
                                    transform="rotate(-90 50 50)"
                                />
                                <text
                                    x="50"
                                    y="50"
                                    fontFamily="sans-serif"
                                    fontSize="22"
                                    textAnchor="middle"
                                    dy="7"
                                    fill="currentColor"
                                >
                                    {overallScore}
                                </text>
                            </svg>
                        </div>
                        <div className="ml-6">
                            <h3 className="font-semibold text-lg">Overall Score</h3>
                            <p className="text-sm text-muted-foreground">
                                {overallScore >= 90
                                    ? "Excellent!"
                                    : overallScore >= 80
                                        ? "Great job!"
                                        : overallScore >= 70
                                            ? "Good work!"
                                            : overallScore >= 60
                                                ? "Room for improvement"
                                                : "Keep practicing"}
                            </p>
                        </div>
                    </div>
                </div>
            </CardHeader>

            <CardContent className="space-y-8 pt-0">
                <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setCurrentStep((prev) => Math.max(0, prev - 1))}
                            disabled={currentStep === 0 || !steps.length}
                            className="h-8 w-8 p-0 rounded-full"
                        >
                            <ChevronUp className="h-4 w-4" />
                            <span className="sr-only">Previous</span>
                        </Button>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={() =>
                                setCurrentStep((prev) => Math.min(steps.length - 1, prev + 1))
                            }
                            disabled={currentStep === steps.length - 1 || !steps.length}
                            className="h-8 w-8 p-0 rounded-full"
                        >
                            <ChevronDown className="h-4 w-4" />
                            <span className="sr-only">Next</span>
                        </Button>
                    </div>

                    <Button
                        onClick={
                            isPlaying
                                ? () => {
                                    audioRef.current?.pause();
                                    setIsPlaying(false);
                                }
                                : startPlayback
                        }
                        disabled={!steps.length}
                        variant="default"
                    >
                        <span className="flex items-center gap-2">
                            {isPlaying ? (
                                <>
                                    <span className="h-2 w-2 rounded-full bg-current animate-pulse" />
                                    Pause Audio
                                </>
                            ) : (
                                <>
                                    <span className="h-0 w-0 border-y-4 border-y-transparent border-l-8 border-l-current" />
                                    Play Enhanced Audio
                                </>
                            )}
                        </span>
                    </Button>
                </div>

                <audio
                    ref={audioRef}
                    src={audioUrl}
                    onEnded={() => setIsPlaying(false)}
                    onError={(e) => {
                        console.error("Audio error:", e);
                        setIsPlaying(false);
                        setAudioError(
                            "Error playing enhanced audio. Please try again later."
                        );
                    }}
                />

                {audioError && (
                    <div className="mb-4 p-3 bg-destructive/10 text-destructive rounded-md">
                        {audioError}
                    </div>
                )}

                {/* Feedback steps */}
                <div className="mt-8 relative">
                    <div className="absolute left-4 inset-y-0 w-0.5 bg-muted" />

                    <AnimatePresence mode="wait">
                        {steps[currentStep] && (
                            <motion.div
                                key={steps[currentStep].id}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                transition={{ duration: 0.3 }}
                                className="ml-6 relative pb-8"
                            >
                                <span className="flex items-center justify-center w-10 h-10 rounded-full bg-primary text-primary-foreground absolute -left-10">
                                    {steps[currentStep].emoji}
                                </span>
                                <div className="bg-card border rounded-lg p-5 shadow-sm hover:shadow-md transition-shadow">
                                    <h3 className="text-lg font-medium mb-3">
                                        {steps[currentStep].title}
                                    </h3>
                                    <p className="text-muted-foreground">
                                        {steps[currentStep].content}
                                    </p>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* Step indicators */}
                    <div className="flex gap-2 mt-6 justify-center">
                        {steps.map((_, i) => (
                            <button
                                key={i}
                                className={`w-2.5 h-2.5 rounded-full transition-colors ${i === currentStep ? "bg-primary" : "bg-muted"
                                    }`}
                                onClick={() => setCurrentStep(i)}
                                aria-label={`Step ${i + 1}`}
                            />
                        ))}
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}
