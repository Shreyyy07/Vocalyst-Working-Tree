"use client";

import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import FloatingPoints from "@/components/ui/FloatingPoints";
import { Upload, Mic, StopCircle, FileAudio, Loader2 } from "lucide-react";

export default function SpeechToTextPage() {
    const [isRecording, setIsRecording] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [transcription, setTranscription] = useState("");
    const [analysis, setAnalysis] = useState<any>(null);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            audioChunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
                await processAudio(audioBlob);
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            setIsRecording(true);
        } catch (error) {
            console.error("Error starting recording:", error);
            alert("Failed to access microphone. Please check permissions.");
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setSelectedFile(file);
            processAudio(file);
        }
    };

    const processAudio = async (audioData: Blob | File) => {
        setIsProcessing(true);
        setTranscription("");
        setAnalysis(null);

        try {
            const formData = new FormData();
            formData.append("file", audioData, "audio.webm");

            console.log("Sending audio to backend...");

            const response = await fetch("http://localhost:5328/api/speech2text", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || "Transcription failed");
            }

            const data = await response.json();
            console.log("Backend response:", data);
            setTranscription(data.text);
            setAnalysis(data.analysis);
        } catch (error: any) {
            console.error("Transcription error:", error);
            alert(`Error: ${error.message}`);
        } finally {
            setIsProcessing(false);
        }
    };

    return (
        <div className="relative min-h-screen bg-black text-white overflow-hidden">
            <div className="absolute inset-0 z-0">
                <FloatingPoints />
            </div>

            <div className="container mx-auto py-20 px-4 z-10 relative">
                <h1 className="text-5xl font-extrabold text-center bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent mb-10">
                    Speech-to-Text Lab
                </h1>

                <div className="max-w-3xl mx-auto space-y-6">
                    {/* Action Buttons */}
                    <Card className="bg-transparent border border-white/20 p-8">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {/* Upload Button */}
                            <div className="space-y-2">
                                <label className="block text-sm font-semibold text-center">Upload File</label>
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept=".mp4,.wav,audio/*,video/*"
                                    onChange={handleFileSelect}
                                    className="hidden"
                                />
                                <Button
                                    onClick={() => fileInputRef.current?.click()}
                                    className="w-full h-12 bg-gradient-to-r from-white to-zinc-300 text-black hover:from-zinc-200 hover:to-white transition-all"
                                    disabled={isProcessing || isRecording}
                                >
                                    <Upload className="mr-2 h-5 w-5" />
                                    {selectedFile ? `${selectedFile.name.substring(0, 20)}...` : "Choose File"}
                                </Button>
                            </div>

                            {/* Record Button */}
                            <div className="space-y-2">
                                <label className="block text-sm font-semibold text-center">
                                    {isRecording ? "Recording..." : "Record Audio"}
                                </label>
                                {!isRecording ? (
                                    <Button
                                        onClick={startRecording}
                                        className="w-full h-12 bg-gradient-to-r from-white to-zinc-300 text-black hover:from-zinc-200 hover:to-white transition-all"
                                        disabled={isProcessing}
                                    >
                                        <Mic className="mr-2 h-5 w-5" />
                                        Start Recording
                                    </Button>
                                ) : (
                                    <Button
                                        onClick={stopRecording}
                                        className="w-full h-12 bg-gradient-to-r from-gray-600 to-gray-800 hover:opacity-90 transition-opacity animate-pulse"
                                    >
                                        <StopCircle className="mr-2 h-5 w-5" />
                                        Stop & Process
                                    </Button>
                                )}
                            </div>
                        </div>

                        {isRecording && (
                            <div className="mt-4 flex items-center justify-center space-x-2">
                                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse delay-75"></div>
                                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse delay-150"></div>
                            </div>
                        )}
                    </Card>

                    {/* Processing Indicator */}
                    {isProcessing && (
                        <Card className="bg-transparent border border-white/20 p-8">
                            <div className="flex flex-col items-center justify-center space-y-3">
                                <Loader2 className="h-10 w-10 animate-spin text-blue-400" />
                                <span className="text-lg font-medium">Processing audio...</span>
                                <span className="text-sm text-gray-400">This may take a moment</span>
                            </div>
                        </Card>
                    )}

                    {/* Transcription Results */}
                    {transcription && (
                        <Card className="bg-transparent border border-white/20 p-8 animate-in fade-in duration-500">
                            <h2 className="text-2xl font-bold mb-4 bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent">
                                Transcription
                            </h2>
                            <div className="bg-white/5 p-6 rounded-lg border border-white/10">
                                <p className="text-white leading-relaxed whitespace-pre-wrap">{transcription}</p>
                            </div>
                        </Card>
                    )}

                    {/* Speech Analysis */}
                    {analysis && (
                        <Card className="bg-transparent border border-white/20 p-8 animate-in fade-in duration-500">
                            <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                                Speech Analysis
                            </h2>

                            {/* Primary Metrics */}
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
                                <div className="bg-gradient-to-br from-blue-500/20 to-blue-600/10 p-4 rounded-xl border border-blue-400/30">
                                    <p className="text-xs text-blue-300 font-medium uppercase tracking-wide mb-1">Total Words</p>
                                    <p className="text-3xl font-bold text-blue-400">{analysis.total_words}</p>
                                </div>
                                <div className="bg-gradient-to-br from-red-500/20 to-red-600/10 p-4 rounded-xl border border-red-400/30">
                                    <p className="text-xs text-red-300 font-medium uppercase tracking-wide mb-1">Filler Words</p>
                                    <p className="text-3xl font-bold text-red-400">
                                        {analysis.filler_count} <span className="text-2xl">{analysis.filler_emoji}</span>
                                    </p>
                                </div>
                                <div className="bg-gradient-to-br from-purple-500/20 to-purple-600/10 p-4 rounded-xl border border-purple-400/30 col-span-2 md:col-span-1">
                                    <p className="text-xs text-purple-300 font-medium uppercase tracking-wide mb-1">Filler %</p>
                                    <p className="text-3xl font-bold text-purple-400">{analysis.filler_percentage}%</p>
                                </div>
                            </div>

                            {/* Advanced Speech Metrics */}
                            {analysis.speaking_metrics && (
                                <>
                                    <h3 className="text-lg font-semibold mb-4 text-white/90">Speaking Quality</h3>
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                                        <div className="bg-gradient-to-br from-green-500/20 to-green-600/10 p-4 rounded-xl border border-green-400/30">
                                            <p className="text-xs text-green-300 font-medium uppercase tracking-wide mb-1">Confidence</p>
                                            <p className="text-2xl font-bold text-green-400">
                                                {analysis.speaking_metrics.confidence_level} {analysis.speaking_metrics.confidence_emoji}
                                            </p>
                                            <p className="text-xs text-green-300/70 mt-1">{analysis.speaking_metrics.confidence_score}/100</p>
                                        </div>
                                        <div className="bg-gradient-to-br from-cyan-500/20 to-cyan-600/10 p-4 rounded-xl border border-cyan-400/30">
                                            <p className="text-xs text-cyan-300 font-medium uppercase tracking-wide mb-1">Vocabulary</p>
                                            <p className="text-2xl font-bold text-cyan-400">{analysis.speaking_metrics.vocabulary_richness}%</p>
                                            <p className="text-xs text-cyan-300/70 mt-1">{analysis.speaking_metrics.unique_words} unique</p>
                                        </div>
                                        <div className="bg-gradient-to-br from-yellow-500/20 to-yellow-600/10 p-4 rounded-xl border border-yellow-400/30">
                                            <p className="text-xs text-yellow-300 font-medium uppercase tracking-wide mb-1">Avg Word Len</p>
                                            <p className="text-2xl font-bold text-yellow-400">{analysis.speaking_metrics.avg_word_length}</p>
                                            <p className="text-xs text-yellow-300/70 mt-1">letters</p>
                                        </div>
                                        <div className="bg-gradient-to-br from-indigo-500/20 to-indigo-600/10 p-4 rounded-xl border border-indigo-400/30">
                                            <p className="text-xs text-indigo-300 font-medium uppercase tracking-wide mb-1">Pauses</p>
                                            <p className="text-2xl font-bold text-indigo-400">{analysis.speaking_metrics.estimated_pauses}</p>
                                            <p className="text-xs text-indigo-300/70 mt-1">detected</p>
                                        </div>
                                    </div>
                                </>
                            )}

                            {/* Found Fillers */}
                            {analysis.found_fillers && analysis.found_fillers.length > 0 && (
                                <div className="bg-white/5 p-6 rounded-xl border border-white/10">
                                    <p className="text-sm text-gray-400 font-medium mb-3">Found Fillers:</p>
                                    <div className="flex flex-wrap gap-2">
                                        {analysis.found_fillers.map((filler: string, idx: number) => (
                                            <span
                                                key={idx}
                                                className="bg-gradient-to-r from-red-500/20 to-pink-500/20 text-red-300 px-4 py-2 rounded-full text-sm font-medium border border-red-400/30"
                                            >
                                                {filler}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </Card>
                    )}
                </div>
            </div>
        </div>
    );
}
