"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, Mic, Square, RotateCcw } from "lucide-react";
import DetailedAnalysis from "@/components/DetailedAnalysis";
import { Analysis, RecordingAnalysis } from "@/components/SpeechFeedback";

// --- Types ---
interface PracticeMode {
  id: string;
  name: string;
  description: string;
  emoji: string;
  prompts: string[];
}

// Reuse Emotion types from CameraPage for consistency
type Emotions = {
  neutral: number;
  happy: number;
  sad: number;
  angry: number;
  fearful: number;
  disgusted: number;
  surprised: number;
};

const PRACTICE_MODES: PracticeMode[] = [
  {
    id: "persuasive",
    name: "Persuasive",
    description: "Learn to convince and influence others effectively.",
    emoji: "🎯",
    prompts: [
      "I believe that remote work is the future of productivity. It allows employees to balance their lives better while reducing the carbon footprint associated with commuting. Companies that embrace this shift will access a global talent pool and reduce overhead costs significantly.",
      "Climate change is not just an environmental issue; it is an economic one. Investing in renewable energy creates jobs, stabilizes energy prices, and ensures a sustainable future for our children. We cannot afford inaction.",
      "Education should be free for everyone. By removing financial barriers, we unlock the potential of millions of brilliant minds who can contribute to science, art, and technology. It is an investment in our collective future."
    ]
  },
  {
    id: "emotive",
    name: "Emotive",
    description: "Express feelings and emotions clearly.",
    emoji: "💝",
    prompts: [
      "I was so incredibly happy when I heard the news! It felt like a weight had been lifted off my shoulders, and suddenly, everything seemed possible again. I just wanted to dance!",
      "It breaks my heart to see so many people struggling. I feel a deep sense of responsibility to help, to do something, anything, to make a difference in their lives.",
      "I was terrified. The darkness seemed to close in around me, and every sound made me jump. I've never felt so alone and vulnerable in my entire life."
    ]
  },
  {
    id: "public-speaking",
    name: "Public Speaking",
    description: "Master speaking in front of audiences.",
    emoji: "🎤",
    prompts: [
      "Good evening everyone. Today, I want to talk to you about the power of community. When we come together, we are stronger, more resilient, and more capable of overcoming any challenge.",
      "Success is not measured by wealth, but by the impact we have on others. As we move forward in our careers, let us not forget to lift others up as we climb.",
      "Thank you for being here. Your presence demonstrates a commitment to change. Together, we can build a world that is more just, equitable, and sustainable for all."
    ]
  },
  {
    id: "formal",
    name: "Formal Conversation",
    description: "Professional and business communication.",
    emoji: "👔",
    prompts: [
      "I appreciate you taking the time to meet with me today. I would like to discuss the quarterly results and propose a new strategy for the upcoming fiscal year.",
      "Regarding the project timeline, we are currently on schedule. However, I recommend allocating additional resources to the QA phase to ensure the highest quality deliverables.",
      "It has been a pleasure doing business with you. We look forward to a continued and mutually beneficial partnership in the years to come."
    ]
  },
  {
    id: "storytelling",
    name: "Storytelling",
    description: "Engaging narrative communication.",
    emoji: "📚",
    prompts: [
      "The old house stood on the hill, silent and imposing. Its windows were like dark eyes watching the village below. Legend had it that no one who entered ever returned.",
      "It was a day like any other, until the letter arrived. The handwriting was familiar, yet I hadn't seen it in twenty years. My hands trembled as I tore open the envelope.",
      "Once upon a time, in a land far away, there lived a young girl with a secret. She could speak to the stars, and they would whisper the secrets of the universe back to her."
    ]
  },
  {
    id: "debate",
    name: "Debating",
    description: "Structured argument and discussion.",
    emoji: "⚖️",
    prompts: [
      "While I understand your point about economic growth, we must prioritize environmental protection. Short-term gains are meaningless if we destroy the planet in the process.",
      "Social media has connected the world, but it has also created an echo chamber. We are losing the ability to have civil discourse with those who disagree with us.",
      "Artificial Intelligence offers immense benefits, but we must implement strict regulations to prevent misuse and ensure it aligns with human values."
    ]
  }
];

export default function PracticePage() {
  // --- State ---
  const [selectedMode, setSelectedMode] = useState<PracticeMode | null>(null);
  const [currentPromptIndex, setCurrentPromptIndex] = useState(0);
  const [view, setView] = useState<'selection' | 'practice' | 'results'>('selection');

  // Recording State
  const [isStreaming, setIsStreaming] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [uploadedVideo, setUploadedVideo] = useState<string | null>(null);
  const [uploadedAudio, setUploadedAudio] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // Analysis State
  const [transcription, setTranscription] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [recordingAnalysis, setRecordingAnalysis] = useState<RecordingAnalysis | null>(null);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [enhancedAudio, setEnhancedAudio] = useState<string | null>(null);

  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const recordingTimerRef = useRef<NodeJS.Timeout | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Data refs for recording accumulation
  const emotionsDataRef = useRef<Array<{ timestamp: number; emotions: Emotions }>>([]);
  const gazeDataRef = useRef<Array<{ timestamp: number; direction: string }>>([]);

  // --- Persistence ---
  useEffect(() => {
    // Only restore if we are structurally mounted and not already holding state
    if (selectedMode) return;

    const savedState = localStorage.getItem("vocalyst-practice-state");
    if (savedState) {
      try {
        const parsed = JSON.parse(savedState);
        if (parsed.modeId && parsed.view) {
          const mode = PRACTICE_MODES.find(m => m.id === parsed.modeId);
          if (mode) {
            console.log("Restoring practice state:", parsed);
            setSelectedMode(mode);
            setCurrentPromptIndex(parsed.promptIndex || 0);
            setView(parsed.view === 'results' ? 'results' : 'practice');

            // Restore data if available
            if (parsed.transcription) setTranscription(parsed.transcription);
            if (parsed.analysis) setAnalysis(parsed.analysis);
            if (parsed.recordingAnalysis) setRecordingAnalysis(parsed.recordingAnalysis);
            if (parsed.uploadedVideo) setUploadedVideo(parsed.uploadedVideo);
            if (parsed.uploadedAudio) setUploadedAudio(parsed.uploadedAudio);

            if (parsed.view === 'practice') {
              setTimeout(() => startCamera(), 500);
            }
          }
        }
      } catch (e) {
        console.error("Failed to restore state", e);
      }
    }
  }, []); // Run ONCE on mount

  useEffect(() => {
    if (selectedMode) {
      const stateToSave = {
        modeId: selectedMode.id,
        promptIndex: currentPromptIndex,
        view: view,
        transcription,
        analysis,
        recordingAnalysis,
        uploadedVideo,
        uploadedAudio
      };
      localStorage.setItem("vocalyst-practice-state", JSON.stringify(stateToSave));
    }
  }, [selectedMode, currentPromptIndex, view, transcription, analysis, recordingAnalysis, uploadedVideo, uploadedAudio]);

  // --- Actions ---

  const handleModeSelect = (mode: PracticeMode) => {
    setSelectedMode(mode);
    setCurrentPromptIndex(0);
    setView('practice');
    startCamera();
  };

  const handleBack = () => {
    stopCamera();
    setView('selection');
    setSelectedMode(null);
    resetState();
    localStorage.removeItem("vocalyst-practice-state");
  };

  const resetState = () => {
    setIsRecording(false);
    setIsUploading(false);
    setRecordingDuration(0);
    setUploadedVideo(null);
    setUploadedAudio(null);
    setTranscription(null);
    setAnalysis(null);
    setRecordingAnalysis(null);
    setEnhancedAudio(null);
    emotionsDataRef.current = [];
    gazeDataRef.current = [];
  };

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  // --- Detection Logic (Copied from CameraPage for consistency) ---
  const detectCombined = async () => {
    if (!videoRef.current || !canvasRef.current || isProcessing) return;

    try {
      setIsProcessing(true);
      const canvas = document.createElement("canvas"); // Use offscreen or ref
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.drawImage(videoRef.current, 0, 0);
      const imageData = canvas.toDataURL("image/jpeg", 0.8);

      const response = await fetch("http://localhost:5328/api/detect-combined", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData }),
      });

      if (!response.ok) return; // Silent fail

      const result = await response.json();

      if (result.success && result.face_detected) {
        if (isRecording) {
          const timestamp = Date.now();
          emotionsDataRef.current.push({ timestamp, emotions: result.emotions });
          if (result.gaze) {
            gazeDataRef.current.push({ timestamp, direction: result.gaze.direction });
          }
        }
      }
    } catch (error) {
      // console.error(error); 
    } finally {
      setIsProcessing(false);
    }
  };

  const processLocalAnalysisData = (duration: number) => {
    // 1. Emotions
    const emotionSums: Record<string, number> = {};
    const emotionCounts: Record<string, number> = {};

    emotionsDataRef.current.forEach(({ emotions }) => {
      Object.entries(emotions).forEach(([e, v]) => {
        emotionSums[e] = (emotionSums[e] || 0) + v;
        emotionCounts[e] = (emotionCounts[e] || 0) + 1;
      });
    });

    const averageEmotions = Object.entries(emotionSums)
      .map(([emotion, sum]) => ({
        emotion,
        average: sum / emotionCounts[emotion]
      }))
      .sort((a, b) => b.average - a.average)
      .slice(0, 3)
      .map(({ emotion, average }) => ({
        emotion,
        percentage: (average * 100).toFixed(1)
      }));

    // Fallback if no emotions detected
    const finalEmotions = averageEmotions.length > 0
      ? averageEmotions
      : [{ emotion: 'neutral', percentage: '100.0' }];

    // 2. Gaze
    const directionCounts: Record<string, number> = {};
    gazeDataRef.current.forEach(({ direction }) => {
      directionCounts[direction] = (directionCounts[direction] || 0) + 1;
    });
    const sortedGaze = Object.entries(directionCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3)
      .map(([direction, count]) => ({
        direction,
        percentage: ((count / Math.max(1, gazeDataRef.current.length)) * 100).toFixed(1)
      }));

    const finalGaze = sortedGaze.length > 0
      ? sortedGaze
      : [{ direction: 'center', percentage: '100.0' }];

    setRecordingAnalysis({
      emotions: finalEmotions as any,
      gaze: finalGaze as any,
      duration: duration
    });
  };

  // --- Camera & Recording Logic ---
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, sampleRate: 44100 }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      streamRef.current = stream;
      setIsStreaming(true);

      // Start background detection loop
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = setInterval(detectCombined, 100);

    } catch (error) {
      console.error("Camera error:", error);
      alert("Could not start camera. Please allow permissions.");
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      if (videoRef.current) videoRef.current.srcObject = null;
      streamRef.current = null;
    }
    if (intervalRef.current) clearInterval(intervalRef.current);
    setIsStreaming(false);
  };

  const startRecording = () => {
    if (!streamRef.current) return;

    emotionsDataRef.current = [];
    gazeDataRef.current = [];
    chunksRef.current = [];

    // Increase frequency during recording
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(detectCombined, 50);

    const mediaRecorder = new MediaRecorder(streamRef.current);
    mediaRecorderRef.current = mediaRecorder;

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    mediaRecorder.onstop = handleRecordingStop;
    mediaRecorder.start(100);
    setIsRecording(true);
    setRecordingDuration(0);
    recordingTimerRef.current = setInterval(() => setRecordingDuration(p => p + 1), 1000);
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);

      // Reset frequency
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = setInterval(detectCombined, 100);
    }
  };

  const handleRecordingStop = async () => {
    const videoBlob = new Blob(chunksRef.current, { type: 'video/mp4' });
    const formData = new FormData();
    formData.append("video", videoBlob, "practice_recording.mp4");

    setIsUploading(true);
    try {
      const response = await fetch("http://localhost:5328/api/upload-video", {
        method: "POST",
        body: formData
      });
      const result = await response.json();

      setUploadedVideo(result.filename);
      if (result.has_audio) {
        setUploadedAudio(result.audio_filename);

        // Process collected LOCAL data for emotions (fixing the "not reflecting" issue)
        processLocalAnalysisData(recordingDuration);

        setView('results');
        processAnalysis(result.audio_filename);
      } else {
        alert("No audio detected in recording.");
      }
    } catch (error) {
      console.error("Upload failed:", error);
      alert("Failed to upload recording.");
    } finally {
      setIsUploading(false);
    }
  };

  const processAnalysis = async (audioVarsFilename: string) => {
    setIsTranscribing(true);
    try {
      // Transcription
      const audioRes = await fetch(`http://localhost:5328/uploads/${audioVarsFilename}`);
      const audioBlob = await audioRes.blob();
      const formData = new FormData();
      formData.append("file", audioBlob, audioVarsFilename);
      if (selectedMode) formData.append("category", selectedMode.id);

      const txResponse = await fetch("http://localhost:5328/api/speech2text", {
        method: "POST", body: formData
      });
      const txResult = await txResponse.json();

      setTranscription(txResult.text || "No speech detected.");
      setAnalysis(txResult.analysis);

      // We do NOT overwrite recordingAnalysis here anymore, 
      // because processLocalAnalysisData already set the definitive emotions/gaze.
      // logic.

      // Enhancement
      setIsEnhancing(true);
      const enResponse = await fetch("http://localhost:5328/api/enhance-audio", {
        method: "POST", body: formData
      });

      if (!enResponse.ok) {
        throw new Error("Enhancement failed");
      }

      const enBlob = await enResponse.blob();
      const outputAudioBlob = new Blob([enBlob], { type: 'audio/wav' }); // Fix playability
      setEnhancedAudio(URL.createObjectURL(outputAudioBlob));

    } catch (e) {
      console.error("Analysis failed:", e);
    } finally {
      setIsTranscribing(false);
      setIsEnhancing(false);
    }
  };

  const handleTryAgain = () => {
    resetState();
    setView('practice');
    startCamera();
  };

  const nextPrompt = () => {
    if (selectedMode) {
      setCurrentPromptIndex((prev) => (prev + 1) % selectedMode.prompts.length);
    }
  };

  // --- Render ---

  // 1. Selection Screen
  if (view === 'selection') {
    return (
      <>
        <div className="container min-h-screen pt-24 pb-12 flex flex-col items-center">
          <h1 className="text-4xl font-bold mb-2 text-white">Choose Your Practice Mode</h1>
          <p className="text-muted-foreground mb-8">Select a category to practice specific speaking skills.</p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 w-full max-w-6xl">
            {PRACTICE_MODES.map((mode) => (
              <Card
                key={mode.id}
                className="hover:border-primary/50 transition-all cursor-pointer bg-card/50 backdrop-blur"
                onClick={() => handleModeSelect(mode)}
              >
                <CardHeader>
                  <div className="text-4xl mb-4">{mode.emoji}</div>
                  <CardTitle>{mode.name}</CardTitle>
                  <CardDescription>{mode.description}</CardDescription>
                </CardHeader>
              </Card>
            ))}
          </div>
        </div>
      </>
    );
  }

  // 2. Practice Screen (Prompt + Camera)
  if (view === 'practice' && selectedMode) {
    return (
      <>
        <div className="container min-h-screen pt-24 pb-12 flex flex-col items-center">
          <div className="w-full max-w-5xl flex justify-between items-center mb-6">
            <Button variant="ghost" onClick={handleBack} disabled={isRecording}>
              <ArrowLeft className="mr-2 h-4 w-4" /> Back
            </Button>
            <Badge variant="outline" className="text-lg px-4 py-1">
              {selectedMode.emoji} {selectedMode.name}
            </Badge>
            <div className="w-20" /> {/* Spacer */}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 w-full max-w-6xl flex-1 min-h-0">
            {/* Left: Prompt Card */}
            <Card className="flex flex-col bg-card/80 backdrop-blur border-primary/20">
              <CardHeader>
                <CardTitle>Read Aloud</CardTitle>
                <CardDescription>Focus on your tone and pacing.</CardDescription>
              </CardHeader>
              <CardContent className="flex-1 flex items-center justify-center p-8">
                <p className="text-2xl leading-relaxed font-serif text-center">
                  "{selectedMode.prompts[currentPromptIndex]}"
                </p>
              </CardContent>
              <CardFooter className="justify-center gap-4">
                <Button variant="ghost" size="sm" onClick={nextPrompt} disabled={isRecording}>
                  Different Prompt
                </Button>
              </CardFooter>
            </Card>

            {/* Right: Camera & Controls */}
            <div className="flex flex-col gap-4 justify-center">
              <div className="relative aspect-video bg-black rounded-xl overflow-hidden border border-white/20 shadow-2xl">
                <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />

                {/* Hidden canvas for offscreen analysis */}
                <canvas ref={canvasRef} className="hidden" />

                {isRecording && (
                  <div className="absolute top-4 right-4 bg-red-600 text-white px-3 py-1 rounded-full animate-pulse font-bold">
                    Recording {formatDuration(recordingDuration)}
                  </div>
                )}
              </div>

              <div className="flex justify-center gap-4 py-4">
                {!isRecording ? (
                  <Button onClick={startRecording} size="lg" className="w-full max-w-xs bg-red-600 hover:bg-red-700 text-lg h-14 rounded-full">
                    <Mic className="mr-2 h-6 w-6" /> Start Practice
                  </Button>
                ) : (
                  <Button onClick={stopRecording} size="lg" variant="destructive" className="w-full max-w-xs text-lg h-14 rounded-full animate-pulse">
                    <Square className="mr-2 h-6 w-6" /> Stop & Analyze
                  </Button>
                )}
              </div>
            </div>
          </div>
        </div>
      </>
    );
  }

  // 3. Results Screen
  if (view === 'results') {
    return (
      <>
        <div className="container min-h-screen pt-24 pb-12 flex flex-col items-center">
          <Button variant="ghost" className="self-start mb-4" onClick={handleBack}>
            <ArrowLeft className="mr-2 h-4 w-4" /> Back to Modes
          </Button>

          <h1 className="text-3xl font-bold mb-8">Performance Analysis</h1>

          <div className="flex flex-col gap-6 w-full max-w-4xl">
            {isTranscribing ? (
              <Card className="p-12 flex flex-col items-center gap-4">
                <div className="text-4xl animate-spin">⏳</div>
                <h2 className="text-xl font-semibold">Analyzing your speech...</h2>
                <p className="text-muted-foreground">Checking emotions, pacing, and clarity.</p>
              </Card>
            ) : (
              transcription && (analysis || recordingAnalysis) ? (
                <Card className="bg-black/40 border border-white/10 backdrop-blur-md">
                  <CardContent className="pt-6">
                    {enhancedAudio && (
                      <div className="mb-6 flex flex-col items-center gap-2">
                        <h3 className="text-sm font-medium text-gray-400">Enhanced Audio</h3>
                        <audio controls src={enhancedAudio} className="w-full max-w-md" />
                      </div>
                    )}

                    <DetailedAnalysis
                      analysis={analysis}
                      recordingAnalysis={recordingAnalysis}
                      transcription={transcription}
                    />

                    <div className="flex justify-center mt-8">
                      <Button onClick={handleTryAgain} variant="outline" size="lg">
                        <RotateCcw className="mr-2 h-4 w-4" /> Practice Again
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <div className="text-center text-red-400">Result loading failed.</div>
              )
            )}
          </div>
        </div>
      </>
    );
  }

  return null;
}
