"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

import DetailedAnalysis from "@/components/DetailedAnalysis";
import SpeechFeedback, { Analysis, RecordingAnalysis, EmotionAnalysis } from "@/components/SpeechFeedback";

type Emotions = {
  neutral: number;
  happy: number;
  sad: number;
  angry: number;
  fearful: number;
  disgusted: number;
  surprised: number;
};

export default function CameraPage() {
  // --- State: Camera & Recording ---
  const [isStreaming, setIsStreaming] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [recordingError, setRecordingError] = useState<string | null>(null);
  const [uploadedVideo, setUploadedVideo] = useState<string | null>(null);
  const [uploadedAudio, setUploadedAudio] = useState<string | null>(null);
  const [recordingDuration, setRecordingDuration] = useState<number>(0);

  // Real-time data containers
  const [emotions, setEmotions] = useState<Emotions>({
    neutral: 0, happy: 0, sad: 0, angry: 0, fearful: 0, disgusted: 0, surprised: 0,
  });
  const [gazeDirection, setGazeDirection] = useState<string>("center");

  // Data refs for recording accumulation
  const emotionsDataRef = useRef<Array<{ timestamp: number; emotions: Emotions }>>([]);
  const gazeDataRef = useRef<Array<{ timestamp: number; direction: string }>>([]);

  // Hardware refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const recordingTimerRef = useRef<NodeJS.Timeout | null>(null);

  // --- State: Analysis & Results ---
  const [showResults, setShowResults] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcription, setTranscription] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [recordingAnalysis, setRecordingAnalysis] = useState<RecordingAnalysis | null>(null);
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [enhancedAudio, setEnhancedAudio] = useState<string | null>(null);


  // --- Helper Functions ---
  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  const getDominantEmotion = (emotions: Record<string, number>): string => {
    if (Object.keys(emotions).length === 0) return "none";
    let maxEmotion = "";
    let maxValue = 0;
    Object.entries(emotions).forEach(([emotion, value]) => {
      if (value > maxValue) {
        maxValue = value;
        maxEmotion = emotion;
      }
    });
    return maxEmotion;
  };

  // --- Core Camera Logic ---
  const testApiConnection = async () => {
    try {
      const response = await fetch("http://localhost:5328/api/test", { headers: { Accept: "application/json" } });
      const data = await response.json();
      return true;
    } catch (error) {
      console.error("API connection test failed:", error);
      setBackendError("Cannot connect to Python server. Make sure it's running on port 5328.");
      return false;
    }
  };

  const detectCombined = async () => {
    if (!videoRef.current || !canvasRef.current || isProcessing) return;

    try {
      setIsProcessing(true);
      setBackendError(null);

      const canvas = document.createElement("canvas");
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

      if (!response.ok) throw new Error("Failed to process frame");

      // Check refs again after async
      if (!videoRef.current || !canvasRef.current) return;

      const result = await response.json();

      if (result.success && result.face_detected) {
        setEmotions(result.emotions);

        if (isRecording) {
          const timestamp = Date.now();
          emotionsDataRef.current.push({ timestamp, emotions: result.emotions });
          if (result.gaze) {
            gazeDataRef.current.push({ timestamp, direction: result.gaze.direction });
          }
        }
      }

      if (result.gaze) {
        setGazeDirection(result.gaze.direction);
        drawOverlays(result);
      }

    } catch (error) {
      console.error("Detection error:", error);
      // Suppress error in UI for smoother experience, just log it
    } finally {
      setIsProcessing(false);
    }
  };

  const drawOverlays = (result: any) => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    if (isRecording) {
      // Clear canvas during recording to keep video clean
      const ctx = canvas.getContext("2d");
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const displayRect = video.getBoundingClientRect();
    if (canvas.width !== displayRect.width || canvas.height !== displayRect.height) {
      canvas.width = displayRect.width;
      canvas.height = displayRect.height;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate scaling
    const scale = Math.min(displayRect.width / video.videoWidth, displayRect.height / video.videoHeight);
    const offsetX = (displayRect.width - video.videoWidth * scale) / 2;
    const offsetY = (displayRect.height - video.videoHeight * scale) / 2;

    ctx.save();
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);

    // Draw Gaze Arrow
    if (result.gaze?.gaze_arrow) {
      const { start, end } = result.gaze.gaze_arrow;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(start.x * video.videoWidth, start.y * video.videoHeight);
      ctx.lineTo(end.x * video.videoWidth, end.y * video.videoHeight);
      ctx.stroke();
    }

    // Draw Gaze Text
    ctx.font = "20px sans-serif";
    ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
    if (result.gaze?.direction) {
      ctx.fillText(result.gaze.direction.toUpperCase(), 20, 40);
    }

    ctx.restore();
  };

  const startCamera = async () => {
    try {
      const isConnected = await testApiConnection();
      if (!isConnected) throw new Error("Backend not reachable");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, sampleRate: 44100 },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await new Promise((resolve) => {
          if (videoRef.current) {
            videoRef.current.onloadedmetadata = () => {
              videoRef.current?.play();
              resolve(true);
            };
          }
        });
      }

      streamRef.current = stream;
      setIsStreaming(true);
      intervalRef.current = setInterval(detectCombined, 100);

    } catch (error) {
      console.error("Camera start error:", error);
      alert("Failed to access camera/mic or backend.");
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      if (videoRef.current) videoRef.current.srcObject = null;
      streamRef.current = null;
    }
    if (intervalRef.current) clearInterval(intervalRef.current);
    setIsStreaming(false);
  };

  // --- Recording Logic ---
  const startRecording = async () => {
    if (!streamRef.current) return;
    setRecordingError(null);
    setRecordingDuration(0);
    emotionsDataRef.current = [];
    gazeDataRef.current = [];

    // Switch to results view? No, stay on camera view until finished.
    setShowResults(false);
    setTranscription(null);
    setAnalysis(null);
    setRecordingAnalysis(null);
    setEnhancedAudio(null);

    // Increase detection frequency for better data
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(detectCombined, 50);

    try {
      const options = { mimeType: 'video/mp4' }; // Fallback handled by browser usually, or specific check
      const mediaRecorder = new MediaRecorder(streamRef.current);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = handleRecordingStop;

      mediaRecorder.start(100);
      setIsRecording(true);
      recordingTimerRef.current = setInterval(() => setRecordingDuration(prev => prev + 1), 1000);

    } catch (e) {
      console.error("Start recording failed:", e);
      setRecordingError("Failed to start recording.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleRecordingStop = async () => {
    if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);

    // Prepare blob
    const videoBlob = new Blob(chunksRef.current, { type: 'video/mp4' });
    const formData = new FormData();
    formData.append("video", videoBlob, "recording.mp4");

    setIsUploading(true);
    try {
      // Upload
      const response = await fetch("http://localhost:5328/api/upload-video", {
        method: "POST",
        body: formData
      });
      if (!response.ok) throw new Error("Upload failed");

      const result = await response.json();
      setUploadedVideo(result.filename);
      if (result.has_audio) setUploadedAudio(result.audio_filename);

      // Process local analysis data immediately
      processLocalAnalysisData(recordingDuration);

      // Switch to results view
      setShowResults(true);

      // Trigger backend analysis
      if (result.has_audio) {
        getTranscription(result.audio_filename);
        enhanceAudio(result.audio_filename);
      } else {
        setTranscription("No audio detected.");
      }

    } catch (error) {
      console.error("Upload error:", error);
      setRecordingError("Failed to process recording.");
    } finally {
      setIsUploading(false);
      // Reset detection interval to normal
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = setInterval(detectCombined, 100);
    }
  };

  const processLocalAnalysisData = (duration: number) => {
    // Calculate top emotions locally from the ref data gathered during recording
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

    // Calculate gaze
    const directionCounts: Record<string, number> = {};
    gazeDataRef.current.forEach(({ direction }) => {
      directionCounts[direction] = (directionCounts[direction] || 0) + 1;
    });
    const sortedGaze = Object.entries(directionCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3)
      .map(([direction, count]) => ({
        direction,
        percentage: ((count / gazeDataRef.current.length) * 100).toFixed(1)
      }));

    setRecordingAnalysis({
      emotions: averageEmotions,
      gaze: sortedGaze,
      duration: duration
    });
  };

  // --- Backend Analysis Functions (Ported from recordings page) ---
  const getTranscription = async (audioVarsFilename: string) => {
    setIsTranscribing(true);
    try {
      // Fetch blob first
      const audioRes = await fetch(`http://localhost:5328/uploads/${audioVarsFilename}`);
      const audioBlob = await audioRes.blob();

      const formData = new FormData();
      formData.append("file", audioBlob, audioVarsFilename);

      const response = await fetch("http://localhost:5328/api/speech2text", {
        method: "POST",
        body: formData
      });
      const result = await response.json();

      if (result.no_speech_detected) {
        setTranscription("No speech detected.");
      } else {
        setTranscription(result.text || "No transcription available.");
        setAnalysis(result.analysis);
      }

    } catch (e) {
      console.error("Transcription error:", e);
      setTranscription("Failed to transcribe.");
    } finally {
      setIsTranscribing(false);
    }
  };

  const enhanceAudio = async (audioVarsFilename: string) => {
    setIsEnhancing(true);
    try {
      const audioRes = await fetch(`http://localhost:5328/uploads/${audioVarsFilename}`);
      const audioBlob = await audioRes.blob();

      const formData = new FormData();
      formData.append("file", audioBlob, audioVarsFilename);

      const response = await fetch("http://localhost:5328/api/enhance-audio", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error("Enhancement failed");
      }

      const enhancedBlob = await response.blob();
      // Ensure the blob is audio
      const outputAudioBlob = new Blob([enhancedBlob], { type: 'audio/wav' });
      const enhancedUrl = URL.createObjectURL(outputAudioBlob);
      setEnhancedAudio(enhancedUrl);

    } catch (e) {
      console.error("Enhancement error:", e);
    } finally {
      setIsEnhancing(false);
    }
  };


  // --- Helper to reset view ---
  const handleRecordAgain = () => {
    setShowResults(false);
    setRecordingAnalysis(null);
    setAnalysis(null);
    setTranscription(null);
    setEnhancedAudio(null);
    // Camera is likely still running if they didn't navigate away
  };

  // --- Render ---
  return (
    <>
      <div className="fixed inset-0 -z-10 bg-black">
      </div>

      <main className="relative z-10 container py-8 flex flex-col items-center min-h-screen">
        <h1 className="text-4xl font-bold mb-8 text-center text-white">
          {showResults ? "Practice Analysis" : "Camera Practice"}
        </h1>

        <Card className="w-full max-w-4xl bg-card/90 backdrop-blur-sm border-white/10">
          <CardHeader>
            <CardTitle className="text-center">
              {showResults ? "Your Results" : "Live Feed"}
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-6">

            {/* View 1: Camera & Recording Controls */}
            {!showResults && (
              <div className="flex flex-col items-center gap-4">
                <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden border border-white/20 shadow-2xl">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="absolute inset-0 w-full h-full object-contain"
                  />
                  <canvas
                    ref={canvasRef}
                    className="absolute inset-0 w-full h-full pointer-events-none"
                  />
                  {isRecording && (
                    <div className="absolute top-4 right-4 bg-red-600 text-white px-4 py-1.5 rounded-full flex items-center gap-2 shadow-lg animate-pulse">
                      <div className="w-2.5 h-2.5 bg-white rounded-full" />
                      <span className="font-mono font-bold">{formatDuration(recordingDuration)}</span>
                    </div>
                  )}
                </div>

                <div className="flex gap-4 mt-2">
                  {!isStreaming ? (
                    <Button onClick={startCamera} size="lg" className="w-40 font-bold">
                      Turn On Camera
                    </Button>
                  ) : (
                    <>
                      {!isRecording ? (
                        <Button onClick={startRecording} variant="default" size="lg" className="w-40 bg-red-600 hover:bg-red-700 font-bold">
                          Start Recording
                        </Button>
                      ) : (
                        <Button onClick={stopRecording} variant="destructive" size="lg" className="w-40 font-bold animate-pulse">
                          Stop Recording
                        </Button>
                      )}
                    </>
                  )}
                </div>

                {/* Real-time Feedback (Only visible when streaming, not recording) */}
                {isStreaming && !isRecording && (
                  <div className="grid grid-cols-2 gap-4 w-full mt-4">
                    <div className="bg-muted/50 p-4 rounded-xl border border-white/5">
                      <h3 className="text-sm font-semibold mb-2 text-muted-foreground uppercase tracking-wider">Dominant Emotion</h3>
                      <p className="text-2xl font-bold capitalize text-primary">{getDominantEmotion(emotions)}</p>
                    </div>
                    <div className="bg-muted/50 p-4 rounded-xl border border-white/5">
                      <h3 className="text-sm font-semibold mb-2 text-muted-foreground uppercase tracking-wider">Gaze Direction</h3>
                      <p className="text-2xl font-bold capitalize text-primary">{gazeDirection}</p>
                    </div>
                  </div>
                )}

                {isUploading && (
                  <div className="text-primary font-semibold animate-pulse">
                    Processing Recording...
                  </div>
                )}
              </div>
            )}


            {/* View 2: Analysis Results */}
            {showResults && (
              <div className="flex flex-col gap-8 w-full">
                {/* Top Section: Video & Summary */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Recorded Video Player */}
                  <div className="space-y-2">
                    <h3 className="font-semibold text-lg ml-1">Your Recording</h3>
                    <div className="aspect-video bg-black rounded-lg overflow-hidden border border-white/10">
                      {uploadedVideo && (
                        <video
                          src={`http://localhost:5328/uploads/${uploadedVideo}`}
                          controls
                          className="w-full h-full object-contain"
                        />
                      )}
                    </div>
                  </div>


                </div>

                {/* Detailed Analysis Component */}
                {transcription && (analysis || recordingAnalysis) && uploadedAudio && (
                  <Card className="bg-black/40 border border-white/10 backdrop-blur-md mt-4">
                    <CardContent className="pt-6">
                      {/* Audio player removed as per user request */}
                      <DetailedAnalysis
                        analysis={analysis}
                        recordingAnalysis={recordingAnalysis}
                        transcription={transcription}
                      />
                    </CardContent>
                  </Card>
                )}

                {/* Action Buttons */}
                <div className="flex justify-center pt-6">
                  <Button onClick={handleRecordAgain} size="lg" variant="outline" className="px-8">
                    Record Again
                  </Button>
                </div>
              </div>
            )}

            {/* Error Display */}
            {(backendError || recordingError) && (
              <div className="bg-destructive/10 text-destructive p-4 rounded-lg text-center border border-destructive/20">
                {backendError || recordingError}
              </div>
            )}

          </CardContent>
        </Card>
      </main>
    </>
  );
}
