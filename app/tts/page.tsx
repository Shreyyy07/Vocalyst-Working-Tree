// "use client";

// import { useEffect, useState } from "react";

// export default function TTSPage() {
//   const [text, setText] = useState("");
//   const [voices, setVoices] = useState<any[]>([]);

//   const [selectedVoice, setSelectedVoice] = useState("");
//   const [isLoading, setIsLoading] = useState(false);

//   const API_KEY = "sk_8f75ad145b0858db41f25cd5e00b3e986582118266900333"; // Replace with your ElevenLabs API Key

//   useEffect(() => {
//     const fetchVoices = async () => {
//       try {
//         const res = await fetch("https://api.elevenlabs.io/v1/voices", {
//           headers: {
//             "Content-Type": "application/json",
//             "xi-api-key": API_KEY,
//           },
//         });

//         if (!res.ok) {
//           console.error("Failed to fetch voices", await res.text());
//           return;
//         }

//         const data = await res.json();
//         setVoices(data.voices);
//         setSelectedVoice(data.voices[0]?.voice_id);
//       } catch (error) {
//         console.error("Error fetching voices:", error);
//       }
//     };

//     fetchVoices();
//   }, []);

//   const generateSpeech = async (text: string) => {
//   if (!selectedVoice) {
//     alert("No voice selected!");
//     return;
//   }

//   try {
//     setIsLoading(true);

//     const response = await fetch("/api/tts", {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json",
//       },
//       body: JSON.stringify({
//         text,
//         voice_id: selectedVoice,
//       }),
//     });

//     if (!response.ok) {
//       const errorText = await response.text();
//       console.error("TTS error:", errorText);
//       alert(`TTS Error: ${response.status}`);
//       setIsLoading(false);
//       return;
//     }

//     const audioBlob = await response.blob();
//     const audioUrl = URL.createObjectURL(audioBlob);
//     const audio = new Audio(audioUrl);
//     await audio.play();

//     setIsLoading(false);
//   } catch (err) {
//     console.error("Frontend error:", err);
//     alert("Something went wrong while generating speech.");
//     setIsLoading(false);
//   }
// };



//   return (
//     <div className="min-h-screen bg-gradient-to-br from-black via-zinc-900 to-black text-white flex items-center justify-center px-4 font-sans">
//       <div className="backdrop-blur-lg bg-white/5 border border-white/10 shadow-2xl rounded-2xl p-10 w-full max-w-xl transition-all duration-300">
//         <h1 className="text-4xl font-extrabold text-center mb-6 text-white drop-shadow-sm">
//           Text-to-Speech Testing
//         </h1>

//         <h2 className="text-xl font-semibold mb-4 text-white/80">Generate Speech</h2>

//         <label className="block text-sm mb-1 text-white/60">Voice Selection</label>
//         <select
//           value={selectedVoice}
//           onChange={(e) => setSelectedVoice(e.target.value)}
//           className="w-full p-2 rounded-lg bg-[#1f1f1f] text-white border border-white/20 mb-4 focus:outline-none focus:ring-2 focus:ring-white/30"
//         >
//           {voices.map((voice) => (
//             <option
//               key={voice.voice_id}
//               value={voice.voice_id}
//               className="bg-[#1f1f1f] text-white"
//             >
//               {voice.name} ({voice.labels?.accent || "Neutral"})
//             </option>
//           ))}
//         </select>

//         <label className="block text-sm mb-1 text-white/60">Speech Speed</label>
//         <input
//           type="range"
//           min="0.5"
//           max="1.5"
//           step="0.1"
//           defaultValue="1"
//           className="w-full mb-1 accent-white"
//         />
//         <p className="text-sm text-center mb-4 text-white/40">1.0x</p>

//         <label className="block text-sm mb-1 text-white/60">Text Input</label>
//         <textarea
//           className="w-full h-28 p-3 rounded-lg bg-white/10 text-white border border-white/20 mb-4 resize-none focus:outline-none focus:ring-2 focus:ring-white/30 placeholder:text-white/40"
//           placeholder="Enter text to convert to speech..."
//           value={text}
//           onChange={(e) => setText(e.target.value)}
//         ></textarea>

//         <button
//           onClick={() => generateSpeech(text)}
//           disabled={!text || isLoading}
//           className={`w-full py-2 rounded-lg font-semibold tracking-wide transition duration-300 ${
//             !text || isLoading
//               ? "bg-white/10 text-white/40 cursor-not-allowed"
//               : "bg-gradient-to-r from-white to-zinc-300 text-black hover:from-zinc-200 hover:to-white shadow-lg hover:scale-[1.01]"
//           }`}
//         >
//           {isLoading ? "Generating..." : "Generate Speech"}
//         </button>
//       </div>
//     </div>
//   );
// }
"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import FloatingPoints from "@/components/ui/FloatingPoints";

export default function TTSPage() {
  const [text, setText] = useState("");
  const [selectedVoice, setSelectedVoice] = useState("rachel");
  const [isLoading, setIsLoading] = useState(false);
  const [speed, setSpeed] = useState([1.0]);

  // ElevenLabs voices - 8 high-quality options
  const voices = [
    { voice_id: "rachel", name: "Rachel", accent: "Female, American", description: "Clear & professional" },
    { voice_id: "bella", name: "Bella", accent: "Female, Soft", description: "Calm & soothing" },
    { voice_id: "antoni", name: "Antoni", accent: "Male, Energetic", description: "Dynamic & engaging" },
    { voice_id: "josh", name: "Josh", accent: "Male, American", description: "Professional" },
    { voice_id: "charlotte", name: "Charlotte", accent: "Female, British", description: "British accent" },
    { voice_id: "nicole", name: "Nicole", accent: "Female, Australian", description: "Australian accent" },
    { voice_id: "adam", name: "Adam", accent: "Male, American", description: "Deep & authoritative" },
    { voice_id: "default", name: "Default", accent: "Female, American", description: "Rachel (default)" },
  ];

  const generateSpeech = async (e: React.FormEvent) => {
    e.preventDefault();

    console.log("Generate Speech called with text:", text, "length:", text.length);

    if (!text || text.trim().length === 0) {
      alert("Please enter some text first!");
      return;
    }

    try {
      setIsLoading(true);

      const requestBody = {
        text,
        voice: selectedVoice || "rachel",
        speed: speed[0],
      };

      console.log("Sending TTS request:", requestBody);

      const response = await fetch("http://localhost:5328/api/tts", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error("TTS error:", errorData);
        alert(`TTS Error: ${errorData.error || response.statusText}`);
        setIsLoading(false);
        return;
      }

      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);

      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
      };

      await audio.play();
    } catch (err) {
      console.error("Frontend error:", err);
      alert("Something went wrong while generating speech.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen bg-black text-white overflow-hidden">
      <div className="absolute inset-0 z-0">
        <FloatingPoints />
      </div>

      <div className="container mx-auto py-20 px-4 z-10 relative bg-transparent">
        <h1 className="text-5xl font-extrabold text-center bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent mb-10 bg-transparent">
          Text-to-Speech Lab
        </h1>

        <div className="max-w-3xl mx-auto bg-transparent border border-white/20 rounded-2xl shadow-xl p-8 space-y-8">
          <form onSubmit={generateSpeech} className="space-y-6">
            <div>
              <label className="block text-sm font-semibold mb-2 text-cyan-300">Voice Selection</label>
              <p className="text-xs text-white/60 mb-2">Choose from 8 high-quality ElevenLabs AI voices</p>
              <Select value={selectedVoice} onValueChange={setSelectedVoice}>
                <SelectTrigger className="bg-white/10 border border-white/20 hover:bg-white/15 transition">
                  <SelectValue placeholder="Select voice" />
                </SelectTrigger>
                <SelectContent className="bg-black border border-white/20">
                  {voices.map((voice) => (
                    <SelectItem key={voice.voice_id} value={voice.voice_id} className="hover:bg-white/10">
                      {voice.name} - {voice.accent}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="block text-sm font-semibold mb-2 text-purple-300">Speech Speed</label>
              <Slider
                value={speed}
                onValueChange={setSpeed}
                min={0.5}
                max={1.5}
                step={0.1}
                className="mb-2"
              />
              <p className="text-sm text-center text-purple-300 mt-1">{speed[0].toFixed(1)}x</p>
            </div>

            <div>
              <label className="block text-sm font-semibold mb-1">Enter Text</label>
              <Textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Enter something you want to hear..."
                className="h-32 bg-white/10 border border-white/20 text-white"
              />
            </div>

            <Button
              type="submit"
              className="w-full bg-gradient-to-r from-cyan-500 to-purple-600 text-white hover:opacity-90"
              disabled={isLoading || !text.trim() || !selectedVoice}
            >
              {isLoading ? "Generating..." : "Generate Speech"}
            </Button>
          </form>
        </div>
      </div>
    </div>
  );
}