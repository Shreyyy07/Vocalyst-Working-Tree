"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { TrendingUp, AlertCircle, Zap, Activity, MessageSquare, Mic, User } from "lucide-react";

// Mock data to simulate AI analysis
const mockAnalysis = {
  archetype: {
    title: "The Diplomat",
    description: "You speak with empathy and clarity, making you excellent at resolving conflicts and building rapport.",
    traits: ["Empathetic", "Clear", "Balanced"],
    icon: "🤝"
  },
  metrics: {
    clarity: 85,
    wpm: 145,
    fillerWords: 2.1, // percentage
    eyeContact: 78 // percentage
  },
  strengths: [
    "Excellent tonal variety keeps listeners engaged",
    "Strong vocabulary usage in formal contexts",
    "Great pacing - not too fast, not too slow"
  ],
  weaknesses: [
    "Tendency to look away during complex explanations",
    "Occasional use of 'um' and 'like' as fillers",
    "Volume drops slightly at the end of sentences"
  ],
  recentSessions: [
    { id: 1, date: "Today", mode: "Public Speaking", score: 92 },
    { id: 2, date: "Yesterday", mode: "Persuasive", score: 88 },
    { id: 3, date: "2 days ago", mode: "Formal", score: 75 },
  ]
};

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6 } }
};

const stagger = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 }
  }
};

export default function GetInsightsPage() {
  return (
    <div className="min-h-screen bg-black text-white pt-24 pb-12 px-6">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-900/20 via-black to-black -z-10" />

      <div className="max-w-6xl mx-auto space-y-10">
        {/* Header */}
        <motion.div
          initial="hidden"
          animate="visible"
          variants={fadeIn}
          className="text-center space-y-4"
        >
          <h1 className="text-5xl font-extrabold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            Your Communication Profile
          </h1>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            AI-powered analysis of your speaking style, strengths, and areas for growth.
          </p>
        </motion.div>

        {/* Archetype Card */}
        <motion.div
          initial="hidden"
          animate="visible"
          variants={fadeIn}
        >
          <Card className="bg-gradient-to-br from-gray-900 to-black border-2 border-purple-500/30 overflow-hidden relative">
            <div className="absolute top-0 right-0 p-8 opacity-10 text-9xl">
              {mockAnalysis.archetype.icon}
            </div>
            <CardContent className="p-10 flex flex-col md:flex-row items-center gap-10 relative z-10">
              <div className="h-40 w-40 rounded-full bg-gradient-to-tr from-blue-500 to-purple-600 p-[3px] shadow-[0_0_50px_rgba(168,85,247,0.4)]">
                <div className="h-full w-full rounded-full bg-black flex items-center justify-center text-6xl">
                  {mockAnalysis.archetype.icon}
                </div>
              </div>
              <div className="space-y-4 flex-1 text-center md:text-left">
                <div className="space-y-2">
                  <h2 className="text-sm font-medium text-purple-400 uppercase tracking-widest">Communication Archetype</h2>
                  <h3 className="text-4xl font-bold text-white">{mockAnalysis.archetype.title}</h3>
                </div>
                <p className="text-lg text-gray-300 leading-relaxed">
                  {mockAnalysis.archetype.description}
                </p>
                <div className="flex flex-wrap justify-center md:justify-start gap-3 pt-2">
                  {mockAnalysis.archetype.traits.map(trait => (
                    <Badge key={trait} variant="secondary" className="px-4 py-1.5 text-sm bg-white/10 hover:bg-white/20 border-0">
                      {trait}
                    </Badge>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Metrics Grid */}
        <motion.div
          variants={stagger}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        >
          {/* Clarity */}
          <motion.div variants={fadeIn}>
            <Card className="bg-gray-900/50 border-white/10 h-full hover:bg-gray-900/80 transition-colors">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">Clarity Score</CardTitle>
                <Zap className="h-4 w-4 text-yellow-400" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-white mb-2">{mockAnalysis.metrics.clarity}%</div>
                <Progress value={mockAnalysis.metrics.clarity} className="h-2 bg-gray-800" indicatorClassName="bg-yellow-400" />
                <p className="text-xs text-gray-500 mt-2">Top 15% of users</p>
              </CardContent>
            </Card>
          </motion.div>

          {/* WPM */}
          <motion.div variants={fadeIn}>
            <Card className="bg-gray-900/50 border-white/10 h-full hover:bg-gray-900/80 transition-colors">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">Pace (WPM)</CardTitle>
                <Activity className="h-4 w-4 text-blue-400" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-white mb-2">{mockAnalysis.metrics.wpm}</div>
                <div className="flex items-center gap-2">
                  <div className="h-2 w-full bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-400 w-3/4" />
                  </div>
                </div>
                <p className="text-xs text-gray-500 mt-2">Ideal range: 130-150</p>
              </CardContent>
            </Card>
          </motion.div>

          {/* Filler Words */}
          <motion.div variants={fadeIn}>
            <Card className="bg-gray-900/50 border-white/10 h-full hover:bg-gray-900/80 transition-colors">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">Filler Words</CardTitle>
                <MessageSquare className="h-4 w-4 text-red-400" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-white mb-2">{mockAnalysis.metrics.fillerWords}%</div>
                <Progress value={mockAnalysis.metrics.fillerWords * 10} className="h-2 bg-gray-800" indicatorClassName="bg-red-400" />
                <p className="text-xs text-gray-500 mt-2">Decreased by 0.5% this week</p>
              </CardContent>
            </Card>
          </motion.div>

          {/* Eye Contact */}
          <motion.div variants={fadeIn}>
            <Card className="bg-gray-900/50 border-white/10 h-full hover:bg-gray-900/80 transition-colors">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">Eye Contact</CardTitle>
                <User className="h-4 w-4 text-green-400" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-white mb-2">{mockAnalysis.metrics.eyeContact}%</div>
                <Progress value={mockAnalysis.metrics.eyeContact} className="h-2 bg-gray-800" indicatorClassName="bg-green-400" />
                <p className="text-xs text-gray-500 mt-2">Great engagement!</p>
              </CardContent>
            </Card>
          </motion.div>
        </motion.div>

        {/* Breakdown Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Strengths */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="space-y-4"
          >
            <h3 className="text-xl font-semibold flex items-center gap-2">
              <TrendingUp className="text-green-400" /> Key Strengths
            </h3>
            <div className="space-y-3">
              {mockAnalysis.strengths.map((item, i) => (
                <Card key={i} className="bg-green-500/10 border-green-500/20 p-4">
                  <p className="text-green-100 font-medium">{item}</p>
                </Card>
              ))}
            </div>
          </motion.div>

          {/* Weaknesses */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="space-y-4"
          >
            <h3 className="text-xl font-semibold flex items-center gap-2">
              <AlertCircle className="text-red-400" /> Areas to Focus
            </h3>
            <div className="space-y-3">
              {mockAnalysis.weaknesses.map((item, i) => (
                <Card key={i} className="bg-red-500/10 border-red-500/20 p-4">
                  <p className="text-red-100 font-medium">{item}</p>
                </Card>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Recent Sessions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <Card className="bg-gray-900/30 border-white/10">
            <CardHeader>
              <CardTitle>Recent Sessions</CardTitle>
              <CardDescription>Your performance history</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {mockAnalysis.recentSessions.map((session) => (
                  <div key={session.id} className="flex items-center justify-between p-4 rounded-lg bg-white/5 hover:bg-white/10 transition-colors">
                    <div className="flex items-center gap-4">
                      <div className="h-10 w-10 rounded-full bg-blue-500/20 flex items-center justify-center">
                        <Mic className="h-5 w-5 text-blue-400" />
                      </div>
                      <div>
                        <p className="font-medium text-white">{session.mode}</p>
                        <p className="text-sm text-gray-400">{session.date}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <p className="text-2xl font-bold text-white">{session.score}</p>
                        <p className="text-xs text-gray-400">Score</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>

      </div>
    </div>
  );
}