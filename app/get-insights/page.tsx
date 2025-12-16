"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  TrendingUp, AlertCircle, Zap, Activity, MessageSquare, Mic, User,
  Trophy, Flame, Target, ArrowRight, Calendar
} from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area
} from "recharts";

// Mock data to simulate AI analysis
const mockAnalysis = {
  archetype: {
    title: "The Diplomat",
    description: "You speak with empathy and clarity, making you excellent at resolving conflicts and building rapport.",
    traits: ["Empathetic", "Clear", "Balanced"],
    icon: "🤝"
  },
  gamification: {
    level: 5,
    title: "Orator",
    currentXp: 750,
    nextLevelXp: 1000,
    streak: 4,
    totalSessions: 23
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
    { id: 4, date: "3 days ago", mode: "Storytelling", score: 82 },
    { id: 5, date: "5 days ago", mode: "Debate", score: 70 },
  ],
  progressData: [
    { session: '1', score: 65, clarity: 70 },
    { session: '2', score: 68, clarity: 72 },
    { session: '3', score: 75, clarity: 75 },
    { session: '4', score: 72, clarity: 74 },
    { session: '5', score: 82, clarity: 80 },
    { session: '6', score: 88, clarity: 85 },
    { session: '7', score: 92, clarity: 88 },
  ],
  recommendedExercise: {
    title: "The 'Like' Detox",
    description: "Your filler word usage is slightly up. Try to speak for 60 seconds about your favorite hobby without saying 'like' or 'um'.",
    difficulty: "Medium",
    duration: "2 mins"
  }
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

      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <motion.div
          initial="hidden"
          animate="visible"
          variants={fadeIn}
          className="text-center space-y-4 mb-12"
        >
          <h1 className="text-5xl font-extrabold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            Your Communication Profile
          </h1>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            AI-powered analysis of your speaking style, strengths, and areas for growth.
          </p>
        </motion.div>

        {/* Top Section: Archetype + Gamification */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Archetype Card (2 Columns) */}
          <motion.div
            initial="hidden"
            animate="visible"
            variants={fadeIn}
            className="lg:col-span-2"
          >
            <Card className="h-full bg-gradient-to-br from-gray-900 to-black border-2 border-purple-500/30 overflow-hidden relative">
              <div className="absolute top-0 right-0 p-8 opacity-10 text-9xl">
                {mockAnalysis.archetype.icon}
              </div>
              <CardContent className="p-8 flex flex-col md:flex-row items-center gap-8 relative z-10 h-full">
                <div className="flex-shrink-0">
                  <div className="h-32 w-32 rounded-full bg-gradient-to-tr from-blue-500 to-purple-600 p-[3px] shadow-[0_0_50px_rgba(168,85,247,0.4)]">
                    <div className="h-full w-full rounded-full bg-black flex items-center justify-center text-5xl">
                      {mockAnalysis.archetype.icon}
                    </div>
                  </div>
                </div>
                <div className="space-y-4 flex-1 text-center md:text-left">
                  <div className="space-y-1">
                    <h2 className="text-sm font-medium text-purple-400 uppercase tracking-widest">Communication Archetype</h2>
                    <h3 className="text-3xl font-bold text-white">{mockAnalysis.archetype.title}</h3>
                  </div>
                  <p className="text-gray-300 leading-relaxed">
                    {mockAnalysis.archetype.description}
                  </p>
                  <div className="flex flex-wrap justify-center md:justify-start gap-2 pt-2">
                    {mockAnalysis.archetype.traits.map(trait => (
                      <Badge key={trait} variant="secondary" className="px-3 py-1 text-xs bg-white/10 hover:bg-white/20 border-0">
                        {trait}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Gamification Stats (1 Column) */}
          <motion.div
            initial="hidden"
            animate="visible"
            variants={fadeIn}
            transition={{ delay: 0.1 }}
          >
            <Card className="h-full bg-gray-900/40 border-white/10 flex flex-col justify-center">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2">
                  <Trophy className="text-yellow-500 h-5 w-5" /> Your Progress
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Level Progress */}
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Level {mockAnalysis.gamification.level}: <span className="text-white font-semibold">{mockAnalysis.gamification.title}</span></span>
                    <span className="text-gray-500">{mockAnalysis.gamification.currentXp} / {mockAnalysis.gamification.nextLevelXp} XP</span>
                  </div>
                  <Progress value={(mockAnalysis.gamification.currentXp / mockAnalysis.gamification.nextLevelXp) * 100} className="h-2 bg-gray-800" indicatorClassName="bg-gradient-to-r from-yellow-500 to-orange-500" />
                </div>

                {/* Grid stats */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-white/5 rounded-lg p-3 text-center">
                    <div className="flex justify-center mb-1"><Flame className="text-orange-500 h-6 w-6" /></div>
                    <div className="text-2xl font-bold text-white">{mockAnalysis.gamification.streak}</div>
                    <div className="text-xs text-gray-400">Day Streak</div>
                  </div>
                  <div className="bg-white/5 rounded-lg p-3 text-center">
                    <div className="flex justify-center mb-1"><Mic className="text-blue-500 h-6 w-6" /></div>
                    <div className="text-2xl font-bold text-white">{mockAnalysis.gamification.totalSessions}</div>
                    <div className="text-xs text-gray-400">Sessions</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Middle Section: Metrics Grid */}
        <motion.div
          variants={stagger}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"
        >
          {/* Clarity */}
          <motion.div variants={fadeIn}>
            <Card className="bg-gray-900/50 border-white/10 h-full hover:bg-gray-900/80 transition-colors group">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">Clarity Score</CardTitle>
                <Zap className="h-4 w-4 text-yellow-400 group-hover:scale-110 transition-transform" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-white mb-2">{mockAnalysis.metrics.clarity}%</div>
                <Progress value={mockAnalysis.metrics.clarity} className="h-1.5 bg-gray-800" indicatorClassName="bg-yellow-400" />
                <p className="text-xs text-gray-500 mt-2">Top 15% of users</p>
              </CardContent>
            </Card>
          </motion.div>

          {/* WPM */}
          <motion.div variants={fadeIn}>
            <Card className="bg-gray-900/50 border-white/10 h-full hover:bg-gray-900/80 transition-colors group">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">Pace (WPM)</CardTitle>
                <Activity className="h-4 w-4 text-blue-400 group-hover:scale-110 transition-transform" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-white mb-2">{mockAnalysis.metrics.wpm}</div>
                <div className="flex items-center gap-2">
                  <div className="h-1.5 w-full bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-400 w-3/4" />
                  </div>
                </div>
                <p className="text-xs text-gray-500 mt-2">Ideal range: 130-150</p>
              </CardContent>
            </Card>
          </motion.div>

          {/* Filler Words */}
          <motion.div variants={fadeIn}>
            <Card className="bg-gray-900/50 border-white/10 h-full hover:bg-gray-900/80 transition-colors group">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">Filler Words</CardTitle>
                <MessageSquare className="h-4 w-4 text-red-400 group-hover:scale-110 transition-transform" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-white mb-2">{mockAnalysis.metrics.fillerWords}%</div>
                <Progress value={mockAnalysis.metrics.fillerWords * 10} className="h-1.5 bg-gray-800" indicatorClassName="bg-red-400" />
                <p className="text-xs text-gray-500 mt-2">Decreased by 0.5%</p>
              </CardContent>
            </Card>
          </motion.div>

          {/* Eye Contact */}
          <motion.div variants={fadeIn}>
            <Card className="bg-gray-900/50 border-white/10 h-full hover:bg-gray-900/80 transition-colors group">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">Eye Contact</CardTitle>
                <User className="h-4 w-4 text-green-400 group-hover:scale-110 transition-transform" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-white mb-2">{mockAnalysis.metrics.eyeContact}%</div>
                <Progress value={mockAnalysis.metrics.eyeContact} className="h-1.5 bg-gray-800" indicatorClassName="bg-green-400" />
                <p className="text-xs text-gray-500 mt-2">Great engagement!</p>
              </CardContent>
            </Card>
          </motion.div>
        </motion.div>

        {/* Lower Section: Charts & Breakdown */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

          {/* Progress Chart */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
          >
            <Card className="h-full bg-gray-900/30 border-white/10">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="text-purple-400" /> Improvement Trends
                </CardTitle>
                <CardDescription>Your overall score over the last 7 sessions</CardDescription>
              </CardHeader>
              <CardContent className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={mockAnalysis.progressData}>
                    <defs>
                      <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8} />
                        <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                    <XAxis dataKey="session" stroke="#666" />
                    <YAxis stroke="#666" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#000', border: '1px solid #333', borderRadius: '8px' }}
                      itemStyle={{ color: '#fff' }}
                    />
                    <Area type="monotone" dataKey="score" stroke="#8b5cf6" fillOpacity={1} fill="url(#colorScore)" strokeWidth={3} />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </motion.div>

          {/* Recommended Exercise & Weaknesses */}
          <div className="space-y-6">

            {/* Daily Challenge Card */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
            >
              <Card className="bg-gradient-to-r from-blue-900/40 to-purple-900/40 border-blue-500/30">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-blue-300">
                    <Target className="h-5 w-5" /> Recommended Practice
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h3 className="text-xl font-bold text-white mb-2">{mockAnalysis.recommendedExercise.title}</h3>
                      <p className="text-gray-300 text-sm">{mockAnalysis.recommendedExercise.description}</p>
                    </div>
                    <Badge className="bg-blue-500 hover:bg-blue-600">{mockAnalysis.recommendedExercise.difficulty}</Badge>
                  </div>
                  <div className="flex items-center justify-between mt-4">
                    <span className="text-xs text-gray-400 flex items-center gap-1"><Calendar className="h-3 w-3" /> Daily Challenge</span>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="px-4 py-2 bg-white text-black text-sm font-bold rounded-full flex items-center gap-2 hover:bg-gray-200 transition-colors"
                    >
                      Start Now <ArrowRight className="h-4 w-4" />
                    </motion.button>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            {/* Analysis Breakdown */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              className="grid grid-cols-1 sm:grid-cols-2 gap-4"
            >
              <div className="space-y-3">
                <h3 className="text-sm font-semibold flex items-center gap-2 text-gray-400">
                  <TrendingUp className="text-green-400 h-4 w-4" /> Strengths
                </h3>
                {mockAnalysis.strengths.slice(0, 2).map((item, i) => (
                  <div key={i} className="bg-green-500/10 border border-green-500/20 p-3 rounded-md text-xs text-green-100">
                    {item}
                  </div>
                ))}
              </div>
              <div className="space-y-3">
                <h3 className="text-sm font-semibold flex items-center gap-2 text-gray-400">
                  <AlertCircle className="text-red-400 h-4 w-4" /> Areas to Focus
                </h3>
                {mockAnalysis.weaknesses.slice(0, 2).map((item, i) => (
                  <div key={i} className="bg-red-500/10 border border-red-500/20 p-3 rounded-md text-xs text-red-100">
                    {item}
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        </div>

      </div>
    </div>
  );
}