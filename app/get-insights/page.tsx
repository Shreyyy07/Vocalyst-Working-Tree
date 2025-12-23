"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  TrendingUp, AlertCircle, Zap, Activity, MessageSquare, Mic,
  Trophy, Flame, Target, Eye
} from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar
} from "recharts";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

const FloatingPointsCanvas = dynamic(
  () => import("@/components/ui/FloatingPoints"),
  { ssr: false }
);

interface AggregatedAnalytics {
  totalSessions: number;
  totalDuration: number;
  averageEmotions: { emotion: string; percentage: number }[];
  averageGazeDirections: { direction: string; percentage: number }[];
  averageFillerWords: number;
  averageVocabularyScore: number;
  recentSessions?: any[];
}

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
  const [analytics, setAnalytics] = useState<AggregatedAnalytics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAnalytics();
  }, []);

  const fetchAnalytics = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:5328/api/get-analytics');

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.success && data.analytics) {
        setAnalytics(data.analytics);
      } else {
        throw new Error('Invalid response format');
      }
    } catch (err) {
      console.error('Failed to fetch analytics:', err);
      setError(err instanceof Error ? err.message : 'Failed to load analytics');
    } finally {
      setIsLoading(false);
    }
  };

  const formatDuration = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  // Calculate level based on total sessions
  const calculateLevel = (sessions: number) => {
    return Math.floor(sessions / 5) + 1;
  };

  const calculateXP = (sessions: number) => {
    const level = calculateLevel(sessions);
    const sessionsInCurrentLevel = sessions % 5;
    return {
      current: sessionsInCurrentLevel * 200,
      next: 1000,
      level,
      title: level >= 5 ? "Expert" : level >= 3 ? "Intermediate" : "Beginner"
    };
  };

  if (isLoading) {
    return (
      <div className="relative min-h-screen bg-black text-white">
        <div className="fixed inset-0 z-0">
          <FloatingPointsCanvas />
        </div>
        <div className="relative z-10 flex items-center justify-center min-h-screen">
          <div className="text-center">
            <div className="animate-spin text-6xl mb-4">🎯</div>
            <p className="text-xl text-gray-400">Loading your insights...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error || !analytics) {
    return (
      <div className="relative min-h-screen bg-black text-white">
        <div className="fixed inset-0 z-0">
          <FloatingPointsCanvas />
        </div>
        <div className="relative z-10 flex items-center justify-center min-h-screen">
          <div className="text-center space-y-4">
            <AlertCircle className="w-16 h-16 text-red-500 mx-auto" />
            <h2 className="text-2xl font-bold">Failed to Load Insights</h2>
            <p className="text-gray-400">{error}</p>
            <button
              onClick={fetchAnalytics}
              className="px-6 py-3 bg-primary text-white rounded-lg hover:bg-primary/90"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (analytics.totalSessions === 0) {
    return (
      <div className="relative min-h-screen bg-black text-white">
        <div className="fixed inset-0 z-0">
          <FloatingPointsCanvas />
        </div>
        <div className="relative z-10 flex items-center justify-center min-h-screen">
          <div className="text-center space-y-4">
            <Target className="w-16 h-16 text-blue-500 mx-auto" />
            <h2 className="text-2xl font-bold">No Practice Sessions Yet</h2>
            <p className="text-gray-400">Complete your first practice session to see insights!</p>
          </div>
        </div>
      </div>
    );
  }

  const gamification = calculateXP(analytics.totalSessions);

  return (
    <div className="relative min-h-screen bg-black text-white pt-24 pb-12 px-6">
      <div className="fixed inset-0 z-0">
        <FloatingPointsCanvas />
      </div>
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-900/10 via-transparent to-transparent z-[1] pointer-events-none" />

      <div className="relative z-10 max-w-7xl mx-auto space-y-8">
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
            Real-time analysis of your speaking performance and progress
          </p>
        </motion.div>

        {/* Top Section: Stats + Gamification */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Stats Card (2 Columns) */}
          <motion.div
            initial="hidden"
            animate="visible"
            variants={fadeIn}
            className="lg:col-span-2"
          >
            <Card className="h-full bg-gradient-to-br from-gray-900/80 to-black/80 border-2 border-purple-500/30 backdrop-blur-md">
              <CardContent className="p-8">
                <h2 className="text-2xl font-bold mb-6 text-purple-400">Your Progress</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-white/5 rounded-lg p-4 text-center backdrop-blur-sm">
                    <Mic className="text-blue-500 h-8 w-8 mx-auto mb-2" />
                    <div className="text-3xl font-bold text-white">{analytics.totalSessions}</div>
                    <div className="text-sm text-gray-400">Sessions</div>
                  </div>
                  <div className="bg-white/5 rounded-lg p-4 text-center backdrop-blur-sm">
                    <Activity className="text-green-500 h-8 w-8 mx-auto mb-2" />
                    <div className="text-3xl font-bold text-white">{formatDuration(analytics.totalDuration)}</div>
                    <div className="text-sm text-gray-400">Practice Time</div>
                  </div>
                  <div className="bg-white/5 rounded-lg p-4 text-center backdrop-blur-sm">
                    <MessageSquare className="text-yellow-500 h-8 w-8 mx-auto mb-2" />
                    <div className="text-3xl font-bold text-white">{analytics.averageFillerWords.toFixed(1)}%</div>
                    <div className="text-sm text-gray-400">Avg Fillers</div>
                  </div>
                  <div className="bg-white/5 rounded-lg p-4 text-center backdrop-blur-sm">
                    <Zap className="text-purple-500 h-8 w-8 mx-auto mb-2" />
                    <div className="text-3xl font-bold text-white">{analytics.averageVocabularyScore.toFixed(0)}</div>
                    <div className="text-sm text-gray-400">Clarity Score</div>
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
            <Card className="h-full bg-gray-900/80 border-white/10 backdrop-blur-md">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2">
                  <Trophy className="text-yellow-500 h-5 w-5" /> Level Progress
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Level Progress */}
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Level {gamification.level}: <span className="text-white font-semibold">{gamification.title}</span></span>
                    <span className="text-gray-500">{gamification.current} / {gamification.next} XP</span>
                  </div>
                  <Progress value={(gamification.current / gamification.next) * 100} className="h-2 bg-gray-800" />
                </div>

                {/* Grid stats */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-white/5 rounded-lg p-3 text-center backdrop-blur-sm">
                    <div className="flex justify-center mb-1"><Flame className="text-orange-500 h-6 w-6" /></div>
                    <div className="text-2xl font-bold text-white">{Math.min(analytics.totalSessions, 7)}</div>
                    <div className="text-xs text-gray-400">Day Streak</div>
                  </div>
                  <div className="bg-white/5 rounded-lg p-3 text-center backdrop-blur-sm">
                    <div className="flex justify-center mb-1"><Target className="text-blue-500 h-6 w-6" /></div>
                    <div className="text-2xl font-bold text-white">{analytics.totalSessions}</div>
                    <div className="text-xs text-gray-400">Total</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Metrics Grid */}
        <motion.div
          variants={stagger}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-1 md:grid-cols-2 gap-6"
        >
          {/* Emotional Expression */}
          <motion.div variants={fadeIn}>
            <Card className="bg-gray-900/80 border-white/10 backdrop-blur-md h-full">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  😊 Emotional Expression
                </CardTitle>
                <CardDescription>Your average emotional tone</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {analytics.averageEmotions.slice(0, 3).map((emotion) => (
                    <div key={emotion.emotion} className="space-y-1.5">
                      <div className="flex items-center justify-between text-sm">
                        <span className="capitalize text-white">{emotion.emotion}</span>
                        <span className="text-gray-400">{emotion.percentage.toFixed(1)}%</span>
                      </div>
                      <Progress
                        value={emotion.percentage}
                        max={100}
                        className="h-2 bg-gray-800"
                      />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Gaze Patterns */}
          <motion.div variants={fadeIn}>
            <Card className="bg-gray-900/80 border-white/10 backdrop-blur-md h-full">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Eye className="h-5 w-5" /> Gaze Patterns
                </CardTitle>
                <CardDescription>Where you look while speaking</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[200px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={analytics.averageGazeDirections}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis dataKey="direction" stroke="#666" />
                      <YAxis stroke="#666" />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#000', border: '1px solid #333', borderRadius: '8px' }}
                        itemStyle={{ color: '#fff' }}
                      />
                      <Bar dataKey="percentage" fill="#8b5cf6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </motion.div>

        {/* Recent Sessions */}
        {analytics.recentSessions && analytics.recentSessions.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <Card className="bg-gray-900/80 border-white/10 backdrop-blur-md">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="text-green-400" /> Recent Sessions
                </CardTitle>
                <CardDescription>Your latest practice sessions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {analytics.recentSessions.slice(0, 5).map((session, index) => (
                    <div
                      key={session.id || index}
                      className="flex items-center justify-between p-4 bg-white/5 rounded-lg backdrop-blur-sm hover:bg-white/10 transition-colors"
                    >
                      <div className="flex items-center gap-4">
                        <div className="h-10 w-10 rounded-full bg-gradient-to-tr from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold">
                          {index + 1}
                        </div>
                        <div>
                          <div className="font-semibold text-white capitalize">
                            {session.practiceMode.replace('-', ' ')}
                          </div>
                          <div className="text-sm text-gray-400">
                            {new Date(session.timestamp).toLocaleDateString()}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm text-gray-400">Fillers</div>
                        <div className="font-bold text-white">{session.fillerPercentage?.toFixed(1)}%</div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Insights */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-2 gap-6"
        >
          {/* Strengths */}
          <Card className="bg-gradient-to-br from-green-900/30 to-black/80 border-green-500/30 backdrop-blur-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-green-400">
                <TrendingUp className="h-5 w-5" /> Strengths
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {analytics.averageFillerWords < 5 && (
                <div className="bg-green-500/10 border border-green-500/20 p-3 rounded-md text-sm text-green-100">
                  Excellent filler word control - you speak clearly and confidently
                </div>
              )}
              {analytics.averageVocabularyScore > 80 && (
                <div className="bg-green-500/10 border border-green-500/20 p-3 rounded-md text-sm text-green-100">
                  Strong vocabulary and clarity score - your message comes across well
                </div>
              )}
              {analytics.totalSessions >= 5 && (
                <div className="bg-green-500/10 border border-green-500/20 p-3 rounded-md text-sm text-green-100">
                  Great consistency - you're building a strong practice habit
                </div>
              )}
            </CardContent>
          </Card>

          {/* Areas to Focus */}
          <Card className="bg-gradient-to-br from-orange-900/30 to-black/80 border-orange-500/30 backdrop-blur-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-orange-400">
                <AlertCircle className="h-5 w-5" /> Areas to Focus
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {analytics.averageFillerWords > 5 && (
                <div className="bg-orange-500/10 border border-orange-500/20 p-3 rounded-md text-sm text-orange-100">
                  Try to reduce filler words by pausing instead of saying "um" or "like"
                </div>
              )}
              {analytics.averageVocabularyScore < 70 && (
                <div className="bg-orange-500/10 border border-orange-500/20 p-3 rounded-md text-sm text-orange-100">
                  Work on clarity - practice speaking more slowly and enunciating
                </div>
              )}
              {analytics.totalSessions < 5 && (
                <div className="bg-orange-500/10 border border-orange-500/20 p-3 rounded-md text-sm text-orange-100">
                  Keep practicing regularly to build confidence and track progress
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}