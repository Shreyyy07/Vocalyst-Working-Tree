"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  TrendingUp, TrendingDown, Minus, Trophy, Target, Lightbulb,
  BarChart3, Zap, RefreshCw
} from "lucide-react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip
} from "recharts";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

const FloatingPointsCanvas = dynamic(
  () => import("@/components/ui/FloatingPoints"),
  { ssr: false }
);

interface InsightsData {
  strengths: string[];
  weaknesses: string[];
  trends: {
    wpm_trend?: string;
    filler_trend?: string;
  };
  recommendations: string[];
  metrics: {
    avgWpm: number;
    avgFiller: number;
    avgClarity: number;
    avgDuration: number;
    totalSessions: number;
  };
}

export default function GetInsightsPage() {
  const [insights, setInsights] = useState<InsightsData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isResetting, setIsResetting] = useState(false);

  useEffect(() => {
    fetchInsights();
  }, []);

  const fetchInsights = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:5328/api/get-insights');

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.success && data.insights) {
        setInsights(data.insights);
      } else {
        throw new Error('Invalid response format');
      }
    } catch (err) {
      console.error('Failed to fetch insights:', err);
      setError(err instanceof Error ? err.message : 'Failed to load insights');
    } finally {
      setIsLoading(false);
    }
  };

  const handleResetInsights = async () => {
    if (!confirm('Are you sure you want to reset all data? This will archive your current progress and start fresh.')) {
      return;
    }

    setIsResetting(true);
    try {
      const response = await fetch('http://localhost:5328/api/reset-analytics', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to reset data');
      }

      await fetchInsights();
      alert('Data has been reset successfully!');
    } catch (err) {
      console.error('Failed to reset:', err);
      alert('Failed to reset data. Please try again.');
    } finally {
      setIsResetting(false);
    }
  };

  // Prepare radar chart data
  const radarData = insights ? [
    { skill: 'Pace', value: Math.min(100, (insights.metrics.avgWpm / 150) * 100) },
    { skill: 'Clarity', value: insights.metrics.avgClarity },
    { skill: 'Fluency', value: Math.max(0, 100 - insights.metrics.avgFiller * 5) },
    { skill: 'Duration', value: Math.min(100, (insights.metrics.avgDuration / 120) * 100) },
    { skill: 'Consistency', value: Math.min(100, (insights.metrics.totalSessions / 20) * 100) },
  ] : [];

  const getTrendIcon = (trend?: string) => {
    if (trend === 'improving') return <TrendingUp className="w-4 h-4 text-green-400" />;
    if (trend === 'declining') return <TrendingDown className="w-4 h-4 text-red-400" />;
    return <Minus className="w-4 h-4 text-yellow-400" />;
  };

  const getTrendColor = (trend?: string) => {
    if (trend === 'improving') return 'text-green-400';
    if (trend === 'declining') return 'text-red-400';
    return 'text-yellow-400';
  };

  if (isLoading) {
    return (
      <main className="relative min-h-screen">
        <FloatingPointsCanvas />
        <div className="relative z-10 container py-8">
          <div className="flex items-center justify-center h-[60vh]">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
              <p className="text-muted-foreground">Loading your insights...</p>
            </div>
          </div>
        </div>
      </main>
    );
  }

  if (error) {
    return (
      <main className="relative min-h-screen">
        <FloatingPointsCanvas />
        <div className="relative z-10 container py-8">
          <div className="flex flex-col items-center justify-center h-[60vh]">
            <h1 className="text-4xl font-bold mb-2">AI-Powered Insights</h1>
            <p className="text-red-500">{error}</p>
            <button
              onClick={fetchInsights}
              className="mt-4 px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/90"
            >
              Retry
            </button>
          </div>
        </div>
      </main>
    );
  }

  if (!insights || insights.metrics.totalSessions === 0) {
    return (
      <main className="relative min-h-screen">
        <FloatingPointsCanvas />
        <div className="relative z-10 container py-8">
          <div className="flex flex-col items-center justify-center h-[60vh]">
            <h1 className="text-4xl font-bold mb-2">AI-Powered Insights</h1>
            <p className="text-muted-foreground">No practice sessions yet</p>
            <p className="text-sm text-muted-foreground mt-2">
              Complete a practice session to get personalized insights!
            </p>
          </div>
        </div>
      </main>
    );
  }

  const level = Math.floor(insights.metrics.totalSessions / 5) + 1;
  const xpProgress = (insights.metrics.totalSessions % 5) * 20;

  return (
    <main className="relative min-h-screen">
      <div className="fixed inset-0 z-0">
        <FloatingPointsCanvas />
      </div>
      <div className="relative z-10 container py-8 pt-24 space-y-8">
        {/* Header */}
        <div className="flex flex-col items-center">
          <div className="flex items-center gap-4 mb-2">
            <h1 className="text-4xl font-bold">AI-Powered Insights</h1>
            <button
              onClick={handleResetInsights}
              disabled={isResetting}
              className="px-4 py-2 bg-red-500/20 text-red-400 border border-red-500/30 rounded-lg hover:bg-red-500/30 transition-colors disabled:opacity-50"
            >
              {isResetting ? 'Resetting...' : 'Reset Progress'}
            </button>
          </div>
          <p className="text-muted-foreground">
            Personalized recommendations based on your performance
          </p>
        </div>

        {/* Gamification Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="bg-card/80 backdrop-blur-md border-primary/30">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Trophy className="w-5 h-5 text-yellow-400" />
                Level {level}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Progress to Level {level + 1}</span>
                  <span>{xpProgress}%</span>
                </div>
                <Progress value={xpProgress} className="h-2" />
                <p className="text-xs text-muted-foreground">
                  {5 - (insights.metrics.totalSessions % 5)} sessions until next level
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-card/80 backdrop-blur-md border-green-500/30">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="w-5 h-5 text-green-400" />
                Total Sessions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{insights.metrics.totalSessions}</div>
              <p className="text-sm text-muted-foreground mt-2">
                {insights.metrics.totalSessions >= 20 ? 'Expert practitioner!' :
                  insights.metrics.totalSessions >= 10 ? 'Great consistency!' :
                    'Keep building your streak!'}
              </p>
            </CardContent>
          </Card>

          <Card className="bg-card/80 backdrop-blur-md border-blue-500/30">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-blue-400" />
                Avg Performance
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span>WPM:</span>
                  <span className="font-semibold">{insights.metrics.avgWpm.toFixed(0)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Clarity:</span>
                  <span className="font-semibold">{insights.metrics.avgClarity.toFixed(0)}%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Filler Words:</span>
                  <span className="font-semibold">{insights.metrics.avgFiller.toFixed(1)}%</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Skill Breakdown Radar Chart */}
        <Card className="bg-card/80 backdrop-blur-md">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Skill Breakdown
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#444" />
                <PolarAngleAxis dataKey="skill" stroke="#888" />
                <PolarRadiusAxis angle={90} domain={[0, 100]} stroke="#888" />
                <Radar
                  name="Your Skills"
                  dataKey="value"
                  stroke="#8b5cf6"
                  fill="#8b5cf6"
                  fillOpacity={0.6}
                />
              </RadarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Strengths & Weaknesses */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="bg-card/80 backdrop-blur-md border-green-500/30">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-green-400">
                <Trophy className="w-5 h-5" />
                Your Strengths
              </CardTitle>
            </CardHeader>
            <CardContent>
              {insights.strengths.length > 0 ? (
                <ul className="space-y-2">
                  {insights.strengths.map((strength, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-green-400 mt-1">✓</span>
                      <span className="text-sm">{strength}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted-foreground">
                  Complete more sessions to identify your strengths!
                </p>
              )}
            </CardContent>
          </Card>

          <Card className="bg-card/80 backdrop-blur-md border-orange-500/30">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-orange-400">
                <Target className="w-5 h-5" />
                Areas to Focus
              </CardTitle>
            </CardHeader>
            <CardContent>
              {insights.weaknesses.length > 0 ? (
                <ul className="space-y-2">
                  {insights.weaknesses.map((weakness, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-orange-400 mt-1">→</span>
                      <span className="text-sm">{weakness}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted-foreground">
                  You're doing great! Keep up the good work.
                </p>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Trends */}
        {Object.keys(insights.trends).length > 0 && (
          <Card className="bg-card/80 backdrop-blur-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                Performance Trends
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {insights.trends.wpm_trend && (
                  <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                    <div>
                      <p className="text-sm font-medium">Speaking Pace</p>
                      <p className={`text-xs ${getTrendColor(insights.trends.wpm_trend)}`}>
                        {insights.trends.wpm_trend}
                      </p>
                    </div>
                    {getTrendIcon(insights.trends.wpm_trend)}
                  </div>
                )}
                {insights.trends.filler_trend && (
                  <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                    <div>
                      <p className="text-sm font-medium">Filler Words</p>
                      <p className={`text-xs ${getTrendColor(insights.trends.filler_trend)}`}>
                        {insights.trends.filler_trend}
                      </p>
                    </div>
                    {getTrendIcon(insights.trends.filler_trend)}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Personalized Recommendations */}
        <Card className="bg-card/80 backdrop-blur-md border-purple-500/30">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-purple-400">
              <Lightbulb className="w-5 h-5" />
              Personalized Recommendations
            </CardTitle>
          </CardHeader>
          <CardContent>
            {insights.recommendations.length > 0 ? (
              <ul className="space-y-3">
                {insights.recommendations.map((rec, idx) => (
                  <li key={idx} className="flex items-start gap-3 p-3 rounded-lg bg-purple-500/10 border border-purple-500/20">
                    <span className="text-purple-400 font-bold">{idx + 1}.</span>
                    <span className="text-sm">{rec}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-muted-foreground">
                Complete more sessions to get personalized recommendations!
              </p>
            )}
          </CardContent>
        </Card>
      </div>
    </main>
  );
}