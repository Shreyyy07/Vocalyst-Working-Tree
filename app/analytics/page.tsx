"use client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";

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
  averageLogicalFlow: number;
  practiceModes?: { mode: string; count: number }[];
  recentSessions?: any[];
}

const PRACTICE_MODE_COLORS: Record<string, string> = {
  persuasive: "#8b5cf6",
  emotive: "#ec4899",
  debate: "#f59e0b",
  storytelling: "#10b981",
  general: "#6366f1",
};

const PRACTICE_MODE_LABELS: Record<string, string> = {
  persuasive: "Persuasive",
  emotive: "Emotive",
  debate: "Debate",
  storytelling: "Storytelling",
  general: "General",
};

export default function AnalyticsPage() {
  const [analytics, setAnalytics] = useState<AggregatedAnalytics>({
    totalSessions: 0,
    totalDuration: 0,
    averageEmotions: [],
    averageGazeDirections: [],
    averageFillerWords: 0,
    averageVocabularyScore: 0,
    averageLogicalFlow: 0,
    practiceModes: [],
  });
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
    return `${hours}h ${minutes}m`;
  };

  if (isLoading) {
    return (
      <main className="relative min-h-screen">
        <FloatingPointsCanvas />
        <div className="relative z-10 container py-8 space-y-8">
          <div className="flex flex-col items-center">
            <h1 className="text-4xl font-bold mb-2">Communication Analytics</h1>
            <p className="text-muted-foreground">Loading your analytics...</p>
          </div>
          <div className="flex items-center justify-center py-16">
            <div className="animate-spin text-6xl">📊</div>
          </div>
        </div>
      </main>
    );
  }

  if (error) {
    return (
      <main className="relative min-h-screen">
        <FloatingPointsCanvas />
        <div className="relative z-10 container py-8 space-y-8">
          <div className="flex flex-col items-center">
            <h1 className="text-4xl font-bold mb-2">Communication Analytics</h1>
            <p className="text-red-500">{error}</p>
            <button
              onClick={fetchAnalytics}
              className="mt-4 px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/90"
            >
              Retry
            </button>
          </div>
        </div>
      </main>
    );
  }

  if (analytics.totalSessions === 0) {
    return (
      <main className="relative min-h-screen">
        <FloatingPointsCanvas />
        <div className="relative z-10 container py-8 space-y-8">
          <div className="flex flex-col items-center">
            <h1 className="text-4xl font-bold mb-2">Communication Analytics</h1>
            <p className="text-muted-foreground">No practice sessions yet</p>
            <p className="text-sm text-muted-foreground mt-2">
              Complete a practice session to see your analytics here!
            </p>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="relative min-h-screen">
      <div className="fixed inset-0 z-0">
        <FloatingPointsCanvas />
      </div>
      <div className="relative z-10 container py-8 space-y-8">
        <div className="flex flex-col items-center">
          <h1 className="text-4xl font-bold mb-2">Communication Analytics</h1>
          <p className="text-muted-foreground">
            Your speaking performance insights
          </p>
        </div>

        {/* Overview Cards */}
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          <Card className="bg-card/80 backdrop-blur-md border-primary/20">
            <CardHeader>
              <CardTitle className="text-center flex items-center justify-center gap-2">
                <span>Practice Sessions</span>
                <span className="text-2xl">📊</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="text-center">
              <div className="text-3xl font-bold">{analytics.totalSessions}</div>
              <p className="text-muted-foreground">Total recorded sessions</p>
            </CardContent>
          </Card>

          <Card className="bg-card/80 backdrop-blur-md border-primary/20">
            <CardHeader>
              <CardTitle className="text-center flex items-center justify-center gap-2">
                <span>Total Practice Time</span>
                <span className="text-2xl">⏱️</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="text-center">
              <div className="text-3xl font-bold">
                {formatDuration(analytics.totalDuration)}
              </div>
              <p className="text-muted-foreground">Time invested in practice</p>
            </CardContent>
          </Card>

          <Card className="bg-card/80 backdrop-blur-md border-primary/20">
            <CardHeader>
              <CardTitle className="text-center flex items-center justify-center gap-2">
                <span>Average Filler Words</span>
                <span className="text-2xl">💭</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="text-center">
              <div className="text-3xl font-bold">
                {analytics.averageFillerWords.toFixed(1)}%
              </div>
              <p className="text-muted-foreground">Speech clarity rating</p>
            </CardContent>
          </Card>
        </div>

        {/* Detailed Analysis */}
        <div className="grid gap-6 md:grid-cols-2">
          {/* Practice Modes Breakdown */}
          {analytics.practiceModes && analytics.practiceModes.length > 0 && (
            <Card className="bg-card/80 backdrop-blur-md border-primary/20">
              <CardHeader>
                <CardTitle className="flex items-center justify-center gap-2">
                  <span>Practice Modes</span>
                  <span className="text-2xl">🎯</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={analytics.practiceModes}
                        dataKey="count"
                        nameKey="mode"
                        cx="50%"
                        cy="50%"
                        outerRadius={100}
                        label={({ mode, count }) => `${PRACTICE_MODE_LABELS[mode] || mode}: ${count}`}
                      >
                        {analytics.practiceModes.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={PRACTICE_MODE_COLORS[entry.mode] || "#6366f1"}
                          />
                        ))}
                      </Pie>
                      <Tooltip
                        formatter={(value: any, name: any) => [value, PRACTICE_MODE_LABELS[name] || name]}
                      />
                      <Legend
                        formatter={(value: any) => PRACTICE_MODE_LABELS[value] || value}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Emotions Analysis */}
          <Card className="bg-card/80 backdrop-blur-md border-primary/20">
            <CardHeader>
              <CardTitle className="flex items-center justify-center gap-2">
                <span>Emotional Expression</span>
                <span className="text-2xl">😊</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {analytics.averageEmotions.length > 0 ? (
                  analytics.averageEmotions.map((emotion) => (
                    <div key={emotion.emotion} className="space-y-1.5">
                      <div className="flex items-center justify-between text-sm">
                        <span className="capitalize">{emotion.emotion}</span>
                        <span>{emotion.percentage.toFixed(1)}%</span>
                      </div>
                      <Progress
                        value={emotion.percentage}
                        max={100}
                        className="h-2"
                      />
                    </div>
                  ))
                ) : (
                  <p className="text-center text-muted-foreground">No emotion data yet</p>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Gaze Analysis */}
          <Card className="bg-card/80 backdrop-blur-md border-primary/20">
            <CardHeader>
              <CardTitle className="flex items-center justify-center gap-2">
                <span>Gaze Direction Patterns</span>
                <span className="text-2xl">👀</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[300px]">
                {analytics.averageGazeDirections.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={analytics.averageGazeDirections}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="direction" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="percentage" fill="hsl(var(--primary))" />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <p className="text-muted-foreground">No gaze data yet</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Recent Sessions */}
          {analytics.recentSessions && analytics.recentSessions.length > 0 && (
            <Card className="bg-card/80 backdrop-blur-md border-primary/20">
              <CardHeader>
                <CardTitle className="flex items-center justify-center gap-2">
                  <span>Recent Sessions</span>
                  <span className="text-2xl">📝</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3 max-h-[300px] overflow-y-auto">
                  {analytics.recentSessions.map((session, index) => (
                    <div
                      key={session.id || index}
                      className="p-3 bg-primary/5 rounded-lg border border-primary/10"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-semibold capitalize">
                          {PRACTICE_MODE_LABELS[session.practiceMode] || session.practiceMode}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {new Date(session.timestamp).toLocaleDateString()}
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">
                          Duration: {formatDuration(session.duration)}
                        </span>
                        <span className="text-muted-foreground">
                          Fillers: {session.fillerPercentage?.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </main>
  );
}

