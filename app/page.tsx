"use client";

import { motion } from "framer-motion";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import Link from "next/link";
import { useState } from "react";
import dynamic from "next/dynamic";

// Dynamic import for FloatingPointsCanvas (loads asynchronously)
const FloatingPointsCanvas = dynamic(
  () => import("@/components/ui/FloatingPoints"),
  { ssr: false }
);

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.8,
      ease: [0.22, 1, 0.36, 1]
    }
  }
};

const container = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.3,
      delayChildren: 0.2
    }
  }
};

const arrowVariants = {
  rest: {
    x: -4,
    opacity: 0.7,
    scale: 0.9
  },
  hover: {
    x: 4,
    opacity: 1,
    scale: 1,
    transition: {
      duration: 0.3,
      ease: [0.22, 1, 0.36, 1]
    }
  }
};

export default function LandingPage() {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const sections = [
    {
      title: "Practice Speaking",
      description: "Get real-time AI feedback on your communication skills",
      href: "/practice",
      icon: "🎯",
      color: "from-blue-500/10 to-cyan-500/10",
      hoverColor: "from-blue-500/20 to-cyan-500/20"
    },
    {
      title: "Track Progress",
      description: "Visualise your improvement over time with detailed analytics",
      href: "/analytics",
      icon: "📈",
      color: "from-violet-500/10 to-purple-500/10",
      hoverColor: "from-violet-500/20 to-purple-500/20"
    },
    {
      title: "Get Insights",
      description: "Receive personalised tips and recommendations for improvement",
      href: "/get-insights",
      icon: "💡",
      color: "from-amber-500/10 to-orange-500/10",
      hoverColor: "from-amber-500/20 to-orange-500/20"
    }
  ];

  return (
    <>
      {/* background points canvas */}

      <motion.main
        initial="hidden"
        animate="visible"
        className="relative z-10 min-h-screen flex flex-col items-center justify-center p-4 bg-black"
      >
        <div className="absolute inset-0 -z-10">
          <FloatingPointsCanvas />
        </div>
        <div className="w-full max-w-4xl mx-auto">
          {/* Hero Section */}
          <motion.div
            variants={fadeIn}
            className="text-center mb-24 space-y-8"
          >
            <motion.div
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{
                duration: 1,
                ease: [0.34, 1.56, 0.64, 1]
              }}
              className="inline-block text-6xl mb-2"
            >

            </motion.div>
            <motion.h1
              className="text-6xl md:text-7xl font-bold tracking-tight text-white"
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{
                delay: 0.3,
                duration: 0.8,
                ease: [0.22, 1, 0.36, 1]
              }}
            >
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-fuchsia-500">
                Vocalyst
                <span className="text-primary"></span>
              </span>


            </motion.h1>
            <motion.p
              className="text-xl text-muted-foreground/80 font-light max-w-2xl mx-auto text-white/80"
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{
                delay: 0.5,
                duration: 0.8,
                ease: [0.22, 1, 0.36, 1]
              }}
            >
              Master the art of communication with AI-powered feedback and insights
            </motion.p>
          </motion.div>

          {/* Navigation Cards */}
          <motion.div
            variants={container}
            className="grid grid-cols-1 md:grid-cols-3 gap-5 px-4"
          >
            {sections.map((section, index) => (
              <motion.div
                key={section.title}
                variants={fadeIn}
                onHoverStart={() => setHoveredIndex(index)}
                onHoverEnd={() => setHoveredIndex(null)}
              >
                <Link href={section.href}>
                  <Card
                    className={`
                    group relative overflow-hidden rounded-xl p-[2px]
                    bg-gradient-to-br ${section.color}
                    shadow-[0_0_30px_#00f0ff20] transition-all duration-500
                    hover:scale-[1.04] hover:shadow-[0_0_40px_#00f0ff80]
                    before:absolute before:inset-0 before:z-0
                    before:bg-[linear-gradient(120deg,rgba(255,255,255,0)_30%,rgba(255,255,255,0.1)_50%,rgba(255,255,255,0)_70%)]
                    before:animate-shimmer
                  `}
                  >
                    <div
                      className="absolute inset-0 z-0 opacity-0 group-hover:opacity-100
                    bg-[linear-gradient(120deg,rgba(255,255,255,0)_30%,rgba(255,255,255,0.2)_50%,rgba(255,255,255,0)_70%)]
                    animate-shimmer pointer-events-none"
                    />

                    <motion.div
                      className={`absolute inset-0 bg-gradient-to-r ${hoveredIndex === index ? section.hoverColor : section.color
                        } transition-opacity duration-700`}
                      initial={false}
                      animate={{
                        opacity: hoveredIndex === index ? 1 : 0.5
                      }}
                    />
                    <CardHeader className="py-8 relative">
                      <div className="flex flex-col items-center text-center gap-4">
                        <motion.span
                          className="text-4xl"
                          animate={{
                            scale: hoveredIndex === index ? 1.1 : 1,
                            y: hoveredIndex === index ? -2 : 0
                          }}
                          transition={{
                            type: "spring",
                            stiffness: 200,
                            damping: 15
                          }}
                        >
                          {section.icon}
                        </motion.span>
                        <div>
                          <CardTitle className="text-xl font-semibold mb-2 text-white">
                            {section.title}
                          </CardTitle>
                          <CardDescription className="text-sm text-muted-foreground/70 text-white/70">
                            {section.description}
                          </CardDescription>
                        </div>
                        <motion.div
                          className="h-8 flex items-center justify-center mt-2"
                          initial="rest"
                          animate={hoveredIndex === index ? "hover" : "rest"}
                        >
                          <motion.svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="20"
                            height="20"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            className="text-primary"
                            variants={arrowVariants}
                          >
                            <path d="M5 12h14" />
                            <path d="m12 5 7 7-7 7" />
                          </motion.svg>
                        </motion.div>
                      </div>
                    </CardHeader>
                  </Card>
                </Link>
              </motion.div>
            ))}
          </motion.div>

          <motion.footer
            variants={fadeIn}
            className="text-center text-sm text-muted-foreground/60 mt-24 text-white/60"
          >
            <p>Empowering better communication through AI 🚀</p>
          </motion.footer>
        </div>
      </motion.main>
    </>
  );
}