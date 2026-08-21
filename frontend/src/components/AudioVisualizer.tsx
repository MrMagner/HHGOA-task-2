"use client";

import { motion } from "framer-motion";

export function AudioVisualizer({ isRecording }: { isRecording: boolean }) {
  if (!isRecording) return null;

  return (
    <div className="flex items-center justify-center gap-1 h-8 px-4">
      {[...Array(5)].map((_, i) => (
        <motion.div
          key={i}
          className="w-1.5 bg-brand-400 rounded-full"
          initial={{ height: 4 }}
          animate={{
            height: [4, 16 + Math.random() * 16, 4],
          }}
          transition={{
            duration: 0.5,
            repeat: Infinity,
            delay: i * 0.1,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}
