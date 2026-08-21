"use client";

import { useState, useRef } from "react";
import { Mic, Square, Loader2, Send } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { AudioVisualizer } from "./AudioVisualizer";
import { cn } from "@/lib/utils";

interface VoiceRecorderProps {
  onAudioReady: (blob: Blob) => void;
  onTextSubmit: (text: string) => void;
  isProcessing: boolean;
}

export function VoiceRecorder({ onAudioReady, onTextSubmit, isProcessing }: VoiceRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [queryText, setQueryText] = useState("");
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        onAudioReady(blob);
        // Clean up tracks
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing microphone:", err);
      alert("Microphone access is required for voice queries.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (queryText.trim() && !isProcessing && !isRecording) {
      onTextSubmit(queryText);
      setQueryText("");
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto glass-card p-4 relative overflow-hidden transition-all duration-300">
      {/* Background glow when recording */}
      <AnimatePresence>
        {isRecording && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-brand-500/10 mix-blend-screen pointer-events-none"
          />
        )}
      </AnimatePresence>

      <form onSubmit={handleTextSubmit} className="relative z-10 flex flex-col gap-4">
        
        {/* Top area: input field + visualizer */}
        <div className="flex items-center gap-3 bg-dark-surface/50 rounded-xl p-2 border border-white/5 shadow-inner">
          <input
            type="text"
            placeholder={isRecording ? "Listening..." : "Type a query or use voice..."}
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            disabled={isRecording || isProcessing}
            className="flex-1 bg-transparent border-none outline-none px-4 py-2 text-white placeholder-white/40 disabled:opacity-50"
          />
          
          <AudioVisualizer isRecording={isRecording} />
          
          {queryText.trim() && !isRecording && (
            <button
              type="submit"
              disabled={isProcessing}
              className="p-2 rounded-lg bg-brand-500/20 text-brand-400 hover:bg-brand-500/40 transition-colors disabled:opacity-50"
            >
              <Send className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Bottom area: Mic button */}
        <div className="flex justify-center mt-2">
          <motion.button
            type="button"
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isProcessing}
            whileHover={!isProcessing ? { scale: 1.05 } : {}}
            whileTap={!isProcessing ? { scale: 0.95 } : {}}
            className={cn(
              "relative flex items-center justify-center w-16 h-16 rounded-full text-white shadow-lg transition-colors overflow-hidden group disabled:opacity-50 disabled:cursor-not-allowed",
              isRecording 
                ? "bg-red-500/80 hover:bg-red-500" 
                : "bg-brand-600/80 hover:bg-brand-500 backdrop-blur-md border border-white/20"
            )}
          >
            {/* Ripple effect */}
            {isRecording && (
              <motion.div
                className="absolute inset-0 rounded-full border-2 border-red-400"
                animate={{ scale: [1, 1.5, 2], opacity: [1, 0.5, 0] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              />
            )}
            
            <span className="relative z-10">
              {isProcessing ? (
                <Loader2 className="w-7 h-7 animate-spin" />
              ) : isRecording ? (
                <Square className="w-6 h-6 fill-current" />
              ) : (
                <Mic className="w-7 h-7" />
              )}
            </span>
          </motion.button>
        </div>
        
        {/* Status text */}
        <div className="text-center text-xs font-medium tracking-wide uppercase text-white/40 h-4">
          <AnimatePresence mode="wait">
            {isProcessing ? (
              <motion.span key="proc" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                Processing Request...
              </motion.span>
            ) : isRecording ? (
              <motion.span key="rec" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="text-red-400">
                Recording (Tap to Stop)
              </motion.span>
            ) : (
              <motion.span key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                Ready
              </motion.span>
            )}
          </AnimatePresence>
        </div>
      </form>
    </div>
  );
}
