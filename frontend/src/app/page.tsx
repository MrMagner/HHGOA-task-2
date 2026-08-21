"use client";

import { useState } from "react";
import { VoiceRecorder } from "@/components/VoiceRecorder";
import { RAGResult, type RAGResponseData } from "@/components/RAGResult";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, Activity } from "lucide-react";

export default function Home() {
  const [result, setResult] = useState<RAGResponseData | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Use environment variable for production, fallback to localhost for local development
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api/v1";

  const handleAudioReady = async (blob: Blob) => {
    setIsProcessing(true);
    setResult(null);
    
    const formData = new FormData();
    formData.append("audio", blob, "recording.webm");
    formData.append("language", "en"); // Adjust as needed
    formData.append("top_k", "3");
    
    try {
      const res = await fetch(`${API_BASE}/query/voice`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || data.message || "Failed to process voice query");
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setResult({
        status: "error",
        answer: err.message,
        query: "Voice input",
        source: "voice",
        grounded: false,
        sources: [],
        latency: { total_ms: 0 },
        refusal: true,
        refusal_reason: err.message
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleTextSubmit = async (text: string) => {
    setIsProcessing(true);
    setResult(null);
    
    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text, top_k: 3 }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || data.message || "Failed to process text query");
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setResult({
        status: "error",
        answer: err.message,
        query: text,
        source: "text",
        grounded: false,
        sources: [],
        latency: { total_ms: 0 },
        refusal: true,
        refusal_reason: err.message
      });
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <main className="min-h-screen relative flex flex-col pt-16 md:pt-24 px-4 pb-20">
      
      {/* Hero Section */}
      <div className="w-full max-w-4xl mx-auto flex flex-col items-center text-center mb-12 z-10">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs font-medium text-brand-300 mb-6"
        >
          <Activity className="w-3.5 h-3.5" />
          <span>Real-time Audio Processing</span>
        </motion.div>
        
        <motion.h1 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-4xl md:text-6xl font-bold tracking-tight mb-4"
        >
          Voice-Enabled <br className="md:hidden" />
          <span className="text-gradient">Neural Search</span>
        </motion.h1>
        
        <motion.p 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="text-white/50 max-w-xl text-sm md:text-base leading-relaxed"
        >
          Speak your query to retrieve precise, grounded answers backed by MSMARCO-XI.
          Powered by ultra-fast LLMs and hybrid search.
        </motion.p>
      </div>

      {/* Input Section */}
      <div className="relative z-20 w-full mb-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <VoiceRecorder 
            onAudioReady={handleAudioReady} 
            onTextSubmit={handleTextSubmit}
            isProcessing={isProcessing}
          />
        </motion.div>
      </div>

      {/* Results Section */}
      <div className="relative z-10 w-full flex-1 flex flex-col items-center">
        <AnimatePresence mode="wait">
          {result ? (
            <RAGResult key="result" data={result} />
          ) : !isProcessing ? (
            <motion.div 
              key="empty"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="mt-12 flex flex-col items-center justify-center text-white/20"
            >
              <Sparkles className="w-12 h-12 mb-4 opacity-50" />
              <p className="text-sm font-medium uppercase tracking-widest">Awaiting Query</p>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
    </main>
  );
}
