"use client";

import { motion } from "framer-motion";
import { FileText, Clock, ShieldAlert, Cpu, Brain, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

// Assuming types based on backend schema
interface Source {
  id: string;
  text: string;
  score: number;
}

interface Latency {
  stt_ms?: number;
  embedding_ms?: number;
  retrieval_ms?: number;
  generation_ms?: number;
  total_ms: number;
}

export interface RAGResponseData {
  query: string;
  source: "text" | "voice";
  transcript?: string;
  answer: string;
  grounded: boolean;
  confidence?: number;
  sources: Source[];
  latency: Latency;
  refusal?: boolean;
  refusal_reason?: string;
  is_demo?: boolean;
  status?: "success" | "error";
}

export function RAGResult({ data }: { data: RAGResponseData }) {
  if (data.status === "error") {
    return (
      <motion.div 
        initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-4xl mx-auto glass-card p-6 border-red-500/30 bg-red-950/20"
      >
        <div className="flex items-center gap-3 text-red-400 mb-2">
          <ShieldAlert className="w-6 h-6" />
          <h3 className="text-lg font-semibold">Error Processing Request</h3>
        </div>
        <p className="text-white/80">{data.answer}</p>
        
        {data.refusal_reason && (
          <div className="mt-4 p-3 rounded-lg bg-red-950/50 text-sm text-red-200">
            <p>{data.refusal_reason}</p>
          </div>
        )}
      </motion.div>
    );
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
      className="w-full max-w-4xl mx-auto space-y-6"
    >
      {/* User Query */}
      <div className="flex items-start gap-4">
        <div className="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center shrink-0 border border-white/5">
          <span className="text-white/60 font-semibold uppercase">U</span>
        </div>
        <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl rounded-tl-none px-6 py-4 text-white/90 font-medium">
          {data.query}
        </div>
      </div>

      {/* AI Answer Card */}
      <div className="glass-card p-1 relative overflow-hidden group">
        <div className="absolute inset-0 bg-gradient-to-br from-brand-500/5 to-transparent pointer-events-none" />
        
        <div className="bg-dark-surface/80 backdrop-blur-xl rounded-[14px] p-6 relative z-10 border border-white/5">
          {/* Answer Header */}
          <div className="flex items-center justify-between mb-4 pb-4 border-b border-white/10">
            <div className="flex items-center gap-2 text-brand-400">
              <Brain className="w-5 h-5" />
              <span className="font-semibold tracking-wide">RAG Response</span>
            </div>
            
            {/* Grounding Badge */}
            {data.grounded && (
              <div className="flex items-center gap-1 text-xs px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Grounded</span>
              </div>
            )}
          </div>
          
          {/* Answer Text */}
          <div className="prose prose-invert max-w-none text-white/80 leading-relaxed whitespace-pre-wrap mb-6">
            {data.answer}
          </div>
          
          {/* Metrics Footer */}
          <div className="flex flex-wrap gap-3 mt-6 pt-4 border-t border-white/10 text-xs font-medium text-white/50">
            <div className="flex items-center gap-1.5 bg-black/40 px-3 py-1.5 rounded-lg border border-white/5">
              <Clock className="w-3.5 h-3.5" />
              <span>{data.latency.total_ms}ms total</span>
            </div>
            
            {data.latency.retrieval_ms && (
              <div className="flex items-center gap-1.5 bg-black/40 px-3 py-1.5 rounded-lg border border-white/5">
                <FileText className="w-3.5 h-3.5" />
                <span>Retrieval: {data.latency.retrieval_ms}ms</span>
              </div>
            )}
            
            {data.is_demo !== undefined && (
              <div className="flex items-center gap-1.5 bg-black/40 px-3 py-1.5 rounded-lg border border-white/5">
                <Cpu className="w-3.5 h-3.5" />
                <span>{data.is_demo ? "Demo Mode" : "Real Provider"}</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Sources Grid */}
      {data.sources?.length > 0 && (
        <div className="mt-8">
          <h4 className="text-sm font-semibold tracking-widest uppercase text-white/40 mb-4 px-2">Sources Context</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {data.sources.map((source, idx) => (
              <motion.div 
                key={idx}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.1 + (idx * 0.05) }}
                className="bg-white/5 border border-white/10 rounded-xl p-4 hover:bg-white/10 transition-colors"
              >
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-bold text-brand-400 bg-brand-400/10 px-2 py-0.5 rounded">
                    Source [{idx + 1}]
                  </span>
                  <span className="text-xs text-white/30 font-mono">
                    Score: {source.score.toFixed(3)}
                  </span>
                </div>
                <p className="text-sm text-white/60 line-clamp-4 leading-relaxed">
                  {source.text}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
}
