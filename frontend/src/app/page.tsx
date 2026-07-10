"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { PredictionForm } from "@/components/PredictionForm";
import { RiskGauge } from "@/components/RiskGauge";
import { ProbabilityBars } from "@/components/ProbabilityBars";
import { ShapWaterfall } from "@/components/ShapWaterfall";
import { FactorCards } from "@/components/FactorCards";
import { AdviceCards } from "@/components/AdviceCards";
import { predictRisk } from "@/lib/api";
import { PredictionRequest, PredictionResponse } from "@/lib/types";

export default function Home() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (data: PredictionRequest) => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await predictRisk(data);
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Prediction failed. Is the API running?");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-bg">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-border bg-bg/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {/* Logo mark */}
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-teal to-cyan flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-extrabold tracking-tight">
                  <span className="bg-gradient-to-r from-teal to-cyan bg-clip-text text-transparent">
                    EduRisk AI
                  </span>
                </h1>
                <p className="text-[11px] text-text-muted -mt-0.5">
                  Explainable Academic Risk Intelligence
                </p>
              </div>
            </div>
            <div className="hidden sm:flex items-center gap-2">
              <span className="px-2.5 py-1 rounded-lg bg-surface text-text-dim text-[11px] font-medium border border-border">
                Random Forest
              </span>
              <span className="px-2.5 py-1 rounded-lg bg-surface text-text-dim text-[11px] font-medium border border-border">
                ROC-AUC 94.9%
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* ═══ Left Column: Form ═══ */}
          <div className="bg-surface rounded-2xl border border-border p-6 lg:sticky lg:top-24 lg:self-start">
            <PredictionForm onSubmit={handleSubmit} isLoading={isLoading} />
          </div>

          {/* ═══ Right Column: Results ═══ */}
          <div className="space-y-6">
            <AnimatePresence mode="wait">
              {/* Empty state */}
              {!result && !error && !isLoading && (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="bg-surface rounded-2xl border border-border p-8 flex flex-col items-center justify-center min-h-[500px]"
                >
                  <div className="w-20 h-20 rounded-2xl bg-border/50 flex items-center justify-center mb-6">
                    <svg className="w-10 h-10 text-text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
                    </svg>
                  </div>
                  <p className="text-text-dim text-sm font-medium">AI Risk Assessment</p>
                  <p className="text-text-muted text-xs mt-1">
                    Fill in the student profile and click Analyze Risk
                  </p>
                </motion.div>
              )}

              {/* Loading state */}
              {isLoading && (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="bg-surface rounded-2xl border border-border p-8 flex flex-col items-center justify-center min-h-[500px]"
                >
                  <div className="relative mb-6">
                    <div className="w-16 h-16 border-4 border-border rounded-full" />
                    <div className="absolute top-0 left-0 w-16 h-16 border-4 border-teal border-t-transparent rounded-full animate-spin" />
                  </div>
                  <p className="text-text-dim text-sm font-medium">Analyzing risk factors...</p>
                  <p className="text-text-muted text-xs mt-1">Running SHAP explainability</p>
                </motion.div>
              )}

              {/* Error state */}
              {error && (
                <motion.div
                  key="error"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0 }}
                  className="bg-surface rounded-2xl border border-red/30 p-8"
                >
                  <div className="flex items-center gap-3 text-red">
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span className="text-sm font-medium">{error}</span>
                  </div>
                </motion.div>
              )}

              {/* Results */}
              {result && !isLoading && (
                <motion.div
                  key="result"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                  className="space-y-6"
                >
                  {/* Gauge + Confidence */}
                  <div className="bg-surface rounded-2xl border border-border p-6">
                    <div className="flex justify-center">
                      <RiskGauge
                        prediction={result.prediction}
                        confidence={result.confidence_value * 100}
                      />
                    </div>
                  </div>

                  {/* Probability + Advice */}
                  <div className="bg-surface rounded-2xl border border-border p-6">
                    <ProbabilityBars probabilities={result.probabilities} probabilitiesRaw={result.probabilities_raw} />
                  </div>

                  <div className="bg-surface rounded-2xl border border-border p-6">
                    <AdviceCards prediction={result.prediction} />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Factors + SHAP (full width, below fold) */}
            <AnimatePresence>
              {result && !isLoading && (
                <motion.div
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ delay: 0.4, duration: 0.5 }}
                  className="space-y-6"
                >
                  <div className="bg-surface rounded-2xl border border-border p-6">
                    <FactorCards
                      riskFactors={result.shap.top_risk}
                      protectiveFactors={result.shap.top_protective}
                    />
                  </div>

                  <div className="bg-surface rounded-2xl border border-border p-6">
                    <div className="text-[11px] uppercase tracking-[1.5px] text-text-dim font-semibold mb-5">
                      SHAP Feature Contributions
                    </div>
                    <ShapWaterfall
                      features={result.shap.sorted_features}
                      values={result.shap.shap_values}
                    />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-16">
        <div className="max-w-7xl mx-auto px-6 py-5 flex items-center justify-between text-xs text-text-muted">
          <span>Built for CSL 460 — Data Mining | Bahria University</span>
          <a
            href="https://github.com/Khizar525/edurisk-ai"
            target="_blank"
            rel="noopener noreferrer"
            className="text-teal hover:underline"
          >
            GitHub →
          </a>
        </div>
      </footer>
    </div>
  );
}
