"use client";

import { motion } from "framer-motion";
import { ShapFactor } from "@/lib/types";

interface FactorCardsProps {
  riskFactors: ShapFactor[];
  protectiveFactors: ShapFactor[];
}

export function FactorCards({ riskFactors, protectiveFactors }: FactorCardsProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* Risk factors */}
      <div>
        <div className="text-[11px] uppercase tracking-[1.5px] text-red font-semibold mb-3">
          ▲ Increases Risk
        </div>
        <div className="space-y-2">
          {riskFactors.length === 0 && (
            <div className="text-xs text-text-muted italic">None detected</div>
          )}
          {riskFactors.map((f, i) => (
            <motion.div
              key={f.feature}
              className="flex items-center gap-3 px-3 py-2.5 bg-surface rounded-lg border border-border"
              style={{ borderLeftColor: "#ef4444", borderLeftWidth: 3 }}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 + i * 0.08, duration: 0.3 }}
            >
              <div className="flex-1">
                <div className="text-sm text-text">{f.feature}</div>
                <div className="text-[11px] text-text-muted">
                  {Math.abs(f.impact) > 0.3
                    ? "Strongly"
                    : Math.abs(f.impact) > 0.1
                    ? "Moderately"
                    : "Slightly"}{" "}
                  increases risk
                </div>
              </div>
              <div className="text-xs font-mono text-red font-bold">
                +{f.impact.toFixed(3)}
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Protective factors */}
      <div>
        <div className="text-[11px] uppercase tracking-[1.5px] text-green font-semibold mb-3">
          ▼ Decreases Risk
        </div>
        <div className="space-y-2">
          {protectiveFactors.length === 0 && (
            <div className="text-xs text-text-muted italic">None detected</div>
          )}
          {protectiveFactors.map((f, i) => (
            <motion.div
              key={f.feature}
              className="flex items-center gap-3 px-3 py-2.5 bg-surface rounded-lg border border-border"
              style={{ borderLeftColor: "#22c55e", borderLeftWidth: 3 }}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 + i * 0.08, duration: 0.3 }}
            >
              <div className="flex-1">
                <div className="text-sm text-text">{f.feature}</div>
                <div className="text-[11px] text-text-muted">
                  {Math.abs(f.impact) > 0.3
                    ? "Strongly"
                    : Math.abs(f.impact) > 0.1
                    ? "Moderately"
                    : "Slightly"}{" "}
                  decreases risk
                </div>
              </div>
              <div className="text-xs font-mono text-green font-bold">
                {f.impact.toFixed(3)}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}
