"use client";

import { motion } from "framer-motion";

interface ProbabilityBarsProps {
  probabilities: Record<string, number>;
  probabilitiesRaw?: Record<string, number>;
}

const DISPLAY_ORDER = [
  { key: 2, label: "High", color: "#ef4444" },
  { key: 1, label: "Medium", color: "#f59e0b" },
  { key: 0, label: "Low", color: "#22c55e" },
];

export function ProbabilityBars({ probabilities, probabilitiesRaw }: ProbabilityBarsProps) {
  // Use probabilitiesRaw (0-1 decimals) if available, otherwise parse emoji keys (already percentages)
  const values = DISPLAY_ORDER.map((item) => {
    if (probabilitiesRaw) {
      const val = probabilitiesRaw[item.key] ?? probabilitiesRaw[String(item.key)];
      if (val !== undefined) {
        // values are 0-1 decimals from the API, convert to percentage
        return { ...item, value: val * 100 };
      }
    }
    // Fallback: parse emoji keys (already in percentage)
    const match = Object.entries(probabilities).find(([k]) => {
      const num = parseInt(k);
      return num === item.key;
    });
    return { ...item, value: match ? match[1] : 0 };
  });

  return (
    <div className="space-y-3">
      <div className="text-[11px] uppercase tracking-[1.5px] text-text-dim font-semibold">
        Probability
      </div>
      {values.map((item, i) => (
        <div key={item.key} className="flex items-center gap-3">
          <span className="w-14 text-right text-sm text-text-dim">
            {item.label}
          </span>
          <div className="flex-1 h-6 bg-border rounded-md overflow-hidden">
            <motion.div
              className="h-full rounded-md"
              style={{ backgroundColor: item.color }}
              initial={{ width: 0 }}
              animate={{ width: `${item.value}%` }}
              transition={{ delay: 0.3 + i * 0.15, duration: 0.6, ease: "easeOut" }}
            />
          </div>
          <span
            className="w-14 text-right text-sm font-bold"
            style={{ color: item.color }}
          >
            {item.value.toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}
