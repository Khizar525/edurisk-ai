"use client";

import { motion } from "framer-motion";

interface ShapWaterfallProps {
  features: string[];
  values: number[];
  maxBars?: number;
}

export function ShapWaterfall({
  features,
  values,
  maxBars = 10,
}: ShapWaterfallProps) {
  const n = Math.min(maxBars, features.length);
  const displayFeatures = features.slice(0, n).reverse();
  const displayValues = values.slice(0, n).reverse();

  const maxAbs = Math.max(...displayValues.map(Math.abs), 0.01);

  return (
    <div className="space-y-2">
      {displayFeatures.map((feat, i) => {
        const val = displayValues[i];
        const isPositive = val > 0;
        const barWidth = (Math.abs(val) / maxAbs) * 100;
        const color = isPositive ? "#ef4444" : "#22c55e";

        // Nice feature name
        const niceName = feat
          .replace("Have you ever had suicidal thoughts ?", "Suicidal Thoughts")
          .replace("Family History of Mental Illness", "Family History")
          .replace("Work/Study Hours", "Work/Study Hrs")
          .replace("Academic Pressure", "Academic Pressure")
          .replace("Study Satisfaction", "Study Satisfaction")
          .replace("Financial Stress", "Financial Stress")
          .replace("Sleep Duration", "Sleep Duration")
          .replace("Dietary Habits", "Dietary Habits");

        return (
          <div key={feat} className="flex items-center gap-3">
            <span className="w-36 text-right text-xs text-text-dim truncate">
              {niceName}
            </span>
            <div className="flex-1 flex items-center">
              <div className="w-1/2 flex justify-end">
                {!isPositive && (
                  <motion.div
                    className="h-5 rounded-l-sm"
                    style={{ backgroundColor: color, opacity: 0.8 }}
                    initial={{ width: 0 }}
                    animate={{ width: `${barWidth}%` }}
                    transition={{
                      delay: 0.1 + i * 0.05,
                      duration: 0.5,
                      ease: "easeOut",
                    }}
                  />
                )}
              </div>
              <div className="w-px h-5 bg-border-light" />
              <div className="w-1/2">
                {isPositive && (
                  <motion.div
                    className="h-5 rounded-r-sm"
                    style={{ backgroundColor: color, opacity: 0.8 }}
                    initial={{ width: 0 }}
                    animate={{ width: `${barWidth}%` }}
                    transition={{
                      delay: 0.1 + i * 0.05,
                      duration: 0.5,
                      ease: "easeOut",
                    }}
                  />
                )}
              </div>
            </div>
            <span
              className="w-16 text-right text-xs font-mono font-bold"
              style={{ color }}
            >
              {val > 0 ? "+" : ""}
              {val.toFixed(3)}
            </span>
          </div>
        );
      })}
      <div className="flex justify-between text-[10px] text-text-muted mt-1 px-[164px]">
        <span>← Decreases Risk</span>
        <span>Increases Risk →</span>
      </div>
    </div>
  );
}
