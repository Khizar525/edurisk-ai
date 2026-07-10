"use client";

import { motion } from "framer-motion";

interface AdviceCardsProps {
  prediction: number;
}

const ADVICE: Record<number, { icon: string; text: string; color: string }[]> = {
  0: [
    { icon: "✓", text: "Maintain healthy study habits", color: "#22c55e" },
    { icon: "✓", text: "Keep balanced sleep schedule", color: "#22c55e" },
    { icon: "✓", text: "Continue peer support activities", color: "#22c55e" },
  ],
  1: [
    { icon: "⚠", text: "Reduce academic workload if possible", color: "#f59e0b" },
    { icon: "⚠", text: "Speak with faculty advisor", color: "#f59e0b" },
    { icon: "⚠", text: "Improve sleep schedule", color: "#f59e0b" },
    { icon: "⚠", text: "Consider counseling services", color: "#f59e0b" },
  ],
  2: [
    { icon: "!", text: "Reach out to university counselor ASAP", color: "#ef4444" },
    { icon: "!", text: "Reduce course load immediately", color: "#ef4444" },
    { icon: "!", text: "Contact student crisis hotline", color: "#ef4444" },
    { icon: "!", text: "Speak with trusted faculty member", color: "#ef4444" },
  ],
};

export function AdviceCards({ prediction }: AdviceCardsProps) {
  const items = ADVICE[prediction] || ADVICE[0];

  return (
    <div className="space-y-2">
      <div className="text-[11px] uppercase tracking-[1.5px] text-text-dim font-semibold">
        Recommendations
      </div>
      {items.map((item, i) => (
        <motion.div
          key={item.text}
          className="flex items-center gap-3 px-3 py-2.5 bg-surface rounded-lg"
          style={{ borderLeft: `3px solid ${item.color}` }}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 + i * 0.08, duration: 0.3 }}
        >
          <span
            className="text-sm font-bold"
            style={{ color: item.color }}
          >
            {item.icon}
          </span>
          <span className="text-sm text-text">{item.text}</span>
        </motion.div>
      ))}
    </div>
  );
}
