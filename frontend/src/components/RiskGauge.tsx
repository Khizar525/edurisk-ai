"use client";

import { motion } from "framer-motion";
import { RISK_COLORS, RISK_LABELS } from "@/lib/types";

interface RiskGaugeProps {
  prediction: number;
  confidence: number;
}

export function RiskGauge({ prediction, confidence }: RiskGaugeProps) {
  const color = RISK_COLORS[prediction];
  const label = RISK_LABELS[prediction];

  // Semicircle params
  const cx = 150;
  const cy = 130;
  const r = 100;
  const strokeWidth = 14;

  // Arc segments: Low (180-120), Medium (120-60), High (60-0)
  const segments = [
    { start: 180, end: 120, color: "#22c55e", label: "Low" },
    { start: 120, end: 60, color: "#f59e0b", label: "Med" },
    { start: 60, end: 0, color: "#ef4444", label: "High" },
  ];

  // Needle angle: map prediction to angle
  const needleAngles: Record<number, number> = { 0: 150, 1: 90, 2: 30 };
  const needleAngle = needleAngles[prediction];

  function polarToCartesian(angle: number) {
    const rad = (angle * Math.PI) / 180;
    return {
      x: cx + r * Math.cos(rad),
      y: cy - r * Math.sin(rad),
    };
  }

  function arcPath(startAngle: number, endAngle: number) {
    const start = polarToCartesian(startAngle);
    const end = polarToCartesian(endAngle);
    const largeArc = startAngle - endAngle > 180 ? 1 : 0;
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}`;
  }

  const needleRad = (needleAngle * Math.PI) / 180;
  const needleLen = r - 20;
  const needleX = cx + needleLen * Math.cos(needleRad);
  const needleY = cy - needleLen * Math.sin(needleRad);

  return (
    <div className="flex flex-col items-center">
      <svg width="300" height="180" viewBox="0 0 300 180">
        {/* Background segments */}
        {segments.map((seg, i) => (
          <path
            key={i}
            d={arcPath(seg.start, seg.end)}
            fill="none"
            stroke={seg.color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            opacity={0.2}
          />
        ))}

        {/* Active segment */}
        <motion.path
          d={arcPath(segments[prediction].start, segments[prediction].end)}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        />

        {/* Needle */}
        <motion.line
          x1={cx}
          y1={cy}
          x2={needleX}
          y2={needleY}
          stroke="white"
          strokeWidth={3}
          strokeLinecap="round"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.4 }}
        />

        {/* Center dot */}
        <circle cx={cx} cy={cy} r={5} fill="white" />

        {/* Side labels */}
        <text x={cx - r - 15} y={cy + 5} textAnchor="middle" fill="#6b7280" fontSize={11}>
          Low
        </text>
        <text x={cx} y={cy + 20} textAnchor="middle" fill="#6b7280" fontSize={11}>
          Med
        </text>
        <text x={cx + r + 15} y={cy + 5} textAnchor="middle" fill="#6b7280" fontSize={11}>
          High
        </text>
      </svg>

      {/* Risk label */}
      <motion.div
        className="text-center -mt-4"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6, duration: 0.4 }}
      >
        <div
          className="text-2xl font-extrabold tracking-widest"
          style={{ color }}
        >
          {label}
        </div>
        <div className="text-sm text-text-dim mt-1">
          {confidence.toFixed(1)}% confidence
        </div>
      </motion.div>
    </div>
  );
}
