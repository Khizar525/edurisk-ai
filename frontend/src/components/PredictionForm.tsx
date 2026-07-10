"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { PredictionRequest } from "@/lib/types";

interface PredictionFormProps {
  onSubmit: (data: PredictionRequest) => void;
  isLoading: boolean;
}

const SLEEP_OPTIONS = [
  "Less than 5 hours",
  "5-6 hours",
  "7-8 hours",
  "More than 8 hours",
];

const DIET_OPTIONS = ["Healthy", "Moderate", "Unhealthy"];

function SliderInput({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  displayValue,
  lowLabel,
  highLabel,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  displayValue?: string;
  lowLabel?: string;
  highLabel?: string;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs text-text-dim">{label}</span>
        <span className="text-sm font-bold text-teal tabular-nums">
          {displayValue ?? value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 bg-border rounded-full appearance-none cursor-pointer
                   [&::-webkit-slider-thumb]:appearance-none
                   [&::-webkit-slider-thumb]:w-4
                   [&::-webkit-slider-thumb]:h-4
                   [&::-webkit-slider-thumb]:rounded-full
                   [&::-webkit-slider-thumb]:bg-teal
                   [&::-webkit-slider-thumb]:shadow-[0_0_8px_rgba(20,184,166,0.4)]
                   [&::-webkit-slider-thumb]:cursor-pointer
                   [&::-webkit-slider-thumb]:transition-shadow
                   [&::-webkit-slider-thumb]:hover:shadow-[0_0_12px_rgba(20,184,166,0.6)]"
      />
      {(lowLabel || highLabel) && (
        <div className="flex justify-between text-[10px] text-text-muted">
          <span>{lowLabel}</span>
          <span>{highLabel}</span>
        </div>
      )}
    </div>
  );
}

function ToggleGroup({
  label,
  options,
  value,
  onChange,
  activeColor = "teal",
}: {
  label: string;
  options: string[];
  value: string;
  onChange: (v: string) => void;
  activeColor?: string;
}) {
  const colorMap: Record<string, string> = {
    teal: "bg-teal text-white border-teal shadow-[0_0_8px_rgba(20,184,166,0.3)]",
    red: "bg-red text-white border-red shadow-[0_0_8px_rgba(239,68,68,0.3)]",
  };

  return (
    <div className="space-y-2">
      <span className="text-xs text-text-dim">{label}</span>
      <div className="flex gap-2">
        {options.map((opt) => (
          <button
            key={opt}
            onClick={() => onChange(opt)}
            className={`flex-1 py-2.5 px-3 rounded-xl text-sm font-semibold border transition-all duration-200 ${
              value === opt
                ? colorMap[activeColor]
                : "bg-surface text-text-dim border-border hover:border-border-light hover:bg-surface-hover"
            }`}
          >
            {opt}
          </button>
        ))}
      </div>
    </div>
  );
}

export function PredictionForm({ onSubmit, isLoading }: PredictionFormProps) {
  const [form, setForm] = useState<PredictionRequest>({
    gender: "Male",
    age: 20,
    academic_pressure: 3,
    cgpa: 2.5,
    study_satisfaction: 3,
    sleep_duration: "7-8 hours",
    dietary_habits: "Moderate",
    work_study_hours: 6,
    financial_stress: 2,
    family_history: "No",
    suicidal_thoughts: "No",
  });

  const update = (field: keyof PredictionRequest, value: unknown) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  return (
    <div className="space-y-6">
      {/* Section: Demographics */}
      <div>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-6 h-6 rounded-md bg-teal/10 flex items-center justify-center">
            <svg className="w-3.5 h-3.5 text-teal" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
          </div>
          <span className="text-xs uppercase tracking-[1.5px] text-text-dim font-semibold">
            Demographics
          </span>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <ToggleGroup
            label="Gender"
            options={["Male", "Female"]}
            value={form.gender}
            onChange={(v) => update("gender", v)}
          />
          <SliderInput
            label="Age"
            value={form.age}
            min={17}
            max={30}
            onChange={(v) => update("age", v)}
          />
        </div>
      </div>

      {/* Divider */}
      <div className="border-t border-border" />

      {/* Section: Academic */}
      <div>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-6 h-6 rounded-md bg-cyan/10 flex items-center justify-center">
            <svg className="w-3.5 h-3.5 text-cyan" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
          </div>
          <span className="text-xs uppercase tracking-[1.5px] text-text-dim font-semibold">
            Academic
          </span>
        </div>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <SliderInput
              label="Academic Pressure"
              value={form.academic_pressure}
              min={1}
              max={5}
              onChange={(v) => update("academic_pressure", v)}
              lowLabel="Low"
              highLabel="High"
            />
            <SliderInput
              label="CGPA"
              value={form.cgpa}
              min={0}
              max={4}
              step={0.1}
              onChange={(v) => update("cgpa", v)}
              displayValue={form.cgpa.toFixed(1)}
              lowLabel="0.0"
              highLabel="4.0"
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <SliderInput
              label="Study Satisfaction"
              value={form.study_satisfaction}
              min={1}
              max={5}
              onChange={(v) => update("study_satisfaction", v)}
              lowLabel="Dissatisfied"
              highLabel="Satisfied"
            />
            <SliderInput
              label="Work / Study Hours"
              value={form.work_study_hours}
              min={0}
              max={12}
              onChange={(v) => update("work_study_hours", v)}
              displayValue={`${form.work_study_hours}h`}
              lowLabel="0h"
              highLabel="12h"
            />
          </div>
        </div>
      </div>

      {/* Divider */}
      <div className="border-t border-border" />

      {/* Section: Lifestyle */}
      <div>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-6 h-6 rounded-md bg-yellow/10 flex items-center justify-center">
            <svg className="w-3.5 h-3.5 text-yellow" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
            </svg>
          </div>
          <span className="text-xs uppercase tracking-[1.5px] text-text-dim font-semibold">
            Lifestyle
          </span>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <span className="text-xs text-text-dim">Sleep Duration</span>
            <select
              value={form.sleep_duration}
              onChange={(e) => update("sleep_duration", e.target.value)}
              className="w-full py-2.5 px-3 rounded-xl bg-surface border border-border text-text text-sm
                         focus:border-teal focus:outline-none focus:ring-1 focus:ring-teal/30
                         transition-all duration-200 appearance-none cursor-pointer
                         bg-[url('data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2212%22%20height%3D%2212%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236b7280%22%20stroke-width%3D%222%22%3E%3Cpath%20d%3D%22M6%209l6%206%206-6%22%2F%3E%3C%2Fsvg%3E')]
                         bg-no-repeat bg-[right_12px_center]"
            >
              {SLEEP_OPTIONS.map((opt) => (
                <option key={opt} value={opt}>{opt}</option>
              ))}
            </select>
          </div>
          <div className="space-y-2">
            <span className="text-xs text-text-dim">Dietary Habits</span>
            <select
              value={form.dietary_habits}
              onChange={(e) => update("dietary_habits", e.target.value)}
              className="w-full py-2.5 px-3 rounded-xl bg-surface border border-border text-text text-sm
                         focus:border-teal focus:outline-none focus:ring-1 focus:ring-teal/30
                         transition-all duration-200 appearance-none cursor-pointer
                         bg-[url('data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2212%22%20height%3D%2212%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236b7280%22%20stroke-width%3D%222%22%3E%3Cpath%20d%3D%22M6%209l6%206%206-6%22%2F%3E%3C%2Fsvg%3E')]
                         bg-no-repeat bg-[right_12px_center]"
            >
              {DIET_OPTIONS.map((opt) => (
                <option key={opt} value={opt}>{opt}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Divider */}
      <div className="border-t border-border" />

      {/* Section: Risk Factors */}
      <div>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-6 h-6 rounded-md bg-red/10 flex items-center justify-center">
            <svg className="w-3.5 h-3.5 text-red" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <span className="text-xs uppercase tracking-[1.5px] text-text-dim font-semibold">
            Risk Indicators
          </span>
        </div>
        <div className="space-y-4">
          <SliderInput
            label="Financial Stress"
            value={form.financial_stress}
            min={1}
            max={5}
            onChange={(v) => update("financial_stress", v)}
            lowLabel="Low"
            highLabel="High"
          />
          <div className="grid grid-cols-2 gap-4">
            <ToggleGroup
              label="Family History of Mental Illness"
              options={["Yes", "No"]}
              value={form.family_history}
              onChange={(v) => update("family_history", v)}
            />
            <ToggleGroup
              label="Suicidal Thoughts"
              options={["Yes", "No"]}
              value={form.suicidal_thoughts}
              onChange={(v) => update("suicidal_thoughts", v)}
              activeColor="red"
            />
          </div>
        </div>
      </div>

      {/* Submit Button */}
      <motion.button
        onClick={() => onSubmit(form)}
        disabled={isLoading}
        className="w-full py-4 rounded-2xl text-white font-bold text-base tracking-wide
                   bg-gradient-to-r from-teal via-teal-dark to-teal
                   shadow-lg shadow-teal/20
                   hover:shadow-xl hover:shadow-teal/30 hover:brightness-110
                   disabled:opacity-50 disabled:cursor-not-allowed
                   transition-all duration-200
                   flex items-center justify-center gap-2"
        whileHover={{ scale: isLoading ? 1 : 1.01 }}
        whileTap={{ scale: isLoading ? 1 : 0.98 }}
      >
        {isLoading ? (
          <>
            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Analyzing...
          </>
        ) : (
          <>
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Analyze Risk
          </>
        )}
      </motion.button>
    </div>
  );
}
