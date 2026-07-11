export interface PredictionRequest {
  gender: string;
  age: number;
  academic_pressure: number;
  cgpa: number;
  study_satisfaction: number;
  sleep_duration: string;
  dietary_habits: string;
  work_study_hours: number;
  financial_stress: number;
  family_history: string;
  suicidal_thoughts: string;
}

export interface ShapFactor {
  feature: string;
  impact: number;
  direction: string;
}

export interface PredictionResponse {
  prediction: number;
  risk_level: string;
  risk_label: string;
  predicted_probability: string;
  predicted_probability_value: number;
  advice: string;
  probabilities: Record<string, number>;
  probabilities_raw: Record<string, number>;
  shap: {
    sorted_features: string[];
    shap_values: number[];
    bar_colors: string[];
    interpretation: string;
    risk_factors: ShapFactor[];
    protective_factors: ShapFactor[];
    top_risk: ShapFactor[];
    top_protective: ShapFactor[];
  };
}

export const RISK_COLORS: Record<number, string> = {
  0: "#22c55e",
  1: "#f59e0b",
  2: "#ef4444",
};

export const RISK_LABELS: Record<number, string> = {
  0: "LOW RISK",
  1: "MEDIUM RISK",
  2: "HIGH RISK",
};
