"""
EduRisk AI — Gradio Application Entry Point.

Launch the student academic risk prediction dashboard.
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from PIL import Image

from src.inference.predictor import RiskPredictor
from src.inference.logger import PredictionLogger
from src.utils.validators import validate_prediction_inputs
from src.explainability.shap_utils import compute_local_explanation

# ── Initialize Services ────────────────────────────────────────
predictor = RiskPredictor()
logger = PredictionLogger()


def predict_risk(
    gender, age, academic_pressure, cgpa, study_satisfaction,
    sleep_duration, dietary_habits, work_study_hours,
    financial_stress, family_history, suicidal_thoughts,
):
    """Main prediction function called by Gradio."""
    # Validate inputs
    is_valid, error_msg = validate_prediction_inputs(
        gender, age, academic_pressure, cgpa, study_satisfaction,
        sleep_duration, dietary_habits, work_study_hours,
        financial_stress, family_history, suicidal_thoughts,
    )
    if not is_valid:
        return f"❌ {error_msg}", "", "", None

    # Make prediction
    inputs = {
        "gender": gender, "age": age, "academic_pressure": academic_pressure,
        "cgpa": cgpa, "study_satisfaction": study_satisfaction,
        "sleep_duration": sleep_duration, "dietary_habits": dietary_habits,
        "work_study_hours": work_study_hours, "financial_stress": financial_stress,
        "family_history": family_history, "suicidal_thoughts": suicidal_thoughts,
    }

    result = predictor.predict(**inputs)

    # Log prediction
    logger.log(inputs, result)

    # Generate SHAP visualization
    shap_img = None
    try:
        sorted_feats, sorted_vals, bar_colors = compute_local_explanation(
            predictor.explainer,
            result["input_scaled"],
            predictor.feature_names,
            result["prediction"],
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(sorted_feats, sorted_vals, color=bar_colors, edgecolor="none", height=0.6)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_xlabel("SHAP Value (Higher = Increases predicted risk)", fontsize=9)
        ax.set_title(f"Why the model predicted: {result['risk_level']}", fontsize=11)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130)
        buf.seek(0)
        shap_img = Image.open(buf).copy()
        plt.close(fig)
        buf.close()
    except Exception as e:
        print(f"SHAP visualization error: {e}")

    # Format probability output
    prob_text = "\n".join(
        f"  {k}: {v}%" for k, v in result["probabilities"].items()
    )
    confidence_text = f"{result['confidence']}\n\n{prob_text}"

    return result["risk_level"], confidence_text, result["advice"], shap_img


# ── UI Layout ──────────────────────────────────────────────────
with gr.Blocks(
    title="EduRisk AI — Student Risk Predictor",
    theme=gr.themes.Soft(),
) as app:
    gr.Markdown(
        "# EduRisk AI\n"
        "### Student Academic Risk Predictor\n"
        "Predicting academic risk before it becomes a crisis."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gender = gr.Radio(["Male", "Female"], label="Gender", value="Male")
            age = gr.Slider(17, 30, value=20, step=1, label="Age")

        with gr.Column(scale=1):
            cgpa = gr.Slider(0.0, 4.0, value=2.5, step=0.1, label="CGPA (0.0 – 4.0)")
            academic_pressure = gr.Slider(1, 5, value=3, step=1, label="Academic Pressure (1-5)")

    with gr.Row():
        with gr.Column(scale=1):
            study_satisfaction = gr.Slider(1, 5, value=3, step=1, label="Study Satisfaction (1–5)")
            work_study_hours = gr.Slider(0, 12, value=6, step=1, label="Work/Study Hours per Day")

        with gr.Column(scale=1):
            sleep_duration = gr.Dropdown(
                ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"],
                value="7-8 hours", label="Sleep Duration",
            )
            dietary_habits = gr.Dropdown(
                ["Healthy", "Moderate", "Unhealthy"],
                value="Moderate", label="Dietary Habits",
            )

    with gr.Row():
        financial_stress = gr.Slider(1, 5, value=2, step=1, label="Financial Stress (1–5)")
        family_history = gr.Radio(["Yes", "No"], label="Family History of Mental Illness", value="No")
        suicidal_thoughts = gr.Radio(["Yes", "No"], label="Ever had Suicidal Thoughts?", value="No")

    btn = gr.Button("🔍 Predict Risk", variant="primary", size="lg")

    with gr.Row():
        risk_out = gr.Textbox(label="Risk Level", interactive=False)
        conf_out = gr.Textbox(label="Confidence & Probabilities", interactive=False, lines=5)
    advice_out = gr.Textbox(label="Recommendation", interactive=False)
    shap_out = gr.Image(label="Feature Contributions (SHAP)", type="pil", interactive=False)

    btn.click(
        fn=predict_risk,
        inputs=[
            gender, age, academic_pressure, cgpa, study_satisfaction,
            sleep_duration, dietary_habits, work_study_hours,
            financial_stress, family_history, suicidal_thoughts,
        ],
        outputs=[risk_out, conf_out, advice_out, shap_out],
    )


if __name__ == "__main__":
    app.launch(share=True)
