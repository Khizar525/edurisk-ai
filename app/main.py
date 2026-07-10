"""
EduRisk AI — Gradio Application (Modern UI).

Professional dark-mode dashboard for student academic risk prediction.
Design inspired by OpenAI, Linear, Stripe, and Vercel.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
from PIL import Image

from src.inference.predictor import RiskPredictor
from src.inference.logger import PredictionLogger
from src.utils.validators import validate_prediction_inputs
from src.explainability.shap_utils import (
    compute_local_explanation,
    plot_local_waterfall,
    plot_probability_gauge,
)

# ── Initialize Services ────────────────────────────────────────
predictor = RiskPredictor()
logger = PredictionLogger()


# ── Semicircle Gauge ───────────────────────────────────────────
def create_gauge(prediction: int, confidence: float) -> Image.Image:
    """Create a modern semicircle risk gauge."""
    fig, ax = plt.subplots(figsize=(5, 3), facecolor="#111827")
    ax.set_facecolor("#111827")

    # Semicircle background segments
    colors = ["#22c55e", "#f59e0b", "#ef4444"]
    labels = ["Low", "Med", "High"]
    start_angles = [180, 120, 60]
    end_angles = [120, 60, 0]

    for i in range(3):
        wedge = mpatches.Wedge(
            center=(0.5, 0.4), r=0.4,
            theta1=start_angles[i], theta2=end_angles[i],
            facecolor=colors[i], alpha=0.25, edgecolor="none",
        )
        ax.add_patch(wedge)

    # Highlight the active segment
    wedge_active = mpatches.Wedge(
        center=(0.5, 0.4), r=0.4,
        theta1=start_angles[prediction], theta2=end_angles[prediction],
        facecolor=colors[prediction], alpha=0.9, edgecolor="none",
    )
    ax.add_patch(wedge_active)

    # Needle angle
    angle_map = {0: 150, 1: 90, 2: 30}
    needle_angle = np.radians(angle_map[prediction])
    needle_x = 0.5 + 0.32 * np.cos(needle_angle)
    needle_y = 0.4 + 0.32 * np.sin(needle_angle)
    ax.plot([0.5, needle_x], [0.4, needle_y], color="white", linewidth=2.5, zorder=5)
    ax.plot(0.5, 0.4, "o", color="white", markersize=6, zorder=6)

    # Labels
    risk_labels = {0: "LOW RISK", 1: "MEDIUM RISK", 2: "HIGH RISK"}
    risk_colors = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}

    ax.text(0.5, 0.82, risk_labels[prediction],
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=risk_colors[prediction], transform=ax.transAxes)

    ax.text(0.5, 0.70, f"{confidence:.0f}% confidence",
            ha="center", va="center", fontsize=11, color="#9ca3af",
            transform=ax.transAxes)

    # Side labels
    ax.text(0.08, 0.30, "Low", ha="center", fontsize=9, color="#6b7280")
    ax.text(0.50, 0.08, "Med", ha="center", fontsize=9, color="#6b7280")
    ax.text(0.92, 0.30, "High", ha="center", fontsize=9, color="#6b7280")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                facecolor="#111827", edgecolor="none")
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    buf.close()
    return img


# ── Probability Bars ───────────────────────────────────────────
def create_prob_bars(probs: dict) -> Image.Image:
    """Create modern horizontal probability bars."""
    fig, ax = plt.subplots(figsize=(4.5, 2.2), facecolor="#111827")
    ax.set_facecolor("#111827")

    labels = list(probs.keys())
    values = list(probs.values())
    colors = ["#ef4444", "#f59e0b", "#22c55e"]
    display_labels = ["High", "Medium", "Low"]

    for i, (label, val, color, disp) in enumerate(zip(labels, values, colors, display_labels)):
        y = 2 - i
        # Background bar
        ax.barh(y, 100, height=0.5, color="#1f2937", edgecolor="none")
        # Filled bar
        ax.barh(y, val, height=0.5, color=color, edgecolor="none", alpha=0.9)
        # Label
        ax.text(-2, y, disp, ha="right", va="center", fontsize=10, color="#d1d5db")
        # Percentage
        ax.text(val + 1.5, y, f"{val:.1f}%", ha="left", va="center",
                fontsize=11, fontweight="bold", color=color)

    ax.set_xlim(-15, 115)
    ax.set_ylim(0.2, 3.2)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                facecolor="#111827", edgecolor="none")
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    buf.close()
    return img


# ── Prediction Function ────────────────────────────────────────

def predict_risk(
    gender, age, academic_pressure, cgpa, study_satisfaction,
    sleep_duration, dietary_habits, work_study_hours,
    financial_stress, family_history, suicidal_thoughts,
):
    """Main prediction function — returns all outputs."""
    # Validate
    is_valid, error_msg = validate_prediction_inputs(
        gender, age, academic_pressure, cgpa, study_satisfaction,
        sleep_duration, dietary_habits, work_study_hours,
        financial_stress, family_history, suicidal_thoughts,
    )
    if not is_valid:
        return (
            None,  # gauge
            f"Error: {error_msg}",
            "", "", "", "", "", "",
            None,  # shap waterfall
        )

    inputs = {
        "gender": gender, "age": age, "academic_pressure": academic_pressure,
        "cgpa": cgpa, "study_satisfaction": study_satisfaction,
        "sleep_duration": sleep_duration, "dietary_habits": dietary_habits,
        "work_study_hours": work_study_hours, "financial_stress": financial_stress,
        "family_history": family_history, "suicidal_thoughts": suicidal_thoughts,
    }

    result = predictor.predict(**inputs)
    logger.log(inputs, result)

    pred = result["prediction"]
    confidence = result["confidence_value"] * 100

    # ── Gauge Image ────────────────────────────────────────────
    gauge_img = create_gauge(pred, confidence)

    # ── Risk Badge ─────────────────────────────────────────────
    risk_colors = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}
    risk_names = {0: "LOW RISK", 1: "MEDIUM RISK", 2: "HIGH RISK"}
    risk_badge = f"""<div style="text-align:center; padding:12px 0;">
        <span style="font-size:28px; font-weight:800; color:{risk_colors[pred]}; letter-spacing:2px;">
            {risk_names[pred]}
        </span>
    </div>"""

    # ── Confidence ─────────────────────────────────────────────
    conf_bar_len = int(confidence / 2.5)
    confidence_html = f"""<div style="padding:8px 0;">
        <div style="color:#9ca3af; font-size:12px; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">Confidence</div>
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="flex:1; height:8px; background:#1f2937; border-radius:4px; overflow:hidden;">
                <div style="width:{confidence}%; height:100%; background:{risk_colors[pred]}; border-radius:4px;"></div>
            </div>
            <span style="font-size:20px; font-weight:700; color:{risk_colors[pred]};">{confidence:.1f}%</span>
        </div>
    </div>"""

    # ── Recommendation Cards ───────────────────────────────────
    advice_map = {
        0: [
            ("✓", "Maintain healthy study habits", "#22c55e"),
            ("✓", "Keep balanced sleep schedule", "#22c55e"),
            ("✓", "Continue peer support activities", "#22c55e"),
        ],
        1: [
            ("⚠", "Reduce academic workload if possible", "#f59e0b"),
            ("⚠", "Speak with faculty advisor", "#f59e0b"),
            ("⚠", "Improve sleep schedule", "#f59e0b"),
            ("⚠", "Consider counseling services", "#f59e0b"),
        ],
        2: [
            ("!", "Reach out to university counselor ASAP", "#ef4444"),
            ("!", "Reduce course load immediately", "#ef4444"),
            ("!", "Contact student crisis hotline", "#ef4444"),
            ("!", "Speak with trusted faculty member", "#ef4444"),
        ],
    }
    advice_items = advice_map[pred]
    advice_html = '<div style="display:flex; flex-direction:column; gap:6px;">'
    for icon, text, color in advice_items:
        advice_html += f"""<div style="display:flex; align-items:center; gap:10px; padding:8px 12px; background:#1f2937; border-radius:8px; border-left:3px solid {color};">
            <span style="font-size:14px; font-weight:700; color:{color};">{icon}</span>
            <span style="font-size:13px; color:#d1d5db;">{text}</span>
        </div>"""
    advice_html += "</div>"

    # ── Top Factors ────────────────────────────────────────────
    shap_data = result["shap"]
    risk_factors = shap_data.get("top_risk", [])[:3]
    protective_factors = shap_data.get("top_protective", [])[:3]

    factors_html = '<div style="display:flex; gap:16px;">'

    # Risk factors
    factors_html += '<div style="flex:1;"><div style="color:#ef4444; font-size:11px; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; font-weight:600;">▲ Increases Risk</div>'
    for f in risk_factors:
        strength = "strongly" if abs(f["impact"]) > 0.3 else "moderately" if abs(f["impact"]) > 0.1 else "slightly"
        factors_html += f"""<div style="padding:6px 10px; margin-bottom:4px; background:#1f2937; border-radius:6px; border-left:2px solid #ef4444;">
            <span style="color:#d1d5db; font-size:12px;">{f['feature']}</span>
            <span style="color:#6b7280; font-size:11px; margin-left:6px;">{strength}</span>
        </div>"""
    factors_html += '</div>'

    # Protective factors
    factors_html += '<div style="flex:1;"><div style="color:#22c55e; font-size:11px; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; font-weight:600;">▼ Decreases Risk</div>'
    for f in protective_factors:
        strength = "strongly" if abs(f["impact"]) > 0.3 else "moderately" if abs(f["impact"]) > 0.1 else "slightly"
        factors_html += f"""<div style="padding:6px 10px; margin-bottom:4px; background:#1f2937; border-radius:6px; border-left:2px solid #22c55e;">
            <span style="color:#d1d5db; font-size:12px;">{f['feature']}</span>
            <span style="color:#6b7280; font-size:11px; margin-left:6px;">{strength}</span>
        </div>"""
    factors_html += '</div></div>'

    # ── AI Interpretation ──────────────────────────────────────
    interpretation = shap_data.get("interpretation", "")
    interp_html = f"""<div style="padding:12px 16px; background:#1f2937; border-radius:10px; border:1px solid #374151;">
        <div style="color:#9ca3af; font-size:11px; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">AI Explanation</div>
        <div style="color:#d1d5db; font-size:13px; line-height:1.6;">
            The prediction was primarily driven by
            <strong style="color:#f59e0b;">{risk_factors[0]['feature'] if risk_factors else 'N/A'}</strong>
            and <strong style="color:#f59e0b;">{risk_factors[1]['feature'] if len(risk_factors) > 1 else 'N/A'}</strong>.
            {'The high academic pressure and suicidal thoughts significantly increase the risk score.' if pred == 2 else 'Moderate risk indicators detected across multiple factors.' if pred == 1 else 'Protective factors outweigh risk indicators.'}
        </div>
    </div>"""

    # ── SHAP Waterfall ─────────────────────────────────────────
    shap_img = None
    try:
        fig = plot_local_waterfall(
            shap_data["sorted_features"],
            shap_data["shap_values"],
            pred,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor="#111827", edgecolor="none")
        buf.seek(0)
        shap_img = Image.open(buf).copy()
        plt.close(fig)
        buf.close()
    except Exception as e:
        print(f"SHAP waterfall error: {e}")

    return (
        gauge_img,
        risk_badge,
        confidence_html,
        advice_html,
        factors_html,
        interp_html,
        "",  # placeholder for probability (we use HTML below)
        "",  # placeholder
        shap_img,
    )


# ── Analytics Function ─────────────────────────────────────────

def get_analytics():
    """Get prediction analytics."""
    analytics = logger.get_analytics()

    if analytics["total"] == 0:
        return "No predictions yet."

    lines = [
        f"**Total Predictions:** {analytics['total']}",
        f"**Average Confidence:** {analytics['avg_confidence']}%",
        "",
        "**Risk Distribution:**",
    ]
    for level, count in analytics.get("risk_distribution", {}).items():
        pct = count / analytics["total"] * 100
        lines.append(f"- {level}: {count} ({pct:.1f}%)")

    return "\n".join(lines)


def export_predictions():
    """Export predictions to a downloadable CSV."""
    export_path = PROJECT_ROOT / "data" / "prediction_export.csv"
    result = logger.export_csv(export_path)
    if result:
        return str(result)
    return "No predictions to export."


# ── Custom CSS ─────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Global ─────────────────────────────────────────────── */
.gradio-container {
    max-width: 1280px !important;
    background: #0a0a0f !important;
}
.dark { background: #0a0a0f !important; }

/* ── Header ─────────────────────────────────────────────── */
.header-title {
    font-size: 2.2em !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #14b8a6, #38bdf8) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 0 !important;
    letter-spacing: -0.5px;
}
.header-sub {
    color: #6b7280 !important;
    font-size: 0.95em !important;
    margin-top: 4px !important;
}
.header-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    background: #1f2937;
    color: #9ca3af;
    font-size: 12px;
    border: 1px solid #374151;
}

/* ── Panels ─────────────────────────────────────────────── */
.input-panel {
    background: #111827 !important;
    border-radius: 16px !important;
    border: 1px solid #1f2937 !important;
    padding: 24px !important;
}
.output-panel {
    background: #111827 !important;
    border-radius: 16px !important;
    border: 1px solid #1f2937 !important;
    padding: 24px !important;
}
.section-label {
    color: #9ca3af !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    font-weight: 600 !important;
    margin-bottom: 12px !important;
}

/* ── Inputs ─────────────────────────────────────────────── */
.gradio-group { background: transparent !important; }
.gr-input, .gr-textbox, .gr-dropdown, .gr-radio {
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 10px !important;
    color: #e5e7eb !important;
}
.gr-input:focus, .gr-textbox:focus, .gr-dropdown:focus {
    border-color: #14b8a6 !important;
    box-shadow: 0 0 0 2px rgba(20, 184, 166, 0.2) !important;
}
.gr-radio label {
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    color: #d1d5db !important;
}
.gr-radio input:checked + label {
    background: #14b8a6 !important;
    color: white !important;
    border-color: #14b8a6 !important;
}

/* ── Button ─────────────────────────────────────────────── */
.predict-btn {
    background: linear-gradient(135deg, #14b8a6, #0d9488) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 14px 28px !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(20, 184, 166, 0.3) !important;
}
.predict-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(20, 184, 166, 0.4) !important;
}

/* ── Tabs ───────────────────────────────────────────────── */
.tabs > .tab-nav > button {
    background: transparent !important;
    color: #6b7280 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
}
.tabs > .tab-nav > button.selected {
    color: #14b8a6 !important;
    border-bottom-color: #14b8a6 !important;
}

/* ── Markdown ───────────────────────────────────────────── */
.md-container { background: transparent !important; }
"""

# ── UI Layout ──────────────────────────────────────────────────

with gr.Blocks(
    title="EduRisk AI",
    theme=gr.themes.Base(
        primary_hue="teal",
        secondary_hue="slate",
        neutral_hue="gray",
    ).set(
        body_background_fill="#0a0a0f",
        body_background_fill_dark="#0a0a0f",
        block_background_fill="#111827",
        block_background_fill_dark="#111827",
        block_border_color="#1f2937",
        block_border_color_dark="#1f2937",
        block_label_text_color="#9ca3af",
        block_label_text_color_dark="#9ca3af",
        block_title_text_color="#e5e7eb",
        block_title_text_color_dark="#e5e7eb",
        input_background_fill="#1f2937",
        input_background_fill_dark="#1f2937",
        input_border_color="#374151",
        input_border_color_dark="#374151",
        input_text_color="#e5e7eb",
        input_text_color_dark="#e5e7eb",
        button_primary_background_fill="#14b8a6",
        button_primary_background_fill_dark="#14b8a6",
        button_primary_text_color="white",
        button_primary_text_color_dark="white",
        checkbox_background_color="#1f2937",
        checkbox_background_color_dark="#1f2937",
        checkbox_border_color="#374151",
        checkbox_border_color_dark="#374151",
        slider_color="#14b8a6",
        slider_color_dark="#14b8a6",
        font=["Inter", "system-ui", "sans-serif"],
    ),
    css=CUSTOM_CSS,
) as app:

    # ── Header ─────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; padding:24px 0 16px;">
        <div class="header-title">EduRisk AI</div>
        <div class="header-sub">Explainable Academic Risk Intelligence</div>
        <div style="margin-top:10px;">
            <span class="header-badge">Random Forest</span>
            <span class="header-badge">ROC-AUC 94.9%</span>
            <span class="header-badge">3-Class Risk</span>
        </div>
    </div>
    """)

    with gr.Tabs():
        # ═══ TAB 1: PREDICT ═══════════════════════════════════
        with gr.Tab("Predict", id="predict"):
            with gr.Row():
                # ── Left: Input Panel ──────────────────────────
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-label">Student Profile</div>')

                    with gr.Row():
                        gender = gr.Radio(
                            ["Male", "Female"], label="Gender", value="Male",
                        )
                        age = gr.Slider(17, 30, value=20, step=1, label="Age")

                    with gr.Row():
                        academic_pressure = gr.Slider(1, 5, value=3, step=1, label="Academic Pressure")
                        cgpa = gr.Slider(0.0, 4.0, value=2.5, step=0.1, label="CGPA")

                    with gr.Row():
                        study_satisfaction = gr.Slider(1, 5, value=3, step=1, label="Study Satisfaction")
                        work_study_hours = gr.Slider(0, 12, value=6, step=1, label="Work / Study Hours")

                    with gr.Row():
                        sleep_duration = gr.Dropdown(
                            ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"],
                            value="7-8 hours", label="Sleep Duration",
                        )
                        dietary_habits = gr.Dropdown(
                            ["Healthy", "Moderate", "Unhealthy"],
                            value="Moderate", label="Dietary Habits",
                        )

                    with gr.Row():
                        financial_stress = gr.Slider(1, 5, value=2, step=1, label="Financial Stress")
                        family_history = gr.Radio(["Yes", "No"], label="Family History of Mental Illness", value="No")
                        suicidal_thoughts = gr.Radio(["Yes", "No"], label="Suicidal Thoughts?", value="No")

                    predict_btn = gr.Button(
                        "Analyze Risk",
                        variant="primary",
                        size="lg",
                        elem_classes=["predict-btn"],
                    )

                # ── Right: Output Panel ───────────────────────
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-label">AI Risk Assessment</div>')

                    gauge_out = gr.Image(label="", type="pil", interactive=False, height=220)
                    risk_out = gr.HTML(label="")
                    confidence_out = gr.HTML(label="")

                    gr.HTML('<div style="margin-top:12px;"><div class="section-label">Recommendations</div></div>')
                    advice_out = gr.HTML(label="")

            # ── Full Width: Factors + SHAP ─────────────────────
            gr.HTML('<div style="margin-top:20px;"><div class="section-label">Top Contributing Factors</div></div>')
            factors_out = gr.HTML(label="")

            gr.HTML('<div style="margin-top:16px;"><div class="section-label">AI Explanation</div></div>')
            interp_out = gr.HTML(label="")

            gr.HTML('<div style="margin-top:16px;"><div class="section-label">SHAP Visualization</div></div>')
            shap_out = gr.Image(label="", type="pil", interactive=False, height=350)

            # Wire up prediction
            predict_btn.click(
                fn=predict_risk,
                inputs=[
                    gender, age, academic_pressure, cgpa, study_satisfaction,
                    sleep_duration, dietary_habits, work_study_hours,
                    financial_stress, family_history, suicidal_thoughts,
                ],
                outputs=[
                    gauge_out, risk_out, confidence_out,
                    advice_out, factors_out, interp_out,
                    gr.Textbox(visible=False), gr.Textbox(visible=False),
                    shap_out,
                ],
            )

        # ═══ TAB 2: ABOUT ═════════════════════════════════════
        with gr.Tab("About", id="about"):
            gr.Markdown("""
## About EduRisk AI

**EduRisk AI** is a machine learning system that predicts the academic risk level
of university students based on self-reported lifestyle, psychological, and
academic indicators.

### How It Works

1. **Input** — Enter student profile data (demographics, academic metrics, lifestyle)
2. **Model** — A trained Random Forest classifier analyzes the input
3. **SHAP** — Per-prediction explanations show which factors drove the decision
4. **Output** — Risk level (Low / Medium / High) with confidence and recommendations

### Model Performance

| Metric | Value |
|--------|-------|
| Model | Random Forest (100 trees, max_depth=15) |
| Accuracy | 85.58% |
| ROC-AUC | 94.92% |
| Features | 11 (demographic, academic, lifestyle) |
| Target | 3-class risk level |
| Validation | 3-fold stratified cross-validation |

### Team

| Name | Role |
|------|------|
| M. Khizar Akram | App & Deployment (Team Lead) |
| Safwan Marwat | Data Collection & EDA |
| Syed Mughees | Preprocessing & Feature Engineering |
| Ifrahim Yousuf | Model Training & Evaluation |

### Links

- [GitHub Repository](https://github.com/Khizar525/edurisk-ai)
- [Dataset (Kaggle)](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)

---
*Built for CSL 460 — Data Mining | Bahria University Karachi Campus*
""")


if __name__ == "__main__":
    app.launch(share=False)
