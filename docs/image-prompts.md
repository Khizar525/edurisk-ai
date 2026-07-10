# Image Generation Prompts

Generate these images using Gemini, DALL-E, Midjourney, or any image generator.
Save all outputs to: `assets/images/`

---

## 1. Project Banner

**File:** `assets/images/banner.png`
**Use in:** Top of `README.md`

```
A clean, modern tech banner for a machine learning project called "EduRisk AI".
The banner shows a minimalist illustration of a university campus silhouette on the
left, transitioning into a neural network / data visualization on the right.
Color palette: deep navy blue (#1a1a2e) background, teal (#16a085) and coral (#e74c3c)
accent colors. The style is flat design, professional, suitable for a GitHub README.
Wide aspect ratio (1200x400px). No text on the image.
```

---

## 2. System Architecture Diagram

**File:** `assets/images/architecture.png`
**Use in:** `docs/architecture.md` and `README.md`

```
A professional system architecture diagram for a machine learning project.
Five horizontal layers stacked top to bottom, each in a rounded rectangle with
a distinct color:

Top layer (blue): "Data Layer" — icons for CSV file, database, API
Second layer (green): "Processing Layer" — icons for cleaning brush, gears,
    encoding symbol
Third layer (orange): "Training Layer" — icons for neural network, chart,
    magnifying glass
Fourth layer (purple): "Inference Layer" — icons for brain, explanation bubble,
    log file
Bottom layer (red): "Presentation Layer" — icons for web dashboard, REST API,
    mobile phone

Arrows flow downward between layers. Clean flat design, white background,
professional infographic style. No code, no text labels beyond layer names.
```

---

## 3. Data Flow Pipeline

**File:** `assets/images/data-flow.png`
**Use in:** `docs/architecture.md` and `README.md`

```
A horizontal pipeline diagram showing a machine learning data flow with 7 steps
connected by arrows, left to right:

Step 1: "Load" — database icon
Step 2: "Impute" — broom/cleaning icon
Step 3: "Encode" — lock/key icon
Step 4: "Risk Target" — target/bullseye icon
Step 5: "Split" — scissors icon, highlighted in red border (critical step)
Step 6: "Scale" — ruler icon, highlighted in green border (no leakage step)
Step 7: "Train" — rocket icon

Steps 5 and 6 have a warning symbol between them. The split step shows data
diverging into two paths (train and test). Clean flat design, white background,
colorful but professional. No code.
```

---

## 4. Model Comparison Chart

**File:** `assets/images/model-comparison.png`
**Use in:** `docs/results.md` and `README.md`

```
A professional bar chart comparing four machine learning models side by side.
The chart shows Accuracy and ROC-AUC as grouped bars for each model.

Models (x-axis): Random Forest, XGBoost, MLP, SVM
Values (y-axis): 0.0 to 1.0

Approximate values:
- Random Forest: Accuracy 0.856, ROC-AUC 0.949
- XGBoost: Accuracy 0.852, ROC-AUC 0.950
- MLP: Accuracy 0.852, ROC-AUC 0.947
- SVM: Accuracy 0.821, ROC-AUC 0.931

Style: Clean modern chart, teal bars for Accuracy, coral bars for ROC-AUC.
White background, subtle grid lines, legend in top right. Professional
infographic style, not a matplotlib screenshot.
```

---

## 5. Risk Level Breakdown

**File:** `assets/images/risk-breakdown.png`
**Use in:** `docs/results.md` and `README.md`

```
A donut chart / pie chart showing three risk level categories with percentages:

- Low Risk (green, #2ecc71): 37.5%
- Medium Risk (orange, #f39c12): 21.8%
- High Risk (red, #e74c3c): 40.7%

Center of the donut shows "27,901 students". Clean flat design, white background,
professional data visualization style. Each segment has a subtle shadow. Legend
on the right side. No code, no extra text.
```

---

## 6. SHAP Feature Importance

**File:** `assets/images/shap-importance.png`
**Use in:** `docs/results.md` and `README.md`

```
A horizontal bar chart showing feature importance from a SHAP analysis.
Features listed on the y-axis, importance values on the x-axis (0 to 1.0).

Features ranked top to bottom:
1. Academic Pressure — bar length ~0.95 (coral red)
2. Suicidal Thoughts — bar length ~0.85 (coral red)
3. CGPA — bar length ~0.75 (orange)
4. Financial Stress — bar length ~0.60 (orange)
5. Study Satisfaction — bar length ~0.50 (teal)
6. Sleep Duration — bar length ~0.45 (teal)
7. Work Study Hours — bar length ~0.40 (blue)
8. Dietary Habits — bar length ~0.30 (blue)
9. Age — bar length ~0.25 (gray)
10. Gender — bar length ~0.15 (gray)
11. Family History — bar length ~0.10 (gray)

Style: Clean flat design, gradient from red (high importance) to gray (low).
White background, professional infographic. No code.
```

---

## 7. Confusion Matrix

**File:** `assets/images/confusion-matrix.png`
**Use in:** `docs/results.md`

```
A 3x3 confusion matrix heatmap for a machine learning classifier.
Rows are "True" labels, columns are "Predicted" labels.
Classes: Low Risk, Medium Risk, High Risk.

Approximate values:
                Predicted Low   Predicted Medium   Predicted High
True Low           1841            249                 0
True Medium         274            722                223
True High             0             57              2215

Style: Clean heatmap with blue color gradient (light blue for low values,
dark blue for high values). Numbers centered in each cell. Row and column
labels clearly visible. White background, professional data visualization.
No code, no Python imports.
```

---

## 8. ML Pipeline Overview

**File:** `assets/images/pipeline.png`
**Use in:** `README.md` and `docs/methodology.md`

```
A vertical flowchart showing 8 steps of a machine learning pipeline, connected
by downward arrows. Each step is a rounded rectangle with an icon on the left
and text on the right:

1. Data Collection — database icon (blue)
2. EDA — magnifying glass icon (blue)
3. Preprocessing — broom icon (green)
4. Feature Engineering — puzzle piece icon (green)
5. Model Training — brain icon (orange)
6. Evaluation — checkmark icon (orange)
7. SHAP Explainability — lightbulb icon (purple)
8. Deployment — rocket icon (red)

Each step has a subtle number badge (1-8) in the top left corner. Clean flat
design, white background, colorful but professional. The steps progress from
cool colors (top) to warm colors (bottom). No code.
```

---

## 9. Class Distribution

**File:** `assets/images/class-distribution.png`
**Use in:** `docs/results.md`

```
A vertical bar chart showing the distribution of three risk level classes:

- Low Risk: 10,450 students (37.5%) — green bar (#2ecc71)
- Medium Risk: 6,094 students (21.8%) — orange bar (#f39c12)
- High Risk: 11,357 students (40.7%) — red bar (#e74c3c)

Each bar has the count and percentage labeled above it. Y-axis shows number
of students (0 to 12,000). X-axis shows the three risk levels. Clean flat
design, white background, professional data visualization. No code.
```

---

## 10. Project Logo

**File:** `assets/images/logo.png`
**Use in:** Top of `README.md` (optional, smaller than banner)

```
A minimalist logo icon for "EduRisk AI". The design combines a graduation cap
(cap and tassel) with a shield shape, and a small heartbeat/pulse line running
through the center. Color palette: teal (#16a085) primary, white background.
Simple, clean, suitable for a 200x200px favicon or logo. No text.
```

---

## Summary

| # | File | Description | Used In |
|---|------|-------------|---------|
| 1 | `banner.png` | Project banner | README.md |
| 2 | `architecture.png` | System architecture | architecture.md, README.md |
| 3 | `data-flow.png` | Data flow pipeline | architecture.md, README.md |
| 4 | `model-comparison.png` | Model comparison chart | results.md, README.md |
| 5 | `risk-breakdown.png` | Risk level pie chart | results.md, README.md |
| 6 | `shap-importance.png` | SHAP feature importance | results.md, README.md |
| 7 | `confusion-matrix.png` | Confusion matrix heatmap | results.md |
| 8 | `pipeline.png` | ML pipeline overview | README.md, methodology.md |
| 9 | `class-distribution.png` | Class distribution bars | results.md |
| 10 | `logo.png` | Project logo icon | README.md |

After generating, place all files in `assets/images/` and tell me — I'll update all markdown files to reference them.
