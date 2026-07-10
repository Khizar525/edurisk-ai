# AI-Generated Assets — Prompts

Only two images in this project should be AI-generated: the banner and logo.
Everything else is generated from real data or Mermaid diagrams.

---

## 1. Banner

**File:** `assets/images/banner.png`
**Use in:** Top of `README.md`

```
Design a premium GitHub hero banner for an AI product called EduRisk AI.

Style:
Modern AI SaaS branding with a clean, minimal aesthetic inspired by GitHub, Stripe, Linear, Vercel, and OpenAI.

Scene:
On the left, abstract geometric shapes subtly resembling a university campus and connected academic pathways.
These gradually transition across the banner into flowing data streams, neural network nodes, and intelligent prediction graphs on the right, symbolizing the transformation of educational data into actionable insights.

Color palette:
Background: deep navy (#111827)
Primary accent: teal (#14b8a6)
Secondary accent: cyan (#38bdf8)
Small highlight accents: coral (#f97316)

Lighting:
Soft gradients, subtle glow effects around network nodes, plenty of negative space.

Style:
Flat vector illustration with slight depth.
No 3D.
No people.
No laptops.
No stock-style illustrations.

Composition:
Wide 3:1 GitHub banner (1600x500).
Leave the center visually clean for README overlays if needed.

Absolutely no text, logos, watermarks, UI mockups, or screenshots.
```

---

## 2. Logo

**File:** `assets/images/logo.png`
**Use in:** Top of `README.md`, favicon

```
Minimal vector logo.

A modern geometric shield containing a subtle graduation cap formed from negative space.

Inside the shield, three connected AI nodes forming a small neural network.

Primary color: teal (#14b8a6)

White background.

Flat design.

Scalable SVG icon.

No text.

Suitable for GitHub avatar, favicon and documentation.
```

---

## What NOT to Generate with AI

| Asset | Why | Generate With |
|-------|-----|---------------|
| Architecture diagram | Must reflect actual code | Mermaid in `docs/architecture.md` |
| Data flow | Must reflect actual pipeline | Mermaid in `docs/architecture.md` |
| ML pipeline | Must reflect actual steps | Mermaid in `README.md` |
| Model comparison | Must use real metrics | `scripts/generate_real_charts.py` |
| Confusion matrix | Must use real predictions | `scripts/generate_real_charts.py` |
| SHAP importance | Must use real SHAP values | `scripts/generate_real_charts.py` |
| Class distribution | Must use real counts | `scripts/generate_real_charts.py` |
| Risk breakdown | Must use real distribution | `scripts/generate_real_charts.py` |
| Per-class metrics | Must use real report | `scripts/generate_real_charts.py` |
| App screenshots | Must show real running app | Manual capture |
