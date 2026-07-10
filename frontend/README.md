# EduRisk AI — Frontend

Dark-mode web interface for the EduRisk AI academic risk prediction system.

## Tech Stack

- **Next.js 15** — React framework with App Router
- **TypeScript** — end-to-end type safety
- **Tailwind CSS v4** — utility-first styling
- **Framer Motion** — animations and transitions

## Components

| Component | Description |
|-----------|-------------|
| `PredictionForm` | Student profile input (sliders, toggles, dropdowns) |
| `RiskGauge` | Animated SVG semicircular risk gauge |
| `ProbabilityBars` | Animated probability bars for High/Medium/Low |
| `ShapWaterfall` | Horizontal bar chart of SHAP feature contributions |
| `FactorCards` | Risk factors (red) and protective factors (green) |
| `AdviceCards` | Personalized recommendations based on risk level |

## Getting Started

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Backend Required

The frontend connects to the FastAPI backend at `http://localhost:8000`. Start it first:

```bash
# From project root
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

## Build

```bash
npm run build
npm start
```

## Project Structure

```
src/
├── app/
│   ├── page.tsx          # Main page
│   ├── layout.tsx        # Root layout
│   └── globals.css       # Global styles + Tailwind
├── components/
│   ├── PredictionForm.tsx
│   ├── RiskGauge.tsx
│   ├── ProbabilityBars.tsx
│   ├── ShapWaterfall.tsx
│   ├── FactorCards.tsx
│   └── AdviceCards.tsx
└── lib/
    ├── api.ts            # API client
    └── types.ts          # TypeScript types
```
