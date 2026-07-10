import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EduRisk AI — Explainable Academic Risk Intelligence",
  description: "ML-powered student academic risk prediction with SHAP explainability",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-bg antialiased">
        {children}
      </body>
    </html>
  );
}
