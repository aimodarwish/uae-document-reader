import type { Metadata } from "next";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";

export const metadata: Metadata = {
  title: "UAE Document Intelligence Suite | Smart OCR & Verification",
  description: "Enterprise-grade UAE Vehicle License (Mulkiya), Passport, and Emirates ID extraction with instant in-memory processing.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div className="ambient-glow-top" />
        <div className="ambient-glow-bottom" />
        {children}
        <Analytics />
      </body>
    </html>
  );
}
