import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DocQuery",
  description: "Citation-grounded hybrid-search RAG over Postgres + pgvector",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
