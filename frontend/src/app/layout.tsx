import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Medical Disease Detection",
  description: "AI-Powered Medical Diagnosis Support System",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <div className="app-container">
          {children}
        </div>
      </body>
    </html>
  );
}
