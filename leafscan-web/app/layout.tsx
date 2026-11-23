import type { Metadata, Viewport } from "next";
import "@/styles/globals.css";
import { Footer } from "@/components/layout/Footer";
import { Header } from "@/components/layout/Header";
import { Providers } from "@/components/layout/Providers";

export const metadata: Metadata = {
  title: "LeafScan V3 — Botanical Field Console",
  description: "Hierarchical Two-Tier Foliar Screening for Solanaceous Crops",
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "LeafScan V3",
  },
};

export const viewport: Viewport = {
  themeColor: "#2F6B3A",
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full antialiased">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;600;700;800&family=Geist+Mono:wght@400;500;600;700&family=Geist:wght@300;400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="flex min-h-full flex-col bg-leaf-50 text-bark-900 selection:bg-leaf-200">
        <Providers>
          <Header />
          <main className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 py-6 md:py-8">
            {children}
          </main>
          <Footer />
        </Providers>
      </body>
    </html>
  );
}
