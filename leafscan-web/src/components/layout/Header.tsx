"use client";

import { flushOfflineQueue } from "@/features/history/historyStore";
import { fetchHealth, fetchReady } from "@/lib/api/diagnoseClient";
import { Activity, Leaf, Wifi, WifiOff } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import type React from "react";
import { useEffect, useState } from "react";

export const Header: React.FC = () => {
  const pathname = usePathname();

  const [systemState, setSystemState] = useState<
    "checking" | "ready" | "degraded" | "offline"
  >("checking");
  const [engineInfo, setEngineInfo] = useState<string | null>(null);
  const [isOnline, setIsOnline] = useState<boolean>(true);

  // Monitor connectivity
  useEffect(() => {
    if (typeof window === "undefined") return;

    const updateOnline = () => {
      const online = navigator.onLine;
      setIsOnline(online);
      if (online) {
        // Auto-flush offline queue when connection is restored
        flushOfflineQueue().catch(console.error);
      }
    };

    window.addEventListener("online", updateOnline);
    window.addEventListener("offline", updateOnline);
    updateOnline();

    return () => {
      window.removeEventListener("online", updateOnline);
      window.removeEventListener("offline", updateOnline);
    };
  }, []);

  // Poll system health every 60 seconds (paused when tab is hidden)
  useEffect(() => {
    let timer: NodeJS.Timeout | null = null;

    const checkSystem = async () => {
      if (document.hidden || !navigator.onLine) {
        if (!navigator.onLine) setSystemState("offline");
        return;
      }

      try {
        const [health, ready] = await Promise.all([
          fetchHealth(),
          fetchReady(),
        ]);
        if (ready.status === "ready") {
          setSystemState("ready");
          setEngineInfo(ready.engine || "ONNX Runtime");
        } else if (health.status === "healthy") {
          setSystemState("ready");
        } else {
          setSystemState("degraded");
        }
      } catch {
        setSystemState(navigator.onLine ? "degraded" : "offline");
      }
    };

    const handleVisibilityChange = () => {
      if (!document.hidden) {
        checkSystem();
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    checkSystem();
    timer = setInterval(checkSystem, 60000);

    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      if (timer) clearInterval(timer);
    };
  }, []);

  const navLinks = [
    { href: "/diagnose", label: "Diagnose" },
    { href: "/history", label: "History" },
    { href: "/model", label: "Model" },
    { href: "/feedback", label: "Feedback" },
  ];

  return (
    <header className="sticky top-0 z-40 w-full border-b border-bark-200/60 bg-white/90 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
        {/* Brand */}
        <Link
          href="/"
          className="flex items-center gap-2.5 text-bark-900 group"
        >
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-leaf-600 text-white shadow-sm group-hover:bg-leaf-700 transition-colors">
            <Leaf className="h-5 w-5" />
          </div>
          <div>
            <span className="block text-base font-serif font-bold tracking-tight">
              LeafScan V3
            </span>
            <span className="block text-[10px] font-mono font-medium text-bark-500 uppercase tracking-wider">
              Botanical Field Console
            </span>
          </div>
        </Link>

        {/* Navigation */}
        <nav
          className="hidden md:flex items-center gap-1 bg-bark-100/60 p-1 rounded-xl"
          aria-label="Main Navigation"
        >
          {navLinks.map((link) => {
            const isActive =
              pathname === link.href ||
              (link.href !== "/" && pathname?.startsWith(link.href));
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`rounded-lg px-3.5 py-1.5 text-xs font-semibold transition-all ${
                  isActive
                    ? "bg-white text-bark-900 shadow-xs"
                    : "text-bark-600 hover:text-bark-900 hover:bg-white/50"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>

        {/* Status Indicators */}
        <div className="flex items-center">
          {/* Engine & Network Status */}
          <div
            role="status"
            aria-live="polite"
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-medium border bg-bark-50 border-bark-200"
            title={engineInfo ? `Inference Engine: ${engineInfo}` : undefined}
          >
            {isOnline ? (
              systemState === "ready" ? (
                <>
                  <span className="h-2 w-2 rounded-full bg-teal-500 animate-pulse" />
                  <span className="text-bark-800 text-[11px]">V3 Ready</span>
                </>
              ) : systemState === "checking" ? (
                <>
                  <Activity className="h-3 w-3 text-bark-400 animate-spin" />
                  <span className="text-bark-500 text-[11px]">Connecting</span>
                </>
              ) : (
                <>
                  <span className="h-2 w-2 rounded-full bg-amber-500" />
                  <span className="text-amber-800 text-[11px]">Degraded</span>
                </>
              )
            ) : (
              <>
                <WifiOff className="h-3 w-3 text-rust-600" />
                <span className="text-rust-600 text-[11px]">Offline</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Mobile Nav Bar */}
      <nav
        className="flex md:hidden border-t border-bark-200/40 px-4 py-2 justify-around bg-white"
        aria-label="Mobile Navigation"
      >
        {navLinks.map((link) => {
          const isActive =
            pathname === link.href ||
            (link.href !== "/" && pathname?.startsWith(link.href));
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`text-xs font-semibold px-2 py-1 rounded-md ${
                isActive ? "text-leaf-700 bg-leaf-50" : "text-bark-600"
              }`}
            >
              {link.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
};
