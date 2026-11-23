"use client";

import { getQueryClient } from "@/lib/api/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import type React from "react";

export const Providers: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const queryClient = getQueryClient();

  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};
