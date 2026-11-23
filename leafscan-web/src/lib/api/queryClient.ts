import { QueryClient } from "@tanstack/react-query";

export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 1000 * 30, // 30 seconds
        gcTime: 1000 * 60 * 10, // 10 minutes
        retry: (failureCount, error: any) => {
          // Do not retry 4xx errors
          if (
            error?.status >= 400 &&
            error?.status < 500 &&
            error?.status !== 429
          ) {
            return false;
          }
          return failureCount < 3;
        },
        refetchOnWindowFocus: true,
      },
      mutations: {
        retry: (failureCount, error: any) => {
          // If status is 503 (bounded capacity) or 429, retry up to 3 times
          if (error?.status === 503 || error?.status === 429) {
            return failureCount < 3;
          }
          return false;
        },
      },
    },
  });
}

let browserQueryClient: QueryClient | undefined;

export function getQueryClient(): QueryClient {
  if (typeof window === "undefined") {
    // Server: always make a new query client
    return createQueryClient();
  }
  // Browser: make a new query client if we don't already have one
  if (!browserQueryClient) {
    browserQueryClient = createQueryClient();
  }
  return browserQueryClient;
}
