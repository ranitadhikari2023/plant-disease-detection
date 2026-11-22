/**
 * web/src/api/client.ts
 * Configured openapi-fetch client for LeafScan V2 API
 */

import createFetchClient from 'openapi-fetch';
import type { paths } from './schema';

export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? '';

export const apiClient = createFetchClient<paths>({
  baseUrl: API_BASE_URL,
});
