/**
 * Health & Backend Connectivity API
 */

import { apiClient } from './client';

/**
 * Probe FastAPI health check endpoint: GET /health
 */
export async function getHealth(options = {}) {
  return apiClient.get('/health', options);
}

/**
 * Safe health check returning boolean connectivity and latency
 */
export async function checkBackendStatus(options = {}) {
  const start = performance.now();
  const res = await apiClient.safeGet('/health', { timeout: 6000, ...options });
  const latency = Math.round(performance.now() - start);

  if (!res.isLive) {
    console.warn('[BharatWeather Backend Probe] Connection probe failed:', res.error?.message || res.error);
  }

  return {
    isLive: res.isLive && res.data?.status === 'healthy',
    status: res.data?.status || 'offline',
    service: res.data?.service || null,
    version: res.data?.version || null,
    latencyMs: latency,
    error: res.error,
  };
}
