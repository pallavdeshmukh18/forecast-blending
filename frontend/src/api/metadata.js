/**
 * System Metadata & Diagnostics API
 */

import { apiClient } from './client';

/**
 * Retrieve comprehensive project metadata: GET /metadata
 */
export async function getMetadata(options = {}) {
  return apiClient.get('/metadata', options);
}

/**
 * Retrieve system diagnostics and provenance summary: GET /diagnostics
 */
export async function getDiagnostics(options = {}) {
  return apiClient.get('/diagnostics', options);
}

/**
 * Retrieve active system configuration: GET /config
 */
export async function getConfig(options = {}) {
  return apiClient.get('/config', options);
}

/**
 * Retrieve data pipeline status and alignment parameters: GET /data/status
 */
export async function getDataStatus(options = {}) {
  return apiClient.get('/data/status', options);
}
