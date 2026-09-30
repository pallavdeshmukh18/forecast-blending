/**
 * Models & Blending Weights API
 */

import { apiClient } from './client';

/**
 * Retrieve specifications of available NWP and AI models: GET /models
 */
export async function getModels(options = {}) {
  return apiClient.get('/models', options);
}

/**
 * Retrieve active model weights: GET /models/weights
 * @param {Object} params
 * @param {string} params.variable - Target meteorological variable (e.g. 'rainfall', 'temperature', 'wind_speed')
 * @param {number} params.leadTime - Lead time in hours (e.g. 6, 12, 24, 48, 72)
 */
export async function getModelWeights({ variable = 'rainfall', leadTime = 24 } = {}, options = {}) {
  const query = new URLSearchParams({
    variable: String(variable),
    lead_time: String(leadTime),
  });
  return apiClient.get(`/models/weights?${query.toString()}`, options);
}

/**
 * Retrieve dynamically computed model blending weights: GET /blending/weights
 */
export async function getBlendingWeights({ variable = 'rainfall', leadTime = 24, region = 'plains', blender = 'inverse_variance' } = {}, options = {}) {
  const query = new URLSearchParams({
    variable: String(variable),
    lead_time: String(leadTime),
    region: String(region),
    blender: String(blender),
  });
  return apiClient.get(`/blending/weights?${query.toString()}`, options);
}

/**
 * Retrieve historical model error skill metrics: GET /blending/skill
 */
export async function getHistoricalSkill({ variable, leadTime, region } = {}, options = {}) {
  const query = new URLSearchParams();
  if (variable) query.append('variable', String(variable));
  if (leadTime != null) query.append('lead_time', String(leadTime));
  if (region) query.append('region', String(region));

  const qs = query.toString();
  return apiClient.get(`/blending/skill${qs ? `?${qs}` : ''}`, options);
}
