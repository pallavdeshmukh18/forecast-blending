/**
 * Operational Forecast & Blending Pipeline API
 */

import { apiClient } from './client';

/**
 * Execute end-to-end operational forecast blending pipeline: POST /forecast/run
 * @param {Object} params
 * @param {string} [params.mode='synthetic'] - 'synthetic' | 'historical' | 'real'
 * @param {string} [params.variable='rainfall'] - 'rainfall' | 'temperature' | 'wind_speed'
 * @param {number} [params.leadTime=24] - 6 | 12 | 24 | 48 | 72
 * @param {string} [params.region=null] - optional region code
 * @param {string} [params.blenderMethod='regime_blender'] - blender algorithm name
 */
export async function runForecastPipeline({
  mode = 'synthetic',
  variable = 'rainfall',
  leadTime = 24,
  region = null,
  blenderMethod = 'regime_blender',
} = {}, options = {}) {
  const payload = {
    mode,
    variable,
    lead_time: Number(leadTime),
    region,
    blender_method: blenderMethod,
  };
  return apiClient.post('/forecast/run', payload, options);
}

/**
 * Blend forecast arrays: POST /forecast/blend
 */
export async function blendForecasts({
  variable = 'rainfall',
  leadTime = 24,
  region = null,
  blenderMethod = 'regime_blender',
  forecasts = null,
} = {}, options = {}) {
  const payload = {
    variable,
    lead_time: Number(leadTime),
    region,
    blender_method: blenderMethod,
    forecasts,
  };
  return apiClient.post('/forecast/blend', payload, options);
}

/**
 * Generate probabilistic forecast distribution & quantiles: POST /forecast/probability
 */
export async function getForecastProbability({
  variable = 'rainfall',
  leadTime = 24,
  region = null,
  distributionType = 'gaussian',
  quantiles = [0.10, 0.25, 0.50, 0.75, 0.90],
} = {}, options = {}) {
  const payload = {
    variable,
    lead_time: Number(leadTime),
    region,
    distribution_type: distributionType,
    quantiles,
  };
  return apiClient.post('/forecast/probability', payload, options);
}

/**
 * Compute extreme weather exceedance probabilities: POST /forecast/extremes
 */
export async function getForecastExtremes({
  variable = 'rainfall',
  leadTime = 24,
  region = null,
  threshold = null,
} = {}, options = {}) {
  const payload = {
    variable,
    lead_time: Number(leadTime),
    region,
    threshold: threshold != null ? Number(threshold) : null,
  };
  return apiClient.post('/forecast/extremes', payload, options);
}

/**
 * Scan forecasts for active extreme weather alerts: GET /extremes/alerts
 */
export async function getExtremeAlerts(options = {}) {
  return apiClient.get('/extremes/alerts', options);
}

/**
 * Retrieve weather regime classification definitions: GET /regimes
 */
export async function getWeatherRegimes(options = {}) {
  return apiClient.get('/regimes', options);
}

/**
 * Verify forecast accuracy against observations: POST /verify
 */
export async function verifyForecast({
  variable = 'rainfall',
  leadTime = 24,
  blenderMethod = 'regime_blender',
} = {}, options = {}) {
  const payload = {
    variable,
    lead_time: Number(leadTime),
    blender_method: blenderMethod,
  };
  return apiClient.post('/verify', payload, options);
}

/**
 * Summary of forecast verification metrics: GET /metrics
 */
export async function getVerificationMetrics({ variable = 'rainfall', leadTime = 24 } = {}, options = {}) {
  const query = new URLSearchParams({
    variable: String(variable),
    lead_time: String(leadTime),
  });
  return apiClient.get(`/metrics?${query.toString()}`, options);
}
