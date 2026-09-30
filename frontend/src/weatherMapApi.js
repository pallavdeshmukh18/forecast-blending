const API_BASE = import.meta.env.VITE_API_URL || '/api'

const variableMap = {
  rain: 'rainfall',
  temp: 'temperature',
  wind: 'wind_speed',
  uncertainty: 'rainfall',
  weights: 'rainfall',
  extremes: 'rainfall',
  change: 'rainfall'
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options
  })
  if (!response.ok) {
    const message = await response.text()
    throw new Error(message || `API request failed (${response.status})`)
  }
  return response.json()
}

export async function getHealth() {
  return request('/health')
}

export async function runForecast(layer, leadTime, blenderMethod = 'regime_blender') {
  return request('/forecast/run', {
    method: 'POST',
    body: JSON.stringify({
      mode: 'synthetic',
      variable: variableMap[layer] || 'rainfall',
      lead_time: leadTime,
      blender_method: blenderMethod
    })
  })
}

export function forecastToGeoJSON(result, layer) {
  if (!result?.forecast) return { type: 'FeatureCollection', features: [] }
  const dominantModel = Object.entries(result.weights || {}).sort((a, b) => Number(b[1]) - Number(a[1]))[0]?.[0] || 'unknown'
  return {
    type: 'FeatureCollection',
    features: result.forecast.map((row, index) => {
      let value = Number(row.blend_value || 0)
      if (layer === 'uncertainty') value = Number(row.spread || 0)
      if (layer === 'weights') value = index % Math.max(1, Object.keys(result.weights || {}).length)
      if (layer === 'extremes') value = Math.min(100, Math.max(0, value * 2 + Number(row.spread || 0) * 8))
      if (layer === 'change') value = 0
      return {
        type: 'Feature',
        properties: {
          value,
          blendValue: Number(row.blend_value || 0),
          spread: Number(row.spread || 0),
          regime: row.regime || result.regime?.dominant_regime || 'stable',
          dominantModel,
          timestamp: row.timestamp || result.timestamp,
          isCenter: 0
        },
        geometry: { type: 'Point', coordinates: [Number(row.longitude), Number(row.latitude)] }
      }
    })
  }
}

export { API_BASE }
