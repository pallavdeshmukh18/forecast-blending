import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  MapPin,
  CloudRain,
  Thermometer,
  Wind,
  Droplets,
  Plus,
  Minus,
  Crosshair,
  ChevronDown,
  AlertTriangle
} from 'lucide-react';
import L from 'leaflet';
import WindParticleOverlay from './WindParticleOverlay';
import LiveStatusBadge from './LiveStatusBadge';
import useLiveStatus from '../hooks/useLiveStatus';
import { getLiveForecast, getLiveWindGrid } from '../api';
import { SCALAR_LAYERS, DISAGREEMENT_VARS, WIND_STOPS, disagreementLayer, msToKmh } from '../windField';
export const CITIES_DATA = {
  mumbai: {
    city: 'Mumbai',
    state: 'Maharashtra',
    lat: 19.0760,
    lng: 72.8777,
    rainfall: 82,
    rainfallBlended: '82 mm',
    temp: 27.4,
    wind: 28,
    humidity: 72,
    condition: 'Moderate Rain'
  },
  delhi: {
    city: 'Delhi',
    state: 'NCR',
    lat: 28.6139,
    lng: 77.2090,
    rainfall: 12,
    rainfallBlended: '12 mm',
    temp: 34.2,
    wind: 16,
    humidity: 58,
    condition: 'Partly Cloudy'
  },
  bengaluru: {
    city: 'Bengaluru',
    state: 'Karnataka',
    lat: 12.9716,
    lng: 77.5946,
    rainfall: 28,
    rainfallBlended: '28 mm',
    temp: 24.1,
    wind: 19,
    humidity: 68,
    condition: 'Scattered Showers'
  },
  kolkata: {
    city: 'Kolkata',
    state: 'West Bengal',
    lat: 22.5726,
    lng: 88.3639,
    rainfall: 64,
    rainfallBlended: '64 mm',
    temp: 29.8,
    wind: 24,
    humidity: 84,
    condition: 'Heavy Rain'
  },
  chennai: {
    city: 'Chennai',
    state: 'Tamil Nadu',
    lat: 13.0827,
    lng: 80.2707,
    rainfall: 18,
    rainfallBlended: '18 mm',
    temp: 31.5,
    wind: 22,
    humidity: 76,
    condition: 'Humid & Overcast'
  },
  guwahati: {
    city: 'Guwahati',
    state: 'Assam',
    lat: 26.1445,
    lng: 91.7362,
    rainfall: 94,
    rainfallBlended: '94 mm',
    temp: 26.2,
    wind: 15,
    humidity: 89,
    condition: 'Intense Monsoon Rain'
  }
};

const HORIZONS = [6, 12, 24, 48, 72];
const LAYER_TABS = [
  { key: 'wind', label: '🌬️ Windy Vector Flow' },
  { key: 'rain', label: '🌧️ Precipitation Radar' },
  { key: 'temp', label: '🌡️ Temperature Gradient' },
  { key: 'disagreement', label: '⚠️ Model Disagreement' },
];

const fmtCI = (v, ci, unit = '°C') => (v == null ? '—' : ci ? `${v}${unit} [${ci[0]}${unit} – ${ci[1]}${unit}]` : `${v}${unit}`);
const escapeHtml = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function windLegendRows() {
  return WIND_STOPS.map((s, i) => {
    const lo = i === 0 ? null : WIND_STOPS[i - 1].max;
    const hi = s.max === Infinity ? null : s.max;
    let label;
    if (lo == null) label = `< ${hi} m/s · < ${Math.round(hi * 3.6)} km/h`;
    else if (hi == null) label = `> ${lo} m/s · > ${Math.round(lo * 3.6)} km/h`;
    else label = `${lo} – ${hi} m/s · ${Math.round(lo * 3.6)} – ${Math.round(hi * 3.6)} km/h`;
    return { color: s.color, label };
  }).reverse();
}

export default function InteractiveMapSection({ currentCityKey, onSelectCity }) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markerRef = useRef(null);
  const gridCacheRef = useRef({});

  const [map, setMap] = useState(null);
  const [activeLayer, setActiveLayer] = useState('wind'); // wind | rain | temp | disagreement
  const [disagreementVar, setDisagreementVar] = useState('wind_speed_ms');
  const [horizon, setHorizon] = useState(24);
  const [isCityDropdownOpen, setIsCityDropdownOpen] = useState(false);
  const [grid, setGrid] = useState(null);
  const [gridState, setGridState] = useState('loading'); // loading | ok | error
  const [gridNonce, setGridNonce] = useState(0);
  const [live, setLive] = useState(null); // LiveForecastResult for the selected city
  const [liveError, setLiveError] = useState(false);

  const currentData = CITIES_DATA[currentCityKey] || CITIES_DATA.mumbai;

  // ---- Live city forecast (re-fetched on every city change) -------------------
  const loadCity = useCallback(async () => {
    try {
      const res = await getLiveForecast({ latitude: currentData.lat, longitude: currentData.lng });
      setLive(res.status === 'success' ? res : null);
      setLiveError(res.status !== 'success');
    } catch {
      setLive(null);
      setLiveError(true);
    }
  }, [currentData.lat, currentData.lng]);

  const liveStatus = useLiveStatus({
    onRefresh: async () => {
      gridCacheRef.current = {};
      setGridNonce((n) => n + 1);
      await loadCity();
    },
  });
  const { poll } = liveStatus;

  useEffect(() => {
    setLive(null);
    loadCity().then(poll);
  }, [loadCity, poll]);

  // ---- Gridded field (wind u/v, rain, temp, disagreement) ---------------------
  useEffect(() => {
    const wantDis = activeLayer === 'disagreement';
    const key = wantDis ? 'dis' : 'base';
    const cached = gridCacheRef.current[key];
    if (cached) {
      setGrid(cached);
      setGridState('ok');
      return undefined;
    }
    let cancelled = false;
    setGridState('loading');
    getLiveWindGrid({ includeDisagreement: wantDis })
      .then((g) => {
        if (cancelled) return;
        gridCacheRef.current[key] = g;
        setGrid(g);
        setGridState('ok');
      })
      .catch(() => {
        if (cancelled) return;
        setGrid(null);
        setGridState('error');
      });
    return () => {
      cancelled = true;
    };
  }, [activeLayer, gridNonce]);

  // ---- Map init -------------------------------------------------------------
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return undefined;
    const m = L.map(mapContainerRef.current, {
      center: [20.5937, 78.9629],
      zoom: 5,
      minZoom: 4,
      maxZoom: 18,
      zoomControl: false,
      attributionControl: false,
      maxBounds: [[-5, 55], [45, 110]],
    });

    // Esri World Dark Gray canvas: keyless, high contrast for particle streams.
    // Native tiles stop at z16; deeper zooms upscale. Falls back to OSM if Esri tiles fail.
    const esri = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
      { minZoom: 4, maxZoom: 18, maxNativeZoom: 16, attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ' }
    ).addTo(m);
    let swapped = false;
    esri.on('tileerror', () => {
      if (swapped) return;
      swapped = true;
      m.removeLayer(esri);
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        minZoom: 4,
        maxZoom: 18,
        subdomains: 'abc',
        attribution: '&copy; OpenStreetMap contributors',
      }).addTo(m);
    });
    L.control.attribution({ position: 'bottomleft', prefix: false }).addTo(m);

    mapInstanceRef.current = m;
    setMap(m);
    return () => {
      m.remove();
      mapInstanceRef.current = null;
      setMap(null);
    };
  }, []);

  // ---- Marker + alert popup + fly-to ---------------------------------------
  const alerts = live?.active_alerts || [];
  const alertCount = alerts.length;
  useEffect(() => {
    if (!map) return;
    map.flyTo([currentData.lat, currentData.lng], 6, { duration: 1.2, easeLinearity: 0.25 });
    if (markerRef.current) map.removeLayer(markerRef.current);

    const hasAlert = alertCount > 0;
    const color = hasAlert ? '#ef4444' : '#1e70eb';
    const icon = L.divIcon({
      className: 'custom-map-pin',
      html: `<div style="background:${color};border:2px solid #fff;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;box-shadow:0 0 ${hasAlert ? 18 : 10}px ${color};transform:translate(-11px,-11px)"><div style="width:8px;height:8px;border-radius:50%;background:#fff"></div></div>`,
      iconSize: [22, 22],
    });
    const marker = L.marker([currentData.lat, currentData.lng], { icon }).addTo(map);
    if (hasAlert) {
      const rows = (live?.active_alerts || [])
        .slice(0, 4)
        .map((a) => `<div style="margin-top:6px;font-size:12px"><b>${escapeHtml(a.headline || a.hazard_type)}</b><br/>${escapeHtml(a.observed_or_forecast_value)} (threshold ${escapeHtml(a.threshold)})</div>`)
        .join('');
      marker.bindPopup(
        `<div style="min-width:200px"><strong style="color:#b91c1c">⚠ ${alertCount} active alert${alertCount > 1 ? 's' : ''} – ${escapeHtml(currentData.city)}</strong>${rows}</div>`
      );
      marker.openPopup();
    }
    markerRef.current = marker;
  }, [map, currentData, alertCount, live]);

  const handleZoomIn = () => mapInstanceRef.current?.zoomIn();
  const handleZoomOut = () => mapInstanceRef.current?.zoomOut();
  const handleLocateMe = () => mapInstanceRef.current?.flyTo([currentData.lat, currentData.lng], 7);

  // ---- Card values: live when available, static preview otherwise ----------
  const step = live?.forecast_steps?.find((s) => s.lead_time_hours === horizon) || null;
  const cur = live?.current_conditions;
  const card = live && step
    ? {
        rain: `${step.rainfall_mm} mm/6h`,
        temp: fmtCI(step.temperature_c, step.temperature_ci90),
        wind: `${Math.round(msToKmh(step.wind_speed_ms))} km/h (${step.wind_speed_ms} m/s)`,
        humidity: cur?.relative_humidity_pct != null ? `${cur.relative_humidity_pct}%` : '—',
        source: 'LIVE',
      }
    : {
        rain: currentData.rainfallBlended,
        temp: `${currentData.temp} °C`,
        wind: `${currentData.wind} km/h`,
        humidity: `${currentData.humidity}%`,
        source: 'PREVIEW',
      };

  const legendDef = activeLayer === 'wind' ? null : activeLayer === 'disagreement' ? disagreementLayer(disagreementVar) : SCALAR_LAYERS[activeLayer];
  const legendRows = activeLayer === 'wind' ? windLegendRows() : legendDef.legend.map(([label, color]) => ({ label, color }));

  return (
    <section id="forecast" className="map-dashboard-section">
      <div className="container">
        {/* Controls Bar */}
        <div className="map-controls-bar" style={{ flexWrap: 'wrap' }}>
          {/* Location Selector Dropdown */}
          <div style={{ position: 'relative' }}>
            <button
              className="location-selector-input"
              onClick={() => setIsCityDropdownOpen(!isCityDropdownOpen)}
              type="button"
            >
              <MapPin className="location-pin-icon" />
              <span>{currentData.city}, {currentData.state}</span>
              <ChevronDown size={16} style={{ marginLeft: 'auto', color: '#64748b' }} />
            </button>

            {isCityDropdownOpen && (
              <div style={{
                position: 'absolute', top: '110%', left: 0, width: '100%', background: '#ffffff',
                border: '1px solid #e2e8f0', borderRadius: '12px', boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
                zIndex: 1000, overflow: 'hidden'
              }}>
                {Object.keys(CITIES_DATA).map((key) => (
                  <div
                    key={key}
                    onClick={() => {
                      onSelectCity(key);
                      setIsCityDropdownOpen(false);
                    }}
                    style={{
                      padding: '10px 16px', fontSize: '13.5px', fontWeight: 600, cursor: 'pointer',
                      background: currentCityKey === key ? '#eff6ff' : 'transparent',
                      color: currentCityKey === key ? '#1e70eb' : '#334155',
                      borderBottom: '1px solid #f1f5f9'
                    }}
                  >
                    {CITIES_DATA[key].city}, {CITIES_DATA[key].state}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Layer Mode Tabs */}
          <div className="layer-tabs" role="tablist" aria-label="Map layer">
            {LAYER_TABS.map((t) => (
              <button
                key={t.key}
                role="tab"
                aria-selected={activeLayer === t.key}
                className={`layer-tab-btn ${activeLayer === t.key ? 'active' : ''}`}
                onClick={() => setActiveLayer(t.key)}
              >
                {t.label}
              </button>
            ))}
          </div>

          {/* Horizon selector */}
          <div className="layer-tabs" aria-label="Forecast horizon">
            {HORIZONS.map((h) => (
              <button
                key={h}
                className={`layer-tab-btn ${horizon === h ? 'active' : ''}`}
                onClick={() => setHorizon(h)}
              >
                +{h}h
              </button>
            ))}
          </div>

          <LiveStatusBadge live={liveStatus} />
        </div>

        {/* Map Viewport */}
        <div className="map-viewport-wrapper">
          <div id="india-weather-map" ref={mapContainerRef}></div>

          {grid && gridState === 'ok' && (
            <WindParticleOverlay map={map} grid={grid} layer={activeLayer} disagreementVar={disagreementVar} />
          )}

          {gridState !== 'ok' && (
            <div style={{ position: 'absolute', top: 12, left: '50%', transform: 'translateX(-50%)', zIndex: 1000, background: 'rgba(15,23,42,.88)', color: '#e2e8f0', padding: '6px 14px', borderRadius: 999, fontSize: 12.5 }}>
              {gridState === 'loading' ? 'Loading live field from backend…' : 'Live field unavailable — start the FastAPI backend (python run_server.py)'}
            </div>
          )}

          {activeLayer === 'disagreement' && gridState === 'ok' && (
            <div style={{ position: 'absolute', top: 12, left: 12, zIndex: 1000, display: 'flex', gap: 4, background: 'rgba(15,23,42,.88)', padding: 4, borderRadius: 8 }}>
              {Object.entries(DISAGREEMENT_VARS).map(([k, v]) => (
                <button
                  key={k}
                  onClick={() => setDisagreementVar(k)}
                  style={{ border: 0, borderRadius: 6, padding: '4px 10px', fontSize: 12, cursor: 'pointer', fontWeight: 600, background: disagreementVar === k ? '#2563eb' : 'transparent', color: '#fff' }}
                >
                  {v.label}
                </button>
              ))}
            </div>
          )}

          {/* Map Custom Controls (+, -, locate) */}
          <div className="map-custom-controls">
            <button className="map-control-btn" onClick={handleZoomIn} title="Zoom In"><Plus size={18} /></button>
            <button className="map-control-btn" onClick={handleZoomOut} title="Zoom Out"><Minus size={18} /></button>
            <button className="map-control-btn" onClick={handleLocateMe} title="Center on Location"><Crosshair size={18} /></button>
          </div>

          {/* Legend */}
          <div className="map-rainfall-legend">
            <div className="legend-title">
              {activeLayer === 'wind' ? 'Wind Speed' : legendDef.title}
            </div>
            <div className="legend-items">
              {legendRows.map((r) => (
                <div className="legend-row" key={r.label}>
                  <div className="legend-color-swatch" style={{ background: r.color }}></div>
                  <span>{r.label}</span>
                </div>
              ))}
            </div>
            {grid?.valid_time_utc && (
              <div style={{ fontSize: 10.5, color: '#64748b', marginTop: 6 }}>
                Analysis {grid.valid_time_utc.replace('T', ' ')} UTC
                {activeLayer === 'disagreement' && ' · ECMWF IFS vs GFS'}
              </div>
            )}
          </div>

          {/* Floating Weather Detail Card on Top Right */}
          <div className="map-floating-stats-card">
            <h4 className="mf-location">{currentData.city}, {currentData.state}</h4>
            <p className="mf-period">
              +{horizon}h · {card.source}
              {liveError && ' (backend offline)'}
            </p>

            <div className="mf-items-list">
              <div className="mf-item-row">
                <div className="mf-icon-wrapper" style={{ background: '#e0f2fe' }}><CloudRain size={18} color="#0284c7" /></div>
                <div className="mf-item-info"><span className="mf-item-val">{card.rain}</span><span className="mf-item-lbl">Rainfall (Blended)</span></div>
              </div>
              <div className="mf-item-row">
                <div className="mf-icon-wrapper" style={{ background: '#fee2e2' }}><Thermometer size={18} color="#dc2626" /></div>
                <div className="mf-item-info"><span className="mf-item-val" style={{ fontSize: card.temp.length > 12 ? 12 : undefined }}>{card.temp}</span><span className="mf-item-lbl">Temperature (EMOS 90% CI)</span></div>
              </div>
              <div className="mf-item-row">
                <div className="mf-icon-wrapper" style={{ background: '#ecfdf5' }}><Wind size={18} color="#059669" /></div>
                <div className="mf-item-info"><span className="mf-item-val">{card.wind}</span><span className="mf-item-lbl">Wind Speed</span></div>
              </div>
              <div className="mf-item-row">
                <div className="mf-icon-wrapper" style={{ background: '#eff6ff' }}><Droplets size={18} color="#2563eb" /></div>
                <div className="mf-item-info"><span className="mf-item-val">{card.humidity}</span><span className="mf-item-lbl">Humidity</span></div>
              </div>
              {alertCount > 0 && (
                <div className="mf-item-row">
                  <div className="mf-icon-wrapper" style={{ background: '#fee2e2' }}><AlertTriangle size={18} color="#b91c1c" /></div>
                  <div className="mf-item-info"><span className="mf-item-val" style={{ color: '#b91c1c' }}>{alertCount} active</span><span className="mf-item-lbl">{alerts[0].headline || alerts[0].hazard_type}</span></div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
