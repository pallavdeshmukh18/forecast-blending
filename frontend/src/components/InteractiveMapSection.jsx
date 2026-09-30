import React, { useEffect, useRef, useState } from 'react';
import { 
  MapPin, 
  CloudRain, 
  Thermometer, 
  Wind, 
  Droplets, 
  Plus, 
  Minus, 
  Crosshair,
  ChevronDown
} from 'lucide-react';
import L from 'leaflet';

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

export default function InteractiveMapSection({ currentCityKey, onSelectCity }) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markerRef = useRef(null);

  const [activeLayer, setActiveLayer] = useState('rainfall'); // rainfall | temp | wind
  const [selectedHorizon, setSelectedHorizon] = useState('Next 24 hours');
  const [isCityDropdownOpen, setIsCityDropdownOpen] = useState(false);

  const currentData = CITIES_DATA[currentCityKey] || CITIES_DATA.mumbai;

  // Initialize Leaflet Map
  useEffect(() => {
    if (!mapContainerRef.current) return;

    if (!mapInstanceRef.current) {
      // Create Leaflet map centered on Central/West India
      const map = L.map(mapContainerRef.current, {
        center: [20.5937, 78.9629],
        zoom: 5,
        zoomControl: false,
        attributionControl: false
      });

      // CartoDB Dark Matter / Voyager base layer with satellite terrain aesthetics
      L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
        maxZoom: 18,
        subdomains: 'abcd',
      }).addTo(map);

      // Add custom SVG radar weather heatmap overlay simulating India Doppler radar coverage
      const bounds = [[6.0, 67.0], [36.0, 98.0]];
      
      // Canvas overlay for meteorological radar intensity simulation
      const canvasOverlay = L.canvasOverlay ? null : null; // fallback or custom image
      
      // Create SVG meteorological simulation overlay
      const svgRadarOverlay = L.svgOverlay(createRadarSvgElement(), bounds, {
        opacity: 0.68,
        interactive: false
      });
      svgRadarOverlay.addTo(map);

      mapInstanceRef.current = map;
    }

    const map = mapInstanceRef.current;
    if (map) {
      // Pan to selected city
      map.flyTo([currentData.lat, currentData.lng], 6, {
        duration: 1.2,
        easeLinearity: 0.25
      });

      // Update City Marker
      if (markerRef.current) {
        map.removeLayer(markerRef.current);
      }

      const customIcon = L.divIcon({
        className: 'custom-map-pin',
        html: `
          <div style="
            background: #1e70eb;
            color: white;
            border: 2px solid white;
            border-radius: 50%;
            width: 22px;
            height: 22px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 12px rgba(30,112,235,0.6);
            transform: translate(-11px, -11px);
          ">
            <div style="width: 8px; height: 8px; border-radius: 50%; background: white;"></div>
          </div>
        `,
        iconSize: [22, 22]
      });

      markerRef.current = L.marker([currentData.lat, currentData.lng], { icon: customIcon }).addTo(map);
    }
  }, [currentCityKey, currentData]);

  const handleZoomIn = () => {
    if (mapInstanceRef.current) mapInstanceRef.current.zoomIn();
  };

  const handleZoomOut = () => {
    if (mapInstanceRef.current) mapInstanceRef.current.zoomOut();
  };

  const handleLocateMe = () => {
    if (mapInstanceRef.current) {
      mapInstanceRef.current.flyTo([currentData.lat, currentData.lng], 7);
    }
  };

  return (
    <section id="forecast" className="map-dashboard-section">
      <div className="container">
        {/* Controls Bar */}
        <div className="map-controls-bar">
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
                position: 'absolute',
                top: '110%',
                left: 0,
                width: '100%',
                background: '#ffffff',
                border: '1px solid #e2e8f0',
                borderRadius: '12px',
                boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
                zIndex: 50,
                overflow: 'hidden'
              }}>
                {Object.keys(CITIES_DATA).map((key) => (
                  <div
                    key={key}
                    onClick={() => {
                      onSelectCity(key);
                      setIsCityDropdownOpen(false);
                    }}
                    style={{
                      padding: '10px 16px',
                      fontSize: '13.5px',
                      fontWeight: 600,
                      cursor: 'pointer',
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
          <div className="layer-tabs">
            <button
              className={`layer-tab-btn ${activeLayer === 'rainfall' ? 'active' : ''}`}
              onClick={() => setActiveLayer('rainfall')}
            >
              Rainfall
            </button>
            <button
              className={`layer-tab-btn ${activeLayer === 'temp' ? 'active' : ''}`}
              onClick={() => setActiveLayer('temp')}
            >
              Temperature
            </button>
            <button
              className={`layer-tab-btn ${activeLayer === 'wind' ? 'active' : ''}`}
              onClick={() => setActiveLayer('wind')}
            >
              Wind
            </button>
          </div>

          {/* Timeframe Dropdown */}
          <div className="time-dropdown">
            <select
              value={selectedHorizon}
              onChange={(e) => setSelectedHorizon(e.target.value)}
            >
              <option value="Next 6 hours">Next 6 hours</option>
              <option value="Next 12 hours">Next 12 hours</option>
              <option value="Next 24 hours">Next 24 hours</option>
              <option value="Next 48 hours">Next 48 hours</option>
              <option value="Next 7 days">Next 7 days</option>
            </select>
          </div>
        </div>

        {/* Map Viewport */}
        <div className="map-viewport-wrapper">
          <div id="india-weather-map" ref={mapContainerRef}></div>

          {/* Map Custom Controls (+, -, locate) */}
          <div className="map-custom-controls">
            <button className="map-control-btn" onClick={handleZoomIn} title="Zoom In">
              <Plus size={18} />
            </button>
            <button className="map-control-btn" onClick={handleZoomOut} title="Zoom Out">
              <Minus size={18} />
            </button>
            <button className="map-control-btn" onClick={handleLocateMe} title="Center on Location">
              <Crosshair size={18} />
            </button>
          </div>

          {/* Rainfall / Variable Legend */}
          <div className="map-rainfall-legend">
            <div className="legend-title">
              {activeLayer === 'rainfall' && 'Rainfall (mm)'}
              {activeLayer === 'temp' && 'Temperature (°C)'}
              {activeLayer === 'wind' && 'Wind Speed (km/h)'}
            </div>
            
            {activeLayer === 'rainfall' && (
              <div className="legend-items">
                <div className="legend-row">
                  <div className="legend-color-swatch" style={{ background: '#7e22ce' }}></div>
                  <span>&gt; 200</span>
                </div>
                <div className="legend-row">
                  <div className="legend-color-swatch" style={{ background: '#ef4444' }}></div>
                  <span>100 – 200</span>
                </div>
                <div className="legend-row">
                  <div className="legend-color-swatch" style={{ background: '#f97316' }}></div>
                  <span>50 – 100</span>
                </div>
                <div className="legend-row">
                  <div className="legend-color-swatch" style={{ background: '#eab308' }}></div>
                  <span>20 – 50</span>
                </div>
                <div className="legend-row">
                  <div className="legend-color-swatch" style={{ background: '#22c55e' }}></div>
                  <span>5 – 20</span>
                </div>
                <div className="legend-row">
                  <div className="legend-color-swatch" style={{ background: '#06b6d4' }}></div>
                  <span>1 – 5</span>
                </div>
                <div className="legend-row">
                  <div className="legend-color-swatch" style={{ background: '#2563eb' }}></div>
                  <span>&lt; 1</span>
                </div>
              </div>
            )}

            {activeLayer === 'temp' && (
              <div className="legend-items">
                <div className="legend-row">
                  <div className="legend-color-swatch" style={{ background: '#b91c1c' }}></div>
                  <span>&gt; 42°C</span>
                </div>
                <div className="legend-row">
                  <div className="legend-color-swatch" style={{ background: '#ea580c' }}></div>
                  <span>36 – 42°C</span>
                </div>
                <div className="legend-row">
                  <div className="legend-color-swatch" style={{ background: '#facc15' }}></div>
                  <span>30 – 36°C</span>
                </div>
                <div className="legend-row">
                  <div className="legend-color-swatch" style={{ background: '#4ade80' }}></div>
                  <span>24 – 30°C</span>
                </div>
                <div className="legend-row">
                  <div className="legend-color-swatch" style={{ background: '#38bdf8' }}></div>
                  <span>&lt; 24°C</span>
                </div>
              </div>
            )}

            {activeLayer === 'wind' && (
              <div className="legend-items">
                <div className="legend-row">
                  <div className="legend-color-swatch" style={{ background: '#dc2626' }}></div>
                  <span>&gt; 60 km/h</span>
                </div>
                <div className="legend-row">
                  <div className="legend-color-swatch" style={{ background: '#f59e0b' }}></div>
                  <span>40 – 60 km/h</span>
                </div>
                <div className="legend-row">
                  <div className="legend-color-swatch" style={{ background: '#10b981' }}></div>
                  <span>20 – 40 km/h</span>
                </div>
                <div className="legend-row">
                  <div className="legend-color-swatch" style={{ background: '#0284c7' }}></div>
                  <span>&lt; 20 km/h</span>
                </div>
              </div>
            )}
          </div>

          {/* Floating Weather Detail Card on Top Right */}
          <div className="map-floating-stats-card">
            <h4 className="mf-location">{currentData.city}, {currentData.state}</h4>
            <p className="mf-period">{selectedHorizon}</p>

            <div className="mf-items-list">
              <div className="mf-item-row">
                <div className="mf-icon-wrapper" style={{ background: '#e0f2fe' }}>
                  <CloudRain size={18} color="#0284c7" />
                </div>
                <div className="mf-item-info">
                  <span className="mf-item-val">{currentData.rainfallBlended}</span>
                  <span className="mf-item-lbl">Rainfall (Blended)</span>
                </div>
              </div>

              <div className="mf-item-row">
                <div className="mf-icon-wrapper" style={{ background: '#fee2e2' }}>
                  <Thermometer size={18} color="#dc2626" />
                </div>
                <div className="mf-item-info">
                  <span className="mf-item-val">{currentData.temp} °C</span>
                  <span className="mf-item-lbl">Temperature</span>
                </div>
              </div>

              <div className="mf-item-row">
                <div className="mf-icon-wrapper" style={{ background: '#ecfdf5' }}>
                  <Wind size={18} color="#059669" />
                </div>
                <div className="mf-item-info">
                  <span className="mf-item-val">{currentData.wind} km/h</span>
                  <span className="mf-item-lbl">Wind Speed</span>
                </div>
              </div>

              <div className="mf-item-row">
                <div className="mf-icon-wrapper" style={{ background: '#eff6ff' }}>
                  <Droplets size={18} color="#2563eb" />
                </div>
                <div className="mf-item-info">
                  <span className="mf-item-val">{currentData.humidity}%</span>
                  <span className="mf-item-lbl">Humidity</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// Helper function to generate an SVG radar heatmap mimicking Indian weather patterns
function createRadarSvgElement() {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 1000 1000');
  svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  svg.innerHTML = `
    <defs>
      <radialGradient id="heavyRainMumbai" cx="30%" cy="55%" r="35%">
        <stop offset="0%" stop-color="#ef4444" stop-opacity="0.9" />
        <stop offset="35%" stop-color="#f59e0b" stop-opacity="0.8" />
        <stop offset="60%" stop-color="#10b981" stop-opacity="0.65" />
        <stop offset="85%" stop-color="#0284c7" stop-opacity="0.45" />
        <stop offset="100%" stop-color="#2563eb" stop-opacity="0" />
      </radialGradient>
      <radialGradient id="heavyRainNE" cx="75%" cy="35%" r="28%">
        <stop offset="0%" stop-color="#dc2626" stop-opacity="0.85" />
        <stop offset="40%" stop-color="#f97316" stop-opacity="0.75" />
        <stop offset="70%" stop-color="#22c55e" stop-opacity="0.6" />
        <stop offset="100%" stop-color="#3b82f6" stop-opacity="0" />
      </radialGradient>
      <radialGradient id="cycloneBay" cx="65%" cy="65%" r="30%">
        <stop offset="0%" stop-color="#7e22ce" stop-opacity="0.85" />
        <stop offset="30%" stop-color="#3b82f6" stop-opacity="0.7" />
        <stop offset="65%" stop-color="#06b6d4" stop-opacity="0.5" />
        <stop offset="100%" stop-color="#0284c7" stop-opacity="0" />
      </radialGradient>
    </defs>
    <rect x="0" y="0" width="1000" height="1000" fill="transparent" />
    <circle cx="320" cy="550" r="240" fill="url(#heavyRainMumbai)" />
    <circle cx="750" cy="350" r="190" fill="url(#heavyRainNE)" />
    <circle cx="650" cy="650" r="220" fill="url(#cycloneBay)" />
  `;
  return svg;
}
