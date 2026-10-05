/**
 * Wind / scalar field utilities shared by the Leaflet landing map and the MapLibre workspace.
 * Grid input is the payload of GET /api/live/wind-grid (regular lat/lng axes, row-major cells).
 */

export const INDIA_BOUNDS = { latMin: 6, latMax: 36, lngMin: 67, lngMax: 98 };

// Wind speed colour ramp (m/s)
export const WIND_STOPS = [
  { max: 5, color: '#00f2fe', label: '< 5 m/s' },
  { max: 15, color: '#4facfe', label: '5 – 15 m/s' },
  { max: 25, color: '#f6d365', label: '15 – 25 m/s' },
  { max: Infinity, color: '#ff0844', label: '> 25 m/s (gale)' },
];

export function windColor(speedMs) {
  for (const s of WIND_STOPS) if (speedMs < s.max) return s.color;
  return WIND_STOPS[WIND_STOPS.length - 1].color;
}

export const msToKmh = (ms) => ms * 3.6;

// Scalar layer definitions: stops are [value, [r,g,b,a]] interpolated linearly.
export const SCALAR_LAYERS = {
  rain: {
    title: 'Precipitation (mm/h)',
    unit: 'mm/h',
    accessor: (c) => c.precipitation_mm,
    stops: [
      [0, [37, 99, 235, 0]],
      [0.1, [6, 182, 212, 150]],
      [1, [34, 197, 94, 185]],
      [3, [234, 179, 8, 205]],
      [8, [249, 115, 22, 220]],
      [20, [126, 34, 206, 235]],
    ],
    legend: [['> 20', '#7e22ce'], ['8 – 20', '#f97316'], ['3 – 8', '#eab308'], ['1 – 3', '#22c55e'], ['0.1 – 1', '#06b6d4'], ['dry', '#2563eb']],
  },
  temp: {
    title: 'Temperature (°C)',
    unit: '°C',
    accessor: (c) => c.temperature_c,
    stops: [
      [0, [56, 189, 248, 170]],
      [15, [56, 189, 248, 170]],
      [24, [74, 222, 128, 180]],
      [30, [250, 204, 21, 195]],
      [36, [234, 88, 12, 210]],
      [44, [185, 28, 28, 225]],
    ],
    legend: [['> 42', '#b91c1c'], ['36 – 42', '#ea580c'], ['30 – 36', '#facc15'], ['24 – 30', '#4ade80'], ['< 24', '#38bdf8']],
  },
};

export const DISAGREEMENT_VARS = {
  wind_speed_ms: { label: 'Wind', unit: 'm/s', max: 4 },
  temperature_c: { label: 'Temp', unit: '°C', max: 3 },
  precipitation_mm: { label: 'Rain', unit: 'mm/h', max: 2 },
};

export function disagreementLayer(variable) {
  const v = DISAGREEMENT_VARS[variable] || DISAGREEMENT_VARS.wind_speed_ms;
  const m = v.max;
  return {
    title: `|ECMWF − GFS| ${v.label} (${v.unit})`,
    unit: v.unit,
    accessor: (c) => c.disagreement?.[variable],
    stops: [
      [0, [34, 197, 94, 0]],
      [m * 0.15, [34, 197, 94, 130]],
      [m * 0.4, [250, 204, 21, 185]],
      [m * 0.7, [249, 115, 22, 210]],
      [m, [220, 38, 38, 230]],
    ],
    legend: [[`> ${m}`, '#dc2626'], [`${(m * 0.4).toFixed(1)} – ${m}`, '#f97316'], [`${(m * 0.15).toFixed(1)} – ${(m * 0.4).toFixed(1)}`, '#facc15'], ['low', '#22c55e']],
  };
}

function interpStops(stops, v) {
  if (v <= stops[0][0]) return stops[0][1];
  for (let i = 1; i < stops.length; i++) {
    if (v <= stops[i][0]) {
      const [v0, c0] = stops[i - 1];
      const [v1, c1] = stops[i];
      const t = (v - v0) / (v1 - v0 || 1);
      return [c0[0] + (c1[0] - c0[0]) * t, c0[1] + (c1[1] - c0[1]) * t, c0[2] + (c1[2] - c0[2]) * t, c0[3] + (c1[3] - c0[3]) * t];
    }
  }
  return stops[stops.length - 1][1];
}

/**
 * Build a bilinear sampler over the regular grid. `read(cell)` returns a number (or null).
 * Returns fn(lat, lng) -> number | null (null outside the grid or where data is missing).
 */
export function createScalarSampler(grid, read) {
  if (!grid?.cells?.length) return () => null;
  const { lat_axis: la, lon_axis: lo, cells } = grid;
  const nLon = lo.length;
  const at = (i, j) => {
    const v = read(cells[i * nLon + j]);
    return typeof v === 'number' ? v : null;
  };
  return (lat, lng) => {
    if (lat < la[0] || lat > la[la.length - 1] || lng < lo[0] || lng > lo[nLon - 1]) return null;
    let i = 0;
    while (i < la.length - 2 && lat > la[i + 1]) i++;
    let j = 0;
    while (j < nLon - 2 && lng > lo[j + 1]) j++;
    const ty = (lat - la[i]) / (la[i + 1] - la[i]);
    const tx = (lng - lo[j]) / (lo[j + 1] - lo[j]);
    const a = at(i, j), b = at(i, j + 1), c = at(i + 1, j), d = at(i + 1, j + 1);
    if (a == null || b == null || c == null || d == null) return null;
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
  };
}

/** Wind sampler -> fn(lat,lng) => {u,v,speed} | null */
export function createWindSampler(grid) {
  const su = createScalarSampler(grid, (c) => c?.u10);
  const sv = createScalarSampler(grid, (c) => c?.v10);
  return (lat, lng) => {
    const u = su(lat, lng);
    const v = sv(lat, lng);
    if (u == null || v == null) return null;
    return { u, v, speed: Math.hypot(u, v) };
  };
}

/** Wind direction the air moves TOWARD, as atan2(-u, -v) convention gives the FROM bearing (deg). */
export function windFromBearing(u, v) {
  return (Math.atan2(-u, -v) * 180 / Math.PI + 360) % 360;
}

/**
 * Pixel-space lookup (cell = `step` css px) of a function of (lat,lng) built via `unproject(x,y)->[lat,lng]`.
 */
function buildPixelField(w, h, step, unproject, sample) {
  const cols = Math.ceil(w / step), rows = Math.ceil(h / step);
  const out = new Array(cols * rows);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const [lat, lng] = unproject(c * step + step / 2, r * step + step / 2);
      out[r * cols + c] = sample(lat, lng);
    }
  }
  return { cols, rows, step, data: out };
}

/** Render a scalar layer into `canvas` (sized w x h css px, DPR handled by caller via ctx transform). */
export function drawScalarLayer(ctx, w, h, unproject, sample, stops, opacity = 0.7) {
  const step = 6;
  const f = buildPixelField(w, h, step, unproject, sample);
  const off = document.createElement('canvas');
  off.width = f.cols;
  off.height = f.rows;
  const octx = off.getContext('2d');
  const img = octx.createImageData(f.cols, f.rows);
  for (let i = 0; i < f.data.length; i++) {
    const v = f.data[i];
    if (v == null) continue;
    const [r, g, b, a] = interpStops(stops, v);
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = a * opacity / 0.7;
  }
  octx.putImageData(img, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(off, 0, 0, f.cols * step, f.rows * step);
}

/**
 * Animated particle streamlines (Windy-style). Particles advect through a pixel-space (u,v) field
 * rebuilt whenever the view changes; trails fade through destination-out compositing.
 */
export class WindParticleEngine {
  constructor(canvas, { getSize, unproject, sampleWind, density = 0.0016, maxParticles = 4500, speedScale = 0.32 }) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.getSize = getSize;
    this.unproject = unproject;
    this.sampleWind = sampleWind;
    this.density = density;
    this.maxParticles = maxParticles;
    this.speedScale = speedScale;
    this.raf = null;
    this.particles = [];
    this.field = null;
    this.running = false;
  }

  resize() {
    const { w, h } = this.getSize();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = w;
    this.h = h;
    this.canvas.width = Math.max(1, Math.round(w * dpr));
    this.canvas.height = Math.max(1, Math.round(h * dpr));
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.min(this.maxParticles, Math.round(w * h * this.density));
    this.particles = Array.from({ length: n }, () => this.spawn({}));
    this.rebuild();
  }

  rebuild() {
    this.field = buildPixelField(this.w, this.h, 6, this.unproject, this.sampleWind);
    this.ctx.clearRect(0, 0, this.w, this.h);
  }

  spawn(p) {
    p.x = Math.random() * this.w;
    p.y = Math.random() * this.h;
    p.age = Math.floor(Math.random() * 40);
    p.max = 50 + Math.random() * 50; // 50-100 frames
    p.px = p.x;
    p.py = p.y;
    return p;
  }

  lookup(x, y) {
    const f = this.field;
    const c = Math.floor(x / f.step), r = Math.floor(y / f.step);
    if (c < 0 || r < 0 || c >= f.cols || r >= f.rows) return null;
    return f.data[r * f.cols + c];
  }

  frame = () => {
    if (!this.running) return;
    const ctx = this.ctx;
    // fade existing trails
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = 'rgba(0,0,0,0.07)';
    ctx.fillRect(0, 0, this.w, this.h);
    ctx.globalCompositeOperation = 'source-over';
    ctx.lineWidth = 1.25;
    ctx.lineCap = 'round';

    // batch strokes per colour bucket
    const buckets = WIND_STOPS.map(() => []);
    for (const p of this.particles) {
      const wnd = this.lookup(p.x, p.y);
      if (!wnd || p.age >= p.max) {
        this.spawn(p);
        continue;
      }
      const nx = p.x + wnd.u * this.speedScale;
      const ny = p.y - wnd.v * this.speedScale; // screen y grows downward
      const bi = WIND_STOPS.findIndex((s) => wnd.speed < s.max);
      buckets[bi].push(p.x, p.y, nx, ny);
      p.px = p.x;
      p.py = p.y;
      p.x = nx;
      p.y = ny;
      p.age++;
    }
    buckets.forEach((seg, i) => {
      if (!seg.length) return;
      ctx.strokeStyle = WIND_STOPS[i].color;
      ctx.beginPath();
      for (let k = 0; k < seg.length; k += 4) {
        ctx.moveTo(seg[k], seg[k + 1]);
        ctx.lineTo(seg[k + 2], seg[k + 3]);
      }
      ctx.stroke();
    });
    this.raf = requestAnimationFrame(this.frame);
  };

  start() {
    if (this.running) return;
    this.running = true;
    this.raf = requestAnimationFrame(this.frame);
  }

  stop() {
    this.running = false;
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = null;
  }

  clear() {
    this.ctx?.clearRect(0, 0, this.w || 0, this.h || 0);
  }
}
