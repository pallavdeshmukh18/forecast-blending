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
    return Number.isFinite(v) ? v : null;
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
 * rebuilt whenever the view changes; white trails fade toward their tails.
 */
export class WindParticleEngine {
  constructor(canvas, { getSize, unproject, project, sampleWind, density = 0.0006, maxParticles = 1400, speedScale = 0.30 }) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.getSize = getSize;
    this.unproject = unproject;
    this.project = project;
    this.sampleWind = sampleWind;
    this.density = density;
    this.maxParticles = maxParticles;
    this.speedScale = speedScale;
    // Longer histories reveal the flow in a sparse grid without extra weather samples.
    this.trailLength = 36;
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
    this.field = buildPixelField(this.w, this.h, 6, this.unproject, (lat, lng) => {
      const wind = this.sampleWind(lat, lng);
      if (!wind || !Number.isFinite(wind.speed)) return null;
      if (!this.project || wind.speed === 0) return { ...wind, dx: wind.u, dy: -wind.v };
      // Project a short east/north displacement, including longitude's latitude scale.
      const origin = this.project(lat, lng);
      const next = this.project(lat + wind.v * 0.0001,
        lng + wind.u * 0.0001 / Math.max(0.01, Math.cos(lat * Math.PI / 180)));
      const dx = next[0] - origin[0], dy = next[1] - origin[1];
      const length = Math.hypot(dx, dy);
      if (!Number.isFinite(length) || length === 0) return null;
      return { ...wind, dx: dx / length * wind.speed, dy: dy / length * wind.speed };
    });
    this.ctx.clearRect(0, 0, this.w, this.h);
  }

  spawn(p) {
    p.x = Math.random() * this.w;
    p.y = Math.random() * this.h;
    p.age = 0;
    p.max = 150 + Math.random() * 90; // 2.5–4 seconds at the fixed 60 Hz simulation rate
    p.trail = [];
    return p;
  }

  lookup(x, y) {
    if (x < 0 || y < 0 || x >= this.w || y >= this.h) return null;
    const f = this.field;
    const gx = Math.max(0, Math.min(f.cols - 1, x / f.step - 0.5));
    const gy = Math.max(0, Math.min(f.rows - 1, y / f.step - 0.5));
    const c = Math.floor(gx), r = Math.floor(gy);
    const tx = gx - c, ty = gy - r;
    const samples = [f.data[r * f.cols + c], f.data[r * f.cols + Math.min(c + 1, f.cols - 1)],
      f.data[Math.min(r + 1, f.rows - 1) * f.cols + c],
      f.data[Math.min(r + 1, f.rows - 1) * f.cols + Math.min(c + 1, f.cols - 1)]];
    if (samples.some(value => !value)) return null;
    const weights = [(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty];
    return samples.reduce((out, value, i) => ({
      dx: out.dx + value.dx * weights[i], dy: out.dy + value.dy * weights[i],
    }), { dx: 0, dy: 0 });
  }

  advance() {
    for (const p of this.particles) {
      const wind = this.lookup(p.x, p.y);
      if (!wind || p.age >= p.max) {
        this.spawn(p);
        continue;
      }
      // Midpoint integration follows curves without abrupt turns at lookup-cell edges.
      const mid = this.lookup(p.x + wind.dx * this.speedScale / 2,
        p.y + wind.dy * this.speedScale / 2);
      if (!mid) { this.spawn(p); continue; }
      p.trail.push([p.x, p.y]);
      if (p.trail.length > this.trailLength) p.trail.shift();
      p.x += mid.dx * this.speedScale;
      p.y += mid.dy * this.speedScale;
      p.age++;
    }
  }

  frame = (time) => {
    if (!this.running) return;
    // Fixed simulation steps give the same movement and trail length on 60/120 Hz displays.
    this.elapsed += this.lastTime == null ? 1000 / 60 : Math.min(50, time - this.lastTime);
    this.lastTime = time;
    while (this.elapsed >= 1000 / 60) {
      this.advance();
      this.elapsed -= 1000 / 60;
    }
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.w, this.h);
    ctx.lineCap = 'round';
    // Draw tails first, then brighter heads. Explicit trails avoid persistent canvas ghosting.
    for (let segment = 0; segment < this.trailLength; segment++) {
      const strength = Math.pow((segment + 1) / this.trailLength, 0.7);
      ctx.beginPath();
      for (const p of this.particles) {
        const index = p.trail.length - this.trailLength + segment;
        if (index < 0) continue;
        const from = p.trail[index];
        const to = p.trail[index + 1] || [p.x, p.y];
        ctx.moveTo(...from);
        ctx.lineTo(...to);
      }
      // A subtle dark edge keeps the white flow visible over light map tiles.
      ctx.lineWidth = 2;
      ctx.strokeStyle = `rgba(25, 48, 62, ${strength * 0.16})`;
      ctx.stroke();
      ctx.lineWidth = 1;
      ctx.strokeStyle = `rgba(245, 252, 255, ${strength * 0.85})`;
      ctx.stroke();
    }
    this.raf = requestAnimationFrame(this.frame);
  };

  start() {
    if (this.running) return;
    this.running = true;
    this.lastTime = null;
    this.elapsed = 0;
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
