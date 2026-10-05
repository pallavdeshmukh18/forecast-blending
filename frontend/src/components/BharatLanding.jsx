import { createElement, useState } from 'react';
import { ArrowRight, MapPin, Layers3, TriangleAlert, ChartNoAxesColumnIncreasing, Sun, CloudRain, Wind, Github, Linkedin, Menu, X, Leaf } from 'lucide-react';
import './bharat-landing.css';

const features = [
  { Icon: Layers3, title: 'Multi-Model Blending', text: <>Combines NWP, ensemble<br />and AI models intelligently</>, color: 'mint' },
  { Icon: MapPin, title: 'Region-Specific Forecasts', text: <>Adaptive to India’s diverse<br />climate and terrain</>, color: 'blue' },
  { Icon: TriangleAlert, title: 'Early Extreme Alerts', text: <>Heavy rainfall, heatwaves,<br />high winds and more</>, color: 'orange' },
  { Icon: ChartNoAxesColumnIncreasing, title: 'Explainable Forecasts', text: <>See why a forecast is made<br />with clear model insights</>, color: 'purple' },
];
const impacts = [
  ['agriculture', 'Agriculture', 'Plan sowing and harvesting with better accuracy.'],
  ['river', 'Disaster Management', 'Early alerts for heavy rainfall, cyclones and floods.'],
  ['wind', 'Infrastructure', 'Plan operations considering weather risks.'],
  ['city', 'Public Use', 'Stay informed with reliable forecasts for your city.'],
];
const series = [
  ['Blended', '#15382f', [25, 52, 99, 58, 44, 26]],
  ['ECMWF', '#349cfa', [24, 48, 133, 64, 38, 26]],
  ['GFS', '#ff6941', [25, 65, 169, 101, 61, 27]],
  ['GraphCast', '#6cd7a3', [25, 56, 79, 49, 48, 25]],
  ['Pangu', '#bb68ec', [25, 101, 85, 84, 37, 25]],
  ['AIFS', '#f3c952', [25, 62, 154, 97, 80, 26]],
];

function Brand({ footer = false }) {
  return <a className="bw-brand" href="#home" aria-label="BharatWeather home"><span className="bw-brand-mark"><Leaf size={19} /><Sun size={12} /></span><span><strong>BharatWeather</strong>{footer && <small>Smarter forecasts. Safer Bharat.</small>}</span></a>;
}
function Note({ children, className = '' }) {
  return <div className={`bw-note ${className}`}>{children}<svg viewBox="0 0 60 42" fill="none" aria-hidden="true"><path d="M42 3C44 22 27 30 9 29m0 0 7-6m-7 6 8 5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" /></svg></div>;
}
function IndiaMap() {
  return <svg className="bw-india" viewBox="0 0 390 490" aria-label="Illustration of India’s weather regions" role="img">
    <defs><linearGradient id="north-fill"><stop stopColor="#ffe96a" /><stop offset="1" stopColor="#fff59c" /></linearGradient></defs>
    <path className="bw-india-outline" d="M111 17 126 12 138 21 143 33 157 37 166 48 183 42 196 44 193 57 180 67 180 79 164 88 169 101 183 113 183 129 199 139 203 151 223 159 239 160 256 174 282 177 289 161 303 166 313 156 329 155 340 144 356 147 352 158 366 166 353 178 354 188 341 196 337 214 326 220 316 214 317 194 304 193 294 203 282 198 275 213 260 218 251 235 234 246 225 264 211 279 202 302 183 321 174 343 163 360 160 387 144 402 137 427 121 446 113 459 105 457 98 439 91 417 88 399 78 378 71 356 65 332 60 304 55 287 49 267 49 250 35 253 20 238 13 224 23 216 40 216 47 207 37 198 41 182 55 174 58 161 70 151 73 137 87 125 95 112 94 97 102 88 98 76 87 66 90 52 102 45 99 33 108 28Z" />
    <path fill="url(#north-fill)" d="M95 112 109 115 117 105 130 115 145 110 151 124 169 121 183 129 199 139 203 151 223 159 239 160 256 174 245 189 225 199 209 200 198 213 180 213 169 232 158 250 153 268 143 264 132 254 119 251 108 242 99 233 97 220 88 211 91 197 82 186 86 172 73 167 70 151 73 137 87 125Z" />
    <g fill="none" stroke="#fff" strokeWidth="1" opacity=".52"><path d="m102 45 18 15 29-7 15 35m-70 9 26-4 23 17m-94 140 30-10 20-7m-50 34 31 5 27-30m-47 62 42-3 24-26 27-7m-83 88 30-9 10-34 31 12 29-9m-48 98 30-19 1-40 23-3m-39 82 25-11m63-174-8-30 14-19m-59 75 34 8 21-14m-48 60 29-9m61-118 12 18 32-1m-7-37 6 19 22-2m-1-27 16 7 19-14" /></g>
    <g fill="none" stroke="#61766b" strokeWidth="1.2"><path d="M167 187q35 35 79 17" /><path d="M89 326q-27 32-52 23" /><path d="M143 389q23 31 52 24" /></g><g fill="#153e33"><circle cx="167" cy="187" r="3" /><circle cx="89" cy="326" r="3" /><circle cx="143" cy="389" r="3" /></g>
  </svg>;
}
function ForecastChart({ city }) {
  const [metric, setMetric] = useState('Rainfall');
  const [hidden, setHidden] = useState([]);
  const units = metric === 'Rainfall' ? 'mm' : '°C';
  const ticks = metric === 'Rainfall' ? [200, 150, 100, 50, 0] : [40, 35, 30, 25, 20];
  return <div className="bw-chart">
    <div className="bw-chart-top"><strong>{metric} Forecast – {city}</strong><label><span className="bw-sr-only">Forecast variable</span><select value={metric} onChange={e => setMetric(e.target.value)}><option value="Rainfall">Rainfall (mm)</option><option value="Temperature">Temperature (°C)</option></select></label></div>
    <svg viewBox="0 0 500 215" role="img" aria-label={`Illustrative ${metric.toLowerCase()} model comparison for ${city}, measured in ${units}`}>
      {ticks.map((tick, i) => <g key={tick}><line x1="34" y1={22 + i * 37} x2="487" y2={22 + i * 37} stroke="#e6edef" /><text x="23" y={26 + i * 37} textAnchor="end">{tick}</text></g>)}
      {[...series].reverse().map(([name, color, values]) => !hidden.includes(name) && <g key={name}><polyline fill="none" stroke={color} strokeWidth="1.8" strokeLinejoin="round" points={values.map((v, i) => `${34 + i * 90.6},${170 - (metric === 'Rainfall' ? v : v * .55 + 35) * .74}`).join(' ')} />{values.map((v, i) => <circle key={i} cx={34 + i * 90.6} cy={170 - (metric === 'Rainfall' ? v : v * .55 + 35) * .74} r="2.4" fill={color} />)}</g>)}
      {['26 Nov', '27 Nov', '27 Nov', '28 Nov', '29 Nov', '30 Nov'].map((day, i) => <text key={i} x={34 + i * 90.6} y="191" textAnchor="middle">{day}</text>)}
    </svg>
    <div className="bw-chart-legend">{series.map(([name, color]) => <button key={name} type="button" aria-pressed={!hidden.includes(name)} onClick={() => setHidden(prev => prev.includes(name) ? prev.filter(n => n !== name) : [...prev, name])} style={{ opacity: hidden.includes(name) ? .35 : 1 }}><i style={{ background: color }} />{name}</button>)}</div>
    <span className="bw-sr-only">Illustrative model data, not a live forecast. Select a legend item to show or hide a model.</span>
  </div>;
}

export default function BharatLanding({ onGetStarted }) {
  const [query, setQuery] = useState('');
  const [city, setCity] = useState('Mumbai');
  const [searchMessage, setSearchMessage] = useState('');
  const [menuOpen, setMenuOpen] = useState(false);
  const cities = ['Mumbai', 'Pune', 'Delhi', 'Bengaluru', 'Chennai'];
  function search(event, chosen) {
    event?.preventDefault();
    const value = chosen || query.trim();
    if (!value) return;
    const match = cities.find(c => c.toLowerCase() === value.toLowerCase());
    if (!match) { setSearchMessage('Choose Mumbai, Pune, Delhi, Bengaluru or Chennai to explore the model preview.'); return; }
    setCity(match); setQuery(match); setSearchMessage('');
    document.getElementById('model-insights')?.scrollIntoView({ behavior: 'smooth' });
  }
  const navigation = <><a href="#home">Home</a><a href="#forecast">Forecast</a><a href="#model-insights">Model Insights</a><a href="#extremes">Extremes</a><a href="#about">About</a></>;
  return <div className="bw-page" id="home">
    <header className="bw-header"><Brand /><nav aria-label="Main navigation" className={menuOpen ? 'bw-nav is-open' : 'bw-nav'} onClick={() => setMenuOpen(false)}>{navigation}</nav><button className="bw-button bw-header-cta" onClick={onGetStarted}>Get Started</button><button className="bw-menu" aria-label={menuOpen ? 'Close menu' : 'Open menu'} aria-expanded={menuOpen} onClick={() => setMenuOpen(!menuOpen)}>{menuOpen ? <X /> : <Menu />}</button></header>
    <main>
      <section className="bw-hero" aria-labelledby="hero-title">
        <div className="bw-hero-content"><h1 id="hero-title">Smarter Weather<br />Forecasts for a Safer Bharat</h1><p>Blending the best of global weather models with AI to deliver<br className="bw-desktop-break" /> accurate, region-specific forecasts and early warnings.</p>
          <form className="bw-search" onSubmit={search}><MapPin size={20} /><label className="bw-sr-only" htmlFor="bw-location">Search for a city, district or state</label><input id="bw-location" value={query} onChange={e => { setQuery(e.target.value); setSearchMessage(''); }} placeholder="Search for a city, district or state..." list="bw-cities" required /><datalist id="bw-cities">{cities.map(c => <option key={c} value={c} />)}</datalist><button type="submit" aria-label="Search forecasts"><ArrowRight /></button></form>
          {searchMessage && <p className="bw-search-message" role="status">{searchMessage}</p>}
          <div className="bw-cities">{cities.map(c => <button key={c} onClick={e => search(e, c)}>{c}</button>)}</div>
        </div><Note className="bw-hero-note">Better forecasts<br />for stronger<br />communities</Note>
      </section>
      <section className="bw-features" aria-label="Forecast capabilities">{features.map(({ Icon, title, text, color }, i) => <a href={i === 2 ? '#extremes' : i === 1 ? '#forecast' : '#model-insights'} className="bw-feature" key={title}><span className={`bw-feature-icon ${color}`}>{createElement(Icon)}</span><h3>{title}</h3><p>{text}</p></a>)}</section>
      <section className="bw-regions" id="forecast" aria-labelledby="region-title"><div className="bw-region-copy"><span className="bw-eyebrow">Built for India</span><h2 id="region-title">Different regions.<br />Different weather<br />behaviours.</h2><p>Our system learns from historical performance<br className="bw-desktop-break" /> across regions, seasons and weather regimes<br className="bw-desktop-break" /> to give you the most reliable forecast.</p><button className="bw-button" onClick={onGetStarted}>Explore Region Forecasts <ArrowRight /></button></div><div className="bw-region-visual"><IndiaMap /><button className="bw-region-card bw-north" onClick={onGetStarted}><Sun className="bw-sun" /><div><h3>North India</h3><p>Higher risk of<br />heatwaves in<br />May-June.</p></div><ArrowRight className="bw-card-arrow" /></button><button className="bw-region-card bw-west" onClick={onGetStarted}><CloudRain className="bw-rain" /><div><h3>West India</h3><p>Heavy rainfall<br />during monsoon.</p></div><ArrowRight className="bw-card-arrow" /></button><button className="bw-region-card bw-east" id="extremes" onClick={onGetStarted}><Wind /><div><h3>East &amp; Northeast</h3><p>High wind events<br />and cyclones.</p></div><ArrowRight className="bw-card-arrow" /></button></div><Note className="bw-region-note">From mountains<br />to coastlines</Note></section>
      <section className="bw-impact" id="about" aria-labelledby="impact-title"><div className="bw-impact-heading"><div><span className="bw-eyebrow">Real impact</span><h2 id="impact-title">Accurate forecasts<br />for real decisions.</h2></div><p>Help farmers, disaster management teams,<br className="bw-desktop-break" /> planners and citizens prepare better with<br className="bw-desktop-break" /> reliable, actionable weather information.</p></div><div className="bw-impact-grid">{impacts.map(([image, title, description]) => <article key={title}><img src={`/images/${image}.jpg`} alt={image === 'agriculture' ? 'Farmer overlooking a green field at sunrise' : image === 'river' ? 'River and bridge surrounded by lush green trees' : image === 'wind' ? 'Wind turbines in open countryside' : 'Mumbai skyline above a leafy neighbourhood'} loading="lazy" /><div><h3>{title}</h3><p>{description}</p></div></article>)}</div></section>
      <div className="bw-bottom"><section className="bw-models" id="model-insights" aria-labelledby="models-title"><div className="bw-model-copy"><span className="bw-eyebrow">Model intelligence</span><h2 id="models-title">Bringing multiple<br />models together</h2><p>We combine physical NWP models<br className="bw-desktop-break" /> (ECMWF, GFS), AI models (GraphCast,<br className="bw-desktop-break" /> Pangu, AIFS) and ensembles to create<br className="bw-desktop-break" /> a single, reliable forecast.</p><button className="bw-text-link" onClick={onGetStarted}>See model comparison <ArrowRight /></button></div><ForecastChart city={city} /></section>
      <section className="bw-cta" aria-labelledby="cta-title"><div><span className="bw-eyebrow">A safer, more resilient tomorrow</span><h2 id="cta-title">Weather intelligence<br />for a stronger Bharat</h2><button className="bw-button" onClick={onGetStarted}>Get Started <ArrowRight /></button></div><Note className="bw-cta-note">Informed today<br />for a safer<br />tomorrow</Note></section></div>
    </main>
    <footer className="bw-footer"><Brand footer /><nav aria-label="Footer navigation">{navigation}</nav><div className="bw-socials"><a href="https://x.com" aria-label="X"><svg viewBox="0 0 24 24" fill="none"><path d="m5 3 14 18h-4L1 3h4Zm-3 18L19 3" stroke="currentColor" strokeWidth="1.5" /></svg></a><a href="https://linkedin.com" aria-label="LinkedIn"><Linkedin /></a><a href="https://github.com" aria-label="GitHub"><Github /></a></div></footer>
  </div>;
}
