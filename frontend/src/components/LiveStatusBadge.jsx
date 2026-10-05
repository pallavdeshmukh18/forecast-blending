import { RefreshCw } from 'lucide-react';

const STATES = {
  fresh: { dot: '🟢', label: 'LIVE / FRESH', color: '#047857', bg: '#d1fae5', border: '#6ee7b7' },
  stale: { dot: '🟡', label: 'STALE', color: '#92400e', bg: '#fef3c7', border: '#fcd34d' },
  offline: { dot: '🔴', label: 'OFFLINE / CACHE FALLBACK', color: '#991b1b', bg: '#fee2e2', border: '#fca5a5' },
};

const fmtCountdown = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

/** Presentational badge; pass the object returned by useLiveStatus(). */
export default function LiveStatusBadge({ live, compact = false }) {
  const { status, state, secondsLeft, refreshNow } = live;
  const s = STATES[state];
  const utc = status?.last_retrieved_utc ? status.last_retrieved_utc.slice(0, 16).replace('T', ' ') + ' UTC' : '— UTC';
  const age = status && Number.isFinite(status.data_age_minutes) && state !== 'offline' ? `${Math.round(status.data_age_minutes)}m old` : null;

  return (
    <div
      className="live-status-badge"
      title={`Provider: ${status?.provider || 'n/a'} · latency ${status?.latency_ms ?? 0} ms`}
      style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '5px 10px', borderRadius: 999, fontSize: 12, fontWeight: 700, color: s.color, background: s.bg, border: `1px solid ${s.border}` }}
    >
      <span>{s.dot}</span>
      <span>{s.label}</span>
      {!compact && <span style={{ fontWeight: 500 }}>{[age, utc].filter(Boolean).join(' · ')}</span>}
      <span style={{ fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>↻ {fmtCountdown(secondsLeft)}</span>
      <button type="button" onClick={refreshNow} aria-label="Refresh now" style={{ border: 0, background: 'transparent', color: 'inherit', cursor: 'pointer', display: 'flex', padding: 0 }}>
        <RefreshCw size={12} />
      </button>
    </div>
  );
}
