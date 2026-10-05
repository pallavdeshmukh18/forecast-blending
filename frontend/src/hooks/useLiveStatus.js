import { useCallback, useEffect, useRef, useState } from 'react';
import { getLiveStatus } from '../api';

export const REFRESH_SECONDS = 300;

/** Map backend freshness + age onto the three operator-facing states. */
export function classifyLiveState(status, failed) {
  if (failed || !status) return 'offline';
  const age = Number(status.data_age_minutes);
  if (status.freshness === 'unavailable' || !Number.isFinite(age)) return 'offline';
  if (age < 60) return 'fresh';
  if (age <= 180) return 'stale';
  return 'offline';
}

/**
 * Polls GET /api/live/status, exposes a countdown to the next auto-refresh and
 * invokes `onRefresh` (page data reload) when the countdown elapses.
 */
export default function useLiveStatus({ onRefresh, refreshSeconds = REFRESH_SECONDS } = {}) {
  const [status, setStatus] = useState(null);
  const [failed, setFailed] = useState(false);
  const [secondsLeft, setSecondsLeft] = useState(refreshSeconds);
  const onRefreshRef = useRef(onRefresh);
  useEffect(() => {
    onRefreshRef.current = onRefresh;
  }, [onRefresh]);

  const poll = useCallback(async () => {
    try {
      setStatus(await getLiveStatus());
      setFailed(false);
    } catch {
      setFailed(true);
    }
  }, []);

  const refreshNow = useCallback(async () => {
    setSecondsLeft(refreshSeconds);
    await onRefreshRef.current?.();
    await poll();
  }, [poll, refreshSeconds]);

  useEffect(() => {
    poll();
    const pollId = setInterval(poll, 30000);
    return () => clearInterval(pollId);
  }, [poll]);

  useEffect(() => {
    const id = setInterval(() => setSecondsLeft((s) => Math.max(0, s - 1)), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (secondsLeft === 0) refreshNow();
  }, [secondsLeft, refreshNow]);

  return { status, failed, state: classifyLiveState(status, failed), secondsLeft, refreshNow, poll };
}
