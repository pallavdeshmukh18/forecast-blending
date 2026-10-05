import { useEffect, useRef } from 'react';
import {
  SCALAR_LAYERS,
  WindParticleEngine,
  createScalarSampler,
  createWindSampler,
  disagreementLayer,
  drawScalarLayer,
} from '../windField';

/**
 * Canvas overlay attached to a Leaflet map.
 *  layer: 'wind' (animated particles) | 'rain' | 'temp' | 'disagreement'
 * The canvas lives in the map container (not a pane), so it never needs re-positioning:
 * it is cleared on movestart/zoomstart and redrawn on moveend/zoomend.
 */
export default function WindParticleOverlay({ map, grid, layer, disagreementVar = 'wind_speed_ms' }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!map || !grid?.cells?.length) return undefined;

    const container = map.getContainer();
    const canvas = document.createElement('canvas');
    canvas.className = 'field-overlay-canvas';
    Object.assign(canvas.style, {
      position: 'absolute',
      top: '0',
      left: '0',
      pointerEvents: 'none',
      zIndex: '450',
    });
    container.appendChild(canvas);
    canvasRef.current = canvas;

    const getSize = () => {
      const s = map.getSize();
      return { w: s.x, h: s.y };
    };
    const unproject = (x, y) => {
      const ll = map.containerPointToLatLng([x, y]);
      return [ll.lat, ll.lng];
    };

    let engine = null;
    let draw = null;

    if (layer === 'wind') {
      engine = new WindParticleEngine(canvas, {
        getSize,
        unproject,
        sampleWind: createWindSampler(grid),
      });
      draw = () => {
        engine.stop();
        engine.resize();
        engine.start();
      };
    } else {
      const def = layer === 'disagreement' ? disagreementLayer(disagreementVar) : SCALAR_LAYERS[layer];
      const sample = createScalarSampler(grid, def.accessor);
      draw = () => {
        const { w, h } = getSize();
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
        canvas.style.width = `${w}px`;
        canvas.style.height = `${h}px`;
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        drawScalarLayer(ctx, w, h, unproject, sample, def.stops);
      };
    }

    const onStart = () => {
      if (engine) {
        engine.stop();
        engine.clear();
      } else {
        canvas.style.opacity = '0';
      }
    };
    const onEnd = () => {
      canvas.style.opacity = '1';
      draw();
    };

    map.on('movestart zoomstart', onStart);
    map.on('moveend zoomend resize', onEnd);
    draw();

    return () => {
      map.off('movestart zoomstart', onStart);
      map.off('moveend zoomend resize', onEnd);
      if (engine) engine.stop();
      canvas.remove();
    };
  }, [map, grid, layer, disagreementVar]);

  return null;
}
