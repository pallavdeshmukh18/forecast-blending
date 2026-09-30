# AtmosVision React forecast map

A full-stack weather map built with React, Vite and MapLibre GL JS. Forecast
fields are loaded from the FastAPI operational pipeline, which runs the
configured multi-model blending, uncertainty, extremes and verification flow.

## Start the app

```bash
cd ..
./scripts/run_full_stack.sh
```

This starts FastAPI at `http://127.0.0.1:8000` and Vite at
`http://localhost:5173`. The Vite `/api` proxy forwards frontend requests to
the operational forecast pipeline.

## Commands

```bash
npm run dev      # local development server
npm run build    # optimized production build
npm run preview  # preview the production build
npm run lint     # ESLint checks
```

## Map stack

- **MapLibre GL JS 6** — WebGL vector map renderer, camera, sources and layers
- **OpenStreetMap** — lightweight raster tiles with no key for this prototype
- **OpenFreeMap glyphs** — labels used by custom weather layers
- **GeoJSON layers** — API forecast fields, heatmaps and risk points
- **Lucide React** — interface icons

The basemap requires internet access, but no map API key. When FastAPI is not
available, the interface clearly reports the connection error and keeps a local
preview field visible instead of covering or breaking the map.
