/* MapLibre owns its imperative lifecycle; refs and callbacks are intentionally stable. */
/* eslint-disable react-hooks/exhaustive-deps */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as maplibregl from 'maplibre-gl'
import { AlertTriangle, BarChart3, Check, Download, ExternalLink, ChevronDown, CloudRain, Cloudy, Crosshair, Droplets, Gauge, Layers3, LocateFixed, Menu, Minus, Pause, Play, Plus, Search, Settings2, Sparkles, ThermometerSun, Upload, Wind, X } from 'lucide-react'
import { cities, cityLayerValue, modelWeights, variables, weatherGeoJSON } from '../../data'
import { changeGeoJSON, forecastToGeoJSON, getHealth, runForecast, variableForLayer } from '../../weatherMapApi'
import { apiClient, getBlendingWeights, getLiveForecast, getLiveMetadata, getLiveStatus, getLiveWindGrid } from '../../api'
import { WindParticleEngine, createWindSampler } from '../../windField'

const rasterPaint={'raster-fade-duration':0}
const BASEMAPS={
  light:{label:'OpenStreetMap Light',layer:'bm-light'},
  dark:{label:'Dark Canvas',layer:'bm-dark'},
  satellite:{label:'Satellite Imagery',layer:'bm-satellite'}
}
const esriTiles=name=>[`https://server.arcgisonline.com/ArcGIS/rest/services/${name}/MapServer/tile/{z}/{y}/{x}`]
const baseStyle={version:8,glyphs:'https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf',
  sources:{
    osm:{type:'raster',tiles:['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],tileSize:256,attribution:'© OpenStreetMap contributors',maxzoom:19},
    dark:{type:'raster',tiles:esriTiles('Canvas/World_Dark_Gray_Base'),tileSize:256,attribution:'Tiles © Esri',maxzoom:16},
    satellite:{type:'raster',tiles:esriTiles('World_Imagery'),tileSize:256,attribution:'Tiles © Esri — Maxar, Earthstar Geographics',maxzoom:18}
  },
  layers:[
    {id:'bm-light',type:'raster',source:'osm',paint:{...rasterPaint,'raster-saturation':-.2,'raster-brightness-min':.18,'raster-brightness-max':.9}},
    {id:'bm-dark',type:'raster',source:'dark',layout:{visibility:'none'},paint:rasterPaint},
    {id:'bm-satellite',type:'raster',source:'satellite',layout:{visibility:'none'},paint:rasterPaint}
  ]}

const layerIcons={rain:CloudRain,temp:ThermometerSun,wind:Wind,uncertainty:BarChart3,weights:Layers3,extremes:AlertTriangle,change:Sparkles,humidity:Droplets,pressure:Gauge,dewpoint:ThermometerSun,cloud:Cloudy}
const layerOrder=['wind','rain','temp','uncertainty','weights','extremes','change']

// Extra layers served from the live gridded feed (GET /api/live/wind-grid)
const LIVE_LAYERS={
  humidity:{accessor:c=>c.humidity_pct},
  pressure:{accessor:c=>c.pressure_hpa},
  dewpoint:{accessor:c=>c.dew_point_c},
  cloud:{accessor:c=>c.cloud_cover_pct}
}
const EXTRA_LAYERS=['humidity','pressure','dewpoint','cloud']

const HOME={center:[78.6,21.2],zoom:4.25}
const FRAME_HOURS=[0,12,24,36,48,60,72]
const ARCHIVE_LEADS=[6,12,24,48] // lead times present in the synthetic forecast archive
const nearestLead=h=>ARCHIVE_LEADS.reduce((b,x)=>Math.abs(x-h)<Math.abs(b-h)?x:b,ARCHIVE_LEADS[0])

const UNITS={kmh:{label:'km/h',perMs:3.6},ms:{label:'m/s',perMs:1},kn:{label:'knots',perMs:1.94384}}
const CYCLES=[0,6,12,18]
const HAZARDS={rain:{label:'Heavy rain',limit:15,color:'#2f7bf0'},wind:{label:'Gale wind',limit:17.2,color:'#8f4fe0'},heat:{label:'Heatwave',limit:40,color:'#e8412f'}}

// Blend method catalogue. `weights` = blender id for GET /blending/weights (what the bars display).
const MAIN_MODELS=[
  {id:'regime_blender',label:'Regime blend',weights:'neural_gating',hint:'Regime classification + neural-gating mixture of experts'},
  {id:'inverse_variance',label:'Inverse variance',weights:'inverse_variance',hint:'w = (1/σ²) / Σ(1/σ²) from historical skill'},
  {id:'equal_weight',label:'Equal weight',weights:null,hint:'Arithmetic mean of all models'}
]
const ADVANCED_MODELS=[
  {id:'spatial_gating',label:'🧠 Spatial CNN Encoder',weights:'neural_gating',hint:'Spatial feature gating'},
  {id:'spatiotemporal_gating',label:'⏱️ Spatio-Temporal ConvGRU',weights:'neural_gating',hint:'Temporal recurrence over lead times'},
  {id:'adaptive_blending',label:'📐 Constrained Simplex Ridge',weights:'ridge_constrained',hint:'Ridge regression on the weight simplex'},
  {id:'emos_gaussian',label:'📊 Gaussian EMOS',weights:'regime_inverse_variance',hint:'Gaussian predictive calibration'},
  {id:'emos_gamma',label:'🌧️ Zero-Inflated Gamma Calibration',weights:'regime_inverse_variance',hint:'Rainfall calibration with dry-day mass'}
]
const MODEL_INDEX=Object.fromEntries([...MAIN_MODELS,...ADVANCED_MODELS].map(m=>[m.id,m]))
const isAdvanced=id=>ADVANCED_MODELS.some(m=>m.id===id)

const MODEL_NAMES={ai_graphcast_ensemble:'GraphCast v2',ecmwf_ifs:'ECMWF HRES',gfs_seamless:'GFS / Baseline',ai_model_c:'GraphCast (AI)',nwp_model_a:'NWP-A',nwp_model_b:'NWP-B'}
const MODEL_COLORS=['#4b9cff','#3cd1a0','#ff9e4a','#ac72f4']
const nameOf=k=>MODEL_NAMES[k]||k.replaceAll('_',' ')

const speedFromMs=(ms,unit)=>ms*UNITS[unit].perMs
const speedFromKmh=(kmh,unit)=>speedFromMs(kmh/3.6,unit)
const fmt=(v,d=1)=>Number(v).toFixed(d).replace(/\.0+$/,'')
const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v))
const pad2=n=>String(n).padStart(2,'0')
const layerForVariable=v=>v==='temperature'?'temp':v==='wind_speed'?'wind':'rain'

function heatColors(v){return['interpolate',['linear'],['heatmap-density'],0,'rgba(0,0,0,0)',.08,'rgba(0,0,0,0)',.22,v.colors[0],.43,v.colors[1],.63,v.colors[2],.82,v.colors[3],1,v.colors[4]]}
const heatWeight=(v,top)=>['interpolate',['linear'],['get','value'],v.stops[0],.08,v.stops.at(-1),top]
const heatRadius=layer=>LIVE_LAYERS[layer]?['interpolate',['linear'],['zoom'],3,70,8,220]:['interpolate',['linear'],['zoom'],3,23,8,46]

function gridToGeoJSON(grid,accessor){
  if(!grid?.cells)return {type:'FeatureCollection',features:[]}
  return {type:'FeatureCollection',features:grid.cells.filter(c=>typeof accessor(c)==='number').map(c=>({type:'Feature',properties:{value:accessor(c),isCenter:0},geometry:{type:'Point',coordinates:[c.lng,c.lat]}}))}
}

function WeatherMap({layer,setSelected,selected,mapRef,fieldGeoJSON,overlayOpacity,labelsVisible,basemap}){
  const node=useRef(null)
  useEffect(()=>{
    const v=variables[layer]
    const map=new maplibregl.Map({container:node.current,style:baseStyle,center:HOME.center,zoom:HOME.zoom,pitch:0,bearing:0,minZoom:3,maxZoom:11,maxPitch:70,attributionControl:false,fadeDuration:0,refreshExpiredTiles:false})
    mapRef.current=map
    map.on('load',()=>{
      map.addSource('field',{type:'geojson',data:fieldGeoJSON})
      map.addLayer({id:'field',type:'heatmap',source:'field',maxzoom:10,paint:{'heatmap-weight':heatWeight(v,.9),'heatmap-intensity':['interpolate',['linear'],['zoom'],3,.5,7,.82],'heatmap-radius':heatRadius(layer),'heatmap-opacity':overlayOpacity,'heatmap-color':heatColors(v)}})
      map.addSource('city-data',{type:'geojson',data:{type:'FeatureCollection',features:cities.map(c=>({type:'Feature',properties:{...c},geometry:{type:'Point',coordinates:c.coordinates}}))}})
      map.addLayer({id:'cities',type:'circle',source:'city-data',paint:{'circle-radius':2.5,'circle-color':'#eefcff','circle-stroke-color':'rgba(10,34,54,.8)','circle-stroke-width':1}})
      map.addLayer({id:'labels',type:'symbol',source:'city-data',layout:{'text-field':['get','name'],'text-font':['Noto Sans Regular'],'text-size':11,'text-offset':[0,1.05],'text-anchor':'top','text-allow-overlap':false},paint:{'text-color':'#fff','text-halo-color':'rgba(12,31,48,.92)','text-halo-width':1.7}})
      map.addControl(new maplibregl.AttributionControl({compact:true}),'bottom-right')
    })
    map.on('click','cities',e=>{const city=cities.find(c=>c.name===e.features?.[0]?.properties?.name);if(city)setSelected(city)})
    map.on('mouseenter','cities',()=>map.getCanvas().style.cursor='pointer')
    map.on('mouseleave','cities',()=>map.getCanvas().style.cursor='')
    return()=>map.remove()
  },[])
  useEffect(()=>{
    const map=mapRef.current
    if(!map?.getSource('field'))return
    const v=variables[layer]
    map.getSource('field').setData(fieldGeoJSON)
    map.setPaintProperty('field','heatmap-color',heatColors(v))
    map.setPaintProperty('field','heatmap-weight',heatWeight(v,.96))
    map.setPaintProperty('field','heatmap-radius',heatRadius(layer))
  },[layer,fieldGeoJSON])
  useEffect(()=>{const map=mapRef.current;if(map?.getLayer('field'))map.setPaintProperty('field','heatmap-opacity',overlayOpacity)},[overlayOpacity])
  useEffect(()=>{const map=mapRef.current;if(map?.getLayer('labels'))map.setLayoutProperty('labels','visibility',labelsVisible?'visible':'none')},[labelsVisible])
  useEffect(()=>{
    const apply=()=>Object.entries(BASEMAPS).forEach(([k,b])=>mapRef.current?.getLayer(b.layer)&&mapRef.current.setLayoutProperty(b.layer,'visibility',k===basemap?'visible':'none'))
    const map=mapRef.current
    if(map?.isStyleLoaded())apply();else map?.once('load',apply)
  },[basemap])
  useEffect(()=>{if(selected)mapRef.current?.flyTo({center:selected.coordinates,zoom:selected.mine?7:6,duration:900})},[selected])
  return <div className="weather-map" ref={node}/>
}

function HazardMarkers({mapRef,hazards,active}){
  useEffect(()=>{
    const map=mapRef.current
    if(!map||!active)return
    const markers=hazards.map(h=>{
      const el=document.createElement('div')
      el.className='hazard-pulse'
      el.style.setProperty('--hz',HAZARDS[h.type].color)
      return new maplibregl.Marker({element:el}).setLngLat([h.lng,h.lat]).setPopup(new maplibregl.Popup({offset:14}).setText(`${HAZARDS[h.type].label}: ${h.text}`)).addTo(map)
    })
    return()=>markers.forEach(m=>m.remove())
  },[hazards,active])
  return null
}

// Wind streamlines driven by the live u10/v10 grid. The procedural field below is the offline fallback.
function LiveWindParticles({enabled,grid,mapRef}){
  const canvasRef=useRef(null)
  useEffect(()=>{
    const map=mapRef.current,canvas=canvasRef.current
    if(!enabled||!map||!grid)return
    const engine=new WindParticleEngine(canvas,{getSize:()=>({w:innerWidth,h:innerHeight}),unproject:(x,y)=>{const p=map.unproject([x,y]);return[p.lat,p.lng]},sampleWind:createWindSampler(grid)})
    const restart=()=>{engine.stop();engine.resize();engine.start()}
    const pause=()=>{engine.stop();engine.clear()}
    map.on('movestart',pause);map.on('moveend',restart);addEventListener('resize',restart)
    restart()
    return()=>{map.off('movestart',pause);map.off('moveend',restart);removeEventListener('resize',restart);engine.stop();engine.clear()}
  },[enabled,grid,mapRef])
  return <canvas className={`particles ${enabled?'visible':''}`} ref={canvasRef}/>
}

function ParticleCanvas({enabled,layer}){
  const canvasRef=useRef(null)
  useEffect(()=>{const canvas=canvasRef.current,ctx=canvas.getContext('2d');let raf,particles=[];const reset=()=>{const dpr=Math.min(devicePixelRatio,2);canvas.width=innerWidth*dpr;canvas.height=innerHeight*dpr;canvas.style.width=innerWidth+'px';canvas.style.height=innerHeight+'px';ctx.setTransform(dpr,0,0,dpr,0,0);particles=Array.from({length:Math.min(520,Math.round(innerWidth*.36))},()=>({x:Math.random()*innerWidth,y:Math.random()*innerHeight,age:Math.random()*100,max:70+Math.random()*90,speed:.45+Math.random()*1.2}))};const draw=()=>{ctx.clearRect(0,0,innerWidth,innerHeight);if(enabled){ctx.lineWidth=.75;ctx.strokeStyle=layer==='wind'?'rgba(238,252,255,.62)':'rgba(235,249,255,.3)';for(const p of particles){const angle=Math.sin(p.y*.006)+Math.cos(p.x*.004)*.45-.2;const vx=Math.cos(angle)*p.speed*1.8,vy=Math.sin(angle)*p.speed;p.age++;if(p.age>p.max||p.x<0||p.x>innerWidth||p.y<0||p.y>innerHeight){p.x=Math.random()*innerWidth;p.y=Math.random()*innerHeight;p.age=0}ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(p.x-vx*6,p.y-vy*6);ctx.stroke();p.x+=vx;p.y+=vy}}raf=requestAnimationFrame(draw)};reset();draw();addEventListener('resize',reset);return()=>{cancelAnimationFrame(raf);removeEventListener('resize',reset)}},[enabled,layer]);return <canvas className={`particles ${enabled?'visible':''}`} ref={canvasRef}/>
}

function SearchBox({onPick,onLocate}){
  const [q,setQ]=useState('')
  const results=useMemo(()=>q?cities.filter(c=>(c.name+c.state).toLowerCase().includes(q.toLowerCase())).slice(0,5):[],[q])
  return <div className="windy-search"><Search/><input value={q} onChange={e=>setQ(e.target.value)} placeholder="Search location..."/><button title="Use my location" onClick={onLocate}><Crosshair/></button>{results.length>0&&<div>{results.map(c=><button key={c.name} onClick={()=>{onPick(c);setQ('')}}><b>{c.name}</b><small>{c.state}</small></button>)}</div>}</div>
}

function LayerRail({layer,setLayer,openDrawer}){
  const keys=EXTRA_LAYERS.includes(layer)?[...layerOrder,layer]:layerOrder
  return <aside className="layer-rail"><div className="rail-menu"><span>Layers</span><button onClick={openDrawer} title="All layers"><Menu/></button></div>{keys.map(key=>{const Icon=layerIcons[key],v=variables[key];return <button key={key} className={layer===key?'active':''} onClick={()=>setLayer(key)}><span>{v.short}</span><i style={{background:`conic-gradient(${v.colors.join(',')})`}}><Icon/></i></button>})}<button className="more" onClick={openDrawer}><span>More layers...</span><i><Plus/></i></button></aside>
}

function LayersDrawer({open,layer,setLayer,onClose}){
  return <aside className={`layers-drawer ${open?'open':''}`} aria-hidden={!open}><header><b>More layers</b><button onClick={onClose} aria-label="Close"><X/></button></header>{EXTRA_LAYERS.map(key=>{const Icon=layerIcons[key],v=variables[key];return <button key={key} className={layer===key?'active':''} onClick={()=>{setLayer(key);onClose()}}><i style={{background:`conic-gradient(${v.colors.join(',')})`}}><Icon/></i><span><b>{v.label}</b><small>{v.unit} · live grid</small></span>{layer===key&&<Check/>}</button>})}<p>Humidity, pressure, dew point and cloud cover come from the live operational feed. Cloud cover is a model-derived infrared proxy, not satellite imagery.</p></aside>
}

function LocationPopup({city,layer,hour,result,live,unit,bars,onClose}){
  const v=variables[layer],idx=Math.max(0,cities.findIndex(c=>c.name===city.name))
  const nearest=result?.forecast?.reduce((best,row)=>{const d=Math.hypot(Number(row.latitude)-city.coordinates[1],Number(row.longitude)-city.coordinates[0]);return !best||d<best.distance?{row,distance:d}:best},null)?.row
  let value,unitLabel=v.unit
  if(layer==='wind'){const ms=nearest?.blend_value;value=ms!=null?speedFromMs(Number(ms),unit):speedFromKmh(cityLayerValue(city,'wind',idx,0)||0,unit);unitLabel=UNITS[unit].label}
  else if(LIVE_LAYERS[layer]){value=live?.liveCell?LIVE_LAYERS[layer].accessor(live.liveCell):null}
  else{value=nearest?.blend_value??cityLayerValue(city,layer,idx,0);if(layer==='uncertainty')value=nearest?.spread??value;if(layer==='extremes')value=Math.round(Math.min(100,Number(nearest?.blend_value||0)*2+Number(nearest?.spread||0)*8));if(layer==='change')value=0}
  value=typeof value==='number'?Math.round(value*10)/10:'—'
  const liveWeights=Object.entries(result?.weights||{})
  const weights=bars?.length?bars.map((b,i)=>({name:b.name,value:b.pct,color:MODEL_COLORS[i%4]})):liveWeights.length?liveWeights.map(([name,weight],i)=>({name:nameOf(name),value:Number(weight)*100,color:MODEL_COLORS[i%4]})):modelWeights
  const spread=Math.round(Number(nearest?.spread||result?.uncertainty?.mean_spread||0)*10)/10
  const confidence=clamp(Math.round(94-spread*8),45,96)
  const fc=live?.forecast
  const step=fc?.forecast_steps?.reduce((b,s)=>!b||Math.abs(s.lead_time_hours-hour)<Math.abs(b.lead_time_hours-hour)?s:b,null)
  const cur=fc?.current_conditions
  const useCur=hour===0||!step
  const alerts=fc?.active_alerts||[]
  return <section className="location-popup"><button className="popup-close" onClick={onClose}><X/></button><small>{result?'LIVE ML BLEND':'LOCAL PREVIEW'}</small><h2>{city.name}</h2><p>{city.state} · +{hour}h</p><div className="popup-value"><b>{value}</b><span>{unitLabel}</span></div><div className="popup-stats"><span><small>Confidence</small><b>{confidence}%</b></span><span><small>Disagreement</small><b>±{spread}</b></span><span><small>Regime</small><b>{nearest?.regime||result?.regime?.dominant_regime||city.risk}</b></span></div><div className="popup-weights">{weights.map(m=><i key={m.name} style={{width:`${m.value}%`,background:m.color}} title={`${m.name} ${Math.round(m.value)}%`}/>)}</div><p className="popup-note">{weights.map(m=>`${m.name} ${Math.round(m.value)}%`).join(' · ')}</p>{live&&<div className="popup-live"><small>LIVE FORECAST · /api/live/forecast</small>{live.loading?<p>Loading…</p>:live.error?<p>Live feed unavailable</p>:(step||cur)&&<p>{useCur?`${cur?.temperature_c}°C`:`${step.temperature_c}°C [${step.temperature_ci90?.join(' – ')}]`} · {fmt(speedFromMs(useCur?cur?.wind_speed_ms:step.wind_speed_ms,unit))} {UNITS[unit].label} · {useCur?cur?.precipitation_mm:step.rainfall_mm} mm</p>}{alerts.length>0&&<p className="popup-alert">⚠ {alerts.map(a=>a.headline||a.hazard_type).join(' · ')}</p>}</div>}</section>
}

function BottomTimeline({frame,setFrame,playing,setPlaying,cycleBase}){
  const days=FRAME_HOURS.map(h=>{const d=new Date(cycleBase+h*3600000);return{h,day:d.toLocaleDateString('en-GB',{weekday:'short',day:'numeric',timeZone:'UTC'}),hh:pad2(d.getUTCHours())}})
  const cur=days[frame]
  return <div className="bottom-timeline"><button className="big-play" onClick={()=>setPlaying(!playing)} title={playing?'Pause':'Play'}>{playing?<Pause/>:<Play/>}</button><span aria-hidden="true"/><div className="hour-tag">{cur.hh}:00 UTC<i/></div><div className="days" style={{gridTemplateColumns:`repeat(${days.length},1fr)`}}>{days.map((d,i)=><button key={d.h} className={i===frame?'active':''} onClick={()=>setFrame(i)} title={`t+${d.h}h`}><b>{d.day}</b><small>{d.hh}h · +{d.h}</small></button>)}</div></div>
}

function ControlDock({layer,particles,setParticles,mapAction,overlayOpacity,setOverlayOpacity,unit,setUnit,is3D,labelsVisible,setLabelsVisible,locate}){
  return <div className="control-dock"><div><button onClick={()=>mapAction('in')} title="Zoom in"><Plus/></button><button onClick={()=>mapAction('out')} title="Zoom out"><Minus/></button><button className={is3D?'on':''} onClick={()=>mapAction('3d')} title="Toggle 3D tilt">3D</button><button className={labelsVisible?'':'dim'} onClick={()=>setLabelsVisible(!labelsVisible)} title="Toggle city labels"><Settings2/></button><button onClick={locate} title="Locate me"><LocateFixed/></button></div><section><label><input type="checkbox" checked={particles} onChange={e=>setParticles(e.target.checked)}/><i/> particles{layer!=='wind'&&<small style={{opacity:.6}}> (wind layer)</small>}</label><label className="opacity-control">Overlay <input type="range" min="0" max="0.75" step="0.05" value={overlayOpacity} onChange={e=>setOverlayOpacity(Number(e.target.value))}/></label><span className="unit-pill" role="group" aria-label="Wind speed unit">{Object.entries(UNITS).map(([k,u])=><button key={k} className={unit===k?'active':''} onClick={()=>setUnit(k)}>{u.label}</button>)}</span>{layer!=='wind'&&<span>{variables[layer].unit}</span>}</section></div>
}

function LegendBar({layer,unit,hazardCount}){
  const v=variables[layer]
  const isWind=layer==='wind'
  const ticks=isWind?v.stops.map(s=>Math.round(speedFromKmh(s,unit)*10)/10):v.stops
  const unitLabel=isWind?UNITS[unit].label:v.unit
  return <div className="legend-bar"><div style={{background:`linear-gradient(90deg,${v.colors.join(',')})`}}>{ticks.map(s=><span key={s}>{s}</span>)}</div><small>{layer==='extremes'&&hazardCount===0?`${unitLabel} · no live breaches`:unitLabel}</small></div>
}

function WeightsPanel({weights,model,note}){
  if(!weights?.length)return null
  return <div className="weights-panel"><b>Blend weights · {MODEL_INDEX[model]?.label.replace(/^\p{Extended_Pictographic}\S*\s/u,'')||model}</b>{weights.map((w,i)=><div key={w.name}><span>{w.name}</span><i><u style={{width:`${w.pct}%`,background:MODEL_COLORS[i%4]}}/></i><em>{Math.round(w.pct)}%</em></div>)}<small>{note}</small></div>
}

function AdvancedBlenderModal({open,model,onSelect,onClose}){
  useEffect(()=>{
    if(!open)return
    const onKey=e=>{if(e.key==='Escape')onClose()}
    addEventListener('keydown',onKey)
    return()=>removeEventListener('keydown',onKey)
  },[open])
  if(!open)return null
  return <div className="blender-modal-backdrop" onClick={onClose}><div className="blender-modal-card" role="dialog" aria-modal="true" aria-label="Advanced blenders" onClick={e=>e.stopPropagation()}>
    <div className="blender-modal-header"><b>Advanced blenders</b><button className="drawer-close" onClick={onClose} aria-label="Close"><X size={20}/></button></div>
    <div className="blender-modal-body">{ADVANCED_MODELS.map(m=><button key={m.id} className={`blender-option ${model===m.id?'active':''}`} onClick={()=>onSelect(m.id)}><span><b>{m.label}</b><small>{m.hint}</small></span>{model===m.id&&<Check size={16}/>}</button>)}
      <p className="blender-note">Methods without a dedicated pipeline implementation run as the regime blender; the status pill reports the effective method.</p></div>
  </div></div>
}

function OperationalMenuDrawer({isOpen,onClose,basemap,setBasemap,health,gridCount,onExportJSON,labelsVisible,setLabelsVisible,hazardsVisible,setHazardsVisible}){
  if(!isOpen)return null
  const docsUrl=`${apiClient.baseUrl}/docs`
  return <div className="menu-drawer-backdrop" onClick={onClose}><div className="menu-drawer-panel" role="dialog" aria-label="BharatWeather operations and settings" onClick={e=>e.stopPropagation()}>
    <div className="drawer-header"><h3>BharatWeather Operations & Settings</h3><button className="drawer-close" onClick={onClose} aria-label="Close"><X size={20}/></button></div>
    <div className="drawer-body">
      <div className="drawer-section"><h4>🗺️ Map Basemap Style</h4><div className="drawer-grid">{Object.entries(BASEMAPS).map(([k,b])=><button key={k} className={`drawer-choice ${basemap===k?'active':''}`} onClick={()=>setBasemap(k)}>{b.label}</button>)}</div></div>
      <div className="drawer-section"><h4>⚡ System Health & Connection</h4><div className="status-box">
        {health.error?<p>Backend unreachable — start <b>python run_server.py</b>.</p>:<>
          <p><strong>Provider:</strong> {health.status?.provider||'…'}</p>
          <p><strong>Status:</strong> {health.status?`${health.status.status} · ${health.status.freshness}${Number.isFinite(health.status.data_age_minutes)&&health.status.freshness!=='unavailable'?` (${Math.round(health.status.data_age_minutes)} min old)`:''}`:'…'}</p>
          <p><strong>Provider latency:</strong> {health.status?.latency_ms!=null?`${fmt(health.status.latency_ms,0)} ms`:'…'}</p>
          <p><strong>Active grid:</strong> {gridCount!=null?`${gridCount.toLocaleString()} forecast nodes`:'—'}</p>
          <p><strong>Engine version:</strong> {health.meta?.version?`v${health.meta.version}`:'…'}</p></>}
      </div></div>
      <div className="drawer-section"><h4>📥 Data Export & Research Logs</h4>
        <button className="drawer-action-btn" onClick={onExportJSON}><Download size={16}/> Export Current Forecast Payload (JSON)</button>
        <button className="drawer-action-btn" onClick={()=>window.open(docsUrl,'_blank','noopener')}><ExternalLink size={16}/> Open FastAPI Swagger Documentation</button>
      </div>
      <div className="drawer-section"><h4>⚙️ Display Preferences</h4>
        <label className="drawer-toggle"><span>Show City Labels</span><input type="checkbox" checked={labelsVisible} onChange={e=>setLabelsVisible(e.target.checked)}/></label>
        <label className="drawer-toggle"><span>Show High-Risk Hazard Radar</span><input type="checkbox" checked={hazardsVisible} onChange={e=>setHazardsVisible(e.target.checked)}/></label>
      </div>
    </div>
  </div></div>
}

export default function WeatherMapWorkspace({onNavigateHome}){
  const [layer,setLayer]=useState('wind'),[frame,setFrame]=useState(2),[playing,setPlaying]=useState(false),[selected,setSelected]=useState(null),[particles,setParticles]=useState(true),[overlayOpacity,setOverlayOpacity]=useState(.42)
  const [model,setModel]=useState('regime_blender'),[unit,setUnit]=useState('kmh'),[is3D,setIs3D]=useState(false),[labelsVisible,setLabelsVisible]=useState(true)
  const [moreOpen,setMoreOpen]=useState(false),[drawerOpen,setDrawerOpen]=useState(false),[cycleOpen,setCycleOpen]=useState(false),[toast,setToast]=useState(null)
  const [cycle,setCycle]=useState(()=>Math.floor(new Date().getUTCHours()/6)*6)
  const [windGrid,setWindGrid]=useState(null),[forecastResult,setForecastResult]=useState(null),[prevResult,setPrevResult]=useState(null)
  const [weightBars,setWeightBars]=useState(null),[liveSel,setLiveSel]=useState(null)
  const [isMenuOpen,setIsMenuOpen]=useState(false),[basemap,setBasemap]=useState('light'),[hazardsVisible,setHazardsVisible]=useState(true),[health,setHealth]=useState({})
  const [apiState,setApiState]=useState({status:'connecting',message:'Connecting to forecast engine'})
  const mapRef=useRef(null),forecastCache=useRef(new Map()),gridCache=useRef(new Map())

  const hour=FRAME_HOURS[frame]
  const effLead=nearestLead(Math.max(hour,1))
  const cycleBase=useMemo(()=>{const d=new Date();d.setUTCHours(cycle,0,0,0);return d.getTime()},[cycle])
  const gridHour=clamp(Math.round((cycleBase+hour*3600000-Date.now())/3600000),0,72)
  const variable=variableForLayer(layer)
  const isLiveLayer=!!LIVE_LAYERS[layer]
  const isChange=layer==='change'
  const notify=useCallback(msg=>{setToast(msg);setTimeout(()=>setToast(null),4200)},[])

  // timeline playback: 1 s per step through t+0 … t+72
  useEffect(()=>{if(!playing)return;const id=setInterval(()=>setFrame(f=>(f+1)%FRAME_HOURS.length),1000);return()=>clearInterval(id)},[playing])

  // live gridded field (wind u/v + scalar layers) for the displayed hour
  useEffect(()=>{
    let active=true
    const cached=gridCache.current.get(gridHour)
    if(cached){setWindGrid(cached);return}
    const t=setTimeout(()=>getLiveWindGrid({hour:gridHour}).then(g=>{gridCache.current.set(gridHour,g);if(active)setWindGrid(g)}).catch(()=>{if(active)setWindGrid(null)}),200)
    return()=>{active=false;clearTimeout(t)}
  },[gridHour])
  useEffect(()=>{const id=setInterval(()=>gridCache.current.clear(),300000);return()=>clearInterval(id)},[])

  // pipeline forecast (cached per variable/lead/method)
  const fetchRun=useCallback(async(v,lead,m)=>{
    const key=`${v}|${lead}|${m}`
    if(forecastCache.current.has(key))return forecastCache.current.get(key)
    const res=await runForecast(layerForVariable(v),lead,m)
    forecastCache.current.set(key,res)
    return res
  },[])
  useEffect(()=>{
    if(isLiveLayer){setApiState({status:windGrid?'online':'loading',message:windGrid?`${windGrid.cells.length} live grid cells · ${windGrid.provider}`:'Loading live grid…'});return}
    let active=true
    setApiState({status:'loading',message:'Running ML blend…'})
    const i=ARCHIVE_LEADS.indexOf(effLead)
    const prevLead=isChange?ARCHIVE_LEADS[i>0?i-1:1]:null
    Promise.all([getHealth(),fetchRun(variable,effLead,model),prevLead?fetchRun(variable,prevLead,model):Promise.resolve(null)]).then(([,result,prev])=>{
      if(!active)return
      setForecastResult(result);setPrevResult(prev)
      const eff=result.diagnostics?.blender_method_effective
      setApiState({status:'online',message:`${result.forecast_count} grid forecasts · ${Math.round(result.diagnostics?.execution_time_ms||0)} ms · +${result.target_lead_time}h${eff&&eff!==model?` · ran as ${eff}`:''}`})
    }).catch(error=>{if(!active)return;setForecastResult(null);setPrevResult(null);setApiState({status:'error',message:'Backend unavailable · showing local preview'});console.error(error)})
    return()=>{active=false}
  },[variable,isChange,effLead,model,isLiveLayer])

  // model weight bars from the backend (GET /blending/weights)
  useEffect(()=>{
    let active=true
    const meta=MODEL_INDEX[model]
    if(!meta?.weights){
      const names=Object.keys(forecastResult?.weights||{ai_graphcast_ensemble:1,ecmwf_ifs:1,gfs_seamless:1})
      setWeightBars({bars:names.map(n=>({name:nameOf(n),pct:100/names.length})),note:'Arithmetic mean · no fitted weights'});return
    }
    getBlendingWeights({variable,leadTime:effLead,blender:meta.weights}).then(r=>{
      if(!active)return
      const entries=Object.entries(r.weights||{})
      const total=entries.reduce((a,[,w])=>a+Number(w),0)||1
      setWeightBars({bars:entries.map(([n,w])=>({name:nameOf(n),pct:Number(w)/total*100})),note:`GET /blending/weights · ${r.blender} · ${variable} +${effLead}h`})
    }).catch(()=>{if(active)setWeightBars(null)})
    return()=>{active=false}
  },[model,variable,effLead])

  // system health + engine metadata for the operations drawer
  useEffect(()=>{
    if(!isMenuOpen)return
    let active=true
    Promise.all([getLiveStatus(),getLiveMetadata()]).then(([status,meta])=>{if(active)setHealth({status,meta})}).catch(()=>{if(active)setHealth({error:true})})
    return()=>{active=false}
  },[isMenuOpen])

  // live point forecast for the selected location (GET /api/live/forecast)
  useEffect(()=>{
    if(!selected){setLiveSel(null);return}
    let active=true
    const [lng,lat]=selected.coordinates
    setLiveSel({loading:true})
    getLiveForecast({latitude:lat,longitude:lng}).then(forecast=>{
      if(active)setLiveSel({forecast,error:forecast.status!=='success'})
    }).catch(()=>{if(active)setLiveSel({error:true})})
    return()=>{active=false}
  },[selected])
  const liveCell=useMemo(()=>{
    if(!selected||!windGrid)return null
    const [lng,lat]=selected.coordinates
    return windGrid.cells.reduce((b,c)=>{const d=Math.hypot(c.lat-lat,c.lng-lng);return !b||d<b.d?{...c,d}:b},null)
  },[selected,windGrid])

  const fieldGeoJSON=useMemo(()=>{
    if(isLiveLayer)return gridToGeoJSON(windGrid,LIVE_LAYERS[layer].accessor)
    if(isChange){const g=changeGeoJSON(forecastResult,prevResult);if(g)return g}
    if(forecastResult){
      const g=forecastToGeoJSON(forecastResult,layer)
      if(layer==='wind')g.features.forEach(f=>{f.properties.value=f.properties.value*3.6}) // m/s → km/h (stop scale)
      return g
    }
    return weatherGeoJSON(layer,frame)
  },[isLiveLayer,windGrid,forecastResult,prevResult,layer,frame])

  const hazards=useMemo(()=>{
    if(!windGrid)return []
    const out=[]
    for(const c of windGrid.cells){
      if((c.precipitation_mm??0)>=HAZARDS.rain.limit)out.push({type:'rain',lat:c.lat,lng:c.lng,text:`${c.precipitation_mm} mm`})
      if((c.wind_speed_ms??0)>=HAZARDS.wind.limit)out.push({type:'wind',lat:c.lat,lng:c.lng,text:`${c.wind_speed_ms} m/s`})
      if((c.temperature_c??0)>=HAZARDS.heat.limit)out.push({type:'heat',lat:c.lat,lng:c.lng,text:`${c.temperature_c} °C`})
    }
    return out
  },[windGrid])
  useEffect(()=>{if(layer==='extremes'&&windGrid&&hazards.length===0)notify('No locations breach warning thresholds in the live grid at this hour')},[layer,windGrid])

  const locate=()=>{
    if(!navigator.geolocation){notify('Geolocation is not available in this browser');return}
    notify('Locating…')
    navigator.geolocation.getCurrentPosition(pos=>{
      const {latitude:lat,longitude:lng}=pos.coords
      setSelected({name:'My location',state:`${lat.toFixed(2)}°N, ${lng.toFixed(2)}°E`,coordinates:[lng,lat],rain:0,temp:0,wind:0,risk:'—',mine:true})
    },err=>notify(`Could not get location: ${err.message}`),{enableHighAccuracy:false,timeout:10000})
  }

  const mapAction=a=>{
    const m=mapRef.current
    if(!m)return
    if(a==='in')m.zoomIn()
    if(a==='out')m.zoomOut()
    if(a==='home')m.flyTo({...HOME,pitch:0,bearing:0})
    if(a==='3d'){const was3D=m.getPitch()>10;m.easeTo({pitch:was3D?0:55,bearing:was3D?0:-15,duration:1000});setIs3D(!was3D)}
  }
  const setLead=h=>setFrame(Math.max(0,FRAME_HOURS.indexOf(h)))
  const exportJSON=()=>{
    if(!forecastResult&&!windGrid){notify('No forecast loaded yet');return}
    const payload={exported_at_utc:new Date().toISOString(),layer,blender_requested:model,lead_hours:hour,cycle_utc:cycle,blend_weights:weightBars,pipeline_forecast:forecastResult,live_grid:windGrid,selected_location:selected?{name:selected.name,coordinates:selected.coordinates,live_forecast:liveSel?.forecast||null}:null}
    const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}))
    const a=document.createElement('a')
    a.href=url;a.download=`bharatweather-forecast-${layer}-${model}-t${hour}h.json`;a.click()
    setTimeout(()=>URL.revokeObjectURL(url),1000)
    notify('Forecast payload exported')
  }
  const share=()=>navigator.clipboard?.writeText(location.href).then(()=>notify('Link copied to clipboard'),()=>notify('Could not copy link'))
  const cycleLabel=`${new Date(cycleBase).toLocaleDateString('en-GB',{day:'2-digit',month:'short',timeZone:'UTC'})} · ${pad2(cycle)} UTC`

  return <main className="windy-app"><WeatherMap {...{layer,selected,setSelected,mapRef,fieldGeoJSON,overlayOpacity,labelsVisible,basemap}}/>{windGrid?<LiveWindParticles enabled={particles&&layer==='wind'} grid={windGrid} mapRef={mapRef}/>:<ParticleCanvas enabled={particles&&layer==='wind'} layer={layer}/>}<HazardMarkers mapRef={mapRef} hazards={hazards} active={layer==='extremes'&&hazardsVisible}/><SearchBox onPick={setSelected} onLocate={locate}/><button className="share" onClick={share} title="Copy link"><Upload/></button><div className="center-brand portal-brand-clickable" role="link" tabIndex={0} title="Return to BharatWeather Landing Page" onClick={()=>onNavigateHome?onNavigateHome():(window.location.hash='#home')} onKeyDown={e=>{if(e.key==='Enter')(onNavigateHome||(()=>{window.location.hash='#home'}))()}}><span><Wind/></span><b>BharatWeather</b><small>weather intelligence</small></div>
    <div className="top-actions"><div className="cycle-wrap"><button className="run-select" onClick={()=>setCycleOpen(!cycleOpen)}>{cycleLabel} <ChevronDown/></button>{cycleOpen&&<div className="cycle-menu">{CYCLES.map(c=><button key={c} className={c===cycle?'active':''} onClick={()=>{setCycle(c);setCycleOpen(false)}}>{pad2(c)} UTC{c===cycle&&<Check/>}</button>)}<small>The cycle shifts valid times and the live-grid hour. The synthetic forecast archive has no per-cycle runs.</small></div>}</div><button className={`api-status ${apiState.status}`} title="Forecast engine telemetry"><i/>{apiState.message}{hour>48&&!isLiveLayer?' · archive max +48h':''}</button><button className="login" onClick={()=>setIsMenuOpen(true)}><Menu size={18}/> Menu</button></div>
    <LayerRail {...{layer,setLayer}} openDrawer={()=>setDrawerOpen(true)}/><LayersDrawer open={drawerOpen} layer={layer} setLayer={setLayer} onClose={()=>setDrawerOpen(false)}/>
    {selected&&<LocationPopup city={selected} layer={layer} hour={hour} result={forecastResult} live={liveSel&&{...liveSel,liveCell}} unit={unit} bars={weightBars?.bars} onClose={()=>setSelected(null)}/>}
    <div className="lead-picker"><span>Lead time</span>{[24,48,72].map(x=><button key={x} className={hour===x?'active':''} onClick={()=>setLead(x)}>+{x}h</button>)}</div>
    <WeightsPanel weights={weightBars?.bars} model={model} note={weightBars?.note}/>
    <LegendBar layer={layer} unit={unit} hazardCount={hazards.length}/>
    <div className="model-picker">{MAIN_MODELS.map(m=><button key={m.id} className={model===m.id?'active':''} title={m.hint} onClick={()=>setModel(m.id)}>{m.label}</button>)}<button className={isAdvanced(model)?'active':''} onClick={()=>setMoreOpen(!moreOpen)}>{isAdvanced(model)?MODEL_INDEX[model].label.replace(/^\p{Extended_Pictographic}\S*\s/u,''):'More…'}</button>
</div>
    <ControlDock {...{layer,particles,setParticles,mapAction,overlayOpacity,setOverlayOpacity,unit,setUnit,is3D,labelsVisible,setLabelsVisible,locate}}/>
    <button className="zoom-fab in" onClick={()=>mapAction('in')}><Plus/></button><button className="zoom-fab out" onClick={()=>mapAction('out')}><Minus/></button>
    <BottomTimeline {...{frame,setFrame,playing,setPlaying,cycleBase}}/><AdvancedBlenderModal open={moreOpen} model={model} onSelect={id=>{setModel(id);setMoreOpen(false)}} onClose={()=>setMoreOpen(false)}/><OperationalMenuDrawer isOpen={isMenuOpen} onClose={()=>setIsMenuOpen(false)} basemap={basemap} setBasemap={setBasemap} health={health} gridCount={forecastResult?.forecast_count??windGrid?.cells?.length??null} onExportJSON={exportJSON} labelsVisible={labelsVisible} setLabelsVisible={setLabelsVisible} hazardsVisible={hazardsVisible} setHazardsVisible={setHazardsVisible}/>{toast&&<div className="map-toast" role="status">{toast}</div>}</main>
}
