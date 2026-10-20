import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, Brush, Check, Circle, Eraser, Eye, Hand, Minus, Redo2, RotateCcw, Save, Undo2 } from 'lucide-react';

const pointFromPointer = (event, frame, sourceWidth, sourceHeight, rotation) => {
  const rect = frame.getBoundingClientRect();
  const displayWidth = frame.offsetWidth;
  const displayHeight = frame.offsetHeight;
  const radians = rotation * Math.PI / 180;
  const bw = Math.abs(displayWidth * Math.cos(radians)) + Math.abs(displayHeight * Math.sin(radians));
  const bh = Math.abs(displayWidth * Math.sin(radians)) + Math.abs(displayHeight * Math.cos(radians));
  const dx = event.clientX - (rect.left + bw / 2);
  const dy = event.clientY - (rect.top + bh / 2);
  const x = dx * Math.cos(radians) + dy * Math.sin(radians) + displayWidth / 2;
  const y = -dx * Math.sin(radians) + dy * Math.cos(radians) + displayHeight / 2;
  return [Math.max(0, Math.min(sourceWidth - 1, Math.round(x * sourceWidth / displayWidth))), Math.max(0, Math.min(sourceHeight - 1, Math.round(y * sourceHeight / displayHeight)))];
};
const asPath = (points) => points.map(([x, y]) => `${x},${y}`).join(' ');

export default function ManualEditor({ apiUrl, jobId, sourceUrl, candidateUrl, width, height, onClose, onSaved }) {
  const [tool, setTool] = useState('brush');
  const [brushSize, setBrushSize] = useState(24);
  const [color, setColor] = useState('#ffffff');
  const [zoom, setZoom] = useState(100);
  const [rotation, setRotation] = useState(0);
  const [comparison, setComparison] = useState('candidate');
  const [operations, setOperations] = useState([]);
  const [savedOperations, setSavedOperations] = useState([]);
  const [redoStack, setRedoStack] = useState([]);
  const [selectedPolygon, setSelectedPolygon] = useState(null);
  const [activePoints, setActivePoints] = useState(null);
  const [pointerPosition, setPointerPosition] = useState(null);
  const [repairSample, setRepairSample] = useState(null);
  const [sampling, setSampling] = useState(false);
  const [version, setVersion] = useState(0);
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState('');
  const [repairPreviewUrl, setRepairPreviewUrl] = useState('');
  const [previewHash, setPreviewHash] = useState('');
  const frameRef = useRef(null);
  const isDirty = JSON.stringify(operations) !== JSON.stringify(savedOperations);
  const pendingHash = JSON.stringify(operations.slice(savedOperations.length));
  const repairNeedsPreview = operations.slice(savedOperations.length).some((op) => op.tool === 'repair') && previewHash !== pendingHash;
  const candidateSrc = `${candidateUrl}${candidateUrl?.includes('?') ? '&' : '?'}manual=${version}`;
  const previewStroke = activePoints || (operations.length > savedOperations.length ? operations.at(-1)?.points : null);

  useEffect(() => {
    let alive = true;
    fetch(apiUrl(`/api/jobs/${jobId}/manual-edits`)).then((r) => r.ok ? r.json() : Promise.reject(new Error('Could not load edit history.'))).then((data) => {
      if (!alive) return;
      setVersion(data.candidate_version || 0);
      setOperations(data.operations || []);
      setSavedOperations(data.operations || []);
    }).catch((e) => setNotice(e.message));
    return () => { alive = false; };
  }, [apiUrl, jobId]);

  const mapPoint = (event) => pointFromPointer(event, frameRef.current, width, height, rotation);
  const pointerDown = (event) => {
    if (tool === 'pan') return;
    if (sampling) { setRepairSample(mapPoint(event)); setSampling(false); setTool('repair'); setNotice('Repair source sampled. Draw a repair stroke, review it, then save.'); return; }
    event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId);
    const p = mapPoint(event);
    if (tool === 'select-rect' || tool === 'select-freehand') setActivePoints([p]);
    else setActivePoints([p]);
  };
  const pointerMove = (event) => {
    const p = mapPoint(event); setPointerPosition(p);
    if (!activePoints) return;
    setActivePoints((points) => tool === 'line' || tool === 'select-rect' ? [points[0], p] : [...points, p]);
  };
  const pointerUp = () => {
    if (!activePoints?.length) return;
    if (tool === 'select-rect') {
      const [a, b] = activePoints; setSelectedPolygon([[a[0],a[1]],[b[0],a[1]],[b[0],b[1]],[a[0],b[1]]]);
    } else if (tool === 'select-freehand') setSelectedPolygon(activePoints);
    else {
      const op = { tool, points: activePoints, width: brushSize, color, ...(tool === 'repair' ? { sample: repairSample } : {}) };
      setOperations((current) => [...current, op]); setRedoStack([]);
    }
    setActivePoints(null);
  };
  const addSelectionEdit = (erase) => {
    if (!selectedPolygon) return;
    setOperations((current) => [...current, { tool: 'selection-clear', points: selectedPolygon, color: erase ? '#ffffff' : color, width: brushSize }]);
    setSelectedPolygon(null); setRedoStack([]);
  };
  const undo = () => { if (!operations.length) return; setRedoStack((r) => [...r, operations.at(-1)]); setOperations((o) => o.slice(0, -1)); };
  const redo = () => { if (!redoStack.length) return; setOperations((o) => [...o, redoStack.at(-1)]); setRedoStack((r) => r.slice(0, -1)); };
  const resetUnsaved = () => { setOperations(savedOperations); setSelectedPolygon(null); setRedoStack([]); };
  const discardAll = async () => {
    if (!operations.length) return;
    if (!window.confirm('Discard every manual edit and return to the automatic candidate?')) return;
    setSaving(true);
    try {
      const response=await fetch(apiUrl(`/api/jobs/${jobId}/manual-edits`),{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({candidate_version:version,operations:[]})});
      const data=await response.json(); if(!response.ok) throw new Error(data.detail||'Could not discard edits.');
      setVersion(data.candidate_version);setOperations([]);setSavedOperations([]);setRedoStack([]);setSelectedPolygon(null);setNotice('All manual edits discarded. Approval cleared.');onSaved?.(data);
    } catch(e) { setNotice(e.message); }
    finally { setSaving(false); }
  };
  const save = async () => {
    if (!isDirty) return;
    if (repairNeedsPreview) { setNotice('Preview the repair result and confirm it before saving.'); return; }
    if (operations.some((op) => op.tool === 'repair' && !op.sample)) { setNotice('Choose a repair source before saving repair strokes.'); return; }
    setSaving(true); setNotice('Applying edits to a full-resolution candidate…');
    try {
      const response = await fetch(apiUrl(`/api/jobs/${jobId}/manual-edits`), { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ candidate_version: version, operations }) });
      const data = await response.json(); if (!response.ok) throw new Error(data.detail || 'Could not save edits.');
      setVersion(data.candidate_version); setSavedOperations(operations); setNotice(`Saved candidate version ${data.candidate_version}. Approval was cleared.`); onSaved?.(data);
    } catch (e) { setNotice(e.message); }
    finally { setSaving(false); }
  };
  const makeRepairPreview = async () => {
    try {
      const img=new Image(); img.crossOrigin='anonymous'; img.src=candidateSrc;
      if (!img.complete) await new Promise((resolve,reject)=>{img.onload=resolve;img.onerror=()=>reject(new Error('Candidate image could not be loaded for preview.'));});
      const frame=frameRef.current; const canvas=document.createElement('canvas'); canvas.width=Math.max(1,frame.offsetWidth); canvas.height=Math.max(1,frame.offsetHeight);
      const ctx=canvas.getContext('2d'); ctx.drawImage(img,0,0,canvas.width,canvas.height);
      const sx=canvas.width/width, sy=canvas.height/height;
      for (const op of operations.slice(savedOperations.length)) {
        const pts=op.points||[]; if (!pts.length) continue;
        if (op.tool==='repair'&&op.sample) { const r=(op.width||1)/2,[sampleX,sampleY]=op.sample; for(const [x,y] of pts)ctx.drawImage(img,sampleX-r,sampleY-r,2*r,2*r,(x-r)*sx,(y-r)*sy,2*r*sx,2*r*sy); continue; }
        ctx.beginPath(); ctx.lineWidth=(op.width||1)*sx; ctx.lineCap='round'; ctx.lineJoin='round'; ctx.strokeStyle=op.tool==='erase'?'#ffffff':(op.color||'#ffffff'); ctx.fillStyle=ctx.strokeStyle;
        if(op.tool==='selection-clear'){pts.forEach(([x,y],i)=>i?ctx.lineTo(x*sx,y*sy):ctx.moveTo(x*sx,y*sy));ctx.closePath();ctx.fill();continue;}
        if(pts.length===1){ctx.arc(pts[0][0]*sx,pts[0][1]*sy,Math.max(1,ctx.lineWidth/2),0,Math.PI*2);ctx.fill();}
        else {pts.forEach(([x,y],i)=>i?ctx.lineTo(x*sx,y*sy):ctx.moveTo(x*sx,y*sy));ctx.stroke();}
      }
      setRepairPreviewUrl(canvas.toDataURL('image/png')); setPreviewHash(pendingHash); setNotice('Review this downscaled repair preview before saving the full-resolution edit.');
    } catch(e) { setNotice(e.message); }
  };
  const actions = [
    ['brush','Brush',Brush], ['background','Background',Brush], ['erase','Eraser',Eraser], ['repair','Repair',Circle], ['line','Boundary line',Minus], ['select-rect','Rectangle select',Circle], ['select-freehand','Freehand select',Circle], ['pan','Pan',Hand],
  ];
  const operationPaths = useMemo(() => operations.slice(savedOperations.length), [operations, savedOperations.length]);
  const renderPaths = [...operationPaths, ...(activePoints ? [{ tool, points: activePoints, width: brushSize, color }] : [])];
  return <section className="manual-editor panel" aria-label="Manual candidate editor">
    <div className="manual-editor-head"><div><h2>Edit candidate</h2><p>Edits use source-image coordinates. Original upload remains immutable.</p></div><button className="secondary-button" onClick={onClose}><ArrowLeft size={15}/> Back to review</button></div>
    <div className="manual-toolbar" role="toolbar" aria-label="Manual editing tools">
      {actions.map(([id,label,Icon]) => <button key={id} className={`secondary-button ${tool === id ? 'active' : ''}`} onClick={() => { setTool(id); setSampling(false); }} title={label}><Icon size={15}/>{label}</button>)}
      <button className="secondary-button" onClick={undo} disabled={!operations.length}><Undo2 size={15}/>Undo</button>
      <button className="secondary-button" onClick={redo} disabled={!redoStack.length}><Redo2 size={15}/>Redo</button>
    </div>
    <div className="manual-controls">
      <label>Brush size <input type="range" min="2" max="600" value={brushSize} onChange={(e)=>setBrushSize(Number(e.target.value))}/> {brushSize}px</label>
      <label>Paint <input type="color" value={color} onChange={(e)=>setColor(e.target.value)}/></label>
      {tool === 'repair' && <button className="secondary-button" onClick={()=>setSampling(true)}>Sample nearby pixels</button>}
      <label>Zoom <input type="range" min="50" max="300" value={zoom} onChange={(e)=>setZoom(Number(e.target.value))}/> {zoom}%</label>
      <label>Rotate <input type="range" min="-180" max="180" value={rotation} onChange={(e)=>setRotation(Number(e.target.value))}/> {rotation}°</label>
      <select aria-label="Original comparison" value={comparison} onChange={(e)=>setComparison(e.target.value)}><option value="candidate">Candidate</option><option value="original">Original</option><option value="difference">Difference overlay</option></select>
    </div>
    <div className="manual-canvas-scroll" style={{maxHeight:'68vh',overflow:'auto'}}>
      <div ref={frameRef} className="manual-edit-frame" style={{width:`${zoom}%`,position:'relative',margin:'12px auto',transform:`rotate(${rotation}deg)`,transformOrigin:'center center',lineHeight:0}}>
        <img src={comparison==='original'?sourceUrl:candidateSrc} alt={comparison==='original'?'Original image':'Current candidate'} style={{display:'block',width:'100%',height:'auto',userSelect:'none',pointerEvents:'none'}}/>
        {comparison==='difference' && <img src={sourceUrl} alt="Original difference reference" style={{position:'absolute',inset:0,width:'100%',height:'100%',objectFit:'fill',opacity:.72,mixBlendMode:'difference',pointerEvents:'none'}}/>}
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{position:'absolute',inset:0,width:'100%',height:'100%',touchAction:'none',cursor:tool==='pan'?'grab':'crosshair',pointerEvents:comparison==='original'?'none':'auto'}} onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerUp} onPointerCancel={pointerUp}>
          {renderPaths.map((op,i)=><polyline key={i} points={asPath(op.points||[])} fill={op.tool==='selection-clear'?'rgba(255,120,45,.2)':'none'} stroke={op.tool==='erase'?'#ff6655':op.tool==='repair'?'#29d8ff':'#ad45ec'} strokeWidth={op.width||brushSize} strokeLinecap="round" strokeLinejoin="round" opacity=".8"/>)}
          {selectedPolygon && <polygon points={asPath(selectedPolygon)} fill="rgba(255,160,40,.18)" stroke="#ff9d2e" strokeWidth="3" vectorEffect="non-scaling-stroke"/>}
          {pointerPosition && tool!=='pan' && <circle cx={pointerPosition[0]} cy={pointerPosition[1]} r={brushSize/2} fill="rgba(190,70,225,.13)" stroke="#ad45ec" strokeWidth={Math.max(2,brushSize/20)}/>}
        </svg>
      </div>
    </div>
    {selectedPolygon && <div className="manual-selection-actions"><span>Selected region ready for review at {zoom}% zoom.</span><button className="secondary-button" onClick={()=>addSelectionEdit(true)}>Erase selection</button><button className="secondary-button" onClick={()=>addSelectionEdit(false)}>Fill selection</button><button className="secondary-button" onClick={()=>setSelectedPolygon(null)}>Clear selection</button></div>}
    {repairNeedsPreview && <div className="manual-selection-actions"><span>Repair strokes stay pending until you inspect their result.</span><button className="secondary-button" onClick={makeRepairPreview}>Preview repair</button></div>}
    <div className="manual-editor-foot"><span>Tool: <strong>{tool}</strong> · {width} × {height}px · {zoom}% · {isDirty?'Unsaved edits':'Saved'} · {operations.length} operations</span><div><button className="secondary-button" onClick={resetUnsaved} disabled={!isDirty}>Reset to saved</button><button className="secondary-button" onClick={discardAll} disabled={!operations.length}>Discard all</button><button className="primary-button" onClick={save} disabled={!isDirty||saving}><Save size={15}/>{saving?'Applying…':'Save edits'}</button></div></div>
    {notice && <p role="status" className="manual-notice">{notice}</p>}
    {repairPreviewUrl && <div className="repair-preview-modal" role="dialog" aria-modal="true" aria-label="Repair preview"><div><h3>Repair preview</h3><p>Downscaled visual preview. Inspect for boundary crossings; save applies the same strokes at source resolution.</p><img src={repairPreviewUrl} alt="Pending repair preview"/><div><button className="secondary-button" onClick={()=>{setRepairPreviewUrl('');setPreviewHash('');}}>Continue editing</button><button className="primary-button" onClick={()=>{setRepairPreviewUrl('');setNotice('Repair preview confirmed. Save edits to apply at full resolution.');}}>Confirm preview</button></div></div></div>}
  </section>;
}
