import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity,
  AlertCircle,
  ArrowDownToLine,
  Check,
  ChevronRight,
  CircleHelp,
  FileImage,
  Files,
  ImagePlus,
  LoaderCircle,
  Map,
  Maximize2,
  Plus,
  RefreshCw,
  ScanLine,
  Settings2,
  ShieldCheck,
  Trash2,
  UploadCloud,
  X,
} from 'lucide-react';
import './styles.css';
import ManualEditor from './ManualEditor.jsx';

const API_BASE = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const apiUrl = (path) => `${API_BASE}${path}`;
const formatBytes = (size) =>
  size < 1024 * 1024
    ? `${Math.max(1, Math.round(size / 1024))} KB`
    : `${(size / (1024 * 1024)).toFixed(1)} MB`;
const filenameOf = (map) => map?.filename || 'Original filename unavailable';

function ProgressBar({ value = 0, tone = 'green', indeterminate = false }) {
  return (
    <div
      className={`progress-track ${indeterminate ? 'is-indeterminate' : ''}`}
      role="progressbar"
      aria-valuemin="0"
      aria-valuemax="100"
      aria-valuenow={indeterminate ? undefined : value}
    >
      <div
        className={`progress-fill ${tone}`}
        style={indeterminate ? undefined : { width: `${value}%` }}
      />
    </div>
  );
}

function App() {
  const [maps, setMaps] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [uploadStates, setUploadStates] = useState({});
  const [isDragging, setIsDragging] = useState(false);
  const [filter, setFilter] = useState('all');
  const [apiOnline, setApiOnline] = useState(false);
  const [notice, setNotice] = useState('');
  const [approved, setApproved] = useState({});
  const [assembly, setAssembly] = useState({});
  const [assemblyMode, setAssemblyMode] = useState(false);
  const [finalPreview, setFinalPreview] = useState(false);
  const [compare, setCompare] = useState(false);
  const [zoom, setZoom] = useState(100);
  const [candidateLoaded, setCandidateLoaded] = useState(false);
  const [manualEditMode, setManualEditMode] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const primaryInput = useRef(null);
  const sheetsInput = useRef(null);
  const selected = maps.find((map) => map.id === selectedId) || maps[0] || null;
  const primary = maps.find((map) => map.map_type === 'primary') || null;
  const sheets = maps.filter((map) => map.map_type === 'sheet');
  const visibleMaps = useMemo(
    () => (filter === 'all' ? maps : maps.filter((map) => map.map_type === filter)),
    [maps, filter],
  );

  const refreshMaps = useCallback(async () => {
    try {
      const res = await fetch(apiUrl('/api/maps'));
      if (!res.ok) throw new Error('API unavailable');
      const data = await res.json();
      const hydrated = await Promise.all(
        data.map(async (map) => {
          if (!map.latest_job_id) return map;
          try {
            const jobResponse = await fetch(apiUrl(`/api/jobs/${map.latest_job_id}`));
            if (jobResponse.ok) return { ...map, job: await jobResponse.json() };
          } catch {
            /* A missing in-memory job does not make the source unavailable. */
          }
          return map;
        }),
      );
      setMaps(hydrated);
      setApiOnline(true);
      setSelectedId((current) =>
        current && hydrated.some((map) => map.id === current) ? current : hydrated[0]?.id || null,
      );
    } catch {
      setApiOnline(false);
    }
  }, []);

  useEffect(() => {
    refreshMaps();
  }, [refreshMaps]);

  const activeJobKey = maps
    .filter((map) => map.latest_job_id && ['queued', 'processing'].includes(map.job?.status))
    .map((map) => `${map.id}:${map.latest_job_id}`)
    .join('|');
  useEffect(() => {
    const active = activeJobKey
      ? activeJobKey.split('|').map((pair) => {
          const [id, latest_job_id] = pair.split(':');
          return { id, latest_job_id };
        })
      : [];
    if (!active.length) return undefined;
    let cancelled = false;
    const poll = async () =>
      Promise.all(
        active.map(async (map) => {
          try {
            const response = await fetch(apiUrl(`/api/jobs/${map.latest_job_id}`));
            if (!response.ok) return;
            const job = await response.json();
            if (!cancelled)
              setMaps((current) =>
                current.map((item) => (item.id === map.id ? { ...item, job } : item)),
              );
          } catch {
            /* transient status errors are retried on the next poll */
          }
        }),
      );
    const timer = window.setInterval(poll, 1500);
    poll();
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [activeJobKey]);

  const setFileStatus = (key, value) =>
    setUploadStates((prev) => ({ ...prev, [key]: { ...prev[key], ...value } }));

  const uploadOne = (file, mapType) =>
    new Promise((resolve) => {
      const tempId = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
      setUploadStates((prev) => ({
        ...prev,
        [tempId]: {
          name: file.name,
          mapType,
          progress: 0,
          state: 'uploading',
          message: 'Starting upload…',
        },
      }));
      const xhr = new XMLHttpRequest();
      xhr.open('POST', apiUrl('/api/maps/upload'));
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable)
          setFileStatus(tempId, {
            progress: Math.round((event.loaded / event.total) * 100),
            message: `Uploading ${formatBytes(event.loaded)} of ${formatBytes(event.total)}…`,
          });
      };
      xhr.onload = async () => {
        try {
          const payload = JSON.parse(xhr.responseText || '{}');
          if (xhr.status < 200 || xhr.status >= 300)
            throw new Error(payload.detail || 'Upload failed.');
          setFileStatus(tempId, { progress: 100, state: 'done', message: 'Upload complete' });
          const added = { ...payload, localSize: file.size };
          setMaps((prev) => [...prev.filter((m) => m.id !== added.id), added]);
          setSelectedId(added.id);
          setApiOnline(true);
          resolve(added);
        } catch (err) {
          setFileStatus(tempId, { state: 'error', message: err.message || 'Upload failed.' });
          resolve(null);
        }
      };
      xhr.onerror = () => {
        setFileStatus(tempId, {
          state: 'error',
          message: 'Network error. Is the FastAPI server running?',
        });
        setApiOnline(false);
        resolve(null);
      };
      const form = new FormData();
      form.append('file', file);
      form.append('map_type', mapType);
      xhr.send(form);
    });

  const handleFiles = async (fileList, mapType) => {
    const files = Array.from(fileList || []);
    if (!files.length) return;
    const accepted = files.filter((file) => /\.(png|jpe?g|tiff?|bmp|webp)$/i.test(file.name));
    const rejected = files.length - accepted.length;
    if (rejected)
      setNotice(
        `${rejected} unsupported file${rejected > 1 ? 's' : ''} skipped. Use PNG, JPG, TIFF, BMP or WEBP.`,
      );
    else setNotice('');
    if (mapType === 'primary' && accepted.length > 1)
      setNotice(
        'Only the first selected image is used as the primary map. Remaining files were skipped.',
      );
    const items = mapType === 'primary' ? accepted.slice(0, 1) : accepted;
    for (const file of items) await uploadOne(file, mapType);
    await refreshMaps();
  };

  const runPreview = async (map) => {
    setMaps((prev) =>
      prev.map((item) =>
        item.id === map.id
          ? {
              ...item,
              job: { status: 'queued', progress: 0, message: 'Adding to processing queue…' },
            }
          : item,
      ),
    );
    try {
      const response = await fetch(apiUrl(`/api/maps/${map.id}/process`), { method: 'POST' });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Could not start processing.');
      const jobId = payload.job_id;
      const jobResponse = await fetch(apiUrl(`/api/jobs/${jobId}`));
      if (!jobResponse.ok)
        throw new Error(
          'Job started, but its status could not be loaded. Use Refresh to reconnect.',
        );
      const job = await jobResponse.json();
      setMaps((prev) =>
        prev.map((item) => (item.id === map.id ? { ...item, job, latest_job_id: jobId } : item)),
      );
    } catch (err) {
      setMaps((prev) =>
        prev.map((item) =>
          item.id === map.id
            ? {
                ...item,
                job: { status: 'failed', progress: 0, message: err.message, error: err.message },
              }
            : item,
        ),
      );
    }
  };

  const activeJob = selected?.job;
  const selectedImageUrl = selected?.image_url ? apiUrl(selected.image_url) : null;
  const previewUrl =
    activeJob?.preview_available && selected?.latest_job_id
      ? `${apiUrl(`/api/jobs/${selected.latest_job_id}/preview`)}?v=${activeJob.updated_at || ''}`
      : null;
  const approveJob = async () => {
    if (!selected?.latest_job_id) {
      throw new Error('No survey sheet selected.');
    }

    const jobId = selected.latest_job_id;
    const response = await fetch(apiUrl(`/api/jobs/${jobId}/approve`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ approved: true }),
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || 'Approval failed.');
    }

    setApproved((prev) => ({ ...prev, [jobId]: true }));

    setMaps((prev) =>
      prev.map((item) =>
        item.id === selected.id
          ? {
              ...item,
              approved: true,
              job: item.job
                ? { ...item.job, approved: true, message: 'Cleaned sheet approved' }
                : item.job,
            }
          : item,
      ),
    );

    return data;
  };

  const approveOnly = async () => {
    try {
      await approveJob();
      setNotice('Survey sheet approved. No file downloaded.');
      setFinalPreview(false);
    } catch (e) {
      setNotice(e.message || 'Approval failed.');
    }
  };

  const approveCandidate = async (format) => {
    try {
      await approveJob();

      const response = await fetch(apiUrl(`/api/jobs/${selected.latest_job_id}/export.${format}`));

      if (!response.ok) {
        throw new Error('Approved artifact download failed.');
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');

      a.href = url;
      a.download = `${(selected.filename || 'survey-map').replace(
        /\.[^.]+$/,
        '',
      )}-cleaned.${format}`;

      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setFinalPreview(false);
    } catch (e) {
      setNotice(e.message || 'Approval or download failed.');
    }
  };
  const openFinalPreview = () => {
    setCandidateLoaded(false);
    setFinalPreview(true);
  };
  const invalidateApproval = async () => {
    if (!selected?.latest_job_id) return;
    setApproved((prev) => ({ ...prev, [selected.latest_job_id]: false }));
    await fetch(apiUrl(`/api/jobs/${selected.latest_job_id}/edit`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ edited: true }),
    });
  };
  const saveAssembly = async () => {
    const layout = { sheets: assembly, primary: primary?.id || null };
    try {
      const r = await fetch(apiUrl('/api/projects/default'), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ layout }),
      });
      if (!r.ok) throw new Error('Could not save assembly.');
      setNotice('Assembly layout saved locally.');
    } catch (e) {
      setNotice(e.message);
    }
  };
  const confirmDeleteMap = async () => {
    if (!deleteTarget?.id || deleting) return;
    setDeleting(true);
    try {
      const response = await fetch(apiUrl(`/api/maps/${encodeURIComponent(deleteTarget.id)}`), {
        method: 'DELETE',
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(result.detail || 'Could not delete this map.');
      setNotice(`${filenameOf(deleteTarget)} deleted.`);
      if (selectedId === deleteTarget.id) setSelectedId(null);
      setDeleteTarget(null);
      await refreshMaps();
    } catch (error) {
      setNotice(error.message || 'Could not delete this map.');
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Map size={20} strokeWidth={1.8} />
          </div>
          <div>
            <div className="brand-name">
              SURVEY<span>STUDIO</span>
            </div>
            <div className="brand-caption">CADASTRAL WORKSPACE</div>
          </div>
        </div>
        <div className="workspace-label">WORKSPACE</div>
        <button className="nav-item active">
          <ScanLine size={17} />
          <span>Map workspace</span>
          <span className="nav-dot" />
        </button>
        <button
          className={`nav-item ${assemblyMode ? 'active' : ''}`}
          onClick={() => setAssemblyMode(!assemblyMode)}
        >
          <Files size={17} />
          <span>Sheet assembly</span>
        </button>
        <button
          className="nav-item"
          onClick={() => setNotice('Settings will be added after the first processing milestone.')}
        >
          <Settings2 size={17} />
          <span>Settings</span>
        </button>
        <div className="sidebar-bottom">
          <div className="local-note">
            <ShieldCheck size={16} />
            <div>
              <strong>Local-first workspace</strong>
              <span>Files stay on this computer while running locally.</span>
            </div>
          </div>
          <div className="version-line">
            <span>Survey Map Studio</span>
            <span>v0.1.0</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumbs">
            <span>Projects</span>
            <ChevronRight size={14} />
            <strong>Village map cleanup</strong>
          </div>
          <div className="topbar-right">
            <span className={`connection ${apiOnline ? 'online' : ''}`}>
              <span className="connection-dot" />
              {apiOnline ? 'API connected' : 'API offline'}
            </span>
            <div className="avatar">SM</div>
          </div>
        </header>

        <section className="page-heading">
          <div>
            <div className="eyebrow">
              <span className="eyebrow-line" /> SURVEY MAPPING / PROJECT 001
            </div>
            <h1>
              Map workspace<span className="heading-period">.</span>
            </h1>
            <p className="subtitle">
              Clean survey sheets without losing the boundaries that matter.
            </p>
          </div>
          <button
            className="secondary-button help-button"
            onClick={() =>
              setNotice(
                'Start by adding one primary village map, then upload the individual numbered survey sheets. Original images are never overwritten.',
              )
            }
          >
            <CircleHelp size={16} /> How it works
          </button>
        </section>

        {notice && (
          <div className="notice">
            <AlertCircle size={16} />
            <span>{notice}</span>
            <button aria-label="Dismiss message" onClick={() => setNotice('')}>
              <X size={15} />
            </button>
          </div>
        )}

        {!apiOnline && (
          <div className="api-warning">
            <AlertCircle size={17} />
            <div>
              <strong>Backend not connected</strong>
              <span>
                Start the FastAPI server using the commands in README.md, then refresh this page.
              </span>
            </div>
            <button onClick={refreshMaps}>
              <RefreshCw size={15} /> Retry
            </button>
          </div>
        )}

        <section className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon primary-icon">
              <Map size={17} />
            </div>
            <div className="stat-label">PRIMARY MAP</div>
            <div className="stat-value">
              {primary ? '01' : '—'}
              <span className="stat-tail"> / 01</span>
            </div>
            <div className="stat-caption">
              {primary ? filenameOf(primary) : 'Reference village map'}
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon sheets-icon">
              <Files size={17} />
            </div>
            <div className="stat-label">SURVEY SHEETS</div>
            <div className="stat-value">
              {String(sheets.length).padStart(2, '0')}
              <span className="stat-tail"> uploaded</span>
            </div>
            <div className="stat-caption">Individual survey-number maps</div>
          </div>
          <div className="stat-card">
            <div className="stat-icon process-icon">
              <Activity size={17} />
            </div>
            <div className="stat-label">PREVIEWS READY</div>
            <div className="stat-value">
              {String(maps.filter((m) => m.job?.status === 'completed').length).padStart(2, '0')}
            </div>
            <div className="stat-caption">Contrast-enhanced review previews</div>
          </div>
        </section>

        <div className="section-header">
          <div>
            <h2>Project files</h2>
            <p>Upload the overview map and the detailed sheets.</p>
          </div>
          <div className="file-actions">
            <button className="secondary-button" onClick={() => primaryInput.current?.click()}>
              <ImagePlus size={16} /> Add primary
            </button>
            <button className="primary-button" onClick={() => sheetsInput.current?.click()}>
              <Plus size={17} /> Add survey sheets
            </button>
            <input
              ref={primaryInput}
              hidden
              type="file"
              accept="image/png,image/jpeg,image/tiff,image/bmp,image/webp"
              onChange={(e) => {
                handleFiles(e.target.files, 'primary');
                e.target.value = '';
              }}
            />
            <input
              ref={sheetsInput}
              hidden
              type="file"
              multiple
              accept="image/png,image/jpeg,image/tiff,image/bmp,image/webp"
              onChange={(e) => {
                handleFiles(e.target.files, 'sheet');
                e.target.value = '';
              }}
            />
          </div>
        </div>

        {assemblyMode && (
          <section className="panel" style={{ padding: 20, marginBottom: 20 }}>
            <div className="panel-head">
              <div className="panel-title">
                <Files size={16} />
                <strong>Puzzle-style assembly</strong>
              </div>
              <button className="secondary-button" onClick={saveAssembly}>
                Save layout
              </button>
            </div>
            <p>
              Drag approved sheet layers to place them over the primary map. Adjust scale and
              rotation without changing source images.
            </p>
            <div
              style={{
                position: 'relative',
                height: 520,
                overflow: 'hidden',
                background: '#e8e5dc',
                border: '1px solid #ddd',
                backgroundImage: primary ? `url(${apiUrl(primary.image_url)})` : undefined,
                backgroundSize: 'contain',
                backgroundRepeat: 'no-repeat',
                backgroundPosition: 'center',
              }}
            >
              {sheets.map((sheet, i) => {
                const pos = assembly[sheet.id] || {
                  x: 25 + i * 3,
                  y: 20 + i * 4,
                  scale: 1,
                  rotation: 0,
                  opacity: 0.72,
                };
                return (
                  <div
                    key={sheet.id}
                    draggable
                    onDragEnd={(e) =>
                      setAssembly((v) => ({
                        ...v,
                        [sheet.id]: {
                          ...pos,
                          x: Math.max(0, Math.min(80, pos.x + e.movementX / 8)),
                          y: Math.max(0, Math.min(80, pos.y + e.movementY / 5)),
                        },
                      }))
                    }
                    style={{
                      position: 'absolute',
                      left: `${pos.x}%`,
                      top: `${pos.y}%`,
                      width: `${Math.round(25 * pos.scale)}%`,
                      transform: `rotate(${pos.rotation}deg)`,
                      opacity: pos.opacity,
                      border: '1px dashed #426d57',
                      cursor: 'move',
                    }}
                  >
                    <img
                      src={apiUrl(sheet.preview_url)}
                      alt={filenameOf(sheet)}
                      style={{ width: '100%', display: 'block' }}
                    />
                    <small style={{ background: 'white' }}>{filenameOf(sheet)}</small>
                    <input
                      aria-label={`${filenameOf(sheet)} scale`}
                      type="range"
                      min="0.5"
                      max="2"
                      step="0.05"
                      value={pos.scale}
                      onChange={(e) =>
                        setAssembly((v) => ({
                          ...v,
                          [sheet.id]: { ...pos, scale: Number(e.target.value) },
                        }))
                      }
                    />
                    <input
                      aria-label={`${filenameOf(sheet)} rotation`}
                      type="range"
                      min="-180"
                      max="180"
                      value={pos.rotation}
                      onChange={(e) =>
                        setAssembly((v) => ({
                          ...v,
                          [sheet.id]: { ...pos, rotation: Number(e.target.value) },
                        }))
                      }
                    />
                  </div>
                );
              })}
            </div>
            <p>
              Placement is manual; choose locations using map evidence. Full-map preview and export
              approval are required before export.
            </p>
          </section>
        )}
        <div className="workspace-grid">
          <section className="files-panel panel">
            <div className="panel-head">
              <div className="panel-title">
                <Files size={16} />
                <strong>Map library</strong>
                <span className="count-pill">{maps.length}</span>
              </div>
              <button className="icon-button" title="Refresh files" onClick={refreshMaps}>
                <RefreshCw size={15} />
              </button>
            </div>
            <div className="filter-row">
              <button
                className={filter === 'all' ? 'filter active-filter' : 'filter'}
                onClick={() => setFilter('all')}
              >
                All files <span>{maps.length}</span>
              </button>
              <button
                className={filter === 'primary' ? 'filter active-filter' : 'filter'}
                onClick={() => setFilter('primary')}
              >
                Primary <span>{primary ? 1 : 0}</span>
              </button>
              <button
                className={filter === 'sheet' ? 'filter active-filter' : 'filter'}
                onClick={() => setFilter('sheet')}
              >
                Sheets <span>{sheets.length}</span>
              </button>
            </div>
            <div className="files-list">
              {visibleMaps.map((map) => (
                <div
                  key={map.id || `invalid-${visibleMaps.indexOf(map)}`}
                  className={`file-row ${selected?.id === map.id ? 'selected-file' : ''}`}
                  role="button"
                  tabIndex={0}
                  onClick={() => map.id && setSelectedId(map.id)}
                  onKeyDown={(event) => {
                    if (
                      (event.key === 'Enter' || event.key === ' ') &&
                      event.target === event.currentTarget
                    ) {
                      event.preventDefault();
                      if (map.id) setSelectedId(map.id);
                    }
                  }}
                >
                  <div
                    className={`file-thumb ${map.map_type === 'primary' ? 'overview-thumb' : ''}`}
                  >
                    {map.thumbnail_url || map.image_url ? (
                      <img src={apiUrl(map.thumbnail_url || map.image_url)} alt="" />
                    ) : (
                      <span className="file-thumb-placeholder">Unavailable</span>
                    )}
                    <span className="type-mini">
                      {map.map_type === 'primary'
                        ? 'MAP'
                        : map.map_type === 'sheet'
                          ? 'SHEET'
                          : '?'}
                    </span>
                  </div>
                  <div className="file-info">
                    <strong title={filenameOf(map)}>{filenameOf(map)}</strong>
                    <span>
                      {map.map_type === 'primary'
                        ? 'Primary map'
                        : map.map_type === 'sheet'
                          ? 'Survey sheet'
                          : 'Map type unavailable'}{' '}
                      ·{' '}
                      {map.width && map.height
                        ? `${map.width} × ${map.height}`
                        : 'Dimensions unavailable'}
                    </span>
                    <span className="file-status">
                      <i
                        className={
                          map.job?.status === 'completed'
                            ? 'status-ready'
                            : map.job?.status === 'failed'
                              ? 'status-error'
                              : ''
                        }
                      />
                      {map.job?.status === 'completed'
                        ? 'Preview ready'
                        : map.job?.status === 'processing'
                          ? 'Processing'
                          : map.job?.status === 'failed'
                            ? 'Needs attention'
                            : 'Original saved'}
                    </span>
                  </div>
                  <ChevronRight className="file-chevron" size={15} />
                  <button
                    className="delete-map-button"
                    type="button"
                    disabled={!map.id}
                    aria-label={`Delete ${filenameOf(map)}`}
                    title={map.id ? 'Delete map' : 'Cannot delete: stable asset ID unavailable'}
                    onClick={(event) => {
                      event.stopPropagation();
                      if (map.id) setDeleteTarget(map);
                    }}
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              ))}
              {!visibleMaps.length && (
                <div className="empty-files">
                  <div className="empty-icon">
                    <UploadCloud size={21} />
                  </div>
                  <strong>No maps uploaded</strong>
                  <span>Add the primary map, then the individual survey sheets.</span>
                  <button className="text-button" onClick={() => primaryInput.current?.click()}>
                    Upload primary map <ChevronRight size={14} />
                  </button>
                </div>
              )}
              {Object.entries(uploadStates)
                .filter(([, state]) => state.state !== 'done')
                .map(([id, state]) => (
                  <div className="upload-row" key={id}>
                    <div className="upload-row-top">
                      <span>
                        <LoaderCircle
                          size={14}
                          className={state.state === 'uploading' ? 'spin' : ''}
                        />
                        {state.name}
                      </span>
                      <span>{state.state === 'error' ? 'ERROR' : `${state.progress}%`}</span>
                    </div>
                    <ProgressBar
                      value={state.progress}
                      tone={state.state === 'error' ? 'red' : 'green'}
                    />
                    <small>{state.message}</small>
                  </div>
                ))}
            </div>
            <button
              className="dropzone"
              onClick={() => sheetsInput.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragging(false);
                handleFiles(e.dataTransfer.files, 'sheet');
              }}
              data-dragging={isDragging}
            >
              <div className="drop-icon">
                <UploadCloud size={19} />
              </div>
              <div>
                <strong>Drop survey sheets here</strong>
                <span>
                  or <u>browse files</u> · up to 40 MB each
                </span>
              </div>
              <Plus size={16} className="drop-plus" />
            </button>
          </section>

          <section className="preview-panel panel">
            <div className="panel-head">
              <div className="panel-title">
                <ScanLine size={16} />
                <strong>Sheet preview</strong>
              </div>
              <span className="review-label">
                <span /> REVIEW MODE
              </span>
            </div>
            {selected ? (
              <>
                <div className="preview-meta">
                  <div>
                    <span className="meta-label">SELECTED FILE</span>
                    <strong title={filenameOf(selected)}>{filenameOf(selected)}</strong>
                    {selected.metadata_warnings?.length > 0 && (
                      <small className="metadata-warning">
                        {selected.metadata_warnings.join(' ')}
                      </small>
                    )}
                  </div>
                  <span className={`kind-pill ${selected.map_type || 'unknown'}`}>
                    {selected.map_type === 'primary'
                      ? 'PRIMARY MAP'
                      : selected.map_type === 'sheet'
                        ? 'SURVEY SHEET'
                        : 'UNCLASSIFIED'}
                  </span>
                </div>
                <div className="image-stage">
                  <div className="stage-toolbar">
                    <span>
                      <span className="stage-dot" /> ORIGINAL
                    </span>
                    <span>
                      {selected.width && selected.height
                        ? `${selected.width} × ${selected.height} px`
                        : 'Dimensions unavailable'}
                    </span>
                  </div>
                  <div className="image-holder">
                    {selectedImageUrl ? (
                      <img src={selectedImageUrl} alt={`Original map: ${filenameOf(selected)}`} />
                    ) : (
                      <span>Original file unavailable</span>
                    )}
                  </div>
                </div>
                {activeJob && (
                  <div
                    className={`job-status ${activeJob.status === 'failed' ? 'job-error' : activeJob.status === 'completed' ? 'job-complete' : ''}`}
                  >
                    <div className="job-status-line">
                      <div className="job-status-label">
                        {activeJob.status === 'completed' ? (
                          <Check size={15} />
                        ) : activeJob.status === 'failed' ? (
                          <AlertCircle size={15} />
                        ) : (
                          <LoaderCircle size={15} className="spin" />
                        )}
                        <strong>
                          {activeJob.status === 'completed'
                            ? 'Preview generated'
                            : activeJob.status === 'failed'
                              ? 'Processing failed'
                              : activeJob.message || activeJob.stage || 'Preparing preview'}
                        </strong>
                      </div>
                      <span>
                        {activeJob.progress || 0}% ·{' '}
                        {Math.floor((activeJob.elapsed_seconds || 0) / 60)}m{' '}
                        {Math.floor((activeJob.elapsed_seconds || 0) % 60)}s
                      </span>
                    </div>
                    <ProgressBar
                      value={activeJob.progress || 0}
                      tone={activeJob.status === 'failed' ? 'red' : 'green'}
                    />
                    <p>{activeJob.error || activeJob.message}</p>
                    {activeJob.status === 'failed' && (
                      <button className="secondary-button" onClick={() => runPreview(selected)}>
                        Retry cleaning
                      </button>
                    )}
                  </div>
                )}
                <div className="preview-actions">
                  <button
                    className="primary-button process-button"
                    disabled={!apiOnline || ['queued', 'processing'].includes(activeJob?.status)}
                    onClick={() => runPreview(selected)}
                  >
                    <ScanLine size={16} />
                    {activeJob?.status === 'completed'
                      ? 'Rerun cleaning'
                      : 'Generate cleaned candidate'}
                  </button>
                  {previewUrl && (
                    <>
                      <button className="secondary-button" onClick={() => setManualEditMode(true)}>
                        <Settings2 size={15} /> Edit candidate
                      </button>
                      <button className="secondary-button view-preview" onClick={openFinalPreview}>
                        <Maximize2 size={15} /> Review / Download Cleaned Map
                      </button>
                    </>
                  )}
                </div>
                {previewUrl && (
                  <div className="preview-result">
                    <div className="result-heading">
                      <Check size={15} />
                      <strong>Latest candidate</strong>
                    </div>
                    <div className="result-image" style={{ maxHeight: 400, overflow: 'auto' }}>
                      <img
                        onLoad={() => setCandidateLoaded(true)}
                        style={{ maxWidth: 'none', width: '100%' }}
                        src={previewUrl}
                        alt={`Cleaned candidate preview of ${filenameOf(selected)}`}
                      />
                    </div>
                    <p>
                      Pixel-verified changes:{' '}
                      {activeJob.review?.qc?.verified_measurement_removals || 0} measurements and{' '}
                      {activeJob.review?.qc?.verified_red_removals || 0} red annotations. Protected
                      principal labels: {activeJob.review?.protected_principal_labels?.length || 0}.
                      Unresolved: {activeJob.review?.unresolved_decimal_like?.length || 0} decimal
                      candidates and {activeJob.review?.dashed_candidates?.length || 0} line
                      candidates. Box symbols and dashed guides remain for manual review.
                    </p>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      {Object.entries(activeJob.review?.diagnostics || {})
                        .filter(([key, value]) => key.endsWith('_url') && value)
                        .map(([key, value]) => (
                          <a
                            key={key}
                            className="secondary-button"
                            href={apiUrl(value)}
                            target="_blank"
                            rel="noreferrer"
                          >
                            {key.replaceAll('_', ' ')}
                          </a>
                        ))}
                    </div>
                  </div>
                )}
                <div className="safety-note">
                  <ShieldCheck size={15} />
                  <p>
                    <strong>Original stays untouched.</strong> Candidate cleanup is conservative;
                    ambiguous small labels and line candidates remain visible for manual inspection.
                    Approval is required before downloads.
                  </p>
                </div>
              </>
            ) : (
              <div className="preview-empty">
                <div className="preview-empty-icon">
                  <FileImage size={27} />
                </div>
                <h3>Your map preview appears here</h3>
                <p>Upload the primary village map or an individual sheet to inspect it.</p>
                <button className="primary-button" onClick={() => primaryInput.current?.click()}>
                  <Plus size={16} /> Upload first map
                </button>
              </div>
            )}
          </section>
        </div>

        {deleteTarget && (
          <div
            className="delete-dialog-backdrop"
            role="presentation"
            onMouseDown={(event) => {
              if (event.target === event.currentTarget && !deleting) setDeleteTarget(null);
            }}
          >
            <section
              className="delete-dialog"
              role="alertdialog"
              aria-modal="true"
              aria-labelledby="delete-map-title"
              aria-describedby="delete-map-description"
            >
              <div className="delete-dialog-icon">
                <Trash2 size={19} />
              </div>
              <h2 id="delete-map-title">Delete map?</h2>
              <p className="delete-file-name">{filenameOf(deleteTarget)}</p>
              <p id="delete-map-description">
                {deleteTarget.original_available
                  ? 'This permanently deletes the original image, thumbnail, and cleaning candidates associated with this map.'
                  : 'The original file is already unavailable. This removes its library record and any associated generated files.'}{' '}
                This cannot be undone.
              </p>
              <div className="delete-dialog-actions">
                <button
                  className="secondary-button"
                  disabled={deleting}
                  onClick={() => setDeleteTarget(null)}
                >
                  Cancel
                </button>
                <button
                  className="delete-confirm-button"
                  disabled={deleting}
                  onClick={confirmDeleteMap}
                >
                  <Trash2 size={15} />
                  {deleting ? 'Deleting…' : 'Delete permanently'}
                </button>
              </div>
            </section>
          </div>
        )}
        {finalPreview && selected && previewUrl && (
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Final cleaned map preview"
            style={{
              position: 'fixed',
              inset: 0,
              zIndex: 50,
              background: 'rgba(12,18,17,.88)',
              display: 'flex',
              flexDirection: 'column',
              padding: 24,
              color: 'white',
            }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: 16,
              }}
            >
              <div>
                <h2 style={{ margin: 0 }}>Final cleaned map preview</h2>
                <p style={{ margin: '6px 0' }}>
                  This is the current candidate artifact. Inspect it before approving a download.
                </p>
              </div>
              <button className="secondary-button" onClick={() => setFinalPreview(false)}>
                <X size={16} /> Close
              </button>
            </div>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center', padding: '12px 0' }}>
              <button className="secondary-button" onClick={() => setCompare((v) => !v)}>
                {compare ? 'Show cleaned' : 'Compare original'}
              </button>
              <label>
                Zoom{' '}
                <input
                  type="range"
                  min="50"
                  max="250"
                  value={zoom}
                  onChange={(e) => setZoom(Number(e.target.value))}
                />
                {zoom}%
              </label>
            </div>
            <div style={{ flex: 1, overflow: 'auto', background: '#e8e8e8', textAlign: 'center' }}>
              <img
                onLoad={() => setCandidateLoaded(true)}
                src={compare ? selectedImageUrl : previewUrl}
                alt={compare ? 'Original reference' : 'Exact final cleaned candidate'}
                style={{ width: `${zoom}%`, maxWidth: 'none', display: 'inline-block' }}
              />
            </div>
            <div style={{ padding: '12px 0' }}>
              <strong>Review status:</strong>{' '}
              {activeJob.review?.qc?.verified_measurement_removals || 0} pixel-verified measurement
              removals; {activeJob.review?.protected_principal_labels?.length || 0} protected
              adjacent principal labels; {activeJob.review?.unresolved_decimal_like?.length || 0}{' '}
              unresolved decimal candidates; {activeJob.review?.dashed_candidates?.length || 0} line
              candidates. Box symbols and confirmed guide removal remain unfinished; inspect the
              candidate before approval.
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
              <button
                className="secondary-button"
                onClick={async () => {
                  await invalidateApproval();
                  setFinalPreview(false);
                }}
              >
                Back to Edit
              </button>
              <div style={{display:'flex', gap:8, flexWrap:'wrap'}}>
                <button
                  className="secondary-button"
                  disabled={!candidateLoaded}
                  onClick={approveOnly}
                >
                  Approve Only
                </button>

                <button
                  className="primary-button"
                  disabled={!candidateLoaded}
                  onClick={() => approveCandidate('png')}
                >
                  Approve and Download PNG
                </button>

                <button
                  className="primary-button"
                  disabled={!candidateLoaded}
                  onClick={() => approveCandidate('pdf')}
                >
                  Approve and Download PDF
                </button>
              </div>
            </div>
          </div>
        )}
        {manualEditMode && selected && activeJob?.status === 'completed' && (
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Manual candidate editor"
            className="manual-editor-modal"
          >
            <ManualEditor
              apiUrl={apiUrl}
              jobId={selected.latest_job_id}
              sourceUrl={selectedImageUrl}
              candidateUrl={previewUrl}
              width={selected.width}
              height={selected.height}
              onClose={() => setManualEditMode(false)}
              onSaved={(data) => {
                setApproved((prev) => ({ ...prev, [selected.latest_job_id]: false }));
                setCandidateLoaded(false);
                setMaps((prev) =>
                  prev.map((item) =>
                    item.id === selected.id
                      ? {
                          ...item,
                          job: {
                            ...item.job,
                            candidate_version: data.candidate_version,
                            approved: false,
                            updated_at: new Date().toISOString(),
                            message: 'Manual edits saved; approval cleared.',
                          },
                        }
                      : item,
                  ),
                );
              }}
            />
          </div>
        )}
        <footer className="footer">
          <span>© 2026 SURVEY MAP STUDIO</span>
          <span>
            <span className="footer-dot" /> Original geometry preservation is the priority
          </span>
          <button
            onClick={() =>
              setNotice('Candidate cleaning and manual assembly tools are available for review.')
            }
          >
            PROJECT STATUS <ChevronRight size={13} />
          </button>
        </footer>
      </main>
    </div>
  );
}

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
