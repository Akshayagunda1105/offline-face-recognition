import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { API_BASE_URL, ApiError, galleryApi, healthApi } from "./api/client.js";
import { inspectGalleryFiles, MAX_GALLERY_IMAGES } from "./gallery/files.js";

const initialChecks = {
  backend: { state: "loading", data: null, error: null },
  models: { state: "loading", data: null, error: null },
  recognition: { state: "loading", data: null, error: null },
};

function StatusDot({ tone = "neutral" }) {
  return <span className={`status-dot status-dot--${tone}`} aria-hidden="true" />;
}

function StatusTile({ title, state, detail }) {
  const tone = state === "Ready" || state === "Connected" ? "green" : state === "Loading" ? "neutral" : "amber";
  return (
    <div className="health-tile" aria-live="polite">
      <StatusDot tone={tone} />
      <div className="health-copy"><span>{title}</span><strong>{state}</strong>{detail ? <small>{detail}</small> : null}</div>
    </div>
  );
}

function friendlyError(error) {
  if (!(error instanceof ApiError)) return error?.message || "The request failed. Please try again.";
  const messages = {
    network_error: "Could not reach the local backend. Your current gallery is unchanged.",
    reference_images_required: "Select at least one supported reference image.",
    empty_gallery: "Select at least one reference image.",
    too_many_identities: `A gallery can contain at most ${MAX_GALLERY_IMAGES} reference images.`,
    invalid_identity: "A filename does not provide a valid identity label. Rename the file and try again.",
    duplicate_identity: "More than one file has the same identity label. Use one reference image per person and unique filenames.",
    invalid_reference_image: "A reference image is invalid, unsupported, corrupted, or its extension does not match its contents.",
    zero_faces: "A reference image contains no face. Each reference photo must contain exactly one face.",
    multiple_faces: "A reference image contains more than one face. Each reference photo must contain exactly one face.",
    invalid_face_detection: "A reference image produced an invalid face detection. Review the image and try again.",
    embedding_alignment_failure: "A face could not be aligned for one of the references. Review the image and try again.",
    invalid_embedding: "A reference face could not be converted into a valid representation.",
    face_processing_failed: "Face processing failed for a reference image. The previous gallery remains active.",
    gallery_too_large: "The selected images exceed the backend upload limit.",
    gallery_session_capacity: "The backend has reached its active-gallery limit. Delete an unused gallery or wait for one to expire.",
    processing_busy: "The backend is processing another request. Wait briefly and try again.",
    models_not_ready: "The local face models are not available. Check backend model readiness.",
    gallery_session_unavailable: "This gallery has expired, was deleted, or was cleared by a backend restart.",
    gallery_store_closed: "The backend gallery service is unavailable. Try again after it restarts.",
  };
  return messages[error.code] || error.message || "The backend request failed. Please try again.";
}

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Unavailable"
    : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function remainingText(expiresAt, now) {
  const remaining = new Date(expiresAt).getTime() - now;
  if (!Number.isFinite(remaining) || remaining <= 0) return "Expiry time reached — check session status";
  const minutes = Math.floor(remaining / 60_000);
  const seconds = Math.floor((remaining % 60_000) / 1000);
  return minutes > 0 ? `${minutes}m ${seconds}s remaining` : `${seconds}s remaining`;
}

export default function App() {
  const inputRef = useRef(null);
  const [checks, setChecks] = useState(initialChecks);
  const [selection, setSelection] = useState(null);
  const [gallery, setGallery] = useState(null);
  const [now, setNow] = useState(Date.now());
  const [busy, setBusy] = useState(null);
  const [notice, setNotice] = useState(null);

  const fileAnalysis = useMemo(() => selection || inspectGalleryFiles([]), [selection]);

  const refreshHealth = useCallback(async () => {
    setChecks(initialChecks);
    const [backend, models, recognition] = await Promise.allSettled([
      healthApi.liveness(), healthApi.models(), healthApi.recognition(),
    ]);
    const result = {};
    for (const [key, item] of Object.entries({ backend, models, recognition })) {
      result[key] = item.status === "fulfilled"
        ? { state: "loaded", data: item.value.data, status: item.value.status, error: null }
        : { state: "error", data: null, error: friendlyError(item.reason) };
    }
    setChecks(result);
  }, []);

  useEffect(() => { refreshHealth(); }, [refreshHealth]);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  function onFolderSelected(event) {
    setSelection(inspectGalleryFiles(event.target.files));
    setNotice(null);
    event.target.value = "";
  }

  async function createGallery() {
    if (busy || fileAnalysis.files.length === 0 || fileAnalysis.exceedsLimit ||
        fileAnalysis.duplicateIdentities.length || fileAnalysis.invalidFilenames.length) return;
    const previous = gallery;
    setBusy("creating");
    setNotice({ tone: "info", text: "Uploading reference photos and building the gallery…" });
    try {
      const created = await galleryApi.create(fileAnalysis.files);
      setGallery({ ...created });
      setSelection(null);
      setNotice({ tone: "success", text: `Gallery created with ${created.identity_count} identities.` });

      if (previous?.session_id) {
        try {
          await galleryApi.deleteCurrent(previous.session_id);
          setNotice({ tone: "success", text: "New gallery is active; the previous gallery was deleted." });
        } catch (error) {
          if (error.code === "gallery_session_unavailable") {
            setNotice({ tone: "success", text: "New gallery is active; the previous session had already expired or been cleared." });
          } else {
            setNotice({ tone: "warning", text: `New gallery is active, but previous-session cleanup failed: ${friendlyError(error)}` });
          }
        }
      }
    } catch (error) {
      setNotice({ tone: "error", text: friendlyError(error) });
    } finally {
      setBusy(null);
    }
  }

  async function inspectGallery() {
    if (!gallery || busy) return;
    setBusy("inspecting");
    setNotice(null);
    try {
      const info = await galleryApi.getCurrent(gallery.session_id);
      setGallery((current) => current?.session_id === gallery.session_id ? { ...current, ...info } : current);
      setNotice({ tone: "success", text: "Gallery session is active." });
    } catch (error) {
      if (error.code === "gallery_session_unavailable") {
        setGallery((current) => current?.session_id === gallery.session_id ? null : current);
        setNotice({ tone: "warning", text: "This gallery expired or was cleared by a backend restart. Select the reference folder again to create a new session." });
      } else {
        setNotice({ tone: "error", text: friendlyError(error) });
      }
    } finally {
      setBusy(null);
    }
  }

  async function deleteGallery() {
    if (!gallery || busy) return;
    const deleting = gallery;
    setBusy("deleting");
    setNotice(null);
    try {
      await galleryApi.deleteCurrent(deleting.session_id);
      setGallery((current) => current?.session_id === deleting.session_id ? null : current);
      setNotice({ tone: "success", text: "Gallery deleted from the backend session store." });
    } catch (error) {
      if (error.code === "gallery_session_unavailable") {
        setGallery((current) => current?.session_id === deleting.session_id ? null : current);
        setNotice({ tone: "warning", text: "The gallery was already expired or cleared. Local session state has been removed." });
      } else {
        setNotice({ tone: "error", text: friendlyError(error) });
      }
    } finally {
      setBusy(null);
    }
  }

  const backend = checks.backend;
  const models = checks.models;
  const recognition = checks.recognition;
  const canCreate = fileAnalysis.files.length > 0 && !fileAnalysis.exceedsLimit &&
    fileAnalysis.duplicateIdentities.length === 0 && fileAnalysis.invalidFilenames.length === 0 && !busy;
  const directorySupported = typeof document !== "undefined" && "webkitdirectory" in document.createElement("input");

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#home" aria-label="Offline Face Recognition home">
          <span className="brand-mark" aria-hidden="true">OF</span><span>Offline Face Recognition</span>
        </a>
        <div className="topbar-right"><span className="local-tag"><span className="local-dot" /> Local runtime</span><span className="local-caption">No cloud services</span></div>
      </header>

      <section className="intro" id="home">
        <div className="eyebrow">REFERENCE GALLERY</div>
        <h1>Choose a reference folder</h1>
        <p>Build a temporary, local gallery from one reference photo per person. Filenames provide identity labels.</p>
      </section>

      <section className="workspace" aria-label="Reference gallery management">
        <div className="panel selection-panel">
          <div className="panel-heading"><div><div className="step-label">01 / SELECT</div><h2>Reference photos</h2></div><span className="limit-label">Up to {MAX_GALLERY_IMAGES}</span></div>
          <input ref={inputRef} className="visually-hidden" type="file" accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp" multiple webkitdirectory="" directory="" onChange={onFolderSelected} aria-label="Select a reference folder" />
          <button className="folder-picker" type="button" onClick={() => inputRef.current?.click()} disabled={Boolean(busy)}>
            <span className="folder-icon" aria-hidden="true">▱</span><span className="picker-title">Select reference folder</span><span className="picker-caption">JPG · JPEG · PNG · WebP</span><span className="picker-action">Browse folders</span>
          </button>
          <p className="picker-help">{directorySupported ? "Choose a folder in a Chromium-based browser. If directory selection is unavailable, select all reference images together." : "This browser may not support directory selection. Use the file picker to select the reference images together."}</p>
          <div className="rule-note"><span className="rule-icon">i</span><span>Each image must contain <strong>exactly one face</strong>. The filename stem becomes the person’s label.</span></div>

          {selection ? (
            <div className="file-review">
              <div className="file-review-heading"><strong>{fileAnalysis.files.length} supported image{fileAnalysis.files.length === 1 ? "" : "s"}</strong><span>{fileAnalysis.selectedCount} file{fileAnalysis.selectedCount === 1 ? "" : "s"} selected</span></div>
              {fileAnalysis.files.length > 0 ? <ul className="file-list" aria-label="Selected reference filenames">{fileAnalysis.files.map((file, index) => <li key={`${file.name}-${index}`}><span className="file-badge">IMG</span><span className="filename">{file.name}</span><span className="identity-preview">{file.name.replace(/\.[^.]+$/, "")}</span></li>)}</ul> : <p className="selection-message selection-message--error">{fileAnalysis.selectedCount === 0 ? "The selected folder is empty. Choose a folder containing reference images." : "No supported images were found. Use JPG, JPEG, PNG, or WebP files."}</p>}
              {fileAnalysis.unsupportedFiles.length > 0 ? <p className="selection-message selection-message--warning">Ignored {fileAnalysis.unsupportedFiles.length} unsupported file{fileAnalysis.unsupportedFiles.length === 1 ? "" : "s"}: {fileAnalysis.unsupportedFiles.join(", ")}</p> : null}
              {fileAnalysis.exceedsLimit ? <p className="selection-message selection-message--error">The selection has {fileAnalysis.files.length} supported images; the limit is {MAX_GALLERY_IMAGES}. Reduce the selection before uploading.</p> : null}
              {fileAnalysis.invalidFilenames.length > 0 ? <p className="selection-message selection-message--error">These filenames do not provide a valid identity label: {fileAnalysis.invalidFilenames.join(", ")}</p> : null}
              {fileAnalysis.duplicateIdentities.length > 0 ? <div className="selection-message selection-message--error" role="alert"><strong>Duplicate identity labels (case-insensitive):</strong>{fileAnalysis.duplicateIdentities.map((duplicate) => <div key={duplicate.label}>{duplicate.filenames.join(" and ")} → “{duplicate.label}”</div>)}<span>Use one reference photo per person and unique filenames.</span></div> : null}
            </div>
          ) : null}

          <div className="upload-row"><button className="primary-button" onClick={createGallery} disabled={!canCreate}>{busy === "creating" ? <><span className="button-spinner" /> Building gallery…</> : gallery ? "Create replacement gallery" : "Create gallery"}</button>{selection && fileAnalysis.files.length > 0 ? <button className="text-button" onClick={() => { setSelection(null); setNotice(null); }} disabled={Boolean(busy)}>Clear selection</button> : null}</div>
        </div>

        <aside className="panel active-panel" aria-live="polite">
          <div className="panel-heading"><div><div className="step-label">02 / ACTIVE SESSION</div><h2>Current gallery</h2></div>{gallery ? <span className="active-indicator"><StatusDot tone="green" /> Active</span> : <span className="inactive-indicator">None</span>}</div>
          {gallery ? <>
            <div className="gallery-count"><strong>{gallery.identity_count}</strong><span>identit{gallery.identity_count === 1 ? "y" : "ies"} in this session</span></div>
            <div className="gallery-dates"><div><span>Created</span><strong>{formatDate(gallery.created_at)}</strong></div><div><span>Expires</span><strong>{formatDate(gallery.expires_at)}</strong></div></div>
            <div className="expiry-line"><span className="expiry-pulse" />{remainingText(gallery.expires_at, now)}</div>
            <div className="identity-section"><div className="identity-heading"><strong>Identity labels</strong><span>{gallery.identity_labels.length}</span></div><ul className="identity-list">{gallery.identity_labels.map((label, index) => <li key={`${label}-${index}`}><span className="identity-avatar">{label.slice(0, 1).toLocaleUpperCase()}</span><span>{label}</span></li>)}</ul></div>
            <div className="gallery-actions"><button className="secondary-button" onClick={inspectGallery} disabled={Boolean(busy)}>{busy === "inspecting" ? "Checking…" : "Check session"}</button><button className="danger-button" onClick={deleteGallery} disabled={Boolean(busy)}>{busy === "deleting" ? "Deleting…" : "Delete gallery"}</button></div>
          </> : <div className="empty-gallery"><span className="empty-icon" aria-hidden="true">◎</span><strong>No active gallery</strong><p>Select the reference folder and create a gallery. The session token stays in memory and is not saved in the browser.</p></div>}
        </aside>
      </section>

      {notice ? <div className={`notice notice--${notice.tone}`} role={notice.tone === "error" ? "alert" : "status"}><StatusDot tone={notice.tone === "success" ? "green" : notice.tone === "warning" || notice.tone === "info" ? "amber" : "red"} /><span>{notice.text}</span><button className="notice-close" aria-label="Dismiss message" onClick={() => setNotice(null)}>×</button></div> : null}

      <section className="system-section" aria-label="Local backend status">
        <div className="system-heading"><div><div className="step-label">LOCAL SERVICE</div><h2>System status</h2></div><button className="refresh-button" onClick={refreshHealth}><span aria-hidden="true">↻</span> Refresh</button></div>
        <div className="health-grid">
          <StatusTile title="Backend" state={backend.state === "loading" ? "Loading" : backend.state === "error" ? "Unavailable" : backend.data?.status === "alive" ? "Connected" : "Unavailable"} detail={backend.state === "error" ? backend.error : "Local API"} />
          <StatusTile title="Face models" state={models.state === "loading" ? "Loading" : models.state === "error" ? "Unavailable" : models.data?.models_ready ? "Ready" : "Unavailable"} detail={models.state === "error" ? models.error : models.data?.error_code || "YuNet · SFace"} />
          <StatusTile title="Recognition" state={recognition.state === "loading" ? "Loading" : recognition.state === "error" ? "Unavailable" : recognition.data?.recognition_ready ? "Ready" : "Not ready"} detail={recognition.state === "error" ? recognition.error : recognition.data?.reason || "Policy validation pending"} />
        </div>
        <p className="api-address">API <code>{API_BASE_URL}</code></p>
      </section>

      <footer className="page-footer"><span>Runs locally on this device</span><span>Gallery embeddings are held in backend memory and expire automatically.</span></footer>
    </main>
  );
}
