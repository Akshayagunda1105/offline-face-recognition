import { formatFileSize } from "../group-photo/file.js";
import RecognitionResults from "./RecognitionResults.jsx";

export default function GroupPhotoWorkflow({
  selection,
  previewUrl,
  result,
  resultImageUrl,
  hasGallery,
  busy,
  onSelect,
  onClear,
  onRecognize,
}) {
  return (
    <section className="group-workflow" aria-labelledby="group-photo-heading">
      <div className="workflow-heading">
        <div><div className="eyebrow">RECOGNITION</div><h2 id="group-photo-heading">Group photo</h2><p>Select one group photograph to process against the active reference gallery.</p></div>
        <span className="limit-label">Maximum 20 MiB</span>
      </div>

      <div className="group-photo-grid">
        <div className="photo-input-card">
          <label className="group-file-picker">
            <span className="photo-icon" aria-hidden="true">▧</span>
            <span className="picker-title">{selection ? "Choose a different photo" : "Select group photo"}</span>
            <span className="picker-caption">JPG · JPEG · PNG · WebP</span>
            <input type="file" accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp" onChange={onSelect} disabled={Boolean(busy)} />
          </label>

          {selection ? (
            <div className="photo-file-details">
              <div><span>Filename</span><strong title={selection.file.name}>{selection.file.name}</strong></div>
              <div><span>File size</span><strong>{formatFileSize(selection.file.size)}</strong></div>
              {selection.error ? <p className="selection-message selection-message--error" role="alert">{selection.error}</p> : null}
              {previewUrl ? <button className="text-button" onClick={onClear} disabled={Boolean(busy)}>Clear selected photo</button> : null}
            </div>
          ) : <p className="picker-help">The original image is previewed locally and is not changed.</p>}

          <button className="primary-button recognize-button" onClick={onRecognize} disabled={!selection?.valid || !hasGallery || Boolean(busy)}>
            {busy === "recognizing" ? <><span className="button-spinner" /> Uploading and processing…</> : "Recognize group photo"}
          </button>
          {!hasGallery ? <p className="inline-guidance">Create a reference gallery before starting recognition.</p> : null}
        </div>

        <div className="original-preview-card">
          <div className="preview-title"><strong>Original photo preview</strong><span>Unmodified</span></div>
          {previewUrl ? <img className="original-preview" src={previewUrl} alt="Selected original group photo preview" /> : <div className="photo-placeholder"><span aria-hidden="true">▧</span><p>Your selected photo preview will appear here.</p></div>}
        </div>
      </div>

      <RecognitionResults result={result} imageUrl={resultImageUrl} />
    </section>
  );
}
