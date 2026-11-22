export default function RecognitionResults({ result, imageUrl }) {
  if (!result) return null;
  return (
    <section className="result-panel" aria-labelledby="recognition-result-heading" aria-live="polite">
      <div className="result-heading">
        <div><div className="step-label">RECOGNITION RESULT</div><h2 id="recognition-result-heading">Annotated group photo</h2></div>
        <span className="face-count"><strong>{result.faceCount}</strong> {result.faceCount === 1 ? "face" : "faces"}</span>
      </div>
      {result.faceCount === 0 ? <p className="no-faces-message" role="status">No faces detected in this photo.</p> : null}
      <div className="result-layout">
        <div className="annotated-preview"><img src={imageUrl} alt="Group photo annotated by the local recognition system" /></div>
        <div className="identification-panel">
          <h3>Identifications</h3>
          {result.identifications.length > 0 ? (
            <ol className="identification-list">
              {result.identifications.map((item) => (
                <li key={`${item.detection_index}-${item.status}`}>
                  <span className="detection-number">{item.detection_index + 1}</span>
                  <span className={`identification-label identification-label--${item.status.toLowerCase()}`}>
                    {item.status === "Known" ? item.identity_label : "Unknown"}
                  </span>
                </li>
              ))}
            </ol>
          ) : <p className="muted-result">There are no face results to list.</p>}
        </div>
      </div>
    </section>
  );
}
