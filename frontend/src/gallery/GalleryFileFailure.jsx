export default function GalleryFileFailure({ report }) {
  if (!report?.failures?.length) return null;
  return (
    <div className="gallery-error-detail">
      <strong>Gallery not created: {report.invalidImageCount} invalid image{report.invalidImageCount === 1 ? "" : "s"}</strong>
      <ul className="gallery-error-list">
        {report.failures.map((failure, index) => (
          <li key={`${failure.filename}-${failure.code}-${index}`}>
            <strong>{failure.filename}</strong>
            <span>Reason: {failure.reason}</span>
            <span>Fix: {failure.fix}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
