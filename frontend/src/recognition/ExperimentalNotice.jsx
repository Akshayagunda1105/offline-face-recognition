export default function ExperimentalNotice({ check }) {
  if (check?.state !== "loaded" || check.data?.experimental !== true) return null;
  return (
    <section className="experimental-banner" role="status" aria-live="polite">
      <strong>Experimental recognition — not calibrated</strong>
      <span>Results are for local evaluation only and must not be treated as validated identity decisions.</span>
    </section>
  );
}
