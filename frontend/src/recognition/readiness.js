const reasons = {
  models_not_ready: "Local face models are unavailable.",
  experimental_opt_in_disabled: "Experimental recognition is disabled in backend configuration.",
  matching_thresholds_not_configured: "Set both experimental matching thresholds in backend configuration.",
};

export function recognitionAvailabilityMessage(check) {
  if (check?.state === "loading") return "Checking backend recognition status…";
  if (check?.state === "error") return `Recognition status unavailable: ${check.error}`;
  if (check?.data?.experimental_recognition_available === true) {
    return "Experimental mode enabled; decisions are not calibrated.";
  }
  return reasons[check?.data?.reason] || "Experimental recognition is unavailable.";
}
