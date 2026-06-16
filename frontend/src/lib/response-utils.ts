import type { AnalyzeResponse } from "./api";

export function getProviderDebugInfo(result: AnalyzeResponse | null) {
  return {
    provider:
      result?.metadata?.provider_used ??
      result?.final_answer?.provider_used ??
      "Unavailable",
    liveLlmMode:
      result?.metadata?.live_llm_mode ??
      result?.final_answer?.live_llm_mode ??
      "Unavailable",
    openRouterCalls: result?.metadata?.openrouter_call_count ?? 0,
    fallbackReason: result?.metadata?.fallback_reason || "None",
  };
}

export function getSourceQuality(result: AnalyzeResponse | null) {
  if (result?.final_answer?.source_quality) return result.final_answer.source_quality;
  if (!result?.sources.length) return "weak";
  const topRelevance = Math.max(...result.sources.map((source) => source.relevance_score));
  const averageRelevance =
    result.sources.reduce((total, source) => total + source.relevance_score, 0) /
    result.sources.length;
  if (result.sources.length >= 2 && topRelevance >= 0.7 && averageRelevance >= 0.58) {
    return "strong";
  }
  if (topRelevance >= 0.45) return "partial";
  return "weak";
}

export function shouldShowNeedsMoreInfo(result: AnalyzeResponse | null) {
  const intake = result?.metadata?.custom_intake;
  return getSourceQuality(result) === "weak" || intake?.intent === "triage";
}

export function getScoreDisplay(
  result: AnalyzeResponse | null,
  score: "confidence" | "agreement",
) {
  if (!result) return "Needs more info";
  if (shouldShowNeedsMoreInfo(result)) return "Needs more info";
  const value = score === "confidence" ? result.confidence_score : result.agreement_score;
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
}
