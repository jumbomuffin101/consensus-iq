import type { AnalyzeResponse } from "@/lib/api";

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
