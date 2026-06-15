import type { AnalyzeResponse } from "@/lib/api";
import { getProviderDebugInfo } from "@/lib/response-utils";

const responseWithoutOptionalDebug: AnalyzeResponse = {
  consensus: "Use a cautious recommendation.",
  scenario_label: "Custom",
  confidence_score: 0.4,
  agreement_score: 0.7,
  reasoning_summary: "Partial evidence only.",
  agent_outputs: [],
  disagreements: [],
  sources: [],
  metadata: {
    execution_time_ms: 10,
    retrieval_time_ms: 1,
    agent_time_ms: 2,
    consensus_time_ms: 3,
  },
};

const debugInfo = getProviderDebugInfo(responseWithoutOptionalDebug);

if (
  debugInfo.provider !== "Unavailable" ||
  debugInfo.liveLlmMode !== "Unavailable" ||
  debugInfo.openRouterCalls !== 0 ||
  debugInfo.fallbackReason !== "None"
) {
  throw new Error("Optional debug field fallback changed unexpectedly.");
}
