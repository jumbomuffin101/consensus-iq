import type { AnalyzeResponse } from "./api";
import { getProviderDebugInfo, getScoreDisplay } from "./response-utils";

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

const weakTriageResponse: AnalyzeResponse = {
  ...responseWithoutOptionalDebug,
  final_answer: {
    summary: "No strong sources were retrieved for this custom prompt.",
    recommendation: "Contact a veterinarian if symptoms are unusual or severe.",
    key_findings: [],
    risks_or_limitations: [],
    follow_up_questions: [],
    source_quality: "weak",
    provider_used: "fast-deterministic",
    live_llm_mode: "off",
  },
  metadata: {
    execution_time_ms: 10,
    retrieval_time_ms: 1,
    agent_time_ms: 2,
    consensus_time_ms: 3,
    custom_intake: {
      domain: "pet_health",
      intent: "triage",
      urgency: "moderate",
      missing_information: ["age"],
      retrieval_queries: [],
      answer_style: "Safe veterinary triage guidance.",
      confidence: 0.9,
    },
  },
};

if (
  getScoreDisplay(weakTriageResponse, "confidence") !== "Needs more info" ||
  getScoreDisplay(weakTriageResponse, "agreement") !== "Needs more info"
) {
  throw new Error("Weak triage responses should hide numeric score display.");
}
