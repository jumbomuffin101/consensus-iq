export type AgentOutput = {
  agent: string;
  role: string;
  stance: "support" | "caution" | "alternative" | "synthesis";
  recommendation: string;
  conclusion: string;
  rationale: string[];
  confidence_score: number;
  evidence_refs: string[];
  missing_evidence: string[];
  limitations: string[];
};

export type Disagreement = {
  topic: string;
  kind:
    | "conflicting_recommendation"
    | "differing_confidence"
    | "missing_evidence";
  positions: string[];
  severity: "low" | "medium" | "high";
  suggested_resolution: string;
};

export type Source = {
  id?: string;
  citation_id: string;
  title: string;
  source: string;
  url: string;
  snippet: string;
  relevance_score: number;
  retrieved_at?: string;
};

export type AnalyzeResponse = {
  consensus: string;
  scenario_label:
    | "Clinical"
    | "Cybersecurity"
    | "Research"
    | "Enterprise"
    | "Finance"
    | "Health / Sports Injury"
    | "Custom";
  confidence_score: number;
  agreement_score: number;
  reasoning_summary: string;
  agent_outputs: AgentOutput[];
  disagreements: Disagreement[];
  sources: Source[];
  metadata?: {
    execution_time_ms: number;
    retrieval_time_ms: number;
    agent_time_ms: number;
    consensus_time_ms: number;
  };
};

function getApiBaseUrl() {
  const configuredUrl = process.env.NEXT_PUBLIC_API_URL?.trim();
  if (configuredUrl) {
    return configuredUrl.replace(/\/$/, "");
  }

  return "https://consensusiq-api.onrender.com";
}

export async function analyzeQuestion(
  question: string,
): Promise<AnalyzeResponse> {
  let response: Response;
  try {
    const apiBaseUrl = getApiBaseUrl();
    response = await fetch(`${apiBaseUrl}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch {
    throw new Error(
      "Unable to reach the ConsensusIQ API. Confirm the backend is deployed and NEXT_PUBLIC_API_URL is set.",
    );
  }

  if (!response.ok) {
    let message = "Analysis failed. Please check that the backend API is running.";
    try {
      const errorBody = await response.json();
      if (typeof errorBody.detail === "string") {
        message = errorBody.detail;
      }
    } catch {
      if (response.status >= 500) {
        message = "The reasoning service is temporarily unavailable.";
      }
    }
    throw new Error(message);
  }

  return response.json();
}
