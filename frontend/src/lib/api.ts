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
  citation_validity?: {
    valid: boolean;
    invalid_citations: string[];
    available_sources: string[];
  };
  provider_used?: "openrouter" | "mock" | "azure";
  fallback_reason?: string | null;
  metadata?: {
    execution_time_ms: number;
    retrieval_time_ms: number;
    agent_time_ms: number;
    consensus_time_ms: number;
  };
};

const LOCAL_DEV_API_URL = "http://localhost:8000";

export function normalizeApiBaseUrl(value: string) {
  return value.trim().replace(/\/+$/, "");
}

export function getApiBaseUrl() {
  const configuredUrl = process.env.NEXT_PUBLIC_API_URL?.trim();
  if (configuredUrl) {
    return normalizeApiBaseUrl(configuredUrl);
  }

  if (process.env.NODE_ENV !== "production") {
    return LOCAL_DEV_API_URL;
  }

  throw new Error(
    "NEXT_PUBLIC_API_URL is not configured. Set NEXT_PUBLIC_API_URL to the deployed FastAPI backend URL, for example https://consensusiq-api.onrender.com.",
  );
}

export async function analyzeQuestion(
  question: string,
): Promise<AnalyzeResponse> {
  const apiBaseUrl = getApiBaseUrl();
  await checkBackendHealth(apiBaseUrl);

  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch (error) {
    throw new Error(
      [
        "Unable to reach the ConsensusIQ API.",
        `Configured API base URL: ${apiBaseUrl}`,
        "Failure type: network/CORS",
        "Expected env var: NEXT_PUBLIC_API_URL",
        error instanceof Error ? `Details: ${error.message}` : null,
      ]
        .filter(Boolean)
        .join(" "),
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

async function checkBackendHealth(apiBaseUrl: string) {
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}/health`, {
      method: "GET",
      cache: "no-store",
    });
  } catch (error) {
    throw new Error(
      [
        "Backend health check failed.",
        `Configured API base URL: ${apiBaseUrl}`,
        "Failure type: network/CORS",
        "Expected endpoint: /health",
        error instanceof Error ? `Details: ${error.message}` : null,
      ]
        .filter(Boolean)
        .join(" "),
    );
  }

  if (!response.ok) {
    throw new Error(
      [
        "Backend health check failed.",
        `Configured API base URL: ${apiBaseUrl}`,
        "Failure type: non-200",
        `Status: ${response.status}`,
      ].join(" "),
    );
  }
}
