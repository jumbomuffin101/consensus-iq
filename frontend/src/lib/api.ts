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
  citation_id: string;
  title: string;
  source: string;
  url: string;
  snippet: string;
  relevance_score: number;
};

export type AnalyzeResponse = {
  consensus: string;
  confidence_score: number;
  agreement_score: number;
  reasoning_summary: string;
  agent_outputs: AgentOutput[];
  disagreements: Disagreement[];
  sources: Source[];
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function analyzeQuestion(
  question: string,
): Promise<AnalyzeResponse> {
  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    throw new Error(`Analysis failed with status ${response.status}`);
  }

  return response.json();
}
