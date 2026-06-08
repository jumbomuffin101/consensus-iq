export type AgentOutput = {
  agent: string;
  role: string;
  stance: "support" | "caution" | "alternative" | "synthesis";
  summary: string;
  confidence: number;
  evidence_refs: string[];
};

export type Disagreement = {
  topic: string;
  positions: string[];
  severity: "low" | "medium" | "high";
};

export type AnalyzeResponse = {
  consensus: string;
  confidence: number;
  agreement_score: number;
  agent_outputs: AgentOutput[];
  disagreements: Disagreement[];
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
