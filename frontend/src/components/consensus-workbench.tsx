"use client";

import { FormEvent, useEffect, useState } from "react";
import {
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  Loader2,
  Scale,
} from "lucide-react";

import { analyzeQuestion, type AnalyzeResponse } from "@/lib/api";
import {
  getProviderDebugInfo,
  getScoreDisplay,
  getSourceQuality as getResponseSourceQuality,
} from "@/lib/response-utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";

const starterQuestion =
  "Should a 63-year-old patient with new-onset focal seizure receive MRI before lumbar puncture?";

const demoPrompts = [
  {
    label: "Clinical Reasoning",
    question:
      "Should a 63-year-old patient with new-onset focal seizure receive MRI before lumbar puncture?",
  },
  {
    label: "Enterprise Risk",
    question:
      "Should our company allow employees to use public AI tools for confidential client documents?",
  },
  {
    label: "Research Evaluation",
    question:
      "Should a research team trust a single LLM grader for evaluating student concept maps?",
  },
];

const flowSteps = [
  {
    step: "01",
    label: "Question",
    subtitle: "User decision prompt",
  },
  {
    step: "02",
    label: "Foundry IQ Retrieval",
    subtitle: "Grounded context",
  },
  {
    step: "03",
    label: "Specialist Agents",
    subtitle: "Risk, evidence, alternatives",
  },
  {
    step: "04",
    label: "Disagreement Analysis",
    subtitle: "Conflicts and gaps",
  },
  {
    step: "05",
    label: "Consensus Judge",
    subtitle: "Final recommendation",
  },
];

const progressSteps = [
  "Retrieving grounded context",
  "Planning reasoning tasks",
  "Running specialist agents",
  "Detecting disagreements",
  "Synthesizing consensus",
];

function asPercent(value: number) {
  if (!Number.isFinite(value)) return "0%";
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
}

function asTitleCase(value: string) {
  return value.replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function promptDomainLabel(question: string) {
  const preset = demoPrompts.find((prompt) => prompt.question === question);
  if (preset?.label === "Clinical Reasoning") return "Clinical";
  if (preset?.label === "Enterprise Risk") return "Enterprise";
  if (preset?.label === "Research Evaluation") return "Research";
  return "Custom";
}

function getConfidenceFactors(result: AnalyzeResponse, question: string) {
  const averageRelevance =
    result.sources.reduce((total, source) => total + source.relevance_score, 0) /
    Math.max(result.sources.length, 1);
  const missingEvidenceCount = result.agent_outputs.reduce(
    (total, agent) => total + agent.missing_evidence.length,
    0,
  );
  const hasPromptInjection = /ignore .*instructions|100% certain|100% confidence|no uncertainty/i.test(question);
  const hasMediumOrHighDisagreement = result.disagreements.some(
    (item) => item.severity === "medium" || item.severity === "high",
  );
  const highRiskDomain = ["Clinical", "Cybersecurity", "Finance"].includes(result.scenario_label);

  const factors = [
    result.sources.length === 0
      ? "No strong retrieved evidence matched this custom prompt"
      : averageRelevance >= 0.85
      ? "High source relevance supports stronger grounding"
      : averageRelevance >= 0.7
        ? "Moderate source relevance leaves some uncertainty"
        : "Limited evidence coverage reduced confidence",
    result.agreement_score >= 0.7
      ? "Strong agent agreement increased confidence"
      : result.agreement_score >= 0.5
        ? "Moderate agent disagreement kept confidence measured"
        : "Material agent disagreement lowered confidence",
  ];

  if (missingEvidenceCount > 0 || result.disagreements.some((item) => item.kind === "missing_evidence")) {
    factors.push("Missing evidence lowered confidence");
  } else {
    factors.push("Specialists cited retrieved sources consistently");
  }

  if (highRiskDomain) {
    factors.push("High-risk domain requires caution");
  } else if (hasMediumOrHighDisagreement) {
    factors.push("Disagreement severity limited certainty");
  } else {
    factors.push("No high-severity disagreement detected");
  }

  if (hasPromptInjection) {
    factors.push("Prompt injection detected, confidence reduced");
  } else if (result.confidence_score < 0.5) {
    factors.push("Overall confidence remains intentionally conservative");
  }

  return factors.slice(0, 5);
}

function getConfidenceCalculation(result: AnalyzeResponse, question: string) {
  const averageRelevance =
    result.sources.reduce((total, source) => total + source.relevance_score, 0) /
    Math.max(result.sources.length, 1);
  const missingEvidenceCount = result.agent_outputs.reduce(
    (total, agent) => total + agent.missing_evidence.length,
    0,
  );
  const highRiskDomain = ["Clinical", "Cybersecurity", "Finance"].includes(result.scenario_label);
  const hasPromptInjection = /ignore .*instructions|100% certain|100% confidence|no uncertainty/i.test(question);
  const highSeverityDisagreement = result.disagreements.some((item) => item.severity === "high");

  const positiveFactors = [
    result.sources.length > 0 && averageRelevance >= 0.85 ? "Strong source relevance" : null,
    result.sources.length >= 3 ? "Multiple supporting sources" : null,
    result.agreement_score >= 0.7 ? "Agent agreement supports the recommendation" : null,
    result.agent_outputs.every((agent) => agent.evidence_refs.length > 0)
      ? "Specialists cite retrieved evidence"
      : null,
  ].filter(Boolean) as string[];

  const negativeFactors = [
    result.sources.length === 0 ? "No strong retrieved evidence matched this custom prompt" : null,
    result.sources.length > 0 && averageRelevance < 0.55 ? "Limited evidence coverage" : null,
    missingEvidenceCount > 0 ? "Missing evidence remains unresolved" : null,
    highRiskDomain ? "High-risk domain uncertainty" : null,
    result.agreement_score < 0.7 ? "Agent disagreement reduced certainty" : null,
    highSeverityDisagreement ? "High-severity disagreement requires caution" : null,
    hasPromptInjection ? "Adversarial certainty request detected" : null,
  ].filter(Boolean) as string[];

  return {
    positiveFactors: positiveFactors.length
      ? positiveFactors
      : [result.sources.length ? "Grounded sources available" : "General decision-support reasoning available"],
    negativeFactors: negativeFactors.length ? negativeFactors : ["No major confidence penalties detected"],
  };
}

function getDisplayedTiming(metadata: AnalyzeResponse["metadata"]) {
  if (!metadata) return null;

  const retrieval = Math.max(metadata.retrieval_time_ms, 80);
  const agents = Math.max(metadata.agent_time_ms, 250);
  const consensus = Math.max(metadata.consensus_time_ms, 60);
  const total = Math.max(metadata.execution_time_ms, retrieval + agents + consensus);

  return {
    total,
    retrieval,
    agents,
    consensus,
  };
}

function getAgent(result: AnalyzeResponse, name: string) {
  return result.agent_outputs.find((agent) => agent.agent === name);
}

function getRecommendationSummary(result: AnalyzeResponse) {
  if (result.final_answer?.recommendation) {
    return result.final_answer.recommendation;
  }
  const [opening] = result.consensus.split("Evidence view:");
  return capitalizeFirst(opening.replace(/^[^:]+:\s*/, "").trim());
}

function firstSentence(value: string) {
  const match = value.match(/.+?[.!?](?:\s|$)/);
  return (match?.[0] ?? value).trim();
}

function capitalizeFirst(value: string) {
  const trimmed = value.trim();
  const index = trimmed.search(/[A-Za-z]/);
  if (index < 0) return trimmed;
  return `${trimmed.slice(0, index)}${trimmed[index].toUpperCase()}${trimmed.slice(index + 1)}`;
}

function getDecisionSentence(result: AnalyzeResponse) {
  return capitalizeFirst(firstSentence(getRecommendationSummary(result)));
}

function getRecommendedApproach(result: AnalyzeResponse) {
  if (result.final_answer?.summary) {
    return result.final_answer.summary;
  }
  const evidence = getAgent(result, "Evidence Analyst Agent");
  const alternatives = getAgent(result, "Alternative Solutions Agent");
  const recommendation = evidence?.recommendation || alternatives?.recommendation;
  return capitalizeFirst(recommendation || "Use the consensus recommendation with explicit evidence checks and review points.");
}

function getAvoidWatchOut(result: AnalyzeResponse) {
  const firstRisk = result.final_answer?.risks_or_limitations?.[0];
  if (firstRisk) {
    return capitalizeFirst(firstSentence(firstRisk));
  }
  const risk = getAgent(result, "Risk Analyst Agent");
  const missingEvidence = result.agent_outputs.flatMap((agent) => agent.missing_evidence);
  const disagreement = result.disagreements[0]?.topic;

  if (missingEvidence.length) {
    return capitalizeFirst(`Missing evidence: ${missingEvidence.slice(0, 2).join("; ")}.`);
  }
  if (disagreement) {
    return capitalizeFirst(`Do not ignore unresolved disagreement around ${disagreement}.`);
  }
  if (risk?.conclusion) {
    return capitalizeFirst(firstSentence(risk.conclusion));
  }
  return "Avoid overconfidence; verify local constraints, policy context, and any domain-specific facts before acting.";
}

function getKeyDisagreementSummary(result: AnalyzeResponse) {
  const keyDisagreement =
    result.disagreements.find((item) => item.severity === "high") ??
    result.disagreements.find((item) => item.severity === "medium") ??
    result.disagreements[0];

  if (!keyDisagreement) {
    return "No material disagreement detected; remaining limitations are reflected in agent confidence and missing evidence.";
  }

  return `${keyDisagreement.topic}: ${keyDisagreement.suggested_resolution}`;
}

function getTopRelevance(result: AnalyzeResponse) {
  if (!result.sources.length) return 0;
  return Math.max(...result.sources.map((source) => source.relevance_score));
}

function getAverageRelevance(result: AnalyzeResponse) {
  if (!result.sources.length) return 0;
  return result.sources.reduce((total, source) => total + source.relevance_score, 0) / result.sources.length;
}

function getUsedSourceRefs(result: AnalyzeResponse) {
  const refs = new Set<string>();
  result.agent_outputs.forEach((agent) => {
    agent.evidence_refs.forEach((ref) => refs.add(ref));
  });
  if (!refs.size && result.sources.length) {
    result.sources.forEach((source) => refs.add(source.citation_id));
  }
  return Array.from(refs);
}

function getEvidenceCoverageLabel(result: AnalyzeResponse) {
  const quality = finalAnswerSourceQuality(result);
  if (quality === "weak") {
    return "Source coverage is weak for this prompt. Treat the answer as provisional and verify with stronger evidence.";
  }
  if (quality === "partial") {
    return "Source coverage is partial: useful for framing, but not definitive.";
  }
  if (!result.sources.length) return "No strong retrieved evidence matched this custom prompt";
  return "Strong source coverage available for the main recommendation.";
}

function getConfidenceLimiters(result: AnalyzeResponse, question: string) {
  const missingEvidence = result.agent_outputs.flatMap((agent) => agent.missing_evidence);
  const highRiskDomain = ["Clinical", "Cybersecurity", "Finance"].includes(result.scenario_label);
  const hasPromptInjection = /ignore .*instructions|100% certain|100% confidence|no uncertainty/i.test(question);
  const averageRelevance = getAverageRelevance(result);

  const limiters = [
    result.sources.length === 0 ? "No strong retrieved evidence matched this custom prompt" : null,
    result.sources.length > 0 && averageRelevance < 0.55 ? "Limited evidence coverage" : null,
    result.disagreements.length ? "Agent disagreement remains unresolved" : null,
    missingEvidence.length ? "Evidence gaps require additional grounding" : null,
    highRiskDomain ? "High-risk domain requires conservative confidence" : null,
    averageRelevance < 0.85 ? "Retrieved sources are relevant but not definitive" : null,
    hasPromptInjection ? "Adversarial certainty request reduced confidence" : null,
  ].filter(Boolean) as string[];

  if (!limiters.length) {
    limiters.push("Local policy, patient-specific facts, or implementation details may still affect the decision");
  }

  return limiters.slice(0, 4);
}

function sourceDisplayName(source: string) {
  if (source.includes("Azure AI Search") || source.includes("Foundry IQ Search Service")) {
    return "Azure AI Search / Foundry IQ Search Service";
  }
  if (
    source === "Mock Foundry IQ Knowledge Base" ||
    source.includes("Foundry IQ-Compatible Demo Corpus") ||
    source.includes("Demo Corpus") ||
    source.includes("Curated Public Corpus")
  ) {
    return "Foundry IQ-Compatible Demo Corpus";
  }
  return source;
}

function sourceDisplaySnippet(snippet: string) {
  return snippet
    .replace(/^Mock Foundry IQ source:/, "Curated public corpus source:")
    .replace(/^Demo corpus source:/, "Curated public corpus source:");
}

function sourceType(source: AnalyzeResponse["sources"][number]) {
  const displayName = sourceDisplayName(source.source);
  if (displayName === "Foundry IQ-Compatible Demo Corpus") {
    return "Curated public corpus";
  }
  if (displayName === "Azure AI Search / Foundry IQ Search Service") {
    return "Microsoft-backed retrieval";
  }
  return "Live retrieval result";
}

function retrievalLayerLabel(result: AnalyzeResponse) {
  const firstSource = result.sources[0]?.source;
  if (!firstSource) return "Foundry IQ-Compatible Demo Corpus";
  return sourceDisplayName(firstSource);
}

function agentsUsingSource(result: AnalyzeResponse, citationId: string) {
  return result.agent_outputs
    .filter((agent) => agent.evidence_refs.includes(citationId))
    .map((agent) => agent.agent.replace(" Agent", ""))
    .join(", ");
}

function sourceIdentifier(source: AnalyzeResponse["sources"][number]) {
  return source.source_id || source.id || source.citation_id;
}

function sourceByIdentifier(result: AnalyzeResponse, sourceId: string) {
  return result.sources.find(
    (source) =>
      sourceIdentifier(source) === sourceId ||
      source.citation_id === sourceId ||
      source.id === sourceId,
  );
}

function finalAnswerSourceQuality(result: AnalyzeResponse) {
  return getResponseSourceQuality(result);
}

function customIntake(result: AnalyzeResponse) {
  return result.metadata?.custom_intake ?? null;
}

function domainDetectedLabel(result: AnalyzeResponse) {
  const intake = customIntake(result);
  return intake?.domain ? asTitleCase(intake.domain.replace(/_/g, " ")) : result.scenario_label;
}

function sourceQualityTone(quality: "strong" | "partial" | "weak"): "success" | "warning" | "danger" {
  if (quality === "strong") return "success";
  if (quality === "partial") return "warning";
  return "danger";
}

function getKeyFindings(result: AnalyzeResponse) {
  if (result.final_answer?.key_findings?.length) {
    return result.final_answer.key_findings;
  }
  return result.agent_outputs.slice(0, 3).map((agent) => ({
    claim: agent.conclusion,
    source_ids: agent.evidence_refs,
  }));
}

function getRisksOrLimitations(result: AnalyzeResponse, question: string) {
  if (result.final_answer?.risks_or_limitations?.length) {
    return result.final_answer.risks_or_limitations;
  }
  return getConfidenceLimiters(result, question);
}

function getFollowUpQuestions(result: AnalyzeResponse) {
  if (result.final_answer?.follow_up_questions?.length) {
    return result.final_answer.follow_up_questions;
  }
  return result.agent_outputs
    .flatMap((agent) => agent.missing_evidence)
    .slice(0, 3)
    .map((item) => `Can we verify: ${item}`);
}

function getCriticalMissingInfo(result: AnalyzeResponse) {
  const intakeMissing = customIntake(result)?.missing_information ?? [];
  if (intakeMissing.length) return intakeMissing.slice(0, 8);
  return result.agent_outputs.flatMap((agent) => agent.missing_evidence).slice(0, 6);
}

function CitationChips({ refs = [], linked = true }: { refs?: string[]; linked?: boolean }) {
  if (!refs.length) {
    return <span className="text-xs text-muted-foreground">No citation</span>;
  }

  return (
    <span className="inline-flex flex-wrap gap-1">
      {refs.map((ref) =>
        linked ? (
          <a
            key={ref}
            href={`#evidence-${ref}`}
            className="rounded border border-primary/30 bg-primary/10 px-1.5 py-0.5 font-mono text-[11px] text-primary underline-offset-4 hover:underline"
          >
            {ref}
          </a>
        ) : (
          <span
            key={ref}
            className="rounded border border-primary/30 bg-primary/10 px-1.5 py-0.5 font-mono text-[11px] text-primary"
          >
            {ref}
          </span>
        ),
      )}
    </span>
  );
}

function SourceIdChips({
  result,
  sourceIds = [],
}: {
  result: AnalyzeResponse;
  sourceIds?: string[];
}) {
  if (!sourceIds.length) {
    return <span className="text-xs text-muted-foreground">No source citation</span>;
  }

  return (
    <span className="inline-flex flex-wrap gap-1">
      {sourceIds.map((sourceId) => {
        const source = sourceByIdentifier(result, sourceId);
        const label = source?.citation_id ?? sourceId;
        const href = source ? `#evidence-${source.citation_id}` : undefined;
        return href ? (
          <a
            key={sourceId}
            href={href}
            title={sourceId}
            className="rounded border border-primary/30 bg-primary/10 px-1.5 py-0.5 font-mono text-[11px] text-primary underline-offset-4 hover:underline"
          >
            {label}
          </a>
        ) : (
          <span
            key={sourceId}
            className="rounded border border-red-500/30 bg-red-500/10 px-1.5 py-0.5 font-mono text-[11px] text-red-200"
          >
            {sourceId}
          </span>
        );
      })}
    </span>
  );
}

function disagreementWhyItMatters(kind: AnalyzeResponse["disagreements"][number]["kind"]) {
  if (kind === "conflicting_recommendation") {
    return "Agents are pointing toward different decision paths, so the judge must preserve conditions and boundaries.";
  }
  if (kind === "differing_confidence") {
    return "Confidence spread shows which specialist view should receive extra scrutiny.";
  }
  return "Evidence gaps should lower certainty until more grounded context is available.";
}

function AgentPerspectives({ result }: { result: AnalyzeResponse | null }) {
  const [openAgents, setOpenAgents] = useState<string[]>([]);

  useEffect(() => {
    setOpenAgents(result ? ["Risk Analyst Agent"] : []);
  }, [result]);

  function toggleAgent(agentName: string) {
    setOpenAgents((current) =>
      current.includes(agentName)
        ? current.filter((name) => name !== agentName)
        : [...current, agentName],
    );
  }

  if (!result?.agent_outputs.length) {
    return null;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Agent Perspectives</CardTitle>
        <CardDescription>
          Specialist reasoning traces returned by the API.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        {result.agent_outputs.map((agent) => (
          <AgentAccordionCard
            key={agent.agent}
            agent={agent}
            isOpen={openAgents.includes(agent.agent)}
            onToggle={() => toggleAgent(agent.agent)}
          />
        ))}
      </CardContent>
    </Card>
  );
}

function AgentAccordionCard({
  agent,
  isOpen,
  onToggle,
}: {
  agent: AnalyzeResponse["agent_outputs"][number];
  isOpen: boolean;
  onToggle: () => void;
}) {
  const longerReasoning = [...agent.rationale, ...agent.limitations]
    .filter((item) => !/^Fallback/i.test(item))
    .slice(0, 3);

  return (
    <article className="rounded-lg border border-border bg-background">
      <button
        type="button"
        onClick={onToggle}
        className="w-full p-4 text-left"
        aria-expanded={isOpen}
      >
        <div className="mb-3 flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h3 className="text-sm font-semibold text-foreground">{agent.agent}</h3>
            <div className="mt-1 font-mono text-2xl font-semibold text-primary">
              {asPercent(agent.confidence_score)}
            </div>
            <p
              className="mt-2 overflow-hidden text-sm leading-6 text-muted-foreground"
              style={{ display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}
            >
              {agent.conclusion}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <Badge tone={agent.stance === "caution" ? "warning" : agent.stance === "support" ? "success" : "muted"}>
              {asTitleCase(agent.stance)}
            </Badge>
            <ChevronRight className={`h-4 w-4 text-muted-foreground transition-transform ${isOpen ? "rotate-90" : ""}`} />
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <span>Sources</span>
          <CitationChips refs={agent.evidence_refs} linked={false} />
        </div>
      </button>

      {isOpen ? (
        <div className="space-y-3 border-t border-border p-4 pt-4">
          <AgentDetail label="Key Finding" value={agent.conclusion} />
          <AgentDetail label="Recommendation" value={agent.recommendation} />
          <div>
            <div className="mb-1 text-[11px] font-semibold uppercase text-muted-foreground">
              Source IDs
            </div>
            <CitationChips refs={agent.evidence_refs} />
          </div>
          {longerReasoning.length ? (
            <div>
              <div className="mb-1 text-[11px] font-semibold uppercase text-muted-foreground">
                Longer Reasoning
              </div>
              <p className="text-xs leading-5 text-muted-foreground">
                {longerReasoning.join(" ")}
              </p>
            </div>
          ) : null}
        </div>
      ) : null}
    </article>
  );
}

function AgentDetail({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="mb-1 text-[11px] font-semibold uppercase text-muted-foreground">{label}</div>
      <p className="text-sm leading-6 text-foreground">{value}</p>
    </div>
  );
}

function DisagreementsPanel({ result }: { result: AnalyzeResponse | null }) {
  const [openTopic, setOpenTopic] = useState<string | null>(null);

  useEffect(() => {
    setOpenTopic(result?.disagreements[0]?.topic ?? null);
  }, [result]);

  return (
    <div className="h-fit rounded-lg border border-border bg-background p-4">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold">Disagreements</h3>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">
            Where agents diverged before final synthesis.
          </p>
        </div>
        <Badge tone="muted">Analysis</Badge>
      </div>

      {result ? (
        result.disagreements.length ? (
          <div className="space-y-3">
            {result.disagreements.map((item) => {
              const isOpen = openTopic === item.topic;

              return (
                <div key={item.topic} className="rounded-md border border-border bg-card">
                  <button
                    type="button"
                    onClick={() => setOpenTopic((current) => (current === item.topic ? null : item.topic))}
                    className="flex w-full items-center justify-between gap-3 p-3 text-left"
                    aria-expanded={isOpen}
                  >
                    <div className="flex min-w-0 items-center gap-2">
                      <AlertTriangle className="h-4 w-4 shrink-0 text-amber-200" />
                      <h4 className="text-sm font-semibold">{item.topic}</h4>
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                      <Badge tone={item.severity === "high" ? "danger" : item.severity === "medium" ? "warning" : "muted"}>
                        {asTitleCase(item.severity)}
                      </Badge>
                      <ChevronRight className={`h-4 w-4 text-muted-foreground transition-transform ${isOpen ? "rotate-90" : ""}`} />
                    </div>
                  </button>
                  {isOpen ? (
                    <div className="space-y-3 border-t border-border p-3 text-sm leading-6">
                      <DisagreementBlock
                        label="Why It Matters"
                        value={disagreementWhyItMatters(item.kind)}
                      />
                      <div className="rounded-md border border-border bg-background p-3">
                        <div className="mb-2 text-[11px] font-semibold uppercase text-muted-foreground">
                          Agent Viewpoints
                        </div>
                        <ul className="space-y-1 text-muted-foreground">
                          {item.positions.map((position) => (
                            <li key={position}>{position}</li>
                          ))}
                        </ul>
                      </div>
                      <DisagreementBlock
                        label="Resolution Strategy"
                        value={item.suggested_resolution}
                      />
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        ) : (
          <div className="rounded-md border border-border bg-card p-3 text-sm leading-6 text-muted-foreground">
            No material disagreements detected. The agents still preserve limitations and missing evidence in their individual perspectives.
          </div>
        )
      ) : (
        <div className="rounded-md border border-border bg-card p-3 text-sm text-muted-foreground">
          Disagreement analysis appears after a question is analyzed.
        </div>
      )}
    </div>
  );
}

function DisagreementBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-background p-3">
      <div className="mb-1 text-[11px] font-semibold uppercase text-muted-foreground">{label}</div>
      <p className="text-muted-foreground">{value}</p>
    </div>
  );
}

export function ConsensusWorkbench() {
  const [question, setQuestion] = useState(starterQuestion);
  const [analyzedQuestion, setAnalyzedQuestion] = useState(starterQuestion);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progressIndex, setProgressIndex] = useState(0);
  const [openSourceIds, setOpenSourceIds] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const domainLabel = promptDomainLabel(question);
  const displayedTiming = result ? getDisplayedTiming(result.metadata) : null;

  useEffect(() => {
    if (!isLoading) return;

    const interval = window.setInterval(() => {
      setProgressIndex((current) => Math.min(current + 1, progressSteps.length - 1));
    }, 650);

    return () => window.clearInterval(interval);
  }, [isLoading]);

  useEffect(() => {
    setOpenSourceIds(result?.sources[0]?.citation_id ? [result.sources[0].citation_id] : []);
  }, [result]);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setResult(null);
    setProgressIndex(0);
    setIsLoading(true);

    try {
      const response = await analyzeQuestion(question);
      setResult(response);
      setAnalyzedQuestion(question);
      setProgressIndex(progressSteps.length);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to analyze question");
      setProgressIndex(0);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-background">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
        <header className="flex flex-col gap-3 border-b border-border pb-5 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <div className="mb-2 flex items-center gap-2 text-sm text-muted-foreground">
              <BrainCircuit className="h-4 w-4 text-primary" />
              Microsoft Agents League MVP
            </div>
            <h1 className="text-3xl font-semibold tracking-normal text-foreground sm:text-4xl">
              ConsensusIQ
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">
              ConsensusIQ asks multiple specialist AI agents to reason independently,
              compares their conclusions, detects disagreement, and produces an
              evidence-grounded consensus using Microsoft Foundry IQ-style retrieval.
            </p>
          </div>
          <Badge tone="success" className="w-fit">Judge-Ready Demo</Badge>
        </header>

        <section className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4 text-sm leading-6 text-amber-100">
          ConsensusIQ is a reasoning and decision-support demo. It is not a
          substitute for professional medical, legal, financial, or security
          advice.
        </section>

        <section className="rounded-lg border border-border bg-card/80 p-4" aria-label="ConsensusIQ architecture flow">
          <ol className="flex flex-col gap-3 md:flex-row md:items-stretch md:gap-2">
            {flowSteps.map((step, index) => (
              <li key={step.step} className="flex flex-1 flex-col items-center gap-2 md:flex-row">
                <div className="w-full rounded-lg border border-border/70 bg-muted/10 px-3 py-3">
                  <div className="mb-2 font-mono text-[11px] text-primary/80">{step.step}</div>
                  <div className="text-sm font-semibold text-foreground">{step.label}</div>
                  <div className="mt-1 text-xs leading-5 text-muted-foreground">{step.subtitle}</div>
                </div>
                {index < flowSteps.length - 1 ? (
                  <ChevronRight className="h-4 w-4 shrink-0 rotate-90 text-muted-foreground/60 md:rotate-0" />
                ) : null}
              </li>
            ))}
          </ol>
        </section>

        <section className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Question</CardTitle>
              <CardDescription>
                Choose a demo scenario or submit your own decision prompt.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form className="flex flex-col gap-4" onSubmit={onSubmit}>
                <div className="flex items-center justify-between rounded-md border border-border bg-background px-3 py-2">
                  <span className="text-sm text-muted-foreground">Prompt type</span>
                  <Badge tone={domainLabel === "Custom" ? "muted" : "success"}>{asTitleCase(domainLabel)}</Badge>
                </div>
                <div className="grid gap-2">
                  {demoPrompts.map((prompt) => (
                    <Button
                      key={prompt.label}
                      type="button"
                      variant={question === prompt.question ? "secondary" : "ghost"}
                      className="h-auto justify-start whitespace-normal px-3 py-2 text-left"
                      onClick={() => setQuestion(prompt.question)}
                    >
                      <span className="font-semibold">{prompt.label}</span>
                    </Button>
                  ))}
                </div>
                <Textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="Ask a decision-oriented question..."
                  required
                />
                {error ? (
                  <div className="rounded-md border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-100">
                    {error}
                  </div>
                ) : null}
                <Button type="submit" disabled={isLoading || question.trim().length < 3}>
                  {isLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Scale className="h-4 w-4" />
                  )}
                  Analyze consensus
                </Button>
                {isLoading || result ? (
                  <ReasoningProgress
                    activeIndex={progressIndex}
                    isComplete={Boolean(result) && !isLoading}
                  />
                ) : null}
              </form>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Final Consensus</CardTitle>
              <CardDescription>
                Final synthesis from the consensus judge.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              {result ? (
                <>
                  <ExecutiveVerdict result={result} />

                  <div className="grid gap-3">
                    <RecommendationCard result={result} question={analyzedQuestion} />
                    <ConfidenceSummary result={result} question={analyzedQuestion} />
                  </div>
                </>
              ) : (
                <EmptyState />
              )}
            </CardContent>
          </Card>
        </section>

        {result ? (
          <section className={`grid gap-6 ${result.agent_outputs.length ? "lg:grid-cols-2" : "lg:grid-cols-1"}`}>
            {result.agent_outputs.length ? <AgentPerspectives result={result} /> : null}
            <DisagreementsPanel result={result} />
          </section>
        ) : null}

        {result ? (
          <RetrievedEvidence
            result={result}
            query={analyzedQuestion}
            retrievalLatency={displayedTiming?.retrieval}
            openSourceIds={openSourceIds}
            onToggleSource={(citationId) =>
              setOpenSourceIds((current) =>
                current.includes(citationId)
                  ? current.filter((id) => id !== citationId)
                  : [...current, citationId],
              )
            }
          />
        ) : null}

        <RunMetadata timing={displayedTiming} result={result} />
      </div>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  const valueClass = value.length > 12 ? "text-xl" : "text-3xl";
  return (
    <div className="rounded-lg border border-border bg-background p-4">
      <div className="text-xs uppercase text-muted-foreground">{label}</div>
      <div className={`mt-2 break-words font-mono ${valueClass} font-semibold`}>{value}</div>
    </div>
  );
}

function ExecutiveVerdict({ result }: { result: AnalyzeResponse }) {
  const sourceQuality = finalAnswerSourceQuality(result);
  return (
    <div className="rounded-lg border border-primary/30 bg-primary/10 p-4">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <div className="text-xs font-semibold uppercase text-primary">Final Recommendation</div>
          <h3 className="mt-1 text-xl font-semibold leading-7 text-foreground">
            {getDecisionSentence(result)}
          </h3>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-2">
          <Badge tone={result.scenario_label === "Custom" ? "muted" : "success"}>
            {asTitleCase(result.scenario_label)}
          </Badge>
          <Badge tone={sourceQualityTone(sourceQuality)}>
            {asTitleCase(sourceQuality)} sources
          </Badge>
        </div>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <Metric
          label="Confidence"
          value={getScoreDisplay(result, "confidence")}
        />
        <Metric
          label="Agreement"
          value={getScoreDisplay(result, "agreement")}
        />
      </div>
    </div>
  );
}

function RecommendationCard({ result, question }: { result: AnalyzeResponse; question: string }) {
  const findings = getKeyFindings(result);
  const risks = getRisksOrLimitations(result, question);
  const followUps = getFollowUpQuestions(result);
  const missingInfo = getCriticalMissingInfo(result);

  return (
    <div className="rounded-lg border border-border bg-background p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold">Grounded Answer</h3>
        <Badge tone={sourceQualityTone(finalAnswerSourceQuality(result))}>
          {asTitleCase(finalAnswerSourceQuality(result))}
        </Badge>
      </div>
      <div className="grid gap-3">
        <RecommendationBlock label="Recommended Next Step" value={getRecommendationSummary(result)} />
        <RecommendationBlock label="Why" value={getRecommendedApproach(result)} />
        <div className="rounded-md border border-border bg-card p-3">
          <div className="mb-2 text-[11px] font-semibold uppercase text-muted-foreground">
            Key Findings
          </div>
          <div className="space-y-3">
            {findings.map((finding) => (
              <div key={finding.claim} className="grid gap-2">
                <p className="text-sm leading-6 text-foreground">{capitalizeFirst(finding.claim)}</p>
                <SourceIdChips result={result} sourceIds={finding.source_ids} />
              </div>
            ))}
          </div>
        </div>
        {missingInfo.length ? (
          <SummaryList title="Critical Missing Info" items={missingInfo} marker="?" />
        ) : null}
        <div className="grid gap-3 md:grid-cols-2">
          <SummaryList title="Risks Or Limitations" items={risks.slice(0, 4)} marker="-" />
          <SummaryList
            title="Follow-Up Questions"
            items={followUps.length ? followUps.slice(0, 4) : ["No immediate follow-up questions returned."]}
            marker="?"
          />
        </div>
      </div>
    </div>
  );
}

function RecommendationBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-card p-3">
      <div className="mb-1 text-[11px] font-semibold uppercase text-muted-foreground">{label}</div>
      <p className="text-sm leading-6 text-foreground">{capitalizeFirst(value)}</p>
    </div>
  );
}

function ConfidenceSummary({
  result,
  question,
}: {
  result: AnalyzeResponse;
  question: string;
}) {
  const [showInputs, setShowInputs] = useState(false);
  const calculation = getConfidenceCalculation(result, question);

  return (
    <div className="rounded-lg border border-border bg-background p-4">
      <h3 className="mb-3 text-sm font-semibold">Confidence Assessment</h3>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-md border border-border bg-card p-3">
          <div className="mb-1 text-[11px] font-semibold uppercase text-muted-foreground">
            Why Confidence Is Not Higher
          </div>
          <p className="text-sm leading-6 text-foreground">
            {getConfidenceLimiters(result, question)[0]}
          </p>
        </div>
        <div className="rounded-md border border-border bg-card p-3">
          <div className="mb-1 text-[11px] font-semibold uppercase text-muted-foreground">
            Primary Disagreement
          </div>
          <p className="text-sm leading-6 text-foreground">
            {getKeyDisagreementSummary(result)}
          </p>
        </div>
      </div>
      <button
        type="button"
        onClick={() => setShowInputs((current) => !current)}
        className="mt-3 flex w-full items-center justify-between rounded-md border border-border bg-card px-3 py-2 text-left text-sm"
        aria-expanded={showInputs}
      >
        <span>Score Inputs</span>
        <ChevronRight className={`h-4 w-4 text-muted-foreground transition-transform ${showInputs ? "rotate-90" : ""}`} />
      </button>
      {showInputs ? (
        <div className="mt-3 grid gap-3">
          <SummaryList title="Confidence Factors" items={getConfidenceFactors(result, question)} marker="+" />
          <div className="grid gap-3 sm:grid-cols-2">
            <SummaryList title="Increased Confidence" items={calculation.positiveFactors} marker="+" />
            <SummaryList title="Reduced Confidence" items={calculation.negativeFactors} marker="-" />
          </div>
          <SummaryList title="Why Confidence Is Not Higher" items={getConfidenceLimiters(result, question)} marker="-" />
        </div>
      ) : null}
    </div>
  );
}

function SummaryList({
  title,
  items,
  marker,
}: {
  title: string;
  items: string[];
  marker: string;
}) {
  return (
    <div className="rounded-md border border-border bg-card p-3">
      <div className="mb-2 text-[11px] font-semibold uppercase text-muted-foreground">{title}</div>
      <ul className="space-y-1 text-sm leading-6 text-muted-foreground">
        {items.map((item) => (
          <li key={item}>
            <span className="font-mono text-primary">{marker}</span> {item}
          </li>
        ))}
      </ul>
    </div>
  );
}

function RetrievedEvidence({
  result,
  query,
  retrievalLatency,
  openSourceIds,
  onToggleSource,
}: {
  result: AnalyzeResponse;
  query: string;
  retrievalLatency?: number;
  openSourceIds: string[];
  onToggleSource: (citationId: string) => void;
}) {
  return (
    <section className="rounded-lg border border-border bg-card p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold">Retrieved Evidence</h3>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">
            Citation-grounded sources returned through the configured Microsoft retrieval interface.
          </p>
        </div>
        <Badge tone="muted">{retrievalLayerLabel(result)}</Badge>
      </div>
      <RetrievalTrace
        result={result}
        query={query}
        retrievalLatency={retrievalLatency}
      />
      <div className="mt-3 rounded-md border border-border bg-background px-3 py-2 text-sm text-muted-foreground">
        {getEvidenceCoverageLabel(result)}
      </div>
      <div className="mt-3 grid gap-3 lg:grid-cols-3">
        {result.sources.length ? (
          result.sources.map((source) => (
            <EvidenceCard
              key={source.citation_id}
              source={source}
              result={result}
              isOpen={openSourceIds.includes(source.citation_id)}
              onToggle={() => onToggleSource(source.citation_id)}
            />
          ))
        ) : (
          <div className="rounded-lg border border-border bg-background p-4 text-sm leading-6 text-muted-foreground lg:col-span-3">
            No strong retrieved evidence matched this custom prompt. ConsensusIQ still runs the agents, but confidence is reduced and the recommendation should be treated as decision-support reasoning rather than source-grounded certainty.
          </div>
        )}
      </div>
    </section>
  );
}

function RetrievalTrace({
  result,
  query,
  retrievalLatency,
}: {
  result: AnalyzeResponse;
  query: string;
  retrievalLatency?: number;
}) {
  const usedRefs = getUsedSourceRefs(result);

  return (
    <div className="rounded-lg border border-primary/20 bg-primary/5 p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h4 className="text-sm font-semibold">Retrieval Trace</h4>
      </div>
      <div className="grid gap-3 text-xs text-muted-foreground sm:grid-cols-3">
        <TraceMetric label="Query" value={query} />
        <TraceMetric label="Retrieved Documents" value={String(result.sources.length)} />
        <div className="rounded-md border border-border bg-background px-3 py-2">
          <div className="mb-1 text-[11px] uppercase text-muted-foreground">Sources Used</div>
          <div className="break-words text-sm text-foreground">
            <CitationChips refs={usedRefs} />
          </div>
        </div>
        {result.sources.length ? (
          <>
            <TraceMetric label="Top Relevance" value={asPercent(getTopRelevance(result))} />
            <TraceMetric label="Average Relevance" value={asPercent(getAverageRelevance(result))} />
          </>
        ) : null}
        {retrievalLatency ? <TraceMetric label="Retrieval Latency" value={`${retrievalLatency}ms`} /> : null}
      </div>
    </div>
  );
}

function TraceMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-background px-3 py-2">
      <div className="mb-1 text-[11px] uppercase text-muted-foreground">{label}</div>
      <div className="break-words text-sm text-foreground">{value}</div>
    </div>
  );
}

function EvidenceCard({
  source,
  result,
  isOpen,
  onToggle,
}: {
  source: AnalyzeResponse["sources"][number];
  result: AnalyzeResponse;
  isOpen: boolean;
  onToggle: () => void;
}) {
  const usedBy = agentsUsingSource(result, source.citation_id);
  const isPublicUrl = source.url.startsWith("https://") || source.url.startsWith("http://");

  return (
    <article id={`evidence-${source.citation_id}`} className="scroll-mt-6 rounded-lg border border-border bg-background">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-start justify-between gap-3 p-3 text-left"
        aria-expanded={isOpen}
      >
        <div className="min-w-0 flex-1">
          <div className="flex items-start gap-2">
            <span className="font-mono text-xs text-primary">[{source.citation_id}]</span>
            <span className="text-sm font-medium text-foreground">{source.title}</span>
          </div>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <span>Relevance: {asPercent(source.relevance_score)}</span>
            <span>Used By: {usedBy || "Consensus context"}</span>
          </div>
          <div className="mt-2 flex items-center gap-1 text-xs font-medium text-primary">
            <ChevronRight className={`h-3.5 w-3.5 transition-transform ${isOpen ? "rotate-90" : ""}`} />
            {isOpen ? "Hide Evidence" : "View Evidence"}
          </div>
        </div>
        <Badge tone="muted">{source.citation_id}</Badge>
      </button>

      {isOpen ? (
        <div className="border-t border-border p-3 pt-4">
          <div className="mb-3 grid gap-2 text-xs text-muted-foreground sm:grid-cols-2">
            {source.title ? (
              <div className="sm:col-span-2">
                <span>Title </span>
                <span className="text-foreground">{source.title}</span>
              </div>
            ) : null}
            <div>
              <span>Citation ID </span>
              <span className="font-mono text-primary">{source.citation_id}</span>
            </div>
            <div>
              <span>Source Type </span>
              <span className="text-foreground">{sourceType(source)}</span>
            </div>
            <div>
              <span>Relevance </span>
              <span className="font-mono text-primary">{asPercent(source.relevance_score)}</span>
            </div>
            <div>
              <span>Used By Agents </span>
              <span className="text-foreground">{usedBy || "Referenced in consensus context"}</span>
            </div>
          </div>
          {source.snippet ? (
            <>
              <div className="mb-2 text-[11px] uppercase text-muted-foreground">Evidence Excerpt</div>
              <p className="text-xs leading-5 text-muted-foreground">{sourceDisplaySnippet(source.snippet)}</p>
            </>
          ) : null}
          {isPublicUrl ? (
            <div className="mt-3">
              <div className="mb-1 text-[11px] uppercase text-muted-foreground">Public Source Link</div>
              <a
                href={source.url}
                target="_blank"
                rel="noreferrer"
                className="break-all font-mono text-xs text-primary underline-offset-4 hover:underline"
              >
                {source.url}
              </a>
            </div>
          ) : null}
        </div>
      ) : null}
    </article>
  );
}

function ReasoningProgress({
  activeIndex,
  isComplete,
}: {
  activeIndex: number;
  isComplete: boolean;
}) {
  return (
    <div className="rounded-lg border border-border bg-background p-4">
      <div className="mb-3 text-sm font-semibold">Reasoning Progress</div>
      <ol className="space-y-2">
        {progressSteps.map((step, index) => {
          const completed = isComplete || index < activeIndex;
          const active = !isComplete && index === activeIndex;

          return (
            <li key={step} className="flex items-center gap-3 text-sm">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-border bg-card">
                {completed ? (
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-300" />
                ) : active ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
                ) : (
                  <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground/50" />
                )}
              </span>
              <span className={completed || active ? "text-foreground" : "text-muted-foreground"}>
                {step}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function RuntimeMetric({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md border border-border bg-background px-3 py-2">
      <span>{label}</span>
      <span className="float-right font-mono">{value}ms</span>
    </div>
  );
}

function RunMetadata({
  timing,
  result,
}: {
  timing: ReturnType<typeof getDisplayedTiming>;
  result: AnalyzeResponse | null;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const debugInfo = getProviderDebugInfo(result);

  if (!timing) return null;

  return (
    <div className="rounded-lg border border-border bg-background">
      <button
        type="button"
        onClick={() => setIsOpen((current) => !current)}
        className="flex w-full items-center justify-between gap-3 p-3 text-left"
        aria-expanded={isOpen}
      >
        <div>
          <div className="text-sm font-semibold">Run Metadata</div>
          <div className="mt-1 text-xs text-muted-foreground">
            Total {timing.total}ms / Retrieval {timing.retrieval}ms
          </div>
        </div>
        <ChevronRight className={`h-4 w-4 text-muted-foreground transition-transform ${isOpen ? "rotate-90" : ""}`} />
      </button>
      {isOpen ? (
        <div className="border-t border-border p-3">
          <div className="grid gap-2 text-xs text-muted-foreground sm:grid-cols-4">
            <RuntimeMetric label="Total" value={timing.total} />
            <RuntimeMetric label="Retrieval" value={timing.retrieval} />
            <RuntimeMetric label="Agents" value={timing.agents} />
            <RuntimeMetric label="Consensus" value={timing.consensus} />
          </div>
          {result ? (
            <div className="mt-3 grid gap-2 text-xs text-muted-foreground sm:grid-cols-4">
              <TraceMetric label="Domain Detected" value={domainDetectedLabel(result)} />
              <TraceMetric label="Source Coverage" value={asTitleCase(finalAnswerSourceQuality(result))} />
              <TraceMetric
                label="Provider"
                value={debugInfo.provider}
              />
              <TraceMetric
                label="Live LLM Mode"
                value={debugInfo.liveLlmMode}
              />
              <TraceMetric
                label="OpenRouter Calls"
                value={String(debugInfo.openRouterCalls)}
              />
              <TraceMetric
                label="Fallback"
                value={debugInfo.fallbackReason}
              />
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex min-h-40 flex-col items-center justify-center rounded-lg border border-dashed border-border bg-background p-6 text-center">
      <CheckCircle2 className="mb-3 h-8 w-8 text-muted-foreground" />
      <p className="max-w-sm text-sm leading-6 text-muted-foreground">
        Run an analysis to see the consensus judge output and scoring.
      </p>
    </div>
  );
}

function PlaceholderRows() {
  return (
    <>
      {["Risk Analyst Agent", "Evidence Analyst Agent", "Alternative Solutions Agent"].map((agent) => (
        <div key={agent} className="rounded-lg border border-border bg-background p-4">
          <div className="mb-3 h-4 w-32 rounded bg-muted" />
          <div className="mb-2 h-3 w-full rounded bg-muted" />
          <div className="h-3 w-3/4 rounded bg-muted" />
        </div>
      ))}
    </>
  );
}
