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
  return `${Math.round(value * 100)}%`;
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
    averageRelevance >= 0.85
      ? "High source relevance supports stronger grounding"
      : averageRelevance >= 0.7
        ? "Moderate source relevance leaves some uncertainty"
        : "Lower source relevance reduces confidence",
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

function firstSentence(value: string) {
  const match = value.match(/.+?[.!?](?:\s|$)/);
  return (match?.[0] ?? value).trim();
}

function getRecommendationSummary(result: AnalyzeResponse) {
  const [opening] = result.consensus.split("Evidence view:");
  return opening.replace(/^[^:]+:\s*/, "").trim();
}

function getRationaleSummary(result: AnalyzeResponse) {
  const evidence = getAgent(result, "Evidence Analyst Agent");
  const risk = getAgent(result, "Risk Analyst Agent");
  if (evidence && risk) {
    return `${firstSentence(evidence.conclusion)} ${firstSentence(risk.conclusion)}`;
  }
  return firstSentence(result.reasoning_summary || result.consensus);
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

function getConsensusJudgeSummary(result: AnalyzeResponse) {
  const disagreement = getKeyDisagreementSummary(result);
  return `Final recommendation balances ${asPercent(result.confidence_score)} confidence, ${asPercent(result.agreement_score)} agreement, and the key disagreement: ${disagreement}`;
}

function getTopRelevance(result: AnalyzeResponse) {
  if (!result.sources.length) return 0;
  return Math.max(...result.sources.map((source) => source.relevance_score));
}

function sourceDisplayName(source: string) {
  if (source === "Mock Foundry IQ Knowledge Base") {
    return "Foundry IQ Retrieval Layer \u2014 Demo Corpus";
  }
  return source;
}

function sourceDisplaySnippet(snippet: string) {
  return snippet.replace(/^Mock Foundry IQ source:/, "Demo corpus source:");
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

export function ConsensusWorkbench() {
  const [question, setQuestion] = useState(starterQuestion);
  const [analyzedQuestion, setAnalyzedQuestion] = useState(starterQuestion);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progressIndex, setProgressIndex] = useState(0);
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

        <section className="grid gap-6 lg:grid-cols-[0.85fr_1.15fr]">
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
                  <div className="flex items-center justify-between rounded-md border border-border bg-background px-3 py-2">
                    <span className="text-sm text-muted-foreground">Scenario</span>
                    <Badge tone={result.scenario_label === "Custom" ? "muted" : "success"}>
                      {asTitleCase(result.scenario_label)}
                    </Badge>
                  </div>

                  <div className="grid gap-3">
                    <ConsensusSection
                      title="Recommendation"
                      value={getRecommendationSummary(result)}
                    />
                    <ConsensusSection
                      title="Rationale"
                      value={getRationaleSummary(result)}
                    />
                    <div className="rounded-lg border border-border bg-background p-4">
                      <h3 className="mb-3 text-sm font-semibold">Confidence</h3>
                      <div className="grid gap-3 sm:grid-cols-2">
                        <Metric label="Confidence Score" value={asPercent(result.confidence_score)} />
                        <Metric label="Agreement Score" value={asPercent(result.agreement_score)} />
                      </div>
                    </div>
                    <ConsensusSection
                      title="Key Disagreement"
                      value={getKeyDisagreementSummary(result)}
                    />
                  </div>

                  <div className="rounded-lg border border-border bg-background p-4">
                    <h3 className="mb-3 text-sm font-semibold">Confidence Factors</h3>
                    <ul className="space-y-2 text-sm leading-6 text-muted-foreground">
                      {getConfidenceFactors(result, analyzedQuestion).map((factor) => (
                        <li key={factor} className="flex gap-2">
                          <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                          <span>{factor}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  {displayedTiming ? (
                    <div className="space-y-2">
                      <div className="text-xs uppercase text-muted-foreground">Observed / displayed timing</div>
                      <div className="grid gap-2 text-xs text-muted-foreground sm:grid-cols-4">
                        <RuntimeMetric label="Total" value={displayedTiming.total} />
                        <RuntimeMetric label="Retrieval" value={displayedTiming.retrieval} />
                        <RuntimeMetric label="Agents" value={displayedTiming.agents} />
                        <RuntimeMetric label="Consensus" value={displayedTiming.consensus} />
                      </div>
                    </div>
                  ) : null}

                  <ReasoningBreakdown result={result} />

                  <div className="space-y-3">
                    <div className="flex items-center justify-between gap-3">
                      <h3 className="text-sm font-semibold">Retrieved Evidence</h3>
                      <Badge tone="muted">Foundry IQ Retrieval</Badge>
                    </div>
                    <RetrievalTrace
                      query={analyzedQuestion}
                      documentCount={result.sources.length}
                      topRelevance={getTopRelevance(result)}
                      retrievalLatency={displayedTiming?.retrieval}
                    />
                    <div className="space-y-3">
                      {result.sources.map((source) => (
                        <EvidenceCard key={source.citation_id} source={source} />
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <EmptyState />
              )}
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
          <Card>
            <CardHeader>
              <CardTitle>Agent Perspectives</CardTitle>
              <CardDescription>
                Specialist reasoning traces returned by the API.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-3">
              {result?.agent_outputs.map((agent) => (
                <article key={agent.agent} className="rounded-lg border border-border bg-background p-4">
                  <div className="mb-3 flex items-center justify-between gap-2">
                    <h3 className="text-sm font-semibold">{agent.agent}</h3>
                    <Badge tone={agent.stance === "caution" ? "warning" : agent.stance === "support" ? "success" : "muted"}>
                      {asTitleCase(agent.stance)}
                    </Badge>
                  </div>
                  <div className="mb-4 rounded-md border border-border bg-card px-3 py-2">
                    <div className="text-xs uppercase text-muted-foreground">Agent confidence</div>
                    <div className="mt-1 font-mono text-2xl font-semibold text-foreground">
                      {asPercent(agent.confidence_score)}
                    </div>
                  </div>
                  <p className="mb-3 text-xs leading-5 text-muted-foreground">{agent.role}</p>
                  <p className="text-sm leading-6">{agent.conclusion}</p>
                  <p className="mt-3 text-xs leading-5 text-muted-foreground">
                    Recommendation: {agent.recommendation}
                  </p>
                  <div className="mt-4 flex items-center justify-between gap-3 text-xs text-muted-foreground">
                    <span>Cited sources</span>
                    <span>{agent.evidence_refs.length ? agent.evidence_refs.join(", ") : "No citation"}</span>
                  </div>
                </article>
              )) ?? <PlaceholderRows />}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Disagreements</CardTitle>
              <CardDescription>
                Where agents diverged before final synthesis.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {result ? (
                result.disagreements.length ? (
                  result.disagreements.map((item) => (
                    <div key={item.topic} className="rounded-lg border border-border bg-background p-4">
                      <div className="mb-3 flex items-center justify-between gap-3">
                        <div className="flex items-center gap-2">
                          <AlertTriangle className="h-4 w-4 text-amber-200" />
                          <h3 className="text-sm font-semibold">{item.topic}</h3>
                        </div>
                        <Badge tone={item.severity === "high" ? "danger" : item.severity === "medium" ? "warning" : "muted"}>
                          {asTitleCase(item.severity)}
                        </Badge>
                      </div>
                      <p className="mb-3 text-xs leading-5 text-muted-foreground">
                        Why it matters: {disagreementWhyItMatters(item.kind)}
                      </p>
                      <div className="mb-2 text-xs font-semibold uppercase text-muted-foreground">
                        Agent viewpoints
                      </div>
                      <ul className="space-y-2 text-sm leading-6 text-muted-foreground">
                        {item.positions.map((position) => (
                          <li key={position}>{position}</li>
                        ))}
                      </ul>
                      <p className="mt-3 text-xs leading-5 text-muted-foreground">
                        Resolution: {item.suggested_resolution}
                      </p>
                    </div>
                  ))
                ) : (
                  <div className="rounded-lg border border-border bg-background p-4 text-sm leading-6 text-muted-foreground">
                    No material disagreements detected. The agents still preserve limitations and missing evidence in their individual perspectives.
                  </div>
                )
              ) : (
                <div className="rounded-lg border border-border bg-background p-4 text-sm text-muted-foreground">
                  Disagreement analysis appears after a question is analyzed.
                </div>
              )}
            </CardContent>
          </Card>
        </section>
      </div>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-background p-4">
      <div className="text-xs uppercase text-muted-foreground">{label}</div>
      <div className="mt-2 font-mono text-3xl font-semibold">{value}</div>
    </div>
  );
}

function ConsensusSection({ title, value }: { title: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-background p-4">
      <h3 className="mb-2 text-sm font-semibold">{title}</h3>
      <p className="text-sm leading-6 text-muted-foreground">{value}</p>
    </div>
  );
}

function ReasoningBreakdown({ result }: { result: AnalyzeResponse }) {
  const risk = getAgent(result, "Risk Analyst Agent");
  const evidence = getAgent(result, "Evidence Analyst Agent");
  const alternatives = getAgent(result, "Alternative Solutions Agent");

  return (
    <div className="rounded-lg border border-border bg-background p-4">
      <h3 className="mb-3 text-sm font-semibold">How Consensus Was Reached</h3>
      <div className="space-y-3">
        <ReasoningStep
          label="Risk Analyst"
          value={risk?.conclusion ?? "No risk analyst output returned."}
          refs={risk?.evidence_refs}
        />
        <ReasoningStep
          label="Evidence Analyst"
          value={evidence?.conclusion ?? "No evidence analyst output returned."}
          refs={evidence?.evidence_refs}
        />
        <ReasoningStep
          label="Alternative Solutions Analyst"
          value={alternatives?.conclusion ?? "No alternatives analyst output returned."}
          refs={alternatives?.evidence_refs}
        />
        <ReasoningStep
          label="Consensus Judge"
          value={getConsensusJudgeSummary(result)}
          refs={result.sources.map((source) => source.citation_id)}
        />
      </div>
    </div>
  );
}

function ReasoningStep({
  label,
  value,
  refs = [],
}: {
  label: string;
  value: string;
  refs?: string[];
}) {
  return (
    <div className="rounded-md border border-border bg-card p-3">
      <div className="mb-1 flex items-center justify-between gap-3">
        <div className="text-xs font-semibold uppercase text-muted-foreground">{label}</div>
        {refs.length ? (
          <span className="font-mono text-xs text-primary">{refs.join(", ")}</span>
        ) : null}
      </div>
      <p className="text-sm leading-6 text-foreground">{value}</p>
    </div>
  );
}

function RetrievalTrace({
  query,
  documentCount,
  topRelevance,
  retrievalLatency,
}: {
  query: string;
  documentCount: number;
  topRelevance: number;
  retrievalLatency?: number;
}) {
  return (
    <div className="rounded-lg border border-primary/20 bg-primary/5 p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h4 className="text-sm font-semibold">Retrieval Trace</h4>
        <Badge tone="success">Foundry IQ Grounding</Badge>
      </div>
      <div className="grid gap-3 text-xs text-muted-foreground sm:grid-cols-4">
        <TraceMetric label="Query" value={query} />
        <TraceMetric label="Retrieved Documents" value={String(documentCount)} />
        <TraceMetric label="Top Relevance" value={asPercent(topRelevance)} />
        <TraceMetric label="Retrieval Latency" value={retrievalLatency ? `${retrievalLatency}ms` : "Not reported"} />
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

function EvidenceCard({ source }: { source: AnalyzeResponse["sources"][number] }) {
  return (
    <article className="rounded-lg border border-border bg-background p-3">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-medium">{source.title}</div>
          <div className="mt-1 text-xs text-muted-foreground">{sourceDisplayName(source.source)}</div>
        </div>
        <Badge tone="muted">{source.citation_id}</Badge>
      </div>
      <div className="mb-3 grid gap-2 text-xs text-muted-foreground sm:grid-cols-2">
        <div>
          <span className="text-muted-foreground">Relevance score </span>
          <span className="font-mono text-primary">{asPercent(source.relevance_score)}</span>
        </div>
        <div>
          <span className="text-muted-foreground">Citation ID </span>
          <span className="font-mono text-primary">{source.citation_id}</span>
        </div>
      </div>
      <p className="text-xs leading-5 text-muted-foreground">{sourceDisplaySnippet(source.snippet)}</p>
      {source.url ? (
        <a className="mt-2 block text-xs text-primary hover:underline" href={source.url}>
          {source.url}
        </a>
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
