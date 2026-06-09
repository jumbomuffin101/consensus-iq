"use client";

import { FormEvent, useState } from "react";
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
  "Question",
  "Foundry IQ Retrieval",
  "Specialist Agents",
  "Disagreement Analysis",
  "Consensus Judge",
];

function asPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function promptDomainLabel(question: string) {
  const preset = demoPrompts.find((prompt) => prompt.question === question);
  if (preset?.label === "Clinical Reasoning") return "Clinical";
  if (preset?.label === "Enterprise Risk") return "Enterprise";
  if (preset?.label === "Research Evaluation") return "Research";
  return "Custom";
}

export function ConsensusWorkbench() {
  const [question, setQuestion] = useState(starterQuestion);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const domainLabel = promptDomainLabel(question);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      setResult(await analyzeQuestion(question));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to analyze question");
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
          <Badge tone="success" className="w-fit">Judge-ready demo</Badge>
        </header>

        <section className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4 text-sm leading-6 text-amber-100">
          ConsensusIQ is a reasoning and decision-support demo. It is not a
          substitute for professional medical, legal, financial, or security
          advice.
        </section>

        <section className="rounded-lg border border-border bg-card p-4">
          <div className="grid gap-2 md:grid-cols-5">
            {flowSteps.map((step, index) => (
              <div key={step} className="flex items-center gap-2">
                <div className="flex min-h-12 flex-1 items-center rounded-md border border-border bg-background px-3 text-sm">
                  {step}
                </div>
                {index < flowSteps.length - 1 ? (
                  <ChevronRight className="hidden h-4 w-4 text-muted-foreground md:block" />
                ) : null}
              </div>
            ))}
          </div>
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
                  <Badge tone={domainLabel === "Custom" ? "muted" : "success"}>{domainLabel}</Badge>
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
                  <p className="text-sm leading-6 text-foreground">{result.consensus}</p>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <Metric label="Confidence Score" value={asPercent(result.confidence_score)} />
                    <Metric label="Agreement Score" value={asPercent(result.agreement_score)} />
                  </div>
                  {result.metadata ? (
                    <div className="grid gap-2 text-xs text-muted-foreground sm:grid-cols-4">
                      <RuntimeMetric label="Total" value={result.metadata.execution_time_ms} />
                      <RuntimeMetric label="Retrieval" value={result.metadata.retrieval_time_ms} />
                      <RuntimeMetric label="Agents" value={result.metadata.agent_time_ms} />
                      <RuntimeMetric label="Consensus" value={result.metadata.consensus_time_ms} />
                    </div>
                  ) : null}
                  <div className="rounded-lg border border-border bg-background p-4">
                    <h3 className="mb-2 text-sm font-semibold">Reasoning Summary</h3>
                    <p className="text-sm leading-6 text-muted-foreground">
                      {result.reasoning_summary}
                    </p>
                  </div>
                  <div className="space-y-3">
                    <h3 className="text-sm font-semibold">Sources</h3>
                    <div className="space-y-3">
                      {result.sources.map((source) => (
                        <div key={source.citation_id} className="rounded-lg border border-border bg-background p-3">
                          <div className="mb-1 flex items-center justify-between gap-3">
                            <span className="font-mono text-xs text-primary">{source.citation_id}</span>
                            <span className="text-xs text-muted-foreground">
                              {asPercent(source.relevance_score)}
                            </span>
                          </div>
                          <div className="text-sm font-medium">{source.title}</div>
                          <div className="text-xs text-muted-foreground">{source.source}</div>
                          <p className="mt-2 text-xs leading-5 text-muted-foreground">{source.snippet}</p>
                          {source.url ? (
                            <a className="mt-2 block text-xs text-primary hover:underline" href={source.url}>
                              {source.url}
                            </a>
                          ) : null}
                        </div>
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
                      {agent.stance}
                    </Badge>
                  </div>
                  <p className="mb-3 text-xs leading-5 text-muted-foreground">{agent.role}</p>
                  <p className="text-sm leading-6">{agent.conclusion}</p>
                  <p className="mt-3 text-xs leading-5 text-muted-foreground">
                    Recommendation: {agent.recommendation}
                  </p>
                  <div className="mt-4 flex items-center justify-between text-xs text-muted-foreground">
                    <span>Confidence {asPercent(agent.confidence_score)}</span>
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
              {result?.disagreements.map((item) => (
                <div key={item.topic} className="rounded-lg border border-border bg-background p-4">
                  <div className="mb-3 flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="h-4 w-4 text-amber-200" />
                      <h3 className="text-sm font-semibold">{item.topic}</h3>
                    </div>
                    <Badge tone={item.severity === "high" ? "danger" : item.severity === "medium" ? "warning" : "muted"}>
                      {item.severity}
                    </Badge>
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
              )) ?? (
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
