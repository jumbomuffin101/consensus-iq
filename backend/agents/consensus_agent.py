from agents.prompting import agent_outputs_payload, disagreements_payload
from llm.base import BaseLLMProvider
from llm.mock import MockLLMProvider
from models.reasoning import ConsensusJudgment, ReasoningState
from reasoning.disagreement import DisagreementDetector
from reasoning.domain import (
    bounded_score,
    build_domain_profile,
    missing_information_load,
    prompt_injection_risk,
)


class ConsensusJudgeNode:
    """Synthesizes agent outputs into a transparent final recommendation.

    Future Azure OpenAI integration point: use the model only after deterministic
    disagreement detection has produced auditable inputs for the final judgment.
    """

    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
        detector: DisagreementDetector | None = None,
    ) -> None:
        self.provider = provider or MockLLMProvider()
        self.detector = detector or DisagreementDetector()

    def __call__(self, state: ReasoningState) -> ReasoningState:
        disagreements = self.detector.detect(state.agent_outputs)
        agreement_score = self.detector.calculate_agreement_score(state.agent_outputs)
        profile = build_domain_profile(state)
        missing_penalty = missing_information_load(state)
        injection_penalty = prompt_injection_risk(state.question)
        avg_confidence = (
            sum(output.confidence_score for output in state.agent_outputs)
            / len(state.agent_outputs)
            if state.agent_outputs
            else 0.0
        )
        raw_confidence = (
            (avg_confidence * 0.38)
            + (agreement_score * 0.3)
            + (profile.evidence_quality * 0.24)
            + (profile.source_certainty * 0.1)
            - (profile.ambiguity * 0.18)
            - (profile.risk_level * 0.08)
            - (0.12 if not state.retrieved_context else 0)
            - (0.06 if state.retrieved_context and profile.evidence_quality < 0.55 else 0)
            - missing_penalty
            - injection_penalty
        )
        if not state.retrieved_context:
            raw_confidence = max(raw_confidence, 0.18)
        confidence_score = bounded_score(raw_confidence)

        state_with_disagreements = state.copy(update={"disagreements": disagreements})
        fallback_judgment = ConsensusJudgment(
            consensus=self._build_consensus(state, profile.domain),
            confidence_score=confidence_score,
            agreement_score=agreement_score,
            reasoning_summary=self._build_reasoning_summary(
                state_with_disagreements, disagreements, profile.domain
            ),
        )
        payload = self.provider.complete_json(
            system_prompt=(
                "You are the ConsensusIQ Consensus Judge. Synthesize independent "
                "specialist outputs and the disagreement report into a transparent, "
                "decision-ready recommendation. Preserve uncertainty. Use retrieved "
                "context citations already present in evidence_refs whenever making "
                "evidence-based claims."
            ),
            user_prompt=(
                f"Question: {state.question}\n\n"
                f"Retrieved context:\n{[item.dict() for item in state.retrieved_context]}\n\n"
                f"Agent outputs:\n{agent_outputs_payload(state)}\n\n"
                f"Disagreements:\n{disagreements_payload(state_with_disagreements)}\n\n"
                "Return JSON with keys: consensus, confidence_score, "
                "agreement_score, reasoning_summary. Scores must be numbers "
                "between 0 and 1."
            ),
            fallback=fallback_judgment.dict(),
        )
        try:
            judgment = ConsensusJudgment.parse_obj(payload)
        except Exception:
            judgment = fallback_judgment

        return state.copy(
            update={
                "disagreements": disagreements,
                "scenario_label": profile.scenario_label,
                "consensus": judgment.consensus,
                "confidence_score": confidence_score,
                "agreement_score": agreement_score,
                "reasoning_summary": judgment.reasoning_summary,
            }
        )

    def _build_consensus(self, state: ReasoningState, domain: str) -> str:
        outputs = {output.agent: output for output in state.agent_outputs}
        risk = outputs.get("Risk Analyst Agent")
        evidence = outputs.get("Evidence Analyst Agent")
        alternatives = outputs.get("Alternative Solutions Agent")

        domain_opening = {
            "clinical": "Clinical consensus",
            "cybersecurity": "Cybersecurity consensus",
            "research": "Research evaluation consensus",
            "enterprise": "Enterprise risk consensus",
            "finance": "Finance consensus",
            "custom": "Custom decision consensus",
        }[domain]

        clauses = []
        if evidence:
            clauses.append(f"Evidence view: {evidence.recommendation}")
        if risk:
            clauses.append(f"Risk view: {risk.recommendation}")
        if alternatives:
            clauses.append(f"Alternative view: {alternatives.recommendation}")

        question = state.question.lower()
        if domain == "clinical" and any(
            term in question for term in ["stroke", "thrombolytic", "aphasia", "weakness"]
        ):
            recommendation = (
                "activate an urgent stroke pathway and consider thrombolysis only after imaging excludes hemorrhage and contraindications are checked"
            )
        elif domain == "clinical":
            recommendation = "prioritize patient safety and diagnostic sequencing before invasive steps"
        elif domain == "enterprise":
            if "replace software engineers" in question or "every company" in question:
                recommendation = (
                    "reject a universal engineer-replacement claim and use AI agents only where task-level evidence, human accountability, and quality gates support them"
                )
            else:
                recommendation = "use governance, approved tooling, and confidentiality controls before allowing broad AI use"
        elif domain == "cybersecurity":
            recommendation = "contain the personal device, preserve evidence, assess exposure, and then decide notification and remediation duties"
        elif domain == "research":
            recommendation = "avoid single-LLM grading dependence until validity, bias, reliability, and appeal checks are demonstrated"
        elif domain == "finance":
            recommendation = "avoid putting all savings into one AI stock and prefer diversified, liquidity-aware investing"
        else:
            recommendation = "make the decision conditional on the strongest cited evidence and unresolved risks"

        uncertainty_note = ""
        if prompt_injection_risk(state.question):
            uncertainty_note = " The request for certainty is treated as a reliability risk, so the answer preserves uncertainty rather than claiming 100% confidence."

        if not state.retrieved_context:
            grounding_note = (
                " No strong retrieved evidence was found, so the final recommendation is decision-support reasoning with explicit evidence gaps rather than source-grounded certainty."
            )
        else:
            average_relevance = sum(item.relevance_score for item in state.retrieved_context) / len(state.retrieved_context)
            grounding_note = (
                " Retrieved evidence coverage is limited, so confidence is reduced and the recommendation should be verified with better source coverage."
                if average_relevance < 0.55
                else " The final recommendation is grounded in the cited retrieved sources and preserves the main disagreement points."
            )

        return (
            f"{domain_opening}: {recommendation}. "
            + " ".join(clauses)
            + uncertainty_note
            + grounding_note
        )

    def _build_reasoning_summary(
        self, state: ReasoningState, disagreements: list, domain: str
    ) -> str:
        task_summary = "; ".join(task.description for task in state.reasoning_tasks)
        disagreement_summary = (
            "; ".join(item.topic for item in disagreements)
            if disagreements
            else "No material disagreements detected."
        )
        agent_summary = " ".join(
            f"{output.agent}: {output.conclusion}" for output in state.agent_outputs
        )
        return (
            f"Scenario detected: {domain}. Planner tasks: {task_summary}. "
            f"Independent findings: {agent_summary} "
            f"Detected disagreement areas: {disagreement_summary}."
        )
