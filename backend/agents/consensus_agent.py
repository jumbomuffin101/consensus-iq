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
        injection_penalty = prompt_injection_risk(state.question)
        avg_confidence = (
            sum(output.confidence_score for output in state.agent_outputs)
            / len(state.agent_outputs)
            if state.agent_outputs
            else 0.35
        )
        confidence_values = [output.confidence_score for output in state.agent_outputs]
        confidence_spread = (
            max(confidence_values) - min(confidence_values)
            if confidence_values
            else 0.3
        )
        evidence_quality = self._evidence_quality_score(state, profile.evidence_quality)
        answer_completeness = self._answer_completeness_score(state, avg_confidence)
        prompt_clarity = max(0.25, min(0.9, 1.0 - profile.ambiguity))
        safety_score = max(0.0, 1.0 - (injection_penalty * 4.0))
        raw_confidence = (
            (evidence_quality * 0.30)
            + (agreement_score * 0.25)
            + (answer_completeness * 0.20)
            + (prompt_clarity * 0.15)
            + (safety_score * 0.10)
            - min(0.08, confidence_spread * 0.08)
            - min(0.06, missing_information_load(state) * 0.25)
        )
        confidence_score = self._bounded_confidence_range(
            raw_confidence, profile.domain, evidence_quality, injection_penalty
        )

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
            agent_name="consensus judge",
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
            recommendation = self._custom_recommendation(state.question)

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

    def _custom_recommendation(self, question: str) -> str:
        topic = question.strip().rstrip("?.!")
        return (
            f"evaluate this custom proposal: '{topic or 'this proposal'}' as a conditional decision: clarify the goal, "
            "expected benefits, participants or stakeholders, downside risks, and a "
            "small reversible trial before making a broad commitment"
        )

    def _evidence_quality_score(
        self, state: ReasoningState, profile_evidence_quality: float
    ) -> float:
        source_count = len(state.retrieved_context)
        if source_count == 0:
            return 0.25
        source_bonus = min(0.12, source_count * 0.04)
        return max(0.25, min(0.9, profile_evidence_quality + source_bonus))

    def _answer_completeness_score(
        self, state: ReasoningState, avg_agent_confidence: float
    ) -> float:
        if not state.agent_outputs:
            return 0.25
        populated_outputs = sum(
            1
            for output in state.agent_outputs
            if output.recommendation and output.conclusion
        )
        coverage = populated_outputs / len(state.agent_outputs)
        missing_count = sum(len(output.missing_evidence) for output in state.agent_outputs)
        missing_penalty = min(0.22, missing_count * 0.025)
        return max(
            0.25,
            min(0.9, (coverage * 0.55) + (avg_agent_confidence * 0.45) - missing_penalty),
        )

    def _bounded_confidence_range(
        self,
        raw_confidence: float,
        domain: str,
        evidence_quality: float,
        injection_penalty: float,
    ) -> float:
        if injection_penalty:
            return round(max(0.1, min(0.3, raw_confidence)), 2)
        if domain == "custom":
            upper_bound = 0.5
            return round(max(0.3, min(upper_bound, raw_confidence)), 2)
        upper_bound = 0.75 if domain == "clinical" and evidence_quality >= 0.55 else 0.65
        lower_bound = 0.45 if evidence_quality >= 0.35 else 0.35
        return round(max(lower_bound, min(upper_bound, raw_confidence)), 2)

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
