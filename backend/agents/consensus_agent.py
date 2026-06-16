from agents.prompting import agent_outputs_payload, disagreements_payload
from llm.base import BaseLLMProvider
from llm.mock import MockLLMProvider
from models.reasoning import ConsensusJudgment, FinalAnswer, KeyFinding, ReasoningState
from policies.custom_domains import policy_for_domain
from prompts.final_judge import (
    FINAL_ANSWER_SCHEMA_INSTRUCTIONS,
    FINAL_JUDGE_SYSTEM_PROMPT,
    STRICT_CITATION_RETRY_INSTRUCTIONS,
)
from reasoning.disagreement import DisagreementDetector
from reasoning.domain import (
    bounded_score,
    build_domain_profile,
    missing_information_load,
    prompt_injection_risk,
)
from reasoning.general_decision import build_general_decision_frame


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
        fallback_final_answer = self._build_final_answer(
            state_with_disagreements,
            profile.domain,
            confidence_score,
            agreement_score,
        )
        payload = self.provider.complete_json(
            system_prompt=FINAL_JUDGE_SYSTEM_PROMPT,
            user_prompt=self._final_judge_prompt(
                state_with_disagreements,
                profile.domain,
                confidence_score,
                agreement_score,
            ),
            fallback=fallback_final_answer.dict(),
            agent_name="consensus judge",
        )
        final_answer = self._parse_final_answer(payload, fallback_final_answer)
        final_answer, invalid_citations = self._validate_final_answer_citations(
            final_answer, state.retrieved_context
        )
        final_answer = self._cap_source_quality(final_answer, state_with_disagreements)
        if invalid_citations:
            retry_payload = self.provider.complete_json(
                system_prompt=f"{FINAL_JUDGE_SYSTEM_PROMPT} {STRICT_CITATION_RETRY_INSTRUCTIONS}",
                user_prompt=self._final_judge_prompt(
                    state_with_disagreements,
                    profile.domain,
                    confidence_score,
                    agreement_score,
                    invalid_citations=invalid_citations,
                ),
                fallback=final_answer.dict(),
                agent_name="consensus judge",
            )
            final_answer = self._parse_final_answer(retry_payload, final_answer)
            final_answer, invalid_citations = self._validate_final_answer_citations(
                final_answer, state.retrieved_context
            )
            final_answer = self._cap_source_quality(final_answer, state_with_disagreements)
            if invalid_citations:
                final_answer = self._remove_invalid_citations(
                    final_answer, state.retrieved_context
                )

        judgment = ConsensusJudgment(
            consensus=final_answer.recommendation,
            confidence_score=confidence_score,
            agreement_score=agreement_score,
            reasoning_summary=final_answer.summary,
        )

        return state.copy(
            update={
                "disagreements": disagreements,
                "scenario_label": profile.scenario_label,
                "consensus": judgment.consensus,
                "confidence_score": confidence_score,
                "agreement_score": agreement_score,
                "reasoning_summary": judgment.reasoning_summary,
                "final_answer": final_answer,
            }
        )

    def _final_judge_prompt(
        self,
        state: ReasoningState,
        domain: str,
        confidence_score: float,
        agreement_score: float,
        invalid_citations: list[str] | None = None,
    ) -> str:
        retrieved_sources = [
            {
                "source_id": item.source_id,
                "citation_id": item.citation_id,
                "title": item.title,
                "source": item.source,
                "url": item.url,
                "snippet": item.snippet,
                "relevance_score": item.relevance_score,
            }
            for item in state.retrieved_context
        ]
        invalid_note = (
            f"\nInvalid source IDs from the previous answer: {invalid_citations}. "
            "Do not use them."
            if invalid_citations
            else ""
        )
        return (
            f"Question: {state.question}\n"
            f"Scenario/domain: {domain}\n"
            f"Custom intake:\n{state.custom_intake.dict() if state.custom_intake else None}\n"
            f"Custom domain policy:\n{policy_for_domain(state.custom_intake.domain).__dict__ if state.custom_intake else None}\n"
            f"Computed confidence_score: {confidence_score}\n"
            f"Computed agreement_score: {agreement_score}\n"
            f"Source quality estimate: {self._source_quality(state)}\n"
            f"Retrieved_sources:\n{retrieved_sources}\n\n"
            f"Deterministic specialist outputs:\n{agent_outputs_payload(state)}\n\n"
            f"Disagreements:\n{disagreements_payload(state)}\n\n"
            f"{FINAL_ANSWER_SCHEMA_INSTRUCTIONS}{invalid_note}"
        )

    def _parse_final_answer(
        self, payload: object, fallback: FinalAnswer
    ) -> FinalAnswer:
        try:
            return FinalAnswer.parse_obj(payload)
        except Exception:
            return fallback

    def _validate_final_answer_citations(
        self, answer: FinalAnswer, sources: list
    ) -> tuple[FinalAnswer, list[str]]:
        source_id_lookup = {source.source_id: source.source_id for source in sources}
        source_id_lookup.update({source.citation_id: source.source_id for source in sources})
        invalid: list[str] = []
        normalized_findings: list[KeyFinding] = []
        for finding in answer.key_findings:
            normalized_ids: list[str] = []
            for source_id in finding.source_ids:
                mapped = source_id_lookup.get(source_id)
                if mapped:
                    normalized_ids.append(mapped)
                else:
                    invalid.append(source_id)
            normalized_findings.append(
                finding.copy(update={"source_ids": list(dict.fromkeys(normalized_ids))})
            )
        return answer.copy(update={"key_findings": normalized_findings}), list(dict.fromkeys(invalid))

    def _remove_invalid_citations(
        self, answer: FinalAnswer, sources: list
    ) -> FinalAnswer:
        sanitized, _ = self._validate_final_answer_citations(answer, sources)
        downgraded_quality = "weak" if not sources else "partial"
        return sanitized.copy(update={"source_quality": downgraded_quality})

    def _cap_source_quality(
        self, answer: FinalAnswer, state: ReasoningState
    ) -> FinalAnswer:
        retrieval_quality = self._source_quality(state)
        rank = {"weak": 0, "partial": 1, "strong": 2}
        if rank[answer.source_quality] <= rank[retrieval_quality]:
            return answer
        return answer.copy(update={"source_quality": retrieval_quality})

    def _build_final_answer(
        self,
        state: ReasoningState,
        domain: str,
        confidence_score: float,
        agreement_score: float,
    ) -> FinalAnswer:
        source_quality = self._source_quality(state)
        if state.custom_intake and state.custom_intake.domain == "pet_health":
            consensus = build_general_decision_frame(state.question).recommendation
        else:
            consensus = self._build_consensus(state, domain)
        findings = self._fallback_key_findings(state)
        risks = self._fallback_risks(state)
        if source_quality == "weak":
            risks = [
                "No strong sources were retrieved for this custom prompt. Treat the recommendation as provisional.",
                *risks,
            ]
        return FinalAnswer(
            summary=self._build_reasoning_summary(state, state.disagreements, domain),
            recommendation=consensus,
            key_findings=findings,
            risks_or_limitations=risks[:5],
            follow_up_questions=self._fallback_follow_up_questions(state),
            source_quality=source_quality,
            provider_used="fast-deterministic",
            live_llm_mode="off",
        )

    def _fallback_key_findings(self, state: ReasoningState) -> list[KeyFinding]:
        source_ids = [source.source_id for source in state.retrieved_context]
        findings: list[KeyFinding] = []
        for output in state.agent_outputs[:3]:
            cited_source_ids = [
                source.source_id
                for source in state.retrieved_context
                if source.citation_id in output.evidence_refs
                or source.source_id in output.evidence_refs
            ]
            if not cited_source_ids and source_ids:
                cited_source_ids = source_ids[:1]
            findings.append(
                KeyFinding(
                    claim=output.conclusion,
                    source_ids=cited_source_ids,
                )
            )
        if not findings:
            findings.append(
                KeyFinding(
                    claim="The retrieved evidence is insufficient for a strongly sourced recommendation.",
                    source_ids=[],
                )
            )
        return findings

    def _fallback_risks(self, state: ReasoningState) -> list[str]:
        risks = []
        for output in state.agent_outputs:
            risks.extend(output.missing_evidence[:2])
            risks.extend(output.limitations[:1])
        if state.disagreements:
            risks.extend(item.suggested_resolution for item in state.disagreements[:2])
        return list(dict.fromkeys(risks)) or [
            "Local context and implementation constraints may change the decision."
        ]

    def _fallback_follow_up_questions(self, state: ReasoningState) -> list[str]:
        missing = []
        for output in state.agent_outputs:
            missing.extend(output.missing_evidence)
        if missing:
            return [f"Can we verify: {item}" for item in list(dict.fromkeys(missing))[:3]]
        return [
            "What decision threshold would make this recommendation actionable?",
            "Which local constraints or policies could change the answer?",
        ]

    def _source_quality(self, state: ReasoningState) -> str:
        if not state.retrieved_context:
            return "weak"
        top_relevance = max(source.relevance_score for source in state.retrieved_context)
        average_relevance = sum(source.relevance_score for source in state.retrieved_context) / len(state.retrieved_context)
        if len(state.retrieved_context) >= 2 and top_relevance >= 0.7 and average_relevance >= 0.58:
            return "strong"
        if top_relevance >= 0.45:
            return "partial"
        return "weak"

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
            "sports_injury": "Health / Sports Injury consensus",
            "custom": "Custom decision consensus",
        }[domain]
        if state.custom_intake and state.custom_intake.domain == "pet_health":
            domain_opening = "Pet health triage"

        clauses = []
        if evidence:
            clauses.append(f"Evidence view: {evidence.recommendation}")
        if risk:
            clauses.append(f"Risk view: {risk.recommendation}")
        if alternatives:
            clauses.append(f"Alternative view: {alternatives.recommendation}")

        question = state.question.lower()
        if state.custom_intake and state.custom_intake.domain == "pet_health":
            recommendation = build_general_decision_frame(state.question).recommendation
        elif domain == "clinical" and any(
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
        elif domain == "sports_injury":
            recommendation = build_general_decision_frame(state.question).recommendation
        else:
            recommendation = build_general_decision_frame(state.question).recommendation
        recommendation = recommendation.rstrip(".")

        uncertainty_note = ""
        if prompt_injection_risk(state.question):
            uncertainty_note = " The request for certainty is treated as a reliability risk, so the answer preserves uncertainty rather than claiming 100% confidence."

        if not state.retrieved_context:
            if state.custom_intake and state.custom_intake.domain == "pet_health":
                grounding_note = (
                    " No strong sources were retrieved for this custom prompt, so this is safe triage guidance rather than a source-grounded diagnosis."
                )
            else:
                grounding_note = (
                    " No strong sources were retrieved for this custom prompt, so the final recommendation is provisional and should be verified with stronger evidence."
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
        return build_general_decision_frame(question).recommendation

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
        if state.custom_intake and state.custom_intake.domain == "pet_health":
            return (
                "This is a pet-health triage question. No diagnosis can be made from "
                "the prompt alone, so the safest answer is to look for red flags, "
                "check how unusual and prolonged the sleepiness is, and contact a "
                "veterinarian promptly if symptoms are severe, prolonged, or paired "
                "with appetite, breathing, pain, gum-color, collapse, toxin, vomiting, "
                "or diarrhea concerns."
            )
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
