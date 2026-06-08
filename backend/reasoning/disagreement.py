from models.reasoning import AgentOutput, Disagreement


class DisagreementDetector:
    """Compares independent agent outputs before the consensus judge decides.

    Future LLM agents should keep returning AgentOutput so this detector remains
    testable and provider-neutral.
    """

    CONFIDENCE_GAP_THRESHOLD = 0.15

    def detect(self, outputs: list[AgentOutput]) -> list[Disagreement]:
        disagreements: list[Disagreement] = []
        disagreements.extend(self._detect_conflicting_recommendations(outputs))
        confidence_gap = self._detect_confidence_gap(outputs)
        if confidence_gap:
            disagreements.append(confidence_gap)
        disagreements.extend(self._detect_missing_evidence(outputs))
        return disagreements

    def calculate_agreement_score(self, outputs: list[AgentOutput]) -> float:
        if not outputs:
            return 0.0

        disagreements = self.detect(outputs)
        severity_penalty = sum(
            {"low": 0.05, "medium": 0.12, "high": 0.22}[item.severity]
            for item in disagreements
        )
        stance_count = len({output.stance for output in outputs})
        diversity_penalty = max(0, stance_count - 1) * 0.04
        return round(max(0.0, min(1.0, 1.0 - severity_penalty - diversity_penalty)), 2)

    def _detect_conflicting_recommendations(
        self, outputs: list[AgentOutput]
    ) -> list[Disagreement]:
        recommendation_groups: dict[str, list[AgentOutput]] = {}
        for output in outputs:
            key = self._recommendation_group(output.recommendation)
            recommendation_groups.setdefault(key, []).append(output)

        if len(recommendation_groups) <= 1:
            return []

        return [
            Disagreement(
                topic="Recommendation direction",
                kind="conflicting_recommendation",
                severity="medium",
                positions=[
                    f"{output.agent}: {output.recommendation}" for output in outputs
                ],
                suggested_resolution=(
                    "Preserve the shared direction while making rollout scope "
                    "and review gates explicit."
                ),
            )
        ]

    def _detect_confidence_gap(
        self, outputs: list[AgentOutput]
    ) -> Disagreement | None:
        if len(outputs) < 2:
            return None

        sorted_outputs = sorted(outputs, key=lambda output: output.confidence_score)
        lowest = sorted_outputs[0]
        highest = sorted_outputs[-1]
        gap = highest.confidence_score - lowest.confidence_score

        if gap < self.CONFIDENCE_GAP_THRESHOLD:
            return None

        return Disagreement(
            topic="Confidence spread",
            kind="differing_confidence",
            severity="low" if gap < 0.25 else "medium",
            positions=[
                f"{lowest.agent}: {lowest.confidence_score:.2f}",
                f"{highest.agent}: {highest.confidence_score:.2f}",
            ],
            suggested_resolution=(
                "Treat lower-confidence limitations as conditions for the final "
                "recommendation."
            ),
        )

    def _detect_missing_evidence(
        self, outputs: list[AgentOutput]
    ) -> list[Disagreement]:
        missing_items = [
            f"{output.agent}: {item}"
            for output in outputs
            for item in output.missing_evidence
        ]
        outputs_without_refs = [
            output.agent for output in outputs if not output.evidence_refs
        ]

        if not missing_items and not outputs_without_refs:
            return []

        positions = missing_items + [
            f"{agent}: no supporting evidence references provided"
            for agent in outputs_without_refs
        ]

        return [
            Disagreement(
                topic="Evidence completeness",
                kind="missing_evidence",
                severity="medium" if len(positions) > 2 else "low",
                positions=positions,
                suggested_resolution=(
                    "Request additional Foundry IQ retrieval or lower confidence "
                    "until missing evidence is resolved."
                ),
            )
        ]

    def _recommendation_group(self, recommendation: str) -> str:
        normalized = recommendation.lower()
        if "do not" in normalized or "avoid" in normalized:
            return "avoid"
        if "pilot" in normalized or "limited" in normalized:
            return "pilot"
        if "proceed" in normalized or "support" in normalized:
            return "proceed"
        if "caution" in normalized or "gate" in normalized:
            return "caution"
        return normalized[:40]
