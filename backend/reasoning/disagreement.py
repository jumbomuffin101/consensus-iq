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
        similarities = [
            self._token_similarity(left.recommendation, right.recommendation)
            for index, left in enumerate(outputs)
            for right in outputs[index + 1 :]
        ]
        semantic_alignment = sum(similarities) / len(similarities) if similarities else 0.5
        confidence_values = [output.confidence_score for output in outputs]
        confidence_spread = max(confidence_values) - min(confidence_values)
        evidence_coverage = sum(1 for output in outputs if output.evidence_refs) / len(outputs)
        severity_penalty = sum(
            {"low": 0.05, "medium": 0.12, "high": 0.22}[item.severity]
            for item in disagreements
        )
        raw_score = (
            0.35
            + (semantic_alignment * 0.34)
            + (evidence_coverage * 0.18)
            + self._domain_alignment_bonus(outputs)
            - (confidence_spread * 0.22)
            - severity_penalty
        )
        return round(max(0.05, min(0.98, raw_score)), 2)

    def _detect_conflicting_recommendations(
        self, outputs: list[AgentOutput]
    ) -> list[Disagreement]:
        recommendation_groups: dict[str, list[AgentOutput]] = {}
        for output in outputs:
            key = self._recommendation_group(output.recommendation)
            recommendation_groups.setdefault(key, []).append(output)

        if len(recommendation_groups) <= 1 or self._recommendations_are_complementary(outputs):
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
        if "do not" in normalized or "avoid" in normalized or "restrict" in normalized:
            return "avoid"
        if "pilot" in normalized or "limited" in normalized or "controlled" in normalized:
            return "pilot"
        if "proceed" in normalized or "support" in normalized or "favor" in normalized:
            return "proceed"
        if "caution" in normalized or "gate" in normalized:
            return "caution"
        if "compare" in normalized or "consider" in normalized:
            return "compare"
        return normalized[:40]

    def _recommendations_are_complementary(self, outputs: list[AgentOutput]) -> bool:
        text = " ".join(output.recommendation.lower() for output in outputs)
        if "do not" in text and ("support" in text or "favor" in text):
            return False
        if "restrict" in text and "unrestricted" in text:
            return False
        return any(
            term in text
            for term in ["controlled", "calibration", "imaging", "approved", "hybrid"]
        )

    def _token_similarity(self, left: str, right: str) -> float:
        stopwords = {
            "a",
            "an",
            "and",
            "or",
            "the",
            "to",
            "for",
            "with",
            "before",
            "after",
            "use",
            "using",
        }
        left_tokens = {token for token in left.lower().replace("-", " ").split() if token not in stopwords}
        right_tokens = {token for token in right.lower().replace("-", " ").split() if token not in stopwords}
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def _domain_alignment_bonus(self, outputs: list[AgentOutput]) -> float:
        text = " ".join(
            f"{output.recommendation} {output.conclusion}".lower()
            for output in outputs
        )
        if all(term in text for term in ["mri", "lp"]) or "lumbar puncture" in text:
            return -0.02
        if all(term in text for term in ["public ai", "confidential"]):
            return 0.08
        if all(term in text for term in ["grader", "validity"]):
            return 0.03
        if "custom question" in text or "decision criteria" in text:
            return -0.03
        return 0.0
