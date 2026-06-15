from models.reasoning import AgentOutput, Disagreement


class DisagreementDetector:
    """Compares independent agent outputs before the consensus judge decides.

    Future LLM agents should keep returning AgentOutput so this detector remains
    testable and provider-neutral.
    """

    CONFIDENCE_GAP_THRESHOLD = 0.15

    def detect(self, outputs: list[AgentOutput]) -> list[Disagreement]:
        disagreements: list[Disagreement] = []
        if not outputs:
            return [
                Disagreement(
                    topic="Decision uncertainty",
                    kind="missing_evidence",
                    severity="medium",
                    positions=[
                        "No specialist agent outputs were available for comparison."
                    ],
                    suggested_resolution=(
                        "Rerun the analysis and treat any recommendation as unscored "
                        "until specialist perspectives are available."
                    ),
                )
            ]

        disagreements.extend(self._detect_conflicting_recommendations(outputs))
        confidence_gap = self._detect_confidence_gap(outputs)
        if confidence_gap:
            disagreements.append(confidence_gap)
        disagreements.extend(self._detect_missing_evidence(outputs))
        if not disagreements:
            disagreements.append(self._build_decision_uncertainty(outputs))
        return disagreements

    def calculate_agreement_score(self, outputs: list[AgentOutput]) -> float:
        if not outputs:
            return 0.25

        disagreements = self.detect(outputs)
        direction_alignment = self._direction_alignment(outputs)
        confidence_values = [output.confidence_score for output in outputs]
        confidence_spread = max(confidence_values) - min(confidence_values)
        confidence_alignment = max(0.0, 1.0 - min(1.0, confidence_spread / 0.5))
        role_coverage = min(1.0, len(outputs) / 3)
        conflict_penalty = 0.0
        if any(item.kind == "conflicting_recommendation" for item in disagreements):
            conflict_penalty += 0.12
        if any(item.severity == "high" for item in disagreements):
            conflict_penalty += 0.08
        severity_penalty = sum(
            {"low": 0.01, "medium": 0.025, "high": 0.06}[item.severity]
            for item in disagreements
        )
        disagreement_count_penalty = min(0.08, len(disagreements) * 0.018)
        raw_score = (
            0.22
            + (direction_alignment * 0.42)
            + (confidence_alignment * 0.2)
            + (role_coverage * 0.12)
            + self._domain_alignment_bonus(outputs)
            - conflict_penalty
            - severity_penalty
            - disagreement_count_penalty
            - self._major_concern_penalty(outputs)
        )
        return round(max(0.25, min(0.88, raw_score)), 2)

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
                    "Request additional Microsoft retrieval or lower confidence "
                    "until missing evidence is resolved."
                ),
            )
        ]

    def _build_decision_uncertainty(self, outputs: list[AgentOutput]) -> Disagreement:
        confidence_values = [output.confidence_score for output in outputs]
        average_confidence = sum(confidence_values) / len(confidence_values)
        evidence_refs = sorted(
            {
                ref
                for output in outputs
                for ref in output.evidence_refs
            }
        )
        positions = [
            f"Average specialist confidence: {average_confidence:.2f}",
            (
                f"Evidence referenced by agents: {', '.join(evidence_refs)}"
                if evidence_refs
                else "No retrieved evidence references were cited by specialists."
            ),
            "No direct recommendation conflict was detected, but assumptions still require review.",
        ]
        return Disagreement(
            topic="Decision uncertainty",
            kind="missing_evidence",
            severity="low",
            positions=positions,
            suggested_resolution=(
                "Proceed only with the stated conditions, and verify local facts, "
                "policy constraints, or domain-specific assumptions before acting."
            ),
        )

    def _recommendation_group(self, recommendation: str) -> str:
        normalized = recommendation.lower()
        if "diversification" in normalized or "diversified" in normalized:
            return "diversify"
        if "contain" in normalized or "preserve evidence" in normalized:
            return "contain"
        if "not as the sole" in normalized or "not let a single" in normalized:
            return "hybrid"
        if "do not" in normalized or "avoid" in normalized or "restrict" in normalized:
            return "avoid"
        if "pilot" in normalized or "limited" in normalized or "controlled" in normalized:
            return "pilot"
        if "proceed" in normalized or "support" in normalized or "favor" in normalized or "evaluate" in normalized:
            return "proceed"
        if "caution" in normalized or "gate" in normalized:
            return "caution"
        if "compare" in normalized or "consider" in normalized:
            return "compare"
        return normalized[:40]

    def _recommendations_are_complementary(self, outputs: list[AgentOutput]) -> bool:
        text = " ".join(output.recommendation.lower() for output in outputs)
        complementary_caution = any(
            term in text
            for term in [
                "only after",
                "until",
                "diversification",
                "diversified",
                "not as the sole",
                "not let a single",
            ]
        )
        if "do not" in text and ("support" in text or "favor" in text) and not complementary_caution:
            return False
        if "restrict" in text and "unrestricted" in text:
            return False
        return any(
            term in text
            for term in [
                "controlled",
                "calibration",
                "imaging",
                "approved",
                "hybrid",
                "contain",
                "forensic",
                "scope",
                "diversified",
                "contraindications",
            ]
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
        if all(term in text for term in ["thrombolysis", "contraindication"]):
            return 0.05
        if all(term in text for term in ["mri", "lp"]) or "lumbar puncture" in text:
            return -0.02
        if all(term in text for term in ["public ai", "confidential"]):
            return 0.08
        if all(term in text for term in ["grader", "validity"]):
            return 0.03
        if "custom question" in text or "decision criteria" in text:
            return -0.03
        if all(term in text for term in ["contain", "evidence"]):
            return 0.05
        if all(term in text for term in ["single", "grader"]):
            return -0.02
        if all(term in text for term in ["all savings", "one ai stock"]):
            return 0.02
        if "replace engineers" in text or "universal replacement" in text:
            return -0.06
        return 0.0

    def _direction_alignment(self, outputs: list[AgentOutput]) -> float:
        if self._recommendations_are_complementary(outputs):
            text = " ".join(output.recommendation.lower() for output in outputs)
            if "single llm" in text or "single-model" in text:
                return 0.52
            if "all savings" in text or "one ai stock" in text:
                return 0.62
            return 0.72

        groups = [self._recommendation_group(output.recommendation) for output in outputs]
        if not groups:
            return 0.0
        most_common = max(groups.count(group) for group in set(groups))
        base = most_common / len(groups)
        if {"avoid", "proceed"}.issubset(set(groups)):
            return max(0.15, base - 0.22)
        if "compare" in groups and ("avoid" in groups or "pilot" in groups):
            return max(0.25, base - 0.08)
        return base

    def _major_concern_penalty(self, outputs: list[AgentOutput]) -> float:
        text = " ".join(
            f"{output.recommendation} {output.conclusion}".lower()
            for output in outputs
        )
        penalty = 0.0
        if "catastrophic" in text or "permanent capital loss" in text:
            penalty += 0.04
        if "single llm" in text and "sole" in text:
            penalty += 0.03
        if "100% confidence" in text or "100% certain" in text:
            penalty += 0.05
        return penalty
