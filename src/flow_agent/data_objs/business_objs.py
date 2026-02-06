from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

DecisionID = Literal[
    "reset_vpn_profile",
    "restart_sso_session",
    "run_connectivity_diagnostics",
    "update_internal_record",
    "send_notification",
    "approval_required",
]

DECISION_TRIGGERS = {
    "reset_vpn_profile": (
        "Triggered by repeated VPN disconnects, corrupted profiles, or authentication failures."
    ),
    "restart_sso_session": (
        "Triggered by expired sessions, login loops, or token-related issues."
    ),
    "run_connectivity_diagnostics": (
        "Triggered by network failures, unreachable services, or packet loss indications."
    ),
    "update_internal_record": (
        "Triggered by status changes, onboarding/offboarding events, or asset updates."
    ),
    "send_notification": (
        "Triggered when escalation, alerting, or user communication is required."
    ),
    "approval_required": (
        "Triggered when policies, compliance conditions, elevated access, or exceptions apply."
    ),
}


class DecisionContext(BaseModel):
    decision_id: str
    context: str


class DecisionOutput(BaseModel):
    decision_id: DecisionID
    decision: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    model: Optional[str] = 'default'
    notes: Optional[str] = None
    latency_ms: Optional[int] = Field(None, ge=0)
    thread_identifier: str = None

    # Normalize/round confidence at validation time
    @field_validator("confidence", mode="before")
    @classmethod
    def _round_confidence(cls, v):
        try:
            return round(float(v), 3)
        except Exception:
            raise ValueError("confidence must be a numeric value between 0.0 and 1.0")


class CombinedPlan(BaseModel):
    # Decision flags
    reset_vpn_profile: bool = False
    restart_sso_session: bool = False
    run_connectivity_diagnostics: bool = False
    update_internal_record: bool = False
    send_notification: bool = False
    approval_required: bool = False
    generated_summary: str | None = ''
    thread_identifier: Optional[str] = None

    # Aggregated metadata
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    task_specific_notes: Optional[list[str]] = []

    @classmethod
    def assemble_from_evaluators(
            cls,
            evals: List[DecisionOutput],
            conf_threshold: float = 0.6,
            summary_notes: Optional[list[str]] = None,
            additional_summary: Optional[str] = None
    ) -> "CombinedPlan":
        """
        Deterministic combiner:
        - Enable a decision if any evaluator for that decision_id returned decision==True
          with confidence >= conf_threshold.
        - approval_required follows same rule.
        - confidence = max(confidence of enabled decisions) or 0.0.
        - notes = summary_notes (combiner LLM or deterministic concat).
        """
        summary_notes = list(summary_notes or [])  # make a fresh list

        if summary_notes is None:
            summary_notes = []
        keys = {
            "reset_vpn_profile",
            "restart_sso_session",
            "run_connectivity_diagnostics",
            "update_internal_record",
            "send_notification",
            "approval_required",
        }

        mapping = {k: False for k in keys}
        true_confidences: List[float] = []
        _thread_identifier = ''
        for d in evals:
            if d.decision and d.confidence >= conf_threshold:
                mapping[d.decision_id] = True
                true_confidences.append(d.confidence)
                summary_notes.append(f"[{d.decision_id}] {d.notes or ''}".strip())
                _thread_identifier = d.thread_identifier

        overall_conf = round(max(true_confidences) if true_confidences else 0.0, 3)
        return cls(
            generated_summary=additional_summary if additional_summary else '',
            reset_vpn_profile=mapping["reset_vpn_profile"],
            restart_sso_session=mapping["restart_sso_session"],
            run_connectivity_diagnostics=mapping["run_connectivity_diagnostics"],
            update_internal_record=mapping["update_internal_record"],
            send_notification=mapping["send_notification"],
            approval_required=mapping["approval_required"],
            confidence=overall_conf,
            task_specific_notes=summary_notes,
            thread_identifier=_thread_identifier if _thread_identifier else None
        )


# ---- Minimal example usage ----
if __name__ == "__main__":
    examples = [
        DecisionOutput(
            decision_id="reset_vpn_profile",
            decision=True,
            confidence=0.87,
            model="gpt-5-mini-v1",
            notes=["Detected repeated VPN disconnects"],
            latency_ms=320,
        ),
        DecisionOutput(
            decision_id="restart_sso_session",
            decision=True,
            confidence=0.58,
            model="gpt-5-mini-v1",
            notes=["SSO token refresh errors ambiguous"],
            latency_ms=280,
        ),
    ]

    plan = CombinedPlan.assemble_from_evaluators(examples, conf_threshold=0.6,
                                                 summary_notes=["Reset VPN recommended; SSO ambiguous."],
                                                 additional_summary='additional_summary')
    print(plan.model_dump_json(indent=2))
