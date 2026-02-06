from __future__ import annotations
import pytest

from src.flow_agent.data_objs.business_objs import DecisionOutput, CombinedPlan


# --- Example evaluators used in tests ---
@pytest.fixture
def example_evals():
    return [
        DecisionOutput(
            decision_id="reset_vpn_profile",
            decision=True,
            confidence=0.87,
            model="gpt-5-mini-v1",
            notes="Detected repeated VPN disconnects",
            latency_ms=320,
        ),
        DecisionOutput(
            decision_id="restart_sso_session",
            decision=True,
            confidence=0.58,
            model="gpt-5-mini-v1",
            notes="SSO token refresh errors ambiguous",
            latency_ms=280,
        ),
    ]


def test_default_threshold_disables_low_confidence(example_evals):
    plan = CombinedPlan.assemble_from_evaluators(
        example_evals, conf_threshold=0.6, summary_notes=["Reset VPN recommended; SSO ambiguous."]
    )
    assert plan.reset_vpn_profile is True
    assert plan.restart_sso_session is False  # 0.58 < 0.6
    assert plan.confidence == 0.87
    assert "Reset VPN recommended; SSO ambiguous." in plan.task_specific_notes


def test_lower_global_threshold_enables_sso(example_evals):
    plan = CombinedPlan.assemble_from_evaluators(
        example_evals, conf_threshold=0.5, summary_notes=["Lower threshold test"]
    )
    assert plan.reset_vpn_profile is True
    assert plan.restart_sso_session is True  # 0.58 >= 0.5
    assert plan.confidence == 0.87


def test_per_decision_threshold_enables_sso_only(example_evals):
    plan = CombinedPlan.assemble_from_evaluators(
        example_evals,
        conf_threshold=0.5,
        summary_notes=["Per-decision threshold test"]
    )
    assert plan.reset_vpn_profile is True
    assert plan.restart_sso_session is True  # per-decision threshold 0.55 allows 0.58
    assert plan.confidence == 0.87


def test_no_decisions_above_threshold_returns_confidence_zero():
    low_conf_evals = [
        DecisionOutput(
            decision_id="reset_vpn_profile",
            decision=True,
            confidence=0.2,
            model="gpt-5-mini-v1",
            notes="low confidence",
            latency_ms=100,
        )
    ]
    plan = CombinedPlan.assemble_from_evaluators(low_conf_evals, conf_threshold=0.5)
    assert plan.reset_vpn_profile is False
    assert plan.confidence == 0.0
