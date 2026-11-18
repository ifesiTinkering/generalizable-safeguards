"""Keep prepaid semantic API usage separate from Tinker cash budgets.

The rubric, Astra/Luna attempt policy, and once-only physical IDs are unchanged.
Cash-funded execution continues to use the frozen original adapter.
"""

from decimal import Decimal

from evidence_budget import GrantLedger
from evidence_judge_fallback import RetryingEvidenceJudge
from judge_provider_adapter import MeteredJudgeAdapter

from study.vendor.shared.judge_factory import (
    MODELS,
)
from study.vendor.shared.judge_factory import (
    grading_upper as cash_upper,
)
from study.vendor.shared.judge_factory import (
    make_judge as cash_judge,
)

_PREPAID = False
_MAX_CALLS = 1000000


def configure_prepaid(enabled, max_calls=1000000):
    global _PREPAID, _MAX_CALLS
    if type(enabled) is not bool or type(max_calls) is not int or max_calls < 1:
        raise ValueError("Explicit prepaid flag and positive judge-call allowance required")
    _PREPAID, _MAX_CALLS = enabled, max_calls


class PrepaidAdapter(MeteredJudgeAdapter):
    def reserve(self, key, body):
        # A depleted prepaid allowance stops admission; it never falls back to cash.
        if not self.grant.reserve(
            key,
            {"request_sha256": key, "model": body["model"], "max_output_tokens": 8192},
        ):
            return False
        self.funding[key] = "grant"
        return self.meter.reserve(
            key,
            Decimal(".000001"),
            {
                "kind": "judge",
                "model": body["model"],
                "funding": "grant",
                "cash_expense_usd": "0",
                "unknown_hold_preserved": True,
            },
        )


def make_judge(root, meter, grant=None):
    if not _PREPAID:
        return cash_judge(root, meter, grant)
    from pathlib import Path

    grant = grant or GrantLedger(Path(root) / "prepaid_judge_usage.sqlite", max_calls=_MAX_CALLS)
    return RetryingEvidenceJudge(
        root, {k: PrepaidAdapter(v, meter, grant) for k, v in MODELS.items()}, MODELS
    )


def grading_upper(row):
    # The tiny admission holds cover at most six attempts. Known usage settles at zero cash.
    return Decimal(".000006") if _PREPAID else cash_upper(row)
