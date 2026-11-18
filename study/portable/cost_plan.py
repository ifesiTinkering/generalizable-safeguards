"""Historical nominal token accounting; pricing verification is an explicit input."""

import json
import os
from decimal import Decimal
from pathlib import Path

FAMILY = os.environ["POLICY_STUDY_MODEL_FAMILY"]
RATES = json.loads(
    (Path(__file__).resolve().parents[1] / "vendor" / FAMILY / "rates.json").read_text()
)
MODEL = RATES["model"]


def tokens(value):
    if type(value) is not int or value < 0:
        raise ValueError("Nonnegative integer tokens required")
    return value


def sampling_cost(input_tokens, output_tokens, *, cached_input_tokens=0):
    i, o, c = map(tokens, (input_tokens, output_tokens, cached_input_tokens))
    if c > i:
        raise ValueError("Cached tokens exceed input")
    r = RATES["tinker"]
    return (
        Decimal(r["prefill"]) * (i - c)
        + Decimal(r["cached_prefill"]) * c
        + Decimal(r["sample"]) * o
    ) / 1000000


def sampling_upper_bound(prompt_tokens, max_output_tokens=23000, *, n=1):
    if type(n) is not int or n < 1:
        raise ValueError("Positive sample count required")
    return n * sampling_cost(prompt_tokens, max_output_tokens)


def training_upper_bound(sequence_lengths):
    return Decimal(RATES["tinker"]["train"]) * sum(tokens(n) for n in sequence_lengths) / 1000000
