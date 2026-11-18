"""Relative-path access to immutable, scored evaluation records."""

from functools import lru_cache

from .io import ROOT, read_json
from .metrics import strict_mean, summarize

OVERRIDES = {}


def set_override(panel, family, path):
    OVERRIDES[panel, family.lower()] = path
    cells.cache_clear()


@lru_cache(maxsize=8)
def cells(panel, family):
    """Index scored answer groups without pooling different arms or seeds.

    Qwen's out-of-distribution SFT observations were archived separately and are
    merged here for a uniform interface. Every original row/draw/conversation
    identifier remains attached to its scores.
    """
    family = family.lower()
    if family not in ("qwen", "gpt") or panel not in ("ood", "id", "policy"):
        raise ValueError("Use panel ood/id/policy and family qwen/gpt")
    override = OVERRIDES.get((panel, family))
    data = read_json(
        override if override else ROOT / "data" / "observations" / f"{panel}-{family}.json.gz"
    )["cells"]
    if not override and panel == "ood" and family == "qwen":
        data += read_json(ROOT / "data/observations/ood-qwen-sft.json.gz")["cells"]
    result = {}
    for c in data:
        i = c["identity"]
        arm = None if i["target"] == "SFT" else i["arm"]
        key = (i["target"], arm, i["seed"], i["mode"], i["cell"])
        if key in result:
            raise ValueError("Duplicate evaluation cell: " + str(key))
        result[key] = c
    return result


def cell_summaries(panel, family):
    """Yield per-checkpoint/domain metrics before any cross-domain averaging."""
    for key, c in sorted(cells(panel, family).items(), key=lambda x: str(x[0])):
        target, arm, seed, mode, domain = key
        yield dict(
            panel=panel,
            family=family,
            checkpoint_id=c["checkpoint_id"],
            target=target,
            arm=arm,
            seed=seed,
            mode=mode,
            source=domain.split(":")[0],
            domain=domain.split(":")[1],
            rows=len(set(c["row_ids"])),
            answers=len(c["values"]),
            metrics=summarize(c["values"]),
        )


def paired_points(panel, family, target, source, mode, low, high, seeds):
    """Equal domain and paired seed means. Missing cells are an error."""
    index = cells(panel, family)
    domains = (
        ("privacy", "fraud", "defamation")
        if panel == "id"
        else ("content_moderation", "specialized_advice", "intellectual_property")
    )
    arms = {}
    for arm in (low, high):
        ms = [
            summarize(index[target, arm, seed, mode, source + ":" + domain]["values"])
            for seed in seeds
            for domain in domains
        ]
        arms[arm] = {m: strict_mean(x[m] for x in ms) for m in ms[0]}
    delta = {
        m: arms[high][m] - arms[low][m]
        if arms[high][m] is not None and arms[low][m] is not None
        else None
        for m in arms[low]
    }
    return arms, delta
