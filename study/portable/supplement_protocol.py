"""Fresh-study authorization replaces private historical admission chains only."""

_admitted = None


def authorize(configuration):
    global _admitted
    if _admitted is not None:
        raise RuntimeError("Only one fresh study per process")
    if configuration.get("explicit_paid_opt_in") is not True:
        raise PermissionError("Paid execution was not explicitly enabled")
    if configuration.get("source_verification_passed") is not True:
        raise PermissionError("Unverified scientific source")
    _admitted = dict(configuration)


def require_authorized():
    if _admitted is None:
        raise PermissionError("No portable fresh-study admission")
    return {"specification": dict(_admitted)}
