"""Driver wall-clock accounting for GiGPO/WiGPO (seconds, not GPU-seconds).

The four shared model stages alias the original trainer timers verbatim.
Advantage timers operate on CPU tensors returned by the blocking worker calls.
"""

from contextlib import contextmanager
from time import perf_counter


@contextmanager
def budget_timer(name, timings):
    """Accumulate disjoint intervals; a None collector disables measurement."""
    if timings is None:
        yield
        return
    start = perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + perf_counter() - start


def budget_metrics(timing_raw, advantage_timings, train_total):
    """Report component sum separately from measured training wall time.

    train_total ends before validation, checkpointing and generation dumps.
    Other retains bookkeeping, reward processing, diagnostics and advantage
    composition, rather than charging those to inter-step or intra-step.
    """
    sections = {
        "Rollout": timing_raw.get("gen", 0.0),
        "OldProb": timing_raw.get("old_log_prob", 0.0),
        "RefProb": timing_raw.get("ref", 0.0),
        "Update": timing_raw.get("update_actor", 0.0),
        "AE": advantage_timings.get("AE", 0.0),
        "inter-step": advantage_timings.get("inter-step", 0.0),
        "intra-step": advantage_timings.get("intra-step", 0.0),
    }
    # GiGPO's original additional components, when profiling that baseline.
    for name in ("Grouping", "A_step"):
        if name in advantage_timings:
            sections[name] = advantage_timings[name]
    measured = sum(sections.values())
    sections.update(MeasuredTotal=measured, TrainTotal=train_total,
                    Other=train_total - measured)
    return {f"time_breakdown/{name}": value for name, value in sections.items()}
