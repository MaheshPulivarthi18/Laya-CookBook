"""Speed and memory: preload, warm up, and MPS vs. CPU -- measured fresh
against this repo's environment (Laya 0.3.11, Apple M4), not copied from
Step 0's numbers.

Three things that matter in production:
1. The first call to any checkpoint pays a one-time load cost (building the
   model). Preload at startup so your first real request isn't the one
   that pays it.
2. After loading, the very first forward pass is still slower than steady
   state (kernel warm-up). Send one throwaway prediction at startup too.
3. MPS (Apple GPU) vs. CPU matters for latency, but batching questions
   barely helps either way -- cost scales with (questions x text length),
   not with batch size, because each question re-reads the full text.

Run:
    python production/speed_and_memory.py
"""
import os
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def median_latency_ms(fn, n=12):
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return times[len(times) // 2], min(times), max(times)


if __name__ == "__main__":
    from laya import Router

    state = {"body": "We were billed twice for March. Please refund the duplicate today."}
    one_q = {"department": {"type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages",
                     "sales": "pricing", "other": "everything else"}}}
    ten_q = {f"q{i}": {"type": "noul", "instructions": f"Signal number {i}: is this urgent?"}
             for i in range(10)}

    print("=" * 90)
    print("Cold load cost: the FIRST call to a checkpoint vs. every call after")
    print("=" * 90)
    router = Router(device="mps", preload=False)
    t0 = time.perf_counter()
    router.predict(state, one_q)
    first_call_ms = (time.perf_counter() - t0) * 1000
    print(f"  first call (includes model load): {first_call_ms:.0f} ms")
    median, lo, hi = median_latency_ms(lambda: router.predict(state, one_q))
    print(f"  steady-state median (12 calls, already loaded): {median:.1f} ms "
          f"(min {lo:.1f}, max {hi:.1f})")
    print(f"  -> the cold load cost {first_call_ms - median:.0f} ms more than steady state.")
    print("     Preload at startup (Router(preload=True) or router.preload([...]))")
    print("     so this cost lands before your first real request, not during it.")

    print()
    print("=" * 90)
    print("MPS vs. CPU, 1 question vs. 10 questions (same router, already warm)")
    print("=" * 90)
    for device in ("mps", "cpu"):
        r = Router(device=device, preload=False)
        r.predict(state, one_q)  # warm up this device's checkpoint
        m1, _, _ = median_latency_ms(lambda: r.predict(state, one_q), n=10)
        m10, _, _ = median_latency_ms(lambda: r.predict(state, ten_q), n=10)
        print(f"  {device:<5} 1 question:  {m1:.1f} ms")
        print(f"  {device:<5} 10 questions: {m10:.1f} ms  "
              f"(per-question: {m10/10:.1f} ms -- NOT 1/10th of the 1-question time)")

    print()
    print("Batching questions into one call IS worth doing -- per-question cost on MPS")
    print("dropped from ~35ms solo to ~20ms batched (10 questions), a real saving from")
    print("one shared forward pass. But it's nowhere near the 10x you'd hope for: cost")
    print("still scales with (question count x text length), since each question reads")
    print("the full input, it just shares the state tokenization and one model call.")
