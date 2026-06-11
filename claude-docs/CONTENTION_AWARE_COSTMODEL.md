# Contention-Aware Reuse-Distance Cost Model

**Date:** 2026-06-08
**Status:** design + incremental implementation
**Origin:** the cost-model review (`COSTMODEL_REVIEW_FINDINGS.md`) + the fission
contention study. The model was always intended to reason in **reuse distance**
(`estimateInterveningFootprint`); this extends that primitive to be
**contention-aware** and unifies the fission decision onto it.

## The principle

> Model the cache you are **guaranteed**, not the cache you nominally have; price a
> reuse that exceeds it at **worst-case (contended) bandwidth/latency**.

A shared LLC is not yours — a co-tenant can evict your line between its production and
its reuse. So residency must be judged against an **effective** cache: private levels
(L1, L2 on Zen4) at (nearly) full size; the shared LLC derated by the number of
sharers. This is the axis BLIS / Pluto / roofline all omit (they assume exclusive
cache), and it is the regime real deployments (SPEC rate, multi-tenant, parallel make)
actually run in.

## The primitive already in the model: reuse distance

`estimateInterveningFootprint(def, use)` = distinct bytes touched between a value's
production and its consumption = **reuse distance**. `decideBufferStrategy` already
prices:
- **keep**  = `numConsumers · estimateLoadLatency(bufferSize + reuseDistance)`
- **recompute** = `numConsumers · (leafLoad + operandReloadPenalty)`  + `numConsumers · alu`

where `operandReloadPenalty` is itself a reuse-distance test on the *source* (is the
recompute's input still resident when re-read?). The keep side asks "does the buffered
value survive its reuse distance?"; the recompute side asks "does the source survive
being re-read N times?". **Both are reuse-distance vs cache.** MemoryFission currently
ignores all of this (hard-wired `l1Latency`); the fix is to route it through the same
machinery.

## The one change that makes it contention-aware

`estimateLoadLatency(bytes)` maps a reuse-distance/working-set to a latency tier by
comparing against cache sizes. Make those sizes **effective**:

```
L1 (private):  full l1Size
L2 (private):  full l2Size            (optionally × occupancy margin)
L3 (shared):   l3Size / llcSharers    <-- derated by co-tenants
> effective L3: memLatency
```

One change; every consumer of the reuse-distance machinery (DR keep/recompute, fission,
tiling) becomes contention-aware for free. `llcSharers = 1` reproduces the old
(isolated) model exactly — backward-compatible.

## How it captures the three measured fission regimes

Measured (Ryzen 7950X3D core 8, vs recompute, 3 consumers, sqrt+div):

| total working set | isolated | contended | model verdict (effective cache) |
|---|---|---|---|
| ≤ private L2 (≤~0.5MB) | 2.0× help | **2.0× help** | reuse distance ≤ l2 (private, never derated) → keep cheap → **FISSION** ✓ |
| ≈ L2 … few×L3 (0.8–1.6MB) | ~1.1× | **0.4–0.5× hurt** | buffer tips set past private L2 into derated L3 → keep load = memLat → **SKIP** ✓ (near-boundary; calibration-sensitive) |
| ≫ L3 (4.6MB) | 0.72× hurt iso | **2.58× help** | both buffer *and* re-read source exceed effective cache → load tiers equal → fission wins on **saved compute** (1× vs N×) → **FISSION** ✓ |

The **WS4.6M reversal** falls out of the reuse-distance framing: when recompute must
re-read the source N times at a reuse distance that exceeds the effective cache, its
`operandReloadPenalty` is paid N times; fission reads the source **once**. With both
buffer and source evicted the per-load tiers match, and the `(N−1)×` saved compute
decides for fission — the original recomputation-avoidance rationale, re-emerging under
contention. The model gets this right *because* it prices source re-reads by reuse
distance, not by a flat assumption.

## Parameters (added to `CacheParams` / pass options)

- `llcSharers` (unsigned, default 1): co-tenants of the shared LLC. Derates effective
  L3. Set to the LLC's core count (Zen4 CCD = 8) for a worst-case shared estimate;
  1 for a dedicated/isolated run.
- (future) `l2Occupancy` fraction for the near-boundary mid regime (the aggregate-
  pressure effect the per-value reuse distance under-counts); a calibration knob.
- (future) per-level `bandwidth` for a roofline traffic term, derated by `llcSharers`.

## Acceptance test (grounded in measurement, not assertion)

With `llcSharers` set to a contended value, the fission decision must:
- FISSION when total WS ≤ private L2,
- SKIP in the mid regime (buffer overflows private L2, relies on contended L3),
- FISSION at large WS where recompute re-reads an evicted source N× (compute saving).

`llcSharers = 1` must reproduce today's isolated decisions (no regression on the
existing lit tests).

## Implementation status

1. **[foundation — DONE]** `CacheParams.llcSharers` + effective-L3 in
   `estimateLoadLatency` (`l3Size / llcSharers`; private L1/L2 full). `llcSharers=1`
   reproduces the isolated model. Lands the contention axis for every reuse-distance
   consumer at once.
2. **[fission — DONE + VALIDATED]** MemoryFission's `l1Latency` fiction replaced by the
   reuse-distance rule: `FISSION iff totalWS ≤ l2OccupancyPct·privateL2  OR
   perConsumerReuseDist > l3Size/llcSharers`. `totalWS` and the source re-read distance
   are estimated from the distinct memrefs the consumer loops touch. New options
   `l3-size`, `llc-sharers`, `l2-occupancy-pct`; `l2-size` default corrected to the
   private-L2 size (1 MiB). Dynamic-trip candidates fall back to the legacy estimate
   (existing lit tests unchanged).

   **Acceptance test PASSED** — decision vs measured help/hurt (7950X3D), all five WS
   points at both contention levels:

   | WS | iso measured | model (s=1) | contended measured | model (s=20) |
   |---|---|---|---|---|
   | 200K | help | FISSION ✓ | help | FISSION ✓ |
   | 400K | help | FISSION ✓ | help | FISSION ✓ |
   | 800K | help (1.24×, marginal) | SKIP (safe) | hurt 0.47× | SKIP ✓ |
   | 1.6M | hurt | SKIP ✓ | hurt | SKIP ✓ |
   | 4.6M | hurt 0.72× | SKIP ✓ | **help 2.58×** | **FISSION ✓** |

   The WS4.6M row is the headline: the *same* candidate flips SKIP→FISSION as
   `llcSharers` rises, because the source's re-read distance crosses the derated LLC —
   the reuse-distance term doing exactly the job it was designed for. Pinned by
   `test/MemoryFission/fission-contention-aware.mlir`. Only the 800K-isolated cell is
   off (model SKIP vs marginal 1.24× help) — the safe direction, since it regresses
   0.47× the moment the machine is loaded.

3. **[generalize — in progress]** DR keep/recompute already consults
   `estimateLoadLatency`, so it inherits the effective-L3 derating; a `dr-llc-sharers`
   option threads the contention assumption into `CacheParams`. AffineRegisterBlock
   cache-tiling has no cost-model hook yet (separate work — would size tiles against the
   guaranteed/private budget).
4. **[calibrate — future]** `l2OccupancyPct` and a per-level bandwidth term via the
   Nelder-Mead harness against the measured regime table; the 800K boundary is the
   calibration target.
