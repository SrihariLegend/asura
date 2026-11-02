# Asura correctness audit — 2026-11-02

## Scope and outcome

Baseline: `33a9fd5`. The complete extracted user/assistant design narrative and
compaction summaries from the prior design session were reviewed in four
chronological passes. Historical tool-result records were not included in that
extraction; historical execution, proof, and strength claims were not assumed true.

**Concrete correctness defects were reproduced and repaired. The playing-strength
problem is not established as solved.** The search-fix match scored 46.875% against
the original build, with a negative point estimate and a confidence interval that
includes zero.

The ATN-1 architecture, bundled weights, search tuning parameters, and
`scale_atn1_evaluation` leaf-scaling function were preserved. No material floors,
root optimism, tablebase bonuses, retraining, or parameter tuning were introduced.

Checkpoint SHA-256, unchanged:

```text
1402eeb715752a7ec4c85f72ea2d0042c819157588643525f6bb432d727cc807
```

## Engine repairs

| Defect | Reproduction and repair | Main files |
| --- | --- | --- |
| Compensation changes on a TT hit | `ThreadData::evaluate_psqt_only` returned ordinary material instead of learned PSQT. In the first 10,000 dataset records, mean absolute discrepancy was 20.35 cp, maximum 209 cp, with 113 discrepancies at least 100 cp. It now uses the network component. Negative quantization also uses `/ 32`, matching the full forward pass, rather than arithmetic right shift. | `src/thread.rs`, `src/nnue.rs`, `tests/nnue_search.rs` |
| Correction applied twice during singular verification | Same-ply verification reused an already-corrected stack evaluation as raw, corrected it again, and overwrote its parent's value. A material-only fixture changed the expected −205 to −110; native search also reproduced the defect. Verification now reuses the corrected value directly. | `src/search.rs` |
| TT key/payload races | Distinct verification signatures returned one another's complete payloads under contention. Concurrent packed-key stores could also revert another slot's signature. A cluster try-lock now protects both reads and writes, with RAII release on early returns. Contention becomes a cache miss/dropped write instead of corrupted data; the 32-byte layout is retained. | `src/transposition.rs`, `tests/transposition.rs` |
| Expired mate becomes a fake exact score | A cached mate at halfmove 96 became an `Exact` +31743 at halfmove 99, although every legal move immediately drew. A zeroing-move mate demonstrated that simply lowering its score was also wrong. When clock compatibility cannot be established, the score/bound is discarded, retaining other usable entry information. | `src/transposition.rs`, `tests/transposition.rs` |
| Reuse of clock-dependent raw evaluation | Different clocks share a TT hash bucket, but NNUE raw evaluation already contains rule-50 damping. A cached +570 at clock 24 was reused instead of +517 at clock 31. Search now refreshes NNUE raw evaluation outside the first, wholly undamped clock bucket. Tests cover both directions across the damping threshold. | `src/thread.rs`, `src/search.rs`, `tests/nnue_search.rs` |
| Lost quiescence PV | A reported mate-in-three returned only `d2h6`, omitting searched `h8g8 h6g7`. PV quiescence nodes now clear and update their PV rows. | `src/search.rs`, `tests/quiescence.rs` |
| Fixed-depth stop and depth reporting | Plain single-thread `go depth` bypassed the shared stop mechanism. It now uses the managed worker path. Limited/aborted searches no longer advertise an uncompleted requested depth. | `src/uci.rs`, `tests/uci.rs` |
| Worker lifetime/completion unsoundness | The public safe scope extension allowed forgotten handles to outlive borrowed captures; a shared completion Boolean also let one task satisfy another's join. Borrowing dispatch is now private and explicitly unsafe, with per-job completion and internally owned join-on-Drop handles. Unwinding stops sibling searches before joining them. Persistent OS workers and state are retained. | `src/threadpool.rs`, `src/threadpool/worker_tests.rs` |
| Scalar build failure | Non-AVX2 PK inference referenced an unimported `PK_HIDDEN`. The conditional import is restored. | `src/nnue/accumulator.rs` |
| Misleading validation passes | The external-oracle test silently succeeded without fixtures. It is now explicitly ignored by default and fails if requested without its required environment variables. The bundled-model test uses the manifest directory and fails if its model is missing. The Chess960 fixture now actually places the king on f1 and verifies that the promised king/rook swap is generated. | `tests/nnue.rs` |

Worker API safety was not demonstrated to cause normal match losses. Ordinary
TT signature collisions remain possible by design. These repairs are not a
proof that every selective-search or cross-clock bound approximation is sound.

## External trainer repairs

These two files **outside the repository were modified**:

- `/home/tom/.cache/asura_spsa/nnue_lab/train_atn1_cuda.cu`
- `/home/tom/.cache/asura_spsa/nnue_lab/cuda_trainer_atn1.cu`

The reproducible patch is retained in [`trainer-contract.patch`](trainer-contract.patch).
It has already been applied locally; do not apply it twice. The source version
that originally produced the bundled checkpoint is not established, so these
current-source failures do not prove historical checkpoint corruption.

1. **FT truncation:** a position reached after 136 legal plies has 129 active rows
   per perspective; both trainers silently kept only 128. With fixed neural
   weights, the omitted features changed an evaluation from −177 to −148 cp.
   Capacity is now 512, FT counts are 16-bit, and insertion is checked rather
   than silently truncated.
2. **PK overwrite:** a structurally valid stress position activates 100 PK
   features. The 96-entry buffer overwrote FT counts: 58/58 became 98/1.
   UBSan reproduced the out-of-bounds write. Capacity is now 128 with checked
   insertion. A complete reachability trace was not constructed for this PK fixture.
3. **Scaling disagreement:** production training allowed fortress damping with
   rooks and pawn levers, unlike the intended engine condition. The predicate
   now agrees. A minimized fixture changed from production 153 / engine 204 cp
   to 204 / 204. The previously observed flag disagreements disappeared in a
   resample of 393,216 records across six packed files.

The conservative capacity assumptions are at most 32 PSQ rows, 8 occupied attack
targets per piece, 120 pawn pairs and 16 passers; PK has at most 16 pawns, 2 kings,
96 hole indicators and 6 other indicators. Checked insertion aborts/traps if a
capacity is nevertheless exceeded. These structural changes do not change the
PackedBoard or `.nnue` file formats.

Run the durable, CPU-only regression checker:

```sh
python3 tools/check_trainer_contract.py /path/to/nnue_lab
# Individual failures can be isolated with --case ft, --case pk or --case scaling.
```

The checker compiles the actual extraction functions from both supplied trainer
sources with UBSan. All three cases fail against the original sources and pass
against the repaired sources. Both complete CUDA source files also compiled with
CUDA 13.3 for `sm_120`. **No GPU training kernels or optimizer steps were run.**

## Validation

- `cargo test`: **163 passed, 1 explicitly ignored**.
- Generic x86-64, AVX2 disabled: **163 passed, 1 explicitly ignored**.
- Explicit external-oracle test: **10,000 CPU-oracle comparisons passed** with the
  bundled checkpoint. This is not production GPU/QAT/gradient/export parity.
- **412,960** additional exact comparisons between incremental and refreshed
  scalar evaluation, including accumulator lanes and PSQT, using random and
  trained weights through lazy move/null/undo/branch/reset paths and the snapshot ring.
- Native abort sweep: **1,040 cases, 778 actual stops**; board, exclusion markers,
  NMP state, score range, and returned PV legality passed.
- Extreme TT contention: **zero mismatched payloads in 50,769 returned hits** over
  8 × 100,000 write/read iterations. This deliberately high-contention hit rate is
  not representative of real games.
- Native worker smoke: **12 searches using 1/2/4 workers**, checking legal PVs,
  node-shard totals, reset behavior, and status.
- All four initial legal conversion probes were won by both baseline and repaired
  search at 20,000 nodes/move against a defending engine. This small sample does
  not establish universal conversion strength or support a blanket failure claim.
- `lean proofs/search_invariants.lean` passed. These are narrow arithmetic and
  capacity implications, **not a verification of the Rust program or chess strength**.
- `git diff --check` passed. Existing whole-repository formatting drift was not
  broadly rewritten.

Re-run scalar tests without replacing native build artifacts:

```sh
CARGO_TARGET_DIR=/tmp/asura-scalar \
RUSTFLAGS='-C target-cpu=x86-64 -C target-feature=-avx2' cargo test
```

The optional oracle test requires matching checkpoint/data/result fixtures:

```sh
ASURA_GATE2_MODEL=/path/to/checkpoint.nnue \
ASURA_GATE2_DATA=/path/to/packed-positions.bin \
ASURA_GATE2_ORACLE=/path/to/10000-little-endian-i32-results.bin \
cargo test --test nnue gate2 -- --ignored
```

The normal test profile exercises panic unwinding. Release still uses
`panic=abort`; no release panic-recovery guarantee is made.

## Playing strength and throughput

The saved search-correctness snapshot played the baseline using:

- **10+0.1**, one thread and 16 MiB hash per engine;
- **128 games / 64 color-reversed opening pairs**, seed `20261102`;
- the existing development opening book, not a fresh held-out book;
- no score-based draw/resignation adjudication; a 300-move safety limit.

Result: **19 wins, 27 losses, 82 draws; 60/128 points (46.875%)**.
Runner-reported estimate: **−21.74 ±30.76 Elo**.
Pentanomial counts: **[1, 17, 36, 9, 1]**.

An independent PGN replay verified all 128 games, both colors and identical
starting FENs within each pair, every win ending in mate, and every draw having
an ordinary chess termination. No unexplained adjudications were found.

**This does not establish improvement. The point estimate is worse.** Do not
attribute that outcome to a particular fix without controlled follow-up testing.
The match snapshot predates the worker API hardening. The final build matched
its score, completed depth, node count and best move on 24 fixed-50,000-node
searches; this is a compatibility check, not a substitute for a new clocked match.

Six alternating, single-core `bench 15` runs of baseline/final gave median
**901,379 / 901,796 NPS**: effectively unchanged. Node totals were deterministic
within each build, 782,804 / 540,109. Different fixed-depth tree sizes are not
playing-strength evidence.

No new external-opponent gauntlet or LTC strength result is claimed.

## Remaining work

- Establish production GPU forward/QAT/export parity, not just independent
  quantized-oracle agreement, before further training.
- Define and validate endgame retention/target policy. The current packer still
  rejects all three-piece positions and raw teacher scores beyond ±1500 cp.
  Those exclusions were reproduced, but their causal contribution to losses was
  not established; simply deleting every filter is not a justified repair.
- Measure the corrected search on held-out paired openings and representative
  node/time controls before retuning its coupled heuristics.
- Keep accuracy/correctness regressions separate from strength-selection criteria.

Detailed local investigation artifacts and frozen binaries are under
`/tmp/asura-audit/`; history reports are under `/tmp/asura-audit-history/`.
These are temporary artifacts, whereas the regression tests, trainer checker,
patch, and arithmetic specifications are retained in the repository.
