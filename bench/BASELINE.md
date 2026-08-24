# Benchmark baseline of record

`bench/bench_track.sh compare` with no arguments compares the current commit
against whatever `bench/results/baseline.json` points at. That directory is not
tracked — results are local to a machine, because a timing recorded on one box
says nothing on another. This file is the tracked half: which commit the
baseline of record was taken at, on what box, and under what protocol, so a
comparison against it can be reproduced or refused.

Re-record with `bench/bench_track.sh baseline` and update the entry below in
the same commit.

## Current

| | |
| --- | --- |
| Commit | `7bb672fd` (`bench: refuse to measure on a busy box, and add an alternated compare`) |
| Recorded | 2026-08-23 |
| Box | Linux x86_64, Fedora 44, 32 GB |
| Toolchain | conda-forge clang (`linux-clang-conda` preset), Release |
| Protocol | sequential, 9 repetitions, google-benchmark medians |
| One-minute load at start | 0.49, below the 0.6 idle gate (recorded in the result's own metadata) |

Library-identical to `main` at `d22ec85c`: every commit between the two touches
only tests, the example CTest registration, and this script, none of which
`bench_all` compiles.

The previous baseline of record was `fbeb8c7a`, which predates the idle gate
and the load metadata, so a comparison against it warns rather than refuses.
Against it, this recording shows no regressions on any of the 128 benchmarks,
the largest movement being `BM_Phase_Transcribe_16seg` at +1.2%.

## Reading a comparison against this baseline

`BM_Phase_Transcribe_64seg` is bimodal on this box: it settles into a mode for
the length of a run, and which mode it picks varies between runs of the same
binary. Three recordings of the identical binary, made within an hour of each
other, put it at 252 us, 322 us and 367 us. A sequential comparison can
therefore report a large change on it with no code change behind it — the 367 us
recording, taken with five repetitions while the box had not fully settled,
reads as a 45% regression against the 252 us one.

Two things keep that out of a comparison. The idle gate refuses to record until
the box is quiet, which is what the 252 us recording above was taken under. And
`compare --alternate` runs both arms interleaved, one repetition each per round,
so a mode flip lands in both arms rather than in the difference between them:
alternating this build against itself over nine rounds put the cell at 322 us on
both arms, 0.0% apart.

**The consequence for the pre-merge gate, stated rather than left to be
rediscovered.** This baseline happens to have been captured in the low mode.
A plain `compare` against it will therefore read anywhere from about 0% to
about +45% on `BM_Phase_Transcribe_64seg`, depending only on which mode the
new recording lands in — with nothing behind it. On a gate step whose rule is
"justify any regressions in the PR description", that cell is the one to expect
to have to justify, and the justification is this paragraph.

The practical rule: a difference on a transcription cell is not evidence until
`compare --alternate` reproduces it. Re-recording the baseline does not fix
this, because the mode is not a property of the recording that can be pinned —
three recordings of one binary landed in three different places.
