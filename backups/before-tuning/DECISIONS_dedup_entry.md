## Decision: repeated bookings are flagged at load, deduplicated for modelling, and reported both ways

**Date:** 2026-11-16   **Status:** open until the client says whether repeats are real bookings

**What we found.** 31,994 of 119,390 rows (26.8%) are exact repeats of an earlier row in every column the export holds. `is_dup` marks them; nothing is deleted at loading.

**What we chose.** The deduplicated data is the primary training and scoring set. `row_key` (hash of every column except `is_canceled`) groups identical rows. The split stays by arrival date, so identical rows always fall in the same period (checked in the split stage).

**What it is worth (same code and seed on each variant).**

| | Deduplicated (primary) | Raw |
|---|---|---|
| Base rate, train period | 22.2% | 35.8% |
| Headline model and buffer | 3 Logistic with constant 0 | 3 Logistic with constant 0 |
| Proof: walks per 1,000 / ceiling met | 6.72 / True | 9.26 / True |
| Proof: saving vs the flat rule | 34.6% | 52.3% |

**What the repeats look like.** Repeated rows cancel at 63.1% against 27.5% for first copies, and 92.9% of Non Refund rows are repeats.

**Does the buffer recommendation flip?** No. The same headline model and buffer rule come out on both variants. Buffers that differ between variants: 1 Flat rule, 2 Segment table.

**Still open.** Ask the client whether repeated rows are separate bookings. If yes, the raw variant is closer to their books.
