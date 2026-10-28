# External GPU hash-table comparisons

The sources here are HKV-side adapters to external libraries; those libraries
are **not** copied into this repository. `CMakeLists.txt` pins new builds to
specific external commits resolved on 2026-10-27:

| Library | Repository | Pinned commit | Citation to retain |
|---|---|---|---|
| WarpCore | [sleeepyjack/warpcore](https://github.com/sleeepyjack/warpcore) | `1a2fe03d438e8dfa1dbd8d9da73f183bd2d72051` | Jünger et al., WarpCore (2020) |
| BGHT/P2BHT | [owensgroup/BGHT](https://github.com/owensgroup/BGHT) | `7584547fcba6e980e8c80beb1694e52310ade16c` | Awad et al., Better GPU Hash Tables (2022) |
| cuCollections | [NVIDIA/cuCollections](https://github.com/NVIDIA/cuCollections) | `3f7ff1043ff538a9941123075a67edaed637c572` | NVIDIA cuCollections project and version |

Use each upstream project's license from its pinned source checkout. These
revisions are **new-run pins**, not evidence that earlier measurements used
them. Before publishing any cross-system number, record each adapter's value
size, successful-operation denominator, failure/overflow behavior, source
hash, dependency revision and full-payload validation. Key-only membership
and 128-byte value-returning lookup are separate contracts.

Configure this subproject explicitly if the external dependencies are
available. CMake `FETCHCONTENT_SOURCE_DIR_WARPCORE`, `..._BGHT` and
`..._CUCO` can point to already-fetched, identity-checked local sources;
missing dependencies mean `NOT_RUN`, not a result. No download or build was
performed during this migration.
