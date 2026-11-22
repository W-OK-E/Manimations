# Hello world — technical smoke test

Status: om approved this design in chat on 2026-11-20 ("Alright proceed with it.").
Low-quality preview rendered and verified. Final render requires preview feedback.

## Claim

This short technical test will check whether this machine can render ordinary text,
LaTeX maths and a simple transition into a playable MP4. It teaches no mathematical
result and is not a 60–90 second gallery animation. Proposed duration: 4 seconds.
Successful rendering alone will not certify mathematical accuracy or the final gallery.

## Maths

Proposed formula: `1 + 1 = 2`, rendered using MathTex. The symbols denote the
integers one and two, addition and equality. Hand check: joining one item with
one item gives two items. Code check to run before scene implementation:
`python3 -c 'assert 1 + 1 == 2'`. No algorithm, dataset, randomness, graph or
inferred scientific claim. Code verification status: passed on 2026-11-20
using Python 3.10.12, agreeing with the hand check.

## Decisions

om approved choices 1–5 below on 2026-11-20; the proposal wording is retained
to preserve the options and rationale reviewed.

1. Content: plain text alone versus text plus a tiny formula. Propose both,
   because plain text would leave the LaTeX path untested. The arithmetic is exact;
   the clip is a technical demonstration, not an illustration of a larger result.
2. Motion: static scene, fade, or shape transformation. Propose a single fade-in
   and a hold. It checks frame generation without implying any mathematical
   transformation. No arrows, morphs or shape-to-equation correspondence.
3. Appearance: propose white text on black, DejaVu Sans for the title,
   default LaTeX typeface for the formula. No external assets or audio.
   Title `Hello, Manim!` centered above the formula; no overlapping objects.
4. Preview: propose Cairo renderer, `-ql` (480p, 15 fps), fixed seed 0,
   local ignored `build/hello-world/` output and no automatic video player.
   The gallery's eventual 720p/30 fps delivery settings are a separate decision.
5. Scope: keep this smoke test out of the gallery manifest. Alternatives were
   using an old TTC_ML clip or starting Huffman now; neither would isolate this
   new environment's basic text/LaTeX/render path.

## Storyboard

| Time | Visible content and motion | Purpose |
| --- | --- | --- |
| 0–1 s | Title and formula fade in together, in fixed positions. | Check text, LaTeX and animation frames. Simultaneous appearance avoids suggesting one text transforms into the other. |
| 1–4 s | Both remain still. | Allow inspection of glyphs, spacing and playback. |

After approval: create scene.py, render the preview,
inspect representative frames and ffprobe metadata, report output size and any
differences from this storyboard. A final render requires om's feedback first.

## Sources

Opened and checked 2026-11-20:

- [Manim CE 0.19.0 package metadata and official usage example](https://pypi.org/pypi/manim/0.19.0/json): Community Edition, supported Python versions, and low-quality CLI usage.
- Versioned documentation URLs for installation/quickstart were inaccessible via
  the web tool; no assertions are based on their unread contents.

Arithmetic needs no research citation. Installed Manim CE 0.19.0 source and CLI
were also checked: `manim/mobject/text/tex_mobject.py` defines MathTex;
`manim/utils/tex.py` uses standalone with babel, amsmath and amssymb;
`manim.constants.QUALITIES` reports low quality as 854 × 480 at 15 fps.
CLI and Text/MathTex/FadeIn imports pass, and PyAV discovers libx264.
No text objects or scene were instantiated. Imports alone are not a successful render.

## Open questions for om

1. After inspecting the low-quality preview, does om want any changes before
   closing the smoke test and proceeding to step 2?

## Preview result (2026-11-20)

Command (from repository root, using the locked project `.venv`):

```sh
UV_CACHE_DIR=.cache/uv uv run --locked manim -ql --renderer cairo --media_dir build/hello-world --disable_caching --format mp4 animations/hello-world/scene.py HelloWorld
```

Output: `build/hello-world/videos/scene/480p15/HelloWorld.mp4`, 17,138 bytes,
H.264/yuv420p, 854 × 480, 15 fps, 60 frames, exactly 4 seconds. The file and
generated intermediates are ignored by git. No final render or publication.

What it actually shows: a black first frame, both lines partially visible at
0.47 seconds, then a white title above a readable LaTeX formula at 1.00 and
3.93 seconds. Both lines remain centered with a clear gap and no clipping.
The scene computes the displayed sum from two integer operands; there is no
core algorithm beyond addition. Implementation uses a fixed seed, groups the
two lines for simultaneous FadeIn, then waits three seconds.

Checks: ffprobe metadata above; all 60 frames decoded with PyAV; FFmpeg decoded
the full video without errors; visually inspected a contact sheet at frames
0, 7, 15 and 59 (`build/hello-world/contact-sheet.png`). The compressed hold
frames are not pixel-identical, so this is not a lossless-output claim. No
visible change in layout or content was observed. At this preview resolution,
the LaTeX formula appears smaller than the title despite both using font_size=48;
it remains readable. No storyboard change was needed.

Conclusion: the smoke-test claim is met for text, LaTeX, transition and MP4
encoding/decoding on this machine. Interactive playback in om's player is still
for om to review. Step 1's environment verification is complete; awaiting that
feedback before further animation work.
