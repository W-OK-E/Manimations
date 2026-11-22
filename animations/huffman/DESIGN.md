# Huffman encoding — 480p inspection preview

Status: preview authorized by om on 2026-11-22: "proceed and ask the next
question only once you have finished the animation at 480p for me to inspect."
This supersedes the earlier per-decision stop for this inspection preview.
Established mathematical choices are retained; remaining visual/text choices
below are provisional for om to inspect. No final render/publication authorized.
Earlier discussion is preserved in DESIGN_DISCUSSION.md.

## Claim

Huffman builds a decodable variable-length code by repeatedly merging the two
least frequent nodes. Shorter encoded text does not guarantee a smaller complete
file: the codebook costs space. More skewed frequencies shorten average codes,
and longer messages amortize tree overhead. Preview target about 110 seconds;
expanded scope includes the requested second example and distribution charts,
so this provisional cut may run longer than the original 60–90 second target.

## Maths

Inputs are ASCII, including case, spaces and punctuation, excluding text-file
newlines. First: `You cannot build what you do not understand`, exactly 43
characters and 18 distinct text symbols. Second: long_text.txt, an original
constructed English paragraph chosen to contain many e's/spaces, exactly 286
characters and 23 distinct symbols. Label this constructed input on screen;
it is not an external corpus or evidence for all English text.

For each symbol s, f(s) is its count. Add END (ID 128) with count one; text
IDs are ASCII 0–127. Priority-queue order is (frequency, minimum descendant
symbol ID). Extract two smallest nodes, place first left/0 and second right/1,
then reinsert their sum. Repeat until one root remains. Symbol codes are the
full root-to-leaf bit strings, retaining leading zeros. Codes are prefix-free
because symbols occupy leaves only. An invalid toy assignment A=0, B=01, C=1
is explicitly labelled invalid; 01 can mean B or AC. Do not show it as Huffman.

Message cost is sum_s f(s)*length(s), including END when indicated. A merge
adds f+g bits. The greedy rule admits an optimal prefix tree by exchanging the
two smallest weights into deepest sibling leaves and contracting that pair.
No claim of optimality among all possible file formats is made.

Wire format, provisionally completed for the authorized preview: preorder tree,
internal marker 0, leaf marker 1 + eight-bit ID, then text codewords, END,
then zero padding to the next byte. No padding between tree and message. The
decoder recursively parses the tree and stops the message on the END leaf,
ignoring final padding. There is no length header, magic number, version or
checksum; comparisons concern this explicit educational format, not ZIP.

A full binary tree with L leaves has I=L-1 internal nodes: 2I edges equal
I+L-1. Tree bits = (2L-1)+8L = 10L-1. Final padding is (-total_bits) mod 8.

| Component | First input | Longer input |
| --- | ---: | ---: |
| ASCII characters / bytes | 43 | 286 |
| 7-bit ASCII baseline | 301 bits | 2002 bits |
| 8-bit byte-storage baseline | 344 bits | 2288 bits |
| Leaves including END | 19 | 24 |
| Tree | 189 bits | 239 bits |
| Encoded text | 166 bits | 967 bits |
| END | 5 bits | 8 bits |
| Final padding | 0 bits | 2 bits |
| Complete | 360 bits / 45 bytes | 1216 bits / 152 bytes |

Hand checks: first word lengths 3+6+5+4+3+2+3+10 plus 7 spaces = 43.
First 18 merge weights: 2,2,2,2,2,2,4,4,4,7,8,8,9,12,15,17,27,44 sum to
171 = 166+5. First tree: 37+19*8=189. Total: 189+171=360=45*8.
Longer histogram grouped hand sum: 81+58+33+26+18+9+8+7+7+6+5+4+4+4+
3+3+3+2+1+1+1+1+1=286. Longer tree: 47+24*8=239; tree+message
239+975=1214, plus 2 padding bits = 1216=152*8. Code verifies all values,
sum of merge weights equals weighted path cost, all-pairs prefix checks, tree
round-trip and byte-packed end-to-end decoding for both inputs.

Longer hand merge-weight trace: 2,2,2,4,4,6,7,8,8,11,13,14,16,17,24,30,
35,50,63,85,121,166,287. Cumulative sums end at 975; this agrees with the
independent weighted-length sum (967 text + 8 END) from the actual codebook.

For text-only distribution charts, p(s)=f(s)/N, with N text characters,
excluding END. Both panels use percentages on the same 0–35% scale; symbols
are ranked within each panel, space shown as SP, and all symbols included.
There is no raw-count scale mismatch. Alphabet sizes differ and are disclosed.
H(p)=-sum p(s)log2 p(s), in bits/character, is empirical single-character
entropy, not a bound on every compression method for this particular sentence.
Text entropy/mean text code length are respectively 3.761/3.860 (short) and
3.348/3.381 (long). Mean excludes END's own bits but uses the END-inclusive
tree. The source bound is tested separately with END-inclusive probabilities
q(s)=f(s)/(N+1): H(q)<=mean_with_END<H(q)+1. Header/padding excluded.

Computed assets: prepare_preview.py -> preview_results.json and ignored
build/huffman/{short,long}.huff. check_example.py retains the established first
example. Scene reads the cache; displayed counts, codes and costs come from it.
No numerical/randomly generated data are hand-drawn. Seed fixed to 0.

## Decisions

1. **Exact:** retain om's sentence, both ASCII baselines, tree merge/tie/edge
   conventions, eight-bit identities, END termination and serialized-tree cost.
   Alternatives and reasoning are in DESIGN_DISCUSSION.md.
2. **Provisional, exact computation on constructed text:** use long_text.txt
   rather than an external excerpt or controlled equal-length string. It follows
   om's request for a longer second input; disclose intentional repetition,
   changed length and changed alphabet. Measure rather than presume skew.
3. **Provisional format:** contiguous tree+text+END, final-only zero padding.
   Alternative separate section alignment wastes bits and needs a stated rule.
   Provisional output is a simple educational container, not a standardized one.
4. **Provisional visual:** dark navy, off-white text, teal for text/code paths,
   amber for tree overhead, red for ambiguity/expansion, purple for END.
   Alternatives: neutral monochrome or theme matching the website. Colour is
   redundant with text labels; no external graphics/fonts. DejaVu Sans/Mono.
5. **Provisional visual:** show all 18 exact merges, early merges slowly and
   later ones faster. Queue boxes show leaf names or explicit subtree labels;
   the final full tree is computed from the codebook. A highlighted traversal
   makes selected codes legible even when the full 19-leaf overview is dense.
   Do not morph a leaf into an ancestor while claiming it remains a character.
6. **Provisional charts:** two ranked percentage histograms with identical
   vertical scales; show all characters and disclose END is excluded. Separate
   exact byte/bits breakdown explains overhead. Alternatives raw counts or
   separately auto-scaled charts risk mixing length and distribution.
7. **Provisional narrative:** silent, readable on-screen explanations; no voice,
   audio or third-party figures. Include the surprise "Wait—what just happened?"
   and entropy/average-length comparison. Explicitly state both length and skew
   changed; never claim uniform/short strings universally fail to compress.
8. **Preview only:** Cairo -ql, 854×480 at 15 fps. About 110–125 s to fit the
   expanded story. No 720p/final delivery export or gallery integration yet.

## Storyboard

| Approx. time | Shot and motion | Why |
| --- | --- | --- |
| 0–6 s | Sentence appears under Huffman title. | Establish exact input before numbers. |
| 6–14 s | You -> three eight-bit ASCII codes; both full-message baselines. | Show fixed field width, including leading zeros. |
| 14–21 s | Actual frequencies in ranked tiles, frequent/rare symbols emphasized. | Motivate short codes without yet assigning an invalid codebook. |
| 21–31 s | Labelled invalid A/B/C code; 01 splits as B or AC; leaf-only valid toy tree. | Demonstrate ambiguity and the structural prefix condition. |
| 31–51 s | Sorted queue, highlight two minima, show sum/parent, reinsert; all 18 merges. | Show the real greedy algorithm, including internal subtree weights. |
| 51–60 s | Full exact tree, then highlighted root-to-leaf paths and codes. | Ground variable lengths in actual edges. |
| 60–68 s | Encode You and show complete text bit total; END stops decoding. | Connect tree to message, distinguish identity 128 from END code. |
| 68–79 s | Payload shrinks, then tree/END enter size sum; 45 bytes exceeds 43. | Reveal real accounting rather than imply payload equals file. |
| 79–86 s | Show complete longer constructed paragraph and input sizes. | Make the new example inspectable and provenance explicit. |
| 86–98 s | Side-by-side percentage histograms rise on same scale. | Compare skew without confounding raw counts; disclose alphabet sizes. |
| 98–108 s | Full second breakdown including padding; ASCII 286 vs Huffman 152. | Count every bit of the chosen format. |
| 108–120 s | Compare entropy/mean and overhead per character; conclusion and credits. | Separate shorter average codes from amortized header costs. |

Timing is provisional and will be measured from the rendered scene. Motion
always follows real computation or is purely a labelled transition; no data
curve is invented. Viewer should pause a preview to inspect the full tree.

## Sources

Opened and checked, no third-party figures reused:

- [RFC 20](https://www.rfc-editor.org/info/rfc20/): seven-bit ASCII and its
  eight-bit-byte embedding. Previously checked 2026-11-20, reopened 2026-11-22.
- [MIT 6.046J, lecture 19 (2012)](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2012/388115265a456321c4a5d19dc9e05281_MIT6_046JS12_lec19.pdf):
  prefix trees, weighted depth, greedy merges and exchange/contraction proof.
  Previously read 2026-11-21, reopened 2026-11-22.
- [MIT 18.200, lecture 19 (2024)](https://ocw.mit.edu/courses/18-200-principles-of-discrete-applied-mathematics-spring-2024/mit18_200_s24_lec19.pdf):
  opened/read 2026-11-22; empirical distribution/entropy interpretation and
  expected prefix-code length bound. Do not present memoryless-source bounds
  as a universal minimum file size for correlated natural language.

Original Huffman 1952 paper was located but PDF access failed earlier; do not
claim to have read it. Our serialization and examples are original computations.

## Open questions for om — after preview only

Inspect pacing, crowded tree labels, second-input choice, percentage charts,
the expansion reveal and provisional packing format. No additional question
will be asked before the authorized 480p preview is complete. Final render
and any gallery publication remain pending om feedback.

## Rendered preview report — 2026-11-22

Preview complete: `build/huffman/videos/scene/480p15/HuffmanEncoding.mp4`.
H.264, 854×480, 15 fps, 1,796 frames, 119.734 seconds, 2,699,524 bytes.
Silent with on-screen text. No final-quality render, gallery insertion, commit
of generated binaries, push or publication.

Actual result versus claim: the rendered cut presents the exact first input,
both ASCII baselines, actual counts, a clearly invalid toy prefix collision,
a valid leaf-only toy, all 18 real merges including END, the full computed tree,
highlighted actual codewords, 166 text bits plus END, and the full 45-byte versus
43-byte expansion. It shows the complete 286-character constructed longer input,
both same-scale percentage distributions, all longer-file components and the
152-byte versus 286-byte comparison. Closing values distinguish average code
length/empirical entropy from tree overhead per character, and disclose that
alphabet and length also changed.

Validation: all numeric data regenerated and byte-packed decoding assertions
passed for both inputs. Full final MP4 decoded with PyAV and FFmpeg without
errors; ffprobe verified resolution/fps/codec/duration. One full-size sample
from every shot was visually inspected (12 shots, contact sheet plus dedicated
chart/tree/closing-table views). First cut exposed misaligned closing column
headers and a crowded peak annotation; both were corrected and the revised
cut rerendered/rechecked. No visible overlap or clipping was found in final
samples. This is not a claim to have watched every transition in real time.

Limitations for om review: the full-tree overview and some axis labels are small
at 480p; larger selected-path code labels supplement the overview. The cut is
about two minutes rather than the original 60–90 second target, to cover the
expanded story. The longer text and colour/layout/packing choices are still
provisional under om's preview-first authorization. Full serialization is
accounted for numerically rather than animated bit by bit. No narration/audio.

Reproduce:

```sh
python3 animations/huffman/prepare_preview.py
UV_CACHE_DIR=.cache/uv uv run --locked manim -ql --renderer cairo --media_dir build/huffman --disable_caching --progress_bar none animations/huffman/scene.py HuffmanEncoding
.venv/bin/python scripts/inspect_huffman_preview.py
```

Local ignored QA artifacts: `build/huffman/shot_times.json`,
`inspection-report.json`, `inspection-01.png` through `inspection-12.png`,
`contact-sheet.png`, and the exact 45-/152-byte packed examples. Await om's
inspection/changes before final rendering or resuming decision quizzes.
