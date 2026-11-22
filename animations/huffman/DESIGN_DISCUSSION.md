# Huffman encoding — design in discussion

Status: INCOMPLETE, NOT APPROVED FOR IMPLEMENTATION OR RENDERING.
2026-11-20: om requested starting the concepts after accepting the smoke test,
and wants quizzes and participation in every maths/CS/animation decision.
This file records the discussion; it is not yet the completed design required
by CODEX_BRIEF.md. No scene code until all sections are complete and approved.

## Claim

om's chosen narrative (2026-11-20): encode a concrete sentence using ASCII;
compare it with Huffman while showing the construction under the hood. Motivate
short codes for frequent symbols, expose ambiguous variable-length codes, and
explain why assigning symbols only to leaves prevents prefix collisions.
Both ASCII baselines and the payload-first/separate-overhead presentation are
approved. END termination and eight-bit leaf identities are approved (2026-11-22):
ASCII 0–127 plus END 128. Current checked results: 166 text bits + 5 END bits +
189 tree bits = 360 bits before any additional format overhead. Final packing
convention, claim wording, storyboard and runtime remain pending. The earlier
164-message/161-tree/325-combined-bit results retained below are the historical
**without-END intermediate example**; example_results.json now contains the
END-inclusive results and a labelled without_end_reference.

Narrative extension selected by om, 2026-11-22: reveal that the short sentence's
complete representation is larger, ask "Wait—what just happened?", explain
overhead and frequency skew, then show a second message that compresses better.
om subsequently chose the longer-text direction with a graph/distribution chart
to explain what changed. The actual second input and chart design remain to choose. Do not promise
that every short or equal-frequency input expands: length amortizes tree cost,
skew affects weighted code lengths, and alphabet size affects comparisons to
ASCII. Even a uniform small alphabet can beat ASCII for a sufficiently long input.

## Maths

Chosen text, verbatim, excluding quotation marks and without a trailing newline:

`You cannot build what you do not understand`

Preserve the uppercase Y, lowercase y, and every space. No normalization was
requested. Hand count: word lengths 3 + 6 + 5 + 4 + 3 + 2 + 3 + 10 = 36,
plus seven spaces = 43 characters. Python Counter/ASCII encoding independently
agrees: 43 characters, 18 distinct symbols, all in ASCII. Verified 2026-11-20.

Frequencies checked by code: space 7; n and o each 5; d, t and u each 4;
a 3; Y, b, c, e, h, i, l, r, s, w and y each 1. The grouped sum is
7 + 2×5 + 3×4 + 3 + 11 = 43. A reusable verification script accompanies this design.

Approved baselines (om, 2026-11-21): 7-bit ASCII gives 43×7 = 301 bits;
ASCII stored in 8-bit bytes gives 43×8 = 344 bits (43 bytes). ASCII's fixed
symbol assignments do not adapt to this sentence's frequencies. Do not imply
that ASCII itself requires eight bits, or that byte padding is the sole
opportunity for compression. See RFC 20 below.

om's leaf argument is correct with the following explicit conventions: bits
label edges (distinct 0/1 labels for the two children), a codeword is a
root-to-symbol path, and symbols label leaves only. If one distinct symbol's
codeword were a prefix of another's, its node would be an ancestor of the
other symbol's node and therefore would have a child. It could not be a leaf.
Thus no symbol's codeword is a prefix of another's. This is a structural
prefix-code argument, not by itself a proof that a particular tree minimizes
encoded length. om approved the two-smallest-weights merge rule on 2026-11-21;
the optimality reasoning and source are recorded below.

The eventual comparison must distinguish encoded message bits from any
codebook/tree, padding and framing overhead. On 2026-11-21 om chose sending a
serialized tree for a self-contained message and approved showing message bits
first, then the separate decoding overhead. Tree format is now chosen below;
message framing and padding remain to choose. No complete-file savings claim
is established yet. A tree
already shared out of band would be a different accounting scenario.

Remaining definitions, formulas, algorithm and Huffman worked example: pending
discussion with om. Every final example must be checked by hand and code.

Merge-cost discussion, 2026-11-21: om correctly reasoned that combining nodes
covering f and g occurrences adds one bit to each occurrence, hence f + g bits
to the whole message. A zero edge costs one bit just as a one edge does; leading
zeros are part of the codeword. Internal node weights count all occurrences
under that subtree, rather than necessarily a single character's frequency.
Left/right bit assignment remains undecided. om has resumed the Huffman design
discussion after the gallery-learning exercises.

Greedy merge rule, approved by om 2026-11-21: repeatedly remove the two
smallest-weight current nodes, combine them under a new parent whose weight is
their sum, and reinsert that parent until one root remains. Weights are occurrence
counts for the chosen text, including subtree totals after merges. Characters
label leaves, never internal nodes; depth means number of edges from the root.
Shorter paths reduce the weighted cost B = sum_s f_s * depth_s, where s ranges
over symbols and f_s is the occurrence count of s. This is message-bit cost,
excluding serialization overhead. Deterministic tie-breaking is defined below;
edge-bit labels are now approved: first extracted child goes left with edge 0,
second goes right with edge 1 (om, 2026-11-21).

Tie rule (2026-11-21): om accepts either proposed reproducible method; chose
character-code order. Queue key is (occurrence weight, smallest descendant ASCII
character code). Each active subtree contains disjoint symbols, so the secondary
keys are distinct. This rule handles leaf/leaf, leaf/subtree and subtree/subtree
ties without insertion-order dependence. The first pair is Y and b, each weight 1.
The same input/tie rule fixes merge groups and lengths; approved edge labels
now also fix the bitstrings. om correctly explained that arbitrary equal-weight
choices can change codewords but preserve the optimal total message bits.

Worked merge-weight trace, checked by hand and by `check_example.py`:
2, 2, 2, 2, 2, 3, 4, 4, 6, 8, 8, 9, 11, 15, 17, 26, 43.
Hand sequence: five 1+1 merges; 1+2; two 2+2 merges; 3+3; two 4+4
merges; 4+5; 5+6; 7+8; 8+9; 11+15; 17+26. The 17 parent weights
sum to 164 bits, matching the independent weighted-length sum in code.
Cached exact output, including all merge groups and lengths, is
`example_results.json`; regenerate with
`python3 animations/huffman/check_example.py > animations/huffman/example_results.json`.
This result excludes serialized tree, framing and padding costs; it is not a
complete-file size. Exact codewords and concatenated message bits are now in the
cached output; no scene code or rendered frames yet.

Edge convention approved 2026-11-21: om correctly notes that either 0/1 labelling
preserves prefix-freeness if full codeword strings, including leading zeros, are
retained. Left=0/right=1 was preferred for readability. Computed examples:
Y=101010, b=101011, space=110, l=00000, n=011, u=010. In particular l's
five zeros are five bits; these strings must never be displayed as integer values.
The first-merge siblings Y/b share their entire code prefix except for the last bit.
`check_example.py` checks all code lengths against the existing independent
merge-depth counts, all distinct codeword pairs for prefix collisions, encoded
length 164, and exact decoding back to the original 43-character sentence.

Tree format, approved after discussion on 2026-11-22: preorder traversal
(node, left, right). Internal node: marker 0 followed recursively by both children.
Leaf: marker 1 followed by the character's fixed-width seven-bit ASCII code.
This is our educational wire format, not a named external compression format.
The fixed-width ASCII fields delimit symbol identities while parsing the tree;
the Huffman codebook derived from that tree delimits codewords in the message.
Alternatives discussed: variable-width symbol identities without a boundary rule
(ambiguous), eight-bit symbol fields or a code-length table (not selected).

Hand proof, also worked through with om: let I be internal-node count and L leaf
count. A full binary tree has 2I edges, and any tree with I+L nodes has I+L-1
edges. Thus I=L-1. Here L=18 and I=17; there are 35 one-bit markers and
18×7=126 symbol bits, giving 161 serialized-tree bits. Message bits remain 164,
so tree+message is 325 bits before message framing/padding. Code checks agree.
`check_example.py` now serializes the tree, reconstructs it, checks equality of
all recovered codewords, verifies that parsing consumes exactly 161 bits from
tree+message, and decodes the remaining message back to the original sentence.
The stopping condition in this test is the known bitstring boundary; a real
byte-packed self-contained message still needs an agreed termination scheme.

Message termination, selected after discussion 2026-11-22: append a special END
symbol once, with frequency one, and include it as a leaf in the Huffman tree.
The receiver stops on that leaf without emitting a character. END terminates
the message; it does not terminate the tree serialization, whose recursive
structure already tells the parser where the tree ends. An arbitrary zero run
is not a safe delimiter because zeros can occur in valid codewords or across
their boundaries. ASCII NUL (value 0) differs from the character '0' (value 48).
Chosen alternative to storing original character count. The following choice
and recomputation now supersede the earlier without-END example.

END identity/leaf width, approved 2026-11-22: eight-bit leaf identities, ASCII
0–127 and END 128 (binary 10000000). Node markers remain one bit. This revises
the earlier seven-bit leaf format, preserving all ASCII inputs without reserving
NUL. Ties extend the existing rule to (weight, minimum descendant symbol ID);
END is ID 128. This does not make all message codewords eight bits long.

Current worked example, checked by hand and code: 19 leaves (18 characters plus
END), 18 internal nodes, 37 markers. Tree size = 37 + 19×8 = 189 bits.
END frequency is one; input has 43 characters plus one END occurrence.
Parent-weight trace: 2, 2, 2, 2, 2, 2, 4, 4, 4, 7, 8, 8, 9, 12, 15, 17,
27, 44. Hand construction: six 1+1 merges, three 2+2 merges, 3+4, two 4+4,
4+5, 5+7, 7+8, 8+9, 12+15, 17+27. Their sum is 171 encoded bits,
matching the weighted-length sum. END=00011 costs five bits; text alone costs
166 bits with this new codebook. Tree+text+END = 189+166+5 = 360 bits,
which is divisible by eight (45 bytes if packed contiguously). This is two
bytes larger than the original ASCII byte storage (43 bytes), despite the shorter
encoded text. Do not present the full standalone message as compression savings.

`check_example.py` and example_results.json now use this END-inclusive format.
Checks pass for code lengths, prefix-freeness, message cost, 189-bit recursive
tree parse/reconstruction, recovered codebook and original-text decoding until
END. Extra zeros after END do not emit extra characters. The old l=00000
example above belongs to the historical codebook; the new one has s=00000.
Byte packing/padding is a proposal below, not yet implemented or approved.

END clarification, 2026-11-22: END is not ASCII NUL. Its chosen identity is
128, serialized as 10000000; its current variable-length Huffman code is 00011.
The decoder follows the reconstructed tree and stops when it reaches that leaf;
it never searches for a raw zero run, and it does not emit END as a text character.
Bits after END are not decoded. This strategy does not require a character-count
header. A count header is a valid alternative, not selected in addition to END.
om's renewed question about a header is treated as a conceptual question, not as
approval of a new framing format. Final padding still awaits agreement.

Why the greedy choice is globally safe (mathematical explanation, not yet an
approved shot/proof sequence): in an optimal full binary prefix tree, take a pair
of deepest sibling leaves. Exchanging symbols can place the two least frequent
symbols there without increasing weighted path cost: moving a lower weight deeper
and a higher weight shallower cannot increase their combined cost. Thus an optimum
exists with the two smallest symbols as siblings. Contract that pair to a single
symbol of combined weight f+g. Expanding it always adds f+g bits, so solving the
contracted problem optimally also solves the original. Repeat by induction.
Scope: minimum message-bit cost among binary, per-symbol prefix codes for the
given frequencies; this is not a claim of smallest possible compressed file.
Verified against MIT lecture 19, section 19.2.2, below. No paper figures will be reused.

Opening quiz (a diagnostic example, not the chosen animation example):
assign A → 0, B → 01, C → 1. Can the received bits `01` be decoded uniquely?
What constraint would om place on the codewords?
om chose placing the ambiguity illustration after the frequency intuition,
before the tree. The exact ambiguity example is still a proposal.

## Decisions

1. Narrative: om chose sentence → ASCII → frequency/short-code intuition →
   prefix ambiguity → tree construction and leaf explanation → comparison.
   Earlier alternatives: opening with the ambiguity puzzle or tree construction.
   Rationale: a concrete message establishes the size problem first.
   The sentence and eventual algorithm are exact, not synthetic measurement;
   any invalid-code counterexample will be explicitly labelled as a deliberately
   invalid proposal, not as a Huffman result.
2. Text: om selected the sentence above. Preserve it exactly; computed counts
   are exact for that input. Examples and on-screen numbers will come from code.
3. Leaf explanation: om requested explaining the prefix property using leaves
   not being ancestors of other leaves. Formal reasoning is recorded above;
   the exact visual demonstration awaits review.
4. ASCII baseline: om chose "Show both" on 2026-11-21 in response to showing
   7-bit ASCII alongside byte storage, with byte storage as the main comparison.
   Show labels explicitly: 301 bits (7-bit ASCII) and 344 bits (8-bit storage).
   Alternatives were showing either baseline alone. Rationale discussed: fixed
   width makes boundaries and direct indexing straightforward; Huffman adapts
   codeword lengths to frequencies. These totals are exact for the chosen text.
   Exact shot layout, emphasis, motion and pacing remain for om's review.
5. Decoder and accounting: om explained that bits alone do not identify the
   symbol mapping, chose transmitting a serialized tree, and approved message
   bits first followed by separately counted decoding overhead (2026-11-21).
   Alternative: compare complete serialized outputs from the outset, or assume
   a codebook already shared. Chosen presentation keeps the coding intuition
   visible while avoiding an unsupported standalone-file savings claim. All
   eventual sizes will be exact for an explicitly agreed serialization format.
   Tree wire format subsequently approved in decision 9; end-of-message scheme
   and padding rule remain pending.
6. Merge rule: om selected the two smallest current weights, reasoning that
   rare symbols can bear longer codewords while frequent ones should have
   shorter root-to-leaf paths (2026-11-21). Alternatives discussed: merging
   smallest or largest weights. Exact algorithm, using real sentence counts.
   Parent weights are reinserted; "higher" does not mean placing a character
   at an internal node. Global optimality rationale is above; animation treatment
   still needs review. The first pair is fixed by decision 7 below.
7. Ties: om delegated the choice between character-code order and stable
   insertion order, emphasizing reproducibility. Chose character-code order,
   extended to a subtree by its minimum descendant character code, as defined
   above. Rationale: explicit input-derived key independent of initial dictionary
   iteration order. Exact, not illustrative. om's prediction about invariant
   optimum bit cost is correct; tied choices can also change individual lengths.
8. Edge labels and child placement: om approved left=0/right=1 with first extracted
   node on the left (2026-11-21). Alternative: left=1/right=0; both preserve lengths,
   total cost and prefix-freeness. Reason: om finds this representation clearer.
   Exact codewords computed from approved merges, checked by round-trip decoding.
   This does not approve a visual layout or traversal motion yet.
9. Tree serialization: om accepted preorder, internal marker 0, leaf marker 1
   plus seven-bit ASCII, after distinguishing tree parsing from message decoding
   (2026-11-22). Alternatives and exact 161-bit cost are above. Exact for this
   chosen tree and format. No decision yet on showing the full serialization
   on screen versus a focused example; that is an animation decision for review.
10. Message termination: om understands that END is decoded from the message,
    rather than appended merely as tree metadata, and selects including END in
    the Huffman tree (2026-11-22). Alternative: original-character-count header.
    Frequency one because exactly one END is appended. Exact strategy for this
    educational format; END's serialized identity and resulting new costs still
    need agreement/calculation. This changes the 18-symbol intermediate tree.
11. END identity: om approved eight-bit leaf identities after explaining that
    seven bits have only 128 available values (2026-11-22). Chosen mapping:
    ASCII 0–127; END 128. Alternative: reserve NUL and exclude it from input.
    Reason: preserve every ASCII character without input restrictions. Exact
    results recomputed above, including the 45-byte contiguous-packing prospect.
12. Reveal and second example: om chose using the larger complete result as a
    narrative surprise, followed by an explanation and a more skewed message
    (2026-11-22). The first comparison is exact; second message not yet selected.
    Need to distinguish distribution, alphabet size and amortization effects.
    om selected the longer-message direction with a distribution chart, declining
    the recommendation to hold length/alphabet fixed (2026-11-22). Alternative
    considered: controlled synthetic text with the same 43 characters and
    18-symbol alphabet. Actual second text, graph type, scale, ordering and motion
    remain unapproved. Do not assume the second distribution is more skewed:
    compute it. Any benefit from length/overhead must be identified separately.
13. Further decisions to quiz and agree, one at a time: final learning claim;
   byte packing and padding; average length and entropy scope;
   on-screen notation; tree layout; motion, pacing, colour and captions.
   This list is a discussion agenda, not a chosen algorithm or storyboard.

## Storyboard

Agreed narrative order only (shot timing and motion are not approved):

1. Show the sentence and encode its characters using the chosen ASCII baseline.
2. Count symbol frequencies and motivate shorter codes for frequent symbols.
3. Show that a careless variable-length assignment can create ambiguity.
4. Construct the Huffman tree step by step; explain root-to-leaf codes and why
   stopping only at leaves prevents prefix collisions.
5. Encode the same exact sentence; first show 166 text bits (with END separately
   identified as 5 bits), then add 189 tree bits. Reveal 360 bits versus the
   original 344-bit byte storage; the contiguous 45-byte interpretation still
   awaits final packing approval. om's intended beat: "Wait—what just happened?"
6. Explain why a shorter payload need not mean a smaller complete representation:
   tree overhead, message length and the character-frequency distribution.
7. Show a second input with skewed frequencies, recomputing its tree, text, END,
   padding and complete size. Input, fairness controls, motion and timing pending.
   om requests a graph/distribution chart to explain the changed input. Show
   measured distribution rather than assume greater skew; a separate size
   comparison is proposed to explain overhead amortization. Neither chart layout
   nor second text has been selected yet.

Tree size (19 leaves including END), how many merges to show in detail, readability and pacing
need discussion with om before choosing shots or a 60–90 second duration.

## Sources

- [MIT 6.046J, lecture 19: Compression and Huffman Coding](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2012/388115265a456321c4a5d19dc9e05281_MIT6_046JS12_lec19.pdf),
  opened and read 2026-11-21: sections 19.1.6 and 19.2 define prefix codes,
  weighted path cost and the merge algorithm; section 19.2.2 (lemmas 19.1–19.2)
  gives the exchange and contraction arguments, including arbitrary frequency ties.
  This verifies the claims above while the original-paper PDF remains inaccessible.
- [RFC 20: ASCII format for Network Interchange](https://www.rfc-editor.org/info/rfc20/),
  opened and checked 2026-11-20: introduction distinguishes seven-bit ASCII from
  embedding in eight-bit bytes; sections 2–3 give the table and seven-bit representation.
- David A. Huffman (1952), [A Method for the Construction of Minimum-Redundancy Codes](https://web.stanford.edu/class/ee398a/handouts/papers/Huffman%20-%20Min%20Redundancy%20Codes%20-%20IRE52.pdf).
  Located 2026-11-20; opening the Stanford PDF failed with HTTP 502, and an
  IIT Delhi mirror timed out. Full-paper verification is pending; do not treat
  the search excerpt as a checked paper. Further source verification and precise
  claim-to-source mapping will be added as the mathematical scope is agreed.

## Open questions for om

1. Next chart quiz/decision: when comparing different-length texts, should the
   frequency chart use raw occurrence counts or relative frequencies (count
   divided by text length)? Proposed: comparable normalized character-frequency
   bars; keep message-length and byte-cost information separate. Shared axes,
   character order, chart layout and transitions still need review.
2. Packing/padding proposal, not approved: concatenate serialized tree and
   encoded text+END directly, without padding between sections; append only enough
   zero bits after END to reach a whole byte (zero to seven bits). This sentence
   already totals 360 bits and needs none. Quiz: why can the decoder ignore those
   trailing padding bits without storing a padding count? Alternative: separately
   byte-align tree and message sections, which adds space here.
3. Subsequent choices (including motion) will be asked
   individually after discussing each answer. Merge groups/lengths and codewords
   and END-inclusive tree serialization computed and verified; byte packing,
   storyboard and tree drawing remain pending.
4. Exact longer second text remains to be proposed/chosen with om before any
   claims about its skew, alphabet, codeword lengths or compressed size.
