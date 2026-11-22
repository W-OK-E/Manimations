"""480p inspection cut; computation and provisional choices are in DESIGN.md."""

import json
import textwrap
from pathlib import Path

from manim import (
    DOWN, LEFT, RIGHT, UP, Circle, Create, FadeIn, FadeOut, GrowFromEdge,
    LaggedStart, Line, Rectangle, ReplacementTransform, Scene, Text, VGroup,
)

HERE = Path(__file__).resolve().parent
BG = "#101827"
INK = "#F1F5F9"
MUTED = "#9DAFC5"
DIM = "#33465F"
TEAL = "#64E3C1"
BLUE = "#83B9FF"
AMBER = "#FFCE78"
RED = "#FF8B95"
PURPLE = "#C6A0FF"


def text(value, size=26, color=INK, width=12.5, mono=False):
    result = Text(str(value), font="DejaVu Sans Mono" if mono else "DejaVu Sans",
                  font_size=size, color=color, disable_ligatures=True)
    if result.width > width:
        result.scale_to_fit_width(width)
    return result


def symbol(value):
    return "SP" if value == " " else value


def identity(value):
    return 128 if value == "END" else ord(value)


class HuffmanEncoding(Scene):
    def __init__(self, **kwargs):
        kwargs.setdefault("random_seed", 0)
        super().__init__(**kwargs)
        self.shots = []

    def begin(self, title, subtitle=""):
        if self.mobjects:
            self.play(*[FadeOut(item) for item in list(self.mobjects)], run_time=0.25)
        self.shot_start = self.time
        self.shots.append({"title": title, "start": round(self.time, 3)})
        heading = text(title, 36).move_to([0, 3.1, 0])
        rule = Line([-6.35, 2.7, 0], [6.35, 2.7, 0], color=DIM, stroke_width=1)
        self.play(FadeIn(heading), Create(rule), run_time=0.4)
        if subtitle:
            self.add(text(subtitle, 21, MUTED).move_to([0, 2.35, 0]))

    def footer(self, value, color=MUTED):
        result = text(value, 21, color).move_to([0, -3.4, 0])
        self.add(result)
        return result

    def finish(self, target):
        self.wait(max(0.1, target - (self.time - self.shot_start)))

    def construct(self):
        self.camera.background_color = BG
        data = json.loads((HERE / "preview_results.json").read_text())
        self.short, self.long = data["short"], data["long"]
        self.introduction()
        self.ascii()
        self.frequencies()
        self.ambiguity()
        self.merging()
        self.tree_paths()
        self.encoding()
        self.surprise()
        self.longer_text()
        self.distributions()
        self.longer_cost()
        self.conclusion()
        destination = HERE.parents[1] / "build/huffman"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "shot_times.json").write_text(json.dumps({"shots": self.shots, "duration": self.time}, indent=2))

    def introduction(self):
        self.begin("Huffman encoding", "A smaller bitstream. But a smaller file?")
        quote = VGroup(*[text(line, 34, TEAL) for line in textwrap.wrap(self.short["text"], 28)])
        quote.arrange(DOWN, buff=0.25).move_to([0, 0.7, 0])
        self.play(FadeIn(quote, shift=UP * 0.15), run_time=0.7)
        self.add(text(f'{self.short["characters"]} characters, including spaces', 25, MUTED).move_to([0, -0.85, 0]))
        self.footer("480p inspection preview · exact computed examples · no narration")
        self.finish(6)

    def ascii(self):
        d = self.short
        self.begin("Start with fixed-width ASCII", "Seven-bit codes are commonly stored in eight-bit bytes.")
        groups = []
        for i, (letter, code) in enumerate(zip(d["text"].split()[0], d["ascii_first_word"])):
            box = Rectangle(width=3.3, height=1.55, color=DIM, stroke_width=1)
            group = VGroup(box, text(letter, 42, BLUE).shift(UP * 0.35),
                           text(code, 27, INK, mono=True).shift(DOWN * 0.4))
            group.move_to([-3.8 + i * 3.8, 0.9, 0])
            groups.append(group)
        self.play(LaggedStart(*[FadeIn(group) for group in groups], lag_ratio=0.25), run_time=1)
        totals = VGroup(
            text(f'7-bit ASCII: {d["characters"]} × 7 = {d["ascii_7_bits"]} bits', 29, MUTED),
            text(f'Byte storage: {d["characters"]} × 8 = {d["ascii_8_bits"]} bits = {d["ascii_bytes"]} bytes', 29, BLUE),
        ).arrange(DOWN, buff=0.4).move_to([0, -1.4, 0])
        self.play(FadeIn(totals), run_time=0.5)
        self.footer("Every character occupies the same width, regardless of frequency.")
        self.finish(8)

    def tile(self, name, count, width=1.2, height=1):
        color = PURPLE if name == "END" else INK
        box = Rectangle(width=width, height=height, color=DIM, stroke_width=1,
                        fill_color="#172338", fill_opacity=1)
        label_size = 20 if name == "subtree" else 25
        labels = VGroup(text(symbol(name), label_size, color, width=width * 0.9, mono=True),
                        text(count, 21, MUTED)).arrange(DOWN, buff=0.08)
        return VGroup(box, labels)

    def frequencies(self):
        self.begin("Some characters work much harder", "Count the exact text first. SP denotes a space.")
        items = sorted(self.short["frequencies"].items(), key=lambda pair: (-pair[1], ord(pair[0])))
        tiles = VGroup()
        for i, (name, count) in enumerate(items):
            tile = self.tile(name, count)
            tile.move_to([-5.85 + (i % 9) * 1.46, 1.1 - (i // 9) * 1.3, 0])
            tiles.add(tile)
        self.play(LaggedStart(*[FadeIn(tile) for tile in tiles], lag_ratio=0.04), run_time=0.8)
        rule = text("Frequent symbols should get shorter codewords.", 29, TEAL).move_to([0, -1.8, 0])
        self.play(FadeIn(rule), run_time=0.5)
        self.add(text("But variable-length codes need recognizable boundaries.", 24, MUTED).move_to([0, -2.4, 0]))
        self.footer("END will be added once as a separate symbol.")
        self.finish(7)

    def ambiguity(self):
        self.begin("Short codes must still be decodable", "A prefix-free code never makes one symbol's code the start of another's.")
        left = VGroup(
            text("INVALID toy code", 25, RED),
            text("A = 0    B = 01    C = 1", 24, INK, width=6, mono=True),
            text("01", 46, RED, mono=True),
            text("B  or  A + C ?", 28, RED),
        ).arrange(DOWN, buff=0.3).move_to([-3.25, 0.2, 0])
        self.play(FadeIn(left), run_time=0.6)
        coords = {"root": (3.2, 1.1), "A": (1.95, -0.25), "inner": (4.35, -0.25),
                  "B": (3.7, -1.65), "C": (5.6, -1.65)}
        toy = VGroup(text("VALID toy tree", 25, TEAL).move_to([3.5, 1.95, 0]))
        for a, b, bit in (("root", "A", "0"), ("root", "inner", "1"), ("inner", "B", "0"), ("inner", "C", "1")):
            p, q = [*coords[a], 0], [*coords[b], 0]
            line = Line(p, q, color=DIM, stroke_width=2)
            toy.add(line, text(bit, 20, MUTED, mono=True).move_to(line.get_center() + LEFT * 0.18))
        for name, coord in coords.items():
            toy.add(Circle(radius=0.13, color=TEAL if name in "ABC" else MUTED,
                           fill_color=BG, fill_opacity=1).move_to([*coord, 0]))
            if name in ("A", "B", "C"):
                code = {"A": "0", "B": "10", "C": "11"}[name]
                toy.add(text(f"{name}: {code}", 21, TEAL, mono=True).move_to([coord[0], coord[1] - 0.43, 0]))
        self.play(FadeIn(toy), run_time=0.8)
        self.footer("Symbols occupy leaves only. A leaf has no descendants to continue to.", TEAL)
        self.finish(10)

    def queue_view(self, queue):
        result = VGroup()
        for i, item in enumerate(queue):
            name = item["symbols"][0] if len(item["symbols"]) == 1 else "subtree"
            tile = self.tile(name, item["weight"], width=1.13, height=0.85)
            tile.move_to([-5.85 + (i % 10) * 1.3, 1.35 - (i // 10) * 1.08, 0])
            result.add(tile)
        return result

    def mini_merge(self, merge):
        root, left, right = [0, -0.7, 0], [-2, -1.9, 0], [2, -1.9, 0]
        result = VGroup()
        for destination, bit in ((left, "0"), (right, "1")):
            edge = Line(root, destination, color=TEAL, stroke_width=2)
            result.add(edge, text(bit, 18, TEAL, mono=True).move_to(edge.get_center() + UP * 0.15))
        for location, name, count in ((root, "parent", merge["combined_weight"]),
                                      (left, "child", merge["weights"][0]), (right, "child", merge["weights"][1])):
            circle = Circle(radius=0.29, color=TEAL, fill_color=BG, fill_opacity=1).move_to(location)
            result.add(circle, text(count, 22).move_to(location))
        for location, group in ((left, merge["groups"][0]), (right, merge["groups"][1])):
            name = symbol(group[0]) if len(group) == 1 else f"subtree ({len(group)} symbols)"
            result.add(text(name, 18, MUTED, width=3.4).move_to([location[0], -2.4, 0]))
        result.add(text(f'{merge["weights"][0]} + {merge["weights"][1]} = {merge["combined_weight"]}', 26, TEAL).move_to([0, -2.95, 0]))
        return result

    def merging(self):
        self.begin("Build the actual Huffman tree", "Take the two lightest nodes, join them, and reinsert their combined weight.")
        queue = [{"symbols": [name], "weight": count} for name, count in self.short["symbol_frequencies"].items()]
        key = lambda item: (item["weight"], min(identity(s) for s in item["symbols"]))
        queue.sort(key=key)
        drawing = self.queue_view(queue)
        self.play(FadeIn(drawing), run_time=0.3)
        self.footer("Exact counts, including END=1 · ties: smallest descendant symbol ID")
        for i, merge in enumerate(self.short["merges"]):
            assert queue[0]["symbols"] == merge["groups"][0]
            assert queue[1]["symbols"] == merge["groups"][1]
            self.play(drawing[0][0].animate.set_stroke(TEAL, width=3),
                      drawing[1][0].animate.set_stroke(TEAL, width=3), run_time=0.18)
            mini = self.mini_merge(merge)
            self.play(FadeIn(mini), run_time=0.24)
            self.wait(0.5 if i < 3 else 0.13)
            combined = sorted(queue[0]["symbols"] + queue[1]["symbols"], key=identity)
            queue = queue[2:] + [{"symbols": combined, "weight": merge["combined_weight"]}]
            queue.sort(key=key)
            updated = self.queue_view(queue)
            self.play(ReplacementTransform(drawing, updated), FadeOut(mini), run_time=0.24)
            drawing = updated
        self.finish(21)

    def actual_tree(self):
        codes, weights = self.short["codewords"], self.short["symbol_frequencies"]
        ordered = sorted(codes, key=lambda name: codes[name])
        xs = {codes[name]: -6.1 + i * 12.2 / (len(ordered) - 1) for i, name in enumerate(ordered)}
        prefixes = {code[:i] for code in codes.values() for i in range(len(code) + 1)}
        positions = {}
        for prefix in sorted(prefixes, key=len, reverse=True):
            x = xs[prefix] if prefix in xs else (positions[prefix + "0"][0] + positions[prefix + "1"][0]) / 2
            positions[prefix] = [x, 1.7 - 0.62 * len(prefix), 0]
        edges, nodes, names, bits = VGroup(), VGroup(), VGroup(), VGroup()
        lookup = {}
        for prefix in sorted(prefixes, key=lambda p: (len(p), p)):
            position = positions[prefix]
            if prefix:
                edge = Line(positions[prefix[:-1]], position, color=DIM, stroke_width=1.3)
                lookup[prefix] = edge
                edges.add(edge)
                bits.add(text(prefix[-1], 12, MUTED, mono=True).move_to(edge.get_center() + LEFT * 0.09))
            leaf = prefix in xs
            name = ordered[list(codes[n] for n in ordered).index(prefix)] if leaf else None
            weight = weights[name] if leaf else sum(weights[n] for n, code in codes.items() if code.startswith(prefix))
            nodes.add(Circle(radius=0.12, color=PURPLE if name == "END" else TEAL if leaf else MUTED,
                             fill_color=BG, fill_opacity=1, stroke_width=1).move_to(position))
            nodes.add(text(weight, 11, INK).move_to(position))
            if leaf:
                names.add(text(symbol(name), 17 if name != "END" else 13, PURPLE if name == "END" else TEAL,
                               mono=True).move_to([position[0], position[1] - 0.3, 0]))
        return VGroup(edges, bits, nodes, names), lookup

    def tree_paths(self):
        self.begin("A codeword is a root-to-leaf path", "Left edge = 0. Right edge = 1. Every edge counts, including leading zeros.")
        tree, lookup = self.actual_tree()
        self.play(FadeIn(tree), run_time=0.7)
        code_label = None
        for name in (" ", "Y", "s"):
            for edge in lookup.values():
                edge.set_stroke(DIM, width=1.3)
            code = self.short["codewords"][name]
            highlighted = [lookup[code[:i]] for i in range(1, len(code) + 1)]
            current = text(f'{symbol(name)}  →  {code}  ({len(code)} bits)', 28, TEAL, mono=True).move_to([0, -2.9, 0])
            animations = [edge.animate.set_stroke(AMBER, width=3) for edge in highlighted]
            if code_label is not None:
                animations.append(FadeOut(code_label))
            animations.append(FadeIn(current))
            self.play(*animations, run_time=0.55)
            code_label = current
            self.wait(1)
        self.footer("Exact 19-leaf tree · internal numbers are subtree occurrence totals")
        self.finish(8)

    def encoding(self):
        d = self.short
        self.begin("Use the tree to encode the message", "The same characters, now with variable-length codewords.")
        cards = []
        for i, letter in enumerate(d["text"].split()[0]):
            group = VGroup(text(letter, 42, TEAL), text(d["codewords"][letter], 31, INK, mono=True)).arrange(DOWN, buff=0.35)
            group.move_to([-3.8 + i * 3.8, 1, 0])
            cards.append(group)
        self.play(LaggedStart(*[FadeIn(card) for card in cards], lag_ratio=0.4), run_time=1.1)
        self.add(text(f'Whole text: {d["text_bits"]} bits, instead of {d["ascii_8_bits"]}', 31, TEAL).move_to([0, -0.6, 0]))
        self.add(text(f'Append END: {d["encoded_end"]} ({d["end_bits"]} bits)', 26, PURPLE, mono=True).move_to([0, -1.5, 0]))
        self.add(text("END's leaf identity is 128; its Huffman codeword is different.", 22, MUTED).move_to([0, -2.25, 0]))
        self.footer("On reaching END, decoding stops. Padding is not decoded as text.")
        self.finish(8)

    def stacked_bar(self, parts, total_scale, y, width=10.8):
        result = VGroup()
        cursor = -5.4
        for value, color in parts:
            if value == 0:
                continue
            segment_width = value / total_scale * width
            segment = Rectangle(width=segment_width, height=0.42, stroke_width=0,
                                fill_color=color, fill_opacity=1)
            segment.move_to([cursor + segment_width / 2, y, 0])
            cursor += segment_width
            result.add(segment)
        return result

    def surprise(self):
        d = self.short
        self.begin("Wait—what just happened?", "The shorter payload is not the complete self-contained file.")
        top = text(f'ASCII byte storage: {d["ascii_bytes"]} bytes', 27, BLUE).move_to([0, 1.7, 0])
        ascii_bar = self.stacked_bar([(d["ascii_8_bits"], BLUE)], 380, 1.05)
        payload_label = text(f'Encoded text alone: {d["text_bits"]} bits', 27, TEAL).move_to([0, 0.25, 0])
        payload_bar = self.stacked_bar([(d["text_bits"], TEAL)], 380, -0.4)
        self.play(FadeIn(top), FadeIn(ascii_bar), FadeIn(payload_label), FadeIn(payload_bar), run_time=0.7)
        self.wait(1.5)
        complete = self.stacked_bar([(d["text_bits"], TEAL), (d["tree_bits"], AMBER), (d["end_bits"], PURPLE)], 380, -0.4)
        complete_label = text(f'Complete Huffman: {d["bytes"]} bytes', 27, RED).move_to([0, 0.25, 0])
        self.play(ReplacementTransform(payload_bar, complete), ReplacementTransform(payload_label, complete_label), run_time=0.8)
        equation = text(f'{d["text_bits"]} text + {d["tree_bits"]} tree + {d["end_bits"]} END + {d["padding_bits"]} padding = {d["total_bits"]} bits', 25).move_to([0, -1.4, 0])
        self.play(FadeIn(equation), run_time=0.5)
        self.add(text(f'{d["bytes"]} bytes > {d["ascii_bytes"]} bytes', 39, RED).move_to([0, -2.35, 0]))
        self.footer("Teal: text · amber: serialized tree · purple: END · our explicit format")
        self.finish(10)

    def longer_text(self):
        d = self.long
        self.begin("Try a longer, more skewed message", "Constructed English sample, intentionally rich in e and spaces.")
        badge = text(f'{d["characters"]} characters · {d["alphabet"]} distinct text symbols', 26, AMBER).move_to([0, 1.65, 0])
        lines = VGroup(*[text(line, 24, INK, width=12) for line in textwrap.wrap(d["text"], 76)])
        lines.arrange(DOWN, buff=0.22).move_to([0, -0.1, 0])
        self.play(FadeIn(badge), FadeIn(lines), run_time=0.7)
        self.footer("New length and alphabet too: this is not a one-variable experiment.")
        self.finish(8)

    def histogram(self, data, left, color, title):
        width, height, bottom = 5.55, 2.55, -1.65
        ranked = sorted(data["percentages"].items(), key=lambda pair: (-pair[1], identity(pair[0])))
        static = VGroup(text(title, 25, color, width=5.5).move_to([left + width / 2, 1.8, 0]),
                        text(f'{data["characters"]} chars · {data["alphabet"]} symbols', 20, MUTED).move_to([left + width / 2, 1.3, 0]))
        for tick in (0, 10, 20, 30):
            y = bottom + tick / 35 * height
            static.add(Line([left, y, 0], [left + width, y, 0], color=DIM, stroke_width=0.6),
                       text(f"{tick}%", 15, MUTED).move_to([left - 0.35, y, 0]))
        bars = VGroup()
        pitch = width / len(ranked)
        for i, (name, percentage) in enumerate(ranked):
            x = left + (i + 0.5) * pitch
            bar_height = percentage / 35 * height
            bars.add(Rectangle(width=pitch * 0.67, height=bar_height, stroke_width=0,
                               fill_color=color, fill_opacity=1).move_to([x, bottom + bar_height / 2, 0]))
            static.add(text(symbol(name), 14, INK, mono=True, width=pitch * 0.95).move_to([x, bottom - 0.25, 0]))
            if i == 0:
                # Keep the peak annotation clear of the y-axis tick labels.
                static.add(text(f"{percentage:.1f}%", 15, color).move_to([x + 0.25, bottom + bar_height + 0.2, 0]))
        static.add(text("Characters ranked by frequency", 18, MUTED).move_to([left + width / 2, -2.3, 0]))
        return static, bars

    def distributions(self):
        self.begin("Compare proportions, not raw counts", "Same percentage scale on both panels · all text symbols shown · END excluded")
        left, bars_left = self.histogram(self.short, -5.95, BLUE, "Original sentence")
        right, bars_right = self.histogram(self.long, 0.9, TEAL, "Longer sample")
        self.play(FadeIn(left), FadeIn(right), run_time=0.6)
        self.play(LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars_left], lag_ratio=0.025),
                  LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars_right], lag_ratio=0.025), run_time=1.4)
        self.footer(f'Largest share: {max(self.short["percentages"].values()):.1f}% → {max(self.long["percentages"].values()):.1f}% · the longer sample is more concentrated')
        self.finish(11)

    def longer_cost(self):
        d = self.long
        self.begin("Now count the complete second file", "The tree is still included. So are END and the final byte padding.")
        self.add(text(f'ASCII byte storage: {d["ascii_bytes"]} bytes ({d["ascii_8_bits"]} bits)', 28, BLUE).move_to([0, 1.65, 0]))
        original = self.stacked_bar([(d["ascii_8_bits"], BLUE)], d["ascii_8_bits"], 1)
        encoded = self.stacked_bar([(d["text_bits"], TEAL), (d["tree_bits"], AMBER),
                                    (d["end_bits"], PURPLE), (d["padding_bits"], MUTED)], d["ascii_8_bits"], -0.4)
        self.add(text(f'Complete Huffman: {d["bytes"]} bytes', 28, TEAL).move_to([0, 0.25, 0]))
        self.play(FadeIn(original), FadeIn(encoded), run_time=0.8)
        self.add(text(f'{d["text_bits"]} text + {d["tree_bits"]} tree + {d["end_bits"]} END + {d["padding_bits"]} padding = {d["total_bits"]} bits', 24).move_to([0, -1.35, 0]))
        self.add(text(f'{d["saved_bytes"]} bytes smaller, including the codebook', 33, TEAL).move_to([0, -2.3, 0]))
        self.footer("This educational format has no extra magic/version/checksum header.")
        self.finish(9)

    def conclusion(self):
        a, b = self.short, self.long
        self.begin("Two effects, one honest comparison", "A shorter average code, and less tree cost per character.")
        # Explicit columns: leading spaces are not reliable SVG layout anchors.
        rows = VGroup(text("Short", 23, MUTED).move_to([2.1, 1.65, 0]),
                      text("Longer", 23, MUTED).move_to([4.65, 1.65, 0]))
        for i, (name, key, color) in enumerate((
            ("Text entropy H(p)", "entropy_text", INK),
            ("Mean text code length", "average_text", TEAL),
            ("Tree bits / text character", "tree_bits_per_character", AMBER),
        )):
            y = 1.0 - i * 0.63
            rows.add(text(name, 23, color, width=6).move_to([-4.8, y, 0], aligned_edge=LEFT),
                     text(f"{a[key]:.3f}", 25, color, mono=True).move_to([2.1, y, 0]),
                     text(f"{b[key]:.3f}", 25, color, mono=True).move_to([4.65, y, 0]))
        self.play(FadeIn(rows), run_time=0.7)
        self.add(text("H(p) = −Σ p log₂ p · empirical text frequencies; END excluded", 21, MUTED).move_to([0, -1.15, 0]))
        self.add(text("Huffman optimizes the codewords. The complete format still costs space.", 25, TEAL).move_to([0, -2.05, 0]))
        self.add(text("Uniform small alphabets can also beat ASCII; skew is not the only factor.", 21, MUTED).move_to([0, -2.65, 0]))
        self.footer("Sources: RFC 20 · MIT 6.046J L19 (2012) · MIT 18.200 L19 (2024)")
        self.finish(11)
