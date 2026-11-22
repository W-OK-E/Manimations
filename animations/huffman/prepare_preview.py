"""Compute and verify every numeric value used by the 480p preview."""

import json
import math
from collections import Counter
from pathlib import Path

from check_example import (END, TEXT, codes_from_tree, decode_until_end,
                           huffman_lengths, label, labelled, parse_tree,
                           serialize_tree, tree_from_codes)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def analyse(text):
    raw = text.encode("ascii")
    counts = Counter(text)
    weights = counts.copy()
    weights[END] = 1
    lengths, merges, codes = huffman_lengths(weights)
    tree = tree_from_codes(codes)
    tree_bits = serialize_tree(tree)
    payload = "".join(codes[s] for s in text)
    message = payload + codes[END]
    unpadded = tree_bits + message
    padding = (-len(unpadded)) % 8
    packed_bits = unpadded + "0" * padding
    packed = bytes(int(packed_bits[i:i + 8], 2) for i in range(0, len(packed_bits), 8))
    recovered_bits = "".join(format(byte, "08b") for byte in packed)
    recovered_tree, offset = parse_tree(recovered_bits)
    recovered, message_bits = decode_until_end(recovered_bits[offset:], recovered_tree)
    assert recovered == text
    assert offset == len(tree_bits) and message_bits == len(message)
    assert codes_from_tree(recovered_tree) == codes
    assert sum(weights[s] * lengths[s] for s in weights) == len(message)
    assert sum(merge["combined_weight"] for merge in merges) == len(message)
    assert len(tree_bits) == 10 * len(weights) - 1
    assert all(not b.startswith(a) for x, a in codes.items() for y, b in codes.items() if x != y)
    probabilities = {s: count / len(text) for s, count in counts.items()}
    q = {s: count / (len(text) + 1) for s, count in weights.items()}
    entropy_with_end = -sum(p * math.log2(p) for p in q.values())
    average_with_end = len(message) / (len(text) + 1)
    assert entropy_with_end <= average_with_end < entropy_with_end + 1
    result = {
        "text": text, "characters": len(text), "alphabet": len(counts),
        "frequencies": labelled(counts), "symbol_frequencies": labelled(weights),
        "percentages": labelled({s: p * 100 for s, p in probabilities.items()}),
        "codewords": labelled(codes), "lengths": labelled(lengths), "merges": merges,
        "tree_bits": len(tree_bits), "text_bits": len(payload), "end_bits": len(codes[END]),
        "padding_bits": padding, "total_bits": len(packed_bits), "bytes": len(packed),
        "ascii_7_bits": 7 * len(text), "ascii_8_bits": 8 * len(text),
        "ascii_bytes": len(raw), "ascii_first_word": [format(ord(s), "08b") for s in text.split()[0]],
        "encoded_text": payload, "encoded_end": codes[END],
        "serialized_tree": tree_bits, "packed_hex": packed.hex(),
        "entropy_text": -sum(p * math.log2(p) for p in probabilities.values()),
        "entropy_with_end": entropy_with_end, "average_text": len(payload) / len(text),
        "average_with_end": average_with_end,
        "tree_bits_per_character": len(tree_bits) / len(text),
        "saved_bytes": len(raw) - len(packed),
    }
    return result, packed


def main():
    # Only the text-file newline is excluded; internal punctuation/case/spaces stay.
    longer = (HERE / "long_text.txt").read_text().removesuffix("\n")
    short, short_bytes = analyse(TEXT)
    long, long_bytes = analyse(longer)
    assert (short["characters"], short["tree_bits"], short["text_bits"], short["end_bits"], short["bytes"]) == (43, 189, 166, 5, 45)
    # Hand checks: 24 leaves incl. END -> 47 markers + 24*8 identity bits.
    assert long["characters"] == 286 and long["alphabet"] == 23
    assert long["frequencies"]["e"] == 81 and long["frequencies"][" "] == 58
    assert long["tree_bits"] == 47 + 24 * 8 == 239
    assert long["text_bits"] + long["end_bits"] == 975
    assert long["padding_bits"] == 2 and long["bytes"] == 152
    result = {"short": short, "long": long, "long_text_origin": "Original constructed English sample, intentionally rich in e and spaces; not a corpus benchmark.",
              "format": "preorder tree (internal 0; leaf 1 + 8-bit ID), text, END, final zero padding only; ASCII 0–127, END 128; no magic/version/checksum"}
    (HERE / "preview_results.json").write_text(json.dumps(result, indent=2) + "\n")
    output = ROOT / "build/huffman"
    output.mkdir(parents=True, exist_ok=True)
    (output / "short.huff").write_bytes(short_bytes)
    (output / "long.huff").write_bytes(long_bytes)
    print(json.dumps({name: {key: value[key] for key in ("characters", "alphabet", "tree_bits", "text_bits", "end_bits", "padding_bits", "bytes", "entropy_text", "average_text")}
                      for name, value in (("short", short), ("long", long))}, indent=2))


if __name__ == "__main__":
    main()
