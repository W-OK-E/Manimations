"""Verify approved Huffman codes and round-trip decoding; no scene or rendering."""

import heapq
import json
from collections import Counter

TEXT = "You cannot build what you do not understand"
END = chr(128)  # Symbol identity 128 is outside ASCII 0–127.


def label(symbol):
    return "END" if symbol == END else symbol


def labelled(mapping):
    return {label(symbol): value for symbol, value in sorted(mapping.items())}


def huffman_lengths(counts):
    # Each queue item is (weight, smallest descendant character code, symbols).
    # Active groups are disjoint, so their smallest character codes are unique.
    queue = [(weight, ord(symbol), (symbol,)) for symbol, weight in counts.items()]
    heapq.heapify(queue)
    lengths = dict.fromkeys(counts, 0)
    codes = dict.fromkeys(counts, "")
    merges = []
    while len(queue) > 1:
        first = heapq.heappop(queue)
        second = heapq.heappop(queue)
        symbols = tuple(sorted(first[2] + second[2]))
        weight = first[0] + second[0]
        # Every occurrence in these groups gains one bit, regardless of its value.
        for symbol in symbols:
            lengths[symbol] += 1
        # Preserve leading zeros: codewords are strings, never integer values.
        # A later merge adds an edge above these nodes, hence prepending bits.
        for symbol in first[2]:
            codes[symbol] = "0" + codes[symbol]
        for symbol in second[2]:
            codes[symbol] = "1" + codes[symbol]
        merges.append({
            "groups": [[label(symbol) for symbol in first[2]],
                       [label(symbol) for symbol in second[2]]],
            "weights": [first[0], second[0]],
            "combined_weight": weight,
        })
        heapq.heappush(queue, (weight, min(first[1], second[1]), symbols))
    return lengths, merges, codes


def decode(bits, codes):
    inverse = {code: symbol for symbol, code in codes.items()}
    decoded = []
    pending = ""
    for bit in bits:
        pending += bit
        if pending in inverse:
            decoded.append(inverse[pending])
            pending = ""
    if pending:
        raise ValueError("Incomplete or invalid trailing codeword")
    return "".join(decoded)


def decode_until_end(bits, tree):
    node = tree
    decoded = []
    for consumed, bit in enumerate(bits, start=1):
        node = node[bit]
        if isinstance(node, str):
            if node == END:
                return "".join(decoded), consumed
            decoded.append(node)
            node = tree
    raise ValueError("Missing END symbol")


def tree_from_codes(codes):
    root = {}
    for symbol, code in codes.items():
        node = root
        for bit in code[:-1]:
            node = node.setdefault(bit, {})
        node[code[-1]] = symbol
    return root


def serialize_tree(node, symbol_bits=8):
    if isinstance(node, str):
        # Fixed-width identity, distinct from the variable-length message code.
        assert 0 <= ord(node) < 2 ** symbol_bits
        return "1" + format(ord(node), f"0{symbol_bits}b")
    return "0" + serialize_tree(node["0"], symbol_bits) + serialize_tree(node["1"], symbol_bits)


def parse_tree(bits, offset=0, symbol_bits=8):
    if offset >= len(bits):
        raise ValueError("Truncated tree marker")
    marker = bits[offset]
    offset += 1
    if marker == "1":
        if offset + symbol_bits > len(bits):
            raise ValueError("Truncated symbol identity")
        identity = int(bits[offset:offset + symbol_bits], 2)
        if identity > 128:
            raise ValueError("Unknown symbol identity")
        return chr(identity), offset + symbol_bits
    if marker != "0":
        raise ValueError("Invalid tree marker")
    left, offset = parse_tree(bits, offset, symbol_bits)
    right, offset = parse_tree(bits, offset, symbol_bits)
    return {"0": left, "1": right}, offset


def codes_from_tree(node, prefix=""):
    if isinstance(node, str):
        return {node: prefix}
    return {**codes_from_tree(node["0"], prefix + "0"),
            **codes_from_tree(node["1"], prefix + "1")}


def main():
    counts = Counter(TEXT)
    # Independent hand checks recorded in DESIGN.md.
    assert [len(word) for word in TEXT.split(" ")] == [3, 6, 5, 4, 3, 2, 3, 10]
    assert len(TEXT) == 43 and len(counts) == 18
    assert counts[" "] == 7 and sum(counts.values()) == 43
    assert len(TEXT.encode("ascii")) == len(TEXT)
    # Preserve and verify the earlier seven-bit/no-END intermediate example.
    old_lengths, old_merges, old_codes = huffman_lengths(counts)
    assert sum(counts[s] * old_lengths[s] for s in counts) == 164
    assert len(serialize_tree(tree_from_codes(old_codes), symbol_bits=7)) == 161
    symbol_counts = counts.copy()
    symbol_counts[END] = 1
    lengths, merges, codes = huffman_lengths(symbol_counts)
    total_bits = sum(symbol_counts[symbol] * length for symbol, length in lengths.items())
    # Independent hand trace in DESIGN.md, including END's frequency one.
    expected_merge_weights = [2, 2, 2, 2, 2, 2, 4, 4, 4, 7, 8, 8, 9, 12, 15, 17, 27, 44]
    assert [merge["combined_weight"] for merge in merges] == expected_merge_weights
    assert total_bits == sum(expected_merge_weights) == 171
    assert all(len(codes[symbol]) == length for symbol, length in lengths.items())
    assert all(not second.startswith(first)
               for a, first in codes.items() for b, second in codes.items() if a != b)
    text_bits = "".join(codes[symbol] for symbol in TEXT)
    encoded = text_bits + codes[END]
    assert len(encoded) == total_bits
    assert decode(encoded, codes) == TEXT + END
    tree = tree_from_codes(codes)
    tree_bits = serialize_tree(tree)
    # Hand calculation: 19 leaves + 18 internal markers + 19 eight-bit identities.
    assert len(tree_bits) == (19 + 18) + 19 * 8 == 189
    reconstructed, consumed = parse_tree(tree_bits + encoded)
    reconstructed_codes = codes_from_tree(reconstructed)
    assert consumed == 189 and reconstructed == tree
    assert reconstructed_codes == codes
    decoded_text, message_consumed = decode_until_end((tree_bits + encoded)[consumed:], reconstructed)
    assert decoded_text == TEXT and message_consumed == len(encoded)
    # END stops before trailing bits; no final byte-packing convention chosen yet.
    assert decode_until_end(encoded + "00000000", reconstructed) == (TEXT, len(encoded))
    print(json.dumps({
        "text": TEXT,
        "characters": len(TEXT),
        "frequencies": dict(sorted(counts.items())),
        "ascii_7bit_bits": 7 * len(TEXT),
        "ascii_byte_storage_bits": 8 * len(TEXT.encode("ascii")),
        "ascii_7bit_codewords": [format(ord(char), "07b") for char in TEXT],
        "ascii_byte_codewords": [format(byte, "08b") for byte in TEXT.encode("ascii")],
        "tie_break": "weight, then smallest descendant symbol ID (ASCII 0–127; END 128)",
        "huffman_symbol_frequencies": labelled(symbol_counts),
        "huffman_lengths": labelled(lengths),
        "huffman_merge_steps": merges,
        "huffman_message_bits": total_bits,
        "huffman_text_bits": len(text_bits),
        "huffman_end_bits": len(codes[END]),
        "edge_bits_assigned": True,
        "edge_convention": "first extracted child left=0; second extracted child right=1",
        "huffman_codewords": labelled(codes),
        "huffman_message_bitstream": encoded,
        "decoded_text": decoded_text,
        "tree_serialization": "preorder: internal=0, leaf=1 followed by eight-bit ID (ASCII 0–127, END 128)",
        "serialized_tree_bits": tree_bits,
        "serialized_tree_length": len(tree_bits),
        "tree_and_message_bits": len(tree_bits) + len(encoded),
        "without_end_reference": {"message_bits": 164, "seven_bit_leaf_tree_bits": 161},
        "framing_and_padding_finalized": False,
    }, indent=2))


if __name__ == "__main__":
    main()
