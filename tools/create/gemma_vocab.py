#!/usr/bin/env python3
"""Writes Gemma's SentencePiece vocabulary in the simple form Brekwol Upscale's engine reads
(engine/gemma_tokenizer.hpp): "GVOC", u32 version 1, u32 count, then per piece u8 type, f32 score,
u16 byte length, UTF-8 bytes.

    python gemma_vocab.py tokenizer.model gemma-vocab.bin [--check cases.txt]

With --check, every line of cases.txt (and a built-in set) is also written with SentencePiece's
own ids to gemma-vocab.cases.bin for the engine's test: u32 count, then per case u32 text length,
text, u32 id count, i32 ids.
"""
import struct
import sys

from sentencepiece import SentencePieceProcessor
from sentencepiece import sentencepiece_model_pb2 as pb

BUILT_IN = [
    "", " ", "  ", "a", "A red fox sitting in fresh snow in a forest, sharp focus on the eyes",
    "BREKWOL CAFE", "Portrait of a young African woman with braided hair smiling, 85mm lens",
    "Numbers 1234567890 and 3.14159, prices $19.99 and 50%", "Tabs\tand\nnew lines\n\nand   spaces    here",
    "Emoji 🦊🌲❄️ and accents: café, naïve, Zürich, São Paulo", "Somali: Magaalada Baydhabo waa magaalo qurux badan.",
    "Swahili: Habari za asubuhi, rafiki yangu.", "Arabic: مرحبا بالعالم", "Chinese: 你好，世界", "Japanese: こんにちは世界",
    "<start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n", "trailing space ", " leading space",
    "UPPER lower MiXeD", "a​zero width", "rare \U0001F9FF\U0001FAE0 symbols", "x" * 200,
]


def main():
    model, out = sys.argv[1], sys.argv[2]
    m = pb.ModelProto()
    m.ParseFromString(open(model, 'rb').read())
    with open(out, 'wb') as f:
        f.write(b'GVOC' + struct.pack('<II', 1, len(m.pieces)))
        for p in m.pieces:
            b = p.piece.encode('utf-8')
            f.write(struct.pack('<BfH', p.type, p.score, len(b)) + b)
    print(out, len(m.pieces), 'pieces')
    if '--check' in sys.argv:
        sp = SentencePieceProcessor(model_file=model)
        cases = list(BUILT_IN)
        extra = sys.argv[sys.argv.index('--check') + 1]
        cases += [line.rstrip('\n') for line in open(extra, encoding='utf-8')]
        with open(out.replace('.bin', '.cases.bin'), 'wb') as f:
            f.write(struct.pack('<I', len(cases)))
            for c in cases:
                b = c.encode('utf-8')
                ids = sp.encode(c)
                f.write(struct.pack('<I', len(b)) + b + struct.pack('<I', len(ids)) + struct.pack(f'<{len(ids)}i', *ids))
        print(len(cases), 'cases')


if __name__ == '__main__':
    main()
