#!/usr/bin/env python3
"""Check actual external trainer extraction on CPU, without training or writing models.

Usage: python3 tools/check_trainer_contract.py /path/to/nnue_lab
Requires g++ with UBSan. This is not a GPU, gradient, or export-parity test.
"""

import argparse
from pathlib import Path
import struct
import subprocess
import tempfile


FIXTURES = [
    # FT fixture was reached by a legal 136-ply sequence; both perspectives have 129 rows.
    ("r3b3/7k/1ppppr2/bPnnq1pp/pQNPPpPP/PBP1RN2/1B3P2/R2K4 w - - 0 1", (129, 129, 38)),
    # Structurally valid PK stress position; no full reachability claim.
    ("8/kP1pP1pK/8/pP1pP1pP/8/8/pP1pP1pP/8 w - - 0 1", (58, 58, 100)),
    # Rooks must disable the locked-minor/pawn fortress multiplier, either perspective.
    ("4k3/8/8/1pp2pp1/8/8/R7/4K3 w - - 0 1", None),
    ("4k3/8/8/1pp2pp1/8/8/R7/4K3 b - - 0 1", None),
    # A pawn lever must also disable fortress scaling.
    ("4k3/8/8/1pp2pp1/3P4/8/8/4K3 w - - 0 1", None),
    ("4k3/8/8/1pp2pp1/3P4/8/8/4K3 b - - 0 1", None),
]


def packed_board(fen):
    placement, stm, *_ = fen.split()
    squares = {}
    for rank, row in enumerate(reversed(placement.split("/"))):
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                squares[rank * 8 + file] = "PNBRQKpnbrqk".index(ch)
                file += 1
        assert file == 8
    ordered = sorted(squares)
    assert len(ordered) <= 32
    occ = sum(1 << sq for sq in ordered)
    codes = sum(squares[sq] << (4 * i) for i, sq in enumerate(ordered))
    white_king = next(sq for sq in ordered if squares[sq] == 5)
    black_king = next(sq for sq in ordered if squares[sq] == 11)
    return struct.pack("<QQQhhBBBB", occ, codes & ((1 << 64) - 1), codes >> 64,
                       0, 0, int(stm == "b"), white_king, black_king, 0)


def source(lab):
    result = r"""
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#define __host__
#define __device__
#define __constant__
#define __align__(x) alignas(x)
"""
    for namespace, name in [("production", "train_atn1_cuda.cu"),
                            ("oracle", "cuda_trainer_atn1.cu")]:
        text = (lab / name).read_text()
        # Fail loudly if the source layout changes; never substitute a copied extractor.
        start = text.index("constexpr int L1_SIZE")
        end = text.index("struct FeatureRow", start)
        result += f"namespace {namespace} {{\n{text[start:end]}\n}}\n"
    result += r"""
template<class A, class B>
bool equal_rows(const A* a, int na, const B* b, int nb) {
    std::vector<int> x(a, a + na), y(b, b + nb);
    std::sort(x.begin(), x.end()); std::sort(y.begin(), y.end());
    return x == y;
}
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    FILE* f = fopen(argv[1], "rb");
    if (!f) return 2;
    production::ThreatLookupTables pt;
    oracle::ThreatLookupTables ot;
    production::init_threat_tables(pt); oracle::init_threat_tables(ot);
    unsigned char raw[32];
    while (fread(raw, 1, sizeof(raw), f) == sizeof(raw)) {
        production::PackedBoard p; oracle::PackedBoard o;
        memcpy(&p, raw, 32); memcpy(&o, raw, 32);
        production::DecodedFeatures a = {}; oracle::DecodedFeatures b = {};
        production::extract_board_features(p, pt, a);
        oracle::extract_board_features(o, ot, b);
        if (!equal_rows(a.us_rows, a.n_us, b.us_rows, b.n_us) ||
            !equal_rows(a.them_rows, a.n_them, b.them_rows, b.n_them) ||
            !equal_rows(a.pk_rows, a.n_pk, b.pk_rows, b.n_pk) || a.flags != b.flags) {
            fprintf(stderr, "production/oracle feature contract disagreement\n");
            return 1;
        }
        printf("%d %d %d %d\n", int(a.n_us), int(a.n_them), int(a.n_pk), int(a.flags));
    }
    fclose(f);
}
"""
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lab", type=Path)
    parser.add_argument("--case", choices=["ft", "pk", "scaling"])
    args = parser.parse_args()
    fixtures = list(enumerate(FIXTURES))
    if args.case == "ft":
        fixtures = fixtures[:1]
    elif args.case == "pk":
        fixtures = fixtures[1:2]
    elif args.case == "scaling":
        fixtures = fixtures[2:]
    with tempfile.TemporaryDirectory(prefix="asura-trainer-contract-") as name:
        directory = Path(name)
        cpp = directory / "contract.cpp"
        cpp.write_text(source(args.lab))
        data = directory / "fixtures.bin"
        data.write_bytes(b"".join(packed_board(fen) for _, (fen, _) in fixtures))
        executable = directory / "contract"
        subprocess.run(["g++", "-std=c++17", "-O1", "-g", "-fsanitize=undefined",
                        "-fno-sanitize-recover=all", str(cpp), "-o", str(executable)], check=True)
        result = subprocess.run([str(executable), str(data)], text=True, capture_output=True)
        if result.returncode:
            raise SystemExit(result.stderr or f"extractor exited {result.returncode}")
        rows = [tuple(map(int, line.split())) for line in result.stdout.splitlines()]
        assert len(rows) == len(fixtures)
        for (i, (fen, counts)), actual in zip(fixtures, rows):
            if counts is not None:
                assert actual[:3] == counts, (fen, counts, actual)
            if i >= 2:
                assert actual[3] == 6, ("fortress scaling must be disabled", fen, actual)
            print(f"fixture {i + 1}: FT={actual[0]}/{actual[1]} PK={actual[2]} flags={actual[3]} OK")
    print("CPU extraction/flags match; bounds sanitization passed. GPU/QAT/export remain separate gates.")


if __name__ == "__main__":
    main()
