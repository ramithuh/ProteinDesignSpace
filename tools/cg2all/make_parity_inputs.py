"""Build the three parity inputs: 16-aa ideal helix, crambin (1AB1) Ca trace, two-chain trace with a break.

usage: python -I make_parity_inputs.py <out_dir> <cg2all checkout>/tests
"""
import math, sys
out = sys.argv[1]
ref = sys.argv[2]

def atom_line(serial, name, resn, chain, resi, x, y, z, het=False):
    rec = "HETATM" if het else "ATOM  "
    name4 = f" {name:<3s}" if len(name) < 4 and not het else f"{name:<4s}"
    return f"{rec}{serial:5d} {name4} {resn:>3s} {chain}{resi:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {(name if het else name[0]):>2s}\n"

THREE = dict(A="ALA", C="CYS", D="ASP", E="GLU", F="PHE", G="GLY", H="HIS", I="ILE", K="LYS", L="LEU",
             M="MET", N="ASN", P="PRO", Q="GLN", R="ARG", S="SER", T="THR", V="VAL", W="TRP", Y="TYR")
seq = "MKHWQEALKKHMWQGL"
lines = []
for i, aa in enumerate(seq):
    t = math.radians(100.0 * i)
    lines.append(atom_line(i + 1, "CA", THREE[aa], "A", i + 1, 2.3 * math.cos(t), 2.3 * math.sin(t), 1.5 * i))
open(f"{out}/helix16.pdb", "w").write("".join(lines) + "TER\nEND\n")

src = [l for l in open(f"{ref}/1ab1_A.calpha.pdb") if l.startswith(("SSBOND", "ATOM"))]
open(f"{out}/crambin46.pdb", "w").write("".join(src) + "TER\nEND\n")

atoms = [l for l in src if l.startswith("ATOM")]
mc = []
serial = 1
for l in atoms:
    resi = int(l[22:26])
    if 33 <= resi <= 36:
        continue  # chain break inside chain B
    chain = "A" if resi <= 22 else "B"
    mc.append(l[:6] + f"{serial:5d}" + l[11:21] + chain + l[22:])
    serial += 1
    if resi == 22:
        mc.append("TER\n")
mc.append("TER\n")
mc.append(atom_line(serial, "ZN", " ZN", "C", 101, 10.0, 10.0, 10.0, het=True))
open(f"{out}/twochain.pdb", "w").write("".join(mc) + "END\n")
