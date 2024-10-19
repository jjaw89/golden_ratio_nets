#!/usr/bin/env python3

import argparse
import pathlib
import subprocess
import sys
import fileinput

parser = argparse.ArgumentParser()
parser.add_argument("datadir", type=pathlib.Path, help="datadir where inputs/outputs are kept")
parser.add_argument("executable", type=pathlib.Path, help="executable to generate pointsets")

args = parser.parse_args()

datadir = args.datadir
exe = args.executable

if not datadir.exists():
    print(f"Error: datadir '{datadir}' does not exist", file=sys.stderr)
    sys.exit(1)

if not exe.exists():
    print(f"Error: executable '{exe}' does not exist", file=sys.stderr)
    sys.exit(1)

exitcode = 0

def gen_data(d, n):
    global exitcode
    sequences = ["halton", "niederreiter", "reversehalton", "sobol"]
    for seq in sequences:
        out = datadir / f"data_{d:02d}_{n:04d}_{seq}.txt"
        with open(out, "w") as f:
            output = subprocess.run([exe, str(d), str(n), seq], stdout=f)
            if output.returncode != 0:
                print(f"Error: generation of {out} gave return code {output.returncode}")
                exitcode = 1

gen_data(2, 5)
gen_data(2, 20)
gen_data(2, 50)
gen_data(2, 100)
gen_data(2, 200)
gen_data(3, 5)
gen_data(3, 20)
gen_data(3, 50)
gen_data(3, 80)
gen_data(4, 5)
gen_data(4, 15)
gen_data(4, 30)
gen_data(5, 5)
gen_data(5, 15)

sys.exit(exitcode)
