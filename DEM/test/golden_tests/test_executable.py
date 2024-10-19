#!/usr/bin/env python3

import argparse
import pathlib
import subprocess
import sys
import fileinput

parser = argparse.ArgumentParser()
parser.add_argument("datadir", type=pathlib.Path, help="datadir where inputs/outputs are kept")
parser.add_argument("executable", type=pathlib.Path, help="executable to test")

args = parser.parse_args()

datadir = args.datadir
exe = args.executable

if not datadir.exists():
    print(f"Error: datadir '{datadir}' does not exist", file=sys.stderr)
    sys.exit(1)

if not exe.exists():
    print(f"Error: executable '{exe}' does not exist", file=sys.stderr)
    sys.exit(1)

exitcode=0
for inp in datadir.glob("data_*.txt"):
    out = datadir / inp.name.replace("data_", "output_")
    expected = next(fileinput.input(out)).strip()
    fileinput.close()
    f = open(inp, "rb")
    inputdata = f.read()
    f.close()
    output = subprocess.run([exe], input=inputdata, capture_output=True)
    if output.returncode != 0:
        print(f"Error: executable gave exit code {output.returncode} for input {inp}")
        exitcode=1
    output = output.stdout.decode().strip()
    if output != expected:
        print(f"Error: {output} != {expected} for input {inp}")
        exitcode=1

sys.exit(exitcode)
