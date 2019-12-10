import sys
import tqdm
from random import randrange

EXPECTARGC = 2
alphabet = ['A', 'C', 'G', 'T']
read_length = 64

try:
    assert (len(sys.argv) == EXPECTARGC + 1)
except AssertionError as e:
    print("-----", file=sys.stderr)
    if len(sys.argv) > EXPECTARGC + 1:
        print("Expects only TWO arguments.", file=sys.stderr)
    else:
        print("Expects TWO arguments.", file=sys.stderr)
    print("-----", file=sys.stderr)
    e
    sys.exit()

filename = sys.argv[1] + ".txt"

try:
    # Touch file
    f = open(filename, mode="x")
except FileExistsError as e:
    print(f"-----\nFile: {filename} already exists.\n-----", file=sys.stderr)
    e
    sys.exit()

try:
    assert sys.argv[2].isdigit()
except AssertionError as e:
    print("Expects a number for the second argument", file=sys.stderr)
    e
    sys.exit()
read_count = int(sys.argv[2])

with open(filename, "w") as f:
    for i in tqdm.tqdm(range(read_count)):
        read = ""
        for j in range(read_length):
            read += alphabet[randrange(4)]
        print(read, file=f)
