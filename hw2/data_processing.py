#!/usr/bin/env python
# -*- coding: utf-8 -*-


def transform(line):
    s = [n for n in line.split(",")]
    new_line = []
    for idx, char in enumerate(s):
        if idx == 57:
            new_line.append(int(char[0]))
        if idx < 54:
            if not char == "0":
                new_line.append(1)
            else:
                new_line.append(0)
        else:
            pass
    return str(new_line).strip("[]").replace(" ", "")+"\n"


def process_datafile(datafile):
    with open(datafile, "r") as f:
        # line = f.read()
        # for line in f:
        # current_line = f.read()
        content = [x for x in f.readlines()]
        # for l in
        for line in content:
            new_line = transform(line)
            with open("spambase.dat", "a") as new_file:
                new_file.write(new_line)


if __name__ == "__main__":
    process_datafile("spambase.data")

