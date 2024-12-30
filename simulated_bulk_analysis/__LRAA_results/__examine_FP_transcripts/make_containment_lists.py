#!/usr/bin/env python3

import sys, os, re
import json

print("\t".join(["contain-er", "contain-ey"]))

with open("sim_bulk_down_10.quant.expr", "rt") as fh:
    header = next(fh)
    for line in fh:
        line = line.rstrip()
        vals = line.split("\t")
        transcript_id = vals[1]

        if len(vals) < 8:
            continue

        containment_list = vals[7]

        if containment_list == "":
            continue

        containments_set = eval(containment_list)
        for contained in containments_set:
            print("\t".join([transcript_id, contained]))
