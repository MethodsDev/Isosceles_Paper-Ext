#!/usr/bin/env python3

import sys, os, re


TP_FP_table = "LRAA.insuff.TP_FP_classes"

LRAA_gtf = f"LRAA.insufficient.clean.gtf"


FP_ids = set()
with open(TP_FP_table) as fh:
    header = next(fh)
    for line in fh:
        line = line.rstrip()
        transcript_id, transcript_class = line.split("\t")
        if transcript_class == "FP":
            FP_ids.add(transcript_id)


ofh_TPs = open("LRAA-TPs.gtf", "wt")
ofh_FPs = open("LRAA-FPs.gtf", "wt")


with open(LRAA_gtf) as fh:
    for line in fh:
        if line[0] == "\n" or line[0] == "#":
            continue

        line = line.rstrip()
        m = re.search('transcript_id "([^"]+)"', line)
        if m is not None:
            transcript_id = m.group(1)
            if transcript_id in FP_ids:
                print(line, file=ofh_FPs)
            else:
                print(line, file=ofh_TPs)

        else:
            raise RuntimeError("No transcript_id parsed from line: " + line)


print("Done")
sys.exit(0)
