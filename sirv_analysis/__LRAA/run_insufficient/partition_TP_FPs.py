#!/usr/bin/env python3

import sys, os, re


TP_FP_table = "LRAA.insuff.TP_FP_classes"

LRAA_gtf = "LRAA.insufficient.clean.gtf"
correct_annots = "../../reference_data/sirvome_correct_annotations.gtf"

TP_ids = set()
FP_ids = set()
with open(TP_FP_table) as fh:
    header = next(fh)
    for line in fh:
        line = line.rstrip()
        vals = line.split("\t")
        transcript_id, transcript_class = vals[7], vals[1]

        if transcript_class == "FP":
            FP_ids.add(transcript_id)

        ref_transcript_id = vals[3]
        TP_ids.add(ref_transcript_id)


ofh_TPs = open("LRAA-TPs.gtf", "wt")
ofh_FPs = open("LRAA-FPs.gtf", "wt")
ofh_FNs = open("LRAA-FNs.gtf", "wt")

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


with open(correct_annots) as fh:
    for line in fh:
        if line[0] == "\n" or line[0] == "#":
            continue

        vals = line.split("\t")
        if vals[2] != "exon":
            continue

        line = line.rstrip()
        m = re.search('transcript_id "([^"]+)"', line)
        if m is not None:
            transcript_id = m.group(1)
            if transcript_id not in TP_ids:
                print(line, file=ofh_FNs)

        else:
            raise RuntimeError("No transcript_id parsed from line: " + line)


print("Done")

sys.exit(0)
