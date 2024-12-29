#!/usr/bin/env python3

import sys, os, re


if len(sys.argv) < 2:
    sys.exit("usage: provide version string as param")

LRAA_version = sys.argv[1]


FPs_table = "/home/bhaas/projects/Isosceles_review/Isosceles_Paper/reports/__compare_LRAA_Isosceles/LRAA_fps_with_tx_ids.tsv"

LRAA_gtf = f"/home/bhaas/projects/Isosceles_review/Isosceles_Paper/simulated_bulk_analysis/__LRAA_results/{LRAA_version}/sim_bulk_down_10.gtf"


FP_ids = set()
with open(FPs_table) as fh:
    header = next(fh)
    for line in fh:
        line = line.rstrip()
        transcript_id = line.split("\t")[7]
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
