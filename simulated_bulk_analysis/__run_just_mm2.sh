#! /usr/bin/env bash

# Set the number of CPUs/threads for the analysis
export ncpu=8

# Set the sample ID
export sample_id="truncated_bulk_rnaseq"

# Align Nanopore reads to the reference genome with minimap2
mkdir -p bam
../software/bin/minimap2 -t $ncpu \
  -ax splice --secondary=no \
  --junc-bed ../reference_data/known_introns.bed --junc-bonus 15 \
  ../reference_data/genome.fasta \
  ../input_data/fastq_sim/${sample_id}.fastq.gz \
  | samtools sort -o bam/${sample_id}.bam
samtools index bam/${sample_id}.bam

