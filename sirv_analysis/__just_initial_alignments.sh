#! /usr/bin/env bash

# Set the number of CPUs/threads for the analysis
export ncpu=1

# Set the sample ID
export sample_id="cdna"

# Download SIRV annotation data
mkdir -p reference_data
wget -O reference_data/sirvome.fasta \
  http://s3.amazonaws.com/nanopore-human-wgs/rna/referenceFastaFiles/sirv/SIRVome_isoforms_ERCCs_170612a.fasta
wget -O reference_data/sirvome_correct_annotations.gtf \
  http://s3.amazonaws.com/nanopore-human-wgs/rna/referenceFastaFiles/sirv/SIRVome_isoforms_ERCCs_Lot001485_C_170612a.gtf
wget -O reference_data/sirvome_insufficient_annotations.gtf \
  http://s3.amazonaws.com/nanopore-human-wgs/rna/referenceFastaFiles/sirv/SIRVome_isoforms_ERCCs_Lot001485_I_170612a.gtf
wget -O reference_data/sirvome_over_annotations.gtf \
  http://s3.amazonaws.com/nanopore-human-wgs/rna/referenceFastaFiles/sirv/SIRVome_isoforms_ERCCs_Lot001485_O_170612a.gtf

# Download the SIRV BAM file
mkdir -p bam
wget -O bam/sirvome_cdna.bam \
  https://s3.amazonaws.com/nanopore-human-wgs/rna/bamFiles/NA12878-cDNA-1D.pass.dedup.fastq.SIRVome.minimap2.sorted.bam
wget -O bam/sirvome_cdna.bam.bai \
  https://s3.amazonaws.com/nanopore-human-wgs/rna/bamFiles/NA12878-cDNA-1D.pass.dedup.fastq.SIRVome.minimap2.sorted.bam.bai
