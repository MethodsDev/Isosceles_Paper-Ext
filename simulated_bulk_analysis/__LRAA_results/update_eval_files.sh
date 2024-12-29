
# update non-downsampled quants
cat sim_bulk_all.quant.expr | print.pl 1 6 > /home/bhaas/projects/Isosceles_review/Isosceles_Paper/simulated_bulk_analysis/report_data/truncated_bulk_rnaseq_quant/LRAA.tsv 


# update gtfs for downsampled
cat sim_bulk_down_10.gtf | egrep -v ^\# | perl -lane 'if (/\w/) { print;}' > /home/bhaas/projects/Isosceles_review/Isosceles_Paper/simulated_bulk_analysis/report_data/truncated_bulk_rnaseq_denovo/LRAA_down10.gtf 
cat sim_bulk_down_20.gtf | egrep -v ^\# | perl -lane 'if (/\w/) { print;}' > /home/bhaas/projects/Isosceles_review/Isosceles_Paper/simulated_bulk_analysis/report_data/truncated_bulk_rnaseq_denovo/LRAA_down20.gtf 
cat sim_bulk_down_30.gtf | egrep -v ^\# | perl -lane 'if (/\w/) { print;}' > /home/bhaas/projects/Isosceles_review/Isosceles_Paper/simulated_bulk_analysis/report_data/truncated_bulk_rnaseq_denovo/LRAA_down30.gtf 


# update quants for downsampled
cat sim_bulk_down_10.quant.expr | print.pl 1 6 > /home/bhaas/projects/Isosceles_review/Isosceles_Paper/simulated_bulk_analysis/report_data/truncated_bulk_rnaseq_denovo/LRAA_down10.tsv 
cat sim_bulk_down_20.quant.expr | print.pl 1 6 > /home/bhaas/projects/Isosceles_review/Isosceles_Paper/simulated_bulk_analysis/report_data/truncated_bulk_rnaseq_denovo/LRAA_down20.tsv
cat sim_bulk_down_30.quant.expr | print.pl 1 6 > /home/bhaas/projects/Isosceles_review/Isosceles_Paper/simulated_bulk_analysis/report_data/truncated_bulk_rnaseq_denovo/LRAA_down30.tsv

