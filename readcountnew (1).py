

import pysam
import re

bamfile = pysam.AlignmentFile("nKP_merged_sorted.bam", "rb")

r1_bed = open("R1.filtered.bed", "w")
r2_bed = open("R2.filtered.bed", "w")

valid_chroms = {str(i) for i in range(1, 23)} | {"X", "Y", "M"}

for read in bamfile.fetch(until_eof=True):
    if (read.is_unmapped or read.is_secondary or read.is_supplementary or
        read.query_sequence is None or len(read.query_sequence) != 98):
        continue

    chrom = bamfile.get_reference_name(read.reference_id)
    chrom_clean = chrom.replace("chr", "")  # normalize if "chr1", "chrX", etc.

    if chrom_clean not in valid_chroms:
        continue

    start = read.reference_start
    end = read.reference_end

    if read.cigartuples:
        if read.cigartuples[0][0] == 4:
            start -= read.cigartuples[0][1]
        if read.cigartuples[-1][0] == 4:
            end += read.cigartuples[-1][1]

    start = max(0, start)
    strand = "-" if read.is_reverse else "+"

    name = read.query_name
    bed_line = f"{chrom}\t{start}\t{end}\t{name}\t0\t{strand}\n"

    if read.is_read1:
        r1_bed.write(bed_line)
    elif read.is_read2:
        r2_bed.write(bed_line)

r1_bed.close()
r2_bed.close()
bamfile.close()
