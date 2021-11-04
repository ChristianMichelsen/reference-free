using ReferenceFree
using Serialization

filename = "../data/gargamell_small_human.bam"
# filename = "../data/MMS8_HGDP00521_French.paired.qualfilt.rmdup.entropy1.0.sort.bam"
# filename = "../data/AltaiNea.hg19_1000g.1.dq.bam"

max_reads = 1_000_000

name = join(split(basename(filename), ".")[1:end-1], ".")
filename_out = "./data/" * name * "__$max_reads.data"


function file_exists(filename; forced = false)
    if forced
        return false
    end
    return isfile(filename)
end


if !file_exists(filename_out)
    reads = compute_reads(filename; max_reads = max_reads)
    save_reads(filename_out, reads)
else
    reads = load_reads(filename_out)
end


