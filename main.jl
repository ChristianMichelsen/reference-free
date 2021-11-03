using ReferenceFree
using Serialization

filename = "../data/smallsmall.bam"
filename = "../data/gargamell_small_human.bam"
filename = "../data/MMS8_HGDP00521_French.paired.qualfilt.rmdup.entropy1.0.sort.bam"
filename = "../data/AltaiNea.hg19_1000g.1.dq.bam"

max_length = 1_000_000

name = join(split(basename(filename), ".")[1:end-1], ".")
filename_out = "./data/" * name * "__$max_length.data"


function file_exists(filename; forced = false)
    if forced
        return false
    end
    return isfile(filename)
end


if !file_exists(filename_out)
    sequences, PMDs = load_data(filename, max_length = max_length)
    serialize(filename_out, (sequences = sequences, PMDs = PMDs))

else
    objects = deserialize(filename_out)
    for (k, v) in pairs(objects)
        @eval $k = $v
    end

end
