module PMDTools

export compute_PMD_score

#%%

using XAM
using BioSequences


include("likelihoods.jl")
include("align.jl")



function get_MD(record)::String
    if haskey(record, "MD")
        md = record["MD"]::String
    else
        md = ""
    end
    return md
end


function compute_reference(read, cigar, md, quals, forward = true)

    seq, ref = readcigarmd2seqref(read, cigar, md)

    seq = LongDNASeq(seq)
    ref = LongDNASeq(ref)

    if !forward
        reverse_complement!(seq)
        reverse_complement!(ref)
        reverse!(quals)
    end

    return seq, ref, quals

end


function record2reference(record::BAM.Record)
    # flag = BAM.flag(record)
    read = BAM.sequence(record)
    cigar = BAM.cigar(record)
    md = record["MD"]::String
    # md = get_MD(record)
    quals = BAM.quality(record)
    forward = BAM.ispositivestrand(record)

    sequence, reference, qualities = compute_reference(read, cigar, md, quals, forward)
    return sequence, reference, qualities
end



function compute_PMD_score(record::BAM.Record, max_position::Int = -1)
    sequence, reference, qualities = record2reference(record)
    PMD = compute_PMD_score(sequence, reference, qualities, max_position)
    return PMD, sequence
end


end