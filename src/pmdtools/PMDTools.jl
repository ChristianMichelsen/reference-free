module PMDTools

export compute

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


function make_alignment(read, cigar, md, quals, forward = true)

    seq_aligned, ref_aligned = align(read, cigar, md)

    seq_aligned = LongDNASeq(seq_aligned)
    ref_aligned = LongDNASeq(ref_aligned)
    qual_aligned = quals

    if !forward
        reverse_complement!(seq_aligned)
        reverse_complement!(ref_aligned)
        reverse!(qual_aligned)
    end

    return seq_aligned, ref_aligned, qual_aligned

end


function make_alignment(record::BAM.Record)
    # flag = BAM.flag(record)
    read = BAM.sequence(record)
    cigar = BAM.cigar(record)
    md = record["MD"]::String
    # md = get_MD(record)
    quals = BAM.quality(record)
    forward = BAM.ispositivestrand(record)

    seq_aligned, ref_aligned, qual_aligned = make_alignment(read, cigar, md, quals, forward)
    return seq_aligned, ref_aligned, qual_aligned
end



function compute(record::BAM.Record, max_position::Int = -1)

    # get original sequence
    seq_original = BAM.sequence(record)

    # get aligned sequence, reference and quality score
    seq_aligned, ref_aligned, qual_aligned = make_alignment(record)

    # compute the PMD score
    PMD = compute_PMD_score(seq_aligned, ref_aligned, qual_aligned, max_position)

    # if max_position is set, then return slices
    if max_position > 0
        seq_original =
            seq_original[1:max_position], seq_aligned = seq_aligned[1:max_position]
    end

    return seq_original, seq_aligned, PMD
end


end