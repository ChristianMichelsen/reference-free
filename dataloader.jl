
using XAM
using BioSequences
using ProgressMeter
using DataFrames
using Serialization
# using Serialization
# using GLMakie
# using StatsBase


#%%

filename_ancient = "../data/AltaiNea.hg19_1000g.1.dq.bam"
filename_modern = "../data/MMS8_HGDP00521_French.paired.qualfilt.rmdup.entropy1.0.sort.bam"
# do_augment_data = false
do_augment_data = true


#%%

# function get_seq_lengths(filename)
#     reader = open(BAM.Reader, filename)
#     progress = ProgressUnknown("Getting sequence lengths for $filename:", spinner = true)
#     Ls = Int64[]
#     record = BAM.Record()
#     while !eof(reader)
#         empty!(record)
#         read!(reader, record)
#         ProgressMeter.next!(progress)
#         push!(Ls, BAM.seqlength(record))
#     end
#     ProgressMeter.finish!(progress)
#     close(reader)
#     return Ls
# end
# # Ls = get_seq_lengths(filename_ancient)


function is_reverse(record::BAM.Record)::Bool
    return !BAM.ispositivestrand(record)
end


function contains_N(seq::LongDNASeq)::Bool
    return !isnothing(findfirst(DNA_N, seq))
end

function has_ACGT(seq::LongDNASeq)::Bool
    return occursin(biore"[A]"dna, seq) &&
           occursin(biore"[C]"dna, seq) &&
           occursin(biore"[G]"dna, seq) &&
           occursin(biore"[T]"dna, seq)
end

function is_specific_chromosome(record::BAM.Record, chromosome)
    if chromosome == ""
        return true
    elseif chromosome == BAM.refname(record)
        return true
    else
        return false
    end
end

function passes_requirements(
    seq::LongDNASeq,
    record::BAM.Record,
    chromosome;
    seqlen = 76,
)::Bool
    return is_specific_chromosome(record, chromosome) &&
           length(seq) == seqlen &&
           has_ACGT(seq) &&
           !contains_N(seq)
end

function get_sequences(
    filename;
    get_forward_sequence = true,
    max_sequences = -1,
    chromosome = "",
)

    # Open a BAM file.
    reader = open(BAM.Reader, filename)

    # Initialize a record, to be reused later
    record = BAM.Record()

    # Prepare vector of sequences
    sequences = LongDNASeq[]

    # Initialize spinner
    progress = ProgressUnknown("Reading $filename:", spinner = true)

    # Iterate over BAM records.
    while !eof(reader)
        empty!(record)
        read!(reader, record)
        ProgressMeter.next!(progress)

        sequence = BAM.sequence(record)
        if get_forward_sequence && is_reverse(record)
            reverse_complement!(sequence)
        end

        if passes_requirements(sequence, record, chromosome)
            push!(sequences, sequence)
        end

        if 0 < max_sequences && max_sequences <= length(sequences)
            break
        end

    end

    ProgressMeter.finish!(progress)
    close(reader)

    return sequences
end

seqs_ancient = get_sequences(filename_ancient; chromosome = "1")
seqs_modern = get_sequences(
    filename_modern;
    chromosome = "chr1",
    max_sequences = length(seqs_ancient),
)

if do_augment_data
    seqs_ancient = [seqs_ancient; reverse_complement.(seqs_ancient)]
    seqs_modern = [seqs_modern; reverse_complement.(seqs_modern)]
end

sequences = [seqs_ancient; seqs_modern]
labels = [
    ones(Int8, length(seqs_ancient))
    zeros(Int8, length(seqs_modern))
]


# hist(
#     Ls,
#     bins = 50,
#     color = :red,
#     strokewidth = 1,
#     strokecolor = :black,
#     normalization = :pdf,
# )


#%%

df_y = DataFrame(y = labels)
df_X = DataFrame(permutedims(hcat(collect.(sequences)...)), :auto)

df = hcat(df_y, df_X)

filename_out = "./df.data"
serialize(filename_out, (df = df,))
