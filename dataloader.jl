
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


function contains_N(seq)
    return !isnothing(findfirst(DNA_N, seq))
end

function has_ACGT(seq)
    return occursin(biore"A+C+G+T+"dna, seq)
end


function get_seq(record)
    # chr_flag = BAM.refname(record) == "chr1" || BAM.refname(record) == "chr2"
    # if chr_flag && (BAM.seqlength(record) == 76)
    if BAM.seqlength(record) == 76
        seq = BAM.sequence(record)
        if !contains_N(seq) && has_ACGT(seq)
            return seq
        end
    end
    return dna""
end


function get_seq_lengths(filename)
    reader = open(BAM.Reader, filename)
    progress = ProgressUnknown("Getting sequence lengths for $filename:", spinner = true)
    Ls = Int64[]
    record = BAM.Record()
    while !eof(reader)
        empty!(record)
        read!(reader, record)
        ProgressMeter.next!(progress)
        push!(Ls, BAM.seqlength(record))
    end
    ProgressMeter.finish!(progress)
    close(reader)
    return Ls
end
# Ls = get_seq_lengths(filename_ancient)



function get_sequences(filename; max_length = -1)
    reader = open(BAM.Reader, filename)
    progress = ProgressUnknown("Reading $filename:", spinner = true)
    seqs = LongDNASeq[]
    record = BAM.Record()
    while !eof(reader)
        empty!(record)
        read!(reader, record)
        ProgressMeter.next!(progress)
        seq = get_seq(record)
        if length(seq) > 0
            push!(seqs, seq)
        end
        if 0 < max_length && max_length <= length(seqs)
            break
        end
    end
    ProgressMeter.finish!(progress)
    close(reader)
    return seqs
end

seqs_ancient = get_sequences(filename_ancient)
seqs_modern = get_sequences(filename_modern, max_length = length(seqs_ancient))

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
