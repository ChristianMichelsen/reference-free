

using XAM
using BioSequences
using ProgressMeter
# using Serialization
using GLMakie
using StatsBase


#%%

filename_ancient = "../data/SLVi33.16.hg18.bam"
filename_modern = "../data/MMS8_HGDP00521_French.paired.qualfilt.rmdup.entropy1.0.sort.bam"
do_augment_data = false
# do_augment_data = true
max_cols = 5
# max_cols = 20


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

using GLM, DataFrames


df = hcat(
    DataFrame(y = labels),
    DataFrame(permutedims(hcat(collect.(sequences)...))[:, 1:max_cols], :auto),
)

levels = [DNA_A, DNA_C, DNA_G, DNA_T]
coding = DummyCoding(levels = levels)

variable_names = names(df, Not(:y))

contrast = Dict(Symbol(name) => coding for name in variable_names)


formula = @formula(y ~ 1 + x1 * x2 * x3)
# formula = term(:y) ~ sum(term.([1; variable_names]))


logistic = glm(formula, df, Bernoulli(), LogitLink(); contrasts = contrast)
# exp.(coef(logistic))

prediction = predict(logistic, df);
prediction_class = map(x -> Int(x > 0.5), prediction);


prediction_df = DataFrame(
    y_actual = df.y,
    y_predicted = prediction_class,
    prob_predicted = prediction,
    correctly_classified = df.y .== prediction_class,
);
# prediction_df.correctly_classified = prediction_df.y_actual .== prediction_df.y_predicted


accuracy = mean(prediction_df.correctly_classified)
println(
    "Using $(length(variable_names)) variables, the accuracy is ",
    round(100 * accuracy, digits = 2),
    "%",
)
