using XAM
using BenchmarkTools
using BioSequences


function readcigarmd2seqref_sam2pairwise(read, cigar, md)
    seq_str = convert(String, read)
    command = `./readcigarmd2seqref $seq_str $cigar $md`
    output = readchomp(command)
    seq, ref = split(output)
    return seq, ref
end


function is_integer(s)
    return isa(tryparse(Int, s), Int)
end

function is_alpha(text)
    "Faster version of isletter function. Only works for ASCI, not unicode."
    all(c -> 'a' <= c <= 'z' || 'A' <= c <= 'Z', text)
end


# %%

function readcigarmd2seqref(read::String, cigar::String, md::String)

    ref_seq = ""
    newread = ""

    MDlist = collect(m.match for m in eachmatch(r"(\d+|\D+)", md))

    MDcounter = 1
    alignment = ""


    for e in MDlist

        if is_integer(e)
            i = tryparse(Int, e)
            alignment *= "."^i
            ref_seq *= read[MDcounter:MDcounter+i-1]
            newread *= read[MDcounter:MDcounter+i-1]
            MDcounter += i

        elseif occursin("^", e)
            ef = lstrip(e, '^')
            alignment *= ef
            continue

        elseif is_alpha(e)
            alignment *= e
            ref_seq *= e
            newread *= read[MDcounter]
            MDcounter += length(e)
        end

    end


    if occursin("I", cigar) | occursin("S", cigar)

        # find insertions and clips in cigar
        insertions = Int[]
        softclips = Int[]
        cigarcount = 1

        cigarcomponents = collect(c.captures for c in eachmatch(r"([0-9]*)([A-Z])", cigar))

        for p in cigarcomponents

            cigaraddition = tryparse(Int, p[1])

            if occursin("I", p[2])
                for c = cigarcount:cigarcount+cigaraddition-1
                    push!(insertions, c)
                end

            elseif occursin("S", p[2])
                for c = cigarcount:cigarcount+cigaraddition-1
                    push!(softclips, c)
                end
            end

            cigarcount += cigaraddition

        end
        # end cigar parsing


        # redo the read and ref using indel and clip info
        ref_seq = ""
        # newread = ""
        alignmentcounter = 1

        for (x, r) in zip(1:length(read), read)

            if x in insertions
                ref_seq *= "-"
                # newread *= read[x]

            elseif x in softclips
                ref_seq *= "-"
                # newread *= read[x]

            else
                # newread *= read[x]
                refbasealn = alignment[alignmentcounter]
                if refbasealn == '.'
                    ref_seq *= read[x]
                else
                    ref_seq *= refbasealn
                end
                alignmentcounter += 1

            end
        end

    end

    return read, ref_seq

end


function readcigarmd2seqref(read::LongDNASeq, cigar::String, md::String)
    return readcigarmd2seqref(convert(String, read), cigar, md)
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

# %%


function record2reference(record::BAM.Record)
    # flag = BAM.flag(record)
    read = BAM.sequence(record)
    cigar = BAM.cigar(record)
    md = record["MD"]::String
    quals = BAM.quality(record)
    forward = BAM.ispositivestrand(record)

    sequence, reference, qualities = compute_reference(read, cigar, md, quals, forward)
    return sequence, reference, qualities
end


# parameter p in geometric probability distribution of PMD
const PMDpparam = 0.3

# constant C in geometric probability distribution of PMD
const PMDconstant = 0.01

# True biological polymorphism between the ancient individual and the reference sequence
const polymorphism_ancient = 0.001

# True biological polymorphism between the contaminants and the reference sequence
const polymorphism_contamination = 0.001


function phred2prob(Q)
    q = Q % Int
    return 10.0^(-q / 10.0)
end

function phreds2probs(Qs)
    return phred2prob.(Qs)
end

function damage_model_modern(z)
    return 0.001
end

function damage_model_ancient(z, p = PMDpparam, C = PMDconstant)
    return Dz = p * (1 - p)^(z - 1) + C
end

function L_match(z, damage_model, quality, polymorphism)
    P_damage = damage_model(z)
    P_error = phred2prob(quality) / 3
    P_poly = polymorphism
    P_match =
        (1.0 - P_damage) * (1.0 - P_error) * (1.0 - P_poly) +
        (P_damage * P_error * (1.0 - P_poly)) +
        (P_error * P_poly * (1.0 - P_damage))
    return P_match
end

function L_mismatch(z, damage_model, quality, polymorphism)
    P_match = L_match(z, damage_model, quality, polymorphism)
    P_mismatch = 1 - P_match
    return P_mismatch
end


function compute_likelihood_ratio(
    sequence::LongDNASeq,
    reference::LongDNASeq,
    qualities::Vector{UInt8},
    max_position::Int = -1,
)::Float64

    if max_position < 1
        max_position = length(sequence)
    end

    L_D = 1.0
    L_M = 1.0

    z = 1
    for (s, r, q) in zip(sequence, reference, qualities)

        if s == DNA_N | r == DNA_N
            continue
        end

        if r == DNA_C
            if s == DNA_T
                L_D *= L_mismatch(z, damage_model_ancient, q, polymorphism_ancient)
                L_M *= L_mismatch(z, damage_model_modern, q, polymorphism_contamination)
            elseif s == DNA_C
                L_D *= L_match(z, damage_model_ancient, q, polymorphism_ancient)
                L_M *= L_match(z, damage_model_modern, q, polymorphism_contamination)
            end

        elseif r == DNA_G

            if s == DNA_A
                L_D *= L_mismatch(z, damage_model_ancient, q, polymorphism_ancient)
                L_M *= L_mismatch(z, damage_model_modern, q, polymorphism_contamination)
            elseif s == DNA_G
                L_D *= L_match(z, damage_model_ancient, q, polymorphism_ancient)
                L_M *= L_match(z, damage_model_modern, q, polymorphism_contamination)
            end
        end

        z += 1

        if z > max_position
            break
        end

    end

    LR = log(L_D / L_M)
    return LR

end

function compute_likelihood_ratio(record::BAM.Record, max_position::Int = -1)
    sequence, reference, qualities = record2reference(record)
    LR = compute_likelihood_ratio(sequence, reference, qualities, max_position)
    return LR, sequence
end


# LR, sequence = compute_likelihood_ratio(record)

# sequence, reference, qualities = record2reference(record)

function load_data(filename)

    # Open a BAM file.
    reader = open(BAM.Reader, filename)

    LRs = Float64[]
    sequences = LongDNASeq[]

    record = BAM.Record()
    while !eof(reader)
        empty!(record)
        read!(reader, record)

        # `record` is a BAM.Record object.
        if BAM.ismapped(record)
            LR, sequence = compute_likelihood_ratio(record)
            push!(LRs, LR)
            push!(sequences, sequence)
        end
    end

    # println(counter)
    close(reader)

    return sequences, LRs
end


function load_record(filename)
    "Load record with many errors"

    # Open a BAM file.
    reader = open(BAM.Reader, filename)

    record = BAM.Record()
    while !eof(reader)
        empty!(record)
        read!(reader, record)

        # `record` is a BAM.Record object.
        if BAM.ismapped(record)
            if length(BAM.cigar(record)) > 10 | length(record["MD"]::String) > 10
                break
            end
        end
    end

    # println(counter)
    close(reader)

    return record
end

#%%


filename = "smallsmall.bam"


do_run = false
if do_run
    sequences, LRs = load_data(filename)

    # serialize/save to file
    using Serialization
    serialize("test", (sequences = sequences, LRs = LRs))

else
    # deserialize
    using Serialization
    objects = deserialize("test")
    for (k, v) in pairs(objects)
        @eval $k = $v
    end

end

# save("test.jld", (sequences, LRs))


lengths = length.(sequences)

#%%

if false

    using Plots
    gr()

    histogram(LRs)
    histogram(lengths)
    histogram2d(lengths, LRs)
end

# %%

# record = load_record(filename)
# sequence, reference, qualities = record2reference(record)
# compute_likelihood_ratio(sequence, reference, qualities)
# compute_likelihood_ratio(complement(sequence), complement(reference), qualities)

# %%

min_length = minimum(lengths)
const dna_letters = [DNA_A, DNA_C, DNA_G, DNA_T]

function onehot(sequence)
    return dna_letters .== permutedims(collect(sequence))
end


# X = [onehot(sequence[1:min_length]) for sequence in sequences];

#%%


