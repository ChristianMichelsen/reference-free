using XAM
using BioSequences
using ProgressMeter
# using ResumableFunctions

filename = "../data/AltaiNea.hg19_1000g.1.dq.bam"
# filename = "Mez-subsampled.bam"
# filename = "SLMez1.hg18.bam"

get_record_channel(filename) =
    Channel(ctype = BAM.Record) do c

        reader = open(BAM.Reader, filename)
        record = BAM.Record()
        while !eof(reader)
            empty!(record)
            read!(reader, record)
            put!(c, copy(record))
        end
        close(reader)
    end
records = Iterators.take(get_record_channel(filename), 10) |> collect;
record = records[1]
BAM.refname(record)

# @resumable function get_record_yield(filename)::BAM.Record
#     while !eof(reader)
#         @yield record
#     end
# end


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

    close(reader)
    ProgressMeter.finish!(progress)

    return sequences
end

sequences = get_sequences(filename; get_forward_sequence = true, max_sequences = 10_000);
length(sequences)
sequences = get_sequences(
    filename;
    get_forward_sequence = true,
    # max_sequences = 10_000,
    # chromosome = "chr1",
);
sequence = sequences[1]

