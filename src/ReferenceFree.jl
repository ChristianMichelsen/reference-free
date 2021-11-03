module ReferenceFree


export load_data

#%%

using XAM
using BioSequences

include("utils/Utils.jl")
using .Utils

include("pmdtools/PMDTools.jl")
using .PMDTools

#%%

function load_data(filename; max_length = -1)

    # Open a BAM file.
    reader = open(BAM.Reader, filename)

    PMDs = Float64[]
    sequences = LongDNASeq[]
    counter = 0

    record = BAM.Record()
    while !eof(reader)
        empty!(record)
        read!(reader, record)

        # `record` is a BAM.Record object.
        if BAM.ismapped(record)
            PMD, sequence = compute_PMD_score(record)
            push!(PMDs, PMD)
            push!(sequences, sequence)
            counter += 1
            if (max_length > 0) & (counter >= max_length)
                break
            end
        end
    end

    return sequences, PMDs
end

#%%

function load_damaged_record(filename)
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


end
