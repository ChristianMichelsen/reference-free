module ReferenceFree


export compute_reads, save_reads, load_reads

#%%

using XAM
using BioSequences
using ProgressMeter
using Serialization

include("utils/Utils.jl")
using .Utils

include("pmdtools/PMDTools.jl")
using .PMDTools

#%%

struct Reads
    sequences_original::Vector{LongDNASeq}
    sequences_aligned::Vector{LongDNASeq}
    PMDs::Vector{Float64}
end


# this is used to handle a call to `print`
function Base.show(io::IO, r::Reads)
    println("""Instance of Reads with fields:""")
    println("""   "sequences_original" """)
    println("""   "sequences_aligned" """)
    println("""   "PMDs" \n """)
    println(io, "sequences_original = ", r.sequences_original[1:5], "\n")
    println(io, "sequences_aligned = ", r.sequences_aligned[1:5], "\n")
    println(io, "PMDs = ", r.PMDs[1:5], "\n")
end


function Base.show(io::IO, ::MIME"text/plain", r::Reads)
    print(io, "Instance of Reads(sequences_original, sequences_aligned, PMDs)")
end


function save_reads(filename::String, reads::Reads)
    named_tuple = (
        sequences_original = reads.sequences_original,
        sequences_aligned = reads.sequences_aligned,
        PMDs = reads.PMDs,
    )
    serialize(filename, named_tuple)
end

function load_reads(filename::String)
    object = deserialize(filename)
    # for (k, v) in pairs(objects)
    #     @eval $k = $v
    # end
    return Reads(object.sequences_original, object.sequences_aligned, object.PMDs)
end


#%%

function compute_reads(filename; max_reads = -1)::Reads

    # Open a BAM file.
    reader = open(BAM.Reader, filename)

    sequences_original = LongDNASeq[]
    sequences_aligned = LongDNASeq[]
    PMDs = Float64[]

    counter = 0

    progress = ProgressUnknown("Reading $filename:", spinner = true)

    record = BAM.Record()
    while !eof(reader)
        empty!(record)
        read!(reader, record)
        ProgressMeter.next!(progress)

        # `record` is a BAM.Record object.
        if BAM.ismapped(record)
            seq_original, seq_aligned, PMD = compute(record)
            push!(sequences_original, seq_original)
            push!(sequences_aligned, seq_aligned)
            push!(PMDs, PMD)
            counter += 1
            if (max_reads > 0) & (counter >= max_reads)
                break
            end
        end

    end

    ProgressMeter.finish!(progress)
    close(reader)

    return Reads(sequences_original, sequences_aligned, PMDs)
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
