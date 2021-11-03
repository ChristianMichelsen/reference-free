
include("../utils/Utils.jl")
using .Utils


function readcigarmd2seqref(read::String, cigar::String, md::String)

    ref_seq = ""
    newread = ""

    # # if empty md tag
    # if md == ""
    #     md = string(length(read))
    # end

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

