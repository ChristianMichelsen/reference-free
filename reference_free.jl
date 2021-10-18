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


function get_record(filename)

    # Open a BAM file.
    reader = open(BAM.Reader, filename)

    counter = 1

    local last_record

    record = BAM.Record()
    while !eof(reader)
        empty!(record)
        read!(reader, record)

        # `record` is a BAM.Record object.
        if BAM.ismapped(record)
            last_record = record
            # push!(seqs, seq)
            # push!(refs, ref)
            counter += 1
        end
    end

    println(counter)
    close(reader)

    return last_record
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

filename = "smallsmall.bam"
record = get_record(filename)


# flag = BAM.flag(record)
read = BAM.sequence(record)
cigar = BAM.cigar(record)
md = record["MD"]::String
quals = BAM.quality(record)
forward = BAM.ispositivestrand(record)

# readcigarmd2seqref_sam2pairwise(read, cigar, md)
# readcigarmd2seqref(read, cigar, md)

sequence, reference, quality = compute_reference(read, cigar, md, quals, forward)

