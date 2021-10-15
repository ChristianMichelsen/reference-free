using XAM
using Printf

function get_record(filename)

    # Open a BAM file.
    reader = open(BAM.Reader, filename)

    local flag
    local cigar
    local seq
    local md

    record = BAM.Record()
    while !eof(reader)
        empty!(record)
        read!(reader, record)

        # `record` is a BAM.Record object.
        if BAM.ismapped(record)
            flag = BAM.flag(record)
            cigar = BAM.cigar(record)
            seq = BAM.sequence(record)
            md = record["MD"]::String
            break
        end
    end

    close(reader)

    # return record
    # return flag
    return record, flag, cigar, seq, md
    # return record

end

filename = "smallsmall.bam"
record, flag, cigar, seq, md = get_record(filename)
record

seq = "GCCTGAGAACAAGTGAGAAAGAAACTCATTCCTGTCTTTCAATGAGTGCTTTTGTGCATTTAGGAGAACTAGGCAGCACACATTTAGGGCTGAAAGATGNA"
cigar = "1S15M1D65M2I18M"
md = "15^T35A30C7C6G1"


# %%
# BAM.refname(record)
# BAM.reflen(record)
# BAM.position(record)
# BAM.mappingquality(record)
# BAM.alignment(record)
# BAM.alignlength(record)
# BAM.tempname(record)
# BAM.seqlength(record)
# BAM.quality(record)

# // Initialize variables for the three substring positions, the content of the
# // CIGAR and MD, and the sequences for the new read, the reference, and the matches
# seq = BAM.sequence(record)

subpos = 0
cigarpos = 0
mdpos = 0

# // Flags to tell if an integer or character should be next in the MD tag
mdintnext_flag = 1


function get_matches_cigar(s)

    matches = collect(m.match for m in eachmatch(r"(\d+|\D)", s))

    numbers = Array{Int}(undef, 0)
    letters = Array{String}(undef, 0)

    for i = 1:2:length(matches)
        number = parse(Int64, matches[i])
        letter = matches[i + 1]
        push!(numbers, number)
        push!(letters, letter)
    end

    return numbers, letters
end

cigar_numbers, cigar_letters = get_matches_cigar(cigar)
cigar_counter = 1


function isa_int(s)
    return isa(tryparse(Int, s), Int)
end

function get_matches_md(s)

    matches = collect(m.match for m in eachmatch(r"(\d+|\D)", md))

    flags = Array{Int}(undef, 0)
    numbers = Array{Int}(undef, 0)
    letters = Array{String}(undef, 0)

    number = 0
    letter = ""

    for match in matches
        flag = 0

        if isa_int(match)
            number = parse(Int64, match)
        else
            number = 0
            letter = match
            flag = 1
        end

        push!(flags, flag)
        push!(numbers, number)
        push!(letters, letter)

    end

    return flags, numbers, letters

end

md_flags, md_numbers, md_letters = get_matches_md(md)
md_counter = 1


modseq = ""
matches = ""
reference = ""

# %%

# // While the substring counter is less than the length of the read
while subpos < length(seq)

    # If cigar position is 0, shift the first element(s) of the CIGAR vector
    # to get the distance and the letter
    if cigarpos == 0
        cigar_number = cigar_numbers[cigar_counter]
        cigar_letter = cigar_letters[cigar_counter]
        cigar_counter += 1
    end

    if cigar_letter == 'H'
        println("Got an H")
    end
            # If you find hard clipping at the beginning of the read,
            # just skip it. Back up and take the next element.
        # if mdstream.rdbuf() -> in_avail() != 0
        #     # goto start;
        # else
        #     # If there's nothing more in the stream, it's hard clipping
        #     # at the end, and you can just finish up.
        #     break;
        # end

    if mdpos == 0 && cigar_letter != 'S' && cigar_letter != 'I'
        mdintnext_flag = md_flags[md_counter]
        md_number = md_numbers[md_counter]
        md_letter = md_letters[md_counter]
    end



    insert_flag = 0
    nonmatch_flag = 0
    n_flag = 0
    pad_flag = 0


    if cigarpos < cigar_number
        # translate_cigar(modseq, read, cigar_letter, subpos, cigarpos, nonmatch_flag, insert_flag, n_flag, pad_flag);
    end


    subpos += 1

end

