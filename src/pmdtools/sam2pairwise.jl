
function readcigarmd2seqref_sam2pairwise(read, cigar, md)
    seq_str = convert(String, read)
    command = `./readcigarmd2seqref $seq_str $cigar $md`
    output = readchomp(command)
    seq, ref = split(output)
    return seq, ref
end

