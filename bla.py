line = "CRATOS:145:D1UH5ACXX:2:1308:6211:53528	153	Zv9_scaffold3453	49562	26	1S15M1D65M2I18M	=	49562	0	GCCTGAGAACAAGTGAGAAAGAAACTCATTCCTGTCTTTCAATGAGTGCTTTTGTGCATTTAGGAGAACTAGGCAGCACACATTTAGGGCTGAAAGATGNA	(CCDCCCCAEECCDFFFFFFFHECHFHJIHDJIIIIJJJJJJJJJJJJIHGIFJJIGJJJIIIIIIJIJIJIGIIHGFCCJJJJJIJJHGHHHFFFDA1#C	PG:Z:novoalign	AS:i:206	UQ:i:206	NM:i:7	MD:Z:15^T35A30C7C6G1"
reverse = False

import re

cigarparser = re.compile("([0-9]*)([A-Z])")


line = line.rstrip("\n")
col = line.split("\t")
readname = col[0]
position = int(col[3])
chromosome = col[2]

MAPQ = int(col[4])
read = col[9]
quals = col[10]
cigar = col[5]


read = col[9]
ref_seq = ""
newread = ""

MD = line.split("MD:Z:")[1].split()[0].rstrip("\n")

MDlist = re.findall("(\d+|\D+)", MD)

MDcounter = 0
alignment = ""
for e in MDlist:
    if e.isdigit():
        e = int(e)
        alignment += "." * e
        ref_seq += read[MDcounter : MDcounter + e]
        newread += read[MDcounter : MDcounter + e]
        MDcounter += int(e)

    elif "^" in e:
        ef = e.lstrip("^")
        alignment += ef
        continue

    elif e.isalpha():
        alignment += e
        ref_seq += e
        newread += read[MDcounter]
        MDcounter += len(e)

if "I" in cigar or "S" in cigar:

    # find insertions and clips in cigar
    insertions = []
    softclips = []
    cigarcount = 0
    cigarcomponents = cigarparser.findall(cigar)
    for p in cigarcomponents:
        cigaraddition = int(p[0])
        if "I" in p[1]:
            for c in range(cigarcount, cigarcount + cigaraddition):
                insertions.append(c)
        elif "S" in p[1]:
            for c in range(cigarcount, cigarcount + cigaraddition):
                softclips.append(c)
        cigarcount += int(p[0])
    # end cigar parsing

    # redo the read and ref using indel and clip info
    ref_seq = ""
    newread = ""
    alignmentcounter = 0
    for x, r in zip(range(0, len(col[9])), read):
        # break
        if x in insertions:
            ref_seq += "-"
            newread += read[x]
        elif x in softclips:
            ref_seq += "-"
            newread += read[x]
        else:
            newread += read[x]
            refbasealn = alignment[alignmentcounter]
            if refbasealn == ".":
                ref_seq += read[x]
            else:
                ref_seq += refbasealn
            alignmentcounter += 1

if reverse:
    read = revcomp(read)
    ref_seq = revcomp(ref_seq)
    quals = quals[::-1]

real_read = read
real_ref_seq = ref_seq


#%%


print(real_read)
print(real_ref_seq)

# # true
# GCCTGAGAACAAGTGA-GAAAGAAACTCATTCCTGTCTTTCAATGAGTGCTTTTGTGCATTTAGGAGAACTAGGCAGCACACATTTAGGGCTGAAAGATGNA
# NCCTGAGAACAAGTGATGAAAGAAACTCATTCCTGTCTTTCAATGAGTGCTTATGTGCATTTAGGAGAACTAGGCAGCACAC--TCAGGGCTGCAAGATGGA

# # pred
# GCCTGAGAACAAGTGAGAAAGAAACTCATTCCTGTCTTTCAATGAGTGCTTTTGTGCATTTAGGAGAACTAGGCAGCACACATTTAGGGCTGAAAGATGNA
# -CCTGAGAACAAGTGATAAAGAAACTCATTCCTGTCTTTCAATGAGTGCTTTAGTGCATTTAGGAGAACTAGGCAGCACACA--TCGGGCTGACAGATGNG
