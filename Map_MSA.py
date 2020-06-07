"""
This code mapping human protein A and human protein B from same organism
"""
import numpy as np
import sys

def get_sequences_ids(msafile):
    ids=[]
    seqs=[]
    ors=[]
    with open(msafile, 'r') as f:
        for count, line in enumerate(f, start=1):
            if count % 2 == 0:
                seqs.append(line.rstrip('\n'))
            else: 
                ids.append(line.rstrip('\n').split('/')[0])
                ors.append(line.rstrip('\n').split('/')[0].split('_')[1])
    seqs=np.array(seqs)
    seqs_uni,seqs_i=np.unique(seqs,return_index=True)
    ids=np.array(ids)
    ors=np.array(ors)
    ids1=ids[seqs_i]
    ors1=ors[seqs_i]
    return seqs_uni,ids1,ors1

def map_seqs(ors1,ors2,ids1,ids2,seqs_uni1,seqs_uni2):
    seq_m=[]
    id_m=[]
    ors12=np.intersect1d(ors1,ors2)
    for i in ors12:
        i1=np.where(ors1==i)

        i2=np.where(ors2==i)
        if len(i1) < len(i2):
            for j in range(len(i1)):
                seq_m.append(seqs_uni1[i1[j]].tolist()[0]+seqs_uni2[i2[j]].tolist()[0])
                id_m.append(ids1[i1[j]].tolist()[0]+ids2[i2[j]].tolist()[0])
        else:
            for j in range(len(i2)):
                seq_m.append(seqs_uni1[i1[j]].tolist()[0]+seqs_uni2[i2[j]].tolist()[0])
                id_m.append(ids1[i1[j]].tolist()[0]+ids2[i2[j]].tolist()[0])
    return seq_m, id_m

#main function
msafile1=sys.argv[1]
msafile2=sys.argv[2]
outfile=sys.argv[3]

seqs_uni1,ids1,ors1=get_sequences_ids(msafile1)

seqs_uni2,ids2,ors2=get_sequences_ids(msafile2)
seq_m,id_m=map_seqs(ors1,ors2,ids1,ids2,seqs_uni1,seqs_uni2)

outf = open(outfile, "w")

for i in range(len(seq_m)):
    outf.write(seq_m[i] + "\n")

outf.close()
