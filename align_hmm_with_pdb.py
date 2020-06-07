import os
import sys


#hmmfile='ecTbetaR2.hmm'
hmmfile=sys.argv[1]
hmm_press='/home/hts/Downloads/finalhmmer/bin/hmmpress'+' '+hmmfile
os.system(hmm_press)

fastafile=sys.argv[2]
scanfile=sys.argv[3]
hmmscan='/home/hts/Downloads/finalhmmer/bin/hmmscan'+' -o '+scanfile+' --notextw '+hmmfile+' '+fastafile
print(hmmscan)
os.system(hmmscan)
