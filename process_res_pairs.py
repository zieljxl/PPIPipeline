import linecache 
import numpy as np
import matplotlib.pyplot as plt
import sys
def get_inter_pairs(inputf,l1):

	r1=[]
	r2=[]
	s=[]
	ls=[]
	with open(inputf,'r') as f:
    		for count, line in enumerate(f):    
	#l=f.readline().rstrip('\n').split(' ')
        		l=line.rstrip('\n').split(' ')
        		if int(l[0])<=l1 and int(l[1])>l1:
            			r1.append(int(l[0]))
            			r2.append(int(l[1])-l1)
            			s.append(float(l[2]))

	return r1,r2,s

def get_3d_pairs(contact_file):

	r1=[]
	r2=[]
	s=[]
	with open(contact_file,'r') as f:
    		for count, line in enumerate(f):
        		if count!=0:
	#l=f.readline().rstrip('\n').split(' ')
        			l=line.rstrip('\n').split(' ')
        			l=list(filter(None,l))

        			r1.append(int(l[0]))
        			r2.append(int(l[3]))

	return r1,r2

def get_domain_protein_mapping_reference(align):
	#take the name of files
	
	#ranked = input("Ranked DI file name: ")

	#get information from manual alignment file
	domain = linecache.getline(align, 1)
	protein_id = linecache.getline(align, 6)[:4]

	d_init = int(linecache.getline(align, 2))
	d_end = int(linecache.getline(align, 4)) 
	p_init = int(linecache.getline(align, 7))
	p_end = int(linecache.getline(align, 9)) 

	#domain sequence string
	l1 = linecache.getline(align, 3)
	#protein sequence string
	l2 = linecache.getline(align, 8)
	x1 = len(l1)
	x2 = len(l2)
	delta = max(x1,x2)
	#get the difference between initial positions	delta = max(x1,x2)
	#domain code and respective number
	d = []
	dn = []
	#protein code and respective number
	p = []
	pn = []
	#fill d and p arrays with domain and protein sequences
	for i in range(0,delta-1):
		d.append(l1[i])
		p.append(l2[i])
	#compute the original positions in the system
	j1=-1
	j2=-1
	references=[]
	#output1.write('Domain \t n \t Protein \t n\n')
	for i in range(0,len(d)):
		if d[i]!='.':
			j1+=1
			dn.append(str(d_init+j1))
		if d[i]=='.':
			dn.append('')
		if p[i]!='-':
			j2+=1
			pn.append(str(p_init+j2))
		if p[i]=='-':
			pn.append('')
	#output1.write(d[i]+'\t'+str(dn[i])+'\t'+p[i]+'\t'+str(pn[i])+'\n')
		references.append([d[i],dn[i],p[i],pn[i]])
	return d,dn,p,pn


#########main function############
#get residue pairs with calculated scores
score_files_ranked=sys.argv[1]
len_protein1=int(sys.argv[2])
res1,res2,score=get_inter_pairs(score_files_ranked,len_protein1)


#get mapping reference between domains and proteins, [domain aa, domain no.,protein aa,protein no.]
#protein 1
alignmentfile1 =sys.argv[3]
d1,dn1,p1,pn1=get_domain_protein_mapping_reference(alignmentfile1)
#protein 2, same order with joined MSA
alignmentfile2 = sys.argv[4]
d2,dn2,p2,pn2=get_domain_protein_mapping_reference(alignmentfile2)

#map residue positions in domain (with score calculation) to pdb structural positions

mapped_res1=[]
mapped_res2=[]
mapped_score=[]
for i in range(len(res1)):
	if str(res1[i]) in dn1 and str(res2[i]) in dn2:
		if pn1[dn1.index(str(res1[i]))] and pn2[dn2.index(str(res2[i]))]:
			mapped_res1.append(int(pn1[dn1.index(str(res1[i]))]))
			mapped_res2.append(int(pn2[dn2.index(str(res2[i]))]))
			mapped_score.append(float(score[i]))	
	
contact_file=sys.argv[5]
top=int(sys.argv[6])
g1,g2=get_3d_pairs(contact_file)
fig=plt.figure
print(mapped_score)
output_f=sys.argv[7]
fw=open(output_f,'w')
for i in range(top):
	fw.write(str(mapped_res1[i])+' '+str(mapped_res2[i])+' '+str(mapped_score[i])+'\n')
fw.close()


h1=plt.scatter(g1,g2,s=2)
h2=plt.scatter(mapped_res1[:top],mapped_res2[:top],s=2,marker='*')

plt.title('ResPRE contact prediction')
plt.xlabel('Protein 1')
plt.ylabel('Protein 2')
plt.legend((h1,h2),('Real contacts','Predicted contacts'))

plt.show()

