from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
candidate_list=[]
map_imageName_to_index = {}
map_imageName_to_char = {'c.jpg':'s','d.jpg':'e','eb.jpg':'b','g.jpg':'s','g1.jpg':'s',
							'h.jpg':'h','lg.jpg':'s','lm.jpg':'m','m.jpg':'m','mario1.jpg':'s',
							'mario2.jpg':'s','p.jpg':'p','q.jpg':'q','rg.jpg':'s','rm.jpg':'m',
							'rmu.jpg':'n','s.jpg':'s','sb.jpg':'a','st.jpg':'d','vb.jpg':'v',
							'ymu.jpg':'n','A.jpg':'A','B.jpg':'B','C.jpg':'C','D.jpg':'D','E.jpg':'E','LG.jpg':'G','RG.jpg':'G'
							,'new1g.jpg':'s','new2g.jpg':'s','za.jpg':'s','zz.jpg':'s','bel.jpg':'s','UH.jpg':'H','LH.jpg':'H',
							'I1.jpg':'I','I2.jpg':'I','I3.jpg':'I','I4.jpg':'I','I5.jpg':'I','I6.jpg':'I'
							,'I7.jpg':'I','J.jpg':'J','K.jpg':'K','L.jpg':'L','xy.jpg':'s','yz.jpg':'s'
							,'xz.jpg':'s','ab.jpg':'s','bc.jpg':'s','ab.jpg':'s','N1.jpg':'N','N2.jpg':'N'
							,'LM.jpg':'M','M.jpg':'M','RM.jpg':'M','aa.jpg':'s','bb.jpg':'s','abc.jpg':'D'
							,'mno.jpg':'D','xyz.jpg':'D','new11g.jpg':'s','new21g.jpg':'s','p1.jpg':'p','p2.jpg':'p'
							,'p3.jpg':'p','p4.jpg':'p','p5.jpg':'p','p6.jpg':'p','p7.jpg':'p','O1.jpg':'O'
							,'O2.jpg':'O','qw.jpg':'s'}


w, h = 182, 13;
a = np.chararray((h,w))
a[:] = 'a'

i = 0
for filename in os.listdir("candidate_image"):
	#print filename + "a"
	map_imageName_to_index[i] = filename
	x = plt.imread('/home/akash/Desktop/Major_2/candidate_image/'+filename)
	candidate_list.append(x)
	i = i + 1

row_index=0
col_index=0
files=[]
for image in os.listdir("Major_image"):
	files.append(image)

def func(s):
	num1=""
	num2=""
	i=0
	while(s[i]!='_'):
		i=i+1
	i=i+1
	while(s[i]!='_'):
		num1=num1+s[i]
		i=i+1
	i=i+1
	while(s[i]!='.'):
		num2=num2+s[i]
		i=i+1
	n1=int(num1)
	n2=int(num2)
	return tuple((n1,n2))


files=sorted(files,key=func)

for image in files:
	print image
	imageA = plt.imread('/home/akash/Desktop/Major_2/Major_image'+'/'+image)
	#print imageA
	max_val=-1;
	index=-1;
	i=0
	while i<len(candidate_list):
		imageB=candidate_list[i]
		#print type(imageB)
		sc = ssim(imageA,imageB,multichannel=True)
		#print sc
		if(sc>max_val):
			max_val=sc;
			index=i
		i=i+1
	print max_val	
	file_name=map_imageName_to_index[index]
	print file_name
	ch=map_imageName_to_char[file_name]
	a[row_index][col_index]=ch
	col_index=(col_index+1)%182
	if(col_index==0):
		row_index=row_index+1
with open('test.txt','w') as f:
	np.savetxt(f,a,fmt='%s')	
# np.savetxt('text.txt',a)










