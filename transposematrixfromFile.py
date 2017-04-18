import numpy as np
from io import StringIO
i=13
while i<=13:
	print i
	filename = "level"+str(i)+".txt"
	raw_text = open(filename).read()
	raw_text=unicode(raw_text,"utf-8")
	c = StringIO(raw_text)
	a = np.loadtxt(c,dtype=np.str)
	a = np.transpose(a)
	with open('train'+str(i)+'.txt','w') as f:
		np.savetxt(f,a,fmt='%s',delimiter="")
	i = i + 1	