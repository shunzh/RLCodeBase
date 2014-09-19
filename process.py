import random
import pprint
import sys
import numpy
import math

# output average of every vector in the matrix, and 95% confidence interval
for fName in sys.argv[1:]:
	dataSet = dict()
	f	= open(fName, 'r')
	line = f.readline()
	while (line):
		data = line.split(" ")
		x = float(data[0])
		y = float(data[1])

		if x in dataSet.keys():
			dataSet[x].append(y)
		else:
			dataSet[x] = [y]

		line = f.readline()

	fout = open(fName + ".out", 'w')
	dataList = sorted(dataSet.items())
	for x, vector in dataList:
		mean = numpy.mean(vector)
		ci = 1.96 * numpy.std(vector) / numpy.sqrt(len(vector)) # 95% confidence interval

		fout.write(str(x) + ' ' + str(mean) + ' ' + str(ci) + '\n')

	f.close()
	fout.close()
