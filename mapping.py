import os
import sys
f = open("map_clsloc.txt")
imagenet = {}
for lines in f:
	splitted = lines.strip().split()
	imagenet[splitted[0]] = splitted[1]
f.close()
f = open("wnids.txt")
final_indices = {}
i = 1
for lines in f:
	final_indices[int(imagenet[lines.strip()])] = i
	i+=1
print(final_indices)
