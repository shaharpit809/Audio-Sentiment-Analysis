from os import listdir
from os.path import isfile, join
import os
import subprocess
import sys

old_path = sys.argv[1]
new_path = sys.argv[2]

files = []
data_folders = os.listdir(old_path)

#Separating audio clips on the basis of 8 emotions and storing them together accordingly
#Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
for i in data_folders:
	temp_join = join(old_path,i)
	for j in listdir(temp_join):
		if isfile(join(temp_join,j)):
			files.append(j)
			temp = j.split('.')[0].split('-')
			subprocess.call("mv %s %s" % (join(temp_join,j),join(new_path,temp[2])),shell=True)