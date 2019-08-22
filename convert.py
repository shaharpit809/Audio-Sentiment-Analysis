import sys
import glob
from pydub import AudioSegment

#Path to all mp3 files
files_path = sys.argv[1] + '/*.mp3'

for file in sorted(glob.iglob(files_path)):
    name = file.split('.mp3')[0]
    sound=AudioSegment.from_mp3(file)
    sound.export(name + '.wav',format='wav')
