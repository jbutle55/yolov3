import os
import sys

txt_file = sys.argv[1]
new_file = 'out.txt'

with open(txt_file) as fin:
    with open(new_file, 'w') as fout:
        for line in fin:
            line = line.lstrip('.')
            line = '/data2/DATA_justin/COCO2017' + line
            fout.write(line)
