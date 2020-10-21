import os
import sys

txt_file = sys.argv[1]
new_file = 'out.txt'

with open(txt_file) as fin:
    with open(new_file, 'w') as fout:
        for line in fin:
            line = line.lstrip('.')
            line = '/Users/justinbutler/Desktop/school/Calgary/ML_Work/Datasets/COCO2017' + line
            fout.write(line)
