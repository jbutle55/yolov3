import os
import sys

txt_file = sys.argv[1]
#new_file = 'out.txt'
new_file = sys.argv[2]

with open(txt_file) as fin:
    with open(new_file, 'w') as fout:
        for line in fin:
            #line = line.lstrip('.')
            #line = '/Users/justinbutler/Desktop/school/Calgary/ML_Work/Datasets/COCO2017' + line
            line = line.rstrip()
            line = '/data2/DATA_justin/stanford_dataset/sdd/images/' + line + '.jpg\n'

            fout.write(line)
