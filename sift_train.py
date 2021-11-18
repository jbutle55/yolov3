from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *

import kornia

batch_size = 8
path = '/Users/justinbutler/Desktop/school/Calgary/ML_Work/Datasets/COCO2017/images/train2017'
imgsz = 416
dataset = LoadImagesAndLabels(path, imgsz, batch_size, rect=True, single_cls=False, pad=0.5)
batch_size = min(batch_size, len(dataset))
dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                        pin_memory=True,
                        collate_fn=dataset.collate_fn)

input = dataset[0]
descrip = kornia.feature.siftdesc.sift_describe(input[0])

print('Finished')

