import shutil
import os
from tqdm import tqdm
 

# file_List = ["train2007", "val2007", "test2007"]
file_List = ["train2007", "val2007", "test2007"]
for file in file_List:
    if not os.path.exists('./datasets/VOC/images/%s' % file):
        os.makedirs('./datasets/VOC/images/%s' % file)
    if not os.path.exists('../datasets/VOC/labels/%s' % file):
        os.makedirs('./datasets/VOC/labels/%s' % file)

    f = open('./datasets/%s.txt' % file, 'r')
    lines = f.readlines()
    for line in tqdm(lines):
        line = line.strip()

        if os.path.exists(line):
            shutil.copy(line, "./datasets/VOC/images/%s" % file)
            line = line.replace('JPEGImages', 'labels')
            line = line.replace('jpg', 'txt')
            shutil.copy(line, "./datasets/VOC/labels/%s/" % file)
        else:
            print(line)