import xml.etree.ElementTree as ET
import os
from tqdm import tqdm

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(root,year, image_id):
    in_file = open(os.path.join(root,'VOC%s/Annotations/%s.xml' % (year, image_id)))
    out_file = open(os.path.join(root,'VOC%s/labels/%s.txt' % (year, image_id)), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

if __name__ == '__main__':

    sets_2007 = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
    #sets_2012 = [('2012', 'train'), ('2012', 'val')]

    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
            "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
            
    root = r'./datasets_old/VOC/VOCdevkit'
    output_dir = r'./datasets'
    #sets_07_12 = [sets_2007,sets_2012]
    sets_07_12 = [sets_2007]
    for sets in sets_07_12:
        for year, image_set in sets:
            if not os.path.exists(os.path.join(root,'VOC%s/labels/' % year)):
                os.makedirs(os.path.join(root,'VOC%s/labels/' % year))
            image_ids = open(os.path.join(root,'VOC%s/ImageSets/Main/%s.txt' % (year, image_set))).read().strip().split()
            list_file = open('%s/%s%s.txt' % (output_dir,image_set,year), 'a')
            for image_id in tqdm(image_ids):
                list_file.write('%s/VOC%s/JPEGImages/%s.jpg\n' % (root, year, image_id))
                convert_annotation(root,year, image_id)
            list_file.close()