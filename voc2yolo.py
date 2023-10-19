
'''
根据voc数据集，划分出 训练集、验证集、和测试集

'''

import os, shutil, random
from tqdm import tqdm
import os
import shutil
import cv2
from lxml import etree

def VOC2Yolo(class_num, voc_img_path, voc_xml_path, yolo_txt_save_path, yolo_img_save_path=None):
    '''
    将VOC 格式转换为 YOLO 格式
    '''
    xmls = os.listdir(voc_xml_path)
    xmls = [x for x in xmls if x.endswith('.xml')]
    if yolo_img_save_path is not None:
        if not os.path.exists(yolo_img_save_path):
            os.mkdir(yolo_img_save_path)
    if not os.path.exists(yolo_txt_save_path):
        os.mkdir(yolo_txt_save_path)
    all_xmls = len(xmls)
    for idx, one_xml in tqdm(enumerate(xmls),desc='VOC2Yolo', total=len(xmls)):
        xl = etree.parse(os.path.join(voc_xml_path, one_xml)) # os.path.join(a,b)会自动在ab之间加/  
        root = xl.getroot()
        objects = root.findall('object')
        img_size = root.find('size')
        img_w = 0
        img_h = 0
        if img_size:
            img_width = img_size.find('width')
            if img_width is not None:
                img_w = int(img_width.text)
            img_height = img_size.find('height')
            if img_height is not None:
                img_h = int(img_height.text)
        label_lines = []
        for ob in objects:
            one_annotation = {}
            label = ob.find('name').text
            one_annotation['tag'] = label
            one_annotation['flag'] = False
            bbox = ob.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            xmin,xmax = min(xmin,xmax), max(xmin,xmax)
            ymin,ymax = min(ymin,ymax), max(ymin,ymax)
            if img_w == 0 or img_h == 0:
                img = cv2.imread(os.path.join(voc_img_path, one_xml.replace('.xml', '.jpg')))
                img_h, img_w = img.shape[:2]
            bbox_w = (xmax - xmin) / img_w
            bbox_h = (ymax - ymin) / img_h
            bbox_cx = (xmin + xmax) / 2 / img_w
            bbox_cy = (ymin + ymax) / 2 / img_h
            try:
                bbox_label = class_num[label]
                label_lines.append(f'{bbox_label} {bbox_cx} {bbox_cy} {bbox_w} {bbox_h}' + '\n')
            except Exception as e:
                print("not find number label in class_num ", e, one_xml)
                label_lines = []
                break
        if len(label_lines):
            with open(os.path.join(yolo_txt_save_path, one_xml.replace('.xml', '.txt')), 'w') as fp:
                fp.writelines(label_lines)
            if yolo_img_save_path is not None:
                shutil.copy(os.path.join(voc_img_path, one_xml.replace('.xml', '.jpg')),
                            os.path.join(yolo_img_save_path))
        #print(f"processing: {idx+1}/{all_xmls}")

"""
标注文件是yolo格式（txt文件）
训练集：验证集：测试集 （7：2：1） 
"""
def split_img(img_path, label_path, split_list):
    try:
        Data = 'G:/orange_yolo_dataset'
        # Data是你要将要创建的文件夹路径（路径一定是相对于你当前的这个脚本而言的）
        # os.mkdir(Data)

        train_img_dir = Data + '/images/train'
        val_img_dir = Data + '/images/val'
        test_img_dir = Data + '/images/test'

        train_label_dir = Data + '/labels/train'
        val_label_dir = Data + '/labels/val'
        test_label_dir = Data + '/labels/test'

        # 创建文件夹
        os.makedirs(train_img_dir)
        os.makedirs(train_label_dir)
        os.makedirs(val_img_dir)
        os.makedirs(val_label_dir)
        os.makedirs(test_img_dir)
        os.makedirs(test_label_dir)

    except:
        print('文件目录已存在')

    train, val, test = split_list
    all_img = os.listdir(img_path)
    all_img_path = [os.path.join(img_path, img) for img in all_img]
    # all_label = os.listdir(label_path)
    # all_label_path = [os.path.join(label_path, label) for label in all_label]
    train_img = random.sample(all_img_path, int(train * len(all_img_path)))
    train_img_copy = [os.path.join(train_img_dir, img.split('\\')[-1]) for img in train_img]
    train_label = [toLabelPath(img, label_path) for img in train_img]
    train_label_copy = [os.path.join(train_label_dir, label.split('\\')[-1]) for label in train_label]
    for i in tqdm(range(len(train_img)), desc='train ', ncols=80, unit='img'):
        _copy(train_img[i], train_img_dir)
        _copy(train_label[i], train_label_dir)
        all_img_path.remove(train_img[i])
    val_img = random.sample(all_img_path, int(val / (val + test) * len(all_img_path)))
    val_label = [toLabelPath(img, label_path) for img in val_img]
    for i in tqdm(range(len(val_img)), desc='val ', ncols=80, unit='img'):
        _copy(val_img[i], val_img_dir)
        _copy(val_label[i], val_label_dir)
        all_img_path.remove(val_img[i])
    test_img = all_img_path
    test_label = [toLabelPath(img, label_path) for img in test_img]
    for i in tqdm(range(len(test_img)), desc='test ', ncols=80, unit='img'):
        _copy(test_img[i], test_img_dir)
        _copy(test_label[i], test_label_dir)


def _copy(from_path, to_path):
    shutil.copy(from_path, to_path)


def toLabelPath(img_path, label_path):
    img = img_path.split('\\')[-1]
    label = img.split('.jpg')[0] + '.txt'
    return os.path.join(label_path, label)


if __name__ == '__main__':

    # 先将voc数据集 转成 yolo格式的数据集['seed','bivalve','broken','diseased','membrance','peel', 'spongy']
    # VOC2Yolo(
    #     class_num={'seed':0,'bivalve':1,'broken':2,'diseased':3,'membrance':4,'peel':5, 'spongy':6},  # 标签种类
    #     voc_img_path='G:/orange_voc2/VOC2007/JPEGImages',  # 数据集图片文件夹存储路径
    #     voc_xml_path='G:/orange_voc2/VOC2007/Annotations',  # 标签xml文件夹存储路径
    #     yolo_txt_save_path='D:/QR/vscodeSpace/yolov7-main/dataset/YoloLabels'  # 将要生成的txt文件夹存储路径
    # )



    img_path = 'G:/orange_voc2/VOC2007/JPEGImages'  # 你的图片存放的路径（路径一定是相对于你当前的这个脚本文件而言的）
    label_path = 'D:/QR/vscodeSpace/yolov7-main/dataset/YoloLabels'  # 你的txt文件存放的路径（路径一定是相对于你当前的这个脚本文件而言的）
    split_list = [0.8, 0.1, 0.1]  # 数据集划分比例[train:val:test]
    split_img(img_path, label_path, split_list)

