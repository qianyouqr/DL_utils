
'''
在图像中显示xml标签文件中保存的检测框
'''
import os
import random
from tqdm import tqdm
import cv2
import xml.etree.ElementTree as ET

def draw_bounding_boxes(image, xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        xmin,xmax = min(xmin,xmax), max(xmin,xmax)
        ymin,ymax = min(ymin,ymax), max(ymin,ymax)
        print(f"(xmin, ymin)=({xmin},{ymin})--- (xmax, ymax)=({xmax},{ymax})")
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_bounding_boxes_dir(img_dir, xml_dir, saved_dir, o_p):
    '''
    将img_dir目录下的o_p%的图片，根据xml_dir目录下的标注文件进行真实框的绘制，将结果图片保存到 saved_dir 目录下
    '''
    # 获取目录下的所有文件名
    all_files = os.listdir(img_dir)
    # 筛选出所有的图片文件名
    image_files = [file for file in all_files if file.endswith('.jpg') or file.endswith('.png')]

    # 计算需要随机选择多少个
    n = int(len(image_files)*o_p/100)
    selected_images = random.sample(image_files, n)
    print("选择的图片数：", len(selected_images))
    # print(selected_images)
    # 从xml_dir 中获取所有的标记文件
    all_xml_files = os.listdir(xml_dir)
    if "IMG_20230518_214249_vertical_flip.xml" in all_xml_files:
        print("ZAI")
    else:
        print("no zai")
    # 遍历每个 图片
    for img_name in tqdm(selected_images):
        xml_name = img_name.split('.')[0]+".xml"
        if xml_name not in all_xml_files:
            print(f"{img_name}对应的标签文件{xml_name}不存在")
            continue
        # 标记文件的绝对路径
        xml_file = xml_dir+"/"+xml_name
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # 图片的绝对路径
        image_path = img_dir+"/"+img_name
        image = cv2.imread(image_path)
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        saved_path = saved_dir+'/'+img_name
        cv2.imwrite(saved_path, image)



if __name__ == '__main__':
    # 单个xml文件和单个图像的操作
    image_path = 'G:\orange_voc2/VOC2007/JPEGImages/IMG_20230510_191321_vertical_flip.jpg'
    xml_path = 'G:\orange_voc2/VOC2007/Annotations/IMG_20230510_191321_vertical_flip.xml'
    image = cv2.imread(image_path)
    draw_bounding_boxes(image, xml_path)

    # 将某个目录下的所有真实框gt标注出来
    # img_dir = "G:/orange_voc/test_cat_dog/img"
    # xml_dir = "G:/DogandCatDetection/VOC2007/Annotations"  # xml的目录包含img_dir中图片真实框的标准信息
    # saved_dir = "G:/orange_voc/test_cat_dog/showed2"
    # draw_bounding_boxes_dir(img_dir,xml_dir,saved_dir, 100)
