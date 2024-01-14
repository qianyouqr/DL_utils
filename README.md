# DL_utils
深度学习的工具箱，包括以下内容：
## 1 在图像上绘制voc标注格式的 标注框  
* showLabel.py
https://github.com/qianyouqr/DL_utils/blob/main/showLabel.py
显示voc格式的标注框

## 2 yolov7-训练自己的数据集
数据集来自于标注的voc格式的数据集，数据集的目录形式：  
![image](https://github.com/qianyouqr/DL_utils/assets/68421771/a5829b5d-b65d-4813-9d8a-6c1d81c68e02)  
使用`voc2yolo.py`文件将voc格式的数据集转成yolo格式的数据集。使用方式：  
**第一步**
```
    # step1:
    # # 先将voc数据集 转成 yolo格式的数据集['seed','bivalve','broken','diseased','membrance','peel', 'spongy']
    # VOC2Yolo(
    #     class_num={'seed':0,'bivalve':1,'broken':2,'diseased':3,'membrance':4,'peel':5, 'spongy':6},  # 标签种类
    #     voc_img_path='C:/datasets/orange_v1/VOC2007_all/VOC2007/JPEGImages',  # 数据集图片文件夹存储路径
    #     voc_xml_path='C:/datasets/orange_v1/VOC2007_all/VOC2007/Annotations',  # 标签xml文件夹存储路径
    #     yolo_txt_save_path='C:/datasets/orange_v1/VOC2007_all/VOC2007/YoloLabels'  # 将要生成的标注文件  txt文件夹存储路径  
    # )
```
把voc_img_path 改成自己voc数据集中保存所有图片的目录  
把voc_xml_path 改成自己voc数据集中保存所有标注结果的目录  
把yolo_txt_save_path 改成自己定义的 保存转换后的标注文件 的目录  
这样执行完第一步后得到了 yolo 格式标注的所有标注文件  
**第二步**，将转换后的yolo格式数据集进行划分。    
*使用第二步就把第一步的代码注释掉*  
```
    # step2:
    # 所分割的数据集所有的 图片文件目录
    img_path = 'C:/datasets/orange_v1/VOC2007/JPEGImages'  # 你的图片存放的路径（路径一定是相对于你当前的这个脚本文件而言的）
    # 所分割的数据集所有的 标签文件目录
    label_path = 'C:/datasets/orange_v1/VOC2007/YoloLabels'  # 你的txt文件存放的路径 这个目录是step1中设置的yolo_txt_save_path（路径一定是相对于你当前的这个脚本文件而言的）
    split_list = [0.8, 0.15, 0.15]  # 数据集划分比例[train:val:test]
    # 在方法里面设置 要保存到的目录
    split_img(img_path, label_path, split_list)
```
把img_path 改成自己voc数据集中保存所有图片的目录  
把label_path 改成自己定义的 保存转换后的标注文件 的目录，就是yolo标注文件目录  
在**split_img**函数内Data变量设置为*yolo数据集的目录*，这一步你可以将Data变成传入的参数。  
![image](https://github.com/qianyouqr/DL_utils/assets/68421771/2abd99ea-6bf4-4f14-bb6c-3b19f7c3592b)

这样执行完第一步后得到了 yolo格式标注的所有标注文件  
![image](https://github.com/qianyouqr/DL_utils/assets/68421771/23298a4f-c7a4-418d-8d0a-29b650f3a98e)  
**第三步**，修改yolov7代码中的yaml文件  
![image](https://github.com/qianyouqr/DL_utils/assets/68421771/7bb42c96-9d7f-4d24-ad0e-3ed5dcc72ce3)  
之后调整参数进行训练即可。  
## 3 计算模型的mAP  
### 3.1计算效果  
![image](https://github.com/qianyouqr/DL_utils/assets/68421771/6cd48042-1f96-4d18-8c04-3f780452f2c1)  
除此之外，还有结果图像的绘制  
### 3.2数据格式要求  
真实值标签数据格式:  
【类名 left, top, right, bottom】  
![image](https://github.com/qianyouqr/DL_utils/assets/68421771/0191d4fc-ca46-4d44-af0d-5bf9fb270209)  
推理后形成的txt文件格式：  
【类名编号 置信度 left, top, right, bottom】  
![image](https://github.com/qianyouqr/DL_utils/assets/68421771/529e026e-e6cb-45d4-a324-7f965dbffe10)  
### 3.3代码路径设置  
1. `base_path`
保存mAP结果的目录：  
```
base_path = 'D:/code/orin_nano/map_result/yolov7'
```
![image](https://github.com/qianyouqr/DL_utils/assets/68421771/0077e929-8cd7-4be2-82f4-806523a489af)  
2. `test_gt_impoved`  
真实值标签的路径  
```
test_gt_impoved = 'C:/datasets/orange_v3/yolo/labels/test_impoved'
```
![image](https://github.com/qianyouqr/DL_utils/assets/68421771/e59acabf-1051-4e73-837b-6be66089e3cf)  
3. `label_path`  
需要计算mAP的检测结果的路径  
```
label_path = 'D:/code/orin_nano/best_YoloV7_FP32_result_txt_official'
```
![image](https://github.com/qianyouqr/DL_utils/assets/68421771/66e9025e-9fd6-4f2d-8355-1e1c888e59b2)  
4. `image_path`  
所推理的图片的目录  
```
image_path = 'C:/datasets/orange_v3/yolo/images/test'
```
用来获取图像的尺度信息。  
5. 修改检测类型信息  
在`cls_name`函数中修改类名字典，类名编号和类名字符串对应关系要和训练时设置的一样。
![image](https://github.com/qianyouqr/DL_utils/assets/68421771/8739ebee-66ef-4d0a-b765-2096dbed12da)  

修改完这些路径信息，执行`compute_mAP.py`即可。











