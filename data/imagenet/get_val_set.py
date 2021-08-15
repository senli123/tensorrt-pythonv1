import numpy as np
import pickle 
import os
import shutil
import xml.etree.ElementTree as ET
import json

from numpy.lib.function_base import _cov_dispatcher
def get_val_set(img_path, xml_path, des_img_path, des_xml_path):
    img_list = os.listdir(img_path)
    img_num = len(img_list)
    val_img_index = np.random.randint(0, img_num, 5000)
    for i in val_img_index:
        img_name = img_list[i]
        xml_name = img_name.split('.')[0] + '.xml'
        img_file = os.path.join(img_path, img_name)
        xml_file = os.path.join(xml_path, xml_name)
        shutil.copy(img_file, des_img_path)
        shutil.copy(xml_file, des_xml_path)
def parse_synsets():
    # with open("./data/imagenet/imagenet_synsets.txt", "rb") as f:
    #     lines = f.readlines()
    # for line in lines:
    #     line = line.decode().strip().split(' ', 1)
    #     #print(line)
    # with open("./data/imagenet/labels_map.txt", "rb") as f1:
    #     dict = f1.read()
    #     dict = json.loads(dict)
    # q=list(dict.keys())[list(dict.values()).index("part, portion, component part, component, constituent")]
    # print(q)
    with open("./data/imagenet/imagenet_classes.txt", "rb") as f:
        lines = f.readlines()
        lines1 = [line.decode().strip() for line in lines]
        return lines1
def save_name_label():
    label_dict = parse_synsets()
    img_path = "./data/imagenet/val_img"
    xml_path = "./data/imagenet/val_xml"
    save_path = "./data/imagenet"
    img_name_pkl_list = []
    label_pkl_list = []
    img_name_list = os.listdir(img_path)
    for img_name in img_name_list:
        xml_name = img_name.split('.')[0] + '.xml'
        xml_file = os.path.join(xml_path, xml_name)
        classes_names = []
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
        classes_names = list(set(classes_names))
        try:
            index = label_dict.index(classes_names[0])
            print(index)
            img_name_pkl_list.append(os.path.join(img_path, img_name))
            label_pkl_list.append(index)
        except:
            print("there no class id")
    img_save_path = os.path.join(save_path, "img_name.pickle")
    label_save_path = os.path.join(save_path, "label.pickle")

    name_file = open(img_save_path, 'wb')
    pickle.dump(img_name_pkl_list, name_file)
    name_file.close()

    label_file = open(label_save_path, 'wb')
    pickle.dump(label_pkl_list, label_file)
    label_file.close()

if __name__ == '__main__':
   #save_name_label()
   with open('./data/imagenet/img_name.pickle', 'rb') as file:   #用with的优点是可以不用写关闭文件操作
        dict_get = pickle.load(file)
        print(type(dict_get[0]))
    




