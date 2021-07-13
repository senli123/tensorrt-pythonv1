import numpy as np
import pickle 
import os
import shutil
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



