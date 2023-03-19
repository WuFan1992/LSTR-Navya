#!/usr/bin/env python

import json

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm

from scipy.interpolate import InterpolatedUnivariateSpline


class TransformCAMMA2TuSimpleFormat():
    def __init__(self, num_xml_eachsdb = 20, 
                 num_sdb = 100,
                 dataset_path = f'/home/fan.wu/Downloads/data_set/comma2k1/' ):
       self.num_xml_eachsdb = num_xml_eachsdb
       self.map_anno = {}
       self.num_sdb = num_sdb
       self.dataset_path = dataset_path
       self.global_index = 0

    def display_result(self, index_sdb):
        if(index_sdb < self.num_sdb):
            sdb_path = self.dataset_path + "scb{index_sdb}/ "
            imgs = []
            plt.figure(figsize=(40, 4))
            for frame in range(21):
                plt.subplot(1, 21, frame + 1)
                img_path = sdb_path + f"imgs/{frame}.png"
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                img = self._camera2model(img)
                plt.imshow(img)
                anno = self._read_content(path + f'{frame}.xml')
                left = anno['left']
                right = anno['right']
                plt.plot(left[:, 0], left[:, 1])
                plt.plot(right[:, 0], right[:, 1])
                plt.ylim(720, 0)
                plt.axis('off')
        plt.tight_layout()
        plt.show()


    def write_tusimple_json(self):
        with open('comma2k19ld_tusimple_annotation_raw.json', 'w') as f:
            f.write(json.dumps(self.map_anno))
    
    def create_map_annos(self):
        for sc in tqdm(range(1,self.num_sdb+1)):
            scb_name = "scb" + str(sc) + "/"
            sdb_path =  self.dataset_path + scb_name
            self._create_anno(sdb_path)


    def _create_anno(self, sdb_path):
        for frame in range(self.num_xml_eachsdb+1):   #21
            img_path = sdb_path + f"imgs/{frame}.png"
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = self._camera2model(img)
        

            anno = self._read_content(sdb_path + f'{frame}.xml')
            left = anno['left']
            right = anno['right']

            lines = []
            y_all = np.arange(160, 711, 10)
        
            vaid_y_idx = y_all >= max(left[:, 1].min(), right[:, 1].min())

            idx = np.argsort(left[:, 1])
            cs = InterpolatedUnivariateSpline(
                                                left[idx, 1], left[idx, 0],
                                                k=min(1, left.shape[0] - 1))
            lx = cs(y_all)
            lx = np.where(vaid_y_idx, lx, -2)
            lx = np.where((lx < 0) | (lx > 1279), -2, lx)
        
            lines.append(lx.astype(int).tolist())

            idx = np.argsort(right[:, 1])
            cs = InterpolatedUnivariateSpline(
                                                right[idx, 1], right[idx, 0],
                                                k=min(1, right.shape[0] - 1))
            rx = cs(y_all)
            rx = np.where(vaid_y_idx, rx, -2)
            rx = np.where((rx < 0) | (rx > 1280), -2, rx)
            lines.append(rx.astype(int).tolist())
        
            self.global_index +=1
            self.map_anno[self.global_index] = {
               'lanes': lines,
               'h_samples': y_all.astype(int).tolist(),
               'raw_file': img_path
        }
            
    def _camera2model(self, img):
        assert img.shape == (874, 1164, 3)
        img = img[200:-220, 106:-106]
        img = cv2.resize(img, (1280, 720))#.astype(np.float64)
        return img
    

    def _read_content(self, xml_file: str):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        list_left = []
        list_right = []
    
        filename = root.find('path').text

        for boxes in root.iter('object'):
            ymin, xmin, ymax, xmax = None, None, None, None

            ymin = int(boxes.find("bndbox/ymin").text)
            xmin = int(boxes.find("bndbox/xmin").text)
            ymax = int(boxes.find("bndbox/ymax").text)
            xmax = int(boxes.find("bndbox/xmax").text)

            list_with_single_boxes = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
            list_with_single_boxes -= [106, 200]
            list_with_single_boxes /= [952, 454]
            list_with_single_boxes *= [1280, 720]
        
            list_with_single_boxes = list_with_single_boxes.astype(int)
        
            name_ = boxes.find("name").text
            if name_ == 'l':
                list_left.append(list_with_single_boxes)
            elif name_ == 'r':
                list_right.append(list_with_single_boxes)
            else:
                raise Exception(f'no {name_}')
            
        return {'path': filename, 'left': np.stack(list_left), 'right': np.stack(list_right)}
         

if __name__ == "__main__":
    trans = TransformCAMMA2TuSimpleFormat()
    trans.create_map_annos()
    trans.write_tusimple_json()
