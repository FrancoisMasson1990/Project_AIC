import os
from viewer_3D import Viewer3D
from viewer_2D import Viewer2D
import argparse
import yaml 

def main_3D(data,folder_mask,folder_npy,multi_label):
    viewer = Viewer3D(data,mode=4,label=folder_mask,npy=folder_npy,multi_label=multi_label)
    viewer.show()

def main_2D(data,folder_mask):
    Viewer2D(data,folder_mask)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Project AIC')
    arg = parser.parse_args()

    with open('./data_info.yml') as f:
       # The FullLoader parameter handles the conversion from YAML
       # scalar values to Python the dictionary format
       config = yaml.load(f, Loader=yaml.FullLoader)

    arg.data_path = config.get("data_path",None)
    arg.labels_2D = config.get("labels_2D",None)
    arg.folder_mask = config.get("folder_mask",None)
    arg.npy_folder = config.get("npy_folder",None)
    arg.multi_label = config.get("multi_label",None)

    sub_folders = os.listdir(arg.data_path)
    data = []
    for sub_folder in sub_folders:
        root = os.path.join(arg.data_path,sub_folder)
        sub_ = os.listdir(root)
        for sub in sub_ :
            data.append(os.path.join(root,sub))
    
    if arg.labels_2D :
        main_2D(data,arg.folder_mask)
    else :
        main_3D(data,arg.folder_mask,arg.npy_folder,arg.multi_label)
