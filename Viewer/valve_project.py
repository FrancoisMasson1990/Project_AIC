import os
from viewer_3D import Viewer3D
from viewer_2D import Viewer2D
import argparse

def main_3D(data):
    viewer = Viewer3D(data,mode=2)
    viewer.show()

def main_2D(data,folder_mask):
    Viewer2D(data,folder_mask)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Project AIC')
    parser.add_argument('--folder_dcm', help='Dicom dataset with the .dcm extension files')
    parser.add_argument('--labels_2D',action='store_true',help='Activate label 2D tools')
    parser.add_argument('--folder_mask',help='folder with the label_mask')

    arg = parser.parse_args()
    data_path = arg.folder_dcm
    #data_path = '/home/francoismasson/Projet_AIC/label_mask/'
    #data_path = '/home/francoismasson/Projet_AIC/data_preprocess/'

    sub_folders = os.listdir(data_path)
    data = []
    for sub_folder in sub_folders:
        root = os.path.join(data_path,sub_folder)
        sub_ = os.listdir(root)
        for sub in sub_ :
            data.append(os.path.join(root,sub))

    if arg.labels_2D :
        main_2D(data,arg.folder_mask)
    else :
        main_3D(data)