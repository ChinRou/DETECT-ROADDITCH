import os
import cv2
import numpy as np
import glob
import pandas
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_dir():
    if not os.path.exists(os.path.join('dataset', 'trainannot')):
        os.mkdir(os.path.join('dataset', 'trainannot'))
    if not os.path.exists(os.path.join('dataset', 'valannot')):
        os.mkdir(os.path.join('dataset', 'valannot'))
    if not os.path.exists(os.path.join('dataset', 'testannot')):
        os.mkdir(os.path.join('dataset', 'testannot'))

def main():
    data_type = ['trainannot_origin', 'testannot_origin', 'valannot_origin']
    # save_path = ['trainannot', 'testannot', 'valannot']
    dir_path = 'dataset'
    for dtype in tqdm(data_type[:]):
        file_list = glob.glob(os.path.join(dir_path, dtype, '*.png'))
        for file_name in tqdm(file_list[:]):
            img = cv2.imread(file_name)
            height, width = img.shape[:2]
            #［B G R］
            #背景
            greenBound = [np.array([0, 255, 0]), np.array([0, 255, 0])]
            #側牆、底版之模板組立
            yellowBound = [np.array([0, 255, 255]), np.array([0, 255, 255])]
            #拆模
            blueBound = [np.array([255, 0, 0]), np.array([255, 0, 0])]
            #溝蓋板鋼筋組立
            orangeBound = [np.array([0, 180, 255]), np.array([0, 180, 255])]
            #完成
            redBound = [np.array([0, 0, 255]), np.array([0, 0, 255])]

            green_mask = (cv2.inRange(img, greenBound[0], greenBound[1]))
            yellow_mask = (cv2.inRange(img, yellowBound[0], yellowBound[1]))
            blue_mask = (cv2.inRange(img, blueBound[0], blueBound[1]))
            orange_mask = (cv2.inRange(img, orangeBound[0], orangeBound[1]))
            red_mask = (cv2.inRange(img, redBound[0], redBound[1]))
            mask = np.zeros((height, width))

            mask[green_mask == 255] = 0
            mask[yellow_mask == 255] = 1
            mask[blue_mask == 255] = 2
            mask[orange_mask == 255] = 3
            mask[red_mask == 255] = 4

            mask_list = [green_mask, yellow_mask, blue_mask, orange_mask, red_mask]

            #greenBound = [np.array([0, 220, 0]), np.array([0, 255, 0])]
            #blueBound = [np.array([220, 0, 0]), np.array([255, 0, 0])]
            #green_mask = (cv2.inRange(img, greenBound[0], greenBound[1]))
            #blue_mask = (cv2.inRange(img, blueBound[0], blueBound[1]))
            #mask = np.zeros((height, width))
            #mask[green_mask == 255] = 0
            #mask[blue_mask == 255] = 1
            #mask_list = [green_mask, blue_mask]
            '''
            for i in range(len(mask_list[:-1])):
               for x in range(height):
                    for y in range(width):
                        if mask_list[i][x, y] == 255:
                            mask[x, y] = i
            '''
            # print(os.path.join(dir_path, dtype.split('_')[0], os.path.basename(file_name)))
            cv2.imwrite(os.path.join(dir_path, dtype.split('_')[0], os.path.basename(file_name)), mask) 
            """
            mask_list.append(mask)
            mask_title = ['green_mask', 'blue_mask', 'mask']
            for i in range(len(mask_list)):
                plt.subplot(1, 3, i+1)
                plt.imshow(mask_list[i])
                plt.axis('off')
                plt.title(mask_title[i])
            plt.tight_layout()
            plt.show()
            print(np.unique(green_mask))
            print(np.unique(blue_mask))
            print(np.unique(mask))
            """
if __name__ == '__main__':
    create_dir()
    main()
