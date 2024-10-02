import os
import glob
import argparse
import shutil
import random
import ntpath



def create_dataset_dir():
    if not os.path.exists('dataset'):
        os.mkdir('dataset')
    if not os.path.exists(os.path.join('dataset', 'train')):
        os.mkdir(os.path.join('dataset', 'train'))
    if not os.path.exists(os.path.join('dataset', 'val')):
        os.mkdir(os.path.join('dataset', 'val'))
    if not os.path.exists(os.path.join('dataset', 'test')):
        os.mkdir(os.path.join('dataset', 'test'))
    if not os.path.exists(os.path.join('dataset', 'trainannot_origin')):
        os.mkdir(os.path.join('dataset', 'trainannot_origin'))
    if not os.path.exists(os.path.join('dataset', 'valannot_origin')):
        os.mkdir(os.path.join('dataset', 'valannot_origin'))
    if not os.path.exists(os.path.join('dataset', 'testannot_origin')):
        os.mkdir(os.path.join('dataset', 'testannot_origin'))
    if not os.path.exists(os.path.join('dataset', 'trainannot')):
        os.mkdir(os.path.join('dataset', 'trainannot')) 
    if not os.path.exists(os.path.join('dataset', 'valannot')):
        os.mkdir(os.path.join('dataset', 'valannot'))
    if not os.path.exists(os.path.join('dataset', 'testannot')):
        os.mkdir(os.path.join('dataset', 'testannot'))

    
def copy_and_move(dataset):
    for key in dataset.keys():
        for data in dataset[key]:
            file_name = os.path.basename(data)
            target = os.path.join('dataset', key, file_name)
            original = data
            shutil.copyfile(original, target)

def json2jpeg(json_list):
    jpeg_list = []
    for json_file in json_list:
        jpeg_file = json_file.split('.')[0] + '.png'
        jpeg_list.append(jpeg_file)
    return jpeg_list

def png2jpeg(png_list):
    jpeg_list = []
    for png_file in png_list:
        jpeg_file = png_file.split('.')[0] + '.png'
        jpeg_list.append(jpeg_file)
    return jpeg_list

def mask2origin(mask_list):
    origin_list = []
    jpeg_list = png2jpeg(mask_list)
    for jpeg_file in jpeg_list:
        file_name = os.path.basename(jpeg_file)
        origin_file = os.path.join(args.dataset, file_name)
        origin_list.append(origin_file)
    return origin_list

def generate_file_list(dataset, train_size, val_size):
    train_list = random.sample(dataset, train_size)
    val_list = random.sample(set(dataset).difference(train_list), val_size)
    test_list = set(dataset).difference(train_list, val_list) 
    return train_list, val_list, test_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d' ,'--dataset', default='origin_data', help='dataset path')
    parser.add_argument('-mask', default='mask', help='mask dataset path')
    parser.add_argument('--train', type=float, default=0.7, help='percentage of train, default is 0.7')
    parser.add_argument('--val', type=float, default=0.2, help='percentage of val, default is 0.2')
    parser.add_argument('--test', type=float, default=0.1, help='percentage of test, default is 0.1')
    parser.add_argument('--seed', type=int, default=1000, help='random seed')
    args = parser.parse_args()
    random.seed(args.seed)
    print(os.path.join(args.mask, '*.png'))
    dataset = glob.glob(os.path.join(args.mask, '*.png'))
    size = len(dataset)
    val_size = int(size * args.val)
    test_size = int(size * args.test)
    train_size = size - val_size - test_size
    train_list, val_list, test_list = generate_file_list(dataset, train_size, val_size)
    jpeg_dict = {
        'train': mask2origin(train_list),
        'val': mask2origin(val_list),
        'test': mask2origin(test_list),
    }
    png_dict = {
        'trainannot_origin': train_list,
        'valannot_origin': val_list,
        'testannot_origin': test_list,
    }
    create_dataset_dir()     
    copy_and_move(jpeg_dict)
    copy_and_move(png_dict)
    print('complete train/val/test split!')
    print('train size   :', train_size)
    print('val size     :', val_size)
    print('test size    :', test_size)
