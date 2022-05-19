% -----------------------------------------------------------------------------------------------------------------------------------------------------%
% @编写：wtx
% @更新日期：2022/1/1
% @概述：读取原始ubyte格式MNIST、FASHION_MNIST数据集，进行简单去噪处理后以mat文件格式输出
% @原始数据格式：magic_number, image_number, rows, colums, 每张图像依次左上到右下的像素值
% @原始标签格式：magic_number, image_number, 每张图像的标签
% @输出格式：eg.训练集图像和标签：[60000, 784], [60000, 1]的矩阵，以mat格式存储
% @备注：mat比txt读写速度快得多，且大小小了10倍左右
% -----------------------------------------------------------------------------------------------------------------------------------------------------%
clear;clc;
load('.\Definition_pkg');
%----------------------------------------------------------------------参数配置区-------------------------------------------------------------------%
DATASET = 'MNIST';  %可选：'MNIST'或'FASHION_MNIST'
%----------------------------------------------------------------------参数配置区-------------------------------------------------------------------%

TRAIN_IMG_DIR = fopen(strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_ubyte\train-images.idx3-ubyte'), 'r');
TEST_IMG_DIR = fopen(strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_ubyte\t10k-images.idx3-ubyte'), 'r');
TRAIN_LABEL_DIR = fopen(strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_ubyte\train-labels.idx1-ubyte'), 'r');
TEST_LABEL_DIR = fopen(strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_ubyte\t10k-labels.idx1-ubyte'), 'r');

TRAIN_IMG_SAVE_DIR = strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_Denoising_mat\train_images');
TEST_IMG_SAVE_DIR = strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_Denoising_mat\test_images');
TRAIN_LABEL_SAVE_DIR = strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_Denoising_mat\train_labels');
TEST_LABEL_SAVE_DIR = strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_Denoising_mat\test_labels');

%读取训练集图像，去噪后保存
train_img_magic_number = fread(TRAIN_IMG_DIR, 1, 'uint32', 0, 'ieee-be');
train_img_number = fread(TRAIN_IMG_DIR, 1, 'uint32', 0, 'ieee-be');
train_img_rows = fread(TRAIN_IMG_DIR, 1, 'uint32', 0, 'ieee-be');
train_img_colums = fread(TRAIN_IMG_DIR, 1, 'uint32', 0, 'ieee-be');

train_imgs = zeros(MNIST_TRAIN_IMG, MNIST_SIZE * MNIST_SIZE);
for i = 1 : MNIST_TRAIN_IMG
    temp = fread(TRAIN_IMG_DIR,(MNIST_SIZE * MNIST_SIZE), 'uchar');
    train_imgs(i, :) = temp';
end
train_imgs(train_imgs < DENOISING_THRESHOLD) = 0;
save([TRAIN_IMG_SAVE_DIR, '.mat'], 'train_imgs');

%读取测试集图像，去噪后保存
test_img_magic_number = fread(TEST_IMG_DIR, 1, 'uint32', 0, 'ieee-be');
test_img_number = fread(TEST_IMG_DIR, 1, 'uint32', 0, 'ieee-be');
test_img_rows = fread(TEST_IMG_DIR, 1, 'uint32', 0, 'ieee-be');
test_img_colums = fread(TEST_IMG_DIR, 1, 'uint32', 0, 'ieee-be');

test_imgs = zeros(MNIST_TEST_IMG, MNIST_SIZE * MNIST_SIZE);
for i = 1 : MNIST_TEST_IMG
    temp = fread(TEST_IMG_DIR,(MNIST_SIZE * MNIST_SIZE), 'uchar');
    test_imgs(i, :) = temp';
end
test_imgs(test_imgs < DENOISING_THRESHOLD) = 0;
save([TEST_IMG_SAVE_DIR, '.mat'], 'test_imgs');

%读取训练集标签，并保存
train_label_magic_number = fread(TRAIN_LABEL_DIR, 1, 'uint32', 0, 'ieee-be');
train_label_number = fread(TRAIN_LABEL_DIR, 1, 'uint32', 0, 'ieee-be');

train_labels = zeros(MNIST_TRAIN_IMG, 1);
for i = 1 : MNIST_TRAIN_IMG
    temp = fread(TRAIN_LABEL_DIR, 1, 'uchar');
    train_labels(i, :) = temp';
end
save([TRAIN_LABEL_SAVE_DIR, '.mat'], 'train_labels');

%读取测试集标签，并保存
test_label_magic_number = fread(TEST_LABEL_DIR, 1, 'uint32', 0, 'ieee-be');
test_label_number = fread(TEST_LABEL_DIR, 1, 'uint32', 0, 'ieee-be');

test_labels = zeros(MNIST_TEST_IMG, 1);
for i = 1 : MNIST_TEST_IMG
    temp = fread(TEST_LABEL_DIR, 1, 'uchar');
    test_labels(i, :) = temp';
end
save([TEST_LABEL_SAVE_DIR, '.mat'], 'test_labels');

