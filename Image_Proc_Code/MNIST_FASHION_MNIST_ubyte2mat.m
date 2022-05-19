% -----------------------------------------------------------------------------------------------------------------------------------------------------%
% @��д��wtx
% @�������ڣ�2022/1/1
% @��������ȡԭʼubyte��ʽMNIST��FASHION_MNIST���ݼ������м�ȥ�봦�����mat�ļ���ʽ���
% @ԭʼ���ݸ�ʽ��magic_number, image_number, rows, colums, ÿ��ͼ���������ϵ����µ�����ֵ
% @ԭʼ��ǩ��ʽ��magic_number, image_number, ÿ��ͼ��ı�ǩ
% @�����ʽ��eg.ѵ����ͼ��ͱ�ǩ��[60000, 784], [60000, 1]�ľ�����mat��ʽ�洢
% @��ע��mat��txt��д�ٶȿ�ö࣬�Ҵ�СС��10������
% -----------------------------------------------------------------------------------------------------------------------------------------------------%
clear;clc;
load('.\Definition_pkg');
%----------------------------------------------------------------------����������-------------------------------------------------------------------%
DATASET = 'MNIST';  %��ѡ��'MNIST'��'FASHION_MNIST'
%----------------------------------------------------------------------����������-------------------------------------------------------------------%

TRAIN_IMG_DIR = fopen(strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_ubyte\train-images.idx3-ubyte'), 'r');
TEST_IMG_DIR = fopen(strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_ubyte\t10k-images.idx3-ubyte'), 'r');
TRAIN_LABEL_DIR = fopen(strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_ubyte\train-labels.idx1-ubyte'), 'r');
TEST_LABEL_DIR = fopen(strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_ubyte\t10k-labels.idx1-ubyte'), 'r');

TRAIN_IMG_SAVE_DIR = strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_Denoising_mat\train_images');
TEST_IMG_SAVE_DIR = strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_Denoising_mat\test_images');
TRAIN_LABEL_SAVE_DIR = strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_Denoising_mat\train_labels');
TEST_LABEL_SAVE_DIR = strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_Denoising_mat\test_labels');

%��ȡѵ����ͼ��ȥ��󱣴�
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

%��ȡ���Լ�ͼ��ȥ��󱣴�
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

%��ȡѵ������ǩ��������
train_label_magic_number = fread(TRAIN_LABEL_DIR, 1, 'uint32', 0, 'ieee-be');
train_label_number = fread(TRAIN_LABEL_DIR, 1, 'uint32', 0, 'ieee-be');

train_labels = zeros(MNIST_TRAIN_IMG, 1);
for i = 1 : MNIST_TRAIN_IMG
    temp = fread(TRAIN_LABEL_DIR, 1, 'uchar');
    train_labels(i, :) = temp';
end
save([TRAIN_LABEL_SAVE_DIR, '.mat'], 'train_labels');

%��ȡ���Լ���ǩ��������
test_label_magic_number = fread(TEST_LABEL_DIR, 1, 'uint32', 0, 'ieee-be');
test_label_number = fread(TEST_LABEL_DIR, 1, 'uint32', 0, 'ieee-be');

test_labels = zeros(MNIST_TEST_IMG, 1);
for i = 1 : MNIST_TEST_IMG
    temp = fread(TEST_LABEL_DIR, 1, 'uchar');
    test_labels(i, :) = temp';
end
save([TEST_LABEL_SAVE_DIR, '.mat'], 'test_labels');

