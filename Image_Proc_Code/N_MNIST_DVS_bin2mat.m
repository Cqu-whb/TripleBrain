% -----------------------------------------------------------------------------------------------------------------------------------------------------%
% @编写：wtx
% @更新日期：2022/1/1
% @运行时间：300s
% @概述：网上直接下载的N_MNIST数据集每张图像都用了个二进制bin文件保存，不仅文件大且不易读取，本脚本先将训练集和测试集全转换到两个mat文件中，格式和其它DVS数据集一致
% @原始数据格式：每张图像都是一个单独的二进制文件，文件名代表其在原始MNIST数据集中的编号，其由事件列表组成，每个事件占用40位，第39-32位：x地址，第31-24位：y地址
%               第23位：极性，第22-0位：时间戳（以us为单位）     
% @原始标签格式：0-9的目录下分别存放0-9标签的数据
% @输出格式：train mat中包含60000个cell，test mat中包含10000个cell，每个cell的第1、4、5列分别表示时间、像素横坐标、像素纵坐标，其它列都填充0
% @备注1：转换完发现文件居然还大了一点...但起码还是简化了读取方法，统一了格式吧
% @备注2：N_MNIST图像分辨率竟然是34x34的，原文件也不说，fo了
% -----------------------------------------------------------------------------------------------------------------------------------------------------%
clear;clc;
load('.\Definition_pkg');

%原始数据目录
TRAIN_IMG_DIR = '..\..\Common_Datasets\DVS\N_MNIST\Original_files\Train\';
TEST_IMG_DIR = '..\..\Common_Datasets\DVS\N_MNIST\Original_files\Test\';

%转换为mat后保存目录
SAVE_DIR = '..\..\Common_Datasets\DVS\N_MNIST\Original_mat\';

train_cells = cell(1, MNIST_TRAIN_IMG);
test_cells = cell(1, MNIST_TEST_IMG);
train_labels = zeros(MNIST_TRAIN_IMG, 1);
test_labels = zeros(MNIST_TEST_IMG, 1);

%读入每张训练图像，整合进Cell中，并手动生成标签
img_cnt = 0;
for i = 1 : MNIST_NCLASS %10
        img_list = dir(strcat(TRAIN_IMG_DIR, num2str(i-1), '\', '*.bin')); %获取该目录下所有bin格式图像信息
        img_num = length(img_list);  %该目录下图像数量
        for n = 1 : img_num
            img_cnt = img_cnt + 1;
            img_name = img_list(n).name;
            TD = Read_N_MNIST(strcat(TRAIN_IMG_DIR, num2str(i-1), '\', img_name)); %读取每张图像的事件流
            train_cells{img_cnt}(:, 1) = TD.ts(:);
            train_cells{img_cnt}(:, 4) = TD.x(:);
            train_cells{img_cnt}(:, 5) = TD.y(:);
            train_labels(img_cnt, 1) = i - 1; %0-9
        end
end
%保存为mat格式
save([strcat(SAVE_DIR, 'N_MNIST_DVS_34x34_60000'), '.mat'], 'train_cells', '-v7.3');
save([strcat(SAVE_DIR, 'train_labels'), '.mat'], 'train_labels');
clear train_cells;  %防止内存不足

%读入每张测试图像，整合进Cell中，并手动生成标签
img_cnt = 0;
for i = 1 : MNIST_NCLASS %10
        img_list = dir(strcat(TEST_IMG_DIR, num2str(i-1), '\', '*.bin')); %获取该目录下所有bin格式图像信息
        img_num = length(img_list);  %该目录下图像数量
        for n = 1 : img_num
            img_cnt = img_cnt + 1;
            img_name = img_list(n).name;
            TD = Read_N_MNIST(strcat(TEST_IMG_DIR, num2str(i-1), '\', img_name)); %读取每张图像的事件流
            test_cells{img_cnt}(:, 1) = TD.ts(:);
            test_cells{img_cnt}(:, 4) = TD.x(:);
            test_cells{img_cnt}(:, 5) = TD.y(:);
            test_labels(img_cnt, 1) = i - 1; %0-9
        end
end
%保存为mat格式
save([strcat(SAVE_DIR, 'N_MNIST_DVS_34x34_10000'), '.mat'], 'test_cells');
save([strcat(SAVE_DIR, 'test_labels'), '.mat'], 'test_labels'); 
