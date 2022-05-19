% -----------------------------------------------------------------------------------------------------------------------------------------------------%
% @编写：wtx
% @更新日期：2022/1/1
% @运行时间：resize为16x16后15s左右
% @概述：读取ETH80数据集（png格式），并将其编码，支持Temporal_Coding、Rate_Coding、任意尺寸Resize、以及高斯差分滤波处理（Dog）
% @输入数据格式：png图像
% @输出格式：1行n列数组[addr1 t1 addr2 t2 ... -1 addr1 t1 addr2 t2 ... -1 ...], 地址addr范围：[1, img_size^2], 时间戳t范围：[1, TIME_WINDOW - 1],
%           时间t以升序排序，每张图像之间以-1分隔
% @备注1：注意matlab输出的地址是从1开始的，C#等语言处理时需要将地址减1
% @备注2：每次运行脚本都将打乱数据一次，得到不同的训练集和测试集
% @备注3：Dog处理时取消相应位置代码注释可对比原始图像和Dog后的图像
% @备注4：编码输出目录需根据需求手动在电脑中创建，但无需在代码中修改目录，配置好参数自动修改 eg.在Temporal_Coding目录下创建名为16x16_Dog的文件夹
% -----------------------------------------------------------------------------------------------------------------------------------------------------%
clear;clc;
load('.\Definition_pkg');
%----------------------------------------------------------------------参数配置区-------------------------------------------------------------------%
%数据集、预处理方法、编码方法选择
PRE_PROC = 'Gray';                   % 预处理方法，可选：'Gray'或'Dog'
CODING_METHOD = 'Temporal_Coding';  % 编码方法，可选：Temporal_Coding'或'Rate_Coding'
TRAIN_RATIO = 0.8;                  % 训练集所占比例

%Resize参数
IF_RESIZE = 1;                      % 是否做resize处理，1--True, 0--False
RESIZE_SIZE = 16;                   % resize后的图像大小

%高斯差分滤波（Dog）参数
SIGMA1 = 1; SIGMA2 = 3; WINDOW = 15;

%Temporal_Coding参数, t = TIME_WINDOW1 - floor(K1 * pixel)
TIME_WINDOW1 = 256;                   %时间窗口
K1 = TIME_WINDOW1 / MAX_PIXEL_VALUE;  %编码系数

%Rate_Coding参数, 以K2*像素值为lmd参数生成泊松序列
TIME_WINDOW2 = 256;                   %时间窗口
SPIKE_PER_PIXEL = 5;                  %每个像素值生成的泊松脉冲个数
K2 = TIME_WINDOW2 / MAX_PIXEL_VALUE;  %编码系数
%----------------------------------------------------------------------参数配置区-------------------------------------------------------------------%
IMG_DIR = '..\..\Common_Datasets\Frame_based\ETH80\Original_png\';

%编码后保存目录（第一次需根据自己需求创建，eg.在Temporal_Coding目录下创建名为16x16_Dog的文件夹）
folder_name = strcat(num2str(TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)), 'x', num2str(TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)), '_', PRE_PROC);
TRAIN_ENCODING_SAVE_DIR = strcat('..\..\Common_Datasets\Frame_based\ETH80\Encoding\', CODING_METHOD, '\', folder_name, '\');
TEST_ENCODING_SAVE_DIR = strcat('..\..\Common_Datasets\Frame_based\ETH80\Encoding\', CODING_METHOD, '\', folder_name, '\');

all_imgs_2d = zeros(ETH80_IMG, TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE), TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE));
all_labels = zeros(ETH80_IMG, 1);
all_imgs_2d_rand = zeros(ETH80_IMG, TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE), TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE));
all_labels_rand = zeros(ETH80_IMG, 1);
train_imgs_2d = zeros(round(ETH80_IMG * TRAIN_RATIO), TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE), TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE));
test_imgs_2d = zeros(round(ETH80_IMG * (1 - TRAIN_RATIO)), TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE), TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE));
train_labels = zeros(round(ETH80_IMG * TRAIN_RATIO), 1);
test_labels = zeros(round(ETH80_IMG * (1 - TRAIN_RATIO)), 1);

%读入每张图像，并手动生成标签
img_cnt = 0;
for i = 1 : ETH80_NCLASS %8
    for j = 1 : ETH80_NFOLDER_PER_CLASS %10
        img_list = dir(strcat(IMG_DIR, num2str(i), '\', num2str(j), '\', '*.png')); %获取该目录下所有png格式图像信息
        for n = 1 : ETH80_NIMG_PER_FOLDER %41
            img_name = img_list(n).name;
            img = imread(strcat(IMG_DIR, num2str(i), '\', num2str(j), '\', img_name));  %读取每张图像
            img_gray = rgb2gray(img);
            if (IF_RESIZE)  %resize
                img_gray = floor(imresize(img_gray, [RESIZE_SIZE, RESIZE_SIZE], 'bilinear'));  %双线性插值（比最邻近插值效果好）              
            end
            img_cnt = img_cnt + 1;
            all_imgs_2d(img_cnt, :, :) = img_gray;
            all_labels(img_cnt, 1) = i - 1; %0-7
        end
    end
end

%打乱图像，并分配训练集和测试集
rand_num = randperm(ETH80_IMG, ETH80_IMG);  %返回一行从1到x的整数中的x个，且这x个数各不相同
for i = 1 : ETH80_IMG
    all_imgs_2d_rand(i, :, :) = all_imgs_2d(rand_num(i), :, :);
    all_labels_rand(i, 1) = all_labels(rand_num(i), 1);
end
for i = 1 : round(ETH80_IMG * TRAIN_RATIO)
    train_imgs_2d(i, :, :) = all_imgs_2d_rand(i, :, :);
    train_labels(i, 1) = all_labels_rand(i, 1);
end
for i = 1 : round(ETH80_IMG * (1 - TRAIN_RATIO))
    test_imgs_2d(i, :, :) = all_imgs_2d_rand(i + round(ETH80_IMG * TRAIN_RATIO), :, :);
    test_labels(i, 1) = all_labels_rand(i + round(ETH80_IMG * TRAIN_RATIO), 1);
end

%判断是否高斯滤波
%type= 'gaussian'，为高斯低通滤波器，模板有两个，sigma表示滤波器的标准差（单位为像素，默认值为 0.5），window表示模版尺寸，默认值为[3,3]
if (strcmp(PRE_PROC, 'Dog'))
    H1 = fspecial('gaussian', WINDOW, SIGMA1);
    H2 = fspecial('gaussian', WINDOW, SIGMA2);
    DiffGauss = H1 - H2;
    %滤波每张训练图像
    for n = 1 : round(ETH80_IMG * TRAIN_RATIO)
         dog_img = abs(imfilter(squeeze(train_imgs_2d(n, :, :)), DiffGauss, 'replicate'));   %对任意类型数组或多维图像进行滤波
         dog_img = mat2gray(dog_img);      % 将图像矩阵归一化0到1范围内(包括0和1)
         dog_img = floor(dog_img * 255);   % 再乘255变成正常灰度值    
         
         %检查Dog后的图像是否合理
%          subplot(1, 2, 1); imshow(squeeze(train_imgs_2d(n, :, :))/255); title('Ori');
%          subplot(1, 2, 2); imshow(dog_img/255); title('Dog');
%          debug_temp = 0;   %此处设置断点对比Dog前后图像     
         
         train_imgs_2d(n, :, :) = dog_img;     
    end
    %滤波每张测试图像
    for n = 1 : round(ETH80_IMG * (1 - TRAIN_RATIO))
         dog_img = abs(imfilter(squeeze(test_imgs_2d(n, :, :)), DiffGauss, 'replicate'));   %对任意类型数组或多维图像进行滤波
         dog_img = mat2gray(dog_img);      % 将图像矩阵归一化0到1范围内(包括0和1)
         dog_img = floor(dog_img * 255);   % 再乘255变成正常灰度值
         test_imgs_2d(n, :, :) = dog_img;
    end
end

%Encoding
%----------------------------Temporal_Coding------------------------------%
if(strcmp(CODING_METHOD, 'Temporal_Coding'))
    train_coding = zeros(1, (round(ETH80_IMG * TRAIN_RATIO) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * 2)); %按最大情况预先分配内存，以加快运行速度
    test_coding = zeros(1, (round(ETH80_IMG * (1 - TRAIN_RATIO)) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * 2)); %按最大情况预先分配内存，以加快运行速度
    %编码训练集图像
    train_cnt = 0;
    for n = 1 : round(ETH80_IMG * TRAIN_RATIO)
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE));  %存储每张图编码后脉冲的地址
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE));     %存储每张图编码后脉冲的时间
        spike_num = 0;   %统计每张图像脉冲个数
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)
                if (train_imgs_2d(n, x, y) ~= 0) %忽略0像素(背景黑)处
                    spike_num = spike_num + 1;
                    addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) + y; %让此处地址从1开始，方便后续处理
                    t(spike_num) = TIME_WINDOW1 - floor(K1 * train_imgs_2d(n, x, y));                   
                end
            end
        end
        %去除数组中的0
        addr(addr == 0) = [];
        t(t == 0) = [];
        %将脉冲发射时间排序（升序）
        [t, index] = sort (t, 'ascend');  %升序排序，并返回排序后数组的索引
        addr = addr(index);               %地址跟着排序
        %将地址和脉冲依次添加进编码数组中，每张图像以-1结尾
        for i = 1 : spike_num
            train_cnt = train_cnt + 1;
            train_coding(train_cnt) = addr(i);
            train_cnt = train_cnt + 1;
            train_coding(train_cnt) = t(i);
        end
        train_cnt = train_cnt + 1;
        train_coding(train_cnt) = -1;
    end
    %编码测试集图像
    test_cnt = 0;
    for n = 1 : round(ETH80_IMG * (1 - TRAIN_RATIO))
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE));  %存储每张图编码后脉冲的地址
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE));     %存储每张图编码后脉冲的时间
        spike_num = 0;   %统计每张图像脉冲个数
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)
                if (test_imgs_2d(n, x, y) ~= 0) %忽略0像素(背景黑)处
                    spike_num = spike_num + 1;
                    addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) + y; %让此处地址从1开始，方便后续处理
                    t(spike_num) = TIME_WINDOW1 - floor(K1 * test_imgs_2d(n, x, y));                   
                end
            end
        end
        %去除数组中的0
        addr(addr == 0) = [];
        t(t == 0) = [];
        %将脉冲发射时间排序（升序）
        [t, index] = sort (t, 'ascend');  %升序排序，并返回排序后数组的索引
        addr = addr(index);               %地址跟着排序
        %将地址和脉冲依次添加进编码数组中，每张图像以-1结尾
        for i = 1 : spike_num
            test_cnt = test_cnt + 1;
            test_coding(test_cnt) = addr(i);
            test_cnt = test_cnt + 1;
            test_coding(test_cnt) = t(i);
        end
        test_cnt = test_cnt + 1;
        test_coding(test_cnt) = -1;
    end
    %保存编码后的数据集和标签
    train_coding(train_coding == 0) = [];  %去除数组中多余的0
    test_coding(test_coding == 0) = [];    %去除数组中多余的0
    save([strcat(TRAIN_ENCODING_SAVE_DIR, 'train_coding'), '.mat'], 'train_coding');
    save([strcat(TEST_ENCODING_SAVE_DIR, 'test_coding'), '.mat'], 'test_coding');
    save([strcat(TRAIN_ENCODING_SAVE_DIR, 'train_labels'), '.mat'], 'train_labels');
    save([strcat(TRAIN_ENCODING_SAVE_DIR, 'test_labels'), '.mat'], 'test_labels'); 
%-------------------------------Rate_Coding--------------------------------%
elseif(strcmp(CODING_METHOD, 'Rate_Coding'))
    train_coding = zeros(1, (round(ETH80_IMG * TRAIN_RATIO) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * 2 * SPIKE_PER_PIXEL)); %按最大情况预先分配内存，以加快运行速度
    test_coding = zeros(1, (round(ETH80_IMG * (1 - TRAIN_RATIO)) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * 2 * SPIKE_PER_PIXEL)); %按最大情况预先分配内存，以加快运行速度
    %编码训练集图像
    train_cnt = 0;
    for n = 1 : round(ETH80_IMG * TRAIN_RATIO)
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * SPIKE_PER_PIXEL);  %存储每张图编码后脉冲的地址
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * SPIKE_PER_PIXEL);     %存储每张图编码后脉冲的时间
        spike_num = 0;   %统计每张图像脉冲个数
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)
                pixel = TriOp(IF_RESIZE, train_resize_imgs_2d(n, x, y), train_imgs_2d(n, x, y));
                if (pixel ~= 0)  %忽略0像素(背景黑)处
                    poiss_tran = poissrnd(K2 * pixel, 1, SPIKE_PER_PIXEL);  %生成以K2*像素值为lmd参数的泊松脉冲序列
                    poiss_tran(poiss_tran <= 0) = 1; poiss_tran(poiss_tran >= TIME_WINDOW2 - 1) = TIME_WINDOW2 - 1; %截断
                    for i = 1 : SPIKE_PER_PIXEL
                        spike_num = spike_num + 1;
                        addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) + y; %让此处地址从1开始，方便后续处理
                        t(spike_num) = poiss_tran(1, i);
                    end
                end
            end
        end
        %去除数组中的0
        addr(addr == 0) = [];
        t(t == 0) = [];
        %将脉冲发射时间排序（升序）
        [t, index] = sort (t, 'ascend');  %升序排序，并返回排序后数组的索引
        addr = addr(index);               %地址跟着排序
        %将地址和脉冲依次添加进编码数组中，每张图像以-1结尾
        for i = 1 : spike_num
            train_cnt = train_cnt + 1;
            train_coding(train_cnt) = addr(i);
            train_cnt = train_cnt + 1;
            train_coding(train_cnt) = t(i);
        end
        train_cnt = train_cnt + 1;
        train_coding(train_cnt) = -1;
    end
    %编码测试集图像
    test_cnt = 0;
    for n = 1 : round(ETH80_IMG * (1 - TRAIN_RATIO))
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * SPIKE_PER_PIXEL);  %存储每张图编码后脉冲的地址
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * SPIKE_PER_PIXEL);     %存储每张图编码后脉冲的时间
        spike_num = 0;   %统计每张图像脉冲个数
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)
                pixel = TriOp(IF_RESIZE, test_resize_imgs_2d(n, x, y), test_imgs_2d(n, x, y));
                if (pixel ~= 0)  %忽略0像素(背景黑)处
                    poiss_tran = poissrnd(K2 * pixel, 1, SPIKE_PER_PIXEL);  %生成以K2*像素值为lmd参数的泊松脉冲序列
                    poiss_tran(poiss_tran <= 0) = 1; poiss_tran(poiss_tran >= TIME_WINDOW2 - 1) = TIME_WINDOW2 - 1; %截断
                    for i = 1 : SPIKE_PER_PIXEL
                        spike_num = spike_num + 1;
                        addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) + y; %让此处地址从1开始，方便后续处理
                        t(spike_num) = poiss_tran(1, i);
                    end
                end
            end
        end
        %去除数组中的0
        addr(addr == 0) = [];
        t(t == 0) = [];
        %将脉冲发射时间排序（升序）
        [t, index] = sort (t, 'ascend');  %升序排序，并返回排序后数组的索引
        addr = addr(index);               %地址跟着排序
        %将地址和脉冲依次添加进编码数组中，每张图像以-1结尾
        for i = 1 : spike_num
            test_cnt = test_cnt + 1;
            test_coding(test_cnt) = addr(i);
            test_cnt = test_cnt + 1;
            test_coding(test_cnt) = t(i);
        end
        test_cnt = test_cnt + 1;
        test_coding(test_cnt) = -1;
    end
    %保存编码后的数据集和标签
    train_coding(train_coding == 0) = [];  %去除数组中多余的0
    test_coding(test_coding == 0) = [];    %去除数组中多余的0
    save([strcat(TRAIN_ENCODING_SAVE_DIR, 'train_coding'), '.mat'], 'train_coding');
    save([strcat(TEST_ENCODING_SAVE_DIR, 'test_coding'), '.mat'], 'test_coding');
    save([strcat(TRAIN_ENCODING_SAVE_DIR, 'train_labels'), '.mat'], 'train_labels');
    save([strcat(TRAIN_ENCODING_SAVE_DIR, 'test_labels'), '.mat'], 'test_labels'); 
end

