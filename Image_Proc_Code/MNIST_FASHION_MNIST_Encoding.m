% -----------------------------------------------------------------------------------------------------------------------------------------------------%
% @编写：wtx
% @更新日期：2022/1/1
% @运行时间：Temporal_Coding 5min以内, Rate_Coding时间较长, 由配置参数决定，一般在1h以内
% @概述：读取MNIST或FAHION_MNIST数据集（mat格式），并将其编码，支持Temporal_Coding、Rate_Coding、任意尺寸Resize、以及高斯差分滤波处理（Dog）
% @输入数据格式：eg.训练集图像和标签：[60000, 784], [60000, 1]的矩阵，以mat格式存储
% @输出格式：1行n列数组[addr1 t1 addr2 t2 ... -1 addr1 t1 addr2 t2 ... -1 ...], 地址addr范围：[1, img_size^2], 时间戳t范围：[1, TIME_WINDOW - 1],
%           时间t以升序排序，每张图像之间以-1分隔
% @备注1：注意matlab输出的地址是从1开始的，C#等语言处理时需要将地址减1
% @备注2：Dog处理时取消相应位置代码注释可对比原始图像和Dog后的图像
% @备注3：Temporal_Coding和Rate_Coding有很多种实现方式，但本质相同，此处各提供一种方法作为参考
% @备注4：编码输出目录需根据需求手动在电脑中创建，但无需在代码中修改目录，配置好参数自动修改 eg.在Temporal_Coding目录下创建名为28x28_Dog的文件夹
% @备注5：使用imshow显示图像会发现图像逆时针旋转了90°，但图像的旋转不影响编码的结果
% -----------------------------------------------------------------------------------------------------------------------------------------------------%
clear;clc;
load('.\Definition_pkg');
%----------------------------------------------------------------------参数配置区-------------------------------------------------------------------%
%数据集、预处理方法、编码方法选择
DATASET = 'MNIST';                  % 数据集，可选'MNIST'或'FASHION_MNIST'
PRE_PROC = 'Gray';                   % 预处理方法，可选：'Gray'或'Dog'
CODING_METHOD = 'Temporal_Coding';  % 编码方法，可选：Temporal_Coding'、'Rate_Coding'或'MultiSpike_Temporal_Coding'

%Resize参数
IF_RESIZE = 1;                      % 是否做resize处理，1--True, 0--False
RESIZE_SIZE = 16;                   % resize后的图像大小

%高斯差分滤波（Dog）参数
SIGMA1 = 1; SIGMA2 = 5; WINDOW = 15;

%Temporal_Coding参数, t = TIME_WINDOW1 - floor(K1 * pixel)
TIME_WINDOW1 = 256;                   %时间窗口
K1 = TIME_WINDOW1 / MAX_PIXEL_VALUE;  %编码系数

%Rate_Coding参数, 以K2*像素值为lmd参数生成泊松序列
TIME_WINDOW2 = 256;                   %时间窗口
SPIKE_PER_PIXEL = 3;                  %每个像素值生成的泊松脉冲个数
K2 = TIME_WINDOW2 / MAX_PIXEL_VALUE;  %编码系数

%MultiSpike_Temporal_Coding参数
TIME_WINDOW3 = 256;
BETA = 8e-5;  %T=256, beta=8e-5时像素255可发射5个脉冲，像素50可发射一个脉冲
VTH = 1;
%----------------------------------------------------------------------参数配置区-------------------------------------------------------------------%

%导入数据
load(strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_Denoising_mat\train_images'));
load(strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_Denoising_mat\test_images'));

%编码后保存目录（第一次需根据自己需求创建，eg.在Temporal_Coding目录下创建名为28x28_Dog的文件夹）
folder_name = strcat(num2str(TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)), 'x', num2str(TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)), '_', PRE_PROC);
TRAIN_ENCODING_SAVE_DIR = strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Encoding\', CODING_METHOD, '\', folder_name, '\');
TEST_ENCODING_SAVE_DIR = strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Encoding\', CODING_METHOD, '\', folder_name, '\');

%为方便后续处理，先将原始图像数据转为2维
train_imgs_2d = zeros(MNIST_TRAIN_IMG, MNIST_SIZE, MNIST_SIZE);
test_imgs_2d = zeros(MNIST_TEST_IMG, MNIST_SIZE, MNIST_SIZE);
for n = 1 : MNIST_TRAIN_IMG
    train_imgs_2d(n, :, :) = reshape(train_imgs(n, :), MNIST_SIZE, MNIST_SIZE);
end
for n = 1 : MNIST_TEST_IMG
    test_imgs_2d(n, :, :) = reshape(test_imgs(n, :), MNIST_SIZE, MNIST_SIZE);
end

%先resize，再dog
train_resize_imgs_2d = zeros(MNIST_TRAIN_IMG, TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE), TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE));
test_resize_imgs_2d = zeros(MNIST_TEST_IMG, TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE), TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE));
if (IF_RESIZE == 1) 
    %reshape每张训练集图像
    for n = 1 : MNIST_TRAIN_IMG
        temp_img_2d = squeeze(train_imgs_2d(n, :, :));  % 删除为1的维度
        resize_img_2d = floor(imresize(temp_img_2d, [RESIZE_SIZE, RESIZE_SIZE], 'bilinear'));  %双线性插值（比最邻近插值效果好）
        train_resize_imgs_2d(n, :, :) = resize_img_2d;
    end
     %reshape每张测试图像
    for n = 1 : MNIST_TEST_IMG
        temp_img_2d = squeeze(test_imgs_2d(n, :, :));   % 删除为1的维度
        resize_img_2d = floor(imresize(temp_img_2d, [RESIZE_SIZE, RESIZE_SIZE], 'bilinear'));  %双线性插值（比最邻近插值效果好）
        test_resize_imgs_2d(n, :, :) = resize_img_2d;
    end
end

%判断是否高斯滤波
%type= 'gaussian'，为高斯低通滤波器，模板有两个，sigma表示滤波器的标准差（单位为像素，默认值为 0.5），window表示模版尺寸，默认值为[3,3]
if (strcmp(PRE_PROC, 'Dog'))
    H1 = fspecial('gaussian', WINDOW, SIGMA1);
    H2 = fspecial('gaussian', WINDOW, SIGMA2);
    DiffGauss = H1 - H2;
    %滤波每张训练图像
    for n = 1 : MNIST_TRAIN_IMG
         dog_img = abs(imfilter(TriOp(IF_RESIZE, squeeze(train_resize_imgs_2d(n, :, :)), squeeze(train_imgs_2d(n, :, :))), DiffGauss, 'replicate'));   %对任意类型数组或多维图像进行滤波
         dog_img = mat2gray(dog_img);      % 将图像矩阵归一化0到1范围内(包括0和1)
         dog_img = floor(dog_img * 255);   % 再乘255变成正常灰度值         
         %检查Dog后的图像是否合理
%          subplot(1, 2, 1); imshow(TriOp(IF_RESIZE, squeeze(train_resize_imgs_2d(n, :, :)), squeeze(train_imgs_2d(n, :, :)))/255); title('Ori');
%          subplot(1, 2, 2); imshow(dog_img/255); title('Dog');
%          debug_temp = 0;   %此处设置断点对比Dog前后图像     
         if (IF_RESIZE)
             train_resize_imgs_2d(n, :, :) = dog_img;
         else
             train_imgs_2d(n, :, :) = dog_img;
         end       
    end
    %滤波每张测试图像
    for n = 1 : MNIST_TEST_IMG
         dog_img = abs(imfilter(TriOp(IF_RESIZE, squeeze(test_resize_imgs_2d(n, :, :)), squeeze(test_imgs_2d(n, :, :))), DiffGauss, 'replicate'));   %对任意类型数组或多维图像进行滤波
         dog_img = mat2gray(dog_img);      % 将图像矩阵归一化0到1范围内(包括0和1)
         dog_img = floor(dog_img * 255);   % 再乘255变成正常灰度值
         if (IF_RESIZE)
             test_resize_imgs_2d(n, :, :) = dog_img;
         else
             test_imgs_2d(n, :, :) = dog_img;
         end
    end
end

%Encoding
%----------------------------Temporal_Coding------------------------------%
if(strcmp(CODING_METHOD, 'Temporal_Coding'))
    train_coding = zeros(1, (MNIST_TRAIN_IMG * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * 2)); %按最大情况预先分配内存，以加快运行速度
    test_coding = zeros(1, (MNIST_TEST_IMG * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * 2)); %按最大情况预先分配内存，以加快运行速度
    %编码训练集图像
    train_cnt = 0;
    for n = 1 : MNIST_TRAIN_IMG
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE));  %存储每张图编码后脉冲的地址
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE));     %存储每张图编码后脉冲的时间
        spike_num = 0;   %统计每张图像脉冲个数
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
                if (TriOp(IF_RESIZE, train_resize_imgs_2d(n, x, y), train_imgs_2d(n, x, y)) ~= 0) %忽略0像素(背景黑)处
                    spike_num = spike_num + 1;
                    addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) + y; %让此处地址从1开始，方便后续处理
                    t(spike_num) = TIME_WINDOW1 - floor(K1 * TriOp(IF_RESIZE, train_resize_imgs_2d(n, x, y), train_imgs_2d(n, x, y)));                   
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
    for n = 1 : MNIST_TEST_IMG
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE));  %存储每张图编码后脉冲的地址
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE));     %存储每张图编码后脉冲的时间
        spike_num = 0;   %统计每张图像脉冲个数
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
                if (TriOp(IF_RESIZE, test_resize_imgs_2d(n, x, y), test_imgs_2d(n, x, y)) ~= 0) %忽略0像素(背景黑)处
                    spike_num = spike_num + 1;
                    addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) + y; %让此处地址从1开始，方便后续处理
                    t(spike_num) = TIME_WINDOW1 - floor(K1 * TriOp(IF_RESIZE, test_resize_imgs_2d(n, x, y), test_imgs_2d(n, x, y)));                   
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
    %去除数组中多余的0
    train_coding(train_coding == 0) = []; 
    test_coding(test_coding == 0) = [];
%-------------------------------Rate_Coding--------------------------------%
elseif(strcmp(CODING_METHOD, 'Rate_Coding'))
    train_coding = zeros(1, (MNIST_TRAIN_IMG * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * 2 * SPIKE_PER_PIXEL)); %按最大情况预先分配内存，以加快运行速度
    test_coding = zeros(1, (MNIST_TEST_IMG * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * 2 * SPIKE_PER_PIXEL)); %按最大情况预先分配内存，以加快运行速度
    %编码训练集图像
    train_cnt = 0;
    for n = 1 : MNIST_TRAIN_IMG
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * SPIKE_PER_PIXEL);  %存储每张图编码后脉冲的地址
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * SPIKE_PER_PIXEL);     %存储每张图编码后脉冲的时间
        spike_num = 0;   %统计每张图像脉冲个数
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
                pixel = TriOp(IF_RESIZE, train_resize_imgs_2d(n, x, y), train_imgs_2d(n, x, y));
                if (pixel ~= 0)  %忽略0像素(背景黑)处
                    poiss_tran = poissrnd(K2 * pixel, 1, SPIKE_PER_PIXEL);  %生成以K2*像素值为lmd参数的泊松脉冲序列
                    poiss_tran(poiss_tran <= 0) = 1; poiss_tran(poiss_tran >= TIME_WINDOW2 - 1) = TIME_WINDOW2 - 1; %截断
                    for i = 1 : SPIKE_PER_PIXEL
                        spike_num = spike_num + 1;
                        addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) + y; %让此处地址从1开始，方便后续处理
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
    for n = 1 : MNIST_TEST_IMG
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * SPIKE_PER_PIXEL);  %存储每张图编码后脉冲的地址
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * SPIKE_PER_PIXEL);     %存储每张图编码后脉冲的时间
        spike_num = 0;   %统计每张图像脉冲个数
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
                pixel = TriOp(IF_RESIZE, test_resize_imgs_2d(n, x, y), test_imgs_2d(n, x, y));
                if (pixel ~= 0)  %忽略0像素(背景黑)处
                    poiss_tran = poissrnd(K2 * pixel, 1, SPIKE_PER_PIXEL);  %生成以K2*像素值为lmd参数的泊松脉冲序列
                    poiss_tran(poiss_tran <= 0) = 1; poiss_tran(poiss_tran >= TIME_WINDOW2 - 1) = TIME_WINDOW2 - 1; %截断
                    for i = 1 : SPIKE_PER_PIXEL
                        spike_num = spike_num + 1;
                        addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) + y; %让此处地址从1开始，方便后续处理
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
    %去除数组中多余的0
    train_coding(train_coding == 0) = [];
    test_coding(test_coding == 0) = [];
%-------------------------------MultiSpike_Temporal_Coding--------------------------------%
elseif(strcmp(CODING_METHOD, 'MultiSpike_Temporal_Coding'))
    train_coding = zeros(1, 1); %直接这样申明也慢不了多少
    test_coding = zeros(1, 1); 
    %编码训练集图像
    train_cnt = 0;
    for n = 1 : MNIST_TRAIN_IMG
        addr = zeros(1, 1);  %存储每张图编码后脉冲的地址
        t = zeros(1, 1);     %存储每张图编码后脉冲的时间
        spike_num = 0;   %统计每张图像脉冲个数
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
                pixel = TriOp(IF_RESIZE, train_resize_imgs_2d(n, x, y), train_imgs_2d(n, x, y));
                if (pixel ~= 0)  %忽略0像素(背景黑)处
                    V = 0; t0 = 0;
                    for ti = 1 : TIME_WINDOW3
                        V = BETA * pixel * (ti - t0);
                        if V >= VTH
                            spike_num = spike_num + 1;
                            addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) + y; %让此处地址从1开始，方便后续处理
                            t(spike_num) = ti;
                            t0 = t;
                        end
                    end
                end
            end
        end
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
    for n = 1 : MNIST_TEST_IMG
        addr = zeros(1, 1);  %存储每张图编码后脉冲的地址
        t = zeros(1,1);     %存储每张图编码后脉冲的时间
        spike_num = 0;   %统计每张图像脉冲个数
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
                pixel = TriOp(IF_RESIZE, test_resize_imgs_2d(n, x, y), test_imgs_2d(n, x, y));
                if (pixel ~= 0)  %忽略0像素(背景黑)处
                    V = 0; t0 = 0;
                    for ti = 1 : TIME_WINDOW3
                        V = BETA * pixel * (ti - t0);
                        if V >= VTH
                            spike_num = spike_num + 1;
                            addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) + y; %让此处地址从1开始，方便后续处理
                            t(spike_num) = ti;
                            t0 = t;
                        end
                    end
                end
            end
        end
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
end
%保存编码后的数据集
save([strcat(TRAIN_ENCODING_SAVE_DIR, 'train_coding'), '.mat'], 'train_coding');
save([strcat(TEST_ENCODING_SAVE_DIR, 'test_coding'), '.mat'], 'test_coding');
copyfile(strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_Denoising_mat\train_labels.mat'),TRAIN_ENCODING_SAVE_DIR); %拷贝标签方便训练调用
copyfile(strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_Denoising_mat\test_labels.mat'),TEST_ENCODING_SAVE_DIR);   %拷贝标签方便训练调用
