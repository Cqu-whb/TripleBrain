% -----------------------------------------------------------------------------------------------------------------------------------------------------%
% @��д��wtx
% @�������ڣ�2022/1/1
% @����ʱ�䣺resizeΪ16x16��15s����
% @��������ȡETH80���ݼ���png��ʽ������������룬֧��Temporal_Coding��Rate_Coding������ߴ�Resize���Լ���˹����˲�����Dog��
% @�������ݸ�ʽ��pngͼ��
% @�����ʽ��1��n������[addr1 t1 addr2 t2 ... -1 addr1 t1 addr2 t2 ... -1 ...], ��ַaddr��Χ��[1, img_size^2], ʱ���t��Χ��[1, TIME_WINDOW - 1],
%           ʱ��t����������ÿ��ͼ��֮����-1�ָ�
% @��ע1��ע��matlab����ĵ�ַ�Ǵ�1��ʼ�ģ�C#�����Դ���ʱ��Ҫ����ַ��1
% @��ע2��ÿ�����нű�������������һ�Σ��õ���ͬ��ѵ�����Ͳ��Լ�
% @��ע3��Dog����ʱȡ����Ӧλ�ô���ע�ͿɶԱ�ԭʼͼ���Dog���ͼ��
% @��ע4���������Ŀ¼����������ֶ��ڵ����д������������ڴ������޸�Ŀ¼�����úò����Զ��޸� eg.��Temporal_CodingĿ¼�´�����Ϊ16x16_Dog���ļ���
% -----------------------------------------------------------------------------------------------------------------------------------------------------%
clear;clc;
load('.\Definition_pkg');
%----------------------------------------------------------------------����������-------------------------------------------------------------------%
%���ݼ���Ԥ�����������뷽��ѡ��
PRE_PROC = 'Gray';                   % Ԥ����������ѡ��'Gray'��'Dog'
CODING_METHOD = 'Temporal_Coding';  % ���뷽������ѡ��Temporal_Coding'��'Rate_Coding'
TRAIN_RATIO = 0.8;                  % ѵ������ռ����

%Resize����
IF_RESIZE = 1;                      % �Ƿ���resize����1--True, 0--False
RESIZE_SIZE = 16;                   % resize���ͼ���С

%��˹����˲���Dog������
SIGMA1 = 1; SIGMA2 = 3; WINDOW = 15;

%Temporal_Coding����, t = TIME_WINDOW1 - floor(K1 * pixel)
TIME_WINDOW1 = 256;                   %ʱ�䴰��
K1 = TIME_WINDOW1 / MAX_PIXEL_VALUE;  %����ϵ��

%Rate_Coding����, ��K2*����ֵΪlmd�������ɲ�������
TIME_WINDOW2 = 256;                   %ʱ�䴰��
SPIKE_PER_PIXEL = 5;                  %ÿ������ֵ���ɵĲ����������
K2 = TIME_WINDOW2 / MAX_PIXEL_VALUE;  %����ϵ��
%----------------------------------------------------------------------����������-------------------------------------------------------------------%
IMG_DIR = '..\..\Common_Datasets\Frame_based\ETH80\Original_png\';

%����󱣴�Ŀ¼����һ��������Լ����󴴽���eg.��Temporal_CodingĿ¼�´�����Ϊ16x16_Dog���ļ��У�
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

%����ÿ��ͼ�񣬲��ֶ����ɱ�ǩ
img_cnt = 0;
for i = 1 : ETH80_NCLASS %8
    for j = 1 : ETH80_NFOLDER_PER_CLASS %10
        img_list = dir(strcat(IMG_DIR, num2str(i), '\', num2str(j), '\', '*.png')); %��ȡ��Ŀ¼������png��ʽͼ����Ϣ
        for n = 1 : ETH80_NIMG_PER_FOLDER %41
            img_name = img_list(n).name;
            img = imread(strcat(IMG_DIR, num2str(i), '\', num2str(j), '\', img_name));  %��ȡÿ��ͼ��
            img_gray = rgb2gray(img);
            if (IF_RESIZE)  %resize
                img_gray = floor(imresize(img_gray, [RESIZE_SIZE, RESIZE_SIZE], 'bilinear'));  %˫���Բ�ֵ�������ڽ���ֵЧ���ã�              
            end
            img_cnt = img_cnt + 1;
            all_imgs_2d(img_cnt, :, :) = img_gray;
            all_labels(img_cnt, 1) = i - 1; %0-7
        end
    end
end

%����ͼ�񣬲�����ѵ�����Ͳ��Լ�
rand_num = randperm(ETH80_IMG, ETH80_IMG);  %����һ�д�1��x�������е�x��������x����������ͬ
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

%�ж��Ƿ��˹�˲�
%type= 'gaussian'��Ϊ��˹��ͨ�˲�����ģ����������sigma��ʾ�˲����ı�׼���λΪ���أ�Ĭ��ֵΪ 0.5����window��ʾģ��ߴ磬Ĭ��ֵΪ[3,3]
if (strcmp(PRE_PROC, 'Dog'))
    H1 = fspecial('gaussian', WINDOW, SIGMA1);
    H2 = fspecial('gaussian', WINDOW, SIGMA2);
    DiffGauss = H1 - H2;
    %�˲�ÿ��ѵ��ͼ��
    for n = 1 : round(ETH80_IMG * TRAIN_RATIO)
         dog_img = abs(imfilter(squeeze(train_imgs_2d(n, :, :)), DiffGauss, 'replicate'));   %����������������άͼ������˲�
         dog_img = mat2gray(dog_img);      % ��ͼ������һ��0��1��Χ��(����0��1)
         dog_img = floor(dog_img * 255);   % �ٳ�255��������Ҷ�ֵ    
         
         %���Dog���ͼ���Ƿ����
%          subplot(1, 2, 1); imshow(squeeze(train_imgs_2d(n, :, :))/255); title('Ori');
%          subplot(1, 2, 2); imshow(dog_img/255); title('Dog');
%          debug_temp = 0;   %�˴����öϵ�Ա�Dogǰ��ͼ��     
         
         train_imgs_2d(n, :, :) = dog_img;     
    end
    %�˲�ÿ�Ų���ͼ��
    for n = 1 : round(ETH80_IMG * (1 - TRAIN_RATIO))
         dog_img = abs(imfilter(squeeze(test_imgs_2d(n, :, :)), DiffGauss, 'replicate'));   %����������������άͼ������˲�
         dog_img = mat2gray(dog_img);      % ��ͼ������һ��0��1��Χ��(����0��1)
         dog_img = floor(dog_img * 255);   % �ٳ�255��������Ҷ�ֵ
         test_imgs_2d(n, :, :) = dog_img;
    end
end

%Encoding
%----------------------------Temporal_Coding------------------------------%
if(strcmp(CODING_METHOD, 'Temporal_Coding'))
    train_coding = zeros(1, (round(ETH80_IMG * TRAIN_RATIO) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * 2)); %��������Ԥ�ȷ����ڴ棬�Լӿ������ٶ�
    test_coding = zeros(1, (round(ETH80_IMG * (1 - TRAIN_RATIO)) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * 2)); %��������Ԥ�ȷ����ڴ棬�Լӿ������ٶ�
    %����ѵ����ͼ��
    train_cnt = 0;
    for n = 1 : round(ETH80_IMG * TRAIN_RATIO)
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE));  %�洢ÿ��ͼ���������ĵ�ַ
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE));     %�洢ÿ��ͼ����������ʱ��
        spike_num = 0;   %ͳ��ÿ��ͼ���������
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)
                if (train_imgs_2d(n, x, y) ~= 0) %����0����(������)��
                    spike_num = spike_num + 1;
                    addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) + y; %�ô˴���ַ��1��ʼ�������������
                    t(spike_num) = TIME_WINDOW1 - floor(K1 * train_imgs_2d(n, x, y));                   
                end
            end
        end
        %ȥ�������е�0
        addr(addr == 0) = [];
        t(t == 0) = [];
        %�����巢��ʱ����������
        [t, index] = sort (t, 'ascend');  %�������򣬲�������������������
        addr = addr(index);               %��ַ��������
        %����ַ������������ӽ����������У�ÿ��ͼ����-1��β
        for i = 1 : spike_num
            train_cnt = train_cnt + 1;
            train_coding(train_cnt) = addr(i);
            train_cnt = train_cnt + 1;
            train_coding(train_cnt) = t(i);
        end
        train_cnt = train_cnt + 1;
        train_coding(train_cnt) = -1;
    end
    %������Լ�ͼ��
    test_cnt = 0;
    for n = 1 : round(ETH80_IMG * (1 - TRAIN_RATIO))
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE));  %�洢ÿ��ͼ���������ĵ�ַ
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE));     %�洢ÿ��ͼ����������ʱ��
        spike_num = 0;   %ͳ��ÿ��ͼ���������
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)
                if (test_imgs_2d(n, x, y) ~= 0) %����0����(������)��
                    spike_num = spike_num + 1;
                    addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) + y; %�ô˴���ַ��1��ʼ�������������
                    t(spike_num) = TIME_WINDOW1 - floor(K1 * test_imgs_2d(n, x, y));                   
                end
            end
        end
        %ȥ�������е�0
        addr(addr == 0) = [];
        t(t == 0) = [];
        %�����巢��ʱ����������
        [t, index] = sort (t, 'ascend');  %�������򣬲�������������������
        addr = addr(index);               %��ַ��������
        %����ַ������������ӽ����������У�ÿ��ͼ����-1��β
        for i = 1 : spike_num
            test_cnt = test_cnt + 1;
            test_coding(test_cnt) = addr(i);
            test_cnt = test_cnt + 1;
            test_coding(test_cnt) = t(i);
        end
        test_cnt = test_cnt + 1;
        test_coding(test_cnt) = -1;
    end
    %������������ݼ��ͱ�ǩ
    train_coding(train_coding == 0) = [];  %ȥ�������ж����0
    test_coding(test_coding == 0) = [];    %ȥ�������ж����0
    save([strcat(TRAIN_ENCODING_SAVE_DIR, 'train_coding'), '.mat'], 'train_coding');
    save([strcat(TEST_ENCODING_SAVE_DIR, 'test_coding'), '.mat'], 'test_coding');
    save([strcat(TRAIN_ENCODING_SAVE_DIR, 'train_labels'), '.mat'], 'train_labels');
    save([strcat(TRAIN_ENCODING_SAVE_DIR, 'test_labels'), '.mat'], 'test_labels'); 
%-------------------------------Rate_Coding--------------------------------%
elseif(strcmp(CODING_METHOD, 'Rate_Coding'))
    train_coding = zeros(1, (round(ETH80_IMG * TRAIN_RATIO) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * 2 * SPIKE_PER_PIXEL)); %��������Ԥ�ȷ����ڴ棬�Լӿ������ٶ�
    test_coding = zeros(1, (round(ETH80_IMG * (1 - TRAIN_RATIO)) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * 2 * SPIKE_PER_PIXEL)); %��������Ԥ�ȷ����ڴ棬�Լӿ������ٶ�
    %����ѵ����ͼ��
    train_cnt = 0;
    for n = 1 : round(ETH80_IMG * TRAIN_RATIO)
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * SPIKE_PER_PIXEL);  %�洢ÿ��ͼ���������ĵ�ַ
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * SPIKE_PER_PIXEL);     %�洢ÿ��ͼ����������ʱ��
        spike_num = 0;   %ͳ��ÿ��ͼ���������
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)
                pixel = TriOp(IF_RESIZE, train_resize_imgs_2d(n, x, y), train_imgs_2d(n, x, y));
                if (pixel ~= 0)  %����0����(������)��
                    poiss_tran = poissrnd(K2 * pixel, 1, SPIKE_PER_PIXEL);  %������K2*����ֵΪlmd�����Ĳ�����������
                    poiss_tran(poiss_tran <= 0) = 1; poiss_tran(poiss_tran >= TIME_WINDOW2 - 1) = TIME_WINDOW2 - 1; %�ض�
                    for i = 1 : SPIKE_PER_PIXEL
                        spike_num = spike_num + 1;
                        addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) + y; %�ô˴���ַ��1��ʼ�������������
                        t(spike_num) = poiss_tran(1, i);
                    end
                end
            end
        end
        %ȥ�������е�0
        addr(addr == 0) = [];
        t(t == 0) = [];
        %�����巢��ʱ����������
        [t, index] = sort (t, 'ascend');  %�������򣬲�������������������
        addr = addr(index);               %��ַ��������
        %����ַ������������ӽ����������У�ÿ��ͼ����-1��β
        for i = 1 : spike_num
            train_cnt = train_cnt + 1;
            train_coding(train_cnt) = addr(i);
            train_cnt = train_cnt + 1;
            train_coding(train_cnt) = t(i);
        end
        train_cnt = train_cnt + 1;
        train_coding(train_cnt) = -1;
    end
    %������Լ�ͼ��
    test_cnt = 0;
    for n = 1 : round(ETH80_IMG * (1 - TRAIN_RATIO))
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * SPIKE_PER_PIXEL);  %�洢ÿ��ͼ���������ĵ�ַ
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) * SPIKE_PER_PIXEL);     %�洢ÿ��ͼ����������ʱ��
        spike_num = 0;   %ͳ��ÿ��ͼ���������
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)
                pixel = TriOp(IF_RESIZE, test_resize_imgs_2d(n, x, y), test_imgs_2d(n, x, y));
                if (pixel ~= 0)  %����0����(������)��
                    poiss_tran = poissrnd(K2 * pixel, 1, SPIKE_PER_PIXEL);  %������K2*����ֵΪlmd�����Ĳ�����������
                    poiss_tran(poiss_tran <= 0) = 1; poiss_tran(poiss_tran >= TIME_WINDOW2 - 1) = TIME_WINDOW2 - 1; %�ض�
                    for i = 1 : SPIKE_PER_PIXEL
                        spike_num = spike_num + 1;
                        addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE) + y; %�ô˴���ַ��1��ʼ�������������
                        t(spike_num) = poiss_tran(1, i);
                    end
                end
            end
        end
        %ȥ�������е�0
        addr(addr == 0) = [];
        t(t == 0) = [];
        %�����巢��ʱ����������
        [t, index] = sort (t, 'ascend');  %�������򣬲�������������������
        addr = addr(index);               %��ַ��������
        %����ַ������������ӽ����������У�ÿ��ͼ����-1��β
        for i = 1 : spike_num
            test_cnt = test_cnt + 1;
            test_coding(test_cnt) = addr(i);
            test_cnt = test_cnt + 1;
            test_coding(test_cnt) = t(i);
        end
        test_cnt = test_cnt + 1;
        test_coding(test_cnt) = -1;
    end
    %������������ݼ��ͱ�ǩ
    train_coding(train_coding == 0) = [];  %ȥ�������ж����0
    test_coding(test_coding == 0) = [];    %ȥ�������ж����0
    save([strcat(TRAIN_ENCODING_SAVE_DIR, 'train_coding'), '.mat'], 'train_coding');
    save([strcat(TEST_ENCODING_SAVE_DIR, 'test_coding'), '.mat'], 'test_coding');
    save([strcat(TRAIN_ENCODING_SAVE_DIR, 'train_labels'), '.mat'], 'train_labels');
    save([strcat(TRAIN_ENCODING_SAVE_DIR, 'test_labels'), '.mat'], 'test_labels'); 
end

