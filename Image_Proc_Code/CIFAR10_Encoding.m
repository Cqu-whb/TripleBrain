% -----------------------------------------------------------------------------------------------------------------------------------------------------%
% @��д��wtx
% @�������ڣ�2022/1/1
% @����ʱ�䣺Temporal_Coding��Resize��16x16��70s����Resize 150s����
% @��������ȡCIFAR10���ݼ���mat��ʽ������������룬֧��Temporal_Coding��Rate_Coding������ߴ�Resize���Լ���˹����˲�����Dog��
% @�������ݸ�ʽ��ѵ����5��batch�����Լ�1��batch��ÿ��batch����10000��32x32����ͨ��ͼ�����ǩ��ͼ����[10000, 3072]��int8�����ʾ��ͨ��˳��Ϊ�졢�̡���
% @�����ʽ��1��n������[addr1 t1 addr2 t2 ... -1 addr1 t1 addr2 t2 ... -1 ...], ��ַaddr��Χ��[1, img_size^2], ʱ���t��Χ��[1, TIME_WINDOW - 1],
%           ʱ��t����������ÿ��ͼ��֮����-1�ָ�
% @��ע1���ð汾��֧��CIFAR10ת�Ҷȼ�ת�ҶȺ�Dog������Ҫ���ڲ�ɫͼ�����������޸Ľű�
% @��ע2��ע��matlab����ĵ�ַ�Ǵ�1��ʼ�ģ�C#�����Դ���ʱ��Ҫ����ַ��1
% @��ע3��Dog����ʱȡ����Ӧλ�ô���ע�ͿɶԱ�ԭʼͼ���Dog���ͼ��
% @��ע4���������Ŀ¼����������ֶ��ڵ����д������������ڴ������޸�Ŀ¼�����úò����Զ��޸� eg.��Temporal_CodingĿ¼�´�����Ϊ16x16_Dog���ļ���
% -----------------------------------------------------------------------------------------------------------------------------------------------------%
clear;clc;
load('.\Definition_pkg');
%----------------------------------------------------------------------����������-------------------------------------------------------------------%
%���ݼ���Ԥ�����������뷽��ѡ��
PRE_PROC = 'Gray';                   % Ԥ����������ѡ��'Gray'��'Dog'
CODING_METHOD = 'Temporal_Coding';  % ���뷽������ѡ��Temporal_Coding'��'Rate_Coding'

%Resize����
IF_RESIZE = 1;                      % �Ƿ���resize����1--True, 0--False
RESIZE_SIZE = 16;                   % resize���ͼ���С

%��˹����˲���Dog������
SIGMA1 = 1; SIGMA2 = 3; WINDOW = 3;

%Temporal_Coding����, t = TIME_WINDOW1 - floor(K1 * pixel)
TIME_WINDOW1 = 256;                   %ʱ�䴰��
K1 = TIME_WINDOW1 / MAX_PIXEL_VALUE;  %����ϵ��

%Rate_Coding����, ��K2*����ֵΪlmd�������ɲ�������
TIME_WINDOW2 = 256;                   %ʱ�䴰��
SPIKE_PER_PIXEL = 5;                  %ÿ������ֵ���ɵĲ����������
K2 = TIME_WINDOW2 / MAX_PIXEL_VALUE;  %����ϵ��
%----------------------------------------------------------------------����������-------------------------------------------------------------------%

%����󱣴�Ŀ¼����һ��������Լ����󴴽���eg.��Temporal_CodingĿ¼�´�����Ϊ16x16_Dog���ļ��У�
folder_name = strcat(num2str(TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE)), 'x', num2str(TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE)), '_', PRE_PROC);
TRAIN_ENCODING_SAVE_DIR = strcat('..\..\Common_Datasets\Frame_based\CIFAR10\Encoding\', CODING_METHOD, '\', folder_name, '\');
TEST_ENCODING_SAVE_DIR = strcat('..\..\Common_Datasets\Frame_based\CIFAR10\Encoding\', CODING_METHOD, '\', folder_name, '\');

%��������
train_imgs_2d = zeros(CIFAR10_TRAIN_IMG, TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE), TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE));
test_imgs_2d = zeros(CIFAR10_TEST_IMG, TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE), TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE));
train_labels = zeros(CIFAR10_TRAIN_IMG, 1);
test_labels = zeros(CIFAR10_TEST_IMG, 1);

%����ѵ����
train_cnt = 0;
temp_img_3d = zeros(CIFAR10_SIZE, CIFAR10_SIZE, 3);
for i = 1 : CIFAR10_TRAIN_NBATCH %5
    load(strcat('..\..\Common_Datasets\Frame_based\CIFAR10\Original_mat\data_batch_', num2str(i)));
    for n = 1 : CIFAR10_NIMG_PER_BATCH %10000
        temp_img_3d = reshape(data(n, :, :), CIFAR10_SIZE, CIFAR10_SIZE, 3);
        temp_img_3d = rot90(temp_img_3d, 3);    %��ͼ��ת��
        temp_img_gray = rgb2gray(temp_img_3d);  %ת�Ҷ�
        if (IF_RESIZE)  %resize
            temp_img_gray = floor(imresize(temp_img_gray, [RESIZE_SIZE, RESIZE_SIZE], 'bilinear'));  %˫���Բ�ֵ�������ڽ���ֵЧ���ã� 
        end
        train_cnt = train_cnt + 1;
        train_imgs_2d(train_cnt, :, :) = temp_img_gray;
        train_labels(train_cnt, 1) = labels(n, 1);
    end
end
%������Լ�
test_cnt = 0;
%temp_img_3d = zeros(CIFAR10_SIZE, CIFAR10_SIZE, 3);
load('..\..\Common_Datasets\Frame_based\CIFAR10\Original_mat\test_batch');
for n = 1 : CIFAR10_NIMG_PER_BATCH %10000
    temp_img_3d = reshape(data(n, :, :), CIFAR10_SIZE, CIFAR10_SIZE, 3);
    temp_img_3d = rot90(temp_img_3d, 3);       %��ͼ��ת��
    temp_img_gray = rgb2gray(temp_img_3d);  %ת�Ҷ�
    if (IF_RESIZE)  %resize
        temp_img_gray = floor(imresize(temp_img_gray, [RESIZE_SIZE, RESIZE_SIZE], 'bilinear'));  %˫���Բ�ֵ�������ڽ���ֵЧ���ã� 
    end
    test_cnt = test_cnt + 1;
    test_imgs_2d(test_cnt, :, :) = temp_img_gray;
    test_labels(test_cnt, 1) = labels(n, 1);
end

%�ж��Ƿ��˹�˲�
%type= 'gaussian'��Ϊ��˹��ͨ�˲�����ģ����������sigma��ʾ�˲����ı�׼���λΪ���أ�Ĭ��ֵΪ 0.5����window��ʾģ��ߴ磬Ĭ��ֵΪ[3,3]
if (strcmp(PRE_PROC, 'Dog'))
    H1 = fspecial('gaussian', WINDOW, SIGMA1);
    H2 = fspecial('gaussian', WINDOW, SIGMA2);
    DiffGauss = H1 - H2;
    %�˲�ÿ��ѵ��ͼ��
    for n = 1 : CIFAR10_TRAIN_IMG
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
    for n = 1 : CIFAR10_TEST_IMG
         dog_img = abs(imfilter(squeeze(test_imgs_2d(n, :, :)), DiffGauss, 'replicate'));   %����������������άͼ������˲�
         dog_img = mat2gray(dog_img);      % ��ͼ������һ��0��1��Χ��(����0��1)
         dog_img = floor(dog_img * 255);   % �ٳ�255��������Ҷ�ֵ
         test_imgs_2d(n, :, :) = dog_img;
    end
end

%Encoding
%----------------------------Temporal_Coding------------------------------%
if(strcmp(CODING_METHOD, 'Temporal_Coding'))
    train_coding = zeros(1, (CIFAR10_TRAIN_IMG * TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * 2)); %��������Ԥ�ȷ����ڴ棬�Լӿ������ٶ�
    test_coding = zeros(1, (CIFAR10_TEST_IMG * TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * 2)); %��������Ԥ�ȷ����ڴ棬�Լӿ������ٶ�
    %����ѵ����ͼ��
    train_cnt = 0;
    for n = 1 : CIFAR10_TRAIN_IMG
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE));  %�洢ÿ��ͼ���������ĵ�ַ
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE));     %�洢ÿ��ͼ����������ʱ��
        spike_num = 0;   %ͳ��ÿ��ͼ���������
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE)
                if (train_imgs_2d(n, x, y) ~= 0) %����0����(������)��
                    spike_num = spike_num + 1;
                    addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) + y; %�ô˴���ַ��1��ʼ�������������
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
    for n = 1 : CIFAR10_TEST_IMG
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE));  %�洢ÿ��ͼ���������ĵ�ַ
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE));     %�洢ÿ��ͼ����������ʱ��
        spike_num = 0;   %ͳ��ÿ��ͼ���������
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE)
                if (test_imgs_2d(n, x, y) ~= 0) %����0����(������)��
                    spike_num = spike_num + 1;
                    addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) + y; %�ô˴���ַ��1��ʼ�������������
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
    train_coding = zeros(1, (CIFAR10_TRAIN_IMG * TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * 2 * SPIKE_PER_PIXEL)); %��������Ԥ�ȷ����ڴ棬�Լӿ������ٶ�
    test_coding = zeros(1, (CIFAR10_TEST_IMG * TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * 2 * SPIKE_PER_PIXEL)); %��������Ԥ�ȷ����ڴ棬�Լӿ������ٶ�
    %����ѵ����ͼ��
    train_cnt = 0;
    for n = 1 : CIFAR10_TRAIN_IMG
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * SPIKE_PER_PIXEL);  %�洢ÿ��ͼ���������ĵ�ַ
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * SPIKE_PER_PIXEL);     %�洢ÿ��ͼ����������ʱ��
        spike_num = 0;   %ͳ��ÿ��ͼ���������
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE)
                pixel = TriOp(IF_RESIZE, train_resize_imgs_2d(n, x, y), train_imgs_2d(n, x, y));
                if (pixel ~= 0)  %����0����(������)��
                    poiss_tran = poissrnd(K2 * pixel, 1, SPIKE_PER_PIXEL);  %������K2*����ֵΪlmd�����Ĳ�����������
                    poiss_tran(poiss_tran <= 0) = 1; poiss_tran(poiss_tran >= TIME_WINDOW2 - 1) = TIME_WINDOW2 - 1; %�ض�
                    for i = 1 : SPIKE_PER_PIXEL
                        spike_num = spike_num + 1;
                        addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) + y; %�ô˴���ַ��1��ʼ�������������
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
    for n = 1 : CIFAR10_TEST_IMG
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * SPIKE_PER_PIXEL);  %�洢ÿ��ͼ���������ĵ�ַ
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) * SPIKE_PER_PIXEL);     %�洢ÿ��ͼ����������ʱ��
        spike_num = 0;   %ͳ��ÿ��ͼ���������
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE)
                pixel = TriOp(IF_RESIZE, test_resize_imgs_2d(n, x, y), test_imgs_2d(n, x, y));
                if (pixel ~= 0)  %����0����(������)��
                    poiss_tran = poissrnd(K2 * pixel, 1, SPIKE_PER_PIXEL);  %������K2*����ֵΪlmd�����Ĳ�����������
                    poiss_tran(poiss_tran <= 0) = 1; poiss_tran(poiss_tran >= TIME_WINDOW2 - 1) = TIME_WINDOW2 - 1; %�ض�
                    for i = 1 : SPIKE_PER_PIXEL
                        spike_num = spike_num + 1;
                        addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, CIFAR10_SIZE) + y; %�ô˴���ַ��1��ʼ�������������
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
