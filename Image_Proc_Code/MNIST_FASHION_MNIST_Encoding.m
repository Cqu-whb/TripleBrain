% -----------------------------------------------------------------------------------------------------------------------------------------------------%
% @��д��wtx
% @�������ڣ�2022/1/1
% @����ʱ�䣺Temporal_Coding 5min����, Rate_Codingʱ��ϳ�, �����ò���������һ����1h����
% @��������ȡMNIST��FAHION_MNIST���ݼ���mat��ʽ������������룬֧��Temporal_Coding��Rate_Coding������ߴ�Resize���Լ���˹����˲�����Dog��
% @�������ݸ�ʽ��eg.ѵ����ͼ��ͱ�ǩ��[60000, 784], [60000, 1]�ľ�����mat��ʽ�洢
% @�����ʽ��1��n������[addr1 t1 addr2 t2 ... -1 addr1 t1 addr2 t2 ... -1 ...], ��ַaddr��Χ��[1, img_size^2], ʱ���t��Χ��[1, TIME_WINDOW - 1],
%           ʱ��t����������ÿ��ͼ��֮����-1�ָ�
% @��ע1��ע��matlab����ĵ�ַ�Ǵ�1��ʼ�ģ�C#�����Դ���ʱ��Ҫ����ַ��1
% @��ע2��Dog����ʱȡ����Ӧλ�ô���ע�ͿɶԱ�ԭʼͼ���Dog���ͼ��
% @��ע3��Temporal_Coding��Rate_Coding�кܶ���ʵ�ַ�ʽ����������ͬ���˴����ṩһ�ַ�����Ϊ�ο�
% @��ע4���������Ŀ¼����������ֶ��ڵ����д������������ڴ������޸�Ŀ¼�����úò����Զ��޸� eg.��Temporal_CodingĿ¼�´�����Ϊ28x28_Dog���ļ���
% @��ע5��ʹ��imshow��ʾͼ��ᷢ��ͼ����ʱ����ת��90�㣬��ͼ�����ת��Ӱ�����Ľ��
% -----------------------------------------------------------------------------------------------------------------------------------------------------%
clear;clc;
load('.\Definition_pkg');
%----------------------------------------------------------------------����������-------------------------------------------------------------------%
%���ݼ���Ԥ�����������뷽��ѡ��
DATASET = 'MNIST';                  % ���ݼ�����ѡ'MNIST'��'FASHION_MNIST'
PRE_PROC = 'Gray';                   % Ԥ����������ѡ��'Gray'��'Dog'
CODING_METHOD = 'Temporal_Coding';  % ���뷽������ѡ��Temporal_Coding'��'Rate_Coding'��'MultiSpike_Temporal_Coding'

%Resize����
IF_RESIZE = 1;                      % �Ƿ���resize����1--True, 0--False
RESIZE_SIZE = 16;                   % resize���ͼ���С

%��˹����˲���Dog������
SIGMA1 = 1; SIGMA2 = 5; WINDOW = 15;

%Temporal_Coding����, t = TIME_WINDOW1 - floor(K1 * pixel)
TIME_WINDOW1 = 256;                   %ʱ�䴰��
K1 = TIME_WINDOW1 / MAX_PIXEL_VALUE;  %����ϵ��

%Rate_Coding����, ��K2*����ֵΪlmd�������ɲ�������
TIME_WINDOW2 = 256;                   %ʱ�䴰��
SPIKE_PER_PIXEL = 3;                  %ÿ������ֵ���ɵĲ����������
K2 = TIME_WINDOW2 / MAX_PIXEL_VALUE;  %����ϵ��

%MultiSpike_Temporal_Coding����
TIME_WINDOW3 = 256;
BETA = 8e-5;  %T=256, beta=8e-5ʱ����255�ɷ���5�����壬����50�ɷ���һ������
VTH = 1;
%----------------------------------------------------------------------����������-------------------------------------------------------------------%

%��������
load(strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_Denoising_mat\train_images'));
load(strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_Denoising_mat\test_images'));

%����󱣴�Ŀ¼����һ��������Լ����󴴽���eg.��Temporal_CodingĿ¼�´�����Ϊ28x28_Dog���ļ��У�
folder_name = strcat(num2str(TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)), 'x', num2str(TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)), '_', PRE_PROC);
TRAIN_ENCODING_SAVE_DIR = strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Encoding\', CODING_METHOD, '\', folder_name, '\');
TEST_ENCODING_SAVE_DIR = strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Encoding\', CODING_METHOD, '\', folder_name, '\');

%Ϊ������������Ƚ�ԭʼͼ������תΪ2ά
train_imgs_2d = zeros(MNIST_TRAIN_IMG, MNIST_SIZE, MNIST_SIZE);
test_imgs_2d = zeros(MNIST_TEST_IMG, MNIST_SIZE, MNIST_SIZE);
for n = 1 : MNIST_TRAIN_IMG
    train_imgs_2d(n, :, :) = reshape(train_imgs(n, :), MNIST_SIZE, MNIST_SIZE);
end
for n = 1 : MNIST_TEST_IMG
    test_imgs_2d(n, :, :) = reshape(test_imgs(n, :), MNIST_SIZE, MNIST_SIZE);
end

%��resize����dog
train_resize_imgs_2d = zeros(MNIST_TRAIN_IMG, TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE), TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE));
test_resize_imgs_2d = zeros(MNIST_TEST_IMG, TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE), TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE));
if (IF_RESIZE == 1) 
    %reshapeÿ��ѵ����ͼ��
    for n = 1 : MNIST_TRAIN_IMG
        temp_img_2d = squeeze(train_imgs_2d(n, :, :));  % ɾ��Ϊ1��ά��
        resize_img_2d = floor(imresize(temp_img_2d, [RESIZE_SIZE, RESIZE_SIZE], 'bilinear'));  %˫���Բ�ֵ�������ڽ���ֵЧ���ã�
        train_resize_imgs_2d(n, :, :) = resize_img_2d;
    end
     %reshapeÿ�Ų���ͼ��
    for n = 1 : MNIST_TEST_IMG
        temp_img_2d = squeeze(test_imgs_2d(n, :, :));   % ɾ��Ϊ1��ά��
        resize_img_2d = floor(imresize(temp_img_2d, [RESIZE_SIZE, RESIZE_SIZE], 'bilinear'));  %˫���Բ�ֵ�������ڽ���ֵЧ���ã�
        test_resize_imgs_2d(n, :, :) = resize_img_2d;
    end
end

%�ж��Ƿ��˹�˲�
%type= 'gaussian'��Ϊ��˹��ͨ�˲�����ģ����������sigma��ʾ�˲����ı�׼���λΪ���أ�Ĭ��ֵΪ 0.5����window��ʾģ��ߴ磬Ĭ��ֵΪ[3,3]
if (strcmp(PRE_PROC, 'Dog'))
    H1 = fspecial('gaussian', WINDOW, SIGMA1);
    H2 = fspecial('gaussian', WINDOW, SIGMA2);
    DiffGauss = H1 - H2;
    %�˲�ÿ��ѵ��ͼ��
    for n = 1 : MNIST_TRAIN_IMG
         dog_img = abs(imfilter(TriOp(IF_RESIZE, squeeze(train_resize_imgs_2d(n, :, :)), squeeze(train_imgs_2d(n, :, :))), DiffGauss, 'replicate'));   %����������������άͼ������˲�
         dog_img = mat2gray(dog_img);      % ��ͼ������һ��0��1��Χ��(����0��1)
         dog_img = floor(dog_img * 255);   % �ٳ�255��������Ҷ�ֵ         
         %���Dog���ͼ���Ƿ����
%          subplot(1, 2, 1); imshow(TriOp(IF_RESIZE, squeeze(train_resize_imgs_2d(n, :, :)), squeeze(train_imgs_2d(n, :, :)))/255); title('Ori');
%          subplot(1, 2, 2); imshow(dog_img/255); title('Dog');
%          debug_temp = 0;   %�˴����öϵ�Ա�Dogǰ��ͼ��     
         if (IF_RESIZE)
             train_resize_imgs_2d(n, :, :) = dog_img;
         else
             train_imgs_2d(n, :, :) = dog_img;
         end       
    end
    %�˲�ÿ�Ų���ͼ��
    for n = 1 : MNIST_TEST_IMG
         dog_img = abs(imfilter(TriOp(IF_RESIZE, squeeze(test_resize_imgs_2d(n, :, :)), squeeze(test_imgs_2d(n, :, :))), DiffGauss, 'replicate'));   %����������������άͼ������˲�
         dog_img = mat2gray(dog_img);      % ��ͼ������һ��0��1��Χ��(����0��1)
         dog_img = floor(dog_img * 255);   % �ٳ�255��������Ҷ�ֵ
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
    train_coding = zeros(1, (MNIST_TRAIN_IMG * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * 2)); %��������Ԥ�ȷ����ڴ棬�Լӿ������ٶ�
    test_coding = zeros(1, (MNIST_TEST_IMG * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * 2)); %��������Ԥ�ȷ����ڴ棬�Լӿ������ٶ�
    %����ѵ����ͼ��
    train_cnt = 0;
    for n = 1 : MNIST_TRAIN_IMG
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE));  %�洢ÿ��ͼ���������ĵ�ַ
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE));     %�洢ÿ��ͼ����������ʱ��
        spike_num = 0;   %ͳ��ÿ��ͼ���������
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
                if (TriOp(IF_RESIZE, train_resize_imgs_2d(n, x, y), train_imgs_2d(n, x, y)) ~= 0) %����0����(������)��
                    spike_num = spike_num + 1;
                    addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) + y; %�ô˴���ַ��1��ʼ�������������
                    t(spike_num) = TIME_WINDOW1 - floor(K1 * TriOp(IF_RESIZE, train_resize_imgs_2d(n, x, y), train_imgs_2d(n, x, y)));                   
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
    for n = 1 : MNIST_TEST_IMG
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE));  %�洢ÿ��ͼ���������ĵ�ַ
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE));     %�洢ÿ��ͼ����������ʱ��
        spike_num = 0;   %ͳ��ÿ��ͼ���������
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
                if (TriOp(IF_RESIZE, test_resize_imgs_2d(n, x, y), test_imgs_2d(n, x, y)) ~= 0) %����0����(������)��
                    spike_num = spike_num + 1;
                    addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) + y; %�ô˴���ַ��1��ʼ�������������
                    t(spike_num) = TIME_WINDOW1 - floor(K1 * TriOp(IF_RESIZE, test_resize_imgs_2d(n, x, y), test_imgs_2d(n, x, y)));                   
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
    %ȥ�������ж����0
    train_coding(train_coding == 0) = []; 
    test_coding(test_coding == 0) = [];
%-------------------------------Rate_Coding--------------------------------%
elseif(strcmp(CODING_METHOD, 'Rate_Coding'))
    train_coding = zeros(1, (MNIST_TRAIN_IMG * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * 2 * SPIKE_PER_PIXEL)); %��������Ԥ�ȷ����ڴ棬�Լӿ������ٶ�
    test_coding = zeros(1, (MNIST_TEST_IMG * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * 2 * SPIKE_PER_PIXEL)); %��������Ԥ�ȷ����ڴ棬�Լӿ������ٶ�
    %����ѵ����ͼ��
    train_cnt = 0;
    for n = 1 : MNIST_TRAIN_IMG
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * SPIKE_PER_PIXEL);  %�洢ÿ��ͼ���������ĵ�ַ
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * SPIKE_PER_PIXEL);     %�洢ÿ��ͼ����������ʱ��
        spike_num = 0;   %ͳ��ÿ��ͼ���������
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
                pixel = TriOp(IF_RESIZE, train_resize_imgs_2d(n, x, y), train_imgs_2d(n, x, y));
                if (pixel ~= 0)  %����0����(������)��
                    poiss_tran = poissrnd(K2 * pixel, 1, SPIKE_PER_PIXEL);  %������K2*����ֵΪlmd�����Ĳ�����������
                    poiss_tran(poiss_tran <= 0) = 1; poiss_tran(poiss_tran >= TIME_WINDOW2 - 1) = TIME_WINDOW2 - 1; %�ض�
                    for i = 1 : SPIKE_PER_PIXEL
                        spike_num = spike_num + 1;
                        addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) + y; %�ô˴���ַ��1��ʼ�������������
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
    for n = 1 : MNIST_TEST_IMG
        addr = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * SPIKE_PER_PIXEL);  %�洢ÿ��ͼ���������ĵ�ַ
        t = zeros(1,TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) * SPIKE_PER_PIXEL);     %�洢ÿ��ͼ����������ʱ��
        spike_num = 0;   %ͳ��ÿ��ͼ���������
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
                pixel = TriOp(IF_RESIZE, test_resize_imgs_2d(n, x, y), test_imgs_2d(n, x, y));
                if (pixel ~= 0)  %����0����(������)��
                    poiss_tran = poissrnd(K2 * pixel, 1, SPIKE_PER_PIXEL);  %������K2*����ֵΪlmd�����Ĳ�����������
                    poiss_tran(poiss_tran <= 0) = 1; poiss_tran(poiss_tran >= TIME_WINDOW2 - 1) = TIME_WINDOW2 - 1; %�ض�
                    for i = 1 : SPIKE_PER_PIXEL
                        spike_num = spike_num + 1;
                        addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) + y; %�ô˴���ַ��1��ʼ�������������
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
    %ȥ�������ж����0
    train_coding(train_coding == 0) = [];
    test_coding(test_coding == 0) = [];
%-------------------------------MultiSpike_Temporal_Coding--------------------------------%
elseif(strcmp(CODING_METHOD, 'MultiSpike_Temporal_Coding'))
    train_coding = zeros(1, 1); %ֱ����������Ҳ�����˶���
    test_coding = zeros(1, 1); 
    %����ѵ����ͼ��
    train_cnt = 0;
    for n = 1 : MNIST_TRAIN_IMG
        addr = zeros(1, 1);  %�洢ÿ��ͼ���������ĵ�ַ
        t = zeros(1, 1);     %�洢ÿ��ͼ����������ʱ��
        spike_num = 0;   %ͳ��ÿ��ͼ���������
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
                pixel = TriOp(IF_RESIZE, train_resize_imgs_2d(n, x, y), train_imgs_2d(n, x, y));
                if (pixel ~= 0)  %����0����(������)��
                    V = 0; t0 = 0;
                    for ti = 1 : TIME_WINDOW3
                        V = BETA * pixel * (ti - t0);
                        if V >= VTH
                            spike_num = spike_num + 1;
                            addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) + y; %�ô˴���ַ��1��ʼ�������������
                            t(spike_num) = ti;
                            t0 = t;
                        end
                    end
                end
            end
        end
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
    for n = 1 : MNIST_TEST_IMG
        addr = zeros(1, 1);  %�洢ÿ��ͼ���������ĵ�ַ
        t = zeros(1,1);     %�洢ÿ��ͼ����������ʱ��
        spike_num = 0;   %ͳ��ÿ��ͼ���������
        for x = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
            for y = 1 : TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE)
                pixel = TriOp(IF_RESIZE, test_resize_imgs_2d(n, x, y), test_imgs_2d(n, x, y));
                if (pixel ~= 0)  %����0����(������)��
                    V = 0; t0 = 0;
                    for ti = 1 : TIME_WINDOW3
                        V = BETA * pixel * (ti - t0);
                        if V >= VTH
                            spike_num = spike_num + 1;
                            addr(spike_num) = (x-1)*TriOp(IF_RESIZE, RESIZE_SIZE, MNIST_SIZE) + y; %�ô˴���ַ��1��ʼ�������������
                            t(spike_num) = ti;
                            t0 = t;
                        end
                    end
                end
            end
        end
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
end
%������������ݼ�
save([strcat(TRAIN_ENCODING_SAVE_DIR, 'train_coding'), '.mat'], 'train_coding');
save([strcat(TEST_ENCODING_SAVE_DIR, 'test_coding'), '.mat'], 'test_coding');
copyfile(strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_Denoising_mat\train_labels.mat'),TRAIN_ENCODING_SAVE_DIR); %������ǩ����ѵ������
copyfile(strcat('..\..\Common_Datasets\Frame_based\',DATASET,'\Original_Denoising_mat\test_labels.mat'),TEST_ENCODING_SAVE_DIR);   %������ǩ����ѵ������
