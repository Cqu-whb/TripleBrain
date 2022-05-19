% -----------------------------------------------------------------------------------------------------------------------------------------------------%
% @��д��wtx
% @�������ڣ�2022/1/1
% @����ʱ�䣺300s
% @����������ֱ�����ص�N_MNIST���ݼ�ÿ��ͼ�����˸�������bin�ļ����棬�����ļ����Ҳ��׶�ȡ�����ű��Ƚ�ѵ�����Ͳ��Լ�ȫת��������mat�ļ��У���ʽ������DVS���ݼ�һ��
% @ԭʼ���ݸ�ʽ��ÿ��ͼ����һ�������Ķ������ļ����ļ�����������ԭʼMNIST���ݼ��еı�ţ������¼��б���ɣ�ÿ���¼�ռ��40λ����39-32λ��x��ַ����31-24λ��y��ַ
%               ��23λ�����ԣ���22-0λ��ʱ�������usΪ��λ��     
% @ԭʼ��ǩ��ʽ��0-9��Ŀ¼�·ֱ���0-9��ǩ������
% @�����ʽ��train mat�а���60000��cell��test mat�а���10000��cell��ÿ��cell�ĵ�1��4��5�зֱ��ʾʱ�䡢���غ����ꡢ���������꣬�����ж����0
% @��ע1��ת���귢���ļ���Ȼ������һ��...�����뻹�Ǽ��˶�ȡ������ͳһ�˸�ʽ��
% @��ע2��N_MNISTͼ��ֱ��ʾ�Ȼ��34x34�ģ�ԭ�ļ�Ҳ��˵��fo��
% -----------------------------------------------------------------------------------------------------------------------------------------------------%
clear;clc;
load('.\Definition_pkg');

%ԭʼ����Ŀ¼
TRAIN_IMG_DIR = '..\..\Common_Datasets\DVS\N_MNIST\Original_files\Train\';
TEST_IMG_DIR = '..\..\Common_Datasets\DVS\N_MNIST\Original_files\Test\';

%ת��Ϊmat�󱣴�Ŀ¼
SAVE_DIR = '..\..\Common_Datasets\DVS\N_MNIST\Original_mat\';

train_cells = cell(1, MNIST_TRAIN_IMG);
test_cells = cell(1, MNIST_TEST_IMG);
train_labels = zeros(MNIST_TRAIN_IMG, 1);
test_labels = zeros(MNIST_TEST_IMG, 1);

%����ÿ��ѵ��ͼ�����Ͻ�Cell�У����ֶ����ɱ�ǩ
img_cnt = 0;
for i = 1 : MNIST_NCLASS %10
        img_list = dir(strcat(TRAIN_IMG_DIR, num2str(i-1), '\', '*.bin')); %��ȡ��Ŀ¼������bin��ʽͼ����Ϣ
        img_num = length(img_list);  %��Ŀ¼��ͼ������
        for n = 1 : img_num
            img_cnt = img_cnt + 1;
            img_name = img_list(n).name;
            TD = Read_N_MNIST(strcat(TRAIN_IMG_DIR, num2str(i-1), '\', img_name)); %��ȡÿ��ͼ����¼���
            train_cells{img_cnt}(:, 1) = TD.ts(:);
            train_cells{img_cnt}(:, 4) = TD.x(:);
            train_cells{img_cnt}(:, 5) = TD.y(:);
            train_labels(img_cnt, 1) = i - 1; %0-9
        end
end
%����Ϊmat��ʽ
save([strcat(SAVE_DIR, 'N_MNIST_DVS_34x34_60000'), '.mat'], 'train_cells', '-v7.3');
save([strcat(SAVE_DIR, 'train_labels'), '.mat'], 'train_labels');
clear train_cells;  %��ֹ�ڴ治��

%����ÿ�Ų���ͼ�����Ͻ�Cell�У����ֶ����ɱ�ǩ
img_cnt = 0;
for i = 1 : MNIST_NCLASS %10
        img_list = dir(strcat(TEST_IMG_DIR, num2str(i-1), '\', '*.bin')); %��ȡ��Ŀ¼������bin��ʽͼ����Ϣ
        img_num = length(img_list);  %��Ŀ¼��ͼ������
        for n = 1 : img_num
            img_cnt = img_cnt + 1;
            img_name = img_list(n).name;
            TD = Read_N_MNIST(strcat(TEST_IMG_DIR, num2str(i-1), '\', img_name)); %��ȡÿ��ͼ����¼���
            test_cells{img_cnt}(:, 1) = TD.ts(:);
            test_cells{img_cnt}(:, 4) = TD.x(:);
            test_cells{img_cnt}(:, 5) = TD.y(:);
            test_labels(img_cnt, 1) = i - 1; %0-9
        end
end
%����Ϊmat��ʽ
save([strcat(SAVE_DIR, 'N_MNIST_DVS_34x34_10000'), '.mat'], 'test_cells');
save([strcat(SAVE_DIR, 'test_labels'), '.mat'], 'test_labels'); 
