% -----------------------------------------------------------------------------------------------------------------------------------------------------%
% @��д��wtx
% @�������ڣ�2022/1/1
% @����ʱ�䣺6000/1000��ͼ350s����
% @����������Ӳ�ֶεķ�ʽ����DVS���ݼ������¼������ֶΣ�ʱ��ѹ����ʽ֧�֡���ѹ��(None)������������(Compact)���͡�����(Linear)�����ַ�ʽ���ɸ���ʵ�����ѡ��
% @�������ݸ�ʽ��ת�����mat�ļ��а���60000��ѵ��cell��10000������cell��ÿ��cell��ʾһ��ͼ������壬���1��4��5�зֱ��ʾʱ�䡢���غ����ꡢ���������꣬ÿ��ͼԼ300ms
% @�����ʽ��1��n������[addr1 t1 addr2 t2 ... -1 addr1 t1 addr2 t2 ... -1 ...], ��ַaddr��Χ��[1, img_size^2], ʱ���t��Χ��[0, ...]
% @��ע1����DVS���ݼ���һ�Σ�Segment���൱��һ��ͼ���ֶη���������ֶκ�Ӳ�ֶΣ����ű����ý�Ϊ�򵥵Ļ����¼�����Ӳ�ֶη�ʽ������һ�ε������ӵ���
% @��ע2��ʱ��ѹ����ʽ����ѹ��(None)��ָ�Ͱ�ԭʼ���ݼ�ʱ���������λus
% @��ע3��ʱ��ѹ����ʽ��������(Compact)��ָһ���¼���Ӧһ��ʱ�䲽�������巢��ʱ����Ϊ1��ÿ��ʱ�䲽�������壬��λ���Լ�����
% @��ע4��ʱ��ѹ����ʽ������(Linear)��ָ������us���������һ���䣨��ѹ���߶�Ϊ��λ����ȡ��������ͬλ�����ط����εİ�һ���㣬��λ��ѹ���߶ȼ���
% @��ע5��ԭʼ�����ǰ���ǩ�������еģ�������ȫ�����Һ��ٽ��зֶ�
% @��ע6���ֶκ�ѵ�����Ͳ��Լ���Segment�����ֱ���txt��ʽ��������ͬĿ¼��
% @��ע7������DVS���ݼ���Resize��Ϊ�ټ��Ҹ��ӣ��ð汾�ű���֧�ֲ�Resize��Resize��16x16�������ߴ�Resize�ɲο�����������޸�
% -----------------------------------------------------------------------------------------------------------------------------------------------------%
clear;clc;
load('.\Definition_pkg');
load('..\..\Common_Datasets\DVS\N_MNIST\Original_mat\N_MNIST_DVS_34x34_60000');
load('..\..\Common_Datasets\DVS\N_MNIST\Original_mat\N_MNIST_DVS_34x34_10000');
load('..\..\Common_Datasets\DVS\N_MNIST\Original_mat\train_labels');
load('..\..\Common_Datasets\DVS\N_MNIST\Original_mat\test_labels');
LOCAL_N_MNIST_SIZE = 32;
%----------------------------------------------------------------------����������-------------------------------------------------------------------%
%ѵ����ռ��(N_MNIST_DVS����̫�࣬����ȡһ���ֳ���ѵ��/���Լ��ɣ�����ֵ��6000/1000)
TRAIN_CELL = 6000;
TEST_CELL = 1000;

%�ֶβ���
% NEVENT_PER_SEGMENT = 1000;  %ÿ��Segment���¼���
NEVENT_PER_SEGMENT = 1000;  %ÿ��Segment���¼���

%ʱ��ѹ��ģʽ
TIME_COMP_MODE = 'Linear';  %��ѡ��'None', 'Compact', 'Linear'

%����ѹ���߶ȣ���λ��us��
LINEAR_COMP_SCALE = 500; 

%�Ƿ�Resize��ע������DVS���ݼ���Resize��Ϊ�ټ��Ҹ��ӣ��ð汾�ű���֧�ֲ�Resize��Resize��16x16��
%ע����34x34 Resize��16x16,Ϊ���������ͼ��ü�Ϊ32x32������������ΧһȦ���أ�Ȼ��ԭ(x, y)����-1���ɣ�����Resize������֮ǰһ��
IF_RESIZE = 0;   % �Ƿ�ResizeΪ16x16��1--True, 0--False
%----------------------------------------------------------------------����������-------------------------------------------------------------------%
%ѹ���󱣴�Ŀ¼����DVS���ݼ���ѹ��Ŀ¼����ǰ�����ã�
folder_name = strcat('Comp_', TIME_COMP_MODE);
SAVE_DIR = strcat('..\..\Common_Datasets\DVS\N_MNIST\', folder_name, '\');

train_coding = zeros(1, 1); %��matlab��������������������������չ��ʵ��������ٶ�Ӱ�첻��
test_coding = zeros(1, 1);
train_labels_rand = zeros(1, 1);
test_labels_rand = zeros(1, 1);

train_segment_num = 0;
test_segment_num = 0;

%����ѵ���Ͳ�������
rand_num_train = randperm(N_MNIST_TRAIN_NCELL, N_MNIST_TRAIN_NCELL); 
rand_num_test = randperm(N_MNIST_TEST_NCELL, N_MNIST_TEST_NCELL); 
train_cells_rand = cell(1, N_MNIST_TRAIN_NCELL); %���Һ��ѵ��cell
test_cells_rand = cell(1, N_MNIST_TEST_NCELL);   %���Һ�Ĳ���cell
for i = 1 : (N_MNIST_TRAIN_NCELL)
    train_cells_rand{i} = train_cells{rand_num_train(i)};
end
for i = 1 : (N_MNIST_TEST_NCELL)
    test_cells_rand{i} = test_cells{rand_num_test(i)};
end

%�ֶ�ѵ�������ݣ�������ñ�ǩ
cnt = 0;
for i = 1 : TRAIN_CELL
    img_spike_num = length(train_cells_rand{i}(:, 1));           %����ͼ����������
    img_segment_num = floor(img_spike_num / NEVENT_PER_SEGMENT); %����ͼ��ֶκ����
    t_pre = train_cells_rand{i}(1, 1);  %ȷ��ÿ��ͼ����0ʱ�̿�ʼ
    t_compact = 0;
    t_linear_pre = floor(train_cells_rand{i}(1, 1) / LINEAR_COMP_SCALE);
    t_linear_addr_temp = [];
    for j = 1 : img_segment_num
        for k = 1 : NEVENT_PER_SEGMENT
            x =  train_cells_rand{i}( (j-1) * NEVENT_PER_SEGMENT + k, 4);
            y =  train_cells_rand{i}( (j-1) * NEVENT_PER_SEGMENT + k, 5);
            t =  train_cells_rand{i}( (j-1) * NEVENT_PER_SEGMENT + k, 1);
            t_linear = floor(t / LINEAR_COMP_SCALE);
            t_linear_post = TriOp( ((j-1) * NEVENT_PER_SEGMENT + k) < img_spike_num, ...
                                   floor( train_cells_rand{i}( (j-1) * NEVENT_PER_SEGMENT + k + 1 ) / LINEAR_COMP_SCALE), ...
                                   -1); %��һ���¼���ʱ��������ѵ����һ���¼�����-1��ʾ
            %��ѹ��(None)
            if (strcmp(TIME_COMP_MODE, 'None') && (x > 1 && x < LOCAL_N_MNIST_SIZE) && (y > 1 && y < LOCAL_N_MNIST_SIZE))
                cnt = cnt + 1;
                %train_coding(cnt) = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1) * LOCAL_N_MNIST_SIZE + y); %��ַ
                train_coding(cnt) = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1-1) * LOCAL_N_MNIST_SIZE + y-1); %��ַ
                cnt = cnt + 1;
                train_coding(cnt) = t - t_pre;  %ʱ�����ȷ��ÿ�ζ���0��ʼ
            end
            %������(Compact)
            if (strcmp(TIME_COMP_MODE, 'Compact') && (x > 1 && x < LOCAL_N_MNIST_SIZE) && (y > 1 && y < LOCAL_N_MNIST_SIZE))
                cnt = cnt + 1;
                %train_coding(cnt) = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1) * LOCAL_N_MNIST_SIZE + y); %��ַ
                train_coding(cnt) = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1-1) * LOCAL_N_MNIST_SIZE + y-1); %��ַ
                cnt = cnt + 1;
                train_coding(cnt) = t_compact;  %ѹ��ʱ�����ÿ�ζ���0��ʼ��ÿ���¼�+1
                t_compact = t_compact + 1;
            end
            %����(Linear)
            if (strcmp(TIME_COMP_MODE, 'Linear') && (x > 1 && x < LOCAL_N_MNIST_SIZE) && (y > 1 && y < LOCAL_N_MNIST_SIZE))
                %addr = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1) * LOCAL_N_MNIST_SIZE + y);
                addr = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1-1) * LOCAL_N_MNIST_SIZE + y-1);
                t_linear_addr_temp = [t_linear_addr_temp addr];  %����Ԫ�ظ�������1000ʱ������������ַ�ʽ���Ԫ�أ�Ӧ��ǰ�����ڴ�
                if (t_linear_post ~=  t_linear) %��ǰʱ�䲽�¼���ַ��������
                    t_linear_addr_temp = unique(t_linear_addr_temp);  %ȥ����ͬ�ĵ�ַ
                    len = length(t_linear_addr_temp); %��ȡ�¼�����
                    for n = 1 : len
                         cnt = cnt + 1;
                         train_coding(cnt) = t_linear_addr_temp(n);
                         cnt = cnt + 1;
                         train_coding(cnt) = t_linear - t_linear_pre;
                    end
                    t_linear_pretimestep = t_linear;  %����t_linear_pre
                    t_linear_addr_temp = [];
                end
            end
        end
        cnt = cnt + 1;
        train_coding(cnt) = -1;  %ÿ����-1��β
        t_pre = t;
        t_linear_pre = t_linear;
        t_compact = 0;       
        train_segment_num = train_segment_num + 1;
        train_labels_rand(train_segment_num, 1) = train_labels(rand_num_train(i));
    end
end

%�ֶβ��Լ����ݣ�������ñ�ǩ
cnt = 0;
for i = 1 : TEST_CELL
    img_spike_num = length(test_cells_rand{i}(:, 1));            %����ͼ����������
    img_segment_num = floor(img_spike_num / NEVENT_PER_SEGMENT); %����ͼ��ֶκ����
    t_pre = test_cells_rand{i}(1, 1);  %ȷ��ÿ��ͼ����0ʱ�̿�ʼ
    t_compact = 0;
    t_linear_pre = floor(test_cells_rand{i}(1, 1) / LINEAR_COMP_SCALE);
    t_linear_addr_temp = [];
    for j = 1 : img_segment_num
        for k = 1 : NEVENT_PER_SEGMENT
            x =  test_cells_rand{i}( (j-1) * NEVENT_PER_SEGMENT + k, 4);
            y =  test_cells_rand{i}( (j-1) * NEVENT_PER_SEGMENT + k, 5);
            t =  test_cells_rand{i}( (j-1) * NEVENT_PER_SEGMENT + k, 1);
            t_linear = floor(t / LINEAR_COMP_SCALE);
            t_linear_post = TriOp( ((j-1) * NEVENT_PER_SEGMENT + k) < img_spike_num, ...
                                   floor( test_cells_rand{i}( (j-1) * NEVENT_PER_SEGMENT + k + 1 ) / LINEAR_COMP_SCALE), ...
                                   -1); %��һ���¼���ʱ��������ѵ����һ���¼�����-1��ʾ
            %��ѹ��(None)
            if (strcmp(TIME_COMP_MODE, 'None') && (x > 1 && x < LOCAL_N_MNIST_SIZE) && (y > 1 && y < LOCAL_N_MNIST_SIZE))
                cnt = cnt + 1;
                %test_coding(cnt) = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1) * LOCAL_N_MNIST_SIZE + y); %��ַ
                test_coding(cnt) = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1 - 1) * LOCAL_N_MNIST_SIZE + y-1); %��ַ
                cnt = cnt + 1;
                test_coding(cnt) = t - t_pre;  %ʱ�����ȷ��ÿ�ζ���0��ʼ
            end
            %������(Compact)
            if (strcmp(TIME_COMP_MODE, 'Compact') && (x > 1 && x < LOCAL_N_MNIST_SIZE) && (y > 1 && y < LOCAL_N_MNIST_SIZE))
                cnt = cnt + 1;
                %test_coding(cnt) = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1) * LOCAL_N_MNIST_SIZE + y); %��ַ
                test_coding(cnt) = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1-1) * LOCAL_N_MNIST_SIZE + y-1); %��ַ
                cnt = cnt + 1;
                test_coding(cnt) = t_compact;  %ѹ��ʱ�����ÿ�ζ���0��ʼ��ÿ���¼�+1
                t_compact = t_compact + 1;
            end
            %����(Linear)
            if (strcmp(TIME_COMP_MODE, 'Linear') && (x > 1 && x < LOCAL_N_MNIST_SIZE) && (y > 1 && y < LOCAL_N_MNIST_SIZE))
%                 addr = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1) * LOCAL_N_MNIST_SIZE + y);
                addr = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1-1) * LOCAL_N_MNIST_SIZE + y-1);
                t_linear_addr_temp = [t_linear_addr_temp addr];  %����Ԫ�ظ�������1000ʱ������������ַ�ʽ���Ԫ�أ�Ӧ��ǰ�����ڴ�
                if (t_linear_post ~=  t_linear) %��ǰʱ�䲽�¼���ַ��������
                    t_linear_addr_temp = unique(t_linear_addr_temp);  %ȥ����ͬ�ĵ�ַ
                    len = length(t_linear_addr_temp); %��ȡ�¼�����
                    for n = 1 : len
                         cnt = cnt + 1;
                         test_coding(cnt) = t_linear_addr_temp(n);
                         cnt = cnt + 1;
                         test_coding(cnt) = t_linear - t_linear_pre;
                    end
                    t_linear_pretimestep = t_linear;  %����t_linear_pre
                    t_linear_addr_temp = [];
                end
            end
        end
        cnt = cnt + 1;
        test_coding(cnt) = -1;  %ÿ����-1��β
        t_pre = t;
        t_linear_pre = t_linear;
        t_compact = 0;       
        test_segment_num = test_segment_num + 1;
        test_labels_rand(test_segment_num, 1) = test_labels(rand_num_test(i));
    end
end
%��mat��ʽ����
save([strcat(SAVE_DIR, 'train_coding'), '.mat'], 'train_coding');
save([strcat(SAVE_DIR, 'test_coding'), '.mat'], 'test_coding');
save([strcat(SAVE_DIR, 'train_labels'), '.mat'], 'train_labels_rand');
save([strcat(SAVE_DIR, 'test_labels'), '.mat'], 'test_labels_rand');
fprintf(fopen(strcat(SAVE_DIR, 'train_segment_num.txt'),'wt'), '%d\t', train_segment_num);
fprintf(fopen(strcat(SAVE_DIR, 'test_segment_num.txt'),'wt'), '%d\t', test_segment_num);


