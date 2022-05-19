% -----------------------------------------------------------------------------------------------------------------------------------------------------%
% @��д��wtx
% @�������ڣ�2022/1/1
% @����ʱ�䣺200s����
% @����������Ӳ�ֶεķ�ʽ����DVS���ݼ������¼������ֶΣ�ʱ��ѹ����ʽ֧�֡���ѹ��(None)������������(Compact)���͡�����(Linear)�����ַ�ʽ���ɸ���ʵ�����ѡ��
% @�������ݸ�ʽ��ԭʼmat�ļ��а���100��cell��ÿ��cell��ʾһ��ͼ������壬���1��4��5�зֱ��ʾʱ�䡢���غ����ꡢ���������꣬ÿ��ͼԼ2s��4s,��ǩ��0-2
% @�����ʽ��1��n������[addr1 t1 addr2 t2 ... -1 addr1 t1 addr2 t2 ... -1 ...], ��ַaddr��Χ��[1, img_size^2], ʱ���t��Χ��[0, ...]����ǩ0-2
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
load('..\..\Common_Datasets\DVS\POSTURE_DVS\Original_mat\posture_2timesIn1CIN_32x32');
%----------------------------------------------------------------------����������-------------------------------------------------------------------%
%ѵ����ռ��
TRAIN_RATIO = 0.8; 

%�ֶβ���
NEVENT_PER_SEGMENT = 500;  %ÿ��Segment���¼���

%ʱ��ѹ��ģʽ
TIME_COMP_MODE = 'Linear';  %��ѡ��'None', 'Compact', 'Linear'

%����ѹ���߶ȣ���λ��us��
LINEAR_COMP_SCALE = 1000; 

%�Ƿ�Resize��ע������DVS���ݼ���Resize��Ϊ�ټ��Ҹ��ӣ��ð汾�ű���֧�ֲ�Resize��Resize��16x16��
%ע1����32x32Resize��16x16���ǰ���ʱ�䲽�����ν�ͼ��maxpooling��16x16�����ڡ���ѹ��(None)���͡�������(Compact)��������ÿ��ʱ�䲽��1���¼���
%     ��˲�����ĶԶ�ά������maxpooling��ֱ�Ӷ�����x, y����ӳ�伴�ɣ���16x16_x = ceil(32x32_x / 2)��16x16_y = ceil(32x32_y / 2)
%ע2�����ڡ�����(Linear)��ѹ����Ҳ�ݲ�������maxpooling�ķ���������ע������һ��ʱ�䲽����������壬����һ��ʱ�䲽�ڼ�������һ����16x16_addr������һ���������ձ�������
IF_RESIZE = 0;   % �Ƿ�ResizeΪ16x16��1--True, 0--False
%----------------------------------------------------------------------����������-------------------------------------------------------------------%

%ѹ���󱣴�Ŀ¼����DVS���ݼ���ѹ��Ŀ¼����ǰ�����ã�
folder_name = strcat('Comp_', TIME_COMP_MODE);
SAVE_DIR = strcat('..\..\Common_Datasets\DVS\POSTURE_DVS\', folder_name, '\');

train_coding = zeros(1, 1); %��matlab��������������������������չ��ʵ��������ٶ�Ӱ�첻��
test_coding = zeros(1, 1);
train_labels = zeros(1, 1);
test_labels = zeros(1, 1);

train_segment_num = 0;
test_segment_num = 0;

%����ԭʼ����
%����һ�д�1��484�������е�484����������484����������ͬ  ---->�Ķ� 1w - 484
rand_num = randperm(POSTURE_DVS_NCELL, POSTURE_DVS_NCELL);  
CINs_rand = cell(1, POSTURE_DVS_NCELL); %���Һ������
for i = 1 : (POSTURE_DVS_NCELL)
    CINs_rand{i} = CINs{rand_num(i)};
end

%�ֶ�ѵ�������ݣ�������ñ�ǩ
cnt = 0;
for i = 1 : round(POSTURE_DVS_NCELL * TRAIN_RATIO)
    img_spike_num = length(CINs_rand{i}(:, 1));                  %����ͼ����������
    img_segment_num = floor(img_spike_num / NEVENT_PER_SEGMENT); %����ͼ��ֶκ����
    t_pre = CINs_rand{i}(1, 1);  %ȷ��ÿ��ͼ����0ʱ�̿�ʼ
    t_compact = 0;
    t_linear_pre = floor(CINs_rand{i}(1, 1) / LINEAR_COMP_SCALE);
    t_linear_addr_temp = [];
    for j = 1 : img_segment_num
        for k = 1 : NEVENT_PER_SEGMENT
            x =  CINs_rand{i}( (j-1) * NEVENT_PER_SEGMENT + k, 4);
            y =  CINs_rand{i}( (j-1) * NEVENT_PER_SEGMENT + k, 5);
            t =  CINs_rand{i}( (j-1) * NEVENT_PER_SEGMENT + k, 1);
            t_linear = floor(t / LINEAR_COMP_SCALE);
            t_linear_post = TriOp( ((j-1) * NEVENT_PER_SEGMENT + k) < img_spike_num, ...
                                   floor( CINs_rand{i}( (j-1) * NEVENT_PER_SEGMENT + k + 1 ) / LINEAR_COMP_SCALE), ...
                                   -1); %��һ���¼���ʱ��������ѵ����һ���¼�����-1��ʾ
            %��ѹ��(None)
            if (strcmp(TIME_COMP_MODE, 'None'))
                cnt = cnt + 1;
                train_coding(cnt) = TriOp(IF_RESIZE, (ceil(x/2) - 1) * DVS_RESIZE_SIZE + ceil(y/2), (x - 1) * POSTURE_SIZE + y); %��ַ
                cnt = cnt + 1;
                train_coding(cnt) = t - t_pre;  %ʱ�����ȷ��ÿ�ζ���0��ʼ
            end
            %������(Compact)
            if (strcmp(TIME_COMP_MODE, 'Compact'))
                cnt = cnt + 1;
                train_coding(cnt) = TriOp(IF_RESIZE, (ceil(x/2) - 1) * DVS_RESIZE_SIZE + ceil(y/2), (x - 1) * POSTURE_SIZE + y); %��ַ
                cnt = cnt + 1;
                train_coding(cnt) = t_compact;  %ѹ��ʱ�����ÿ�ζ���0��ʼ��ÿ���¼�+1
                t_compact = t_compact + 1;
            end
            %����(Linear)
            if (strcmp(TIME_COMP_MODE, 'Linear'))
                addr = TriOp(IF_RESIZE, (ceil(x/2) - 1) * DVS_RESIZE_SIZE + ceil(y/2), (x - 1) * POSTURE_SIZE + y);
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
        train_labels(train_segment_num, 1) = Labels(rand_num(i));
    end
end

%�ֶβ��Լ����ݣ�������ñ�ǩ
cnt = 0;
for i = round(POSTURE_DVS_NCELL * TRAIN_RATIO) + 1 : POSTURE_DVS_NCELL
    img_spike_num = length(CINs_rand{i}(:, 1));                  %����ͼ����������
    img_segment_num = floor(img_spike_num / NEVENT_PER_SEGMENT); %����ͼ��ֶκ����
    t_pre = CINs_rand{i}(1, 1);  %ȷ��ÿ��ͼ����0ʱ�̿�ʼ
    t_compact = 0;
    t_linear_pre = floor(CINs_rand{i}(1, 1) / LINEAR_COMP_SCALE);
    t_linear_addr_temp = [];
    for j = 1 : img_segment_num
        for k = 1 : NEVENT_PER_SEGMENT
            x =  CINs_rand{i}( (j-1) * NEVENT_PER_SEGMENT + k, 4);
            y =  CINs_rand{i}( (j-1) * NEVENT_PER_SEGMENT + k, 5);
            t =  CINs_rand{i}( (j-1) * NEVENT_PER_SEGMENT + k, 1);
            t_linear = floor(t / LINEAR_COMP_SCALE);
            t_linear_post = TriOp( ((j-1) * NEVENT_PER_SEGMENT + k) < img_spike_num, ...
                                   floor( CINs_rand{i}( (j-1) * NEVENT_PER_SEGMENT + k + 1 ) / LINEAR_COMP_SCALE), ...
                                   -1); %��һ���¼���ʱ��������ѵ����һ���¼�����-1��ʾ
            %��ѹ��(None)
            if (strcmp(TIME_COMP_MODE, 'None'))
                cnt = cnt + 1;
                test_coding(cnt) = TriOp(IF_RESIZE, (ceil(x/2) - 1) * DVS_RESIZE_SIZE + ceil(y/2), (x - 1) * POSTURE_SIZE + y); %��ַ
                cnt = cnt + 1;
                test_coding(cnt) = t - t_pre;  %ʱ�����ȷ��ÿ�ζ���0��ʼ
            end
            %������(Compact)
            if (strcmp(TIME_COMP_MODE, 'Compact'))
                cnt = cnt + 1;
                test_coding(cnt) = TriOp(IF_RESIZE, (ceil(x/2) - 1) * DVS_RESIZE_SIZE + ceil(y/2), (x - 1) * POSTURE_SIZE + y); %��ַ
                cnt = cnt + 1;
                test_coding(cnt) = t_compact;  %ѹ��ʱ�����ÿ�ζ���0��ʼ��ÿ���¼�+1
                t_compact = t_compact + 1;
            end
            %����(Linear)
            if (strcmp(TIME_COMP_MODE, 'Linear'))
                addr = TriOp(IF_RESIZE, (ceil(x/2) - 1) * DVS_RESIZE_SIZE + ceil(y/2), (x - 1) * POSTURE_SIZE + y);
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
        test_labels(test_segment_num, 1) = Labels(rand_num(i));
    end
end

%��mat��ʽ����
save([strcat(SAVE_DIR, 'train_coding'), '.mat'], 'train_coding');
save([strcat(SAVE_DIR, 'test_coding'), '.mat'], 'test_coding');
save([strcat(SAVE_DIR, 'train_labels'), '.mat'], 'train_labels');
save([strcat(SAVE_DIR, 'test_labels'), '.mat'], 'test_labels');
fprintf(fopen(strcat(SAVE_DIR, 'train_segment_num.txt'),'wt'), '%d\t', train_segment_num);
fprintf(fopen(strcat(SAVE_DIR, 'test_segment_num.txt'),'wt'), '%d\t', test_segment_num);