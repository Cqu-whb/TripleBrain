% -----------------------------------------------------------------------------------------------------------------------------------------------------%
% @编写：wtx
% @更新日期：2022/1/1
% @运行时间：5s
% @概述：将半导体所SPAD传感器拍摄的真实手写数据集基于事件个数分段，时间压缩方式为“紧凑型(Compact)”
% @输入数据格式：[28, 28, 288, 10]的数组，用1/0表示在x, y, t, label处是否有/无脉冲
% @输出格式：1行n列数组[addr1 t1 addr2 t2 ... -1 addr1 t1 addr2 t2 ... -1 ...]
% @备注：该数据集不是开源数据集，本脚本仅对其进行一种配置的处理
% -----------------------------------------------------------------------------------------------------------------------------------------------------%
clear;clc;
load('.\Definition_pkg');
load('..\..\Common_Datasets\DVS\SPAD_MNIST\Original_mat\imdb28o');
%----------------------------------------------------------------------参数配置区-------------------------------------------------------------------%
%训练集占比
TRAIN_RATIO = 0.9; 

%分段参数
NEVENT_PER_SEGMENT = 500;  %每个Segment的事件数 

%是否Resize
IF_RESIZE = 1;   % 是否Resize为16x16，1--True, 0--False
%----------------------------------------------------------------------参数配置区-------------------------------------------------------------------%
n_spike_per_img = [27950, 15773, 24575, 29625, 18893, 29047, 25078, 21420, 27197, 20130]; %每张图像脉冲个数
SAVE_DIR = '..\..\Common_Datasets\DVS\SPAD_MNIST\Comp_Compact\';

%计算一共有多少段，并为每段分配标签
n_all_segment = 0;
Labels = zeros(1, 1);
cnt = 1;
for i = 1 : 10
    n_all_segment = n_all_segment + floor(n_spike_per_img(i)/NEVENT_PER_SEGMENT);
    for j = 1 : floor(n_spike_per_img(i)/NEVENT_PER_SEGMENT)
        Labels(cnt) = i - 1;
        cnt = cnt + 1;
    end
end
train_segment_num = floor(n_all_segment * TRAIN_RATIO);
test_segment_num = n_all_segment - train_segment_num;

CINs = cell(1, n_all_segment);

%先将图像换一种格式
cell_data = cell(1, 10);
cnt = 1;
for n = 1 : 10
    for t = 1 : 288
        for x = 1 : 28
            for y = 1 : 28
                if (imdb.data(x, y, t, n) == 1)
                    cell_data{n}(cnt, 1) = t;
                    cell_data{n}(cnt, 2) = x;
                    cell_data{n}(cnt, 3) = y;
                    cnt = cnt + 1;
                end
            end
        end
    end
    cnt = 1;
end

cnt = 1;
for i = 1 : 10
    n_segment_per_img = floor(n_spike_per_img(i)/NEVENT_PER_SEGMENT);
    for j = 1 : n_segment_per_img
        CINs{cnt} = cell_data{i}( ((j-1)*NEVENT_PER_SEGMENT + 1) : (j * NEVENT_PER_SEGMENT), :);
        cnt = cnt + 1;
    end
end

%打乱数据
rand_num = randperm(n_all_segment, n_all_segment); 
CINs_rand = cell(1, n_all_segment); %打乱后的数据
for i = 1 : (n_all_segment)
    CINs_rand{i} = CINs{rand_num(i)};
end

train_coding = zeros(1, 1); 
test_coding = zeros(1, 1);
train_labels = zeros(1, 1);
test_labels = zeros(1, 1);

%将训练集数据变为最终格式
cnt = 1;
for i = 1 : train_segment_num
    t_compact = 0;
    for j = 1 : NEVENT_PER_SEGMENT
        t = CINs_rand{i}(j, 1);
        x = CINs_rand{i}(j, 2);
        y = CINs_rand{i}(j, 3);
        train_coding(cnt) = TriOp(IF_RESIZE, (ceil((x+2)/2) - 1) * DVS_RESIZE_SIZE + ceil((y+2)/2), (x - 1) * MNIST_SIZE + y); %地址
        cnt = cnt + 1;
        train_coding(cnt) = t_compact;  %压缩时间戳，每段都从0开始，每个事件+1
        cnt = cnt + 1;
        t_compact = t_compact + 1;
    end
    train_labels(i, 1) = Labels(rand_num(i));
    train_coding(cnt) = -1;   %每段以-1结尾
    cnt = cnt + 1;
end

%将测试集数据变为最终格式
cnt = 1;
for i = train_segment_num + 1 : n_all_segment
    t_compact = 0;
    for j = 1 : NEVENT_PER_SEGMENT
        t = CINs_rand{i}(j, 1);
        x = CINs_rand{i}(j, 2);
        y = CINs_rand{i}(j, 3);
        test_coding(cnt) = TriOp(IF_RESIZE, (ceil((x+2)/2) - 1) * DVS_RESIZE_SIZE + ceil((y+2)/2), (x - 1) * MNIST_SIZE + y); %地址
        cnt = cnt + 1;
        test_coding(cnt) = t_compact;  %压缩时间戳，每段都从0开始，每个事件+1
        cnt = cnt + 1;
        t_compact = t_compact + 1;
    end
    test_labels(i - train_segment_num, 1) = Labels(rand_num(i));
    test_coding(cnt) = -1;  %每段以-1结尾
    cnt = cnt + 1;
end

%以mat格式保存
save([strcat(SAVE_DIR, 'train_coding'), '.mat'], 'train_coding');
save([strcat(SAVE_DIR, 'test_coding'), '.mat'], 'test_coding');
save([strcat(SAVE_DIR, 'train_labels'), '.mat'], 'train_labels');
save([strcat(SAVE_DIR, 'test_labels'), '.mat'], 'test_labels');
fprintf(fopen(strcat(SAVE_DIR, 'train_segment_num.txt'),'wt'), '%d\t', train_segment_num);
fprintf(fopen(strcat(SAVE_DIR, 'test_segment_num.txt'),'wt'), '%d\t', test_segment_num);

