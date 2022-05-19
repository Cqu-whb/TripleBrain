% -----------------------------------------------------------------------------------------------------------------------------------------------------%
% @编写：wtx
% @更新日期：2022/1/1
% @运行时间：1s
% @概述：采用硬分段的方式，将DVS数据集基于事件个数分段，时间压缩方式支持“不压缩(None)”、“紧凑型(Compact)”和“线性(Linear)”三种方式，可根据实际情况选择
% @输入数据格式：原始mat文件中包含100个cell，每个cell表示一张图像的脉冲，其第1、4、5列分别表示时间、像素横坐标、像素纵坐标，每张图约2ms,标签是1-4
% @输出格式：1行n列数组[addr1 t1 addr2 t2 ... -1 addr1 t1 addr2 t2 ... -1 ...], 地址addr范围：[1, img_size^2], 时间戳t范围：[0, ...]，标签0-3
% @备注1：在DVS数据集中一段（Segment）相当于一张图，分段方法包括软分段和硬分段，本脚本采用较为简单的基于事件数的硬分段方式（不够一段的脉冲扔掉）
% @备注2：时间压缩方式“不压缩(None)”指就按原始数据集时间输出，单位us
% @备注3：时间压缩方式“紧凑型(Compact)”指一个事件对应一个时间步，即脉冲发射时间间距为1，每个时间步都有脉冲，单位可自己定义
% @备注4：时间压缩方式“线性(Linear)”指把若干us的脉冲合在一起发射（以压缩尺度为单位向下取整），相同位置像素发射多次的按一次算，单位按压缩尺度计算
% @备注5：原始数据是按标签依次排列的，首先需全部打乱后再进行分段
% @备注6：分段后训练集和测试集的Segment数量分别以txt格式保存在相同目录下
% @备注7：由于DVS数据集的Resize较为少见且复杂，该版本脚本仅支持不Resize或Resize成16x16，其它尺寸Resize可参考下面代码做修改
% -----------------------------------------------------------------------------------------------------------------------------------------------------%
clear;clc;
load('.\Definition_pkg');
load('..\..\Common_Datasets\DVS\CARD_DVS\Original_mat\cards_100_seq');
%----------------------------------------------------------------------参数配置区-------------------------------------------------------------------%
%训练集占比
TRAIN_RATIO = 0.8; 

%分段参数
NEVENT_PER_SEGMENT = 200;  %每个Segment的事件数

%时间压缩模式
TIME_COMP_MODE = 'Linear';  %可选：'None', 'Compact', 'Linear'

%线性压缩尺度（单位：us）
LINEAR_COMP_SCALE = 10; 

%是否Resize（注：由于DVS数据集的Resize较为少见且复杂，该版本脚本仅支持不Resize或Resize成16x16）
%注1：由32x32Resize成16x16就是按照时间步，依次将图像maxpooling到16x16，对于“不压缩(None)”和“紧凑型(Compact)”，由于每个时间步就1个事件，
%     因此不必真的对二维矩阵做maxpooling，直接对坐标x, y进行映射即可，即16x16_x = ceil(32x32_x / 2)，16x16_y = ceil(32x32_y / 2)
%注2：对于“线性(Linear)”压缩，也暂不采用真maxpooling的方法，但需注意由于一个时间步包含多个脉冲，若在一个时间步内计算出多个一样的16x16_addr，保留一个放入最终变量即可
IF_RESIZE = 0;   % 是否Resize为16x16，1--True, 0--False
%----------------------------------------------------------------------参数配置区-------------------------------------------------------------------%
%压缩后保存目录（各DVS数据集各压缩目录已提前创建好）
folder_name = strcat('Comp_', TIME_COMP_MODE);
SAVE_DIR = strcat('..\..\Common_Datasets\DVS\CARD_DVS\', folder_name, '\');

train_coding = zeros(1, 1); %在matlab中先随意声明变量后再任意拓展，实测对运行速度影响不大
test_coding = zeros(1, 1);
train_labels = zeros(1, 1);
test_labels = zeros(1, 1);

train_segment_num = 0;
test_segment_num = 0;

%打乱原始数据
%返回一行从1到100的整数中的1w个，而且这100个数各不相同
rand_num = randperm(CARD_DVS_NCELL, CARD_DVS_NCELL); 
CINs_rand = cell(1, CARD_DVS_NCELL); %打乱后的数据
for i = 1 : (CARD_DVS_NCELL)
    CINs_rand{i} = CINs{rand_num(i)};
end

%分段训练集数据，并分配好标签
cnt = 0;
for i = 1 : round(CARD_DVS_NCELL * TRAIN_RATIO)
    img_spike_num = length(CINs_rand{i}(:, 1));                  %该张图像脉冲总数
    img_segment_num = floor(img_spike_num / NEVENT_PER_SEGMENT); %该张图像分段后段数
    t_pre = CINs_rand{i}(1, 1);  %确保每张图都从0时刻开始
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
                                   -1); %下一个事件的时间戳，若已到最后一个事件就用-1表示
            %不压缩(None)
            if (strcmp(TIME_COMP_MODE, 'None'))
                cnt = cnt + 1;
                train_coding(cnt) = TriOp(IF_RESIZE, (ceil(x/2) - 1) * DVS_RESIZE_SIZE + ceil(y/2), (x - 1) * CARD_SIZE + y); %地址
                cnt = cnt + 1;
                train_coding(cnt) = t - t_pre;  %时间戳，确保每段都从0开始
            end
            %紧凑型(Compact)
            if (strcmp(TIME_COMP_MODE, 'Compact'))
                cnt = cnt + 1;
                train_coding(cnt) = TriOp(IF_RESIZE, (ceil(x/2) - 1) * DVS_RESIZE_SIZE + ceil(y/2), (x - 1) * CARD_SIZE + y); %地址
                if (train_coding(cnt) > 256)
                    temp = train_coding(cnt);
                    trmp = 0;
                end
                cnt = cnt + 1;
                train_coding(cnt) = t_compact;  %压缩时间戳，每段都从0开始，每个事件+1
                t_compact = t_compact + 1;
            end
            %线性(Linear)
            if (strcmp(TIME_COMP_MODE, 'Linear'))
                addr = TriOp(IF_RESIZE, (ceil(x/2) - 1) * DVS_RESIZE_SIZE + ceil(y/2), (x - 1) * CARD_SIZE + y);
                t_linear_addr_temp = [t_linear_addr_temp addr];  %数组元素个数超过1000时不建议采用这种方式添加元素，应提前分配内存
                if (t_linear_post ~=  t_linear) %当前时间步事件地址均添加完毕
                    t_linear_addr_temp = unique(t_linear_addr_temp);  %去除相同的地址
                    len = length(t_linear_addr_temp); %获取事件个数
                    for n = 1 : len
                         cnt = cnt + 1;
                         train_coding(cnt) = t_linear_addr_temp(n);
                         cnt = cnt + 1;
                         train_coding(cnt) = t_linear - t_linear_pre;
                    end
                    t_linear_pretimestep = t_linear;  %更新t_linear_pre
                    t_linear_addr_temp = [];
                end
            end
        end
        cnt = cnt + 1;
        train_coding(cnt) = -1;  %每段以-1结尾
        t_pre = t;
        t_linear_pre = t_linear;
        t_compact = 0;       
        train_segment_num = train_segment_num + 1;
        train_labels(train_segment_num, 1) = Labels(rand_num(i)) - 1; %1-4变为0-3
    end
end

%分段测试集数据，并分配好标签
cnt = 0;
for i = round(CARD_DVS_NCELL * TRAIN_RATIO) + 1 : CARD_DVS_NCELL
    img_spike_num = length(CINs_rand{i}(:, 1));                  %该张图像脉冲总数
    img_segment_num = floor(img_spike_num / NEVENT_PER_SEGMENT); %该张图像分段后段数
    t_pre = CINs_rand{i}(1, 1);  %确保每张图都从0时刻开始
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
                                   -1); %下一个事件的时间戳，若已到最后一个事件就用-1表示
            %不压缩(None)
            if (strcmp(TIME_COMP_MODE, 'None'))
                cnt = cnt + 1;
                test_coding(cnt) = TriOp(IF_RESIZE, (ceil(x/2) - 1) * DVS_RESIZE_SIZE + ceil(y/2), (x - 1) * CARD_SIZE + y); %地址
                cnt = cnt + 1;
                test_coding(cnt) = t - t_pre;  %时间戳，确保每段都从0开始
            end
            %紧凑型(Compact)
            if (strcmp(TIME_COMP_MODE, 'Compact'))
                cnt = cnt + 1;
                test_coding(cnt) = TriOp(IF_RESIZE, (ceil(x/2) - 1) * DVS_RESIZE_SIZE + ceil(y/2), (x - 1) * CARD_SIZE + y); %地址
                cnt = cnt + 1;
                test_coding(cnt) = t_compact;  %压缩时间戳，每段都从0开始，每个事件+1
                t_compact = t_compact + 1;
            end
            %线性(Linear)
            if (strcmp(TIME_COMP_MODE, 'Linear'))
                addr = TriOp(IF_RESIZE, (ceil(x/2) - 1) * DVS_RESIZE_SIZE + ceil(y/2), (x - 1) * CARD_SIZE + y);
                t_linear_addr_temp = [t_linear_addr_temp addr];  %数组元素个数超过1000时不建议采用这种方式添加元素，应提前分配内存
                if (t_linear_post ~=  t_linear) %当前时间步事件地址均添加完毕
                    t_linear_addr_temp = unique(t_linear_addr_temp);  %去除相同的地址
                    len = length(t_linear_addr_temp); %获取事件个数
                    for n = 1 : len
                         cnt = cnt + 1;
                         test_coding(cnt) = t_linear_addr_temp(n);
                         cnt = cnt + 1;
                         test_coding(cnt) = t_linear - t_linear_pre;
                    end
                    t_linear_pretimestep = t_linear;  %更新t_linear_pre
                    t_linear_addr_temp = [];
                end
            end
        end
        cnt = cnt + 1;
        test_coding(cnt) = -1;  %每段以-1结尾
        t_pre = t;
        t_linear_pre = t_linear;
        t_compact = 0;       
        test_segment_num = test_segment_num + 1;
        test_labels(test_segment_num, 1) = Labels(rand_num(i)) - 1;  %1-4变为0-3
    end
end

%以mat格式保存
save([strcat(SAVE_DIR, 'train_coding'), '.mat'], 'train_coding');
save([strcat(SAVE_DIR, 'test_coding'), '.mat'], 'test_coding');
save([strcat(SAVE_DIR, 'train_labels'), '.mat'], 'train_labels');
save([strcat(SAVE_DIR, 'test_labels'), '.mat'], 'test_labels');
fprintf(fopen(strcat(SAVE_DIR, 'train_segment_num.txt'),'wt'), '%d\t', train_segment_num);
fprintf(fopen(strcat(SAVE_DIR, 'test_segment_num.txt'),'wt'), '%d\t', test_segment_num);

