% -----------------------------------------------------------------------------------------------------------------------------------------------------%
% @编写：wtx
% @更新日期：2022/1/1
% @运行时间：6000/1000张图350s以内
% @概述：采用硬分段的方式，将DVS数据集基于事件个数分段，时间压缩方式支持“不压缩(None)”、“紧凑型(Compact)”和“线性(Linear)”三种方式，可根据实际情况选择
% @输入数据格式：转换后的mat文件中包括60000个训练cell，10000个测试cell，每个cell表示一张图像的脉冲，其第1、4、5列分别表示时间、像素横坐标、像素纵坐标，每张图约300ms
% @输出格式：1行n列数组[addr1 t1 addr2 t2 ... -1 addr1 t1 addr2 t2 ... -1 ...], 地址addr范围：[1, img_size^2], 时间戳t范围：[0, ...]
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
load('..\..\Common_Datasets\DVS\N_MNIST\Original_mat\N_MNIST_DVS_34x34_60000');
load('..\..\Common_Datasets\DVS\N_MNIST\Original_mat\N_MNIST_DVS_34x34_10000');
load('..\..\Common_Datasets\DVS\N_MNIST\Original_mat\train_labels');
load('..\..\Common_Datasets\DVS\N_MNIST\Original_mat\test_labels');
LOCAL_N_MNIST_SIZE = 32;
%----------------------------------------------------------------------参数配置区-------------------------------------------------------------------%
%训练集占比(N_MNIST_DVS数据太多，建议取一部分出来训练/测试即可，建议值：6000/1000)
TRAIN_CELL = 6000;
TEST_CELL = 1000;

%分段参数
% NEVENT_PER_SEGMENT = 1000;  %每个Segment的事件数
NEVENT_PER_SEGMENT = 1000;  %每个Segment的事件数

%时间压缩模式
TIME_COMP_MODE = 'Linear';  %可选：'None', 'Compact', 'Linear'

%线性压缩尺度（单位：us）
LINEAR_COMP_SCALE = 500; 

%是否Resize（注：由于DVS数据集的Resize较为少见且复杂，该版本脚本仅支持不Resize或Resize成16x16）
%注：由34x34 Resize成16x16,为简单起见，将图像裁剪为32x32，即忽略最外围一圈像素，然后将原(x, y)坐标-1即可，后续Resize方法与之前一致
IF_RESIZE = 0;   % 是否Resize为16x16，1--True, 0--False
%----------------------------------------------------------------------参数配置区-------------------------------------------------------------------%
%压缩后保存目录（各DVS数据集各压缩目录已提前创建好）
folder_name = strcat('Comp_', TIME_COMP_MODE);
SAVE_DIR = strcat('..\..\Common_Datasets\DVS\N_MNIST\', folder_name, '\');

train_coding = zeros(1, 1); %在matlab中先随意声明变量后再任意拓展，实测对运行速度影响不大
test_coding = zeros(1, 1);
train_labels_rand = zeros(1, 1);
test_labels_rand = zeros(1, 1);

train_segment_num = 0;
test_segment_num = 0;

%打乱训练和测试数据
rand_num_train = randperm(N_MNIST_TRAIN_NCELL, N_MNIST_TRAIN_NCELL); 
rand_num_test = randperm(N_MNIST_TEST_NCELL, N_MNIST_TEST_NCELL); 
train_cells_rand = cell(1, N_MNIST_TRAIN_NCELL); %打乱后的训练cell
test_cells_rand = cell(1, N_MNIST_TEST_NCELL);   %打乱后的测试cell
for i = 1 : (N_MNIST_TRAIN_NCELL)
    train_cells_rand{i} = train_cells{rand_num_train(i)};
end
for i = 1 : (N_MNIST_TEST_NCELL)
    test_cells_rand{i} = test_cells{rand_num_test(i)};
end

%分段训练集数据，并分配好标签
cnt = 0;
for i = 1 : TRAIN_CELL
    img_spike_num = length(train_cells_rand{i}(:, 1));           %该张图像脉冲总数
    img_segment_num = floor(img_spike_num / NEVENT_PER_SEGMENT); %该张图像分段后段数
    t_pre = train_cells_rand{i}(1, 1);  %确保每张图都从0时刻开始
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
                                   -1); %下一个事件的时间戳，若已到最后一个事件就用-1表示
            %不压缩(None)
            if (strcmp(TIME_COMP_MODE, 'None') && (x > 1 && x < LOCAL_N_MNIST_SIZE) && (y > 1 && y < LOCAL_N_MNIST_SIZE))
                cnt = cnt + 1;
                %train_coding(cnt) = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1) * LOCAL_N_MNIST_SIZE + y); %地址
                train_coding(cnt) = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1-1) * LOCAL_N_MNIST_SIZE + y-1); %地址
                cnt = cnt + 1;
                train_coding(cnt) = t - t_pre;  %时间戳，确保每段都从0开始
            end
            %紧凑型(Compact)
            if (strcmp(TIME_COMP_MODE, 'Compact') && (x > 1 && x < LOCAL_N_MNIST_SIZE) && (y > 1 && y < LOCAL_N_MNIST_SIZE))
                cnt = cnt + 1;
                %train_coding(cnt) = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1) * LOCAL_N_MNIST_SIZE + y); %地址
                train_coding(cnt) = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1-1) * LOCAL_N_MNIST_SIZE + y-1); %地址
                cnt = cnt + 1;
                train_coding(cnt) = t_compact;  %压缩时间戳，每段都从0开始，每个事件+1
                t_compact = t_compact + 1;
            end
            %线性(Linear)
            if (strcmp(TIME_COMP_MODE, 'Linear') && (x > 1 && x < LOCAL_N_MNIST_SIZE) && (y > 1 && y < LOCAL_N_MNIST_SIZE))
                %addr = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1) * LOCAL_N_MNIST_SIZE + y);
                addr = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1-1) * LOCAL_N_MNIST_SIZE + y-1);
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
        train_labels_rand(train_segment_num, 1) = train_labels(rand_num_train(i));
    end
end

%分段测试集数据，并分配好标签
cnt = 0;
for i = 1 : TEST_CELL
    img_spike_num = length(test_cells_rand{i}(:, 1));            %该张图像脉冲总数
    img_segment_num = floor(img_spike_num / NEVENT_PER_SEGMENT); %该张图像分段后段数
    t_pre = test_cells_rand{i}(1, 1);  %确保每张图都从0时刻开始
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
                                   -1); %下一个事件的时间戳，若已到最后一个事件就用-1表示
            %不压缩(None)
            if (strcmp(TIME_COMP_MODE, 'None') && (x > 1 && x < LOCAL_N_MNIST_SIZE) && (y > 1 && y < LOCAL_N_MNIST_SIZE))
                cnt = cnt + 1;
                %test_coding(cnt) = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1) * LOCAL_N_MNIST_SIZE + y); %地址
                test_coding(cnt) = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1 - 1) * LOCAL_N_MNIST_SIZE + y-1); %地址
                cnt = cnt + 1;
                test_coding(cnt) = t - t_pre;  %时间戳，确保每段都从0开始
            end
            %紧凑型(Compact)
            if (strcmp(TIME_COMP_MODE, 'Compact') && (x > 1 && x < LOCAL_N_MNIST_SIZE) && (y > 1 && y < LOCAL_N_MNIST_SIZE))
                cnt = cnt + 1;
                %test_coding(cnt) = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1) * LOCAL_N_MNIST_SIZE + y); %地址
                test_coding(cnt) = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1-1) * LOCAL_N_MNIST_SIZE + y-1); %地址
                cnt = cnt + 1;
                test_coding(cnt) = t_compact;  %压缩时间戳，每段都从0开始，每个事件+1
                t_compact = t_compact + 1;
            end
            %线性(Linear)
            if (strcmp(TIME_COMP_MODE, 'Linear') && (x > 1 && x < LOCAL_N_MNIST_SIZE) && (y > 1 && y < LOCAL_N_MNIST_SIZE))
%                 addr = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1) * LOCAL_N_MNIST_SIZE + y);
                addr = TriOp(IF_RESIZE, (ceil((x-1)/2) - 1) * DVS_RESIZE_SIZE + ceil((y-1)/2), (x - 1-1) * LOCAL_N_MNIST_SIZE + y-1);
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
        test_labels_rand(test_segment_num, 1) = test_labels(rand_num_test(i));
    end
end
%以mat格式保存
save([strcat(SAVE_DIR, 'train_coding'), '.mat'], 'train_coding');
save([strcat(SAVE_DIR, 'test_coding'), '.mat'], 'test_coding');
save([strcat(SAVE_DIR, 'train_labels'), '.mat'], 'train_labels_rand');
save([strcat(SAVE_DIR, 'test_labels'), '.mat'], 'test_labels_rand');
fprintf(fopen(strcat(SAVE_DIR, 'train_segment_num.txt'),'wt'), '%d\t', train_segment_num);
fprintf(fopen(strcat(SAVE_DIR, 'test_segment_num.txt'),'wt'), '%d\t', test_segment_num);


