clc
clear all;

ImageSize = 28;
OneDimLen = ImageSize * ImageSize + 1;

CSVDataMatrix = [638,OneDimLen]; %与是否resize有关
file_index = ["01","02","03","04","05","06","07","08","09","10"];
csv_filepath = "face.csv"; 
newcsv_filepath = "newface.csv";
traincsv_filepath = "trainface.csv";
testcsv_filepath = "testface.csv";



file_row_index = 1;  %csv文件的第几行
for ClassNum = 1 : 10
    ID = file_index(ClassNum);
    file_path = char(strcat("D:\Acedemic Research\算法\STDP Sparse Coding\人脸数据集\Yale\yaleB",ID,"\")); % 图像文件夹路径

    img_path_list = dir(strcat(file_path,'*.pgm'));%获取该文件夹中所有jpg格式的图像
    img_num = length(img_path_list); %获取图像总数量
    
    for j = 1:img_num %逐一读取图像
        image_name = img_path_list(j).name;% 图像名
        image = imread(strcat(file_path,image_name));
        image = imresize(image,[ImageSize,ImageSize],'bilinear');  %resize 图像
        image=histeq(image,4096); %直方图均衡化
%         image=imadjust(image,[0.2 0.9],[0 1],1.0); %灰度调整
%         filter=fspecial('gaussian',[9,9],0.8);  %平滑去噪
%         image = imfilter(image,filter);
        
        image = dog_proc(image);
        image = uint8(image*256);
        imshow(image);
%         pause();
        %fprintf('%d %s\n',j,strcat(file_path,image_name));% 显示正在处理的图像名
        %图像处理过程(加一个标签之后写入csv文件)
        CSVDataMatrix(file_row_index,1) = ClassNum - 1; %类别 
        [maxrow,maxcol] = size(image);
        for row = 1 : maxrow
            for col = 1 : maxcol
                index = (row - 1)*maxcol + col + 1;
                CSVDataMatrix(file_row_index,index) = image(row,col);
            end
        end
        file_row_index = file_row_index + 1;
    end
end

multiple = 1; %把数据翻multiple倍
len = 638*multiple;
NewMultipleDataMatrix = zeros(len,OneDimLen);
for i = 1 : multiple
     NewMultipleDataMatrix((i-1) * 638+1:i * 638,:) =CSVDataMatrix(1:638,:);
end



NewCsvDataMatrix =zeros(len,OneDimLen);
%随机排列组合

Rndindex = randperm(len);
for j = 1 : len
    x = Rndindex(j);
    NewCsvDataMatrix(x,:) = NewMultipleDataMatrix(j,:);
end

% for k = 1 : len
%     I = NewCsvDataMatrix(k,2:32257);
%     I = (reshape(I,168,192))';
%     imshow(I,[])
%     pause();
% end


%将数据集分为训练集和测试集
train_index = fix(len * 0.6);

csvwrite(csv_filepath,CSVDataMatrix);
csvwrite(newcsv_filepath,NewCsvDataMatrix);

csvwrite(traincsv_filepath,NewCsvDataMatrix(1:train_index,:));
csvwrite(testcsv_filepath,NewCsvDataMatrix(train_index : len,:));





