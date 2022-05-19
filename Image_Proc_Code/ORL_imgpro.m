clc
clear all;

ImageSize = 32;
OneDimLen = ImageSize * ImageSize + 1;

total_num = 600;

CSVDataMatrix = [total_num,OneDimLen]; %与是否resize有关
file_index = ["1","2","3","4","5","6","7","8","9","10"];
csv_filepath = "orl_face.csv"; 
newcsv_filepath = "orl_newface.csv";
traincsv_filepath = "orl_trainface.csv";
testcsv_filepath = "orl_testface.csv";

file_row_index = 1;  %csv文件的第几行
for ClassNum = 1 : 10
    ID = file_index(ClassNum);
    file_path = char(strcat("..\..\..\ORL_Faces\s",ID,"\")); % 图像文件夹路径

    img_path_list = dir(strcat(file_path,'*.pgm'));%获取该文件夹中所有jpg格式的图像
    img_num = length(img_path_list); %获取图像总数量
    
    for j = 1:img_num %逐一读取图像
        image_name = img_path_list(j).name;% 图像名
        origal_image = imread(strcat(file_path,image_name));
        
        for k = 1 : 6
            %变化类型
            if k == 1
                change_image = origal_image; 
            end
            if k == 2  
                change_image = flip(origal_image, 2);  %水平镜像
            end
             if k == 3 
                change_image = origal_image(1:90,1:90); %左上
            end           
             if k == 4
                change_image = origal_image(2:92,2:92);  %右上
            end             
             if k == 5 
                change_image = origal_image(22:112,1:90);  %左下
            end             
             if k == 6 
                change_image = origal_image(22:112,2:92);  %右下
            end             
            
            imwrite(change_image,char(strcat("..\..\..\ORL_Faces\change_img\s",ID,int2str(j),int2str(k),".png")))

            image = imresize(change_image,[ImageSize,ImageSize],'bilinear');  %resize 图像
            %image=histeq(image,4096); %直方图均衡化
            image = dog_proc(image);
             image = uint8(image*256);
            imshow(image);
%             图像处理过程(加一个标签之后写入csv文件)
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
end

multiple = 1; %把数据翻multiple倍
len = total_num*multiple;
NewMultipleDataMatrix = zeros(len,OneDimLen);
for i = 1 : multiple
     NewMultipleDataMatrix((i-1) * total_num+1:i * total_num,:) =CSVDataMatrix(1:total_num,:);
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
csvwrite(testcsv_filepath,NewCsvDataMatrix(train_index+1 : len,:));





