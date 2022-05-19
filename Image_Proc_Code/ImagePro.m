clc
clear all;

ImageSize = 28;
OneDimLen = ImageSize * ImageSize + 1;

CSVDataMatrix = [638,OneDimLen]; %���Ƿ�resize�й�
file_index = ["01","02","03","04","05","06","07","08","09","10"];
csv_filepath = "face.csv"; 
newcsv_filepath = "newface.csv";
traincsv_filepath = "trainface.csv";
testcsv_filepath = "testface.csv";



file_row_index = 1;  %csv�ļ��ĵڼ���
for ClassNum = 1 : 10
    ID = file_index(ClassNum);
    file_path = char(strcat("D:\Acedemic Research\�㷨\STDP Sparse Coding\�������ݼ�\Yale\yaleB",ID,"\")); % ͼ���ļ���·��

    img_path_list = dir(strcat(file_path,'*.pgm'));%��ȡ���ļ���������jpg��ʽ��ͼ��
    img_num = length(img_path_list); %��ȡͼ��������
    
    for j = 1:img_num %��һ��ȡͼ��
        image_name = img_path_list(j).name;% ͼ����
        image = imread(strcat(file_path,image_name));
        image = imresize(image,[ImageSize,ImageSize],'bilinear');  %resize ͼ��
        image=histeq(image,4096); %ֱ��ͼ���⻯
%         image=imadjust(image,[0.2 0.9],[0 1],1.0); %�Ҷȵ���
%         filter=fspecial('gaussian',[9,9],0.8);  %ƽ��ȥ��
%         image = imfilter(image,filter);
        
        image = dog_proc(image);
        image = uint8(image*256);
        imshow(image);
%         pause();
        %fprintf('%d %s\n',j,strcat(file_path,image_name));% ��ʾ���ڴ����ͼ����
        %ͼ�������(��һ����ǩ֮��д��csv�ļ�)
        CSVDataMatrix(file_row_index,1) = ClassNum - 1; %��� 
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

multiple = 1; %�����ݷ�multiple��
len = 638*multiple;
NewMultipleDataMatrix = zeros(len,OneDimLen);
for i = 1 : multiple
     NewMultipleDataMatrix((i-1) * 638+1:i * 638,:) =CSVDataMatrix(1:638,:);
end



NewCsvDataMatrix =zeros(len,OneDimLen);
%����������

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


%�����ݼ���Ϊѵ�����Ͳ��Լ�
train_index = fix(len * 0.6);

csvwrite(csv_filepath,CSVDataMatrix);
csvwrite(newcsv_filepath,NewCsvDataMatrix);

csvwrite(traincsv_filepath,NewCsvDataMatrix(1:train_index,:));
csvwrite(testcsv_filepath,NewCsvDataMatrix(train_index : len,:));





