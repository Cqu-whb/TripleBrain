clc
clear all;

ImageSize = 32;
OneDimLen = ImageSize * ImageSize + 1;

total_num = 600;

CSVDataMatrix = [total_num,OneDimLen]; %���Ƿ�resize�й�
file_index = ["1","2","3","4","5","6","7","8","9","10"];
csv_filepath = "orl_face.csv"; 
newcsv_filepath = "orl_newface.csv";
traincsv_filepath = "orl_trainface.csv";
testcsv_filepath = "orl_testface.csv";

file_row_index = 1;  %csv�ļ��ĵڼ���
for ClassNum = 1 : 10
    ID = file_index(ClassNum);
    file_path = char(strcat("..\..\..\ORL_Faces\s",ID,"\")); % ͼ���ļ���·��

    img_path_list = dir(strcat(file_path,'*.pgm'));%��ȡ���ļ���������jpg��ʽ��ͼ��
    img_num = length(img_path_list); %��ȡͼ��������
    
    for j = 1:img_num %��һ��ȡͼ��
        image_name = img_path_list(j).name;% ͼ����
        origal_image = imread(strcat(file_path,image_name));
        
        for k = 1 : 6
            %�仯����
            if k == 1
                change_image = origal_image; 
            end
            if k == 2  
                change_image = flip(origal_image, 2);  %ˮƽ����
            end
             if k == 3 
                change_image = origal_image(1:90,1:90); %����
            end           
             if k == 4
                change_image = origal_image(2:92,2:92);  %����
            end             
             if k == 5 
                change_image = origal_image(22:112,1:90);  %����
            end             
             if k == 6 
                change_image = origal_image(22:112,2:92);  %����
            end             
            
            imwrite(change_image,char(strcat("..\..\..\ORL_Faces\change_img\s",ID,int2str(j),int2str(k),".png")))

            image = imresize(change_image,[ImageSize,ImageSize],'bilinear');  %resize ͼ��
            %image=histeq(image,4096); %ֱ��ͼ���⻯
            image = dog_proc(image);
             image = uint8(image*256);
            imshow(image);
%             ͼ�������(��һ����ǩ֮��д��csv�ļ�)
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
end

multiple = 1; %�����ݷ�multiple��
len = total_num*multiple;
NewMultipleDataMatrix = zeros(len,OneDimLen);
for i = 1 : multiple
     NewMultipleDataMatrix((i-1) * total_num+1:i * total_num,:) =CSVDataMatrix(1:total_num,:);
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
csvwrite(testcsv_filepath,NewCsvDataMatrix(train_index+1 : len,:));





