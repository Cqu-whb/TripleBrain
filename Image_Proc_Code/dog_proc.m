
function DogImg = dog_proc(img)
%     gray = double(rgb2gray(img));
    gray = double(img);
    %mask_img = uint8(gray .* (mask > 0));  %��ʽ����Ϊuint����Ȼ˫���ȻᱻĬ�ϵ�0-1֮��
    %figure;imshow(img);

    resize_img = gray; %imresize(gray, [32, 32], 'bilinear', 'Antialiasing', true);  %���ٽ���ֵ��ע����resize��Ȼ��ֱ��Dog���Ȳ�����Ĥ��
    % figure;imshow(resize_img);

    sigma1 = 3; %3  xf 3
    sigma2 = 7; %7  xf 7
    window = 15; %3 yale % orl 15 xf 15
    % fspecial�������ڽ���Ԥ������˲����ӣ����﷨��ʽΪ��
    % h = fspecial(type,parameters,sigma)
    %   ����typeָ�����ӵ����ͣ�paraָ����Ӧ�Ĳ�����
    %   type= 'gaussian'��Ϊ��˹��ͨ�˲�����������������n��ʾģ��ߴ磬Ĭ��ֵΪ[3,3]��sigma��ʾ�˲����ı�׼���λΪ���أ�Ĭ��ֵΪ 0.5
    %   G=fspecial('gaussian',5)----����Ϊ5����ʾ���� 5*5 ��gaussian�������û�У�Ĭ��Ϊ 3*3 �ľ���
    H1 = fspecial('gaussian', window, sigma1);
    H2 = fspecial('gaussian', window, sigma2);

    % ����˹���
    DiffGauss = H1 - H2;
    % g = imfilter(f, w, filtering_mode, boundary_options, size_options)
    %   fΪ����ͼ��wΪ�˲���ģ��gΪ�˲���ͼ��
    %   filtering_mode����ָ�����˲���������ʹ�á���ء����ǡ��������
    %     ��corr�� ͨ��ʹ���������ɣ���ֵΪĬ�ϡ�
    %     ��conv�� ͨ��ʹ�þ�������
    %   boundary_options���ڴ���߽�������⣬�߽�Ĵ�С���˲����Ĵ�Сȷ����
    %     ��replicate�� ͼ���Сͨ��������߽��ֵ����չ
    %     ��symmetric�� ͼ���Сͨ����������߽�����չ
    out = abs(imfilter(resize_img, DiffGauss, 'replicate'));   %����������������άͼ������˲�

    % I = mat2gray(A, [amin amax])
    % ��ͼ�����A�н���amin��amax�����ݹ�һ������ ����С��amin��Ԫ�ض���Ϊ0�� ����amax��Ԫ�ض���Ϊ1��

    % I = mat2gray(A)
    % ��ͼ�����A��һ��Ϊͼ�����I�� ��һ���������ÿ��Ԫ�ص�ֵ����0��1��Χ��(����0��1)������0��ʾ��ɫ��1��ʾ��ɫ��
    out = mat2gray(out);
    DogImg = out;
    % figure;imshow(out);
end