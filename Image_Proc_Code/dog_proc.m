
function DogImg = dog_proc(img)
%     gray = double(rgb2gray(img));
    gray = double(img);
    %mask_img = uint8(gray .* (mask > 0));  %显式声明为uint，不然双精度会被默认到0-1之间
    %figure;imshow(img);

    resize_img = gray; %imresize(gray, [32, 32], 'bilinear', 'Antialiasing', true);  %最临近插值（注：先resize，然后直接Dog，先不用掩膜）
    % figure;imshow(resize_img);

    sigma1 = 3; %3  xf 3
    sigma2 = 7; %7  xf 7
    window = 15; %3 yale % orl 15 xf 15
    % fspecial函数用于建立预定义的滤波算子，其语法格式为：
    % h = fspecial(type,parameters,sigma)
    %   其中type指定算子的类型，para指定相应的参数；
    %   type= 'gaussian'，为高斯低通滤波器，参数有两个，n表示模版尺寸，默认值为[3,3]，sigma表示滤波器的标准差，单位为像素，默认值为 0.5
    %   G=fspecial('gaussian',5)----参数为5，表示产生 5*5 的gaussian矩阵，如果没有，默认为 3*3 的矩阵。
    H1 = fspecial('gaussian', window, sigma1);
    H2 = fspecial('gaussian', window, sigma2);

    % 作高斯差分
    DiffGauss = H1 - H2;
    % g = imfilter(f, w, filtering_mode, boundary_options, size_options)
    %   f为输入图像，w为滤波掩模，g为滤波后图像
    %   filtering_mode用于指定在滤波过程中是使用“相关”还是“卷积”。
    %     ‘corr’ 通过使用相关来完成，该值为默认。
    %     ‘conv’ 通过使用卷积来完成
    %   boundary_options用于处理边界充零问题，边界的大小由滤波器的大小确定。
    %     ‘replicate’ 图像大小通过复制外边界的值来扩展
    %     ‘symmetric’ 图像大小通过镜像反射其边界来扩展
    out = abs(imfilter(resize_img, DiffGauss, 'replicate'));   %对任意类型数组或多维图像进行滤波

    % I = mat2gray(A, [amin amax])
    % 将图像矩阵A中介于amin和amax的数据归一化处理， 其余小于amin的元素都变为0， 大于amax的元素都变为1。

    % I = mat2gray(A)
    % 将图像矩阵A归一化为图像矩阵I， 归一化后矩阵中每个元素的值都在0到1范围内(包括0和1)。其中0表示黑色，1表示白色。
    out = mat2gray(out);
    DogImg = out;
    % figure;imshow(out);
end