% -----------------------------------------------------------------------------------------------------------------------------------------------------%
% @��д��wtx
% @�������ڣ�2022/1/1
% @����������ȫ�ֱ�����������mat�ļ��й������ű�ʹ��
% @��ע1�����ļ��û������޸�
% @��ע2���˷���ÿ������µı�����ʵʱ���и��£���������һ�����õĵط�
% -----------------------------------------------------------------------------------------------------------------------------------------------------%
MAX_PIXEL_VALUE = 256;  %�������ֵ���ϸ�˵��255��256��Ϊ�˷�����㣩

MNIST_TRAIN_IMG = 60000;  %MNIST��FASHION_MNISTӵ����ͬ�����ݼ�������ͼ��ߴ�
MNIST_TEST_IMG = 10000;
MNIST_SIZE = 28;
MNIST_NCLASS = 10;
DENOISING_THRESHOLD = 16;  %С��ȥ����ֵ������ֵ����

ETH80_IMG = 3280;  %ETH80���ݼ�ͼ������
ETH80_SIZE = 256;  %ETH80���ݼ��ߴ�: 256x256��ͨ��ͼ��
ETH80_NCLASS = 8;  %ETH80���ݼ������
ETH80_NFOLDER_PER_CLASS = 10; %ETH80���ݼ�ÿ������°������ļ�����
ETH80_NIMG_PER_FOLDER = 41;   %ETH80���ݼ�ÿ������ÿ���ļ����°�����ͼ����

CIFAR10_TRAIN_NBATCH = 5;        %CIFAR10ԭʼѵ����������5��batch
CIFAR10_TEST_NBATCH = 1;         %CIFAR10ԭʼ���Լ�������1��batch
CIFAR10_NIMG_PER_BATCH = 10000;  %CIFAR10ÿ��batch����ͼ����
CIFAR10_TRAIN_IMG = 50000;       %CIFAR10ѵ����ͼ������
CIFAR10_TEST_IMG = 10000;        %CIFAR10���Լ�ͼ������
CIFAR10_SIZE = 32;               %CIFAR10�ߴ�

DVS_RESIZE_SIZE = 16;            %�ð汾DVS���ݼ���֧��Resize��16x16

MNIST_DVS_NCELL = 10000;          %MNIST_DVS���ݼ�cell����
MNIST_DVS_TRAIN_MAX_SPIKE = 1e7;  %MNIST_DVSѵ�������������
MNIST_DVS_TEST_MAX_SPIKE = 1e7;   %MNIST_DVS���Լ����������

CARD_DVS_NCELL = 100;  %CARD_DVS���ݼ�cell����
CARD_SIZE = 32;        %CARDԭʼͼ��ߴ�

POSTURE_DVS_NCELL = 484;
POSTURE_SIZE = 32;

N_MNIST_TRAIN_NCELL = 60000;
N_MNIST_TEST_NCELL = 10000;
N_MNIST_SIZE = 34;


save(['.\Definition_pkg', '.mat'], 'MNIST_TRAIN_IMG', 'MNIST_TEST_IMG', 'DENOISING_THRESHOLD', 'MNIST_SIZE', 'MNIST_NCLASS', 'MAX_PIXEL_VALUE' ...
      , 'ETH80_IMG', 'ETH80_SIZE', 'ETH80_NCLASS', 'ETH80_NFOLDER_PER_CLASS', 'ETH80_NIMG_PER_FOLDER', 'CIFAR10_TRAIN_NBATCH' ...
      , 'CIFAR10_TEST_NBATCH', 'CIFAR10_NIMG_PER_BATCH', 'CIFAR10_TRAIN_IMG','CIFAR10_TEST_IMG', 'CIFAR10_SIZE', 'MNIST_DVS_NCELL' ...
      , 'MNIST_DVS_TRAIN_MAX_SPIKE', 'MNIST_DVS_TEST_MAX_SPIKE', 'DVS_RESIZE_SIZE', 'CARD_DVS_NCELL', 'CARD_SIZE', 'POSTURE_DVS_NCELL' ...
      , 'POSTURE_SIZE', 'N_MNIST_TRAIN_NCELL', 'N_MNIST_TEST_NCELL', 'N_MNIST_SIZE');