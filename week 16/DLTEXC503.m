%%  ����˵��
% ʵ�� 5.1-3
% ���ܣ�����VGG16����������ͼ����з���
% ���ߣ�zhaoxch_mail@sina.com
% ʱ�䣺2020��3��15��
% �汾��DLTEXC503-V1

%%  ����Ԥѵ���õ�VGG16���������,��ȷ������������ͼ��Ĵ�С�Լ��������������
net = vgg16;                                   % ��VGG16��������絼�빤����
inputSize = net.Layers(1).InputSize;             % ��ȡVGG16��������������������ͼ��Ĵ�С
classNames = net.Layers(end).ClassNames;         % ��ȡVGG16���������������еķ���

%% ����RGBͼ�񣬲���ͼ��Ĵ�С�任����VGG16��������������������ͼ����ͬ�Ĵ�С
I = imread('glassdog.jpg');
figure
imshow(I)
I = imresize(I,inputSize(1:2));


%% ����VGG16���������������ͼ����з���
[label1,scores1] = classify(net,I);

%% ��ͼ������ʾ������������
figure
imshow(I)
title(string(label1) + ", " + num2str(100*scores1(classNames == label1),3) + "%");

