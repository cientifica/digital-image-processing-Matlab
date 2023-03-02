%%  程序说明
% 实例 5.1-3
% 功能：基于VGG16卷积神经网络对图像进行分类
% 作者：zhaoxch_mail@sina.com
% 时间：2020年3月15日
% 版本：DLTEXC503-V1

%%  导入预训练好的VGG16卷积神经网络,并确定该网络输入图像的大小以及分类种类的名称
net = vgg16;                                   % 将VGG16卷积神经网络导入工作区
inputSize = net.Layers(1).InputSize;             % 获取VGG16卷积神经网络输入层中输入图像的大小
classNames = net.Layers(end).ClassNames;         % 获取VGG16卷积神经网络输出层中的分类

%% 读入RGB图像，并将图像的大小变换成与VGG16卷积神经网络输入层中输入图像相同的大小
I = imread('glassdog.jpg');
figure
imshow(I)
I = imresize(I,inputSize(1:2));


%% 基于VGG16卷积神经网络对输入的图像进行分类
[label1,scores1] = classify(net,I);

%% 在图像上显示分类结果及概率
figure
imshow(I)
title(string(label1) + ", " + num2str(100*scores1(classNames == label1),3) + "%");

