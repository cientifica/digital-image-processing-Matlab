%%  ����˵��
% ʵ�� 4.8-1
% ���ܣ���������·���ӵľ�������磬�Ժ���0~9���ֵĶ�ֵͼ������Ϊ28��28�����з��࣬���������׼ȷ��
% ���ߣ�zhaoxch_mail@sina.com
% ʱ�䣺2020��3��22��
% �汾��DLTEX801-V1

%% ����ڴ桢�����Ļ
clear
clc

%% �������������
layers = [
    imageInputLayer([28 28 1],'Name','input')
    
    convolution2dLayer(5,16,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')
    
    convolution2dLayer(3,32,'Padding','same','Stride',2,'Name','conv_2')
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','relu_2')
    convolution2dLayer(3,32,'Padding','same','Name','conv_3')
    batchNormalizationLayer('Name','BN_3')
    reluLayer('Name','relu_3')
    
    additionLayer(2,'Name','add')
    
    averagePooling2dLayer(2,'Stride',2,'Name','avpool')
    fullyConnectedLayer(10,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classOutput')];

% ��������ʾ����
lgraph = layerGraph(layers);
figure
plot(lgraph)

%% ������·���Ӳ�
skipConv = convolution2dLayer(1,32,'Padding','same','Stride',2,'Name','skipConv');
lgraph = addLayers(lgraph,skipConv);
figure
plot(lgraph)

%% �������Ӳ���������ṹͼ��
lgraph = connectLayers(lgraph,'relu_1','skipConv');
lgraph = connectLayers(lgraph,'skipConv','add/in2');
figure
plot(lgraph);

%% ����ѵ������֤����
[XTrain,YTrain] = digitTrain4DArrayData;
[XValidation,YValidation] = digitTest4DArrayData;

%% ����ѵ��������ѵ������
options = trainingOptions('sgdm', ...
    'MaxEpochs',5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(XTrain,YTrain,lgraph,options);

%% ��ʾ������Ϣ
net

%% ����֤�����з��ಢ����׼ȷ��
YPredicted = classify(net,XValidation);
accuracy = mean(YPredicted == YValidation)
