%%  ����˵��
% ʵ�� 4.8-3
% ���ܣ���������·���ӵľ�������磬�������ͼ����з���
% ���ߣ�zhaoxch_mail@sina.com
% ʱ�䣺2020��3��22��
% �汾��DLTEX803-V1

%% ����ڴ桢�����Ļ
clear
clc
%% �������������
layers = [
    imageInputLayer([28 28 1],'Name','input')
    
    convolution2dLayer(5,16,'Padding','same','Name','conv_1')
    reluLayer('Name','relu_1')
    
    convolution2dLayer(3,16,'Padding','same','Stride',1,'Name','conv_2')
    reluLayer('Name','relu_2') 
    additionLayer(2,'Name','add1')
    
    convolution2dLayer(3,16,'Padding','same','Stride',2,'Name','conv_3')
    reluLayer('Name','relu_3') 
    additionLayer(2,'Name','add2')
    
    fullyConnectedLayer(10,'Name','fc')
    softmaxLayer('Name','softmax');
    classificationLayer('Name','output')];

lgraph = layerGraph(layers);
 
%% ������·���Ӳ�
skipConv = convolution2dLayer(1,16,'Stride',2,'Name','skipConv');
lgraph = addLayers(lgraph,skipConv);

%% ��������
lgraph = connectLayers(lgraph,'relu_1','add1/in2');
lgraph = connectLayers(lgraph,'add1','skipConv');
lgraph = connectLayers(lgraph,'skipConv','add2/in2');


%% ����ѵ������֤����
[XTrain,YTrain] = digitTrain4DArrayData;
[XValidation,YValidation] = digitTest4DArrayData;

%% ����ѵ��������ѵ������
options = trainingOptions('sgdm', ...
    'MaxEpochs',50, ...
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