%%  程序说明

% 实例 4.5-1
% 功能：对lgraph_1卷积神经网路进行训练 
% 作者：zhaoxch_mail@sina.com
% 时间：2020年3月6日
% 版本：DLTEX5-V1

%% 步骤1：加载图像数据，并将其划分为训练集和验证集

% 加载图像数据
unzip('MerchData.zip');
imds = imageDatastore('MerchData', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% 划分验证集和训练集
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

% 随机显示训练集中的部分图像
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

%% 步骤2：调整数据集

% 查看网络输入层的大小和通道数
inputSize = lgraph_1(1).InputSize;

% 将批量训练图像的大小调整为与输入层的大小相同
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
% 将批量验证图像的大小调整为与输入层的大小相同
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%% 步骤3：对网络进行训练

% 对训练参数进行设置
options = trainingOptions('sgdm', ...
    'MiniBatchSize',15, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',0.00005, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',true, ...
'Plots','training-progress');

%  用训练图像对网络进行训练
    netTransfer = trainNetwork(augimdsTrain, lgraph_1 ,options);
    
%% 步骤4：验证并显示结果

% 对训练好的网络采用验证数据集进行验证
[YPred,scores] = classify(netTransfer,augimdsValidation);

% 随机显示验证效果
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

%% 计算分类准确率
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)

%% 创建并显示混淆矩阵
figure
confusionchart(YValidation,YPred)
