%%  程序说明

% 实例 4.4-1
% 功能：对covnet1卷积神经网路进行训练 
% 作者：zhaoxch_mail@sina.com
% 时间：2020年3月7日
% 版本：DLTEX4-V1

%% 步骤1：加载图像样本数据，并显示其中的部分图像
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
figure;
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end

%% 步骤2：将加载的图像样本分为训练集和测试集（注：在本例中，训练集的数量为750幅，剩余的为测试集）
numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

%% 步骤3：配置训练选项并开始训练
    
    % 配置训练选项 
    options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.0005, ...
    'MaxEpochs',6, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',true, ...
    'Plots','training-progress');

    % 对网络进行训练                                           
    net = trainNetwork(imdsTrain,covnet1,options); 
    
   %% 步骤4：将训练好的网络用于对新的输入图像进行分类，并计算准确率
   YPred = classify(net,imdsValidation);
   YValidation = imdsValidation.Labels;
   accuracy = sum(YPred == YValidation)/numel(YValidation)
