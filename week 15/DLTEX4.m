%%  ����˵��

% ʵ�� 4.4-1
% ���ܣ���covnet1�������·����ѵ�� 
% ���ߣ�zhaoxch_mail@sina.com
% ʱ�䣺2020��3��7��
% �汾��DLTEX4-V1

%% ����1������ͼ���������ݣ�����ʾ���еĲ���ͼ��
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

%% ����2�������ص�ͼ��������Ϊѵ�����Ͳ��Լ���ע���ڱ����У�ѵ����������Ϊ750����ʣ���Ϊ���Լ���
numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

%% ����3������ѵ��ѡ���ʼѵ��
    
    % ����ѵ��ѡ�� 
    options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.0005, ...
    'MaxEpochs',6, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',true, ...
    'Plots','training-progress');

    % ���������ѵ��                                           
    net = trainNetwork(imdsTrain,covnet1,options); 
    
   %% ����4����ѵ���õ��������ڶ��µ�����ͼ����з��࣬������׼ȷ��
   YPred = classify(net,imdsValidation);
   YValidation = imdsValidation.Labels;
   accuracy = sum(YPred == YValidation)/numel(YValidation)
