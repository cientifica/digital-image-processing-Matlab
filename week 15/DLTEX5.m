%%  ����˵��

% ʵ�� 4.5-1
% ���ܣ���lgraph_1�������·����ѵ�� 
% ���ߣ�zhaoxch_mail@sina.com
% ʱ�䣺2020��3��6��
% �汾��DLTEX5-V1

%% ����1������ͼ�����ݣ������仮��Ϊѵ��������֤��

% ����ͼ������
unzip('MerchData.zip');
imds = imageDatastore('MerchData', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% ������֤����ѵ����
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

% �����ʾѵ�����еĲ���ͼ��
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

%% ����2���������ݼ�

% �鿴���������Ĵ�С��ͨ����
inputSize = lgraph_1(1).InputSize;

% ������ѵ��ͼ��Ĵ�С����Ϊ�������Ĵ�С��ͬ
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
% ��������֤ͼ��Ĵ�С����Ϊ�������Ĵ�С��ͬ
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%% ����3�����������ѵ��

% ��ѵ��������������
options = trainingOptions('sgdm', ...
    'MiniBatchSize',15, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',0.00005, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',true, ...
'Plots','training-progress');

%  ��ѵ��ͼ����������ѵ��
    netTransfer = trainNetwork(augimdsTrain, lgraph_1 ,options);
    
%% ����4����֤����ʾ���

% ��ѵ���õ����������֤���ݼ�������֤
[YPred,scores] = classify(netTransfer,augimdsValidation);

% �����ʾ��֤Ч��
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

%% �������׼ȷ��
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)

%% ��������ʾ��������
figure
confusionchart(YValidation,YPred)
