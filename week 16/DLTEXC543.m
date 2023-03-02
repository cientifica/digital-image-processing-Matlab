%%  程序说明
% 实例 5.4-3
% 功能：基于Image Labeler输出数据的R-CNN目标检测器构建
% 作者：zhaoxch_mail@sina.com
% 时间：2020年4月19日
% 版本：DLTEXC543-V1

%%  进行数据类型的转化
trainingdate=objectDetectorTrainingData(gTruth);
%%  导入网络
net=alexnet;
%%  设置训练策略参数并进行训练
% 设置训练策略参数
options = trainingOptions('sgdm', ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 100, ...
        'MaxEpochs',10, ...
        'Verbose', true);
    
  % 训练网络.    
    rcnn = trainRCNNObjectDetector(trainingdate, net, options, ...
    'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1]) 

%%  显示测试结果
% 读取数据
I = imread('stoptest.jpg');
% 用检测器测试
[bboxes,scores] = detect(rcnn,I);
% 标注测试结果并显示
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)
