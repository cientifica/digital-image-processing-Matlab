%%  ����˵��
% ʵ�� 5.4-3
% ���ܣ�����Image Labeler������ݵ�R-CNNĿ����������
% ���ߣ�zhaoxch_mail@sina.com
% ʱ�䣺2020��4��19��
% �汾��DLTEXC543-V1

%%  �����������͵�ת��
trainingdate=objectDetectorTrainingData(gTruth);
%%  ��������
net=alexnet;
%%  ����ѵ�����Բ���������ѵ��
% ����ѵ�����Բ���
options = trainingOptions('sgdm', ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 100, ...
        'MaxEpochs',10, ...
        'Verbose', true);
    
  % ѵ������.    
    rcnn = trainRCNNObjectDetector(trainingdate, net, options, ...
    'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1]) 

%%  ��ʾ���Խ��
% ��ȡ����
I = imread('stoptest.jpg');
% �ü��������
[bboxes,scores] = detect(rcnn,I);
% ��ע���Խ������ʾ
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)
