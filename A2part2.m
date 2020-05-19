imds = imageDatastore('C:\Users\diego\OneDrive - UTS\UTS\Year 4\Sem 1\Neural Net - fuss logic\A2\images', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); % load the images and labels
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.5   ); %split training and 
%validation datasets, training is 60%

layers = [
    imageInputLayer([32 32 3])
    
    convolution2dLayer(3,24,'Stride',1,'Padding',1)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)% the pooling kernel size is 2*2, stride is 2
    
    convolution2dLayer(3,30,'Stride',1,'Padding',1)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,36,'Stride',1,'Padding',1)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,36,'Stride',1,'Padding',1)    
    reluLayer
    fullyConnectedLayer(10)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
 %  'InitialLearnRate',0.001, ...
 %'InitialLearnRate',0.001, 'Momentum',0.4, ...
 options = trainingOptions('adam', ...
     'InitialLearnRate',0.01, ...
     'MaxEpochs',30, ...
     'ValidationData',imdsValidation, ...
     'ValidationFrequency',2, ...
     'Plots','training-progress');
 net = trainNetwork(imdsTrain,layers,options);
 
 net.Layers  % display all the layers of this structure
    
    
    
    
    