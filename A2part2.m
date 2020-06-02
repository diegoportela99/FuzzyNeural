currentFolder = pwd;
currentData = append(currentFolder, '\dogs_vs_cats');

imds = imageDatastore(currentData, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); % load the images and labels

imds.ReadFcn = @customReadDatastoreImage;
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7); %split training and 
%validation datasets

layers = [
    imageInputLayer([64 64 3])
    convolution2dLayer(5,24,'Stride',1,'Padding',2)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)% the pooling kernel size is 2*2, stride is 2
    
    convolution2dLayer(5,28,'Stride',1,'Padding',2)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(5,32,'Stride',1,'Padding',2)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(5,36,'Stride',1,'Padding',2)    
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(10)
    fullyConnectedLayer(2)
    softmaxLayer
    
    classificationLayer];
 %  'InitialLearnRate',0.001, ...
 %'InitialLearnRate',0.001, 'Momentum',0.4, ...
 options = trainingOptions('adam', ...
     'ExecutionEnvironment','auto',...
     'InitialLearnRate',0.001, ...
     'MaxEpochs',50, ...
     'MiniBatch',50, ...
     'ValidationData',imdsValidation, ...
     'shuffle', 'every-epoch', ...
     'ValidationFrequency',10, ...
     'Plots','training-progress');
 net = trainNetwork(imdsTrain,layers,options);
 net.Layers  % display all the layers of this structure
    
    
function data = customReadDatastoreImage(filename)
    % code from default function: 
    onState = warning('off', 'backtrace'); 
    c = onCleanup(@() warning(onState)); 
    data = imread(filename); % added lines: 
    %turn_gray = rgb2gray(reading);
    %data = imsharpen(reading); %sharpen the image
    data = data(:,:,min(1:3, end)); 
    data = imresize(data,[64 64]);
end