%%% Team Members: Tiffany Hamstreet, Ahmet Narman, Junke Yao, Luxi Zhang
%%% BMI Competition Spring 2019 

function [x, y] = positionEstimator(test_data, param)

    % Preprocessing the testing data to use it in the FNN
    teIn = getTestingData(test_data,param.winSize,param.binSize);
    
    % Feedforward
    IN = [teIn 1]'; % The input of NN with bias
    Y1 = param.W1*IN; % Input multiplied with weights matrix
    U1 = [expReLu(Y1); 1]; % 1st hidden layer output after activation
    Y2 = param.W2*U1; % 1st hidden layer output multiplied with weights
    U2 = [expReLu(Y2); 1]; % 2nd hidden layer output after activation
    Y3 = param.W3*U2; % 2nd hidden layer output multiplied with weights
    U3 = [expReLu(Y3); 1]; % 3rd hidden layer output after activation
    Y4 = param.W4*U3; % 3rd hidden layer output multiplied with weights
    U4 = [expReLu(Y4); 1]; % 4th hidden layer output after activation
    OUT = param.W5*U4; % Final output (dx, dy) 
    
    
    if isempty(test_data.decodedHandPos)
        % If there is no decoded hand position yet, this is the start of
        % the testing and the next position will be calculated with respect
        % to the starting position
        
        x = test_data.startHandPos(1) + OUT(1);
        y = test_data.startHandPos(2) + OUT(2);
    else
        % If decoded hand position exists, the next position will be
        % calculated according to that
        
        x = test_data.decodedHandPos(1,end) + OUT(1);
        y = test_data.decodedHandPos(2,end) + OUT(2);
    end

end

% Exponential Rectified Linear Unit Activation function
function out = expReLu(in)
    out = max(in, 0);
    idx = out==0;
    out(idx) = 0.1*(exp(in(idx))-1);
end

function teIn = getTestingData(data,winSize,binSize)
% This function returns the testing inputs that will
% be used in the neural network architecture. It 
% generates one testing feature set that will be fed
% to the neural network
    
    [N, len] = size(data(1,1).spikes); 

    numBins = winSize/binSize; 
    
    teIn = zeros(1,N*numBins);
    
    for n = 1:numBins
        bin = [len-n*binSize:len-(n-1)*binSize];
        teIn(N*(n-1)+1:N*n) = sum(data(1,1).spikes(:,bin)');
    end
    
end

