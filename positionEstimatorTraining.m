%%% Team Members: Tiffany Hamstreet, Ahmet Narman, Junke Yao, Luxi Zhang
%%% BMI Competition Spring 2019 

function modelParameters = positionEstimatorTraining(training_data)
    
    tic;

    winSize = 300;
    binSize = 20;
    
    % Preprocessing the training data and acquiring the input-output pairs
    % that will be used to train the FNN
    [trIn, trOut] = getTrainingData(training_data, winSize, binSize);

    % Parameters
    L1 = 100; % First layer neurons
    L2 = 150; % Second layer neurons
    L3 = 100; % Third layer neurons
    L4 = 50; % Fourth layer neurons
    O = size(trOut,2); % Output dimensions
    lr = 0.01; % Learning Rate
    batch = 30; % Minibatch size
    epoch = 50; % Amount of epochs
    [Len, Dim] = size(trIn); % Amount of datapoints and input dimensions

    % The weights below include bias terms as well
    W1 = normrnd(0,0.1,[L1 Dim+1]); % First layer of weights 
    W2 = normrnd(0,0.1,[L2 L1+1]); % Second layer of weights
    W3 = normrnd(0,0.1,[L3 L2+1]); % Third layer of weights
    W4 = normrnd(0,0.1,[L4 L3+1]); % Output layer of weights 
    W5 = normrnd(0,0.1,[O L4+1]); % Output layer of weights 
    
    % ADAM hyperparameters
    Beta1 = 0.4;
    Beta2 = 0.999;
    eps = 10^-8;
    
    % Momentum terms initialized (mean estimate)
    VdW1 = zeros(L1, Dim+1); 
    VdW2 = zeros(L2, L1+1); 
    VdW3 = zeros(L3, L2+1); 
    VdW4 = zeros(L4, L3+1); 
    VdW5 = zeros(O, L4+1); 
    
    % RSMProp terms initialized (variance estimate)
    SdW1 = zeros(L1, Dim+1); 
    SdW2 = zeros(L2, L1+1); 
    SdW3 = zeros(L3, L2+1); 
    SdW4 = zeros(L4, L3+1); 
    SdW5 = zeros(O, L4+1); 

    iter = 1; % Will be used to track how many batches have passed
    BatchErrVec = zeros(1,batch*epoch); % Storing errors for every batch

    for ep = 1:epoch % Run this loop for every epoch

        ind = randperm(Len); % Randomized data indices for training

        for bat = 1:Len/batch % Run this loop for every batch in epoch

            % Inputs for this batch
            input = trIn(ind((bat-1)*batch+1:bat*batch),:); 
            % Desired outputs for this batch
            desired = trOut(ind((bat-1)*batch+1:bat*batch),:); 
            
            dW1 = zeros(L1, Dim+1); % Change in First layer of weights 
            dW2 = zeros(L2, L1+1); % Change in Second layer of weights
            dW3 = zeros(L3, L2+1); % Change in Third layer of weights 
            dW4 = zeros(L4, L3+1); % Change in Fourth layer of weights 
            dW5 = zeros(O, L4+1); % Change in Output layer of weights 

            MSE = 0; % Mean Squared Error for one datapoint
            totBatchErr = 0; % Total error of the minibatch

            for i= 1:batch % Run this loop for every datapoint in the batch
                % Find weight updates for every datapoint, but don't update

                % Feedforward
                IN = [input(i,:) 1]'; % The input of NN with bias
                Y1 = W1*IN; % Input multiplied with weights matrix
                U1 = [expReLu(Y1); 1]; % 1st hidden layer output after activation
                Y2 = W2*U1; % 1st hidden layer output multiplied with weights
                U2 = [expReLu(Y2); 1]; % 2nd hidden layer output after activation
                Y3 = W3*U2; % 2nd hidden layer output multiplied with weights
                U3 = [expReLu(Y3); 1]; % 3rd hidden layer output after activation
                Y4 = W4*U3; % 1st hidden layer output multiplied with weights
                U4 = [expReLu(Y4); 1]; % 2nd hidden layer output after activation
                OUT = W5*U4; % Final output

                MSE = sum((desired(i,:)' - OUT).^2)/2; % mean squared error
                totBatchErr = totBatchErr + MSE; % Total error of the minibatch

                % Backpropagation

                % Intermediate updating values
                dE = desired(i,:)' - OUT; 
                del1 = dE'*W5(:,1:L4).*dexpReLu(Y4)';
                del2 = del1*W4(:,1:L3).*dexpReLu(Y3)';
                del3 = del2*W3(:,1:L2).*dexpReLu(Y2)';
                del4 = del3*W2(:,1:L1).*dexpReLu(Y1)';

                % Change of weights are sum for a minibatch
                dW5 = dW5 + dE*U4';
                dW4 = dW4 + del1'*U3';
                dW3 = dW3 + del2'*U2';
                dW2 = dW2 + del3'*U1';
                dW1 = dW1 + del4'*IN';
            end

            % Weight update done at the end of minibatch

            % Momentum terms updated
            VdW1 = Beta1*VdW1 + (1-Beta1)*(dW1/batch);
            VdW2 = Beta1*VdW2 + (1-Beta1)*(dW2/batch);
            VdW3 = Beta1*VdW3 + (1-Beta1)*(dW3/batch);
            VdW4 = Beta1*VdW4 + (1-Beta1)*(dW4/batch);
            VdW5 = Beta1*VdW5 + (1-Beta1)*(dW5/batch);
            
            % RSMProp terms updated
            SdW1 = Beta2*SdW1 + (1-Beta2)*(dW1.^2/batch); 
            SdW2 = Beta2*SdW2 + (1-Beta2)*(dW2.^2/batch); 
            SdW3 = Beta2*SdW3 + (1-Beta2)*(dW3.^2/batch); 
            SdW4 = Beta2*SdW4 + (1-Beta2)*(dW4.^2/batch); 
            SdW5 = Beta2*SdW5 + (1-Beta2)*(dW5.^2/batch); 
            
            % WEights are updated using ADAM optimization algorithm
            W5 = W5 + lr*(VdW5/(1-Beta1^iter))./(sqrt(SdW5/(1-Beta2^iter))+eps);
            W4 = W4 + lr*(VdW4/(1-Beta1^iter))./(sqrt(SdW4/(1-Beta2^iter))+eps);
            W3 = W3 + lr*(VdW3/(1-Beta1^iter))./(sqrt(SdW3/(1-Beta2^iter))+eps);
            W2 = W2 + lr*(VdW2/(1-Beta1^iter))./(sqrt(SdW2/(1-Beta2^iter))+eps);
            W1 = W1 + lr*(VdW1/(1-Beta1^iter))./(sqrt(SdW1/(1-Beta2^iter))+eps);

            BatchErrVec(iter) = totBatchErr/batch; % Average MSE error for the batch
            iter = iter + 1; % the iteration number updated for every batch
        end
    end
    trTime = toc; % Training time
    
    % Model parameters are updated and returned
    modelParameters.W1 = W1;
    modelParameters.W2 = W2;
    modelParameters.W3 = W3;
    modelParameters.W4 = W4;
    modelParameters.W5 = W5;
    modelParameters.winSize = winSize;
    modelParameters.binSize = binSize;
end

% Exponential Rectified Linear Unit Activation function
function out = expReLu(in)
    out = max(in, 0);
    idx = out==0;
    out(idx) = 0.1*(exp(in(idx))-1);
end

% Derivative of the activation function
function out = dexpReLu(in)
    out = double(in>=0);
    idx = out==0;
    out(idx) = 0.1*exp(in(idx));
end

function [trIn, trOut]= getTrainingData(data,winSize,binSize)
% This function returns the training inputs and training outputs that will
% be used in the neural network training. 


    [T,A] = size(data); % T is the number of trials, A is the number of angles
    
    N = size(data(1,1).spikes,1); % Can be changed with the results of ANOVA

    numBins = winSize/binSize; % Number of time bins for each neuron
    
    trIn = zeros(100000, N*numBins);    % Alllocating extended memory
    trOut = zeros(100000, 2);           % (Will be cut later)
    
    tempData = zeros(1,N*numBins); % temporary variable to hold data
    
    iter = 1; % To count how many datapoints are generated
    for i = 1:T
        for j = 1:A
            
            len = size(data(i,j).spikes,2); % Length of trial
            times=320:20:len; % Times divided to bins
            
            for t=times
                for n = 1:numBins
                    bin = [t-n*binSize:t-(n-1)*binSize]; % This code looks back from time 't'
                    % Spike sum of every bin was put in the datapoint
                    tempData(N*(n-1)+1:N*n) = sum(data(i,j).spikes(:,bin)');
                    
                end
                % Training input 
                trIn(iter,:) = tempData; 
                
                dx = data(i,j).handPos(1,t)-data(i,j).handPos(1,t-20); % Change in x in 20 ms
                dy = data(i,j).handPos(2,t)-data(i,j).handPos(2,t-20); % Change in y in 20 ms
                
                % Training output (Change in position)
                trOut(iter,:) = [dx dy];
                
                iter = iter+1;
            end                   
        end
    end
    trIn = trIn(1:iter-1,:);    % Removing the unused part of the memory
    trOut = trOut(1:iter-1,:); 
end