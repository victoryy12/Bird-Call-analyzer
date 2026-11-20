function P = mlpPredict(model, F)
%MLPPREDICT Class probabilities from the trained bird call network.
%   P = mlpPredict(model, F) takes raw feature rows F (N x D, as returned by
%   extractBirdFeatures) and returns N x K class probabilities.

    X = (F - model.mu) ./ model.sigma;
    H = max(X * model.W1 + model.b1, 0);       % ReLU hidden layer
    Z = H * model.W2 + model.b2;
    Z = Z - max(Z, [], 2);
    P = exp(Z) ./ sum(exp(Z), 2);              % softmax
end
