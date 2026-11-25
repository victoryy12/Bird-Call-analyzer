function model = trainModel(X, y)
    model.classes = categories(y);
    model.mu      = mean(X);
    model.sigma   = std(X);

    X = (X - model.mu) ./ model.sigma;
    target = double(y) == 1:numel(model.classes);   
    numHidden = 64;
    learnRate = 0.5;
    lambda    = 0.01;   
    n = size(X, 1);

    model.W1 = randn(size(X, 2), numHidden) * sqrt(2 / size(X, 2));
    model.b1 = zeros(1, numHidden);
    model.W2 = randn(numHidden, numel(model.classes)) * sqrt(1 / numHidden);
    model.b2 = zeros(1, numel(model.classes));

    for step = 1:1000
        hidden = max(0, X * model.W1 + model.b1);
        scores = hidden * model.W2 + model.b2;
        probs  = exp(scores - max(scores, [], 2));
        probs  = probs ./ sum(probs, 2);

        dScores = (probs - target) / n;
        dHidden = (dScores * model.W2') .* (hidden > 0);

        model.W2 = model.W2 - learnRate * (hidden' * dScores + lambda * model.W2);
        model.b2 = model.b2 - learnRate * sum(dScores);
        model.W1 = model.W1 - learnRate * (X' * dHidden + lambda * model.W1);
        model.b1 = model.b1 - learnRate * sum(dHidden);
    end
end
