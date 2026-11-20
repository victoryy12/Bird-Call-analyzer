%TRAINBIRDCLASSIFIER Train the bird call classifier on the Bird Call Dataset.
%   Run this script once. It:
%     1. Extracts features from every labelled clip (cached in birdFeatures.mat)
%     2. Splits the data 70/15/15 by *recording* so clips from the same
%        recording never appear in both training and test sets
%     3. Trains a one-hidden-layer neural network (base MATLAB, no toolboxes)
%     4. Reports test accuracy and a confusion matrix
%     5. Saves the model to birdModel.mat for use by birdCallAnalyzer

rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir), rootDir = pwd; end
dataDir  = fullfile(rootDir, 'Bird Call Dataset');
wavDir   = fullfile(dataDir, 'wavfiles');
featFile = fullfile(rootDir, 'birdFeatures.mat');
rng(0);

%% 1. Load metadata and extract features
meta = readtable(fullfile(dataDir, 'bird_songs_metadata.csv'), ...
                 'TextType', 'string', 'Delimiter', ',');
classes = ["Bewick's Wren", "Northern Cardinal", "American Robin", ...
           "Song Sparrow", "Northern Mockingbird"];
keep = ismember(meta.name, classes);
meta = meta(keep, :);
[~, y] = ismember(meta.name, classes);
recId = meta.id;
N = height(meta);

if isfile(featFile)
    S = load(featFile);
    if isequal(S.files, meta.filename)
        F = S.F;
        fprintf('Loaded cached features for %d clips.\n', N);
    end
end
if ~exist('F', 'var')
    fprintf('Extracting features from %d clips...\n', N);
    F = [];
    tic;
    for i = 1:N
        [x, fs] = audioread(fullfile(wavDir, meta.filename(i)));
        f = extractBirdFeatures(mean(x, 2), fs);
        if isempty(F), F = zeros(N, numel(f)); end
        F(i, :) = f;
        if mod(i, 500) == 0
            fprintf('  %d / %d  (%.0f s)\n', i, N, toc);
        end
    end
    files = meta.filename;
    save(featFile, 'F', 'files');
end

%% 2. Recording-level stratified split (70% train / 15% val / 15% test)
split = zeros(N, 1);   % 1 = train, 2 = val, 3 = test
for c = 1:numel(classes)
    ids = unique(recId(y == c));
    ids = ids(randperm(numel(ids)));
    nId = numel(ids);
    nTe = round(0.15*nId);  nVa = round(0.15*nId);
    split(ismember(recId, ids(1:nTe)))           = 3;
    split(ismember(recId, ids(nTe+1:nTe+nVa)))   = 2;
    split(ismember(recId, ids(nTe+nVa+1:end)))   = 1;
end
tr = split == 1;  va = split == 2;  te = split == 3;
fprintf('Clips: %d train, %d validation, %d test (%d recordings total)\n', ...
        nnz(tr), nnz(va), nnz(te), numel(unique(recId)));

%% 3. Train the neural network
mu    = mean(F(tr, :), 1);
sigma = std(F(tr, :), 0, 1) + 1e-8;
Xtr = (F(tr,:) - mu) ./ sigma;   ytr = y(tr);
Xva = (F(va,:) - mu) ./ sigma;   yva = y(va);

net = trainMLP(Xtr, ytr, Xva, yva, numel(classes));

model = net;
model.mu = mu;
model.sigma = sigma;
model.classes = classes;
model.fs = 22050;

%% 4. Evaluate on the held-out test set
P    = mlpPredict(model, F(te, :));
[~, yhat] = max(P, [], 2);
yte  = y(te);
acc  = mean(yhat == yte);
fprintf('\nTest accuracy (per 3-second clip): %.1f%%\n', 100*acc);

% Recording-level accuracy: average the clip probabilities per recording
teIds = unique(recId(te));
recCorrect = false(numel(teIds), 1);
teRec = recId(te);
for k = 1:numel(teIds)
    m = teRec == teIds(k);
    [~, p] = max(mean(P(m, :), 1));
    recCorrect(k) = p == mode(yte(m));
end
fprintf('Test accuracy (per whole recording): %.1f%%  (%d recordings)\n', ...
        100*mean(recCorrect), numel(teIds));

K = numel(classes);
C = accumarray([yte yhat], 1, [K K]);
fprintf('\nConfusion matrix (rows = true, columns = predicted):\n');
fprintf('%-22s', ''); fprintf('%8s', 'BEWR', 'NOCA', 'AMRO', 'SOSP', 'NOMO'); fprintf('\n');
for c = 1:K
    fprintf('%-22s', classes(c)); fprintf('%8d', C(c, :));
    fprintf('   (%.0f%% correct)\n', 100*C(c,c)/sum(C(c,:)));
end

model.testAccuracy = acc;
model.confusion = C;
save(fullfile(rootDir, 'birdModel.mat'), 'model');
fprintf('\nModel saved to birdModel.mat\n');

if usejava('desktop')
    figure('Name', 'Test set confusion matrix');
    confusionchart(C, classes, 'RowSummary', 'row-normalized', ...
                   'Title', sprintf('Test accuracy %.1f%%', 100*acc));
end

%% ------------------------------------------------------------------------
function net = trainMLP(X, y, Xva, yva, K)
% One hidden layer (ReLU) + softmax, trained with Adam, dropout, weight
% decay, input noise, and early stopping on validation loss.
    H        = 256;
    lr       = 1e-3;
    wd       = 1e-4;
    dropP    = 0.4;
    noiseStd = 0.15;
    batch    = 64;
    maxEpoch = 300;
    patience = 25;

    [N, D] = size(X);
    Y = full(sparse(1:N, y, 1, N, K));

    W1 = randn(D, H) * sqrt(2/D);  b1 = zeros(1, H);
    W2 = randn(H, K) * sqrt(2/H);  b2 = zeros(1, K);
    params = {W1, b1, W2, b2};
    m = cellfun(@(p) zeros(size(p)), params, 'UniformOutput', false);
    v = m;  t = 0;  beta1 = 0.9;  beta2 = 0.999;

    best = struct('loss', inf, 'params', {params}, 'epoch', 0);
    for epoch = 1:maxEpoch
        order = randperm(N);
        for s = 1:batch:N
            id = order(s:min(s+batch-1, N));
            Xb = X(id,:) + noiseStd*randn(numel(id), D);
            Yb = Y(id,:);
            [W1, b1, W2, b2] = params{:};

            Z1 = Xb*W1 + b1;  A1 = max(Z1, 0);
            M  = (rand(size(A1)) > dropP) / (1 - dropP);
            A1 = A1 .* M;
            Z2 = A1*W2 + b2;  Z2 = Z2 - max(Z2, [], 2);
            Pb = exp(Z2) ./ sum(exp(Z2), 2);

            n   = numel(id);
            dZ2 = (Pb - Yb) / n;
            gW2 = A1'*dZ2 + wd*W2;   gb2 = sum(dZ2, 1);
            dZ1 = (dZ2*W2') .* M .* (Z1 > 0);
            gW1 = Xb'*dZ1 + wd*W1;   gb1 = sum(dZ1, 1);
            grads = {gW1, gb1, gW2, gb2};

            t = t + 1;
            for j = 1:4
                m{j} = beta1*m{j} + (1-beta1)*grads{j};
                v{j} = beta2*v{j} + (1-beta2)*grads{j}.^2;
                mh = m{j} / (1 - beta1^t);
                vh = v{j} / (1 - beta2^t);
                params{j} = params{j} - lr * mh ./ (sqrt(vh) + 1e-8);
            end
        end

        % validation
        [W1, b1, W2, b2] = params{:};
        Zv = max(Xva*W1 + b1, 0)*W2 + b2;  Zv = Zv - max(Zv, [], 2);
        Pv = exp(Zv) ./ sum(exp(Zv), 2);
        vLoss = -mean(log(Pv(sub2ind(size(Pv), (1:numel(yva))', yva)) + 1e-12));
        [~, pv] = max(Pv, [], 2);
        if vLoss < best.loss
            best.loss = vLoss;  best.params = params;  best.epoch = epoch;
        end
        if mod(epoch, 10) == 0
            fprintf('  epoch %3d  val loss %.3f  val acc %.1f%%\n', ...
                    epoch, vLoss, 100*mean(pv == yva));
        end
        if epoch - best.epoch >= patience, break; end
    end
    fprintf('Best validation loss %.3f at epoch %d\n', best.loss, best.epoch);
    [net.W1, net.b1, net.W2, net.b2] = best.params{:};
end
