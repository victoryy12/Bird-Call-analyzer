%% Bird Call Classifier
% Identifies bird species from 3-second audio clips

clear; clc;
rng(42);    % makes the random train/test split repeatable

dataFolder = "Bird Call Dataset";


%% 1. Load the labels
clips = readtable(fullfile(dataFolder, "bird_songs_metadata.csv"), "TextType", "string");
clips.path = fullfile(dataFolder, "wavfiles", clips.filename);
clips = clips(isfile(clips.path), :);  

species = categorical(clips.name);
summary(species)


%% 2. Extracts features from every clip and caches them for shorter run time
cacheFile = fullfile(dataFolder, "features_cache.mat");
if isfile(cacheFile)
    load(cacheFile, "features", "cachedPaths");
    if ~isequal(cachedPaths, clips.path)
        clear features  
    end
end
if ~exist("features", "var")
    numClips = height(clips);
    features = zeros(numClips, 36);
    for i = 1:numClips          
        [audio, fs] = audioread(clips.path(i));
        features(i, :) = getFeatures(audio, fs);
    end
    cachedPaths = clips.path;
    save(cacheFile, "features", "cachedPaths");
end


%% 3. Split into training and test sets
recordings = unique(clips.id);
numTest = round(0.2 * numel(recordings));
testRecordings = recordings(randperm(numel(recordings), numTest));
isTest = ismember(clips.id, testRecordings);


%% 4. Train
model = trainModel(features(~isTest, :), species(~isTest));


%% 5. Test
predicted = predictSpecies(model, features(isTest, :));
accuracy = mean(predicted == species(isTest));
fprintf("Test accuracy: %.1f%%\n", 100 * accuracy);


%% 6. Classify audio files
while true
    [file, folder] = uigetfile({'*.wav;*.mp3;*.flac;*.ogg;*.m4a', 'Audio files'}, ...
                               'Choose a bird recording (Cancel to stop)');
    if isequal(file, 0)
        break   
    end

    [audio, fs] = audioread(fullfile(folder, file));
    audio = mean(audio, 2);

    % Cut into 3 second chunks because the model was trained on 3 sec clips
    chunkLen  = round(3 * fs);
    numChunks = max(1, floor(numel(audio) / chunkLen));
    chunkFeatures = zeros(numChunks, 36);
    chunkPeak     = zeros(numChunks, 1);
    for c = 1:numChunks
        chunk = audio((c-1)*chunkLen + 1 : min(c*chunkLen, numel(audio)));
        chunkFeatures(c, :) = getFeatures(chunk, fs);
        chunkPeak(c) = max(abs(chunk));
    end

    loudChunks = chunkPeak >= 0.1 * max(chunkPeak);
    votes = predictSpecies(model, chunkFeatures(loudChunks, :));
    guess = mode(votes);
    result = sprintf("%s (%d of %d chunks)", guess, sum(votes == guess), numel(votes));

    fprintf("%s -> %s\n", file, result);
    msgbox(sprintf("%s\n\nThis sounds like a: %s", file, result), "Bird Classifier");
end



%% ===================== Helper funcs =====================

function f = getFeatures(audio, fs)
    audio = mean(audio, 2);                   
    audio = changeSampleRate(audio, fs, 22050); 
    fs    = 22050;
    audio = audio / (max(abs(audio)) + eps);    

    frameLen = 1024;
    hop      = 512;
    starts   = 1:hop:(numel(audio) - frameLen + 1);
    frames   = audio(starts + (0:frameLen-1)');         
    window   = 0.5 - 0.5 * cos(2*pi*(0:frameLen-1)' / frameLen);
    power    = abs(fft(frames .* window)).^2;
    power    = power(1:frameLen/2 + 1, :);

    melEnergy = log(melFilters(40, frameLen, fs) * power + 1e-10);
    dctMatrix = cos(pi * (0:12)' * ((0:39) + 0.5) / 40);
    mfcc      = dctMatrix * melEnergy;

    freqs = (0:frameLen/2)' * fs / frameLen;
    p     = power(freqs >= 500, :);
    fr    = freqs(freqs >= 500);
    total = sum(p) + eps;

    centroid = sum(fr .* p) ./ total;                           
    spread   = sqrt(sum((fr - centroid).^2 .* p) ./ total);     
    flatness = exp(mean(log(p + eps))) ./ (mean(p) + eps);      
    [~, bin] = max(cumsum(p) ./ total >= 0.85);
    rolloff  = fr(bin)';                                         
    flux     = [0, sqrt(sum(diff(p ./ total, 1, 2).^2))];       
    spectral = [centroid; spread; flatness; rolloff; flux];

    loudness = sum(power);
    loud     = loudness >= median(loudness);
    mfcc     = mfcc(:, loud);
    spectral = spectral(:, loud);

    f = [mean(mfcc, 2); std(mfcc, 0, 2); mean(spectral, 2); std(spectral, 0, 2)]';
end


function audio = changeSampleRate(audio, fs, newFs)
    if fs == newFs
        return
    end
    n    = numel(audio);
    newN = round(n * newFs / fs);
    keep = floor(min(n, newN) / 2);   

    spectrum = fft(audio);
    newSpectrum = zeros(newN, 1);
    newSpectrum(1:keep+1)         = spectrum(1:keep+1);       
    newSpectrum(end-keep+1:end)   = spectrum(end-keep+1:end);    
    audio = real(ifft(newSpectrum)) * newN / n;
end


function filters = melFilters(numFilters, frameLen, fs)
    toMel = @(hz)  2595 * log10(1 + hz / 700);
    toHz  = @(mel) 700 * (10.^(mel / 2595) - 1);

    edges = toHz(linspace(toMel(500), toMel(fs/2), numFilters + 2));
    freqs = (0:frameLen/2) * fs / frameLen;

    filters = zeros(numFilters, numel(freqs));
    for k = 1:numFilters
        rising  = (freqs - edges(k))   / (edges(k+1) - edges(k));
        falling = (edges(k+2) - freqs) / (edges(k+2) - edges(k+1));
        filters(k, :) = max(0, min(rising, falling));
    end
end


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


function label = predictSpecies(model, X)
    X = (X - model.mu) ./ model.sigma;
    hidden = max(0, X * model.W1 + model.b1);
    [~, best] = max(hidden * model.W2 + model.b2, [], 2);
    label = categorical(model.classes(best), model.classes);
end
