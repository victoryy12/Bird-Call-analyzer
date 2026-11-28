%% Bird Call Classifier
% Identifies bird species from 3-second audio clips

clear; clc;
rng(42);    % makes the random train/test split repeatable

dataFolder = "Bird Call Dataset";


%% 1. Load the labels
clips = readtable(fullfile(dataFolder, "bird_songs_metadata.csv"), "TextType", "string");
clips.path = fullfile(dataFolder, "wavfiles", clips.filename);
clips = clips(isfile(clips.path), :);  
if ~any(isfile(clips.path))
    error("Download the dataset into ""Bird Call Dataset/wavfiles"" first (see README).")
end

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
save("birdModel.mat", "model")

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
