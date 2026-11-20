function results = testGenerator(nPerSpecies, seedOffset)
%TESTGENERATOR Closed-loop test: does the analyzer recognize synthetic calls?
%   testGenerator          writes 10 synthetic songs per species to
%                          'Synthetic Calls/', classifies each one with
%                          birdCallAnalyzer, and prints a confusion matrix.
%   testGenerator(n)       n songs per species.
%   testGenerator(n, k)    shift every seed by k to get a fresh set of songs.
%   results = testGenerator(...)  also returns one struct per song
%                          (file, species, predicted, confidence).
%
%   If the real dataset is in 'Bird Call Dataset/wavfiles', it also draws a
%   real recording next to a synthetic one for each species, which makes a
%   good slide for the presentation.
%
%   Requires birdModel.mat (run trainBirdClassifier first).
%
%   See also birdCallGenerator, birdCallAnalyzer.

    if nargin < 1 || isempty(nPerSpecies), nPerSpecies = 10; end
    if nargin < 2 || isempty(seedOffset),  seedOffset  = 0;  end
    rootDir = fileparts(mfilename('fullpath'));
    if isempty(rootDir), rootDir = pwd; end
    outDir  = fullfile(rootDir, 'Synthetic Calls');

    % ---- 1. Generate -----------------------------------------------------
    [~, ~, info] = birdCallGenerator('all', 'Count', nPerSpecies, 'OutDir', outDir, ...
                                     'Seed', seedOffset);

    % ---- 2. Classify each file with the real analyzer ---------------------
    S = load(fullfile(rootDir, 'birdModel.mat'), 'model');
    classes = cellstr(S.model.classes);
    K = numel(classes);
    N = numel(info.files);
    truth = zeros(N, 1);  pred = zeros(N, 1);  conf = zeros(N, 1);
    results = struct('file', info.files, 'species', info.species, ...
                     'predicted', '', 'confidence', 0);
    sp = '';  p = [];          % filled in by evalc below (it hides the analyzer's printout)
    fprintf('Classifying %d synthetic calls', N);
    for i = 1:N
        file = info.files{i};                                    %#ok<NASGU> used in evalc
        txt  = evalc('[sp, p] = birdCallAnalyzer(file, false);'); %#ok<NASGU>
        truth(i) = find(strcmp(classes, info.species{i}));
        pred(i)  = find(strcmp(classes, char(sp)));
        conf(i)  = p(truth(i));
        results(i).predicted  = char(sp);
        results(i).confidence = p(pred(i));
        if mod(i, 5) == 0, fprintf('.'); end
    end
    fprintf('\n');

    % ---- 3. Report -------------------------------------------------------
    C = accumarray([truth pred], 1, [K K]);
    codes = {'BEWR', 'NOCA', 'AMRO', 'SOSP', 'NOMO'};
    fprintf('\nSynthetic calls recognized by the analyzer: %d of %d (%.0f%%)\n', ...
            trace(C), N, 100*trace(C)/N);
    fprintf('\nConfusion matrix (rows = species generated, columns = species predicted):\n');
    fprintf('%-22s', '');  fprintf('%7s', codes{:});  fprintf('   correct   avg P(true)\n');
    for c = 1:K
        m = truth == c;
        fprintf('%-22s', classes{c});  fprintf('%7d', C(c, :));
        fprintf('   %5.0f%%     %.2f\n', 100*C(c,c)/max(1, sum(C(c,:))), mean(conf(m)));
    end

    % ---- 4. Optional figure: real vs synthetic ----------------------------
    wavDir = fullfile(rootDir, 'Bird Call Dataset', 'wavfiles');
    if exist(wavDir, 'dir') && usejava('desktop')
        plotRealVsSynthetic(rootDir, wavDir, classes, info);
    end
end

%% ------------------------------------------------------------------------
function plotRealVsSynthetic(rootDir, wavDir, classes, info)
% Top row: one real clip per species. Bottom row: one synthetic clip.
    meta = readtable(fullfile(rootDir, 'Bird Call Dataset', 'bird_songs_metadata.csv'), ...
                     'TextType', 'string', 'Delimiter', ',');
    figure('Name', 'Real vs synthetic bird calls', 'Position', [50 100 1500 520]);
    K = numel(classes);
    for c = 1:K
        realFile = fullfile(wavDir, meta.filename(find(meta.name == classes{c}, 1)));
        synFile  = info.files{find(strcmp(info.species, classes{c}), 1)};
        files    = {realFile, synFile};
        labels   = {'real', 'synthetic'};
        for r = 1:2
            [x, fs] = audioread(files{r});
            x = mean(x, 2);
            [Sx, tt, ff] = stftDb(x, fs);
            ax = subplot(2, K, (r-1)*K + c);
            imagesc(ax, tt, ff/1000, Sx);
            axis(ax, 'xy');  colormap(ax, parula);
            mx = max(Sx(:));  set(ax, 'CLim', [mx-70 mx]);
            ylim(ax, [0 10]);
            title(ax, sprintf('%s (%s)', classes{c}, labels{r}));
            if c == 1, ylabel(ax, 'Frequency (kHz)'); end
            if r == 2, xlabel(ax, 'Time (s)'); end
        end
    end
end

function [S, tt, ff] = stftDb(x, fs)
    nfft = 512;  hop = 128;
    w  = 0.5 - 0.5*cos(2*pi*(0:nfft-1)'/(nfft-1));
    nF = max(1, 1 + floor((numel(x) - nfft)/hop));
    xp = [x(:); zeros(max(0, nfft - numel(x)), 1)];
    idx = (1:nfft)' + (0:nF-1)*hop;
    S  = abs(fft(xp(idx) .* w, nfft));
    S  = 20*log10(S(1:nfft/2+1, :) + 1e-6);
    tt = ((0:nF-1)*hop + nfft/2) / fs;
    ff = (0:nfft/2) * fs / nfft;
end
