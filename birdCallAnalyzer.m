function [species, probs, classes] = birdCallAnalyzer(audioFile, showPlot)
%BIRDCALLANALYZER Identify a bird species from a recording of its call.
%   birdCallAnalyzer                 opens a file picker
%   birdCallAnalyzer('call.wav')     analyzes the given file
%   [species, probs, classes] = birdCallAnalyzer(file, false)   no figure
%
%   Recognized species: Bewick's Wren, Northern Cardinal, American Robin,
%   Song Sparrow, Northern Mockingbird.
%
%   The recording can be any length and sample rate (WAV, MP3, FLAC, ...).
%   It is cut into 3-second windows (the length of the training clips),
%   near-silent windows are skipped, and the class probabilities of the
%   remaining windows are averaged.
%
%   Requires birdModel.mat - run trainBirdClassifier first.

    rootDir = fileparts(mfilename('fullpath'));
    modelFile = fullfile(rootDir, 'birdModel.mat');
    if ~isfile(modelFile)
        error('birdCallAnalyzer:noModel', ...
              'birdModel.mat not found. Run trainBirdClassifier first.');
    end
    S = load(modelFile, 'model');
    model = S.model;
    classes = model.classes;

    if nargin < 1 || isempty(audioFile)
        [fn, fp] = uigetfile({'*.wav;*.mp3;*.flac;*.ogg;*.m4a', 'Audio files'}, ...
                             'Select a bird call recording', ...
                             fullfile(rootDir, 'Bird Call Dataset', 'wavfiles'));
        if isequal(fn, 0), species = ""; probs = []; return; end
        audioFile = fullfile(fp, fn);
    end
    if nargin < 2, showPlot = true; end

    % ---- Load, convert to mono, resample to the training rate -----------
    [x, fsIn] = audioread(audioFile);
    x = mean(x, 2);
    fs = model.fs;
    if fsIn ~= fs
        x = fftResample(x, fsIn, fs);
    end

    % ---- Split into 3-second windows and classify each ------------------
    winLen = 3*fs;
    hop    = round(1.5*fs);
    if numel(x) < winLen, x(end+1:winLen) = 0; end
    starts = 1:hop:(numel(x) - winLen + 1);
    if starts(end) + winLen - 1 < numel(x)        % cover the tail
        starts(end+1) = numel(x) - winLen + 1;
    end
    nSeg = numel(starts);
    rmsSeg = zeros(nSeg, 1);
    F = [];
    for k = 1:nSeg
        seg = x(starts(k):starts(k)+winLen-1);
        rmsSeg(k) = sqrt(mean(seg.^2));
        f = extractBirdFeatures(seg, fs);
        if isempty(F), F = zeros(nSeg, numel(f)); end
        F(k, :) = f;
    end
    use = rmsSeg >= 0.25*max(rmsSeg);             % skip quiet windows
    Pseg  = mlpPredict(model, F);
    probs = mean(Pseg(use, :), 1);
    [~, best] = max(probs);
    species = classes(best);

    % ---- Report ---------------------------------------------------------
    [~, name, ext] = fileparts(audioFile);
    fprintf('\nFile: %s%s  (%.1f s, %d of %d windows used)\n', ...
            name, ext, numel(x)/fs, nnz(use), nSeg);
    [ps, order] = sort(probs, 'descend');
    for k = 1:numel(order)
        fprintf('  %-22s %5.1f%%\n', classes(order(k)), 100*ps(k));
    end
    fprintf('Prediction: %s\n', species);

    if showPlot
        plotResult(x, fs, classes, probs, best, species, [name ext]);
    end
end

%% ------------------------------------------------------------------------
function plotResult(x, fs, classes, probs, best, species, titleStr)
    fig = figure('Name', 'Bird Call Analyzer', 'Position', [100 100 1000 650]);
    tl = tiledlayout(fig, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, sprintf('%s  \\rightarrow  %s (%.0f%%)', ...
          strrep(titleStr, '_', '\_'), species, 100*probs(best)), ...
          'FontWeight', 'bold');

    t = (0:numel(x)-1)/fs;
    ax1 = nexttile(tl);
    plot(ax1, t, x, 'Color', [0.2 0.4 0.7]);
    xlim(ax1, [0 t(end)]); ylabel(ax1, 'Amplitude'); title(ax1, 'Waveform');

    % Spectrogram (manual STFT - no Signal Processing Toolbox needed)
    nfft = 512; hop = 128;
    w = 0.5 - 0.5*cos(2*pi*(0:nfft-1)'/(nfft-1));
    nF = max(1, 1 + floor((numel(x)-nfft)/hop));
    xp = [x; zeros(max(0, nfft - numel(x)), 1)];
    idx = (1:nfft)' + (0:nF-1)*hop;
    Sx = abs(fft(xp(idx).*w, nfft));
    Sx = 20*log10(Sx(1:nfft/2+1, :) + 1e-6);
    ax2 = nexttile(tl);
    imagesc(ax2, ((0:nF-1)*hop + nfft/2)/fs, (0:nfft/2)*fs/nfft/1000, Sx);
    axis(ax2, 'xy'); colormap(ax2, 'parula');
    mx = max(Sx(:)); clim(ax2, [mx-80 mx]);
    ylabel(ax2, 'Frequency (kHz)'); xlabel(ax2, 'Time (s)');
    title(ax2, 'Spectrogram'); linkaxes([ax1 ax2], 'x');

    ax3 = nexttile(tl);
    b = barh(ax3, categorical(classes, classes), 100*probs, 'FaceColor', 'flat');
    b.CData = repmat([0.7 0.7 0.7], numel(classes), 1);
    b.CData(best, :) = [0.85 0.33 0.1];
    xlim(ax3, [0 100]); xlabel(ax3, 'Probability (%)');
    title(ax3, 'Species probabilities');
end

function y = fftResample(x, fsIn, fsOut)
% Band-limited resampling via the FFT (base MATLAB replacement for resample).
    n  = numel(x);
    m  = round(n * fsOut / fsIn);
    X  = fft(x);
    Y  = zeros(m, 1);
    h  = floor(min(n, m)/2);
    Y(1:h) = X(1:h);
    Y(end-h+1:end) = X(end-h+1:end);
    y  = real(ifft(Y)) * (m / n);
end
