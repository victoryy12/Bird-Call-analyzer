function f = extractBirdFeatures(x, fs)
%EXTRACTBIRDFEATURES Fixed-length acoustic feature vector for a bird call.
%   f = extractBirdFeatures(x, fs) returns a 1xD row vector describing the
%   audio signal x (mono) sampled at fs Hz. Uses only base MATLAB.
%
%   Features (computed on the louder, "active" frames of the clip):
%     - 20 MFCCs: mean and std
%     - 20 delta-MFCCs: std
%     - 40 log-mel band energies: mean (spectral shape) and std
%     - spectral centroid, bandwidth, roll-off, flatness: mean and std
%     - dominant (peak) frequency: mean, std, median, mean |change|
%     - zero-crossing rate: mean and std
%     - temporal: energy std, fraction of active frames, syllable rate

    nfft  = 1024;   % ~46 ms at 22.05 kHz
    hop   = 256;
    nMel  = 40;
    nMfcc = 20;
    fmin  = 500;    % most songbird energy is above ~500 Hz
    fmax  = fs/2;

    x = double(x(:));
    x = x - mean(x);
    pk = max(abs(x));
    if pk > 0, x = x / pk; end
    if numel(x) < nfft, x(end+1:nfft) = 0; end

    % ---- Short-time Fourier transform -----------------------------------
    win = 0.5 - 0.5*cos(2*pi*(0:nfft-1)'/(nfft-1));      % Hann window
    nFrames = 1 + floor((numel(x) - nfft)/hop);
    idx = (1:nfft)' + (0:nFrames-1)*hop;
    frames = x(idx) .* win;
    S = abs(fft(frames, nfft)).^2;
    S = S(1:nfft/2+1, :);                                 % power spectrum
    freqs = (0:nfft/2)' * fs / nfft;

    % ---- Mel spectrum and MFCCs -----------------------------------------
    melFB  = melFilterbank(nMel, nfft, fs, fmin, fmax);
    logMel = log10(melFB*S + 1e-10);                      % nMel x nFrames
    mfcc   = dctMatrix(nMfcc, nMel) * logMel;             % nMfcc x nFrames
    dmfcc  = [zeros(nMfcc,1), diff(mfcc, 1, 2)];

    % ---- Pick active frames (where the bird is actually calling) --------
    band   = freqs >= fmin;
    Sb     = S(band, :);
    fb     = freqs(band);
    logE   = log10(sum(Sb, 1) + 1e-10);
    thresh = median(logE);
    act    = logE >= thresh;
    if nnz(act) < 3, act = true(1, nFrames); end

    % ---- Spectral descriptors (within the bird band) --------------------
    P        = Sb ./ (sum(Sb, 1) + eps);
    centroid = fb' * P;
    bwidth   = sqrt(sum(((fb - centroid).^2) .* P, 1));
    cumP     = cumsum(P, 1);
    [~, ri]  = max(cumP >= 0.85, [], 1);
    rolloff  = fb(ri)';
    flatness = exp(mean(log(Sb + 1e-12), 1)) ./ (mean(Sb, 1) + 1e-12);
    [~, pi_] = max(Sb, [], 1);
    peakF    = fb(pi_)';
    zcr      = sum(abs(diff(sign(frames), 1, 1)), 1) / (2*nfft);

    a  = act;
    kHz = 1e-3;   % scale frequencies to kHz so values are comparable
    pf = peakF(a) * kHz;
    spectral = [mean(centroid(a))*kHz, std(centroid(a))*kHz, ...
                mean(bwidth(a))*kHz,   std(bwidth(a))*kHz, ...
                mean(rolloff(a))*kHz,  std(rolloff(a))*kHz, ...
                mean(flatness(a)),     std(flatness(a)), ...
                mean(pf), std(pf), median(pf), mean(abs(diff([pf pf(end)]))), ...
                mean(zcr(a)), std(zcr(a))];

    % ---- Temporal descriptors -------------------------------------------
    dur = numel(x) / fs;
    rising = sum(diff(double(act)) == 1);                 % syllable onsets
    temporal = [std(logE), mean(act), rising/dur];

    % ---- Assemble -------------------------------------------------------
    lm = logMel(:, a);
    lm = lm - mean(lm(:));                                % level-independent
    f = [mean(mfcc(:,a), 2)', std(mfcc(:,a), 0, 2)', ...
         std(dmfcc(:,a), 0, 2)', ...
         mean(lm, 2)', std(lm, 0, 2)', ...
         spectral, temporal];
    f(~isfinite(f)) = 0;
end

function fb = melFilterbank(nMel, nfft, fs, fmin, fmax)
    persistent cache key
    k = [nMel nfft fs fmin fmax];
    if isequal(k, key), fb = cache; return; end
    hz2mel = @(h) 2595*log10(1 + h/700);
    mel2hz = @(m) 700*(10.^(m/2595) - 1);
    edges = mel2hz(linspace(hz2mel(fmin), hz2mel(fmax), nMel+2));
    freqs = (0:nfft/2) * fs / nfft;
    fb = zeros(nMel, nfft/2+1);
    for m = 1:nMel
        lo = edges(m); ce = edges(m+1); hi = edges(m+2);
        up   = (freqs - lo) / (ce - lo);
        down = (hi - freqs) / (hi - ce);
        fb(m,:) = max(0, min(up, down));
    end
    cache = fb; key = k;
end

function D = dctMatrix(nOut, nIn)
    [kk, nn] = ndgrid(0:nOut-1, 0:nIn-1);
    D = sqrt(2/nIn) * cos(pi * kk .* (2*nn + 1) / (2*nIn));
    D(1,:) = D(1,:) / sqrt(2);
end
