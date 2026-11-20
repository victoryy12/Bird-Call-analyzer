function [y, fs, info] = birdCallGenerator(species, varargin)
%BIRDCALLGENERATOR Synthesize a bird song with FM/AM synthesis.
%   y = birdCallGenerator('Northern Cardinal') returns 3 seconds of
%   synthetic cardinal song at 22,050 Hz, the sample rate of the training
%   data, so the output can go straight into birdCallAnalyzer.
%
%   [y, fs, info] = birdCallGenerator(species, Name, Value, ...)
%
%   species   'Northern Cardinal' | 'American Robin' | 'Song Sparrow' |
%             'Bewick''s Wren' | 'Northern Mockingbird' | 'all'
%             Four-letter codes (NOCA, AMRO, SOSP, BEWR, NOMO) and single
%             words ('cardinal', 'robin', 'sparrow', 'wren', 'mockingbird')
%             also work.
%
%   Options
%     'Seed'      random seed. Each seed is a different song; the same seed
%                 always gives the same song. Default: random.
%     'Duration'  length in seconds (default 3, the training clip length)
%     'SNR'       background level, in dB below the song. Default: what is
%                 typical for that species in the training recordings
%                 (wren 11, cardinal 13, robin 15, sparrow 17,
%                 mockingbird 25). Inf = no background at all.
%     'Reverb'    amount of echo, 0 (none) to 1 (default 0.5)
%     'OutFile'   also write the song to this WAV file
%     'Play'      play it through the speakers (default false)
%     'Plot'      show waveform + spectrogram (default: true when called
%                 with no output arguments)
%     'Count'     with 'all': songs per species (default 3)
%     'OutDir'    with 'all': folder for the WAV files
%                 (default 'Synthetic Calls')
%
%   Examples
%     birdCallGenerator('cardinal', 'Play', true)
%     birdCallGenerator('NOCA', 'Seed', 4, 'OutFile', 'fake_cardinal.wav');
%     birdCallAnalyzer('fake_cardinal.wav')    % does the classifier buy it?
%     birdCallGenerator('all', 'Count', 5)     % 25 WAV files
%
%   How it works
%     Every note is a sinusoid whose instantaneous frequency follows a
%     contour f(t). The phase is the running integral of frequency,
%         phi(n) = 2*pi * cumsum(f) / fs,
%     so a falling f(t) gives a downslurred whistle (an FM chirp).
%     Harmonics add timbre, a raised-cosine envelope shapes each note's
%     attack and decay, fast sinusoidal FM + AM (100-300 Hz) turns a tone
%     into a buzz, and band-limited noise makes harsh calls. Each species
%     is a recipe that arranges these notes in time. Finally the song gets
%     the things every field recording has: slight pitch wobble, echoes
%     (reverb by FFT convolution), and background hiss with faint ticks.
%     Without them the analyzer hears pure tones in silence and calls
%     almost everything a mockingbird.
%
%   Uses only base MATLAB (no toolboxes).
%
%   See also birdCallAnalyzer, testGenerator.

    if nargin < 1 || isempty(species), species = 'Northern Cardinal'; end
    opt = parseOptions(varargin, nargout);
    fs  = 22050;

    if strcmpi(char(species), 'all')
        info = generateAll(opt, fs);
        if nargout > 0, y = []; end
        return;
    end

    [name, code, typicalSnr] = resolveSpecies(species);
    snr = opt.SNR;
    if isempty(snr), snr = typicalSnr; end
    seed = opt.Seed;
    if isempty(seed), seed = randi(1e6); end
    setSeed(seed);

    n = round(opt.Duration * fs);
    switch code
        case 'NOCA', y = cardinalSong(n, fs);
        case 'AMRO', y = robinSong(n, fs);
        case 'SOSP', y = songSparrowSong(n, fs);
        case 'BEWR', y = wrenSong(n, fs);
        case 'NOMO', y = mockingbirdSong(n, fs);
    end
    y = addReverb(y, fs, opt.Reverb);
    y = addBackground(y, fs, snr);
    y = 0.9 * y / max(abs(y));

    info = struct('species', name, 'code', code, 'fs', fs, 'seed', seed, ...
                  'snr', snr, 'file', opt.OutFile);
    if ~isempty(opt.OutFile)
        audiowrite(opt.OutFile, y, fs);
    end
    if opt.Play
        sound(y, fs);
    end
    if opt.Plot
        plotSong(y, fs, sprintf('Synthetic %s (seed %d)', name, seed));
    end
    if nargout == 0
        clear y      % don't print 66,000 samples when called without a semicolon
    end
end

%% ========================================================================
%  Species recipes. Frequencies in Hz, times in seconds. u(a,b) draws a
%  uniform random number, so every call varies a little, like real birds.
%  ========================================================================

function y = cardinalSong(n, fs)
% Loud, clear slurred whistles: a phrase of long slurs ("cheer cheer" going
% down, or "wheet wheet" going up), then a fast run of short downslurs
% ("chew-chew-chew"). Almost a pure tone; energy spread over 1.5-4.4 kHz.
    y = zeros(n, 1);  dur = n / fs;
    harm = [1 0.04 0.01];
    t = u(0.05, 0.25);
    while t < dur - 0.3
        % Phrase A: 2-4 long slurs
        goingDown = rand < 0.65;
        if goingDown
            fA = u(3800, 4400);  fB = u(1500, 1800);
        else
            fA = u(1500, 1800);  fB = u(3600, 4100);
        end
        len = u(0.22, 0.32);  gap = u(0.06, 0.12);
        for k = 1:randi([2 4])
            if goingDown
                x = note(fs, len, fA, fB, 'exp', harm, 0.05, 0.15);
            else
                x = note(fs, len, fA, fB, 'exp', harm, 0.15, 0.05);
            end
            y = place(y, u(0.85, 1) * x, t, fs);
            t = t + len + gap;
        end
        % Phrase B: fast, short downslurs
        t = t + u(0.05, 0.15);
        len = u(0.07, 0.11);  gap = u(0.03, 0.05);
        f1 = u(3500, 4000);   f2 = u(1600, 2000);
        for k = 1:randi([4 9])
            x = note(fs, len, f1, f2, 'exp', harm, 0.1, 0.2);
            y = place(y, u(0.8, 1) * x, t, fs);
            t = t + len + gap;
        end
        t = t + u(0.3, 0.7);
    end
end

function y = robinSong(n, fs)
% "Cheerily, cheer-up, cheerio": short phrases of 2-4 warbled syllables,
% each rising and falling within about 1.7-3.7 kHz, then a pause. The
% syllables in a phrase run together without breaks: one continuous
% voice that dips in loudness between syllables. Rounder and slower than
% a cardinal (about 2-3 phrases a second).
    y = zeros(n, 1);  dur = n / fs;
    harm = [1 0.08 0.02];
    t = u(0.05, 0.3);
    while t < dur - 0.2
        base = u(1900, 2300);                 % each phrase has its own pitch
        fc = [];  ac = [];
        for k = 1:randi([2 4])
            m  = round(u(0.10, 0.16) * fs);
            lo = base * u(0.9, 1.0);
            hi = base * u(1.35, 1.6);
            switch randi(4)
                case {1, 2}, fk = sweep(m, lo, hi, 'arch');
                case 3,      fk = sweep(m, hi, lo, 'lin');
                case 4,      fk = sweep(m, lo, hi, 'lin');
            end
            fc = [fc; fk];                                     %#ok<AGROW>
            ac = [ac; 0.55 + 0.45 * env(m, 0.3, 0.35)];        %#ok<AGROW>
        end
        x = tone(fc, fs, harm) .* ac .* env(numel(fc), 0.05, 0.1);
        y = place(y, u(0.8, 1) * x, t, fs);
        t = t + numel(fc)/fs + u(0.25, 0.5);
    end
end

function y = songSparrowSong(n, fs)
% Classic pattern: 2-4 clear introductory notes ("sweet sweet sweet"),
% a buzz, a rapid trill, then a jumble of short, varied notes. Brighter
% and busier than the whistlers; energy mostly 3-6 kHz.
    y = zeros(n, 1);  dur = n / fs;
    harm = [1 0.25 0.06];
    t = u(0.05, 0.3);
    while t < dur - 0.5
        % 1) Introductory notes
        f = u(2600, 3300);
        for k = 1:randi([2 4])
            len = u(0.07, 0.11);
            x = note(fs, len, 1.1*f, 0.9*f, 'lin', harm, 0.1, 0.3);
            y = place(y, x, t, fs);
            t = t + len + u(0.08, 0.14);
        end
        % 2) Buzz
        len = u(0.25, 0.45);
        x = buzz(fs, len, u(4100, 5000), u(800, 1400), u(120, 220), harm);
        y = place(y, 0.7 * x, t, fs);
        t = t + len + u(0.04, 0.08);
        % 3) Trill: fast, steep downsweeps
        rate = u(14, 20);
        fA = u(5500, 6800);  fB = u(2800, 3600);
        for k = 1:randi([6 12])
            x = note(fs, 0.6/rate, fA, fB, 'exp', harm, 0.1, 0.4);
            y = place(y, 0.8 * x, t, fs);
            t = t + 1/rate;
        end
        % 4) Terminal jumble
        for k = 1:randi([4 8])
            len = u(0.03, 0.08);
            f1 = u(2300, 6400);  f2 = f1 * u(0.6, 1.4);
            x = note(fs, len, f1, f2, 'lin', harm, 0.1, 0.4);
            y = place(y, u(0.6, 1) * x, t, fs);
            t = t + len + u(0.02, 0.06);
        end
        t = t + u(0.8, 1.5);                  % pause before the next song
    end
end

function y = wrenSong(n, fs)
% Bewick's Wren: thin and high. One or two buzzy or whistled intro notes,
% then two trills with different note shapes and speeds. Lives in the same
% 2.6-5.5 kHz range as the Song Sparrow, but is built from longer trills.
    y = zeros(n, 1);  dur = n / fs;
    harm = [1 0.2 0.05];
    t = u(0.05, 0.3);
    while t < dur - 0.5
        % Intro: high buzz or high downslurred whistle
        for k = 1:randi([1 2])
            len = u(0.15, 0.25);
            if rand < 0.5
                x = buzz(fs, len, u(4000, 5200), u(500, 800), u(150, 250), harm);
            else
                x = note(fs, len, u(5000, 5800), u(3300, 3900), 'exp', harm, 0.1, 0.3);
            end
            y = place(y, 0.8 * x, t, fs);
            t = t + len + u(0.06, 0.12);
        end
        % Trill 1: fast upsweeps
        rate = u(9, 13);
        fA = u(2900, 3400);  fB = u(4500, 5300);
        for k = 1:round(u(0.5, 0.9) * rate)
            x = note(fs, 0.6/rate, fA, fB, 'exp', harm, 0.2, 0.3);
            y = place(y, x, t, fs);
            t = t + 1/rate;
        end
        t = t + u(0.03, 0.08);
        % Trill 2: slower, lower downsweeps
        rate = u(6, 10);
        fA = u(4000, 4800);  fB = u(2600, 3000);
        for k = 1:round(u(0.4, 0.8) * rate)
            x = note(fs, 0.6/rate, fA, fB, 'exp', harm, 0.1, 0.35);
            y = place(y, 0.9 * x, t, fs);
            t = t + 1/rate;
        end
        t = t + u(0.6, 1.2);
    end
end

function y = mockingbirdSong(n, fs)
% The mimic: a run of different phrase types, each repeated 3-5 times
% before switching. Phrases are borrowed sounds:
% cardinal-like slurs, robin-like warbles, quick trills, two-note
% whistles, and the odd harsh "chack". Mostly tonal, around 1.8-3.8 kHz.
    y = zeros(n, 1);  dur = n / fs;
    harm = [1 0.06 0.015];
    t = u(0.0, 0.15);
    while t < dur
        x = mockPhrase(fs, harm);
        for r = 1:randi([3 5])
            y = place(y, u(0.8, 1) * x, t, fs);
            t = t + numel(x)/fs + u(0.06, 0.1);
        end
        t = t + u(0.15, 0.3);
    end
end

function x = mockPhrase(fs, harm)
% One phrase for the mockingbird to repeat.
    switch randi(6)
        case 1   % borrowed cardinal slur
            x = note(fs, u(0.15, 0.25), u(3200, 3800), u(1700, 2100), 'exp', harm, 0.08, 0.25);
        case 2   % borrowed robin warble: two arched syllables
            b = u(2000, 2500);
            x = [note(fs, u(0.08, 0.12), b, 1.2*b, 'arch', harm, 0.2, 0.3); ...
                 zeros(round(0.02*fs), 1); ...
                 note(fs, u(0.08, 0.12), 0.95*b, 1.4*b, 'arch', harm, 0.2, 0.3)];
        case 3   % quick trill
            r = u(18, 28);  f1 = u(2600, 3600);  f2 = f1 * u(0.75, 0.9);
            x = [];
            for k = 1:randi([4 7])
                x = [x; note(fs, 0.6/r, f1, f2, 'lin', harm, 0.1, 0.3); zeros(round(0.4*fs/r), 1)]; %#ok<AGROW>
            end
        case 4   % two-note whistle ("tee-oo")
            fA = u(2800, 3600);
            x = [note(fs, u(0.06, 0.1), fA, fA*1.05, 'lin', harm, 0.2, 0.2); ...
                 note(fs, u(0.08, 0.12), fA*0.8, fA*0.7, 'lin', harm, 0.1, 0.3)];
        case 5   % rising-falling whistle
            fA = u(1700, 2200);
            x = note(fs, u(0.15, 0.25), fA, fA*u(1.4, 1.7), 'arch', harm, 0.15, 0.25);
        case 6   % harsh "chack", softened by a tonal core
            len = u(0.05, 0.08);
            nz  = bandNoise(round(len*fs), fs, 1500, 4500);
            x   = (0.35*nz + note(fs, len, 2600, 2200, 'lin', harm, 0.05, 0.05)) ...
                  .* env(round(len*fs), 0.05, 0.6);
    end
end

%% ========================================================================
%  Building blocks
%  ========================================================================

function x = note(fs, dur, f1, f2, shape, harm, att, rel)
% One note: a frequency sweep from f1 to f2 with an attack/release envelope.
    n = max(2, round(dur * fs));
    x = tone(sweep(n, f1, f2, shape), fs, harm) .* env(n, att, rel);
end

function f = sweep(n, f1, f2, shape)
% Frequency contour over n samples.
%   'lin'   straight line from f1 to f2
%   'exp'   constant ratio per second (a straight line on a log axis)
%   'arch'  rises from f1 to f2 at the midpoint, then falls back to f1
    t = linspace(0, 1, n)';
    switch shape
        case 'lin',  f = f1 + (f2 - f1) * t;
        case 'exp',  f = f1 * (f2 / f1) .^ t;
        case 'arch', f = f1 + (f2 - f1) * sin(pi * t);
        otherwise,   error('birdCallGenerator:shape', 'Unknown sweep shape "%s".', shape);
    end
end

function y = tone(f, fs, harm)
% Harmonic tone following the frequency contour f (one value per sample).
% Phase = running integral of frequency, so the pitch follows f exactly.
% A slight random pitch wander is added because no real bird
% holds a perfectly steady pitch; without it every note is an unnaturally
% thin spectral line. (About 2% wander, at rates up to 120 Hz.)
    f   = f(:);
    f   = f .* (1 + 0.02 * smoothNoise(numel(f), fs, 120));
    phi = 2*pi * cumsum(f) / fs;
    y   = zeros(size(f));
    for k = 1:numel(harm)
        if harm(k) == 0, continue; end
        ok = k * f < 0.45 * fs;                  % drop harmonics that would alias
        y  = y + harm(k) * sin(k * phi) .* ok;
    end
end

function e = env(n, att, rel)
% Raised-cosine attack and release; att and rel are fractions of the note.
    e  = ones(n, 1);
    na = max(1, round(att * n));
    nr = max(1, round(rel * n));
    e(1:na) = 0.5 - 0.5*cos(pi * (0:na-1)' / na);
    e(end-nr+1:end) = e(end-nr+1:end) .* (0.5 + 0.5*cos(pi * (1:nr)' / nr));
end

function x = buzz(fs, dur, fc, depth, rate, harm)
% Buzz: a carrier whose pitch and loudness wobble 'rate' times a second.
    n  = max(2, round(dur * fs));
    t  = (0:n-1)' / fs;
    f  = fc + depth * sin(2*pi*rate*t);
    am = 0.55 + 0.45 * cos(2*pi*rate*t);
    x  = tone(f, fs, harm) .* am .* env(n, 0.1, 0.2);
end

function w = smoothNoise(n, fs, bw)
% Random signal in [-1, 1] containing only frequencies below bw Hz.
    W  = fft(randn(n, 1));
    fr = (0:n-1)' * fs / n;
    fr = min(fr, fs - fr);
    W(fr > bw) = 0;
    w  = real(ifft(W));
    w  = w / (max(abs(w)) + eps);
end

function y = addReverb(y, fs, wet)
% Echoes off trees and ground: convolve with an impulse response of
% exponentially decaying noise (60 dB of decay in rt60 seconds), done with
% the FFT because direct convolution of 66,000 x 5,000 samples is slow.
    if wet <= 0, return; end
    rt60 = 0.25;
    nIr  = round(rt60 * fs);
    t    = (0:nIr-1)' / fs;
    ir   = randn(nIr, 1) .* exp(-6.9 * t / rt60);
    nfft = 2^nextpow2(numel(y) + nIr - 1);
    r    = real(ifft(fft(y, nfft) .* fft(ir, nfft)));
    r    = r(1:numel(y));
    r    = r * sqrt(mean(y.^2) / (mean(r.^2) + eps));      % match loudness
    y    = (1 - wet) * y + wet * r;
end

function x = bandNoise(n, fs, fLo, fHi)
% White noise band-limited to [fLo, fHi] Hz by zeroing FFT bins.
    X = fft(randn(n, 1));
    f = (0:n-1)' * fs / n;
    f = min(f, fs - f);                          % frequency of each bin
    X(f < fLo | f > fHi) = 0;
    x = real(ifft(X));
    x = x / (max(abs(x)) + eps);
end

function y = place(y, x, t0, fs)
% Mix note x into y starting at t0 seconds; anything past the end is cut.
    i0 = round(t0 * fs) + 1;
    if i0 > numel(y), return; end
    i1 = min(numel(y), i0 + numel(x) - 1);
    y(i0:i1) = y(i0:i1) + x(1:i1-i0+1);
end

function y = addBackground(y, fs, snrDb)
% Real recordings always have sound under the bird. Two layers, scaled to
% sit snrDb below the song:
%   1) steady hiss whose power falls off with frequency (~1/f^0.6, between
%      white and pink noise), and
%   2) scattered faint ticks in the 3.5-10 kHz range (insects, distant
%      birds) at random times, about 8 a second, so the high bands
%      flicker the way they do in field recordings instead of sitting flat.
    if isinf(snrDb), return; end
    n = numel(y);
    sig   = y(abs(y) > 1e-3 * max(abs(y)));      % RMS of the song itself
    level = sqrt(mean(sig.^2)) * 10^(-snrDb/20);

    X = fft(randn(n, 1));
    f = (0:n-1)' * fs / n;
    f = min(f, fs - f);
    f(1) = f(2);                                 % avoid dividing by zero at DC
    hiss = real(ifft(X ./ f.^0.3));              % amplitude ~ f^-0.3, power ~ f^-0.6
    hiss = hiss / sqrt(mean(hiss.^2));

    ticks = zeros(n, 1);
    t = 0;
    while true
        t = t - log(rand) / 8;                   % random (Poisson) arrival times
        if t >= n/fs, break; end
        m = round(u(0.01, 0.05) * fs);
        x = bandNoise(m, fs, u(3500, 6000), u(7000, 10500)) .* env(m, 0.5, 0.5);
        ticks = place(ticks, u(0.5, 1.5) * x, t, fs);
    end
    ticks = ticks / sqrt(mean(ticks.^2) + eps);

    y = y + level * (hiss + 2 * ticks);
end

function r = u(a, b)
% Uniform random number in [a, b].
    r = a + (b - a) * rand;
end

function setSeed(seed)
    if exist('rng', 'file') || exist('rng', 'builtin')
        rng(seed);
    else
        rand('seed', seed); randn('seed', seed); %#ok<RAND>
    end
end

%% ========================================================================
%  Options, batch mode, plotting
%  ========================================================================

function [name, code, snr] = resolveSpecies(s)
% Name, code, and typical recording SNR (dB) for each species. The SNRs
% were calibrated against the training recordings: wren and sparrow clips
% are the noisiest (spectral flatness ~0.015), mockingbird clips the
% cleanest (~0.005; they sing loudly from exposed perches).
    names = {'Bewick''s Wren', 'Northern Cardinal', 'American Robin', ...
             'Song Sparrow', 'Northern Mockingbird'};
    codes = {'BEWR', 'NOCA', 'AMRO', 'SOSP', 'NOMO'};
    words = {'wren', 'cardinal', 'robin', 'sparrow', 'mockingbird'};
    snrs  = [11 13 15 17 25];
    s = lower(strtrim(char(s)));
    for k = 1:numel(names)
        if strcmpi(s, names{k}) || strcmpi(s, codes{k}) || ~isempty(strfind(s, words{k}))
            name = names{k};  code = codes{k};  snr = snrs(k);
            return;
        end
    end
    error('birdCallGenerator:species', 'Unknown species "%s". Use one of: %s, or ''all''.', ...
          s, strjoin(names, ', '));
end

function opt = parseOptions(args, nOut)
    opt = struct('Seed', [], 'Duration', 3, 'SNR', [], 'Reverb', 0.5, 'OutFile', '', ...
                 'Play', false, 'Plot', nOut == 0, 'Count', 3, ...
                 'OutDir', 'Synthetic Calls');
    if mod(numel(args), 2) ~= 0
        error('birdCallGenerator:options', 'Options must come in Name, Value pairs.');
    end
    names = fieldnames(opt);
    for k = 1:2:numel(args)
        hit = names(strcmpi(names, char(args{k})));
        if isempty(hit)
            error('birdCallGenerator:options', 'Unknown option "%s".', char(args{k}));
        end
        opt.(hit{1}) = args{k+1};
    end
    opt.OutFile = char(opt.OutFile);
    opt.OutDir  = char(opt.OutDir);
    if ~isscalar(opt.Duration) || opt.Duration < 1
        error('birdCallGenerator:options', 'Duration must be at least 1 second.');
    end
end

function info = generateAll(opt, fs)
% Write opt.Count songs per species to opt.OutDir. Seeds are fixed
% (species index * 1000 + song number, plus 'Seed' if given), so the same
% call always produces the same files.
    codes = {'BEWR', 'NOCA', 'AMRO', 'SOSP', 'NOMO'};
    base  = opt.Seed;
    if isempty(base), base = 0; end
    if ~exist(opt.OutDir, 'dir'), mkdir(opt.OutDir); end
    files = {};  species = {};
    for c = 1:numel(codes)
        for k = 1:opt.Count
            file = fullfile(opt.OutDir, sprintf('synthetic_%s_%02d.wav', codes{c}, k));
            [~, ~, one] = birdCallGenerator(codes{c}, 'Seed', base + 1000*c + k, ...
                'Duration', opt.Duration, 'SNR', opt.SNR, 'Reverb', opt.Reverb, ...
                'OutFile', file, 'Plot', false);
            files{end+1, 1}   = file;         %#ok<AGROW>
            species{end+1, 1} = one.species;  %#ok<AGROW>
        end
    end
    info = struct('files', {files}, 'species', {species}, 'fs', fs);
    fprintf('Wrote %d synthetic calls to "%s".\n', numel(files), opt.OutDir);
end

function plotSong(y, fs, titleStr)
% Waveform + spectrogram (manual STFT, same settings as birdCallAnalyzer).
    figure('Name', 'Bird Call Generator', 'Position', [100 100 1000 500]);
    t = (0:numel(y)-1) / fs;
    ax1 = subplot(2, 1, 1);
    plot(ax1, t, y, 'Color', [0.2 0.4 0.7]);
    xlim(ax1, [0 t(end)]);  ylabel(ax1, 'Amplitude');
    title(ax1, strrep(titleStr, '_', '\_'));

    [S, tt, ff] = stftDb(y, fs);
    ax2 = subplot(2, 1, 2);
    imagesc(ax2, tt, ff/1000, S);
    axis(ax2, 'xy');  colormap(ax2, parula);
    mx = max(S(:));  set(ax2, 'CLim', [mx-80 mx]);
    ylabel(ax2, 'Frequency (kHz)');  xlabel(ax2, 'Time (s)');
    title(ax2, 'Spectrogram');
    linkaxes([ax1 ax2], 'x');
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
