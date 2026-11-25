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
