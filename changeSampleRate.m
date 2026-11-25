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
