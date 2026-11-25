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
