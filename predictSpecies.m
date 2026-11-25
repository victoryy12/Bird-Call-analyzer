function label = predictSpecies(model, X)
    X = (X - model.mu) ./ model.sigma;
    hidden = max(0, X * model.W1 + model.b1);
    [~, best] = max(hidden * model.W2 + model.b2, [], 2);
    label = categorical(model.classes(best), model.classes);
end
