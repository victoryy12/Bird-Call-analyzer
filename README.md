# Bird Call Analyzer & Generator

**CSCI 4810 Digital Information Processing · University of North Georgia · Fall 2026**

MATLAB tools that (1) identify a bird's species from a recording and (2) synthesize bird songs from scratch. The two halves check each other: generated songs go back through the analyzer to test whether they sound like the species they're supposed to be.

**Species:** Bewick's Wren, Northern Cardinal, American Robin, Song Sparrow, Northern Mockingbird

## Team

| Member | Contribution |
| --- | --- |
| Victor | Analyzer: feature extraction, neural network training, classification (`extractBirdFeatures.m`, `trainBirdClassifier.m`, `mlpPredict.m`, `birdCallAnalyzer.m`) |
| Jack | Generator and closed-loop test (`birdCallGenerator.m`, `testGenerator.m`), README |
| Caleb | _add your part_ |
| Zack | _add your part_ |

## Requirements

- **MATLAB R2022a or newer.** Everything runs on base MATLAB with no toolboxes. The one exception is the confusion-matrix figure at the very end of `trainBirdClassifier`, which needs the Statistics and Machine Learning Toolbox or the Deep Learning Toolbox. The model is saved before that step, so the figure is optional.
- **The dataset is only needed to retrain.** The trained model (`birdModel.mat`) is already in the repo, so the analyzer and generator work right after cloning.

## Setup

1. Clone the repo and open the folder in MATLAB as the current folder:
   ```
   git clone https://github.com/victoryy12/Bird-Call-analyzer.git
   ```
2. *(Only to retrain, or to analyze clips from the dataset)* Download the [bird song data set](https://www.kaggle.com/datasets/vinayshanbhag/bird-song-data-set) from Kaggle (about 2.4 GB). Unzip it and move its `wavfiles` folder into `Bird Call Dataset/`, so clips end up at paths like `Bird Call Dataset/wavfiles/557838-0.wav`. Git ignores that folder because it's too big for GitHub.

## Quick start

```matlab
% Identify a species
birdCallAnalyzer                                     % opens a file picker
birdCallAnalyzer('Bird Call Dataset/wavfiles/557838-0.wav')

% Generate a song
birdCallGenerator('cardinal', 'Play', true)          % plays it and shows the spectrogram
birdCallGenerator('NOCA', 'Seed', 4, 'OutFile', 'fake_cardinal.wav');
birdCallAnalyzer('fake_cardinal.wav')                % does the analyzer buy it?

% Closed-loop test: 10 synthetic songs per species, all classified
testGenerator

% Retrain the classifier (needs the dataset)
trainBirdClassifier
```

`birdCallGenerator` options: `'Seed'` (same seed, same song), `'Duration'` (seconds, default 3), `'SNR'` (background level in dB), `'Reverb'` (0 to 1), `'OutFile'`, `'Play'`, `'Plot'`. Species can be full names, four-letter codes (`NOCA`, `AMRO`, `SOSP`, `BEWR`, `NOMO`), or one word (`'robin'`). `birdCallGenerator('all', 'Count', 5)` writes 5 songs per species to `Synthetic Calls/`. Run `help birdCallGenerator` for details.

## How it works

### Analyzer

Each 3-second window becomes 157 features, computed only on the louder "active" frames where the bird is singing: MFCC means and spread, delta-MFCC spread, 40 log-mel band energies, spectral centroid, bandwidth, roll-off and flatness, peak-frequency statistics, zero-crossing rate, and energy and syllable rate. A one-hidden-layer neural network (256 ReLU units, softmax output) classifies them. It's trained from scratch in base MATLAB with Adam, dropout, and early stopping.

The data is split 70/15/15 **by recording**, not by clip, so clips cut from the same recording never appear in both training and testing. Longer recordings are cut into overlapping 3-second windows (1.5 s hop). Quiet windows are skipped and the remaining windows' probabilities are averaged.

### Generator

Every note is a sinusoid whose frequency follows a contour f(t). The phase is the running sum of frequency, φ[n] = 2π · Σ f[k] / fs, so a falling f(t) produces a downslurred whistle (an FM chirp). Each species is a recipe built from a few DSP building blocks:

| Building block | DSP idea | Used for |
| --- | --- | --- |
| Frequency sweep | Linear, exponential, or arched f(t) | Whistles, slurs, trill notes |
| Harmonics | Sum of sin(k·φ) with small weights | Timbre |
| Envelope | Raised-cosine attack and release | Note onsets and offsets |
| Buzz | Fast sinusoidal FM + AM at 100–300 Hz | Sparrow and wren buzzes |
| Band-limited noise | Zero the FFT bins outside a band | Harsh calls, background ticks |
| Pitch wander | Low-pass noise added to f(t), about 2% | Natural, non-robotic notes |
| Reverb | Convolution with decaying noise, via FFT | Echoes off trees and ground |
| Background | 1/f^0.6 hiss + random high-pitched ticks | Field-recording conditions |

| Species | Recipe |
| --- | --- |
| Northern Cardinal | 2–4 long slurs ("cheer" falling from ~4.4 to ~1.5 kHz, or "wheet" rising), then 4–9 fast "chew" downslurs |
| American Robin | Phrases of 2–4 connected, arched syllables (1.7–3.7 kHz), with a pause after each phrase |
| Song Sparrow | 2–4 intro notes → buzz → rapid trill (14–20 notes/s) → jumble of short notes |
| Bewick's Wren | High buzz or whistle → fast upsweep trill → slower downsweep trill |
| Northern Mockingbird | Borrowed phrases (cardinal slur, robin warble, trill, two-note whistle, harsh "chack"), each repeated 3–5 times |

**Calibration.** Frequency ranges, reverb, and background level were tuned against the features of the real training clips (`birdFeatures.mat`). The first version made pure tones in near-silence, and the analyzer called almost everything a mockingbird. Real songs are smeared by pitch wobble, echoes, MP3 compression, and background noise. Adding pitch wander, reverb, and a realistic noise floor is what made the synthetic songs recognizable. Recording quality also differs by species in this dataset: the mockingbird clips are the cleanest and the wren and sparrow clips the noisiest. So each species gets its typical recording SNR by default (wren 11 dB, cardinal 13, robin 15, sparrow 17, mockingbird 25).

## Results

### Analyzer on real recordings

**76.5% of 848 held-out test clips** (3 s each), all from recordings never seen in training.

| Species | Correct | Most common mistake |
| --- | --- | --- |
| Northern Cardinal | 82% | Mockingbird (23 of 168) |
| Northern Mockingbird | 80% | American Robin (19 of 193) |
| American Robin | 78% | Mockingbird (23 of 152) |
| Song Sparrow | 74% | Bewick's Wren (39 of 197) |
| Bewick's Wren | 68% | Song Sparrow (28 of 138) |

The mistakes make biological sense. Cardinals and robins get confused with the mockingbird, which imitates other birds, cardinals and robins included. The wren and sparrow both sing buzzy trills in the same 3–6 kHz range.

### Synthetic songs through the analyzer

**183 of 200 synthetic songs (92%) were classified as the intended species**, using 40 songs per species from seeds never used during tuning: wren 95%, robin 95%, sparrow 92%, mockingbird 92%, cardinal 82% (its misses went to robin).

> These numbers come from running the generator with Victor's feature extraction and saved model in GNU Octave. MATLAB's random number generator makes different songs from the same seeds, so `testGenerator(40, 50000)` in MATLAB will give similar but not identical results. Replace this note with the MATLAB numbers once someone runs it.

## Files

| File | What it does |
| --- | --- |
| `birdCallAnalyzer.m` | Classifies a recording of any length and sample rate; plots waveform, spectrogram, and species probabilities |
| `extractBirdFeatures.m` | 157-feature vector for one clip (base MATLAB STFT, mel filterbank, MFCCs) |
| `trainBirdClassifier.m` | Extracts features, splits by recording, trains the network, reports test accuracy, saves `birdModel.mat` |
| `mlpPredict.m` | Forward pass of the trained network |
| `birdCallGenerator.m` | Synthesizes songs for all five species |
| `testGenerator.m` | Closed-loop test: generates songs, classifies them, prints a confusion matrix; with the dataset present, also plots real vs. synthetic spectrograms |
| `birdModel.mat` | Trained network weights and normalization constants |
| `birdFeatures.mat` | Cached training features, so retraining skips feature extraction |
| `Bird Call Dataset/bird_songs_metadata.csv` | Species, recordist, location, and license for every clip |
| `Synthetic Calls/` | Created by `testGenerator` (git-ignored) |

## Data and credits

The recordings come from [xeno-canto](https://xeno-canto.org) contributors, via the Kaggle [bird song data set](https://www.kaggle.com/datasets/vinayshanbhag/bird-song-data-set) by Vinay Shanbhag: 5,422 three-second clips from 477 recordings made in the United States, 22,050 Hz mono, songs only, top xeno-canto quality grade only. Every clip's recordist and Creative Commons license are listed in `Bird Call Dataset/bird_songs_metadata.csv`. All of the licenses allow non-commercial use with attribution (mostly CC BY-NC-SA).
