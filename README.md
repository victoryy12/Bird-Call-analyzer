# Bird Call Analyzer & Generator

CSCI 4810 (Digital Information Processing) group project, University of North Georgia, Fall 2026.

This project has two parts. The classifier listens to a recording and figures out which bird is singing. The generator goes the other direction and builds bird songs from scratch. We use each half to check the other: generated songs get run back through the classifier to see if they actually sound like the species they're supposed to be.

It covers five species: American Robin, Bewick's Wren, Northern Cardinal, Northern Mockingbird, and Song Sparrow.

## Team

| Member | Contribution |
| --- | --- |
| Victor | Classifier: feature extraction, neural network, training and classification (`bird_classifier.m`, `getFeatures.m`, `trainModel.m`, `predictSpecies.m`, `changeSampleRate.m`, `melFilters.m`) |
| Jack | Generator and closed-loop test (`birdCallGenerator.m`, `testGenerator.m`), README |
| Caleb |Paper revision, project revision, results overseer |
| Zac | Code revision, drafting research paper, data analysis |

## What you need

You'll need MATLAB R2019b or newer. It all runs on base MATLAB, so no toolboxes.

The generator and `testGenerator` work as soon as you clone, because the trained model (`birdModel.mat`) is already in the repo. The dataset (about 2.4 GB) is only needed for `bird_classifier`, since that's what it trains on. Download it before you run that script. Right now, without the WAV files, it stops with a message telling you to download them.

## Setup

1. Clone the repo, then open the folder in MATLAB and make it your current folder:
   ```
   git clone https://github.com/victoryy12/Bird-Call-analyzer.git
   ```
2. If you want to retrain or classify clips from the dataset, download the [bird song data set](https://www.kaggle.com/datasets/vinayshanbhag/bird-song-data-set) from Kaggle. Unzip it and move the `wavfiles` folder into `Bird Call Dataset/`, so a clip ends up at a path like `Bird Call Dataset/wavfiles/557838-0.wav`. Git ignores that folder because it's too big for GitHub.

## Running it

```matlab
% Train, print the test accuracy, then pick recordings to classify
% (needs the dataset; Cancel the file picker to stop)
bird_classifier

% Generate a song
birdCallGenerator('cardinal', 'Play', true)          % plays it and shows the spectrogram
birdCallGenerator('NOCA', 'Seed', 4, 'OutFile', 'fake_cardinal.wav');

% Closed-loop test: 10 synthetic songs per species, all classified
testGenerator
```

`birdCallGenerator` takes a few options. `'Seed'` picks the song, and the same seed always gives the same song. `'Duration'` is the length in seconds (3 by default), `'SNR'` is the background level in dB, `'Dynamics'` is how much the loudness can vary from note to note in dB, and `'Reverb'` goes from 0 to 1. There's also `'OutFile'`, `'Play'`, and `'Plot'`. You can name a species by its full name, its four-letter code (`AMRO`, `BEWR`, `NOCA`, `NOMO`, `SOSP`), or one word like `'robin'`. `birdCallGenerator('all', 'Count', 5)` writes 5 songs per species to `Synthetic Calls/`. Type `help birdCallGenerator` for the details.

To classify one of the generated files, run `bird_classifier` and pick it in the file picker.

## How it works

### The classifier

Every clip gets resampled to 22,050 Hz (using the FFT) and split into frames of 1,024 samples with a hop of 512. We only keep the louder half of those frames, since that's where the bird is actually singing. From them we get 36 features: the mean and spread of 13 MFCCs, plus the mean and spread of five spectral measures (centroid, spread, flatness, roll-off, and flux). Everything is computed above 500 Hz.

The features go into a neural network with one hidden layer of 64 ReLU units and a softmax output. We train it from scratch with full-batch gradient descent for 1,000 steps, with a learning rate of 0.5 and an L2 penalty of 0.01. The data is split 80/20 by recording, not by clip, so clips cut from the same recording never end up in both training and testing. Features are cached in `Bird Call Dataset/features_cache.mat` so later runs can skip extraction, and the trained model gets saved to `birdModel.mat`.

To classify a full file, the script chops it into 3-second chunks, skips any chunk quieter than 10% of the loudest one, classifies each remaining chunk, and goes with the majority vote.

### The generator

Every note is a sine wave whose frequency follows a curve f(t). The phase is the running sum of the frequency, φ[n] = 2π · Σ f[k] / fs, so when f(t) falls you get a whistle that slides down in pitch (an FM chirp). Each species is a recipe put together from a handful of building blocks:

| Building block | DSP idea | Used for |
| --- | --- | --- |
| Frequency sweep | Linear, exponential, or arched f(t) | Whistles, slurs, trill notes |
| Harmonics | Sum of sin(k·φ) with small weights | Timbre |
| Envelope | Raised-cosine attack and release | Note onsets and offsets |
| Buzz | Fast sinusoidal FM + AM at 100–300 Hz | Sparrow and wren buzzes |
| Band-limited noise | Zero the FFT bins outside a band | Harsh calls, background ticks |
| Pitch wander | Low-pass noise added to f(t), about 2% | Natural, non-robotic notes |
| Uneven loudness | Each note a random 0 to N dB below full | Loud and soft notes, a bird turning its head |
| Reverb | Convolution with decaying noise, via FFT | Echoes off trees and ground |
| Background | 1/f^0.6 hiss + random high-pitched ticks | Field-recording conditions |

| Species | Recipe |
| --- | --- |
| Northern Cardinal | 2–4 long slurs ("cheer" falling from ~4.4 to ~1.5 kHz, or "wheet" rising), then 4–9 fast "chew" downslurs |
| American Robin | Phrases of 2–4 connected, arched syllables (1.7–3.7 kHz), with a pause after each phrase |
| Song Sparrow | 2–4 intro notes, then a buzz, a rapid trill (14–20 notes/s), and a jumble of short notes |
| Bewick's Wren | A high buzz or whistle, then a fast upsweep trill, then a slower downsweep trill |
| Northern Mockingbird | Borrowed phrases (cardinal slur, robin warble, trill, two-note whistle, harsh "chack"), each repeated 3–5 times |

### Calibration

We tuned the generator by comparing what it makes against the features of the real training clips, and we learned two things doing it.

First, realistic beats clean. Our first version made pure tones over near-silence, and the classifier called almost everything a mockingbird. Real recordings are messier. The pitch wobbles, there are echoes, MP3 compression, and background noise, and some notes are a lot louder than others. Once we added those effects, the synthetic songs started getting recognized.

Second, recording conditions aren't the same for every species. In this dataset the mockingbird recordings are the cleanest and the wren and sparrow recordings are the noisiest, and the species also differ in how loud the singing is compared with each clip's peak. So each species gets its own default background level and loudness range, both set so its synthetic songs match its real recordings:

| | Wren | Cardinal | Robin | Sparrow | Mockingbird |
| --- | --- | --- | --- | --- | --- |
| Background (SNR, dB) | 11 | 13 | 15 | 17 | 25 |
| Loudness range (dB) | 40 | 20 | 25 | 10 | 10 |

The calibration also has to keep up with the classifier. When the classifier was rebuilt with a different set of features, the share of synthetic songs it recognized dropped from 92% to about 70%, because almost every synthetic wren got called a Song Sparrow. Re-matching each species' loudness range to the new features brought it back up to about 90%.

## Results

### The classifier on real recordings

`bird_classifier` prints its test accuracy on its fixed split (`rng(42)`): **72%** 

We also ran the same pipeline on 10 other random recording-level splits, and it averaged 72% (between 66% and 75%). Here's how each species did, totaled over those 10 splits:

| Species | Correct | Most common mistake |
| --- | --- | --- |
| Northern Cardinal | 79% | Mockingbird (7%) |
| Song Sparrow | 76% | Bewick's Wren (16%) |
| American Robin | 70% | Mockingbird (15%) |
| Northern Mockingbird | 68% | American Robin (18%) |
| Bewick's Wren | 63% | Song Sparrow (30%) |

The mistakes line up with the biology. Robins and mockingbirds get mixed up because mockingbirds imitate other birds, robins included. The wren and the sparrow both sing buzzy trills in the same 3–6 kHz range.

### Synthetic songs through the classifier

We ran `testGenerator(40, 50000)`, which makes 40 songs per species from seeds we never used while tuning. The trained model in `birdModel.mat` classified 181 of the 200 songs (90%) as the right species: mockingbird 98%, sparrow 98%, robin 90%, cardinal 85%, and wren 82%. That's higher than on real recordings, which makes sense, since the synthetic songs are textbook examples of each species.


## Files

| File | What it does |
| --- | --- |
| `bird_classifier.m` | The main script. Loads the labels, extracts or loads cached features, splits by recording, trains, saves `birdModel.mat`, prints the test accuracy, then classifies files you pick |
| `getFeatures.m` | Computes the 36 features for one clip (FFT, mel filterbank, MFCCs, spectral measures) |
| `trainModel.m` | Trains the neural network |
| `predictSpecies.m` | Classifies feature vectors with a trained model |
| `changeSampleRate.m` | Resamples audio to 22,050 Hz in the frequency domain |
| `melFilters.m` | Builds the triangular mel filterbank |
| `birdModel.mat` | Trained network weights and normalization constants |
| `birdCallGenerator.m` | Synthesizes songs for all five species |
| `testGenerator.m` | The closed-loop test. Generates songs, classifies them, and prints a confusion matrix. If the dataset is there, it also plots real and synthetic spectrograms side by side |
| `Bird Call Dataset/bird_songs_metadata.csv` | Species, recordist, location, and license for every clip |
| `Bird Call Dataset/features_cache.mat` | Cached features for all 5,422 clips |
| `Synthetic Calls/` | Created by `testGenerator` (git-ignored) |

## Data and credits

All of the recordings come from [xeno-canto](https://xeno-canto.org) contributors, through the Kaggle [bird song data set](https://www.kaggle.com/datasets/vinayshanbhag/bird-song-data-set) put together by Vinay Shanbhag. It has 5,422 three-second clips from 477 recordings made in the United States, all 22,050 Hz mono, songs only, and only recordings with xeno-canto's top quality grade. Each clip's recordist and Creative Commons license are listed in `Bird Call Dataset/bird_songs_metadata.csv`. Every license allows non-commercial use with attribution (most are CC BY-NC-SA).
