# Environmental Sound Classification Research

## ESC-50 Dataset

> ###### [Overview](#esc-50-dataset-for-environmental-sound-classification) | [Download](#download) | [Results](#results) | [Repository content](#repository-content) | [License](#license) | [Citing](#citing) | [Caveats](#caveats) | [Changelog](#changelog)
>
> <a href="https://circleci.com/gh/karoldvl/ESC-50"><img src="https://circleci.com/gh/karoldvl/ESC-50.svg?style=svg" /></a>&nbsp;
<a href="LICENSE"><img src="https://img.shields.io/badge/license-CC%20BY--NC-blue.svg" />&nbsp;
<a href="https://github.com/karoldvl/ESC-50/archive/master.zip"><img src="https://img.shields.io/badge/download-.zip-ff69b4.svg" alt="Download" /></a>&nbsp;

<img src="esc50.gif" alt="ESC-50 clip preview" title="ESC-50 clip preview" align="right" />

The **ESC-50 dataset** is a labeled collection of 2000 environmental audio recordings suitable for benchmarking methods of environmental sound classification.

The dataset consists of 5-second-long recordings organized into 50 semantical classes (with 40 examples per class) loosely arranged into 5 major categories:

| <sub>Animals</sub> | <sub>Natural soundscapes & water sounds </sub> | <sub>Human, non-speech sounds</sub> | <sub>Interior/domestic sounds</sub> | <sub>Exterior/urban noises</sub> |
| :--- | :--- | :--- | :--- | :--- |
| <sub>101 - Dog</sub> | <sub>201 - Rain</sub> | <sub>Crying baby</sub> | <sub>Door knock</sub> | <sub>Helicopter</sub></sub> |
| <sub>102 - Rooster</sub> | <sub>202 - Sea waves</sub> | <sub>Sneezing</sub> | <sub>Mouse click</sub> | <sub>Chainsaw</sub> |
| <sub>103 - Pig</sub> | <sub>203 - Crackling fire</sub> | <sub>Clapping</sub> | <sub>Keyboard typing</sub> | <sub>Siren</sub> |
| <sub>104 - Cow</sub> | <sub>204 - Crickets</sub> | <sub>Breathing</sub> | <sub>Door, wood creaks</sub> | <sub>Car horn</sub> |
| <sub>105 - Frog</sub> | <sub>205 - Chirping birds</sub> | <sub>Coughing</sub> | <sub>Can opening</sub> | <sub>Engine</sub> |
| <sub>106 - Cat</sub> | <sub>206 - Water drops</sub> | <sub>Footsteps</sub> | <sub>Washing machine</sub> | <sub>Train</sub> |
| <sub>107 - Hen</sub> | <sub>207 - Wind</sub> | <sub>Laughing</sub> | <sub>Vacuum cleaner</sub> | <sub>Church bells</sub> |
| <sub>108 - Insects (flying)</sub> | <sub>208 - Pouring water</sub> | <sub>Brushing teeth</sub> | <sub>Clock alarm</sub> | <sub>Airplane</sub> |
| <sub>109 - Sheep</sub> | <sub>209 - Toilet flush</sub> | <sub>Snoring</sub> | <sub>Clock tick</sub> | <sub>Fireworks</sub> |
| <sub>110 - Crow</sub> | <sub>210 - Thunderstorm</sub> | <sub>Drinking, sipping</sub> | <sub>Glass breaking</sub> | <sub>Hand saw</sub> |

Clips in this dataset have been manually extracted from public field recordings gathered by the **[Freesound.org project](http://freesound.org/)**. The dataset has been prearranged into 5 folds for comparable cross-validation, making sure that fragments from the same original source file are contained in a single fold.

A more thorough description of the dataset is available in the original [paper](http://karol.piczak.com/papers/Piczak2015-ESC-Dataset.pdf) with some supplementary materials on GitHub: **[ESC: Dataset for Environmental Sound Classification - paper replication data](https://github.com/karoldvl/paper-2015-esc-dataset)**.

## License

The dataset is available under the terms of the [Creative Commons Attribution Non-Commercial license](http://creativecommons.org/licenses/by-nc/3.0/).

A smaller subset (clips tagged as *ESC-10*) is distributed under CC BY (Attribution).

Attributions for each clip are available in the [ LICENSE file](LICENSE).

### General description
This repo is used for ESC50 and UrbanSound8K classification using a convolutional neural net with Tensorflow. Developed on unix but should work on windows with a few tweaks. **Note currently only ESC50 supported and fully tested.**

Supported transformations:
* Linear/Mel scaled Short-time fourier transform (STFT)
* Constant-Q transform (CQT)
* Continuous Wavelet transform (CWT)
* MFCC

### Data Preparation
Required libraries:
* Python 3.5
* Jupyter Notebook
* pillow
* [librosa 0.5.1](https://librosa.github.io/librosa/install.html)
* pyWavelets (only for CWT)
* scipy
* Tensorflow (ver >1.0) for TFRecords

As input we will first generate time-frequency representations of the audio signal (i.e. spectrograms) from the .ogg or .wav files then convert them to TFRecords, Tensorflow's native data representation.
1. Go into DataPrep folder, launch and run WavToSpecConversion.ipynb
2. A small subset of the ESC50 dataset has been included in toy_data to get you started. Full datasets can be found at [ESC50](https://github.com/karoldvl/ESC-50) and [UrbanSound8K](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html).
3. If all cells were run correctly, you will now have additional folders in DataPrep. {transform}_png and {transform}_tif holds the png and tif linear spectrograms respectively where {transform} is the name of the transformation chosen. Under these folders, spectrograms have been split into their respective folds for cross-validation during training. A labels.txt file containing class labels should also have been generated.
4. Next run spect2TFRecords.sh and follow the prompts to convert the images into TFRecords. When asked, specify the folder containing the folds i.e.{transform}_png and the number of shards per fold desired. Only point to the png folder since tif is not supported. TFRecords should now be generated in {transform}_png.

### Classification
Required libraries 
* Tensorflow (ver >1.0)

1. Go to the Training folder. Inside there should be a esc50_us8K_classification.py that allows one to do classification on the ESC50 dataset with a single fold held out as a validation set.
2. params.py holds some important parameters including data paths, model and network hyperparameters. You can change the hyperparameters to suit your experiment in this file.
3. In the same params file you can also specify the number of neurons for each layer in the model. Two models are provided - model1 (1 conv layer + 2 fc) and model3 (3 conv layer + 2 fc). If you want to make more drastic changes to the model do it in the individual model files.
4. When ready, decide whether you want to run validation on a single fold or do k-fold cross validation. For single fold, run esc50_us8K_classification.py. You can specify the folder containing the TFRecords, validation fold, whether to do 1D (freq bins as channels) or 2D (freq bins as height of input) convolution and the model to use in the arguments. 
5. A bash script kfold_classification.sh has been provided for your convenience to do k-fold. Run it without any arguments and follow the prompts.

TO DO: saveState() in pickledModel.py under utils to take in dictionaries. Also double check compatibility of parameters list to the variables used for style transfer.