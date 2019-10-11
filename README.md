# Environmental Sound Classification Research

### ESC-50 Dataset

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

### License

The dataset is available under the terms of the [Creative Commons Attribution Non-Commercial license](http://creativecommons.org/licenses/by-nc/3.0/).

A smaller subset (clips tagged as *ESC-10*) is distributed under CC BY (Attribution).

Attributions for each clip are available in the [ LICENSE file](LICENSE).