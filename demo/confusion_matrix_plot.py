import numpy as np

A = np.load('./fold5_ch2.npy')
target_names = [
    'Pig',
    'Glass breaking',
    'Engine',
    'Door knock',
    'Sneezing',
    'Keyboard typing',
    'Sheep',
    'Sea waves',
    'Coughing',
    'Washing machine',

    'Church bells',
    'Cow',
    'Vaccum cleaner',
    'Frog',
    'Rooster',
    'Thunderstorm',
    'Crackling fire',
    'Clock tick',
    'Rain',
    'Insects',

    'Airplane',
    'Fireworks',
    'Hen',
    'Brushing teeth',
    'Footsteps',
    'Crow',
    'Crickets',
    'Pouring water',
    'Car horn',
    'Siren',

    'Water drops',
    'Clock alarm',
    'Door - wood creaks',
    'Wind',
    'Chainsaw',
    'Snoring',
    'Helicopter',
    'Can opening',
    'Chirping birds',
    'Hand saw',

    'Dog',
    'Laughing',
    'Clapping',
    'Crying baby',
    'Toilet flush',
    'Train',
    'Breathing',
    'Cat',
    'Mouse click',
    'Drinking - sipping'
]
def plot_confusion_matrix(cm,
                          title='Confusion matrix',
                          target_names=None,
                          cmap=None):

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Reds')

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90, fontsize=5)
        plt.yticks(tick_marks, target_names, fontsize=5)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                 horizontalalignment="center",
                 fontsize=5,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

plot_confusion_matrix(
    cm= A,
    target_names=target_names
)