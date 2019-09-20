import os
import numpy as np
import numpy.matlib
import librosa
import scipy
import pngspect
import pywt


# Set some project parameters
K_SR = 22050 # sampling rate
K_FFTSIZE = 512 # also used for window length where that parameter is called for
K_HOP = 256 # fft hop length
K_DUR = 5.0 # make all files this duration (sec)

K_SCALEW = None
K_SCALEH = 65

# include trailing slash!!
main_dir = '/home/201850854/compareTF/DataPrep/'

# location of subdirectories of ogg files organized with subdirectories named by category
K_OGGDIR = main_dir + 'ESC-50'

# location to write the wav files (converted from ogg)
K_WAVEDIR = main_dir + 'ESC-50-wav'

# Probably
K_QSPECTDIR = main_dir + 'constQ_v2' # name to indicate when you do the const Q? (see assignment below for "small" switch)

def get_subdirs(a_dir):
    """ Returns a list of sub directory names in a_dir """
    return [
        name for name in os.listdir(a_dir)
        if (os.path.isdir(os.path.join(a_dir, name)) and not (name.startswith('.')))
    ]

def listDirectory(directory, fileExtList):
    """ Returns a list of file info objects in directory that extension in the list fileExtList - include the . in your extension string """
    fnameList = [
        os.path.normcase(f)
        for f in os.listdir(directory)
            if (not(f.startswith('.')))
    ]

    fileList = [
        os.path.join(directory, f)
        for f in fnameList
            if os.path.splitext(f)[1] in fileExtList
    ]

    return fileList, fnameList

def dirs2labelfile(parentdir, labelfile):
    """ takes subdirectories of parentdir and writes them, one per line, to labelfile """
    namelist = get_subdirs(parentdir)
    with open(labelfile, mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(namelist))


# First choose the desired transform
transform = 'stft' # choose the signal processing technique: ['stft', 'mel', 'cqt', 'cwt', 'mfcc']

if transform == 'stft':
    # location to write the linear spectrogram files (converted from wave files)
    K_SPECTDIR = main_dir + 'stft'

    # If resampling spectrogram, write the params in the directory name
    if (K_SCALEH != None) :
        K_SPECTDIR = K_SPECTDIR + '.' + str(K_SCALEH)
    if (K_SCALEW != None) :
        K_SPECTDIR = K_SPECTDIR + '.' + str(K_SCALEW)

elif transform == 'mel':
    # location to write the mel spec files (converted from wav files)
    K_SPECTDIR = main_dir + 'mel'

elif transform == 'cqt':
    # location to write the CQT spec files (converted from wav files)
    K_SPECTDIR = main_dir + 'cqt'

elif transform == 'cwt':
    # location to write the wavelet files (converted from wav files)
    K_SPECTDIR = main_dir + 'cwt'

elif transform == 'mfcc':
    # location to write the mfcc files (converted from wav file)
    K_SPECTDIR = main_dir + 'mfcc'

else:
    raise ValueError('Transform not supported! Please choose from ["stft", "mel", "cqt", "cwt", "mfcc"]')


# routines to convert wav to spectrograms using Librosa as backend. For wavelets we are using Pywavlet library.from
def wav2stft(fname, srate, fftSize, fftHop, y_scale='linear', dur=None, showplt=False, dcbin=True):
    try:
        audiodata, samplerate = librosa.load(fname, sr=srate, mono=True, duration=dur)
    except:
        print('can not read ' + fname)
        return
    S = np.abs(librosa.stft(audiodata, n_fft=fftSize, hop_length=fftHop, win_length=fftSize, center=False))

    if (dcbin == False) :
        S = np.delete(S, (0), axis=0) # delete freq 0 row
        # note: a pure DC input signal bleeds into bin 1, too.

    D = librosa.amplitude_to_db(S, ref=np.max)

    return D

def wav2cqt(fname, srate, fftSize, fftHop, dur=None, showplt=False) :
    try:
        audiodata, samplerate = librosa.load(fname, sr=srate, mono=True, duration=dur)
    except:
        print('can not read ' + fname)
        return

    S = librosa.cqt(audiodata, hop_length=fftHop, n_bins=fftSize//2, bins_per_octave=32)

    D = librosa.amplitude_to_db(S, ref=np.max)

    return D

def wav2cwt(fname, srate, freq_bins, wavelet='morl', dur=None, showplt=False) :

    try:
        audiodata, samplerate = librosa.load(fname, sr=srate, mono=True, duration=dur)
    except:
        print('can not read ' + fname)
        return

    widths = np.arange(1, freq_bins+1) # no.of freq bins

    S, freqs = pywt.cwt(audiodata, widths, wavelet, sampling_period=1/srate)

    D = librosa.amplitude_to_db(S, ref=np.max)

    return D

def wav2mel(fname, srate, fftSize, fftHop, n_mels=128, dur=None, showplt=False):
    try:
        audiodata, samplerate = librosa.load(fname, sr=srate, mono=True, duration=dur)
    except:
        print('can not read ' + fname)
        return

    S = librosa.feature.melspectrogram(audiodata, sr=srate, n_fft=fftSize, hop_length=fftHop, n_mels=n_mels)

    D = librosa.power_to_db(S, ref=np.max)

    return D

def wav2mfcc(fname, srate, fftSize, fftHop, n_mfcc=128, dur=None, showplt=False):
    try:
        audiodata, samplerate = librosa.load(fname, sr=srate, mono=True, duration=dur)
    except:
        print('can not read ' + fname)
        return

    S = librosa.feature.mfcc(audiodata, sr=srate, n_fft=fftSize, hop_length=fftHop, n_mfcc=n_mfcc)

    return S

def esc50get_fold(string):
    """ get fold no. from ESC-50 dataset using the filename. Labels #1-5 """
    fold = ''
    try:
        label_pos = string.index('-')
        fold=string[label_pos-1]
    except:
        fold='1'
    return fold

def wav2Spect(topdir, outdir, dur, srate, fftSize, fftHop, transform, neww=None, newh=None) :
    """
        Creates spectrograms for subfolder-labeled wavfiles.
        Create class folders for the spectrogram files in outdir with the same structure found in topdir.

        Parameters
            topdir - the dir containing class folders containing wav files.
            outdir - the top level directory to write wave files to (written in to class subfolders)
            dur - (in seconds) all files will be truncated or zeropadded to have this duration given the srate
            srate - input files will be resampled to srate as they are read in before being saved as wav files
            transform - desired transform, defined at the top of this code

        Scaling Params
            neww - scale width of spectrogram (in pixels) to this value
            newh - scale height of spectrogram (in pixels) to this value. maintains aspect ratio if only neww is specified
            * if both neww and newh is left unspecified no scaling is carried out
    """
    subdirs = get_subdirs(topdir)
    count = 0
    for subdir in subdirs:

        fullpaths, _ = listDirectory(topdir + '/' + subdir, '.wav')

        for idx in range(len(fullpaths)):
            fname = os.path.basename(fullpaths[idx])
            fold = esc50get_fold(fname)

            if transform == "stft":
                D = wav2stft(fullpaths[idx], srate, fftSize, fftHop)
            elif transform == "mel":
                D = wav2mel(fullpaths[idx], srate, fftSize, fftHop)
            elif transform == "cqt":
                D = wav2cqt(fullpaths[idx], srate, fftSize, fftHop)
            elif transform == "cwt":
                D = wav2cwt(fullpaths[idx], srate, 64, 'morl')
            elif transform == "mfcc":
                D = wav2mfcc(fullpaths[idx], srate, fftSize, fftHop)

            try:
                os.stat(outdir + '_png/' + fold + '/' + subdir)  # test for existence
            except:
                os.makedirs(outdir + '_png/' + fold + '/' + subdir)  # create if necessary

            pngspect.logSpect2PNG(D, outdir + '_png/' + fold + '/' + subdir + '/' + os.path.splitext(fname)[0] + '.png', neww, newh)
            print(str(count) + ': ' + subdir + '/' + os.path.splitext(fname)[0])
            count += 1
    print("COMPLETE")

# alternative CQT transform below, which converts from a linear spectrogram
# from lonce wyse: https://github.com/lonce/audioST/blob/master/dataPrep/ESC50_Convert.ipynb
def logfmap(I, L, H):
    """

        % [M, N] = logfmap(I, L, H)

        :param I: number of rows in the original spectrogram
        :param L: low bin to preserve
        :param H: high bin to preserve

        :return:

        Return a matrix for premultiplying spectrogram to map the rows into a log frequency space.
        Output map covers bins L to H of input L must be larger than 1, since the lowest bin of the FFT
        (corresponding to 0 Hz) cannot be represented on a log frequency axis.

        Including bins close to 1 makes the number of output rows exponetially larger.
        N returns the recovery matrix such that N*M is approximately I(for dimensions L to H).

        Ported from MATLAB code written by Dan Ellis: 2004-05-21 dpwe@ee.columbia.edu
    """
    ratio = (H-1)/H;
    opr = np.int(np.round(np.log(L/H)/np.log(ratio)))   # number of frequency bins in log req + 1
    print('opr is ' + str(opr))
    ibin = L*np.exp(list(range(0,opr)*-np.log(ratio)))  # fractional bin numbers (len(ibin) = opr - 1)

    M = np.zeros((opr, I))
    eps = np.finfo(float).eps

    for i in range(0, opr):
        # Where do we sample this output bin?
        # Idea is to make them 1:1 at top, and progressively denser below
        # i.e. i = max -> bin = topbin, i = max-1 -> bin = topbin-1,
        # but general form is bin = A exp (i/B)

        tt = np.multiply(np.pi, (list(range(0,I)) - ibin[i]))
        M[i,:] = np.divide((np.sin(tt)+eps), (tt+eps))

    # Normalize rows, but only if they are boosted by the operation
    G = np.ones((I))
    print('H is ' + str(H))
    G[0:H] = np.divide(list(range(0,H)), H)

    N = np.transpose(np.multiply(M,np.matlib.repmat(G,opr,1)))

    return M, N

def esc50Spect2logFreqSpect(topdir, outdir, srate, fftSize, fftHop, lowRow=1, neww=None, newh=None):
    """

        Create psuedo constant-Q spectrograms from lineaqr frequency spectrograms.
        Create class folders in outdir with the same structure found in topdir.


        :param topdir: the dir containing class folders containing png (log magnitude) spectrogram files.
        :param outdir: the top level directory to write psuedo constan-Q files to (written in to class subforlders)
        :param lowRow: is the lowest row in the FFT that you want to include in the psuedo constant-Q spectrogram

    """
    # First lets get the logf map we want
    LIN_FREQ_BINS = int(fftSize/2+1)    # number of bins in original linear frequency mag spectrogram
    LOW_ROW = lowRow
    LOG_FREQ_BINS = int(fftSize/2+1)    # resample the lgfmapped psuedo constant-Q matrix to have this many frequency bins
    M, N = logfmap(LIN_FREQ_BINS, LOW_ROW, LOG_FREQ_BINS)

    folds = get_subdirs(topdir)
    count = 0
    for fold in folds:
        subdir = get_subdirs(topdir + '/' + fold)
        for classes in subdir:
            try:
                os.stat(outdir + '_png/' + fold + '/' + classes)        # test for existence
            except:
                os.makedirs(outdir + '_png/' + fold + '/' + classes)    # create if necessary
            fullpaths, _ = listDirectory(topdir + '/' + fold + '/' + classes, '.png')

            for idx in range(len(fullpaths)):
                fname = os.path.basename(fullpaths[idx])
                D, pnginfo = pngspect.PNG2LogSpect(fullpaths[idx])

                # Here's the beef
                MD = np.dot(M, D)
                MD = scipy.signal.resample(MD, LIN_FREQ_BINS)   # downsample to something reasonable

                # save
                # info = {}
                pnginfo['linFreqBins'] = LIN_FREQ_BINS
                pnginfo['lowRow'] = LOW_ROW
                pnginfo['logFreqBins'] = LOG_FREQ_BINS
                pngspect.logSpect2PNG(MD, outdir + '_png/' + fold + '/' + classes + '/' + os.path.splitext(fname)[0] + '.png', neww, newh, lwinfo=pnginfo)
                print(str(count) + ': ' + classes + '/' + os.path.splitext(fname)[0])
                count += 1
    print('COMPLETE')




# wav2Spect(K_WAVEDIR, K_SPECTDIR, K_DUR, K_SR, K_FFTSIZE, K_HOP, transform, neww=K_SCALEW, newh=K_SCALEH)
dirs2labelfile(K_SPECTDIR + '_png/1', main_dir + '/lables.txt')

# Probably
# if (K_SCALEH != None):
#     K_QSPECTDIR = K_QSPECTDIR + '.' + str(K_SCALEH)
# if (K_SCALEW != None):
#     K_QSPECTDIR = K_QSPECTDIR + '.' + str(K_SCALEW)
#
# esc50Spect2logFreqSpect(K_SPECTDIR + '_png', K_QSPECTDIR, K_SR, K_FFTSIZE, K_HOP, lowRow=10)
