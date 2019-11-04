MAIN_DIR = '/home/201850854/compareTF'
RES_DIR = MAIN_DIR + '/res/'
WAV_DIR = MAIN_DIR + '/res/ESC-50-wav'


# Single channel input
MAG = RES_DIR + 'Mag/'
MAG_H = RES_DIR + 'Mag+H/'
MAG_MEL = RES_DIR + 'Mag-Mel/'
MAG_MEL_H = RES_DIR + 'Mag-Mel+H/'


# Double channel input
IF = RES_DIR + 'IF/'                # case 1
IF_MEL = RES_DIR + 'IF-Mel/'        # case 2
PHASE = RES_DIR + 'Phase/'           # case 3

IF_H = RES_DIR + 'IF+H/'             # case 4
IF_MEL_H = RES_DIR + 'IF-Mel+H/'    # case 5
PHASE_H = RES_DIR + 'Phase+H/'      # case 6

PHASE_MEL = RES_DIR + 'Phase-Mel/'
PHASE_MEL = RES_DIR + 'Phsae-Mel+H/'


OVERLAP_RATIO = 0.75
MEL_DOWNSCALE = 2