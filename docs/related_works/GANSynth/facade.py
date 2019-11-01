import os
os.environ['CUDA_DIVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'
from pre_process import PreProcessor


wav_dir = './NSynth'

A_dist_list = []
A = PreProcessor()
A.experiment_A(wav_dir=wav_dir)
A.get_dist_meanstd(dist_list=A_dist_list)

B_dist_list = []
B = PreProcessor()
B.experiment_B(wav_dir=wav_dir)
B.get_dist_meanstd(dist_list=B_dist_list)