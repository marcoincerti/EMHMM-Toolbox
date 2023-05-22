import numpy as np
from emhmm_v2.util.read_xls_fixations import read_xls_fixations
from emhmm_v2.hmm.vbhmm_learn import vbhmm_learn

# demo_faces - example of eye gaze analysis for face recognition
#
# ---
# Eye-Movement analysis with HMMs (emhmm-toolbox)
# Copyright (c) 2017-01-13
# Antoni B. Chan, Janet H. Hsiao, Tim Chuk
# City University of Hong Kong, University of Hong Kong

np.random.seed(101)
np.random.seed(101)

# Load data from xls
data, SubjNames, TrialNames = read_xls_fixations('/Users/marcoincerti/Desktop/EMHMM-Toolbox/emhmm_v2/src/demo/demodata.xls')

# the data is read and separated by subject and trial, and stored in a list:
# data[i]         = i-th subject
# data[i][j]      = ... j-th trial
# data[i][j][t,:] = ... [x y] location of t-th fixation

# the same data is stored in a mat file.
# load demodata.mat

# the number of subjects
N = len(data)

# VB Parameters
K = range(2, 4)  # automatically select from K=2 to 3
vbopt = {
    'alpha': 0.1,
    'mu': np.array([256, 192]),
    'W': 0.005,
    'beta': 1,
    'v': 5,
    'epsilon': 0.1,
    'showplot': 0,
}
faceimg = 'face.jpg'

# Learn Subject's HMMs
hmms = []
# estimate for each subject
for i in range(N):
    print(f'=== running Subject {i+1} ===')
    hmms.append(vbhmm_learn(data[i], K, vbopt))

# show subject 1
vbhmm_plot(hmms[0], data[0], faceimg)
vbhmm_plot_compact(hmms[0], faceimg)

# plot each subject
import matplotlib.pyplot as plt
fig, axes = plt.subplots(4, 3, figsize=(10, 10))
for i in range(N):
    ax = axes.flatten()[i]
    vbhmm_plot_compact(hmms[i], faceimg)
    ax.set_title(f'Subject {i+1}')
plt.tight_layout()

""" # Run HEM clustering (1 group)
# summarize all subjects with one HMM
print('=== Clustering (1 group) ===')
hemopt = {'sortclusters': 'f'}
all_hmms1 = vhem_cluster(hmms, 1, 3, hemopt)  # 1 group, 3 hidden states
vhem_plot(all_hmms1, faceimg)

# Run HEM Clustering (2 groups)
# cluster subjects into 2 groups
print('=== Clustering (2 groups) ===')
group_hmms2 = vhem_cluster(hmms, 2, 3, hemopt)  # 2 groups, 3 hidden states
vhem_plot(group_hmms2, faceimg)
vhem_plot_clusters(group_hmms2, hmms, faceimg)

# show group membership
print('Group membership:')
for j, group in enumerate(group_hmms2.groups):
    print(f'  group {j+1} = {group}')

# Statistical test
# collect data for group 1 and group 2
data1 = [data[idx] for idx in group_hmms2.groups[0]]
data2 = [data[idx] for idx in group_hmms2.groups[1]]

# run t-test for hmm1
p, info, lld = stats_ttest(group_hmms2.hmms[0], group_hmms2.hmms[1], data1)
print(f'test group hmm1 different from group hmm2: t({info["df"]})={info["tstat"]}; p={p}')

# run t-test for hmm2
p, info, lld = stats_ttest(group_hmms2.hmms[1], group_hmms2.hmms[0], data2)
print(f'test group hmm2 different from group hmm1: t({info["df"]})={info["tstat"]}; p={p}')
 """