import numpy as np
from emhmm_toolbox import vbhmm_learn, vbhmm_plot_compact, stats_ttest

# demo_faces_jov_compare - example of comparing HMMs
#
# ---
# For each subject, we separated trials that led to correct responses
# and trials that led to incorrect (wrong) responses.
# A separate HMM is learned for correct trials and wrong trials.
# Then, we compare the "correct" HMM and "wrong" HMM.
#
# Eye-Movement analysis with HMMs (emhmm-toolbox)
# Copyright (c) 2017-01-13
# Antoni B. Chan, Janet H. Hsiao, Tim Chuk
# City University of Hong Kong, University of Hong Kong

np.random.seed(101)
np.random.seed(101)

## Load data
jov_data = np.load('jov_data.npy', allow_pickle=True)

# jov_data contains the data that was used for
# Chuk T., Chan, A. B., & Hsiao, J. H. (2014).
# Understanding eye movements in face recognition using
# hidden Markov models. Journal of Vision 14(8).
# doi:10.1167/14.11.8.

# data is stored in a list
# data[i]         = i-th subject
# data[i][j]      = ... j-th trial
# data[i][j][t,:] = ... [x y] location of t-th fixation

# the number of subjects
N = len(jov_data)

## VB Parameters
K = range(2, 4)
vbopt = {
    'alpha': 1,
    'mu': np.array([160, 210]),
    'W': 0.001,
    'beta': 1,
    'v': 10,
    'epsilon': 1,
    'showplot': 0,
}
faceimg = 'ave_face120.png'

## Learn Subject's HMMs
hmms_correct = []
hmms_wrong = []

# estimate for each subject
for i in range(N):
    print(f'=== running Subject {i+1} ===')

    # learn HMM for correct trials
    hmms_correct.append(vbhmm_learn(jov_data[i]['data_correct'], K, vbopt))

    # learn HMM for wrong trials
    hmms_wrong.append(vbhmm_learn(jov_data[i]['data_wrong'], K, vbopt))

    # plot the HMMs
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title(f'Subject {i+1} - correct')
    vbhmm_plot_compact(hmms_correct[i], faceimg, ax=axes[0])
    axes[1].set_title(f'Subject {i+1} - wrong')
    vbhmm_plot_compact(hmms_wrong[i], faceimg, ax=axes[1])
    plt.show()

## Run statistical tests
# see if correct HMMs are different from wrong HMMs.
print('=== correct vs. wrong ===')
p, info, lld = stats_ttest(hmms_correct, hmms_wrong, jov_data['data_correct'])
print('p:', p)
print('info:', info)

print('=== wrong vs. correct ===')
p, info, lld = stats_ttest(hmms_wrong, hmms_correct, jov_data['data_wrong'])
print('p:', p)
print('info:', info)
