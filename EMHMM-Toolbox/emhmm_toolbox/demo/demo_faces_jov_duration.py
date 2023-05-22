import numpy as np
from emhmm_toolbox import (
    read_xls_fixations,
    vbhmm_auto_hyperparam,
    vbhmm_learn,
    vbhmm_plot,
    vbhmm_plot_compact,
    vhem_cluster,
    vhem_plot,
)

# demo_faces_jov_duration - example of HMMs with fixation location and duration
#
# ---
# For each subject, we train an HMM. The subjects' HMMs are clustered
# using VHEM to obtain the common strategies used.
#
# Eye-Movement analysis with HMMs (emhmm-toolbox)
# Copyright (c) 2017-01-19
# Antoni B. Chan, Janet H. Hsiao, Tim Chuk
# City University of Hong Kong, University of Hong Kong

np.random.seed(101)
np.random.seed(101)

## Load data with fixation location and duration
data, SubjectIDs, TrialIDs = read_xls_fixations('jov_duration.xls')

# data is stored in a list
# data[i]         = i-th subject
# data[i][j]      = ... j-th trial
# data[i][j][t,:] = ... [x y d] location of t-th fixation. duration "d" is in milliseconds

# data is also in this mat file:
# data = np.load('jov_duration.npy', allow_pickle=True)

# the number of subjects
N = len(data)
N = 10  # only look at 10 in this demo

## VB Parameters
K = range(2, 4)
vbopt = {
    'alpha': 1,
    'mu': np.array([160, 210, 250]),  # the image center, and 250ms duration
    'W': np.array([0.001, 0.001, 0.0016]),  # stdev of 31 pixels for ROIs, and stddev of 25ms for duration
    'beta': 1,
    'v': 10,
    'epsilon': 1,
    'showplot': 0,
}

faceimg = 'ave_face120.png'

vbopt = vbhmm_auto_hyperparam(vbopt, data[:N], faceimg, 'd')

## Learn Subject's HMMs
hmms = []

# estimate for each subject
for i in range(N):
    print(f'=== running Subject {i+1} ===')
    hmms.append(vbhmm_learn(data[i], K, vbopt))

    # plot the HMMs
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title(f'Subject {i+1}')
    vbhmm_plot(hmms[i], data[i], faceimg, ax=ax)
    plt.show()

# plot each subject
for i in range(N):
    if i % 16 == 0:
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.tight_layout()

    axes[i % 16 // 4, i % 16 % 4].set_title(f'Subject {i+1}')
    vbhmm_plot_compact(hmms[i], faceimg, ax=axes[i % 16 // 4, i % 16 % 4])

plt.show()

## Run HEM clustering (1 cluster)
print('=== Clustering 1 ===')
hemopt = {
    'sortclusters': 'f',
}
group_hmms1 = vhem_cluster(hmms, 1, 3, hemopt)

# plot the groups
vhem_plot(group_hmms1, faceimg)

## Run HEM Clustering (2 clusters)
print('=== Clustering 2 ===')
group_hmms2 = vhem_cluster(hmms, 2, 3, hemopt)

# plot the groups
vhem_plot(group_hmms2, faceimg)

# show group membership
print('Group membership:')
for j, group in enumerate(group_hmms2['groups']):
    print(f'  group {j+1} = {group}')
