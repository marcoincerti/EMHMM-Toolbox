import numpy as np
from emhmm_toolbox import vbhmm_learn, vbhmm_plot_compact, vhem_cluster, vhem_plot

# demo_faces_jov_clustering - example of HMM clustering
#
# ---
# For each subject, we train an HMM. The subjects' HMMs are clustered
# using VHEM to obtain the common strategies used.
#
# Eye-Movement analysis with HMMs (emhmm-toolbox)
# Copyright (c) 2017-01-13
# Antoni B. Chan, Janet H. Hsiao, Tim Chuk
# City University of Hong Kong, University of Hong Kong

np.random.seed(101)
np.random.seed(101)

## Load data
jov_data = np.load('jov_data.npy', allow_pickle=True)

# jov_data contains the data that was used in
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
hmms = []

# estimate for each subject
for i in range(N):
    print(f'=== running Subject {i+1} ===')
    hmms.append(vbhmm_learn(jov_data[i], K, vbopt))

    # plot the HMMs
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title(f'Subject {i+1}')
    vbhmm_plot_compact(hmms[i], faceimg, ax=ax)
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
vhem_plot(group_hmms1, faceimg)

## Run HEM Clustering (2 clusters)
print('=== Clustering 2 ===')
group_hmms2 = vhem_cluster(hmms, 2, 3, hemopt)
vhem_plot(group_hmms2, faceimg)

# show group membership
print('Group membership:')
for j, group in enumerate(group_hmms2['groups']):
    print(f'  group {j+1} = {group}')

## Run HEM Clustering (3 clusters)
print('=== Clustering 3 ===')
group_hmms3 = vhem_cluster(hmms, 3, 3, hemopt)
vhem_plot(group_hmms3, faceimg)

# show group membership
print('Group membership:')
for j, group in enumerate(group_hmms3['groups']):
    print(f'  group {j+1} = {group}')
