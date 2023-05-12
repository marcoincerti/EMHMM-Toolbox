# demo_faces - example of eye gaze analysis for face recognition
#
# ---
# Eye-Movement analysis with HMMs (emhmm-toolbox)
# Copyright (c) 2017-01-13
# Antoni B. Chan, Janet H. Hsiao, Tim Chuk
# City University of Hong Kong, University of Hong Kong

import numpy as np
import matplotlib.pyplot as plt
clear
close_('all')
# set random state to be able to replicate results
np.random.rand('state',101)
np.random.randn('state',101)
## Load data from xls ##########################
# see the xls file for the format
data,SubjNames,TrialNames = read_xls_fixations('demodata.xls')
# the data is read and separated by subject and trial, and stored in a cell array:
# data{i}         = i-th subject
# data{i}{j}      = ... j-th trial
# data{i}{j}(t,:) = ... [x y] location of t-th fixation

# the same data is stored in a mat file.
# load demodata.mat

# the number of subjects
N = len(data)
## VB Parameters ################################

K = np.arange(2,3+1)

vbopt.alpha = 0.1
vbopt.mu = np.array([[256],[192]])
vbopt.W = 0.005
vbopt.beta = 1
vbopt.v = 5
vbopt.epsilon = 0.1
vbopt.showplot = 0
faceimg = 'face.jpg'
## Learn Subject's HMMs #####################
# estimate for each subject
for i in np.arange(1,N+1).reshape(-1):
    print('=== running Subject %d ===\n' % (i))
    hmms[i] = vbhmm_learn(data[i],K,vbopt)

# show subject 1
vbhmm_plot(hmms[0],data[0],faceimg)
figure
vbhmm_plot_compact(hmms[0],faceimg)
# plot each subject
figure
for i in np.arange(1,N+1).reshape(-1):
    subplot(4,3,i)
    vbhmm_plot_compact(hmms[i],faceimg)
    plt.title(sprintf('Subject %d',i))

## Run HEM clustering (1 group) ############################
# summarize all subjects with one HMM
print('=== Clustering (1 group) ===\n' % ())
hemopt.sortclusters = 'f'
all_hmms1 = vhem_cluster(hmms,1,3,hemopt)
# plot the overall HMM
vhem_plot(all_hmms1,faceimg)
## Run HEM Clustering (2 groups) ##########################
# cluster subjects into 2 groups
print('=== Clustering (2 groups) ===\n' % ())
group_hmms2 = vhem_cluster(hmms,2,3,hemopt)
# plot the groups
vhem_plot(group_hmms2,faceimg)
# plot the groups and cluster members
vhem_plot_clusters(group_hmms2,hmms,faceimg)
# show group membership
print('Group membership: \n' % ())
for j in np.arange(1,len(group_hmms2.groups)+1).reshape(-1):
    print('  group %d = %s\n' % (j,mat2str(group_hmms2.groups[j])))

## Statistical test ###########################
# collect data for group 1 and group 2
data1 = np.array([data[group_hmms2.groups[0]]])
data2 = np.array([data[group_hmms2.groups[2]]])
# run t-test for hmm1
p,info,lld = stats_ttest(group_hmms2.hmms[0],group_hmms2.hmms[2],data1)
print('test group hmm1 different from group hmm2: t(%d)=%g; p=%g\n' % (info.df,info.tstat,p))
# run t-test for hmm2
p,info,lld = stats_ttest(group_hmms2.hmms[2],group_hmms2.hmms[0],data2)
print('test group hmm2 different from group hmm1: t(%d)=%g; p=%g\n' % (info.df,info.tstat,p))