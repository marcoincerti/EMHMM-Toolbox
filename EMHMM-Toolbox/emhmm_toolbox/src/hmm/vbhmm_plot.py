import numpy as np
import matplotlib.pyplot as plt


def vbhmm_plot(hmm, data, imgfile='', grpnames=[]):
    if imgfile:
        img = plt.imread(imgfile)
    else:
        img = None

    D = len(hmm['pdf'][0]['mean'])

    if 'group_ids' in hmm:
        usegroups = True
        py = 1
    else:
        usegroups = False
        py = 2

    if usegroups:
        G = len(hmm['group_ids'])
        px = 2
        py = int(np.ceil(G / px))

        fig1, axs1 = plt.subplots(py, px)
        fig2, axs2 = plt.subplots(py, px)
        fig3, axs3 = plt.subplots(py, px)
        fig4, axs4 = plt.subplots(py, px)
        fig5, axs5 = plt.subplots(py, px)

        if D > 2:
            fig6, axs6 = plt.subplots(py, px)

        for g in range(G):
            if grpnames:
                glab = grpnames[g]
            else:
                glab = f"group {hmm['group_ids'][g]}"

            gdata = data[hmm['group_inds'][g]]
            ggamma = hmm['gamma'][hmm['group_inds'][g]]
            gN = hmm['Ng'][g]

            ax1 = axs1.flatten()[g]
            plot_fixations(gdata, img, 0, ax1)
            ax1.set_title(f"fixations {glab}")

            ax2 = axs2.flatten()[g]
            plot_emissions(gdata, ggamma, hmm['pdf'], img, ax2)
            ax2.set_title(f"emissions {glab}")

            ax3 = axs3.flatten()[g]
            plot_emcounts(gN, ax3)
            ax3.set_title(f"counts {glab}")

            ax4 = axs4.flatten()[g]
            plot_transprob(hmm['trans'][g], ax4)
            ax4.set_title(f"trans {glab}")

            ax5 = axs5.flatten()[g]
            plot_prior(hmm['prior'][g], ax5)
            ax5.set_title(f"prior {glab}")

            if D > 2:
                ax6 = axs6.flatten()[g]
                plot_emissions_dur(data, ggamma, hmm['pdf'], ax6)
                ax6.set_title(f"emissions duration {glab}")

        plt.show()
        return

    if D == 2:
        px = 3
    else:
        px = 4

    fig, axs = plt.subplots(py, px)
    ind = 1

    ax1 = axs.flatten()[ind-1]
    plot_fixations(data, img, hmm['LL'], ax1)
    ind += 1

    ax2 = axs.flatten()[ind-1]
    plot_emissions(data, hmm['gamma'], hmm['pdf'], img, ax2)
    ind += 1

    if D > 2:
        ax3 = axs.flatten()[ind-1]
        plot_emissions_dur(data, hmm['gamma'], hmm['pdf'], ax3)
        ind += 1

    ax4 = axs.flatten()[ind-1]
    plot_emcounts(hmm['N'], ax4)
    ind += 1

    ax5 = axs.flatten()[ind-1]
    plot_transcount(hmm['M'], ax5)
    ind += 1

    ax6 = axs.flatten()[ind-1]
    plot_transprob(hmm['trans'], ax6)
    ind += 1

    ax7 = axs.flatten()[ind-1]
    plot_prior(hmm['prior'], ax7)

    plt.show()


def plot_fixations(data, img, LL, ax):
    ax.plot(data[:, 0], data[:, 1], 'kx')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Fixations (Log-likelihood: {LL})')

    if img is not None:
        ax.imshow(img, extent=[0, 1, 0, 1], alpha=0.5)


def plot_emissions(data, gamma, pdf, img, ax):
    for i, point in enumerate(data):
        x, y = point
        color = np.array([0, 0, 0]) if img is None else img[int(y * img.shape[0]), int(x * img.shape[1]), :]
        ax.plot(x, y, 'o', markersize=6, markerfacecolor=color, markeredgecolor='k')

        for j, comp in enumerate(pdf):
            mu = comp['mean']
            cov = comp['cov']
            weight = gamma[i, j]
            draw_ellipse(mu, cov, weight, ax)

    ax.set_aspect('equal', 'box')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Emissions')


def plot_emissions_dur(data, gamma, pdf, ax):
    for i, point in enumerate(data):
        x, y = point
        duration = point[2]
        color = duration_to_color(duration)
        ax.plot(x, y, 'o', markersize=6, markerfacecolor=color, markeredgecolor='k')

        for j, comp in enumerate(pdf):
            mu = comp['mean']
            cov = comp['cov']
            weight = gamma[i, j]
            draw_ellipse(mu, cov, weight, ax)

    ax.set_aspect('equal', 'box')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Emissions Duration')


def plot_emcounts(N, ax):
    ax.bar(range(len(N)), N)
    ax.set_xlabel('Component')
    ax.set_ylabel('Count')
    ax.set_title('Component Counts')


def plot_transcount(M, ax):
    ax.bar(range(len(M)), M)
    ax.set_xlabel('From')
    ax.set_ylabel('Count')
    ax.set_title('Transition Counts')


def plot_transprob(trans, ax):
    ax.imshow(trans, cmap='viridis')
    ax.set_xlabel('To')
    ax.set_ylabel('From')
    ax.set_title('Transition Probabilities')


def plot_prior(prior, ax):
    ax.bar(range(len(prior)), prior)
    ax.set_xlabel('Component')
    ax.set_ylabel('Prior')
    ax.set_title('Priors')


def draw_ellipse(mu, cov, weight, ax):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(5.991 * eigenvalues)

    ax.add_patch(plt.Ellipse(mu, width, height, angle, alpha=weight, edgecolor='k', facecolor='none'))


def duration_to_color(duration):
    color_map = plt.cm.get_cmap('plasma')
    normalized_duration = (duration - np.min(duration)) / (np.max(duration) - np.min(duration))
    return color_map(normalized_duration)

