import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import norm
from PIL import Image

def vbhmm_plot_compact(hmm, imgfile='', plotmode='r'):
    """
    vbhmm_plot_compact - plot the compact representation of an HMM

    Args:
        hmm (dict): The HMM.
        imgfile (str): Path to the image file (optional).
        plotmode (str): Plot mode ('r' for right, 'b' for bottom) (optional).

    Returns:
        None

    ---
    Eye-Movement analysis with HMMs (emhmm-toolbox)
    Copyright (c) 2017-01-13
    Antoni B. Chan, Janet H. Hsiao, Tim Chuk
    City University of Hong Kong, University of Hong Kong
    """

    colors, colorfull = get_color_list()

    if imgfile != '':
        img = np.array(Image.open(imgfile))
        if img.ndim == 2:
            img = np.stack((img,) * 3, axis=-1)
    else:
        img = None

    if img is not None:
        xmin_img = 0
        xmax_img = img.shape[1]
        ymin_img = 0
        ymax_img = img.shape[0]
    else:
        raise NotImplementedError("Not supported yet")

    D = len(hmm['pdf'][0]['mean'])
    K = len(hmm['trans'])

    if plotmode == 'r':
        xmin_trans = xmax_img
        xmax_trans = xmax_img + min(xmax_img, ymax_img)
        ymin_trans = ymin_img
        ymax_trans = ymax_img
        if D > 2:
            xmin_dur = xmin_trans
            xmax_dur = xmax_trans
            ymin_dur = ymin_trans
            ymax_dur = ymax_trans

            w = xmax_dur - xmin_dur
            xmin_trans += w
            xmax_trans += w
    elif plotmode == 'b':
        raise NotImplementedError("Not supported yet")
    else:
        raise ValueError("Invalid plot mode")

    plot_emissions([], [], hmm['pdf'], img)
    plt.title('')

    if plotmode == '':
        plt.show()
        return

    plt.hold(True)

    if D > 2:
        xw = xmax_dur - xmin_dur
        yh = ymax_dur - ymin_dur

        dur_z = []
        dur_t = []
        dur_mu_z = []
        legs = []
        for k in range(K):
            mu = hmm['pdf'][k]['mean'][2]
            sigma2 = hmm['pdf'][k]['cov'][2, 2]
            ss = np.sqrt(sigma2)
            tmin = max(mu - 3 * ss, 0)
            tmax = mu + 3 * ss
            t = np.linspace(tmin, tmax, 100)
            z = norm.pdf(t, mu, ss)

            dur_t.append(t)
            dur_z.append(z)
            dur_mu_z.append([mu, norm.pdf(mu, mu, ss)])

            legs.append(f"\\color{{{colorfull[k]}}} {k}: {round(mu)}±{round(ss)}")

        tmin = 0
        tmax = max(np.concatenate(dur_t))
        zmin = 0
        zmax = max(np.concatenate(dur_z))

        padding = 20
        yaxpadding = 65
        textpadding = 5
        textoffset = 20
        t_map = lambda t: ((t - tmin) / (tmax - tmin)) * (xw - padding * 2) + xmin_dur + padding
        z_map = lambda z: ymax_dur - yaxpadding - ((z - zmin) / (zmax - zmin)) * (yh - yaxpadding)

        plt.fill([xmin_dur + padding, xmin_dur + padding, xmax_dur - padding, xmax_dur - padding],
                 [ymin_dur, ymax_dur - yaxpadding, ymax_dur - yaxpadding, ymin_dur],
                 'w', linewidth=0.5)

        ytext = ymax_dur - yaxpadding + textpadding
        plt.text(xmin_dur + padding, ytext, str(int(tmin)),
                 horizontalalignment='center', fontsize=7, verticalalignment='top')
        plt.text(xmax_dur - padding, ytext, str(int(tmax)),
                 horizontalalignment='center', fontsize=7, verticalalignment='top')

        for k in range(K):
            mytext = ytext + (k - 1) * textoffset
            plt.plot(t_map(dur_t[k]), z_map(dur_z[k]), color=colors[k], linewidth=2)
            plt.plot([t_map(dur_mu_z[k][0])] * 2, [mytext, z_map([dur_mu_z[k][1]])], '--', color=colors[k])
            plt.text(t_map(dur_mu_z[k][0]), mytext, str(round(dur_mu_z[k][0])),
                     horizontalalignment='center', fontsize=7, verticalalignment='top')

        for k in range(K):
            plt.text(t_map(dur_mu_z[k][0]), z_map(dur_mu_z[k][1] / 2), str(k), color=colors[k],
                     horizontalalignment='center')

        textprops = dict(color='black', horizontalalignment='right', verticalalignment='top',
                         backgroundcolor='white', fontsize=7, edgecolor='black', margin=1)
        plt.text(xmax_dur - padding, ymin_dur, "\n".join(legs), **textprops)

    padding = 40

    tx = np.linspace(xmin_trans + padding, xmax_trans, K + 1)
    tx = 0.5 * (tx[1:] + tx[:-1])

    ty = np.linspace(ymin_trans + padding, ymax_trans, K + 2)
    ty = 0.5 * (ty[1:] + ty[:-1])

    typ = ty[0] - padding
    tyt = ty[1:]

    if len(tyt) > 1:
        dy = tyt[1] - tyt[0]
    else:
        dy = ymax_trans - ymin_trans - padding
    if len(tx) > 1:
        dx = tx[1] - tx[0]
    else:
        dx = xmax_trans - xmin_trans - padding

    plt.imshow(hmm['prior'][:, np.newaxis], aspect='auto', extent=(xmin_trans + padding, xmax_trans, typ - dy / 4, typ + dy / 4), vmin=0, vmax=1)
    plt.imshow(hmm['trans'], aspect='auto', extent=(xmin_trans + padding, xmax_trans, np.min(tyt) - dy / 2, np.max(tyt)), vmin=0, vmax=1)

    for j in range(K):
        mycolor = getcolor(hmm['prior'][j])
        plt.text(tx[j], typ, f"{hmm['prior'][j]:.2f}",
                 horizontalalignment='center', fontsize=7, color=mycolor)

    for j in range(K):
        for k in range(K):
            mycolor = getcolor(hmm['trans'][j, k])
            plt.text(tx[k], tyt[j], f"{hmm['trans'][j, k]:.2f}",
                     horizontalalignment='center', fontsize=7, color=mycolor)

    for j in range(K):
        plt.text(xmin_trans + padding * 2.5 / 4, tyt[j], str(j),
                 horizontalalignment='center', fontsize=7)
        plt.text(tx[j], tyt[0] - dy / 2 - padding * 1.5 / 4, f"to {j}",
                 horizontalalignment='center', fontsize=7)

    plt.text(xmin_trans + padding * 2.5 / 4, typ, 'prior', rotation=90, fontsize=7, horizontalalignment='center')

    plt.axis([min(xmin_img, xmin_trans), max(xmax_img, xmax_trans),
              min(ymin_img, ymin_trans), max(ymax_img, ymax_trans)])
    plt.set_cmap('gray')

    plt.show()


def getcolor(p):
    if p > 0.3:
        return 'k'
    else:
        return 'w'
