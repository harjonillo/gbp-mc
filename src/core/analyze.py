# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 15:17:36 2018

@author: hannah

visualizer version 2
"""

import numpy as np
import matplotlib.pyplot as pl
import gaussian_beam_propagation as gbp
import intersect
import zipfile

dir1 = 'data/iplt1arjonillo-6/'
dir0 = 'data/no_curve/'

NA_list = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
cross_corr_nocorrect = np.array([0.87, 0.93, 0.95, 0.96, 0.96, 0.96, 0.96])

def cross_correlate_1d(signal_a, signal_b):
    a = (signal_a - np.mean(signal_a))/(np.std(signal_a)*len(signal_a))
    b = (signal_b - np.mean(signal_b))/(np.std(signal_b))
    cross_correlation = np.correlate(a, b, 'same')

    # pl.figure()
    # pl.plot(signal_a, label='Signal A')
    # pl.plot(signal_b, label='Signal B')
    # pl.plot(cross_correlation, label='Cross correlation')
    # pl.legend(loc='best')
    # pl.savefig('cross_correlation_test.png', dpi=200)
    # pl.close()

    return cross_correlation[int(len(cross_correlation)//2)]


def optimize_step_parameter(curvature_correction, num_photons, NA):
    filename = dir1 + 'gbp_c%d_N%.0E_NA0%d' % \
        (int(curvature_correction), num_photons, NA*10)
    filename = filename.replace('+', '')
    filename = filename.replace('_', '')
    filename = filename + '_correlation-info.npz'
    file = np.load(filename)
    keys = file.files
    step_param_array = file[keys[0]]
    cross_correlation_array = file[keys[1]]

    cross_correlation_array[np.isnan(cross_correlation_array)] = 0
    # print(cross_correlation_array)
    cross_correlation_ave = np.mean(cross_correlation_array, axis=1)
    cross_correlation_std = np.std(cross_correlation_array, axis=1)
    max_corr = np.argmax(cross_correlation_ave)

    y_err = (cross_correlation_std, cross_correlation_std)

    crosscorr0 = cross_corr_nocorrect[np.int_(NA_list*10) == int(NA*10)]
    print (crosscorr0)

    pl.figure()
    label = '%.2f: %.2E, %.2f' % (NA, step_param_array[max_corr],
                                  cross_correlation_ave[max_corr])
    # print(step_param_array.shape)
    pl.plot(step_param_array, crosscorr0*np.ones(shape=step_param_array.shape),
            'gray', label='Cross correlation for no curvature correction')
    pl.semilogx(step_param_array, cross_correlation_ave, ':', linewidth=0.2,
                label=label)
    pl.errorbar(step_param_array, cross_correlation_ave, yerr=y_err,
                linestyle=':', linewidth=1.0)
    pl.legend(loc='best')
    filename = filename.replace('npz', 'png')
    pl.savefig(filename, dpi=300)

    return step_param_array[max_corr], cross_correlation_ave[max_corr]


def see_vars(curvature_correction, N_photons, NA, trials, stepparam):
    filename = dir+'gbp_c%d_N%.0E_NA0%d_sp%.0E' % \
        (curvature_correction, N_photons, NA*10, stepparam)
    filename = filename.replace('+', "")
    datafilename = filename+'_t%d.npz' % (1)
    filename = filename.replace('_', '')
    npzfile = np.load(datafilename)
    vars = npzfile['vars'][0]
    print(vars)


def average_profiles(curvature_correction, N_photons, NA, trials, stepparam=None,
                     save_fig=True):
    # filename = dir+'gbp_c%d_N%.0E_NA0%d_sp%.0E' % \
    #     (curvature_correction, N_photons, NA*10, stepparam)
    # filename = filename.replace('+', "")
    # datafilename = filename+'_t%d.npz' % (1)
    # datafilename = datafilename.replace('_', '')

    if curvature_correction == 0:
        dir = dir0
    else:
        dir = dir1

    filename = dir+f'gbpc%dN%.0ENA0%dsp%.0Et%d.npz' % \
        (curvature_correction, N_photons, NA*10, stepparam,1)
    filename = filename.replace('+','')
    npzfile = np.load(filename)
    vars = npzfile['vars'][0]

    # print('processing '+filename+'...')

    ax_res = vars['axial_resolution']
    path_bins = np.linspace(0, 1e7, int(np.cbrt(N_photons)))
    # print(path_bins.shape)
    H_ave = np.zeros(shape=(trials, ax_res, ax_res))
    ax_int = np.zeros(shape=(trials, ax_res))
    trans_int = np.zeros(shape=(trials, ax_res))
    path_lengths = np.zeros(shape=(trials, path_bins.shape[0]-1))
    edges = np.zeros(shape=(trials, path_bins.shape[0]))


    for t in np.arange(0, trials, 1):
        # print('trial ' + str(t))
        filename = dir+f'gbpc%dN%.0ENA0%dsp%.0Et%d.npz' % \
            (curvature_correction, N_photons, NA*10, stepparam,t)
        filename = filename.replace('+','')

        # print(filename)
        try:
            npzfile = np.load(filename)
        except (zipfile.BadZipFile, AttributeError) as e:
            print(f'Attribute Error for N{N_photons} stepparam{stepparam}')
            return

        NA = vars['NA']
        H_ave[t] = npzfile['intensity_distribution']
        transverse_edges = npzfile['transverse_edges']
        axial_edges = npzfile['axial_edges']
        # path_lengths[t], edges[t] = np.histogram(npzfile['path_lengths'],
        #                                          bins=path_bins)
        trans_int[t] = H_ave[t, :, H_ave[t].shape[0]//2]
        ax_int[t] = H_ave[t, H_ave[t].shape[0]//2, :]

    H_ave = np.mean(H_ave, axis=0)
    if np.amax(H_ave) != 0:
        H_ave = H_ave/np.amax(H_ave)
    # ax_int_stdev = np.std(ax_int, axis=0)
    ax_int = np.mean(ax_int, axis=0)
    # trans_int_stdev = np.std(ax_int, axis=0)
    trans_int = np.mean(trans_int, axis=0)

    return H_ave, ax_int, axial_edges, trans_int, transverse_edges, vars


def linfoot(signal_expt, signal_theo):
    '''
    C: relative structural content
    F: fidelity
    Q: correlation quality
    '''
    C = np.mean(signal_expt**2) / np.mean(signal_theo**2)
    F = 1 - np.mean((signal_expt - signal_theo)**2)/np.mean(signal_theo**2)
    Q = np.mean(signal_expt*signal_theo) / np.mean(signal_theo**2)

    return C, F, Q


def plot_profiles(NA, step_param, save_fig=False):
    H0, ax0, axe0, trans0, transe0, vars0 = average_profiles(curvature_correction=0, N_photons=1E5, NA=NA,
                                              stepparam=0, trials=10)
    H1, ax1, axe1, trans1, transe1, vars1 = average_profiles(curvature_correction=1, N_photons=1E5, NA=NA,
                                              stepparam=step_param, trials=10)

    trans_img_range = vars1['trans_img_range']
    axial_img_range = vars1['axial_img_range']
    ax_res = vars1['axial_resolution']
    N_photons = vars1['num_photons']
    filename = 'gbpN%.0ENA0%dsp%.0E' % (N_photons, NA*10, step_param)
    filename = dir1 + filename
    filename = filename.replace('+','')

    x = np.linspace(trans_img_range[0], trans_img_range[1], ax_res)
    z = np.linspace(axial_img_range[0], axial_img_range[1], ax_res)

    z2 = np.linspace(axial_img_range[0], axial_img_range[1], ax0.shape[0])

    theo_beam = gbp.FocusedGaussianBeam(NA=NA)
    axint_theo = theo_beam.axial_intensity(z2)

    # normalize such that line integral of ax0 = line integral of axint_theo
    dz = z[1] - z[0]
    axint_theo_integral = np.sum(axint_theo * dz)
    ax0_integral = np.sum(ax0 * dz)
    ax1_integral = np.sum(ax1 * dz)
    print('######################')
    print(ax0_integral)
    print(ax1_integral)
    print('######################')
    if axint_theo_integral != 0:
        axint_theo /= axint_theo_integral
    if ax0_integral != 0:
        ax0 /= ax0_integral
    if ax1_integral != 0:
        ax1 /= ax1_integral

    axint_theo_integral = np.sum(axint_theo * dz)
    ax0_integral = np.sum(ax0 * dz)
    ax1_integral = np.sum(ax1 * dz)
    print('######################')
    print(ax0_integral)
    print(ax1_integral)
    print('######################')
    cross_corr0 = cross_correlate_1d(ax0, axint_theo)
    cross_corr1 = cross_correlate_1d(ax1, axint_theo)
    c0, f0, q0 = linfoot(ax0, axint_theo)
    c1, f1, q1 = linfoot(ax1, axint_theo)

    pl.figure()
    pl.plot(z2, axint_theo, 'b-', linewidth=0.8,
            label='Scalar diffraction theory')
    mc_label0 = 'MC1, C = %.2f, F = %.2f, Q = %.2f' % (c0, f0, q0)
    mc_label1 = 'MC1, C = %.2f, F = %.2f, Q = %.2f' % (c1, f1, q1)
    pl.plot(z, ax0, 'r-', linewidth=0.8, label=mc_label0)
    pl.plot(z, ax1, 'g-', linewidth=0.8, label=mc_label1)
    pl.plot(vars1['focal_tolerance'] * np.ones(shape=(z.shape)), z, 'gray',
            linewidth=0.5)
    pl.plot(-vars1['focal_tolerance'] * np.ones(shape=(z.shape)), z, 'gray',
            linewidth=0.5)
    # pl.errorbar(z*1000, ax0/np.amax(ax0), xerr=0, yerr=ax0_stdev)
    pl.xlabel(r'z ($\mu$m)')
    pl.xlim(axial_img_range[0], axial_img_range[1])
    pl.ylim(0, 1.25)
    pl.ylabel('Normalized intensity')
    pl.title('Axial photon distribution \n NA = %.1f, N = %d'
             % (NA, N_photons))
    pl.legend(loc='best')
    if save_fig is True:
        pl.savefig(filename+'_ax.png', dpi=300)
    pl.close()


    pl.figure()
    expected = theo_beam.transverse_intensity(r=x, z=0)
    expected = expected/np.amax(expected)
    pl.plot(x, expected, 'b-', linewidth=0.8, label='Scalar \
            diffraction theory')
    pl.plot(x, trans0/np.amax(trans0), 'r-', linewidth=0.8,
            label='MC')
    pl.xlabel(r'x ($\mu$m)')
    pl.ylabel('Normalized intensity')
    pl.title('Transverse photon distribution \n NA = %.1f, N = %d'
             % (NA, N_photons))
    pl.legend(loc='best')
    if save_fig is True:
        pl.savefig(filename+'_trans-2.png', dpi=300)
    pl.close()


    pl.figure()
    ub = (transe1[:-1] + transe1[1:])/2
    vb = (axe1[:-1] + axe1[1:])/2
    xv, zv = np.meshgrid(ub, vb)

    if np.amax(H1) != 0:
        H_normed = H1.T/np.amax(H1)
    else:
        H_normed = H1.T

    levels = H_normed[:, H_normed.shape[0]//2]/(np.e**2)
#    out_waist = np.int(H_normed <= levels)
#    print(out_waist)
    CS = pl.contourf(zv, xv, H_normed, cmap=pl.cm.viridis,
                     origin='lower')
    CS2 = pl.contour(CS, levels=levels, colors='#8C73A3', linewidths=0.1,
                     origin='lower')
    pl.colorbar(CS)
#    pl.clabel(CS2, inline=True, fontsize=5)
    geometrical_shadow = theo_beam.beam_radius(z)
    pl.plot(z, geometrical_shadow, 'w', linewidth=0.5)
    pl.plot(z, -geometrical_shadow, 'w', linewidth=0.5)
    pl.plot(vars1['focal_tolerance'] * np.ones(shape=(x.shape)), x, 'r',
            linewidth=0.5)
    pl.plot(-vars1['focal_tolerance'] * np.ones(shape=(x.shape)), x, 'r',
            linewidth=0.5)
    pl.ylim(trans_img_range[0], trans_img_range[1])
    pl.xlim(axial_img_range[0], axial_img_range[1])
    pl.xlabel(r'z ($\mu$m)')
    pl.ylabel(r'x ($\mu$m)')
    pl.title('Unscattered photon distribution \n NA = %.1f, N = %d'
             % (NA, N_photons))
    if save_fig is True:
        pl.savefig(filename+'_contour.png', dpi=300)
    pl.close()

    # pl.figure()
    # for t in np.arange(0, trials, 1):
    #     ticks = (edges[t][1:] + edges[t][:-1])/2
    #     pl.plot(ticks, path_lengths[t])
    # pl.title('Path length distribution')
    # if save_fig is True:
    #     pl.savefig(filename + '_paths.png', dpi=300)
    # pl.close()

    # pl.figure()
    # pl.matshow(H1)
    # pl.show()

    # print('done.\n')

    return c0, f0, q0, c1, f1, q1, cross_corr0, cross_corr1


def evaluate_linfootcriterion(NA):
    # for NA in np.arange(0.2,0.9,0.1):
    #     best_param, cross_corr = optimize_step_parameter(curvature_correction=1, num_photons=int(1E5), NA=NA)
    #     print(best_param)
    #     print(cross_corr)
    #     plot_profiles(NA=NA, step_param=best_param, save_fig=True)

    # NA = 0.2
    step_param_list = [3E-19, 2E-19, 1E-19, 9E-20, 8E-20, 7E-20, 6E-20, 5E-20, 4E-20, 3E-20, 2E-20, 1E-20]
    c = []
    f = []
    q = []
    c0 = 0
    f0 = 0
    q0 = 0
    cross_corr_wcorrect = []
    for step_param in step_param_list:
        c0, f0, q0, c1, f1, q1, cross_corr0, cross_corr1 = plot_profiles(NA=NA, step_param=step_param, save_fig=False)
        c.append(c1)
        f.append(f1)
        q.append(q1)
        cross_corr_wcorrect.append(cross_corr1)
    # c0, f0, q0 = cross_corr_nocorrect[np.int_(NA_list*10) == int(NA*10)]

    figname = 'NA_%.2f' % NA

    pl.figure()
    pl.plot(step_param_list, c, '.:', label='MC2')
    pl.plot(step_param_list, c0*np.ones(shape=(len(step_param_list),)),
            'gray', label='MC1')
    pl.plot(step_param_list, cross_corr_wcorrect, '.:', label='MC2, cross correlation')
    pl.xlabel('Step size parameter')
    pl.ylabel('C')
    pl.ylim(0,1)
    pl.legend(loc='best')
    pl.title('Relative structural content, NA = %.2f' % (NA))
    pl.savefig(figname+'_-c.png', dpi=300)

    pl.figure()
    pl.plot(step_param_list, f, '.:', label='MC2')
    pl.plot(step_param_list, f0*np.ones(shape=(len(step_param_list),)),
            'gray', label='MC1')
    pl.plot(step_param_list, cross_corr_wcorrect, '.:', label='MC2, cross correlation')
    pl.xlabel('Step size parameter')
    pl.ylabel('F')
    pl.ylim(0,1)
    pl.legend(loc='best')
    pl.title('Fidelity, NA = %.2f' % (NA))
    pl.savefig(figname+'_-f.png', dpi=300)

    pl.figure()
    pl.plot(step_param_list, q, '.:', label='MC2')
    pl.plot(step_param_list, q0*np.ones(shape=(len(step_param_list),)),
            'gray', label='MC1')
    pl.plot(step_param_list, cross_corr_wcorrect, '.:', label='MC2, cross correlation')
    pl.xlabel('Step size parameter')
    pl.ylabel('Q')
    pl.ylim(0,1)
    pl.legend(loc='best')
    pl.title('Correlation quality, NA = %.2f' % (NA))
    pl.savefig(figname+'_-q.png', dpi=300)


if __name__ == "__main__":
    for NA in np.arange(0.2, 0.9, 0.1):
        evaluate_linfootcriterion(NA)
