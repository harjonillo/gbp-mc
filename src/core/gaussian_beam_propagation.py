# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 17:16:05 2018

@author: hannah

Monte Carlo simulation of a focused Gaussian beam
"""

import numpy as np
import multiprocessing
import sys
import os
from scipy.special import expit, jv
from scipy.integrate import quad

from src.core import intersect


class FocusedGaussianBeam:
    def __init__(self, NA=None, aperture=None, z_f=0, wavelength=632.8E-3, n=1,
                 n_list=None, trunc_coeff=0, layer_bounds=None,
                 at_zero='focus'):
        self.wavelength0 = wavelength  # 633E-3 for He:Ne laser
        self.wavelength = self.wavelength0 / n
        self.n = n  # index of refraction
        if NA is not None:
            self.NA = NA
            self.aperture = 1500  # diameter
            theta = np.arcsin(self.NA/self.n)
            self.f = (self.aperture / 2) / np.tan(theta)
        elif aperture is not None:
            self.aperture = aperture
            self.f = 4000
            theta = np.arctan2(aperture, 2*self.f)
            self.NA = self.n * np.sin(theta)

        # for beams propagating in layers of different refractive indices
        # self.n_list = n_list
        # self.layer_bounds = layer_bounds

        self.k = 2*np.pi/self.wavelength

        if at_zero == 'lens':
            self.z_lens = 0
            self.z_f = self.z_lens + self.f
        else:
            self.z_f = z_f  # location of focus
            self.z_lens = self.z_f - self.f

        # self.w0 = (2/np.pi) * (self.wavelength*self.f / self.aperture)
        if trunc_coeff >= 4:
            self.w_incident = (self.aperture / 2) / np.sqrt(trunc_coeff)
            Nw = self.w_incident**2 / (self.wavelength * self.f)
            self.w0 = self.w_incident / np.sqrt(1 + (np.pi * Nw)**2)
        else:
            self.w_incident = self.aperture / np.sqrt(trunc_coeff)
            self.w0 = (2/np.pi) * (self.wavelength*self.f / self.aperture)
            # NOTE: change this. needs to be calculated from intensity.
            # find plane of waist first then get r where 1/e^2 intensity

        self.pinhole_radius = self.w0/2
        self.lensradius = self.f * np.tan(np.arcsin(self.NA/self.n))
        self.trunc_coeff = trunc_coeff # see horvath (2003) for details
        # truncation coefficient > 4 means not truncated; 0 for spherical beam

        self.c = 2.998e8 * 1e6  # speed of light, in microns
        self.z_R = np.pi * self.w0**2 / self.wavelength
        self.focal_tolerance = 2 * self.z_R

    def curvature_radius(self, z):
        '''
        Radius of curvature of a Gaussian beam.
        Parameters:
        -----------
            z: float or array-like object
        Returns:
        --------
            R: float or array-like object
        '''
        R = -(z-self.z_f) * (1+(self.z_R / (z - self.z_f))**2)
        return R

    def T(self, z):
        return 1 / self.curvature_radius(z)

    def beam_radius(self, z):
        # w0 = (2/np.pi) * (self.wavelength*self.f / self.aperture)
        return self.w0 * np.sqrt(1+((z-self.z_f)/self.z_R)**2)

    def transverse_intensity(self, r, z):
        return np.exp(-2*r**2 / self.beam_radius(z)**2)

    def epsilon(self, z):
        k = 2*np.pi/self.wavelength
        ws = self.beam_radius(z=0)
        return 2*(z-self.z_f)/(k*ws**2)

    def kappa(self, z):
        return np.sqrt(2) / (self.w0 * np.sqrt(1 + self.epsilon(z)**2))

    def sigmasqr(self, z):
        return 1 + 1j*self.epsilon(z)

    def efield(self, r, z):
        wz = self.beam_radius(z)
        Rz = self.curvature_radius(z)
        amp = self.w0 / wz
        gouy = np.arctan2(z, self.z_R)
        expterm = np.exp((-r**2 / wz**2)
                         - 1j*(self.k*((r**2/(2*Rz)) + (z-self.z_f)) - gouy))

        return np.abs(amp * expterm)**2

    def axial_intensity(self, z):
        '''
        Axial intensity of a focused Gaussian beam through a circular aperture.
        IMPORTANT NOTE: requires lens located at z = 0.
        Reference: Tanaka et al., 1985.
        '''
        k = 2*np.pi/self.wavelength
        ws = self.beam_radius(z=0)  # untruncated
        alpha = (self.aperture / 2) / ws
        P = k * ws**2 / self.f
        Z = (z-self.z_f)/self.f

        eps = self.epsilon(z=0)

        s1 = (P * alpha**2 * (1-Z) / (2*Z)) + (alpha**2 * eps / (1 + eps**2))
        s2 = -alpha**2 / (1 + eps**2)

        f1 = P**2 * alpha**4 / (4 * Z**2 * (s1**2 + s2**2))
        f2 = 1 + expit(2*s2) - 2*expit(s2)*np.cos(s1)
        intensity = f1*f2

        z_is_zero = z == 0
        intensity[z_is_zero] = 1
        return intensity

    def diffraction_field(self, r, z):
        '''
        Diffraction field of a focused Gaussian beam through a circular
        aperture. IMPORTANT NOTE: requires lens located at z = 0.
        '''
        kappa0 = self.kappa(z=0)
        epsilon0 = self.epsilon(z=0)
        sigmasqr0 = self.sigmasqr(z=0)
        t1 = 2j * np.sqrt(np.pi) * kappa0 / (self.wavelength * z)
        t2 = np.exp(-1j * (self.k * (z-self.z_f) + np.arctan(epsilon0)
                           + self.k * r**2 / (2*z)))

        innerterm = kappa0**2 * sigmasqr0 + 1j * self.k * ((1/z) - (1/self.f))
        f1 = lambda r0: r0 * jv(0, self.k * r * r0 / z) \
            * np.exp(-(1/2) * innerterm * r0**2)
        integral, err = quad(f1, 0, self.aperture)

        return t1 * t2 * integral


    def intensity_3d(self, r, z):
        '''
        Intensity distribution of a focused Gaussian beam through a circular
        aperture. Reference: Tanaka et al., 1985.
        '''
        k = 2*np.pi/self.wavelength
        ws = self.beam_radius(z=0)
        alpha = (self.aperture / 2) / ws
        P = k * ws**2 / self.f
        Z = z/self.f
        R = r / ws

        eps = self.epsilon(z=0)

        s1 = (P * alpha**2 * (1-Z) / (2*Z)) + (alpha**2 * eps / (1 + eps**2))
        s2 = -alpha**2 / (1 + eps**2)

        factor = P * alpha ** 2 / Z ** 2

        f1 = lambda r0: r0 * jv(0, P * alpha * R / Z * r0) * \
            expit(-alpha**2 * r0**2 / (1 + eps**2)) * np.cos(s1 * r0**2)
        f2 = lambda r0: r0 * jv(0, P * alpha * R / Z * r0) * \
            expit(-alpha**2 * r0**2 / (1 + eps**2)) * np.sin(s1 * r0**2)
        firstterm, err = quad(f1, 0, 1)
        secondterm, err = quad(f2, 0, 1)

        intensity = factor * (firstterm**2 + secondterm**2)

        return intensity

    def change_coords(self, r, z):
        aof = (self.aperture / 2)/self.f
        u = self.k * (aof)**2 * z
        v = self.k * (aof) * r

        return v, u

    def change_coords_reverse(self, v, u):
        aof = (self.aperture / 2)/self.f
        z = u / (self.k * (aof)**2)
        r = v / (self.k * (aof))

        return r, z

    def bessel_m(self, x, m, n):
        '''
        Returns the mth term of the bessel function of the first kind, order n
        '''
        j = ((x/2)**((2*m+n))) * (-1)**m / (np.math.factorial(m)
                                            * np.math.factorial(m+n))
        return j

    def bessel(self, x, n, M):
        '''
        Returns the bessel function of the first kind of order n, approximate
        up to M
        '''
        J_n = 0
        for m in range(M):
            J_n = J_n + self.bessel_m(x, m, n)
        return J_n

    def lommel_U(self, n, p, v):
        # s = np.arange(0, 100, 1, dtype=np.int)
        U_n = 0
        S = 30
        for s in range(S):
            U_n += (-1)**s * (p/v)**(n+2*s) * self.bessel(v, n + 2*s, 30)

        return U_n

    def lommel_V(self, n, v, p):
        V_n = 0
        S = 30
        for s in range(S):
            V_n += (-1)**s * (v/p)**(n+2*s) * self.bessel(v, n + 2*s, 30)

        return V_n

    def lommel_K(self, p, v):
        a = -1j * p / 2
        K = (np.exp(a) / a) * (self.lommel_U(2, p, v)
                               - 1j*self.lommel_U(1, p, v))

        return K

    def debye_approx_field(self, v, u):
        '''
        Debye approximation to the diffraction field of a focused Gaussian beam
        through a circular aperture that truncates the incident beam.
        Reference: Horvath & Bor, 2003
        '''
        aof = (self.aperture / 2)/self.f
        # V, U = self.change_coords(r, z)
        U, V = np.meshgrid(u, v)

        P = U - 2j * self.trunc_coeff
        factor = (-1j * np.pi * self.n / self.wavelength) * (aof)**2
        field = factor * np.exp(1j * (1/aof)**2 * U) * self.lommel_K(P, V)
        return V, U, field


class FocusedGaussianBeamMC(FocusedGaussianBeam):
    def __init__(self, num_photons, NA, z_f, num_iterations,
                 curvature_correction, axial_resolution,
                 beam_dist, num_steps, wavelength=632.8E-3, step_param=None,
                 trunc_coeff=4, n=1):
        FocusedGaussianBeam.__init__(self, NA, z_f, n=n, trunc_coeff=trunc_coeff)

        self.curvature_correction = curvature_correction
        self.n = n
        if curvature_correction is True:
            if step_param is None:
                if NA == 0.2 or 0.5:
                    self.step_param = 120E-17
                elif NA == 0.8 or 0.9:
                    self.step_param = 64E-17
                elif NA == 0.4:
                    self.step_param = 4E-20
                else:
                    self.step_param = 120E-17
            else:
                self.step_param = step_param
        else:
            self.step_param = 0
        # Parameters, in microns
        self.num_photons = int(num_photons)  # number of photons
        self.num_iter = int(num_iterations)
        self.num_steps = num_steps
        self.axial_resolution = int(axial_resolution)
        img_limit = self.focal_tolerance * 30
        self.axial_img_range = (z_f-img_limit, z_f+img_limit)
        self.trans_img_range = (-img_limit, img_limit)

        filename = 'gbpc%dN%.0ENA0%d' % \
            (int(self.curvature_correction), self.num_photons, self.NA*10)
        self.filename = filename.replace("+", "")

        self.beam_dist = beam_dist

    def init_collimated(self, some_photons):
        # photon coordinates in collimated space, when it reaches the focusing
        # lens
        stdev_at_lens = self.beam_radius(z=self.z_lens)/2
        # D4\sigma beam width. w = 2*radius = 4*stdev

        r_coll = np.zeros(shape=(3, some_photons))
        if self.beam_dist == 'gaussian':
            r_coll[0] = np.random.normal(scale=stdev_at_lens, size=some_photons)
            r_coll[1] = np.random.normal(scale=stdev_at_lens, size=some_photons)
        elif self.beam_dist == 'uniform':
            r_coll[0] = stdev_at_lens * np.random.random(size=some_photons)
            r_coll[1] = stdev_at_lens * np.random.random(size=some_photons)
        r_coll[2] = self.z_lens*np.ones(shape=(some_photons, ))
        rLens2 = r_coll[0]**2 + r_coll[1]**2
        blocked_photons = (rLens2 > self.aperture**2)
        blocked_count = np.sum(blocked_photons)

        if self.beam_dist == 'gaussian':
            xregen = np.random.normal(scale=stdev_at_lens, size=blocked_count)
            yregen = np.random.normal(scale=stdev_at_lens, size=blocked_count)
        elif self.beam_dist == 'uniform':
            xregen = stdev_at_lens * np.random.random(size=blocked_count)
            yregen = stdev_at_lens * np.random.random(size=blocked_count)
        for i in range(blocked_count):
            while (xregen[i]**2 + yregen[i]**2) > self.aperture**2:
                if self.beam_dist == 'gaussian':
                    xregen[i] = np.random.normal(scale=stdev_at_lens)
                    yregen[i] = np.random.normal(scale=stdev_at_lens)
                elif self.beam_dist == 'uniform':
                    xregen[i] = stdev_at_lens * np.random.random()
                    yregen[i] = stdev_at_lens * np.random.random()
        r_coll[0, blocked_photons] = xregen
        r_coll[1, blocked_photons] = yregen

        del xregen, yregen, rLens2, blocked_photons

        return r_coll

    def init_focus(self, r_coll):
        '''
        Focuses rays of a collimated beam to some focus f located at z = z_f.
        '''
        r_focused = np.zeros(r_coll.shape)
        const = (r_coll[2] - self.z_f) \
            / np.sqrt(r_coll[0]**2 + r_coll[1]**2 + self.f**2)
        r_focused[0] = -r_coll[0]*const
        r_focused[1] = -r_coll[1]*const
        r_focused[2] = self.z_f + self.f*const

        return r_focused

    def ddR(self, r):
        x = r[0]
        y = r[1]
        z = r[2]
        inf_curvature = (z - self.z_f) == 0

        Tz = np.zeros(shape=z.shape)
        Tz[~inf_curvature] = self.T(z[~inf_curvature])

        const = (self.c/self.n) * self.z_R**2 \
            / ((1 + (Tz**2 * (x**2 + y**2)))**2 * ((z - self.z_f)**2
                                                   + self.z_R**2)**2)
        ddr = np.zeros(shape=r.shape)
        ddr[0] = const * x
        ddr[1] = const * y
        ddr[2] = const * Tz * (x**2 + y**2)

        return ddr

    def launch(self, some_photons):
        '''
        Parameters:
        -----------
        num_photons: int
        distribution: string
            either airy or gaussian

        Returns:
        --------
        r, mu: numpy arrays
            r and mu both have shape (num_photons, 3, num_steps)
        '''
        # print('Initializing photons...')
        # array of positions
        r = np.zeros(shape=(3, some_photons, self.num_steps), dtype=float)
        # array of direction cosines
        mu = np.zeros(shape=(3, some_photons, self.num_steps-1),
                      dtype=float)

        stdev_at_focus = self.w0/2  # w0 is radius at z=0

        r[:, :, 0] = self.init_collimated(some_photons)
        if self.curvature_correction is True:
            r[:, :, 0] = self.init_focus(r[:, :, 0])

        if self.beam_dist == 'gaussian':
            at_focus_x = np.random.normal(scale=stdev_at_focus,
                                          size=some_photons)
            at_focus_y = np.random.normal(scale=stdev_at_focus,
                                          size=some_photons)
        elif self.beam_dist == 'uniform':
            at_focus_x = stdev_at_focus * np.random.random(size=some_photons)
            at_focus_y = stdev_at_focus * np.random.random(size=some_photons)

        mu[0, :, 0] = at_focus_x - r[0, :, 0]
        mu[1, :, 0] = at_focus_y - r[1, :, 0]
        mu[2, :, 0] = self.z_f - r[2, :, 0]

        norm_factor = np.sqrt(mu[0, :, 0]**2 + mu[1, :, 0]**2 + mu[2, :, 0]**2)
        mu[0, :, 0] = mu[0, :, 0]/norm_factor
        mu[1, :, 0] = mu[1, :, 0]/norm_factor
        mu[2, :, 0] = mu[2, :, 0]/norm_factor

        return r, mu, at_focus_x, at_focus_y

    def move(self, r, mu, stepsize):
        norm_factor = np.sqrt(mu[0]**2 + mu[1]**2 + mu[2]**2)
        norm_factor_nonzero = norm_factor != 0
        mu[:, norm_factor_nonzero] = mu[:, norm_factor_nonzero] / \
            norm_factor[norm_factor_nonzero]

        # update new positions
        rprime = r + (mu * stepsize)

        return rprime

    def update_in_free_space(self, r, at_focus_x, at_focus_y):
        if self.curvature_correction is True:
            ddr = self.ddR(r)
            s = np.amax(np.abs(ddr), axis=0) * self.c * self.step_param \
                / self.n
        else:
            s = np.sqrt((at_focus_x - r[0])**2 +
                        (at_focus_y - r[1])**2 +
                        (self.z_f - r[2])**2) / (self.num_steps/2)

        mu = np.zeros(shape=r.shape)
        # determine direction cosines
        mu[0] = at_focus_x - r[0]
        mu[1] = at_focus_y - r[1]
        mu[2] = self.z_f - r[2]
        if self.curvature_correction is False:
            after_focus = r[2] > 0
            mu[0:2, after_focus] = -mu[0:2, after_focus]

        return mu, s

    def propagate(self, r, mu, at_focus_x, at_focus_y):
        # print("propagating...")
        for step in range(0, self.num_steps-1):
            mu[:, :, step], s = self.update_in_free_space(r[:, :, step],
                                                          mu[:, :, step],
                                                          at_focus_x,
                                                          at_focus_y)
            r[:, :, step+1] = self.move(r[:, :, step], mu[:, :, step],
                                        at_focus_x, at_focus_y, s)
        return r

    def calculate_paths(self, R):
        dx = R[0, :, 1:] - R[0, :, :-1]
        dy = R[1, :, 1:] - R[1, :, :-1]
        dz = R[2, :, 1:] - R[2, :, :-1]
        dr2 = dx**2 + dy**2 + dz**2
        r = np.sqrt(np.cumsum(dr2, axis=0)[:, -1])

        return r

    def unwrap_hot_loop(self, some_photons):
        np.random.seed()

        r, mu, at_focus_x, at_focus_y = self.launch(some_photons)
        R = self.propagate(r, mu, at_focus_x, at_focus_y)
        total_path = self.calculate_paths(R)

        return R[0], R[1], R[2], total_path

    def rays_to_intensity(self, rays, trans_img_range=None,
                          axial_img_range=None, axial_resolution=None):
        if trans_img_range is None:
            trans_img_range = self.trans_img_range
        if axial_img_range is None:
            axial_img_range = self.axial_img_range
        if axial_resolution is None:
            axial_resolution = self.axial_resolution

        xb = np.linspace(trans_img_range[0], trans_img_range[1],
                         axial_resolution+1)
        zb = np.linspace(axial_img_range[0], axial_img_range[1],
                         axial_resolution+1)
        if self.curvature_correction is True:
            to_plot = [np.ravel(rays[0, :, :]), np.ravel(rays[2, :, :])]
        else:
            # print('Intersections...')
            some_event = int(self.num_steps/2)
            rendered_1 = intersect.render(resolution=axial_resolution//2,
                                          z_bounds=[axial_img_range[0], 0],
                                          r1=rays[:, :, 0],
                                          r2=rays[:, :, some_event-5])
            rendered_2 = intersect.render(resolution=axial_resolution//2,
                                          z_bounds=[0, axial_img_range[1]],
                                          r1=rays[:, :, some_event+5],
                                          r2=rays[:, :, -1])
            rendered = np.hstack((rendered_1, rendered_2))
            to_plot = [rendered[0], rendered[2]]

        H, trans_edges, ax_edges = np.histogram2d(to_plot[0], to_plot[1],
                                                  bins=[xb, zb])

        return H, trans_edges, ax_edges

    def save_data(self, trial_num, **kwargs):
        if kwargs is not None:
            # save variables for this object too
            vars = np.array([self.__dict__])
            print(vars)

            filename = self.filename + 't%d.npz' % (self.step_param, trial_num)
            filename = filename.replace('+', "")
            my_file = f"data/random_scattering/{filename}"
            print('saving '+str(my_file))
            with open(my_file, "wb") as input_file:
                np.savez(input_file, vars=vars, **kwargs)
            print('contour saved.')
        else:
            print('No arguments to save.')

        sys.stdout.flush()

    def sim(self, pool):
        '''
        Parameters:
        -----------
        num_scat: int
            number of scattering events or steps
        '''

        num_photons_per_iter = self.num_photons//self.num_iter
        x = np.zeros(shape=(self.num_iter, num_photons_per_iter,
                            self.num_steps))
        y = np.zeros(shape=x.shape)
        z = np.zeros(shape=x.shape)
        paths = np.zeros(shape=(self.num_iter, num_photons_per_iter))
        N_jobs = int(multiprocessing.cpu_count())
        nppi = num_photons_per_iter
        for i in range(self.num_iter):
            coupled_sets = pool.map(self.unwrap_hot_loop,
                                    N_jobs*[num_photons_per_iter//N_jobs + 1])
            # print('Gathering data from processes...')
            x[i] = np.concatenate(np.array(list(zip(*coupled_sets))[0]))[:nppi]
            y[i] = np.concatenate(np.array(list(zip(*coupled_sets))[1]))[:nppi]
            z[i] = np.concatenate(np.array(list(zip(*coupled_sets))[2]))[:nppi]
            paths[i] = np.concatenate(np.array(list(zip(*coupled_sets))[3]))[:nppi]
        # print('Gathering data from iterations...')
        X = np.concatenate(x, axis=0)
        Y = np.concatenate(y, axis=0)
        Z = np.concatenate(z, axis=0)
        L = np.concatenate(paths, axis=0)

        return np.stack((X, Y, Z)), L


def cross_correlate_1d(signal_a, signal_b):
    a = (signal_a - np.mean(signal_a))/(np.std(signal_a)*len(signal_a))
    b = (signal_b - np.mean(signal_b))/(np.std(signal_b))
    cross_correlation = np.correlate(a, b, 'same')

    return cross_correlation[int(len(cross_correlation)//2)]


def calibration(curve_correct):
    NA_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    trials = 10
    for NA in NA_list:
        for trial in range(trials):
            newbeam = FocusedGaussianBeamMC(num_photons=int(1E2), NA=NA, z_f=0,
                                            curvature_correction=curve_correct,
                                            axial_resolution=101,
                                            trial=trial, beam_dist='gaussian')
            newbeam.sim()


def calibrate_step_size_parameter(NA, pool):
    num_trials = 10
    num_iterations = 10
    num_photons = int(1E5)
    step_param_list = [3E-19, 2E-19, 1E-19, 9E-20, 8E-20, 7E-20, 6E-20, 5E-20,
                       4E-20, 3E-20, 2E-20, 1E-20]
    correlation_array = np.zeros(shape=(len(step_param_list), int(num_trials)))

    param_count = 0
    for param in step_param_list:
        newbeam = FocusedGaussianBeamMC(num_photons=num_photons, NA=NA, z_f=0,
                                        num_iterations=num_iterations,
                                        curvature_correction=True,
                                        axial_resolution=101,
                                        step_param=param, beam_dist='gaussian')
        rays = np.zeros(shape=(num_iterations, 3, num_photons, newbeam.num_steps))
        paths = np.zeros(shape=(num_iterations, num_photons))
        for trial in range(num_trials):
            rays, paths = newbeam.sim(pool)
            print('Ray matrix size:')
            print(rays.shape)
            H, trans_edges, ax_edges = newbeam.rays_to_intensity(rays)
            ax_int = H[H.shape[0]//2, :]
            newbeam.save_data(trial, intensity_distribution=H,
                              transverse_edges=trans_edges,
                              axial_edges=ax_edges, paths=paths)

            print('Calculating cross correlation...')
            z2 = np.linspace(newbeam.axial_img_range[0],
                             newbeam.axial_img_range[1], ax_int.shape[0])
            theo_beam = FocusedGaussianBeam(NA=NA, z_f=0)
            axint_theo = theo_beam.axial_intensity(z2)
            axint_theo /= np.amax(axint_theo)

            ax_int[np.isinf(ax_int)] = 1
            ax_int[np.isnan(ax_int)] = 0
            cross_corr = cross_correlate_1d(ax_int, axint_theo)
            correlation_array[param_count][trial] = cross_corr
            print('Cross correlation added to matrix.')
        param_count += 1

    correlation_data_filename = newbeam.filename + '_correlation-info.npz'
    np.savez(correlation_data_filename, np.array(step_param_list),
             correlation_array)
    print('saved correlation data.')


def all_no_curvature_correction(num_photons, pool):
    num_iterations = 1
    num_trials = 10
    for NA in np.arange(0.2, 1.0, 0.1):
        newbeam = FocusedGaussianBeamMC(num_photons=num_photons, NA=NA, z_f=0,
                                        num_iterations=num_iterations,
                                        curvature_correction=False,
                                        axial_resolution=101,
                                        step_param=None, beam_dist='gaussian')
        rays = np.zeros(shape=(num_iterations, 3, num_photons,
                               newbeam.num_steps))
        paths = np.zeros(shape=(num_iterations, num_photons))
        for trial in range(num_trials):
            rays, paths = newbeam.sim(pool)
            print('Ray matrix size:')
            # print(rays.shape)
            H, trans_edges, ax_edges = newbeam.rays_to_intensity(rays)
            newbeam.save_data(trial, intensity_distribution=H,
                              transverse_edges=trans_edges,
                              axial_edges=ax_edges, paths=paths)


if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
        all_no_curvature_correction(int(1E5), pool)
