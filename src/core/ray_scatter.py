# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# import pyximport
# pyximport.install()

import numpy as np
import multiprocessing
import matplotlib.pyplot as pl
import time
# import cProfile

# own modules
import intersect
import scatterer
import gaussian_beam_propagation as gbp


def normalize(x, y, z):
    dR_unrotated = np.array([x, y, z])
    dRnorm_unrotated = np.sqrt(np.sum(dR_unrotated**2, axis=0))
    dR_unrotated = np.divide(dR_unrotated, dRnorm_unrotated)
    one = np.ones(mu.shape[1])
    dR_unrotated = np.vstack((dR_unrotated, one))

    return dR_unrotated


def henyey_greenstein(mu, g):
    '''
    The Henyey-Greenstein phase function approximates the Mie phase function
    for particle radii of less than ten times smaller than the wavelength
    (difference is always less than 0.2%). Reference: Toublanc, 1996.
    '''
    np.random.seed()

    E = np.random.random(size=mu.shape[1])
    costheta = np.zeros(shape=E.shape)
    zero_g = g == 0
    costheta[zero_g] = 2*E[zero_g] - 1
    j = g[~zero_g]
    costheta[~zero_g] = (1+j**2)/(2*j) \
        - ((1-j**2)**2) / ((2*j)*(1-j+2*j*E[~zero_g])**2)
    sintheta = np.sqrt(1-costheta**2)

    phi = 2 * np.pi * np.random.random(size=mu.shape[1])
    x = sintheta * np.cos(phi)
    y = sintheta * np.sin(phi)
    z = costheta

    # Renormalize them
    dR_unrotated = normalize(x, y, z)

    return dR_unrotated


def euler_rotate(mu):
    # Rotate them
    # Rotation matrices:
    a = mu[0]
    b = mu[1]
    c = mu[2]
    d = np.sqrt(b**2 + c**2)
    zero = np.zeros(len(a))

    # Find where d = 0 (it means rotation axis is along x axis)
    diszeroindices = np.where(d == 0)[0]
    d[diszeroindices] = 1

    # Vectorized matrix multiplication
    Rmat1 = np.vstack((d, zero, a, zero))
    Rmat2 = np.vstack((-a*b/d, c/d, b, zero))
    Rmat3 = np.vstack((-a*c/d, -b/d, c, zero))
    Rmat4 = np.vstack((zero, zero, zero, one))
    # DO YOU REALLY NEED Rmat4??
    if len(diszeroindices) != 0:
        Rmat2[0, diszeroindices] = 0.
        Rmat2[1, diszeroindices] = 1.
        Rmat2[2, diszeroindices] = 0.
        Rmat3[0, diszeroindices] = -a[diszeroindices]
        Rmat3[1, diszeroindices] = 1.
        Rmat3[2, diszeroindices] = d[diszeroindices]

    stack1 = Rmat1 * dR_unrotated
    stack1 = np.sum(stack1, axis=0)
    stack2 = Rmat2 * dR_unrotated
    stack2 = np.sum(stack2, axis=0)
    stack3 = Rmat3 * dR_unrotated
    stack3 = np.sum(stack3, axis=0)
    stack4 = Rmat4 * dR_unrotated
    stack4 = np.sum(stack4, axis=0)

    dR_rotated = np.vstack((stack1, stack2, stack3))

    del zero, one, stack1, stack2, stack3, stack4
    del Rmat1, Rmat2, Rmat3, Rmat4

    # Normalize
    dRnorm_rotated = np.sqrt(np.sum(dR_rotated**2,
                                    axis=0))
    dR_rotated = np.divide(dR_rotated, dRnorm_rotated)

    return dR_rotated


def sim2_unwrap(beam, scat_medium):
    np.random.seed()

    scat_centers = scat_medium.particle_centers
    anisotropies = scat_medium.particle_anisotropies
    radii = scat_medium.radii_eff # effective radii based on scattering cross section
    bounds = scat_medium.bounds

    some_photons = beam.num_photons
    num_steps = beam.num_steps

    # launch photons
    r, mu, at_focus_x, at_focus_y = beam.launch(some_photons)
    theta = np.zeros(shape=mu.shape)

    # make first proposal
    mu[:, :, 0], stepsize = \
        beam.update_in_free_space(r[:, :, 0], at_focus_x, at_focus_y)
    r[:, :, 1] = beam.move(r[:, :, 0], mu[:, :, 0], stepsize)

    # move default position of r from center of scatterer to some point far
    # away. this will be neglected later when considering scattering out of the
    # medium, and will be replaced with propagating in the direction upon
    # scattering out
    r[:, :, 2:] = np.inf * np.ones(shape=r[:, :, 2:].shape)

    scattered = np.zeros(shape=(some_photons,), dtype=bool)
    num_times_scattered = np.zeros(shape=(some_photons,), dtype=np.int)
    scattered_out = np.zeros(shape=(some_photons, num_steps), dtype=bool)
    scattered_history = np.zeros(shape=(some_photons, num_steps), dtype=bool)

    for step in range(1, num_steps-1, 1):
        # print(f'\n step {step}')
        # print(f'ray z: {r[2, :, step]}')
        was_just_in_medium = intersect.points_inside_box(r[:, :, step-1],
                                                              bounds)
        proposed_step_in_medium = intersect.points_inside_box(r[:, :, step],
                                                              bounds)
        hits_medium_within_step = was_just_in_medium | proposed_step_in_medium
        crosses_medium_within_step = intersect.rays_box(r[:, :, step-1:step+1],
                                                        bounds, segment=True)
        # print(f'hits_medium_within_step: {hits_medium_within_step}')
        # print(f'proposed_step_in_medium: {proposed_step_in_medium}')
        # print(f'crosses_medium_within_step: {crosses_medium_within_step}')

        valid_intersect = (hits_medium_within_step | crosses_medium_within_step)

        scattered_out[:, step] = scattered_out[:, step-1] | (~valid_intersect & scattered)

        # check for scattering event between last two steps
        # print(r[:, :, step-1:step+1].shape)
        intersects_index_pairs = intersect.rays_spheres(r[:, :, step-1:step+1],
                                                        scat_centers,
                                                        radii=radii)
        # intersects: boolean matrix showing which ray intersected which sphere
        # intersects_index_pairs: array of ray indices (1st col) and sphere
        #                         indices (2nd col)
        intersects_any = np.zeros(shape=r.shape[1], dtype=bool)  # checks if ray intersects any sphere
        if len(intersects_index_pairs) > 0:
            scat_i = intersects_index_pairs[:, 1]
            ray_i = intersects_index_pairs[:, 0]
            # print(f'scat_i: {scat_i}; ray_i: {ray_i}')
            intersects_any[ray_i] = 1
            intersects_any = intersects_any & (~scattered_out[:, step] & valid_intersect)
        # print(f'intersects_any: {intersects_any}')
        num_times_scattered += np.int_(intersects_any)
        scattered = num_times_scattered > 0

        state1 = (~scattered & ~intersects_any) & ~scattered_out[:, step]
        state2 = intersects_any & ~scattered_out[:, step]
        state3 = (scattered & ~intersects_any) & ~scattered_out[:, step]

        if np.sum(state1) > 0:
            # keep proposed position for this step.
            # check and update if outside medium
            out = ~(intersect.points_inside_box(r[:, state1, step], bounds)) & scattered[state1]
            scattered_out[state1, step] = scattered_out[state1, step] | out

            # for all rays that have not been scattered, generate new proposed
            # target to be tested for the next step by propagating as though
            # there is no scattering medium
            if beam.curvature_correction == 1:
                r_ref = r[:, state1, step]
            else:
                r_ref = r[:, state1, 0]

            mu[:, state1, step], stepsize = \
                beam.update_in_free_space(r_ref, at_focus_x[state1],
                                          at_focus_y[state1])
            r[:, state1, step+1] = beam.move(r[:, state1, step],
                                             mu[:, state1, step],
                                             stepsize)

        if np.sum(state2) > 0:
            # for rays that have entered the medium and collide with a
            # scatterer, update position to scatterer center
            # print(intersects_index_pairs)
            r[:, ray_i, step] = scat_centers[:, scat_i]

            # generate new target to be tested for the next step from the g of the scatterer
            stepsize = scat_medium.height

            x = mu[0, ray_i, step-1]
            y = mu[1, ray_i, step-1]
            z = mu[2, ray_i, step-1]
            norm = np.sqrt(x**2 + y**2 + z**2)
            wherenormzero = norm == 0
            theta0 = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
            phi0 = np.arctan2(y, x) * 180 / np.pi
            theta1 = scat_medium.gen_mie(size=ray_i.shape[0]) + theta0
            phi1 = 360 * np.random.random(size=theta0.shape[0]) - 180 + phi0
            theta1 = theta1 * np.pi / 180
            phi1 = phi1 * np.pi / 180
            x = np.sin(theta1) * np.cos(phi1)
            y = np.sin(theta1) * np.sin(phi1)
            z = np.cos(theta1)
            norm = np.sqrt(x**2 + y**2 + z**2)

            mu[0, ray_i, step] = x / norm
            mu[1, ray_i, step] = y / norm
            mu[2, ray_i, step] = z / norm
            r[:, ray_i, step+1] = beam.move(r[:, ray_i, step],
                                             mu[:, ray_i, step], stepsize)

        if np.sum(state3) > 0:
            # keep proposed position for this step.
            # check and update if outside medium
            out = ~(intersect.points_inside_box(r[:, state3, step], bounds))
            out = out & scattered[state3]
            scattered_out[state3, step] = scattered_out[state3, step] | out

            # generate new target to be tested for the next step
            # using previous position
            mu[:, state3, step] = mu[:, state3, step-1]
            stepsize = scat_medium.height
            r[:, state2, step+1] = beam.move(r[:, state2, step],
                                             mu[:, state2, step], stepsize)

        scattered_history[:, step] = intersects_any

    return r, scattered_out, scattered_history


def wrap(args):
    temp = sim2_unwrap(*args)
    # temp = cProfile.runctx('sim2_unwrap(*args)', globals(), locals())
    return temp


def sim2(num_photons, scattering_degree, num_steps=100):
    wavelength = 632.8E-3
    # define scattering medium
    # glist = [0.1]
    # g_tag = int(glist[0] * 10)
    g_percent = [1.0]
    bounds = np.array([[-5, 5], [-5, 5], [-0.5, 0.5]])
    surround_index = 1.33
    radius = 0.06969697
    bounds[:, 0] = bounds[:, 0] + radius
    bounds[:, 1] = bounds[:, 1] + radius
    size_param = 2 * np.pi * radius * surround_index / wavelength
    print(f'size param: {size_param}')
    structure = 'random'
    particle_index = 2.488  # polystyrene in water

    # initialize scattering medium
    med_time_start = time.process_time_ns()
    scat_medium = scatterer.RandomScatteringMedium(bounds=bounds,
                                                   particle_radii=radius,
                                                   size_param=size_param,
                                                   scattering_degree=scattering_degree,
                                                   particle_index=particle_index,
                                                   env_index=surround_index,
                                                   no_overlaps=False)
    med_time_end = time.process_time_ns()
    med_time = med_time_end - med_time_start

    if scattering_degree == 0:
        g_tag = 0
    else:
        g_tag = int(scat_medium.particle_anisotropies[0] * 10)

    num_trials = 5
    num_iterations = 10
    num_jobs = int(multiprocessing.cpu_count())
    # num_steps = 100
    some_photons = np.int((num_photons/num_iterations)//num_jobs)
    NA = 0.4

    beam = gbp.FocusedGaussianBeamMC(num_photons=some_photons, NA=NA, z_f=0,
                                     num_iterations=num_iterations,
                                     curvature_correction=True,
                                     axial_resolution=101, num_steps=num_steps,
                                     step_param=8.2e-18, beam_dist='gaussian',
                                     wavelength=wavelength, n=surround_index)
    print(beam.__dict__)

    runtime = np.zeros(shape=(num_trials + 1,))
    runtime[0] = med_time

    for i in range(num_trials):
        time_start = time.process_time_ns()
        print(f'trial {i}')
        r = np.zeros(shape=(num_iterations, 3, some_photons*num_jobs, num_steps))
        scattered_out = np.zeros(shape=(num_iterations,
                                    some_photons*num_jobs, num_steps))
        scattered_history = np.zeros(shape=scattered_out.shape)

        for j in range(num_iterations):
            with multiprocessing.Pool() as pool:
                coupled_sets = pool.map(wrap, num_jobs*[(beam, scat_medium)])
            print(f'Gathering data from processes for iteration {j}')
            r[j] = np.concatenate(np.array(list(zip(*coupled_sets))[0]),
                                     axis=1)[:num_photons]
            # print(f'r shape: {r.shape}')
            scattered_out[j] = np.concatenate(np.array(list(zip(*coupled_sets))[1]), axis=0)[:num_photons]
            scattered_history[j] = np.concatenate(np.array(list(zip(*coupled_sets))[2]), axis=0)[:num_photons]

        # print('Gathering data from processes...')
        r = np.concatenate(r, axis=1)
        print(f'r shape: {r.shape}')
        scattered_out = np.concatenate(scattered_out, axis=0)
        print(f'scattered out shape: {scattered_out.shape}')
        scattered_history = np.concatenate(scattered_history, axis=0)
        print(f'scattered hist shape: {scattered_history.shape}')
        np.save(f'rayN{num_photons}s{num_steps}R{int(scattering_degree)}g{g_tag}t{i}.npy', r)
        np.save(f'outN{num_photons}s{num_steps}R{int(scattering_degree)}g{g_tag}t{i}.npy', scattered_out)
        np.save(f'medN{num_photons}s{num_steps}R{int(scattering_degree)}g{g_tag}t{i}.npy', scat_medium.particle_centers)
        np.save(f'hisN{num_photons}s{num_steps}R{int(scattering_degree)}g{g_tag}t{i}.npy', scattered_history)
        np.save(f'medpropN{num_photons}s{num_steps}R{int(scattering_degree)}g{g_tag}t{i}.npy', np.array(scat_medium.__basic_props__))
        np.save(f'beampropsN{num_photons}s{num_steps}R{int(scattering_degree)}t{i}.npy', np.array(beam.__dict__))

        time_end = time.process_time_ns()
        runtime[i+1] = time_end - time_start
        print(f'runtime: {runtime[i+1]}')
        del r, scattered_out, scattered_history

    np.save(f'runtime-{num_photons}-{int(scattering_degree)}-{num_steps}.npy', runtime)


if __name__ == "__main__":
    # num_photons_list = np.logspace(3, 6, 7)
    num_photons = np.int(3E3)
    for scattering_degree in np.arange(0, 5, 2):
        sim2(int(num_photons), scattering_degree)

    # for num_photons in np.logspace(3, 5, 5):
    #     for scattering_degree in np.arange(4, 9, 2):
    #         sim2(int(num_photons), scattering_degree)

    # num_photons = np.int(1E4)
    # scattering_degree = 2
    # for num_steps in np.arange(160, 201, 10):
    #     sim2(num_photons, scattering_degree, num_steps)
