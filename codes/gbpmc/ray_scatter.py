# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyximport
pyximport.install()

import numpy as np
import multiprocessing
import matplotlib.pyplot as pl
import time
import cProfile

# own modules
import intersectFast as intersect
from . import scatterer
from . import gaussian_beam_propagation as gbp


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
        was_just_in_medium = intersect.points_inside_box(r[:, :, step-1],
                                                              bounds)
        proposed_step_in_medium = intersect.points_inside_box(r[:, :, step],
                                                              bounds)
        hits_medium_within_step = was_just_in_medium | proposed_step_in_medium
        crosses_medium_within_step = intersect.rays_box(r[:, :, step-1:step+1],
                                                        bounds, segment=True)

        valid_intersect = (hits_medium_within_step | crosses_medium_within_step)

        scattered_out[:, step] = scattered_out[:, step-1] | (~valid_intersect & scattered)

        # check for scattering event between last two steps
        intersects_index_pairs = intersect.rays_spheres(r[:, :, step-1:step+1],
                                                        scat_centers,
                                                        radii=radii)
        # intersects: boolean matrix showing which ray intersected which sphere
        # intersects_index_pairs: array of ray indices (1st col) and sphere
        #                         indices (2nd col)
        intersects_any = np.zeros(shape=r.shape[1], dtype=bool)
        # checks if ray intersects any sphere

        if len(intersects_index_pairs) > 0:
            scat_i = intersects_index_pairs[:, 1]
            ray_i = intersects_index_pairs[:, 0]

            intersects_any[ray_i] = 1
            intersects_any = intersects_any & (~scattered_out[:, step] & valid_intersect)

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
            r[:, ray_i, step] = scat_centers[:, scat_i]

            # generate new target to be tested for the next step from the g of
            # the scatterer
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
            # keep proposed position for this step. check and update if outside
            # medium
            out = ~(intersect.points_inside_box(r[:, state3, step], bounds))
            out = out & scattered[state3]
            scattered_out[state3, step] = scattered_out[state3, step] | out

            # generate new target to be tested for the next step using previous
            # position
            mu[:, state3, step] = mu[:, state3, step-1]
            stepsize = scat_medium.height
            r[:, state2, step+1] = beam.move(r[:, state2, step],
                                             mu[:, state2, step], stepsize)

        scattered_history[:, step] = intersects_any

    return r, scattered_out, scattered_history


def wrap(args):
    temp = sim2_unwrap(*args)

    return temp


if __name__ == "__main__":
    # num_photons_list = np.logspace(3, 6, 7)
    num_photons = np.int(3E3)
    for scattering_degree in np.arange(0, 5, 2):
        sim2(int(num_photons), scattering_degree)
