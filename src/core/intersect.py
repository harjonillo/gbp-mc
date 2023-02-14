#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 16:44:20 2018

@author: hannaharjonillo

"""

import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from numba import jit


@jit
def ray_xyplane(plane, r1, r2):
    '''
    Finds the intersection I of a ray on plane P normal to the z-axis
    given ray starting point r1 and endpoint r2
    P: float
    r1: numpy array (3, N)
    r2: numpy array (3, N)

    returns I: numpy array (3, N)
    '''

    intersection = np.zeros(shape=r1.shape)

    u = (r1[2] - plane)/(r1[2] - r2[2])
    intersection[0] = r1[0] + u * (r2[0] - r1[0])
    intersection[1] = r1[1] + u * (r2[1] - r1[1])
    intersection[2] = plane * np.ones(shape=intersection[2].shape)

    return intersection


def render(resolution, z_bounds, r1, r2):
    '''
    Gives the intersection of several rays on planes between the bounds to
    satisfy required resolution
    '''
    xyplanes = np.linspace(z_bounds[0], z_bounds[1], resolution+1)
    rendered_points = np.zeros(shape=(resolution, 3, r1.shape[1]))
    # print(xyplanes.shape)
    for p in range(xyplanes.shape[0]-1):
        rendered_points[p] = ray_xyplane(xyplanes[p], r1, r2)

    return np.hstack(rendered_points)


def rayBoxIntersection(origin, direction, vmin, vmax):
    '''
    Ray-box intersection using the smits algorithm.
    Parameters:
        origin: array or tuple, shape(3,)
        direction: array or tuple, shape(3,)
        vmin
        vmax
    Returns:
        flag: boolean
            1 means intersection occurs
        tmin: float
            distance from the ray origin
    '''
    flag = 1

    if direction[0]>0:
        tmin = (vmin[0] - origin[0]) / direction[0]
        tmax = (vmax[0] - origin[0]) / direction[0]
    else:
        tmin = (vmax[0] - origin[0]) / direction[0]
        tmax = (vmin[0] - origin[0]) / direction[0]

    if direction[1] >= 0:
        tymin = (vmin[1] - origin[1]) / direction[1]
        tymax = (vmax[1] - origin[1]) / direction[1]
    else:
        tymin = (vmax[1] - origin[1]) / direction[1]
        tymax = (vmin[1] - origin[1]) / direction[1]

    if (tmin > tymax) | (tymin > tmax):
        flag = 0
        tmin = -1
        return flag, tmin

    if (tymin > tmin):
        tmin = tymin

    if (tymax < tmax):
        tmax = tymax

    if direction[2] >= 0:
        tzmin = (vmin[2] - origin[2]) / direction[2]
        tzmax = (vmax[2] - origin[2]) / direction[2]
    else:
        tzmin = (vmax[2] - origin[2]) / direction[2]
        tzmax = (vmin[2] - origin[2]) / direction[2]

    if (tmin > tzmax) | (tzmin > tmax):
        flag = 0
        tmin = -1
        return flag, tmin

    if tzmin > tmin:
        tmin = tzmin

    if tzmax < tmax:
        tmax = tzmax

    print(flag, tmin)

    return flag, tmin


@jit
def rays_spheres(rays, spheres, radii, segment=False, debug_mode=False):
    '''
    Parameters:
        rays: numpy array
            shape (3 for coordinate axes, number of rays, 2 for endpoints)
        spheres: numpy array
            shape (3 for coordinate axes, number of spheres)

    Returns:
        first_intersects: numpy array, boolean,
            shape (number of rays, number of spheres)
    '''
    r0 = rays[:, :, 0]
    r1 = rays[:, :, 1]

    EPS = np.finfo(np.float64).eps

    intersects = np.zeros(shape=(rays.shape[1], spheres.shape[1]),
                          dtype=np.bool)
    intersect_index_pairs = []

    for r in range(rays.shape[1]):
        # print(f'\n ray {r}')
        x0, y0, z0 = rays[:, r, 0]
        x1, y1, z1 = rays[:, r, 1]
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0

        # normalize ray direction
        a = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if a != 0:
            dx, dy, dz = dx/a, dy/a, dz/a

        min_distance_squared = np.inf
        min_distance_indices = None
        for s in range(spheres.shape[1]):
            line_intersect = False
            xs, ys, zs = spheres[:, s]
            tol = radii[s]//100

            l2oc = (xs - x0)**2 + (ys - y0)**2 + (zs - z0)**2
            if l2oc < radii[s]**2:  # if ray starts from inside sphere
                continue

            tca = (dx * (xs - x0)) + (dy * (ys - y0)) + (dz * (zs - z0))
            if tca < 0:  # if ray points away from sphere
                continue

            d2 = l2oc - tca**2
            if np.abs(d2) > radii[s]**2:
                continue

            t2hc = radii[s]**2 - d2
            # print(f'\t t2hc: {t2hc}')
            if t2hc < 0:
                continue

            line_intersect = True

            if l2oc < min_distance_squared:
                min_distance_indices = [r, s]
                min_distance_squared = l2oc

            intersects[r, s] = line_intersect

        if min_distance_indices != None:
            intersect_index_pairs.append(min_distance_indices)

    return np.array(intersect_index_pairs)

@jit
def sphere_sphere(sphere0, all_spheres, radius0, all_radii):
    x0, y0, z0 = sphere0
    d2 = (all_spheres[0] - x0)**2 + (all_spheres[1] - y0)**2 + \
        (all_spheres[2] - z0)**2
    intersects = d2 < ((radius0 + all_radii)**2)

    return intersects

@jit
def get_indices(intersects, exclude_self=True):
    # remove double counting and self intersection by zeroing out elements
    # below diagonal
    intersects = np.tril(intersects, -1)
    intersects_indices = np.where(intersects == 1)
    intersects_indices = np.vstack((intersects_indices[0],
                                    intersects_indices[1])).T

    return intersects_indices

@jit
def among_spheres(spheres, radii, self_intersect=False):
    '''
    Checks for if any two (or more) spheres overlap in a system of multiple
    stationary spheres. Returns indices of intersecting spheres (excluding
    self intersection)
    '''
    intersects = np.zeros(shape=(spheres.shape[1], spheres.shape[1]),
                          dtype=bool)
    sphere_indices = np.arange(spheres.shape[1], dtype=int)

    for s in sphere_indices:
        intersects[s] = sphere_sphere(spheres[:, s], spheres, radii[s], radii)

    return intersects

@jit
def ray_box(ray, box_bounds, segment=False):
    bmin, bmax = box_bounds[:, 0], box_bounds[:, 1]

    x0, y0, z0 = ray[:, 0]
    x1, y1, z1 = ray[:, 1]
    dx, dy, dz = ray[:, 1] - ray[:, 0]

    parallel_to_any_plane = (dx == 0) | (dy == 0) | (dz == 0)

    for axis in range(3):
        r0, r1 = ray[axis, 0], ray[axis, 1]
        dr = r1 - r0
        # print(f'dr: {dr}')
        if dr == 0 or (dr == np.nan):
            return False
        tnear = -np.inf
        tfar = np.inf
        if parallel_to_any_plane == 1:
            not_from_inside_box = (r0 < bmin[axis]) | (r1 > bmax[axis])
            if not_from_inside_box == 1:
                return False
        else:
            t1 = (bmin[axis] - r0) / dr
            t2 = (bmax[axis] - r1) / dr
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > tnear:
                tnear = t1
            if t2 < tfar:
                tfar = t2


            if tnear > tfar:
                return False
            if tfar < 0:
                return False

            if segment is True:
                if r1 < r0:
                    startpoint, endpoint = r1, r0
                else:
                    startpoint, endpoint = r0, r1
                if endpoint < bmin[axis] or startpoint > bmax[axis]:
                    return False

    return True

@jit
def rays_box(rays, box_bounds, segment=False):
    intersects = np.zeros(shape=(rays.shape[1],), dtype=bool)
    ray_indices = np.arange(rays.shape[1], dtype=int)
    for r in ray_indices:
        intersects[r] = ray_box(rays[:, r], box_bounds, segment)

    intersect_index_pairs = np.where(intersects==1)

    return intersects

@jit
def points_inside_box(points, box_bounds):
    xn = points[0] > box_bounds[0, 0]
    xp = points[0] < box_bounds[0, 1]
    yn = points[1] > box_bounds[1, 0]
    yp = points[1] < box_bounds[1, 1]
    zn = points[2] > box_bounds[2, 0]
    zp = points[2] < box_bounds[2, 1]

    inside = xn & xp & yn & yp & zn & zp

    return inside


if __name__ == "__main__":
    print('')
