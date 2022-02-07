#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 16:44:20 2018

@author: hannaharjonillo

"""
cimport numpy as np
import numpy as np
from numpy.math cimport INFINITY

import matplotlib.pyplot as pl
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


ctypedef np.float64_t DTYPE
ctypedef np.uint8_t DTYPE_i
ctypedef np.int DTYPE_int


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


def traverse_voxels(origin, direction, grid3D, verbose=0):
    '''
    A voxel-traversal function based on the amanatides-woo algorithm (1987).
    Parameters:
    origin: array or tuple, shape (3,)
    direction: array or tuple, shape (3,)
    grid3D: grid dimensions (nx, ny, nz, minBound, maxBound)

    '''
    nx = grid3D['nx']
    ny = grid3D['ny']
    nz = grid3D['nz']
    minBound = grid3D['minBound']
    maxBound = grid3D['maxBound']

    flag, tmin = rayBoxIntersection(origin, direction, minBound, maxBound)

    voxels = []

    if flag == 0:
        print('\n The ray does not intersect the grid')
    else:
        if tmin < 0:
            tmin = 0

        start = origin + tmin*direction
        boxSize = maxBound - minBound

        # if (verbose)
        #     plot3(start(1), start(2), start(3), 'r.', 'MarkerSize', 15)
        # end

        x = np.floor(((start[0]-minBound[0])/boxSize[0])*nx)+1
        y = np.floor(((start[1]-minBound[1])/boxSize[1])*ny)+1
        z = np.floor(((start[2]-minBound[2])/boxSize[2])*nx)+1

        if x == nx+1:
            x = x-1
        if y == ny+1:
            y = y-1
        if z == nz+1:
            z = z-1

        if direction[0] >= 0:
            tVoxelX = x / nx
            stepX = 1
        else:
            tVoxelX = (x-1) / nx
            stepX = -1

        if direction[1] >= 0:
            tVoxelY = y / ny
            stepY = 1
        else:
            tVoxelY = (y-1) / ny
            stepY = -1

        if direction[2] >= 0:
            tVoxelZ = z / nz
            stepZ = 1
        else:
            tVoxelZ = (z-1) / nz
            stepZ = -1

        voxelMaxX = minBound[0] + tVoxelX*boxSize[0]
        voxelMaxY = minBound[1] + tVoxelY*boxSize[1]
        voxelMaxZ = minBound[2] + tVoxelZ*boxSize[2]

        tMaxX = tmin + (voxelMaxX-start[0]) / direction[0]
        tMaxY = tmin + (voxelMaxY-start[1]) / direction[1]
        tMaxZ = tmin + (voxelMaxZ-start[2]) / direction[2]

        voxelSizeX = boxSize[0] / nx
        voxelSizeY = boxSize[1] / ny
        voxelSizeZ = boxSize[2] / nz

        tDeltaX = voxelSizeX/abs(direction[0])
        tDeltaY = voxelSizeY/abs(direction[1])
        tDeltaZ = voxelSizeZ/abs(direction[2])

        while ((x <= nx) & (x >= 1) & (y <= ny) & (y >= 1) & (z <= nz) & (z >= 1)):
            # ----------------------------------------------------------
            # check if voxel [x,y,z] contains any intersection with the ray
            #
            #   if ( intersection )
            #       break
            #   end
            # ----------------------------------------------------------

            print(f'\nIntersection: voxel = {x, y, z}')
            voxels.append(np.array([x, y, z]))

            if (tMaxX < tMaxY):
                if (tMaxX < tMaxZ):
                    x = x + stepX
                    tMaxX = tMaxX + tDeltaX
                else:
                    z = z + stepZ
                    tMaxZ = tMaxZ + tDeltaZ
            else:
                if (tMaxY < tMaxZ):
                    y = y + stepY
                    tMaxY = tMaxY + tDeltaY
                else:
                    z = z + stepZ
                    tMaxZ = tMaxZ + tDeltaZ

    return voxels


def show_voxels(voxels, grid3D, origin, endpoint):
    print(voxels)
    nx = grid3D['nx']
    ny = grid3D['ny']
    nz = grid3D['nz']
    minBound = grid3D['minBound']
    maxBound = grid3D['maxBound']

    mpl.style.use('default')
    fig = pl.figure()
    ax = fig.gca(projection='3d')
    for voxel in voxels:
        x, y, z = voxel[0], voxel[1], voxel[2]
        t1 = [(x-1)/nx, (y-1)/ny, (z-1)/nz]
        t2 = [x/nx, y/ny, z/nz]

        boxSize = maxBound - minBound
        vmin = minBound + t1 * boxSize
        vmax = minBound + t2 * boxSize
        print(vmin)

        # show voxels
        # prepare some coordinates
        xi, yi, zi = np.indices((maxBound[0], maxBound[1], maxBound[2]))
        print(xi)

        # draw cuboids
        cube1_lb = (xi <= vmax[0]) & (yi <= vmax[1]) & (zi <= vmax[2])
        cube1_ub = (xi > vmin[0]) & (yi > vmin[1]) & (zi > vmin[2])
        cube1 = cube1_lb & cube1_ub

        voxel_i = cube1

        ax.voxels(voxel_i, facecolors=[0.2, 0.3, 0.6, 0.4], edgecolors=[0.2, 0.3, 0.6, 0.5])
    ax.plot([origin[0], endpoint[0]], [origin[1], endpoint[1]], [origin[2], endpoint[2]], color='C1')
    pl.show()


def rays_spheres(double[:, :, :] rays, double[:, :] spheres, double[:] radii,
                 bint segment=False, bint debug_mode=False):
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
    cdef int num_rays = rays.shape[1]
    cdef int num_spheres = spheres.shape[1]
    cdef np.uint8_t [:, :] intersects = np.zeros(shape=(num_rays, num_spheres), dtype=np.uint8)
    # intersects_coords = np.inf*np.ones(shape=(rays.shape[1], spheres.shape[1]))
    cdef list intersect_index_pairs = []

    cdef int line_intersect
    cdef double l2oc, tca, d2, t2hc
    cdef double x0, y0, z0, x1, y1, z1, dx, dy, dz, a
    cdef double min_distance_squared
    cdef list min_distance_indices = []

    cdef Py_ssize_t r, s, i, j, k
    i, j, k = 0, 1, 2

    for r in range(num_rays):
        # print(f'\n ray {r}')
        x0, y0, z0 = rays[i, r, 0], rays[j, r, 0], rays[k, r, 0]
        x1, y1, z1 = rays[i, r, 1], rays[j, r, 1], rays[k, r, 1]
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0

        # normalize ray direction
        a = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if a != 0:
            dx, dy, dz = dx/a, dy/a, dz/a

        min_distance_squared = INFINITY
        min_distance_indices = None

        for s in range(num_spheres):
            # if debug_mode:
            #     print(f'\n \t sphere {s}')
            line_intersect = False
            xs, ys, zs = spheres[i, s], spheres[j, s], spheres[k, s]
            tol = radii[s]//100

            l2oc = (xs - x0)**2 + (ys - y0)**2 + (zs - z0)**2
            # if debug_mode:
            #     print(f'\n \t l2oc: {np.sqrt(l2oc), radii[s]}')
            if l2oc < radii[s]**2:  # if ray starts from inside sphere
                continue

            tca = (dx * (xs - x0)) + (dy * (ys - y0)) + (dz * (zs - z0))
            # if debug_mode:
            #     print(f'\t tca: {tca}')
            if tca < 0:  # if ray points away from sphere
                continue

            d2 = l2oc - tca**2
            # if debug_mode:
            #     print(f'\t distance ray to center of sphere: {np.sqrt(np.abs(d2))} \n \t radius: {radii[s]}')

            if np.abs(d2) > radii[s]**2:
                continue

            t2hc = radii[s]**2 - d2
            # print(f'\t t2hc: {t2hc}')
            if t2hc < 0:
                continue

            line_intersect = 1
            # if debug_mode:
            #     print(f'\t intersection: {line_intersect}')
            #     print(f'\t intersected sphere at {xs, ys, zs}')

            if l2oc < min_distance_squared:
                min_distance_indices = [r, s]
                min_distance_squared = l2oc

            intersects[r, s] = line_intersect

        if min_distance_indices != None:
            intersect_index_pairs.append(min_distance_indices)

    intersect_index_pairs_nparr = np.array(intersect_index_pairs)
    return intersect_index_pairs_nparr


def sphere_sphere(sphere0, all_spheres, radius0, all_radii):
    x0, y0, z0 = sphere0
    d2 = (all_spheres[0] - x0)**2 + (all_spheres[1] - y0)**2 + \
        (all_spheres[2] - z0)**2
    intersects = d2 < ((radius0 + all_radii)**2)

    return intersects


def get_indices(intersects, exclude_self=True):
    # remove double counting and self intersection by zeroing out
    # elements below diagonal
    intersects = np.tril(intersects, -1)
    intersects_indices = np.where(intersects == 1)
    intersects_indices = np.vstack((intersects_indices[0], intersects_indices[1])).T

    return intersects_indices


def among_spheres(spheres, radii, self_intersect=False):
    '''
    Checks for if any two (or more) spheres overlap in a system of multiple
    stationary spheres. Returns indices of intersecting spheres (excluding
    self intersection)
    '''
    intersects = np.zeros(shape=(spheres.shape[1], spheres.shape[1]),
                          dtype=bool)
    sphere_indices = np.arange(spheres.shape[1], dtype=int)
    # print(radii[0])

    for s in sphere_indices:
        intersects[s] = sphere_sphere(spheres[:, s], spheres, radii[s], radii)

    return intersects


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


def rays_box(rays, box_bounds, segment=False):
    intersects = np.zeros(shape=(rays.shape[1],), dtype=bool)
    ray_indices = np.arange(rays.shape[1], dtype=int)
    for r in ray_indices:
        intersects[r] = ray_box(rays[:, r], box_bounds, segment)

    intersect_index_pairs = np.where(intersects==1)

    return intersects


def points_inside_box(points, box_bounds):
    xn = points[0] > box_bounds[0, 0]
    xp = points[0] < box_bounds[0, 1]
    yn = points[1] > box_bounds[1, 0]
    yp = points[1] < box_bounds[1, 1]
    zn = points[2] > box_bounds[2, 0]
    zp = points[2] < box_bounds[2, 1]

    inside = xn & xp & yn & yp & zn & zp

    return inside


def test():
    # A fast and simple voxel traversal algorithm through a 3D space partition (grid)
    # proposed by J. Amanatides and A. Woo (1987).

    # Test Nro. 1
    origin = np.array([15, 15, 15])
    direction = np.array([-0.3, -0.5, -0.7])

    # Test Nro. 2
    # origin = [-8.5, -4.5, -9.5]
    # direction = [0.5, 0.5, 0.7]

    # Grid: dimensions
    grid3D = {'nx': 10, 'ny': 15, 'nz': 20, 'minBound': np.array([0, 0, 0]),
              'maxBound': np.array([20,  20,  20])}

    voxels = traverse_voxels(origin, direction, grid3D, verbose=1)
    endpoint = origin + 30*direction
    show_voxels(voxels, grid3D, origin, endpoint)


if __name__ == "__main__":
    test()
