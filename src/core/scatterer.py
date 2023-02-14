#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 16:11:32 2018

@author: hannah

selects points in a given three dimensional range for scatterer location
random locations and at bravais lattice points
"""

import numpy as np
import matplotlib.pyplot as pl
from src.core import intersect
import miepython as mp
from tqdm import tqdm


class InitializationException(Exception):
    pass


class ScatteringMedium(object):
    def __init__(self, bounds, particle_radii, size_param, particle_index):
        # self.g_list = g_list
        # self.g_percent = g_percent
        self.bounds = bounds
        self.depth = abs(bounds[2][1] - bounds[2][0])
        self.width = abs(bounds[0][1] - bounds[0][0])
        self.height = abs(bounds[1][1] - bounds[1][0])
        self.volume = self.depth * self.width * self.height
        self.structure = ''

        self.particle_radii = particle_radii
        self.particle_index = particle_index
        self.particle_size_param = size_param

        self.particle_index = particle_index
        self.size_param = size_param
        self.thetarange, self.scat_dist = self.calc_miescatterdist()

        self.particle_centers = []
        self.particle_anisotropies = []


    def calc_miescatterdist(self):
        '''
        Generates the angular distribution using the Mie phase function, for a
        given particle size parameter and refractive index.
        '''
        m = self.particle_index
        x = self.size_param

        theta = np.linspace(-180, 180, 1000)
        mu = np.cos(theta/180*np.pi)
        scat = mp.i_unpolarized(m, x, mu)
        # normalize scat over area
        scat = scat / np.sum(scat)

        return theta, scat

    def gen_mie(self, size):
        theta, scat = self.thetarange, self.scat_dist
        theta0 = np.random.choice(theta, size=size, p=scat)

        return theta0

    def update_info(self):
        self.__basic_props__ = dict(self.__dict__)
        if self.structure == 'lattice':
            del self.__basic_props__['space_lattice']
            del self.__basic_props__['basis_lattice']

        print(self.__basic_props__)


class RandomScatteringMedium(ScatteringMedium):
    def __init__(self, bounds, particle_radii, size_param, particle_index,
                 env_index, scattering_degree=None, density=None,
                 no_overlaps=True):
        '''
        Parameters:
            scattering_degree: int
                positive integer values greater than 0. usually in the range
                [0, 10].
            density: int or float
                number of particles per unit volume (microns cube). neglected
                if scattering_degree is defined.
            heterogenous: bool
                homogeniety of particles in the medium. default False.
            anisotropies: tuple or 2D array
                anisotropy and concentration of anisotropies in the medium.
                required if heterogenous is False.
        '''
        ScatteringMedium.__init__(self, bounds, particle_radii, size_param,
                                  particle_index)
        self.structure = "random"
        self.no_overlaps = no_overlaps


        if scattering_degree is None:
            raise InitializationException("Scattering degree should be \
                                          user defined for random medium.")
        else:
            # d_s = 1 / (particle density * scattering cross section) where
            # scattering cross section (rayleigh) is 8/3 q**4 pi r**2
            self.scattering_degree = scattering_degree
            qext, qsca, qback, g = mp.mie(self.particle_index/env_index,
                                          self.size_param)
            self.qsca = qsca
            self.qext = qext
            self.g = g
            self.env_index = env_index
            A = np.pi * self.particle_radii ** 2
            self.scattering_cross_section = qsca * A
            self.radii_eff = np.sqrt(self.scattering_cross_section / np.pi)
            self.density = self.scattering_degree / \
                (self.depth*10 * self.scattering_cross_section)
            self.num_particles = np.int(self.density * self.volume)
            self.particle_anisotropies = g * \
                np.ones(shape=(self.num_particles,))
            if isinstance(self.particle_radii, (list, tuple, np.ndarray)) is False:
                self.particle_radii = self.particle_radii * \
                    np.ones(shape=(self.num_particles,))
                self.radii_eff = self.radii_eff * \
                    np.ones(shape=(self.num_particles,))

        self.init_structure()
        self.update_info()

    def init_structure(self):
        self.particle_centers = self.random()
        if self.no_overlaps is True:
            # so far, this is just for a homogenous medium.
            # first double check if all those particles would fit at all.
            # for a random medium, packing density is 65%.
            # reference: http://mathworld.wolfram.com/SpherePacking.html
            dim_vals = self.bounds[:, 1] - self.bounds[:, 0]
            volume_box = dim_vals[0] * dim_vals[1] * dim_vals[2]
            volume_sphere = (4/3) * np.pi * np.average(self.particle_radii**2)
            limit = 0.65 * volume_box / volume_sphere
            if self.num_particles > limit:
                print('I can\'t impose having no overlaps. Too many particles \
                      for a random medium.')

            radii = self.particle_radii * np.ones(shape=(self.num_particles,))
            centers = self.relocate_overlaps(radii)
            self.particle_centers = centers

    def random(self):
        '''
        Parameters:
            bounds: planes along x, y, and z; shape(3, 2)
            scat_density: number of scatterers per unit volume
        Returns:
            centers: centers of scatterers in cartersian coordinates. shape is
            (N, 3)
        '''
        xb, yb, zb = self.bounds[0], self.bounds[1], self.bounds[2]

        # scatterer density is related to h/ds
        N = self.num_particles
        n = np.arange(0, N, dtype=np.int)
        centers = np.zeros(shape=(3, N))

        centers[0] = np.random.uniform(xb[0], xb[1], size=(N,))
        centers[1] = np.random.uniform(yb[0], yb[1], size=(N,))
        centers[2] = np.random.uniform(zb[0], zb[1], size=(N,))

        return centers


    def relocate_overlaps(self, radii):
        centers = self.particle_centers
        # make sure no spheres overlap
        xb, yb, zb = self.bounds[0], self.bounds[1], self.bounds[2]

        intersects = intersect.among_spheres(centers, radii)
        intersect_index_pairs = intersect.get_indices(intersects)
        num_overlaps = intersect_index_pairs.shape[0]

        print(f'num overlaps: {num_overlaps}')

        if num_overlaps > 0 and num_overlaps < int(1E5):
            for p in tqdm(range(num_overlaps), desc="Removing overlaps"):
                two_overlapping_spheres = intersect_index_pairs[p]
                tagged = two_overlapping_spheres[0]
                limit_attempts = int(1E2)
                success = 0
                # print(f'Relocating sphere {tagged}')
                for attempt in range(limit_attempts):
                    new_center = np.empty(shape=(3,))
                    new_center[0] = np.random.uniform(xb[0], xb[1])
                    new_center[1] = np.random.uniform(yb[0], yb[1])
                    new_center[2] = np.random.uniform(zb[0], zb[1])

                    # make sure no spheres overlap
                    new_intersects = intersect.sphere_sphere(new_center,
                                                             centers,
                                                             radii[tagged],
                                                             radii)
                    new_intersect_indices = intersect.get_indices(new_intersects)
                    new_num_overlaps = new_intersect_indices.shape[0]

                    if new_num_overlaps < 1:
                        centers[:, tagged] = new_center
                        success = 1
                        break

                if success == 0:
                    print(f'Limit of attempts reached. Will not change \
                          location of sphere {tagged}.')

        elif num_overlaps > int(1E4):
            print('Too many overlaps. Will ignore this function call.')
        else:
            print('No overlaps found.')

        intersects = intersect.among_spheres(centers, radii)
        intersect_index_pairs = intersect.get_indices(intersects)
        num_overlaps = intersect_index_pairs.shape[0]

        print(f'num overlaps after relocation: {num_overlaps}')

        return centers

    def assign_anisotropy(self, g_list=None, g_percent=None):
        centers = self.particle_centers

        if self.heterogenous == True:
            num_particles = centers.shape[1]
            particle_g = np.zeros(shape=num_particles)
            last = 0
            for i in range(len(g_list)):
                g = g_list[i]
                num_g = int(g_percent[i] * num_particles)
                particle_g[last:last+num_g] = g*np.ones(num_g)
                last += num_g
        else:
            # particle_g = g_list[0] * np.ones(shape=self.num_particles)
            particle_g = self.g * np.ones(shape=self.num_particles)

        return particle_g


class LatticeScatteringMedium(ScatteringMedium):
    def __init__(self, bounds, space_structure, particle_radii, size_param,
                 particle_index, env_index, basis_structure=None,
                 lattice_constants=None, scattering_degree=None,
                 conv_cell=False):
        ScatteringMedium.__init__(self, bounds=bounds, size_param=size_param,
                                  particle_radii=particle_radii,
                                  particle_index=particle_index)
        self.structure = 'lattice'
        self.space_structure = space_structure
        self.basis_structure = basis_structure
        self.particle_radii = np.array(particle_radii)
        self.N_particles_per_ccell = 1

        if basis_structure is not None:
            self.N_particles_per_ccell += basis_structure.shape[0]

        if lattice_constants is None:
            if scattering_degree is not None:
                self.scattering_degree = scattering_degree
                qext, qsca, qback, g = mp.mie(self.particle_index/env_index,
                                              self.size_param)
                self.qsca = qsca
                self.qext = qext
                self.g = g
                self.env_index = env_index

                A = np.pi * self.particle_radii ** 2

                self.scattering_cross_section = qsca * A
                self.radii_eff = np.sqrt(self.scattering_cross_section / np.pi)

                self.density = self.scattering_degree / \
                    (self.depth*10 * self.scattering_cross_section)
                self.num_particles = np.int(self.density * self.volume)
                self.particle_anisotropies = g \
                    * np.ones(shape=(self.num_particles,))

                if isinstance(self.particle_radii,
                              (list, tuple, np.ndarray)) is False:
                    self.particle_radii = self.particle_radii \
                        * np.ones(shape=(self.num_particles,))

                if isinstance(self.radii_eff,
                              (list, tuple, np.ndarray)) is False:
                    self.radii_eff = self.radii_eff \
                        * np.ones(shape=(self.num_particles,))
                self.lattice_constants = None
            else:
                self.density = density
                self.lattice_constants = None
                # to be determined by init_structure
        else:
            self.lattice_constants = lattice_constants
            self.density = None  # to be determiend by init_structure

        self.space_lattice = []
        self.basis_lattice = []

        self.init_structure()
        self.update_info()

    def init_structure(self):
        if self.space_structure in ['cubic', 'orthorhombic', 'tetragonal']:
            a, angles = self.cuboid()
            nx = int(self.width / a[0, 0])
            ny = int(self.height / a[1, 1])
            nz = int(self.depth / a[2, 2])
            space_lattice = self.create_space_lattice(a, nx, ny, nz)
            
            sc_centers = self.ravel_ijk(space_lattice)
            sc_centers = sc_centers.T

            # shift particles
            sc_centers[0] += self.bounds[0, 0]
            sc_centers[1] += self.bounds[1, 0]
            sc_centers[2] += self.bounds[2, 0]

            self.particle_centers = sc_centers

            if self.basis_structure is not None:
                basis_lattice = self.create_basis_lattice()

        else:
            raise InitializationException
            print('Structure not defined in this class.')

    def ravel_ijk(self, R):
        return np.concatenate(np.concatenate(R, axis=0), axis=0)

    def cuboid(self):
        '''
        Can be cubic, orthorhombic, or tetragonal.
        alpha = beta = gamma = 90 deg
        Parameter:
        ----------
            lattice_constants: tuple
                length of each side (a1, a2, a3)
        Returns:
        --------
            a: numpy array
                array of the magnitude each side
            angles: numpy array
                array of angles alpha, beta, and gamma in radians
        '''

        if self.lattice_constants is None:
            vol_per_ccell = self.N_particles_per_ccell / self.density
            lattice_constants = np.ones(shape=(3,))
            a1 = np.amax(self.particle_radii) * 2  # minimum possible side len
            if self.space_structure == 'cubic':
                lattice_constants = lattice_constants * np.cbrt(vol_per_ccell)
            elif self.space_structure == 'tetragonal':
                lattice_constants[0:2] = a1
                lattice_constants[2] = vol_per_ccell / a1**2
                if lattice_constants[2] < a1:
                    print ('Spheres overlap.')
            elif self.space_structure == 'orthorhombic':
                lattice_constants[0] = a1
                excess = vol_per_ccell - a1 ** 2
                if excess < 0:
                    print ('Spheres overlap.')
                part_excess = np.random.random() * excess
                lattice_constants[1] = a1 + part_excess
                lattice_constnats[2]
            self.lattice_constants = lattice_constants

        angles = (np.pi/2) * np.ones(shape=(3,))
        a = np.diag(self.lattice_constants[0]*np.ones(shape=(3,)))

        return a, angles

    def create_space_lattice(self, a, nx, ny, nz):
        R = np.zeros(shape=(nx, ny, nz, 3))
        # the lattice points are described by R. Mathematically,
        # R[n1,n2,n3] = n1*a1 + n2*a2 + n3*a3, where ai are vectors.
        # R[n1,n2,n3] give the cartesian coordinates of the lattice point
        # indexed by n1, n2, and n3, ie R[n1,n2,n3] has the shape (3,).
        x0, y0, z0 = self.bounds[:, 0]

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    R[i, j, k] = i * a[0, :] + j * a[1, :] + k * a[2, :]

        self.space_lattice = R
        
        return R

    def create_basis_lattice(self):
        '''
        Creates another lattice from the basis such that the complete lattice
        is obtained by superimposing the output and the space lattice S.

        Parameters:
            basis: numpy array, shape (N, 3)
                Array containing the three constants [m, n, q] for each atom
                in the basis. The position of each atom is given by
                B = m * a1 + n * a2 + q * a3,
                where a1, a2, and a3 are the space lattice constants. Since the
                basis is applied on each primitive cell, then m, n, and q are
                each expected to be in the range [0, 1). The output will
                exclude atoms that coincide with the space lattice.
        Returns:
            B: lattice minus space lattice.
        '''
        basis = self.basis_structure
        M = basis.shape[0]
        centers = self.ravel_ijk(self.space_lattice)
        N = centers.shape[0]
        basis_latt = np.zeros(shape=(M * N, 3))
        m = 0
        for atom in basis:
            basis_latt[m:m+N] = np.sum([centers, atom],
                                                  axis=0)
            m += N

        xmax, xmin = np.amax(centers[:, 0]), np.amin(centers[:, 0])
        ymax, ymin = np.amax(centers[:, 1]), np.amin(centers[:, 1])
        zmax, zmin = np.amax(centers[:, 2]), np.amin(centers[:, 2])
        in_ccell = (basis_latt[:, 0] <= xmax) & (basis_latt[:, 0] >= xmin) \
            & (basis_latt[:, 1] <= ymax) & (basis_latt[:, 1] >= ymin) \
            & (basis_latt[:, 2] <= zmax) & (basis_latt[:, 2] >= zmin)

        self.basis_lattice = basis_latt[in_ccell]

        return basis_latt[in_ccell]

    def get_conventional_cell(self):
        space_cc = self.space_lattice[:2, :2, :2]
        centers = self.ravel_ijk(space_cc)
        if self.basis_structure is None:
            basis_structure = np.zeros(shape=(1, 3))
        else:
            basis_structure = self.basis_structure

        M = basis_structure.shape[0]
        N = centers.shape[0]
        basis_cc = np.zeros(shape=(M * N, 3))
        m = 0
        for atom in basis_structure:
            basis_cc[m:m+N] = np.sum([centers, atom], axis=0)
            m += N

        xmax, xmin = np.amax(centers[:, 0]), np.amin(centers[:, 0])
        ymax, ymin = np.amax(centers[:, 1]), np.amin(centers[:, 1])
        zmax, zmin = np.amax(centers[:, 2]), np.amin(centers[:, 2])

        in_ccell_s = (
            (centers[:, 0] <= xmax) & (centers[:, 0] >= xmin)
            & (centers[:, 1] <= ymax) & (centers[:, 1] >= ymin)
            & (centers[:, 2] <= zmax) & (centers[:, 2] >= zmin)
        )
        in_ccell_b = (
            (basis_cc[:, 0] <= xmax) & (basis_cc[:, 0] >= xmin)
            & (basis_cc[:, 1] <= ymax) & (basis_cc[:, 1] >= ymin)
            & (basis_cc[:, 2] <= zmax) & (basis_cc[:, 2] >= zmin)
        )

        return centers[in_ccell_s].T, basis_cc[in_ccell_b].T

    def plot(self, space_lattice, basis_lattice=None):
        xs, ys, zs = space_lattice.T

        fig = pl.figure()
        ax = pl.subplot2grid((2, 3), (0, 1), projection='3d',
                             colspan=2, rowspan=2)
        ax.scatter(xs, ys, zs, color='C0')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax2 = pl.subplot2grid((2,3), (0,0))
        ax2.plot(xs, ys, '.', color='C0')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax3 = pl.subplot2grid((2,3), (1,0))
        ax3.plot(xs, zs, '.', color='C0')
        ax3.set_xlabel('x')
        ax3.set_ylabel('z')

        if basis_lattice is not None:
            if self.basis_lattice is None:
                print("Initial object originally has no basis lattice")

            xb, yb, zb = basis_lattice.T
            ax.scatter(xb, yb, zb, color='C2')
            ax2.plot(xb, yb, '.', color='C2')
            ax3.plot(xb, zb, '.', color='C2')
