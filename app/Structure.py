#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for computing radial and angular structural features for particles in a simulation.

This module provides functions to calculate radial structures, angular structures, 
spherical harmonics features, and higher-order structural descriptors for particle 
positions in a periodic simulation box. It uses NumPy and SciPy for numerical 
computations and is optimized with Numba for performance.

Author: Tomi
Created: May 2, 2024
"""

import numpy as np
from numba import njit
from scipy.special import sph_harm
from typing import List, Tuple


@njit
def radial_structure(particle_idx: int, positions: np.ndarray, box: np.ndarray, sigma: float) -> List[float]:
    """
    Calculate radial structure features for a particle based on its distance to others.

    Args:
        particle_idx (int): Index of the particle.
        positions (np.ndarray): Array of particle positions (N, 3).
        box (np.ndarray): Simulation box dimensions (3,).
        sigma (float): Gaussian kernel width.

    Returns:
        List[float]: Radial structure features for AA and AB particle interactions.
    """
    features = []
    cross_position = positions - positions[particle_idx]
    cross_position = np.abs(cross_position)
    cross_position = np.where(cross_position > box / 2, cross_position - box, cross_position)
    distances = np.sqrt(np.sum(cross_position ** 2, axis=1))
    distances = distances[distances != 0]  # Exclude self-distance
    distances_aa = distances[:3276]  # AA interactions
    distances_ab = distances[3276:]  # AB interactions

    for r in np.arange(0, 5, 0.1):
        structure_aa = np.sum(np.exp(-((r - distances_aa) / (np.sqrt(2) * sigma)) ** 2))
        features.append(structure_aa)
    for r in np.arange(0, 5, 0.1):
        structure_ab = np.sum(np.exp(-((r - distances_ab) / (np.sqrt(2) * sigma)) ** 2))
        features.append(structure_ab)
    
    return features


@njit
def angular_structure(particle_idx: int, positions: np.ndarray) -> List[float]:
    """
    Calculate angular structure features based on three-point structure function for a particle based on AA, AB, and BB interactions.

    Args:
        particle_idx (int): Index of the particle.
        positions (np.ndarray): Array of particle positions (N, 3).

    Returns:
        List[float]: Angular structure features for AA, AB, and BB interactions.
    """
    xi_params = [
        14.633, 14.633, 14.638, 14.638, 2.554, 2.554, 2.554, 2.554, 1.648, 1.648,
        1.204, 1.204, 1.204, 1.204, 0.933, 0.933, 0.933, 0.933, 0.695, 0.695, 0.695, 0.695
    ]
    zeta_params = [1, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 4, 16, 1, 2, 4, 16, 1, 2, 4, 16]
    lambd_params = [-1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    features = []
    n_particles = positions.shape[0]
    n_particles_a = round(0.8 * n_particles)
    range_j = np.delete(np.arange(n_particles), particle_idx)
    range_ja = range_j[:(n_particles_a - 1)]  # Type A particles
    range_jb = range_j[(n_particles_a - 1):]  # Type B particles

    # AA interactions
    for xi, zeta, lambd in zip(xi_params, zeta_params, lambd_params):
        sum_ = 0.0
        for idx_j, j in enumerate(range_ja):
            range_k = np.delete(range_ja, idx_j)
            for k in range_k:
                rjk = positions[j] - positions[k]
                rpk = positions[particle_idx] - positions[k]
                rpj = positions[particle_idx] - positions[j]
                rjk2 = np.dot(rjk, rjk)
                rpk2 = np.dot(rpk, rpk)
                rpj2 = np.dot(rpj, rpj)
                cos_theta = np.dot(rpj, rpk) / np.sqrt(rpj2 * rpk2)
                sum_ += np.exp(-(rjk2 + rpk2 + rpj2) / (xi ** 2)) * ((1 + lambd * cos_theta) ** zeta)
        features.append(sum_)

    # AB interactions
    for xi, zeta, lambd in zip(xi_params, zeta_params, lambd_params):
        sum_ = 0.0
        for j in range_ja:
            for k in range_jb:
                rjk = positions[j] - positions[k]
                rpk = positions[particle_idx] - positions[k]
                rpj = positions[particle_idx] - positions[j]
                rjk2 = np.dot(rjk, rjk)
                rpk2 = np.dot(rpk, rpk)
                rpj2 = np.dot(rpj, rpj)
                cos_theta = np.dot(rpj, rpk) / np.sqrt(rpj2 * rpk2)
                sum_ += np.exp(-(rjk2 + rpk2 + rpj2) / (xi ** 2)) * ((1 + lambd * cos_theta) ** zeta)
        features.append(sum_)

    # BB interactions
    for xi, zeta, lambd in zip(xi_params, zeta_params, lambd_params):
        sum_ = 0.0
        for idx_j, j in enumerate(range_jb):
            range_k = np.delete(range_jb, idx_j)
            for k in range_k:
                rjk = positions[j] - positions[k]
                rpk = positions[particle_idx] - positions[k]
                rpj = positions[particle_idx] - positions[j]
                rjk2 = np.dot(rjk, rjk)
                rpk2 = np.dot(rpk, rpk)
                rpj2 = np.dot(rpj, rpj)
                cos_theta = np.dot(rpj, rpk) / np.sqrt(rpj2 * rpk2)
                sum_ += np.exp(-(rjk2 + rpk2 + rpj2) / (xi ** 2)) * ((1 + lambd * cos_theta) ** zeta)
        features.append(sum_)

    return features


@njit
def angular_structure_optimized(particle_idx: int, positions: np.ndarray) -> List[float]:
    """
    Optimized version of angular structure calculation with reduced parameters and distance cutoff.

    Args:
        particle_idx (int): Index of the particle.
        positions (np.ndarray): Array of particle positions (N, 3).

    Returns:
        List[float]: Optimized angular structure features for AA, AB, and BB interactions.
    """
    xi_params = [2.554]
    zeta_params = [1]
    lambd_params = [-1]

    features = []
    n_particles = positions.shape[0]
    n_particles_a = round(0.8 * n_particles)
    range_j = np.delete(np.arange(n_particles), particle_idx)
    range_ja = range_j[:(n_particles_a - 1)]  # Type A particles
    range_jb = range_j[(n_particles_a - 1):]  # Type B particles

    # AA interactions
    for xi, zeta, lambd in zip(xi_params, zeta_params, lambd_params):
        sum_ = 0.0
        for idx_j, j in enumerate(range_ja):
            rpj = positions[particle_idx] - positions[j]
            rpj2 = np.dot(rpj, rpj)
            if rpj2 < 15 * xi * xi:
                range_k = np.delete(range_ja, idx_j)
                for k in range_k:
                    rjk = positions[j] - positions[k]
                    rpk = positions[particle_idx] - positions[k]
                    rjk2 = np.dot(rjk, rjk)
                    rpk2 = np.dot(rpk, rpk)
                    cos_theta = np.dot(rpj, rpk) / np.sqrt(rpj2 * rpk2)
                    sum_ += np.exp(-(rjk2 + rpk2 + rpj2) / (xi ** 2)) * ((1 + lambd * cos_theta) ** zeta)
        features.append(sum_)

    # AB interactions
    for xi, zeta, lambd in zip(xi_params, zeta_params, lambd_params):
        sum_ = 0.0
        for j in range_ja:
            rpj = positions[particle_idx] - positions[j]
            rpj2 = np.dot(rpj, rpj)
            if rpj2 < 15 * xi * xi:
                for k in range_jb:
                    rjk = positions[j] - positions[k]
                    rpk = positions[particle_idx] - positions[k]
                    rjk2 = np.dot(rjk, rjk)
                    rpk2 = np.dot(rpk, rpk)
                    cos_theta = np.dot(rpj, rpk) / np.sqrt(rpj2 * rpk2)
                    sum_ += np.exp(-(rjk2 + rpk2 + rpj2) / (xi ** 2)) * ((1 + lambd * cos_theta) ** zeta)
        features.append(sum_)

    # BB interactions
    for xi, zeta, lambd in zip(xi_params, zeta_params, lambd_params):
        sum_ = 0.0
        for idx_j, j in enumerate(range_jb):
            rpj = positions[particle_idx] - positions[j]
            rpj2 = np.dot(rpj, rpj)
            if rpj2 < 15 * xi * xi:
                range_k = np.delete(range_jb, idx_j)
                for k in range_k:
                    rjk = positions[j] - positions[k]
                    rpk = positions[particle_idx] - positions[k]
                    rjk2 = np.dot(rjk, rjk)
                    rpk2 = np.dot(rpk, rpk)
                    cos_theta = np.dot(rpj, rpk) / np.sqrt(rpj2 * rpk2)
                    sum_ += np.exp(-(rjk2 + rpk2 + rpj2) / (xi ** 2)) * ((1 + lambd * cos_theta) ** zeta)
        features.append(sum_)

    return features


def spherical_harmonics_features(particle_idx: int, positions: np.ndarray, box: np.ndarray) -> List[float]:
    """
    Calculate spherical harmonics features for a particle within specified shells.

    Args:
        particle_idx (int): Index of the particle.
        positions (np.ndarray): Array of particle positions (N, 3).
        box (np.ndarray): Simulation box dimensions (3,).

    Returns:
        List[float]: Spherical harmonics features for different l values and shells.
    """
    features = []
    l_values = [2, 4, 6, 8, 10, 12, 14]
    inner_shells = [1.0, 1.5, 2.0, 2.5, 3.0]

    displacements = positions - positions[particle_idx]
    displacements = np.where(np.abs(displacements) >= box * 0.5,
                             displacements - np.sign(displacements) * box, displacements)
    distances = np.linalg.norm(displacements, axis=-1)

    for r_i in inner_shells:
        filter_r = np.where((distances > r_i) & (distances < r_i + 0.5))[0]
        dr_pj = displacements[filter_r]
        r_pj = distances[filter_r]

        theta = np.arccos(dr_pj[:, 2] / r_pj)
        phi = np.arctan2(dr_pj[:, 1], dr_pj[:, 0])

        for l in l_values:
            q_lm_list = []
            for m in range(-l, l + 1):
                y_lm = sph_harm(m, l, phi, theta)
                q_lm = np.abs(np.mean(y_lm))
                q_lm_list.append(q_lm * q_lm)
            q_l = np.sqrt((4 * np.pi / (2 * l + 1)) * sum(q_lm_list))
            features.append(q_l)

    return features


def get_structure(particle_idx: int, positions: np.ndarray, box: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute radial structure features for a particle.

    Args:
        particle_idx (int): Index of the particle.
        positions (np.ndarray): Array of particle positions (N, 3).
        box (np.ndarray): Simulation box dimensions (3,).
        sigma (float): Gaussian kernel width.

    Returns:
        np.ndarray: Radial structure features.
    """
    return np.array(radial_structure(particle_idx, positions, box, sigma))


def get_radial_and_angular_structure(particle_idx: int, positions: np.ndarray, box: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute combined radial and spherical harmonics features for a particle.

    Args:
        particle_idx (int): Index of the particle.
        positions (np.ndarray): Array of particle positions (N, 3).
        box (np.ndarray): Simulation box dimensions (3,).
        sigma (float): Gaussian kernel width.

    Returns:
        np.ndarray: Combined radial and spherical harmonics features.
    """
    return np.concatenate((
        radial_structure(particle_idx, positions, box, sigma),
        spherical_harmonics_features(particle_idx, positions, box)
    ), axis=None)

