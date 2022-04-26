#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Library for fitting function
"""

from cylinder_fitting import fit
from scipy.spatial import ConvexHull as cvxh


def fitting_cylinder(points,
                     guess_angles=None):
    """Fit a cylinder between 3D points."""
    if points.shape[1] == 4:
        points = points[:, :3]
    w_fit, C_fit, r_fit, fit_err = \
        fit(points,
            guess_angles=guess_angles)
    return w_fit, C_fit, r_fit, fit_err


def convex_hull(points):
    """Set a convec hull fit."""
    if points.shape[1] == 4:
        points = points[:, :3]
    return cvxh(points)
