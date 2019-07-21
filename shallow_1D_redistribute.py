#!/usr/bin/env python
# encoding: utf-8
r"""
Riemann solvers for the shallow water equations.

The available solvers are:
 * Roe - Use Roe averages to caluclate the solution to the Riemann problem
 * HLL - Use a HLL solver
 * Exact - Use a newton iteration to calculate the exact solution to the
        Riemann problem

.. math::
    q_t + f(q)_x = 0

where

.. math::
    q(x,t) = \left [ \begin{array}{c} h \\ h u \end{array} \right ],

the flux function is

.. math::
    f(q) = \left [ \begin{array}{c} h u \\ hu^2 + 1/2 g h^2 \end{array}\right ].

and :math:`h` is the water column height, :math:`u` the velocity and :math:`g`
is the gravitational acceleration.

:Authors:
    Kyle T. Mandli (2009-02-05): Initial version
"""
# ============================================================================
#      Copyright (C) 2009 Kyle T. Mandli <mandli@amath.washington.edu>
#
#  Distributed under the terms of the Berkeley Software Distribution (BSD)
#  license
#                     http://www.opensource.org/licenses/
# ============================================================================

import numpy as np

num_eqn = 2
num_waves = 2

def riemanntype(hL, hR, uL, uR, maxiter, drytol, g):
    h_min = min(hR,hL)
    h_max = max(hR,hL)
    delu = uR - uL

    if (h_min <= drytol):
        hm = 0.0
        um = 0.0
        s1m = uR + uL - 2.0 * np.sqrt(g * hR) + 2.0 * np.sqrt(g * hL)
        s2m = uR + uL - 2.0 * np.sqrt(g * hR) + 2.0 * np.sqrt(g * hL)
        if (hL <= 0.0):
            rare2 = True
            rare1 = False
        else:
            rare1 = True
            rare2 = False
    else:
        F_min = delu + 2.0 * (np.sqrt(g * h_min) - np.sqrt(g * h_max))
        F_max = delu + (h_max - h_min) * (np.sqrt(0.5 * g * (h_max + h_min) / (h_max * h_min)))

        if (F_min > 0.0): #2-rarefactions
            hm = (1.0 / (16.0 * g)) * max(0.0, - delu + 2.0 * (np.sqrt(g * hL) + np.sqrt(g * hR)))**2
            um = uL + 2.0 * (np.sqrt(g * hL) - np.sqrt(g * hm))
            s1m = uL + 2.0 * np.sqrt(g * hL) - 3.0 * np.sqrt(g * hm)
            s2m = uR - 2.0 * np.sqrt(g * hR) + 3.0 * np.sqrt(g * hm)
            rare1 = True
            rare2 = True

        elif (F_max <= 0.0): # !2 shocks
            # root finding using a Newton iteration on sqrt(h)===
            h0 = h_max
            for iter in xrange(maxiter):
                gL = np.sqrt(0.5 * g * (1 / h0 + 1 / hL))
                gR = np.sqrt(0.5 * g * (1 / h0 + 1 / hR))
                F0 = delu + (h0 - hL) * gL + (h0 - hR) * gR
                dfdh = gL - g * (h0 - hL) / (4.0 * (h0**2) * gL) + gR - g * (h0 - hR) / (4.0 * (h0**2) * gR)
                slope = 2.0 * np.sqrt(h0) * dfdh
                h0 = (np.sqrt(h0) - F0 / slope)**2

            hm = h0
            u1m = uL - (hm-hL) * np.sqrt((0.5 * g) * (1 / hm + 1 / hL))
            u2m = uR + (hm - hR) * np.sqrt((0.5 * g) * (1 / hm + 1 / hR))
            um = 0.5 * (u1m + u2m)
            s1m = u1m - np.sqrt(g * hm)
            s2m = u2m + np.sqrt(g * hm)
            rare1 = False
            rare2 = False

        else: #one shock one rarefaction
            h0 = h_min
            for iter in xrange(maxiter):
                F0 = delu + 2.0 * (np.sqrt(g * h0) - np.sqrt(g * h_max)) + (h0 - h_min) * np.sqrt(0.5 * g * (1 / h0 + 1 / h_min))
                slope = (F_max - F0) / (h_max - h_min)
                h0 = h0 - F0 / slope

            hm = h0
            if (hL > hR):
                um = uL + 2.0 * np.sqrt(g * hL) - 2.0 * np.sqrt(g * hm)
                s1m = uL + 2.0 * np.sqrt(g * hL) - 3.0 * np.sqrt(g * hm)
                s2m = uL + 2.0 * np.sqrt(g * hL) - np.sqrt(g * hm)
                rare1 = True
                rare2 = False
            else:
                s2m = uR - 2.0 * np.sqrt(g * hR) + 3.0 * np.sqrt(g * hm)
                s1m = uR - 2.0 * np.sqrt(g * hR) + np.sqrt(g * hm)
                um = uR - 2.0 * np.sqrt(g * hR) + 2.0 * np.sqrt(g * hm)
                rare2 = True
                rare1 = False

    return hm, s1m, s2m, rare1, rare2


def shallow_fwave_1d(q_l, q_r, aux_l, aux_r, problem_data):
    r"""Shallow water Riemann solver using fwaves

    Also includes support for bathymetry but be wary if you think you might have
    dry states as this has not been tested.

    *problem_data* should contain:
     - *grav* - (float) Gravitational constant
     - *sea_level* - (float) Datum from which the dry-state is calculated.

    :Version: 1.0 (2014-09-05)
    """

    g = problem_data['grav']

    num_rp = q_l.shape[1]
    num_eqn = 2
    num_waves = 2

    # Output arrays
    fwave = np.zeros( (num_eqn, num_waves, num_rp) )
    s = np.zeros( (num_waves, num_rp) )
    amdq = np.zeros( (num_eqn, num_rp) )
    apdq = np.zeros( (num_eqn, num_rp) )

    # Extract state
    u_l = np.where(q_l[0,:] - problem_data['sea_level'] > 1e-3,
                   q_l[1,:] / q_l[0,:], 0.0)
    u_r = np.where(q_r[0,:] - problem_data['sea_level'] > 1e-3,
                   q_r[1,:] / q_r[0,:], 0.0)
    phi_l = q_l[0,:] * u_l**2 + 0.5 * g * q_l[0,:]**2
    phi_r = q_r[0,:] * u_r**2 + 0.5 * g * q_r[0,:]**2

    # Speeds
    s[0,:] = u_l - np.sqrt(g * q_l[0,:])
    s[1,:] = u_r + np.sqrt(g * q_r[0,:])

    delta1 = q_r[1,:] - q_l[1,:]
    delta2 = phi_r - phi_l + g * 0.5 * (q_r[0,:] + q_l[0,:]) * (aux_r[0,:] - aux_l[0,:])

    beta1 = (s[1,:] * delta1 - delta2) / (s[1,:] - s[0,:])
    beta2 = (delta2 - s[0,:] * delta1) / (s[1,:] - s[0,:])

    fwave[0,0,:] = beta1
    fwave[1,0,:] = beta1 * s[0,:]
    fwave[0,1,:] = beta2
    fwave[1,1,:] = beta2 * s[1,:]

    for m in xrange(num_eqn):
        for mw in xrange(num_waves):
            amdq[m,:] += (s[mw,:] < 0.0) * fwave[m,mw,:]
            apdq[m,:] += (s[mw,:] >= 0.0) * fwave[m,mw,:]

    return fwave, s, amdq, apdq



def riemann_fwave_1d(hL, hR, huL, huR, bL, bR, uL, uR, phiL, phiR, s1, s2, g):
    num_eqn = 2
    num_waves = 2
    fw = np.zeros((num_eqn, num_waves))

    delh = hR - hL
    delhu = huR - huL
    delb = bR - bL
    delphidecomp = phiR - phiL + g * 0.5 * (hL + hR) * delb

    beta1 = (s2 * delhu - delphidecomp) / (s2 - s1)
    beta2 = (delphidecomp - s1 * delhu) / (s2 - s1)

    # 1st nonlinear wave
    fw[0,0] = beta1
    fw[1,0] = beta1 * s1

    # 2nd nonlinear wave
    fw[0,1] = beta2
    fw[1,1] = beta2 * s2

    return fw


def barrier_passing(hL, hR, huL, huR, bL, bR, wall_height, drytol, g, maxiter):

    L2R = False
    R2L = False
    hstarL = 0.0
    hstarR = 0.0

    if (hL > drytol):
        uL = huL / hL
        hstar,_,_,_,_ = riemanntype(hL, hL, uL, -uL, maxiter, drytol, g)
        hstartest = max(hL, hstar)
        if (hstartest + bL > 0.5*(bL+bR)+wall_height):
            L2R = True
            hstarL = hstartest + bL - 0.5*(bL+bR) - wall_height

    if (hR > drytol):
        uR = huR / hR
        hstar,_,_,_,_ = riemanntype(hR, hR, -uR, uR, maxiter, drytol, g)
        hstartest = max(hR, hstar)
        if (hstartest + bR > 0.5*(bL+bR)+wall_height):
            R2L = True
            hstarR = hstartest + bR - 0.5*(bL+bR) - wall_height

    return L2R, R2L, hstarL, hstarR


def redistribute_fwave(q_l, q_r, aux_l, aux_r, wall_height, drytol, g, maxiter):

    fwave = np.zeros((2, 2, 2))
    s = np.zeros((2, 2))
    amdq = np.zeros((2, 2))
    apdq = np.zeros((2, 2))

    q_wall = np.zeros((2,3))
    aux_wall = np.zeros((1,3))
    s_wall = np.zeros(2)
    gamma = np.zeros((2,2))
    amdq_wall = np.zeros(2)
    apdq_wall = np.zeros(2)

    # hbox method
    q_wall[:,0] = q_l[:,0].copy()
    q_wall[:,2] = q_r[:,1].copy()
    # print ("aux_l.shape: ", aux_l.shape)

    aux_wall[0,0] = aux_l[0].copy()
    aux_wall[0,2] = aux_r[1].copy()
    aux_wall[0,1] = 0.5*(aux_wall[0,0] + aux_wall[0,2]) + wall_height

    L2R, R2L, hstarL, hstarR = barrier_passing(q_wall[0,0], q_wall[0,2], q_wall[1,0], q_wall[1,2], aux_wall[0,0], aux_wall[0,2], wall_height, drytol, g, maxiter)
    if (L2R==True and R2L==True):
        q_wall[0,1] = 0.5*(hstarL+hstarR)
        q_wall[1,1] = q_wall[0,1]  * (q_wall[1,0] + q_wall[1,2])/(q_wall[0,0] + q_wall[0,2])


    q_wall_l = q_wall[:,:-1].copy()
    q_wall_r = q_wall[:,1:].copy()
    aux_wall_l = aux_wall[:,:-1].copy()
    aux_wall_r = aux_wall[:,1:].copy()


    for i in xrange(2):
        hL = q_wall_l[0,i]
        hR = q_wall_r[0,i]
        huL = q_wall_l[1,i]
        huR = q_wall_r[1,i]
        bL = aux_wall_l[0,i]
        bR = aux_wall_r[0,i]

        # Check wet/dry states
        if (hR > drytol): # right state is not dry
            uR = huR / hR
            phiR = 0.5 * g * hR**2 + huR**2 / hR
        else:
            hR = 0.0
            huR = 0.0
            uR = 0.0
            phiR = 0.0

        if (hL > drytol):
            uL = huL / hL
            phiL = 0.5 * g * hL**2 + huL**2 / hL
        else:
            hL = 0.0
            huL = 0.0
            uL = 0.0
            phiL = 0.0

        if (hL > drytol or hR > drytol):
            wall = np.ones(2)
            if (hR <= drytol):
                hstar,_,_,_,_ = riemanntype(hL, hL, uL, -uL, maxiter, drytol, g)
                hstartest = max(hL, hstar)
                if (hstartest + bL <= bR):
                    wall[1] = 0.0
                    hR = hL
                    huR = -huL
                    bR = bL
                    phiR = phiL
                    uR = -uL
                elif (hL + bL <= bR):
                    bR = hL + bL

            if (hL <= drytol):
                hstar,_,_,_,_ = riemanntype(hR, hR, -uR, uR, maxiter, drytol, g)
                hstartest = max(hR, hstar)
                if (hstartest + bR <= bL):
                    wall[0] = 0.0
                    hL = hR
                    huL = -huR
                    bL = bR
                    phiL = phiR
                    uL = -uR
                elif (hR + bR <= bL):
                    bL = hR + bR

            sL = uL - np.sqrt(g * hL)
            sR = uR + np.sqrt(g * hR)
            uhat = (np.sqrt(g * hL) * uL + np.sqrt(g * hR) * uR) / (np.sqrt(g * hR) + np.sqrt(g * hL))
            chat = np.sqrt(g * 0.5 * (hR + hL))
            sRoe1 = uhat - chat
            sRoe2 = uhat + chat
            s1 = min(sL, sRoe1)
            s2 = max(sR, sRoe2)
            fw = riemann_fwave_1d(hL, hR, huL, huR, bL, bR, uL, uR, phiL, phiR, s1, s2, g)

            s[0,i] = s1 * wall[0]
            s[1,i] = s2 * wall[1]
            fwave[:,0,i] = fw[:,0] * wall[0]
            fwave[:,1,i] = fw[:,1] * wall[1]

            for mw in xrange(num_waves):
                if (s[mw,i] < 0):
                    amdq[:,i] += fwave[:,mw,i]
                elif (s[mw,i] > 0):
                    apdq[:,i] += fwave[:,mw,i]


    s_wall[0] = np.min(s)
    s_wall[1] = np.max(s)

    # s_wall[0] = s[0,0]
    # s_wall[1] = s[1,1]

    if s_wall[1] - s_wall[0] != 0.0:
        gamma[0,0] = (s_wall[1] * np.sum(fwave[0,:,:]) - np.sum(fwave[1,:,:])) / (s_wall[1] - s_wall[0])
        gamma[0,1] = (np.sum(fwave[1,:,:]) - s_wall[0] * np.sum(fwave[0,:,:])) / (s_wall[1] - s_wall[0])
        gamma[1,0] = gamma[0,0] * s_wall[0]
        gamma[1,1] = gamma[0,1] * s_wall[1]

    wave_wall = gamma
    # print("gamma[0,:]: ", gamma[0,:])
    for mw in xrange(2):
        if (s_wall[mw] < 0):
            amdq_wall[:] += gamma[:,mw]
        elif (s_wall[mw] > 0):
            apdq_wall[:] += gamma[:,mw]

    return wave_wall, s_wall, amdq_wall, apdq_wall



def shallow_fwave_dry_1d(q_l, q_r, aux_l, aux_r, problem_data):
    # print("shallow_fwave_hbox_dry_1d")
    g = problem_data['grav']
    drytol = problem_data['dry_tolerance']
    maxiter = problem_data['max_iteration']

    num_rp = q_l.shape[1]
    num_eqn = 2
    num_waves = 2
    num_ghost = 2


    # Output arrays
    fwave = np.zeros((num_eqn, num_waves, num_rp))
    s = np.zeros((num_waves, num_rp))
    amdq = np.zeros((num_eqn, num_rp))
    apdq = np.zeros((num_eqn, num_rp))

    for i in xrange(num_rp):
        hL = q_l[0,i]
        hR = q_r[0,i]
        huL = q_l[1,i]
        huR = q_r[1,i]
        bL = aux_l[0,i]
        bR = aux_r[0,i]

        # Check wet/dry states
        if (hR > drytol): # right state is not dry
            uR = huR / hR
            phiR = 0.5 * g * hR**2 + huR**2 / hR
        else:
            hR = 0.0
            huR = 0.0
            uR = 0.0
            phiR = 0.0

        if (hL > drytol):
            uL = huL / hL
            phiL = 0.5 * g * hL**2 + huL**2 / hL
        else:
            hL = 0.0
            huL = 0.0
            uL = 0.0
            phiL = 0.0

        if (hL > drytol or hR > drytol):
            wall = np.ones(2)
            if (hR <= drytol):
                hstar,_,_,_,_ = riemanntype(hL, hL, uL, -uL, maxiter, drytol, g)
                hstartest = max(hL, hstar)
                if (hstartest + bL <= bR):
                    wall[1] = 0.0
                    hR = hL
                    huR = -huL
                    bR = bL
                    phiR = phiL
                    uR = -uL
                elif (hL + bL <= bR):
                    bR = hL + bL

            if (hL <= drytol):
                hstar,_,_,_,_ = riemanntype(hR, hR, -uR, uR, maxiter, drytol, g)
                hstartest = max(hR, hstar)
                if (hstartest + bR <= bL):
                    wall[0] = 0.0
                    hL = hR
                    huL = -huR
                    bL = bR
                    phiL = phiR
                    uL = -uR
                elif (hR+ bR <= bL):
                    bL = hR + bR

            sL = uL - np.sqrt(g * hL)
            sR = uR + np.sqrt(g * hR)
            uhat = (np.sqrt(g * hL) * uL + np.sqrt(g * hR) * uR) / (np.sqrt(g * hR) + np.sqrt(g * hL))
            chat = np.sqrt(g * 0.5 * (hR + hL))
            sRoe1 = uhat - chat
            sRoe2 = uhat + chat
            s1 = min(sL, sRoe1)
            s2 = max(sR, sRoe2)
            fw = riemann_fwave_1d(hL, hR, huL, huR, bL, bR, uL, uR, phiL, phiR, s1, s2, g)

            s[0,i] = s1 * wall[0]
            s[1,i] = s2 * wall[1]
            fwave[:,0,i] = fw[:,0] * wall[0]
            fwave[:,1,i] = fw[:,1] * wall[1]

            for mw in xrange(num_waves):
                if (s[mw,i] < 0):
                    amdq[:,i] += fwave[:,mw,i]
                elif (s[mw,i] > 0):
                    apdq[:,i] += fwave[:,mw,i]
                else:
                    amdq[:,i] += 0.5 * fwave[:,mw,i]
                    apdq[:,i] += 0.5 * fwave[:,mw,i]

    if problem_data['zero_width'] == True:
        nw = problem_data['wall_position']
        wall_height = problem_data['wall_height']
        iw = nw + num_ghost - 1
        fwave[:,:,iw], s[:,iw], amdq[:,iw], apdq[:,iw] = redistribute_fwave(q_l[:,iw:iw+2].copy(), q_r[:,iw-1:iw+1].copy(), aux_l[0,iw:iw+2].copy(), aux_r[0,iw-1:iw+1].copy(), wall_height, drytol, g, maxiter)


    return fwave, s, amdq, apdq

