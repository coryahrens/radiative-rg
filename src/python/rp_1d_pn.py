#!/usr/bin/env python
# encoding: utf-8
r"""
Riemann solvers for for the 1D, Cartesian geometry P_n equations.

.. math:: phi_t + A phi_z = 0
  
where 

.. math:: 
    phi(z,t) = \left [ \begin{array}{c} \phi_0(z,t)...\phi_n(z,t) \end{array} \right ]
  
and the coefficient matrix is 

.. math::

    A = \left [\begin{matrix}
    0      & K\\
    1/\rho & 0 
    \end{matrix} \right ]

:Authors:
    Cory Ahrens (2013-02-25): Initial version
"""
# ============================================================================
#  Distributed under the terms of the Berkeley Software Distribution (BSD) 
#  license
#                     http://www.opensource.org/licenses/
# ============================================================================

num_eqn = 2
num_waves = 2

def rp_acoustics_1d(q_l,q_r,aux_l,aux_r,problem_data):
    r"""
    Basic 1d acoustics riemann solver, with interleaved arrays
    
    *problem_data* is expected to contain -
     - *zz* - (float) Impedence
     - *cc* - (float) Speed of sound
    
    See :ref:`pyclaw_rp` for more details.
    
    :Version: 1.0 (2009-02-03)
    """
    import numpy as np

    # Convenience
    num_rp = np.size(q_l,1)

    # Return values
    wave = np.empty( (num_eqn, num_waves, num_rp) )
    s = np.empty( (num_waves, num_rp) )
    amdq = np.empty( (num_eqn, num_rp) )
    apdq = np.empty( (num_eqn, num_rp) )
    
    # Local values
    delta = np.empty(np.shape(q_l))
    
    delta = q_r - q_l
    a1 = (-delta[0,:] + problem_data['zz']*delta[1,:]) / (2.0 * problem_data['zz'])
    a2 = ( delta[0,:] + problem_data['zz']*delta[1,:]) / (2.0 * problem_data['zz'])
        
    # Compute the waves
    # 1-Wave
    wave[0,0,:] = -a1 * problem_data['zz']
    wave[1,0,:] =  a1
    s[0,:] = -problem_data['cc']
        
    # 2-Wave
    wave[0,1,:] = a2 * problem_data['zz']
    wave[1,1,:] = a2
    s[1,:] = problem_data['cc']
    
    # Compute the left going and right going fluctuations
    for m in xrange(num_eqn):
        amdq[m,:] = s[0,:] * wave[m,0,:]
        apdq[m,:] = s[1,:] * wave[m,1,:]
    
    return wave, s, amdq, apdq
