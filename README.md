# hbox-examples
H-box method for Shallow Water Examples - 1D

This directory contains a number of examples for solving the shallow
water equations in one dimension using hbox method.  If you have a valid install of 
PyClaw you should be able to run the examples from this directory with some
tweaks such as where you want the output and plots to go.

Note that many of these scripts were set up to make nice movies and output 
much more often than is probably necessary.

Examples
========
 - Well-Balancing and Ghost Fluid (sill_edge.py): Test which uses a setting of 
   a steady state including a wall assigning at the edge.
 - Conservation of mass (mass_conservation.ipynb): Demonstration that the 
   wave redistribution method maintains conservation. Includes a jump in depth at 
   x = -0.2 with zero momentum, which is the initial condition of the classic 
   dam-break problem.
 - Leaky Barriers (sill_h_box_wave.py): Test that the barrier does indeed keep
   water from flowing past it provided that the barrier is high enough. The barrier
   is assigned within a cell at x=-0.024.
 - Over-Topped Barrier on a Sloping Beach (sill_h_box_wave.py): Example of coastal flood 
   modeling with a sloping bathymetry and wet-dry interface. In this case the incoming 
   wave has enough momentum so that the wave overcomes the barrier and leads to flooding  
   on the other side of the barrier. 
