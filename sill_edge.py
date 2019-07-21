#!/usr/bin/env python
# encoding: utf-8

r"""
Shallow water flow
==================
Solve the one-dimensional shallow water equations including bathymetry:
.. math::
    h_t + (hu)_x & = 0 \\
    (hu)_t + (hu^2 + \frac{1}{2}gh^2)_x & = -g h b_x.
Here h is the depth, u is the velocity, g is the gravitational constant, and b
the bathymetry.
"""
import sys
import numpy
import matplotlib.pyplot as plt
from clawpack import riemann
import shallow_1D_redistribute
from clawpack.pyclaw.plot import plot

def before_step(solver, states):
    drytol = states.problem_data['dry_tolerance']
    for i in xrange(len(states.q[0,:])):
        states.q[0,i] = max(states.q[0,i], 0.0)
        if states.q[0,i] < drytol:
            states.q[1,i] = 0.0
    return states

def load_parameters(fileName):
    fileObj = open(fileName)
    params = {}
    for line in fileObj:
        line = line.strip()
        key_value = line.split('=')
        params[key_value[0]] = key_value[1]
    return params

def setup(kernel_language='Python',use_petsc=False, outdir='./_output', solver_type='classic'):

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    params = load_parameters('parameters_edge.txt')
    xlower = float(params['xlower'])
    xupper = float(params['xupper'])
    cells_number = int(params['cells_number'])
    nw = int(params['wall_position']) # index of the edge used to present the wall
    wall_height = float(params['wall_height'])

    x = pyclaw.Dimension(xlower, xupper, cells_number, name='x')
    domain = pyclaw.Domain(x)
    state = pyclaw.State(domain, 2, 1)
    xc = state.grid.x.centers

    # Gravitational constant
    state.problem_data['grav'] = 9.8
    state.problem_data['sea_level'] = 0.0

    # Wall position
    state.problem_data['wall_position'] = nw
    state.problem_data['wall_height'] = wall_height
    state.problem_data['dry_tolerance'] = 0.001
    state.problem_data['max_iteration'] = 1
    state.problem_data['zero_width'] = True


    solver = pyclaw.ClawSolver1D(shallow_1D_redistribute.shallow_fwave_dry_1d)

    solver.limiters = pyclaw.limiters.tvd.vanleer
    solver.order = 1
    solver.cfl_max = 0.8
    solver.cfl_desired = 0.7
    solver.kernel_language = "Python"
    solver.fwave = True
    solver.num_waves = 2
    solver.num_eqn = 2
    solver.before_step = before_step
    # # # JCP solver require 3 waves
    # # solver.num_waves = 3
    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.wall
    solver.aux_bc_lower[0] = pyclaw.BC.wall
    solver.aux_bc_upper[0] = pyclaw.BC.wall

    # Initial Conditions
    state.aux[0, :] = - 0.8 * numpy.ones(xc.shape)

    ## slope bathymetry
    # bathymetry = numpy.linspace(-0.8, -0.0, xc.shape[0] - 1, endpoint=True)
    # state.aux[0,:nw-1] = bathymetry[:nw-1]
    # state.aux[0,nw-1] = bathymetry[nw-1]
    # state.aux[0,nw:] = bathymetry[nw-1:]
    # state.aux[0, :] = numpy.linspace(-0.8, -0.0, xc.shape[0], endpoint=True)

    state.q[0, :] = 0.0 - state.aux[0, :]
    state.q[0,:] = state.q[0,:].clip(min=0)
    state.q[0, :nw-5] += 0.4
    # state.q[0, nw:] = 0.0 #dry state in the right of wall
    state.q[1, :] = 0.0


    claw = pyclaw.Controller()
    claw.keep_copy = True
    claw.tfinal = 1.0
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver
    claw.setplot = setplot
    claw.write_aux_init = True

    claw.output_style = 1
    claw.num_output_times = 20
    claw.nstepout = 1

    if outdir is not None:
        claw.outdir = outdir
    else:
        claw.output_format = None

    return claw



#--------------------------
def setplot(plotdata):
#--------------------------
    """
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of visclaw.data.ClawPlotData.
    Output: a modified version of plotdata.
    """
    plotdata.clearfigures()  # clear any old figures,axes,items data

    params = load_parameters('parameters_edge.txt')
    nw = int(params['wall_position'])
    xlower = float(params['xlower'])
    xupper = float(params['xupper'])
    regular_cells_number = int(params['cells_number'])
    wall_height = float(params['wall_height'])
    delta_x = (xupper - xlower) / regular_cells_number

    # Plot variables
    def bathy(current_data):
        return current_data.aux[0, :]

    def eta(current_data):
        return current_data.q[0, :] + bathy(current_data)

    def velocity(current_data):
        return current_data.q[1, :] / current_data.q[0, :]

    def momentum(current_data):
        return current_data.q[1, :]

    def cell_ref_lines(current_data):
        x_edge = numpy.linspace(xlower, xupper, regular_cells_number+1, endpoint=True)
        y_edge_1 =  -1.0
        y_edge_2 =  1.0
        axis = plt.gca()
        axis.plot([x_edge,x_edge],mass_ylimits,'b--',linewidth=0.5)
        x_wall = x_edge[nw]
        y1 =  0.5*(current_data.aux[0,nw-1] + current_data.aux[0,nw])
        y2 =  y1 + wall_height
        axis.plot([x_wall,x_wall],[y1,y2],'r',linewidth=2)

    def momentum_ref_lines(current_data):
        x_edge = numpy.linspace(xlower, xupper, regular_cells_number+1, endpoint=True)
        y_edge_1 =  -1.0
        y_edge_2 =  1.0
        axis = plt.gca()
        axis.plot([x_edge,x_edge],momentum_ylimits,'b--',linewidth=0.5)
        axis.plot([xlower,xupper],[0.0,0.0],'r--',linewidth=0.5)

    def wall_ref_lines(current_data):
        x_edge = numpy.linspace(xlower, xupper, regular_cells_number+1, endpoint=True)
        x_wall = x_edge[nw]
        y1 =  0.5*(current_data.aux[0,nw-1] + current_data.aux[0,nw])
        y2 =  y1 + wall_height
        axis = plt.gca()
        axis.plot([x_wall,x_wall],[y1,y2],'r',linewidth=2)

    rgb_converter = lambda triple: [float(rgb) / 255.0 for rgb in triple]
    mass_ylimits = [-1.0, 1.0]
    x_limits = [xlower, xupper]
    momentum_ylimits = [-1.0, 1.0]

    # Figure for depth
    plotfigure = plotdata.new_plotfigure(name='Depth', figno=0)
    plotfigure.kwargs = {'figsize': [6,6.4]}

    # Axes for water depth
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = x_limits
    plotaxes.ylimits = mass_ylimits
    plotaxes.title = 'Water Depth'
    plotaxes.afteraxes = wall_ref_lines
    plotaxes.axescmd = 'subplot(211)'
    plotType = '1d_pwconst'

    plotitem = plotaxes.new_plotitem(plot_type='1d_fill_between')
    plotitem.plot_var = eta
    plotitem.plot_var2 = bathy
    plotitem.color = rgb_converter((67,183,219))

    plotitem = plotaxes.new_plotitem(plot_type=plotType)
    plotitem.plot_var = bathy
    plotitem.color = 'k'

    plotitem = plotaxes.new_plotitem(plot_type=plotType)
    plotitem.plot_var = eta
    plotitem.color = 'k'

    # Axes for velocity
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(212)'
    plotaxes.xlimits = x_limits
    plotaxes.ylimits = momentum_ylimits
    plotaxes.title = 'Momentum'

    plotitem = plotaxes.new_plotitem(plot_type=plotType)
    plotitem.plot_var = momentum
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':3}


    # Figure for depth (zoom in)
    plotfigure = plotdata.new_plotfigure(name='Depth_zoom', figno=1)
    plotfigure.kwargs = {'figsize': [6,6.4]}

    # Axes for water depth
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [delta_x * (nw-5)+xlower, delta_x * (nw+5)+xlower]
    plotaxes.ylimits = mass_ylimits
    plotaxes.title = 'Water Depth'
    plotaxes.afteraxes = cell_ref_lines
    plotaxes.axescmd = 'subplot(211)'


    plotitem = plotaxes.new_plotitem(plot_type=plotType)
    # plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = bathy
    plotitem.color = 'k'

    plotitem = plotaxes.new_plotitem(plot_type=plotType)
    # plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = eta
    plotitem.color = 'k'

    # Axes for momentum
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [delta_x * (nw-5)+xlower, delta_x * (nw+5)+xlower]
    plotaxes.ylimits = momentum_ylimits
    plotaxes.title = 'Momentum'
    plotaxes.afteraxes = momentum_ref_lines
    plotaxes.axescmd = 'subplot(212)'

    plotitem = plotaxes.new_plotitem(plot_type=plotType)
    # plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = momentum
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':3}

    # plotdata.print_format = 'pdf'            # file format

    return plotdata


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)
