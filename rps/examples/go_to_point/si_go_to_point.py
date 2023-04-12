import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time

# Instantiate Robotarium object
N = 4
# initial_conditions = np.array(np.mat('5 0.5 -0.5 0 0.28; 0.8 -0.3 -0.75 0.1 0.34; 0 0 0 0 0'))
initial_conditions = generate_initial_conditions(N)
# for i in initial_conditions[]
print(initial_conditions.shape)

print(initial_conditions)

r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=False)

# Define goal points by removing orientation from poses
goal_points = generate_initial_conditions(N)

# Create single integrator position controller
single_integrator_position_controller = create_si_position_controller()

# Create barrier certificates to avoid collision
#si_barrier_cert = create_single_integrator_barrier_certificate()
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

_, uni_to_si_states = create_si_to_uni_mapping()

# Create mapping from single integrator velocity commands to unicycle velocity commands
si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()

# Create Walls of Map

# Vertices
l1 = [-7.0, 9.5]
l2 = [-7.0, 3]

l3 = [0, 9.5]
l4 = [0, 3]

l5 = [5, 1]
l6 = [8, 1]

l7 = [2, -9.5]
l8 = [2, -3]

l9 = [-3.5, -9.5]
l10 = [-3.5, -3]

# Convert to line segments by x- or y-
xl12 = [l1[0], l2[0]]
yl12 = [l1[1], l2[1]]

xl34 = [l3[0], l4[0]]
yl34 = [l3[1], l4[1]]

xl56 = [l5[0], l6[0]]
yl56 = [l5[1], l6[1]]

xl78 = [l7[0], l8[0]]
yl78 = [l7[1], l8[1]]

xl91 = [l9[0], l10[0]]
yl91 = [l9[1], l10[1]]

# Plot Walls
plt.plot(xl12, yl12, 'k', linewidth=7)
plt.plot(xl34, yl34, 'k', linewidth=7)
plt.plot(xl56, yl56, 'k', linewidth=7)
plt.plot(xl78, yl78, 'k', linewidth=7)
plt.plot(xl91, yl91, 'k', linewidth=7)

# Plot Room Labels
plt.text(-3.5, 8.5, "Room 1", size=10,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )

plt.text(5, 8.5, "Room 2", size=10,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )

plt.text(6, -8.5, "Room 3", size=10,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )

plt.text(-0.75, -8.5, "Room 4", size=10,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )

plt.text(-8.5, 8.5, "Garage", size=10,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )


# define x initially
x = r.get_poses()
x_si = uni_to_si_states(x)
r.step()

# While the number of robots at the required poses is less
# than N...
while (np.size(at_pose(np.vstack((x_si,x[2,:])), goal_points, rotation_error=100)) != N):

    # Get poses of agents
    x = r.get_poses()
    x_si = uni_to_si_states(x)

    # Create single-integrator control inputs
    dxi = single_integrator_position_controller(x_si, goal_points[:2][:])

    # Create safe control inputs (i.e., no collisions)
    dxi = si_barrier_cert(dxi, x_si)

    # Transform single integrator velocity commands to unicycle
    dxu = si_to_uni_dyn(dxi, x)

    # Set the velocities by mapping the single-integrator inputs to unciycle inputs
    r.set_velocities(np.arange(N), dxu)
    # Iterate the simulation
    r.step()

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
