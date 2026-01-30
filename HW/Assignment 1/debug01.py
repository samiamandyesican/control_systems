#%%
# make a list of x, y, z - velocities: 
arm_vel = [2, 3, 2.5]

# print the last entry in vel: 
print(arm_vel[-1])

# Now imagine that this is the velocity of the tip of a robot arm based on how its joints are rotating.
# If we want to add the velocity of a mobile base that it is attached to the arm to get the total velocity 
# of the robot, we can add the base velocity to the arm velocity.
base_vel = [1, 0, 0]

#%%
# add the base velocity to the arm velocity:
total_robot_vel = arm_vel + base_vel
print("total velocity is:", total_robot_vel)

# TODO if the output is not what you expected, how you can fix it?
# You can either do a formatted string or separate the terms as I did above.
# You also need to add the vectors element wise. By default the lists are
# just appended to one another for normal python lists.

#%%
# Now let's introduce the numpy library to make linear algebra operations much easier.
import numpy as np

# we can make arrays from the previous lists
arm_vel_np = np.array(arm_vel)
base_vel_np = np.array(base_vel)

# add the base velocity to the arm velocity:
total_robot_vel_np = arm_vel_np + base_vel_np

print('total velocity is:', total_robot_vel_np)

#TODO answer this question as a comment below:
# did this behave as you expected?
# yes.



#%%
# Now we want to calculate the magnitude of the total velocity of the tip of the robot arm.
# we can do this by calculating the norm of the total velocity vector as follows: 
total_robot_vel_magnitude = np.sqrt(total_robot_vel_np*total_robot_vel_np)
print("magnitude of velocity at arm tip:", total_robot_vel_magnitude)

# did this give the answer you expected? If not, why not? Can you fix it? 
# TODO put your answer here.
# It did not behave as desired because np.sqrt acted element-wise and returned a vector
# Here is a correct way to do it
total_robot_vel_magnitude = np.linalg.norm(total_robot_vel_np)
print("magnitude of velocity at arm tip:", total_robot_vel_magnitude)

# or equivalently
total_robot_vel_magnitude = np.sqrt(np.sum(total_robot_vel_np**2))
print("magnitude of velocity at arm tip:", total_robot_vel_magnitude)


# %%
