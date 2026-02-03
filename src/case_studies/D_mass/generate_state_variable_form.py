# %%
# local (controlbook)
from case_studies.common import sym_utils as su

# local imports (from this folder)
from .generate_KE import *

# %%[markdown]
# The code imported from above shows how we defined q, q_dot, and necessary system parameters.
# Then we used position, velocity, and angular velocity to calculate kinetic energy.

PE = sp.Rational(1,2) * k * z**2 

L = sp.simplify(K - PE)

# %%
# Solution for Euler-Lagrange equations, but this does not include right-hand side (like friction and tau)
EL_case_studyD = sp.simplify(sp.diff(sp.diff(L, sp.diff(q, t)), t) - sp.diff(L, q)) # type: ignore

# %%
############################################################
### Including friction and generalized forces, then solving for highest order derivatives
############################################################

# these are just convenience variables
zd = z.diff(t)
zdd = zd.diff(t)

# defining symbols for external force and friction
tau, b = sp.symbols("tau, b")

# defining the right-hand side of the equation and combining it with E-L part
RHS = sp.Matrix([[tau - b * zd]])
full_eom = EL_case_studyD - RHS

# finding and assigning zdd and thetadd
# if our eom were more complicated, we could rearrange, solve for the mass matrix, and invert it to move it to the other side and find qdd and thetadd
result = sp.simplify(sp.solve(full_eom, (zdd)))

# result is a Python dictionary, we get to the entries we are interested in
# by using the name of the variable that we were solving for
zdd_eom = result[zdd]  # EOM for thetadd, as a function of states and inputs


# %% [markdown]
# OK, now we can get the state variable form of the equations of motion.

# %%
from . import params as PE
import numpy as np

# defining fixed parameters that are not states or inputs (like g, ell, m, b)
# can be done like follows:
# params = [(m, P.m), (ell, P.ell), (g, P.g), (b, P.b)]

# now defining the state variables that will be passed into f(x,u)
# state = np.array([theta, thetad])
# ctrl_input = np.array([tau])

state = sp.Matrix([[z], [zd]])
ctrl_input = sp.Matrix([[tau]])

# defining the function that will be called to get the derivatives of the states
state_dot = sp.Matrix([[zd], [zdd_eom]])

sp.pprint(state_dot)

# %%

# converting the function to a callable function that uses numpy to evaluate and
# return a list of state derivatives
eom = sp.lambdify([state, ctrl_input, m, k, b], state_dot, "numpy")

# calling the function as a test to see if it works:
cur_state = np.array([0, 0])
cur_input = np.array([1])
print("x_dot = ", eom(cur_state, cur_input, P.m, P.k, P.b))

if __name__ == "__main__":
    from case_studies import D_mass

    # make sure printing only happens when running this file directly
    su.enable_printing(__name__ == "__main__")

    su.write_eom_to_file(state, ctrl_input, [m, k, b], D_mass, eom=state_dot)

    import numpy as np
    from case_studies.D_mass import eom_generated
    import importlib

    importlib.reload(eom_generated)  # reload in case it was just generated/modified
    P = D_mass.params
    param_vals = {
        "m": P.m,
        "k": P.k,
        "b": P.b,
    }

    x_test = np.array([0.0, 0.0])
    u_test = np.array([1.0])

    x_dot_test = eom_generated.calculate_eom(x_test, u_test, **param_vals)
    print("\nx_dot_test from generated function = ", x_dot_test)
    # should match what was printed earlier when we called eom directly
