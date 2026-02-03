import numpy as np

# local (controlbook)
from case_studies import E_blockbeam
from case_studies.common import sym_utils as su

# local imports (from this folder)
from .generate_KE import *

# potential energy
PE = m2 * g * ell / 2 * sp.sin(theta) + m1 * g * z * sp.sin(theta)

# lagrangian
L = sp.simplify(K - PE)

# Euler-lagrangian equation solution
EL_case_studyE = sp.simplify(sp.diff(sp.diff(L, sp.diff(q, t)), t) - sp.diff(L, q)) # type: ignore

# convenience variables
zd = z.diff(t)
zdd = zd.diff(t)
thetad = theta.diff(t)
thetadd = thetad.diff(t)

# external force and friction
F = sp.symbols("F")

# defining the right-hand side of the equation and combining it with E-L part
RHS = sp.Matrix([[0], [F * ell * sp.cos(theta)]])
full_eom = EL_case_studyE - RHS

# solving for highest order
result = sp.simplify(sp.solve(full_eom, (zdd, thetadd)))
zdd_eom = sp.simplify(result[zdd])
thetadd_eom = sp.simplify(result[thetadd])

print("ZDD------------------------------------------------")
sp.pprint(zdd_eom)
print("THETADD----------------------------")
sp.pprint(thetadd_eom)

# defining inputs
state = sp.Matrix([[z], [theta], [zd], [thetad]])
ctrl_input = sp.Matrix([[F]])

# function call to get state_dot
state_dot = sp.Matrix([[zd], [thetad], [zdd_eom], [thetadd_eom]])

# converting the function to a callable function that uses numpy to evaluate and
# return a list of state derivatives
eom = sp.lambdify([state, ctrl_input, m1, m2, ell, g], state_dot, "numpy")

# test input
cur_state = np.array([0, 0, 0, 0])
cur_input = np.array([2])

# import params
P = E_blockbeam.params

print("X_dot = ", eom(cur_state, cur_input, P.m1, P.m2, P.ell, P.g))

if __name__ == "__main__":
    from case_studies import D_mass

    # make sure printing only happens when running this file directly
    su.enable_printing(__name__ == "__main__")

    su.write_eom_to_file(state, ctrl_input, [m1, m2, ell, g], E_blockbeam, eom=state_dot)

    import numpy as np
    from case_studies.E_blockbeam import eom_generated
    import importlib

    importlib.reload(eom_generated)  # reload in case it was just generated/modified
    P = E_blockbeam.params

    param_vals = {
        "m1": P.m1,
        "m2": P.m2,
        "ell": P.ell,
        "g": P.g,
    }

    # test input
    x_test = np.array([0., 0., 0., 0.])
    u_test = np.array([2.])

    x_dot_test = eom_generated.calculate_eom(x_test, u_test, **param_vals)
    print("\nx_dot_test from generated function = ", x_dot_test)
    # should match what was printed earlier when we called eom directly