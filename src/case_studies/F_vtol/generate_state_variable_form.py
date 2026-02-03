import numpy as np

# local (controlbook)
from case_studies import F_vtol
from case_studies.common import sym_utils as su

# local imports (from this folder)
from .generate_KE import *

# potential energy
PE = (m_c + m_l + m_r) * g * h

# lagrangian
L = sp.simplify(KE - PE)

#q
q = sp.Matrix([[z_v], [h], [theta]])

# Euler-lagrangian equation solution
EL_case_studyE = sp.simplify(sp.diff(sp.diff(L, sp.diff(q, t)), t) - sp.diff(L, q)) # type: ignore

# convenience variables
zd = z_v.diff(t)
zdd = zd.diff(t)
hd = h.diff(t)
hdd = hd.diff(t)
thetad = theta.diff(t)
thetadd = thetad.diff(t)

# external force and friction
fr, fl = sp.symbols("fr fl")

# defining the right-hand side of the equation and combining it with E-L part
RHS = sp.Matrix([[-(fr + fl) * sp.sin(theta)], [(fr + fl) * sp.cos(theta)], [(fr - fl) * d]]) + sp.Matrix([[-mu * zd], [0], [0]])
full_eom = EL_case_studyE - RHS

# solving for highest order
result = sp.simplify(sp.solve(full_eom, (zdd, hdd, thetadd)))
zdd_eom = sp.simplify(result[zdd])
hdd_eom = sp.simplify(result[hdd])
thetadd_eom = sp.simplify(result[thetadd])

print("ZDD------------------------------------------------")
sp.pprint(zdd_eom)
print("HDD------------------------------------------------")
sp.pprint(hdd_eom)
print("THETADD----------------------------")
sp.pprint(thetadd_eom)

# defining inputs
state = sp.Matrix([[z_v], [h], [theta], [zd], [hd], [thetad]])
ctrl_input = sp.Matrix([[fr, fl]])

# function call to get state_dot
state_dot = sp.Matrix([[zd], [hd], [thetad], [zdd_eom], [hdd_eom], [thetadd_eom]])

# converting the function to a callable function that uses numpy to evaluate and
# return a list of state derivatives
eom = sp.lambdify([state, ctrl_input, m_c, J_c, m_r, m_l, d, mu, g], state_dot, "numpy")

# test input
cur_state = np.array([0., 0, 0, 0, 0, 0])
cur_input = np.array([1., 1])

# import params
P = F_vtol.params

print("X_dot = ", eom(cur_state, cur_input, P.mc, P.Jc, P.mr, P.ml, P.d, P.mu, P.g))

if __name__ == "__main__":
    from case_studies import D_mass

    # make sure printing only happens when running this file directly
    su.enable_printing(__name__ == "__main__")

    su.write_eom_to_file(state, ctrl_input, [m_c, J_c, m_r, m_l, d, mu, g], F_vtol, eom=state_dot)

    import numpy as np
    from case_studies.F_vtol import eom_generated
    import importlib

    importlib.reload(eom_generated)  # reload in case it was just generated/modified

    param_vals = {
        "m_c": P.mc,
        "J_c": P.Jc,
        "m_r": P.mr,
        "m_l": P.ml,
        "d": P.d,
        "mu": P.mu,
        "g": P.g,
    }

    # test input
    x_test = np.array([0., 0., 0., 0., 0, 0])
    u_test = np.array([1., 1])

    x_dot_test = eom_generated.calculate_eom(x_test, u_test, **param_vals)
    print("\nx_dot_test from generated function = ", x_dot_test)
    # should match what was printed earlier when we called eom directly