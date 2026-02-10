# %%
# local (controlbook)
from case_studies.common import sym_utils as su

# local imports (from this folder)
from .generate_state_variable_form import *

# This makes it so printing from su only happens when running this file directly
su.enable_printing(__name__ == "__main__")

# %%
############################################################
### Defining vectors for x_dot, x, and u, then taking partial derivatives
############################################################


# defining derivative of states, states, and inputs symbolically
### for this first one, keep in mind that zdd_eom is actually a full
### row of equations, while zdd is just the symbolic variable itself.
F, tau = sp.symbols("F tau")
force_mixing = [
    (fr, sp.Rational(1, 2) * F + sp.Rational(1,2) * tau / d),
    (fl, sp.Rational(1, 2) * F - sp.Rational(1,2) * tau / d)
    ]
state_variable_form = sp.Matrix([[zd], [hd], [thetad], [zdd_eom.subs(force_mixing)], [hdd_eom.subs(force_mixing)], [thetadd_eom.subs(force_mixing)]])
inputs = sp.Matrix([[F], [tau]])

# find equilibriums
equilibrium_vars = [z_v, h, theta, zd, hd, thetad, F, tau]
equilibrium_sol = sp.solve(state_variable_form, equilibrium_vars)

sp.pprint(equilibrium_sol)

# finding the jacobian with respect to states (A) and inputs (B)
A = state_variable_form.jacobian(state)
B = state_variable_form.jacobian(inputs)

# sp.pprint(A)
# sp.pprint(B)

# sub in values for equilibrium points (x_e, u_e) or (x_0, u_0)
equilibrium = [(z_v, 0.0), (h, 0.0), (theta, 0.0), (zd, 0.0), (hd, 0.0), (thetad, 0.0), (F, sp.symbols("F_e")), (tau, 0.0)]
A_lin = sp.simplify(A.subs(equilibrium))
B_lin = sp.simplify(B.subs(equilibrium))

sp.pprint(A_lin)
sp.pprint(B_lin)

sp.pprint(sp.simplify(A_lin @ state + B_lin @ inputs))


# %%
# TODO - this form of substitution is OK because they are all 0, however, substituting
# 0 (or any constant) for theta, will also make thetad zero. This is a known issue - (see "substitution"
# at https://docs.sympy.org/latest/modules/physics/mechanics/advanced.html). Future
# examples should substitute for dynamic symbols after the EOM are derived, and before
# substituting in the equilibrium points. It may be OK because velocity variables like
# thetad and zd should be zero at an equilibrium point, and would be for any constant value
# of the generalized coordinates in the state vector, but it is something to be aware of.

# see also use of "msubs" at link above for speed up in substitution in large expressions.
