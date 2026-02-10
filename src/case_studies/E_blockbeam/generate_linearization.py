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
state_variable_form = sp.Matrix([[zd], [thetad], [zdd_eom], [thetadd_eom]])
states = sp.Matrix([[z], [theta], [zd], [thetad]])
inputs = sp.Matrix([[F]])


# finding the jacobian with respect to states (A) and inputs (B)
A = state_variable_form.jacobian(states)
B = state_variable_form.jacobian(inputs)

# sp.pprint(A)
# sp.pprint(B)

# sub in values for equilibrium points (x_e, u_e) or (x_0, u_0)
x_e = sp.Matrix([[sp.Rational(1, 2) * ell], [0], [0], [0]])
u_e = sp.Matrix([[sp.Rational(1, 2) * (m1+m2) * g]])
z_e = sp.symbols("z_e")
equilibrium = [(thetad, 0.0), (theta, 0.0), (zd, 0.0), (z, z_e), (F, g * (sp.Rational(1, 2) * m2 + m1 * z_e / ell))]
A_lin = sp.simplify(A.subs(equilibrium))
B_lin = sp.simplify(B.subs(equilibrium))

sp.pprint(A_lin)
sp.pprint(B_lin)

sp.pprint(sp.simplify(A_lin @ (states - x_e) + B_lin @ (inputs - u_e)))


# %%
# TODO - this form of substitution is OK because they are all 0, however, substituting
# 0 (or any constant) for theta, will also make thetad zero. This is a known issue - (see "substitution"
# at https://docs.sympy.org/latest/modules/physics/mechanics/advanced.html). Future
# examples should substitute for dynamic symbols after the EOM are derived, and before
# substituting in the equilibrium points. It may be OK because velocity variables like
# thetad and zd should be zero at an equilibrium point, and would be for any constant value
# of the generalized coordinates in the state vector, but it is something to be aware of.

# see also use of "msubs" at link above for speed up in substitution in large expressions.
