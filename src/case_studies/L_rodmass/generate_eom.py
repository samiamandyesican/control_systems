import numpy as np
import sympy as sp
from case_studies.common.sym_utils import dynamicsymbols, DynamicSymbol, write_eom_to_file
from case_studies.L_rodmass import params as P

t, b, m, g, ell, m, k1, k2, tau = sp.symbols("t, b, m, g, ell, m, k1, k2, tau")
theta = DynamicSymbol("theta")

# generate KE
# position of mass
p = sp.Matrix([[ell*sp.cos(theta)], [ell*sp.sin(theta)]])
pdot = p.diff(t)

KE = sp.Rational(1, 2) * m * pdot.T @ pdot
KE = sp.simplify(KE[0])

print("KE:")
sp.pprint(KE)

# generate PE
PE_mass = m*g*sp.sin(theta)
PE_spring = sp.Rational(1, 2)*k1*theta**2 + sp.Rational(1,4)*k2*theta**4
PE = PE_mass + PE_spring
PE = sp.simplify(PE)

print("PE:")
sp.pprint(PE)

# generate L
L = KE - PE
L = sp.simplify(L)

print("L:")
sp.pprint(L)

# generate non-conservative forces
RHS = tau - b*theta.diff(t)

# Euler legrange
thetad = theta.diff(t)
thetadd = thetad.diff(t)
LHS = L.diff(thetad).diff(t) - L.diff(theta)

full_eom = RHS - LHS
result = sp.solve(full_eom, thetadd)
thetadd_eom = sp.simplify(result[0])

print("thetadd = ")
sp.pprint(thetadd_eom)

