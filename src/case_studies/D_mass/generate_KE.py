import sympy as sp
from case_studies.common.sym_utils import enable_printing, dynamicsymbols, printeq, DynamicSymbol

# define sympy symbols (dependent and independent)
t, m, k, b = sp.symbols("t m k b")
z = DynamicSymbol("z")

# define positions and velocities in terms of origin
p1 = sp.Matrix([[z]])
p1_dot = p1.diff(t)

# define q
q = sp.Matrix([[z]])

# define rotational inertia tensor
# ONLY WORKS FOR ROTATION ABOUT khat! Otherwise you need rotation matrices
K = sp.Rational(1, 2) * m * p1_dot.T * p1_dot
K = K[0,0]