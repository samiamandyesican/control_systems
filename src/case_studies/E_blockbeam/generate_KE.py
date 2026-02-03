import sympy as sp
from case_studies.common.sym_utils import dynamicsymbols, printeq, DynamicSymbol

# define sympy symbols (dependent and independent)
t, m1, m2, ell, g = sp.symbols("t m1 m2 ell g")
z, theta = dynamicsymbols("z theta")

# define positions and velocities in terms of origin
p1 = sp.Matrix([[z*sp.cos(theta)], [z*sp.sin(theta)], [0]]) # type: ignore
p1_dot = p1.diff(t)

p2 = sp.Matrix([[ell/sp.Integer(2) * sp.cos(theta)],
                [ell/sp.Integer(2) * sp.sin(theta)],
                [0]])
p2_dot = p2.diff(t)

# define angular velocity vector
w = sp.Matrix([
    [0],
    [0],
    [sp.diff(theta, t)]
    ])

# define q
q = sp.Matrix([[z], [theta]])

# define rotational inertia tensor
# ONLY WORKS FOR ROTATION ABOUT khat! Otherwise you need rotation matrices
J2 = sp.Matrix([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, sp.Rational(1, 12) * m2 * ell**2]
])
K_1 = sp.Rational(1, 2) * m1 * p1_dot.T * p1_dot
K_2_lin = sp.Rational(1, 2) * m2 * p2_dot.T * p2_dot
K_2_rot =  sp.Rational(1, 2) * w.T * J2 * w

K = K_1 + K_2_lin + K_2_rot 
K = K[0, 0]