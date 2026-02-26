import sympy as sp
from case_studies.D_mass import params as P
import numpy as np

s, m, b, k = sp.symbols('s m b k')
char_eq = s**2 + (b/m)*s + (k/m)

# Find the roots of the characteristic equation
roots = sp.solve(char_eq, s)
vals = {
    m: P.m,
    b: P.b,
    k: P.k
}

print("mass system poles:")
for i, root in enumerate(roots):
    print(f"Root {i+1}: {root.subs(vals)}")


k_p, b_0, a_1, a_0, k_D, k_P = sp.symbols('k_p b_0 a_1 a_0 k_D k_P')
tf_PD = k_p * b_0 / (s**2 + (a_1 + b_0 * k_D)*s + (a_0 + b_0 * k_P))
vals2 = {
    b_0: 1/m,
    a_1: b/m,
    a_0: k/m,
}
for key, val in vals2.items():
    print(f"{key} = {val.subs(vals)}")
num, den = tf_PD.subs(vals2).as_numer_denom()

char_eq_PD = den
roots_PD = sp.solve(char_eq_PD, s)

print("\nPD controlled mass system poles:")
sp.pprint(roots_PD)
for i, root in enumerate(roots_PD):
    print(f"Root {i+1}: {root.subs(vals)}")


p1 = -1
p2 = -1.5
print(f"\nDesired poles: {p1}, {p2}")
desired_char_eq = sp.Poly((s - p1) * (s - p2), s).monic()
actual_char_eq = sp.Poly(char_eq_PD.subs(vals2).subs(vals), s).monic()

print(f"\nDesired characteristic equation: {desired_char_eq.as_expr()}")
print(f"\nCharacteristic equation in terms of k_P and k_D: {actual_char_eq.as_expr()}")
sol_gains_alt = sp.solve(sp.Matrix(desired_char_eq.coeffs()) - sp.Matrix(actual_char_eq.coeffs()), (k_P, k_D))
print("\nPD gains for desired poles:")
sp.pprint(sol_gains_alt)

# tuning parameters
# p1 = -10
# p2 = -10
t_r = 2.0 # 2 second rise time
zeta = 0.707 # damping ratio for 5% overshoot
w_n = np.pi / (2*t_r * np.sqrt(1-zeta**2)) # natural frequency

# system parameters
b0 = P.tf_num[-1]
a1, a0 = P.tf_den[-2:]

# desired characteristic equation parameters
# s^2 + alpha1*s + alpha0 = s^2 + (a1 + b0*kd)s + (a0 + b0*kp)
# des_CE = np.poly([p1, p2])
des_CE = [1, 2*zeta*w_n, w_n**2]
alpha1, alpha0 = des_CE[-2:]

print(f"\nDesired characteristic equation: {des_CE}")
print(des_CE)

des_roots = np.roots(des_CE)
print(f"\nDesired poles: {des_roots}")

# find gains
kp = (alpha0 - a0) / b0
kd = (alpha1 - a1) / b0
print(f"{kp = :.2f}, {kd = :.3f}")