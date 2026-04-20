import numpy as np
import sympy as sp
from case_studies.common.sym_utils import dynamicsymbols, DynamicSymbol, write_eom_to_file
from case_studies.L_rodmass import params as P

t, b, m, g, ell, k1, k2, tau = sp.symbols("t, b, m, g, ell, k1, k2, tau")
theta = DynamicSymbol("theta")
thetad = theta.diff(t)
thetadd = thetad.diff(t)

def generate_KE():
    # generate KE
    # position of mass
    p = sp.Matrix([[ell*sp.cos(theta)], [ell*sp.sin(theta)]])
    pdot = p.diff(t)

    KE = sp.Rational(1, 2) * m * pdot.T @ pdot
    KE = sp.simplify(KE[0])

    print("KE:")
    sp.pprint(KE)

    return KE


def generate_PE():
    # generate PE
    PE_mass = m*g*sp.sin(theta)
    PE_spring = sp.Rational(1, 2)*k1*theta**2 + sp.Rational(1,4)*k2*theta**4
    PE = PE_mass + PE_spring
    PE = sp.simplify(PE)

    print("PE:")
    sp.pprint(PE)

    return PE

def generate_eom(KE, PE, test_state=np.array([0., 3.]), test_input=np.array([2.0]), regenerate_eom_file=True):

    # generate L
    L = KE - PE
    L = sp.simplify(L)

    print("L:")
    sp.pprint(L)

    # generate non-conservative forces
    RHS = tau - b*theta.diff(t)

    # Euler legrange
    LHS = L.diff(thetad).diff(t) - L.diff(theta)

    full_eom = RHS - LHS
    result = sp.solve(full_eom, thetadd)
    thetadd_eom = sp.simplify(result[0])

    print("thetadd = ")
    sp.pprint(thetadd_eom)

    # generate eom file
    state = sp.Matrix([[theta], [thetad]])
    state_dot = sp.Matrix([[thetad], [thetadd_eom]])
    input = sp.Matrix([[tau]])
    eom = sp.lambdify([state, input, m, g, ell, b, k1, k2], state_dot, "numpy")

    if regenerate_eom_file:
        # write to file
        from case_studies import L_rodmass
        write_eom_to_file(state, input, [m, g, ell, b, k1, k2], L_rodmass, eom=state_dot)
        from case_studies.L_rodmass import eom_generated

        # test file
        import importlib
        importlib.reload(eom_generated)  # reload in case it was just generated/modified


        print("X_dot_func = ", eom(test_state, test_input, P.m, P.g, P.ell, P.b, P.k1, P.k2))

        param_vals = {
            "m": P.m,
            "g": P.g,
            "ell": P.ell,
            "b": P.b,
            "k1": P.k1,
            "k2": P.k2
        }

        x_dot_file_test = eom_generated.calculate_eom(test_state, test_input, **param_vals)
        print("\nx_dot_test from generated file = ", x_dot_file_test)

    return state, input, state_dot, eom

def generate_linearization(state, state_dot, input):
    theta_eq, tau_eq = sp.symbols("theta_eq, tau_eq")
    equilibrium = [(theta, theta_eq), (thetad, 0.0), (tau, tau_eq)]
    equilibrium_eq = sp.Equality(0.0, state_dot[1].subs(equilibrium))
    tau_eq_sol = sp.solve(equilibrium_eq, tau_eq)[0]
    tau_eq_sol = sp.simplify(tau_eq_sol)

    x_eq = sp.Matrix([[theta_eq], [0.0]])
    print("x_eq:")
    sp.pprint(x_eq)

    u_eq = sp.Matrix([[tau_eq_sol]])
    print("u_eq:")
    sp.pprint(u_eq)

    # find jacobians for linearization
    A = state_dot.jacobian(state)
    B = state_dot.jacobian(input)

    # sub in equilibrium
    equilibrium = [(theta, theta_eq), (thetad, 0.0), (tau, tau_eq_sol)]
    A_lin = sp.simplify(A.subs(equilibrium))
    B_lin = sp.simplify(B.subs(equilibrium))

    print("A_lin:")
    sp.pprint(A_lin)

    print("B_lin:")
    sp.pprint(B_lin)

    return A_lin, B_lin


def generate_transfer_function(A_lin, B_lin):
    # generate symbols for use
    s = sp.symbols("s")
    I = sp.eye(A_lin.shape[0])

    # generate C and D
    C = sp.Matrix(P.Cr)
    D = sp.Matrix([[0]]) # same # rows as C, # columns as u

    # generate transfer function H(s) = Y(s)/U(s) with theta_eq value subbed in
    H = C @ (s * I - A_lin).LUsolve(B_lin) + D
    H = H.subs({"theta_eq": 0.0})
    H = sp.simplify(H[0])
    print("H(s) (NOT MONIC):")
    sp.pprint(H)


def main():
    KE = generate_KE()
    PE = generate_PE()
    state, input, state_dot, eom = generate_eom(KE, PE, regenerate_eom_file=False)
    A_lin, B_lin = generate_linearization(state, state_dot, input)
    generate_transfer_function(A_lin, B_lin)

if __name__ == "__main__":
    main()