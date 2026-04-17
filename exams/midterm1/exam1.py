import numpy as np
import sympy as sp



def Q2():
    m1, m2, r, k, tau, t, J1 = sp.symbols("m1, m2, r, k, tau, t, J1")
    theta_f, phi_f = sp.symbols("theta_f phi_f", cls=sp.Function)
    theta = theta_f(t)
    phi = phi_f(t)

    q = sp.Matrix([[theta], [phi]])
    thetadot = theta.diff(t)
    phidot = phi.diff(t)
    qdot = sp.Matrix([[thetadot], [phidot]])

    # kinetic energy of disk
    KE_D = sp.Rational(1, 2) * J1 * thetadot**2

    # velocity from position
    p_b = sp.Matrix([[r*sp.cos(theta + phi)], [r*sp.sin(theta + phi)], [0]])
    v_b = p_b.diff(t)

    # sum of kinetic energies
    KE_b = sp.Rational(1,2) * m2 * v_b.T @ v_b
    KE = KE_D + sp.simplify(KE_b)[0]

    return KE


def Q7():
    m1, m2, ell, k1, F, t = sp.symbols("m1, m2, ell, k1, F, t")
    x1_f, theta_f = sp.symbols("x1_f, theta_f", cls=sp.Function)
    x1 = x1_f(t)
    theta = theta_f(t)
    p_m2 = sp.Matrix([[ell*sp.cos(theta + sp.pi/2)], [ell*sp.sin(theta + sp.pi/2) + x1], [0]])
    v_m2 = p_m2.diff(t)

    KE_1 = sp.Rational(1, 2)*m1*x1.diff(t)**2 
    KE_2 = sp.Rational(1, 2) * m2 * v_m2.T @ v_m2
    KE = KE_1 + KE_2[0]
    KE = sp.simplify(KE)

    sp.pprint(KE)

def Q8():

    Ja, mc, ell, g, t = sp.symbols("Ja, mc, ell, g, t")
    theta_f, x_f = sp.symbols("theta, x", cls=sp.Function)
    theta = theta_f(t)
    x = x_f(t)


    thetadot = theta.diff(t)
    xdot = x.diff(t)
    L = 0.5*Ja*thetadot**2 + 0.5*mc*xdot**2 + mc*ell*thetadot*xdot - mc*g*ell*sp.sin(theta)

    q = sp.Matrix([[theta], [x]])
    qdot = q.diff(t)

    L = sp.diff(sp.diff(L, qdot), t) - sp.diff(L, q)
    L = sp.simplify(L)

    sp.pprint(L)

def Q9():
    F, tau, t = sp.symbols("F, tau, t")
    theta_f, z_f = sp.symbols("theta, z", cls=sp.Function)
    theta = theta_f(t)
    z = z_f(t)

    thetadot = theta.diff(t)
    zdot = z.diff(t)
    zddot = 3*F - 1.5*theta - 2.5*thetadot - 2*zdot - 4*z
    thetaddot = 1.5*tau + 0.5*F - 0.25*zdot + 3.2*theta

    x = sp.Matrix([[z], [theta], [zdot], [thetadot]])
    inputs = sp.Matrix([[F], [tau]])

    xdot = sp.Matrix([
        [zdot],
        [thetadot],
        [zddot],
        [thetaddot]
    ])
    
    A = xdot.jacobian(x)
    B = xdot.jacobian(inputs)

    sp.pprint(sp.simplify(A))
    sp.pprint(sp.simplify(B))


def Q11():
    L, R, m, k, g, t, V = sp.symbols("L, R, m, k, g, t, V")
    i_f, z_f = sp.symbols("i, z", cls=sp.Function)
    i = i_f(t)
    z = z_f(t)

    zdot = z.diff(t)
    x = sp.Matrix([[i], [zdot], [z]])
    u = sp.Matrix([[V]])

    xdot = sp.Matrix([
        [(V - R*i) / L],
        [k/m*(i/z)**2 -g],
        [zdot]
    ])

    A = xdot.jacobian(x)
    B = xdot.jacobian(u)

    z0, i0, V0 = sp.symbols("z_0, i_0, V_0")
    equilibrium_vals = {
        z: z0,
        i: i0,
        V: V0,
    }
    A_lin = A.subs(equilibrium_vals)
    B_lin = B.subs(equilibrium_vals)

    A_lin = sp.simplify(A_lin)
    B_lin = sp.simplify(B_lin)

    sp.pprint(A)
    sp.pprint(B)


def Q12():
    G1, G2, G3, G4, G5, G6, E1, E2, E3, Y, Yr, U = sp.symbols("G_1, G_2, G_3, G_4, G_5, G_6, E_1, E_2, E_3, Y, Y_r, U")
    Y = U*G1
    U = E2*G4 + E2*G5
    E2 = E1*G2 - E3*G3
    E1 = Yr - E3
    E3 = Y*G6
    H = Y/Yr

    Y = (E2*G4 + E2*G5) * G1
    Y = ((E1*G2 - E3*G3)*G4 + (E1*G2 - E3*G3)*G5) * G1
    Y = (((Yr - E3)*G2 - (Y*G6)*G3)*G4 + ((Yr - E3)*G2 - (Y*G6)*G3)*G5) * G1
    Y = (((Yr - (Y*G6))*G2 - (Y*G6)*G3)*G4 + ((Yr - (Y*G6))*G2 - (Y*G6)*G3)*G5) * G1
    expr = (((Yr - (Y*G6))*G2 - (Y*G6)*G3)*G4 + ((Yr - (Y*G6))*G2 - (Y*G6)*G3)*G5) * G1 - Y
    Y = sp.solve(expr, Y)[0]
    H = Y/Yr

    sp.pprint(sp.simplify(H))

def Q13():
    A = sp.Matrix([
        [0.4, 0.6],
        [1, -0.2]
    ])

    B = sp.Matrix([
        [0.5],
        [0]
    ])

    C = sp.Matrix([
        [1, 0],
        [0, 1]
    ])

    D = sp.Matrix([
        [0],
        [0]
    ])

    s = sp.symbols("s")
    I = sp.eye(A.shape[0])
    H = C @ (s*I - A).LUsolve(B) + D

    sp.pprint(sp.simplify(H))

    print(-0.4 + 0.2)
    print(-0.4*0.2-0.6)

def Q16():
    eq = (-2 + np.sqrt(4 - 4*6*(-9))) / (2*6)
    print(eq)

def Q17():
    zeta = 0.95
    tr = 1.5

    wn = np.pi / (2*tr*np.sqrt(1-zeta**2))

    print("Q17")
    print(wn**2)
    print(2*zeta*wn)

    b0 = -3/6
    a1 = 2/6
    a0 = -9/6
    alpha0 = wn**2
    alpha1 = 2*zeta*wn
    kp = (alpha0 - a0) / b0
    kd = (alpha1 - a1) / b0

    print("Q18")
    print(kp)
    # print(17/8)
    print(kd)

    b0 = -3/6
    a1 = 2/6
    a0 = -9/6
    alpha0 = wn**2
    alpha1 = 2*zeta*wn
    kp = (alpha0 - a0) / b0
    kd = (alpha1 - a1) / b0

    kp = -15
    kd = -9 + 1/3

    print("Q19")
    print(b0*kp / (a0 + b0*kp))

    b0 = 8
    a1 = 4
    a0 = -5
    alpha0 = wn**2
    alpha1 = 2*zeta*wn
    kp = (alpha0 - a0) / b0
    kd = (alpha1 - a1) / b0


    kp = -15
    kd = -9 - 1/3
    print("Q20")
    print((-(a1+b0*kd) + np.sqrt(-((a1+b0*kd)**2 - 4*(a0+b0*kp)))) / 2)
    print(np.sqrt(-((a1+b0*kd)**2 - 4*(a0+b0*kp))))

def Q20():
    print("Q20")

    # givens
    kp = -15
    kd = -9.333333

    # from problem 19
    b0 = -1/2
    a1 = 1/3
    a0 = -3/2

    # quadratic formula for closed loop t.f.
    a = 1
    b = a1 + b0*kd
    c = a0 + b0*kp

    # separate out radical in case imaginary
    real = -b / (2*a)
    rad = np.sqrt(b**2 - 4*a*c) / (2*a)
    print(real+rad)
    print(real-rad)

def main():
    # KE = Q2()
    # Q7()
    # Q8()
    # Q9()
    # Q11()
    # Q12()
    # Q13()
    # Q16()
    # Q17()
    Q20()

if __name__ == "__main__":
    main()