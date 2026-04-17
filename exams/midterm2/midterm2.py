import numpy as np
import sympy as sp

def Q2to5():
    print("Q 2 thru 5 ==========================")
    s = sp.symbols("s")
    P = 4 / (s**2 + 2*s - 3)
    C = (5*s**2 + 6*s + 7) / (s*(0.01*s + 1))
    input3 = 3/s**1
    e_ss_i = 1/(1+P*C)*input3

    print("e_ss_i")
    sp.pprint(sp.simplify(e_ss_i))

    input5 = 2/1 # 2/s
    e_ss_d = P/(1+P*C)*input5
    
    print("e_ss_d:")
    sp.pprint(sp.simplify(e_ss_d))

def Q6():
    print("Q 6 =================================")
    s, k_i, Y, R = sp.symbols("s, k_i, Y, R")
    P = 2 / (s*(s+1))

    E = R - Y
    U = E*k_i/s + 3*E - 2*s*Y
    equation = sp.Equality(Y, U*P)
    
    sol = sp.solve(equation, Y)
    print("transfer function:")
    sp.pprint(sp.simplify(sol[0]))

def Q7():
    print("Q 7 =================================")
    s = sp.symbols("s")
    A = sp.Matrix([[1, 2, -4, -8],
                   [1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
    I = sp.eye(A.shape[0])

    char_eq = sp.det(s*I - A)
    print("characteristic equation:")
    sp.pprint(char_eq)

def Q9():
    print("Q 9 =========================================")
    C_AB = np.array([[1, -1.5, 1],
                     [0.5, -0.75, 0.5],
                     [2, -3, 3]])
    print("det(C_AB):")
    print(np.linalg.det(C_AB))

def Q10to11():
    print("Q 10 thru 11 =============================")
    A = np.array([[1, 2],
                  [0, 1]])
    B = np.array([[1], [-1]])

    C_AB = np.block([B, A@B])
    
    print("C_AB:")
    print(C_AB)

def Q12to15():
    print("Q 12 thru 15 =============================")
    s = sp.symbols("s")
    A = sp.Matrix([[-1, 4],
                   [1, 0]])
    B = sp.Matrix([[1],
                   [0]])
    C = sp.Matrix([[1, -1]])
    I = sp.eye(A.shape[0])

    # ol_char = sp.det(s*I - A)
    # root_eq = sp.Equality(ol_char, 0)
    # roots = sp.solve(root_eq, s)
    # sp.pprint(ol_char)
    # sp.pprint(roots)

    K = sp.Matrix([[2, 5]])
    cl_char = sp.det(s*I - (A-B@K))
    root_eq = sp.Equality(cl_char, 0)
    roots = sp.solve(root_eq, s)
    sp.pprint(roots)

    k_r = -sp.inv_quick(C @ sp.inv_quick(A - B@K) @ B)
    print("k_r:")
    sp.pprint(k_r)

    O_AC = sp.Matrix([[C], [C@A]])
    print("O_AC:")
    sp.pprint(O_AC)

def Q17():
    print("Q 17 =================================")
    s, k_i, R, Y, K = sp.symbols("s, k_i, R, Y, K")
    P = -1 / (s+5)
    E = R - Y
    U = -E*k_i/s - K*Y
    eqn = sp.Equality(Y, U*P)
    sol = sp.solve(eqn, Y)[0]

    print("transfer function Y(s):")
    sp.pprint(sp.simplify(sol))

    den = s**2 + (5-K)*s + k_i
    root_eqn = sp.Equality(den, 0)
    roots = sp.solve(root_eqn, s)

    gains = {K: 2, k_i: 2, R:1}
    
    print("roots:")
    sp.pprint(roots[0].subs(gains))
    sp.pprint(roots[1].subs(gains))

def Q19():
    print("Q 19 ==================================")
    A = np.array([[-1, 2],
                  [1, 0]])
    B = np.array([[1, 0],
                  [0, 1]])
    C = np.array([[1, -1]])

    inputs = B.shape[1]
    A2 = np.block([[A, B],
                   [np.zeros((inputs, A.shape[1])), np.zeros((inputs, inputs))]])
    C2 = np.block([[C, np.zeros((C.shape[0], inputs))]])
    O_A2C2 = np.block([[C2], [C2@A2], [C2@A2@A2], [C2@A2@A2@A2]])

    print(f"{A2 = }")
    print(f"{C2 = }")
    print(f"{O_A2C2 = }")

def main():
    # Q2to5()
    # Q6()
    # Q7()
    # Q9()
    # Q10to11()
    # Q12to15()
    # Q17()
    Q19()

if __name__ == "__main__":
    main()