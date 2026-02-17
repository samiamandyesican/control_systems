# %%
from .generate_linearization import *
from sympy.physics.control import TransferFunction

# define 's' as symbolic variable for the Laplace domain
s = sp.symbols("s")
I = sp.eye(A_lin.shape[0])

# %%
# define C and D
C = sp.Matrix([[1, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0]])
D = sp.Matrix([[0,0],
               [0,0],
               [0,0]])

# calc H(s)
H = C @ (s * I - A_lin).LUsolve(B_lin) + D

H = sp.simplify(H)

print("H(s) = ")
sp.pprint(sp.cancel(H))

# %%
# substituting for numerical values helps to simplify things.
H = H.subs([(g, P.g), (J_c, P.Jc), (m_c, P.mc), (m_r, P.mr), (d, P.d), (m_l, P.ml), (mu, P.mu)])
sp.pprint(sp.cancel(H))


# %%
# this is sufficient, but if we want to get the xfer function
# in a better form (monic form), we can do the following:
# Separate numerator and denominator

# for i in range(len(H)):
#     num, den = H[i, 0].as_numer_denom()

#     # Expand all paranetheses in denominator
#     expanded_den = sp.expand(den)

#     # Extract coefficient of highest-order term
#     order = sp.degree(expanded_den, s)
#     highest_order_term = expanded_den.coeff(s, order)  # Extract the s^2 term

#     # Divide through by the highest_order_term
#     num_monic = sp.simplify(num / highest_order_term)
#     den_monic = sp.collect(sp.simplify(expanded_den / highest_order_term), s)

#     # we don't need this "TransferFunction" object, but it helps
#     # to print things properly without undoing the term rearrangement.
#     cancel_terms = sp.cancel(num_monic / den_monic)
#     tf_monic = TransferFunction(
#         cancel_terms.as_numer_denom()[0], cancel_terms.as_numer_denom()[1], s
#     )
#     sp.pprint((tf_monic))


# Iterate through each output (row) and each input (column)
for i in range(H.rows):
    for j in range(H.cols):
        # Access element at row i, column j
        element = H[i, j]
        
        # Skip zero elements to avoid errors
        if element == 0:
            print(f"Transfer Function H_{i+1}{j+1} is 0")
            continue

        num, den = element.as_numer_denom()

        # ... (rest of your monic logic)
        expanded_den = sp.expand(den)
        order = sp.degree(expanded_den, s)
        highest_order_term = expanded_den.coeff(s, order)

        num_monic = sp.simplify(num / highest_order_term)
        den_monic = sp.collect(sp.simplify(expanded_den / highest_order_term), s)

        cancel_terms = sp.cancel(num_monic / den_monic)
        tf_monic = TransferFunction(
            cancel_terms.as_numer_denom()[0], cancel_terms.as_numer_denom()[1], s
        )
        
        print(f"\n--- Output {i+1}, Input {j+1} ---")
        sp.pprint(tf_monic)

# %%
