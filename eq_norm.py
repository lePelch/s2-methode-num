import numpy as np


def poly_approx_eq_norm(x, y, deg):
    # Validation
    if len(x) != len(y):
        raise ValueError(f"x et y pas la même taille")
    # Étape 1 calculer <phi i,phi j>
    M = deg + 1
    Phi = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            Phi[i, j] = sum(x ** (i + j))
    
    # Étape 2 calculer  <phi i,y>
    Psi = np.zeros((M, 1))
    for i in range(M):
        Psi[i] = sum(x**i * y)
    # A = Psi / Phi sur matlab
    A = np.linalg.solve(Phi, Psi)
    return A

def fct_from_coeff(A, xn):
    coef = 0
    fx = 0
    for a in A:
        fx += xn * a ** coef 
        coef += 1
    return fx

if __name__ == "__main__":
    xn = np.array([1, 3, 4, 6, 7])
    yn = np.array([-1.6, 4.8, 6.1, 14.6, 15.1])
    for d in range(1, len(xn) - 1):
        print(f"Le degrée de liberté est de {d}")
        A = poly_approx_eq_norm(xn, yn, d)
        print(f"Les coef sont de \n {A}")
        err = 0
        for x, y in zip(xn, yn):
            err += (fct_from_coeff(A, x) - y) ** 2
        print(f"L'erreur quadratique est de {err}")
        N = len(xn)
        RMSE = np.sqrt((1/N*err))
        print(f"Le RMSE est de {RMSE}")

    print("\n\n\n")
    for x, y in zip(xn, yn):
        print(f"({x}, {y})")


