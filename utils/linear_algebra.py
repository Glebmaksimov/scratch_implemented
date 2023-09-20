import numpy as np


def is_square(M):
    return len(M) == len(M[0])


def multiply(A, B):  # https://www.youtube.com/watch?v=sZxjuT1kUd0
    return (
        "Error A cols != B rows"
        if len(B) != len(A[0])
        else [
            [sum(r_i * c_i for r_i, c_i in zip(row, col)) for col in zip(*B)]
            for row in A
        ]
    )

def transpose(M):
    # return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]
    return list(map(list, zip(*M)))


def get_I(M):
    return (
        "Error A cols != B rows"
        if len(M) != len(M[0])
        else [[1 if i == j else 0 for j in range((len(M)))] for i in range((len(M)))]
    )


def is_invertable(M, M_i):
    return True if multiply(M, M_i) == multiply(M_i, M) == get_I(M) else False


def make_square(M):
    return multiply(transpose(M), M)


def make_diagonal(x):
    """Converts a vector into an diagonal matrix"""
    m = np.zeros((len(x), len(x)))
    for i in range(len(m[0])):
        m[i, i] = x[i]
    return m


# ======================================================= TO_DO =========================================================
# https://www.mathsisfun.com/algebra/matrix-inverse-minors-cofactors-adjugate.html


def get_determinant(M):
    if len(M) == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]

    determinant = 0
    for c in range(len(M)):
        determinant += (
            ((-1) ** c) * M[0][c] * get_determinant(get_minor(M, 0, c))
        )  # noqa
    return determinant


def get_minor(M, i, j):
    return [row[:j] + row[j + 1 :] for row in (M[:i] + M[i + 1 :])]


def inverse(M):  # https://www.youtube.com/watch?v=P3l7gGeHXC8
    determinant = get_determinant(M)

    if determinant != 0 and is_square(M):
        if len(M) == 2:
            return [
                [M[1][1] / determinant, -1 * M[0][1] / determinant],
                [-1 * M[1][0] / determinant, M[0][0] / determinant],
            ]

        cofactors = []
        for r in range(len(M)):
            cofactorRow = []
            for c in range(len(M)):
                minor = get_minor(M, r, c)
                cofactorRow.append(((-1) ** (r + c)) * get_determinant(minor))  # noqa
            cofactors.append(cofactorRow)
        cofactors = transpose(cofactors)
        for r in range(len(cofactors)):
            for c in range(len(cofactors)):
                cofactors[r][c] = cofactors[r][c] / determinant
        return cofactors

    return "Error!Matrix is not square or it`s determinant = 0"
