import copy
import pathlib


def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    a_matrix = []
    b_vector = []

    with path.open('r') as file:
        lines = file.readlines()

        for line in lines:
            equation = line.replace(' ', '').split('=')

            terms = equation[0]
            constant = int(equation[1])
            coefficients = []

            for variable in ['x', 'y', 'z']:
                parts = terms.split(variable)

                coefficient = parts[0]
                terms = parts[1]

                if coefficient == '' or coefficient == '+':
                    coefficient_value = 1
                elif coefficient == '-':
                    coefficient_value = -1
                else:
                    coefficient_value = int(coefficient)

                coefficients.append(coefficient_value)

            a_matrix.append(coefficients)
            b_vector.append(constant)

    return a_matrix, b_vector


def determinant(matrix: list[list[float]]) -> float:
    term1 = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
    term2 = matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
    term3 = matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])

    return term1 - term2 + term3


def trace(matrix: list[list[float]]) -> float:
    return matrix[0][0] + matrix[1][1] + matrix[2][2]


def norm(vector: list[float]) -> float:
    return (vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2) ** 0.5


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[0][0], matrix[1][0], matrix[2][0]],
            [matrix[0][1], matrix[1][1], matrix[2][1]],
            [matrix[0][2], matrix[1][2], matrix[2][2]]]


def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    result = []

    for row in matrix:

        dot_product = 0
        for i in range(len(vector)):
            dot_product += row[i] * vector[i]

        result.append(dot_product)

    return result


def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    result = []

    det_matrix = determinant(matrix)

    for i in range(0, 3):

        new_matrix = copy.deepcopy(matrix)
        col_index = i

        for j in range(0, 3):
            new_matrix[j][col_index] = vector[j]

        result.append(determinant(new_matrix) / det_matrix)

    return result


def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    result = []
    for i1 in range(0, 3):
        if i1 == i:
            continue

        row = []

        for j1 in range(0, 3):
            if j1 == j:
                continue

            row.append(matrix[i1][j1])

        result.append(row)

    return result


def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    result = []

    for i in range(0, 3):
        row = []

        for j in range(0, 3):
            factor = (-1) ** (i + j)  #1 if i+j % 2 == 0 else -1
            minor_matrix = minor(matrix, i, j)
            det_minor = minor_matrix[0][0] * minor_matrix[1][1] - minor_matrix[0][1] * minor_matrix[1][0]

            row.append(factor * det_minor)

        result.append(row)

    return result


def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return transpose(cofactor(matrix))


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det_matrix = determinant(matrix)
    result = multiply(adjoint(matrix), vector)
    return [result[0] / det_matrix, result[1] / det_matrix, result[2] / det_matrix]


if __name__ == '__main__':
    A, B = load_system(pathlib.Path("system.txt"))
    print(f"{A=} {B=}")

    print(f"{determinant(A)=}")
    print(f"{trace(A)=}")
    print(f"{norm(B)=}")

    print(f"{transpose(A)=}")
    print(f"{multiply(A, B)=}")

    print(f"{solve_cramer(A, B)=}")
    print(f"{solve(A, B)=}")
