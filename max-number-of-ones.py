import numpy as np


def find_pivot_row(column, i):
    # return np.argmax(column[i:] != 0) + i
    return np.argmax(column[i:] != False) + i


def clean_column(matrix, pivot_position):
    row_count, _ = matrix.shape
    # matrix[pivot_position[0]] /= matrix[pivot_position]

    for i in range(row_count):
        if i == pivot_position[0] or matrix[i, pivot_position[1]] == 0:
            continue
        # matrix[i] -= matrix[i, pivot_position[1]] * matrix[pivot_position[0]]
        matrix[i] ^= matrix[pivot_position[0]]


def gauss_jordan(coef_matrix, aug_matrix=None):
    augmented_matrix = (
        np.copy(coef_matrix)
        if aug_matrix is None
        else np.concatenate((coef_matrix, aug_matrix), axis=1)
    )
    row_cursor = 0
    row_count, col_count = augmented_matrix.shape
    for j in range(col_count):
        if row_cursor >= row_count:
            break
        i = find_pivot_row(augmented_matrix[:, j], row_cursor)
        if i != row_cursor:
            augmented_matrix[[row_cursor, i]] = augmented_matrix[[i, row_cursor]]
        if not augmented_matrix[(row_cursor, j)]:
            continue

        clean_column(augmented_matrix, (row_cursor, j))
        row_cursor += 1

    return augmented_matrix

# input must be in row reduced echelon form
def calc_rank(rref_matrix):
    rank = 0
    for _, row in enumerate(rref_matrix):
        leading_entry_index = np.argmax(row != False)
        rank += 1 if row[leading_entry_index] else 0
    return rank

n, m = list(map(int, input().split()))
A = []
for i in range(n):
    A.append(list(map(int, list(input()))))
A = np.asarray(A, dtype=bool)

if np.all(A == 0):
    # all zero matrix
    print(m*n)
else:
    min_num_of_zeros = calc_rank((gauss_jordan(A))) - 1
    print(m*n - min_num_of_zeros)
