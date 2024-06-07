# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9981785
# No pseudocode or github

from munkres import Munkres, print_matrix

def LNS_PBS(case_requests: dict, robots: list):
    matrix = [[5, 9, 1],
              [10, 3, 2],
              [8, 7, 4]]
    m = Munkres()
    indexes = m.compute(matrix)
    print_matrix(matrix, msg='Lowest cost through this matrix:')
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        print(f'({row}, {column}) -> {value}')
    print(f'total cost: {total}')


if __name__ == "__main__":
    LNS_PBS({}, [])