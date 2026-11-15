
# N-Queens using Backtracking

N = int(input("Enter value of N: "))

board = [-1] * N

row = int(input("Enter first queen row: ")) - 1
col = int(input("Enter first queen column: ")) - 1

if row < 0 or row >= N or col < 0 or col >= N:
    print("Invalid position")

else:
    board[row] = col
    first_row = row

    # Check safe position
    def is_safe(row, col):
        for r in range(N):
            if board[r] != -1:
                if board[r] == col or abs(board[r] - col) == abs(r - row):
                    return False
        return True

    # Solve using backtracking
    def solve(row):
        if row == N:
            return True

        if row == first_row:
            return solve(row + 1)

        for col in range(N):
            if is_safe(row, col):
                board[row] = col

                if solve(row + 1):
                    return True

                board[row] = -1

        return False

    if solve(0):
        print("\nSolution:")
        for i in range(N):
            for j in range(N):
                if board[i] == j:
                    print("Q", end=" ")
                else:
                    print(".", end=" ")
            print()
    else:
        print("No solution exists")
