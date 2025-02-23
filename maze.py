import random


def dfs_generate(n, seed=None):
    """Generates a random n by n maze using the DFS algorithm.

    Args:
        n: The number of cells. To make room for walls between cells, the map
           size is 2n + 1.
        seed: Random number generator seed to make the results reproducible.

    Returns
        A list of lists each of length 2n + 1. The element values are 1 for
        wall and 0 for open.
    """
    # Algorithm from:
    # https://www.cs.cmu.edu/~112-s23/notes/student-tp-guides/Mazes.pdf
    rng = random.Random(seed)
    # The initial map has walls at every index where the row or the column is
    # even and is open everywhere both the column and row are odd. Note:
    # indexing starts from 0.
    # e.g. n = 2
    # 1 1 1 1 1
    # 1 0 1 0 1
    # 1 1 1 1 1
    # 1 0 1 0 1
    # 1 1 1 1 1
    #
    # Write this as an expression using the bitwise and '&' and bitwise xor '^'
    # operators.
    # r & 1 - 1 for odd rows
    # c & 1 - 1 for odd columns
    # (r & 1) & (c & 1) - 1 for odd rows and columns
    # 1 ^ ((r & 1) & (c & 1)) - 0 for odd rows and columns. Bitwise ^ to flip
    #                           the value.
    # Simplify by making use of the fact that '&' is associative and 1 & 1 = 1
    # 1 ^ (r & c & 1) - 0 for odd rows and columns.
    maze_map = [[1 ^ (r & c & 1) for c in range(2 * n + 1)] for r in range(2 * n + 1)]

    # DFS uses recursive backtracking to tunnel through the cells until every
    # cell has been visited.
    visited = [[False for _ in range(n)] for _ in range(n)]

    def backtracker(x, y):
        visited[x][y] = True
        # Visit the neighbors in random order
        neighbors = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        rng.shuffle(neighbors)
        for dx, dy in neighbors:
            nx = x + dx
            ny = y + dy
            if nx < 0 or ny < 0 or nx >= n or ny >= n:
                # No neighbor
                continue
            if visited[nx][ny]:
                continue
            # Add a passage from the current location to the neighbor.
            maze_map[2 * x + 1 + dx][2 * y + 1 + dy] = 0
            backtracker(nx, ny)

    # Start from a random location
    backtracker(rng.randrange(n), rng.randrange(n))
    return maze_map
