import maze
import unittest


class TestDfsGenerate(unittest.TestCase):

    def test_generates_a_maze(self):
        # There is a unique maze for n = 1
        self.assertEqual(maze.dfs_generate(1), [[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    def test_is_random(self):
        a = maze.dfs_generate(20, seed=123)
        b = maze.dfs_generate(20, seed=456)
        self.assertNotEqual(a, b)

    def test_is_reproducible(self):
        a = maze.dfs_generate(20, seed=123)
        b = maze.dfs_generate(20, seed=123)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
