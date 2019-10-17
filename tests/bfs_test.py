import unittest
from env.intspace import IntSpace
from search.bfs import BFS


class BFSTest(unittest.TestCase):

    def test_visit_intspace(self):
        s = BFS(IntSpace())
        path = []
        for _ in range(16):
            path.append(s.next()[0])
        self.assertListEqual(path, [0,-10,-7,7,10,-20,-17,-3,-14,3,14,17,20,-30,-27,-13])
        self.assertListEqual(list(s.frontier), [-24, 4, -21, -4, 13, 21, 24, 27, 30, -40, -37, -23, -34, -6])
        self.assertListEqual(s.trace(-10), [0, -10])
        self.assertListEqual(s.trace(-3), [0, -10, -3])
        self.assertListEqual(s.trace(-24), [0, -10, -17, -24])
        self.assertListEqual(s.trace(-13), [0, -10, -20, -13])


if __name__ == '__main__':
    unittest.main()