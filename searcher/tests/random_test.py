import unittest
from env.intspace import IntSpace
from search.random import RandomRollout


class GreedyTest(unittest.TestCase):

    def test_visit_intspace(self):
        s = RandomRollout(IntSpace(), seed=1)
        path = []
        for _ in range(16): path.append(s.next()[0])
        self.assertListEqual(path, [0,-7,3,-4,-11,-21,-28,-35,-42,-32,-25,-18,-8,-1,6,16])

        s = RandomRollout(IntSpace(), seed=0, max_depth=3)
        path = []
        for _ in range(16): path.append(s.next()[0])
        self.assertListEqual(path, [0,10,17,7,0,7,17,24,0,10,17,24,0,7,17,10])
        self.assertListEqual(s.trace(7), [0, 7])
        self.assertListEqual(s.trace(10), [0, 10])
        self.assertListEqual(s.trace(17), [0, 10, 17])


        s = RandomRollout(IntSpace(), seed=48, max_depth=9)
        path = []
        for _ in range(50): path.append(s.next()[0])
        self.assertListEqual(s.trace(21), [0, 7, -3, 4, 14, 21])
        for _ in range(500): path.append(s.next()[0])
        self.assertListEqual(s.trace(21), [0, 7, 14, 21])





if __name__ == '__main__':
    unittest.main()