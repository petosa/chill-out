import unittest
from env.intspace import IntSpace
from search.beam import Beam
import heapq as pq


class BeamTest(unittest.TestCase):

    def test_visit_intspace(self):
        s = Beam(IntSpace())
        path = []
        for _ in range(24):
            path.append(s.next()[0])
        my_path = ([
            0,-10,-7,7,10,-20,-17,-3,-14,3,14,17,
            -24,-21,-4,-27,-30,-13,-23,-6,-11,6,-31,-28
        ])
        self.assertListEqual(path, my_path)
        frontier = [pq.heappop(s.frontier)[1] for _ in range(len(s.frontier))]
        self.assertListEqual(s.trace(-10), [0, -10])
        self.assertListEqual(s.trace(-3), [0, -10, -3])
        self.assertListEqual(s.trace(-24), [0, -7, -14, -24])
        self.assertListEqual(s.trace(-13), [0, -10, -20, -13])
        self.assertListEqual(frontier, [-11,-6,-23,-28,6,-31])
        self.assertListEqual(s.trace(-21), [0, -7, -14, -21])
        self.assertListEqual(s.trace(-11), [0, -7, -14, -4, -11])




if __name__ == '__main__':
    unittest.main()