import unittest
from env.intspace import IntSpace
from search.beam import Beam
import heapq as pq

class BeamTest(unittest.TestCase):

    def test_unvisit_intspace(self):
        s = Beam(IntSpace(), visited_set=False)
        path = []
        for _ in range(53):
            path.append(s.next()[0])
        my_path = ([
            0,-10,-7,7,10,-20,-17,-3,0,-17,-14,0,3,-3,0,14,
            17,-24,-21,-7,-4,-27,-24,-10,-7,-27,-24,-10,-7,
            -20,-17,-3,0,-20,-17,-3,0,-17,-14,0,3,-24,-21,-7,-4,
            -27,-24,-10,-7,-27,-24,-10,-7
        ])
        self.assertListEqual(path, my_path)
        frontier = [pq.heappop(s.frontier)[1] for _ in range(len(s.frontier))]
        self.assertListEqual(frontier, [-10,-10,-7,-7,-7,-4,-21,-24,-24,-24,-27,-27])


    def test_visit_intspace(self):
        s = Beam(IntSpace(), visited_set=True)
        path = []
        for _ in range(24):
            path.append(s.next()[0])
        my_path = ([
            0,-10,-7,7,10,-20,-17,-3,-14,3,14,17,
            -24,-21,-4,-27,-30,-13,-23,-6,-11,6,-31,-28

        ])
        self.assertListEqual(path, my_path)
        frontier = [pq.heappop(s.frontier)[1] for _ in range(len(s.frontier))]
        self.assertListEqual(frontier, [-11,-6,-23,-28,6,-31])




if __name__ == '__main__':
    unittest.main()