import unittest
from env.intspace import IntSpace
from search.bestfirst import BestFirst
import heapq as pq


class BestFirstTest(unittest.TestCase):

    def test_visit_intspace(self):
        s = BestFirst(IntSpace())
        path = []
        for _ in range(16):
            path.append(s.next()[0])
        self.assertListEqual(path, [0,-10,-7,7,10,-20,-17,-3,-27,-24,-14,3,-21,-4,-30,-13])
        frontier = [pq.heappop(s.frontier)[1] for _ in range(len(s.frontier))]
        self.assertListEqual(frontier, [-13,-4,-21,-3,-24,-27,3,-30,7,10])
        self.assertListEqual(s.trace(-10), [0, -10])
        self.assertListEqual(s.trace(-3), [0, -10, -3])
        self.assertListEqual(s.trace(-24), [0, -10, -17, -24])
        self.assertListEqual(s.trace(-13), [0, -10, -20, -13])


if __name__ == '__main__':
    unittest.main()