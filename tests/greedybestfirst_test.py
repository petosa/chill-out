import unittest
from env.intspace import IntSpace
from search.greedybestfirst import GreedyBestFirst
import heapq as pq

class GreedyBestFirstTest(unittest.TestCase):

    def test_unvisit_intspace(self):
        s = GreedyBestFirst(IntSpace(), visited_set=False)
        path = []
        for _ in range(17):
            path.append(s.next()[0])
        self.assertListEqual(path, [0,-10,-7,7,10,-20,-17,-3,0,-27,-24,-10,-7,-20,-17,-3,0])
        frontier = [pq.heappop(s.frontier)[1] for _ in range(len(s.frontier))]
        self.assertListEqual(frontier, [-17,-7,-7,-20,-20,-3,-3,-24,0,0,-27,7,10])

    def test_visit_intspace(self):
        s = GreedyBestFirst(IntSpace(), visited_set=True)
        path = []
        for _ in range(16):
            path.append(s.next()[0])
        self.assertListEqual(path, [0,-10,-7,7,10,-20,-17,-3,-27,-24,-14,3,-21,-4,-30,-13])
        frontier = [pq.heappop(s.frontier)[1] for _ in range(len(s.frontier))]
        self.assertListEqual(frontier, [-13,-4,-21,-3,-24,-27,3,-30,7,10])


if __name__ == '__main__':
    unittest.main()