import unittest
from env.intspace import IntSpace
from search.greedy import Greedy
import heapq as pq


class RandomRolloutTest(unittest.TestCase):

    def test_visit_intspace(self):
        s = Greedy(IntSpace())
        path = []
        for _ in range(16):
            path.append(s.next()[0])
        self.assertListEqual(path, [0,-10,-7,7,10,-20,-17,-3,-27,-24,-34,-31,-14,-21,-4,-11])
        self.assertListEqual(s.trace(-34), [0, -10, -17, -24, -34])
        self.assertListEqual(s.trace(-11), [0, -10, -17, -24, -14, -4, -11])


if __name__ == '__main__':
    unittest.main()