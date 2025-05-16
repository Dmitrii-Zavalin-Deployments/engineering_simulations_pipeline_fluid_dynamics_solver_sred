import time
import multiprocessing as mp
import unittest

class TestParallelExecution(unittest.TestCase):
    def test_parallel_scaling_efficiency(self):
        """Ensure multi-core execution scales effectively"""
        num_cores = mp.cpu_count()

        start_single = time.time()
        solver.run_simulation(num_threads=1)
        single_core_time = time.time() - start_single

        start_parallel = time.time()
        solver.run_simulation(num_threads=num_cores)
        multi_core_time = time.time() - start_parallel

        efficiency_ratio = single_core_time / multi_core_time
        assert efficiency_ratio >= num_cores * 0.75, "Parallel efficiency too low!"

if __name__ == "__main__":
    unittest.main()



