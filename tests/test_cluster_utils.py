import unittest
from unittest.mock import patch, MagicMock, mock_open

from pyERGM.cluster_utils import *

class TestClusterUtils(unittest.TestCase):

    @patch("subprocess.run")
    def test_should_check_output_files_marks_false_for_running_jobs(self, mock_run):
        # Simulate bjobs output: JOBID USER STAT QUEUE ... JOB_NAME ...
        mock_run.return_value.stdout = (
            "87920 tomta RUN short login1 host *leNets[1] Aug 31 14:53\n"
            "87920 tomta DONE short login1 host *leNets[2] Aug 31 14:53\n"
        )

        res = should_check_output_files(["87920"], 2)

        # Index 0 should be False because job [1] is RUN
        # Index 1 should be True because job [2] is DONE
        expected = np.array([False, True])
        np.testing.assert_array_equal(res, expected)

    @patch("subprocess.run")
    def test_should_check_output_files_all_done(self, mock_run):
        mock_run.return_value.stdout = (
            "87920 tomta DONE short login1 host *leNets[1] Aug 31 14:53\n"
            "87920 tomta EXIT short login1 host *leNets[2] Aug 31 14:53\n"
        )

        res = should_check_output_files(["87920"], 2)

        # Both jobs finished → all True
        expected = np.array([True, True])
        np.testing.assert_array_equal(res, expected)

    @patch("subprocess.run")
    @patch("os.path.join", side_effect=lambda x,y,z : "ClusterScripts/distributed_logistic_regression.sh")
    def test_resend_failed_jobs_batches_and_parses(self,
                                                   mock_join,
                                                   mock_run):
        # Simulate subprocess returning stdout for bsub
        mock_run.side_effect = [
            MagicMock(stdout=b"Job <1234> is submitted to default queue.\n"),
            MagicMock(stdout=b"Job <7865> is submitted to default queue.\n"),
        ]


        ids = resend_failed_jobs(Path(""), list(range(150)), "arrayname")

        # It should split into 2 batches (LSF_ID_LIST_LEN_LIMIT = 5)
        self.assertEqual(len(ids), 2)
        self.assertEqual([1234, 7865], ids)


if __name__ == "__main__":
    unittest.main()