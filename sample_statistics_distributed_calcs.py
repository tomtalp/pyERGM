import argparse
from utils import *


def parse_cmd_args(arg_list: list = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir_path', type=str)
    parser.add_argument('--num_samples_per_job', type=int)
    parser.add_argument('--p', type=float)
    if arg_list is not None:
        args = parser.parse_args(arg_list)
    else:
        args = parser.parse_args()
    return args


def main():
    args = parse_cmd_args()
    out_dir_path = args.out_dir_path
    num_samples_per_job = args.num_samples_per_job
    p = args.p
    func_id = int(os.environ['LSB_JOBINDEX']) - 1
    with open(os.path.join(out_dir_path, 'data', 'metric_collection.pkl'), 'rb') as f:
        metric_collection = pickle.load(f)

    sample_statistics = metric_collection.calc_statistics_for_binomial_tensor_local(num_samples_per_job, p=p)
    statistics_dir_path = os.path.join(out_dir_path, "sample_statistics")
    os.makedirs(statistics_dir_path, exist_ok=True)
    with open(os.path.join(statistics_dir_path, f'{func_id}.pkl'), 'wb') as f:
        pickle.dump(sample_statistics, f)
