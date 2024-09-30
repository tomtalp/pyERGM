import argparse
import numpy as np
import os
import pickle
from utils import *


class StoreNpArr(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)


def parse_cmd_args(arg_list: list = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir_path', type=str)
    parser.add_argument('--function', type=str)
    parser.add_argument('--num_edges_per_job', type=int)
    parser.add_argument('--thetas', action=StoreNpArr, type=float, nargs='+')
    if arg_list is not None:
        args = parser.parse_args(arg_list)
    else:
        args = parser.parse_args()
    return args


def main():
    args = parse_cmd_args()
    out_dir_path = args.out_dir_path
    num_edges_per_job = args.num_edges_per_job
    thetas = args.thetas
    func_id = int(os.environ['LSB_JOBINDEX']) - 1
    with open(os.path.join(out_dir_path, 'data', 'metric_collection.pkl'), 'rb') as f:
        metric_collection = pickle.load(f)
    with open(os.path.join(out_dir_path, 'data', 'observed_network.pkl'), 'rb') as f:
        observed_network = pickle.load(f)

    # Get data chunk
    edge_indices = (func_id * num_edges_per_job, (func_id + 1) * num_edges_per_job)
    Xs_chunk, ys_chunk = metric_collection.prepare_mple_data(observed_network, edge_indices)

    # Calculate a chunk of the predictions and store it
    chunk_prediction = calc_logistic_regression_predictions(Xs_chunk, thetas)
    predictions_dir_path = os.path.join(out_dir_path, "prediction")
    os.makedirs(predictions_dir_path, exist_ok=True)
    with open(os.path.join(predictions_dir_path, f'{func_id}.pkl'), 'wb') as f:
        pickle.dump(chunk_prediction, f)

    # Calculate the contributions of the chunk to the log-likelihood, the gradient and the hessian.
    chunk_log_like = calc_logistic_regression_predictions_log_likelihood(chunk_prediction, ys_chunk)
    log_likes_dir_path = os.path.join(out_dir_path, "log_like")
    os.makedirs(log_likes_dir_path, exist_ok=True)
    with open(os.path.join(log_likes_dir_path, f'{func_id}.pkl'), 'wb') as f:
        pickle.dump(chunk_log_like, f)
    chunk_grad = calc_logistic_regression_log_likelihood_grad(Xs_chunk, chunk_prediction, ys_chunk)
    grad_dir_path = os.path.join(out_dir_path, "grad")
    os.makedirs(grad_dir_path, exist_ok=True)
    with open(os.path.join(grad_dir_path, f'{func_id}.pkl'), 'wb') as f:
        pickle.dump(chunk_grad, f)
    chunk_hessian = calc_logistic_regression_log_likelihood_hessian(Xs_chunk, chunk_prediction)
    hessian_dir_path = os.path.join(out_dir_path, "hessian")
    os.makedirs(hessian_dir_path, exist_ok=True)
    with open(os.path.join(hessian_dir_path, f'{func_id}.pkl'), 'wb') as f:
        pickle.dump(chunk_hessian, f)


if __name__ == "__main__":
    main()
