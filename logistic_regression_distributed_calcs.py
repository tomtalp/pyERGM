import argparse
from pyERGM.mple_optimizaiton import *


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir_path', type=str)
    parser.add_argument('--num_edges_per_job', type=int)
    parser.add_argument('--functions', type=str)
    parser.add_argument('--thetas', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_cmd_args()
    out_dir_path = args.out_dir_path
    num_edges_per_job = args.num_edges_per_job
    funcs_to_calc = args.functions.split(',')
    thetas = np.fromstring(args.thetas, sep=',')
    thetas = thetas[:, None]
    func_id = int(os.environ['LSB_JOBINDEX']) - 1
    with open(os.path.join(out_dir_path, 'data', 'metric_collection.pkl'), 'rb') as f:
        metric_collection = pickle.load(f)
    with open(os.path.join(out_dir_path, 'data', 'observed_networks.pkl'), 'rb') as f:
        observed_networks = pickle.load(f)

    # Get data chunk and predictions
    num_nodes = observed_networks.shape[0]

    edge_indices = (func_id * num_edges_per_job,
                    min((func_id + 1) * num_edges_per_job, num_nodes * num_nodes - num_nodes))
    Xs_chunk = metric_collection.prepare_mple_regressors(observed_network=None, edges_indices_lims=edge_indices)
    ys_chunk = metric_collection.prepare_mple_labels(observed_networks, edge_indices)
    chunk_prediction = calc_logistic_regression_predictions(Xs_chunk, thetas)

    for func_to_calc in funcs_to_calc:
        # Create outputs directory to store the calculated chunks
        chunks_dir_path = os.path.join(out_dir_path, func_to_calc)
        os.makedirs(chunks_dir_path, exist_ok=True)

        # Calculate the contributions of the chunk to the log-likelihood, the gradient and the hessian.
        if func_to_calc == 'predictions':
            func_chunk = chunk_prediction
        elif func_to_calc == 'log_likelihood':
            func_chunk = calc_logistic_regression_predictions_log_likelihood(chunk_prediction, ys_chunk)[0][0]
        elif func_to_calc == 'log_likelihood_gradient':
            func_chunk = calc_logistic_regression_log_likelihood_grad(Xs_chunk, chunk_prediction, ys_chunk)
        elif func_to_calc == 'log_likelihood_hessian':
            func_chunk = calc_logistic_regression_log_likelihood_hessian(Xs_chunk, chunk_prediction)
        else:
            raise ValueError(f'Unsupported function to calculate for logistic regression distributed '
                             f'optimization: {func_to_calc}. Possibilities are: '
                             f'predictions, log_likelihood, log_likelihood_gradient, log_likelihood_hessian')

        with open(os.path.join(chunks_dir_path, f'{func_id}.pkl'), 'wb') as f:
            pickle.dump(func_chunk, f)


if __name__ == "__main__":
    main()
