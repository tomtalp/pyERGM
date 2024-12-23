import argparse
from pyERGM.utils import *


class StoreNpArr(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir_path', type=str)
    parser.add_argument('--num_edges_per_job', type=int)
    parser.add_argument('--function', type=str)
    # TODO: it seems like the parser doesn't know to deal with scientific format (e.g. -3.243718829521125e-05).
    parser.add_argument('--thetas', action=StoreNpArr, type=float, nargs='+')
    args = parser.parse_args()
    return args


def main():
    args = parse_cmd_args()
    out_dir_path = args.out_dir_path
    num_edges_per_job = args.num_edges_per_job
    func_to_calc = args.function
    thetas = args.thetas
    thetas = thetas[:, None]
    func_id = int(os.environ['LSB_JOBINDEX']) - 1
    with open(os.path.join(out_dir_path, 'data', 'metric_collection.pkl'), 'rb') as f:
        metric_collection = pickle.load(f)
    with open(os.path.join(out_dir_path, 'data', 'observed_network.pkl'), 'rb') as f:
        observed_network = pickle.load(f)

    # Get data chunk and predictions
    num_nodes = observed_network.shape[0]
    edge_indices = (func_id * num_edges_per_job,
                    min((func_id + 1) * num_edges_per_job, num_nodes * num_nodes - num_nodes))
    Xs_chunk, ys_chunk = metric_collection.prepare_mple_data(observed_network, edge_indices)
    chunk_prediction = calc_logistic_regression_predictions(Xs_chunk, thetas)

    # Create outputs directory to store the calculated chunks
    chunks_dir_path = os.path.join(out_dir_path, func_to_calc)
    os.makedirs(chunks_dir_path, exist_ok=True)

    # Calculate the contributions of the chunk to the log-likelihood, the gradient and the hessian.
    if func_to_calc == 'predictions':
        func_chunk = chunk_prediction
    elif func_to_calc == 'log_likelihood':
        func_chunk = calc_logistic_regression_predictions_log_likelihood(chunk_prediction, ys_chunk)
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
