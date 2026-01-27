import argparse
from pyERGM.mple_optimization import *


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir_path', type=str)
    parser.add_argument('--functions', type=str)
    parser.add_argument('--thetas', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_cmd_args()
    out_dir_path = args.out_dir_path
    funcs_to_calc = args.functions.split(',')
    thetas = np.fromstring(args.thetas, sep=',')
    thetas = thetas[:, None].astype(np.float32)
    func_id = int(os.environ['LSB_JOBINDEX']) - 1

    mple_data_chunk = np.load(os.path.join(out_dir_path, 'mple_data_paged_chunks', f'{func_id}.npz'))
    Xs_chunk = mple_data_chunk['Xs_chunk'].astype(np.float32)
    ys_chunk = mple_data_chunk['ys_chunk'].astype(np.float32)
    if 'weights_chunk' in mple_data_chunk:
        weights_chunk = mple_data_chunk['weights_chunk'].astype(np.float32)
    else:
        weights_chunk = np.empty(0)
    mple_data_chunk.close()

    chunk_prediction = calc_logistic_regression_predictions(Xs_chunk, thetas)

    for func_to_calc in funcs_to_calc:
        # Create outputs directory to store the calculated chunks
        log_res_calcs_dir_path = os.path.join(out_dir_path, func_to_calc)
        os.makedirs(log_res_calcs_dir_path, exist_ok=True)

        # Calculate the contributions of the chunk to the log-likelihood, the gradient and the hessian.
        if func_to_calc == 'predictions':
            func_chunk = chunk_prediction
        elif func_to_calc == 'log_likelihood':
            func_chunk = calc_logistic_regression_predictions_log_likelihood(
                chunk_prediction, ys_chunk, sample_weights=weights_chunk)[0][0]
        elif func_to_calc == 'log_likelihood_gradient':
            func_chunk = calc_logistic_regression_log_likelihood_grad(
                Xs_chunk, chunk_prediction, ys_chunk, sample_weights=weights_chunk)
        elif func_to_calc == 'log_likelihood_hessian':
            func_chunk = calc_logistic_regression_log_likelihood_hessian(
                Xs_chunk, chunk_prediction, sample_weights=weights_chunk)
        else:
            raise ValueError(f'Unsupported function to calculate for logistic regression distributed '
                             f'optimization: {func_to_calc}. Possibilities are: '
                             f'predictions, log_likelihood, log_likelihood_gradient, log_likelihood_hessian')

        with open(os.path.join(log_res_calcs_dir_path, f'{func_id}.pkl'), 'wb') as f:
            pickle.dump(func_chunk, f)


if __name__ == "__main__":
    main()
