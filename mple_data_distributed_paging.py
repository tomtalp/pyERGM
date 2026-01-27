import argparse
from pyERGM.mple_optimization import *

def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir_path', type=str)
    parser.add_argument('--num_edges_per_job', type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_cmd_args()
    out_dir_path = args.out_dir_path
    num_edges_per_job = args.num_edges_per_job
    func_id = int(os.environ['LSB_JOBINDEX']) - 1
    with open(os.path.join(out_dir_path, 'data', 'metric_collection.pkl'), 'rb') as f:
        metric_collection = pickle.load(f)
    with open(os.path.join(out_dir_path, 'data', 'observed_networks.pkl'), 'rb') as f:
        observed_networks = pickle.load(f)

    sample_weights_path = os.path.join(out_dir_path, 'data', 'sample_weights.pkl')
    if os.path.exists(sample_weights_path):
        with open(sample_weights_path, 'rb') as f:
            sample_weights = pickle.load(f)
    else:
        sample_weights = None

    num_nodes = observed_networks.shape[0]
    max_edge_idx = metric_collection.mask.sum() if metric_collection.mask is not None else num_nodes ** 2 - num_nodes
    edge_indices = (func_id * num_edges_per_job,
                    min((func_id + 1) * num_edges_per_job, max_edge_idx))
    Xs_chunk = metric_collection.prepare_mple_regressors(observed_network=None, edge_indices_lims=edge_indices)
    ys_chunk = metric_collection.prepare_mple_labels(observed_networks, edge_indices)

    # Flatten and chunk sample weights using the same edge indexing as Xs/ys
    save_dict = dict(Xs_chunk=Xs_chunk, ys_chunk=ys_chunk)
    if sample_weights is not None:
        flat_weights = flatten_square_matrix_to_edge_list(sample_weights, metric_collection.is_directed)
        if metric_collection.mask is not None:
            global_mask = metric_collection.mask
        else:
            global_mask = np.ones(flat_weights.shape[0], dtype=bool)
        weights_chunk = flat_weights[np.where(global_mask)[0][edge_indices[0]:edge_indices[1]]].reshape(-1, 1)
        save_dict['weights_chunk'] = weights_chunk

    chunks_out_path = os.path.join(out_dir_path, 'mple_data_paged_chunks')
    os.makedirs(chunks_out_path, exist_ok=True)

    np.savez_compressed(os.path.join(chunks_out_path, f'{func_id}.npz'), **save_dict)


if __name__ == "__main__":
    main()