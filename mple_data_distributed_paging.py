import argparse
from pyERGM.mple_optimizaiton import *

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

    num_nodes = observed_networks.shape[0]
    max_edge_idx = metric_collection.mask.sum() if metric_collection.mask is not None else num_nodes ** 2 - num_nodes
    edge_indices = (func_id * num_edges_per_job,
                    min((func_id + 1) * num_edges_per_job, max_edge_idx))
    Xs_chunk = metric_collection.prepare_mple_regressors(observed_network=None, edge_indices_lims=edge_indices)
    ys_chunk = metric_collection.prepare_mple_labels(observed_networks, edge_indices)

    chunks_out_path = os.path.join(out_dir_path, 'mple_data_paged_chunks')
    os.makedirs(chunks_out_path, exist_ok=True)

    np.savez_compressed(os.path.join(chunks_out_path, f'{func_id}.npz'), Xs_chunk=Xs_chunk, ys_chunk=ys_chunk)


if __name__ == "__main__":
    main()