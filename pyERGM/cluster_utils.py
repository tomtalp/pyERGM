import os
import numpy as np
from pathlib import Path
from typing import Sequence
import pickle
import subprocess
import numpy.typing as npt
import shutil
import time


LSF_ID_LIST_LEN_LIMIT = 100

def construct_single_batch_bash_cmd_logistic_regression(out_path, cur_thetas, funcs_to_calculate, num_edges_per_job):
    # Construct a string with the current thetas, to pass using the command line to children jobs.
    thetas_str = ''
    for t in cur_thetas:
        thetas_str += f'{t[0]},'
    thetas_str = thetas_str[:-1]

    fns_str = ",".join(funcs_to_calculate)

    cmd_line_for_bsub = (f'python ./logistic_regression_distributed_calcs.py '
                         f'--out_dir_path={out_path} '
                         f'--num_edges_per_job={num_edges_per_job} '
                         f'--functions={fns_str} '
                         f'--thetas={thetas_str}')
    return cmd_line_for_bsub


def wait_for_distributed_children_outputs(num_jobs: int, out_paths: Sequence[Path], job_array_ids: list,
                                          array_name: str):
    children_statuses = np.zeros(num_jobs, dtype=bool)
    while not children_statuses.all():
        should_check_out_files = should_check_output_files(job_array_ids, num_jobs)
        children_to_resend = []
        for i in np.where(should_check_out_files)[0]:
            is_done = True
            for out_path in out_paths:
                if not os.path.exists(os.path.join(out_path, f'{i}.pkl')):
                    is_done = False
                    break
            children_statuses[i] = is_done
            if not is_done:
                children_to_resend.append(i + 1)
        if children_to_resend:
            resent_job_array_ids = resend_failed_jobs(out_paths[0].parent, children_to_resend, array_name)
            job_array_ids += resent_job_array_ids
        time.sleep(60)


def sum_children_jobs_outputs(num_jobs: int, out_path: Path):
    measure = None
    for j in range(num_jobs):
        with open(os.path.join(out_path, f'{j}.pkl'), 'rb') as f:
            content = pickle.load(f)
            if measure is None:
                measure = content
            else:
                measure += content
    return measure


def cat_children_jobs_outputs(num_jobs: int, out_path: Path, axis: int = 0):
    measure = None
    for j in range(num_jobs):
        with open(os.path.join(out_path, f'{j}.pkl'), 'rb') as f:
            content = pickle.load(f)
            if measure is None:
                measure = content
            else:
                measure = np.concatenate((measure, content), axis=axis)
    return measure


def should_check_output_files(job_array_ids: list, num_sent_jobs: int) -> npt.NDArray[np.bool_]:
    should_check_out_files = np.ones(num_sent_jobs, dtype=bool)
    grep_pattern = r"\s|".join([str(jid) for jid in job_array_ids]) + r"\s"
    cmd = f'bjobs -o "jobid stat job_name" -noheader | grep -E "{grep_pattern}"'

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    for i, line in enumerate(lines):
        parts = line.split(None, 2)
        jobid, stat, job_name = parts

        idx = int(job_name[job_name.rfind("[") + 1:job_name.rfind("]")])

        if stat not in {"DONE", "EXIT"}:
            should_check_out_files[idx - 1] = False
    return should_check_out_files


def resend_failed_jobs(out_path: Path, job_indices: list, array_name: str) -> list:
    num_failed_jobs = len(job_indices)
    job_array_ids = []

    for i in range(num_failed_jobs // LSF_ID_LIST_LEN_LIMIT + 1):
        cur_job_indices_str = ''
        for j_idx_in_list in range(i * LSF_ID_LIST_LEN_LIMIT, min((i + 1) * LSF_ID_LIST_LEN_LIMIT, num_failed_jobs)):
            cur_job_indices_str += f'{job_indices[j_idx_in_list]},'
        cur_job_indices_str = cur_job_indices_str[:-1]
        single_batch_bash_path = os.path.join(out_path, "scripts", "single_batch.sh")
        resend_job_command = f'bsub -J {array_name}[{cur_job_indices_str}]'
        jobs_sending_res = subprocess.run(resend_job_command.split(), stdin=open(single_batch_bash_path, 'r'),
                                          stdout=subprocess.PIPE)
        job_array_ids += parse_sent_job_array_ids(jobs_sending_res.stdout)

    return job_array_ids


def parse_sent_job_array_ids(process_stdout) -> list:
    split_array_lines = process_stdout.split(b'\n')[:-1]
    job_array_ids = []
    for line in split_array_lines:
        array_id = int(line.split(b'<')[1].split(b'>')[0])
        job_array_ids.append(array_id)
    return job_array_ids


def run_distributed_children_jobs(out_path, cmd_line_single_batch, single_batch_template_file_name, num_jobs,
                                  array_name):
    # Create current bash scripts to send distributed calculations
    scripts_path = (out_path / "scripts").resolve()
    os.makedirs(scripts_path, exist_ok=True)
    single_batch_bash_path = os.path.join(scripts_path, "single_batch.sh")
    shutil.copyfile(os.path.join(os.getcwd(), "ClusterScripts", single_batch_template_file_name),
                    single_batch_bash_path)
    with open(single_batch_bash_path, 'a') as f:
        f.write(cmd_line_single_batch)

    multiple_batches_bash_path = os.path.join(scripts_path, "multiple_batches.sh")
    with open(multiple_batches_bash_path, 'w') as f:
        num_rows = 1
        while (num_rows - 1) * 2000 < num_jobs:
            f.write(f'bsub < $1 -J {array_name}'
                    f'[{(num_rows - 1) * 2000 + 1}-{min(num_rows * 2000, num_jobs)}]\n')
            num_rows += 1

    # Make sure the logs directory for the children jobs exists and delete previous logs.
    with open(single_batch_bash_path, 'r') as f:
        single_batch_bash_txt = f.read()
    logs_rel_dir_start = single_batch_bash_txt.find('-o') + 3
    log_rel_dir_end = single_batch_bash_txt.find('outs.%J.%I.log')
    logs_rel_dir = single_batch_bash_txt[logs_rel_dir_start:log_rel_dir_end]
    logs_dir = os.path.join(os.getcwd(), logs_rel_dir)
    os.makedirs(logs_dir, exist_ok=True)
    for file_name in os.listdir(logs_dir):
        os.unlink(os.path.join(logs_dir, file_name))

    # Send the jobs
    send_jobs_command = f'bash {multiple_batches_bash_path} {single_batch_bash_path}'
    jobs_sending_res = subprocess.run(send_jobs_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    job_array_ids = parse_sent_job_array_ids(jobs_sending_res.stdout)

    return job_array_ids