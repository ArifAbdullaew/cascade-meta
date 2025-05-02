# Copyright 2023 Flavien Solt, ETH Zurich.
# Licensed under the General Public License, Version 3.0, see LICENSE for details.
# SPDX-License-Identifier: GPL-3.0-only

# Toplevel for a cycle of program generation and RTL simulation.

from common.spike import calibrate_spikespeed
from common.profiledesign import profile_get_medeleg_mask
from cascade.fuzzfromdescriptor import gen_new_test_instance, fuzz_single_from_descriptor
import ray
import time
import multiprocessing as mp
import threading

callback_lock = threading.Lock()
newly_finished_tests = 0
curr_round_id = 0
all_times_to_detection = []

@ray.remote
def run_single_task(memsize, design_name, randseed, num_bbs, authorize_privileges):
    return fuzz_single_from_descriptor(memsize, design_name, randseed, num_bbs, authorize_privileges, None, True)

def test_done_callback(arg):
    global newly_finished_tests
    global callback_lock
    global curr_round_id
    global all_times_to_detection

    print(f"[DEBUG] test_done_callback() called with arg={arg}")

    with callback_lock:
        newly_finished_tests += 1

import ray
import time
from common.spike import calibrate_spikespeed
from common.profiledesign import profile_get_medeleg_mask
from cascade.fuzzfromdescriptor import gen_new_test_instance, fuzz_single_from_descriptor

def fuzzdesign(design_name: str, num_cores: int, seed_offset: int, can_authorize_privileges: bool, num_iterations: int = None):
    num_workers = num_cores
    assert num_workers > 0

    calibrate_spikespeed()
    profile_get_medeleg_mask(design_name)

    print(f"Starting parallel testing of `{design_name}` on {num_workers} workers.")

    process_instance_id = seed_offset
    futures = []
    iterations_done = 0

    for _ in range(num_workers):    
        memsize, _, _, num_bbs, authorize_priv = gen_new_test_instance(
            design_name, process_instance_id, can_authorize_privileges
        )
        future = run_single_task.remote(
            memsize, design_name, process_instance_id, num_bbs, authorize_priv
        )
        futures.append(future)
        process_instance_id += 1
        iterations_done += 1

    while futures:
        done, remaining = ray.wait(futures, num_returns=1, timeout=10)  

        for finished_ref in done:
            try:
                result = ray.get(finished_ref)  
                print(f"[INFO] Task done: {result}")
            except Exception as e:
                print(f"[ERROR] Error in `fuzz_single_from_descriptor`: {e}")

        futures = remaining

        if num_iterations is not None and iterations_done >= num_iterations:
            break

        memsize, _, _, num_bbs, authorize_priv = gen_new_test_instance(
            design_name, process_instance_id, can_authorize_privileges
        )
        new_future = run_single_task.remote(
            memsize, design_name, process_instance_id, num_bbs, authorize_priv
        )
        futures.append(new_future)
        process_instance_id += 1
        iterations_done += 1

        time.sleep(0.1)  


