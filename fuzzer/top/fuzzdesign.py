# Copyright 2023 Flavien Solt, ETH Zurich.
# Licensed under the General Public License, Version 3.0, see LICENSE for details.
# SPDX-License-Identifier: GPL-3.0-only

# Toplevel for a cycle of program generation and RTL simulation.

from common.spike import calibrate_spikespeed
from common.profiledesign import profile_get_medeleg_mask
from cascade.fuzzfromdescriptor import gen_new_test_instance, fuzz_single_from_descriptor
import ray
import time
import threading

callback_lock = threading.Lock()
newly_finished_tests = 0
curr_round_id = 0
all_times_to_detection = []

def test_done_callback(arg):
    global newly_finished_tests
    global callback_lock
    global curr_round_id
    global all_times_to_detection
    
    print(f"[DEBUG] test_done_callback() called with arg={arg}")

    with callback_lock:
        newly_finished_tests += 1

@ray.remote
def fuzz_single_task(memsize, design_name, process_instance_id, num_bbs, authorize_privileges):
    print(f"[DEBUG] Running fuzz_single_from_descriptor for instance {process_instance_id}")
    return fuzz_single_from_descriptor(memsize, design_name, process_instance_id, num_bbs, authorize_privileges, None, True)

def fuzzdesign(design_name: str, num_cores: int, seed_offset: int, can_authorize_privileges: bool):
    global newly_finished_tests, curr_round_id, all_times_to_detection

    newly_finished_tests = 0
    curr_round_id = 0
    all_times_to_detection = []

    calibrate_spikespeed()
    profile_get_medeleg_mask(design_name)

    num_workers = num_cores
    assert num_workers > 0

    print(f"[DEBUG] Starting Ray-based parallel testing on `{design_name}` with {num_workers} workers.")

    process_instance_id = seed_offset

    active_tasks = []
    
    print("[DEBUG] Spawning initial worker processes")
    for _ in range(num_workers):
        memsize, _, _, num_bbs, authorize_privileges = gen_new_test_instance(design_name, process_instance_id, can_authorize_privileges)
        print(f"[DEBUG] Spawning test instance: ID={process_instance_id}, MemSize={memsize}, Num_BBs={num_bbs}, Privileges={authorize_privileges}")
        task = fuzz_single_task.remote(memsize, design_name, process_instance_id, num_bbs, authorize_privileges)
        active_tasks.append(task)
        process_instance_id += 1

    while True:
        time.sleep(2)
        print("[DEBUG] Checking for newly finished tests...")

        ready_tasks, active_tasks = ray.wait(active_tasks, num_returns=1, timeout=0)

        if ready_tasks:
            print(f"[DEBUG] {len(ready_tasks)} test(s) finished, spawning new instances...")
            for _ in range(len(ready_tasks)):
                memsize, _, _, num_bbs, authorize_privileges = gen_new_test_instance(design_name, process_instance_id, can_authorize_privileges)
                print(f"[DEBUG] Spawning test instance: ID={process_instance_id}, MemSize={memsize}, Num_BBs={num_bbs}, Privileges={authorize_privileges}")
                task = fuzz_single_task.remote(memsize, design_name, process_instance_id, num_bbs, authorize_privileges)
                active_tasks.append(task)
                process_instance_id += 1
