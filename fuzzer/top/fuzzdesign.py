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

def test_done_callback(arg):
    global newly_finished_tests
    global callback_lock
    global curr_round_id
    global all_times_to_detection

    print(f"[DEBUG] test_done_callback() called with arg={arg}")

    with callback_lock:
        newly_finished_tests += 1

def fuzzdesign(design_name: str, num_cores: int, seed_offset: int, can_authorize_privileges: bool):
    num_workers = num_cores
    assert num_workers > 0

    calibrate_spikespeed()
    profile_get_medeleg_mask(design_name)

    print(f"Starting parallel testing of `{design_name}` on {num_workers} workers.")

    process_instance_id = seed_offset
    futures = []

    for _ in range(num_workers):
        memsize, _, _, num_bbs, authorize_priv = gen_new_test_instance(
            design_name, process_instance_id, can_authorize_privileges
        )
        # Запускаем функцию ЛОКАЛЬНО, без remote()
        future = fuzz_single_from_descriptor(
            memsize, design_name, process_instance_id, num_bbs, authorize_priv, None, True
        )
        futures.append(future)
        process_instance_id += 1

    while True:
        time.sleep(2)  # Ожидание, чтобы не перегружать CPU

        # Проверяем, завершились ли какие-то задачи
        completed = [f for f in futures if isinstance(f, tuple)]
        if completed:
            for _ in range(len(completed)):
                memsize, _, _, num_bbs, authorize_priv = gen_new_test_instance(
                    design_name, process_instance_id, can_authorize_privileges
                )
                future = fuzz_single_from_descriptor(
                    memsize, design_name, process_instance_id, num_bbs, authorize_priv, None, True
                )
                futures.append(future)
                process_instance_id += 1

            # Удаляем завершенные задачи
            futures = [f for f in futures if not isinstance(f, tuple)]

