# Copyright 2023 Flavien Solt, ETH Zurich.
# Licensed under the General Public License, Version 3.0, see LICENSE for details.
# SPDX-License-Identifier: GPL-3.0-only

# This script executes the fuzzer to find faulting programs.

from common.timeout import timeout
from common.designcfgs import get_design_boot_addr
from params.runparams import DO_ASSERT, NO_REMOVE_TMPFILES
from params.fuzzparams import PROBA_AUTHORIZE_PRIVILEGES
from cascade.basicblock import gen_basicblocks
from cascade.fuzzsim import SimulatorEnum, runtest_simulator
from cascade.genelf import gen_elf_from_bbs
from cascade.spikeresolution import spike_resolution

import os
import random
import time

FUZZ_USE_MODELSIM = False

LOG2_MEMSIZE_UPPERBOUND = 20
NUM_MAX_BBS_UPPERBOUND = 100

# Creates a new program descriptor.
def gen_new_test_instance(design_name: str, randseed: int, can_authorize_privileges: bool, fixed_memsize: int = None, fixed_num_bbs: int = None):
    return random.randrange(1 << 14, 1 << LOG2_MEMSIZE_UPPERBOUND) if fixed_memsize is None else fixed_memsize, design_name, randseed, random.randrange(20, NUM_MAX_BBS_UPPERBOUND) if fixed_num_bbs is None else fixed_num_bbs, can_authorize_privileges and random.random() < PROBA_AUTHORIZE_PRIVILEGES

# The main function for a single fuzzer run. It creates a new fuzzer state, populates it with basic blocks, and then runs the spike resolution. It does not run the RTL simulation.
# @return (fuzzerstate, rtl_elfpath, expected_regvals: list) where expected_regval is a list of num_pickable_regs-1 expected reg values (we ignore x0)
def gen_fuzzerstate_elf_expectedvals(memsize: int, design_name: str, randseed: int, nmax_bbs: int, authorize_privileges: bool, check_pc_spike_again: bool, max_num_instructions: int = None, no_dependency_bias: bool = False):
    from cascade.fuzzerstate import FuzzerState
    if DO_ASSERT:
        assert nmax_bbs is None or nmax_bbs > 0

    start = time.time()
    random.seed(randseed)
    fuzzerstate = FuzzerState(get_design_boot_addr(design_name), design_name, memsize, randseed, nmax_bbs, authorize_privileges, max_num_instructions, no_dependency_bias)
    gen_basicblocks(fuzzerstate)
    time_seconds_spent_in_gen_bbs = time.time() - start

    # spike resolution
    start = time.time()
    expected_regvals = spike_resolution(fuzzerstate, check_pc_spike_again)
    time_seconds_spent_in_spike_resol = time.time() - start

    start = time.time()
    # This is typically quite short
    rtl_elfpath = gen_elf_from_bbs(fuzzerstate, False, 'rtl', fuzzerstate.instance_to_str(), fuzzerstate.design_base_addr)
    time_seconds_spent_in_gen_elf = time.time() - start
    return fuzzerstate, rtl_elfpath, expected_regvals, time_seconds_spent_in_gen_bbs, time_seconds_spent_in_spike_resol, time_seconds_spent_in_gen_elf

###
# Exposed function
###

def run_rtl(memsize: int, design_name: str, randseed: int, nmax_bbs: int, authorize_privileges: bool, check_pc_spike_again: bool, nmax_instructions: int = None, nodependencybias: bool = False, simulator=SimulatorEnum.VERILATOR):
    print(f"[DEBUG] Running RTL simulation for design: {design_name}, seed: {randseed}")
    fuzzerstate, rtl_elfpath, finalregvals_spikeresol, time_seconds_spent_in_gen_bbs, time_seconds_spent_in_spike_resol, time_seconds_spent_in_gen_elf = gen_fuzzerstate_elf_expectedvals(memsize, design_name, randseed, nmax_bbs, authorize_privileges, check_pc_spike_again, nmax_instructions, nodependencybias)

    start = time.time()
    is_success, rtl_msg = runtest_simulator(fuzzerstate, rtl_elfpath, finalregvals_spikeresol, simulator=simulator)
    time_seconds_spent_in_rtl_sim = time.time() - start
    print(f"[DEBUG] RTL simulation completed in {time_seconds_spent_in_rtl_sim} seconds")

    if NO_REMOVE_TMPFILES:
        print(f"[DEBUG] RTL ELF file retained at {rtl_elfpath}")
    else:
        os.remove(rtl_elfpath)
        print(f"[DEBUG] RTL ELF file {rtl_elfpath} removed")

    if not is_success:
        print(f"[ERROR] RTL simulation failed: {rtl_msg}")
        raise Exception(rtl_msg)
    
    return time_seconds_spent_in_gen_bbs, time_seconds_spent_in_spike_resol, time_seconds_spent_in_gen_elf, time_seconds_spent_in_rtl_sim

###
# Some tests
###

# This function runs a single test run from a test descriptor (memsize, design_name, randseed, nmax_bbs) and returns the gathered times (used for the performance evaluation plot).
# Loggers are not yet very tested facilities.
@timeout(seconds=60*60*2)
def fuzz_single_from_descriptor(memsize: int, design_name: str, randseed: int, nmax_bbs: int, authorize_privileges: bool, loggers: list = None, check_pc_spike_again: bool = False, start_time: float = None):
    print(f"[DEBUG] Starting fuzz test: memsize={memsize}, design_name={design_name}, randseed={randseed}, nmax_bbs={nmax_bbs}, authorize_privileges={authorize_privileges}, check_pc_spike_again={check_pc_spike_again}")
    try:
        gathered_times = run_rtl(memsize, design_name, randseed, nmax_bbs, authorize_privileges, check_pc_spike_again)
        print(f"[DEBUG] Fuzz test completed successfully for seed {randseed}")
        if loggers is not None:
            logger = loggers[random.randrange(len(loggers))]
            print(f"[DEBUG] Logging results with logger: {logger}")
            logger.log(True, {'memsize': memsize, 'design_name': design_name, 'randseed': randseed, 'nmax_bbs': nmax_bbs, 'authorize_privileges': authorize_privileges}, False, '')
            print(f"[DEBUG] Logging completed successfully")
        else:
            print(f"[DEBUG] Returning gathered_times: {gathered_times}")
            return gathered_times
    except Exception as e:
        emsg = str(e)
        print(f"[ERROR] Exception occurred during fuzzing: {emsg}")
        if loggers is not None:
            logger = loggers[random.randrange(len(loggers))]
            print(f"[DEBUG] Logging failure with logger: {logger}")
            
            if 'Spike timeout' in emsg:
                print(f"[DEBUG] Logging spike timeout failure")
                logger.log(False, {'memsize': memsize, 'design_name': design_name, 'randseed': randseed, 'nmax_bbs': nmax_bbs, 'authorize_privileges': authorize_privileges}, True, '')
            else:
                print(f"[DEBUG] Logging general failure with message: {emsg}")
                logger.log(False, {'memsize': memsize, 'design_name': design_name, 'randseed': randseed, 'nmax_bbs': nmax_bbs, 'authorize_privileges': authorize_privileges}, False, emsg)
            print(f"[DEBUG] Logging completed with failure")
        else:
            print(f"[DEBUG] Failed test_run_rtl_single for params: memsize={memsize}, design_name={design_name}, check_pc_spike_again={check_pc_spike_again}, randseed={randseed}, nmax_bbs={nmax_bbs}, authorize_privileges={authorize_privileges}")
        print(f"[DEBUG] Returning failure values (0, 0, 0, 0)")
        return 0, 0, 0, 0
