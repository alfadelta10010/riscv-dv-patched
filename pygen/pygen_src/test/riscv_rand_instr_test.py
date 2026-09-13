"""
Copyright 2020 Google LLC
Copyright 2020 PerfectVIPs Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""

import sys
import time
import logging
sys.path.append("pygen/")
from pygen_src.test.riscv_instr_base_test import riscv_instr_base_test
from pygen_src.riscv_instr_gen_config import cfg
from pygen_src.riscv_utils import gen_config_table


class riscv_rand_instr_test(riscv_instr_base_test):
    def __init__(self):
        super().__init__()

    def randomize_cfg(self):
        # Upstream hardcoded cfg.instr_cnt = 10000 and cfg.num_of_sub_program = 5
        # here.  Both are plain Python ints referenced from the default_c
        # constraint in riscv_instr_gen_config.py:319-323, and pyvsc elaborates
        # that constraint with the value the attribute held when cfg was built
        # from argv -- assigning afterwards has no effect on the solver.  The
        # result was cfg.num_of_sub_program == 5 driving gen_sub_program()'s loop
        # while sub_program_instr_cnt was still sized from the command-line value,
        # so riscv_asm_program_gen.py:217 raised IndexError and every test with
        # "gen_test: riscv_rand_instr_test" died.  The command line already
        # carries these values (run.py always passes --instr_cnt and
        # --num_of_sub_program, defaulting to 200 and 5), so just honour it.
        cfg.randomize()
        logging.info("riscv_instr_gen_config is randomized")
        gen_config_table()

    def apply_directed_instr(self):
        # Mix below directed instruction streams with the random instructions
        self.asm.add_directed_instr_stream("riscv_load_store_rand_instr_stream", 4)
        # self.asm.add_directed_instr_stream("riscv_loop_instr", 3)
        self.asm.add_directed_instr_stream("riscv_jal_instr", 4)
        # self.asm.add_directed_instr_stream("riscv_hazard_instr_stream", 4)
        self.asm.add_directed_instr_stream("riscv_load_store_hazard_instr_stream", 4)
        # self.asm.add_directed_instr_stream("riscv_multi_page_load_store_instr_stream", 4)
        # self.asm.add_directed_instr_stream("riscv_mem_region_stress_test", 4)


start_time = time.time()
riscv_rand_test_ins = riscv_rand_instr_test()
riscv_rand_test_ins.run()
end_time = time.time()
logging.info("Total execution time: {}s".format(round(end_time - start_time)))
