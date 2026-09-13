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

import logging
import random
import vsc


@vsc.randobj
class riscv_program:
    def __init__(self, name=""):
        self.name = name
        self.program_id = vsc.rand_bit_t(16)
        self.call_stack_level = vsc.rand_uint32_t()
        self.sub_program_id = vsc.rand_list_t(vsc.bit_t(16))

    @vsc.constraint
    def legel_c(self):
        vsc.unique(self.sub_program_id)
        with vsc.foreach(self.sub_program_id, idx = True) as i:
            self.sub_program_id[i] != self.program_id

    def convert2string(self):
        string = "PID[{}] Lv[{}] :".format(self.program_id, self.call_stack_level)
        for i in range(len(self.sub_program_id)):
            string = "{} {}".format(string, self.sub_program_id[i])
        return string

# -----------------------------------------------------------------------------------------
# RISC-V assembly program call stack generator
#
# The call stack is generated as a tree structure to avoid dead call loop.
# Level 0:                     P0
#                           /  |  \
# Level 1:                 P1  P2  P3
#                          |     \/  \
# Level 2:                 P4    P5   P6
#                                 |
# Level 3:                        P7
#
# Rules: A program can only call the program in the next level.
#        A program can be called many times by other upper level programs.
#        A program can call the same lower level programs multiple times.
# -----------------------------------------------------------------------------------------


@vsc.randobj
class riscv_callstack_gen:
    def __init__(self):
        # Number of programs in the call stack. SV: `int program_cnt = 10`.
        # pyvsc's int_t/bit_t take (width, init), so vsc.int_t(10) would be a
        # 10-bit field holding 0; give the width and the value explicitly.
        self.program_cnt = vsc.bit_t(8, 10)
        # Handles of all programs
        self.program_h = []
        # Maximum call stack level. SV: `int max_stack_level = 50`.
        self.max_stack_level = vsc.bit_t(8, 50)
        # Call stack level of each program
        self.stack_level = vsc.randsz_list_t(vsc.bit_t(11))

    @vsc.constraint
    def program_stack_level_c(self):
        # The stack level is assigned in ascending order to avoid call loop
        self.stack_level.size == self.program_cnt
        self.stack_level[0] == 0
        with vsc.foreach(self.stack_level, idx=True) as i:
            with vsc.if_then(i > 0):
                self.stack_level[i] in vsc.rangelist(vsc.rng(1, self.program_cnt - 1))
                self.stack_level[i] >= self.stack_level[i - 1]
                self.stack_level[i] <= self.stack_level[i - 1] + 1
                self.stack_level[i] <= self.max_stack_level

    # Init all program instances before randomization
    def init(self, program_cnt):
        self.program_cnt = program_cnt
        self.program_h = [0] * program_cnt
        for i in range(len(self.program_h)):
            self.program_h[i] = riscv_program("program_{}".format(i))

# In the randomization stage, only the stack level of each program is specified. The call stack
# generation process is to build the call relationship between different programs. This is
# implemented with post randomize rather than constraints for performance considerations.

    def post_randomize(self):
        program_cnt = int(self.program_cnt)
        last_level = int(self.stack_level[program_cnt - 1])
        for i in range(len(self.program_h)):
            self.program_h[i].program_id = i
            self.program_h[i].call_stack_level = self.stack_level[i]
        # Top-down generate the entire call stack.
        # A program can only call the programs in the next level.
        for i in range(last_level):
            program_list = []
            next_program_list = []
            idx = 0
            for j in range(program_cnt):
                if self.stack_level[j] == i:
                    program_list.append(j)
                if self.stack_level[j] == i + 1:
                    next_program_list.append(j)
            # Randmly duplicate some sub programs in the pool to create a case that
            # one sub program is called by multiple caller. Also it's possible to call
            # the same sub program in one program multiple times.
            # src/riscv_callstack_gen.sv:157 is
            # $urandom_range(size, size + 1) -- inclusive, so it may add one
            # duplicated sub-program. randrange excludes the upper bound and
            # always returned size, disabling that entirely.
            total_sub_program_cnt = random.randrange(len(next_program_list),
                                                     len(next_program_list) + 2)
            # src/riscv_callstack_gen.sv: the first next_program_list.size()
            # entries are the next-level programs themselves, so every one of
            # them gets a caller; an extra entry is a duplicate drawn from the
            # same list.
            sub_program_id_pool = [next_program_list[j] if j < len(next_program_list)
                                   else random.choice(next_program_list)
                                   for j in range(total_sub_program_cnt)]
            random.shuffle(sub_program_id_pool)
            sub_program_cnt = [0] * len(program_list)
            logging.info("{} programs @Lv{}-> {} programs at next level".format(
                         len(program_list), i, len(sub_program_id_pool)))
            # Distribute the programs of the next level among the programs of current level
            # Make sure all program has a caller so that no program is obsolete.

            for j in range(len(sub_program_id_pool)):
                # randrange(0, n - 1) never returns n - 1, so the last
                # program at this level could never be chosen as a caller.
                caller_id = random.randrange(len(sub_program_cnt))
                sub_program_cnt[caller_id] += 1

            for j in range(len(program_list)):
                id = program_list[j]
                self.program_h[id].sub_program_id = [0] * sub_program_cnt[j]
                logging.info("{} sub programs are assigned to program[{}]".format(
                             sub_program_cnt[j], id))
                for k in range(len(self.program_h[id].sub_program_id)):
                    self.program_h[id].sub_program_id[k] = sub_program_id_pool[idx]
                    idx += 1

    def print_call_stack(self, program_id_t, i, string_str):
        if len(self.program_h[i].sub_program_id) == 0:
            logging.info("{}", string_str)
        else:
            for j in range(len(self.program_h[i].sub_program_id)):
                self.print_call_stack(self.program_h[i].sub_program_id[j], "{} -> {}".format(
                    string_str, self.program_h[i].sub_program_id[j]))
