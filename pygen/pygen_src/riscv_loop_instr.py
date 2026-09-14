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

import vsc
import sys
import random
import logging
from pygen_src.riscv_instr_gen_config import cfg
from pygen_src.isa.riscv_instr import riscv_instr
from pygen_src.riscv_instr_stream import riscv_rand_instr_stream
from pygen_src.riscv_instr_pkg import (riscv_reg_t, riscv_instr_name_t, pkg_ins,
                                       riscv_instr_format_t, riscv_instr_category_t,
                                       compressed_gpr)


@vsc.randobj
class riscv_loop_instr(riscv_rand_instr_stream):

    def __init__(self):
        super().__init__()
        self.loop_cnt_reg = vsc.randsz_list_t(vsc.enum_t(riscv_reg_t))
        self.loop_limit_reg = vsc.randsz_list_t(vsc.enum_t(riscv_reg_t))
        self.loop_init_val = vsc.randsz_list_t(vsc.int32_t())
        self.loop_step_val = vsc.randsz_list_t(vsc.int32_t())
        self.loop_limit_val = vsc.randsz_list_t(vsc.int32_t())
        self.num_of_nested_loop = vsc.rand_bit_t(3)
        self.num_of_instr_in_loop = vsc.rand_int32_t(0)
        self.branch_type = vsc.randsz_list_t(vsc.enum_t(riscv_instr_name_t))
        self.loop_init_instr = []
        self.loop_update_instr = []
        self.loop_branch_instr = []
        self.loop_branch_target_instr = []
        # Aggregated loop instruction stream
        self.loop_instr = []

    @vsc.constraint
    def legal_loop_regs_c(self):
        self.num_of_nested_loop.inside(vsc.rangelist(1, 2))
        self.loop_limit_reg.size.inside(vsc.rangelist((1, 32)))
        self.loop_cnt_reg.size.inside(vsc.rangelist((1, 8)))
        vsc.solve_order(self.num_of_nested_loop, self.loop_cnt_reg)
        vsc.solve_order(self.num_of_nested_loop, self.loop_limit_reg)
        with vsc.foreach(self.loop_cnt_reg, idx = True) as i:
            self.loop_cnt_reg[i] != riscv_reg_t.ZERO
            with vsc.foreach(cfg.reserved_regs, idx = True) as j:
                self.loop_cnt_reg[i] != cfg.reserved_regs[j]
        with vsc.foreach(self.loop_limit_reg, idx = True) as i:
            with vsc.foreach(cfg.reserved_regs, idx = True) as j:
                self.loop_limit_reg[i] != cfg.reserved_regs[j]
        vsc.unique(self.loop_cnt_reg)
        vsc.unique(self.loop_limit_reg)
        self.loop_cnt_reg.size == self.num_of_nested_loop
        self.loop_limit_reg.size == self.num_of_nested_loop

    @vsc.constraint
    def loop_c(self):
        vsc.solve_order(self.num_of_nested_loop, self.loop_init_val)
        vsc.solve_order(self.num_of_nested_loop, self.loop_step_val)
        vsc.solve_order(self.num_of_nested_loop, self.loop_limit_val)
        vsc.solve_order(self.loop_limit_val, self.loop_limit_reg)
        vsc.solve_order(self.branch_type, self.loop_init_val)
        vsc.solve_order(self.branch_type, self.loop_step_val)
        vsc.solve_order(self.branch_type, self.loop_limit_val)
        self.num_of_instr_in_loop.inside(vsc.rangelist((1, 25)))
        self.num_of_nested_loop.inside(vsc.rangelist(1, 2))
        self.loop_init_val.size.inside(vsc.rangelist(1, 2))
        self.loop_step_val.size.inside(vsc.rangelist(1, 2))
        self.loop_limit_val.size.inside(vsc.rangelist(1, 2))
        self.branch_type.size.inside(vsc.rangelist(1, 2))
        self.loop_init_val.size == self.num_of_nested_loop
        self.branch_type.size == self.num_of_nested_loop
        self.loop_step_val.size == self.num_of_nested_loop
        self.loop_limit_val.size == self.num_of_nested_loop
        self.branch_type.size == self.num_of_nested_loop
        with vsc.foreach(self.branch_type, idx = True) as i:
            with vsc.if_then(cfg.disable_compressed_instr == 0):
                self.branch_type[i].inside(vsc.rangelist(riscv_instr_name_t.C_BNEZ,
                                                         riscv_instr_name_t.C_BEQZ,
                                                         riscv_instr_name_t.BEQ,
                                                         riscv_instr_name_t.BNE,
                                                         riscv_instr_name_t.BLTU,
                                                         riscv_instr_name_t.BLT,
                                                         riscv_instr_name_t.BGEU,
                                                         riscv_instr_name_t.BGE))
            with vsc.else_then():
                self.branch_type[i].inside(vsc.rangelist(riscv_instr_name_t.BEQ,
                                                         riscv_instr_name_t.BNE,
                                                         riscv_instr_name_t.BLTU,
                                                         riscv_instr_name_t.BLT,
                                                         riscv_instr_name_t.BGEU,
                                                         riscv_instr_name_t.BGE))
        with vsc.foreach(self.loop_init_val, idx = True) as i:
            with vsc.if_then(self.branch_type[i].inside(vsc.rangelist(riscv_instr_name_t.C_BNEZ,
                                                                      riscv_instr_name_t.C_BEQZ))):
                self.loop_limit_val[i] == 0
                self.loop_limit_reg[i] == riscv_reg_t.ZERO
                self.loop_cnt_reg[i].inside(vsc.rangelist(list(compressed_gpr)))
            with vsc.else_then:
                self.loop_limit_val[i].inside(vsc.rangelist((-20, 20)))
                self.loop_limit_reg[i] != riscv_reg_t.ZERO
            with vsc.if_then(self.branch_type[i].inside(vsc.rangelist(riscv_instr_name_t.C_BNEZ,
                                                                      riscv_instr_name_t.C_BEQZ,
                                                                      riscv_instr_name_t.BEQ,
                                                                      riscv_instr_name_t.BNE))):
                self.loop_limit_val[i] != self.loop_init_val[i]
                ((self.loop_limit_val[i] - self.loop_init_val[i]) % self.loop_step_val[i]) == 0
            with vsc.else_if(self.branch_type[i] == riscv_instr_name_t.BGE):
                self.loop_step_val[i] < 0
            with vsc.else_if(self.branch_type[i].inside(vsc.rangelist(riscv_instr_name_t.BGEU))):
                self.loop_step_val[i] < 0
                self.loop_init_val[i] > 0
                # Avoid count to negative
                (self.loop_step_val[i] + self.loop_limit_val[i]) > 0
            with vsc.else_if(self.branch_type[i] == riscv_instr_name_t.BLT):
                self.loop_step_val[i] > 0
            with vsc.else_if(self.branch_type[i] == riscv_instr_name_t.BLTU):
                self.loop_step_val[i] > 0
                self.loop_limit_val[i] > 0
            self.loop_init_val[i].inside(vsc.rangelist((-10, 10)))
            self.loop_step_val[i].inside(vsc.rangelist((-10, 10)))
            with vsc.if_then(self.loop_init_val[i] < self.loop_limit_val[i]):
                self.loop_step_val[i] > 0
            with vsc.else_then:
                self.loop_step_val[i] < 0

    # src/riscv_loop_instr.sv:53-55 closes the loop register set with a cross
    # constraint the port dropped:
    #
    #     foreach (loop_cnt_reg[i]) {
    #       foreach (loop_limit_reg[j]) { loop_cnt_reg[i] != loop_limit_reg[j]; }
    #     }
    #
    # legal_loop_regs_c keeps each list unique but never separates the two, so
    # the solver is free to give a loop the same register as counter and limit.
    # The backward branch then compares that register with itself -- `beq a5,
    # a5, <back>` -- which is unconditional, so the loop cannot exit and the
    # program runs until the testbench's instruction budget stops it. Both
    # models execute it identically, so the comparison passes and the test
    # verifies nothing past the loop. This is the defect behind the
    # `bge x31, x31` loops seen in earlier runs.
    #
    # Applied to the solved values, like legalize_rs1() in
    # riscv_load_store_instr_lib.py: expressing it as a constraint hits the
    # same pyvsc randset-merge crash the comments below describe.
    #
    # A compressed branch (C_BEQZ/C_BNEZ) has no rs2 and loop_c pins its limit
    # register to ZERO on purpose, so those are left alone.
    def legalize_loop_regs(self):
        compressed = (riscv_instr_name_t.C_BEQZ, riscv_instr_name_t.C_BNEZ)
        cnt = [int(self.loop_cnt_reg[i]) for i in range(len(self.loop_cnt_reg))]
        illegal = {int(riscv_reg_t.ZERO)}
        illegal.update(int(r) for r in cfg.reserved_regs)
        for i in range(len(self.loop_limit_reg)):
            if riscv_instr_name_t(int(self.branch_type[i])) in compressed:
                continue
            limit = [int(self.loop_limit_reg[j])
                     for j in range(len(self.loop_limit_reg))]
            if limit[i] not in cnt:
                continue
            taken = set(cnt) | set(limit) | illegal
            choices = [int(r) for r in riscv_reg_t if int(r) not in taken]
            if not choices:
                logging.critical("No register left for a loop limit")
                sys.exit(1)
            new_reg = random.choice(choices)
            logging.info("loop %0d: limit register collided with the counter "
                         "(%0d); moving it to %0d", i, limit[i], new_reg)
            self.loop_limit_reg[i] = new_reg

    # loop_c above is the SV constraint (src/riscv_loop_instr.sv:56-110) and
    # every pyvsc solution satisfies it, but the solutions are heavily skewed:
    # over the 3,048 loops of the ea5573d assembly audit the limit collapsed
    # to init + step, so 80% of BNE and 88% of C.BNEZ loops never iterated,
    # where a uniform draw over the constraint's solutions gives 18% and 37%.
    #
    # With `solve branch_type before loop_init_val/step/limit`, the SV picks
    # the branch type first and then any (init, step, limit) the constraint
    # allows for it. Keep the solved branch type and redraw the values
    # uniformly from that set, enumerated from the SV text:
    #
    #   branch_type inside {C_BNEZ, C_BEQZ}  -> limit == 0
    #   else                                 -> limit inside {[-20:20]}
    #   branch_type inside {C_BNEZ, C_BEQZ, BEQ, BNE} ->
    #       (limit - init) % step == 0 && limit != init
    #   BGE  -> step < 0
    #   BGEU -> step < 0; init > 0; step + limit > 0
    #   BLT  -> step > 0
    #   BLTU -> step > 0; limit > 0
    #   init inside {[-10:10]}; step inside {[-10:10]}
    #   init < limit -> step > 0, else step < 0
    #
    # step == 0 satisfies no branch: the last implication needs step != 0 (and
    # an SV `% 0` is X, which fails the equality constraint).
    _loop_val_space = {}

    @classmethod
    def loop_val_space(cls, branch):
        if branch in cls._loop_val_space:
            return cls._loop_val_space[branch]
        n = riscv_instr_name_t
        sols = []
        for init in range(-10, 11):
            for step in range(-10, 11):
                if step == 0:
                    continue
                limits = [0] if branch in (n.C_BNEZ, n.C_BEQZ) else range(-20, 21)
                for limit in limits:
                    if (step > 0) != (init < limit):
                        continue
                    if branch in (n.C_BNEZ, n.C_BEQZ, n.BEQ, n.BNE):
                        ok = limit != init and (limit - init) % step == 0
                    elif branch == n.BGE:
                        ok = step < 0
                    elif branch == n.BGEU:
                        ok = step < 0 and init > 0 and step + limit > 0
                    elif branch == n.BLT:
                        ok = step > 0
                    elif branch == n.BLTU:
                        ok = step > 0 and limit > 0
                    else:
                        ok = False
                    if ok:
                        sols.append((init, step, limit))
        if not sols:
            logging.critical("No loop values satisfy loop_c for %s", branch.name)
            sys.exit(1)
        cls._loop_val_space[branch] = sols
        return sols

    def legalize_loop_vals(self):
        for i in range(int(self.num_of_nested_loop)):
            branch = riscv_instr_name_t(int(self.branch_type[i]))
            init, step, limit = random.choice(self.loop_val_space(branch))
            self.loop_init_val[i] = init
            self.loop_step_val[i] = step
            self.loop_limit_val[i] = limit

    def post_randomize(self):
        self.legalize_loop_regs()
        self.legalize_loop_vals()
        for i in range(len(self.loop_cnt_reg)):
            self.reserved_rd.append(self.loop_cnt_reg[i])
        for i in range(len(self.loop_limit_reg)):
            self.reserved_rd.append(self.loop_limit_reg[i])
        # Generate instructions that mixed with the loop instructions
        self.initialize_instr_list(self.num_of_instr_in_loop)
        self.gen_instr(1)
        # Randomize the key loop instructions
        self.loop_init_instr = [0] * 2 * self.num_of_nested_loop
        self.loop_update_instr = [0] * self.num_of_nested_loop
        self.loop_branch_instr = [0] * self.num_of_nested_loop
        self.loop_branch_target_instr = [0] * self.num_of_nested_loop
        for i in range(self.num_of_nested_loop):
            # Read the solved list elements into plain Python values *before*
            # entering randomize_with().  Subscripting a vsc list inside a
            # constraint block builds an ExprArraySubscriptModel, and pyvsc's
            # RandInfoBuilder.visit_expr_array_subscript() crashes when it has
            # to merge randsets: rand_info_builder.py:317 does
            # "self._randset_m.pop(idx)" where idx is an int but _randset_m is
            # keyed by RandSet objects (the equivalent code for a plain field
            # reference, line 407, correctly pops by object).  That KeyError is
            # why these constraints were commented out upstream.
            cnt_reg = int(self.loop_cnt_reg[i])
            limit_reg = int(self.loop_limit_reg[i])
            # imm is an unsigned 32-bit field; riscv_instr.extend_imm() sign
            # extends the low imm_len bits afterwards, so hand the solver the
            # 12-bit two's complement of the (possibly negative) loop value.
            init_val = int(self.loop_init_val[i]) & 0xfff
            limit_val = int(self.loop_limit_val[i]) & 0xfff
            step_val = int(self.loop_step_val[i]) & 0xfff
            # Instruction to init the loop counter
            try:
                self.loop_init_instr.insert(2 * i, riscv_instr.get_rand_instr(
                    include_instr = [riscv_instr_name_t.ADDI]))
                with self.loop_init_instr[2 * i].randomize_with():
                    self.loop_init_instr[2 * i].rd == cnt_reg
                    self.loop_init_instr[2 * i].rs1 == riscv_reg_t.ZERO
                    self.loop_init_instr[2 * i].imm == init_val
            except Exception:
                logging.critical("Cannot randomize loop init1 instruction")
                sys.exit(1)
            self.loop_init_instr[2 * i].comment = \
                pkg_ins.format_string("init loop {} counter".format(i))
            # Instruction to init loop limit
            try:
                self.loop_init_instr[2 * i + 1] = riscv_instr.get_rand_instr(
                    include_instr = [riscv_instr_name_t.ADDI])
                with self.loop_init_instr[2 * i + 1].randomize_with():
                    self.loop_init_instr[2 * i + 1].rd == limit_reg
                    self.loop_init_instr[2 * i + 1].rs1 == riscv_reg_t.ZERO
                    self.loop_init_instr[2 * i + 1].imm == limit_val
            except Exception:
                logging.critical("Cannot randomize loop init2 instruction")
                sys.exit(1)
            self.loop_init_instr[2 * i + 1].comment = \
                pkg_ins.format_string("init loop {} limit".format(i))
            # Branch target instruction, can be anything
            self.loop_branch_target_instr[i] = riscv_instr.get_rand_instr(
                include_category = [riscv_instr_category_t.ARITHMETIC.name,
                                    riscv_instr_category_t.LOGICAL.name,
                                    riscv_instr_category_t.COMPARE.name],
                exclude_instr = [riscv_instr_name_t.C_ADDI16SP])
            try:
                with self.loop_branch_target_instr[i].randomize_with():
                    with vsc.if_then(self.loop_branch_target_instr[i].format ==
                                     riscv_instr_format_t.CB_FORMAT):
                        self.loop_branch_target_instr[i].rs1.not_inside(
                            vsc.rangelist(self.reserved_rd))
                        self.loop_branch_target_instr[i].rs1.not_inside(
                            vsc.rangelist(cfg.reserved_regs))
                    with vsc.if_then(self.loop_branch_target_instr[i].has_rd == 1):
                        self.loop_branch_target_instr[i].rd.not_inside(
                            vsc.rangelist(self.reserved_rd))
                        self.loop_branch_target_instr[i].rd.not_inside(
                            vsc.rangelist(cfg.reserved_regs))
            except Exception:
                logging.critical("Cannot randomize branch target instruction")
                sys.exit(1)
            self.loop_branch_target_instr[i].label = pkg_ins.format_string(
                "{}_{}_t".format(self.label, i))
            # Instruction to update loop counter
            self.loop_update_instr[i] = riscv_instr.get_rand_instr(
                include_instr = [riscv_instr_name_t.ADDI])
            try:
                with self.loop_update_instr[i].randomize_with():
                    self.loop_update_instr[i].rd == cnt_reg
                    self.loop_update_instr[i].rs1 == cnt_reg
                    self.loop_update_instr[i].imm == step_val
            except Exception:
                logging.critical("Cannot randomize loop update instruction")
                sys.exit(1)
            self.loop_update_instr[i].comment = pkg_ins.format_string(
                "update loop {} counter".format(i))
            # Backward branch instruction
            branch_name = riscv_instr_name_t(int(self.branch_type[i]))
            self.loop_branch_instr[i] = riscv_instr.get_rand_instr(
                include_instr = [branch_name])
            self.loop_branch_instr[i].randomize()
            with self.loop_branch_instr[i].randomize_with():
                self.loop_branch_instr[i].rs1 == cnt_reg
                # Upstream had this as a vsc.if_then() on branch_type, disabled
                # because pyvsc could not build it.  branch_type is already
                # solved by the time post_randomize() runs, so decide in plain
                # Python.  Without this the backward branch compares the loop
                # counter against a *random* register instead of the loop limit
                # register, so the generated loop has no reliable exit
                # condition.
                if branch_name not in [riscv_instr_name_t.C_BEQZ,
                                       riscv_instr_name_t.C_BNEZ]:
                    self.loop_branch_instr[i].rs2 == limit_reg
            self.loop_branch_instr[i].comment = pkg_ins.format_string(
                "branch for loop {}".format(i))
            self.loop_branch_instr[i].imm_str = self.loop_branch_target_instr[i].label
            self.loop_branch_instr[i].branch_assigned = 1
        # Randomly distribute the loop instruction in the existing instruction stream
        self.build_loop_instr_stream()
        self.mix_instr_stream(self.loop_instr, 1)
        for i in range(len(self.instr_list)):
            if (self.instr_list[i].label != ""):
                self.instr_list[i].has_label = 1
            else:
                self.instr_list[i].has_label = 0
            self.instr_list[i].atomic = 1

    # Build the whole loop structure from innermost loop to the outermost loop
    def build_loop_instr_stream(self):
        self.loop_instr = []
        for i in range(self.num_of_nested_loop):
            # Each iteration wraps the loop built so far (the inner loop) in the
            # next one out, matching src/riscv_loop_instr.sv:194-199.  The port
            # appended self.loop_instr[i] -- an element of the list being built,
            # which for i == 0 is loop_init_instr[0].  That put the "init loop
            # counter" instruction *inside* the loop body, so the counter was
            # reset on every iteration and the generated loop never terminated.
            self.loop_instr = ([self.loop_init_instr[2 * i],
                                self.loop_init_instr[2 * i + 1],
                                self.loop_branch_target_instr[i],
                                self.loop_update_instr[i]] +
                               self.loop_instr +
                               [self.loop_branch_instr[i]])
        logging.info("Totally {} instructions have been added".format(len(self.loop_instr)))
