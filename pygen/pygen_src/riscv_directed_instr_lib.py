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
import random
import logging
import vsc
from importlib import import_module
from enum import IntEnum, auto
from pygen_src.riscv_instr_stream import riscv_rand_instr_stream
from pygen_src.isa.riscv_instr import riscv_instr
from pygen_src.riscv_instr_gen_config import cfg
from pygen_src.riscv_instr_pkg import (riscv_reg_t, riscv_pseudo_instr_name_t,
                                       riscv_instr_name_t, riscv_instr_category_t,
                                       mem_region_t, pkg_ins)
from pygen_src.riscv_pseudo_instr import riscv_pseudo_instr
rcs = import_module("pygen_src.target." + cfg.argv.target + ".riscv_core_setting")


# Base class for directed instruction stream
class riscv_directed_instr_stream(riscv_rand_instr_stream):

    label = ""

    def __init__(self):
        super().__init__()
        self.name = ""

    def post_randomize(self):
        for i in range(len(self.instr_list)):
            self.instr_list[i].has_label = 0
            self.instr_list[i].atomic = 1
        self.instr_list[0].comment = "Start %0s" % (self.name)
        self.instr_list[-1].comment = "End %0s" % (self.name)
        if riscv_directed_instr_stream.label != "":
            self.instr_list[0].label = riscv_directed_instr_stream.label
            self.instr_list[0].has_label = 1


# Base class for memory access stream
@vsc.randobj
class riscv_mem_access_stream(riscv_directed_instr_stream):
    def __init__(self):
        super().__init__()
        self.max_data_page_id = vsc.int32_t()
        self.load_store_shared_memory = 0
        self.data_page = vsc.list_t(mem_region_t())

    def pre_randomize(self):
        self.data_page.clear()
        if self.load_store_shared_memory:
            self.data_page.extend(cfg.amo_region)
        elif self.kernel_mode:
            self.data_page.extend(cfg.s_mem_region)
        else:
            self.data_page.extend(cfg.mem_region)
        self.max_data_page_id = len(self.data_page)

    # Use "la" instruction to initialize the base regiseter
    def add_rs1_init_la_instr(self, gpr, idx, base = 0):
        la_instr = riscv_pseudo_instr()
        la_instr.pseudo_instr_name = riscv_pseudo_instr_name_t.LA
        la_instr.rd = gpr
        if self.load_store_shared_memory:
            la_instr.imm_str = "{}+{}".format(cfg.amo_region[idx].name, base)
        elif self.kernel_mode:
            la_instr.imm_str = "{}{}+{}".format(pkg_ins.hart_prefix(self.hart),
                                                cfg.s_mem_region[idx].name, base)
        else:
            la_instr.imm_str = "{}{}+{}".format(pkg_ins.hart_prefix(self.hart),
                                                cfg.mem_region[idx].name, base)
        self.instr_list.insert(0, la_instr)

    # Insert some other instructions to mix with mem_access instruction
    def add_mixed_instr(self, instr_cnt):
        self.setup_allowed_instr(1, 1)
        for i in range(instr_cnt):
            instr = riscv_instr()
            instr = self.randomize_instr(instr)
            self.insert_instr(instr)


# Jump from one program to another (src/riscv_directed_instr_lib.sv,
# riscv_jump_instr). Like a load/store, JALR needs a base register loaded with
# the target address and an offset.
#
# instr_c's scalars (gpr, imm, mixed_instr_cnt, enable_branch) are drawn in
# Python: the imm bound depends on the jump picked in the same step, which a
# pyvsc constraint elaborated at construction cannot see. The operands of the
# picked instructions are assigned directly for the same reason riscv_instr
# templates are deep-copied: they are plain values once chosen.
class riscv_jump_instr(riscv_directed_instr_stream):
    def __init__(self):
        super().__init__()
        self.jump = None
        self.addi = None
        self.la = riscv_pseudo_instr()
        self.branch = None
        self.gpr = riscv_reg_t.ZERO
        self.imm = 0
        self.enable_branch = 0
        self.mixed_instr_cnt = 0
        self.stack_exit_instr = []
        self.target_program_label = ""
        self.idx = 0
        self.use_jalr = 0

    def gen_jump_instr(self):
        # pre_randomize
        if self.use_jalr:
            self.jump = riscv_instr.get_instr(riscv_instr_name_t.JALR)
        elif cfg.disable_compressed_instr or (int(cfg.ra) != riscv_reg_t.RA):
            self.jump = riscv_instr.get_rand_instr(
                include_instr=[riscv_instr_name_t.JAL, riscv_instr_name_t.JALR])
        else:
            self.jump = riscv_instr.get_rand_instr(
                include_instr=[riscv_instr_name_t.JAL, riscv_instr_name_t.JALR,
                               riscv_instr_name_t.C_JALR])
        self.addi = riscv_instr.get_instr(riscv_instr_name_t.ADDI)
        self.branch = riscv_instr.get_rand_instr(
            include_instr=[riscv_instr_name_t.BEQ, riscv_instr_name_t.BNE,
                           riscv_instr_name_t.BLT, riscv_instr_name_t.BGE,
                           riscv_instr_name_t.BLTU, riscv_instr_name_t.BGEU])
        jump_name = riscv_instr_name_t(int(self.jump.instr_name))
        # instr_c
        reserved = [int(r) for r in cfg.reserved_regs]
        self.gpr = random.choice([r for r in riscv_reg_t
                                  if r != riscv_reg_t.ZERO and int(r) not in reserved])
        if jump_name in (riscv_instr_name_t.C_JR, riscv_instr_name_t.C_JALR):
            self.imm = 0
        else:
            self.imm = random.randint(-1023, 1023)
        self.mixed_instr_cnt = random.randint(5, 10)
        self.enable_branch = random.randint(0, 1)
        # post_randomize
        ra = riscv_reg_t(int(cfg.ra))
        self.jump.rd = ra            # if (has_rd)  rd == cfg.ra
        self.jump.rs1 = self.gpr     # if (has_rs1) rs1 == gpr
        self.addi.rd = self.gpr
        self.addi.rs1 = self.gpr
        self.branch.rs1 = random.choice(list(riscv_reg_t))
        self.branch.rs2 = random.choice(list(riscv_reg_t))
        self.la.pseudo_instr_name = riscv_pseudo_instr_name_t.LA
        self.la.imm_str = self.target_program_label
        self.la.rd = self.gpr
        # Generate some random instructions to mix with jump instructions
        self.reserved_rd = [self.gpr]
        self.initialize_instr_list(self.mixed_instr_cnt)
        self.gen_instr(1)
        if jump_name in (riscv_instr_name_t.JALR, riscv_instr_name_t.C_JALR):
            # JALR is expected to set lsb to 0
            self.addi.imm_str = "{}".format(self.imm + random.randint(0, 1))
        else:
            self.addi.imm_str = "{}".format(self.imm)
        if getattr(cfg, "enable_misaligned_instr", 0):
            # Jump to a misaligned address
            self.jump.imm_str = "{}".format(-self.imm + 2)
        else:
            self.jump.imm_str = "{}".format(-self.imm)
        # The branch is placed after la/addi/stack exit (mix_instr_stream keeps
        # the list order), so taking it cannot skip loading the jump base.
        instr = [self.branch] if self.enable_branch else []
        # Restore stack before unconditional jump
        if ra == riscv_reg_t.ZERO or jump_name == riscv_instr_name_t.C_JR:
            instr = list(self.stack_exit_instr) + instr
        if jump_name == riscv_instr_name_t.JAL:
            self.jump.imm_str = self.target_program_label
        else:
            instr = [self.la, self.addi] + instr
        self.mix_instr_stream(instr)
        self.instr_list.append(self.jump)
        for i in range(len(self.instr_list)):
            self.instr_list[i].has_label = 0
            self.instr_list[i].atomic = 1
        self.jump.has_label = 1
        self.jump.label = "{}_j{}".format(self.label, self.idx)
        self.jump.comment = "jump {} -> {}".format(self.label, self.target_program_label)
        self.branch.imm_str = self.jump.label
        self.branch.comment = "branch to jump instr"
        self.branch.branch_assigned = 1


# Stress back to back jump instruction
@vsc.randobj
class riscv_jal_instr(riscv_rand_instr_stream):
    def __init__(self):
        super().__init__()
        self.name = ""
        self.jump = []
        self.jump_start = riscv_instr()
        self.jump_end = riscv_instr()
        self.num_of_jump_instr = vsc.rand_int_t()
        self.jal = []

    @vsc.constraint
    def instr_c(self):
        self.num_of_jump_instr in vsc.rangelist(vsc.rng(10, 30))

    def post_randomize(self):
        order = []
        RA = cfg.ra
        order = [0] * self.num_of_jump_instr
        self.jump = [0] * self.num_of_jump_instr
        for i in range(len(order)):
            order[i] = i
        random.shuffle(order)
        self.setup_allowed_instr(1, 1)
        jal = [riscv_instr_name_t.JAL]
        if not cfg.disable_compressed_instr:
            jal.append(riscv_instr_name_t.C_J)
            if rcs.XLEN == 32:
                jal.append(riscv_instr_name_t.C_JAL)

        # First instruction
        self.jump_start = riscv_instr.get_instr(riscv_instr_name_t.JAL)
        with self.jump_start.randomize_with():
            self.jump_start.rd == RA
        self.jump_start.imm_str = "{}f".format(order[0])
        self.jump_start.label = self.label

        # Last instruction
        self.jump_end = self.randomize_instr(self.jump_end)
        self.jump_end.label = "{}".format(self.num_of_jump_instr)
        # src/riscv_directed_instr_lib.sv:248-252:
        #   if (has_rd) {
        #     rd dist {RA := 5, T1 := 2, [SP:T0] :/ 1, [T2:T6] :/ 2};
        #     !(rd inside {cfg.reserved_regs});
        #   }
        # The port left out T1 := 2, and vsc.dist only nudges the solver, so
        # draw rd from these weights over the non-reserved registers and pin it,
        # as cfg.post_randomize does for ra_c.
        reserved = {int(r) for r in cfg.reserved_regs}
        rd_weights = {riscv_reg_t.RA: 5.0, riscv_reg_t.T1: 2.0}
        for r in riscv_reg_t:
            if riscv_reg_t.SP <= r <= riscv_reg_t.T0:
                rd_weights[r] = 1.0 / (riscv_reg_t.T0 - riscv_reg_t.SP + 1)
            elif riscv_reg_t.T2 <= r <= riscv_reg_t.T6:
                rd_weights[r] = 2.0 / (riscv_reg_t.T6 - riscv_reg_t.T2 + 1)
        rd_legal = [r for r in rd_weights if int(r) not in reserved]
        for i in range(self.num_of_jump_instr):
            # SV: get_rand_instr(.include_instr({jal})) -- the whole list,
            # {JAL, C_J, C_JAL} on RV32 with compressed instructions enabled.
            self.jump[i] = riscv_instr.get_rand_instr(include_instr = jal)
            if self.jump[i].has_rd:
                rd = random.choices(rd_legal, [rd_weights[r] for r in rd_legal])[0]
                with self.jump[i].randomize_with():
                    self.jump[i].rd == rd
            else:
                self.jump[i].randomize()
            self.jump[i].label = "{}".format(i)

        for i in range(len(order)):
            if i == self.num_of_jump_instr - 1:
                self.jump[order[i]].imm_str = "{}f".format(self.num_of_jump_instr)
            else:
                if order[i + 1] > order[i]:
                    self.jump[order[i]].imm_str = "{}f".format(order[i + 1])
                else:
                    self.jump[order[i]].imm_str = "{}b".format(order[i + 1])
        self.instr_list.append(self.jump_start)
        self.instr_list.extend(self.jump)
        self.instr_list.append(self.jump_end)
        for i in range(len(self.instr_list)):
            self.instr_list[i].has_label = 1
            self.instr_list[i].atomic = 1


class int_numeric_e(IntEnum):
    NormalValue = auto()
    Zero = auto()
    AllOne = auto()
    NegativeMax = auto()


# Strees the numeric corner cases of integer arithmetic operations
@vsc.randobj
class riscv_int_numeric_corner_stream(riscv_directed_instr_stream):
    def __init__(self):
        super().__init__()
        self.num_of_avail_regs = 10
        self.num_of_instr = vsc.rand_uint8_t()
        # src/riscv_directed_instr_lib.sv:431 `rand bit [XLEN-1:0] init_val[]`.
        # XLEN - 1 made a 31-bit field: 1 << (XLEN-1) truncated to 0 and no
        # value with bit 31 set could be drawn.
        self.init_val = vsc.randsz_list_t(vsc.rand_bit_t(rcs.XLEN))
        self.init_val_type = vsc.randsz_list_t(vsc.enum_t(int_numeric_e))
        self.init_instr = []

    @vsc.constraint
    def init_val_c(self):
        # TODO
        vsc.solve_order(self.init_val_type, self.init_val)
        self.init_val_type.size == 10  # self.num_of_avail_regs
        self.init_val.size == 10  # self.num_of_avail_regs
        self.num_of_instr in vsc.rangelist(vsc.rng(15, 30))

    @vsc.constraint
    def avail_regs_c(self):
        # TODO
        self.avail_regs.size == 10  # self.num_of_avail_regs
        vsc.unique(self.avail_regs)
        with vsc.foreach(self.avail_regs, idx = True) as i:
            self.avail_regs[i].not_inside(cfg.reserved_regs)
            self.avail_regs[i] != riscv_reg_t.ZERO

    def pre_randomize(self):
        # TODO
        pass

    def post_randomize(self):
        self.init_instr = [None] * self.num_of_avail_regs
        for i in range(len(self.init_val_type)):
            if self.init_val_type[i] == int_numeric_e.Zero:
                self.init_val[i] = 0
            elif self.init_val_type[i] == int_numeric_e.AllOne:
                # SV: init_val[i] = '1 (every bit set), not 1.
                self.init_val[i] = (1 << rcs.XLEN) - 1
            elif self.init_val_type[i] == int_numeric_e.NegativeMax:
                self.init_val[i] = 1 << (rcs.XLEN - 1)
            self.init_instr[i] = riscv_pseudo_instr()
            self.init_instr[i].rd = self.avail_regs[i]
            self.init_instr[i].pseudo_instr_name = riscv_pseudo_instr_name_t.LI
            self.init_instr[i].imm_str = "0x%0x" % (self.init_val[i])
            self.instr_list.append(self.init_instr[i])
        for i in range(self.num_of_instr):
            instr = riscv_instr.get_rand_instr(
                include_category = ['ARITHMETIC'],
                exclude_group = ['RV32C', 'RV64C', 'RV32F', 'RV64F', 'RV32D', 'RV64D'])
            instr = self.randomize_gpr(instr)
            self.instr_list.append(instr)
        super().post_randomize()


# Push Stack Instructions
class riscv_push_stack_instr(riscv_rand_instr_stream):
    def __init__(self):
        super().__init__()
        self.stack_len = 0
        self.num_of_reg_to_save = 0
        self.num_of_redundant_instr = 0
        self.push_stack_instr = []
        self.saved_regs = []
        self.branch_instr = vsc.attr(riscv_instr())
        self.enable_branch = vsc.rand_bit_t(1)
        self.push_start_label = ''

    def init(self):
        # Save RA, T0
        self.reserved_rd = [cfg.ra]
        self.saved_regs = [cfg.ra]
        self.num_of_reg_to_save = len(self.saved_regs)
        if self.num_of_reg_to_save * (rcs.XLEN / 8) > self.stack_len:
            logging.error('stack len [{}] is not enough to store {} regs'
                          .format(self.stack_len, self.num_of_reg_to_save))
            sys.exit(1)
        # SV $urandom_range(3,10) includes 10; randrange excludes its stop.
        self.num_of_redundant_instr = random.randrange(3, 11)
        self.initialize_instr_list(self.num_of_redundant_instr)

    def gen_push_stack_instr(self, stack_len, allow_branch=1):
        self.stack_len = stack_len
        self.init()
        self.gen_instr(1)
        self.push_stack_instr = [0] * (self.num_of_reg_to_save + 1)
        for i in range(len(self.push_stack_instr)):
            self.push_stack_instr[i] = riscv_instr()
        self.push_stack_instr[0] = \
            riscv_instr.get_instr(riscv_instr_name_t.ADDI)
        with self.push_stack_instr[0].randomize_with():
            self.push_stack_instr[0].rd == cfg.sp
            self.push_stack_instr[0].rs1 == cfg.sp
            self.push_stack_instr[0].imm == ((~self.stack_len + 1) & 0xFFFFFFFF)

        self.push_stack_instr[0].imm_str = '-{}'.format(self.stack_len)
        for i in range(len(self.saved_regs)):
            if rcs.XLEN == 32:
                self.push_stack_instr[i + 1] = riscv_instr.get_instr(riscv_instr_name_t.SW)
                with self.push_stack_instr[i + 1].randomize_with():
                    self.push_stack_instr[i + 1].rs2 == self.saved_regs[i]
                    self.push_stack_instr[i + 1].rs1 == cfg.sp
                    self.push_stack_instr[i + 1].imm == 4 * (i + 1)
            else:
                self.push_stack_instr[i + 1] = riscv_instr.get_instr(riscv_instr_name_t.SD)
                with self.push_stack_instr[i + 1].randomize_with():
                    self.push_stack_instr[i + 1].instr_name == riscv_instr_name_t.SD
                    self.push_stack_instr[i + 1].rs2 == self.saved_regs[i]
                    self.push_stack_instr[i + 1].rs1 == cfg.sp
                    self.push_stack_instr[i + 1].imm == 8 * (i + 1)

            self.push_stack_instr[i + 1].process_load_store = 0
        if allow_branch:
            # src/riscv_directed_instr_lib.sv:334 randomizes this bit.
            # randrange(0, 1) is always 0, so the push-stack sequence never
            # contained a branch.
            self.enable_branch = random.randrange(2)
        else:
            self.enable_branch = 0
        if self.enable_branch:
            self.branch_instr = \
                riscv_instr.get_rand_instr(include_category=[riscv_instr_category_t.BRANCH.name])
            self.branch_instr.randomize()
            self.branch_instr.imm_str = self.push_start_label
            self.branch_instr.branch_assigned = 1
            self.push_stack_instr[0].label = self.push_start_label
            self.push_stack_instr[0].has_label = 1
            self.push_stack_instr = [self.branch_instr] + self.push_stack_instr
        self.mix_instr_stream(self.push_stack_instr)
        for i in range(len(self.instr_list)):
            self.instr_list[i].atomic = 1
            if self.instr_list[i].label == '':
                self.instr_list[i].has_label = 0


# Pop stack instruction stream
class riscv_pop_stack_instr(riscv_rand_instr_stream):
    def __init__(self):
        super().__init__()
        self.stack_len = 0
        self.num_of_reg_to_save = 0
        self.num_of_redundant_instr = 0
        self.pop_stack_instr = []
        self.saved_regs = []

    def init(self):
        self.reserved_rd = [cfg.ra]
        self.num_of_reg_to_save = len(self.saved_regs)
        if self.num_of_reg_to_save * 4 > self.stack_len:
            logging.error('stack len [{}] is not enough to store {} regs'
                          .format(self.stack_len, self.num_of_reg_to_save))
            sys.exit(1)
        # SV $urandom_range(3,10) includes 10; randrange excludes its stop.
        self.num_of_redundant_instr = random.randrange(3, 11)
        self.initialize_instr_list(self.num_of_redundant_instr)

    def gen_pop_stack_instr(self, stack_len, saved_regs):
        self.stack_len = stack_len
        self.saved_regs = saved_regs
        self.init()
        self.gen_instr(1)
        self.pop_stack_instr = [0] * (self.num_of_reg_to_save + 1)
        for i in range(len(self.pop_stack_instr)):
            self.pop_stack_instr[i] = riscv_instr()
        for i in range(len(self.saved_regs)):
            if rcs.XLEN == 32:
                self.pop_stack_instr[i] = riscv_instr.get_instr(riscv_instr_name_t.LW)
                with self.pop_stack_instr[i].randomize_with():
                    self.pop_stack_instr[i].rd == self.saved_regs[i]
                    self.pop_stack_instr[i].rs1 == cfg.sp
                    self.pop_stack_instr[i].imm == 4 * (i + 1)
            else:
                self.pop_stack_instr[i] = riscv_instr.get_instr(riscv_instr_name_t.LD)
                with self.pop_stack_instr[i].randomize_with():
                    self.pop_stack_instr[i].rd == self.saved_regs[i]
                    self.pop_stack_instr[i].rs1 == cfg.sp
                    self.pop_stack_instr[i].imm == 8 * (i + 1)
            self.pop_stack_instr[i].process_load_store = 0
        # addi sp,sp,imm
        self.pop_stack_instr[self.num_of_reg_to_save] = riscv_instr.get_instr(
            riscv_instr_name_t.ADDI)
        with self.pop_stack_instr[self.num_of_reg_to_save].randomize_with():
            self.pop_stack_instr[self.num_of_reg_to_save].rd == cfg.sp
            self.pop_stack_instr[self.num_of_reg_to_save].rs1 == cfg.sp
            self.pop_stack_instr[self.num_of_reg_to_save].imm == self.stack_len
        self.pop_stack_instr[self.num_of_reg_to_save].imm_str = pkg_ins.format_string(
            '{}'.format(self.stack_len))
        self.mix_instr_stream(self.pop_stack_instr)
        for i in range(len(self.instr_list)):
            self.instr_list[i].atomic = 1
            self.instr_list[i].has_label = 0
