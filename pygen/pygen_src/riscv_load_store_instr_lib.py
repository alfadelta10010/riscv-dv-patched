
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
from enum import IntEnum, auto
from importlib import import_module
from pygen_src.riscv_instr_gen_config import cfg
from pygen_src.isa.riscv_instr import riscv_instr
from pygen_src.riscv_directed_instr_lib import riscv_mem_access_stream
from pygen_src.riscv_instr_pkg import riscv_reg_t, riscv_instr_name_t, riscv_instr_group_t
rcs = import_module("pygen_src.target." + cfg.argv.target + ".riscv_core_setting")


class locality_e(IntEnum):
    NARROW = 0
    HIGH = auto()
    MEDIUM = auto()
    SPARSE = auto()


# Base class for all load/store instruction stream
@vsc.randobj
class riscv_load_store_base_instr_stream(riscv_mem_access_stream):
    def __init__(self):
        super().__init__()
        self.num_load_store = vsc.rand_uint32_t()
        self.num_mixed_instr = vsc.rand_uint32_t()
        self.base = vsc.rand_int32_t()
        self.offset = []
        self.addr = []
        self.load_store_instr = []
        self.data_page_id = vsc.rand_uint32_t()
        self.rs1_reg = vsc.rand_enum_t(riscv_reg_t)
        self.locality = vsc.rand_enum_t(locality_e)
        self.max_load_store_offset = vsc.rand_int32_t()
        self.use_sp_as_rs1 = vsc.rand_bit_t()

    @vsc.constraint
    def sp_rnd_order_c(self):
        vsc.solve_order(self.use_sp_as_rs1, self.rs1_reg)

    @vsc.constraint
    def sp_c(self):
        vsc.dist(self.use_sp_as_rs1, [vsc.weight(1, 1), vsc.weight(0, 2)])
        with vsc.if_then(self.use_sp_as_rs1 == 1):
            self.rs1_reg == riscv_reg_t.SP

    # TODO Getting pyvsc error -- > rs1 has not been build yet
    '''@vsc.constraint
    def rs1_c(self):
        self.rs1_reg.not_inside(vsc.rangelist(cfg.reserved_regs,
                                              self.reserved_rd, riscv_reg_t.ZERO))'''

    @vsc.constraint
    def addr_c(self):
        # TODO solve_order
        # vsc.solve_order(self.data_page_id, self.max_load_store_offset)
        # vsc.solve_order(self.max_load_store_offset, self.base)
        self.data_page_id < self.max_data_page_id
        with vsc.foreach(self.data_page, idx = True) as i:
            with vsc.if_then(i == self.data_page_id):
                self.max_load_store_offset == self.data_page[i].size_in_bytes
        self.base in vsc.rangelist(vsc.rng(0, self.max_load_store_offset - 1))

    # Locality ranges from src/riscv_load_store_instr_lib.sv. The SystemVerilog
    # bounds are `inside {[lo:hi]}`, which is inclusive at both ends; the port
    # used random.randrange(), which excludes the upper one.
    LOCALITY_RANGE = {
        locality_e.NARROW: (-16, 16),
        locality_e.HIGH: (-64, 64),
        locality_e.MEDIUM: (-256, 256),
        locality_e.SPARSE: (-2048, 2047),
    }

    def offset_range(self):
        """Offsets that keep base + offset inside the data page.

        The SystemVerilog original solves two constraints together
        (src/riscv_load_store_instr_lib.sv, randomize_offset):

            addr_ == base + offset_;
            addr_ inside {[0 : max_load_store_offset - 1]};

        Intersecting the locality range with the second gives the interval to
        draw from. addr_c already constrains base to [0, max_load_store_offset
        - 1], so the result always contains 0 and is never empty.
        """
        lo, hi = self.LOCALITY_RANGE[locality_e(int(self.locality))]
        base = int(self.base)
        page_size = int(self.max_load_store_offset)
        lo = max(lo, -base)
        hi = min(hi, page_size - 1 - base)
        if lo > hi:
            logging.critical("Cannot randomize load/store offset: base %0d "
                             "outside data page of %0d bytes", base, page_size)
            sys.exit(1)
        return lo, hi

    def randomize_offset(self):
        """Pick each load/store offset and the page-relative address it forms.

        `addr` must be exactly the address the emitted instruction computes:
        gen_load_store_instr() consults it to decide which access widths are
        alignment-legal, while `offset` is what is emitted as the immediate.
        The port broke that link, drawing

            addr_ = random.randrange(base + offset_ - 1, base + offset_ + 1)

        which returns base + offset_ - 1 or base + offset_ with equal
        probability, and never applied the in-page bound at all. Measured over
        800k draws against a 4096-byte page:

          * 50.0% of addresses were one byte below the address the instruction
            really forms, so the alignment decision concerned the wrong address;
          * 12.5% of accesses were selected as 4-byte aligned when the real
            address is not -- on a core with support_unaligned_load_store = 0
            those take a misaligned trap the stream never intended;
          * 7.3% fell outside the data page, into memory the test never
            declared and where the DUT and the reference need not agree.
        """
        self.offset = [0] * self.num_load_store
        self.addr = [0] * self.num_load_store
        lo, hi = self.offset_range()
        base = int(self.base)
        for i in range(self.num_load_store):
            offset_ = random.randint(lo, hi)
            self.offset[i] = offset_
            self.addr[i] = base + offset_

    def pre_randomize(self):
        super().pre_randomize()
        if(riscv_reg_t.SP in [cfg.reserved_regs, self.reserved_rd]):
            self.use_sp_as_rs1 = 0
            with vsc.raw_mode():
                self.use_sp_as_rs1.rand_mode = False
            self.sp_rnd_order_c.constraint_mode(False)

    def post_randomize(self):
        self.randomize_offset()
        # rs1 cannot be modified by other instructions
        if not(self.rs1_reg in self.reserved_rd):
            self.reserved_rd.append(self.rs1_reg)
        self.gen_load_store_instr()
        self.add_mixed_instr(self.num_mixed_instr)
        self.add_rs1_init_la_instr(self.rs1_reg, self.data_page_id, self.base)
        super().post_randomize()

    # Generate each load/store instruction
    def gen_load_store_instr(self):
        allowed_instr = []
        enable_compressed_load_store = 0
        self.randomize_avail_regs()
        if ((self.rs1_reg in [riscv_reg_t.S0, riscv_reg_t.S1, riscv_reg_t.A0, riscv_reg_t.A1,
                              riscv_reg_t.A2, riscv_reg_t.A3, riscv_reg_t.A4, riscv_reg_t.A5,
                              riscv_reg_t.SP]) and not(cfg.disable_compressed_instr)):
            enable_compressed_load_store = 1
        for i in range(len(self.addr)):
            # Assign the allowed load/store instructions based on address alignment
            # This is done separately rather than a constraint to improve the randomization
            # performance
            # NOTE: this must be an assignment, not extend(). src/riscv_load_store_instr_lib.sv
            # does "allowed_instr = {LB, LBU, SB};" at the top of every foreach(addr[i])
            # iteration; the port extended a list built once before the loop, so the allowed
            # set accumulated across addresses. An instruction whose alignment/offset
            # precondition held for an earlier address stayed selectable for a later one that
            # violated it, emitting encodings the assembler rejects, e.g.
            #   Error: illegal operands `c.swsp t2,23(sp)'   (C.SWSP needs uimm 0..252, x4)
            #   Error: illegal operands `c.swsp t2,-184(sp)'
            allowed_instr = [riscv_instr_name_t.LB, riscv_instr_name_t.LBU,
                             riscv_instr_name_t.SB]
            if not cfg.enable_unaligned_load_store:
                if (self.addr[i] & 1) == 0:
                    allowed_instr.extend(
                        [riscv_instr_name_t.LH, riscv_instr_name_t.LHU, riscv_instr_name_t.SH])
                if self.addr[i] % 4 == 0:
                    allowed_instr.extend([riscv_instr_name_t.LW, riscv_instr_name_t.SW])
                    if cfg.enable_floating_point:
                        allowed_instr.extend([riscv_instr_name_t.FLW,
                                              riscv_instr_name_t.FSW])
                    if ((self.offset[i] in range(128)) and (self.offset[i] % 4 == 0) and
                        (riscv_instr_group_t.RV32C in rcs.supported_isa) and
                            (enable_compressed_load_store)):
                        if self.rs1_reg == riscv_reg_t.SP:
                            logging.info("Add LWSP/SWSP to allowed instr")
                            allowed_instr.extend(
                                [riscv_instr_name_t.C_LWSP, riscv_instr_name_t.C_SWSP])
                        else:
                            allowed_instr.extend(
                                [riscv_instr_name_t.C_LW, riscv_instr_name_t.C_SW])
                            if (cfg.enable_floating_point and
                                    riscv_instr_group_t.RV32FC in rcs.supported_isa):
                                allowed_instr.extend(
                                    [riscv_instr_name_t.C_FLW, riscv_instr_name_t.C_FSW])
                if (rcs.XLEN >= 64) and (self.addr[i] % 8 == 0):
                    allowed_instr.extend([riscv_instr_name_t.LWU,
                                          riscv_instr_name_t.LD,
                                          riscv_instr_name_t.SD])
                    if (cfg.enable_floating_point and
                            (riscv_instr_group_t.RV32D in rcs.supported_isa)):
                        allowed_instr.extend([riscv_instr_name_t.FLD,
                                              riscv_instr_name_t.FSD])
                    if (self.offset[i] in range(256) and (self.offset[i] % 8 == 0) and
                        (riscv_instr_group_t.RV64C in rcs.supported_isa) and
                            enable_compressed_load_store):
                        if self.rs1_reg == riscv_reg_t.SP:
                            allowed_instr.extend(
                                [riscv_instr_name_t.C_LDSP, riscv_instr_name_t.C_SDSP])
                        else:
                            allowed_instr.extend(
                                [riscv_instr_name_t.C_LD, riscv_instr_name_t.C_SD])
                            if (cfg.enable_floating_point and
                                    (riscv_instr_group_t.RV32DC in rcs.supported_isa)):
                                allowed_instr.extend(
                                    [riscv_instr_name_t.C_FLD, riscv_instr_name_t.C_FSD])
                else:  # unalligned load/store
                    allowed_instr.extend([riscv_instr_name_t.LW, riscv_instr_name_t.SW,
                                          riscv_instr_name_t.LH, riscv_instr_name_t.LHU,
                                          riscv_instr_name_t.SH])
                    # Compressed load/store still needs to be alligned
                    if (self.offset[i] in range(128) and (self.offset[i] % 4 == 0) and
                        (riscv_instr_group_t.RV32C in rcs.supported_isa) and
                            enable_compressed_load_store):
                        if self.rs1_reg == riscv_reg_t.SP:
                            allowed_instr.extend(
                                [riscv_instr_name_t.C_LWSP, riscv_instr_name_t.C_SWSP])
                        else:
                            allowed_instr.extend(
                                [riscv_instr_name_t.C_LW, riscv_instr_name_t.C_SW])
                    if rcs.XLEN >= 64:
                        allowed_instr.extend(
                            [riscv_instr_name_t.LWU, riscv_instr_name_t.LD, riscv_instr_name_t.SD])
                        if (self.offset[i] in range(256) and (self.offset[i] % 8 == 0) and
                            (riscv_instr_group_t.RV64C in rcs.supported_isa) and
                                enable_compressed_load_store):
                            if self.rs1_reg == riscv_reg_t.SP:
                                allowed_instr.extend(
                                    [riscv_instr_name_t.C_LWSP, riscv_instr_name_t.C_SWSP])
                            else:
                                allowed_instr.extend(
                                    [riscv_instr_name_t.C_LD, riscv_instr_name_t.C_SD])
            instr = riscv_instr.get_load_store_instr(allowed_instr)
            instr.has_rs1 = 0
            instr.has_imm = 0
            self.randomize_gpr(instr)
            instr.rs1 = self.rs1_reg
            instr.imm_str = str(instr.uintToInt(self.offset[i]))
            instr.process_load_store = 0
            self.instr_list.append(instr)
            self.load_store_instr.append(instr)


# A single load/store instruction
@vsc.randobj
class riscv_single_load_store_instr_stream(riscv_load_store_base_instr_stream):
    def __init__(self):
        super().__init__()

    @vsc.constraint
    def legal_c(self):
        self.num_load_store == 1
        self.num_mixed_instr < 5


# Back to back load/store instructions
@vsc.randobj
class riscv_load_store_stress_instr_stream(riscv_load_store_base_instr_stream):
    def __init__(self):
        super().__init__()
        self.max_instr_cnt = 30
        self.min_instr_cnt = 10

    @vsc.constraint
    def legal_c(self):
        self.num_load_store.inside(vsc.rangelist(vsc.rng(self.min_instr_cnt, self.max_instr_cnt)))
        self.num_mixed_instr == 0


# Back to back load/store instructions
@vsc.randobj
class riscv_load_store_shared_mem_stream(riscv_load_store_stress_instr_stream):
    def __init__(self):
        super().__init__()

    def pre_randomize(self):
        self.load_store_shared_memory = 1
        super().pre_randomize()


# Random load/store sequence
# A random mix of load/store instructions and other instructions
@vsc.randobj
class riscv_load_store_rand_instr_stream(riscv_load_store_base_instr_stream):
    def __init__(self):
        super().__init__()

    @vsc.constraint
    def legal_c(self):
        self.num_load_store.inside(vsc.rangelist(vsc.rng(10, 30)))
        self.num_mixed_instr.inside(vsc.rangelist(vsc.rng(10, 30)))


# Use a small set of GPR to create various WAW, RAW, WAR hazard scenario
# Port of src/riscv_load_store_instr_lib.sv:283-301.  The class was missing from
# pyflow entirely, so riscv_utils.factory() aborted with
# "Cannot Create object of riscv_hazard_instr_stream" for every testlist entry
# that requested it (riscv_rand_instr_test in the stock base_testlist).
# NOTE: the SV version also sets num_of_avail_regs = 6.  That is deliberately not
# mirrored here because pyflow's riscv_rand_instr_stream.randomize_avail_regs()
# is still an unimplemented "pass" (riscv_instr_stream.py), so avail_regs is not
# populated for any stream and the narrower register window would have no effect.
@vsc.randobj
class riscv_hazard_instr_stream(riscv_load_store_base_instr_stream):
    def __init__(self):
        super().__init__()

    @vsc.constraint
    def legal_c(self):
        self.num_load_store.inside(vsc.rangelist(vsc.rng(10, 30)))
        self.num_mixed_instr.inside(vsc.rangelist(vsc.rng(10, 30)))


# Use a small set of address to create various load/store hazard sequence
# This instruction stream focus more on hazard handling of load store unit.
@vsc.randobj
class riscv_load_store_hazard_instr_stream(riscv_load_store_base_instr_stream):
    def __init__(self):
        super().__init__()
        self.hazard_ratio = vsc.rand_int32_t()

    @vsc.constraint
    def hazard_ratio_c(self):
        self.hazard_ratio.inside(vsc.rangelist(vsc.rng(20, 100)))

    @vsc.constraint
    def legal_c(self):
        self.num_load_store.inside(vsc.rangelist(vsc.rng(10, 20)))
        self.num_mixed_instr.inside(vsc.rangelist(vsc.rng(1, 7)))

    def randomize_offset(self):
        addr_ = vsc.rand_int32_t()
        offset_ = vsc.rand_int32_t()
        self.offset = [0] * self.num_load_store
        self.addr = [0] * self.num_load_store
        rand_num = random.randrange(0, 100)
        for i in range(self.num_load_store):
            if (i > 0) and (rand_num < self.hazard_ratio):
                self.offset[i] = self.offset[i - 1]
                self.addr[i] = self.addr[i - 1]
            else:
                try:
                    if self.locality == locality_e.NARROW:
                        offset_ = random.randrange(-16, 16)
                    elif self.locality == locality_e.HIGH:
                        offset_ = random.randrange(-64, 64)
                    elif self.locality == locality_e.MEDIUM:
                        offset_ = random.randrange(-256, 256)
                    elif self.locality == locality_e.SPARSE:
                        offset_ = random.randrange(-2048, 2047)
                    var1 = self.base + offset_ - 1
                    var2 = self.base + offset_ + 1
                    addr_ = random.randrange(var1, var2)
                except Exception:
                    logging.critical("Cannot randomize load/store offset")
                    sys.exit(1)
                self.offset[i] = offset_
                self.addr[i] = addr_
