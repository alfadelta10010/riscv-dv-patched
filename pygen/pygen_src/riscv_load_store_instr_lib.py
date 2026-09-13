
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

    # src/riscv_load_store_instr_lib.sv:54-56 constrains the base register:
    #
    #     constraint rs1_c {
    #       !(rs1_reg inside {cfg.reserved_regs, reserved_rd, ZERO});
    #     }
    #
    # The port left it commented out ("pyvsc error --> rs1 has not been build
    # yet"), so nothing stopped the solver picking a base register the stream
    # must not touch. Two distinct failures follow, and both were observed in a
    # single generated riscv_mmu_stress_test program with 496 base-register
    # initialisations:
    #
    #   * rs1_reg == ZERO emits `la zero, region_1+3609`, which the assembler
    #     rejects outright -- the whole program is lost.
    #   * rs1_reg in cfg.reserved_regs rewrites that register with a data page
    #     address. cfg.reserved_regs is {cfg.tp, cfg.sp, cfg.scratch_reg}
    #     (riscv_instr_gen_config.py:472), and those three are themselves
    #     randomized, so they are whichever registers this program picked to
    #     hold the stack pointer, thread pointer and trap scratch value. The
    #     program still assembles and still runs, so it is reported as a pass,
    #     but from that point it executes on a stack pointer aimed into the
    #     data section -- including every trap handler that pushes the GPRs to
    #     the kernel stack. The stimulus the test claims to provide is not what
    #     runs.
    #
    # Measured on one generated riscv_mmu_stress_test program with 496 base
    # register initialisations, whose reserved set was {s9, a4, t6}: 4 used x0
    # and 21 used one of those three. After this patch, a comparable program
    # with 589 initialisations used neither.
    #
    # Enforcing this as a pyvsc constraint hits the same build-order problem the
    # original TODO describes, so it is applied to the solved value instead:
    # legalize_rs1() runs first thing in post_randomize(), before rs1_reg is
    # used to emit anything.
    def legalize_rs1(self):
        illegal = {int(riscv_reg_t.ZERO)}
        illegal.update(int(r) for r in cfg.reserved_regs)
        illegal.update(int(r) for r in self.reserved_rd)
        if int(self.rs1_reg) not in illegal:
            return
        allowed = [r for r in riscv_reg_t if int(r) not in illegal]
        if not allowed:
            logging.critical("No legal rs1 for a load/store stream: every "
                             "register is reserved")
            sys.exit(1)
        self.rs1_reg = random.choice(allowed)

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
        # src/riscv_load_store_instr_lib.sv:101 is
        # `if (SP inside {cfg.reserved_regs, reserved_rd})`, i.e. is SP a member
        # of either list. The port wrote `SP in [cfg.reserved_regs,
        # self.reserved_rd]`, which asks whether SP *is* one of those two list
        # objects and is therefore always False, so use_sp_as_rs1 stayed live
        # and the sp_c dist kept pulling the base register to SP even when SP
        # was reserved.
        #
        # Note the names are misleading: this tests the architectural SP/x2,
        # while cfg.reserved_regs holds cfg.sp (riscv_instr_gen_config.py:472),
        # which is itself a randomized GPR and rarely x2. So the guard is meant
        # to fire only in the small fraction of programs where the randomized
        # sp or tp lands on x2 -- roughly 2% in practice.
        if (riscv_reg_t.SP in cfg.reserved_regs or
                riscv_reg_t.SP in self.reserved_rd):
            self.use_sp_as_rs1 = 0
            with vsc.raw_mode():
                self.use_sp_as_rs1.rand_mode = False
            self.sp_rnd_order_c.constraint_mode(False)

    def post_randomize(self):
        self.legalize_rs1()
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
        """Repeat the previous address hazard_ratio percent of the time.

        Two departures from src/riscv_load_store_instr_lib.sv:

        The SystemVerilog draws the hazard decision inside the loop --
        `if ((i > 0) && ($urandom_range(0, 100) < hazard_ratio))` -- so each
        access independently either repeats its predecessor's address or picks
        a fresh one. The port hoisted that draw above the loop, so a single
        value decided the whole stream: with hazard_ratio constrained to
        [20:100], roughly 60% of streams collapsed to every access using one
        identical address and the other 40% contained no address hazard at all.
        A stream whose entire purpose is a *mix* of load/store address hazards
        produced only the two degenerate cases.

        The fresh-address branch also carried the same broken address
        computation as the base class; it now shares the corrected
        offset_range() helper so both stay inside the data page and `addr`
        matches the address the instruction actually forms.
        """
        self.offset = [0] * self.num_load_store
        self.addr = [0] * self.num_load_store
        lo, hi = self.offset_range()
        base = int(self.base)
        for i in range(self.num_load_store):
            # $urandom_range(0, 100) is inclusive at both ends.
            if (i > 0) and (random.randint(0, 100) < self.hazard_ratio):
                self.offset[i] = self.offset[i - 1]
                self.addr[i] = self.addr[i - 1]
            else:
                offset_ = random.randint(lo, hi)
                self.offset[i] = offset_
                self.addr[i] = base + offset_


# Back to back access to multiple data pages.
# Ported from src/riscv_load_store_instr_lib.sv:355-419, which pyflow never
# carried over -- riscv_utils.factory() aborted with "Cannot Create object of
# riscv_multi_page_load_store_instr_stream" and killed any test requesting it,
# including upstream's own riscv_rand_instr_test and riscv_mmu_stress_test.
#
# The SystemVerilog randomises num_of_instr_stream, data_page_id[] and rs1_reg[]
# together under `unique` constraints over dynamically sized arrays. That is the
# shape of constraint pyvsc handles worst (see the KeyError in
# rand_info_builder.visit_expr_array_subscript() behind patch 05), so the
# selection is done in plain Python here. The three constraints it has to
# satisfy are small and exact, so nothing is lost:
#
#   data_page_id.size() == num_of_instr_stream, unique, each < max_data_page_id
#   rs1_reg.size() == num_of_instr_stream, unique, none in cfg.reserved_regs or ZERO
#   num_of_instr_stream inside {[2:8]}                       (reasonable_c)
@vsc.randobj
class riscv_multi_page_load_store_instr_stream(riscv_mem_access_stream):
    # reasonable_c: each page needs its own register to hold the base address,
    # so the SV caps the stream count rather than running out of registers.
    MIN_NUM_OF_INSTR_STREAM = 2
    MAX_NUM_OF_INSTR_STREAM = 8

    def __init__(self):
        super().__init__()
        self.num_of_instr_stream = 0
        self.data_page_id = []
        self.rs1_reg = []
        self.load_store_instr_stream = []

    def select_data_page_id(self, num_of_instr_stream):
        """page_c: unique {data_page_id}, each < max_data_page_id."""
        return random.sample(range(int(self.max_data_page_id)),
                             num_of_instr_stream)

    def max_num_of_instr_stream(self, num_avail_regs):
        """Largest stream count for which both unique constraints can hold."""
        return min(self.MAX_NUM_OF_INSTR_STREAM,
                   int(self.max_data_page_id), num_avail_regs)

    def post_randomize(self):
        # default_c: rs1_reg values are unique and avoid the reserved set.
        avail_regs = [r for r in riscv_reg_t
                      if r != riscv_reg_t.ZERO and
                      r not in cfg.reserved_regs and
                      r not in self.reserved_rd]
        upper = self.max_num_of_instr_stream(len(avail_regs))
        if upper < self.MIN_NUM_OF_INSTR_STREAM:
            logging.critical("Cannot create a multi-page stream: %0d data "
                             "pages and %0d available registers",
                             int(self.max_data_page_id), len(avail_regs))
            sys.exit(1)
        self.num_of_instr_stream = random.randint(
            self.MIN_NUM_OF_INSTR_STREAM, upper)
        self.data_page_id = self.select_data_page_id(self.num_of_instr_stream)
        self.rs1_reg = random.sample(avail_regs, self.num_of_instr_stream)

        self.load_store_instr_stream = []
        for i in range(self.num_of_instr_stream):
            substream = riscv_load_store_stress_instr_stream()
            substream.hart = self.hart
            substream.kernel_mode = self.kernel_mode
            substream.name = "{}_load_store_instr_stream_{}".format(self.name, i)
            # The SV drops sp_c so rs1 is free to take the value this stream
            # assigns it rather than being pulled to SP by the sp dist.
            substream.sp_c.constraint_mode(False)
            substream.sp_rnd_order_c.constraint_mode(False)
            # legal_c bakes min_instr_cnt/max_instr_cnt into the constraint
            # model when the object is built, so assigning them afterwards --
            # as the SV does -- would not reach the solver (the defect behind
            # patch 07). Impose the SV's 5..10 through randomize_with instead.
            substream.legal_c.constraint_mode(False)
            # Make sure each load/store sequence doesn't override the rs1 of
            # the other sequences.
            for j in range(self.num_of_instr_stream):
                if i != j and self.rs1_reg[j] not in substream.reserved_rd:
                    substream.reserved_rd.append(self.rs1_reg[j])
            try:
                with substream.randomize_with() as it:
                    substream.num_load_store.inside(vsc.rangelist(vsc.rng(5, 10)))
                    substream.num_mixed_instr == 0
                    substream.rs1_reg == self.rs1_reg[i]
                    substream.data_page_id == self.data_page_id[i]
            except Exception:
                logging.critical("Cannot randomize load/store instruction")
                sys.exit(1)
            self.load_store_instr_stream.append(substream)
            # Mix the instruction streams of the different page accesses.
            if i == 0:
                self.instr_list = list(substream.instr_list)
            else:
                self.mix_instr_stream(substream.instr_list)
        super().post_randomize()


# Access different locations of the same memory region.
# src/riscv_load_store_instr_lib.sv:424-438: same as the multi-page stream
# except that every sub-stream targets the same page, and there are fewer of
# them. Overriding page_c in the SV replaces both the range and the uniqueness.
@vsc.randobj
class riscv_mem_region_stress_test(riscv_multi_page_load_store_instr_stream):
    MIN_NUM_OF_INSTR_STREAM = 2
    MAX_NUM_OF_INSTR_STREAM = 5

    def __init__(self):
        super().__init__()

    def select_data_page_id(self, num_of_instr_stream):
        """page_c: data_page_id[i] == data_page_id[i-1] -- one page throughout."""
        return [random.randrange(int(self.max_data_page_id))] * num_of_instr_stream

    def max_num_of_instr_stream(self, num_avail_regs):
        # Only the register uniqueness still binds: every sub-stream shares one
        # page, so max_data_page_id does not cap the count here.
        return min(self.MAX_NUM_OF_INSTR_STREAM, num_avail_regs)
