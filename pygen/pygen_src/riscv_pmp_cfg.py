"""
PMP configuration for the pyflow generator.

Ported from src/riscv_pmp_cfg.sv, which pyflow never carried over:
riscv_instr_gen_config.py:142 has only `# pmp_cfg = riscv_pmp_cfg  # TODO`, and
none of the --pmp_* options existed, so run.py rejected them at argparse with
"unrecognized arguments" and no PMP stimulus could be generated at all.

Scope of this port, and why:

  * Both the directed path (+pmp_region_<i>) and pmp_randomize = 1. The SV
    randomized path is interlocking constraints over a dynamically sized
    array (sanity_c, xwr_c, allow_high_addrs_c, address_modes_c,
    grain_addr_mode_c, addr_range_c, modes_before_addr_c, addr_legal_tor_c,
    addr_napot_mode_c, addr_na4_mode_c) -- the shape pyvsc handles worst --
    so randomize() draws each region in the SV solve order in plain Python.

  * No ePMP. riscv_instr_pkg::support_epmp gates roughly 150 lines of
    gen_pmp_instr, all of it writing mseccfg for Smepmp. FyraCore does not
    implement mseccfg (src/csr.v decodes no 0x747), so support_epmp is 0 and
    that entire branch is unreachable. The code below is the non-ePMP path.

  * gen_pmp_exception_routine is ported, with the divergences its docstring
    lists. After an access fault it finds the matching entry and grants the
    missing permission, so a restrictive region does not re-fault forever.
"""

import logging
import math
import random
import sys

from pygen_src.riscv_instr_pkg import (pmp_addr_mode_t, privileged_reg_t,
                                       exception_cause_t)


class pmp_cfg_reg_t:
    """One pmpcfg byte plus the address that goes with it.

    Mirrors the SystemVerilog struct, which carries the address alongside the
    config byte for convenience. Layout of the byte, from the privileged spec
    and matching src/riscv_pmp_cfg.sv:

        bit 7    L      lock
        bits 6:5 zero   reserved, written as 0
        bits 4:3 A      OFF / TOR / NA4 / NAPOT
        bit 2    X
        bit 1    W
        bit 0    R
    """

    def __init__(self):
        self.l = 0
        self.a = pmp_addr_mode_t.TOR
        self.x = 0
        self.w = 0
        self.r = 0
        self.addr = 0
        self.offset = 0
        # NAPOT region size / TOR overlap control, as in the SV struct.
        self.addr_mode = 0

    def to_byte(self):
        return ((self.l & 0x1) << 7 | (self.a.value & 0x3) << 3 |
                (self.x & 0x1) << 2 | (self.w & 0x1) << 1 | (self.r & 0x1))


class riscv_pmp_cfg:
    # xlen is passed in rather than read from a riscv_core_setting import,
    # because the config module that constructs this object is itself what
    # resolves the target, and importing it here would be circular.
    def __init__(self, xlen):
        self.xlen = xlen
        # Default to a single PMP region, as the SV does.
        self.pmp_num_regions = 1
        # Default granularity of 0, i.e. a 4-byte grain.
        self.pmp_granularity = 0
        self.pmp_num_regions_given = 0
        self.pmp_randomize = 0
        self.pmp_allow_illegal_tor = 0
        # After a PMP access fault the handler finds the matching entry and sets
        # the missing access bit so execution can continue (SV default: on).
        self.enable_pmp_exception_handler = 1
        # Allow regions above the 32-bit address space when XLEN == 32, in
        # high_addr_proportion percent of randomized configurations.
        self.allow_high_addrs = 0
        self.high_addr_proportion = 10
        self.enable_write_pmp_csr = 0
        self.suppress_pmp_setup = 0
        self.pmp_max_offset = (1 << self.xlen) - 1
        # Number of configuration bytes per pmpcfg CSR: 4 in rv32, 8 in rv64.
        self.cfg_per_csr = self.xlen // 8
        self.pmp_cfg = []
        self.pmp_cfg_addr_valid = []
        self.pmp_cfg_already_configured = []
        # Raw +pmp_region_<i>= strings, keyed by region index.
        self.pmp_region_args = {}
        self.end_signature_addr = 0
        self.base_pmp_addr = privileged_reg_t.PMPADDR0
        self.base_pmpcfg_addr = privileged_reg_t.PMPCFG0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def initialize(self, signature_addr):
        self.end_signature_addr = signature_addr - 0x4
        if self.pmp_randomize:
            self.randomize()
            return
        self.pmp_cfg = [pmp_cfg_reg_t() for _ in range(self.pmp_num_regions)]
        self.pmp_cfg_addr_valid = [0] * self.pmp_num_regions
        self.pmp_cfg_already_configured = [0] * self.pmp_num_regions
        self.set_defaults()
        self.setup_pmp()

    # ------------------------------------------------------------------
    # Randomization (pmp_randomize = 1)
    # ------------------------------------------------------------------
    def randomize(self):
        """Draw pmp_cfg[] satisfying the SV constraint blocks, then apply
        command-line overrides as the SV post_randomize() does.

        Each constraint is applied per region in the SV solve order
        (allow_high_addrs, then a and addr_mode, then addr):
          sanity_c           pmp_num_regions in [1:16] unless given
          xwr_c              never W=1 with R=0 (no ePMP, so mml = 0)
          allow_high_addrs_c 1 in high_addr_proportion %, always 1 on RV64
          address_modes_c    addr_mode in [0 : XLEN-3], or [0 : XLEN] if high
          grain_addr_mode_c  granularity >= 1 excludes NA4
          addr_range_c       offset[0] = 0, others in [1 : pmp_max_offset]
          addr_legal_tor_c   TOR above the previous entry's addr (unless
                             pmp_allow_illegal_tor and addr_mode == 0);
                             addr[31:29] == 0 unless high addresses
          addr_napot_mode_c  low addr_mode bits all ones, the next bit zero
          addr_na4_mode_c    addr[31:29] == 0 unless high addresses
        """
        if not self.pmp_num_regions_given:
            self.pmp_num_regions = random.randint(1, 16)
        n = self.pmp_num_regions
        self.pmp_cfg = [pmp_cfg_reg_t() for _ in range(n)]
        self.pmp_cfg_addr_valid = [0] * n
        self.pmp_cfg_already_configured = [0] * n
        if self.xlen == 64:
            self.allow_high_addrs = 1
        else:
            self.allow_high_addrs = int(random.randrange(100) < self.high_addr_proportion)
        max_mode = self.xlen if self.allow_high_addrs else self.xlen - 3
        addr_max = ((1 << self.xlen) - 1 if self.allow_high_addrs
                    else (1 << (self.xlen - 3)) - 1)
        prev_addr = 0
        for i, region in enumerate(self.pmp_cfg):
            region.l = random.randint(0, 1)
            region.x = random.randint(0, 1)
            region.r, region.w = random.choice([(0, 0), (1, 0), (1, 1)])
            region.offset = 0 if i == 0 else random.randint(1, self.pmp_max_offset)
            region.addr_mode = random.randint(0, max_mode)
            tor_ordered = (i > 0 and (not self.pmp_allow_illegal_tor or region.addr_mode > 0))
            modes = list(pmp_addr_mode_t)
            if self.pmp_granularity >= 1:
                modes.remove(pmp_addr_mode_t.NA4)
            if tor_ordered and prev_addr >= addr_max:
                modes.remove(pmp_addr_mode_t.TOR)
            region.a = random.choice(modes)
            if region.a == pmp_addr_mode_t.NAPOT:
                m = region.addr_mode
                if m >= self.xlen:
                    region.addr = (1 << self.xlen) - 1
                else:
                    ones = (1 << m) - 1
                    region.addr = (random.randint(0, (addr_max - ones) >> (m + 1)) << (m + 1)) | ones
            elif region.a == pmp_addr_mode_t.TOR and tor_ordered:
                region.addr = random.randint(prev_addr + 1, addr_max)
            else:
                region.addr = random.randint(0, addr_max)
            prev_addr = region.addr
        self.setup_pmp()

    def set_defaults(self):
        logging.info("MAX OFFSET: 0x%08x", self.pmp_max_offset)
        for i, region in enumerate(self.pmp_cfg):
            region.l = 0
            region.a = pmp_addr_mode_t.TOR
            region.x = 1
            region.w = 1
            region.r = 1
            region.offset = self.assign_default_addr_offset(self.pmp_num_regions, i)
            self.pmp_cfg_addr_valid[i] = 0
            self.pmp_cfg_already_configured[i] = 0

    def assign_default_addr_offset(self, num_regions, index):
        # The SV divides by (num_regions - 1), which is a division by zero for
        # the default single-region case. Guard it: with one region there is no
        # range to spread across, so the offset is 0.
        if num_regions <= 1:
            return 0
        return (self.pmp_max_offset // (num_regions - 1)) * index

    def setup_pmp(self):
        """Apply the +pmp_region_<i>= overrides collected from the command line."""
        for i in range(len(self.pmp_cfg)):
            arg_value = self.pmp_region_args.get(i)
            if arg_value is None:
                continue
            region, addr_valid = self.parse_pmp_config(arg_value, self.pmp_cfg[i])
            self.pmp_cfg[i] = region
            self.pmp_cfg_addr_valid[i] = addr_valid
            logging.info("Configured pmp_cfg[%0d] from command line: "
                         "L:%0d A:%s X:%0d W:%0d R:%0d ADDR:0x%08x",
                         i, region.l, region.a.name, region.x, region.w,
                         region.r, region.addr)

    def parse_pmp_config(self, pmp_region, ref_pmp_cfg):
        """Parse one "L:0,A:TOR,X:1,W:1,R:1,ADDR:FFFFFFFF" string."""
        region = pmp_cfg_reg_t()
        region.__dict__.update(ref_pmp_cfg.__dict__)
        addr_valid = 0
        for field in pmp_region.split(','):
            if not field:
                continue
            parts = field.split(':')
            if len(parts) != 2:
                logging.critical("%s, Invalid PMP configuration field!", field)
                sys.exit(1)
            field_type, field_val = parts[0].strip(), parts[1].strip()
            if field_type == 'L':
                region.l = int(field_val, 2)
            elif field_type == 'A':
                try:
                    region.a = pmp_addr_mode_t[field_val]
                except KeyError:
                    logging.critical("%s, Invalid PMP address mode!", field_val)
                    sys.exit(1)
            elif field_type == 'X':
                region.x = int(field_val, 2)
            elif field_type == 'W':
                region.w = int(field_val, 2)
            elif field_type == 'R':
                region.r = int(field_val, 2)
            elif field_type == 'ADDR':
                # No need to convert to "PMP format" beyond the shift; the rest
                # is masked off in hardware.
                addr_valid = 1
                region.addr = self.format_addr(int(field_val, 16))
            else:
                logging.critical("%s, Invalid PMP configuration field name!",
                                 field_type)
                sys.exit(1)
        return region, addr_valid

    def format_addr(self, addr):
        """pmpaddr never holds the bottom two bits of the address."""
        shifted_addr = addr >> 2
        if self.xlen == 32:
            # pmpaddr is bits [33:2] of a 34-bit address.
            return shifted_addr & 0xFFFFFFFF
        if self.xlen == 64:
            # pmpaddr is bits [55:2] of a 56-bit address, prepended by 10'b0.
            return shifted_addr & ((1 << (self.xlen - 10)) - 1)
        logging.critical("Unsupported XLEN %0s", self.xlen)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Instruction generation
    # ------------------------------------------------------------------
    def gen_pmp_enable_all(self, scratch_reg, instr):
        """One NAPOT region covering all of memory, RWX, unlocked."""
        instr.append("li x{}, 0x1fffffff".format(scratch_reg))
        instr.append("csrw {}, x{}".format(hex(privileged_reg_t.PMPADDR0),
                                           scratch_reg))
        instr.append("csrw {}, 0x1f".format(hex(privileged_reg_t.PMPCFG0)))

    def gen_pmp_instr(self, scratch_reg, instr):
        """Emit the code that programs pmpaddr[i] and pmpcfg[i].

        Several pmpcfg entries share one physical CSR -- four in rv32 -- so the
        bytes are accumulated in pmp_word and written out only when the word is
        full or the list ends, exactly as src/riscv_pmp_cfg.sv does.
        """
        pmp_word = 0
        pmp_id = 0
        for i, region in enumerate(self.pmp_cfg):
            pmp_id = i // self.cfg_per_csr
            cfg_byte = region.to_byte()
            logging.info("cfg_byte: 0x%02x", cfg_byte)
            pmp_word = pmp_word | (cfg_byte << ((i % self.cfg_per_csr) * 8))
            # Only set the address if it was not already configured above.
            if not self.pmp_cfg_already_configured[i] or self.pmp_cfg_addr_valid[i]:
                if self.pmp_cfg_addr_valid[i] or self.pmp_randomize:
                    # An address was supplied by the test, or randomized.
                    instr.append("li x{}, {}".format(scratch_reg[0],
                                                     hex(region.addr)))
                    instr.append("csrw {}, x{}".format(
                        hex(self.base_pmp_addr + i), scratch_reg[0]))
                    logging.info("Value 0x%08x loaded into pmpaddr[%0d], "
                                 "corresponding to address 0x%0x",
                                 region.addr, i, region.addr << 2)
                else:
                    # Add the offset to <main> to get the other pmpaddr values.
                    instr.append("la x{}, main".format(scratch_reg[0]))
                    instr.append("li x{}, {}".format(scratch_reg[1],
                                                     hex(region.offset)))
                    instr.append("add x{}, x{}, x{}".format(
                        scratch_reg[0], scratch_reg[0], scratch_reg[1]))
                    instr.append("srli x{}, x{}, 2".format(scratch_reg[0],
                                                           scratch_reg[0]))
                    instr.append("csrw {}, x{}".format(
                        hex(self.base_pmp_addr + i), scratch_reg[0]))
                    logging.info("Offset of pmp_addr_%0d from main: 0x%08x",
                                 i, region.offset)
            # Write the pmpcfg CSR once its word is complete.
            if i == len(self.pmp_cfg) - 1:
                instr.append("li x{}, {}".format(scratch_reg[0], hex(pmp_word)))
                instr.append("csrw {}, x{}".format(
                    hex(self.base_pmpcfg_addr + pmp_id), scratch_reg[0]))
                break
            elif (i + 1) % self.cfg_per_csr == 0:
                instr.append("li x{}, {}".format(scratch_reg[0], hex(pmp_word)))
                instr.append("csrw {}, x{}".format(
                    hex(self.base_pmpcfg_addr + pmp_id), scratch_reg[0]))
                pmp_word = 0

    def gen_pmp_exception_routine(self, scratch_reg, fault_type, instr):
        """Find the PMP entry matching mtval and grant the missing permission.

        Ported from src/riscv_pmp_cfg.sv gen_pmp_exception_routine, non-ePMP
        branch (FyraCore has no mseccfg). scratch_reg holds 7 registers:
          [0] temporary       [1] pmpaddr[i]      [2] pmpcfg CSR value
          [3] 8-bit cfg byte  [4] temporary / A   [5] pmpaddr[i-1]
          [6] loop counter
        """
        s = scratch_reg
        clog = int(math.log2(self.cfg_per_csr))
        xlen = self.xlen
        g = self.pmp_granularity
        mepc, mtval = hex(privileged_reg_t.MEPC), hex(privileged_reg_t.MTVAL)
        # Initialize loop counter and save to scratch_reg[6]
        instr.extend(("li x{}, 0".format(s[0]),
                      "mv x{}, x{}".format(s[6], s[0]),
                      "li x{}, 0".format(s[5]),
                      # calculate next pmpaddr and pmpcfg CSRs to read
                      "0: mv x{}, x{}".format(s[0], s[6]),
                      "mv x{}, x{}".format(s[4], s[0])))
        for i in range(1, self.pmp_num_regions + 1):
            instr.append("li x{}, {}".format(s[4], i - 1))
            instr.append("beq x{}, x{}, {}f".format(s[0], s[4], i))
        for i in range(1, self.pmp_num_regions + 1):
            instr.append("{}: csrr x{}, {}".format(i, s[1], hex(self.base_pmp_addr + i - 1)))
            instr.append("csrr x{}, {}".format(s[2], hex(self.base_pmpcfg_addr + (i - 1) // 4)))
            instr.append("j 17f")
        instr.extend((
            # get correct 8-bit configuration fields
            "17: li x{}, {}".format(s[3], self.cfg_per_csr),
            "slli x{}, x{}, {}".format(s[0], s[6], xlen - clog),
            "srli x{}, x{}, {}".format(s[0], s[0], xlen - clog),
            "sub x{}, x{}, x{}".format(s[4], s[3], s[0]),
            "addi x{}, x{}, -1".format(s[4], s[4]),
            "slli x{}, x{}, 3".format(s[4], s[4]),
            "sll x{}, x{}, x{}".format(s[3], s[2], s[4]),
            "slli x{}, x{}, 3".format(s[0], s[0]),
            "add x{}, x{}, x{}".format(s[4], s[4], s[0]),
            "srl x{}, x{}, x{}".format(s[3], s[3], s[4]),
            # get pmpcfg[i].A field
            "slli x{}, x{}, {}".format(s[4], s[3], xlen - 5),
            "srli x{}, x{}, {}".format(s[4], s[4], xlen - 2),
            # based on address match mode, branch to appropriate "handler"
            "beqz x{}, 20f".format(s[4]),
            "li x{}, 1".format(s[0]),
            "beq x{}, x{}, 21f".format(s[4], s[0]),
            "li x{}, 2".format(s[0]),
            "beq x{}, x{}, 24f".format(s[4], s[0]),
            "li x{}, 3".format(s[0]),
            "beq x{}, x{}, 25f".format(s[4], s[0]),
            # Error check, if no address modes match, something has gone wrong
            "la x{}, test_done".format(s[0]),
            "jalr x0, x{}, 0".format(s[0]),
            # increment loop counter and branch back to beginning of loop
            "18: mv x{}, x{}".format(s[0], s[6]),
            "mv x{}, x{}".format(s[5], s[1]),
            "addi x{}, x{}, 1".format(s[0], s[0]),
            "mv x{}, x{}".format(s[6], s[0]),
            "li x{}, {}".format(s[1], self.pmp_num_regions),
            "ble x{}, x{}, 19f".format(s[1], s[0]),
            "j 0b",
            # If we reach here, no PMP entry has matched the request: jump to
            # <test_done> ("there is a bug somewhere").
            "19: nop",
            "la x{}, test_done".format(s[0]),
            "jalr x0, x{}, 0".format(s[0]),
            # OFF: continue looping through the other PMP CSRs
            "20: j 18b",
            # TOR
            "21: mv x{}, x{}".format(s[0], s[6]),
            "csrr x{}, {}".format(s[4], mtval),
            "srli x{}, x{}, 2".format(s[4], s[4]),
            "bnez x{}, 22f".format(s[0]),
            "bltz x{}, 18b".format(s[4]),
            "j 23f",
            "22: bgtu x{}, x{}, 18b".format(s[5], s[4]),
            "23: bleu x{}, x{}, 18b".format(s[1], s[4]),
            "j 26f",
            # NA4
            "24: csrr x{}, {}".format(s[0], mtval),
            "srli x{}, x{}, 2".format(s[0], s[0]),
            "slli x{}, x{}, 2".format(s[4], s[1]),
            "srli x{}, x{}, 2".format(s[4], s[4]),
            "bne x{}, x{}, 18b".format(s[0], s[4]),
            "j 26f",
            # NAPOT: mask the bottom pmp_granularity bits of fault_addr and pmpaddr[i]
            "25: csrr x{}, {}".format(s[0], mtval),
            "srli x{}, x{}, 2".format(s[0], s[0]),
            "srli x{}, x{}, {}".format(s[0], s[0], g),
            "slli x{}, x{}, {}".format(s[0], s[0], g),
            "slli x{}, x{}, 2".format(s[4], s[1]),
            "srli x{}, x{}, 2".format(s[4], s[4]),
            "srli x{}, x{}, {}".format(s[4], s[4], g),
            "slli x{}, x{}, {}".format(s[4], s[4], g),
            "bne x{}, x{}, 18b".format(s[0], s[4]),
            "j 26f",
            # address match: check whether the lock bit is set
            "26: nop",
            "andi x{}, x{}, 128".format(s[4], s[3]),
            "bnez x{}, 27f".format(s[4]),
            "j 29f"))

        def skip_faulting_instr():
            return ["lw x{}, 0(x{})".format(s[0], s[0]),
                    "li x{}, 3".format(s[4]),
                    "and x{}, x{}, x{}".format(s[0], s[0], s[4]),
                    "beq x{}, x{}, 28f".format(s[0], s[4]),
                    "csrr x{}, {}".format(s[0], mepc),
                    "addi x{}, x{}, 2".format(s[0], s[0]),
                    "csrw {}, x{}".format(mepc, s[0]),
                    "j 34f",
                    "28: csrr x{}, {}".format(s[0], mepc),
                    "addi x{}, x{}, 4".format(s[0], s[0]),
                    "csrw {}, x{}".format(mepc, s[0]),
                    "j 34f"]
        if fault_type == exception_cause_t.INSTRUCTION_ACCESS_FAULT:
            instr.extend(("27: la x{}, test_done".format(s[0]),
                          "jalr x0, x{}, 0".format(s[0]),
                          "29: ori x{}, x{}, 4".format(s[3], s[3])))
        elif fault_type == exception_cause_t.STORE_AMO_ACCESS_FAULT:
            instr.append("27: csrr x{}, {}".format(s[0], mepc))
            instr.extend(skip_faulting_instr())
            # W:1 with R:0 is reserved, so enabling write enables read too
            instr.append("29: ori x{}, x{}, 3".format(s[3], s[3]))
        elif fault_type == exception_cause_t.LOAD_ACCESS_FAULT:
            instr.extend(("27: csrr x{}, {}".format(s[0], mepc),
                          # MEPC before main: the fault happened in a trap handler
                          "la x{}, main".format(s[4]),
                          "bge x{}, x{}, 40f".format(s[0], s[4]),
                          "la x{}, test_done".format(s[0]),
                          "jalr x0, x{}, 0".format(s[0])))
            seq = skip_faulting_instr()
            seq[0] = "40: " + seq[0]
            instr.extend(seq)
            instr.append("29: ori x{}, x{}, 1".format(s[3], s[3]))
        else:
            logging.critical("Invalid PMP fault type")
            sys.exit(1)
        instr.extend((
            "li x{}, {}".format(s[4], xlen - clog),
            "sll x{}, x{}, x{}".format(s[0], s[6], s[4]),
            "srl x{}, x{}, x{}".format(s[0], s[0], s[4]),
            "slli x{}, x{}, 3".format(s[4], s[0]),
            "sll x{}, x{}, x{}".format(s[3], s[3], s[4]),
            "or x{}, x{}, x{}".format(s[2], s[2], s[3]),
            "mv x{}, x{}".format(s[0], s[6]),
            "srli x{}, x{}, {}".format(s[0], s[0], clog),
            "beqz x{}, 30f".format(s[0]),
            "li x{}, 1".format(s[4]),
            "beq x{}, x{}, 31f".format(s[0], s[4]),
            "li x{}, 2".format(s[4]),
            "beq x{}, x{}, 32f".format(s[0], s[4]),
            "li x{}, 3".format(s[4]),
            "beq x{}, x{}, 33f".format(s[0], s[4]),
            "30: csrw {}, x{}".format(hex(privileged_reg_t.PMPCFG0), s[2]),
            "j 34f",
            "31: csrw {}, x{}".format(hex(privileged_reg_t.PMPCFG1), s[2]),
            "j 34f",
            "32: csrw {}, x{}".format(hex(privileged_reg_t.PMPCFG2), s[2]),
            "j 34f",
            "33: csrw {}, x{}".format(hex(privileged_reg_t.PMPCFG3), s[2]),
            "34: nop"))

    def gen_pmp_write_test(self, scratch_reg, instr):
        """Write random values to every pmpaddr and pmpcfg CSR, then restore.

        This is the WARL exerciser: each CSR is written with csrrw, which also
        reads back the previous value, and a second csrrw puts the original
        value back. What the core chose to store is therefore observable in a
        GPR, so a legalisation that differs from the reference model shows up
        in the comparison rather than being silently absorbed.
        """
        for i in range(self.pmp_num_regions):
            pmp_addr = self.base_pmp_addr + i
            pmpcfg_addr = self.base_pmpcfg_addr + (i // self.cfg_per_csr)
            # Randomize the lower 31 bits and add <main>, so the random value
            # written to pmpaddr[i] cannot interfere with the safe region.
            pmp_val = random.getrandbits(self.xlen) & ~(1 << 31)
            instr.append("li x{}, {}".format(scratch_reg[0], hex(pmp_val)))
            instr.append("la x{}, main".format(scratch_reg[1]))
            instr.append("add x{}, x{}, x{}".format(
                scratch_reg[0], scratch_reg[0], scratch_reg[1]))
            # Write the random address; the old value lands in scratch_reg[0].
            instr.append("csrrw x{}, {}, x{}".format(
                scratch_reg[0], hex(pmp_addr), scratch_reg[0]))
            # Put the original address back; the new value lands in scratch_reg[0].
            instr.append("csrrw x{}, {}, x{}".format(
                scratch_reg[0], hex(pmp_addr), scratch_reg[0]))

            # Randomize the value written to the pmpcfg CSR, with every Lock
            # bit clear -- locking a region would make the rest of the test
            # unable to reconfigure it -- and never W=1 with R=0, which the
            # privileged spec reserves.
            pmp_val = random.getrandbits(self.xlen)
            for byte in range(self.cfg_per_csr):
                shift = byte * 8
                pmp_val &= ~(1 << (shift + 7))          # L = 0
                r_bit = (pmp_val >> shift) & 0x1
                if r_bit == 0:
                    pmp_val &= ~(1 << (shift + 1))      # W = 0 when R = 0
            # Keep the "safe" region fully accessible so the program can
            # continue running after this sequence.
            if pmpcfg_addr == self.base_pmpcfg_addr:
                pmp_val = (pmp_val & ~0xFF) | 0x0F
            instr.append("li x{}, {}".format(scratch_reg[0], hex(pmp_val)))
            instr.append("csrrw x{}, {}, x{}".format(
                scratch_reg[0], hex(pmpcfg_addr), scratch_reg[0]))
            instr.append("csrrw x{}, {}, x{}".format(
                scratch_reg[0], hex(pmpcfg_addr), scratch_reg[0]))
