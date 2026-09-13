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
import copy
import sys
import random
import vsc
from imp import reload
from collections import defaultdict
from bitstring import BitArray
from importlib import import_module
from pygen_src.riscv_instr_pkg import (pkg_ins, riscv_instr_category_t, riscv_reg_t,
                                       riscv_instr_name_t, riscv_instr_format_t,
                                       riscv_instr_group_t, imm_t,
                                       privileged_mode_t, privileged_reg_t)
from pygen_src.riscv_instr_gen_config import cfg
rcs = import_module("pygen_src.target." + cfg.argv.target + ".riscv_core_setting")
reload(logging)
logging.basicConfig(filename='{}'.format(cfg.argv.log_file_name),
                    filemode='w',
                    format="%(asctime)s %(filename)s %(lineno)s %(levelname)s %(message)s",
                    level=logging.DEBUG)


@vsc.randobj
class riscv_instr:
    # All derived instructions
    instr_registry = {}

    # Instruction list
    instr_names = []

    # Categorized instruction list
    instr_group = defaultdict(list)
    instr_category = defaultdict(list)
    basic_instr = []
    instr_template = {}

    # Privileged CSR filter
    exclude_reg = []
    include_reg = []

    def __init__(self):
        # Instruction attributes
        self.group = vsc.enum_t(riscv_instr_group_t)
        self.format = vsc.enum_t(riscv_instr_format_t)
        self.category = vsc.enum_t(riscv_instr_category_t)
        self.instr_name = vsc.enum_t(riscv_instr_name_t)
        self.imm_type = vsc.enum_t(imm_t)
        self.imm_len = vsc.bit_t(5)

        # Operands
        self.csr = vsc.rand_bit_t(12)
        self.rs2 = vsc.rand_enum_t(riscv_reg_t)
        self.rs1 = vsc.rand_enum_t(riscv_reg_t)
        self.rd = vsc.rand_enum_t(riscv_reg_t)
        self.imm = vsc.rand_bit_t(32)

        # Helper Fields
        self.imm_mask = vsc.uint32_t(0xffffffff)
        self.is_branch_target = None
        self.has_label = 1
        self.atomic = 0
        self.branch_assigned = None
        self.process_load_store = 1
        self.is_compressed = None
        self.is_illegal_instr = None
        self.is_hint_instr = None
        self.is_floating_point = None
        self.imm_str = None
        self.comment = ""
        self.label = ""
        self.is_local_numeric_label = None
        self.idx = -1
        self.has_rs1 = vsc.bit_t(1)
        self.has_rs2 = vsc.bit_t(1)
        self.has_rd = vsc.bit_t(1)
        self.has_imm = vsc.bit_t(1)
        self.has_rs1 = 1
        self.has_rs2 = 1
        self.has_rd = 1
        self.has_imm = 1
        self.shift_t = vsc.uint32_t(0xffffffff)
        self.mask = 32
        self.XLEN = vsc.uint32_t(32)  # XLEN is used in constraint throughout the generator.
        # Hence, XLEN should be of PyVSC type in order to use it in a constraint block
        self.XLEN = rcs.XLEN

    @vsc.constraint
    def imm_c(self):
        with vsc.if_then(self.instr_name.inside(vsc.rangelist(riscv_instr_name_t.SLLIW,
                                                              riscv_instr_name_t.SRLIW,
                                                              riscv_instr_name_t.SRAIW))):
            self.imm[11:5] == 0
        with vsc.if_then(self.instr_name.inside(vsc.rangelist(riscv_instr_name_t.SLLI,
                                                              riscv_instr_name_t.SRLI,
                                                              riscv_instr_name_t.SRAI))):
            with vsc.if_then(self.XLEN == 32):
                self.imm[11:5] == 0
            with vsc.if_then(self.XLEN != 32):
                self.imm[11:6] == 0

    @vsc.constraint
    def csr_c(self):
        # TODO
        pass

    @classmethod
    def register(cls, instr_name, instr_group):
        logging.info("Registering {}".format(instr_name.name))
        cls.instr_registry[instr_name] = instr_group
        return 1

    def __deepcopy__(self, memo):
        cls = self.__class__  # Extract the class of the object.
        # Create a new instance of the object based on extracted class.
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ["_ro_int", "tname", "__field_info"]:
                continue  # Skip the fields which are not required.
            else:
                # Copy over attributes by copying directly.
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    # Create the list of instructions based on the supported ISA extensions and configuration
    # of the generator
    @classmethod
    def create_instr_list(cls, cfg):
        cls.instr_names.clear()
        cls.instr_group.clear()
        cls.instr_category.clear()
        for instr_name, instr_group in cls.instr_registry.items():
            if instr_name in rcs.unsupported_instr:
                continue
            instr_inst = cls.create_instr(instr_name, instr_group)
            cls.instr_template[instr_name] = instr_inst

            if not instr_inst.is_supported(cfg):
                continue
            # C_JAL is RV32C only instruction
            if ((rcs.XLEN != 32) and (instr_name == riscv_instr_name_t.C_JAL)):
                continue
            if ((riscv_reg_t.SP in cfg.reserved_regs) and
                    (instr_name == riscv_instr_name_t.C_ADDI16SP)):
                continue
            if (cfg.enable_sfence and instr_name == riscv_instr_name_t.SFENCE_VMA):
                continue
            if instr_name in [riscv_instr_name_t.FENCE, riscv_instr_name_t.FENCE_I,
                              riscv_instr_name_t.SFENCE_VMA]:
                continue
            if (instr_inst.group in rcs.supported_isa and
                    not(cfg.disable_compressed_instr and
                    instr_inst.group.name in ["RV32C", "RV64C", "RV32DC",
                                              "RV32FC", "RV128C"]) and
                    not(not(cfg.enable_floating_point) and instr_inst.group.name in
                    ["RV32F", "RV64F", "RV32D", "RV64D"])):
                cls.instr_category[instr_inst.category.name].append(instr_name)
                cls.instr_group[instr_inst.group.name].append(instr_name)
                cls.instr_names.append(instr_name)
        cls.build_basic_instruction_list(cfg)
        cls.create_csr_filter(cfg)

    @classmethod
    def create_instr(cls, instr_name, instr_group):
        try:
            module_name = import_module("pygen_src.isa." + instr_group.name.lower() + "_instr")
            instr_inst = eval("module_name.riscv_" + instr_name.name + "_instr()")
        except Exception:
            logging.critical("Failed to create instr: {}".format(instr_name.name))
            sys.exit(1)
        return instr_inst

    def is_supported(self, cfg):
        return 1

    @classmethod
    def build_basic_instruction_list(cls, cfg):
        cls.basic_instr = (cls.instr_category["SHIFT"] + cls.instr_category["ARITHMETIC"] +
                           cls.instr_category["LOGICAL"] + cls.instr_category["COMPARE"])
        if cfg.no_ebreak == 0:
            cls.basic_instr.append(riscv_instr_name_t.EBREAK)
            for _ in rcs.supported_isa:
                if(riscv_instr_group_t.RV32C in rcs.supported_isa and
                   not(cfg.disable_compressed_instr)):
                    cls.basic_instr.append(riscv_instr_name_t.C_EBREAK)
                    break
        if cfg.no_dret == 0:
            cls.basic_instr.append(riscv_instr_name_t.DRET)
        # extend, not append: instr_category[...] is a *list* of names, and
        # appending it puts the list itself into basic_instr as one element.
        # get_rand_instr() then indexes instr_template with that list and dies.
        if cfg.no_fence == 0:
            cls.basic_instr.extend(cls.instr_category["SYNCH"])
        # cfg.init_privileged_mode is a privileged_mode_t, so comparing it to
        # the string "MACHINE_MODE" is always False and the CSR category was
        # never added -- pyflow generated no csrrw/csrrs/csrrc at all, whatever
        # no_csr_instr said. src/isa/riscv_instr.sv compares against the enum.
        if (cfg.no_csr_instr == 0 and
                cfg.init_privileged_mode == privileged_mode_t.MACHINE_MODE):
            cls.basic_instr.extend(cls.instr_category["CSR"])
        if cfg.no_wfi == 0:
            cls.basic_instr.append(riscv_instr_name_t.WFI)

    @classmethod
    def create_csr_filter(cls, cfg):
        cls.include_reg.clear()
        cls.exclude_reg.clear()

        if cfg.enable_illegal_csr_instruction:
            # list(), not an alias: assigning rcs.implemented_csr directly means
            # the exclude_reg.clear() above would empty the core setting itself
            # on a later call, and implemented_csr is read elsewhere (e.g.
            # riscv_instr_pkg.push_gpr_to_kernel_stack).
            cls.exclude_reg = list(rcs.implemented_csr)
        elif cfg.enable_access_invalid_csr_level:
            cls.include_reg = list(cfg.invalid_priv_mode_csrs)
        else:
            # Use the scratch register, to avoid the side effect of modifying
            # another privileged-mode CSR. These have to be privileged_reg_t
            # members, not their names: the value is written into the csr field
            # of the generated instruction. Comparing init_privileged_mode
            # against a string, as the port did, also never matched.
            if cfg.init_privileged_mode == privileged_mode_t.MACHINE_MODE:
                cls.include_reg.append(privileged_reg_t.MSCRATCH)
            elif cfg.init_privileged_mode == privileged_mode_t.SUPERVISOR_MODE:
                cls.include_reg.append(privileged_reg_t.SSCRATCH)
            else:
                cls.include_reg.append(privileged_reg_t.USCRATCH)

    def legalize_csr(self):
        """Point a CSR instruction at a CSR the filter allows.

        src/isa/riscv_csr_instr.sv constrains the address directly:

            constraint csr_addr_c {
              if (include_reg.size() > 0) { csr inside {include_reg}; }
              if (exclude_reg.size() > 0) { !(csr inside {exclude_reg}); }
            }

        pyflow has no riscv_csr_instr class at all. Its csr field is a plain
        `vsc.rand_bit_t(12)` whose only constraint, csr_c, is an empty `pass`,
        and include_reg/exclude_reg are populated by create_csr_filter and then
        never read by anything. So the address was a uniform draw over all 4096
        encodings, which would put a random value into whichever machine CSR it
        landed on -- mtvec and mepc included.

        Applied to the solved value rather than as a constraint, for the same
        build-order reason as the load/store rs1 legalisation.
        """
        if self.category != riscv_instr_category_t.CSR:
            return
        if riscv_instr.include_reg:
            self.csr = int(random.choice(riscv_instr.include_reg))
            return
        if riscv_instr.exclude_reg:
            excluded = {int(c) for c in riscv_instr.exclude_reg}
            # 4096 encodings against at most a few dozen exclusions, so a
            # redraw terminates immediately in practice.
            for _ in range(100):
                candidate = random.randrange(0, 1 << 12)
                if candidate not in excluded:
                    self.csr = candidate
                    return
            logging.warning("Could not draw a CSR outside the exclusion list")

    @classmethod
    def get_rand_instr(cls, include_instr=[], exclude_instr=[],
                       include_category=[], exclude_category=[],
                       include_group=[], exclude_group=[]):
        idx = BitArray(uint = 0, length = 32)
        name = ""
        allowed_instr = []
        disallowed_instr = []
        # allowed_categories = []
        for items in include_category:
            allowed_instr.extend(cls.instr_category[items])
        for items in exclude_category:
            if items in cls.instr_category:
                disallowed_instr.extend(cls.instr_category[items])
        for items in include_group:
            allowed_instr.extend(cls.instr_group[items])
        for items in exclude_group:
            if items in cls.instr_group:
                disallowed_instr.extend(cls.instr_group[items])

        disallowed_instr.extend(exclude_instr)

        # TODO Randomization logic needs to be frame with PyVSC library
        if len(disallowed_instr) == 0:
            # random.randrange(0, n - 1) yields 0..n-2, so the last entry of
            # each of these lists was unreachable: one instruction was silently
            # excluded from the generator. It also raises ValueError for a
            # single-element list -- the include_instr branch worked around
            # that with a special case while the other two did not, and the
            # handler below turns it into a fatal "Cannot generate random
            # instruction" rather than the single choice that was intended.
            try:
                if len(include_instr) > 0:
                    name = include_instr[random.randrange(len(include_instr))]
                elif len(allowed_instr) > 0:
                    name = allowed_instr[random.randrange(len(allowed_instr))]
                else:
                    name = cls.instr_names[random.randrange(len(cls.instr_names))]
            except Exception:
                logging.critical("[%s] Cannot generate random instruction", riscv_instr.__name__)
                sys.exit(1)
        else:
            # The SV implementation (src/isa/riscv_instr.sv) constrains the picked
            # name with "!(name inside {disallowed_instr})".  The Python port built
            # disallowed_instr and then never used it, so exclusions were silently
            # ignored and instructions with hard rd/rs1 constraints (e.g. C_ADDI16SP,
            # which forces rd == SP) could be returned even when that register was
            # reserved -- producing an unsatisfiable randomize_gpr() problem.
            # Callers pass either riscv_instr_name_t members or their string names,
            # so normalise both sides before filtering.
            disallowed_set = set()
            for item in disallowed_instr:
                if isinstance(item, str):
                    if item in riscv_instr_name_t.__members__:
                        disallowed_set.add(riscv_instr_name_t[item])
                elif isinstance(item, riscv_instr_name_t):
                    disallowed_set.add(item)
            if len(allowed_instr) > 0:
                candidates = allowed_instr
            elif len(include_instr) > 0:
                candidates = include_instr
            else:
                candidates = cls.instr_names
            candidates = [i for i in candidates if i not in disallowed_set]
            try:
                name = random.choice(candidates)
            except Exception:
                logging.critical("[%s] Cannot generate random instruction", riscv_instr.__name__)
                sys.exit(1)
        # rs1 rs2 values are overwriting and the last generated values are
        # getting assigned for a particular instruction hence creating different
        # object address and id to ratain the randomly generated values.
        instr_h = copy.deepcopy(cls.instr_template[name])
        return instr_h

    @classmethod
    def get_load_store_instr(cls, load_store_instr):
        instr_h = riscv_instr()
        if len(load_store_instr) == 0:
            load_store_instr = cls.instr_category["LOAD"] + \
                cls.instr_category["STORE"]
        # Same off-by-one, and it bites hardest here: gen_load_store_instr()
        # builds allowed_instr in alignment order and appends the widest legal
        # access last, so the widest access for each address was never chosen.
        # For an address where only [LB, LBU, SB] is legal, that removed SB and
        # left the stream unable to emit a byte store at all.
        cls.idx = random.randrange(len(load_store_instr))
        name = load_store_instr[cls.idx]
        instr_h = copy.copy(cls.instr_template[name])
        return instr_h

    @classmethod
    def get_instr(cls, name):
        if not cls.instr_template.get(name):
            logging.critical("Cannot get instr %s", name)
            sys.exit(1)
        instr_h = copy.copy(cls.instr_template[name])
        return instr_h

    def set_rand_mode(self):
        # rand_mode setting for Instruction Format
        if self.format.name == "R_FORMAT":
            self.has_imm = 0
        if self.format.name == "I_FORMAT":
            self.has_rs2 = 0
        if self.format.name in ["S_FORMAT", "B_FORMAT"]:
            self.has_rd = 0
        if self.format.name in ["U_FORMAT", "J_FORMAT"]:
            self.has_rs1 = 0
            self.has_rs2 = 0

        # rand_mode setting for Instruction Category
        if self.category.name == "CSR":
            self.has_rs2 = 0
            if self.format.name == "I_FORMAT":
                self.has_rs1 = 0

    def pre_randomize(self):
        with vsc.raw_mode():
            self.rs1.rand_mode = bool(self.has_rs1)
            self.rs2.rand_mode = bool(self.has_rs2)
            self.rd.rand_mode = bool(self.has_rd)
            self.imm.rand_mode = bool(self.has_imm)
            # int(), because `self.category != <enum>` on a vsc field builds a
            # constraint *expression* rather than evaluating to a Python bool.
            # pre_randomize() runs inside the randomize_with() scope opened by
            # riscv_instr_stream.randomize_gpr(), so that expression was
            # collected into the solve as a real constraint:
            #
            #     vsc.model.solve_failure.SolveFailure: solve failure
            #     Problem Set: 1 constraints
            #       (category != 11);
            #
            # For every non-CSR instruction the constraint happens to be true,
            # which is why it went unnoticed. For a CSR instruction -- category
            # 11 -- it is a flat contradiction and the solve fails, so no test
            # containing a csrrw/csrrs/csrrc could ever be generated. Reading
            # get_val() keeps this a plain Python branch, which is what
            # src/isa/riscv_instr.sv:299 does. (int() does not work here --
            # a non-rand enum_t does not implement __int__.)
            if self.category.get_val() != riscv_instr_category_t.CSR:
                self.csr.rand_mode = False

    def set_imm_len(self):
        if self.format.name in ["U_FORMAT", "J_FORMAT"]:
            self.imm_len = 20
        elif self.format.name in ["I_FORMAT", "S_FORMAT", "B_FORMAT"]:
            if self.imm_type.name == "UIMM":
                self.imm_len = 5
            else:
                self.imm_len = 12
        self.imm_mask = (self.imm_mask << self.imm_len) & self.shift_t

    def extend_imm(self):
        sign = 0
        # self.shift_t = 2 ** 32 -1 is used to limit the width after shift operation
        self.imm = self.imm << (32 - self.imm_len) & self.shift_t
        sign = (self.imm & 0x80000000) >> 31
        self.imm = self.imm >> (32 - self.imm_len) & self.shift_t
        # Signed extension
        if(sign and not((self.format.name == "U_FORMAT") or
                        (self.imm_type.name in ["UIMM", "NZUIMM"]))):
            self.imm = self.imm_mask | self.imm

    def post_randomize(self):
        self.extend_imm()
        self.update_imm_str()

    def convert2asm(self, prefix = " "):
        asm_str = pkg_ins.format_string(string = self.get_instr_name(),
                                        length = pkg_ins.MAX_INSTR_STR_LEN)
        if self.category != riscv_instr_category_t.SYSTEM:
            if self.format == riscv_instr_format_t.J_FORMAT:
                asm_str = '{} {}, {}'.format(asm_str, self.rd.name, self.get_imm())
            elif self.format == riscv_instr_format_t.U_FORMAT:
                asm_str = '{} {}, {}'.format(asm_str, self.rd.name, self.get_imm())
            elif self.format == riscv_instr_format_t.I_FORMAT:
                if self.instr_name == riscv_instr_name_t.NOP:
                    asm_str = "nop"
                elif self.instr_name == riscv_instr_name_t.WFI:
                    asm_str = "wfi"
                elif self.instr_name == riscv_instr_name_t.FENCE:
                    asm_str = "fence"
                elif self.instr_name == riscv_instr_name_t.FENCE_I:
                    asm_str = "fence.i"
                elif self.category == riscv_instr_category_t.LOAD:
                    asm_str = '{} {}, {} ({})'.format(
                        asm_str, self.rd.name, self.get_imm(), self.rs1.name)
                elif self.category == riscv_instr_category_t.CSR:
                    # '0x{}' formats the value in *decimal* behind a hex
                    # prefix, so mscratch (832 == 0x340) was emitted as the
                    # unrelated CSR 0x832. Every CSR address the generator
                    # produced was wrong; it stayed invisible only because no
                    # CSR instruction was ever generated.
                    asm_str = '{} {}, {}, {}'.format(
                        asm_str, self.rd.name, hex(int(self.csr)), self.get_imm())
                else:
                    asm_str = '{} {}, {}, {}'.format(
                        asm_str, self.rd.name, self.rs1.name, self.get_imm())
            elif self.format == riscv_instr_format_t.S_FORMAT:
                if self.category == riscv_instr_category_t.STORE:
                    asm_str = '{} {}, {} ({})'.format(
                        asm_str, self.rs2.name, self.get_imm(), self.rs1.name)
                else:
                    asm_str = '{} {}, {}, {}'.format(
                        asm_str, self.rs1.name, self.rs2.name, self.get_imm())

            elif self.format == riscv_instr_format_t.B_FORMAT:
                if self.category == riscv_instr_category_t.STORE:
                    asm_str = '{} {}, {} ({})'.format(
                        asm_str, self.rs2.name, self.get_imm(), self.rs1.name)
                else:
                    asm_str = '{} {}, {}, {}'.format(
                        asm_str, self.rs1.name, self.rs2.name, self.get_imm())

            elif self.format == riscv_instr_format_t.R_FORMAT:
                if self.category == riscv_instr_category_t.CSR:
                    asm_str = '{} {}, {}, {}'.format(
                        asm_str, self.rd.name, hex(int(self.csr)), self.rs1.name)
                elif self.instr_name == riscv_instr_name_t.SFENCE_VMA:
                    asm_str = "sfence.vma x0, x0"
                else:
                    asm_str = '{} {}, {}, {}'.format(
                        asm_str, self.rd.name, self.rs1.name, self.rs2.name)
            else:
                asm_str = 'Fatal_unsupported_format: {} {}'.format(
                    self.format.name, self.instr_name.name)

        else:
            if self.instr_name == riscv_instr_name_t.EBREAK:
                asm_str = ".4byte 0x00100073 # ebreak"

        if self.comment != "":
            asm_str = asm_str + " #" + self.comment
        return asm_str.lower()

    def get_opcode(self):
        if self.instr_name == "LUI":
            return (BitArray(uint = 55, length = 7).bin)
        elif self.instr_name == "AUIPC":
            return (BitArray(uint = 23, length = 7).bin)
        elif self.instr_name == "JAL":
            return (BitArray(uint = 23, length = 7).bin)
        elif self.instr_name == "JALR":
            return (BitArray(uint = 111, length = 7).bin)
        elif self.instr_name in ["BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU"]:
            return (BitArray(uint = 103, length = 7).bin)
        elif self.instr_name in ["LB", "LH", "LW", "LBU", "LHU", "LWU", "LD"]:
            return (BitArray(uint = 99, length = 7).bin)
        elif self.instr_name in ["SB", "SH", "SW", "SD"]:
            return (BitArray(uint = 35, length = 7).bin)
        elif self.instr_name in ["ADDI", "SLTI", "SLTIU", "XORI", "ORI", "ANDI",
                                 "SLLI", "SRLI", "SRAI", "NOP"]:
            return (BitArray(uint = 19, length = 7).bin)
        elif self.instr_name in ["ADD", "SUB", "SLL", "SLT", "SLTU", "XOR", "SRL",
                                 "SRA", "OR", "AND", "MUL", "MULH", "MULHSU", "MULHU",
                                 "DIV", "DIVU", "REM", "REMU"]:
            return (BitArray(uint = 51, length = 7).bin)
        elif self.instr_name in ["ADDIW", "SLLIW", "SRLIW", "SRAIW"]:
            return (BitArray(uint = 27, length = 7).bin)
        elif self.instr_name in ["MULH", "MULHSU", "MULHU", "DIV", "DIVU", "REM", "REMU"]:
            return (BitArray(uint = 51, length = 7).bin)
        elif self.instr_name in ["FENCE", "FENCE_I"]:
            return (BitArray(uint = 15, length = 7).bin)
        elif self.instr_name in ["ECALL", "EBREAK", "CSRRW", "CSRRS", "CSRRC", "CSRRWI",
                                 "CSRRSI", "CSRRCI"]:
            return (BitArray(uint = 115, length = 7).bin)
        elif self.instr_name in ["ADDW", "SUBW", "SLLW", "SRLW", "SRAW", "MULW", "DIVW",
                                 "DIVUW", "REMW", "REMUW"]:
            return (BitArray(uint = 59, length = 7).bin)
        elif self.instr_name in ["ECALL", "EBREAK", "URET", "SRET", "MRET", "DRET", "WFI",
                                 "SFENCE_VMA"]:
            return (BitArray(uint = 115, length = 7).bin)
        else:
            logging.critical("Unsupported instruction %0s", self.instr_name)
            sys.exit(1)

    def get_func3(self):
        if self.instr_name in ["JALR", "BEQ", "LB", "SB", "ADDI", "NOP", "ADD", "SUB",
                               "FENCE", "ECALL", "EBREAK", "ADDIW", "ADDW", "SUBW", "MUL",
                               "MULW", "ECALL", "EBREAK", "URET", "SRET", "MRET", "DRET",
                               "WFI", "SFENCE_VMA"]:
            return (BitArray(uint = 0, length = 3).bin)
        elif self.instr_name in ["BNE", "LH", "SH", "SLLI", "SLL", "FENCE_I", "CSRRW", "SLLIW",
                                 "SLLW", "MULH"]:
            return (BitArray(uint = 1, length = 3).bin)
        elif self.instr_name in ["LW", "SW", "SLTI", "SLT", "CSRRS", "MULHS"]:
            return (BitArray(uint = 2, length = 3).bin)
        elif self.instr_name in ["SLTIU", "SLTU", "CSRRC", "LD", "SD", "MULHU"]:
            return (BitArray(uint = 3, length = 3).bin)
        elif self.instr_name in ["BLT", "LBU", "XORI", "XOR", "DIV", "DIVW"]:
            return (BitArray(uint = 4, length = 3).bin)
        elif self.instr_name in ["BGE", "LHU", "SRLI", "SRAI", "SRL", "SRA", "CSRRWI", "SRLIW",
                                 "SRAIW", "SRLW",
                                 "SRAW", "DIVU", "DIVUW"]:
            return (BitArray(uint = 5, length = 3).bin)
        elif self.instr_name in ["BLTU", "ORI", "OR", "CSRRSI", "LWU", "REM", "REMW"]:
            return (BitArray(uint = 6, length = 3).bin)
        elif self.instr_name in ["BGEU", "ANDI", "AND", "CSRRCI", "REMU", "REMUW"]:
            return (BitArray(uint = 7, length = 3).bin)
        else:
            logging.critical("Unsupported instruction %0s", self.instr_name)
            sys.exit(1)

    def get_func7(self):
        if self.instr_name in ["SLLI", "SRLI", "ADD", "SLL", "SLT", "SLTU", "XOR",
                               "SRL", "OR", "AND", "FENCE", "FENCE_I", "SLLIW",
                               "SRLIW", "ADDW", "SLLW", "SRLW", "ECALL", "EBREAK", "URET"]:
            return (BitArray(uint = 0, length = 7).bin)
        elif self.instr_name in ["SUB", "SRA", "SRAIW", "SUBW", "SRAW"]:
            return (BitArray(uint = 32, length = 7).bin)
        elif self.instr_name in ["MUL", "MULH", "MULHSU", "MULHU", "DIV", "DIVU", "REM",
                                 "REMU", "MULW", "DIVW", "DIVUW", "REMW", "REMUW"]:
            return (BitArray(uint = 1, length = 7).bin)
        elif self.instr_name in ["SRET", "WFI"]:
            return (BitArray(uint = 8, length = 7).bin)
        elif self.instr_name == "MRET":
            return (BitArray(uint = 24, length = 7).bin)
        elif self.instr_name == "DRET":
            return (BitArray(uint = 61, length = 7).bin)
        elif self.instr_name == "SFENCE_VMA":
            return (BitArray(uint = 9, length = 7).bin)
        else:
            logging.critical("Unsupported instruction %0s", self.instr_name)
            sys.exit(1)

    def convert2bin(self):
        pass  # TODO

    def get_instr_name(self):
        get_instr_name = self.instr_name.name
        get_instr_name = get_instr_name.replace("_", ".")
        return get_instr_name

    def get_c_gpr(self, gpr):
        return self.gpr

    def get_imm(self):
        return self.imm_str

    def clear_unused_label(self):
        if(self.has_label and not(self.is_branch_target) and self.is_local_numeric_label):
            self.has_label = 0

    def do_copy(self):
        pass  # TODO

    def update_imm_str(self):
        self.imm_str = str(self.uintToInt(self.imm))

    def uintToInt(self, x):
        if x < (2 ** self.mask) / 2:
            signed_x = x
        else:
            signed_x = x - 2 ** self.mask
        return signed_x
