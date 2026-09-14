# pyflow-fixes

This branch is upstream riscv-dv (`b7a0b4b`, chipsalliance master) plus fixes
to **pyflow**, the Python/PyVSC port of the SystemVerilog generator under
`pygen/`. Upstream pyflow is a partial hand translation: on the rv32imac
target it generated nothing for most tests, and what it did generate mostly
tested nothing (programs ended after ~2% of themselves and still passed).

The SystemVerilog under `src/` is the oracle. Every commit is one defect,
quotes the SV it was checked against, and says what was measured before and
after. `git log master..pyflow-fixes` is the record.

* Commits 1-31 are the former patch series from AlphaOneSoC's riscv_dv
  plugin, one commit per patch; `patches-01-31.md` is that series' README,
  kept for history (its "still missing" section is superseded below).
* Everything after is described in its own commit message.

## How it was verified

Generated programs were assembled and run on Spike with the flow's options,
and on the FyraCore RTL (iverilog) with the flow's comparison. "It generates"
and "the trace comparison passed" were never taken as proof; the checks were
whether the program runs its body (tohost line vs. static size), whether each
stream uses what it sets up, whether injected encodings trap/retire as
claimed, whether handlers actually execute, and seeded probes of the
constraint solutions against every SV constraint.

## pyvsc

Everything was generated with PyPI pyvsc 0.9.5.27214109393. Two pyvsc defects
that pyflow can reach are fixed in alfadelta10010/pyvsc-patched
(branch riscv-dv-fixes): `_randset_m.pop(idx)` in
`RandInfoBuilder.visit_expr_array_subscript`, and an empty range list making
`inside {}` true. The pyflow code on this branch no longer depends on either.

## Deliberate divergences from the SV

Each is explained in its commit.

* Unexpected traps end the test as a failure, `tohost = (mcause << 1) | 1`;
  the SV falls through to `test_done`, which passes.
* A `riscv_illegal_instr` solve failure is fatal again (no fallback draw).
* AMO: `aq`/`rl` bits are 0 (only ratified A-extension encodings), and LR/SC
  bodies obey the unprivileged spec's eventual-success rules (section 8.3).
* `write_tohost`/`_exit` are emitted ahead of `main` when PMP is supported, so
  a locked entry denying execution over `main` cannot make the exit path fault.
* PMP exception routine: a no-match fault fails the test only when the access
  was checked with M-mode privilege (U-mode no-match is default deny and ends
  through `test_done`, as in the SV); a fault on an entry that already grants
  the access fails instead of retrying forever; NAPOT entries match by region
  size (the SV masks only the grain).
* `pmp_cfg[]` is sized to the randomized `pmp_num_regions` (the SV sizes it
  in `new()`, before randomization).
* `get_invalid_priv_lvl_csr()` is recomputed after the boot mode is drawn (the
  SV computes it in `new()` from the enum default).
* `cfg.ra`, `pmp_reg` and a stream's `avail_regs` are drawn after the solve,
  from the SV's distribution/constraints, because pyvsc does not reproduce
  them in-solve; `randomize_gpr` constrains rs1/rs2/rd to the drawn values.
* Where the port's constraints match the SV but pyvsc's solutions are heavily
  skewed, the value is drawn uniformly from the constraint's solution set (or
  from the SV's `dist` weights) instead, and the constraint is kept:
  * the main/sub-program instruction-count split (uniform composition of
    `instr_cnt`, every part >= 10);
  * call-stack levels (one coin flip per program);
  * loop `(init, step, limit)` for the solved branch type;
  * HINT encodings (a uniform word from `hint_instr_c`, confirmed by the
    solver);
  * `riscv_jal_instr` jump rd (SV `dist` weights);
  * `randomize_gpr` operands: a soft constraint towards a uniformly drawn legal
    register, which every instruction-specific hard constraint overrides.

## Assembly audit of `ea5573d`

An audit of the programs generated at `ea5573d`, against what each test is
meant to exercise, found twelve generator defects: M-mode boot dropping its
mode-switch routine, `.align` padding that made injected illegal/HINT words
unreachable, 31-bit corner values, JAL-only jal streams, lost compressed
comments, an unported `mstatus_vs_c`, an off-by-one push/pop filler count,
and the five solution skews above. Each is fixed in its own commit, which
quotes the SV and gives the measurement.

## Not ported, or not reachable from the rv32imac target

* Debug mode: debug ROM, debug sub-programs, dcsr/dpc handshakes.
* S-mode: supervisor trap handlers, exception/interrupt delegation
  (`gen_delegation_instr`, `delegation_c`), SATP/page tables and the page-fault
  handler, kernel programs (only used with address translation).
* Vector and floating-point init and CSR randomization, bitmanip opcode helpers.
* `require_signature_addr` handshakes in the privileged-mode switch routine
  (the signature helpers themselves are ported and gated on the option).
* `riscv_load_store_rand_addr_instr_stream`.
* ePMP / Smepmp (`mseccfg`).
* `riscv_csr_test` is generated by `scripts/gen_csr_test.py` from a YAML model
  of the target's CSRs; that model is target-specific and not part of this
  branch.
* `push_gpr_to_kernel_stack` keeps pyflow's older scheme (user SP saved with
  `csrrw sp, <xscratch>, sp`) rather than the SV's push onto the kernel stack.
  Both are correct; the difference is visible in traces.
