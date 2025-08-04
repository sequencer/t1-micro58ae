
// Include register initializers in init blocks unless synthesis is set
`ifndef RANDOMIZE
  `ifdef RANDOMIZE_REG_INIT
    `define RANDOMIZE
  `endif // RANDOMIZE_REG_INIT
`endif // not def RANDOMIZE
`ifndef SYNTHESIS
  `ifndef ENABLE_INITIAL_REG_
    `define ENABLE_INITIAL_REG_
  `endif // not def ENABLE_INITIAL_REG_
`endif // not def SYNTHESIS

// Standard header to adapt well known macros for register randomization.

// RANDOM may be set to an expression that produces a 32-bit random unsigned value.
`ifndef RANDOM
  `define RANDOM $random
`endif // not def RANDOM

// Users can define INIT_RANDOM as general code that gets injected into the
// initializer block for modules with registers.
`ifndef INIT_RANDOM
  `define INIT_RANDOM
`endif // not def INIT_RANDOM

// If using random initialization, you can also define RANDOMIZE_DELAY to
// customize the delay used, otherwise 0.002 is used.
`ifndef RANDOMIZE_DELAY
  `define RANDOMIZE_DELAY 0.002
`endif // not def RANDOMIZE_DELAY

// Define INIT_RANDOM_PROLOG_ for use in our modules below.
`ifndef INIT_RANDOM_PROLOG_
  `ifdef RANDOMIZE
    `ifdef VERILATOR
      `define INIT_RANDOM_PROLOG_ `INIT_RANDOM
    `else  // VERILATOR
      `define INIT_RANDOM_PROLOG_ `INIT_RANDOM #`RANDOMIZE_DELAY begin end
    `endif // VERILATOR
  `else  // RANDOMIZE
    `define INIT_RANDOM_PROLOG_
  `endif // RANDOMIZE
`endif // not def INIT_RANDOM_PROLOG_
module T1TokenManager(
  input        clock,
               reset,
               instructionIssue_valid,
  input  [2:0] instructionIssue_bits_instructionIndex,
  input        instructionIssue_bits_writeV0,
               instructionIssue_bits_useV0AsMask,
               instructionIssue_bits_toLane,
               instructionIssue_bits_toMask,
               lsuWriteV0_0_valid,
  input  [2:0] lsuWriteV0_0_bits,
  input        lsuWriteV0_1_valid,
  input  [2:0] lsuWriteV0_1_bits,
  input        lsuWriteV0_2_valid,
  input  [2:0] lsuWriteV0_2_bits,
  input        lsuWriteV0_3_valid,
  input  [2:0] lsuWriteV0_3_bits,
  output       issueAllow,
  input  [7:0] instructionFinish_0,
               instructionFinish_1,
               instructionFinish_2,
               instructionFinish_3,
  output [7:0] v0WriteValid,
  input        maskUnitFree
);

  wire [7:0] issueIndex1H = 8'h1 << instructionIssue_bits_instructionIndex;
  wire [7:0] v0WriteValidVec_lsuWriteSet = lsuWriteV0_0_valid ? 8'h1 << lsuWriteV0_0_bits : 8'h0;
  wire       _maskUnitWriteV0_set_T = instructionIssue_valid & instructionIssue_bits_writeV0;
  wire       _GEN = _maskUnitWriteV0_set_T & instructionIssue_bits_toLane;
  wire       v0WriteValidVec_v0WriteIssue;
  assign v0WriteValidVec_v0WriteIssue = _GEN;
  wire       v0WriteValidVec_v0WriteIssue_1;
  assign v0WriteValidVec_v0WriteIssue_1 = _GEN;
  wire       v0WriteValidVec_v0WriteIssue_2;
  assign v0WriteValidVec_v0WriteIssue_2 = _GEN;
  wire       v0WriteValidVec_v0WriteIssue_3;
  assign v0WriteValidVec_v0WriteIssue_3 = _GEN;
  wire [7:0] v0WriteValidVec_updateOH = v0WriteValidVec_v0WriteIssue ? issueIndex1H : 8'h0;
  reg        v0WriteValidVec_res;
  reg        v0WriteValidVec_res_1;
  reg        v0WriteValidVec_res_2;
  reg        v0WriteValidVec_res_3;
  reg        v0WriteValidVec_res_4;
  reg        v0WriteValidVec_res_5;
  reg        v0WriteValidVec_res_6;
  reg        v0WriteValidVec_res_7;
  wire [1:0] v0WriteValidVec_lo_lo = {v0WriteValidVec_res_1, v0WriteValidVec_res};
  wire [1:0] v0WriteValidVec_lo_hi = {v0WriteValidVec_res_3, v0WriteValidVec_res_2};
  wire [3:0] v0WriteValidVec_lo = {v0WriteValidVec_lo_hi, v0WriteValidVec_lo_lo};
  wire [1:0] v0WriteValidVec_hi_lo = {v0WriteValidVec_res_5, v0WriteValidVec_res_4};
  wire [1:0] v0WriteValidVec_hi_hi = {v0WriteValidVec_res_7, v0WriteValidVec_res_6};
  wire [3:0] v0WriteValidVec_hi = {v0WriteValidVec_hi_hi, v0WriteValidVec_hi_lo};
  wire [7:0] v0WriteValidVec_0 = {v0WriteValidVec_hi, v0WriteValidVec_lo};
  wire [7:0] v0WriteValidVec_lsuWriteSet_1 = lsuWriteV0_1_valid ? 8'h1 << lsuWriteV0_1_bits : 8'h0;
  wire [7:0] v0WriteValidVec_updateOH_1 = v0WriteValidVec_v0WriteIssue_1 ? issueIndex1H : 8'h0;
  reg        v0WriteValidVec_res_8;
  reg        v0WriteValidVec_res_9;
  reg        v0WriteValidVec_res_10;
  reg        v0WriteValidVec_res_11;
  reg        v0WriteValidVec_res_12;
  reg        v0WriteValidVec_res_13;
  reg        v0WriteValidVec_res_14;
  reg        v0WriteValidVec_res_15;
  wire [1:0] v0WriteValidVec_lo_lo_1 = {v0WriteValidVec_res_9, v0WriteValidVec_res_8};
  wire [1:0] v0WriteValidVec_lo_hi_1 = {v0WriteValidVec_res_11, v0WriteValidVec_res_10};
  wire [3:0] v0WriteValidVec_lo_1 = {v0WriteValidVec_lo_hi_1, v0WriteValidVec_lo_lo_1};
  wire [1:0] v0WriteValidVec_hi_lo_1 = {v0WriteValidVec_res_13, v0WriteValidVec_res_12};
  wire [1:0] v0WriteValidVec_hi_hi_1 = {v0WriteValidVec_res_15, v0WriteValidVec_res_14};
  wire [3:0] v0WriteValidVec_hi_1 = {v0WriteValidVec_hi_hi_1, v0WriteValidVec_hi_lo_1};
  wire [7:0] v0WriteValidVec_1 = {v0WriteValidVec_hi_1, v0WriteValidVec_lo_1};
  wire [7:0] v0WriteValidVec_lsuWriteSet_2 = lsuWriteV0_2_valid ? 8'h1 << lsuWriteV0_2_bits : 8'h0;
  wire [7:0] v0WriteValidVec_updateOH_2 = v0WriteValidVec_v0WriteIssue_2 ? issueIndex1H : 8'h0;
  reg        v0WriteValidVec_res_16;
  reg        v0WriteValidVec_res_17;
  reg        v0WriteValidVec_res_18;
  reg        v0WriteValidVec_res_19;
  reg        v0WriteValidVec_res_20;
  reg        v0WriteValidVec_res_21;
  reg        v0WriteValidVec_res_22;
  reg        v0WriteValidVec_res_23;
  wire [1:0] v0WriteValidVec_lo_lo_2 = {v0WriteValidVec_res_17, v0WriteValidVec_res_16};
  wire [1:0] v0WriteValidVec_lo_hi_2 = {v0WriteValidVec_res_19, v0WriteValidVec_res_18};
  wire [3:0] v0WriteValidVec_lo_2 = {v0WriteValidVec_lo_hi_2, v0WriteValidVec_lo_lo_2};
  wire [1:0] v0WriteValidVec_hi_lo_2 = {v0WriteValidVec_res_21, v0WriteValidVec_res_20};
  wire [1:0] v0WriteValidVec_hi_hi_2 = {v0WriteValidVec_res_23, v0WriteValidVec_res_22};
  wire [3:0] v0WriteValidVec_hi_2 = {v0WriteValidVec_hi_hi_2, v0WriteValidVec_hi_lo_2};
  wire [7:0] v0WriteValidVec_2 = {v0WriteValidVec_hi_2, v0WriteValidVec_lo_2};
  wire [7:0] v0WriteValidVec_lsuWriteSet_3 = lsuWriteV0_3_valid ? 8'h1 << lsuWriteV0_3_bits : 8'h0;
  wire [7:0] v0WriteValidVec_updateOH_3 = v0WriteValidVec_v0WriteIssue_3 ? issueIndex1H : 8'h0;
  reg        v0WriteValidVec_res_24;
  reg        v0WriteValidVec_res_25;
  reg        v0WriteValidVec_res_26;
  reg        v0WriteValidVec_res_27;
  reg        v0WriteValidVec_res_28;
  reg        v0WriteValidVec_res_29;
  reg        v0WriteValidVec_res_30;
  reg        v0WriteValidVec_res_31;
  wire [1:0] v0WriteValidVec_lo_lo_3 = {v0WriteValidVec_res_25, v0WriteValidVec_res_24};
  wire [1:0] v0WriteValidVec_lo_hi_3 = {v0WriteValidVec_res_27, v0WriteValidVec_res_26};
  wire [3:0] v0WriteValidVec_lo_3 = {v0WriteValidVec_lo_hi_3, v0WriteValidVec_lo_lo_3};
  wire [1:0] v0WriteValidVec_hi_lo_3 = {v0WriteValidVec_res_29, v0WriteValidVec_res_28};
  wire [1:0] v0WriteValidVec_hi_hi_3 = {v0WriteValidVec_res_31, v0WriteValidVec_res_30};
  wire [3:0] v0WriteValidVec_hi_3 = {v0WriteValidVec_hi_hi_3, v0WriteValidVec_hi_lo_3};
  wire [7:0] v0WriteValidVec_3 = {v0WriteValidVec_hi_3, v0WriteValidVec_lo_3};
  wire       _GEN_0 = instructionIssue_valid & instructionIssue_bits_useV0AsMask & instructionIssue_bits_toLane;
  wire       useV0AsMaskToken_useV0Issue;
  assign useV0AsMaskToken_useV0Issue = _GEN_0;
  wire       useV0AsMaskToken_useV0Issue_1;
  assign useV0AsMaskToken_useV0Issue_1 = _GEN_0;
  wire       useV0AsMaskToken_useV0Issue_2;
  assign useV0AsMaskToken_useV0Issue_2 = _GEN_0;
  wire       useV0AsMaskToken_useV0Issue_3;
  assign useV0AsMaskToken_useV0Issue_3 = _GEN_0;
  wire [7:0] useV0AsMaskToken_updateOH = useV0AsMaskToken_useV0Issue ? issueIndex1H : 8'h0;
  reg        useV0AsMaskToken_res;
  reg        useV0AsMaskToken_res_1;
  reg        useV0AsMaskToken_res_2;
  reg        useV0AsMaskToken_res_3;
  reg        useV0AsMaskToken_res_4;
  reg        useV0AsMaskToken_res_5;
  reg        useV0AsMaskToken_res_6;
  reg        useV0AsMaskToken_res_7;
  wire [1:0] useV0AsMaskToken_lo_lo = {useV0AsMaskToken_res_1, useV0AsMaskToken_res};
  wire [1:0] useV0AsMaskToken_lo_hi = {useV0AsMaskToken_res_3, useV0AsMaskToken_res_2};
  wire [3:0] useV0AsMaskToken_lo = {useV0AsMaskToken_lo_hi, useV0AsMaskToken_lo_lo};
  wire [1:0] useV0AsMaskToken_hi_lo = {useV0AsMaskToken_res_5, useV0AsMaskToken_res_4};
  wire [1:0] useV0AsMaskToken_hi_hi = {useV0AsMaskToken_res_7, useV0AsMaskToken_res_6};
  wire [3:0] useV0AsMaskToken_hi = {useV0AsMaskToken_hi_hi, useV0AsMaskToken_hi_lo};
  wire [7:0] useV0AsMaskToken_updateOH_1 = useV0AsMaskToken_useV0Issue_1 ? issueIndex1H : 8'h0;
  reg        useV0AsMaskToken_res_8;
  reg        useV0AsMaskToken_res_9;
  reg        useV0AsMaskToken_res_10;
  reg        useV0AsMaskToken_res_11;
  reg        useV0AsMaskToken_res_12;
  reg        useV0AsMaskToken_res_13;
  reg        useV0AsMaskToken_res_14;
  reg        useV0AsMaskToken_res_15;
  wire [1:0] useV0AsMaskToken_lo_lo_1 = {useV0AsMaskToken_res_9, useV0AsMaskToken_res_8};
  wire [1:0] useV0AsMaskToken_lo_hi_1 = {useV0AsMaskToken_res_11, useV0AsMaskToken_res_10};
  wire [3:0] useV0AsMaskToken_lo_1 = {useV0AsMaskToken_lo_hi_1, useV0AsMaskToken_lo_lo_1};
  wire [1:0] useV0AsMaskToken_hi_lo_1 = {useV0AsMaskToken_res_13, useV0AsMaskToken_res_12};
  wire [1:0] useV0AsMaskToken_hi_hi_1 = {useV0AsMaskToken_res_15, useV0AsMaskToken_res_14};
  wire [3:0] useV0AsMaskToken_hi_1 = {useV0AsMaskToken_hi_hi_1, useV0AsMaskToken_hi_lo_1};
  wire [7:0] useV0AsMaskToken_updateOH_2 = useV0AsMaskToken_useV0Issue_2 ? issueIndex1H : 8'h0;
  reg        useV0AsMaskToken_res_16;
  reg        useV0AsMaskToken_res_17;
  reg        useV0AsMaskToken_res_18;
  reg        useV0AsMaskToken_res_19;
  reg        useV0AsMaskToken_res_20;
  reg        useV0AsMaskToken_res_21;
  reg        useV0AsMaskToken_res_22;
  reg        useV0AsMaskToken_res_23;
  wire [1:0] useV0AsMaskToken_lo_lo_2 = {useV0AsMaskToken_res_17, useV0AsMaskToken_res_16};
  wire [1:0] useV0AsMaskToken_lo_hi_2 = {useV0AsMaskToken_res_19, useV0AsMaskToken_res_18};
  wire [3:0] useV0AsMaskToken_lo_2 = {useV0AsMaskToken_lo_hi_2, useV0AsMaskToken_lo_lo_2};
  wire [1:0] useV0AsMaskToken_hi_lo_2 = {useV0AsMaskToken_res_21, useV0AsMaskToken_res_20};
  wire [1:0] useV0AsMaskToken_hi_hi_2 = {useV0AsMaskToken_res_23, useV0AsMaskToken_res_22};
  wire [3:0] useV0AsMaskToken_hi_2 = {useV0AsMaskToken_hi_hi_2, useV0AsMaskToken_hi_lo_2};
  wire [7:0] useV0AsMaskToken_updateOH_3 = useV0AsMaskToken_useV0Issue_3 ? issueIndex1H : 8'h0;
  reg        useV0AsMaskToken_res_24;
  reg        useV0AsMaskToken_res_25;
  reg        useV0AsMaskToken_res_26;
  reg        useV0AsMaskToken_res_27;
  reg        useV0AsMaskToken_res_28;
  reg        useV0AsMaskToken_res_29;
  reg        useV0AsMaskToken_res_30;
  reg        useV0AsMaskToken_res_31;
  wire [1:0] useV0AsMaskToken_lo_lo_3 = {useV0AsMaskToken_res_25, useV0AsMaskToken_res_24};
  wire [1:0] useV0AsMaskToken_lo_hi_3 = {useV0AsMaskToken_res_27, useV0AsMaskToken_res_26};
  wire [3:0] useV0AsMaskToken_lo_3 = {useV0AsMaskToken_lo_hi_3, useV0AsMaskToken_lo_lo_3};
  wire [1:0] useV0AsMaskToken_hi_lo_3 = {useV0AsMaskToken_res_29, useV0AsMaskToken_res_28};
  wire [1:0] useV0AsMaskToken_hi_hi_3 = {useV0AsMaskToken_res_31, useV0AsMaskToken_res_30};
  wire [3:0] useV0AsMaskToken_hi_3 = {useV0AsMaskToken_hi_hi_3, useV0AsMaskToken_hi_lo_3};
  wire [7:0] useV0AsMaskToken = {useV0AsMaskToken_hi, useV0AsMaskToken_lo} | {useV0AsMaskToken_hi_1, useV0AsMaskToken_lo_1} | {useV0AsMaskToken_hi_2, useV0AsMaskToken_lo_2} | {useV0AsMaskToken_hi_3, useV0AsMaskToken_lo_3};
  wire       maskUnitWriteV0_set = _maskUnitWriteV0_set_T & instructionIssue_bits_toMask;
  reg        maskUnitWriteV0;
  wire [7:0] _v0WriteValid_output = v0WriteValidVec_0 | v0WriteValidVec_1 | v0WriteValidVec_2 | v0WriteValidVec_3;
  wire       v0Conflict = instructionIssue_bits_writeV0 & (|useV0AsMaskToken) | instructionIssue_bits_useV0AsMask & ((|_v0WriteValid_output) | maskUnitWriteV0);
  always @(posedge clock) begin
    if (reset) begin
      v0WriteValidVec_res <= 1'h0;
      v0WriteValidVec_res_1 <= 1'h0;
      v0WriteValidVec_res_2 <= 1'h0;
      v0WriteValidVec_res_3 <= 1'h0;
      v0WriteValidVec_res_4 <= 1'h0;
      v0WriteValidVec_res_5 <= 1'h0;
      v0WriteValidVec_res_6 <= 1'h0;
      v0WriteValidVec_res_7 <= 1'h0;
      v0WriteValidVec_res_8 <= 1'h0;
      v0WriteValidVec_res_9 <= 1'h0;
      v0WriteValidVec_res_10 <= 1'h0;
      v0WriteValidVec_res_11 <= 1'h0;
      v0WriteValidVec_res_12 <= 1'h0;
      v0WriteValidVec_res_13 <= 1'h0;
      v0WriteValidVec_res_14 <= 1'h0;
      v0WriteValidVec_res_15 <= 1'h0;
      v0WriteValidVec_res_16 <= 1'h0;
      v0WriteValidVec_res_17 <= 1'h0;
      v0WriteValidVec_res_18 <= 1'h0;
      v0WriteValidVec_res_19 <= 1'h0;
      v0WriteValidVec_res_20 <= 1'h0;
      v0WriteValidVec_res_21 <= 1'h0;
      v0WriteValidVec_res_22 <= 1'h0;
      v0WriteValidVec_res_23 <= 1'h0;
      v0WriteValidVec_res_24 <= 1'h0;
      v0WriteValidVec_res_25 <= 1'h0;
      v0WriteValidVec_res_26 <= 1'h0;
      v0WriteValidVec_res_27 <= 1'h0;
      v0WriteValidVec_res_28 <= 1'h0;
      v0WriteValidVec_res_29 <= 1'h0;
      v0WriteValidVec_res_30 <= 1'h0;
      v0WriteValidVec_res_31 <= 1'h0;
      useV0AsMaskToken_res <= 1'h0;
      useV0AsMaskToken_res_1 <= 1'h0;
      useV0AsMaskToken_res_2 <= 1'h0;
      useV0AsMaskToken_res_3 <= 1'h0;
      useV0AsMaskToken_res_4 <= 1'h0;
      useV0AsMaskToken_res_5 <= 1'h0;
      useV0AsMaskToken_res_6 <= 1'h0;
      useV0AsMaskToken_res_7 <= 1'h0;
      useV0AsMaskToken_res_8 <= 1'h0;
      useV0AsMaskToken_res_9 <= 1'h0;
      useV0AsMaskToken_res_10 <= 1'h0;
      useV0AsMaskToken_res_11 <= 1'h0;
      useV0AsMaskToken_res_12 <= 1'h0;
      useV0AsMaskToken_res_13 <= 1'h0;
      useV0AsMaskToken_res_14 <= 1'h0;
      useV0AsMaskToken_res_15 <= 1'h0;
      useV0AsMaskToken_res_16 <= 1'h0;
      useV0AsMaskToken_res_17 <= 1'h0;
      useV0AsMaskToken_res_18 <= 1'h0;
      useV0AsMaskToken_res_19 <= 1'h0;
      useV0AsMaskToken_res_20 <= 1'h0;
      useV0AsMaskToken_res_21 <= 1'h0;
      useV0AsMaskToken_res_22 <= 1'h0;
      useV0AsMaskToken_res_23 <= 1'h0;
      useV0AsMaskToken_res_24 <= 1'h0;
      useV0AsMaskToken_res_25 <= 1'h0;
      useV0AsMaskToken_res_26 <= 1'h0;
      useV0AsMaskToken_res_27 <= 1'h0;
      useV0AsMaskToken_res_28 <= 1'h0;
      useV0AsMaskToken_res_29 <= 1'h0;
      useV0AsMaskToken_res_30 <= 1'h0;
      useV0AsMaskToken_res_31 <= 1'h0;
      maskUnitWriteV0 <= 1'h0;
    end
    else begin
      automatic logic [7:0] _v0WriteValidVec_T = v0WriteValidVec_updateOH | v0WriteValidVec_lsuWriteSet;
      automatic logic [7:0] _v0WriteValidVec_T_25 = v0WriteValidVec_updateOH_1 | v0WriteValidVec_lsuWriteSet_1;
      automatic logic [7:0] _v0WriteValidVec_T_50 = v0WriteValidVec_updateOH_2 | v0WriteValidVec_lsuWriteSet_2;
      automatic logic [7:0] _v0WriteValidVec_T_75 = v0WriteValidVec_updateOH_3 | v0WriteValidVec_lsuWriteSet_3;
      if (_v0WriteValidVec_T[0] | instructionFinish_0[0])
        v0WriteValidVec_res <= _v0WriteValidVec_T[0];
      if (_v0WriteValidVec_T[1] | instructionFinish_0[1])
        v0WriteValidVec_res_1 <= _v0WriteValidVec_T[1];
      if (_v0WriteValidVec_T[2] | instructionFinish_0[2])
        v0WriteValidVec_res_2 <= _v0WriteValidVec_T[2];
      if (_v0WriteValidVec_T[3] | instructionFinish_0[3])
        v0WriteValidVec_res_3 <= _v0WriteValidVec_T[3];
      if (_v0WriteValidVec_T[4] | instructionFinish_0[4])
        v0WriteValidVec_res_4 <= _v0WriteValidVec_T[4];
      if (_v0WriteValidVec_T[5] | instructionFinish_0[5])
        v0WriteValidVec_res_5 <= _v0WriteValidVec_T[5];
      if (_v0WriteValidVec_T[6] | instructionFinish_0[6])
        v0WriteValidVec_res_6 <= _v0WriteValidVec_T[6];
      if (_v0WriteValidVec_T[7] | instructionFinish_0[7])
        v0WriteValidVec_res_7 <= _v0WriteValidVec_T[7];
      if (_v0WriteValidVec_T_25[0] | instructionFinish_1[0])
        v0WriteValidVec_res_8 <= _v0WriteValidVec_T_25[0];
      if (_v0WriteValidVec_T_25[1] | instructionFinish_1[1])
        v0WriteValidVec_res_9 <= _v0WriteValidVec_T_25[1];
      if (_v0WriteValidVec_T_25[2] | instructionFinish_1[2])
        v0WriteValidVec_res_10 <= _v0WriteValidVec_T_25[2];
      if (_v0WriteValidVec_T_25[3] | instructionFinish_1[3])
        v0WriteValidVec_res_11 <= _v0WriteValidVec_T_25[3];
      if (_v0WriteValidVec_T_25[4] | instructionFinish_1[4])
        v0WriteValidVec_res_12 <= _v0WriteValidVec_T_25[4];
      if (_v0WriteValidVec_T_25[5] | instructionFinish_1[5])
        v0WriteValidVec_res_13 <= _v0WriteValidVec_T_25[5];
      if (_v0WriteValidVec_T_25[6] | instructionFinish_1[6])
        v0WriteValidVec_res_14 <= _v0WriteValidVec_T_25[6];
      if (_v0WriteValidVec_T_25[7] | instructionFinish_1[7])
        v0WriteValidVec_res_15 <= _v0WriteValidVec_T_25[7];
      if (_v0WriteValidVec_T_50[0] | instructionFinish_2[0])
        v0WriteValidVec_res_16 <= _v0WriteValidVec_T_50[0];
      if (_v0WriteValidVec_T_50[1] | instructionFinish_2[1])
        v0WriteValidVec_res_17 <= _v0WriteValidVec_T_50[1];
      if (_v0WriteValidVec_T_50[2] | instructionFinish_2[2])
        v0WriteValidVec_res_18 <= _v0WriteValidVec_T_50[2];
      if (_v0WriteValidVec_T_50[3] | instructionFinish_2[3])
        v0WriteValidVec_res_19 <= _v0WriteValidVec_T_50[3];
      if (_v0WriteValidVec_T_50[4] | instructionFinish_2[4])
        v0WriteValidVec_res_20 <= _v0WriteValidVec_T_50[4];
      if (_v0WriteValidVec_T_50[5] | instructionFinish_2[5])
        v0WriteValidVec_res_21 <= _v0WriteValidVec_T_50[5];
      if (_v0WriteValidVec_T_50[6] | instructionFinish_2[6])
        v0WriteValidVec_res_22 <= _v0WriteValidVec_T_50[6];
      if (_v0WriteValidVec_T_50[7] | instructionFinish_2[7])
        v0WriteValidVec_res_23 <= _v0WriteValidVec_T_50[7];
      if (_v0WriteValidVec_T_75[0] | instructionFinish_3[0])
        v0WriteValidVec_res_24 <= _v0WriteValidVec_T_75[0];
      if (_v0WriteValidVec_T_75[1] | instructionFinish_3[1])
        v0WriteValidVec_res_25 <= _v0WriteValidVec_T_75[1];
      if (_v0WriteValidVec_T_75[2] | instructionFinish_3[2])
        v0WriteValidVec_res_26 <= _v0WriteValidVec_T_75[2];
      if (_v0WriteValidVec_T_75[3] | instructionFinish_3[3])
        v0WriteValidVec_res_27 <= _v0WriteValidVec_T_75[3];
      if (_v0WriteValidVec_T_75[4] | instructionFinish_3[4])
        v0WriteValidVec_res_28 <= _v0WriteValidVec_T_75[4];
      if (_v0WriteValidVec_T_75[5] | instructionFinish_3[5])
        v0WriteValidVec_res_29 <= _v0WriteValidVec_T_75[5];
      if (_v0WriteValidVec_T_75[6] | instructionFinish_3[6])
        v0WriteValidVec_res_30 <= _v0WriteValidVec_T_75[6];
      if (_v0WriteValidVec_T_75[7] | instructionFinish_3[7])
        v0WriteValidVec_res_31 <= _v0WriteValidVec_T_75[7];
      if (useV0AsMaskToken_updateOH[0] | instructionFinish_0[0])
        useV0AsMaskToken_res <= useV0AsMaskToken_updateOH[0];
      if (useV0AsMaskToken_updateOH[1] | instructionFinish_0[1])
        useV0AsMaskToken_res_1 <= useV0AsMaskToken_updateOH[1];
      if (useV0AsMaskToken_updateOH[2] | instructionFinish_0[2])
        useV0AsMaskToken_res_2 <= useV0AsMaskToken_updateOH[2];
      if (useV0AsMaskToken_updateOH[3] | instructionFinish_0[3])
        useV0AsMaskToken_res_3 <= useV0AsMaskToken_updateOH[3];
      if (useV0AsMaskToken_updateOH[4] | instructionFinish_0[4])
        useV0AsMaskToken_res_4 <= useV0AsMaskToken_updateOH[4];
      if (useV0AsMaskToken_updateOH[5] | instructionFinish_0[5])
        useV0AsMaskToken_res_5 <= useV0AsMaskToken_updateOH[5];
      if (useV0AsMaskToken_updateOH[6] | instructionFinish_0[6])
        useV0AsMaskToken_res_6 <= useV0AsMaskToken_updateOH[6];
      if (useV0AsMaskToken_updateOH[7] | instructionFinish_0[7])
        useV0AsMaskToken_res_7 <= useV0AsMaskToken_updateOH[7];
      if (useV0AsMaskToken_updateOH_1[0] | instructionFinish_1[0])
        useV0AsMaskToken_res_8 <= useV0AsMaskToken_updateOH_1[0];
      if (useV0AsMaskToken_updateOH_1[1] | instructionFinish_1[1])
        useV0AsMaskToken_res_9 <= useV0AsMaskToken_updateOH_1[1];
      if (useV0AsMaskToken_updateOH_1[2] | instructionFinish_1[2])
        useV0AsMaskToken_res_10 <= useV0AsMaskToken_updateOH_1[2];
      if (useV0AsMaskToken_updateOH_1[3] | instructionFinish_1[3])
        useV0AsMaskToken_res_11 <= useV0AsMaskToken_updateOH_1[3];
      if (useV0AsMaskToken_updateOH_1[4] | instructionFinish_1[4])
        useV0AsMaskToken_res_12 <= useV0AsMaskToken_updateOH_1[4];
      if (useV0AsMaskToken_updateOH_1[5] | instructionFinish_1[5])
        useV0AsMaskToken_res_13 <= useV0AsMaskToken_updateOH_1[5];
      if (useV0AsMaskToken_updateOH_1[6] | instructionFinish_1[6])
        useV0AsMaskToken_res_14 <= useV0AsMaskToken_updateOH_1[6];
      if (useV0AsMaskToken_updateOH_1[7] | instructionFinish_1[7])
        useV0AsMaskToken_res_15 <= useV0AsMaskToken_updateOH_1[7];
      if (useV0AsMaskToken_updateOH_2[0] | instructionFinish_2[0])
        useV0AsMaskToken_res_16 <= useV0AsMaskToken_updateOH_2[0];
      if (useV0AsMaskToken_updateOH_2[1] | instructionFinish_2[1])
        useV0AsMaskToken_res_17 <= useV0AsMaskToken_updateOH_2[1];
      if (useV0AsMaskToken_updateOH_2[2] | instructionFinish_2[2])
        useV0AsMaskToken_res_18 <= useV0AsMaskToken_updateOH_2[2];
      if (useV0AsMaskToken_updateOH_2[3] | instructionFinish_2[3])
        useV0AsMaskToken_res_19 <= useV0AsMaskToken_updateOH_2[3];
      if (useV0AsMaskToken_updateOH_2[4] | instructionFinish_2[4])
        useV0AsMaskToken_res_20 <= useV0AsMaskToken_updateOH_2[4];
      if (useV0AsMaskToken_updateOH_2[5] | instructionFinish_2[5])
        useV0AsMaskToken_res_21 <= useV0AsMaskToken_updateOH_2[5];
      if (useV0AsMaskToken_updateOH_2[6] | instructionFinish_2[6])
        useV0AsMaskToken_res_22 <= useV0AsMaskToken_updateOH_2[6];
      if (useV0AsMaskToken_updateOH_2[7] | instructionFinish_2[7])
        useV0AsMaskToken_res_23 <= useV0AsMaskToken_updateOH_2[7];
      if (useV0AsMaskToken_updateOH_3[0] | instructionFinish_3[0])
        useV0AsMaskToken_res_24 <= useV0AsMaskToken_updateOH_3[0];
      if (useV0AsMaskToken_updateOH_3[1] | instructionFinish_3[1])
        useV0AsMaskToken_res_25 <= useV0AsMaskToken_updateOH_3[1];
      if (useV0AsMaskToken_updateOH_3[2] | instructionFinish_3[2])
        useV0AsMaskToken_res_26 <= useV0AsMaskToken_updateOH_3[2];
      if (useV0AsMaskToken_updateOH_3[3] | instructionFinish_3[3])
        useV0AsMaskToken_res_27 <= useV0AsMaskToken_updateOH_3[3];
      if (useV0AsMaskToken_updateOH_3[4] | instructionFinish_3[4])
        useV0AsMaskToken_res_28 <= useV0AsMaskToken_updateOH_3[4];
      if (useV0AsMaskToken_updateOH_3[5] | instructionFinish_3[5])
        useV0AsMaskToken_res_29 <= useV0AsMaskToken_updateOH_3[5];
      if (useV0AsMaskToken_updateOH_3[6] | instructionFinish_3[6])
        useV0AsMaskToken_res_30 <= useV0AsMaskToken_updateOH_3[6];
      if (useV0AsMaskToken_updateOH_3[7] | instructionFinish_3[7])
        useV0AsMaskToken_res_31 <= useV0AsMaskToken_updateOH_3[7];
      if (maskUnitWriteV0_set | maskUnitFree)
        maskUnitWriteV0 <= maskUnitWriteV0_set;
    end
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:2];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [1:0] i = 2'h0; i < 2'h3; i += 2'h1) begin
          _RANDOM[i] = `RANDOM;
        end
        v0WriteValidVec_res = _RANDOM[2'h0][0];
        v0WriteValidVec_res_1 = _RANDOM[2'h0][1];
        v0WriteValidVec_res_2 = _RANDOM[2'h0][2];
        v0WriteValidVec_res_3 = _RANDOM[2'h0][3];
        v0WriteValidVec_res_4 = _RANDOM[2'h0][4];
        v0WriteValidVec_res_5 = _RANDOM[2'h0][5];
        v0WriteValidVec_res_6 = _RANDOM[2'h0][6];
        v0WriteValidVec_res_7 = _RANDOM[2'h0][7];
        v0WriteValidVec_res_8 = _RANDOM[2'h0][8];
        v0WriteValidVec_res_9 = _RANDOM[2'h0][9];
        v0WriteValidVec_res_10 = _RANDOM[2'h0][10];
        v0WriteValidVec_res_11 = _RANDOM[2'h0][11];
        v0WriteValidVec_res_12 = _RANDOM[2'h0][12];
        v0WriteValidVec_res_13 = _RANDOM[2'h0][13];
        v0WriteValidVec_res_14 = _RANDOM[2'h0][14];
        v0WriteValidVec_res_15 = _RANDOM[2'h0][15];
        v0WriteValidVec_res_16 = _RANDOM[2'h0][16];
        v0WriteValidVec_res_17 = _RANDOM[2'h0][17];
        v0WriteValidVec_res_18 = _RANDOM[2'h0][18];
        v0WriteValidVec_res_19 = _RANDOM[2'h0][19];
        v0WriteValidVec_res_20 = _RANDOM[2'h0][20];
        v0WriteValidVec_res_21 = _RANDOM[2'h0][21];
        v0WriteValidVec_res_22 = _RANDOM[2'h0][22];
        v0WriteValidVec_res_23 = _RANDOM[2'h0][23];
        v0WriteValidVec_res_24 = _RANDOM[2'h0][24];
        v0WriteValidVec_res_25 = _RANDOM[2'h0][25];
        v0WriteValidVec_res_26 = _RANDOM[2'h0][26];
        v0WriteValidVec_res_27 = _RANDOM[2'h0][27];
        v0WriteValidVec_res_28 = _RANDOM[2'h0][28];
        v0WriteValidVec_res_29 = _RANDOM[2'h0][29];
        v0WriteValidVec_res_30 = _RANDOM[2'h0][30];
        v0WriteValidVec_res_31 = _RANDOM[2'h0][31];
        useV0AsMaskToken_res = _RANDOM[2'h1][0];
        useV0AsMaskToken_res_1 = _RANDOM[2'h1][1];
        useV0AsMaskToken_res_2 = _RANDOM[2'h1][2];
        useV0AsMaskToken_res_3 = _RANDOM[2'h1][3];
        useV0AsMaskToken_res_4 = _RANDOM[2'h1][4];
        useV0AsMaskToken_res_5 = _RANDOM[2'h1][5];
        useV0AsMaskToken_res_6 = _RANDOM[2'h1][6];
        useV0AsMaskToken_res_7 = _RANDOM[2'h1][7];
        useV0AsMaskToken_res_8 = _RANDOM[2'h1][8];
        useV0AsMaskToken_res_9 = _RANDOM[2'h1][9];
        useV0AsMaskToken_res_10 = _RANDOM[2'h1][10];
        useV0AsMaskToken_res_11 = _RANDOM[2'h1][11];
        useV0AsMaskToken_res_12 = _RANDOM[2'h1][12];
        useV0AsMaskToken_res_13 = _RANDOM[2'h1][13];
        useV0AsMaskToken_res_14 = _RANDOM[2'h1][14];
        useV0AsMaskToken_res_15 = _RANDOM[2'h1][15];
        useV0AsMaskToken_res_16 = _RANDOM[2'h1][16];
        useV0AsMaskToken_res_17 = _RANDOM[2'h1][17];
        useV0AsMaskToken_res_18 = _RANDOM[2'h1][18];
        useV0AsMaskToken_res_19 = _RANDOM[2'h1][19];
        useV0AsMaskToken_res_20 = _RANDOM[2'h1][20];
        useV0AsMaskToken_res_21 = _RANDOM[2'h1][21];
        useV0AsMaskToken_res_22 = _RANDOM[2'h1][22];
        useV0AsMaskToken_res_23 = _RANDOM[2'h1][23];
        useV0AsMaskToken_res_24 = _RANDOM[2'h1][24];
        useV0AsMaskToken_res_25 = _RANDOM[2'h1][25];
        useV0AsMaskToken_res_26 = _RANDOM[2'h1][26];
        useV0AsMaskToken_res_27 = _RANDOM[2'h1][27];
        useV0AsMaskToken_res_28 = _RANDOM[2'h1][28];
        useV0AsMaskToken_res_29 = _RANDOM[2'h1][29];
        useV0AsMaskToken_res_30 = _RANDOM[2'h1][30];
        useV0AsMaskToken_res_31 = _RANDOM[2'h1][31];
        maskUnitWriteV0 = _RANDOM[2'h2][0];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  assign issueAllow = ~v0Conflict;
  assign v0WriteValid = _v0WriteValid_output;
endmodule

