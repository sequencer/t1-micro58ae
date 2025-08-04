
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
module VRF(
  input          clock,
                 reset,
  output         readRequests_0_ready,
  input          readRequests_0_valid,
  input  [4:0]   readRequests_0_bits_vs,
  input  [1:0]   readRequests_0_bits_readSource,
  input  [4:0]   readRequests_0_bits_offset,
  input  [2:0]   readRequests_0_bits_instructionIndex,
  output         readRequests_1_ready,
  input          readRequests_1_valid,
  input  [4:0]   readRequests_1_bits_vs,
  input  [1:0]   readRequests_1_bits_readSource,
  input  [4:0]   readRequests_1_bits_offset,
  input  [2:0]   readRequests_1_bits_instructionIndex,
  output         readRequests_2_ready,
  input          readRequests_2_valid,
  input  [4:0]   readRequests_2_bits_vs,
  input  [1:0]   readRequests_2_bits_readSource,
  input  [4:0]   readRequests_2_bits_offset,
  input  [2:0]   readRequests_2_bits_instructionIndex,
  output         readRequests_3_ready,
  input          readRequests_3_valid,
  input  [4:0]   readRequests_3_bits_vs,
  input  [1:0]   readRequests_3_bits_readSource,
  input  [4:0]   readRequests_3_bits_offset,
  input  [2:0]   readRequests_3_bits_instructionIndex,
  output         readRequests_4_ready,
  input          readRequests_4_valid,
  input  [4:0]   readRequests_4_bits_vs,
  input  [1:0]   readRequests_4_bits_readSource,
  input  [4:0]   readRequests_4_bits_offset,
  input  [2:0]   readRequests_4_bits_instructionIndex,
  output         readRequests_5_ready,
  input          readRequests_5_valid,
  input  [4:0]   readRequests_5_bits_vs,
  input  [1:0]   readRequests_5_bits_readSource,
  input  [4:0]   readRequests_5_bits_offset,
  input  [2:0]   readRequests_5_bits_instructionIndex,
  output         readRequests_6_ready,
  input          readRequests_6_valid,
  input  [4:0]   readRequests_6_bits_vs,
  input  [1:0]   readRequests_6_bits_readSource,
  input  [4:0]   readRequests_6_bits_offset,
  input  [2:0]   readRequests_6_bits_instructionIndex,
  output         readRequests_7_ready,
  input          readRequests_7_valid,
  input  [4:0]   readRequests_7_bits_vs,
  input  [1:0]   readRequests_7_bits_readSource,
  input  [4:0]   readRequests_7_bits_offset,
  input  [2:0]   readRequests_7_bits_instructionIndex,
  output         readRequests_8_ready,
  input          readRequests_8_valid,
  input  [4:0]   readRequests_8_bits_vs,
  input  [1:0]   readRequests_8_bits_readSource,
  input  [4:0]   readRequests_8_bits_offset,
  input  [2:0]   readRequests_8_bits_instructionIndex,
  output         readRequests_9_ready,
  input          readRequests_9_valid,
  input  [4:0]   readRequests_9_bits_vs,
  input  [1:0]   readRequests_9_bits_readSource,
  input  [4:0]   readRequests_9_bits_offset,
  input  [2:0]   readRequests_9_bits_instructionIndex,
  output         readRequests_10_ready,
  input          readRequests_10_valid,
  input  [4:0]   readRequests_10_bits_vs,
  input  [1:0]   readRequests_10_bits_readSource,
  input  [4:0]   readRequests_10_bits_offset,
  input  [2:0]   readRequests_10_bits_instructionIndex,
  output         readRequests_11_ready,
  input          readRequests_11_valid,
  input  [4:0]   readRequests_11_bits_vs,
  input  [1:0]   readRequests_11_bits_readSource,
  input  [4:0]   readRequests_11_bits_offset,
  input  [2:0]   readRequests_11_bits_instructionIndex,
  output         readRequests_12_ready,
  input          readRequests_12_valid,
  input  [4:0]   readRequests_12_bits_vs,
  input  [1:0]   readRequests_12_bits_readSource,
  input  [4:0]   readRequests_12_bits_offset,
  input  [2:0]   readRequests_12_bits_instructionIndex,
  output         readRequests_13_ready,
  input          readRequests_13_valid,
  input  [4:0]   readRequests_13_bits_vs,
  input  [1:0]   readRequests_13_bits_readSource,
  input  [4:0]   readRequests_13_bits_offset,
  input  [2:0]   readRequests_13_bits_instructionIndex,
  input  [4:0]   readCheck_0_vs,
                 readCheck_0_offset,
  input  [2:0]   readCheck_0_instructionIndex,
  input  [4:0]   readCheck_1_vs,
                 readCheck_1_offset,
  input  [2:0]   readCheck_1_instructionIndex,
  input  [4:0]   readCheck_2_vs,
                 readCheck_2_offset,
  input  [2:0]   readCheck_2_instructionIndex,
  input  [4:0]   readCheck_3_vs,
                 readCheck_3_offset,
  input  [2:0]   readCheck_3_instructionIndex,
  input  [4:0]   readCheck_4_vs,
                 readCheck_4_offset,
  input  [2:0]   readCheck_4_instructionIndex,
  input  [4:0]   readCheck_5_vs,
                 readCheck_5_offset,
  input  [2:0]   readCheck_5_instructionIndex,
  input  [4:0]   readCheck_6_vs,
                 readCheck_6_offset,
  input  [2:0]   readCheck_6_instructionIndex,
  input  [4:0]   readCheck_7_vs,
                 readCheck_7_offset,
  input  [2:0]   readCheck_7_instructionIndex,
  input  [4:0]   readCheck_8_vs,
                 readCheck_8_offset,
  input  [2:0]   readCheck_8_instructionIndex,
  input  [4:0]   readCheck_9_vs,
                 readCheck_9_offset,
  input  [2:0]   readCheck_9_instructionIndex,
  input  [4:0]   readCheck_10_vs,
                 readCheck_10_offset,
  input  [2:0]   readCheck_10_instructionIndex,
  input  [4:0]   readCheck_11_vs,
                 readCheck_11_offset,
  input  [2:0]   readCheck_11_instructionIndex,
  input  [4:0]   readCheck_12_vs,
                 readCheck_12_offset,
  input  [2:0]   readCheck_12_instructionIndex,
  input  [4:0]   readCheck_13_vs,
                 readCheck_13_offset,
  input  [2:0]   readCheck_13_instructionIndex,
  output         readCheckResult_0,
                 readCheckResult_1,
                 readCheckResult_2,
                 readCheckResult_3,
                 readCheckResult_4,
                 readCheckResult_5,
                 readCheckResult_6,
                 readCheckResult_7,
                 readCheckResult_8,
                 readCheckResult_9,
                 readCheckResult_10,
                 readCheckResult_11,
                 readCheckResult_12,
                 readCheckResult_13,
  output [31:0]  readResults_0,
                 readResults_1,
                 readResults_2,
                 readResults_3,
                 readResults_4,
                 readResults_5,
                 readResults_6,
                 readResults_7,
                 readResults_8,
                 readResults_9,
                 readResults_10,
                 readResults_11,
                 readResults_12,
                 readResults_13,
  output         write_ready,
  input          write_valid,
  input  [4:0]   write_bits_vd,
                 write_bits_offset,
  input  [3:0]   write_bits_mask,
  input  [31:0]  write_bits_data,
  input          write_bits_last,
  input  [2:0]   write_bits_instructionIndex,
  input  [4:0]   writeCheck_0_vd,
                 writeCheck_0_offset,
  input  [2:0]   writeCheck_0_instructionIndex,
  input  [4:0]   writeCheck_1_vd,
                 writeCheck_1_offset,
  input  [2:0]   writeCheck_1_instructionIndex,
  input  [4:0]   writeCheck_2_vd,
                 writeCheck_2_offset,
  input  [2:0]   writeCheck_2_instructionIndex,
  input  [4:0]   writeCheck_3_vd,
                 writeCheck_3_offset,
  input  [2:0]   writeCheck_3_instructionIndex,
  input  [4:0]   writeCheck_4_vd,
                 writeCheck_4_offset,
  input  [2:0]   writeCheck_4_instructionIndex,
  input  [4:0]   writeCheck_5_vd,
                 writeCheck_5_offset,
  input  [2:0]   writeCheck_5_instructionIndex,
  input  [4:0]   writeCheck_6_vd,
                 writeCheck_6_offset,
  input  [2:0]   writeCheck_6_instructionIndex,
  output         writeAllow_0,
                 writeAllow_1,
                 writeAllow_2,
                 writeAllow_3,
                 writeAllow_4,
                 writeAllow_5,
                 writeAllow_6,
  input          instructionWriteReport_valid,
                 instructionWriteReport_bits_vd_valid,
  input  [4:0]   instructionWriteReport_bits_vd_bits,
  input          instructionWriteReport_bits_vs1_valid,
  input  [4:0]   instructionWriteReport_bits_vs1_bits,
                 instructionWriteReport_bits_vs2,
  input  [2:0]   instructionWriteReport_bits_instIndex,
  input          instructionWriteReport_bits_ls,
                 instructionWriteReport_bits_st,
                 instructionWriteReport_bits_gather,
                 instructionWriteReport_bits_gather16,
                 instructionWriteReport_bits_crossWrite,
                 instructionWriteReport_bits_crossRead,
                 instructionWriteReport_bits_indexType,
                 instructionWriteReport_bits_ma,
                 instructionWriteReport_bits_onlyRead,
                 instructionWriteReport_bits_slow,
  input  [255:0] instructionWriteReport_bits_elementMask,
  input          instructionWriteReport_bits_state_stFinish,
                 instructionWriteReport_bits_state_wWriteQueueClear,
                 instructionWriteReport_bits_state_wLaneLastReport,
                 instructionWriteReport_bits_state_wTopLastReport,
  input  [7:0]   instructionLastReport,
                 lsuLastReport,
  output [7:0]   vrfSlotRelease,
  input  [7:0]   dataInLane,
                 loadDataInLSUWriteQueue
);

  wire         _writeAllow_6_checkModule_4_checkResult;
  wire         _writeAllow_6_checkModule_3_checkResult;
  wire         _writeAllow_6_checkModule_2_checkResult;
  wire         _writeAllow_6_checkModule_1_checkResult;
  wire         _writeAllow_6_checkModule_checkResult;
  wire         _writeAllow_5_checkModule_4_checkResult;
  wire         _writeAllow_5_checkModule_3_checkResult;
  wire         _writeAllow_5_checkModule_2_checkResult;
  wire         _writeAllow_5_checkModule_1_checkResult;
  wire         _writeAllow_5_checkModule_checkResult;
  wire         _writeAllow_4_checkModule_4_checkResult;
  wire         _writeAllow_4_checkModule_3_checkResult;
  wire         _writeAllow_4_checkModule_2_checkResult;
  wire         _writeAllow_4_checkModule_1_checkResult;
  wire         _writeAllow_4_checkModule_checkResult;
  wire         _writeAllow_3_checkModule_4_checkResult;
  wire         _writeAllow_3_checkModule_3_checkResult;
  wire         _writeAllow_3_checkModule_2_checkResult;
  wire         _writeAllow_3_checkModule_1_checkResult;
  wire         _writeAllow_3_checkModule_checkResult;
  wire         _writeAllow_2_checkModule_4_checkResult;
  wire         _writeAllow_2_checkModule_3_checkResult;
  wire         _writeAllow_2_checkModule_2_checkResult;
  wire         _writeAllow_2_checkModule_1_checkResult;
  wire         _writeAllow_2_checkModule_checkResult;
  wire         _writeAllow_1_checkModule_4_checkResult;
  wire         _writeAllow_1_checkModule_3_checkResult;
  wire         _writeAllow_1_checkModule_2_checkResult;
  wire         _writeAllow_1_checkModule_1_checkResult;
  wire         _writeAllow_1_checkModule_checkResult;
  wire         _writeAllow_0_checkModule_4_checkResult;
  wire         _writeAllow_0_checkModule_3_checkResult;
  wire         _writeAllow_0_checkModule_2_checkResult;
  wire         _writeAllow_0_checkModule_1_checkResult;
  wire         _writeAllow_0_checkModule_checkResult;
  wire         _checkResult_ChainingCheck_readPort13_record4_checkResult;
  wire         _checkResult_ChainingCheck_readPort13_record3_checkResult;
  wire         _checkResult_ChainingCheck_readPort13_record2_checkResult;
  wire         _checkResult_ChainingCheck_readPort13_record1_checkResult;
  wire         _checkResult_ChainingCheck_readPort13_record0_checkResult;
  wire         _readCheckResult_13_checkModule_4_checkResult;
  wire         _readCheckResult_13_checkModule_3_checkResult;
  wire         _readCheckResult_13_checkModule_2_checkResult;
  wire         _readCheckResult_13_checkModule_1_checkResult;
  wire         _readCheckResult_13_checkModule_checkResult;
  wire         _readCheckResult_12_checkModule_4_checkResult;
  wire         _readCheckResult_12_checkModule_3_checkResult;
  wire         _readCheckResult_12_checkModule_2_checkResult;
  wire         _readCheckResult_12_checkModule_1_checkResult;
  wire         _readCheckResult_12_checkModule_checkResult;
  wire         _readCheckResult_11_checkModule_4_checkResult;
  wire         _readCheckResult_11_checkModule_3_checkResult;
  wire         _readCheckResult_11_checkModule_2_checkResult;
  wire         _readCheckResult_11_checkModule_1_checkResult;
  wire         _readCheckResult_11_checkModule_checkResult;
  wire         _readCheckResult_10_checkModule_4_checkResult;
  wire         _readCheckResult_10_checkModule_3_checkResult;
  wire         _readCheckResult_10_checkModule_2_checkResult;
  wire         _readCheckResult_10_checkModule_1_checkResult;
  wire         _readCheckResult_10_checkModule_checkResult;
  wire         _readCheckResult_9_checkModule_4_checkResult;
  wire         _readCheckResult_9_checkModule_3_checkResult;
  wire         _readCheckResult_9_checkModule_2_checkResult;
  wire         _readCheckResult_9_checkModule_1_checkResult;
  wire         _readCheckResult_9_checkModule_checkResult;
  wire         _readCheckResult_8_checkModule_4_checkResult;
  wire         _readCheckResult_8_checkModule_3_checkResult;
  wire         _readCheckResult_8_checkModule_2_checkResult;
  wire         _readCheckResult_8_checkModule_1_checkResult;
  wire         _readCheckResult_8_checkModule_checkResult;
  wire         _readCheckResult_7_checkModule_4_checkResult;
  wire         _readCheckResult_7_checkModule_3_checkResult;
  wire         _readCheckResult_7_checkModule_2_checkResult;
  wire         _readCheckResult_7_checkModule_1_checkResult;
  wire         _readCheckResult_7_checkModule_checkResult;
  wire         _readCheckResult_6_checkModule_4_checkResult;
  wire         _readCheckResult_6_checkModule_3_checkResult;
  wire         _readCheckResult_6_checkModule_2_checkResult;
  wire         _readCheckResult_6_checkModule_1_checkResult;
  wire         _readCheckResult_6_checkModule_checkResult;
  wire         _readCheckResult_5_checkModule_4_checkResult;
  wire         _readCheckResult_5_checkModule_3_checkResult;
  wire         _readCheckResult_5_checkModule_2_checkResult;
  wire         _readCheckResult_5_checkModule_1_checkResult;
  wire         _readCheckResult_5_checkModule_checkResult;
  wire         _readCheckResult_4_checkModule_4_checkResult;
  wire         _readCheckResult_4_checkModule_3_checkResult;
  wire         _readCheckResult_4_checkModule_2_checkResult;
  wire         _readCheckResult_4_checkModule_1_checkResult;
  wire         _readCheckResult_4_checkModule_checkResult;
  wire         _readCheckResult_3_checkModule_4_checkResult;
  wire         _readCheckResult_3_checkModule_3_checkResult;
  wire         _readCheckResult_3_checkModule_2_checkResult;
  wire         _readCheckResult_3_checkModule_1_checkResult;
  wire         _readCheckResult_3_checkModule_checkResult;
  wire         _readCheckResult_2_checkModule_4_checkResult;
  wire         _readCheckResult_2_checkModule_3_checkResult;
  wire         _readCheckResult_2_checkModule_2_checkResult;
  wire         _readCheckResult_2_checkModule_1_checkResult;
  wire         _readCheckResult_2_checkModule_checkResult;
  wire         _readCheckResult_1_checkModule_4_checkResult;
  wire         _readCheckResult_1_checkModule_3_checkResult;
  wire         _readCheckResult_1_checkModule_2_checkResult;
  wire         _readCheckResult_1_checkModule_1_checkResult;
  wire         _readCheckResult_1_checkModule_checkResult;
  wire         _readCheckResult_0_checkModule_4_checkResult;
  wire         _readCheckResult_0_checkModule_3_checkResult;
  wire         _readCheckResult_0_checkModule_2_checkResult;
  wire         _readCheckResult_0_checkModule_1_checkResult;
  wire         _readCheckResult_0_checkModule_checkResult;
  wire         readRequests_0_valid_0 = readRequests_0_valid;
  wire [4:0]   readRequests_0_bits_vs_0 = readRequests_0_bits_vs;
  wire [1:0]   readRequests_0_bits_readSource_0 = readRequests_0_bits_readSource;
  wire [4:0]   readRequests_0_bits_offset_0 = readRequests_0_bits_offset;
  wire [2:0]   readRequests_0_bits_instructionIndex_0 = readRequests_0_bits_instructionIndex;
  wire         readRequests_1_valid_0 = readRequests_1_valid;
  wire [4:0]   readRequests_1_bits_vs_0 = readRequests_1_bits_vs;
  wire [1:0]   readRequests_1_bits_readSource_0 = readRequests_1_bits_readSource;
  wire [4:0]   readRequests_1_bits_offset_0 = readRequests_1_bits_offset;
  wire [2:0]   readRequests_1_bits_instructionIndex_0 = readRequests_1_bits_instructionIndex;
  wire         readRequests_2_valid_0 = readRequests_2_valid;
  wire [4:0]   readRequests_2_bits_vs_0 = readRequests_2_bits_vs;
  wire [1:0]   readRequests_2_bits_readSource_0 = readRequests_2_bits_readSource;
  wire [4:0]   readRequests_2_bits_offset_0 = readRequests_2_bits_offset;
  wire [2:0]   readRequests_2_bits_instructionIndex_0 = readRequests_2_bits_instructionIndex;
  wire         readRequests_3_valid_0 = readRequests_3_valid;
  wire [4:0]   readRequests_3_bits_vs_0 = readRequests_3_bits_vs;
  wire [1:0]   readRequests_3_bits_readSource_0 = readRequests_3_bits_readSource;
  wire [4:0]   readRequests_3_bits_offset_0 = readRequests_3_bits_offset;
  wire [2:0]   readRequests_3_bits_instructionIndex_0 = readRequests_3_bits_instructionIndex;
  wire         readRequests_4_valid_0 = readRequests_4_valid;
  wire [4:0]   readRequests_4_bits_vs_0 = readRequests_4_bits_vs;
  wire [1:0]   readRequests_4_bits_readSource_0 = readRequests_4_bits_readSource;
  wire [4:0]   readRequests_4_bits_offset_0 = readRequests_4_bits_offset;
  wire [2:0]   readRequests_4_bits_instructionIndex_0 = readRequests_4_bits_instructionIndex;
  wire         readRequests_5_valid_0 = readRequests_5_valid;
  wire [4:0]   readRequests_5_bits_vs_0 = readRequests_5_bits_vs;
  wire [1:0]   readRequests_5_bits_readSource_0 = readRequests_5_bits_readSource;
  wire [4:0]   readRequests_5_bits_offset_0 = readRequests_5_bits_offset;
  wire [2:0]   readRequests_5_bits_instructionIndex_0 = readRequests_5_bits_instructionIndex;
  wire         readRequests_6_valid_0 = readRequests_6_valid;
  wire [4:0]   readRequests_6_bits_vs_0 = readRequests_6_bits_vs;
  wire [1:0]   readRequests_6_bits_readSource_0 = readRequests_6_bits_readSource;
  wire [4:0]   readRequests_6_bits_offset_0 = readRequests_6_bits_offset;
  wire [2:0]   readRequests_6_bits_instructionIndex_0 = readRequests_6_bits_instructionIndex;
  wire         readRequests_7_valid_0 = readRequests_7_valid;
  wire [4:0]   readRequests_7_bits_vs_0 = readRequests_7_bits_vs;
  wire [1:0]   readRequests_7_bits_readSource_0 = readRequests_7_bits_readSource;
  wire [4:0]   readRequests_7_bits_offset_0 = readRequests_7_bits_offset;
  wire [2:0]   readRequests_7_bits_instructionIndex_0 = readRequests_7_bits_instructionIndex;
  wire         readRequests_8_valid_0 = readRequests_8_valid;
  wire [4:0]   readRequests_8_bits_vs_0 = readRequests_8_bits_vs;
  wire [1:0]   readRequests_8_bits_readSource_0 = readRequests_8_bits_readSource;
  wire [4:0]   readRequests_8_bits_offset_0 = readRequests_8_bits_offset;
  wire [2:0]   readRequests_8_bits_instructionIndex_0 = readRequests_8_bits_instructionIndex;
  wire         readRequests_9_valid_0 = readRequests_9_valid;
  wire [4:0]   readRequests_9_bits_vs_0 = readRequests_9_bits_vs;
  wire [1:0]   readRequests_9_bits_readSource_0 = readRequests_9_bits_readSource;
  wire [4:0]   readRequests_9_bits_offset_0 = readRequests_9_bits_offset;
  wire [2:0]   readRequests_9_bits_instructionIndex_0 = readRequests_9_bits_instructionIndex;
  wire         readRequests_10_valid_0 = readRequests_10_valid;
  wire [4:0]   readRequests_10_bits_vs_0 = readRequests_10_bits_vs;
  wire [1:0]   readRequests_10_bits_readSource_0 = readRequests_10_bits_readSource;
  wire [4:0]   readRequests_10_bits_offset_0 = readRequests_10_bits_offset;
  wire [2:0]   readRequests_10_bits_instructionIndex_0 = readRequests_10_bits_instructionIndex;
  wire         readRequests_11_valid_0 = readRequests_11_valid;
  wire [4:0]   readRequests_11_bits_vs_0 = readRequests_11_bits_vs;
  wire [1:0]   readRequests_11_bits_readSource_0 = readRequests_11_bits_readSource;
  wire [4:0]   readRequests_11_bits_offset_0 = readRequests_11_bits_offset;
  wire [2:0]   readRequests_11_bits_instructionIndex_0 = readRequests_11_bits_instructionIndex;
  wire         readRequests_12_valid_0 = readRequests_12_valid;
  wire [4:0]   readRequests_12_bits_vs_0 = readRequests_12_bits_vs;
  wire [1:0]   readRequests_12_bits_readSource_0 = readRequests_12_bits_readSource;
  wire [4:0]   readRequests_12_bits_offset_0 = readRequests_12_bits_offset;
  wire [2:0]   readRequests_12_bits_instructionIndex_0 = readRequests_12_bits_instructionIndex;
  wire         readRequests_13_valid_0 = readRequests_13_valid;
  wire [4:0]   readRequests_13_bits_vs_0 = readRequests_13_bits_vs;
  wire [1:0]   readRequests_13_bits_readSource_0 = readRequests_13_bits_readSource;
  wire [4:0]   readRequests_13_bits_offset_0 = readRequests_13_bits_offset;
  wire [2:0]   readRequests_13_bits_instructionIndex_0 = readRequests_13_bits_instructionIndex;
  wire         write_valid_0 = write_valid;
  wire [4:0]   write_bits_vd_0 = write_bits_vd;
  wire [4:0]   write_bits_offset_0 = write_bits_offset;
  wire [3:0]   write_bits_mask_0 = write_bits_mask;
  wire [31:0]  write_bits_data_0 = write_bits_data;
  wire         write_bits_last_0 = write_bits_last;
  wire [2:0]   write_bits_instructionIndex_0 = write_bits_instructionIndex;
  wire         initRecord_bits_vd_valid = instructionWriteReport_bits_vd_valid;
  wire [4:0]   initRecord_bits_vd_bits = instructionWriteReport_bits_vd_bits;
  wire         initRecord_bits_vs1_valid = instructionWriteReport_bits_vs1_valid;
  wire [4:0]   initRecord_bits_vs1_bits = instructionWriteReport_bits_vs1_bits;
  wire [4:0]   initRecord_bits_vs2 = instructionWriteReport_bits_vs2;
  wire [2:0]   initRecord_bits_instIndex = instructionWriteReport_bits_instIndex;
  wire         initRecord_bits_ls = instructionWriteReport_bits_ls;
  wire         initRecord_bits_st = instructionWriteReport_bits_st;
  wire         initRecord_bits_gather = instructionWriteReport_bits_gather;
  wire         initRecord_bits_gather16 = instructionWriteReport_bits_gather16;
  wire         initRecord_bits_crossWrite = instructionWriteReport_bits_crossWrite;
  wire         initRecord_bits_crossRead = instructionWriteReport_bits_crossRead;
  wire         initRecord_bits_indexType = instructionWriteReport_bits_indexType;
  wire         initRecord_bits_ma = instructionWriteReport_bits_ma;
  wire         initRecord_bits_onlyRead = instructionWriteReport_bits_onlyRead;
  wire         initRecord_bits_slow = instructionWriteReport_bits_slow;
  wire [255:0] initRecord_bits_elementMask = instructionWriteReport_bits_elementMask;
  wire         initRecord_bits_state_stFinish = instructionWriteReport_bits_state_stFinish;
  wire         initRecord_bits_state_wWriteQueueClear = instructionWriteReport_bits_state_wWriteQueueClear;
  wire         initRecord_bits_state_wLaneLastReport = instructionWriteReport_bits_state_wLaneLastReport;
  wire         initRecord_bits_state_wTopLastReport = instructionWriteReport_bits_state_wTopLastReport;
  wire [31:0]  readResultS_0 = 32'h0;
  wire         bankReadS_0 = 1'h0;
  wire         firstUsed = 1'h0;
  wire         initRecord_bits_state_wLaneClear = 1'h0;
  wire         portConflictCheck = 1'h1;
  wire         portReady = 1'h1;
  wire         portConflictCheck_1 = 1'h1;
  wire         portConflictCheck_2 = 1'h1;
  wire         portConflictCheck_3 = 1'h1;
  wire         portConflictCheck_4 = 1'h1;
  wire         portConflictCheck_5 = 1'h1;
  wire         portConflictCheck_6 = 1'h1;
  wire         portConflictCheck_7 = 1'h1;
  wire         portConflictCheck_8 = 1'h1;
  wire         portConflictCheck_9 = 1'h1;
  wire         portConflictCheck_10 = 1'h1;
  wire         portConflictCheck_11 = 1'h1;
  wire         portConflictCheck_12 = 1'h1;
  wire         portConflictCheck_13 = 1'h1;
  wire         initRecord_valid = 1'h1;
  wire         bankCorrect = readRequests_0_valid_0;
  wire         bankCorrect_1 = readRequests_1_valid_0;
  wire         bankCorrect_2 = readRequests_2_valid_0;
  wire         bankCorrect_3 = readRequests_3_valid_0;
  wire         bankCorrect_4 = readRequests_4_valid_0;
  wire         bankCorrect_5 = readRequests_5_valid_0;
  wire         bankCorrect_6 = readRequests_6_valid_0;
  wire         bankCorrect_7 = readRequests_7_valid_0;
  wire         bankCorrect_8 = readRequests_8_valid_0;
  wire         bankCorrect_9 = readRequests_9_valid_0;
  wire         bankCorrect_10 = readRequests_10_valid_0;
  wire         bankCorrect_11 = readRequests_11_valid_0;
  wire         bankCorrect_12 = readRequests_12_valid_0;
  reg          sramReady;
  wire         readRequests_0_ready_0 = sramReady;
  reg  [9:0]   sramResetCount;
  wire         resetValid = ~sramReady;
  wire         _pipeFire_T = readRequests_0_ready_0 & readRequests_0_valid_0;
  wire         readRequests_1_ready_0;
  wire         _pipeFire_T_1 = readRequests_1_ready_0 & readRequests_1_valid_0;
  wire         readRequests_2_ready_0;
  wire         _pipeFire_T_2 = readRequests_2_ready_0 & readRequests_2_valid_0;
  wire         readRequests_3_ready_0;
  wire         _pipeFire_T_3 = readRequests_3_ready_0 & readRequests_3_valid_0;
  wire         readRequests_4_ready_0;
  wire         _pipeFire_T_4 = readRequests_4_ready_0 & readRequests_4_valid_0;
  wire         readRequests_5_ready_0;
  wire         _pipeFire_T_5 = readRequests_5_ready_0 & readRequests_5_valid_0;
  wire         readRequests_6_ready_0;
  wire         _pipeFire_T_6 = readRequests_6_ready_0 & readRequests_6_valid_0;
  wire         readRequests_7_ready_0;
  wire         _pipeFire_T_7 = readRequests_7_ready_0 & readRequests_7_valid_0;
  wire         readRequests_8_ready_0;
  wire         _pipeFire_T_8 = readRequests_8_ready_0 & readRequests_8_valid_0;
  wire         readRequests_9_ready_0;
  wire         _pipeFire_T_9 = readRequests_9_ready_0 & readRequests_9_valid_0;
  wire         readRequests_10_ready_0;
  wire         _pipeFire_T_10 = readRequests_10_ready_0 & readRequests_10_valid_0;
  wire         readRequests_11_ready_0;
  wire         _pipeFire_T_11 = readRequests_11_ready_0 & readRequests_11_valid_0;
  wire         readRequests_12_ready_0;
  wire         _pipeFire_T_12 = readRequests_12_ready_0 & readRequests_12_valid_0;
  wire         readRequests_13_ready_0;
  wire         _loadUpdateValidVec_T_27 = readRequests_13_ready_0 & readRequests_13_valid_0;
  wire         write_ready_0;
  wire         _writePipe_valid_T = write_ready_0 & write_valid_0;
  wire [3:0]   portFireCount =
    {1'h0, {1'h0, {1'h0, _pipeFire_T} + {1'h0, _pipeFire_T_1} + {1'h0, _pipeFire_T_2}} + {1'h0, {1'h0, _pipeFire_T_3} + {1'h0, _pipeFire_T_4}} + {1'h0, {1'h0, _pipeFire_T_5} + {1'h0, _pipeFire_T_6}}}
    + {1'h0, {1'h0, {1'h0, _pipeFire_T_7} + {1'h0, _pipeFire_T_8}} + {1'h0, {1'h0, _pipeFire_T_9} + {1'h0, _pipeFire_T_10}}}
    + {1'h0, {1'h0, {1'h0, _pipeFire_T_11} + {1'h0, _pipeFire_T_12}} + {1'h0, {1'h0, _loadUpdateValidVec_T_27} + {1'h0, _writePipe_valid_T}}};
  wire [9:0]   writeIndex = {write_bits_vd_0, write_bits_offset_0};
  reg          chainingRecord_0_valid;
  reg          chainingRecord_0_bits_vd_valid;
  reg  [4:0]   chainingRecord_0_bits_vd_bits;
  reg          chainingRecord_0_bits_vs1_valid;
  reg  [4:0]   chainingRecord_0_bits_vs1_bits;
  reg  [4:0]   chainingRecord_0_bits_vs2;
  reg  [2:0]   chainingRecord_0_bits_instIndex;
  reg          chainingRecord_0_bits_ls;
  reg          chainingRecord_0_bits_st;
  reg          chainingRecord_0_bits_gather;
  reg          chainingRecord_0_bits_gather16;
  reg          chainingRecord_0_bits_crossWrite;
  reg          chainingRecord_0_bits_crossRead;
  reg          chainingRecord_0_bits_indexType;
  reg          chainingRecord_0_bits_ma;
  reg          chainingRecord_0_bits_onlyRead;
  reg          chainingRecord_0_bits_slow;
  reg  [255:0] chainingRecord_0_bits_elementMask;
  reg          chainingRecord_0_bits_state_stFinish;
  reg          chainingRecord_0_bits_state_wWriteQueueClear;
  reg          chainingRecord_0_bits_state_wLaneLastReport;
  reg          chainingRecord_0_bits_state_wTopLastReport;
  reg          chainingRecord_0_bits_state_wLaneClear;
  reg          chainingRecord_1_valid;
  reg          chainingRecord_1_bits_vd_valid;
  reg  [4:0]   chainingRecord_1_bits_vd_bits;
  reg          chainingRecord_1_bits_vs1_valid;
  reg  [4:0]   chainingRecord_1_bits_vs1_bits;
  reg  [4:0]   chainingRecord_1_bits_vs2;
  reg  [2:0]   chainingRecord_1_bits_instIndex;
  reg          chainingRecord_1_bits_ls;
  reg          chainingRecord_1_bits_st;
  reg          chainingRecord_1_bits_gather;
  reg          chainingRecord_1_bits_gather16;
  reg          chainingRecord_1_bits_crossWrite;
  reg          chainingRecord_1_bits_crossRead;
  reg          chainingRecord_1_bits_indexType;
  reg          chainingRecord_1_bits_ma;
  reg          chainingRecord_1_bits_onlyRead;
  reg          chainingRecord_1_bits_slow;
  reg  [255:0] chainingRecord_1_bits_elementMask;
  reg          chainingRecord_1_bits_state_stFinish;
  reg          chainingRecord_1_bits_state_wWriteQueueClear;
  reg          chainingRecord_1_bits_state_wLaneLastReport;
  reg          chainingRecord_1_bits_state_wTopLastReport;
  reg          chainingRecord_1_bits_state_wLaneClear;
  reg          chainingRecord_2_valid;
  reg          chainingRecord_2_bits_vd_valid;
  reg  [4:0]   chainingRecord_2_bits_vd_bits;
  reg          chainingRecord_2_bits_vs1_valid;
  reg  [4:0]   chainingRecord_2_bits_vs1_bits;
  reg  [4:0]   chainingRecord_2_bits_vs2;
  reg  [2:0]   chainingRecord_2_bits_instIndex;
  reg          chainingRecord_2_bits_ls;
  reg          chainingRecord_2_bits_st;
  reg          chainingRecord_2_bits_gather;
  reg          chainingRecord_2_bits_gather16;
  reg          chainingRecord_2_bits_crossWrite;
  reg          chainingRecord_2_bits_crossRead;
  reg          chainingRecord_2_bits_indexType;
  reg          chainingRecord_2_bits_ma;
  reg          chainingRecord_2_bits_onlyRead;
  reg          chainingRecord_2_bits_slow;
  reg  [255:0] chainingRecord_2_bits_elementMask;
  reg          chainingRecord_2_bits_state_stFinish;
  reg          chainingRecord_2_bits_state_wWriteQueueClear;
  reg          chainingRecord_2_bits_state_wLaneLastReport;
  reg          chainingRecord_2_bits_state_wTopLastReport;
  reg          chainingRecord_2_bits_state_wLaneClear;
  reg          chainingRecord_3_valid;
  reg          chainingRecord_3_bits_vd_valid;
  reg  [4:0]   chainingRecord_3_bits_vd_bits;
  reg          chainingRecord_3_bits_vs1_valid;
  reg  [4:0]   chainingRecord_3_bits_vs1_bits;
  reg  [4:0]   chainingRecord_3_bits_vs2;
  reg  [2:0]   chainingRecord_3_bits_instIndex;
  reg          chainingRecord_3_bits_ls;
  reg          chainingRecord_3_bits_st;
  reg          chainingRecord_3_bits_gather;
  reg          chainingRecord_3_bits_gather16;
  reg          chainingRecord_3_bits_crossWrite;
  reg          chainingRecord_3_bits_crossRead;
  reg          chainingRecord_3_bits_indexType;
  reg          chainingRecord_3_bits_ma;
  reg          chainingRecord_3_bits_onlyRead;
  reg          chainingRecord_3_bits_slow;
  reg  [255:0] chainingRecord_3_bits_elementMask;
  reg          chainingRecord_3_bits_state_stFinish;
  reg          chainingRecord_3_bits_state_wWriteQueueClear;
  reg          chainingRecord_3_bits_state_wLaneLastReport;
  reg          chainingRecord_3_bits_state_wTopLastReport;
  reg          chainingRecord_3_bits_state_wLaneClear;
  reg          chainingRecord_4_valid;
  reg          chainingRecord_4_bits_vd_valid;
  reg  [4:0]   chainingRecord_4_bits_vd_bits;
  reg          chainingRecord_4_bits_vs1_valid;
  reg  [4:0]   chainingRecord_4_bits_vs1_bits;
  reg  [4:0]   chainingRecord_4_bits_vs2;
  reg  [2:0]   chainingRecord_4_bits_instIndex;
  reg          chainingRecord_4_bits_ls;
  reg          chainingRecord_4_bits_st;
  reg          chainingRecord_4_bits_gather;
  reg          chainingRecord_4_bits_gather16;
  reg          chainingRecord_4_bits_crossWrite;
  reg          chainingRecord_4_bits_crossRead;
  reg          chainingRecord_4_bits_indexType;
  reg          chainingRecord_4_bits_ma;
  reg          chainingRecord_4_bits_onlyRead;
  reg          chainingRecord_4_bits_slow;
  reg  [255:0] chainingRecord_4_bits_elementMask;
  reg          chainingRecord_4_bits_state_stFinish;
  reg          chainingRecord_4_bits_state_wWriteQueueClear;
  reg          chainingRecord_4_bits_state_wLaneLastReport;
  reg          chainingRecord_4_bits_state_wTopLastReport;
  reg          chainingRecord_4_bits_state_wLaneClear;
  reg          chainingRecordCopy_0_valid;
  reg          chainingRecordCopy_0_bits_vd_valid;
  reg  [4:0]   chainingRecordCopy_0_bits_vd_bits;
  reg          chainingRecordCopy_0_bits_vs1_valid;
  reg  [4:0]   chainingRecordCopy_0_bits_vs1_bits;
  reg  [4:0]   chainingRecordCopy_0_bits_vs2;
  reg  [2:0]   chainingRecordCopy_0_bits_instIndex;
  reg          chainingRecordCopy_0_bits_ls;
  reg          chainingRecordCopy_0_bits_st;
  reg          chainingRecordCopy_0_bits_gather;
  reg          chainingRecordCopy_0_bits_gather16;
  reg          chainingRecordCopy_0_bits_crossWrite;
  reg          chainingRecordCopy_0_bits_crossRead;
  reg          chainingRecordCopy_0_bits_indexType;
  reg          chainingRecordCopy_0_bits_ma;
  reg          chainingRecordCopy_0_bits_onlyRead;
  reg          chainingRecordCopy_0_bits_slow;
  reg  [255:0] chainingRecordCopy_0_bits_elementMask;
  reg          chainingRecordCopy_0_bits_state_stFinish;
  reg          chainingRecordCopy_0_bits_state_wWriteQueueClear;
  reg          chainingRecordCopy_0_bits_state_wLaneLastReport;
  reg          chainingRecordCopy_0_bits_state_wTopLastReport;
  reg          chainingRecordCopy_0_bits_state_wLaneClear;
  reg          chainingRecordCopy_1_valid;
  reg          chainingRecordCopy_1_bits_vd_valid;
  reg  [4:0]   chainingRecordCopy_1_bits_vd_bits;
  reg          chainingRecordCopy_1_bits_vs1_valid;
  reg  [4:0]   chainingRecordCopy_1_bits_vs1_bits;
  reg  [4:0]   chainingRecordCopy_1_bits_vs2;
  reg  [2:0]   chainingRecordCopy_1_bits_instIndex;
  reg          chainingRecordCopy_1_bits_ls;
  reg          chainingRecordCopy_1_bits_st;
  reg          chainingRecordCopy_1_bits_gather;
  reg          chainingRecordCopy_1_bits_gather16;
  reg          chainingRecordCopy_1_bits_crossWrite;
  reg          chainingRecordCopy_1_bits_crossRead;
  reg          chainingRecordCopy_1_bits_indexType;
  reg          chainingRecordCopy_1_bits_ma;
  reg          chainingRecordCopy_1_bits_onlyRead;
  reg          chainingRecordCopy_1_bits_slow;
  reg  [255:0] chainingRecordCopy_1_bits_elementMask;
  reg          chainingRecordCopy_1_bits_state_stFinish;
  reg          chainingRecordCopy_1_bits_state_wWriteQueueClear;
  reg          chainingRecordCopy_1_bits_state_wLaneLastReport;
  reg          chainingRecordCopy_1_bits_state_wTopLastReport;
  reg          chainingRecordCopy_1_bits_state_wLaneClear;
  reg          chainingRecordCopy_2_valid;
  reg          chainingRecordCopy_2_bits_vd_valid;
  reg  [4:0]   chainingRecordCopy_2_bits_vd_bits;
  reg          chainingRecordCopy_2_bits_vs1_valid;
  reg  [4:0]   chainingRecordCopy_2_bits_vs1_bits;
  reg  [4:0]   chainingRecordCopy_2_bits_vs2;
  reg  [2:0]   chainingRecordCopy_2_bits_instIndex;
  reg          chainingRecordCopy_2_bits_ls;
  reg          chainingRecordCopy_2_bits_st;
  reg          chainingRecordCopy_2_bits_gather;
  reg          chainingRecordCopy_2_bits_gather16;
  reg          chainingRecordCopy_2_bits_crossWrite;
  reg          chainingRecordCopy_2_bits_crossRead;
  reg          chainingRecordCopy_2_bits_indexType;
  reg          chainingRecordCopy_2_bits_ma;
  reg          chainingRecordCopy_2_bits_onlyRead;
  reg          chainingRecordCopy_2_bits_slow;
  reg  [255:0] chainingRecordCopy_2_bits_elementMask;
  reg          chainingRecordCopy_2_bits_state_stFinish;
  reg          chainingRecordCopy_2_bits_state_wWriteQueueClear;
  reg          chainingRecordCopy_2_bits_state_wLaneLastReport;
  reg          chainingRecordCopy_2_bits_state_wTopLastReport;
  reg          chainingRecordCopy_2_bits_state_wLaneClear;
  reg          chainingRecordCopy_3_valid;
  reg          chainingRecordCopy_3_bits_vd_valid;
  reg  [4:0]   chainingRecordCopy_3_bits_vd_bits;
  reg          chainingRecordCopy_3_bits_vs1_valid;
  reg  [4:0]   chainingRecordCopy_3_bits_vs1_bits;
  reg  [4:0]   chainingRecordCopy_3_bits_vs2;
  reg  [2:0]   chainingRecordCopy_3_bits_instIndex;
  reg          chainingRecordCopy_3_bits_ls;
  reg          chainingRecordCopy_3_bits_st;
  reg          chainingRecordCopy_3_bits_gather;
  reg          chainingRecordCopy_3_bits_gather16;
  reg          chainingRecordCopy_3_bits_crossWrite;
  reg          chainingRecordCopy_3_bits_crossRead;
  reg          chainingRecordCopy_3_bits_indexType;
  reg          chainingRecordCopy_3_bits_ma;
  reg          chainingRecordCopy_3_bits_onlyRead;
  reg          chainingRecordCopy_3_bits_slow;
  reg  [255:0] chainingRecordCopy_3_bits_elementMask;
  reg          chainingRecordCopy_3_bits_state_stFinish;
  reg          chainingRecordCopy_3_bits_state_wWriteQueueClear;
  reg          chainingRecordCopy_3_bits_state_wLaneLastReport;
  reg          chainingRecordCopy_3_bits_state_wTopLastReport;
  reg          chainingRecordCopy_3_bits_state_wLaneClear;
  reg          chainingRecordCopy_4_valid;
  reg          chainingRecordCopy_4_bits_vd_valid;
  reg  [4:0]   chainingRecordCopy_4_bits_vd_bits;
  reg          chainingRecordCopy_4_bits_vs1_valid;
  reg  [4:0]   chainingRecordCopy_4_bits_vs1_bits;
  reg  [4:0]   chainingRecordCopy_4_bits_vs2;
  reg  [2:0]   chainingRecordCopy_4_bits_instIndex;
  reg          chainingRecordCopy_4_bits_ls;
  reg          chainingRecordCopy_4_bits_st;
  reg          chainingRecordCopy_4_bits_gather;
  reg          chainingRecordCopy_4_bits_gather16;
  reg          chainingRecordCopy_4_bits_crossWrite;
  reg          chainingRecordCopy_4_bits_crossRead;
  reg          chainingRecordCopy_4_bits_indexType;
  reg          chainingRecordCopy_4_bits_ma;
  reg          chainingRecordCopy_4_bits_onlyRead;
  reg          chainingRecordCopy_4_bits_slow;
  reg  [255:0] chainingRecordCopy_4_bits_elementMask;
  reg          chainingRecordCopy_4_bits_state_stFinish;
  reg          chainingRecordCopy_4_bits_state_wWriteQueueClear;
  reg          chainingRecordCopy_4_bits_state_wLaneLastReport;
  reg          chainingRecordCopy_4_bits_state_wTopLastReport;
  reg          chainingRecordCopy_4_bits_state_wLaneClear;
  wire         recordValidVec_0 = ~(&chainingRecord_0_bits_elementMask) & chainingRecord_0_valid;
  wire         recordValidVec_1 = ~(&chainingRecord_1_bits_elementMask) & chainingRecord_1_valid;
  wire         recordValidVec_2 = ~(&chainingRecord_2_bits_elementMask) & chainingRecord_2_valid;
  wire         recordValidVec_3 = ~(&chainingRecord_3_bits_elementMask) & chainingRecord_3_valid;
  wire         recordValidVec_4 = ~(&chainingRecord_4_bits_elementMask) & chainingRecord_4_valid;
  reg          firstReadPipe_0_valid;
  reg  [9:0]   firstReadPipe_0_bits_address;
  reg          writePipe_valid;
  wire         writeValid = writePipe_valid;
  reg  [4:0]   writePipe_bits_vd;
  reg  [4:0]   writePipe_bits_offset;
  reg  [3:0]   writePipe_bits_mask;
  reg  [31:0]  writePipe_bits_data;
  reg          writePipe_bits_last;
  reg  [2:0]   writePipe_bits_instructionIndex;
  wire         _readRecord_T = chainingRecord_0_bits_instIndex == readCheck_0_instructionIndex;
  wire         _readRecord_T_1 = chainingRecord_1_bits_instIndex == readCheck_0_instructionIndex;
  wire         _readRecord_T_2 = chainingRecord_2_bits_instIndex == readCheck_0_instructionIndex;
  wire         _readRecord_T_3 = chainingRecord_3_bits_instIndex == readCheck_0_instructionIndex;
  wire         _readRecord_T_4 = chainingRecord_4_bits_instIndex == readCheck_0_instructionIndex;
  wire         readRecord_state_wLaneClear =
    _readRecord_T & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_1 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_2 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_3 & chainingRecord_3_bits_state_wLaneClear
    | _readRecord_T_4 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_state_wTopLastReport =
    _readRecord_T & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_1 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_2 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_3
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_4 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_state_wLaneLastReport =
    _readRecord_T & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_1 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_2 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_3
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_4 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_state_wWriteQueueClear =
    _readRecord_T & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_1 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_2 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_3
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_4 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_state_stFinish =
    _readRecord_T & chainingRecord_0_bits_state_stFinish | _readRecord_T_1 & chainingRecord_1_bits_state_stFinish | _readRecord_T_2 & chainingRecord_2_bits_state_stFinish | _readRecord_T_3 & chainingRecord_3_bits_state_stFinish
    | _readRecord_T_4 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_elementMask =
    (_readRecord_T ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_1 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_2 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_3 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_4 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_slow =
    _readRecord_T & chainingRecord_0_bits_slow | _readRecord_T_1 & chainingRecord_1_bits_slow | _readRecord_T_2 & chainingRecord_2_bits_slow | _readRecord_T_3 & chainingRecord_3_bits_slow | _readRecord_T_4 & chainingRecord_4_bits_slow;
  wire         readRecord_onlyRead =
    _readRecord_T & chainingRecord_0_bits_onlyRead | _readRecord_T_1 & chainingRecord_1_bits_onlyRead | _readRecord_T_2 & chainingRecord_2_bits_onlyRead | _readRecord_T_3 & chainingRecord_3_bits_onlyRead | _readRecord_T_4
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_ma =
    _readRecord_T & chainingRecord_0_bits_ma | _readRecord_T_1 & chainingRecord_1_bits_ma | _readRecord_T_2 & chainingRecord_2_bits_ma | _readRecord_T_3 & chainingRecord_3_bits_ma | _readRecord_T_4 & chainingRecord_4_bits_ma;
  wire         readRecord_indexType =
    _readRecord_T & chainingRecord_0_bits_indexType | _readRecord_T_1 & chainingRecord_1_bits_indexType | _readRecord_T_2 & chainingRecord_2_bits_indexType | _readRecord_T_3 & chainingRecord_3_bits_indexType | _readRecord_T_4
    & chainingRecord_4_bits_indexType;
  wire         readRecord_crossRead =
    _readRecord_T & chainingRecord_0_bits_crossRead | _readRecord_T_1 & chainingRecord_1_bits_crossRead | _readRecord_T_2 & chainingRecord_2_bits_crossRead | _readRecord_T_3 & chainingRecord_3_bits_crossRead | _readRecord_T_4
    & chainingRecord_4_bits_crossRead;
  wire         readRecord_crossWrite =
    _readRecord_T & chainingRecord_0_bits_crossWrite | _readRecord_T_1 & chainingRecord_1_bits_crossWrite | _readRecord_T_2 & chainingRecord_2_bits_crossWrite | _readRecord_T_3 & chainingRecord_3_bits_crossWrite | _readRecord_T_4
    & chainingRecord_4_bits_crossWrite;
  wire         readRecord_gather16 =
    _readRecord_T & chainingRecord_0_bits_gather16 | _readRecord_T_1 & chainingRecord_1_bits_gather16 | _readRecord_T_2 & chainingRecord_2_bits_gather16 | _readRecord_T_3 & chainingRecord_3_bits_gather16 | _readRecord_T_4
    & chainingRecord_4_bits_gather16;
  wire         readRecord_gather =
    _readRecord_T & chainingRecord_0_bits_gather | _readRecord_T_1 & chainingRecord_1_bits_gather | _readRecord_T_2 & chainingRecord_2_bits_gather | _readRecord_T_3 & chainingRecord_3_bits_gather | _readRecord_T_4
    & chainingRecord_4_bits_gather;
  wire         readRecord_st =
    _readRecord_T & chainingRecord_0_bits_st | _readRecord_T_1 & chainingRecord_1_bits_st | _readRecord_T_2 & chainingRecord_2_bits_st | _readRecord_T_3 & chainingRecord_3_bits_st | _readRecord_T_4 & chainingRecord_4_bits_st;
  wire         readRecord_ls =
    _readRecord_T & chainingRecord_0_bits_ls | _readRecord_T_1 & chainingRecord_1_bits_ls | _readRecord_T_2 & chainingRecord_2_bits_ls | _readRecord_T_3 & chainingRecord_3_bits_ls | _readRecord_T_4 & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_instIndex =
    (_readRecord_T ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_1 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_2 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_3 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_4 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_vs2 =
    (_readRecord_T ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_1 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_2 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_3 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_4 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_vs1_bits =
    (_readRecord_T ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_1 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_2 ? chainingRecord_2_bits_vs1_bits : 5'h0) | (_readRecord_T_3 ? chainingRecord_3_bits_vs1_bits : 5'h0)
    | (_readRecord_T_4 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_vs1_valid =
    _readRecord_T & chainingRecord_0_bits_vs1_valid | _readRecord_T_1 & chainingRecord_1_bits_vs1_valid | _readRecord_T_2 & chainingRecord_2_bits_vs1_valid | _readRecord_T_3 & chainingRecord_3_bits_vs1_valid | _readRecord_T_4
    & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_vd_bits =
    (_readRecord_T ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_1 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_2 ? chainingRecord_2_bits_vd_bits : 5'h0) | (_readRecord_T_3 ? chainingRecord_3_bits_vd_bits : 5'h0)
    | (_readRecord_T_4 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_vd_valid =
    _readRecord_T & chainingRecord_0_bits_vd_valid | _readRecord_T_1 & chainingRecord_1_bits_vd_valid | _readRecord_T_2 & chainingRecord_2_bits_vd_valid | _readRecord_T_3 & chainingRecord_3_bits_vd_valid | _readRecord_T_4
    & chainingRecord_4_bits_vd_valid;
  wire         _readRecord_T_220 = chainingRecord_0_bits_instIndex == readCheck_1_instructionIndex;
  wire         _readRecord_T_221 = chainingRecord_1_bits_instIndex == readCheck_1_instructionIndex;
  wire         _readRecord_T_222 = chainingRecord_2_bits_instIndex == readCheck_1_instructionIndex;
  wire         _readRecord_T_223 = chainingRecord_3_bits_instIndex == readCheck_1_instructionIndex;
  wire         _readRecord_T_224 = chainingRecord_4_bits_instIndex == readCheck_1_instructionIndex;
  wire         readRecord_1_state_wLaneClear =
    _readRecord_T_220 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_221 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_222 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_223
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_224 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_1_state_wTopLastReport =
    _readRecord_T_220 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_221 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_222 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_223
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_224 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_1_state_wLaneLastReport =
    _readRecord_T_220 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_221 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_222 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_223
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_224 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_1_state_wWriteQueueClear =
    _readRecord_T_220 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_221 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_222 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_223
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_224 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_1_state_stFinish =
    _readRecord_T_220 & chainingRecord_0_bits_state_stFinish | _readRecord_T_221 & chainingRecord_1_bits_state_stFinish | _readRecord_T_222 & chainingRecord_2_bits_state_stFinish | _readRecord_T_223 & chainingRecord_3_bits_state_stFinish
    | _readRecord_T_224 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_1_elementMask =
    (_readRecord_T_220 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_221 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_222 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_223 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_224 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_1_slow =
    _readRecord_T_220 & chainingRecord_0_bits_slow | _readRecord_T_221 & chainingRecord_1_bits_slow | _readRecord_T_222 & chainingRecord_2_bits_slow | _readRecord_T_223 & chainingRecord_3_bits_slow | _readRecord_T_224
    & chainingRecord_4_bits_slow;
  wire         readRecord_1_onlyRead =
    _readRecord_T_220 & chainingRecord_0_bits_onlyRead | _readRecord_T_221 & chainingRecord_1_bits_onlyRead | _readRecord_T_222 & chainingRecord_2_bits_onlyRead | _readRecord_T_223 & chainingRecord_3_bits_onlyRead | _readRecord_T_224
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_1_ma =
    _readRecord_T_220 & chainingRecord_0_bits_ma | _readRecord_T_221 & chainingRecord_1_bits_ma | _readRecord_T_222 & chainingRecord_2_bits_ma | _readRecord_T_223 & chainingRecord_3_bits_ma | _readRecord_T_224 & chainingRecord_4_bits_ma;
  wire         readRecord_1_indexType =
    _readRecord_T_220 & chainingRecord_0_bits_indexType | _readRecord_T_221 & chainingRecord_1_bits_indexType | _readRecord_T_222 & chainingRecord_2_bits_indexType | _readRecord_T_223 & chainingRecord_3_bits_indexType | _readRecord_T_224
    & chainingRecord_4_bits_indexType;
  wire         readRecord_1_crossRead =
    _readRecord_T_220 & chainingRecord_0_bits_crossRead | _readRecord_T_221 & chainingRecord_1_bits_crossRead | _readRecord_T_222 & chainingRecord_2_bits_crossRead | _readRecord_T_223 & chainingRecord_3_bits_crossRead | _readRecord_T_224
    & chainingRecord_4_bits_crossRead;
  wire         readRecord_1_crossWrite =
    _readRecord_T_220 & chainingRecord_0_bits_crossWrite | _readRecord_T_221 & chainingRecord_1_bits_crossWrite | _readRecord_T_222 & chainingRecord_2_bits_crossWrite | _readRecord_T_223 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_224 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_1_gather16 =
    _readRecord_T_220 & chainingRecord_0_bits_gather16 | _readRecord_T_221 & chainingRecord_1_bits_gather16 | _readRecord_T_222 & chainingRecord_2_bits_gather16 | _readRecord_T_223 & chainingRecord_3_bits_gather16 | _readRecord_T_224
    & chainingRecord_4_bits_gather16;
  wire         readRecord_1_gather =
    _readRecord_T_220 & chainingRecord_0_bits_gather | _readRecord_T_221 & chainingRecord_1_bits_gather | _readRecord_T_222 & chainingRecord_2_bits_gather | _readRecord_T_223 & chainingRecord_3_bits_gather | _readRecord_T_224
    & chainingRecord_4_bits_gather;
  wire         readRecord_1_st =
    _readRecord_T_220 & chainingRecord_0_bits_st | _readRecord_T_221 & chainingRecord_1_bits_st | _readRecord_T_222 & chainingRecord_2_bits_st | _readRecord_T_223 & chainingRecord_3_bits_st | _readRecord_T_224 & chainingRecord_4_bits_st;
  wire         readRecord_1_ls =
    _readRecord_T_220 & chainingRecord_0_bits_ls | _readRecord_T_221 & chainingRecord_1_bits_ls | _readRecord_T_222 & chainingRecord_2_bits_ls | _readRecord_T_223 & chainingRecord_3_bits_ls | _readRecord_T_224 & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_1_instIndex =
    (_readRecord_T_220 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_221 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_222 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_223 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_224 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_1_vs2 =
    (_readRecord_T_220 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_221 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_222 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_223 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_224 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_1_vs1_bits =
    (_readRecord_T_220 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_221 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_222 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_223 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_224 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_1_vs1_valid =
    _readRecord_T_220 & chainingRecord_0_bits_vs1_valid | _readRecord_T_221 & chainingRecord_1_bits_vs1_valid | _readRecord_T_222 & chainingRecord_2_bits_vs1_valid | _readRecord_T_223 & chainingRecord_3_bits_vs1_valid | _readRecord_T_224
    & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_1_vd_bits =
    (_readRecord_T_220 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_221 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_222 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_223 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_224 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_1_vd_valid =
    _readRecord_T_220 & chainingRecord_0_bits_vd_valid | _readRecord_T_221 & chainingRecord_1_bits_vd_valid | _readRecord_T_222 & chainingRecord_2_bits_vd_valid | _readRecord_T_223 & chainingRecord_3_bits_vd_valid | _readRecord_T_224
    & chainingRecord_4_bits_vd_valid;
  wire         _readRecord_T_440 = chainingRecord_0_bits_instIndex == readCheck_2_instructionIndex;
  wire         _readRecord_T_441 = chainingRecord_1_bits_instIndex == readCheck_2_instructionIndex;
  wire         _readRecord_T_442 = chainingRecord_2_bits_instIndex == readCheck_2_instructionIndex;
  wire         _readRecord_T_443 = chainingRecord_3_bits_instIndex == readCheck_2_instructionIndex;
  wire         _readRecord_T_444 = chainingRecord_4_bits_instIndex == readCheck_2_instructionIndex;
  wire         readRecord_2_state_wLaneClear =
    _readRecord_T_440 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_441 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_442 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_443
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_444 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_2_state_wTopLastReport =
    _readRecord_T_440 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_441 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_442 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_443
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_444 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_2_state_wLaneLastReport =
    _readRecord_T_440 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_441 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_442 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_443
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_444 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_2_state_wWriteQueueClear =
    _readRecord_T_440 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_441 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_442 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_443
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_444 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_2_state_stFinish =
    _readRecord_T_440 & chainingRecord_0_bits_state_stFinish | _readRecord_T_441 & chainingRecord_1_bits_state_stFinish | _readRecord_T_442 & chainingRecord_2_bits_state_stFinish | _readRecord_T_443 & chainingRecord_3_bits_state_stFinish
    | _readRecord_T_444 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_2_elementMask =
    (_readRecord_T_440 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_441 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_442 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_443 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_444 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_2_slow =
    _readRecord_T_440 & chainingRecord_0_bits_slow | _readRecord_T_441 & chainingRecord_1_bits_slow | _readRecord_T_442 & chainingRecord_2_bits_slow | _readRecord_T_443 & chainingRecord_3_bits_slow | _readRecord_T_444
    & chainingRecord_4_bits_slow;
  wire         readRecord_2_onlyRead =
    _readRecord_T_440 & chainingRecord_0_bits_onlyRead | _readRecord_T_441 & chainingRecord_1_bits_onlyRead | _readRecord_T_442 & chainingRecord_2_bits_onlyRead | _readRecord_T_443 & chainingRecord_3_bits_onlyRead | _readRecord_T_444
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_2_ma =
    _readRecord_T_440 & chainingRecord_0_bits_ma | _readRecord_T_441 & chainingRecord_1_bits_ma | _readRecord_T_442 & chainingRecord_2_bits_ma | _readRecord_T_443 & chainingRecord_3_bits_ma | _readRecord_T_444 & chainingRecord_4_bits_ma;
  wire         readRecord_2_indexType =
    _readRecord_T_440 & chainingRecord_0_bits_indexType | _readRecord_T_441 & chainingRecord_1_bits_indexType | _readRecord_T_442 & chainingRecord_2_bits_indexType | _readRecord_T_443 & chainingRecord_3_bits_indexType | _readRecord_T_444
    & chainingRecord_4_bits_indexType;
  wire         readRecord_2_crossRead =
    _readRecord_T_440 & chainingRecord_0_bits_crossRead | _readRecord_T_441 & chainingRecord_1_bits_crossRead | _readRecord_T_442 & chainingRecord_2_bits_crossRead | _readRecord_T_443 & chainingRecord_3_bits_crossRead | _readRecord_T_444
    & chainingRecord_4_bits_crossRead;
  wire         readRecord_2_crossWrite =
    _readRecord_T_440 & chainingRecord_0_bits_crossWrite | _readRecord_T_441 & chainingRecord_1_bits_crossWrite | _readRecord_T_442 & chainingRecord_2_bits_crossWrite | _readRecord_T_443 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_444 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_2_gather16 =
    _readRecord_T_440 & chainingRecord_0_bits_gather16 | _readRecord_T_441 & chainingRecord_1_bits_gather16 | _readRecord_T_442 & chainingRecord_2_bits_gather16 | _readRecord_T_443 & chainingRecord_3_bits_gather16 | _readRecord_T_444
    & chainingRecord_4_bits_gather16;
  wire         readRecord_2_gather =
    _readRecord_T_440 & chainingRecord_0_bits_gather | _readRecord_T_441 & chainingRecord_1_bits_gather | _readRecord_T_442 & chainingRecord_2_bits_gather | _readRecord_T_443 & chainingRecord_3_bits_gather | _readRecord_T_444
    & chainingRecord_4_bits_gather;
  wire         readRecord_2_st =
    _readRecord_T_440 & chainingRecord_0_bits_st | _readRecord_T_441 & chainingRecord_1_bits_st | _readRecord_T_442 & chainingRecord_2_bits_st | _readRecord_T_443 & chainingRecord_3_bits_st | _readRecord_T_444 & chainingRecord_4_bits_st;
  wire         readRecord_2_ls =
    _readRecord_T_440 & chainingRecord_0_bits_ls | _readRecord_T_441 & chainingRecord_1_bits_ls | _readRecord_T_442 & chainingRecord_2_bits_ls | _readRecord_T_443 & chainingRecord_3_bits_ls | _readRecord_T_444 & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_2_instIndex =
    (_readRecord_T_440 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_441 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_442 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_443 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_444 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_2_vs2 =
    (_readRecord_T_440 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_441 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_442 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_443 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_444 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_2_vs1_bits =
    (_readRecord_T_440 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_441 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_442 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_443 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_444 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_2_vs1_valid =
    _readRecord_T_440 & chainingRecord_0_bits_vs1_valid | _readRecord_T_441 & chainingRecord_1_bits_vs1_valid | _readRecord_T_442 & chainingRecord_2_bits_vs1_valid | _readRecord_T_443 & chainingRecord_3_bits_vs1_valid | _readRecord_T_444
    & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_2_vd_bits =
    (_readRecord_T_440 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_441 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_442 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_443 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_444 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_2_vd_valid =
    _readRecord_T_440 & chainingRecord_0_bits_vd_valid | _readRecord_T_441 & chainingRecord_1_bits_vd_valid | _readRecord_T_442 & chainingRecord_2_bits_vd_valid | _readRecord_T_443 & chainingRecord_3_bits_vd_valid | _readRecord_T_444
    & chainingRecord_4_bits_vd_valid;
  wire         _readRecord_T_660 = chainingRecord_0_bits_instIndex == readCheck_3_instructionIndex;
  wire         _readRecord_T_661 = chainingRecord_1_bits_instIndex == readCheck_3_instructionIndex;
  wire         _readRecord_T_662 = chainingRecord_2_bits_instIndex == readCheck_3_instructionIndex;
  wire         _readRecord_T_663 = chainingRecord_3_bits_instIndex == readCheck_3_instructionIndex;
  wire         _readRecord_T_664 = chainingRecord_4_bits_instIndex == readCheck_3_instructionIndex;
  wire         readRecord_3_state_wLaneClear =
    _readRecord_T_660 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_661 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_662 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_663
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_664 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_3_state_wTopLastReport =
    _readRecord_T_660 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_661 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_662 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_663
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_664 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_3_state_wLaneLastReport =
    _readRecord_T_660 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_661 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_662 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_663
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_664 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_3_state_wWriteQueueClear =
    _readRecord_T_660 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_661 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_662 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_663
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_664 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_3_state_stFinish =
    _readRecord_T_660 & chainingRecord_0_bits_state_stFinish | _readRecord_T_661 & chainingRecord_1_bits_state_stFinish | _readRecord_T_662 & chainingRecord_2_bits_state_stFinish | _readRecord_T_663 & chainingRecord_3_bits_state_stFinish
    | _readRecord_T_664 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_3_elementMask =
    (_readRecord_T_660 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_661 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_662 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_663 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_664 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_3_slow =
    _readRecord_T_660 & chainingRecord_0_bits_slow | _readRecord_T_661 & chainingRecord_1_bits_slow | _readRecord_T_662 & chainingRecord_2_bits_slow | _readRecord_T_663 & chainingRecord_3_bits_slow | _readRecord_T_664
    & chainingRecord_4_bits_slow;
  wire         readRecord_3_onlyRead =
    _readRecord_T_660 & chainingRecord_0_bits_onlyRead | _readRecord_T_661 & chainingRecord_1_bits_onlyRead | _readRecord_T_662 & chainingRecord_2_bits_onlyRead | _readRecord_T_663 & chainingRecord_3_bits_onlyRead | _readRecord_T_664
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_3_ma =
    _readRecord_T_660 & chainingRecord_0_bits_ma | _readRecord_T_661 & chainingRecord_1_bits_ma | _readRecord_T_662 & chainingRecord_2_bits_ma | _readRecord_T_663 & chainingRecord_3_bits_ma | _readRecord_T_664 & chainingRecord_4_bits_ma;
  wire         readRecord_3_indexType =
    _readRecord_T_660 & chainingRecord_0_bits_indexType | _readRecord_T_661 & chainingRecord_1_bits_indexType | _readRecord_T_662 & chainingRecord_2_bits_indexType | _readRecord_T_663 & chainingRecord_3_bits_indexType | _readRecord_T_664
    & chainingRecord_4_bits_indexType;
  wire         readRecord_3_crossRead =
    _readRecord_T_660 & chainingRecord_0_bits_crossRead | _readRecord_T_661 & chainingRecord_1_bits_crossRead | _readRecord_T_662 & chainingRecord_2_bits_crossRead | _readRecord_T_663 & chainingRecord_3_bits_crossRead | _readRecord_T_664
    & chainingRecord_4_bits_crossRead;
  wire         readRecord_3_crossWrite =
    _readRecord_T_660 & chainingRecord_0_bits_crossWrite | _readRecord_T_661 & chainingRecord_1_bits_crossWrite | _readRecord_T_662 & chainingRecord_2_bits_crossWrite | _readRecord_T_663 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_664 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_3_gather16 =
    _readRecord_T_660 & chainingRecord_0_bits_gather16 | _readRecord_T_661 & chainingRecord_1_bits_gather16 | _readRecord_T_662 & chainingRecord_2_bits_gather16 | _readRecord_T_663 & chainingRecord_3_bits_gather16 | _readRecord_T_664
    & chainingRecord_4_bits_gather16;
  wire         readRecord_3_gather =
    _readRecord_T_660 & chainingRecord_0_bits_gather | _readRecord_T_661 & chainingRecord_1_bits_gather | _readRecord_T_662 & chainingRecord_2_bits_gather | _readRecord_T_663 & chainingRecord_3_bits_gather | _readRecord_T_664
    & chainingRecord_4_bits_gather;
  wire         readRecord_3_st =
    _readRecord_T_660 & chainingRecord_0_bits_st | _readRecord_T_661 & chainingRecord_1_bits_st | _readRecord_T_662 & chainingRecord_2_bits_st | _readRecord_T_663 & chainingRecord_3_bits_st | _readRecord_T_664 & chainingRecord_4_bits_st;
  wire         readRecord_3_ls =
    _readRecord_T_660 & chainingRecord_0_bits_ls | _readRecord_T_661 & chainingRecord_1_bits_ls | _readRecord_T_662 & chainingRecord_2_bits_ls | _readRecord_T_663 & chainingRecord_3_bits_ls | _readRecord_T_664 & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_3_instIndex =
    (_readRecord_T_660 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_661 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_662 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_663 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_664 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_3_vs2 =
    (_readRecord_T_660 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_661 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_662 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_663 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_664 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_3_vs1_bits =
    (_readRecord_T_660 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_661 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_662 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_663 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_664 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_3_vs1_valid =
    _readRecord_T_660 & chainingRecord_0_bits_vs1_valid | _readRecord_T_661 & chainingRecord_1_bits_vs1_valid | _readRecord_T_662 & chainingRecord_2_bits_vs1_valid | _readRecord_T_663 & chainingRecord_3_bits_vs1_valid | _readRecord_T_664
    & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_3_vd_bits =
    (_readRecord_T_660 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_661 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_662 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_663 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_664 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_3_vd_valid =
    _readRecord_T_660 & chainingRecord_0_bits_vd_valid | _readRecord_T_661 & chainingRecord_1_bits_vd_valid | _readRecord_T_662 & chainingRecord_2_bits_vd_valid | _readRecord_T_663 & chainingRecord_3_bits_vd_valid | _readRecord_T_664
    & chainingRecord_4_bits_vd_valid;
  wire         _readRecord_T_880 = chainingRecord_0_bits_instIndex == readCheck_4_instructionIndex;
  wire         _readRecord_T_881 = chainingRecord_1_bits_instIndex == readCheck_4_instructionIndex;
  wire         _readRecord_T_882 = chainingRecord_2_bits_instIndex == readCheck_4_instructionIndex;
  wire         _readRecord_T_883 = chainingRecord_3_bits_instIndex == readCheck_4_instructionIndex;
  wire         _readRecord_T_884 = chainingRecord_4_bits_instIndex == readCheck_4_instructionIndex;
  wire         readRecord_4_state_wLaneClear =
    _readRecord_T_880 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_881 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_882 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_883
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_884 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_4_state_wTopLastReport =
    _readRecord_T_880 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_881 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_882 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_883
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_884 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_4_state_wLaneLastReport =
    _readRecord_T_880 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_881 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_882 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_883
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_884 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_4_state_wWriteQueueClear =
    _readRecord_T_880 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_881 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_882 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_883
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_884 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_4_state_stFinish =
    _readRecord_T_880 & chainingRecord_0_bits_state_stFinish | _readRecord_T_881 & chainingRecord_1_bits_state_stFinish | _readRecord_T_882 & chainingRecord_2_bits_state_stFinish | _readRecord_T_883 & chainingRecord_3_bits_state_stFinish
    | _readRecord_T_884 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_4_elementMask =
    (_readRecord_T_880 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_881 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_882 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_883 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_884 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_4_slow =
    _readRecord_T_880 & chainingRecord_0_bits_slow | _readRecord_T_881 & chainingRecord_1_bits_slow | _readRecord_T_882 & chainingRecord_2_bits_slow | _readRecord_T_883 & chainingRecord_3_bits_slow | _readRecord_T_884
    & chainingRecord_4_bits_slow;
  wire         readRecord_4_onlyRead =
    _readRecord_T_880 & chainingRecord_0_bits_onlyRead | _readRecord_T_881 & chainingRecord_1_bits_onlyRead | _readRecord_T_882 & chainingRecord_2_bits_onlyRead | _readRecord_T_883 & chainingRecord_3_bits_onlyRead | _readRecord_T_884
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_4_ma =
    _readRecord_T_880 & chainingRecord_0_bits_ma | _readRecord_T_881 & chainingRecord_1_bits_ma | _readRecord_T_882 & chainingRecord_2_bits_ma | _readRecord_T_883 & chainingRecord_3_bits_ma | _readRecord_T_884 & chainingRecord_4_bits_ma;
  wire         readRecord_4_indexType =
    _readRecord_T_880 & chainingRecord_0_bits_indexType | _readRecord_T_881 & chainingRecord_1_bits_indexType | _readRecord_T_882 & chainingRecord_2_bits_indexType | _readRecord_T_883 & chainingRecord_3_bits_indexType | _readRecord_T_884
    & chainingRecord_4_bits_indexType;
  wire         readRecord_4_crossRead =
    _readRecord_T_880 & chainingRecord_0_bits_crossRead | _readRecord_T_881 & chainingRecord_1_bits_crossRead | _readRecord_T_882 & chainingRecord_2_bits_crossRead | _readRecord_T_883 & chainingRecord_3_bits_crossRead | _readRecord_T_884
    & chainingRecord_4_bits_crossRead;
  wire         readRecord_4_crossWrite =
    _readRecord_T_880 & chainingRecord_0_bits_crossWrite | _readRecord_T_881 & chainingRecord_1_bits_crossWrite | _readRecord_T_882 & chainingRecord_2_bits_crossWrite | _readRecord_T_883 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_884 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_4_gather16 =
    _readRecord_T_880 & chainingRecord_0_bits_gather16 | _readRecord_T_881 & chainingRecord_1_bits_gather16 | _readRecord_T_882 & chainingRecord_2_bits_gather16 | _readRecord_T_883 & chainingRecord_3_bits_gather16 | _readRecord_T_884
    & chainingRecord_4_bits_gather16;
  wire         readRecord_4_gather =
    _readRecord_T_880 & chainingRecord_0_bits_gather | _readRecord_T_881 & chainingRecord_1_bits_gather | _readRecord_T_882 & chainingRecord_2_bits_gather | _readRecord_T_883 & chainingRecord_3_bits_gather | _readRecord_T_884
    & chainingRecord_4_bits_gather;
  wire         readRecord_4_st =
    _readRecord_T_880 & chainingRecord_0_bits_st | _readRecord_T_881 & chainingRecord_1_bits_st | _readRecord_T_882 & chainingRecord_2_bits_st | _readRecord_T_883 & chainingRecord_3_bits_st | _readRecord_T_884 & chainingRecord_4_bits_st;
  wire         readRecord_4_ls =
    _readRecord_T_880 & chainingRecord_0_bits_ls | _readRecord_T_881 & chainingRecord_1_bits_ls | _readRecord_T_882 & chainingRecord_2_bits_ls | _readRecord_T_883 & chainingRecord_3_bits_ls | _readRecord_T_884 & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_4_instIndex =
    (_readRecord_T_880 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_881 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_882 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_883 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_884 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_4_vs2 =
    (_readRecord_T_880 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_881 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_882 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_883 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_884 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_4_vs1_bits =
    (_readRecord_T_880 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_881 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_882 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_883 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_884 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_4_vs1_valid =
    _readRecord_T_880 & chainingRecord_0_bits_vs1_valid | _readRecord_T_881 & chainingRecord_1_bits_vs1_valid | _readRecord_T_882 & chainingRecord_2_bits_vs1_valid | _readRecord_T_883 & chainingRecord_3_bits_vs1_valid | _readRecord_T_884
    & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_4_vd_bits =
    (_readRecord_T_880 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_881 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_882 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_883 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_884 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_4_vd_valid =
    _readRecord_T_880 & chainingRecord_0_bits_vd_valid | _readRecord_T_881 & chainingRecord_1_bits_vd_valid | _readRecord_T_882 & chainingRecord_2_bits_vd_valid | _readRecord_T_883 & chainingRecord_3_bits_vd_valid | _readRecord_T_884
    & chainingRecord_4_bits_vd_valid;
  wire         _readRecord_T_1100 = chainingRecord_0_bits_instIndex == readCheck_5_instructionIndex;
  wire         _readRecord_T_1101 = chainingRecord_1_bits_instIndex == readCheck_5_instructionIndex;
  wire         _readRecord_T_1102 = chainingRecord_2_bits_instIndex == readCheck_5_instructionIndex;
  wire         _readRecord_T_1103 = chainingRecord_3_bits_instIndex == readCheck_5_instructionIndex;
  wire         _readRecord_T_1104 = chainingRecord_4_bits_instIndex == readCheck_5_instructionIndex;
  wire         readRecord_5_state_wLaneClear =
    _readRecord_T_1100 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_1101 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_1102 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_1103
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_1104 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_5_state_wTopLastReport =
    _readRecord_T_1100 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_1101 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_1102 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_1103
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_1104 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_5_state_wLaneLastReport =
    _readRecord_T_1100 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_1101 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_1102 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_1103
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_1104 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_5_state_wWriteQueueClear =
    _readRecord_T_1100 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_1101 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_1102 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_1103
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_1104 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_5_state_stFinish =
    _readRecord_T_1100 & chainingRecord_0_bits_state_stFinish | _readRecord_T_1101 & chainingRecord_1_bits_state_stFinish | _readRecord_T_1102 & chainingRecord_2_bits_state_stFinish | _readRecord_T_1103
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_1104 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_5_elementMask =
    (_readRecord_T_1100 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_1101 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_1102 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_1103 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_1104 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_5_slow =
    _readRecord_T_1100 & chainingRecord_0_bits_slow | _readRecord_T_1101 & chainingRecord_1_bits_slow | _readRecord_T_1102 & chainingRecord_2_bits_slow | _readRecord_T_1103 & chainingRecord_3_bits_slow | _readRecord_T_1104
    & chainingRecord_4_bits_slow;
  wire         readRecord_5_onlyRead =
    _readRecord_T_1100 & chainingRecord_0_bits_onlyRead | _readRecord_T_1101 & chainingRecord_1_bits_onlyRead | _readRecord_T_1102 & chainingRecord_2_bits_onlyRead | _readRecord_T_1103 & chainingRecord_3_bits_onlyRead | _readRecord_T_1104
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_5_ma =
    _readRecord_T_1100 & chainingRecord_0_bits_ma | _readRecord_T_1101 & chainingRecord_1_bits_ma | _readRecord_T_1102 & chainingRecord_2_bits_ma | _readRecord_T_1103 & chainingRecord_3_bits_ma | _readRecord_T_1104
    & chainingRecord_4_bits_ma;
  wire         readRecord_5_indexType =
    _readRecord_T_1100 & chainingRecord_0_bits_indexType | _readRecord_T_1101 & chainingRecord_1_bits_indexType | _readRecord_T_1102 & chainingRecord_2_bits_indexType | _readRecord_T_1103 & chainingRecord_3_bits_indexType
    | _readRecord_T_1104 & chainingRecord_4_bits_indexType;
  wire         readRecord_5_crossRead =
    _readRecord_T_1100 & chainingRecord_0_bits_crossRead | _readRecord_T_1101 & chainingRecord_1_bits_crossRead | _readRecord_T_1102 & chainingRecord_2_bits_crossRead | _readRecord_T_1103 & chainingRecord_3_bits_crossRead
    | _readRecord_T_1104 & chainingRecord_4_bits_crossRead;
  wire         readRecord_5_crossWrite =
    _readRecord_T_1100 & chainingRecord_0_bits_crossWrite | _readRecord_T_1101 & chainingRecord_1_bits_crossWrite | _readRecord_T_1102 & chainingRecord_2_bits_crossWrite | _readRecord_T_1103 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_1104 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_5_gather16 =
    _readRecord_T_1100 & chainingRecord_0_bits_gather16 | _readRecord_T_1101 & chainingRecord_1_bits_gather16 | _readRecord_T_1102 & chainingRecord_2_bits_gather16 | _readRecord_T_1103 & chainingRecord_3_bits_gather16 | _readRecord_T_1104
    & chainingRecord_4_bits_gather16;
  wire         readRecord_5_gather =
    _readRecord_T_1100 & chainingRecord_0_bits_gather | _readRecord_T_1101 & chainingRecord_1_bits_gather | _readRecord_T_1102 & chainingRecord_2_bits_gather | _readRecord_T_1103 & chainingRecord_3_bits_gather | _readRecord_T_1104
    & chainingRecord_4_bits_gather;
  wire         readRecord_5_st =
    _readRecord_T_1100 & chainingRecord_0_bits_st | _readRecord_T_1101 & chainingRecord_1_bits_st | _readRecord_T_1102 & chainingRecord_2_bits_st | _readRecord_T_1103 & chainingRecord_3_bits_st | _readRecord_T_1104
    & chainingRecord_4_bits_st;
  wire         readRecord_5_ls =
    _readRecord_T_1100 & chainingRecord_0_bits_ls | _readRecord_T_1101 & chainingRecord_1_bits_ls | _readRecord_T_1102 & chainingRecord_2_bits_ls | _readRecord_T_1103 & chainingRecord_3_bits_ls | _readRecord_T_1104
    & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_5_instIndex =
    (_readRecord_T_1100 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_1101 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_1102 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_1103 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_1104 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_5_vs2 =
    (_readRecord_T_1100 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_1101 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_1102 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_1103 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_1104 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_5_vs1_bits =
    (_readRecord_T_1100 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_1101 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_1102 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_1103 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_1104 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_5_vs1_valid =
    _readRecord_T_1100 & chainingRecord_0_bits_vs1_valid | _readRecord_T_1101 & chainingRecord_1_bits_vs1_valid | _readRecord_T_1102 & chainingRecord_2_bits_vs1_valid | _readRecord_T_1103 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_1104 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_5_vd_bits =
    (_readRecord_T_1100 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_1101 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_1102 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_1103 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_1104 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_5_vd_valid =
    _readRecord_T_1100 & chainingRecord_0_bits_vd_valid | _readRecord_T_1101 & chainingRecord_1_bits_vd_valid | _readRecord_T_1102 & chainingRecord_2_bits_vd_valid | _readRecord_T_1103 & chainingRecord_3_bits_vd_valid | _readRecord_T_1104
    & chainingRecord_4_bits_vd_valid;
  wire         _readRecord_T_1320 = chainingRecord_0_bits_instIndex == readCheck_6_instructionIndex;
  wire         _readRecord_T_1321 = chainingRecord_1_bits_instIndex == readCheck_6_instructionIndex;
  wire         _readRecord_T_1322 = chainingRecord_2_bits_instIndex == readCheck_6_instructionIndex;
  wire         _readRecord_T_1323 = chainingRecord_3_bits_instIndex == readCheck_6_instructionIndex;
  wire         _readRecord_T_1324 = chainingRecord_4_bits_instIndex == readCheck_6_instructionIndex;
  wire         readRecord_6_state_wLaneClear =
    _readRecord_T_1320 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_1321 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_1322 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_1323
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_1324 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_6_state_wTopLastReport =
    _readRecord_T_1320 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_1321 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_1322 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_1323
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_1324 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_6_state_wLaneLastReport =
    _readRecord_T_1320 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_1321 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_1322 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_1323
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_1324 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_6_state_wWriteQueueClear =
    _readRecord_T_1320 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_1321 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_1322 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_1323
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_1324 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_6_state_stFinish =
    _readRecord_T_1320 & chainingRecord_0_bits_state_stFinish | _readRecord_T_1321 & chainingRecord_1_bits_state_stFinish | _readRecord_T_1322 & chainingRecord_2_bits_state_stFinish | _readRecord_T_1323
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_1324 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_6_elementMask =
    (_readRecord_T_1320 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_1321 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_1322 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_1323 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_1324 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_6_slow =
    _readRecord_T_1320 & chainingRecord_0_bits_slow | _readRecord_T_1321 & chainingRecord_1_bits_slow | _readRecord_T_1322 & chainingRecord_2_bits_slow | _readRecord_T_1323 & chainingRecord_3_bits_slow | _readRecord_T_1324
    & chainingRecord_4_bits_slow;
  wire         readRecord_6_onlyRead =
    _readRecord_T_1320 & chainingRecord_0_bits_onlyRead | _readRecord_T_1321 & chainingRecord_1_bits_onlyRead | _readRecord_T_1322 & chainingRecord_2_bits_onlyRead | _readRecord_T_1323 & chainingRecord_3_bits_onlyRead | _readRecord_T_1324
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_6_ma =
    _readRecord_T_1320 & chainingRecord_0_bits_ma | _readRecord_T_1321 & chainingRecord_1_bits_ma | _readRecord_T_1322 & chainingRecord_2_bits_ma | _readRecord_T_1323 & chainingRecord_3_bits_ma | _readRecord_T_1324
    & chainingRecord_4_bits_ma;
  wire         readRecord_6_indexType =
    _readRecord_T_1320 & chainingRecord_0_bits_indexType | _readRecord_T_1321 & chainingRecord_1_bits_indexType | _readRecord_T_1322 & chainingRecord_2_bits_indexType | _readRecord_T_1323 & chainingRecord_3_bits_indexType
    | _readRecord_T_1324 & chainingRecord_4_bits_indexType;
  wire         readRecord_6_crossRead =
    _readRecord_T_1320 & chainingRecord_0_bits_crossRead | _readRecord_T_1321 & chainingRecord_1_bits_crossRead | _readRecord_T_1322 & chainingRecord_2_bits_crossRead | _readRecord_T_1323 & chainingRecord_3_bits_crossRead
    | _readRecord_T_1324 & chainingRecord_4_bits_crossRead;
  wire         readRecord_6_crossWrite =
    _readRecord_T_1320 & chainingRecord_0_bits_crossWrite | _readRecord_T_1321 & chainingRecord_1_bits_crossWrite | _readRecord_T_1322 & chainingRecord_2_bits_crossWrite | _readRecord_T_1323 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_1324 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_6_gather16 =
    _readRecord_T_1320 & chainingRecord_0_bits_gather16 | _readRecord_T_1321 & chainingRecord_1_bits_gather16 | _readRecord_T_1322 & chainingRecord_2_bits_gather16 | _readRecord_T_1323 & chainingRecord_3_bits_gather16 | _readRecord_T_1324
    & chainingRecord_4_bits_gather16;
  wire         readRecord_6_gather =
    _readRecord_T_1320 & chainingRecord_0_bits_gather | _readRecord_T_1321 & chainingRecord_1_bits_gather | _readRecord_T_1322 & chainingRecord_2_bits_gather | _readRecord_T_1323 & chainingRecord_3_bits_gather | _readRecord_T_1324
    & chainingRecord_4_bits_gather;
  wire         readRecord_6_st =
    _readRecord_T_1320 & chainingRecord_0_bits_st | _readRecord_T_1321 & chainingRecord_1_bits_st | _readRecord_T_1322 & chainingRecord_2_bits_st | _readRecord_T_1323 & chainingRecord_3_bits_st | _readRecord_T_1324
    & chainingRecord_4_bits_st;
  wire         readRecord_6_ls =
    _readRecord_T_1320 & chainingRecord_0_bits_ls | _readRecord_T_1321 & chainingRecord_1_bits_ls | _readRecord_T_1322 & chainingRecord_2_bits_ls | _readRecord_T_1323 & chainingRecord_3_bits_ls | _readRecord_T_1324
    & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_6_instIndex =
    (_readRecord_T_1320 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_1321 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_1322 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_1323 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_1324 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_6_vs2 =
    (_readRecord_T_1320 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_1321 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_1322 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_1323 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_1324 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_6_vs1_bits =
    (_readRecord_T_1320 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_1321 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_1322 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_1323 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_1324 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_6_vs1_valid =
    _readRecord_T_1320 & chainingRecord_0_bits_vs1_valid | _readRecord_T_1321 & chainingRecord_1_bits_vs1_valid | _readRecord_T_1322 & chainingRecord_2_bits_vs1_valid | _readRecord_T_1323 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_1324 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_6_vd_bits =
    (_readRecord_T_1320 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_1321 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_1322 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_1323 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_1324 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_6_vd_valid =
    _readRecord_T_1320 & chainingRecord_0_bits_vd_valid | _readRecord_T_1321 & chainingRecord_1_bits_vd_valid | _readRecord_T_1322 & chainingRecord_2_bits_vd_valid | _readRecord_T_1323 & chainingRecord_3_bits_vd_valid | _readRecord_T_1324
    & chainingRecord_4_bits_vd_valid;
  wire         _readRecord_T_1540 = chainingRecord_0_bits_instIndex == readCheck_7_instructionIndex;
  wire         _readRecord_T_1541 = chainingRecord_1_bits_instIndex == readCheck_7_instructionIndex;
  wire         _readRecord_T_1542 = chainingRecord_2_bits_instIndex == readCheck_7_instructionIndex;
  wire         _readRecord_T_1543 = chainingRecord_3_bits_instIndex == readCheck_7_instructionIndex;
  wire         _readRecord_T_1544 = chainingRecord_4_bits_instIndex == readCheck_7_instructionIndex;
  wire         readRecord_7_state_wLaneClear =
    _readRecord_T_1540 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_1541 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_1542 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_1543
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_1544 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_7_state_wTopLastReport =
    _readRecord_T_1540 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_1541 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_1542 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_1543
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_1544 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_7_state_wLaneLastReport =
    _readRecord_T_1540 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_1541 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_1542 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_1543
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_1544 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_7_state_wWriteQueueClear =
    _readRecord_T_1540 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_1541 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_1542 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_1543
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_1544 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_7_state_stFinish =
    _readRecord_T_1540 & chainingRecord_0_bits_state_stFinish | _readRecord_T_1541 & chainingRecord_1_bits_state_stFinish | _readRecord_T_1542 & chainingRecord_2_bits_state_stFinish | _readRecord_T_1543
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_1544 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_7_elementMask =
    (_readRecord_T_1540 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_1541 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_1542 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_1543 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_1544 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_7_slow =
    _readRecord_T_1540 & chainingRecord_0_bits_slow | _readRecord_T_1541 & chainingRecord_1_bits_slow | _readRecord_T_1542 & chainingRecord_2_bits_slow | _readRecord_T_1543 & chainingRecord_3_bits_slow | _readRecord_T_1544
    & chainingRecord_4_bits_slow;
  wire         readRecord_7_onlyRead =
    _readRecord_T_1540 & chainingRecord_0_bits_onlyRead | _readRecord_T_1541 & chainingRecord_1_bits_onlyRead | _readRecord_T_1542 & chainingRecord_2_bits_onlyRead | _readRecord_T_1543 & chainingRecord_3_bits_onlyRead | _readRecord_T_1544
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_7_ma =
    _readRecord_T_1540 & chainingRecord_0_bits_ma | _readRecord_T_1541 & chainingRecord_1_bits_ma | _readRecord_T_1542 & chainingRecord_2_bits_ma | _readRecord_T_1543 & chainingRecord_3_bits_ma | _readRecord_T_1544
    & chainingRecord_4_bits_ma;
  wire         readRecord_7_indexType =
    _readRecord_T_1540 & chainingRecord_0_bits_indexType | _readRecord_T_1541 & chainingRecord_1_bits_indexType | _readRecord_T_1542 & chainingRecord_2_bits_indexType | _readRecord_T_1543 & chainingRecord_3_bits_indexType
    | _readRecord_T_1544 & chainingRecord_4_bits_indexType;
  wire         readRecord_7_crossRead =
    _readRecord_T_1540 & chainingRecord_0_bits_crossRead | _readRecord_T_1541 & chainingRecord_1_bits_crossRead | _readRecord_T_1542 & chainingRecord_2_bits_crossRead | _readRecord_T_1543 & chainingRecord_3_bits_crossRead
    | _readRecord_T_1544 & chainingRecord_4_bits_crossRead;
  wire         readRecord_7_crossWrite =
    _readRecord_T_1540 & chainingRecord_0_bits_crossWrite | _readRecord_T_1541 & chainingRecord_1_bits_crossWrite | _readRecord_T_1542 & chainingRecord_2_bits_crossWrite | _readRecord_T_1543 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_1544 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_7_gather16 =
    _readRecord_T_1540 & chainingRecord_0_bits_gather16 | _readRecord_T_1541 & chainingRecord_1_bits_gather16 | _readRecord_T_1542 & chainingRecord_2_bits_gather16 | _readRecord_T_1543 & chainingRecord_3_bits_gather16 | _readRecord_T_1544
    & chainingRecord_4_bits_gather16;
  wire         readRecord_7_gather =
    _readRecord_T_1540 & chainingRecord_0_bits_gather | _readRecord_T_1541 & chainingRecord_1_bits_gather | _readRecord_T_1542 & chainingRecord_2_bits_gather | _readRecord_T_1543 & chainingRecord_3_bits_gather | _readRecord_T_1544
    & chainingRecord_4_bits_gather;
  wire         readRecord_7_st =
    _readRecord_T_1540 & chainingRecord_0_bits_st | _readRecord_T_1541 & chainingRecord_1_bits_st | _readRecord_T_1542 & chainingRecord_2_bits_st | _readRecord_T_1543 & chainingRecord_3_bits_st | _readRecord_T_1544
    & chainingRecord_4_bits_st;
  wire         readRecord_7_ls =
    _readRecord_T_1540 & chainingRecord_0_bits_ls | _readRecord_T_1541 & chainingRecord_1_bits_ls | _readRecord_T_1542 & chainingRecord_2_bits_ls | _readRecord_T_1543 & chainingRecord_3_bits_ls | _readRecord_T_1544
    & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_7_instIndex =
    (_readRecord_T_1540 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_1541 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_1542 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_1543 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_1544 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_7_vs2 =
    (_readRecord_T_1540 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_1541 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_1542 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_1543 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_1544 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_7_vs1_bits =
    (_readRecord_T_1540 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_1541 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_1542 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_1543 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_1544 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_7_vs1_valid =
    _readRecord_T_1540 & chainingRecord_0_bits_vs1_valid | _readRecord_T_1541 & chainingRecord_1_bits_vs1_valid | _readRecord_T_1542 & chainingRecord_2_bits_vs1_valid | _readRecord_T_1543 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_1544 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_7_vd_bits =
    (_readRecord_T_1540 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_1541 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_1542 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_1543 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_1544 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_7_vd_valid =
    _readRecord_T_1540 & chainingRecord_0_bits_vd_valid | _readRecord_T_1541 & chainingRecord_1_bits_vd_valid | _readRecord_T_1542 & chainingRecord_2_bits_vd_valid | _readRecord_T_1543 & chainingRecord_3_bits_vd_valid | _readRecord_T_1544
    & chainingRecord_4_bits_vd_valid;
  wire         _readRecord_T_1760 = chainingRecord_0_bits_instIndex == readCheck_8_instructionIndex;
  wire         _readRecord_T_1761 = chainingRecord_1_bits_instIndex == readCheck_8_instructionIndex;
  wire         _readRecord_T_1762 = chainingRecord_2_bits_instIndex == readCheck_8_instructionIndex;
  wire         _readRecord_T_1763 = chainingRecord_3_bits_instIndex == readCheck_8_instructionIndex;
  wire         _readRecord_T_1764 = chainingRecord_4_bits_instIndex == readCheck_8_instructionIndex;
  wire         readRecord_8_state_wLaneClear =
    _readRecord_T_1760 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_1761 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_1762 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_1763
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_1764 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_8_state_wTopLastReport =
    _readRecord_T_1760 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_1761 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_1762 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_1763
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_1764 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_8_state_wLaneLastReport =
    _readRecord_T_1760 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_1761 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_1762 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_1763
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_1764 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_8_state_wWriteQueueClear =
    _readRecord_T_1760 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_1761 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_1762 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_1763
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_1764 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_8_state_stFinish =
    _readRecord_T_1760 & chainingRecord_0_bits_state_stFinish | _readRecord_T_1761 & chainingRecord_1_bits_state_stFinish | _readRecord_T_1762 & chainingRecord_2_bits_state_stFinish | _readRecord_T_1763
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_1764 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_8_elementMask =
    (_readRecord_T_1760 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_1761 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_1762 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_1763 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_1764 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_8_slow =
    _readRecord_T_1760 & chainingRecord_0_bits_slow | _readRecord_T_1761 & chainingRecord_1_bits_slow | _readRecord_T_1762 & chainingRecord_2_bits_slow | _readRecord_T_1763 & chainingRecord_3_bits_slow | _readRecord_T_1764
    & chainingRecord_4_bits_slow;
  wire         readRecord_8_onlyRead =
    _readRecord_T_1760 & chainingRecord_0_bits_onlyRead | _readRecord_T_1761 & chainingRecord_1_bits_onlyRead | _readRecord_T_1762 & chainingRecord_2_bits_onlyRead | _readRecord_T_1763 & chainingRecord_3_bits_onlyRead | _readRecord_T_1764
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_8_ma =
    _readRecord_T_1760 & chainingRecord_0_bits_ma | _readRecord_T_1761 & chainingRecord_1_bits_ma | _readRecord_T_1762 & chainingRecord_2_bits_ma | _readRecord_T_1763 & chainingRecord_3_bits_ma | _readRecord_T_1764
    & chainingRecord_4_bits_ma;
  wire         readRecord_8_indexType =
    _readRecord_T_1760 & chainingRecord_0_bits_indexType | _readRecord_T_1761 & chainingRecord_1_bits_indexType | _readRecord_T_1762 & chainingRecord_2_bits_indexType | _readRecord_T_1763 & chainingRecord_3_bits_indexType
    | _readRecord_T_1764 & chainingRecord_4_bits_indexType;
  wire         readRecord_8_crossRead =
    _readRecord_T_1760 & chainingRecord_0_bits_crossRead | _readRecord_T_1761 & chainingRecord_1_bits_crossRead | _readRecord_T_1762 & chainingRecord_2_bits_crossRead | _readRecord_T_1763 & chainingRecord_3_bits_crossRead
    | _readRecord_T_1764 & chainingRecord_4_bits_crossRead;
  wire         readRecord_8_crossWrite =
    _readRecord_T_1760 & chainingRecord_0_bits_crossWrite | _readRecord_T_1761 & chainingRecord_1_bits_crossWrite | _readRecord_T_1762 & chainingRecord_2_bits_crossWrite | _readRecord_T_1763 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_1764 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_8_gather16 =
    _readRecord_T_1760 & chainingRecord_0_bits_gather16 | _readRecord_T_1761 & chainingRecord_1_bits_gather16 | _readRecord_T_1762 & chainingRecord_2_bits_gather16 | _readRecord_T_1763 & chainingRecord_3_bits_gather16 | _readRecord_T_1764
    & chainingRecord_4_bits_gather16;
  wire         readRecord_8_gather =
    _readRecord_T_1760 & chainingRecord_0_bits_gather | _readRecord_T_1761 & chainingRecord_1_bits_gather | _readRecord_T_1762 & chainingRecord_2_bits_gather | _readRecord_T_1763 & chainingRecord_3_bits_gather | _readRecord_T_1764
    & chainingRecord_4_bits_gather;
  wire         readRecord_8_st =
    _readRecord_T_1760 & chainingRecord_0_bits_st | _readRecord_T_1761 & chainingRecord_1_bits_st | _readRecord_T_1762 & chainingRecord_2_bits_st | _readRecord_T_1763 & chainingRecord_3_bits_st | _readRecord_T_1764
    & chainingRecord_4_bits_st;
  wire         readRecord_8_ls =
    _readRecord_T_1760 & chainingRecord_0_bits_ls | _readRecord_T_1761 & chainingRecord_1_bits_ls | _readRecord_T_1762 & chainingRecord_2_bits_ls | _readRecord_T_1763 & chainingRecord_3_bits_ls | _readRecord_T_1764
    & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_8_instIndex =
    (_readRecord_T_1760 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_1761 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_1762 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_1763 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_1764 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_8_vs2 =
    (_readRecord_T_1760 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_1761 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_1762 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_1763 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_1764 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_8_vs1_bits =
    (_readRecord_T_1760 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_1761 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_1762 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_1763 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_1764 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_8_vs1_valid =
    _readRecord_T_1760 & chainingRecord_0_bits_vs1_valid | _readRecord_T_1761 & chainingRecord_1_bits_vs1_valid | _readRecord_T_1762 & chainingRecord_2_bits_vs1_valid | _readRecord_T_1763 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_1764 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_8_vd_bits =
    (_readRecord_T_1760 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_1761 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_1762 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_1763 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_1764 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_8_vd_valid =
    _readRecord_T_1760 & chainingRecord_0_bits_vd_valid | _readRecord_T_1761 & chainingRecord_1_bits_vd_valid | _readRecord_T_1762 & chainingRecord_2_bits_vd_valid | _readRecord_T_1763 & chainingRecord_3_bits_vd_valid | _readRecord_T_1764
    & chainingRecord_4_bits_vd_valid;
  wire         _readRecord_T_1980 = chainingRecord_0_bits_instIndex == readCheck_9_instructionIndex;
  wire         _readRecord_T_1981 = chainingRecord_1_bits_instIndex == readCheck_9_instructionIndex;
  wire         _readRecord_T_1982 = chainingRecord_2_bits_instIndex == readCheck_9_instructionIndex;
  wire         _readRecord_T_1983 = chainingRecord_3_bits_instIndex == readCheck_9_instructionIndex;
  wire         _readRecord_T_1984 = chainingRecord_4_bits_instIndex == readCheck_9_instructionIndex;
  wire         readRecord_9_state_wLaneClear =
    _readRecord_T_1980 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_1981 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_1982 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_1983
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_1984 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_9_state_wTopLastReport =
    _readRecord_T_1980 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_1981 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_1982 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_1983
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_1984 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_9_state_wLaneLastReport =
    _readRecord_T_1980 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_1981 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_1982 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_1983
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_1984 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_9_state_wWriteQueueClear =
    _readRecord_T_1980 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_1981 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_1982 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_1983
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_1984 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_9_state_stFinish =
    _readRecord_T_1980 & chainingRecord_0_bits_state_stFinish | _readRecord_T_1981 & chainingRecord_1_bits_state_stFinish | _readRecord_T_1982 & chainingRecord_2_bits_state_stFinish | _readRecord_T_1983
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_1984 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_9_elementMask =
    (_readRecord_T_1980 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_1981 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_1982 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_1983 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_1984 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_9_slow =
    _readRecord_T_1980 & chainingRecord_0_bits_slow | _readRecord_T_1981 & chainingRecord_1_bits_slow | _readRecord_T_1982 & chainingRecord_2_bits_slow | _readRecord_T_1983 & chainingRecord_3_bits_slow | _readRecord_T_1984
    & chainingRecord_4_bits_slow;
  wire         readRecord_9_onlyRead =
    _readRecord_T_1980 & chainingRecord_0_bits_onlyRead | _readRecord_T_1981 & chainingRecord_1_bits_onlyRead | _readRecord_T_1982 & chainingRecord_2_bits_onlyRead | _readRecord_T_1983 & chainingRecord_3_bits_onlyRead | _readRecord_T_1984
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_9_ma =
    _readRecord_T_1980 & chainingRecord_0_bits_ma | _readRecord_T_1981 & chainingRecord_1_bits_ma | _readRecord_T_1982 & chainingRecord_2_bits_ma | _readRecord_T_1983 & chainingRecord_3_bits_ma | _readRecord_T_1984
    & chainingRecord_4_bits_ma;
  wire         readRecord_9_indexType =
    _readRecord_T_1980 & chainingRecord_0_bits_indexType | _readRecord_T_1981 & chainingRecord_1_bits_indexType | _readRecord_T_1982 & chainingRecord_2_bits_indexType | _readRecord_T_1983 & chainingRecord_3_bits_indexType
    | _readRecord_T_1984 & chainingRecord_4_bits_indexType;
  wire         readRecord_9_crossRead =
    _readRecord_T_1980 & chainingRecord_0_bits_crossRead | _readRecord_T_1981 & chainingRecord_1_bits_crossRead | _readRecord_T_1982 & chainingRecord_2_bits_crossRead | _readRecord_T_1983 & chainingRecord_3_bits_crossRead
    | _readRecord_T_1984 & chainingRecord_4_bits_crossRead;
  wire         readRecord_9_crossWrite =
    _readRecord_T_1980 & chainingRecord_0_bits_crossWrite | _readRecord_T_1981 & chainingRecord_1_bits_crossWrite | _readRecord_T_1982 & chainingRecord_2_bits_crossWrite | _readRecord_T_1983 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_1984 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_9_gather16 =
    _readRecord_T_1980 & chainingRecord_0_bits_gather16 | _readRecord_T_1981 & chainingRecord_1_bits_gather16 | _readRecord_T_1982 & chainingRecord_2_bits_gather16 | _readRecord_T_1983 & chainingRecord_3_bits_gather16 | _readRecord_T_1984
    & chainingRecord_4_bits_gather16;
  wire         readRecord_9_gather =
    _readRecord_T_1980 & chainingRecord_0_bits_gather | _readRecord_T_1981 & chainingRecord_1_bits_gather | _readRecord_T_1982 & chainingRecord_2_bits_gather | _readRecord_T_1983 & chainingRecord_3_bits_gather | _readRecord_T_1984
    & chainingRecord_4_bits_gather;
  wire         readRecord_9_st =
    _readRecord_T_1980 & chainingRecord_0_bits_st | _readRecord_T_1981 & chainingRecord_1_bits_st | _readRecord_T_1982 & chainingRecord_2_bits_st | _readRecord_T_1983 & chainingRecord_3_bits_st | _readRecord_T_1984
    & chainingRecord_4_bits_st;
  wire         readRecord_9_ls =
    _readRecord_T_1980 & chainingRecord_0_bits_ls | _readRecord_T_1981 & chainingRecord_1_bits_ls | _readRecord_T_1982 & chainingRecord_2_bits_ls | _readRecord_T_1983 & chainingRecord_3_bits_ls | _readRecord_T_1984
    & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_9_instIndex =
    (_readRecord_T_1980 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_1981 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_1982 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_1983 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_1984 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_9_vs2 =
    (_readRecord_T_1980 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_1981 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_1982 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_1983 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_1984 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_9_vs1_bits =
    (_readRecord_T_1980 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_1981 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_1982 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_1983 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_1984 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_9_vs1_valid =
    _readRecord_T_1980 & chainingRecord_0_bits_vs1_valid | _readRecord_T_1981 & chainingRecord_1_bits_vs1_valid | _readRecord_T_1982 & chainingRecord_2_bits_vs1_valid | _readRecord_T_1983 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_1984 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_9_vd_bits =
    (_readRecord_T_1980 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_1981 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_1982 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_1983 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_1984 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_9_vd_valid =
    _readRecord_T_1980 & chainingRecord_0_bits_vd_valid | _readRecord_T_1981 & chainingRecord_1_bits_vd_valid | _readRecord_T_1982 & chainingRecord_2_bits_vd_valid | _readRecord_T_1983 & chainingRecord_3_bits_vd_valid | _readRecord_T_1984
    & chainingRecord_4_bits_vd_valid;
  wire         _readRecord_T_2200 = chainingRecord_0_bits_instIndex == readCheck_10_instructionIndex;
  wire         _readRecord_T_2201 = chainingRecord_1_bits_instIndex == readCheck_10_instructionIndex;
  wire         _readRecord_T_2202 = chainingRecord_2_bits_instIndex == readCheck_10_instructionIndex;
  wire         _readRecord_T_2203 = chainingRecord_3_bits_instIndex == readCheck_10_instructionIndex;
  wire         _readRecord_T_2204 = chainingRecord_4_bits_instIndex == readCheck_10_instructionIndex;
  wire         readRecord_10_state_wLaneClear =
    _readRecord_T_2200 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_2201 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_2202 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_2203
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_2204 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_10_state_wTopLastReport =
    _readRecord_T_2200 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_2201 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_2202 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_2203
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_2204 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_10_state_wLaneLastReport =
    _readRecord_T_2200 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_2201 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_2202 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_2203
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_2204 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_10_state_wWriteQueueClear =
    _readRecord_T_2200 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_2201 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_2202 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_2203
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_2204 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_10_state_stFinish =
    _readRecord_T_2200 & chainingRecord_0_bits_state_stFinish | _readRecord_T_2201 & chainingRecord_1_bits_state_stFinish | _readRecord_T_2202 & chainingRecord_2_bits_state_stFinish | _readRecord_T_2203
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_2204 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_10_elementMask =
    (_readRecord_T_2200 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_2201 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_2202 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_2203 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_2204 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_10_slow =
    _readRecord_T_2200 & chainingRecord_0_bits_slow | _readRecord_T_2201 & chainingRecord_1_bits_slow | _readRecord_T_2202 & chainingRecord_2_bits_slow | _readRecord_T_2203 & chainingRecord_3_bits_slow | _readRecord_T_2204
    & chainingRecord_4_bits_slow;
  wire         readRecord_10_onlyRead =
    _readRecord_T_2200 & chainingRecord_0_bits_onlyRead | _readRecord_T_2201 & chainingRecord_1_bits_onlyRead | _readRecord_T_2202 & chainingRecord_2_bits_onlyRead | _readRecord_T_2203 & chainingRecord_3_bits_onlyRead | _readRecord_T_2204
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_10_ma =
    _readRecord_T_2200 & chainingRecord_0_bits_ma | _readRecord_T_2201 & chainingRecord_1_bits_ma | _readRecord_T_2202 & chainingRecord_2_bits_ma | _readRecord_T_2203 & chainingRecord_3_bits_ma | _readRecord_T_2204
    & chainingRecord_4_bits_ma;
  wire         readRecord_10_indexType =
    _readRecord_T_2200 & chainingRecord_0_bits_indexType | _readRecord_T_2201 & chainingRecord_1_bits_indexType | _readRecord_T_2202 & chainingRecord_2_bits_indexType | _readRecord_T_2203 & chainingRecord_3_bits_indexType
    | _readRecord_T_2204 & chainingRecord_4_bits_indexType;
  wire         readRecord_10_crossRead =
    _readRecord_T_2200 & chainingRecord_0_bits_crossRead | _readRecord_T_2201 & chainingRecord_1_bits_crossRead | _readRecord_T_2202 & chainingRecord_2_bits_crossRead | _readRecord_T_2203 & chainingRecord_3_bits_crossRead
    | _readRecord_T_2204 & chainingRecord_4_bits_crossRead;
  wire         readRecord_10_crossWrite =
    _readRecord_T_2200 & chainingRecord_0_bits_crossWrite | _readRecord_T_2201 & chainingRecord_1_bits_crossWrite | _readRecord_T_2202 & chainingRecord_2_bits_crossWrite | _readRecord_T_2203 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_2204 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_10_gather16 =
    _readRecord_T_2200 & chainingRecord_0_bits_gather16 | _readRecord_T_2201 & chainingRecord_1_bits_gather16 | _readRecord_T_2202 & chainingRecord_2_bits_gather16 | _readRecord_T_2203 & chainingRecord_3_bits_gather16 | _readRecord_T_2204
    & chainingRecord_4_bits_gather16;
  wire         readRecord_10_gather =
    _readRecord_T_2200 & chainingRecord_0_bits_gather | _readRecord_T_2201 & chainingRecord_1_bits_gather | _readRecord_T_2202 & chainingRecord_2_bits_gather | _readRecord_T_2203 & chainingRecord_3_bits_gather | _readRecord_T_2204
    & chainingRecord_4_bits_gather;
  wire         readRecord_10_st =
    _readRecord_T_2200 & chainingRecord_0_bits_st | _readRecord_T_2201 & chainingRecord_1_bits_st | _readRecord_T_2202 & chainingRecord_2_bits_st | _readRecord_T_2203 & chainingRecord_3_bits_st | _readRecord_T_2204
    & chainingRecord_4_bits_st;
  wire         readRecord_10_ls =
    _readRecord_T_2200 & chainingRecord_0_bits_ls | _readRecord_T_2201 & chainingRecord_1_bits_ls | _readRecord_T_2202 & chainingRecord_2_bits_ls | _readRecord_T_2203 & chainingRecord_3_bits_ls | _readRecord_T_2204
    & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_10_instIndex =
    (_readRecord_T_2200 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_2201 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_2202 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_2203 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_2204 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_10_vs2 =
    (_readRecord_T_2200 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_2201 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_2202 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_2203 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_2204 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_10_vs1_bits =
    (_readRecord_T_2200 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_2201 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_2202 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_2203 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_2204 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_10_vs1_valid =
    _readRecord_T_2200 & chainingRecord_0_bits_vs1_valid | _readRecord_T_2201 & chainingRecord_1_bits_vs1_valid | _readRecord_T_2202 & chainingRecord_2_bits_vs1_valid | _readRecord_T_2203 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_2204 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_10_vd_bits =
    (_readRecord_T_2200 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_2201 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_2202 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_2203 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_2204 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_10_vd_valid =
    _readRecord_T_2200 & chainingRecord_0_bits_vd_valid | _readRecord_T_2201 & chainingRecord_1_bits_vd_valid | _readRecord_T_2202 & chainingRecord_2_bits_vd_valid | _readRecord_T_2203 & chainingRecord_3_bits_vd_valid | _readRecord_T_2204
    & chainingRecord_4_bits_vd_valid;
  wire         _readRecord_T_2420 = chainingRecord_0_bits_instIndex == readCheck_11_instructionIndex;
  wire         _readRecord_T_2421 = chainingRecord_1_bits_instIndex == readCheck_11_instructionIndex;
  wire         _readRecord_T_2422 = chainingRecord_2_bits_instIndex == readCheck_11_instructionIndex;
  wire         _readRecord_T_2423 = chainingRecord_3_bits_instIndex == readCheck_11_instructionIndex;
  wire         _readRecord_T_2424 = chainingRecord_4_bits_instIndex == readCheck_11_instructionIndex;
  wire         readRecord_11_state_wLaneClear =
    _readRecord_T_2420 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_2421 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_2422 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_2423
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_2424 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_11_state_wTopLastReport =
    _readRecord_T_2420 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_2421 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_2422 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_2423
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_2424 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_11_state_wLaneLastReport =
    _readRecord_T_2420 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_2421 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_2422 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_2423
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_2424 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_11_state_wWriteQueueClear =
    _readRecord_T_2420 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_2421 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_2422 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_2423
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_2424 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_11_state_stFinish =
    _readRecord_T_2420 & chainingRecord_0_bits_state_stFinish | _readRecord_T_2421 & chainingRecord_1_bits_state_stFinish | _readRecord_T_2422 & chainingRecord_2_bits_state_stFinish | _readRecord_T_2423
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_2424 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_11_elementMask =
    (_readRecord_T_2420 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_2421 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_2422 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_2423 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_2424 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_11_slow =
    _readRecord_T_2420 & chainingRecord_0_bits_slow | _readRecord_T_2421 & chainingRecord_1_bits_slow | _readRecord_T_2422 & chainingRecord_2_bits_slow | _readRecord_T_2423 & chainingRecord_3_bits_slow | _readRecord_T_2424
    & chainingRecord_4_bits_slow;
  wire         readRecord_11_onlyRead =
    _readRecord_T_2420 & chainingRecord_0_bits_onlyRead | _readRecord_T_2421 & chainingRecord_1_bits_onlyRead | _readRecord_T_2422 & chainingRecord_2_bits_onlyRead | _readRecord_T_2423 & chainingRecord_3_bits_onlyRead | _readRecord_T_2424
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_11_ma =
    _readRecord_T_2420 & chainingRecord_0_bits_ma | _readRecord_T_2421 & chainingRecord_1_bits_ma | _readRecord_T_2422 & chainingRecord_2_bits_ma | _readRecord_T_2423 & chainingRecord_3_bits_ma | _readRecord_T_2424
    & chainingRecord_4_bits_ma;
  wire         readRecord_11_indexType =
    _readRecord_T_2420 & chainingRecord_0_bits_indexType | _readRecord_T_2421 & chainingRecord_1_bits_indexType | _readRecord_T_2422 & chainingRecord_2_bits_indexType | _readRecord_T_2423 & chainingRecord_3_bits_indexType
    | _readRecord_T_2424 & chainingRecord_4_bits_indexType;
  wire         readRecord_11_crossRead =
    _readRecord_T_2420 & chainingRecord_0_bits_crossRead | _readRecord_T_2421 & chainingRecord_1_bits_crossRead | _readRecord_T_2422 & chainingRecord_2_bits_crossRead | _readRecord_T_2423 & chainingRecord_3_bits_crossRead
    | _readRecord_T_2424 & chainingRecord_4_bits_crossRead;
  wire         readRecord_11_crossWrite =
    _readRecord_T_2420 & chainingRecord_0_bits_crossWrite | _readRecord_T_2421 & chainingRecord_1_bits_crossWrite | _readRecord_T_2422 & chainingRecord_2_bits_crossWrite | _readRecord_T_2423 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_2424 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_11_gather16 =
    _readRecord_T_2420 & chainingRecord_0_bits_gather16 | _readRecord_T_2421 & chainingRecord_1_bits_gather16 | _readRecord_T_2422 & chainingRecord_2_bits_gather16 | _readRecord_T_2423 & chainingRecord_3_bits_gather16 | _readRecord_T_2424
    & chainingRecord_4_bits_gather16;
  wire         readRecord_11_gather =
    _readRecord_T_2420 & chainingRecord_0_bits_gather | _readRecord_T_2421 & chainingRecord_1_bits_gather | _readRecord_T_2422 & chainingRecord_2_bits_gather | _readRecord_T_2423 & chainingRecord_3_bits_gather | _readRecord_T_2424
    & chainingRecord_4_bits_gather;
  wire         readRecord_11_st =
    _readRecord_T_2420 & chainingRecord_0_bits_st | _readRecord_T_2421 & chainingRecord_1_bits_st | _readRecord_T_2422 & chainingRecord_2_bits_st | _readRecord_T_2423 & chainingRecord_3_bits_st | _readRecord_T_2424
    & chainingRecord_4_bits_st;
  wire         readRecord_11_ls =
    _readRecord_T_2420 & chainingRecord_0_bits_ls | _readRecord_T_2421 & chainingRecord_1_bits_ls | _readRecord_T_2422 & chainingRecord_2_bits_ls | _readRecord_T_2423 & chainingRecord_3_bits_ls | _readRecord_T_2424
    & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_11_instIndex =
    (_readRecord_T_2420 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_2421 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_2422 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_2423 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_2424 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_11_vs2 =
    (_readRecord_T_2420 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_2421 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_2422 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_2423 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_2424 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_11_vs1_bits =
    (_readRecord_T_2420 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_2421 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_2422 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_2423 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_2424 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_11_vs1_valid =
    _readRecord_T_2420 & chainingRecord_0_bits_vs1_valid | _readRecord_T_2421 & chainingRecord_1_bits_vs1_valid | _readRecord_T_2422 & chainingRecord_2_bits_vs1_valid | _readRecord_T_2423 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_2424 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_11_vd_bits =
    (_readRecord_T_2420 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_2421 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_2422 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_2423 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_2424 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_11_vd_valid =
    _readRecord_T_2420 & chainingRecord_0_bits_vd_valid | _readRecord_T_2421 & chainingRecord_1_bits_vd_valid | _readRecord_T_2422 & chainingRecord_2_bits_vd_valid | _readRecord_T_2423 & chainingRecord_3_bits_vd_valid | _readRecord_T_2424
    & chainingRecord_4_bits_vd_valid;
  wire         _readRecord_T_2640 = chainingRecord_0_bits_instIndex == readCheck_12_instructionIndex;
  wire         _readRecord_T_2641 = chainingRecord_1_bits_instIndex == readCheck_12_instructionIndex;
  wire         _readRecord_T_2642 = chainingRecord_2_bits_instIndex == readCheck_12_instructionIndex;
  wire         _readRecord_T_2643 = chainingRecord_3_bits_instIndex == readCheck_12_instructionIndex;
  wire         _readRecord_T_2644 = chainingRecord_4_bits_instIndex == readCheck_12_instructionIndex;
  wire         readRecord_12_state_wLaneClear =
    _readRecord_T_2640 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_2641 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_2642 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_2643
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_2644 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_12_state_wTopLastReport =
    _readRecord_T_2640 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_2641 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_2642 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_2643
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_2644 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_12_state_wLaneLastReport =
    _readRecord_T_2640 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_2641 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_2642 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_2643
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_2644 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_12_state_wWriteQueueClear =
    _readRecord_T_2640 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_2641 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_2642 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_2643
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_2644 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_12_state_stFinish =
    _readRecord_T_2640 & chainingRecord_0_bits_state_stFinish | _readRecord_T_2641 & chainingRecord_1_bits_state_stFinish | _readRecord_T_2642 & chainingRecord_2_bits_state_stFinish | _readRecord_T_2643
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_2644 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_12_elementMask =
    (_readRecord_T_2640 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_2641 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_2642 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_2643 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_2644 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_12_slow =
    _readRecord_T_2640 & chainingRecord_0_bits_slow | _readRecord_T_2641 & chainingRecord_1_bits_slow | _readRecord_T_2642 & chainingRecord_2_bits_slow | _readRecord_T_2643 & chainingRecord_3_bits_slow | _readRecord_T_2644
    & chainingRecord_4_bits_slow;
  wire         readRecord_12_onlyRead =
    _readRecord_T_2640 & chainingRecord_0_bits_onlyRead | _readRecord_T_2641 & chainingRecord_1_bits_onlyRead | _readRecord_T_2642 & chainingRecord_2_bits_onlyRead | _readRecord_T_2643 & chainingRecord_3_bits_onlyRead | _readRecord_T_2644
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_12_ma =
    _readRecord_T_2640 & chainingRecord_0_bits_ma | _readRecord_T_2641 & chainingRecord_1_bits_ma | _readRecord_T_2642 & chainingRecord_2_bits_ma | _readRecord_T_2643 & chainingRecord_3_bits_ma | _readRecord_T_2644
    & chainingRecord_4_bits_ma;
  wire         readRecord_12_indexType =
    _readRecord_T_2640 & chainingRecord_0_bits_indexType | _readRecord_T_2641 & chainingRecord_1_bits_indexType | _readRecord_T_2642 & chainingRecord_2_bits_indexType | _readRecord_T_2643 & chainingRecord_3_bits_indexType
    | _readRecord_T_2644 & chainingRecord_4_bits_indexType;
  wire         readRecord_12_crossRead =
    _readRecord_T_2640 & chainingRecord_0_bits_crossRead | _readRecord_T_2641 & chainingRecord_1_bits_crossRead | _readRecord_T_2642 & chainingRecord_2_bits_crossRead | _readRecord_T_2643 & chainingRecord_3_bits_crossRead
    | _readRecord_T_2644 & chainingRecord_4_bits_crossRead;
  wire         readRecord_12_crossWrite =
    _readRecord_T_2640 & chainingRecord_0_bits_crossWrite | _readRecord_T_2641 & chainingRecord_1_bits_crossWrite | _readRecord_T_2642 & chainingRecord_2_bits_crossWrite | _readRecord_T_2643 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_2644 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_12_gather16 =
    _readRecord_T_2640 & chainingRecord_0_bits_gather16 | _readRecord_T_2641 & chainingRecord_1_bits_gather16 | _readRecord_T_2642 & chainingRecord_2_bits_gather16 | _readRecord_T_2643 & chainingRecord_3_bits_gather16 | _readRecord_T_2644
    & chainingRecord_4_bits_gather16;
  wire         readRecord_12_gather =
    _readRecord_T_2640 & chainingRecord_0_bits_gather | _readRecord_T_2641 & chainingRecord_1_bits_gather | _readRecord_T_2642 & chainingRecord_2_bits_gather | _readRecord_T_2643 & chainingRecord_3_bits_gather | _readRecord_T_2644
    & chainingRecord_4_bits_gather;
  wire         readRecord_12_st =
    _readRecord_T_2640 & chainingRecord_0_bits_st | _readRecord_T_2641 & chainingRecord_1_bits_st | _readRecord_T_2642 & chainingRecord_2_bits_st | _readRecord_T_2643 & chainingRecord_3_bits_st | _readRecord_T_2644
    & chainingRecord_4_bits_st;
  wire         readRecord_12_ls =
    _readRecord_T_2640 & chainingRecord_0_bits_ls | _readRecord_T_2641 & chainingRecord_1_bits_ls | _readRecord_T_2642 & chainingRecord_2_bits_ls | _readRecord_T_2643 & chainingRecord_3_bits_ls | _readRecord_T_2644
    & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_12_instIndex =
    (_readRecord_T_2640 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_2641 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_2642 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_2643 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_2644 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_12_vs2 =
    (_readRecord_T_2640 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_2641 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_2642 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_2643 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_2644 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_12_vs1_bits =
    (_readRecord_T_2640 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_2641 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_2642 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_2643 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_2644 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_12_vs1_valid =
    _readRecord_T_2640 & chainingRecord_0_bits_vs1_valid | _readRecord_T_2641 & chainingRecord_1_bits_vs1_valid | _readRecord_T_2642 & chainingRecord_2_bits_vs1_valid | _readRecord_T_2643 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_2644 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_12_vd_bits =
    (_readRecord_T_2640 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_2641 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_2642 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_2643 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_2644 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_12_vd_valid =
    _readRecord_T_2640 & chainingRecord_0_bits_vd_valid | _readRecord_T_2641 & chainingRecord_1_bits_vd_valid | _readRecord_T_2642 & chainingRecord_2_bits_vd_valid | _readRecord_T_2643 & chainingRecord_3_bits_vd_valid | _readRecord_T_2644
    & chainingRecord_4_bits_vd_valid;
  wire         _readRecord_T_2860 = chainingRecord_0_bits_instIndex == readCheck_13_instructionIndex;
  wire         _readRecord_T_2861 = chainingRecord_1_bits_instIndex == readCheck_13_instructionIndex;
  wire         _readRecord_T_2862 = chainingRecord_2_bits_instIndex == readCheck_13_instructionIndex;
  wire         _readRecord_T_2863 = chainingRecord_3_bits_instIndex == readCheck_13_instructionIndex;
  wire         _readRecord_T_2864 = chainingRecord_4_bits_instIndex == readCheck_13_instructionIndex;
  wire         readRecord_13_state_wLaneClear =
    _readRecord_T_2860 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_2861 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_2862 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_2863
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_2864 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_13_state_wTopLastReport =
    _readRecord_T_2860 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_2861 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_2862 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_2863
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_2864 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_13_state_wLaneLastReport =
    _readRecord_T_2860 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_2861 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_2862 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_2863
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_2864 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_13_state_wWriteQueueClear =
    _readRecord_T_2860 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_2861 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_2862 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_2863
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_2864 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_13_state_stFinish =
    _readRecord_T_2860 & chainingRecord_0_bits_state_stFinish | _readRecord_T_2861 & chainingRecord_1_bits_state_stFinish | _readRecord_T_2862 & chainingRecord_2_bits_state_stFinish | _readRecord_T_2863
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_2864 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_13_elementMask =
    (_readRecord_T_2860 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_2861 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_2862 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_2863 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_2864 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_13_slow =
    _readRecord_T_2860 & chainingRecord_0_bits_slow | _readRecord_T_2861 & chainingRecord_1_bits_slow | _readRecord_T_2862 & chainingRecord_2_bits_slow | _readRecord_T_2863 & chainingRecord_3_bits_slow | _readRecord_T_2864
    & chainingRecord_4_bits_slow;
  wire         readRecord_13_onlyRead =
    _readRecord_T_2860 & chainingRecord_0_bits_onlyRead | _readRecord_T_2861 & chainingRecord_1_bits_onlyRead | _readRecord_T_2862 & chainingRecord_2_bits_onlyRead | _readRecord_T_2863 & chainingRecord_3_bits_onlyRead | _readRecord_T_2864
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_13_ma =
    _readRecord_T_2860 & chainingRecord_0_bits_ma | _readRecord_T_2861 & chainingRecord_1_bits_ma | _readRecord_T_2862 & chainingRecord_2_bits_ma | _readRecord_T_2863 & chainingRecord_3_bits_ma | _readRecord_T_2864
    & chainingRecord_4_bits_ma;
  wire         readRecord_13_indexType =
    _readRecord_T_2860 & chainingRecord_0_bits_indexType | _readRecord_T_2861 & chainingRecord_1_bits_indexType | _readRecord_T_2862 & chainingRecord_2_bits_indexType | _readRecord_T_2863 & chainingRecord_3_bits_indexType
    | _readRecord_T_2864 & chainingRecord_4_bits_indexType;
  wire         readRecord_13_crossRead =
    _readRecord_T_2860 & chainingRecord_0_bits_crossRead | _readRecord_T_2861 & chainingRecord_1_bits_crossRead | _readRecord_T_2862 & chainingRecord_2_bits_crossRead | _readRecord_T_2863 & chainingRecord_3_bits_crossRead
    | _readRecord_T_2864 & chainingRecord_4_bits_crossRead;
  wire         readRecord_13_crossWrite =
    _readRecord_T_2860 & chainingRecord_0_bits_crossWrite | _readRecord_T_2861 & chainingRecord_1_bits_crossWrite | _readRecord_T_2862 & chainingRecord_2_bits_crossWrite | _readRecord_T_2863 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_2864 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_13_gather16 =
    _readRecord_T_2860 & chainingRecord_0_bits_gather16 | _readRecord_T_2861 & chainingRecord_1_bits_gather16 | _readRecord_T_2862 & chainingRecord_2_bits_gather16 | _readRecord_T_2863 & chainingRecord_3_bits_gather16 | _readRecord_T_2864
    & chainingRecord_4_bits_gather16;
  wire         readRecord_13_gather =
    _readRecord_T_2860 & chainingRecord_0_bits_gather | _readRecord_T_2861 & chainingRecord_1_bits_gather | _readRecord_T_2862 & chainingRecord_2_bits_gather | _readRecord_T_2863 & chainingRecord_3_bits_gather | _readRecord_T_2864
    & chainingRecord_4_bits_gather;
  wire         readRecord_13_st =
    _readRecord_T_2860 & chainingRecord_0_bits_st | _readRecord_T_2861 & chainingRecord_1_bits_st | _readRecord_T_2862 & chainingRecord_2_bits_st | _readRecord_T_2863 & chainingRecord_3_bits_st | _readRecord_T_2864
    & chainingRecord_4_bits_st;
  wire         readRecord_13_ls =
    _readRecord_T_2860 & chainingRecord_0_bits_ls | _readRecord_T_2861 & chainingRecord_1_bits_ls | _readRecord_T_2862 & chainingRecord_2_bits_ls | _readRecord_T_2863 & chainingRecord_3_bits_ls | _readRecord_T_2864
    & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_13_instIndex =
    (_readRecord_T_2860 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_2861 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_2862 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_2863 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_2864 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_13_vs2 =
    (_readRecord_T_2860 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_2861 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_2862 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_2863 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_2864 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_13_vs1_bits =
    (_readRecord_T_2860 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_2861 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_2862 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_2863 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_2864 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_13_vs1_valid =
    _readRecord_T_2860 & chainingRecord_0_bits_vs1_valid | _readRecord_T_2861 & chainingRecord_1_bits_vs1_valid | _readRecord_T_2862 & chainingRecord_2_bits_vs1_valid | _readRecord_T_2863 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_2864 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_13_vd_bits =
    (_readRecord_T_2860 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_2861 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_2862 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_2863 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_2864 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_13_vd_valid =
    _readRecord_T_2860 & chainingRecord_0_bits_vd_valid | _readRecord_T_2861 & chainingRecord_1_bits_vd_valid | _readRecord_T_2862 & chainingRecord_2_bits_vd_valid | _readRecord_T_2863 & chainingRecord_3_bits_vd_valid | _readRecord_T_2864
    & chainingRecord_4_bits_vd_valid;
  wire         _readRecord_T_3080 = chainingRecord_0_bits_instIndex == readRequests_0_bits_instructionIndex_0;
  wire         _readRecord_T_3081 = chainingRecord_1_bits_instIndex == readRequests_0_bits_instructionIndex_0;
  wire         _readRecord_T_3082 = chainingRecord_2_bits_instIndex == readRequests_0_bits_instructionIndex_0;
  wire         _readRecord_T_3083 = chainingRecord_3_bits_instIndex == readRequests_0_bits_instructionIndex_0;
  wire         _readRecord_T_3084 = chainingRecord_4_bits_instIndex == readRequests_0_bits_instructionIndex_0;
  wire         readRecord_14_state_wLaneClear =
    _readRecord_T_3080 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_3081 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_3082 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_3083
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_3084 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_14_state_wTopLastReport =
    _readRecord_T_3080 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_3081 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_3082 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_3083
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_3084 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_14_state_wLaneLastReport =
    _readRecord_T_3080 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_3081 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_3082 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_3083
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_3084 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_14_state_wWriteQueueClear =
    _readRecord_T_3080 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_3081 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_3082 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_3083
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_3084 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_14_state_stFinish =
    _readRecord_T_3080 & chainingRecord_0_bits_state_stFinish | _readRecord_T_3081 & chainingRecord_1_bits_state_stFinish | _readRecord_T_3082 & chainingRecord_2_bits_state_stFinish | _readRecord_T_3083
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_3084 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_14_elementMask =
    (_readRecord_T_3080 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_3081 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_3082 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_3083 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_3084 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_14_slow =
    _readRecord_T_3080 & chainingRecord_0_bits_slow | _readRecord_T_3081 & chainingRecord_1_bits_slow | _readRecord_T_3082 & chainingRecord_2_bits_slow | _readRecord_T_3083 & chainingRecord_3_bits_slow | _readRecord_T_3084
    & chainingRecord_4_bits_slow;
  wire         readRecord_14_onlyRead =
    _readRecord_T_3080 & chainingRecord_0_bits_onlyRead | _readRecord_T_3081 & chainingRecord_1_bits_onlyRead | _readRecord_T_3082 & chainingRecord_2_bits_onlyRead | _readRecord_T_3083 & chainingRecord_3_bits_onlyRead | _readRecord_T_3084
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_14_ma =
    _readRecord_T_3080 & chainingRecord_0_bits_ma | _readRecord_T_3081 & chainingRecord_1_bits_ma | _readRecord_T_3082 & chainingRecord_2_bits_ma | _readRecord_T_3083 & chainingRecord_3_bits_ma | _readRecord_T_3084
    & chainingRecord_4_bits_ma;
  wire         readRecord_14_indexType =
    _readRecord_T_3080 & chainingRecord_0_bits_indexType | _readRecord_T_3081 & chainingRecord_1_bits_indexType | _readRecord_T_3082 & chainingRecord_2_bits_indexType | _readRecord_T_3083 & chainingRecord_3_bits_indexType
    | _readRecord_T_3084 & chainingRecord_4_bits_indexType;
  wire         readRecord_14_crossRead =
    _readRecord_T_3080 & chainingRecord_0_bits_crossRead | _readRecord_T_3081 & chainingRecord_1_bits_crossRead | _readRecord_T_3082 & chainingRecord_2_bits_crossRead | _readRecord_T_3083 & chainingRecord_3_bits_crossRead
    | _readRecord_T_3084 & chainingRecord_4_bits_crossRead;
  wire         readRecord_14_crossWrite =
    _readRecord_T_3080 & chainingRecord_0_bits_crossWrite | _readRecord_T_3081 & chainingRecord_1_bits_crossWrite | _readRecord_T_3082 & chainingRecord_2_bits_crossWrite | _readRecord_T_3083 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_3084 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_14_gather16 =
    _readRecord_T_3080 & chainingRecord_0_bits_gather16 | _readRecord_T_3081 & chainingRecord_1_bits_gather16 | _readRecord_T_3082 & chainingRecord_2_bits_gather16 | _readRecord_T_3083 & chainingRecord_3_bits_gather16 | _readRecord_T_3084
    & chainingRecord_4_bits_gather16;
  wire         readRecord_14_gather =
    _readRecord_T_3080 & chainingRecord_0_bits_gather | _readRecord_T_3081 & chainingRecord_1_bits_gather | _readRecord_T_3082 & chainingRecord_2_bits_gather | _readRecord_T_3083 & chainingRecord_3_bits_gather | _readRecord_T_3084
    & chainingRecord_4_bits_gather;
  wire         readRecord_14_st =
    _readRecord_T_3080 & chainingRecord_0_bits_st | _readRecord_T_3081 & chainingRecord_1_bits_st | _readRecord_T_3082 & chainingRecord_2_bits_st | _readRecord_T_3083 & chainingRecord_3_bits_st | _readRecord_T_3084
    & chainingRecord_4_bits_st;
  wire         readRecord_14_ls =
    _readRecord_T_3080 & chainingRecord_0_bits_ls | _readRecord_T_3081 & chainingRecord_1_bits_ls | _readRecord_T_3082 & chainingRecord_2_bits_ls | _readRecord_T_3083 & chainingRecord_3_bits_ls | _readRecord_T_3084
    & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_14_instIndex =
    (_readRecord_T_3080 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_3081 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_3082 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_3083 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_3084 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_14_vs2 =
    (_readRecord_T_3080 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_3081 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_3082 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_3083 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_3084 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_14_vs1_bits =
    (_readRecord_T_3080 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_3081 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_3082 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_3083 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_3084 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_14_vs1_valid =
    _readRecord_T_3080 & chainingRecord_0_bits_vs1_valid | _readRecord_T_3081 & chainingRecord_1_bits_vs1_valid | _readRecord_T_3082 & chainingRecord_2_bits_vs1_valid | _readRecord_T_3083 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_3084 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_14_vd_bits =
    (_readRecord_T_3080 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_3081 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_3082 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_3083 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_3084 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_14_vd_valid =
    _readRecord_T_3080 & chainingRecord_0_bits_vd_valid | _readRecord_T_3081 & chainingRecord_1_bits_vd_valid | _readRecord_T_3082 & chainingRecord_2_bits_vd_valid | _readRecord_T_3083 & chainingRecord_3_bits_vd_valid | _readRecord_T_3084
    & chainingRecord_4_bits_vd_valid;
  wire [9:0]   address = {readRequests_0_bits_vs_0, readRequests_0_bits_offset_0};
  reg          pipeBank_pipe_v;
  reg          pipeBank_pipe_pipe_v;
  wire         pipeBank_pipe_pipe_out_valid = pipeBank_pipe_pipe_v;
  reg          pipeBank_pipe_pipe_b;
  wire         pipeBank_pipe_pipe_out_bits = pipeBank_pipe_pipe_b;
  wire         bankReadF_0 = bankCorrect;
  wire         firstUsed_1 = bankCorrect;
  reg          pipeFirstUsed_pipe_v;
  reg          pipeFirstUsed_pipe_pipe_v;
  wire         pipeFirstUsed_pipe_pipe_out_valid = pipeFirstUsed_pipe_pipe_v;
  reg          pipeFirstUsed_pipe_pipe_b;
  wire         pipeFirstUsed_pipe_pipe_out_bits = pipeFirstUsed_pipe_pipe_b;
  reg          pipeFire_pipe_v;
  reg          pipeFire_pipe_b;
  reg          pipeFire_pipe_pipe_v;
  wire         pipeFire_pipe_pipe_out_valid = pipeFire_pipe_pipe_v;
  reg          pipeFire_pipe_pipe_b;
  wire         pipeFire_pipe_pipe_out_bits = pipeFire_pipe_pipe_b;
  wire [31:0]  readResultF_0;
  wire         _readRecord_T_3300 = chainingRecord_0_bits_instIndex == readRequests_1_bits_instructionIndex_0;
  wire         _readRecord_T_3301 = chainingRecord_1_bits_instIndex == readRequests_1_bits_instructionIndex_0;
  wire         _readRecord_T_3302 = chainingRecord_2_bits_instIndex == readRequests_1_bits_instructionIndex_0;
  wire         _readRecord_T_3303 = chainingRecord_3_bits_instIndex == readRequests_1_bits_instructionIndex_0;
  wire         _readRecord_T_3304 = chainingRecord_4_bits_instIndex == readRequests_1_bits_instructionIndex_0;
  wire         readRecord_15_state_wLaneClear =
    _readRecord_T_3300 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_3301 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_3302 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_3303
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_3304 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_15_state_wTopLastReport =
    _readRecord_T_3300 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_3301 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_3302 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_3303
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_3304 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_15_state_wLaneLastReport =
    _readRecord_T_3300 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_3301 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_3302 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_3303
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_3304 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_15_state_wWriteQueueClear =
    _readRecord_T_3300 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_3301 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_3302 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_3303
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_3304 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_15_state_stFinish =
    _readRecord_T_3300 & chainingRecord_0_bits_state_stFinish | _readRecord_T_3301 & chainingRecord_1_bits_state_stFinish | _readRecord_T_3302 & chainingRecord_2_bits_state_stFinish | _readRecord_T_3303
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_3304 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_15_elementMask =
    (_readRecord_T_3300 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_3301 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_3302 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_3303 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_3304 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_15_slow =
    _readRecord_T_3300 & chainingRecord_0_bits_slow | _readRecord_T_3301 & chainingRecord_1_bits_slow | _readRecord_T_3302 & chainingRecord_2_bits_slow | _readRecord_T_3303 & chainingRecord_3_bits_slow | _readRecord_T_3304
    & chainingRecord_4_bits_slow;
  wire         readRecord_15_onlyRead =
    _readRecord_T_3300 & chainingRecord_0_bits_onlyRead | _readRecord_T_3301 & chainingRecord_1_bits_onlyRead | _readRecord_T_3302 & chainingRecord_2_bits_onlyRead | _readRecord_T_3303 & chainingRecord_3_bits_onlyRead | _readRecord_T_3304
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_15_ma =
    _readRecord_T_3300 & chainingRecord_0_bits_ma | _readRecord_T_3301 & chainingRecord_1_bits_ma | _readRecord_T_3302 & chainingRecord_2_bits_ma | _readRecord_T_3303 & chainingRecord_3_bits_ma | _readRecord_T_3304
    & chainingRecord_4_bits_ma;
  wire         readRecord_15_indexType =
    _readRecord_T_3300 & chainingRecord_0_bits_indexType | _readRecord_T_3301 & chainingRecord_1_bits_indexType | _readRecord_T_3302 & chainingRecord_2_bits_indexType | _readRecord_T_3303 & chainingRecord_3_bits_indexType
    | _readRecord_T_3304 & chainingRecord_4_bits_indexType;
  wire         readRecord_15_crossRead =
    _readRecord_T_3300 & chainingRecord_0_bits_crossRead | _readRecord_T_3301 & chainingRecord_1_bits_crossRead | _readRecord_T_3302 & chainingRecord_2_bits_crossRead | _readRecord_T_3303 & chainingRecord_3_bits_crossRead
    | _readRecord_T_3304 & chainingRecord_4_bits_crossRead;
  wire         readRecord_15_crossWrite =
    _readRecord_T_3300 & chainingRecord_0_bits_crossWrite | _readRecord_T_3301 & chainingRecord_1_bits_crossWrite | _readRecord_T_3302 & chainingRecord_2_bits_crossWrite | _readRecord_T_3303 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_3304 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_15_gather16 =
    _readRecord_T_3300 & chainingRecord_0_bits_gather16 | _readRecord_T_3301 & chainingRecord_1_bits_gather16 | _readRecord_T_3302 & chainingRecord_2_bits_gather16 | _readRecord_T_3303 & chainingRecord_3_bits_gather16 | _readRecord_T_3304
    & chainingRecord_4_bits_gather16;
  wire         readRecord_15_gather =
    _readRecord_T_3300 & chainingRecord_0_bits_gather | _readRecord_T_3301 & chainingRecord_1_bits_gather | _readRecord_T_3302 & chainingRecord_2_bits_gather | _readRecord_T_3303 & chainingRecord_3_bits_gather | _readRecord_T_3304
    & chainingRecord_4_bits_gather;
  wire         readRecord_15_st =
    _readRecord_T_3300 & chainingRecord_0_bits_st | _readRecord_T_3301 & chainingRecord_1_bits_st | _readRecord_T_3302 & chainingRecord_2_bits_st | _readRecord_T_3303 & chainingRecord_3_bits_st | _readRecord_T_3304
    & chainingRecord_4_bits_st;
  wire         readRecord_15_ls =
    _readRecord_T_3300 & chainingRecord_0_bits_ls | _readRecord_T_3301 & chainingRecord_1_bits_ls | _readRecord_T_3302 & chainingRecord_2_bits_ls | _readRecord_T_3303 & chainingRecord_3_bits_ls | _readRecord_T_3304
    & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_15_instIndex =
    (_readRecord_T_3300 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_3301 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_3302 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_3303 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_3304 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_15_vs2 =
    (_readRecord_T_3300 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_3301 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_3302 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_3303 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_3304 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_15_vs1_bits =
    (_readRecord_T_3300 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_3301 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_3302 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_3303 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_3304 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_15_vs1_valid =
    _readRecord_T_3300 & chainingRecord_0_bits_vs1_valid | _readRecord_T_3301 & chainingRecord_1_bits_vs1_valid | _readRecord_T_3302 & chainingRecord_2_bits_vs1_valid | _readRecord_T_3303 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_3304 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_15_vd_bits =
    (_readRecord_T_3300 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_3301 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_3302 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_3303 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_3304 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_15_vd_valid =
    _readRecord_T_3300 & chainingRecord_0_bits_vd_valid | _readRecord_T_3301 & chainingRecord_1_bits_vd_valid | _readRecord_T_3302 & chainingRecord_2_bits_vd_valid | _readRecord_T_3303 & chainingRecord_3_bits_vd_valid | _readRecord_T_3304
    & chainingRecord_4_bits_vd_valid;
  wire [9:0]   address_1 = {readRequests_1_bits_vs_0, readRequests_1_bits_offset_0};
  reg          pipeBank_pipe_v_1;
  reg          pipeBank_pipe_pipe_v_1;
  wire         pipeBank_pipe_pipe_out_1_valid = pipeBank_pipe_pipe_v_1;
  reg          pipeBank_pipe_pipe_b_1;
  wire         pipeBank_pipe_pipe_out_1_bits = pipeBank_pipe_pipe_b_1;
  wire         portReady_1;
  assign portReady_1 = ~bankCorrect;
  assign readRequests_1_ready_0 = portReady_1 & sramReady;
  wire         bankReadF_1 = bankCorrect_1 & ~bankCorrect;
  wire         bankReadS_1 = bankCorrect_1 & bankCorrect;
  reg          pipeFirstUsed_pipe_v_1;
  reg          pipeFirstUsed_pipe_b_1;
  reg          pipeFirstUsed_pipe_pipe_v_1;
  wire         pipeFirstUsed_pipe_pipe_out_1_valid = pipeFirstUsed_pipe_pipe_v_1;
  reg          pipeFirstUsed_pipe_pipe_b_1;
  wire         pipeFirstUsed_pipe_pipe_out_1_bits = pipeFirstUsed_pipe_pipe_b_1;
  reg          pipeFire_pipe_v_1;
  reg          pipeFire_pipe_b_1;
  reg          pipeFire_pipe_pipe_v_1;
  wire         pipeFire_pipe_pipe_out_1_valid = pipeFire_pipe_pipe_v_1;
  reg          pipeFire_pipe_pipe_b_1;
  wire         pipeFire_pipe_pipe_out_1_bits = pipeFire_pipe_pipe_b_1;
  wire         firstUsed_2 = bankCorrect | bankCorrect_1;
  wire         _GEN = bankCorrect_1 & bankCorrect;
  wire         _readRecord_T_3520 = chainingRecord_0_bits_instIndex == readRequests_2_bits_instructionIndex_0;
  wire         _readRecord_T_3521 = chainingRecord_1_bits_instIndex == readRequests_2_bits_instructionIndex_0;
  wire         _readRecord_T_3522 = chainingRecord_2_bits_instIndex == readRequests_2_bits_instructionIndex_0;
  wire         _readRecord_T_3523 = chainingRecord_3_bits_instIndex == readRequests_2_bits_instructionIndex_0;
  wire         _readRecord_T_3524 = chainingRecord_4_bits_instIndex == readRequests_2_bits_instructionIndex_0;
  wire         readRecord_16_state_wLaneClear =
    _readRecord_T_3520 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_3521 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_3522 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_3523
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_3524 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_16_state_wTopLastReport =
    _readRecord_T_3520 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_3521 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_3522 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_3523
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_3524 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_16_state_wLaneLastReport =
    _readRecord_T_3520 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_3521 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_3522 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_3523
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_3524 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_16_state_wWriteQueueClear =
    _readRecord_T_3520 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_3521 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_3522 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_3523
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_3524 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_16_state_stFinish =
    _readRecord_T_3520 & chainingRecord_0_bits_state_stFinish | _readRecord_T_3521 & chainingRecord_1_bits_state_stFinish | _readRecord_T_3522 & chainingRecord_2_bits_state_stFinish | _readRecord_T_3523
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_3524 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_16_elementMask =
    (_readRecord_T_3520 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_3521 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_3522 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_3523 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_3524 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_16_slow =
    _readRecord_T_3520 & chainingRecord_0_bits_slow | _readRecord_T_3521 & chainingRecord_1_bits_slow | _readRecord_T_3522 & chainingRecord_2_bits_slow | _readRecord_T_3523 & chainingRecord_3_bits_slow | _readRecord_T_3524
    & chainingRecord_4_bits_slow;
  wire         readRecord_16_onlyRead =
    _readRecord_T_3520 & chainingRecord_0_bits_onlyRead | _readRecord_T_3521 & chainingRecord_1_bits_onlyRead | _readRecord_T_3522 & chainingRecord_2_bits_onlyRead | _readRecord_T_3523 & chainingRecord_3_bits_onlyRead | _readRecord_T_3524
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_16_ma =
    _readRecord_T_3520 & chainingRecord_0_bits_ma | _readRecord_T_3521 & chainingRecord_1_bits_ma | _readRecord_T_3522 & chainingRecord_2_bits_ma | _readRecord_T_3523 & chainingRecord_3_bits_ma | _readRecord_T_3524
    & chainingRecord_4_bits_ma;
  wire         readRecord_16_indexType =
    _readRecord_T_3520 & chainingRecord_0_bits_indexType | _readRecord_T_3521 & chainingRecord_1_bits_indexType | _readRecord_T_3522 & chainingRecord_2_bits_indexType | _readRecord_T_3523 & chainingRecord_3_bits_indexType
    | _readRecord_T_3524 & chainingRecord_4_bits_indexType;
  wire         readRecord_16_crossRead =
    _readRecord_T_3520 & chainingRecord_0_bits_crossRead | _readRecord_T_3521 & chainingRecord_1_bits_crossRead | _readRecord_T_3522 & chainingRecord_2_bits_crossRead | _readRecord_T_3523 & chainingRecord_3_bits_crossRead
    | _readRecord_T_3524 & chainingRecord_4_bits_crossRead;
  wire         readRecord_16_crossWrite =
    _readRecord_T_3520 & chainingRecord_0_bits_crossWrite | _readRecord_T_3521 & chainingRecord_1_bits_crossWrite | _readRecord_T_3522 & chainingRecord_2_bits_crossWrite | _readRecord_T_3523 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_3524 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_16_gather16 =
    _readRecord_T_3520 & chainingRecord_0_bits_gather16 | _readRecord_T_3521 & chainingRecord_1_bits_gather16 | _readRecord_T_3522 & chainingRecord_2_bits_gather16 | _readRecord_T_3523 & chainingRecord_3_bits_gather16 | _readRecord_T_3524
    & chainingRecord_4_bits_gather16;
  wire         readRecord_16_gather =
    _readRecord_T_3520 & chainingRecord_0_bits_gather | _readRecord_T_3521 & chainingRecord_1_bits_gather | _readRecord_T_3522 & chainingRecord_2_bits_gather | _readRecord_T_3523 & chainingRecord_3_bits_gather | _readRecord_T_3524
    & chainingRecord_4_bits_gather;
  wire         readRecord_16_st =
    _readRecord_T_3520 & chainingRecord_0_bits_st | _readRecord_T_3521 & chainingRecord_1_bits_st | _readRecord_T_3522 & chainingRecord_2_bits_st | _readRecord_T_3523 & chainingRecord_3_bits_st | _readRecord_T_3524
    & chainingRecord_4_bits_st;
  wire         readRecord_16_ls =
    _readRecord_T_3520 & chainingRecord_0_bits_ls | _readRecord_T_3521 & chainingRecord_1_bits_ls | _readRecord_T_3522 & chainingRecord_2_bits_ls | _readRecord_T_3523 & chainingRecord_3_bits_ls | _readRecord_T_3524
    & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_16_instIndex =
    (_readRecord_T_3520 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_3521 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_3522 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_3523 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_3524 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_16_vs2 =
    (_readRecord_T_3520 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_3521 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_3522 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_3523 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_3524 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_16_vs1_bits =
    (_readRecord_T_3520 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_3521 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_3522 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_3523 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_3524 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_16_vs1_valid =
    _readRecord_T_3520 & chainingRecord_0_bits_vs1_valid | _readRecord_T_3521 & chainingRecord_1_bits_vs1_valid | _readRecord_T_3522 & chainingRecord_2_bits_vs1_valid | _readRecord_T_3523 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_3524 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_16_vd_bits =
    (_readRecord_T_3520 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_3521 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_3522 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_3523 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_3524 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_16_vd_valid =
    _readRecord_T_3520 & chainingRecord_0_bits_vd_valid | _readRecord_T_3521 & chainingRecord_1_bits_vd_valid | _readRecord_T_3522 & chainingRecord_2_bits_vd_valid | _readRecord_T_3523 & chainingRecord_3_bits_vd_valid | _readRecord_T_3524
    & chainingRecord_4_bits_vd_valid;
  wire [9:0]   address_2 = {readRequests_2_bits_vs_0, readRequests_2_bits_offset_0};
  reg          pipeBank_pipe_v_2;
  reg          pipeBank_pipe_pipe_v_2;
  wire         pipeBank_pipe_pipe_out_2_valid = pipeBank_pipe_pipe_v_2;
  reg          pipeBank_pipe_pipe_b_2;
  wire         pipeBank_pipe_pipe_out_2_bits = pipeBank_pipe_pipe_b_2;
  wire         portReady_2;
  assign portReady_2 = ~firstUsed_2;
  assign readRequests_2_ready_0 = portReady_2 & sramReady;
  wire         bankReadF_2 = bankCorrect_2 & ~firstUsed_2;
  wire         bankReadS_2 = bankCorrect_2 & ~_GEN & firstUsed_2;
  reg          pipeFirstUsed_pipe_v_2;
  reg          pipeFirstUsed_pipe_b_2;
  reg          pipeFirstUsed_pipe_pipe_v_2;
  wire         pipeFirstUsed_pipe_pipe_out_2_valid = pipeFirstUsed_pipe_pipe_v_2;
  reg          pipeFirstUsed_pipe_pipe_b_2;
  wire         pipeFirstUsed_pipe_pipe_out_2_bits = pipeFirstUsed_pipe_pipe_b_2;
  reg          pipeFire_pipe_v_2;
  reg          pipeFire_pipe_b_2;
  reg          pipeFire_pipe_pipe_v_2;
  wire         pipeFire_pipe_pipe_out_2_valid = pipeFire_pipe_pipe_v_2;
  reg          pipeFire_pipe_pipe_b_2;
  wire         pipeFire_pipe_pipe_out_2_bits = pipeFire_pipe_pipe_b_2;
  wire         firstUsed_3 = firstUsed_2 | bankCorrect_2;
  wire         _GEN_0 = bankCorrect_2 & firstUsed_2 | _GEN;
  wire         _readRecord_T_3740 = chainingRecord_0_bits_instIndex == readRequests_3_bits_instructionIndex_0;
  wire         _readRecord_T_3741 = chainingRecord_1_bits_instIndex == readRequests_3_bits_instructionIndex_0;
  wire         _readRecord_T_3742 = chainingRecord_2_bits_instIndex == readRequests_3_bits_instructionIndex_0;
  wire         _readRecord_T_3743 = chainingRecord_3_bits_instIndex == readRequests_3_bits_instructionIndex_0;
  wire         _readRecord_T_3744 = chainingRecord_4_bits_instIndex == readRequests_3_bits_instructionIndex_0;
  wire         readRecord_17_state_wLaneClear =
    _readRecord_T_3740 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_3741 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_3742 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_3743
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_3744 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_17_state_wTopLastReport =
    _readRecord_T_3740 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_3741 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_3742 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_3743
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_3744 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_17_state_wLaneLastReport =
    _readRecord_T_3740 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_3741 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_3742 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_3743
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_3744 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_17_state_wWriteQueueClear =
    _readRecord_T_3740 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_3741 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_3742 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_3743
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_3744 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_17_state_stFinish =
    _readRecord_T_3740 & chainingRecord_0_bits_state_stFinish | _readRecord_T_3741 & chainingRecord_1_bits_state_stFinish | _readRecord_T_3742 & chainingRecord_2_bits_state_stFinish | _readRecord_T_3743
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_3744 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_17_elementMask =
    (_readRecord_T_3740 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_3741 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_3742 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_3743 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_3744 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_17_slow =
    _readRecord_T_3740 & chainingRecord_0_bits_slow | _readRecord_T_3741 & chainingRecord_1_bits_slow | _readRecord_T_3742 & chainingRecord_2_bits_slow | _readRecord_T_3743 & chainingRecord_3_bits_slow | _readRecord_T_3744
    & chainingRecord_4_bits_slow;
  wire         readRecord_17_onlyRead =
    _readRecord_T_3740 & chainingRecord_0_bits_onlyRead | _readRecord_T_3741 & chainingRecord_1_bits_onlyRead | _readRecord_T_3742 & chainingRecord_2_bits_onlyRead | _readRecord_T_3743 & chainingRecord_3_bits_onlyRead | _readRecord_T_3744
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_17_ma =
    _readRecord_T_3740 & chainingRecord_0_bits_ma | _readRecord_T_3741 & chainingRecord_1_bits_ma | _readRecord_T_3742 & chainingRecord_2_bits_ma | _readRecord_T_3743 & chainingRecord_3_bits_ma | _readRecord_T_3744
    & chainingRecord_4_bits_ma;
  wire         readRecord_17_indexType =
    _readRecord_T_3740 & chainingRecord_0_bits_indexType | _readRecord_T_3741 & chainingRecord_1_bits_indexType | _readRecord_T_3742 & chainingRecord_2_bits_indexType | _readRecord_T_3743 & chainingRecord_3_bits_indexType
    | _readRecord_T_3744 & chainingRecord_4_bits_indexType;
  wire         readRecord_17_crossRead =
    _readRecord_T_3740 & chainingRecord_0_bits_crossRead | _readRecord_T_3741 & chainingRecord_1_bits_crossRead | _readRecord_T_3742 & chainingRecord_2_bits_crossRead | _readRecord_T_3743 & chainingRecord_3_bits_crossRead
    | _readRecord_T_3744 & chainingRecord_4_bits_crossRead;
  wire         readRecord_17_crossWrite =
    _readRecord_T_3740 & chainingRecord_0_bits_crossWrite | _readRecord_T_3741 & chainingRecord_1_bits_crossWrite | _readRecord_T_3742 & chainingRecord_2_bits_crossWrite | _readRecord_T_3743 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_3744 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_17_gather16 =
    _readRecord_T_3740 & chainingRecord_0_bits_gather16 | _readRecord_T_3741 & chainingRecord_1_bits_gather16 | _readRecord_T_3742 & chainingRecord_2_bits_gather16 | _readRecord_T_3743 & chainingRecord_3_bits_gather16 | _readRecord_T_3744
    & chainingRecord_4_bits_gather16;
  wire         readRecord_17_gather =
    _readRecord_T_3740 & chainingRecord_0_bits_gather | _readRecord_T_3741 & chainingRecord_1_bits_gather | _readRecord_T_3742 & chainingRecord_2_bits_gather | _readRecord_T_3743 & chainingRecord_3_bits_gather | _readRecord_T_3744
    & chainingRecord_4_bits_gather;
  wire         readRecord_17_st =
    _readRecord_T_3740 & chainingRecord_0_bits_st | _readRecord_T_3741 & chainingRecord_1_bits_st | _readRecord_T_3742 & chainingRecord_2_bits_st | _readRecord_T_3743 & chainingRecord_3_bits_st | _readRecord_T_3744
    & chainingRecord_4_bits_st;
  wire         readRecord_17_ls =
    _readRecord_T_3740 & chainingRecord_0_bits_ls | _readRecord_T_3741 & chainingRecord_1_bits_ls | _readRecord_T_3742 & chainingRecord_2_bits_ls | _readRecord_T_3743 & chainingRecord_3_bits_ls | _readRecord_T_3744
    & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_17_instIndex =
    (_readRecord_T_3740 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_3741 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_3742 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_3743 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_3744 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_17_vs2 =
    (_readRecord_T_3740 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_3741 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_3742 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_3743 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_3744 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_17_vs1_bits =
    (_readRecord_T_3740 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_3741 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_3742 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_3743 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_3744 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_17_vs1_valid =
    _readRecord_T_3740 & chainingRecord_0_bits_vs1_valid | _readRecord_T_3741 & chainingRecord_1_bits_vs1_valid | _readRecord_T_3742 & chainingRecord_2_bits_vs1_valid | _readRecord_T_3743 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_3744 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_17_vd_bits =
    (_readRecord_T_3740 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_3741 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_3742 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_3743 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_3744 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_17_vd_valid =
    _readRecord_T_3740 & chainingRecord_0_bits_vd_valid | _readRecord_T_3741 & chainingRecord_1_bits_vd_valid | _readRecord_T_3742 & chainingRecord_2_bits_vd_valid | _readRecord_T_3743 & chainingRecord_3_bits_vd_valid | _readRecord_T_3744
    & chainingRecord_4_bits_vd_valid;
  wire [9:0]   address_3 = {readRequests_3_bits_vs_0, readRequests_3_bits_offset_0};
  reg          pipeBank_pipe_v_3;
  reg          pipeBank_pipe_pipe_v_3;
  wire         pipeBank_pipe_pipe_out_3_valid = pipeBank_pipe_pipe_v_3;
  reg          pipeBank_pipe_pipe_b_3;
  wire         pipeBank_pipe_pipe_out_3_bits = pipeBank_pipe_pipe_b_3;
  wire         portReady_3;
  assign portReady_3 = ~firstUsed_3;
  assign readRequests_3_ready_0 = portReady_3 & sramReady;
  wire         bankReadF_3 = bankCorrect_3 & ~firstUsed_3;
  wire         bankReadS_3 = bankCorrect_3 & ~_GEN_0 & firstUsed_3;
  reg          pipeFirstUsed_pipe_v_3;
  reg          pipeFirstUsed_pipe_b_3;
  reg          pipeFirstUsed_pipe_pipe_v_3;
  wire         pipeFirstUsed_pipe_pipe_out_3_valid = pipeFirstUsed_pipe_pipe_v_3;
  reg          pipeFirstUsed_pipe_pipe_b_3;
  wire         pipeFirstUsed_pipe_pipe_out_3_bits = pipeFirstUsed_pipe_pipe_b_3;
  reg          pipeFire_pipe_v_3;
  reg          pipeFire_pipe_b_3;
  reg          pipeFire_pipe_pipe_v_3;
  wire         pipeFire_pipe_pipe_out_3_valid = pipeFire_pipe_pipe_v_3;
  reg          pipeFire_pipe_pipe_b_3;
  wire         pipeFire_pipe_pipe_out_3_bits = pipeFire_pipe_pipe_b_3;
  wire         firstUsed_4 = firstUsed_3 | bankCorrect_3;
  wire         _GEN_1 = bankCorrect_3 & firstUsed_3 | _GEN_0;
  wire         _readRecord_T_3960 = chainingRecord_0_bits_instIndex == readRequests_4_bits_instructionIndex_0;
  wire         _readRecord_T_3961 = chainingRecord_1_bits_instIndex == readRequests_4_bits_instructionIndex_0;
  wire         _readRecord_T_3962 = chainingRecord_2_bits_instIndex == readRequests_4_bits_instructionIndex_0;
  wire         _readRecord_T_3963 = chainingRecord_3_bits_instIndex == readRequests_4_bits_instructionIndex_0;
  wire         _readRecord_T_3964 = chainingRecord_4_bits_instIndex == readRequests_4_bits_instructionIndex_0;
  wire         readRecord_18_state_wLaneClear =
    _readRecord_T_3960 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_3961 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_3962 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_3963
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_3964 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_18_state_wTopLastReport =
    _readRecord_T_3960 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_3961 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_3962 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_3963
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_3964 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_18_state_wLaneLastReport =
    _readRecord_T_3960 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_3961 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_3962 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_3963
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_3964 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_18_state_wWriteQueueClear =
    _readRecord_T_3960 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_3961 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_3962 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_3963
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_3964 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_18_state_stFinish =
    _readRecord_T_3960 & chainingRecord_0_bits_state_stFinish | _readRecord_T_3961 & chainingRecord_1_bits_state_stFinish | _readRecord_T_3962 & chainingRecord_2_bits_state_stFinish | _readRecord_T_3963
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_3964 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_18_elementMask =
    (_readRecord_T_3960 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_3961 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_3962 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_3963 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_3964 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_18_slow =
    _readRecord_T_3960 & chainingRecord_0_bits_slow | _readRecord_T_3961 & chainingRecord_1_bits_slow | _readRecord_T_3962 & chainingRecord_2_bits_slow | _readRecord_T_3963 & chainingRecord_3_bits_slow | _readRecord_T_3964
    & chainingRecord_4_bits_slow;
  wire         readRecord_18_onlyRead =
    _readRecord_T_3960 & chainingRecord_0_bits_onlyRead | _readRecord_T_3961 & chainingRecord_1_bits_onlyRead | _readRecord_T_3962 & chainingRecord_2_bits_onlyRead | _readRecord_T_3963 & chainingRecord_3_bits_onlyRead | _readRecord_T_3964
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_18_ma =
    _readRecord_T_3960 & chainingRecord_0_bits_ma | _readRecord_T_3961 & chainingRecord_1_bits_ma | _readRecord_T_3962 & chainingRecord_2_bits_ma | _readRecord_T_3963 & chainingRecord_3_bits_ma | _readRecord_T_3964
    & chainingRecord_4_bits_ma;
  wire         readRecord_18_indexType =
    _readRecord_T_3960 & chainingRecord_0_bits_indexType | _readRecord_T_3961 & chainingRecord_1_bits_indexType | _readRecord_T_3962 & chainingRecord_2_bits_indexType | _readRecord_T_3963 & chainingRecord_3_bits_indexType
    | _readRecord_T_3964 & chainingRecord_4_bits_indexType;
  wire         readRecord_18_crossRead =
    _readRecord_T_3960 & chainingRecord_0_bits_crossRead | _readRecord_T_3961 & chainingRecord_1_bits_crossRead | _readRecord_T_3962 & chainingRecord_2_bits_crossRead | _readRecord_T_3963 & chainingRecord_3_bits_crossRead
    | _readRecord_T_3964 & chainingRecord_4_bits_crossRead;
  wire         readRecord_18_crossWrite =
    _readRecord_T_3960 & chainingRecord_0_bits_crossWrite | _readRecord_T_3961 & chainingRecord_1_bits_crossWrite | _readRecord_T_3962 & chainingRecord_2_bits_crossWrite | _readRecord_T_3963 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_3964 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_18_gather16 =
    _readRecord_T_3960 & chainingRecord_0_bits_gather16 | _readRecord_T_3961 & chainingRecord_1_bits_gather16 | _readRecord_T_3962 & chainingRecord_2_bits_gather16 | _readRecord_T_3963 & chainingRecord_3_bits_gather16 | _readRecord_T_3964
    & chainingRecord_4_bits_gather16;
  wire         readRecord_18_gather =
    _readRecord_T_3960 & chainingRecord_0_bits_gather | _readRecord_T_3961 & chainingRecord_1_bits_gather | _readRecord_T_3962 & chainingRecord_2_bits_gather | _readRecord_T_3963 & chainingRecord_3_bits_gather | _readRecord_T_3964
    & chainingRecord_4_bits_gather;
  wire         readRecord_18_st =
    _readRecord_T_3960 & chainingRecord_0_bits_st | _readRecord_T_3961 & chainingRecord_1_bits_st | _readRecord_T_3962 & chainingRecord_2_bits_st | _readRecord_T_3963 & chainingRecord_3_bits_st | _readRecord_T_3964
    & chainingRecord_4_bits_st;
  wire         readRecord_18_ls =
    _readRecord_T_3960 & chainingRecord_0_bits_ls | _readRecord_T_3961 & chainingRecord_1_bits_ls | _readRecord_T_3962 & chainingRecord_2_bits_ls | _readRecord_T_3963 & chainingRecord_3_bits_ls | _readRecord_T_3964
    & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_18_instIndex =
    (_readRecord_T_3960 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_3961 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_3962 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_3963 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_3964 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_18_vs2 =
    (_readRecord_T_3960 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_3961 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_3962 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_3963 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_3964 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_18_vs1_bits =
    (_readRecord_T_3960 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_3961 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_3962 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_3963 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_3964 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_18_vs1_valid =
    _readRecord_T_3960 & chainingRecord_0_bits_vs1_valid | _readRecord_T_3961 & chainingRecord_1_bits_vs1_valid | _readRecord_T_3962 & chainingRecord_2_bits_vs1_valid | _readRecord_T_3963 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_3964 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_18_vd_bits =
    (_readRecord_T_3960 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_3961 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_3962 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_3963 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_3964 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_18_vd_valid =
    _readRecord_T_3960 & chainingRecord_0_bits_vd_valid | _readRecord_T_3961 & chainingRecord_1_bits_vd_valid | _readRecord_T_3962 & chainingRecord_2_bits_vd_valid | _readRecord_T_3963 & chainingRecord_3_bits_vd_valid | _readRecord_T_3964
    & chainingRecord_4_bits_vd_valid;
  wire [9:0]   address_4 = {readRequests_4_bits_vs_0, readRequests_4_bits_offset_0};
  reg          pipeBank_pipe_v_4;
  reg          pipeBank_pipe_pipe_v_4;
  wire         pipeBank_pipe_pipe_out_4_valid = pipeBank_pipe_pipe_v_4;
  reg          pipeBank_pipe_pipe_b_4;
  wire         pipeBank_pipe_pipe_out_4_bits = pipeBank_pipe_pipe_b_4;
  wire         portReady_4;
  assign portReady_4 = ~firstUsed_4;
  assign readRequests_4_ready_0 = portReady_4 & sramReady;
  wire         bankReadF_4 = bankCorrect_4 & ~firstUsed_4;
  wire         bankReadS_4 = bankCorrect_4 & ~_GEN_1 & firstUsed_4;
  reg          pipeFirstUsed_pipe_v_4;
  reg          pipeFirstUsed_pipe_b_4;
  reg          pipeFirstUsed_pipe_pipe_v_4;
  wire         pipeFirstUsed_pipe_pipe_out_4_valid = pipeFirstUsed_pipe_pipe_v_4;
  reg          pipeFirstUsed_pipe_pipe_b_4;
  wire         pipeFirstUsed_pipe_pipe_out_4_bits = pipeFirstUsed_pipe_pipe_b_4;
  reg          pipeFire_pipe_v_4;
  reg          pipeFire_pipe_b_4;
  reg          pipeFire_pipe_pipe_v_4;
  wire         pipeFire_pipe_pipe_out_4_valid = pipeFire_pipe_pipe_v_4;
  reg          pipeFire_pipe_pipe_b_4;
  wire         pipeFire_pipe_pipe_out_4_bits = pipeFire_pipe_pipe_b_4;
  wire         firstUsed_5 = firstUsed_4 | bankCorrect_4;
  wire         _GEN_2 = bankCorrect_4 & firstUsed_4 | _GEN_1;
  wire         _readRecord_T_4180 = chainingRecord_0_bits_instIndex == readRequests_5_bits_instructionIndex_0;
  wire         _readRecord_T_4181 = chainingRecord_1_bits_instIndex == readRequests_5_bits_instructionIndex_0;
  wire         _readRecord_T_4182 = chainingRecord_2_bits_instIndex == readRequests_5_bits_instructionIndex_0;
  wire         _readRecord_T_4183 = chainingRecord_3_bits_instIndex == readRequests_5_bits_instructionIndex_0;
  wire         _readRecord_T_4184 = chainingRecord_4_bits_instIndex == readRequests_5_bits_instructionIndex_0;
  wire         readRecord_19_state_wLaneClear =
    _readRecord_T_4180 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_4181 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_4182 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_4183
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_4184 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_19_state_wTopLastReport =
    _readRecord_T_4180 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_4181 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_4182 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_4183
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_4184 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_19_state_wLaneLastReport =
    _readRecord_T_4180 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_4181 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_4182 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_4183
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_4184 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_19_state_wWriteQueueClear =
    _readRecord_T_4180 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_4181 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_4182 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_4183
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_4184 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_19_state_stFinish =
    _readRecord_T_4180 & chainingRecord_0_bits_state_stFinish | _readRecord_T_4181 & chainingRecord_1_bits_state_stFinish | _readRecord_T_4182 & chainingRecord_2_bits_state_stFinish | _readRecord_T_4183
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_4184 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_19_elementMask =
    (_readRecord_T_4180 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_4181 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_4182 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_4183 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_4184 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_19_slow =
    _readRecord_T_4180 & chainingRecord_0_bits_slow | _readRecord_T_4181 & chainingRecord_1_bits_slow | _readRecord_T_4182 & chainingRecord_2_bits_slow | _readRecord_T_4183 & chainingRecord_3_bits_slow | _readRecord_T_4184
    & chainingRecord_4_bits_slow;
  wire         readRecord_19_onlyRead =
    _readRecord_T_4180 & chainingRecord_0_bits_onlyRead | _readRecord_T_4181 & chainingRecord_1_bits_onlyRead | _readRecord_T_4182 & chainingRecord_2_bits_onlyRead | _readRecord_T_4183 & chainingRecord_3_bits_onlyRead | _readRecord_T_4184
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_19_ma =
    _readRecord_T_4180 & chainingRecord_0_bits_ma | _readRecord_T_4181 & chainingRecord_1_bits_ma | _readRecord_T_4182 & chainingRecord_2_bits_ma | _readRecord_T_4183 & chainingRecord_3_bits_ma | _readRecord_T_4184
    & chainingRecord_4_bits_ma;
  wire         readRecord_19_indexType =
    _readRecord_T_4180 & chainingRecord_0_bits_indexType | _readRecord_T_4181 & chainingRecord_1_bits_indexType | _readRecord_T_4182 & chainingRecord_2_bits_indexType | _readRecord_T_4183 & chainingRecord_3_bits_indexType
    | _readRecord_T_4184 & chainingRecord_4_bits_indexType;
  wire         readRecord_19_crossRead =
    _readRecord_T_4180 & chainingRecord_0_bits_crossRead | _readRecord_T_4181 & chainingRecord_1_bits_crossRead | _readRecord_T_4182 & chainingRecord_2_bits_crossRead | _readRecord_T_4183 & chainingRecord_3_bits_crossRead
    | _readRecord_T_4184 & chainingRecord_4_bits_crossRead;
  wire         readRecord_19_crossWrite =
    _readRecord_T_4180 & chainingRecord_0_bits_crossWrite | _readRecord_T_4181 & chainingRecord_1_bits_crossWrite | _readRecord_T_4182 & chainingRecord_2_bits_crossWrite | _readRecord_T_4183 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_4184 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_19_gather16 =
    _readRecord_T_4180 & chainingRecord_0_bits_gather16 | _readRecord_T_4181 & chainingRecord_1_bits_gather16 | _readRecord_T_4182 & chainingRecord_2_bits_gather16 | _readRecord_T_4183 & chainingRecord_3_bits_gather16 | _readRecord_T_4184
    & chainingRecord_4_bits_gather16;
  wire         readRecord_19_gather =
    _readRecord_T_4180 & chainingRecord_0_bits_gather | _readRecord_T_4181 & chainingRecord_1_bits_gather | _readRecord_T_4182 & chainingRecord_2_bits_gather | _readRecord_T_4183 & chainingRecord_3_bits_gather | _readRecord_T_4184
    & chainingRecord_4_bits_gather;
  wire         readRecord_19_st =
    _readRecord_T_4180 & chainingRecord_0_bits_st | _readRecord_T_4181 & chainingRecord_1_bits_st | _readRecord_T_4182 & chainingRecord_2_bits_st | _readRecord_T_4183 & chainingRecord_3_bits_st | _readRecord_T_4184
    & chainingRecord_4_bits_st;
  wire         readRecord_19_ls =
    _readRecord_T_4180 & chainingRecord_0_bits_ls | _readRecord_T_4181 & chainingRecord_1_bits_ls | _readRecord_T_4182 & chainingRecord_2_bits_ls | _readRecord_T_4183 & chainingRecord_3_bits_ls | _readRecord_T_4184
    & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_19_instIndex =
    (_readRecord_T_4180 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_4181 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_4182 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_4183 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_4184 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_19_vs2 =
    (_readRecord_T_4180 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_4181 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_4182 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_4183 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_4184 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_19_vs1_bits =
    (_readRecord_T_4180 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_4181 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_4182 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_4183 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_4184 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_19_vs1_valid =
    _readRecord_T_4180 & chainingRecord_0_bits_vs1_valid | _readRecord_T_4181 & chainingRecord_1_bits_vs1_valid | _readRecord_T_4182 & chainingRecord_2_bits_vs1_valid | _readRecord_T_4183 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_4184 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_19_vd_bits =
    (_readRecord_T_4180 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_4181 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_4182 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_4183 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_4184 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_19_vd_valid =
    _readRecord_T_4180 & chainingRecord_0_bits_vd_valid | _readRecord_T_4181 & chainingRecord_1_bits_vd_valid | _readRecord_T_4182 & chainingRecord_2_bits_vd_valid | _readRecord_T_4183 & chainingRecord_3_bits_vd_valid | _readRecord_T_4184
    & chainingRecord_4_bits_vd_valid;
  wire [9:0]   address_5 = {readRequests_5_bits_vs_0, readRequests_5_bits_offset_0};
  reg          pipeBank_pipe_v_5;
  reg          pipeBank_pipe_pipe_v_5;
  wire         pipeBank_pipe_pipe_out_5_valid = pipeBank_pipe_pipe_v_5;
  reg          pipeBank_pipe_pipe_b_5;
  wire         pipeBank_pipe_pipe_out_5_bits = pipeBank_pipe_pipe_b_5;
  wire         portReady_5;
  assign portReady_5 = ~firstUsed_5;
  assign readRequests_5_ready_0 = portReady_5 & sramReady;
  wire         bankReadF_5 = bankCorrect_5 & ~firstUsed_5;
  wire         bankReadS_5 = bankCorrect_5 & ~_GEN_2 & firstUsed_5;
  reg          pipeFirstUsed_pipe_v_5;
  reg          pipeFirstUsed_pipe_b_5;
  reg          pipeFirstUsed_pipe_pipe_v_5;
  wire         pipeFirstUsed_pipe_pipe_out_5_valid = pipeFirstUsed_pipe_pipe_v_5;
  reg          pipeFirstUsed_pipe_pipe_b_5;
  wire         pipeFirstUsed_pipe_pipe_out_5_bits = pipeFirstUsed_pipe_pipe_b_5;
  reg          pipeFire_pipe_v_5;
  reg          pipeFire_pipe_b_5;
  reg          pipeFire_pipe_pipe_v_5;
  wire         pipeFire_pipe_pipe_out_5_valid = pipeFire_pipe_pipe_v_5;
  reg          pipeFire_pipe_pipe_b_5;
  wire         pipeFire_pipe_pipe_out_5_bits = pipeFire_pipe_pipe_b_5;
  wire         firstUsed_6 = firstUsed_5 | bankCorrect_5;
  wire         _GEN_3 = bankCorrect_5 & firstUsed_5 | _GEN_2;
  wire         _readRecord_T_4400 = chainingRecord_0_bits_instIndex == readRequests_6_bits_instructionIndex_0;
  wire         _readRecord_T_4401 = chainingRecord_1_bits_instIndex == readRequests_6_bits_instructionIndex_0;
  wire         _readRecord_T_4402 = chainingRecord_2_bits_instIndex == readRequests_6_bits_instructionIndex_0;
  wire         _readRecord_T_4403 = chainingRecord_3_bits_instIndex == readRequests_6_bits_instructionIndex_0;
  wire         _readRecord_T_4404 = chainingRecord_4_bits_instIndex == readRequests_6_bits_instructionIndex_0;
  wire         readRecord_20_state_wLaneClear =
    _readRecord_T_4400 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_4401 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_4402 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_4403
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_4404 & chainingRecord_4_bits_state_wLaneClear;
  wire         readRecord_20_state_wTopLastReport =
    _readRecord_T_4400 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_4401 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_4402 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_4403
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_4404 & chainingRecord_4_bits_state_wTopLastReport;
  wire         readRecord_20_state_wLaneLastReport =
    _readRecord_T_4400 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_4401 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_4402 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_4403
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_4404 & chainingRecord_4_bits_state_wLaneLastReport;
  wire         readRecord_20_state_wWriteQueueClear =
    _readRecord_T_4400 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_4401 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_4402 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_4403
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_4404 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire         readRecord_20_state_stFinish =
    _readRecord_T_4400 & chainingRecord_0_bits_state_stFinish | _readRecord_T_4401 & chainingRecord_1_bits_state_stFinish | _readRecord_T_4402 & chainingRecord_2_bits_state_stFinish | _readRecord_T_4403
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_4404 & chainingRecord_4_bits_state_stFinish;
  wire [255:0] readRecord_20_elementMask =
    (_readRecord_T_4400 ? chainingRecord_0_bits_elementMask : 256'h0) | (_readRecord_T_4401 ? chainingRecord_1_bits_elementMask : 256'h0) | (_readRecord_T_4402 ? chainingRecord_2_bits_elementMask : 256'h0)
    | (_readRecord_T_4403 ? chainingRecord_3_bits_elementMask : 256'h0) | (_readRecord_T_4404 ? chainingRecord_4_bits_elementMask : 256'h0);
  wire         readRecord_20_slow =
    _readRecord_T_4400 & chainingRecord_0_bits_slow | _readRecord_T_4401 & chainingRecord_1_bits_slow | _readRecord_T_4402 & chainingRecord_2_bits_slow | _readRecord_T_4403 & chainingRecord_3_bits_slow | _readRecord_T_4404
    & chainingRecord_4_bits_slow;
  wire         readRecord_20_onlyRead =
    _readRecord_T_4400 & chainingRecord_0_bits_onlyRead | _readRecord_T_4401 & chainingRecord_1_bits_onlyRead | _readRecord_T_4402 & chainingRecord_2_bits_onlyRead | _readRecord_T_4403 & chainingRecord_3_bits_onlyRead | _readRecord_T_4404
    & chainingRecord_4_bits_onlyRead;
  wire         readRecord_20_ma =
    _readRecord_T_4400 & chainingRecord_0_bits_ma | _readRecord_T_4401 & chainingRecord_1_bits_ma | _readRecord_T_4402 & chainingRecord_2_bits_ma | _readRecord_T_4403 & chainingRecord_3_bits_ma | _readRecord_T_4404
    & chainingRecord_4_bits_ma;
  wire         readRecord_20_indexType =
    _readRecord_T_4400 & chainingRecord_0_bits_indexType | _readRecord_T_4401 & chainingRecord_1_bits_indexType | _readRecord_T_4402 & chainingRecord_2_bits_indexType | _readRecord_T_4403 & chainingRecord_3_bits_indexType
    | _readRecord_T_4404 & chainingRecord_4_bits_indexType;
  wire         readRecord_20_crossRead =
    _readRecord_T_4400 & chainingRecord_0_bits_crossRead | _readRecord_T_4401 & chainingRecord_1_bits_crossRead | _readRecord_T_4402 & chainingRecord_2_bits_crossRead | _readRecord_T_4403 & chainingRecord_3_bits_crossRead
    | _readRecord_T_4404 & chainingRecord_4_bits_crossRead;
  wire         readRecord_20_crossWrite =
    _readRecord_T_4400 & chainingRecord_0_bits_crossWrite | _readRecord_T_4401 & chainingRecord_1_bits_crossWrite | _readRecord_T_4402 & chainingRecord_2_bits_crossWrite | _readRecord_T_4403 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_4404 & chainingRecord_4_bits_crossWrite;
  wire         readRecord_20_gather16 =
    _readRecord_T_4400 & chainingRecord_0_bits_gather16 | _readRecord_T_4401 & chainingRecord_1_bits_gather16 | _readRecord_T_4402 & chainingRecord_2_bits_gather16 | _readRecord_T_4403 & chainingRecord_3_bits_gather16 | _readRecord_T_4404
    & chainingRecord_4_bits_gather16;
  wire         readRecord_20_gather =
    _readRecord_T_4400 & chainingRecord_0_bits_gather | _readRecord_T_4401 & chainingRecord_1_bits_gather | _readRecord_T_4402 & chainingRecord_2_bits_gather | _readRecord_T_4403 & chainingRecord_3_bits_gather | _readRecord_T_4404
    & chainingRecord_4_bits_gather;
  wire         readRecord_20_st =
    _readRecord_T_4400 & chainingRecord_0_bits_st | _readRecord_T_4401 & chainingRecord_1_bits_st | _readRecord_T_4402 & chainingRecord_2_bits_st | _readRecord_T_4403 & chainingRecord_3_bits_st | _readRecord_T_4404
    & chainingRecord_4_bits_st;
  wire         readRecord_20_ls =
    _readRecord_T_4400 & chainingRecord_0_bits_ls | _readRecord_T_4401 & chainingRecord_1_bits_ls | _readRecord_T_4402 & chainingRecord_2_bits_ls | _readRecord_T_4403 & chainingRecord_3_bits_ls | _readRecord_T_4404
    & chainingRecord_4_bits_ls;
  wire [2:0]   readRecord_20_instIndex =
    (_readRecord_T_4400 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_4401 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_4402 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_4403 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_4404 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_20_vs2 =
    (_readRecord_T_4400 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_4401 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_4402 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_4403 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_4404 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_20_vs1_bits =
    (_readRecord_T_4400 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_4401 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_4402 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_4403 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_4404 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire         readRecord_20_vs1_valid =
    _readRecord_T_4400 & chainingRecord_0_bits_vs1_valid | _readRecord_T_4401 & chainingRecord_1_bits_vs1_valid | _readRecord_T_4402 & chainingRecord_2_bits_vs1_valid | _readRecord_T_4403 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_4404 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]   readRecord_20_vd_bits =
    (_readRecord_T_4400 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_4401 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_4402 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_4403 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_4404 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire         readRecord_20_vd_valid =
    _readRecord_T_4400 & chainingRecord_0_bits_vd_valid | _readRecord_T_4401 & chainingRecord_1_bits_vd_valid | _readRecord_T_4402 & chainingRecord_2_bits_vd_valid | _readRecord_T_4403 & chainingRecord_3_bits_vd_valid | _readRecord_T_4404
    & chainingRecord_4_bits_vd_valid;
  wire [9:0]   address_6 = {readRequests_6_bits_vs_0, readRequests_6_bits_offset_0};
  reg          pipeBank_pipe_v_6;
  reg          pipeBank_pipe_pipe_v_6;
  wire         pipeBank_pipe_pipe_out_6_valid = pipeBank_pipe_pipe_v_6;
  reg          pipeBank_pipe_pipe_b_6;
  wire         pipeBank_pipe_pipe_out_6_bits = pipeBank_pipe_pipe_b_6;
  wire         portReady_6;
  assign portReady_6 = ~firstUsed_6;
  assign readRequests_6_ready_0 = portReady_6 & sramReady;
  wire         bankReadF_6 = bankCorrect_6 & ~firstUsed_6;
  wire         bankReadS_6 = bankCorrect_6 & ~_GEN_3 & firstUsed_6;
  reg          pipeFirstUsed_pipe_v_6;
  reg          pipeFirstUsed_pipe_b_6;
  reg          pipeFirstUsed_pipe_pipe_v_6;
  wire         pipeFirstUsed_pipe_pipe_out_6_valid = pipeFirstUsed_pipe_pipe_v_6;
  reg          pipeFirstUsed_pipe_pipe_b_6;
  wire         pipeFirstUsed_pipe_pipe_out_6_bits = pipeFirstUsed_pipe_pipe_b_6;
  reg          pipeFire_pipe_v_6;
  reg          pipeFire_pipe_b_6;
  reg          pipeFire_pipe_pipe_v_6;
  wire         pipeFire_pipe_pipe_out_6_valid = pipeFire_pipe_pipe_v_6;
  reg          pipeFire_pipe_pipe_b_6;
  wire         pipeFire_pipe_pipe_out_6_bits = pipeFire_pipe_pipe_b_6;
  wire         firstUsed_7 = firstUsed_6 | bankCorrect_6;
  wire         _GEN_4 = bankCorrect_6 & firstUsed_6 | _GEN_3;
  wire         _readRecord_T_4620 = chainingRecordCopy_0_bits_instIndex == readRequests_7_bits_instructionIndex_0;
  wire         _readRecord_T_4621 = chainingRecordCopy_1_bits_instIndex == readRequests_7_bits_instructionIndex_0;
  wire         _readRecord_T_4622 = chainingRecordCopy_2_bits_instIndex == readRequests_7_bits_instructionIndex_0;
  wire         _readRecord_T_4623 = chainingRecordCopy_3_bits_instIndex == readRequests_7_bits_instructionIndex_0;
  wire         _readRecord_T_4624 = chainingRecordCopy_4_bits_instIndex == readRequests_7_bits_instructionIndex_0;
  wire         readRecord_21_state_wLaneClear =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_state_wLaneClear | _readRecord_T_4621 & chainingRecordCopy_1_bits_state_wLaneClear | _readRecord_T_4622 & chainingRecordCopy_2_bits_state_wLaneClear | _readRecord_T_4623
    & chainingRecordCopy_3_bits_state_wLaneClear | _readRecord_T_4624 & chainingRecordCopy_4_bits_state_wLaneClear;
  wire         readRecord_21_state_wTopLastReport =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_state_wTopLastReport | _readRecord_T_4621 & chainingRecordCopy_1_bits_state_wTopLastReport | _readRecord_T_4622 & chainingRecordCopy_2_bits_state_wTopLastReport | _readRecord_T_4623
    & chainingRecordCopy_3_bits_state_wTopLastReport | _readRecord_T_4624 & chainingRecordCopy_4_bits_state_wTopLastReport;
  wire         readRecord_21_state_wLaneLastReport =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_state_wLaneLastReport | _readRecord_T_4621 & chainingRecordCopy_1_bits_state_wLaneLastReport | _readRecord_T_4622 & chainingRecordCopy_2_bits_state_wLaneLastReport | _readRecord_T_4623
    & chainingRecordCopy_3_bits_state_wLaneLastReport | _readRecord_T_4624 & chainingRecordCopy_4_bits_state_wLaneLastReport;
  wire         readRecord_21_state_wWriteQueueClear =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_state_wWriteQueueClear | _readRecord_T_4621 & chainingRecordCopy_1_bits_state_wWriteQueueClear | _readRecord_T_4622 & chainingRecordCopy_2_bits_state_wWriteQueueClear | _readRecord_T_4623
    & chainingRecordCopy_3_bits_state_wWriteQueueClear | _readRecord_T_4624 & chainingRecordCopy_4_bits_state_wWriteQueueClear;
  wire         readRecord_21_state_stFinish =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_state_stFinish | _readRecord_T_4621 & chainingRecordCopy_1_bits_state_stFinish | _readRecord_T_4622 & chainingRecordCopy_2_bits_state_stFinish | _readRecord_T_4623
    & chainingRecordCopy_3_bits_state_stFinish | _readRecord_T_4624 & chainingRecordCopy_4_bits_state_stFinish;
  wire [255:0] readRecord_21_elementMask =
    (_readRecord_T_4620 ? chainingRecordCopy_0_bits_elementMask : 256'h0) | (_readRecord_T_4621 ? chainingRecordCopy_1_bits_elementMask : 256'h0) | (_readRecord_T_4622 ? chainingRecordCopy_2_bits_elementMask : 256'h0)
    | (_readRecord_T_4623 ? chainingRecordCopy_3_bits_elementMask : 256'h0) | (_readRecord_T_4624 ? chainingRecordCopy_4_bits_elementMask : 256'h0);
  wire         readRecord_21_slow =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_slow | _readRecord_T_4621 & chainingRecordCopy_1_bits_slow | _readRecord_T_4622 & chainingRecordCopy_2_bits_slow | _readRecord_T_4623 & chainingRecordCopy_3_bits_slow | _readRecord_T_4624
    & chainingRecordCopy_4_bits_slow;
  wire         readRecord_21_onlyRead =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_onlyRead | _readRecord_T_4621 & chainingRecordCopy_1_bits_onlyRead | _readRecord_T_4622 & chainingRecordCopy_2_bits_onlyRead | _readRecord_T_4623 & chainingRecordCopy_3_bits_onlyRead
    | _readRecord_T_4624 & chainingRecordCopy_4_bits_onlyRead;
  wire         readRecord_21_ma =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_ma | _readRecord_T_4621 & chainingRecordCopy_1_bits_ma | _readRecord_T_4622 & chainingRecordCopy_2_bits_ma | _readRecord_T_4623 & chainingRecordCopy_3_bits_ma | _readRecord_T_4624
    & chainingRecordCopy_4_bits_ma;
  wire         readRecord_21_indexType =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_indexType | _readRecord_T_4621 & chainingRecordCopy_1_bits_indexType | _readRecord_T_4622 & chainingRecordCopy_2_bits_indexType | _readRecord_T_4623 & chainingRecordCopy_3_bits_indexType
    | _readRecord_T_4624 & chainingRecordCopy_4_bits_indexType;
  wire         readRecord_21_crossRead =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_crossRead | _readRecord_T_4621 & chainingRecordCopy_1_bits_crossRead | _readRecord_T_4622 & chainingRecordCopy_2_bits_crossRead | _readRecord_T_4623 & chainingRecordCopy_3_bits_crossRead
    | _readRecord_T_4624 & chainingRecordCopy_4_bits_crossRead;
  wire         readRecord_21_crossWrite =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_crossWrite | _readRecord_T_4621 & chainingRecordCopy_1_bits_crossWrite | _readRecord_T_4622 & chainingRecordCopy_2_bits_crossWrite | _readRecord_T_4623
    & chainingRecordCopy_3_bits_crossWrite | _readRecord_T_4624 & chainingRecordCopy_4_bits_crossWrite;
  wire         readRecord_21_gather16 =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_gather16 | _readRecord_T_4621 & chainingRecordCopy_1_bits_gather16 | _readRecord_T_4622 & chainingRecordCopy_2_bits_gather16 | _readRecord_T_4623 & chainingRecordCopy_3_bits_gather16
    | _readRecord_T_4624 & chainingRecordCopy_4_bits_gather16;
  wire         readRecord_21_gather =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_gather | _readRecord_T_4621 & chainingRecordCopy_1_bits_gather | _readRecord_T_4622 & chainingRecordCopy_2_bits_gather | _readRecord_T_4623 & chainingRecordCopy_3_bits_gather
    | _readRecord_T_4624 & chainingRecordCopy_4_bits_gather;
  wire         readRecord_21_st =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_st | _readRecord_T_4621 & chainingRecordCopy_1_bits_st | _readRecord_T_4622 & chainingRecordCopy_2_bits_st | _readRecord_T_4623 & chainingRecordCopy_3_bits_st | _readRecord_T_4624
    & chainingRecordCopy_4_bits_st;
  wire         readRecord_21_ls =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_ls | _readRecord_T_4621 & chainingRecordCopy_1_bits_ls | _readRecord_T_4622 & chainingRecordCopy_2_bits_ls | _readRecord_T_4623 & chainingRecordCopy_3_bits_ls | _readRecord_T_4624
    & chainingRecordCopy_4_bits_ls;
  wire [2:0]   readRecord_21_instIndex =
    (_readRecord_T_4620 ? chainingRecordCopy_0_bits_instIndex : 3'h0) | (_readRecord_T_4621 ? chainingRecordCopy_1_bits_instIndex : 3'h0) | (_readRecord_T_4622 ? chainingRecordCopy_2_bits_instIndex : 3'h0)
    | (_readRecord_T_4623 ? chainingRecordCopy_3_bits_instIndex : 3'h0) | (_readRecord_T_4624 ? chainingRecordCopy_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_21_vs2 =
    (_readRecord_T_4620 ? chainingRecordCopy_0_bits_vs2 : 5'h0) | (_readRecord_T_4621 ? chainingRecordCopy_1_bits_vs2 : 5'h0) | (_readRecord_T_4622 ? chainingRecordCopy_2_bits_vs2 : 5'h0)
    | (_readRecord_T_4623 ? chainingRecordCopy_3_bits_vs2 : 5'h0) | (_readRecord_T_4624 ? chainingRecordCopy_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_21_vs1_bits =
    (_readRecord_T_4620 ? chainingRecordCopy_0_bits_vs1_bits : 5'h0) | (_readRecord_T_4621 ? chainingRecordCopy_1_bits_vs1_bits : 5'h0) | (_readRecord_T_4622 ? chainingRecordCopy_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_4623 ? chainingRecordCopy_3_bits_vs1_bits : 5'h0) | (_readRecord_T_4624 ? chainingRecordCopy_4_bits_vs1_bits : 5'h0);
  wire         readRecord_21_vs1_valid =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_vs1_valid | _readRecord_T_4621 & chainingRecordCopy_1_bits_vs1_valid | _readRecord_T_4622 & chainingRecordCopy_2_bits_vs1_valid | _readRecord_T_4623 & chainingRecordCopy_3_bits_vs1_valid
    | _readRecord_T_4624 & chainingRecordCopy_4_bits_vs1_valid;
  wire [4:0]   readRecord_21_vd_bits =
    (_readRecord_T_4620 ? chainingRecordCopy_0_bits_vd_bits : 5'h0) | (_readRecord_T_4621 ? chainingRecordCopy_1_bits_vd_bits : 5'h0) | (_readRecord_T_4622 ? chainingRecordCopy_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_4623 ? chainingRecordCopy_3_bits_vd_bits : 5'h0) | (_readRecord_T_4624 ? chainingRecordCopy_4_bits_vd_bits : 5'h0);
  wire         readRecord_21_vd_valid =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_vd_valid | _readRecord_T_4621 & chainingRecordCopy_1_bits_vd_valid | _readRecord_T_4622 & chainingRecordCopy_2_bits_vd_valid | _readRecord_T_4623 & chainingRecordCopy_3_bits_vd_valid
    | _readRecord_T_4624 & chainingRecordCopy_4_bits_vd_valid;
  wire [9:0]   address_7 = {readRequests_7_bits_vs_0, readRequests_7_bits_offset_0};
  reg          pipeBank_pipe_v_7;
  reg          pipeBank_pipe_pipe_v_7;
  wire         pipeBank_pipe_pipe_out_7_valid = pipeBank_pipe_pipe_v_7;
  reg          pipeBank_pipe_pipe_b_7;
  wire         pipeBank_pipe_pipe_out_7_bits = pipeBank_pipe_pipe_b_7;
  wire         portReady_7;
  assign portReady_7 = ~firstUsed_7;
  assign readRequests_7_ready_0 = portReady_7 & sramReady;
  wire         bankReadF_7 = bankCorrect_7 & ~firstUsed_7;
  wire         bankReadS_7 = bankCorrect_7 & ~_GEN_4 & firstUsed_7;
  reg          pipeFirstUsed_pipe_v_7;
  reg          pipeFirstUsed_pipe_b_7;
  reg          pipeFirstUsed_pipe_pipe_v_7;
  wire         pipeFirstUsed_pipe_pipe_out_7_valid = pipeFirstUsed_pipe_pipe_v_7;
  reg          pipeFirstUsed_pipe_pipe_b_7;
  wire         pipeFirstUsed_pipe_pipe_out_7_bits = pipeFirstUsed_pipe_pipe_b_7;
  reg          pipeFire_pipe_v_7;
  reg          pipeFire_pipe_b_7;
  reg          pipeFire_pipe_pipe_v_7;
  wire         pipeFire_pipe_pipe_out_7_valid = pipeFire_pipe_pipe_v_7;
  reg          pipeFire_pipe_pipe_b_7;
  wire         pipeFire_pipe_pipe_out_7_bits = pipeFire_pipe_pipe_b_7;
  wire         firstUsed_8 = firstUsed_7 | bankCorrect_7;
  wire         _GEN_5 = bankCorrect_7 & firstUsed_7 | _GEN_4;
  wire         _readRecord_T_4840 = chainingRecordCopy_0_bits_instIndex == readRequests_8_bits_instructionIndex_0;
  wire         _readRecord_T_4841 = chainingRecordCopy_1_bits_instIndex == readRequests_8_bits_instructionIndex_0;
  wire         _readRecord_T_4842 = chainingRecordCopy_2_bits_instIndex == readRequests_8_bits_instructionIndex_0;
  wire         _readRecord_T_4843 = chainingRecordCopy_3_bits_instIndex == readRequests_8_bits_instructionIndex_0;
  wire         _readRecord_T_4844 = chainingRecordCopy_4_bits_instIndex == readRequests_8_bits_instructionIndex_0;
  wire         readRecord_22_state_wLaneClear =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_state_wLaneClear | _readRecord_T_4841 & chainingRecordCopy_1_bits_state_wLaneClear | _readRecord_T_4842 & chainingRecordCopy_2_bits_state_wLaneClear | _readRecord_T_4843
    & chainingRecordCopy_3_bits_state_wLaneClear | _readRecord_T_4844 & chainingRecordCopy_4_bits_state_wLaneClear;
  wire         readRecord_22_state_wTopLastReport =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_state_wTopLastReport | _readRecord_T_4841 & chainingRecordCopy_1_bits_state_wTopLastReport | _readRecord_T_4842 & chainingRecordCopy_2_bits_state_wTopLastReport | _readRecord_T_4843
    & chainingRecordCopy_3_bits_state_wTopLastReport | _readRecord_T_4844 & chainingRecordCopy_4_bits_state_wTopLastReport;
  wire         readRecord_22_state_wLaneLastReport =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_state_wLaneLastReport | _readRecord_T_4841 & chainingRecordCopy_1_bits_state_wLaneLastReport | _readRecord_T_4842 & chainingRecordCopy_2_bits_state_wLaneLastReport | _readRecord_T_4843
    & chainingRecordCopy_3_bits_state_wLaneLastReport | _readRecord_T_4844 & chainingRecordCopy_4_bits_state_wLaneLastReport;
  wire         readRecord_22_state_wWriteQueueClear =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_state_wWriteQueueClear | _readRecord_T_4841 & chainingRecordCopy_1_bits_state_wWriteQueueClear | _readRecord_T_4842 & chainingRecordCopy_2_bits_state_wWriteQueueClear | _readRecord_T_4843
    & chainingRecordCopy_3_bits_state_wWriteQueueClear | _readRecord_T_4844 & chainingRecordCopy_4_bits_state_wWriteQueueClear;
  wire         readRecord_22_state_stFinish =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_state_stFinish | _readRecord_T_4841 & chainingRecordCopy_1_bits_state_stFinish | _readRecord_T_4842 & chainingRecordCopy_2_bits_state_stFinish | _readRecord_T_4843
    & chainingRecordCopy_3_bits_state_stFinish | _readRecord_T_4844 & chainingRecordCopy_4_bits_state_stFinish;
  wire [255:0] readRecord_22_elementMask =
    (_readRecord_T_4840 ? chainingRecordCopy_0_bits_elementMask : 256'h0) | (_readRecord_T_4841 ? chainingRecordCopy_1_bits_elementMask : 256'h0) | (_readRecord_T_4842 ? chainingRecordCopy_2_bits_elementMask : 256'h0)
    | (_readRecord_T_4843 ? chainingRecordCopy_3_bits_elementMask : 256'h0) | (_readRecord_T_4844 ? chainingRecordCopy_4_bits_elementMask : 256'h0);
  wire         readRecord_22_slow =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_slow | _readRecord_T_4841 & chainingRecordCopy_1_bits_slow | _readRecord_T_4842 & chainingRecordCopy_2_bits_slow | _readRecord_T_4843 & chainingRecordCopy_3_bits_slow | _readRecord_T_4844
    & chainingRecordCopy_4_bits_slow;
  wire         readRecord_22_onlyRead =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_onlyRead | _readRecord_T_4841 & chainingRecordCopy_1_bits_onlyRead | _readRecord_T_4842 & chainingRecordCopy_2_bits_onlyRead | _readRecord_T_4843 & chainingRecordCopy_3_bits_onlyRead
    | _readRecord_T_4844 & chainingRecordCopy_4_bits_onlyRead;
  wire         readRecord_22_ma =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_ma | _readRecord_T_4841 & chainingRecordCopy_1_bits_ma | _readRecord_T_4842 & chainingRecordCopy_2_bits_ma | _readRecord_T_4843 & chainingRecordCopy_3_bits_ma | _readRecord_T_4844
    & chainingRecordCopy_4_bits_ma;
  wire         readRecord_22_indexType =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_indexType | _readRecord_T_4841 & chainingRecordCopy_1_bits_indexType | _readRecord_T_4842 & chainingRecordCopy_2_bits_indexType | _readRecord_T_4843 & chainingRecordCopy_3_bits_indexType
    | _readRecord_T_4844 & chainingRecordCopy_4_bits_indexType;
  wire         readRecord_22_crossRead =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_crossRead | _readRecord_T_4841 & chainingRecordCopy_1_bits_crossRead | _readRecord_T_4842 & chainingRecordCopy_2_bits_crossRead | _readRecord_T_4843 & chainingRecordCopy_3_bits_crossRead
    | _readRecord_T_4844 & chainingRecordCopy_4_bits_crossRead;
  wire         readRecord_22_crossWrite =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_crossWrite | _readRecord_T_4841 & chainingRecordCopy_1_bits_crossWrite | _readRecord_T_4842 & chainingRecordCopy_2_bits_crossWrite | _readRecord_T_4843
    & chainingRecordCopy_3_bits_crossWrite | _readRecord_T_4844 & chainingRecordCopy_4_bits_crossWrite;
  wire         readRecord_22_gather16 =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_gather16 | _readRecord_T_4841 & chainingRecordCopy_1_bits_gather16 | _readRecord_T_4842 & chainingRecordCopy_2_bits_gather16 | _readRecord_T_4843 & chainingRecordCopy_3_bits_gather16
    | _readRecord_T_4844 & chainingRecordCopy_4_bits_gather16;
  wire         readRecord_22_gather =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_gather | _readRecord_T_4841 & chainingRecordCopy_1_bits_gather | _readRecord_T_4842 & chainingRecordCopy_2_bits_gather | _readRecord_T_4843 & chainingRecordCopy_3_bits_gather
    | _readRecord_T_4844 & chainingRecordCopy_4_bits_gather;
  wire         readRecord_22_st =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_st | _readRecord_T_4841 & chainingRecordCopy_1_bits_st | _readRecord_T_4842 & chainingRecordCopy_2_bits_st | _readRecord_T_4843 & chainingRecordCopy_3_bits_st | _readRecord_T_4844
    & chainingRecordCopy_4_bits_st;
  wire         readRecord_22_ls =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_ls | _readRecord_T_4841 & chainingRecordCopy_1_bits_ls | _readRecord_T_4842 & chainingRecordCopy_2_bits_ls | _readRecord_T_4843 & chainingRecordCopy_3_bits_ls | _readRecord_T_4844
    & chainingRecordCopy_4_bits_ls;
  wire [2:0]   readRecord_22_instIndex =
    (_readRecord_T_4840 ? chainingRecordCopy_0_bits_instIndex : 3'h0) | (_readRecord_T_4841 ? chainingRecordCopy_1_bits_instIndex : 3'h0) | (_readRecord_T_4842 ? chainingRecordCopy_2_bits_instIndex : 3'h0)
    | (_readRecord_T_4843 ? chainingRecordCopy_3_bits_instIndex : 3'h0) | (_readRecord_T_4844 ? chainingRecordCopy_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_22_vs2 =
    (_readRecord_T_4840 ? chainingRecordCopy_0_bits_vs2 : 5'h0) | (_readRecord_T_4841 ? chainingRecordCopy_1_bits_vs2 : 5'h0) | (_readRecord_T_4842 ? chainingRecordCopy_2_bits_vs2 : 5'h0)
    | (_readRecord_T_4843 ? chainingRecordCopy_3_bits_vs2 : 5'h0) | (_readRecord_T_4844 ? chainingRecordCopy_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_22_vs1_bits =
    (_readRecord_T_4840 ? chainingRecordCopy_0_bits_vs1_bits : 5'h0) | (_readRecord_T_4841 ? chainingRecordCopy_1_bits_vs1_bits : 5'h0) | (_readRecord_T_4842 ? chainingRecordCopy_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_4843 ? chainingRecordCopy_3_bits_vs1_bits : 5'h0) | (_readRecord_T_4844 ? chainingRecordCopy_4_bits_vs1_bits : 5'h0);
  wire         readRecord_22_vs1_valid =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_vs1_valid | _readRecord_T_4841 & chainingRecordCopy_1_bits_vs1_valid | _readRecord_T_4842 & chainingRecordCopy_2_bits_vs1_valid | _readRecord_T_4843 & chainingRecordCopy_3_bits_vs1_valid
    | _readRecord_T_4844 & chainingRecordCopy_4_bits_vs1_valid;
  wire [4:0]   readRecord_22_vd_bits =
    (_readRecord_T_4840 ? chainingRecordCopy_0_bits_vd_bits : 5'h0) | (_readRecord_T_4841 ? chainingRecordCopy_1_bits_vd_bits : 5'h0) | (_readRecord_T_4842 ? chainingRecordCopy_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_4843 ? chainingRecordCopy_3_bits_vd_bits : 5'h0) | (_readRecord_T_4844 ? chainingRecordCopy_4_bits_vd_bits : 5'h0);
  wire         readRecord_22_vd_valid =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_vd_valid | _readRecord_T_4841 & chainingRecordCopy_1_bits_vd_valid | _readRecord_T_4842 & chainingRecordCopy_2_bits_vd_valid | _readRecord_T_4843 & chainingRecordCopy_3_bits_vd_valid
    | _readRecord_T_4844 & chainingRecordCopy_4_bits_vd_valid;
  wire [9:0]   address_8 = {readRequests_8_bits_vs_0, readRequests_8_bits_offset_0};
  reg          pipeBank_pipe_v_8;
  reg          pipeBank_pipe_pipe_v_8;
  wire         pipeBank_pipe_pipe_out_8_valid = pipeBank_pipe_pipe_v_8;
  reg          pipeBank_pipe_pipe_b_8;
  wire         pipeBank_pipe_pipe_out_8_bits = pipeBank_pipe_pipe_b_8;
  wire         portReady_8;
  assign portReady_8 = ~firstUsed_8;
  assign readRequests_8_ready_0 = portReady_8 & sramReady;
  wire         bankReadF_8 = bankCorrect_8 & ~firstUsed_8;
  wire         bankReadS_8 = bankCorrect_8 & ~_GEN_5 & firstUsed_8;
  reg          pipeFirstUsed_pipe_v_8;
  reg          pipeFirstUsed_pipe_b_8;
  reg          pipeFirstUsed_pipe_pipe_v_8;
  wire         pipeFirstUsed_pipe_pipe_out_8_valid = pipeFirstUsed_pipe_pipe_v_8;
  reg          pipeFirstUsed_pipe_pipe_b_8;
  wire         pipeFirstUsed_pipe_pipe_out_8_bits = pipeFirstUsed_pipe_pipe_b_8;
  reg          pipeFire_pipe_v_8;
  reg          pipeFire_pipe_b_8;
  reg          pipeFire_pipe_pipe_v_8;
  wire         pipeFire_pipe_pipe_out_8_valid = pipeFire_pipe_pipe_v_8;
  reg          pipeFire_pipe_pipe_b_8;
  wire         pipeFire_pipe_pipe_out_8_bits = pipeFire_pipe_pipe_b_8;
  wire         firstUsed_9 = firstUsed_8 | bankCorrect_8;
  wire         _GEN_6 = bankCorrect_8 & firstUsed_8 | _GEN_5;
  wire         _readRecord_T_5060 = chainingRecordCopy_0_bits_instIndex == readRequests_9_bits_instructionIndex_0;
  wire         _readRecord_T_5061 = chainingRecordCopy_1_bits_instIndex == readRequests_9_bits_instructionIndex_0;
  wire         _readRecord_T_5062 = chainingRecordCopy_2_bits_instIndex == readRequests_9_bits_instructionIndex_0;
  wire         _readRecord_T_5063 = chainingRecordCopy_3_bits_instIndex == readRequests_9_bits_instructionIndex_0;
  wire         _readRecord_T_5064 = chainingRecordCopy_4_bits_instIndex == readRequests_9_bits_instructionIndex_0;
  wire         readRecord_23_state_wLaneClear =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_state_wLaneClear | _readRecord_T_5061 & chainingRecordCopy_1_bits_state_wLaneClear | _readRecord_T_5062 & chainingRecordCopy_2_bits_state_wLaneClear | _readRecord_T_5063
    & chainingRecordCopy_3_bits_state_wLaneClear | _readRecord_T_5064 & chainingRecordCopy_4_bits_state_wLaneClear;
  wire         readRecord_23_state_wTopLastReport =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_state_wTopLastReport | _readRecord_T_5061 & chainingRecordCopy_1_bits_state_wTopLastReport | _readRecord_T_5062 & chainingRecordCopy_2_bits_state_wTopLastReport | _readRecord_T_5063
    & chainingRecordCopy_3_bits_state_wTopLastReport | _readRecord_T_5064 & chainingRecordCopy_4_bits_state_wTopLastReport;
  wire         readRecord_23_state_wLaneLastReport =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_state_wLaneLastReport | _readRecord_T_5061 & chainingRecordCopy_1_bits_state_wLaneLastReport | _readRecord_T_5062 & chainingRecordCopy_2_bits_state_wLaneLastReport | _readRecord_T_5063
    & chainingRecordCopy_3_bits_state_wLaneLastReport | _readRecord_T_5064 & chainingRecordCopy_4_bits_state_wLaneLastReport;
  wire         readRecord_23_state_wWriteQueueClear =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_state_wWriteQueueClear | _readRecord_T_5061 & chainingRecordCopy_1_bits_state_wWriteQueueClear | _readRecord_T_5062 & chainingRecordCopy_2_bits_state_wWriteQueueClear | _readRecord_T_5063
    & chainingRecordCopy_3_bits_state_wWriteQueueClear | _readRecord_T_5064 & chainingRecordCopy_4_bits_state_wWriteQueueClear;
  wire         readRecord_23_state_stFinish =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_state_stFinish | _readRecord_T_5061 & chainingRecordCopy_1_bits_state_stFinish | _readRecord_T_5062 & chainingRecordCopy_2_bits_state_stFinish | _readRecord_T_5063
    & chainingRecordCopy_3_bits_state_stFinish | _readRecord_T_5064 & chainingRecordCopy_4_bits_state_stFinish;
  wire [255:0] readRecord_23_elementMask =
    (_readRecord_T_5060 ? chainingRecordCopy_0_bits_elementMask : 256'h0) | (_readRecord_T_5061 ? chainingRecordCopy_1_bits_elementMask : 256'h0) | (_readRecord_T_5062 ? chainingRecordCopy_2_bits_elementMask : 256'h0)
    | (_readRecord_T_5063 ? chainingRecordCopy_3_bits_elementMask : 256'h0) | (_readRecord_T_5064 ? chainingRecordCopy_4_bits_elementMask : 256'h0);
  wire         readRecord_23_slow =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_slow | _readRecord_T_5061 & chainingRecordCopy_1_bits_slow | _readRecord_T_5062 & chainingRecordCopy_2_bits_slow | _readRecord_T_5063 & chainingRecordCopy_3_bits_slow | _readRecord_T_5064
    & chainingRecordCopy_4_bits_slow;
  wire         readRecord_23_onlyRead =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_onlyRead | _readRecord_T_5061 & chainingRecordCopy_1_bits_onlyRead | _readRecord_T_5062 & chainingRecordCopy_2_bits_onlyRead | _readRecord_T_5063 & chainingRecordCopy_3_bits_onlyRead
    | _readRecord_T_5064 & chainingRecordCopy_4_bits_onlyRead;
  wire         readRecord_23_ma =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_ma | _readRecord_T_5061 & chainingRecordCopy_1_bits_ma | _readRecord_T_5062 & chainingRecordCopy_2_bits_ma | _readRecord_T_5063 & chainingRecordCopy_3_bits_ma | _readRecord_T_5064
    & chainingRecordCopy_4_bits_ma;
  wire         readRecord_23_indexType =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_indexType | _readRecord_T_5061 & chainingRecordCopy_1_bits_indexType | _readRecord_T_5062 & chainingRecordCopy_2_bits_indexType | _readRecord_T_5063 & chainingRecordCopy_3_bits_indexType
    | _readRecord_T_5064 & chainingRecordCopy_4_bits_indexType;
  wire         readRecord_23_crossRead =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_crossRead | _readRecord_T_5061 & chainingRecordCopy_1_bits_crossRead | _readRecord_T_5062 & chainingRecordCopy_2_bits_crossRead | _readRecord_T_5063 & chainingRecordCopy_3_bits_crossRead
    | _readRecord_T_5064 & chainingRecordCopy_4_bits_crossRead;
  wire         readRecord_23_crossWrite =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_crossWrite | _readRecord_T_5061 & chainingRecordCopy_1_bits_crossWrite | _readRecord_T_5062 & chainingRecordCopy_2_bits_crossWrite | _readRecord_T_5063
    & chainingRecordCopy_3_bits_crossWrite | _readRecord_T_5064 & chainingRecordCopy_4_bits_crossWrite;
  wire         readRecord_23_gather16 =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_gather16 | _readRecord_T_5061 & chainingRecordCopy_1_bits_gather16 | _readRecord_T_5062 & chainingRecordCopy_2_bits_gather16 | _readRecord_T_5063 & chainingRecordCopy_3_bits_gather16
    | _readRecord_T_5064 & chainingRecordCopy_4_bits_gather16;
  wire         readRecord_23_gather =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_gather | _readRecord_T_5061 & chainingRecordCopy_1_bits_gather | _readRecord_T_5062 & chainingRecordCopy_2_bits_gather | _readRecord_T_5063 & chainingRecordCopy_3_bits_gather
    | _readRecord_T_5064 & chainingRecordCopy_4_bits_gather;
  wire         readRecord_23_st =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_st | _readRecord_T_5061 & chainingRecordCopy_1_bits_st | _readRecord_T_5062 & chainingRecordCopy_2_bits_st | _readRecord_T_5063 & chainingRecordCopy_3_bits_st | _readRecord_T_5064
    & chainingRecordCopy_4_bits_st;
  wire         readRecord_23_ls =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_ls | _readRecord_T_5061 & chainingRecordCopy_1_bits_ls | _readRecord_T_5062 & chainingRecordCopy_2_bits_ls | _readRecord_T_5063 & chainingRecordCopy_3_bits_ls | _readRecord_T_5064
    & chainingRecordCopy_4_bits_ls;
  wire [2:0]   readRecord_23_instIndex =
    (_readRecord_T_5060 ? chainingRecordCopy_0_bits_instIndex : 3'h0) | (_readRecord_T_5061 ? chainingRecordCopy_1_bits_instIndex : 3'h0) | (_readRecord_T_5062 ? chainingRecordCopy_2_bits_instIndex : 3'h0)
    | (_readRecord_T_5063 ? chainingRecordCopy_3_bits_instIndex : 3'h0) | (_readRecord_T_5064 ? chainingRecordCopy_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_23_vs2 =
    (_readRecord_T_5060 ? chainingRecordCopy_0_bits_vs2 : 5'h0) | (_readRecord_T_5061 ? chainingRecordCopy_1_bits_vs2 : 5'h0) | (_readRecord_T_5062 ? chainingRecordCopy_2_bits_vs2 : 5'h0)
    | (_readRecord_T_5063 ? chainingRecordCopy_3_bits_vs2 : 5'h0) | (_readRecord_T_5064 ? chainingRecordCopy_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_23_vs1_bits =
    (_readRecord_T_5060 ? chainingRecordCopy_0_bits_vs1_bits : 5'h0) | (_readRecord_T_5061 ? chainingRecordCopy_1_bits_vs1_bits : 5'h0) | (_readRecord_T_5062 ? chainingRecordCopy_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_5063 ? chainingRecordCopy_3_bits_vs1_bits : 5'h0) | (_readRecord_T_5064 ? chainingRecordCopy_4_bits_vs1_bits : 5'h0);
  wire         readRecord_23_vs1_valid =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_vs1_valid | _readRecord_T_5061 & chainingRecordCopy_1_bits_vs1_valid | _readRecord_T_5062 & chainingRecordCopy_2_bits_vs1_valid | _readRecord_T_5063 & chainingRecordCopy_3_bits_vs1_valid
    | _readRecord_T_5064 & chainingRecordCopy_4_bits_vs1_valid;
  wire [4:0]   readRecord_23_vd_bits =
    (_readRecord_T_5060 ? chainingRecordCopy_0_bits_vd_bits : 5'h0) | (_readRecord_T_5061 ? chainingRecordCopy_1_bits_vd_bits : 5'h0) | (_readRecord_T_5062 ? chainingRecordCopy_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_5063 ? chainingRecordCopy_3_bits_vd_bits : 5'h0) | (_readRecord_T_5064 ? chainingRecordCopy_4_bits_vd_bits : 5'h0);
  wire         readRecord_23_vd_valid =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_vd_valid | _readRecord_T_5061 & chainingRecordCopy_1_bits_vd_valid | _readRecord_T_5062 & chainingRecordCopy_2_bits_vd_valid | _readRecord_T_5063 & chainingRecordCopy_3_bits_vd_valid
    | _readRecord_T_5064 & chainingRecordCopy_4_bits_vd_valid;
  wire [9:0]   address_9 = {readRequests_9_bits_vs_0, readRequests_9_bits_offset_0};
  reg          pipeBank_pipe_v_9;
  reg          pipeBank_pipe_pipe_v_9;
  wire         pipeBank_pipe_pipe_out_9_valid = pipeBank_pipe_pipe_v_9;
  reg          pipeBank_pipe_pipe_b_9;
  wire         pipeBank_pipe_pipe_out_9_bits = pipeBank_pipe_pipe_b_9;
  wire         portReady_9;
  assign portReady_9 = ~firstUsed_9;
  assign readRequests_9_ready_0 = portReady_9 & sramReady;
  wire         bankReadF_9 = bankCorrect_9 & ~firstUsed_9;
  wire         bankReadS_9 = bankCorrect_9 & ~_GEN_6 & firstUsed_9;
  reg          pipeFirstUsed_pipe_v_9;
  reg          pipeFirstUsed_pipe_b_9;
  reg          pipeFirstUsed_pipe_pipe_v_9;
  wire         pipeFirstUsed_pipe_pipe_out_9_valid = pipeFirstUsed_pipe_pipe_v_9;
  reg          pipeFirstUsed_pipe_pipe_b_9;
  wire         pipeFirstUsed_pipe_pipe_out_9_bits = pipeFirstUsed_pipe_pipe_b_9;
  reg          pipeFire_pipe_v_9;
  reg          pipeFire_pipe_b_9;
  reg          pipeFire_pipe_pipe_v_9;
  wire         pipeFire_pipe_pipe_out_9_valid = pipeFire_pipe_pipe_v_9;
  reg          pipeFire_pipe_pipe_b_9;
  wire         pipeFire_pipe_pipe_out_9_bits = pipeFire_pipe_pipe_b_9;
  wire         firstUsed_10 = firstUsed_9 | bankCorrect_9;
  wire         _GEN_7 = bankCorrect_9 & firstUsed_9 | _GEN_6;
  wire         _readRecord_T_5280 = chainingRecordCopy_0_bits_instIndex == readRequests_10_bits_instructionIndex_0;
  wire         _readRecord_T_5281 = chainingRecordCopy_1_bits_instIndex == readRequests_10_bits_instructionIndex_0;
  wire         _readRecord_T_5282 = chainingRecordCopy_2_bits_instIndex == readRequests_10_bits_instructionIndex_0;
  wire         _readRecord_T_5283 = chainingRecordCopy_3_bits_instIndex == readRequests_10_bits_instructionIndex_0;
  wire         _readRecord_T_5284 = chainingRecordCopy_4_bits_instIndex == readRequests_10_bits_instructionIndex_0;
  wire         readRecord_24_state_wLaneClear =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_state_wLaneClear | _readRecord_T_5281 & chainingRecordCopy_1_bits_state_wLaneClear | _readRecord_T_5282 & chainingRecordCopy_2_bits_state_wLaneClear | _readRecord_T_5283
    & chainingRecordCopy_3_bits_state_wLaneClear | _readRecord_T_5284 & chainingRecordCopy_4_bits_state_wLaneClear;
  wire         readRecord_24_state_wTopLastReport =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_state_wTopLastReport | _readRecord_T_5281 & chainingRecordCopy_1_bits_state_wTopLastReport | _readRecord_T_5282 & chainingRecordCopy_2_bits_state_wTopLastReport | _readRecord_T_5283
    & chainingRecordCopy_3_bits_state_wTopLastReport | _readRecord_T_5284 & chainingRecordCopy_4_bits_state_wTopLastReport;
  wire         readRecord_24_state_wLaneLastReport =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_state_wLaneLastReport | _readRecord_T_5281 & chainingRecordCopy_1_bits_state_wLaneLastReport | _readRecord_T_5282 & chainingRecordCopy_2_bits_state_wLaneLastReport | _readRecord_T_5283
    & chainingRecordCopy_3_bits_state_wLaneLastReport | _readRecord_T_5284 & chainingRecordCopy_4_bits_state_wLaneLastReport;
  wire         readRecord_24_state_wWriteQueueClear =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_state_wWriteQueueClear | _readRecord_T_5281 & chainingRecordCopy_1_bits_state_wWriteQueueClear | _readRecord_T_5282 & chainingRecordCopy_2_bits_state_wWriteQueueClear | _readRecord_T_5283
    & chainingRecordCopy_3_bits_state_wWriteQueueClear | _readRecord_T_5284 & chainingRecordCopy_4_bits_state_wWriteQueueClear;
  wire         readRecord_24_state_stFinish =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_state_stFinish | _readRecord_T_5281 & chainingRecordCopy_1_bits_state_stFinish | _readRecord_T_5282 & chainingRecordCopy_2_bits_state_stFinish | _readRecord_T_5283
    & chainingRecordCopy_3_bits_state_stFinish | _readRecord_T_5284 & chainingRecordCopy_4_bits_state_stFinish;
  wire [255:0] readRecord_24_elementMask =
    (_readRecord_T_5280 ? chainingRecordCopy_0_bits_elementMask : 256'h0) | (_readRecord_T_5281 ? chainingRecordCopy_1_bits_elementMask : 256'h0) | (_readRecord_T_5282 ? chainingRecordCopy_2_bits_elementMask : 256'h0)
    | (_readRecord_T_5283 ? chainingRecordCopy_3_bits_elementMask : 256'h0) | (_readRecord_T_5284 ? chainingRecordCopy_4_bits_elementMask : 256'h0);
  wire         readRecord_24_slow =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_slow | _readRecord_T_5281 & chainingRecordCopy_1_bits_slow | _readRecord_T_5282 & chainingRecordCopy_2_bits_slow | _readRecord_T_5283 & chainingRecordCopy_3_bits_slow | _readRecord_T_5284
    & chainingRecordCopy_4_bits_slow;
  wire         readRecord_24_onlyRead =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_onlyRead | _readRecord_T_5281 & chainingRecordCopy_1_bits_onlyRead | _readRecord_T_5282 & chainingRecordCopy_2_bits_onlyRead | _readRecord_T_5283 & chainingRecordCopy_3_bits_onlyRead
    | _readRecord_T_5284 & chainingRecordCopy_4_bits_onlyRead;
  wire         readRecord_24_ma =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_ma | _readRecord_T_5281 & chainingRecordCopy_1_bits_ma | _readRecord_T_5282 & chainingRecordCopy_2_bits_ma | _readRecord_T_5283 & chainingRecordCopy_3_bits_ma | _readRecord_T_5284
    & chainingRecordCopy_4_bits_ma;
  wire         readRecord_24_indexType =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_indexType | _readRecord_T_5281 & chainingRecordCopy_1_bits_indexType | _readRecord_T_5282 & chainingRecordCopy_2_bits_indexType | _readRecord_T_5283 & chainingRecordCopy_3_bits_indexType
    | _readRecord_T_5284 & chainingRecordCopy_4_bits_indexType;
  wire         readRecord_24_crossRead =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_crossRead | _readRecord_T_5281 & chainingRecordCopy_1_bits_crossRead | _readRecord_T_5282 & chainingRecordCopy_2_bits_crossRead | _readRecord_T_5283 & chainingRecordCopy_3_bits_crossRead
    | _readRecord_T_5284 & chainingRecordCopy_4_bits_crossRead;
  wire         readRecord_24_crossWrite =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_crossWrite | _readRecord_T_5281 & chainingRecordCopy_1_bits_crossWrite | _readRecord_T_5282 & chainingRecordCopy_2_bits_crossWrite | _readRecord_T_5283
    & chainingRecordCopy_3_bits_crossWrite | _readRecord_T_5284 & chainingRecordCopy_4_bits_crossWrite;
  wire         readRecord_24_gather16 =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_gather16 | _readRecord_T_5281 & chainingRecordCopy_1_bits_gather16 | _readRecord_T_5282 & chainingRecordCopy_2_bits_gather16 | _readRecord_T_5283 & chainingRecordCopy_3_bits_gather16
    | _readRecord_T_5284 & chainingRecordCopy_4_bits_gather16;
  wire         readRecord_24_gather =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_gather | _readRecord_T_5281 & chainingRecordCopy_1_bits_gather | _readRecord_T_5282 & chainingRecordCopy_2_bits_gather | _readRecord_T_5283 & chainingRecordCopy_3_bits_gather
    | _readRecord_T_5284 & chainingRecordCopy_4_bits_gather;
  wire         readRecord_24_st =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_st | _readRecord_T_5281 & chainingRecordCopy_1_bits_st | _readRecord_T_5282 & chainingRecordCopy_2_bits_st | _readRecord_T_5283 & chainingRecordCopy_3_bits_st | _readRecord_T_5284
    & chainingRecordCopy_4_bits_st;
  wire         readRecord_24_ls =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_ls | _readRecord_T_5281 & chainingRecordCopy_1_bits_ls | _readRecord_T_5282 & chainingRecordCopy_2_bits_ls | _readRecord_T_5283 & chainingRecordCopy_3_bits_ls | _readRecord_T_5284
    & chainingRecordCopy_4_bits_ls;
  wire [2:0]   readRecord_24_instIndex =
    (_readRecord_T_5280 ? chainingRecordCopy_0_bits_instIndex : 3'h0) | (_readRecord_T_5281 ? chainingRecordCopy_1_bits_instIndex : 3'h0) | (_readRecord_T_5282 ? chainingRecordCopy_2_bits_instIndex : 3'h0)
    | (_readRecord_T_5283 ? chainingRecordCopy_3_bits_instIndex : 3'h0) | (_readRecord_T_5284 ? chainingRecordCopy_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_24_vs2 =
    (_readRecord_T_5280 ? chainingRecordCopy_0_bits_vs2 : 5'h0) | (_readRecord_T_5281 ? chainingRecordCopy_1_bits_vs2 : 5'h0) | (_readRecord_T_5282 ? chainingRecordCopy_2_bits_vs2 : 5'h0)
    | (_readRecord_T_5283 ? chainingRecordCopy_3_bits_vs2 : 5'h0) | (_readRecord_T_5284 ? chainingRecordCopy_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_24_vs1_bits =
    (_readRecord_T_5280 ? chainingRecordCopy_0_bits_vs1_bits : 5'h0) | (_readRecord_T_5281 ? chainingRecordCopy_1_bits_vs1_bits : 5'h0) | (_readRecord_T_5282 ? chainingRecordCopy_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_5283 ? chainingRecordCopy_3_bits_vs1_bits : 5'h0) | (_readRecord_T_5284 ? chainingRecordCopy_4_bits_vs1_bits : 5'h0);
  wire         readRecord_24_vs1_valid =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_vs1_valid | _readRecord_T_5281 & chainingRecordCopy_1_bits_vs1_valid | _readRecord_T_5282 & chainingRecordCopy_2_bits_vs1_valid | _readRecord_T_5283 & chainingRecordCopy_3_bits_vs1_valid
    | _readRecord_T_5284 & chainingRecordCopy_4_bits_vs1_valid;
  wire [4:0]   readRecord_24_vd_bits =
    (_readRecord_T_5280 ? chainingRecordCopy_0_bits_vd_bits : 5'h0) | (_readRecord_T_5281 ? chainingRecordCopy_1_bits_vd_bits : 5'h0) | (_readRecord_T_5282 ? chainingRecordCopy_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_5283 ? chainingRecordCopy_3_bits_vd_bits : 5'h0) | (_readRecord_T_5284 ? chainingRecordCopy_4_bits_vd_bits : 5'h0);
  wire         readRecord_24_vd_valid =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_vd_valid | _readRecord_T_5281 & chainingRecordCopy_1_bits_vd_valid | _readRecord_T_5282 & chainingRecordCopy_2_bits_vd_valid | _readRecord_T_5283 & chainingRecordCopy_3_bits_vd_valid
    | _readRecord_T_5284 & chainingRecordCopy_4_bits_vd_valid;
  wire [9:0]   address_10 = {readRequests_10_bits_vs_0, readRequests_10_bits_offset_0};
  reg          pipeBank_pipe_v_10;
  reg          pipeBank_pipe_pipe_v_10;
  wire         pipeBank_pipe_pipe_out_10_valid = pipeBank_pipe_pipe_v_10;
  reg          pipeBank_pipe_pipe_b_10;
  wire         pipeBank_pipe_pipe_out_10_bits = pipeBank_pipe_pipe_b_10;
  wire         portReady_10;
  assign portReady_10 = ~firstUsed_10;
  assign readRequests_10_ready_0 = portReady_10 & sramReady;
  wire         bankReadF_10 = bankCorrect_10 & ~firstUsed_10;
  wire         bankReadS_10 = bankCorrect_10 & ~_GEN_7 & firstUsed_10;
  reg          pipeFirstUsed_pipe_v_10;
  reg          pipeFirstUsed_pipe_b_10;
  reg          pipeFirstUsed_pipe_pipe_v_10;
  wire         pipeFirstUsed_pipe_pipe_out_10_valid = pipeFirstUsed_pipe_pipe_v_10;
  reg          pipeFirstUsed_pipe_pipe_b_10;
  wire         pipeFirstUsed_pipe_pipe_out_10_bits = pipeFirstUsed_pipe_pipe_b_10;
  reg          pipeFire_pipe_v_10;
  reg          pipeFire_pipe_b_10;
  reg          pipeFire_pipe_pipe_v_10;
  wire         pipeFire_pipe_pipe_out_10_valid = pipeFire_pipe_pipe_v_10;
  reg          pipeFire_pipe_pipe_b_10;
  wire         pipeFire_pipe_pipe_out_10_bits = pipeFire_pipe_pipe_b_10;
  wire         firstUsed_11 = firstUsed_10 | bankCorrect_10;
  wire         _GEN_8 = bankCorrect_10 & firstUsed_10 | _GEN_7;
  wire         _readRecord_T_5500 = chainingRecordCopy_0_bits_instIndex == readRequests_11_bits_instructionIndex_0;
  wire         _readRecord_T_5501 = chainingRecordCopy_1_bits_instIndex == readRequests_11_bits_instructionIndex_0;
  wire         _readRecord_T_5502 = chainingRecordCopy_2_bits_instIndex == readRequests_11_bits_instructionIndex_0;
  wire         _readRecord_T_5503 = chainingRecordCopy_3_bits_instIndex == readRequests_11_bits_instructionIndex_0;
  wire         _readRecord_T_5504 = chainingRecordCopy_4_bits_instIndex == readRequests_11_bits_instructionIndex_0;
  wire         readRecord_25_state_wLaneClear =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_state_wLaneClear | _readRecord_T_5501 & chainingRecordCopy_1_bits_state_wLaneClear | _readRecord_T_5502 & chainingRecordCopy_2_bits_state_wLaneClear | _readRecord_T_5503
    & chainingRecordCopy_3_bits_state_wLaneClear | _readRecord_T_5504 & chainingRecordCopy_4_bits_state_wLaneClear;
  wire         readRecord_25_state_wTopLastReport =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_state_wTopLastReport | _readRecord_T_5501 & chainingRecordCopy_1_bits_state_wTopLastReport | _readRecord_T_5502 & chainingRecordCopy_2_bits_state_wTopLastReport | _readRecord_T_5503
    & chainingRecordCopy_3_bits_state_wTopLastReport | _readRecord_T_5504 & chainingRecordCopy_4_bits_state_wTopLastReport;
  wire         readRecord_25_state_wLaneLastReport =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_state_wLaneLastReport | _readRecord_T_5501 & chainingRecordCopy_1_bits_state_wLaneLastReport | _readRecord_T_5502 & chainingRecordCopy_2_bits_state_wLaneLastReport | _readRecord_T_5503
    & chainingRecordCopy_3_bits_state_wLaneLastReport | _readRecord_T_5504 & chainingRecordCopy_4_bits_state_wLaneLastReport;
  wire         readRecord_25_state_wWriteQueueClear =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_state_wWriteQueueClear | _readRecord_T_5501 & chainingRecordCopy_1_bits_state_wWriteQueueClear | _readRecord_T_5502 & chainingRecordCopy_2_bits_state_wWriteQueueClear | _readRecord_T_5503
    & chainingRecordCopy_3_bits_state_wWriteQueueClear | _readRecord_T_5504 & chainingRecordCopy_4_bits_state_wWriteQueueClear;
  wire         readRecord_25_state_stFinish =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_state_stFinish | _readRecord_T_5501 & chainingRecordCopy_1_bits_state_stFinish | _readRecord_T_5502 & chainingRecordCopy_2_bits_state_stFinish | _readRecord_T_5503
    & chainingRecordCopy_3_bits_state_stFinish | _readRecord_T_5504 & chainingRecordCopy_4_bits_state_stFinish;
  wire [255:0] readRecord_25_elementMask =
    (_readRecord_T_5500 ? chainingRecordCopy_0_bits_elementMask : 256'h0) | (_readRecord_T_5501 ? chainingRecordCopy_1_bits_elementMask : 256'h0) | (_readRecord_T_5502 ? chainingRecordCopy_2_bits_elementMask : 256'h0)
    | (_readRecord_T_5503 ? chainingRecordCopy_3_bits_elementMask : 256'h0) | (_readRecord_T_5504 ? chainingRecordCopy_4_bits_elementMask : 256'h0);
  wire         readRecord_25_slow =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_slow | _readRecord_T_5501 & chainingRecordCopy_1_bits_slow | _readRecord_T_5502 & chainingRecordCopy_2_bits_slow | _readRecord_T_5503 & chainingRecordCopy_3_bits_slow | _readRecord_T_5504
    & chainingRecordCopy_4_bits_slow;
  wire         readRecord_25_onlyRead =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_onlyRead | _readRecord_T_5501 & chainingRecordCopy_1_bits_onlyRead | _readRecord_T_5502 & chainingRecordCopy_2_bits_onlyRead | _readRecord_T_5503 & chainingRecordCopy_3_bits_onlyRead
    | _readRecord_T_5504 & chainingRecordCopy_4_bits_onlyRead;
  wire         readRecord_25_ma =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_ma | _readRecord_T_5501 & chainingRecordCopy_1_bits_ma | _readRecord_T_5502 & chainingRecordCopy_2_bits_ma | _readRecord_T_5503 & chainingRecordCopy_3_bits_ma | _readRecord_T_5504
    & chainingRecordCopy_4_bits_ma;
  wire         readRecord_25_indexType =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_indexType | _readRecord_T_5501 & chainingRecordCopy_1_bits_indexType | _readRecord_T_5502 & chainingRecordCopy_2_bits_indexType | _readRecord_T_5503 & chainingRecordCopy_3_bits_indexType
    | _readRecord_T_5504 & chainingRecordCopy_4_bits_indexType;
  wire         readRecord_25_crossRead =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_crossRead | _readRecord_T_5501 & chainingRecordCopy_1_bits_crossRead | _readRecord_T_5502 & chainingRecordCopy_2_bits_crossRead | _readRecord_T_5503 & chainingRecordCopy_3_bits_crossRead
    | _readRecord_T_5504 & chainingRecordCopy_4_bits_crossRead;
  wire         readRecord_25_crossWrite =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_crossWrite | _readRecord_T_5501 & chainingRecordCopy_1_bits_crossWrite | _readRecord_T_5502 & chainingRecordCopy_2_bits_crossWrite | _readRecord_T_5503
    & chainingRecordCopy_3_bits_crossWrite | _readRecord_T_5504 & chainingRecordCopy_4_bits_crossWrite;
  wire         readRecord_25_gather16 =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_gather16 | _readRecord_T_5501 & chainingRecordCopy_1_bits_gather16 | _readRecord_T_5502 & chainingRecordCopy_2_bits_gather16 | _readRecord_T_5503 & chainingRecordCopy_3_bits_gather16
    | _readRecord_T_5504 & chainingRecordCopy_4_bits_gather16;
  wire         readRecord_25_gather =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_gather | _readRecord_T_5501 & chainingRecordCopy_1_bits_gather | _readRecord_T_5502 & chainingRecordCopy_2_bits_gather | _readRecord_T_5503 & chainingRecordCopy_3_bits_gather
    | _readRecord_T_5504 & chainingRecordCopy_4_bits_gather;
  wire         readRecord_25_st =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_st | _readRecord_T_5501 & chainingRecordCopy_1_bits_st | _readRecord_T_5502 & chainingRecordCopy_2_bits_st | _readRecord_T_5503 & chainingRecordCopy_3_bits_st | _readRecord_T_5504
    & chainingRecordCopy_4_bits_st;
  wire         readRecord_25_ls =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_ls | _readRecord_T_5501 & chainingRecordCopy_1_bits_ls | _readRecord_T_5502 & chainingRecordCopy_2_bits_ls | _readRecord_T_5503 & chainingRecordCopy_3_bits_ls | _readRecord_T_5504
    & chainingRecordCopy_4_bits_ls;
  wire [2:0]   readRecord_25_instIndex =
    (_readRecord_T_5500 ? chainingRecordCopy_0_bits_instIndex : 3'h0) | (_readRecord_T_5501 ? chainingRecordCopy_1_bits_instIndex : 3'h0) | (_readRecord_T_5502 ? chainingRecordCopy_2_bits_instIndex : 3'h0)
    | (_readRecord_T_5503 ? chainingRecordCopy_3_bits_instIndex : 3'h0) | (_readRecord_T_5504 ? chainingRecordCopy_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_25_vs2 =
    (_readRecord_T_5500 ? chainingRecordCopy_0_bits_vs2 : 5'h0) | (_readRecord_T_5501 ? chainingRecordCopy_1_bits_vs2 : 5'h0) | (_readRecord_T_5502 ? chainingRecordCopy_2_bits_vs2 : 5'h0)
    | (_readRecord_T_5503 ? chainingRecordCopy_3_bits_vs2 : 5'h0) | (_readRecord_T_5504 ? chainingRecordCopy_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_25_vs1_bits =
    (_readRecord_T_5500 ? chainingRecordCopy_0_bits_vs1_bits : 5'h0) | (_readRecord_T_5501 ? chainingRecordCopy_1_bits_vs1_bits : 5'h0) | (_readRecord_T_5502 ? chainingRecordCopy_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_5503 ? chainingRecordCopy_3_bits_vs1_bits : 5'h0) | (_readRecord_T_5504 ? chainingRecordCopy_4_bits_vs1_bits : 5'h0);
  wire         readRecord_25_vs1_valid =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_vs1_valid | _readRecord_T_5501 & chainingRecordCopy_1_bits_vs1_valid | _readRecord_T_5502 & chainingRecordCopy_2_bits_vs1_valid | _readRecord_T_5503 & chainingRecordCopy_3_bits_vs1_valid
    | _readRecord_T_5504 & chainingRecordCopy_4_bits_vs1_valid;
  wire [4:0]   readRecord_25_vd_bits =
    (_readRecord_T_5500 ? chainingRecordCopy_0_bits_vd_bits : 5'h0) | (_readRecord_T_5501 ? chainingRecordCopy_1_bits_vd_bits : 5'h0) | (_readRecord_T_5502 ? chainingRecordCopy_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_5503 ? chainingRecordCopy_3_bits_vd_bits : 5'h0) | (_readRecord_T_5504 ? chainingRecordCopy_4_bits_vd_bits : 5'h0);
  wire         readRecord_25_vd_valid =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_vd_valid | _readRecord_T_5501 & chainingRecordCopy_1_bits_vd_valid | _readRecord_T_5502 & chainingRecordCopy_2_bits_vd_valid | _readRecord_T_5503 & chainingRecordCopy_3_bits_vd_valid
    | _readRecord_T_5504 & chainingRecordCopy_4_bits_vd_valid;
  wire [9:0]   address_11 = {readRequests_11_bits_vs_0, readRequests_11_bits_offset_0};
  reg          pipeBank_pipe_v_11;
  reg          pipeBank_pipe_pipe_v_11;
  wire         pipeBank_pipe_pipe_out_11_valid = pipeBank_pipe_pipe_v_11;
  reg          pipeBank_pipe_pipe_b_11;
  wire         pipeBank_pipe_pipe_out_11_bits = pipeBank_pipe_pipe_b_11;
  wire         portReady_11;
  assign portReady_11 = ~firstUsed_11;
  assign readRequests_11_ready_0 = portReady_11 & sramReady;
  wire         bankReadF_11 = bankCorrect_11 & ~firstUsed_11;
  wire         bankReadS_11 = bankCorrect_11 & ~_GEN_8 & firstUsed_11;
  reg          pipeFirstUsed_pipe_v_11;
  reg          pipeFirstUsed_pipe_b_11;
  reg          pipeFirstUsed_pipe_pipe_v_11;
  wire         pipeFirstUsed_pipe_pipe_out_11_valid = pipeFirstUsed_pipe_pipe_v_11;
  reg          pipeFirstUsed_pipe_pipe_b_11;
  wire         pipeFirstUsed_pipe_pipe_out_11_bits = pipeFirstUsed_pipe_pipe_b_11;
  reg          pipeFire_pipe_v_11;
  reg          pipeFire_pipe_b_11;
  reg          pipeFire_pipe_pipe_v_11;
  wire         pipeFire_pipe_pipe_out_11_valid = pipeFire_pipe_pipe_v_11;
  reg          pipeFire_pipe_pipe_b_11;
  wire         pipeFire_pipe_pipe_out_11_bits = pipeFire_pipe_pipe_b_11;
  wire         firstUsed_12 = firstUsed_11 | bankCorrect_11;
  wire         _GEN_9 = bankCorrect_11 & firstUsed_11 | _GEN_8;
  wire         _readRecord_T_5720 = chainingRecordCopy_0_bits_instIndex == readRequests_12_bits_instructionIndex_0;
  wire         _readRecord_T_5721 = chainingRecordCopy_1_bits_instIndex == readRequests_12_bits_instructionIndex_0;
  wire         _readRecord_T_5722 = chainingRecordCopy_2_bits_instIndex == readRequests_12_bits_instructionIndex_0;
  wire         _readRecord_T_5723 = chainingRecordCopy_3_bits_instIndex == readRequests_12_bits_instructionIndex_0;
  wire         _readRecord_T_5724 = chainingRecordCopy_4_bits_instIndex == readRequests_12_bits_instructionIndex_0;
  wire         readRecord_26_state_wLaneClear =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_state_wLaneClear | _readRecord_T_5721 & chainingRecordCopy_1_bits_state_wLaneClear | _readRecord_T_5722 & chainingRecordCopy_2_bits_state_wLaneClear | _readRecord_T_5723
    & chainingRecordCopy_3_bits_state_wLaneClear | _readRecord_T_5724 & chainingRecordCopy_4_bits_state_wLaneClear;
  wire         readRecord_26_state_wTopLastReport =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_state_wTopLastReport | _readRecord_T_5721 & chainingRecordCopy_1_bits_state_wTopLastReport | _readRecord_T_5722 & chainingRecordCopy_2_bits_state_wTopLastReport | _readRecord_T_5723
    & chainingRecordCopy_3_bits_state_wTopLastReport | _readRecord_T_5724 & chainingRecordCopy_4_bits_state_wTopLastReport;
  wire         readRecord_26_state_wLaneLastReport =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_state_wLaneLastReport | _readRecord_T_5721 & chainingRecordCopy_1_bits_state_wLaneLastReport | _readRecord_T_5722 & chainingRecordCopy_2_bits_state_wLaneLastReport | _readRecord_T_5723
    & chainingRecordCopy_3_bits_state_wLaneLastReport | _readRecord_T_5724 & chainingRecordCopy_4_bits_state_wLaneLastReport;
  wire         readRecord_26_state_wWriteQueueClear =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_state_wWriteQueueClear | _readRecord_T_5721 & chainingRecordCopy_1_bits_state_wWriteQueueClear | _readRecord_T_5722 & chainingRecordCopy_2_bits_state_wWriteQueueClear | _readRecord_T_5723
    & chainingRecordCopy_3_bits_state_wWriteQueueClear | _readRecord_T_5724 & chainingRecordCopy_4_bits_state_wWriteQueueClear;
  wire         readRecord_26_state_stFinish =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_state_stFinish | _readRecord_T_5721 & chainingRecordCopy_1_bits_state_stFinish | _readRecord_T_5722 & chainingRecordCopy_2_bits_state_stFinish | _readRecord_T_5723
    & chainingRecordCopy_3_bits_state_stFinish | _readRecord_T_5724 & chainingRecordCopy_4_bits_state_stFinish;
  wire [255:0] readRecord_26_elementMask =
    (_readRecord_T_5720 ? chainingRecordCopy_0_bits_elementMask : 256'h0) | (_readRecord_T_5721 ? chainingRecordCopy_1_bits_elementMask : 256'h0) | (_readRecord_T_5722 ? chainingRecordCopy_2_bits_elementMask : 256'h0)
    | (_readRecord_T_5723 ? chainingRecordCopy_3_bits_elementMask : 256'h0) | (_readRecord_T_5724 ? chainingRecordCopy_4_bits_elementMask : 256'h0);
  wire         readRecord_26_slow =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_slow | _readRecord_T_5721 & chainingRecordCopy_1_bits_slow | _readRecord_T_5722 & chainingRecordCopy_2_bits_slow | _readRecord_T_5723 & chainingRecordCopy_3_bits_slow | _readRecord_T_5724
    & chainingRecordCopy_4_bits_slow;
  wire         readRecord_26_onlyRead =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_onlyRead | _readRecord_T_5721 & chainingRecordCopy_1_bits_onlyRead | _readRecord_T_5722 & chainingRecordCopy_2_bits_onlyRead | _readRecord_T_5723 & chainingRecordCopy_3_bits_onlyRead
    | _readRecord_T_5724 & chainingRecordCopy_4_bits_onlyRead;
  wire         readRecord_26_ma =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_ma | _readRecord_T_5721 & chainingRecordCopy_1_bits_ma | _readRecord_T_5722 & chainingRecordCopy_2_bits_ma | _readRecord_T_5723 & chainingRecordCopy_3_bits_ma | _readRecord_T_5724
    & chainingRecordCopy_4_bits_ma;
  wire         readRecord_26_indexType =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_indexType | _readRecord_T_5721 & chainingRecordCopy_1_bits_indexType | _readRecord_T_5722 & chainingRecordCopy_2_bits_indexType | _readRecord_T_5723 & chainingRecordCopy_3_bits_indexType
    | _readRecord_T_5724 & chainingRecordCopy_4_bits_indexType;
  wire         readRecord_26_crossRead =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_crossRead | _readRecord_T_5721 & chainingRecordCopy_1_bits_crossRead | _readRecord_T_5722 & chainingRecordCopy_2_bits_crossRead | _readRecord_T_5723 & chainingRecordCopy_3_bits_crossRead
    | _readRecord_T_5724 & chainingRecordCopy_4_bits_crossRead;
  wire         readRecord_26_crossWrite =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_crossWrite | _readRecord_T_5721 & chainingRecordCopy_1_bits_crossWrite | _readRecord_T_5722 & chainingRecordCopy_2_bits_crossWrite | _readRecord_T_5723
    & chainingRecordCopy_3_bits_crossWrite | _readRecord_T_5724 & chainingRecordCopy_4_bits_crossWrite;
  wire         readRecord_26_gather16 =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_gather16 | _readRecord_T_5721 & chainingRecordCopy_1_bits_gather16 | _readRecord_T_5722 & chainingRecordCopy_2_bits_gather16 | _readRecord_T_5723 & chainingRecordCopy_3_bits_gather16
    | _readRecord_T_5724 & chainingRecordCopy_4_bits_gather16;
  wire         readRecord_26_gather =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_gather | _readRecord_T_5721 & chainingRecordCopy_1_bits_gather | _readRecord_T_5722 & chainingRecordCopy_2_bits_gather | _readRecord_T_5723 & chainingRecordCopy_3_bits_gather
    | _readRecord_T_5724 & chainingRecordCopy_4_bits_gather;
  wire         readRecord_26_st =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_st | _readRecord_T_5721 & chainingRecordCopy_1_bits_st | _readRecord_T_5722 & chainingRecordCopy_2_bits_st | _readRecord_T_5723 & chainingRecordCopy_3_bits_st | _readRecord_T_5724
    & chainingRecordCopy_4_bits_st;
  wire         readRecord_26_ls =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_ls | _readRecord_T_5721 & chainingRecordCopy_1_bits_ls | _readRecord_T_5722 & chainingRecordCopy_2_bits_ls | _readRecord_T_5723 & chainingRecordCopy_3_bits_ls | _readRecord_T_5724
    & chainingRecordCopy_4_bits_ls;
  wire [2:0]   readRecord_26_instIndex =
    (_readRecord_T_5720 ? chainingRecordCopy_0_bits_instIndex : 3'h0) | (_readRecord_T_5721 ? chainingRecordCopy_1_bits_instIndex : 3'h0) | (_readRecord_T_5722 ? chainingRecordCopy_2_bits_instIndex : 3'h0)
    | (_readRecord_T_5723 ? chainingRecordCopy_3_bits_instIndex : 3'h0) | (_readRecord_T_5724 ? chainingRecordCopy_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_26_vs2 =
    (_readRecord_T_5720 ? chainingRecordCopy_0_bits_vs2 : 5'h0) | (_readRecord_T_5721 ? chainingRecordCopy_1_bits_vs2 : 5'h0) | (_readRecord_T_5722 ? chainingRecordCopy_2_bits_vs2 : 5'h0)
    | (_readRecord_T_5723 ? chainingRecordCopy_3_bits_vs2 : 5'h0) | (_readRecord_T_5724 ? chainingRecordCopy_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_26_vs1_bits =
    (_readRecord_T_5720 ? chainingRecordCopy_0_bits_vs1_bits : 5'h0) | (_readRecord_T_5721 ? chainingRecordCopy_1_bits_vs1_bits : 5'h0) | (_readRecord_T_5722 ? chainingRecordCopy_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_5723 ? chainingRecordCopy_3_bits_vs1_bits : 5'h0) | (_readRecord_T_5724 ? chainingRecordCopy_4_bits_vs1_bits : 5'h0);
  wire         readRecord_26_vs1_valid =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_vs1_valid | _readRecord_T_5721 & chainingRecordCopy_1_bits_vs1_valid | _readRecord_T_5722 & chainingRecordCopy_2_bits_vs1_valid | _readRecord_T_5723 & chainingRecordCopy_3_bits_vs1_valid
    | _readRecord_T_5724 & chainingRecordCopy_4_bits_vs1_valid;
  wire [4:0]   readRecord_26_vd_bits =
    (_readRecord_T_5720 ? chainingRecordCopy_0_bits_vd_bits : 5'h0) | (_readRecord_T_5721 ? chainingRecordCopy_1_bits_vd_bits : 5'h0) | (_readRecord_T_5722 ? chainingRecordCopy_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_5723 ? chainingRecordCopy_3_bits_vd_bits : 5'h0) | (_readRecord_T_5724 ? chainingRecordCopy_4_bits_vd_bits : 5'h0);
  wire         readRecord_26_vd_valid =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_vd_valid | _readRecord_T_5721 & chainingRecordCopy_1_bits_vd_valid | _readRecord_T_5722 & chainingRecordCopy_2_bits_vd_valid | _readRecord_T_5723 & chainingRecordCopy_3_bits_vd_valid
    | _readRecord_T_5724 & chainingRecordCopy_4_bits_vd_valid;
  wire [9:0]   address_12 = {readRequests_12_bits_vs_0, readRequests_12_bits_offset_0};
  reg          pipeBank_pipe_v_12;
  reg          pipeBank_pipe_pipe_v_12;
  wire         pipeBank_pipe_pipe_out_12_valid = pipeBank_pipe_pipe_v_12;
  reg          pipeBank_pipe_pipe_b_12;
  wire         pipeBank_pipe_pipe_out_12_bits = pipeBank_pipe_pipe_b_12;
  wire         portReady_12;
  assign portReady_12 = ~firstUsed_12;
  assign readRequests_12_ready_0 = portReady_12 & sramReady;
  wire         bankReadF_12 = bankCorrect_12 & ~firstUsed_12;
  wire         bankReadS_12 = bankCorrect_12 & ~_GEN_9 & firstUsed_12;
  reg          pipeFirstUsed_pipe_v_12;
  reg          pipeFirstUsed_pipe_b_12;
  reg          pipeFirstUsed_pipe_pipe_v_12;
  wire         pipeFirstUsed_pipe_pipe_out_12_valid = pipeFirstUsed_pipe_pipe_v_12;
  reg          pipeFirstUsed_pipe_pipe_b_12;
  wire         pipeFirstUsed_pipe_pipe_out_12_bits = pipeFirstUsed_pipe_pipe_b_12;
  reg          pipeFire_pipe_v_12;
  reg          pipeFire_pipe_b_12;
  reg          pipeFire_pipe_pipe_v_12;
  wire         pipeFire_pipe_pipe_out_12_valid = pipeFire_pipe_pipe_v_12;
  reg          pipeFire_pipe_pipe_b_12;
  wire         pipeFire_pipe_pipe_out_12_bits = pipeFire_pipe_pipe_b_12;
  wire         firstUsed_13 = firstUsed_12 | bankCorrect_12;
  wire         _GEN_10 = bankCorrect_12 & firstUsed_12 | _GEN_9;
  wire         _loadUpdateValidVec_T_16 = chainingRecordCopy_0_bits_instIndex == readRequests_13_bits_instructionIndex_0;
  wire         _loadUpdateValidVec_T_19 = chainingRecordCopy_1_bits_instIndex == readRequests_13_bits_instructionIndex_0;
  wire         _loadUpdateValidVec_T_22 = chainingRecordCopy_2_bits_instIndex == readRequests_13_bits_instructionIndex_0;
  wire         _loadUpdateValidVec_T_25 = chainingRecordCopy_3_bits_instIndex == readRequests_13_bits_instructionIndex_0;
  wire         _loadUpdateValidVec_T_28 = chainingRecordCopy_4_bits_instIndex == readRequests_13_bits_instructionIndex_0;
  wire         readRecord_27_state_wLaneClear =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_state_wLaneClear | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_state_wLaneClear | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_state_wLaneClear
    | _loadUpdateValidVec_T_25 & chainingRecordCopy_3_bits_state_wLaneClear | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_state_wLaneClear;
  wire         readRecord_27_state_wTopLastReport =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_state_wTopLastReport | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_state_wTopLastReport | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_state_wTopLastReport
    | _loadUpdateValidVec_T_25 & chainingRecordCopy_3_bits_state_wTopLastReport | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_state_wTopLastReport;
  wire         readRecord_27_state_wLaneLastReport =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_state_wLaneLastReport | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_state_wLaneLastReport | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_state_wLaneLastReport
    | _loadUpdateValidVec_T_25 & chainingRecordCopy_3_bits_state_wLaneLastReport | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_state_wLaneLastReport;
  wire         readRecord_27_state_wWriteQueueClear =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_state_wWriteQueueClear | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_state_wWriteQueueClear | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_state_wWriteQueueClear
    | _loadUpdateValidVec_T_25 & chainingRecordCopy_3_bits_state_wWriteQueueClear | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_state_wWriteQueueClear;
  wire         readRecord_27_state_stFinish =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_state_stFinish | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_state_stFinish | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_state_stFinish | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_state_stFinish | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_state_stFinish;
  wire [255:0] readRecord_27_elementMask =
    (_loadUpdateValidVec_T_16 ? chainingRecordCopy_0_bits_elementMask : 256'h0) | (_loadUpdateValidVec_T_19 ? chainingRecordCopy_1_bits_elementMask : 256'h0) | (_loadUpdateValidVec_T_22 ? chainingRecordCopy_2_bits_elementMask : 256'h0)
    | (_loadUpdateValidVec_T_25 ? chainingRecordCopy_3_bits_elementMask : 256'h0) | (_loadUpdateValidVec_T_28 ? chainingRecordCopy_4_bits_elementMask : 256'h0);
  wire         readRecord_27_slow =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_slow | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_slow | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_slow | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_slow | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_slow;
  wire         readRecord_27_onlyRead =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_onlyRead | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_onlyRead | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_onlyRead | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_onlyRead | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_onlyRead;
  wire         readRecord_27_ma =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_ma | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_ma | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_ma | _loadUpdateValidVec_T_25 & chainingRecordCopy_3_bits_ma
    | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_ma;
  wire         readRecord_27_indexType =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_indexType | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_indexType | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_indexType | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_indexType | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_indexType;
  wire         readRecord_27_crossRead =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_crossRead | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_crossRead | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_crossRead | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_crossRead | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_crossRead;
  wire         readRecord_27_crossWrite =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_crossWrite | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_crossWrite | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_crossWrite | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_crossWrite | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_crossWrite;
  wire         readRecord_27_gather16 =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_gather16 | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_gather16 | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_gather16 | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_gather16 | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_gather16;
  wire         readRecord_27_gather =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_gather | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_gather | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_gather | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_gather | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_gather;
  wire         readRecord_27_st =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_st | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_st | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_st | _loadUpdateValidVec_T_25 & chainingRecordCopy_3_bits_st
    | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_st;
  wire         readRecord_27_ls =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_ls | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_ls | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_ls | _loadUpdateValidVec_T_25 & chainingRecordCopy_3_bits_ls
    | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_ls;
  wire [2:0]   readRecord_27_instIndex =
    (_loadUpdateValidVec_T_16 ? chainingRecordCopy_0_bits_instIndex : 3'h0) | (_loadUpdateValidVec_T_19 ? chainingRecordCopy_1_bits_instIndex : 3'h0) | (_loadUpdateValidVec_T_22 ? chainingRecordCopy_2_bits_instIndex : 3'h0)
    | (_loadUpdateValidVec_T_25 ? chainingRecordCopy_3_bits_instIndex : 3'h0) | (_loadUpdateValidVec_T_28 ? chainingRecordCopy_4_bits_instIndex : 3'h0);
  wire [4:0]   readRecord_27_vs2 =
    (_loadUpdateValidVec_T_16 ? chainingRecordCopy_0_bits_vs2 : 5'h0) | (_loadUpdateValidVec_T_19 ? chainingRecordCopy_1_bits_vs2 : 5'h0) | (_loadUpdateValidVec_T_22 ? chainingRecordCopy_2_bits_vs2 : 5'h0)
    | (_loadUpdateValidVec_T_25 ? chainingRecordCopy_3_bits_vs2 : 5'h0) | (_loadUpdateValidVec_T_28 ? chainingRecordCopy_4_bits_vs2 : 5'h0);
  wire [4:0]   readRecord_27_vs1_bits =
    (_loadUpdateValidVec_T_16 ? chainingRecordCopy_0_bits_vs1_bits : 5'h0) | (_loadUpdateValidVec_T_19 ? chainingRecordCopy_1_bits_vs1_bits : 5'h0) | (_loadUpdateValidVec_T_22 ? chainingRecordCopy_2_bits_vs1_bits : 5'h0)
    | (_loadUpdateValidVec_T_25 ? chainingRecordCopy_3_bits_vs1_bits : 5'h0) | (_loadUpdateValidVec_T_28 ? chainingRecordCopy_4_bits_vs1_bits : 5'h0);
  wire         readRecord_27_vs1_valid =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_vs1_valid | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_vs1_valid | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_vs1_valid | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_vs1_valid | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_vs1_valid;
  wire [4:0]   readRecord_27_vd_bits =
    (_loadUpdateValidVec_T_16 ? chainingRecordCopy_0_bits_vd_bits : 5'h0) | (_loadUpdateValidVec_T_19 ? chainingRecordCopy_1_bits_vd_bits : 5'h0) | (_loadUpdateValidVec_T_22 ? chainingRecordCopy_2_bits_vd_bits : 5'h0)
    | (_loadUpdateValidVec_T_25 ? chainingRecordCopy_3_bits_vd_bits : 5'h0) | (_loadUpdateValidVec_T_28 ? chainingRecordCopy_4_bits_vd_bits : 5'h0);
  wire         readRecord_27_vd_valid =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_vd_valid | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_vd_valid | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_vd_valid | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_vd_valid | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_vd_valid;
  wire         checkResult =
    _checkResult_ChainingCheck_readPort13_record0_checkResult & _checkResult_ChainingCheck_readPort13_record1_checkResult & _checkResult_ChainingCheck_readPort13_record2_checkResult
    & _checkResult_ChainingCheck_readPort13_record3_checkResult & _checkResult_ChainingCheck_readPort13_record4_checkResult;
  wire         validCorrect = readRequests_13_valid_0 & checkResult;
  wire         bankCorrect_13 = validCorrect;
  wire [9:0]   address_13 = {readRequests_13_bits_vs_0, readRequests_13_bits_offset_0};
  reg          pipeBank_pipe_v_13;
  reg          pipeBank_pipe_pipe_v_13;
  wire         pipeBank_pipe_pipe_out_13_valid = pipeBank_pipe_pipe_v_13;
  reg          pipeBank_pipe_pipe_b_13;
  wire         pipeBank_pipe_pipe_out_13_bits = pipeBank_pipe_pipe_b_13;
  wire         portReady_13 = ~firstUsed_13 & checkResult;
  assign readRequests_13_ready_0 = portReady_13 & sramReady;
  wire         bankReadF_13 = bankCorrect_13 & ~firstUsed_13;
  wire         bankReadS_13 = bankCorrect_13 & ~_GEN_10 & firstUsed_13;
  reg          pipeFirstUsed_pipe_v_13;
  reg          pipeFirstUsed_pipe_b_13;
  reg          pipeFirstUsed_pipe_pipe_v_13;
  wire         pipeFirstUsed_pipe_pipe_out_13_valid = pipeFirstUsed_pipe_pipe_v_13;
  reg          pipeFirstUsed_pipe_pipe_b_13;
  wire         pipeFirstUsed_pipe_pipe_out_13_bits = pipeFirstUsed_pipe_pipe_b_13;
  reg          pipeFire_pipe_v_13;
  reg          pipeFire_pipe_b_13;
  reg          pipeFire_pipe_pipe_v_13;
  wire         pipeFire_pipe_pipe_out_13_valid = pipeFire_pipe_pipe_v_13;
  reg          pipeFire_pipe_pipe_b_13;
  wire         pipeFire_pipe_pipe_out_13_bits = pipeFire_pipe_pipe_b_13;
  wire         firstOccupied = firstUsed_13 | bankCorrect_13;
  wire         secondOccupied = bankCorrect_13 & firstUsed_13 | _GEN_10;
  assign write_ready_0 = sramReady & ~firstOccupied;
  wire [31:0]  writeData = resetValid ? 32'h0 : writePipe_bits_data;
  wire [9:0]   writeAddress = resetValid ? sramResetCount : {writePipe_bits_vd, writePipe_bits_offset};
  wire         ramWriteValid;
  assign ramWriteValid = writeValid | resetValid;
  wire         vrfSRAM_0_readwritePorts_0_isWrite = ramWriteValid;
  wire [9:0]   vrfSRAM_0_readwritePorts_0_address = ramWriteValid ? writeAddress : firstReadPipe_0_bits_address;
  wire         vrfSRAM_0_readwritePorts_0_enable = ramWriteValid | firstReadPipe_0_valid;
  wire [35:0]  vrfSRAM_0_readwritePorts_0_readData;
  wire [35:0]  vrfSRAM_0_readwritePorts_0_writeData = {4'h0, writeData};
  assign readResultF_0 = vrfSRAM_0_readwritePorts_0_readData[31:0];
  wire [1:0]   freeRecord_lo = {~chainingRecord_1_valid, ~chainingRecord_0_valid};
  wire [1:0]   freeRecord_hi_hi = {~chainingRecord_4_valid, ~chainingRecord_3_valid};
  wire [2:0]   freeRecord_hi = {freeRecord_hi_hi, ~chainingRecord_2_valid};
  wire [4:0]   freeRecord = {freeRecord_hi, freeRecord_lo};
  wire [3:0]   _recordFFO_T_2 = freeRecord[3:0] | {freeRecord[2:0], 1'h0};
  wire [4:0]   recordFFO = {~(_recordFFO_T_2 | {_recordFFO_T_2[1:0], 2'h0}), 1'h1} & freeRecord;
  wire [4:0]   recordEnq = instructionWriteReport_valid ? recordFFO : 5'h0;
  wire [255:0] writeOH_0 = 256'h1 << {248'h0, writePipe_bits_vd[2:0] - chainingRecord_0_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [255:0] loadReadOH_0 = 256'h1 << {248'h0, readRequests_13_bits_vs_0[2:0] - chainingRecord_0_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]   _GEN_11 = {5'h0, chainingRecord_0_bits_instIndex};
  wire         dataInLsuQueue = |(8'h1 << _GEN_11 & loadDataInLSUWriteQueue);
  wire         writeUpdateValidVec_0 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecord_0_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecord_0_bits_ls);
  wire [255:0] writeUpdate1HVec_0 = writeUpdateValidVec_0 ? writeOH_0 : 256'h0;
  wire         loadUpdateValidVec_0 = _loadUpdateValidVec_T_27 & readRequests_13_bits_instructionIndex_0 == chainingRecord_0_bits_instIndex & chainingRecord_0_bits_st;
  wire [255:0] loadUpdate1HVec_0 = loadUpdateValidVec_0 ? loadReadOH_0 : 256'h0;
  wire         elementUpdateValid = writeUpdateValidVec_0 | loadUpdateValidVec_0;
  wire [255:0] elementUpdate1H = writeUpdate1HVec_0 | loadUpdate1HVec_0;
  wire         dataInLaneCheck = |(8'h1 << _GEN_11 & dataInLane);
  wire         laneLastReport = |(8'h1 << _GEN_11 & instructionLastReport);
  wire         topLastReport = |(8'h1 << _GEN_11 & lsuLastReport);
  wire         waitLaneClear = chainingRecord_0_bits_state_stFinish & chainingRecord_0_bits_state_wWriteQueueClear & chainingRecord_0_bits_state_wLaneLastReport & chainingRecord_0_bits_state_wTopLastReport;
  wire         stateClear = waitLaneClear & chainingRecord_0_bits_state_wLaneClear | (&chainingRecord_0_bits_elementMask) & ~chainingRecord_0_bits_onlyRead;
  wire [255:0] writeOH_0_1 = 256'h1 << {248'h0, writePipe_bits_vd[2:0] - chainingRecord_1_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [255:0] loadReadOH_0_1 = 256'h1 << {248'h0, readRequests_13_bits_vs_0[2:0] - chainingRecord_1_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]   _GEN_12 = {5'h0, chainingRecord_1_bits_instIndex};
  wire         dataInLsuQueue_1 = |(8'h1 << _GEN_12 & loadDataInLSUWriteQueue);
  wire         writeUpdateValidVec_0_1 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecord_1_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecord_1_bits_ls);
  wire [255:0] writeUpdate1HVec_0_1 = writeUpdateValidVec_0_1 ? writeOH_0_1 : 256'h0;
  wire         loadUpdateValidVec_0_1 = _loadUpdateValidVec_T_27 & readRequests_13_bits_instructionIndex_0 == chainingRecord_1_bits_instIndex & chainingRecord_1_bits_st;
  wire [255:0] loadUpdate1HVec_0_1 = loadUpdateValidVec_0_1 ? loadReadOH_0_1 : 256'h0;
  wire         elementUpdateValid_1 = writeUpdateValidVec_0_1 | loadUpdateValidVec_0_1;
  wire [255:0] elementUpdate1H_1 = writeUpdate1HVec_0_1 | loadUpdate1HVec_0_1;
  wire         dataInLaneCheck_1 = |(8'h1 << _GEN_12 & dataInLane);
  wire         laneLastReport_1 = |(8'h1 << _GEN_12 & instructionLastReport);
  wire         topLastReport_1 = |(8'h1 << _GEN_12 & lsuLastReport);
  wire         waitLaneClear_1 = chainingRecord_1_bits_state_stFinish & chainingRecord_1_bits_state_wWriteQueueClear & chainingRecord_1_bits_state_wLaneLastReport & chainingRecord_1_bits_state_wTopLastReport;
  wire         stateClear_1 = waitLaneClear_1 & chainingRecord_1_bits_state_wLaneClear | (&chainingRecord_1_bits_elementMask) & ~chainingRecord_1_bits_onlyRead;
  wire [255:0] writeOH_0_2 = 256'h1 << {248'h0, writePipe_bits_vd[2:0] - chainingRecord_2_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [255:0] loadReadOH_0_2 = 256'h1 << {248'h0, readRequests_13_bits_vs_0[2:0] - chainingRecord_2_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]   _GEN_13 = {5'h0, chainingRecord_2_bits_instIndex};
  wire         dataInLsuQueue_2 = |(8'h1 << _GEN_13 & loadDataInLSUWriteQueue);
  wire         writeUpdateValidVec_0_2 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecord_2_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecord_2_bits_ls);
  wire [255:0] writeUpdate1HVec_0_2 = writeUpdateValidVec_0_2 ? writeOH_0_2 : 256'h0;
  wire         loadUpdateValidVec_0_2 = _loadUpdateValidVec_T_27 & readRequests_13_bits_instructionIndex_0 == chainingRecord_2_bits_instIndex & chainingRecord_2_bits_st;
  wire [255:0] loadUpdate1HVec_0_2 = loadUpdateValidVec_0_2 ? loadReadOH_0_2 : 256'h0;
  wire         elementUpdateValid_2 = writeUpdateValidVec_0_2 | loadUpdateValidVec_0_2;
  wire [255:0] elementUpdate1H_2 = writeUpdate1HVec_0_2 | loadUpdate1HVec_0_2;
  wire         dataInLaneCheck_2 = |(8'h1 << _GEN_13 & dataInLane);
  wire         laneLastReport_2 = |(8'h1 << _GEN_13 & instructionLastReport);
  wire         topLastReport_2 = |(8'h1 << _GEN_13 & lsuLastReport);
  wire         waitLaneClear_2 = chainingRecord_2_bits_state_stFinish & chainingRecord_2_bits_state_wWriteQueueClear & chainingRecord_2_bits_state_wLaneLastReport & chainingRecord_2_bits_state_wTopLastReport;
  wire         stateClear_2 = waitLaneClear_2 & chainingRecord_2_bits_state_wLaneClear | (&chainingRecord_2_bits_elementMask) & ~chainingRecord_2_bits_onlyRead;
  wire [255:0] writeOH_0_3 = 256'h1 << {248'h0, writePipe_bits_vd[2:0] - chainingRecord_3_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [255:0] loadReadOH_0_3 = 256'h1 << {248'h0, readRequests_13_bits_vs_0[2:0] - chainingRecord_3_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]   _GEN_14 = {5'h0, chainingRecord_3_bits_instIndex};
  wire         dataInLsuQueue_3 = |(8'h1 << _GEN_14 & loadDataInLSUWriteQueue);
  wire         writeUpdateValidVec_0_3 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecord_3_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecord_3_bits_ls);
  wire [255:0] writeUpdate1HVec_0_3 = writeUpdateValidVec_0_3 ? writeOH_0_3 : 256'h0;
  wire         loadUpdateValidVec_0_3 = _loadUpdateValidVec_T_27 & readRequests_13_bits_instructionIndex_0 == chainingRecord_3_bits_instIndex & chainingRecord_3_bits_st;
  wire [255:0] loadUpdate1HVec_0_3 = loadUpdateValidVec_0_3 ? loadReadOH_0_3 : 256'h0;
  wire         elementUpdateValid_3 = writeUpdateValidVec_0_3 | loadUpdateValidVec_0_3;
  wire [255:0] elementUpdate1H_3 = writeUpdate1HVec_0_3 | loadUpdate1HVec_0_3;
  wire         dataInLaneCheck_3 = |(8'h1 << _GEN_14 & dataInLane);
  wire         laneLastReport_3 = |(8'h1 << _GEN_14 & instructionLastReport);
  wire         topLastReport_3 = |(8'h1 << _GEN_14 & lsuLastReport);
  wire         waitLaneClear_3 = chainingRecord_3_bits_state_stFinish & chainingRecord_3_bits_state_wWriteQueueClear & chainingRecord_3_bits_state_wLaneLastReport & chainingRecord_3_bits_state_wTopLastReport;
  wire         stateClear_3 = waitLaneClear_3 & chainingRecord_3_bits_state_wLaneClear | (&chainingRecord_3_bits_elementMask) & ~chainingRecord_3_bits_onlyRead;
  wire [255:0] writeOH_0_4 = 256'h1 << {248'h0, writePipe_bits_vd[2:0] - chainingRecord_4_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [255:0] loadReadOH_0_4 = 256'h1 << {248'h0, readRequests_13_bits_vs_0[2:0] - chainingRecord_4_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]   _GEN_15 = {5'h0, chainingRecord_4_bits_instIndex};
  wire         dataInLsuQueue_4 = |(8'h1 << _GEN_15 & loadDataInLSUWriteQueue);
  wire         writeUpdateValidVec_0_4 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecord_4_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecord_4_bits_ls);
  wire [255:0] writeUpdate1HVec_0_4 = writeUpdateValidVec_0_4 ? writeOH_0_4 : 256'h0;
  wire         loadUpdateValidVec_0_4 = _loadUpdateValidVec_T_27 & readRequests_13_bits_instructionIndex_0 == chainingRecord_4_bits_instIndex & chainingRecord_4_bits_st;
  wire [255:0] loadUpdate1HVec_0_4 = loadUpdateValidVec_0_4 ? loadReadOH_0_4 : 256'h0;
  wire         elementUpdateValid_4 = writeUpdateValidVec_0_4 | loadUpdateValidVec_0_4;
  wire [255:0] elementUpdate1H_4 = writeUpdate1HVec_0_4 | loadUpdate1HVec_0_4;
  wire         dataInLaneCheck_4 = |(8'h1 << _GEN_15 & dataInLane);
  wire         laneLastReport_4 = |(8'h1 << _GEN_15 & instructionLastReport);
  wire         topLastReport_4 = |(8'h1 << _GEN_15 & lsuLastReport);
  wire         waitLaneClear_4 = chainingRecord_4_bits_state_stFinish & chainingRecord_4_bits_state_wWriteQueueClear & chainingRecord_4_bits_state_wLaneLastReport & chainingRecord_4_bits_state_wTopLastReport;
  wire         stateClear_4 = waitLaneClear_4 & chainingRecord_4_bits_state_wLaneClear | (&chainingRecord_4_bits_elementMask) & ~chainingRecord_4_bits_onlyRead;
  wire [255:0] writeOH_0_5 = 256'h1 << {248'h0, writePipe_bits_vd[2:0] - chainingRecordCopy_0_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [255:0] loadReadOH_0_5 = 256'h1 << {248'h0, readRequests_13_bits_vs_0[2:0] - chainingRecordCopy_0_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]   _GEN_16 = {5'h0, chainingRecordCopy_0_bits_instIndex};
  wire         dataInLsuQueue_5 = |(8'h1 << _GEN_16 & loadDataInLSUWriteQueue);
  wire         writeUpdateValidVec_0_5 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecordCopy_0_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecordCopy_0_bits_ls);
  wire [255:0] writeUpdate1HVec_0_5 = writeUpdateValidVec_0_5 ? writeOH_0_5 : 256'h0;
  wire         loadUpdateValidVec_0_5 = _loadUpdateValidVec_T_27 & _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_st;
  wire [255:0] loadUpdate1HVec_0_5 = loadUpdateValidVec_0_5 ? loadReadOH_0_5 : 256'h0;
  wire         elementUpdateValid_5 = writeUpdateValidVec_0_5 | loadUpdateValidVec_0_5;
  wire [255:0] elementUpdate1H_5 = writeUpdate1HVec_0_5 | loadUpdate1HVec_0_5;
  wire         dataInLaneCheck_5 = |(8'h1 << _GEN_16 & dataInLane);
  wire         laneLastReport_5 = |(8'h1 << _GEN_16 & instructionLastReport);
  wire         topLastReport_5 = |(8'h1 << _GEN_16 & lsuLastReport);
  wire         waitLaneClear_5 = chainingRecordCopy_0_bits_state_stFinish & chainingRecordCopy_0_bits_state_wWriteQueueClear & chainingRecordCopy_0_bits_state_wLaneLastReport & chainingRecordCopy_0_bits_state_wTopLastReport;
  wire         stateClear_5 = waitLaneClear_5 & chainingRecordCopy_0_bits_state_wLaneClear | (&chainingRecordCopy_0_bits_elementMask) & ~chainingRecordCopy_0_bits_onlyRead;
  wire [7:0]   recordRelease_0 = stateClear_5 & chainingRecordCopy_0_valid ? 8'h1 << _GEN_16 : stateClear & chainingRecord_0_valid ? 8'h1 << _GEN_11 : 8'h0;
  wire [255:0] writeOH_0_6 = 256'h1 << {248'h0, writePipe_bits_vd[2:0] - chainingRecordCopy_1_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [255:0] loadReadOH_0_6 = 256'h1 << {248'h0, readRequests_13_bits_vs_0[2:0] - chainingRecordCopy_1_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]   _GEN_17 = {5'h0, chainingRecordCopy_1_bits_instIndex};
  wire         dataInLsuQueue_6 = |(8'h1 << _GEN_17 & loadDataInLSUWriteQueue);
  wire         writeUpdateValidVec_0_6 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecordCopy_1_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecordCopy_1_bits_ls);
  wire [255:0] writeUpdate1HVec_0_6 = writeUpdateValidVec_0_6 ? writeOH_0_6 : 256'h0;
  wire         loadUpdateValidVec_0_6 = _loadUpdateValidVec_T_27 & _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_st;
  wire [255:0] loadUpdate1HVec_0_6 = loadUpdateValidVec_0_6 ? loadReadOH_0_6 : 256'h0;
  wire         elementUpdateValid_6 = writeUpdateValidVec_0_6 | loadUpdateValidVec_0_6;
  wire [255:0] elementUpdate1H_6 = writeUpdate1HVec_0_6 | loadUpdate1HVec_0_6;
  wire         dataInLaneCheck_6 = |(8'h1 << _GEN_17 & dataInLane);
  wire         laneLastReport_6 = |(8'h1 << _GEN_17 & instructionLastReport);
  wire         topLastReport_6 = |(8'h1 << _GEN_17 & lsuLastReport);
  wire         waitLaneClear_6 = chainingRecordCopy_1_bits_state_stFinish & chainingRecordCopy_1_bits_state_wWriteQueueClear & chainingRecordCopy_1_bits_state_wLaneLastReport & chainingRecordCopy_1_bits_state_wTopLastReport;
  wire         stateClear_6 = waitLaneClear_6 & chainingRecordCopy_1_bits_state_wLaneClear | (&chainingRecordCopy_1_bits_elementMask) & ~chainingRecordCopy_1_bits_onlyRead;
  wire [7:0]   recordRelease_1 = stateClear_6 & chainingRecordCopy_1_valid ? 8'h1 << _GEN_17 : stateClear_1 & chainingRecord_1_valid ? 8'h1 << _GEN_12 : 8'h0;
  wire [255:0] writeOH_0_7 = 256'h1 << {248'h0, writePipe_bits_vd[2:0] - chainingRecordCopy_2_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [255:0] loadReadOH_0_7 = 256'h1 << {248'h0, readRequests_13_bits_vs_0[2:0] - chainingRecordCopy_2_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]   _GEN_18 = {5'h0, chainingRecordCopy_2_bits_instIndex};
  wire         dataInLsuQueue_7 = |(8'h1 << _GEN_18 & loadDataInLSUWriteQueue);
  wire         writeUpdateValidVec_0_7 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecordCopy_2_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecordCopy_2_bits_ls);
  wire [255:0] writeUpdate1HVec_0_7 = writeUpdateValidVec_0_7 ? writeOH_0_7 : 256'h0;
  wire         loadUpdateValidVec_0_7 = _loadUpdateValidVec_T_27 & _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_st;
  wire [255:0] loadUpdate1HVec_0_7 = loadUpdateValidVec_0_7 ? loadReadOH_0_7 : 256'h0;
  wire         elementUpdateValid_7 = writeUpdateValidVec_0_7 | loadUpdateValidVec_0_7;
  wire [255:0] elementUpdate1H_7 = writeUpdate1HVec_0_7 | loadUpdate1HVec_0_7;
  wire         dataInLaneCheck_7 = |(8'h1 << _GEN_18 & dataInLane);
  wire         laneLastReport_7 = |(8'h1 << _GEN_18 & instructionLastReport);
  wire         topLastReport_7 = |(8'h1 << _GEN_18 & lsuLastReport);
  wire         waitLaneClear_7 = chainingRecordCopy_2_bits_state_stFinish & chainingRecordCopy_2_bits_state_wWriteQueueClear & chainingRecordCopy_2_bits_state_wLaneLastReport & chainingRecordCopy_2_bits_state_wTopLastReport;
  wire         stateClear_7 = waitLaneClear_7 & chainingRecordCopy_2_bits_state_wLaneClear | (&chainingRecordCopy_2_bits_elementMask) & ~chainingRecordCopy_2_bits_onlyRead;
  wire [7:0]   recordRelease_2 = stateClear_7 & chainingRecordCopy_2_valid ? 8'h1 << _GEN_18 : stateClear_2 & chainingRecord_2_valid ? 8'h1 << _GEN_13 : 8'h0;
  wire [255:0] writeOH_0_8 = 256'h1 << {248'h0, writePipe_bits_vd[2:0] - chainingRecordCopy_3_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [255:0] loadReadOH_0_8 = 256'h1 << {248'h0, readRequests_13_bits_vs_0[2:0] - chainingRecordCopy_3_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]   _GEN_19 = {5'h0, chainingRecordCopy_3_bits_instIndex};
  wire         dataInLsuQueue_8 = |(8'h1 << _GEN_19 & loadDataInLSUWriteQueue);
  wire         writeUpdateValidVec_0_8 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecordCopy_3_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecordCopy_3_bits_ls);
  wire [255:0] writeUpdate1HVec_0_8 = writeUpdateValidVec_0_8 ? writeOH_0_8 : 256'h0;
  wire         loadUpdateValidVec_0_8 = _loadUpdateValidVec_T_27 & _loadUpdateValidVec_T_25 & chainingRecordCopy_3_bits_st;
  wire [255:0] loadUpdate1HVec_0_8 = loadUpdateValidVec_0_8 ? loadReadOH_0_8 : 256'h0;
  wire         elementUpdateValid_8 = writeUpdateValidVec_0_8 | loadUpdateValidVec_0_8;
  wire [255:0] elementUpdate1H_8 = writeUpdate1HVec_0_8 | loadUpdate1HVec_0_8;
  wire         dataInLaneCheck_8 = |(8'h1 << _GEN_19 & dataInLane);
  wire         laneLastReport_8 = |(8'h1 << _GEN_19 & instructionLastReport);
  wire         topLastReport_8 = |(8'h1 << _GEN_19 & lsuLastReport);
  wire         waitLaneClear_8 = chainingRecordCopy_3_bits_state_stFinish & chainingRecordCopy_3_bits_state_wWriteQueueClear & chainingRecordCopy_3_bits_state_wLaneLastReport & chainingRecordCopy_3_bits_state_wTopLastReport;
  wire         stateClear_8 = waitLaneClear_8 & chainingRecordCopy_3_bits_state_wLaneClear | (&chainingRecordCopy_3_bits_elementMask) & ~chainingRecordCopy_3_bits_onlyRead;
  wire [7:0]   recordRelease_3 = stateClear_8 & chainingRecordCopy_3_valid ? 8'h1 << _GEN_19 : stateClear_3 & chainingRecord_3_valid ? 8'h1 << _GEN_14 : 8'h0;
  wire [255:0] writeOH_0_9 = 256'h1 << {248'h0, writePipe_bits_vd[2:0] - chainingRecordCopy_4_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [255:0] loadReadOH_0_9 = 256'h1 << {248'h0, readRequests_13_bits_vs_0[2:0] - chainingRecordCopy_4_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]   _GEN_20 = {5'h0, chainingRecordCopy_4_bits_instIndex};
  wire         dataInLsuQueue_9 = |(8'h1 << _GEN_20 & loadDataInLSUWriteQueue);
  wire         writeUpdateValidVec_0_9 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecordCopy_4_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecordCopy_4_bits_ls);
  wire [255:0] writeUpdate1HVec_0_9 = writeUpdateValidVec_0_9 ? writeOH_0_9 : 256'h0;
  wire         loadUpdateValidVec_0_9 = _loadUpdateValidVec_T_27 & _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_st;
  wire [255:0] loadUpdate1HVec_0_9 = loadUpdateValidVec_0_9 ? loadReadOH_0_9 : 256'h0;
  wire         elementUpdateValid_9 = writeUpdateValidVec_0_9 | loadUpdateValidVec_0_9;
  wire [255:0] elementUpdate1H_9 = writeUpdate1HVec_0_9 | loadUpdate1HVec_0_9;
  wire         dataInLaneCheck_9 = |(8'h1 << _GEN_20 & dataInLane);
  wire         laneLastReport_9 = |(8'h1 << _GEN_20 & instructionLastReport);
  wire         topLastReport_9 = |(8'h1 << _GEN_20 & lsuLastReport);
  wire         waitLaneClear_9 = chainingRecordCopy_4_bits_state_stFinish & chainingRecordCopy_4_bits_state_wWriteQueueClear & chainingRecordCopy_4_bits_state_wLaneLastReport & chainingRecordCopy_4_bits_state_wTopLastReport;
  wire         stateClear_9 = waitLaneClear_9 & chainingRecordCopy_4_bits_state_wLaneClear | (&chainingRecordCopy_4_bits_elementMask) & ~chainingRecordCopy_4_bits_onlyRead;
  wire [7:0]   recordRelease_4 = stateClear_9 & chainingRecordCopy_4_valid ? 8'h1 << _GEN_20 : stateClear_4 & chainingRecord_4_valid ? 8'h1 << _GEN_15 : 8'h0;
  wire         _hazardVec_isStore_T_6 = chainingRecordCopy_0_valid & chainingRecordCopy_0_bits_ls;
  wire         _GEN_21 = _hazardVec_isStore_T_6 & ~chainingRecordCopy_0_bits_st;
  wire         hazardVec_isLoad_0;
  assign hazardVec_isLoad_0 = _GEN_21;
  wire         hazardVec_isLoad_0_1;
  assign hazardVec_isLoad_0_1 = _GEN_21;
  wire         hazardVec_isLoad_0_2;
  assign hazardVec_isLoad_0_2 = _GEN_21;
  wire         hazardVec_isLoad_0_3;
  assign hazardVec_isLoad_0_3 = _GEN_21;
  wire         _hazardVec_isStore_T_12 = chainingRecordCopy_1_valid & chainingRecordCopy_1_bits_ls;
  wire         _GEN_22 = _hazardVec_isStore_T_12 & ~chainingRecordCopy_1_bits_st;
  wire         hazardVec_isLoad_1;
  assign hazardVec_isLoad_1 = _GEN_22;
  wire         hazardVec_isLoad_0_4;
  assign hazardVec_isLoad_0_4 = _GEN_22;
  wire         hazardVec_isLoad_0_5;
  assign hazardVec_isLoad_0_5 = _GEN_22;
  wire         hazardVec_isLoad_0_6;
  assign hazardVec_isLoad_0_6 = _GEN_22;
  wire         _GEN_23 = _hazardVec_isStore_T_6 & chainingRecordCopy_0_bits_st;
  wire         hazardVec_isStore_0;
  assign hazardVec_isStore_0 = _GEN_23;
  wire         hazardVec_isStore_0_1;
  assign hazardVec_isStore_0_1 = _GEN_23;
  wire         hazardVec_isStore_0_2;
  assign hazardVec_isStore_0_2 = _GEN_23;
  wire         hazardVec_isStore_0_3;
  assign hazardVec_isStore_0_3 = _GEN_23;
  wire         _GEN_24 = _hazardVec_isStore_T_12 & chainingRecordCopy_1_bits_st;
  wire         hazardVec_isStore_1;
  assign hazardVec_isStore_1 = _GEN_24;
  wire         hazardVec_isStore_0_4;
  assign hazardVec_isStore_0_4 = _GEN_24;
  wire         hazardVec_isStore_0_5;
  assign hazardVec_isStore_0_5 = _GEN_24;
  wire         hazardVec_isStore_0_6;
  assign hazardVec_isStore_0_6 = _GEN_24;
  wire         _GEN_25 = chainingRecordCopy_0_valid & chainingRecordCopy_0_bits_slow;
  wire         hazardVec_isSlow_0;
  assign hazardVec_isSlow_0 = _GEN_25;
  wire         hazardVec_isSlow_0_1;
  assign hazardVec_isSlow_0_1 = _GEN_25;
  wire         hazardVec_isSlow_0_2;
  assign hazardVec_isSlow_0_2 = _GEN_25;
  wire         hazardVec_isSlow_0_3;
  assign hazardVec_isSlow_0_3 = _GEN_25;
  wire         _GEN_26 = chainingRecordCopy_1_valid & chainingRecordCopy_1_bits_slow;
  wire         hazardVec_isSlow_1;
  assign hazardVec_isSlow_1 = _GEN_26;
  wire         hazardVec_isSlow_0_4;
  assign hazardVec_isSlow_0_4 = _GEN_26;
  wire         hazardVec_isSlow_0_5;
  assign hazardVec_isSlow_0_5 = _GEN_26;
  wire         hazardVec_isSlow_0_6;
  assign hazardVec_isSlow_0_6 = _GEN_26;
  wire         hazardVec_samVd =
    chainingRecordCopy_0_bits_vd_valid & chainingRecordCopy_1_bits_vd_valid & chainingRecordCopy_0_bits_vd_bits[4:3] == chainingRecordCopy_1_bits_vd_bits[4:3]
    & (chainingRecordCopy_0_bits_elementMask | chainingRecordCopy_1_bits_elementMask) != 256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
  wire         hazardVec_sourceVdEqSinkVs =
    chainingRecordCopy_0_bits_vd_valid & (chainingRecordCopy_0_bits_vd_bits == chainingRecordCopy_1_bits_vs2 | chainingRecordCopy_0_bits_vd_bits == chainingRecordCopy_1_bits_vs1_bits & chainingRecordCopy_1_bits_vs1_valid);
  wire         hazardVec_sinkVdEqSourceVs =
    chainingRecordCopy_1_bits_vd_valid & (chainingRecordCopy_1_bits_vd_bits == chainingRecordCopy_0_bits_vs2 | chainingRecordCopy_1_bits_vd_bits == chainingRecordCopy_0_bits_vs1_bits & chainingRecordCopy_0_bits_vs1_valid);
  wire         hazardVec_older =
    chainingRecordCopy_1_bits_instIndex == chainingRecordCopy_0_bits_instIndex | chainingRecordCopy_1_bits_instIndex[1:0] < chainingRecordCopy_0_bits_instIndex[1:0] ^ chainingRecordCopy_1_bits_instIndex[2]
    ^ chainingRecordCopy_0_bits_instIndex[2];
  wire         hazardVec_hazardForeLoad =
    (hazardVec_older ? hazardVec_isLoad_0 & hazardVec_isSlow_1 : hazardVec_isLoad_1 & hazardVec_isSlow_0) & (hazardVec_samVd | (hazardVec_older ? hazardVec_sourceVdEqSinkVs : hazardVec_sinkVdEqSourceVs));
  wire         hazardVec_rawForeStore = (hazardVec_older ? hazardVec_isStore_0 & hazardVec_isSlow_1 : hazardVec_isStore_1 & hazardVec_isSlow_0) & hazardVec_samVd;
  wire         _hazardVec_isStore_T_16 = chainingRecordCopy_2_valid & chainingRecordCopy_2_bits_ls;
  wire         _GEN_27 = _hazardVec_isStore_T_16 & ~chainingRecordCopy_2_bits_st;
  wire         hazardVec_isLoad_1_1;
  assign hazardVec_isLoad_1_1 = _GEN_27;
  wire         hazardVec_isLoad_1_4;
  assign hazardVec_isLoad_1_4 = _GEN_27;
  wire         hazardVec_isLoad_0_7;
  assign hazardVec_isLoad_0_7 = _GEN_27;
  wire         hazardVec_isLoad_0_8;
  assign hazardVec_isLoad_0_8 = _GEN_27;
  wire         _GEN_28 = _hazardVec_isStore_T_16 & chainingRecordCopy_2_bits_st;
  wire         hazardVec_isStore_1_1;
  assign hazardVec_isStore_1_1 = _GEN_28;
  wire         hazardVec_isStore_1_4;
  assign hazardVec_isStore_1_4 = _GEN_28;
  wire         hazardVec_isStore_0_7;
  assign hazardVec_isStore_0_7 = _GEN_28;
  wire         hazardVec_isStore_0_8;
  assign hazardVec_isStore_0_8 = _GEN_28;
  wire         _GEN_29 = chainingRecordCopy_2_valid & chainingRecordCopy_2_bits_slow;
  wire         hazardVec_isSlow_1_1;
  assign hazardVec_isSlow_1_1 = _GEN_29;
  wire         hazardVec_isSlow_1_4;
  assign hazardVec_isSlow_1_4 = _GEN_29;
  wire         hazardVec_isSlow_0_7;
  assign hazardVec_isSlow_0_7 = _GEN_29;
  wire         hazardVec_isSlow_0_8;
  assign hazardVec_isSlow_0_8 = _GEN_29;
  wire         hazardVec_samVd_1 =
    chainingRecordCopy_0_bits_vd_valid & chainingRecordCopy_2_bits_vd_valid & chainingRecordCopy_0_bits_vd_bits[4:3] == chainingRecordCopy_2_bits_vd_bits[4:3]
    & (chainingRecordCopy_0_bits_elementMask | chainingRecordCopy_2_bits_elementMask) != 256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
  wire         hazardVec_sourceVdEqSinkVs_1 =
    chainingRecordCopy_0_bits_vd_valid & (chainingRecordCopy_0_bits_vd_bits == chainingRecordCopy_2_bits_vs2 | chainingRecordCopy_0_bits_vd_bits == chainingRecordCopy_2_bits_vs1_bits & chainingRecordCopy_2_bits_vs1_valid);
  wire         hazardVec_sinkVdEqSourceVs_1 =
    chainingRecordCopy_2_bits_vd_valid & (chainingRecordCopy_2_bits_vd_bits == chainingRecordCopy_0_bits_vs2 | chainingRecordCopy_2_bits_vd_bits == chainingRecordCopy_0_bits_vs1_bits & chainingRecordCopy_0_bits_vs1_valid);
  wire         hazardVec_older_1 =
    chainingRecordCopy_2_bits_instIndex == chainingRecordCopy_0_bits_instIndex | chainingRecordCopy_2_bits_instIndex[1:0] < chainingRecordCopy_0_bits_instIndex[1:0] ^ chainingRecordCopy_2_bits_instIndex[2]
    ^ chainingRecordCopy_0_bits_instIndex[2];
  wire         hazardVec_hazardForeLoad_1 =
    (hazardVec_older_1 ? hazardVec_isLoad_0_1 & hazardVec_isSlow_1_1 : hazardVec_isLoad_1_1 & hazardVec_isSlow_0_1) & (hazardVec_samVd_1 | (hazardVec_older_1 ? hazardVec_sourceVdEqSinkVs_1 : hazardVec_sinkVdEqSourceVs_1));
  wire         hazardVec_rawForeStore_1 = (hazardVec_older_1 ? hazardVec_isStore_0_1 & hazardVec_isSlow_1_1 : hazardVec_isStore_1_1 & hazardVec_isSlow_0_1) & hazardVec_samVd_1;
  wire         _hazardVec_isStore_T_18 = chainingRecordCopy_3_valid & chainingRecordCopy_3_bits_ls;
  wire         _GEN_30 = _hazardVec_isStore_T_18 & ~chainingRecordCopy_3_bits_st;
  wire         hazardVec_isLoad_1_2;
  assign hazardVec_isLoad_1_2 = _GEN_30;
  wire         hazardVec_isLoad_1_5;
  assign hazardVec_isLoad_1_5 = _GEN_30;
  wire         hazardVec_isLoad_1_7;
  assign hazardVec_isLoad_1_7 = _GEN_30;
  wire         hazardVec_isLoad_0_9;
  assign hazardVec_isLoad_0_9 = _GEN_30;
  wire         _GEN_31 = _hazardVec_isStore_T_18 & chainingRecordCopy_3_bits_st;
  wire         hazardVec_isStore_1_2;
  assign hazardVec_isStore_1_2 = _GEN_31;
  wire         hazardVec_isStore_1_5;
  assign hazardVec_isStore_1_5 = _GEN_31;
  wire         hazardVec_isStore_1_7;
  assign hazardVec_isStore_1_7 = _GEN_31;
  wire         hazardVec_isStore_0_9;
  assign hazardVec_isStore_0_9 = _GEN_31;
  wire         _GEN_32 = chainingRecordCopy_3_valid & chainingRecordCopy_3_bits_slow;
  wire         hazardVec_isSlow_1_2;
  assign hazardVec_isSlow_1_2 = _GEN_32;
  wire         hazardVec_isSlow_1_5;
  assign hazardVec_isSlow_1_5 = _GEN_32;
  wire         hazardVec_isSlow_1_7;
  assign hazardVec_isSlow_1_7 = _GEN_32;
  wire         hazardVec_isSlow_0_9;
  assign hazardVec_isSlow_0_9 = _GEN_32;
  wire         hazardVec_samVd_2 =
    chainingRecordCopy_0_bits_vd_valid & chainingRecordCopy_3_bits_vd_valid & chainingRecordCopy_0_bits_vd_bits[4:3] == chainingRecordCopy_3_bits_vd_bits[4:3]
    & (chainingRecordCopy_0_bits_elementMask | chainingRecordCopy_3_bits_elementMask) != 256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
  wire         hazardVec_sourceVdEqSinkVs_2 =
    chainingRecordCopy_0_bits_vd_valid & (chainingRecordCopy_0_bits_vd_bits == chainingRecordCopy_3_bits_vs2 | chainingRecordCopy_0_bits_vd_bits == chainingRecordCopy_3_bits_vs1_bits & chainingRecordCopy_3_bits_vs1_valid);
  wire         hazardVec_sinkVdEqSourceVs_2 =
    chainingRecordCopy_3_bits_vd_valid & (chainingRecordCopy_3_bits_vd_bits == chainingRecordCopy_0_bits_vs2 | chainingRecordCopy_3_bits_vd_bits == chainingRecordCopy_0_bits_vs1_bits & chainingRecordCopy_0_bits_vs1_valid);
  wire         hazardVec_older_2 =
    chainingRecordCopy_3_bits_instIndex == chainingRecordCopy_0_bits_instIndex | chainingRecordCopy_3_bits_instIndex[1:0] < chainingRecordCopy_0_bits_instIndex[1:0] ^ chainingRecordCopy_3_bits_instIndex[2]
    ^ chainingRecordCopy_0_bits_instIndex[2];
  wire         hazardVec_hazardForeLoad_2 =
    (hazardVec_older_2 ? hazardVec_isLoad_0_2 & hazardVec_isSlow_1_2 : hazardVec_isLoad_1_2 & hazardVec_isSlow_0_2) & (hazardVec_samVd_2 | (hazardVec_older_2 ? hazardVec_sourceVdEqSinkVs_2 : hazardVec_sinkVdEqSourceVs_2));
  wire         hazardVec_rawForeStore_2 = (hazardVec_older_2 ? hazardVec_isStore_0_2 & hazardVec_isSlow_1_2 : hazardVec_isStore_1_2 & hazardVec_isSlow_0_2) & hazardVec_samVd_2;
  wire         _hazardVec_isStore_T_19 = chainingRecordCopy_4_valid & chainingRecordCopy_4_bits_ls;
  wire         _GEN_33 = _hazardVec_isStore_T_19 & ~chainingRecordCopy_4_bits_st;
  wire         hazardVec_isLoad_1_3;
  assign hazardVec_isLoad_1_3 = _GEN_33;
  wire         hazardVec_isLoad_1_6;
  assign hazardVec_isLoad_1_6 = _GEN_33;
  wire         hazardVec_isLoad_1_8;
  assign hazardVec_isLoad_1_8 = _GEN_33;
  wire         hazardVec_isLoad_1_9;
  assign hazardVec_isLoad_1_9 = _GEN_33;
  wire         _GEN_34 = _hazardVec_isStore_T_19 & chainingRecordCopy_4_bits_st;
  wire         hazardVec_isStore_1_3;
  assign hazardVec_isStore_1_3 = _GEN_34;
  wire         hazardVec_isStore_1_6;
  assign hazardVec_isStore_1_6 = _GEN_34;
  wire         hazardVec_isStore_1_8;
  assign hazardVec_isStore_1_8 = _GEN_34;
  wire         hazardVec_isStore_1_9;
  assign hazardVec_isStore_1_9 = _GEN_34;
  wire         _GEN_35 = chainingRecordCopy_4_valid & chainingRecordCopy_4_bits_slow;
  wire         hazardVec_isSlow_1_3;
  assign hazardVec_isSlow_1_3 = _GEN_35;
  wire         hazardVec_isSlow_1_6;
  assign hazardVec_isSlow_1_6 = _GEN_35;
  wire         hazardVec_isSlow_1_8;
  assign hazardVec_isSlow_1_8 = _GEN_35;
  wire         hazardVec_isSlow_1_9;
  assign hazardVec_isSlow_1_9 = _GEN_35;
  wire         hazardVec_samVd_3 =
    chainingRecordCopy_0_bits_vd_valid & chainingRecordCopy_4_bits_vd_valid & chainingRecordCopy_0_bits_vd_bits[4:3] == chainingRecordCopy_4_bits_vd_bits[4:3]
    & (chainingRecordCopy_0_bits_elementMask | chainingRecordCopy_4_bits_elementMask) != 256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
  wire         hazardVec_sourceVdEqSinkVs_3 =
    chainingRecordCopy_0_bits_vd_valid & (chainingRecordCopy_0_bits_vd_bits == chainingRecordCopy_4_bits_vs2 | chainingRecordCopy_0_bits_vd_bits == chainingRecordCopy_4_bits_vs1_bits & chainingRecordCopy_4_bits_vs1_valid);
  wire         hazardVec_sinkVdEqSourceVs_3 =
    chainingRecordCopy_4_bits_vd_valid & (chainingRecordCopy_4_bits_vd_bits == chainingRecordCopy_0_bits_vs2 | chainingRecordCopy_4_bits_vd_bits == chainingRecordCopy_0_bits_vs1_bits & chainingRecordCopy_0_bits_vs1_valid);
  wire         hazardVec_older_3 =
    chainingRecordCopy_4_bits_instIndex == chainingRecordCopy_0_bits_instIndex | chainingRecordCopy_4_bits_instIndex[1:0] < chainingRecordCopy_0_bits_instIndex[1:0] ^ chainingRecordCopy_4_bits_instIndex[2]
    ^ chainingRecordCopy_0_bits_instIndex[2];
  wire         hazardVec_hazardForeLoad_3 =
    (hazardVec_older_3 ? hazardVec_isLoad_0_3 & hazardVec_isSlow_1_3 : hazardVec_isLoad_1_3 & hazardVec_isSlow_0_3) & (hazardVec_samVd_3 | (hazardVec_older_3 ? hazardVec_sourceVdEqSinkVs_3 : hazardVec_sinkVdEqSourceVs_3));
  wire         hazardVec_rawForeStore_3 = (hazardVec_older_3 ? hazardVec_isStore_0_3 & hazardVec_isSlow_1_3 : hazardVec_isStore_1_3 & hazardVec_isSlow_0_3) & hazardVec_samVd_3;
  wire         hazardVec_samVd_4 =
    chainingRecordCopy_1_bits_vd_valid & chainingRecordCopy_2_bits_vd_valid & chainingRecordCopy_1_bits_vd_bits[4:3] == chainingRecordCopy_2_bits_vd_bits[4:3]
    & (chainingRecordCopy_1_bits_elementMask | chainingRecordCopy_2_bits_elementMask) != 256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
  wire         hazardVec_sourceVdEqSinkVs_4 =
    chainingRecordCopy_1_bits_vd_valid & (chainingRecordCopy_1_bits_vd_bits == chainingRecordCopy_2_bits_vs2 | chainingRecordCopy_1_bits_vd_bits == chainingRecordCopy_2_bits_vs1_bits & chainingRecordCopy_2_bits_vs1_valid);
  wire         hazardVec_sinkVdEqSourceVs_4 =
    chainingRecordCopy_2_bits_vd_valid & (chainingRecordCopy_2_bits_vd_bits == chainingRecordCopy_1_bits_vs2 | chainingRecordCopy_2_bits_vd_bits == chainingRecordCopy_1_bits_vs1_bits & chainingRecordCopy_1_bits_vs1_valid);
  wire         hazardVec_older_4 =
    chainingRecordCopy_2_bits_instIndex == chainingRecordCopy_1_bits_instIndex | chainingRecordCopy_2_bits_instIndex[1:0] < chainingRecordCopy_1_bits_instIndex[1:0] ^ chainingRecordCopy_2_bits_instIndex[2]
    ^ chainingRecordCopy_1_bits_instIndex[2];
  wire         hazardVec_hazardForeLoad_4 =
    (hazardVec_older_4 ? hazardVec_isLoad_0_4 & hazardVec_isSlow_1_4 : hazardVec_isLoad_1_4 & hazardVec_isSlow_0_4) & (hazardVec_samVd_4 | (hazardVec_older_4 ? hazardVec_sourceVdEqSinkVs_4 : hazardVec_sinkVdEqSourceVs_4));
  wire         hazardVec_rawForeStore_4 = (hazardVec_older_4 ? hazardVec_isStore_0_4 & hazardVec_isSlow_1_4 : hazardVec_isStore_1_4 & hazardVec_isSlow_0_4) & hazardVec_samVd_4;
  wire         hazardVec_samVd_5 =
    chainingRecordCopy_1_bits_vd_valid & chainingRecordCopy_3_bits_vd_valid & chainingRecordCopy_1_bits_vd_bits[4:3] == chainingRecordCopy_3_bits_vd_bits[4:3]
    & (chainingRecordCopy_1_bits_elementMask | chainingRecordCopy_3_bits_elementMask) != 256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
  wire         hazardVec_sourceVdEqSinkVs_5 =
    chainingRecordCopy_1_bits_vd_valid & (chainingRecordCopy_1_bits_vd_bits == chainingRecordCopy_3_bits_vs2 | chainingRecordCopy_1_bits_vd_bits == chainingRecordCopy_3_bits_vs1_bits & chainingRecordCopy_3_bits_vs1_valid);
  wire         hazardVec_sinkVdEqSourceVs_5 =
    chainingRecordCopy_3_bits_vd_valid & (chainingRecordCopy_3_bits_vd_bits == chainingRecordCopy_1_bits_vs2 | chainingRecordCopy_3_bits_vd_bits == chainingRecordCopy_1_bits_vs1_bits & chainingRecordCopy_1_bits_vs1_valid);
  wire         hazardVec_older_5 =
    chainingRecordCopy_3_bits_instIndex == chainingRecordCopy_1_bits_instIndex | chainingRecordCopy_3_bits_instIndex[1:0] < chainingRecordCopy_1_bits_instIndex[1:0] ^ chainingRecordCopy_3_bits_instIndex[2]
    ^ chainingRecordCopy_1_bits_instIndex[2];
  wire         hazardVec_hazardForeLoad_5 =
    (hazardVec_older_5 ? hazardVec_isLoad_0_5 & hazardVec_isSlow_1_5 : hazardVec_isLoad_1_5 & hazardVec_isSlow_0_5) & (hazardVec_samVd_5 | (hazardVec_older_5 ? hazardVec_sourceVdEqSinkVs_5 : hazardVec_sinkVdEqSourceVs_5));
  wire         hazardVec_rawForeStore_5 = (hazardVec_older_5 ? hazardVec_isStore_0_5 & hazardVec_isSlow_1_5 : hazardVec_isStore_1_5 & hazardVec_isSlow_0_5) & hazardVec_samVd_5;
  wire         hazardVec_samVd_6 =
    chainingRecordCopy_1_bits_vd_valid & chainingRecordCopy_4_bits_vd_valid & chainingRecordCopy_1_bits_vd_bits[4:3] == chainingRecordCopy_4_bits_vd_bits[4:3]
    & (chainingRecordCopy_1_bits_elementMask | chainingRecordCopy_4_bits_elementMask) != 256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
  wire         hazardVec_sourceVdEqSinkVs_6 =
    chainingRecordCopy_1_bits_vd_valid & (chainingRecordCopy_1_bits_vd_bits == chainingRecordCopy_4_bits_vs2 | chainingRecordCopy_1_bits_vd_bits == chainingRecordCopy_4_bits_vs1_bits & chainingRecordCopy_4_bits_vs1_valid);
  wire         hazardVec_sinkVdEqSourceVs_6 =
    chainingRecordCopy_4_bits_vd_valid & (chainingRecordCopy_4_bits_vd_bits == chainingRecordCopy_1_bits_vs2 | chainingRecordCopy_4_bits_vd_bits == chainingRecordCopy_1_bits_vs1_bits & chainingRecordCopy_1_bits_vs1_valid);
  wire         hazardVec_older_6 =
    chainingRecordCopy_4_bits_instIndex == chainingRecordCopy_1_bits_instIndex | chainingRecordCopy_4_bits_instIndex[1:0] < chainingRecordCopy_1_bits_instIndex[1:0] ^ chainingRecordCopy_4_bits_instIndex[2]
    ^ chainingRecordCopy_1_bits_instIndex[2];
  wire         hazardVec_hazardForeLoad_6 =
    (hazardVec_older_6 ? hazardVec_isLoad_0_6 & hazardVec_isSlow_1_6 : hazardVec_isLoad_1_6 & hazardVec_isSlow_0_6) & (hazardVec_samVd_6 | (hazardVec_older_6 ? hazardVec_sourceVdEqSinkVs_6 : hazardVec_sinkVdEqSourceVs_6));
  wire         hazardVec_rawForeStore_6 = (hazardVec_older_6 ? hazardVec_isStore_0_6 & hazardVec_isSlow_1_6 : hazardVec_isStore_1_6 & hazardVec_isSlow_0_6) & hazardVec_samVd_6;
  wire         hazardVec_samVd_7 =
    chainingRecordCopy_2_bits_vd_valid & chainingRecordCopy_3_bits_vd_valid & chainingRecordCopy_2_bits_vd_bits[4:3] == chainingRecordCopy_3_bits_vd_bits[4:3]
    & (chainingRecordCopy_2_bits_elementMask | chainingRecordCopy_3_bits_elementMask) != 256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
  wire         hazardVec_sourceVdEqSinkVs_7 =
    chainingRecordCopy_2_bits_vd_valid & (chainingRecordCopy_2_bits_vd_bits == chainingRecordCopy_3_bits_vs2 | chainingRecordCopy_2_bits_vd_bits == chainingRecordCopy_3_bits_vs1_bits & chainingRecordCopy_3_bits_vs1_valid);
  wire         hazardVec_sinkVdEqSourceVs_7 =
    chainingRecordCopy_3_bits_vd_valid & (chainingRecordCopy_3_bits_vd_bits == chainingRecordCopy_2_bits_vs2 | chainingRecordCopy_3_bits_vd_bits == chainingRecordCopy_2_bits_vs1_bits & chainingRecordCopy_2_bits_vs1_valid);
  wire         hazardVec_older_7 =
    chainingRecordCopy_3_bits_instIndex == chainingRecordCopy_2_bits_instIndex | chainingRecordCopy_3_bits_instIndex[1:0] < chainingRecordCopy_2_bits_instIndex[1:0] ^ chainingRecordCopy_3_bits_instIndex[2]
    ^ chainingRecordCopy_2_bits_instIndex[2];
  wire         hazardVec_hazardForeLoad_7 =
    (hazardVec_older_7 ? hazardVec_isLoad_0_7 & hazardVec_isSlow_1_7 : hazardVec_isLoad_1_7 & hazardVec_isSlow_0_7) & (hazardVec_samVd_7 | (hazardVec_older_7 ? hazardVec_sourceVdEqSinkVs_7 : hazardVec_sinkVdEqSourceVs_7));
  wire         hazardVec_rawForeStore_7 = (hazardVec_older_7 ? hazardVec_isStore_0_7 & hazardVec_isSlow_1_7 : hazardVec_isStore_1_7 & hazardVec_isSlow_0_7) & hazardVec_samVd_7;
  wire         hazardVec_samVd_8 =
    chainingRecordCopy_2_bits_vd_valid & chainingRecordCopy_4_bits_vd_valid & chainingRecordCopy_2_bits_vd_bits[4:3] == chainingRecordCopy_4_bits_vd_bits[4:3]
    & (chainingRecordCopy_2_bits_elementMask | chainingRecordCopy_4_bits_elementMask) != 256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
  wire         hazardVec_sourceVdEqSinkVs_8 =
    chainingRecordCopy_2_bits_vd_valid & (chainingRecordCopy_2_bits_vd_bits == chainingRecordCopy_4_bits_vs2 | chainingRecordCopy_2_bits_vd_bits == chainingRecordCopy_4_bits_vs1_bits & chainingRecordCopy_4_bits_vs1_valid);
  wire         hazardVec_sinkVdEqSourceVs_8 =
    chainingRecordCopy_4_bits_vd_valid & (chainingRecordCopy_4_bits_vd_bits == chainingRecordCopy_2_bits_vs2 | chainingRecordCopy_4_bits_vd_bits == chainingRecordCopy_2_bits_vs1_bits & chainingRecordCopy_2_bits_vs1_valid);
  wire         hazardVec_older_8 =
    chainingRecordCopy_4_bits_instIndex == chainingRecordCopy_2_bits_instIndex | chainingRecordCopy_4_bits_instIndex[1:0] < chainingRecordCopy_2_bits_instIndex[1:0] ^ chainingRecordCopy_4_bits_instIndex[2]
    ^ chainingRecordCopy_2_bits_instIndex[2];
  wire         hazardVec_hazardForeLoad_8 =
    (hazardVec_older_8 ? hazardVec_isLoad_0_8 & hazardVec_isSlow_1_8 : hazardVec_isLoad_1_8 & hazardVec_isSlow_0_8) & (hazardVec_samVd_8 | (hazardVec_older_8 ? hazardVec_sourceVdEqSinkVs_8 : hazardVec_sinkVdEqSourceVs_8));
  wire         hazardVec_rawForeStore_8 = (hazardVec_older_8 ? hazardVec_isStore_0_8 & hazardVec_isSlow_1_8 : hazardVec_isStore_1_8 & hazardVec_isSlow_0_8) & hazardVec_samVd_8;
  wire         hazardVec_samVd_9 =
    chainingRecordCopy_3_bits_vd_valid & chainingRecordCopy_4_bits_vd_valid & chainingRecordCopy_3_bits_vd_bits[4:3] == chainingRecordCopy_4_bits_vd_bits[4:3]
    & (chainingRecordCopy_3_bits_elementMask | chainingRecordCopy_4_bits_elementMask) != 256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
  wire         hazardVec_sourceVdEqSinkVs_9 =
    chainingRecordCopy_3_bits_vd_valid & (chainingRecordCopy_3_bits_vd_bits == chainingRecordCopy_4_bits_vs2 | chainingRecordCopy_3_bits_vd_bits == chainingRecordCopy_4_bits_vs1_bits & chainingRecordCopy_4_bits_vs1_valid);
  wire         hazardVec_sinkVdEqSourceVs_9 =
    chainingRecordCopy_4_bits_vd_valid & (chainingRecordCopy_4_bits_vd_bits == chainingRecordCopy_3_bits_vs2 | chainingRecordCopy_4_bits_vd_bits == chainingRecordCopy_3_bits_vs1_bits & chainingRecordCopy_3_bits_vs1_valid);
  wire         hazardVec_older_9 =
    chainingRecordCopy_4_bits_instIndex == chainingRecordCopy_3_bits_instIndex | chainingRecordCopy_4_bits_instIndex[1:0] < chainingRecordCopy_3_bits_instIndex[1:0] ^ chainingRecordCopy_4_bits_instIndex[2]
    ^ chainingRecordCopy_3_bits_instIndex[2];
  wire         hazardVec_hazardForeLoad_9 =
    (hazardVec_older_9 ? hazardVec_isLoad_0_9 & hazardVec_isSlow_1_9 : hazardVec_isLoad_1_9 & hazardVec_isSlow_0_9) & (hazardVec_samVd_9 | (hazardVec_older_9 ? hazardVec_sourceVdEqSinkVs_9 : hazardVec_sinkVdEqSourceVs_9));
  wire         hazardVec_rawForeStore_9 = (hazardVec_older_9 ? hazardVec_isStore_0_9 & hazardVec_isSlow_1_9 : hazardVec_isStore_1_9 & hazardVec_isSlow_0_9) & hazardVec_samVd_9;
  always @(posedge clock) begin
    if (reset) begin
      sramReady <= 1'h0;
      sramResetCount <= 10'h0;
      chainingRecord_0_valid <= 1'h0;
      chainingRecord_0_bits_vd_valid <= 1'h0;
      chainingRecord_0_bits_vd_bits <= 5'h0;
      chainingRecord_0_bits_vs1_valid <= 1'h0;
      chainingRecord_0_bits_vs1_bits <= 5'h0;
      chainingRecord_0_bits_vs2 <= 5'h0;
      chainingRecord_0_bits_instIndex <= 3'h0;
      chainingRecord_0_bits_ls <= 1'h0;
      chainingRecord_0_bits_st <= 1'h0;
      chainingRecord_0_bits_gather <= 1'h0;
      chainingRecord_0_bits_gather16 <= 1'h0;
      chainingRecord_0_bits_crossWrite <= 1'h0;
      chainingRecord_0_bits_crossRead <= 1'h0;
      chainingRecord_0_bits_indexType <= 1'h0;
      chainingRecord_0_bits_ma <= 1'h0;
      chainingRecord_0_bits_onlyRead <= 1'h0;
      chainingRecord_0_bits_slow <= 1'h0;
      chainingRecord_0_bits_elementMask <= 256'h0;
      chainingRecord_0_bits_state_stFinish <= 1'h0;
      chainingRecord_0_bits_state_wWriteQueueClear <= 1'h0;
      chainingRecord_0_bits_state_wLaneLastReport <= 1'h0;
      chainingRecord_0_bits_state_wTopLastReport <= 1'h0;
      chainingRecord_0_bits_state_wLaneClear <= 1'h0;
      chainingRecord_1_valid <= 1'h0;
      chainingRecord_1_bits_vd_valid <= 1'h0;
      chainingRecord_1_bits_vd_bits <= 5'h0;
      chainingRecord_1_bits_vs1_valid <= 1'h0;
      chainingRecord_1_bits_vs1_bits <= 5'h0;
      chainingRecord_1_bits_vs2 <= 5'h0;
      chainingRecord_1_bits_instIndex <= 3'h0;
      chainingRecord_1_bits_ls <= 1'h0;
      chainingRecord_1_bits_st <= 1'h0;
      chainingRecord_1_bits_gather <= 1'h0;
      chainingRecord_1_bits_gather16 <= 1'h0;
      chainingRecord_1_bits_crossWrite <= 1'h0;
      chainingRecord_1_bits_crossRead <= 1'h0;
      chainingRecord_1_bits_indexType <= 1'h0;
      chainingRecord_1_bits_ma <= 1'h0;
      chainingRecord_1_bits_onlyRead <= 1'h0;
      chainingRecord_1_bits_slow <= 1'h0;
      chainingRecord_1_bits_elementMask <= 256'h0;
      chainingRecord_1_bits_state_stFinish <= 1'h0;
      chainingRecord_1_bits_state_wWriteQueueClear <= 1'h0;
      chainingRecord_1_bits_state_wLaneLastReport <= 1'h0;
      chainingRecord_1_bits_state_wTopLastReport <= 1'h0;
      chainingRecord_1_bits_state_wLaneClear <= 1'h0;
      chainingRecord_2_valid <= 1'h0;
      chainingRecord_2_bits_vd_valid <= 1'h0;
      chainingRecord_2_bits_vd_bits <= 5'h0;
      chainingRecord_2_bits_vs1_valid <= 1'h0;
      chainingRecord_2_bits_vs1_bits <= 5'h0;
      chainingRecord_2_bits_vs2 <= 5'h0;
      chainingRecord_2_bits_instIndex <= 3'h0;
      chainingRecord_2_bits_ls <= 1'h0;
      chainingRecord_2_bits_st <= 1'h0;
      chainingRecord_2_bits_gather <= 1'h0;
      chainingRecord_2_bits_gather16 <= 1'h0;
      chainingRecord_2_bits_crossWrite <= 1'h0;
      chainingRecord_2_bits_crossRead <= 1'h0;
      chainingRecord_2_bits_indexType <= 1'h0;
      chainingRecord_2_bits_ma <= 1'h0;
      chainingRecord_2_bits_onlyRead <= 1'h0;
      chainingRecord_2_bits_slow <= 1'h0;
      chainingRecord_2_bits_elementMask <= 256'h0;
      chainingRecord_2_bits_state_stFinish <= 1'h0;
      chainingRecord_2_bits_state_wWriteQueueClear <= 1'h0;
      chainingRecord_2_bits_state_wLaneLastReport <= 1'h0;
      chainingRecord_2_bits_state_wTopLastReport <= 1'h0;
      chainingRecord_2_bits_state_wLaneClear <= 1'h0;
      chainingRecord_3_valid <= 1'h0;
      chainingRecord_3_bits_vd_valid <= 1'h0;
      chainingRecord_3_bits_vd_bits <= 5'h0;
      chainingRecord_3_bits_vs1_valid <= 1'h0;
      chainingRecord_3_bits_vs1_bits <= 5'h0;
      chainingRecord_3_bits_vs2 <= 5'h0;
      chainingRecord_3_bits_instIndex <= 3'h0;
      chainingRecord_3_bits_ls <= 1'h0;
      chainingRecord_3_bits_st <= 1'h0;
      chainingRecord_3_bits_gather <= 1'h0;
      chainingRecord_3_bits_gather16 <= 1'h0;
      chainingRecord_3_bits_crossWrite <= 1'h0;
      chainingRecord_3_bits_crossRead <= 1'h0;
      chainingRecord_3_bits_indexType <= 1'h0;
      chainingRecord_3_bits_ma <= 1'h0;
      chainingRecord_3_bits_onlyRead <= 1'h0;
      chainingRecord_3_bits_slow <= 1'h0;
      chainingRecord_3_bits_elementMask <= 256'h0;
      chainingRecord_3_bits_state_stFinish <= 1'h0;
      chainingRecord_3_bits_state_wWriteQueueClear <= 1'h0;
      chainingRecord_3_bits_state_wLaneLastReport <= 1'h0;
      chainingRecord_3_bits_state_wTopLastReport <= 1'h0;
      chainingRecord_3_bits_state_wLaneClear <= 1'h0;
      chainingRecord_4_valid <= 1'h0;
      chainingRecord_4_bits_vd_valid <= 1'h0;
      chainingRecord_4_bits_vd_bits <= 5'h0;
      chainingRecord_4_bits_vs1_valid <= 1'h0;
      chainingRecord_4_bits_vs1_bits <= 5'h0;
      chainingRecord_4_bits_vs2 <= 5'h0;
      chainingRecord_4_bits_instIndex <= 3'h0;
      chainingRecord_4_bits_ls <= 1'h0;
      chainingRecord_4_bits_st <= 1'h0;
      chainingRecord_4_bits_gather <= 1'h0;
      chainingRecord_4_bits_gather16 <= 1'h0;
      chainingRecord_4_bits_crossWrite <= 1'h0;
      chainingRecord_4_bits_crossRead <= 1'h0;
      chainingRecord_4_bits_indexType <= 1'h0;
      chainingRecord_4_bits_ma <= 1'h0;
      chainingRecord_4_bits_onlyRead <= 1'h0;
      chainingRecord_4_bits_slow <= 1'h0;
      chainingRecord_4_bits_elementMask <= 256'h0;
      chainingRecord_4_bits_state_stFinish <= 1'h0;
      chainingRecord_4_bits_state_wWriteQueueClear <= 1'h0;
      chainingRecord_4_bits_state_wLaneLastReport <= 1'h0;
      chainingRecord_4_bits_state_wTopLastReport <= 1'h0;
      chainingRecord_4_bits_state_wLaneClear <= 1'h0;
      chainingRecordCopy_0_valid <= 1'h0;
      chainingRecordCopy_0_bits_vd_valid <= 1'h0;
      chainingRecordCopy_0_bits_vd_bits <= 5'h0;
      chainingRecordCopy_0_bits_vs1_valid <= 1'h0;
      chainingRecordCopy_0_bits_vs1_bits <= 5'h0;
      chainingRecordCopy_0_bits_vs2 <= 5'h0;
      chainingRecordCopy_0_bits_instIndex <= 3'h0;
      chainingRecordCopy_0_bits_ls <= 1'h0;
      chainingRecordCopy_0_bits_st <= 1'h0;
      chainingRecordCopy_0_bits_gather <= 1'h0;
      chainingRecordCopy_0_bits_gather16 <= 1'h0;
      chainingRecordCopy_0_bits_crossWrite <= 1'h0;
      chainingRecordCopy_0_bits_crossRead <= 1'h0;
      chainingRecordCopy_0_bits_indexType <= 1'h0;
      chainingRecordCopy_0_bits_ma <= 1'h0;
      chainingRecordCopy_0_bits_onlyRead <= 1'h0;
      chainingRecordCopy_0_bits_slow <= 1'h0;
      chainingRecordCopy_0_bits_elementMask <= 256'h0;
      chainingRecordCopy_0_bits_state_stFinish <= 1'h0;
      chainingRecordCopy_0_bits_state_wWriteQueueClear <= 1'h0;
      chainingRecordCopy_0_bits_state_wLaneLastReport <= 1'h0;
      chainingRecordCopy_0_bits_state_wTopLastReport <= 1'h0;
      chainingRecordCopy_0_bits_state_wLaneClear <= 1'h0;
      chainingRecordCopy_1_valid <= 1'h0;
      chainingRecordCopy_1_bits_vd_valid <= 1'h0;
      chainingRecordCopy_1_bits_vd_bits <= 5'h0;
      chainingRecordCopy_1_bits_vs1_valid <= 1'h0;
      chainingRecordCopy_1_bits_vs1_bits <= 5'h0;
      chainingRecordCopy_1_bits_vs2 <= 5'h0;
      chainingRecordCopy_1_bits_instIndex <= 3'h0;
      chainingRecordCopy_1_bits_ls <= 1'h0;
      chainingRecordCopy_1_bits_st <= 1'h0;
      chainingRecordCopy_1_bits_gather <= 1'h0;
      chainingRecordCopy_1_bits_gather16 <= 1'h0;
      chainingRecordCopy_1_bits_crossWrite <= 1'h0;
      chainingRecordCopy_1_bits_crossRead <= 1'h0;
      chainingRecordCopy_1_bits_indexType <= 1'h0;
      chainingRecordCopy_1_bits_ma <= 1'h0;
      chainingRecordCopy_1_bits_onlyRead <= 1'h0;
      chainingRecordCopy_1_bits_slow <= 1'h0;
      chainingRecordCopy_1_bits_elementMask <= 256'h0;
      chainingRecordCopy_1_bits_state_stFinish <= 1'h0;
      chainingRecordCopy_1_bits_state_wWriteQueueClear <= 1'h0;
      chainingRecordCopy_1_bits_state_wLaneLastReport <= 1'h0;
      chainingRecordCopy_1_bits_state_wTopLastReport <= 1'h0;
      chainingRecordCopy_1_bits_state_wLaneClear <= 1'h0;
      chainingRecordCopy_2_valid <= 1'h0;
      chainingRecordCopy_2_bits_vd_valid <= 1'h0;
      chainingRecordCopy_2_bits_vd_bits <= 5'h0;
      chainingRecordCopy_2_bits_vs1_valid <= 1'h0;
      chainingRecordCopy_2_bits_vs1_bits <= 5'h0;
      chainingRecordCopy_2_bits_vs2 <= 5'h0;
      chainingRecordCopy_2_bits_instIndex <= 3'h0;
      chainingRecordCopy_2_bits_ls <= 1'h0;
      chainingRecordCopy_2_bits_st <= 1'h0;
      chainingRecordCopy_2_bits_gather <= 1'h0;
      chainingRecordCopy_2_bits_gather16 <= 1'h0;
      chainingRecordCopy_2_bits_crossWrite <= 1'h0;
      chainingRecordCopy_2_bits_crossRead <= 1'h0;
      chainingRecordCopy_2_bits_indexType <= 1'h0;
      chainingRecordCopy_2_bits_ma <= 1'h0;
      chainingRecordCopy_2_bits_onlyRead <= 1'h0;
      chainingRecordCopy_2_bits_slow <= 1'h0;
      chainingRecordCopy_2_bits_elementMask <= 256'h0;
      chainingRecordCopy_2_bits_state_stFinish <= 1'h0;
      chainingRecordCopy_2_bits_state_wWriteQueueClear <= 1'h0;
      chainingRecordCopy_2_bits_state_wLaneLastReport <= 1'h0;
      chainingRecordCopy_2_bits_state_wTopLastReport <= 1'h0;
      chainingRecordCopy_2_bits_state_wLaneClear <= 1'h0;
      chainingRecordCopy_3_valid <= 1'h0;
      chainingRecordCopy_3_bits_vd_valid <= 1'h0;
      chainingRecordCopy_3_bits_vd_bits <= 5'h0;
      chainingRecordCopy_3_bits_vs1_valid <= 1'h0;
      chainingRecordCopy_3_bits_vs1_bits <= 5'h0;
      chainingRecordCopy_3_bits_vs2 <= 5'h0;
      chainingRecordCopy_3_bits_instIndex <= 3'h0;
      chainingRecordCopy_3_bits_ls <= 1'h0;
      chainingRecordCopy_3_bits_st <= 1'h0;
      chainingRecordCopy_3_bits_gather <= 1'h0;
      chainingRecordCopy_3_bits_gather16 <= 1'h0;
      chainingRecordCopy_3_bits_crossWrite <= 1'h0;
      chainingRecordCopy_3_bits_crossRead <= 1'h0;
      chainingRecordCopy_3_bits_indexType <= 1'h0;
      chainingRecordCopy_3_bits_ma <= 1'h0;
      chainingRecordCopy_3_bits_onlyRead <= 1'h0;
      chainingRecordCopy_3_bits_slow <= 1'h0;
      chainingRecordCopy_3_bits_elementMask <= 256'h0;
      chainingRecordCopy_3_bits_state_stFinish <= 1'h0;
      chainingRecordCopy_3_bits_state_wWriteQueueClear <= 1'h0;
      chainingRecordCopy_3_bits_state_wLaneLastReport <= 1'h0;
      chainingRecordCopy_3_bits_state_wTopLastReport <= 1'h0;
      chainingRecordCopy_3_bits_state_wLaneClear <= 1'h0;
      chainingRecordCopy_4_valid <= 1'h0;
      chainingRecordCopy_4_bits_vd_valid <= 1'h0;
      chainingRecordCopy_4_bits_vd_bits <= 5'h0;
      chainingRecordCopy_4_bits_vs1_valid <= 1'h0;
      chainingRecordCopy_4_bits_vs1_bits <= 5'h0;
      chainingRecordCopy_4_bits_vs2 <= 5'h0;
      chainingRecordCopy_4_bits_instIndex <= 3'h0;
      chainingRecordCopy_4_bits_ls <= 1'h0;
      chainingRecordCopy_4_bits_st <= 1'h0;
      chainingRecordCopy_4_bits_gather <= 1'h0;
      chainingRecordCopy_4_bits_gather16 <= 1'h0;
      chainingRecordCopy_4_bits_crossWrite <= 1'h0;
      chainingRecordCopy_4_bits_crossRead <= 1'h0;
      chainingRecordCopy_4_bits_indexType <= 1'h0;
      chainingRecordCopy_4_bits_ma <= 1'h0;
      chainingRecordCopy_4_bits_onlyRead <= 1'h0;
      chainingRecordCopy_4_bits_slow <= 1'h0;
      chainingRecordCopy_4_bits_elementMask <= 256'h0;
      chainingRecordCopy_4_bits_state_stFinish <= 1'h0;
      chainingRecordCopy_4_bits_state_wWriteQueueClear <= 1'h0;
      chainingRecordCopy_4_bits_state_wLaneLastReport <= 1'h0;
      chainingRecordCopy_4_bits_state_wTopLastReport <= 1'h0;
      chainingRecordCopy_4_bits_state_wLaneClear <= 1'h0;
      firstReadPipe_0_valid <= 1'h0;
      firstReadPipe_0_bits_address <= 10'h0;
      writePipe_valid <= 1'h0;
      writePipe_bits_vd <= 5'h0;
      writePipe_bits_offset <= 5'h0;
      writePipe_bits_mask <= 4'h0;
      writePipe_bits_data <= 32'h0;
      writePipe_bits_last <= 1'h0;
      writePipe_bits_instructionIndex <= 3'h0;
      pipeBank_pipe_v <= 1'h0;
      pipeBank_pipe_pipe_v <= 1'h0;
      pipeFirstUsed_pipe_v <= 1'h0;
      pipeFirstUsed_pipe_pipe_v <= 1'h0;
      pipeFire_pipe_v <= 1'h0;
      pipeFire_pipe_pipe_v <= 1'h0;
      pipeBank_pipe_v_1 <= 1'h0;
      pipeBank_pipe_pipe_v_1 <= 1'h0;
      pipeFirstUsed_pipe_v_1 <= 1'h0;
      pipeFirstUsed_pipe_pipe_v_1 <= 1'h0;
      pipeFire_pipe_v_1 <= 1'h0;
      pipeFire_pipe_pipe_v_1 <= 1'h0;
      pipeBank_pipe_v_2 <= 1'h0;
      pipeBank_pipe_pipe_v_2 <= 1'h0;
      pipeFirstUsed_pipe_v_2 <= 1'h0;
      pipeFirstUsed_pipe_pipe_v_2 <= 1'h0;
      pipeFire_pipe_v_2 <= 1'h0;
      pipeFire_pipe_pipe_v_2 <= 1'h0;
      pipeBank_pipe_v_3 <= 1'h0;
      pipeBank_pipe_pipe_v_3 <= 1'h0;
      pipeFirstUsed_pipe_v_3 <= 1'h0;
      pipeFirstUsed_pipe_pipe_v_3 <= 1'h0;
      pipeFire_pipe_v_3 <= 1'h0;
      pipeFire_pipe_pipe_v_3 <= 1'h0;
      pipeBank_pipe_v_4 <= 1'h0;
      pipeBank_pipe_pipe_v_4 <= 1'h0;
      pipeFirstUsed_pipe_v_4 <= 1'h0;
      pipeFirstUsed_pipe_pipe_v_4 <= 1'h0;
      pipeFire_pipe_v_4 <= 1'h0;
      pipeFire_pipe_pipe_v_4 <= 1'h0;
      pipeBank_pipe_v_5 <= 1'h0;
      pipeBank_pipe_pipe_v_5 <= 1'h0;
      pipeFirstUsed_pipe_v_5 <= 1'h0;
      pipeFirstUsed_pipe_pipe_v_5 <= 1'h0;
      pipeFire_pipe_v_5 <= 1'h0;
      pipeFire_pipe_pipe_v_5 <= 1'h0;
      pipeBank_pipe_v_6 <= 1'h0;
      pipeBank_pipe_pipe_v_6 <= 1'h0;
      pipeFirstUsed_pipe_v_6 <= 1'h0;
      pipeFirstUsed_pipe_pipe_v_6 <= 1'h0;
      pipeFire_pipe_v_6 <= 1'h0;
      pipeFire_pipe_pipe_v_6 <= 1'h0;
      pipeBank_pipe_v_7 <= 1'h0;
      pipeBank_pipe_pipe_v_7 <= 1'h0;
      pipeFirstUsed_pipe_v_7 <= 1'h0;
      pipeFirstUsed_pipe_pipe_v_7 <= 1'h0;
      pipeFire_pipe_v_7 <= 1'h0;
      pipeFire_pipe_pipe_v_7 <= 1'h0;
      pipeBank_pipe_v_8 <= 1'h0;
      pipeBank_pipe_pipe_v_8 <= 1'h0;
      pipeFirstUsed_pipe_v_8 <= 1'h0;
      pipeFirstUsed_pipe_pipe_v_8 <= 1'h0;
      pipeFire_pipe_v_8 <= 1'h0;
      pipeFire_pipe_pipe_v_8 <= 1'h0;
      pipeBank_pipe_v_9 <= 1'h0;
      pipeBank_pipe_pipe_v_9 <= 1'h0;
      pipeFirstUsed_pipe_v_9 <= 1'h0;
      pipeFirstUsed_pipe_pipe_v_9 <= 1'h0;
      pipeFire_pipe_v_9 <= 1'h0;
      pipeFire_pipe_pipe_v_9 <= 1'h0;
      pipeBank_pipe_v_10 <= 1'h0;
      pipeBank_pipe_pipe_v_10 <= 1'h0;
      pipeFirstUsed_pipe_v_10 <= 1'h0;
      pipeFirstUsed_pipe_pipe_v_10 <= 1'h0;
      pipeFire_pipe_v_10 <= 1'h0;
      pipeFire_pipe_pipe_v_10 <= 1'h0;
      pipeBank_pipe_v_11 <= 1'h0;
      pipeBank_pipe_pipe_v_11 <= 1'h0;
      pipeFirstUsed_pipe_v_11 <= 1'h0;
      pipeFirstUsed_pipe_pipe_v_11 <= 1'h0;
      pipeFire_pipe_v_11 <= 1'h0;
      pipeFire_pipe_pipe_v_11 <= 1'h0;
      pipeBank_pipe_v_12 <= 1'h0;
      pipeBank_pipe_pipe_v_12 <= 1'h0;
      pipeFirstUsed_pipe_v_12 <= 1'h0;
      pipeFirstUsed_pipe_pipe_v_12 <= 1'h0;
      pipeFire_pipe_v_12 <= 1'h0;
      pipeFire_pipe_pipe_v_12 <= 1'h0;
      pipeBank_pipe_v_13 <= 1'h0;
      pipeBank_pipe_pipe_v_13 <= 1'h0;
      pipeFirstUsed_pipe_v_13 <= 1'h0;
      pipeFirstUsed_pipe_pipe_v_13 <= 1'h0;
      pipeFire_pipe_v_13 <= 1'h0;
      pipeFire_pipe_pipe_v_13 <= 1'h0;
    end
    else begin
      sramReady <= resetValid & (&sramResetCount) | sramReady;
      if (resetValid)
        sramResetCount <= sramResetCount + 10'h1;
      chainingRecord_0_valid <= recordEnq[0] | ~stateClear & chainingRecord_0_valid;
      if (recordEnq[0]) begin
        chainingRecord_0_bits_vd_valid <= initRecord_bits_vd_valid;
        chainingRecord_0_bits_vd_bits <= initRecord_bits_vd_bits;
        chainingRecord_0_bits_vs1_valid <= initRecord_bits_vs1_valid;
        chainingRecord_0_bits_vs1_bits <= initRecord_bits_vs1_bits;
        chainingRecord_0_bits_vs2 <= initRecord_bits_vs2;
        chainingRecord_0_bits_instIndex <= initRecord_bits_instIndex;
        chainingRecord_0_bits_ls <= initRecord_bits_ls;
        chainingRecord_0_bits_st <= initRecord_bits_st;
        chainingRecord_0_bits_gather <= initRecord_bits_gather;
        chainingRecord_0_bits_gather16 <= initRecord_bits_gather16;
        chainingRecord_0_bits_crossWrite <= initRecord_bits_crossWrite;
        chainingRecord_0_bits_crossRead <= initRecord_bits_crossRead;
        chainingRecord_0_bits_indexType <= initRecord_bits_indexType;
        chainingRecord_0_bits_ma <= initRecord_bits_ma;
        chainingRecord_0_bits_onlyRead <= initRecord_bits_onlyRead;
        chainingRecord_0_bits_slow <= initRecord_bits_slow;
        chainingRecordCopy_0_bits_vd_valid <= initRecord_bits_vd_valid;
        chainingRecordCopy_0_bits_vd_bits <= initRecord_bits_vd_bits;
        chainingRecordCopy_0_bits_vs1_valid <= initRecord_bits_vs1_valid;
        chainingRecordCopy_0_bits_vs1_bits <= initRecord_bits_vs1_bits;
        chainingRecordCopy_0_bits_vs2 <= initRecord_bits_vs2;
        chainingRecordCopy_0_bits_instIndex <= initRecord_bits_instIndex;
        chainingRecordCopy_0_bits_ls <= initRecord_bits_ls;
        chainingRecordCopy_0_bits_st <= initRecord_bits_st;
        chainingRecordCopy_0_bits_gather <= initRecord_bits_gather;
        chainingRecordCopy_0_bits_gather16 <= initRecord_bits_gather16;
        chainingRecordCopy_0_bits_crossWrite <= initRecord_bits_crossWrite;
        chainingRecordCopy_0_bits_crossRead <= initRecord_bits_crossRead;
        chainingRecordCopy_0_bits_indexType <= initRecord_bits_indexType;
        chainingRecordCopy_0_bits_ma <= initRecord_bits_ma;
        chainingRecordCopy_0_bits_onlyRead <= initRecord_bits_onlyRead;
        chainingRecordCopy_0_bits_slow <= initRecord_bits_slow;
      end
      chainingRecord_0_bits_elementMask <= recordEnq[0] ? initRecord_bits_elementMask : {256{elementUpdateValid}} & elementUpdate1H | chainingRecord_0_bits_elementMask;
      chainingRecord_0_bits_state_stFinish <= recordEnq[0] ? initRecord_bits_state_stFinish : topLastReport | chainingRecord_0_bits_state_stFinish;
      chainingRecord_0_bits_state_wWriteQueueClear <= recordEnq[0] ? initRecord_bits_state_wWriteQueueClear : chainingRecord_0_bits_state_stFinish & ~dataInLsuQueue | chainingRecord_0_bits_state_wWriteQueueClear;
      chainingRecord_0_bits_state_wLaneLastReport <= recordEnq[0] ? initRecord_bits_state_wLaneLastReport : laneLastReport | chainingRecord_0_bits_state_wLaneLastReport;
      chainingRecord_0_bits_state_wTopLastReport <= recordEnq[0] ? initRecord_bits_state_wTopLastReport : topLastReport | chainingRecord_0_bits_state_wTopLastReport;
      chainingRecord_0_bits_state_wLaneClear <= ~(recordEnq[0]) & (waitLaneClear & ~dataInLaneCheck | chainingRecord_0_bits_state_wLaneClear);
      chainingRecord_1_valid <= recordEnq[1] | ~stateClear_1 & chainingRecord_1_valid;
      if (recordEnq[1]) begin
        chainingRecord_1_bits_vd_valid <= initRecord_bits_vd_valid;
        chainingRecord_1_bits_vd_bits <= initRecord_bits_vd_bits;
        chainingRecord_1_bits_vs1_valid <= initRecord_bits_vs1_valid;
        chainingRecord_1_bits_vs1_bits <= initRecord_bits_vs1_bits;
        chainingRecord_1_bits_vs2 <= initRecord_bits_vs2;
        chainingRecord_1_bits_instIndex <= initRecord_bits_instIndex;
        chainingRecord_1_bits_ls <= initRecord_bits_ls;
        chainingRecord_1_bits_st <= initRecord_bits_st;
        chainingRecord_1_bits_gather <= initRecord_bits_gather;
        chainingRecord_1_bits_gather16 <= initRecord_bits_gather16;
        chainingRecord_1_bits_crossWrite <= initRecord_bits_crossWrite;
        chainingRecord_1_bits_crossRead <= initRecord_bits_crossRead;
        chainingRecord_1_bits_indexType <= initRecord_bits_indexType;
        chainingRecord_1_bits_ma <= initRecord_bits_ma;
        chainingRecord_1_bits_onlyRead <= initRecord_bits_onlyRead;
        chainingRecord_1_bits_slow <= initRecord_bits_slow;
        chainingRecordCopy_1_bits_vd_valid <= initRecord_bits_vd_valid;
        chainingRecordCopy_1_bits_vd_bits <= initRecord_bits_vd_bits;
        chainingRecordCopy_1_bits_vs1_valid <= initRecord_bits_vs1_valid;
        chainingRecordCopy_1_bits_vs1_bits <= initRecord_bits_vs1_bits;
        chainingRecordCopy_1_bits_vs2 <= initRecord_bits_vs2;
        chainingRecordCopy_1_bits_instIndex <= initRecord_bits_instIndex;
        chainingRecordCopy_1_bits_ls <= initRecord_bits_ls;
        chainingRecordCopy_1_bits_st <= initRecord_bits_st;
        chainingRecordCopy_1_bits_gather <= initRecord_bits_gather;
        chainingRecordCopy_1_bits_gather16 <= initRecord_bits_gather16;
        chainingRecordCopy_1_bits_crossWrite <= initRecord_bits_crossWrite;
        chainingRecordCopy_1_bits_crossRead <= initRecord_bits_crossRead;
        chainingRecordCopy_1_bits_indexType <= initRecord_bits_indexType;
        chainingRecordCopy_1_bits_ma <= initRecord_bits_ma;
        chainingRecordCopy_1_bits_onlyRead <= initRecord_bits_onlyRead;
        chainingRecordCopy_1_bits_slow <= initRecord_bits_slow;
      end
      chainingRecord_1_bits_elementMask <= recordEnq[1] ? initRecord_bits_elementMask : {256{elementUpdateValid_1}} & elementUpdate1H_1 | chainingRecord_1_bits_elementMask;
      chainingRecord_1_bits_state_stFinish <= recordEnq[1] ? initRecord_bits_state_stFinish : topLastReport_1 | chainingRecord_1_bits_state_stFinish;
      chainingRecord_1_bits_state_wWriteQueueClear <= recordEnq[1] ? initRecord_bits_state_wWriteQueueClear : chainingRecord_1_bits_state_stFinish & ~dataInLsuQueue_1 | chainingRecord_1_bits_state_wWriteQueueClear;
      chainingRecord_1_bits_state_wLaneLastReport <= recordEnq[1] ? initRecord_bits_state_wLaneLastReport : laneLastReport_1 | chainingRecord_1_bits_state_wLaneLastReport;
      chainingRecord_1_bits_state_wTopLastReport <= recordEnq[1] ? initRecord_bits_state_wTopLastReport : topLastReport_1 | chainingRecord_1_bits_state_wTopLastReport;
      chainingRecord_1_bits_state_wLaneClear <= ~(recordEnq[1]) & (waitLaneClear_1 & ~dataInLaneCheck_1 | chainingRecord_1_bits_state_wLaneClear);
      chainingRecord_2_valid <= recordEnq[2] | ~stateClear_2 & chainingRecord_2_valid;
      if (recordEnq[2]) begin
        chainingRecord_2_bits_vd_valid <= initRecord_bits_vd_valid;
        chainingRecord_2_bits_vd_bits <= initRecord_bits_vd_bits;
        chainingRecord_2_bits_vs1_valid <= initRecord_bits_vs1_valid;
        chainingRecord_2_bits_vs1_bits <= initRecord_bits_vs1_bits;
        chainingRecord_2_bits_vs2 <= initRecord_bits_vs2;
        chainingRecord_2_bits_instIndex <= initRecord_bits_instIndex;
        chainingRecord_2_bits_ls <= initRecord_bits_ls;
        chainingRecord_2_bits_st <= initRecord_bits_st;
        chainingRecord_2_bits_gather <= initRecord_bits_gather;
        chainingRecord_2_bits_gather16 <= initRecord_bits_gather16;
        chainingRecord_2_bits_crossWrite <= initRecord_bits_crossWrite;
        chainingRecord_2_bits_crossRead <= initRecord_bits_crossRead;
        chainingRecord_2_bits_indexType <= initRecord_bits_indexType;
        chainingRecord_2_bits_ma <= initRecord_bits_ma;
        chainingRecord_2_bits_onlyRead <= initRecord_bits_onlyRead;
        chainingRecord_2_bits_slow <= initRecord_bits_slow;
        chainingRecordCopy_2_bits_vd_valid <= initRecord_bits_vd_valid;
        chainingRecordCopy_2_bits_vd_bits <= initRecord_bits_vd_bits;
        chainingRecordCopy_2_bits_vs1_valid <= initRecord_bits_vs1_valid;
        chainingRecordCopy_2_bits_vs1_bits <= initRecord_bits_vs1_bits;
        chainingRecordCopy_2_bits_vs2 <= initRecord_bits_vs2;
        chainingRecordCopy_2_bits_instIndex <= initRecord_bits_instIndex;
        chainingRecordCopy_2_bits_ls <= initRecord_bits_ls;
        chainingRecordCopy_2_bits_st <= initRecord_bits_st;
        chainingRecordCopy_2_bits_gather <= initRecord_bits_gather;
        chainingRecordCopy_2_bits_gather16 <= initRecord_bits_gather16;
        chainingRecordCopy_2_bits_crossWrite <= initRecord_bits_crossWrite;
        chainingRecordCopy_2_bits_crossRead <= initRecord_bits_crossRead;
        chainingRecordCopy_2_bits_indexType <= initRecord_bits_indexType;
        chainingRecordCopy_2_bits_ma <= initRecord_bits_ma;
        chainingRecordCopy_2_bits_onlyRead <= initRecord_bits_onlyRead;
        chainingRecordCopy_2_bits_slow <= initRecord_bits_slow;
      end
      chainingRecord_2_bits_elementMask <= recordEnq[2] ? initRecord_bits_elementMask : {256{elementUpdateValid_2}} & elementUpdate1H_2 | chainingRecord_2_bits_elementMask;
      chainingRecord_2_bits_state_stFinish <= recordEnq[2] ? initRecord_bits_state_stFinish : topLastReport_2 | chainingRecord_2_bits_state_stFinish;
      chainingRecord_2_bits_state_wWriteQueueClear <= recordEnq[2] ? initRecord_bits_state_wWriteQueueClear : chainingRecord_2_bits_state_stFinish & ~dataInLsuQueue_2 | chainingRecord_2_bits_state_wWriteQueueClear;
      chainingRecord_2_bits_state_wLaneLastReport <= recordEnq[2] ? initRecord_bits_state_wLaneLastReport : laneLastReport_2 | chainingRecord_2_bits_state_wLaneLastReport;
      chainingRecord_2_bits_state_wTopLastReport <= recordEnq[2] ? initRecord_bits_state_wTopLastReport : topLastReport_2 | chainingRecord_2_bits_state_wTopLastReport;
      chainingRecord_2_bits_state_wLaneClear <= ~(recordEnq[2]) & (waitLaneClear_2 & ~dataInLaneCheck_2 | chainingRecord_2_bits_state_wLaneClear);
      chainingRecord_3_valid <= recordEnq[3] | ~stateClear_3 & chainingRecord_3_valid;
      if (recordEnq[3]) begin
        chainingRecord_3_bits_vd_valid <= initRecord_bits_vd_valid;
        chainingRecord_3_bits_vd_bits <= initRecord_bits_vd_bits;
        chainingRecord_3_bits_vs1_valid <= initRecord_bits_vs1_valid;
        chainingRecord_3_bits_vs1_bits <= initRecord_bits_vs1_bits;
        chainingRecord_3_bits_vs2 <= initRecord_bits_vs2;
        chainingRecord_3_bits_instIndex <= initRecord_bits_instIndex;
        chainingRecord_3_bits_ls <= initRecord_bits_ls;
        chainingRecord_3_bits_st <= initRecord_bits_st;
        chainingRecord_3_bits_gather <= initRecord_bits_gather;
        chainingRecord_3_bits_gather16 <= initRecord_bits_gather16;
        chainingRecord_3_bits_crossWrite <= initRecord_bits_crossWrite;
        chainingRecord_3_bits_crossRead <= initRecord_bits_crossRead;
        chainingRecord_3_bits_indexType <= initRecord_bits_indexType;
        chainingRecord_3_bits_ma <= initRecord_bits_ma;
        chainingRecord_3_bits_onlyRead <= initRecord_bits_onlyRead;
        chainingRecord_3_bits_slow <= initRecord_bits_slow;
        chainingRecordCopy_3_bits_vd_valid <= initRecord_bits_vd_valid;
        chainingRecordCopy_3_bits_vd_bits <= initRecord_bits_vd_bits;
        chainingRecordCopy_3_bits_vs1_valid <= initRecord_bits_vs1_valid;
        chainingRecordCopy_3_bits_vs1_bits <= initRecord_bits_vs1_bits;
        chainingRecordCopy_3_bits_vs2 <= initRecord_bits_vs2;
        chainingRecordCopy_3_bits_instIndex <= initRecord_bits_instIndex;
        chainingRecordCopy_3_bits_ls <= initRecord_bits_ls;
        chainingRecordCopy_3_bits_st <= initRecord_bits_st;
        chainingRecordCopy_3_bits_gather <= initRecord_bits_gather;
        chainingRecordCopy_3_bits_gather16 <= initRecord_bits_gather16;
        chainingRecordCopy_3_bits_crossWrite <= initRecord_bits_crossWrite;
        chainingRecordCopy_3_bits_crossRead <= initRecord_bits_crossRead;
        chainingRecordCopy_3_bits_indexType <= initRecord_bits_indexType;
        chainingRecordCopy_3_bits_ma <= initRecord_bits_ma;
        chainingRecordCopy_3_bits_onlyRead <= initRecord_bits_onlyRead;
        chainingRecordCopy_3_bits_slow <= initRecord_bits_slow;
      end
      chainingRecord_3_bits_elementMask <= recordEnq[3] ? initRecord_bits_elementMask : {256{elementUpdateValid_3}} & elementUpdate1H_3 | chainingRecord_3_bits_elementMask;
      chainingRecord_3_bits_state_stFinish <= recordEnq[3] ? initRecord_bits_state_stFinish : topLastReport_3 | chainingRecord_3_bits_state_stFinish;
      chainingRecord_3_bits_state_wWriteQueueClear <= recordEnq[3] ? initRecord_bits_state_wWriteQueueClear : chainingRecord_3_bits_state_stFinish & ~dataInLsuQueue_3 | chainingRecord_3_bits_state_wWriteQueueClear;
      chainingRecord_3_bits_state_wLaneLastReport <= recordEnq[3] ? initRecord_bits_state_wLaneLastReport : laneLastReport_3 | chainingRecord_3_bits_state_wLaneLastReport;
      chainingRecord_3_bits_state_wTopLastReport <= recordEnq[3] ? initRecord_bits_state_wTopLastReport : topLastReport_3 | chainingRecord_3_bits_state_wTopLastReport;
      chainingRecord_3_bits_state_wLaneClear <= ~(recordEnq[3]) & (waitLaneClear_3 & ~dataInLaneCheck_3 | chainingRecord_3_bits_state_wLaneClear);
      chainingRecord_4_valid <= recordEnq[4] | ~stateClear_4 & chainingRecord_4_valid;
      if (recordEnq[4]) begin
        chainingRecord_4_bits_vd_valid <= initRecord_bits_vd_valid;
        chainingRecord_4_bits_vd_bits <= initRecord_bits_vd_bits;
        chainingRecord_4_bits_vs1_valid <= initRecord_bits_vs1_valid;
        chainingRecord_4_bits_vs1_bits <= initRecord_bits_vs1_bits;
        chainingRecord_4_bits_vs2 <= initRecord_bits_vs2;
        chainingRecord_4_bits_instIndex <= initRecord_bits_instIndex;
        chainingRecord_4_bits_ls <= initRecord_bits_ls;
        chainingRecord_4_bits_st <= initRecord_bits_st;
        chainingRecord_4_bits_gather <= initRecord_bits_gather;
        chainingRecord_4_bits_gather16 <= initRecord_bits_gather16;
        chainingRecord_4_bits_crossWrite <= initRecord_bits_crossWrite;
        chainingRecord_4_bits_crossRead <= initRecord_bits_crossRead;
        chainingRecord_4_bits_indexType <= initRecord_bits_indexType;
        chainingRecord_4_bits_ma <= initRecord_bits_ma;
        chainingRecord_4_bits_onlyRead <= initRecord_bits_onlyRead;
        chainingRecord_4_bits_slow <= initRecord_bits_slow;
        chainingRecordCopy_4_bits_vd_valid <= initRecord_bits_vd_valid;
        chainingRecordCopy_4_bits_vd_bits <= initRecord_bits_vd_bits;
        chainingRecordCopy_4_bits_vs1_valid <= initRecord_bits_vs1_valid;
        chainingRecordCopy_4_bits_vs1_bits <= initRecord_bits_vs1_bits;
        chainingRecordCopy_4_bits_vs2 <= initRecord_bits_vs2;
        chainingRecordCopy_4_bits_instIndex <= initRecord_bits_instIndex;
        chainingRecordCopy_4_bits_ls <= initRecord_bits_ls;
        chainingRecordCopy_4_bits_st <= initRecord_bits_st;
        chainingRecordCopy_4_bits_gather <= initRecord_bits_gather;
        chainingRecordCopy_4_bits_gather16 <= initRecord_bits_gather16;
        chainingRecordCopy_4_bits_crossWrite <= initRecord_bits_crossWrite;
        chainingRecordCopy_4_bits_crossRead <= initRecord_bits_crossRead;
        chainingRecordCopy_4_bits_indexType <= initRecord_bits_indexType;
        chainingRecordCopy_4_bits_ma <= initRecord_bits_ma;
        chainingRecordCopy_4_bits_onlyRead <= initRecord_bits_onlyRead;
        chainingRecordCopy_4_bits_slow <= initRecord_bits_slow;
      end
      chainingRecord_4_bits_elementMask <= recordEnq[4] ? initRecord_bits_elementMask : {256{elementUpdateValid_4}} & elementUpdate1H_4 | chainingRecord_4_bits_elementMask;
      chainingRecord_4_bits_state_stFinish <= recordEnq[4] ? initRecord_bits_state_stFinish : topLastReport_4 | chainingRecord_4_bits_state_stFinish;
      chainingRecord_4_bits_state_wWriteQueueClear <= recordEnq[4] ? initRecord_bits_state_wWriteQueueClear : chainingRecord_4_bits_state_stFinish & ~dataInLsuQueue_4 | chainingRecord_4_bits_state_wWriteQueueClear;
      chainingRecord_4_bits_state_wLaneLastReport <= recordEnq[4] ? initRecord_bits_state_wLaneLastReport : laneLastReport_4 | chainingRecord_4_bits_state_wLaneLastReport;
      chainingRecord_4_bits_state_wTopLastReport <= recordEnq[4] ? initRecord_bits_state_wTopLastReport : topLastReport_4 | chainingRecord_4_bits_state_wTopLastReport;
      chainingRecord_4_bits_state_wLaneClear <= ~(recordEnq[4]) & (waitLaneClear_4 & ~dataInLaneCheck_4 | chainingRecord_4_bits_state_wLaneClear);
      chainingRecordCopy_0_valid <= recordEnq[0] | ~stateClear_5 & chainingRecordCopy_0_valid;
      chainingRecordCopy_0_bits_elementMask <= recordEnq[0] ? initRecord_bits_elementMask : {256{elementUpdateValid_5}} & elementUpdate1H_5 | chainingRecordCopy_0_bits_elementMask;
      chainingRecordCopy_0_bits_state_stFinish <= recordEnq[0] ? initRecord_bits_state_stFinish : topLastReport_5 | chainingRecordCopy_0_bits_state_stFinish;
      chainingRecordCopy_0_bits_state_wWriteQueueClear <= recordEnq[0] ? initRecord_bits_state_wWriteQueueClear : chainingRecordCopy_0_bits_state_stFinish & ~dataInLsuQueue_5 | chainingRecordCopy_0_bits_state_wWriteQueueClear;
      chainingRecordCopy_0_bits_state_wLaneLastReport <= recordEnq[0] ? initRecord_bits_state_wLaneLastReport : laneLastReport_5 | chainingRecordCopy_0_bits_state_wLaneLastReport;
      chainingRecordCopy_0_bits_state_wTopLastReport <= recordEnq[0] ? initRecord_bits_state_wTopLastReport : topLastReport_5 | chainingRecordCopy_0_bits_state_wTopLastReport;
      chainingRecordCopy_0_bits_state_wLaneClear <= ~(recordEnq[0]) & (waitLaneClear_5 & ~dataInLaneCheck_5 | chainingRecordCopy_0_bits_state_wLaneClear);
      chainingRecordCopy_1_valid <= recordEnq[1] | ~stateClear_6 & chainingRecordCopy_1_valid;
      chainingRecordCopy_1_bits_elementMask <= recordEnq[1] ? initRecord_bits_elementMask : {256{elementUpdateValid_6}} & elementUpdate1H_6 | chainingRecordCopy_1_bits_elementMask;
      chainingRecordCopy_1_bits_state_stFinish <= recordEnq[1] ? initRecord_bits_state_stFinish : topLastReport_6 | chainingRecordCopy_1_bits_state_stFinish;
      chainingRecordCopy_1_bits_state_wWriteQueueClear <= recordEnq[1] ? initRecord_bits_state_wWriteQueueClear : chainingRecordCopy_1_bits_state_stFinish & ~dataInLsuQueue_6 | chainingRecordCopy_1_bits_state_wWriteQueueClear;
      chainingRecordCopy_1_bits_state_wLaneLastReport <= recordEnq[1] ? initRecord_bits_state_wLaneLastReport : laneLastReport_6 | chainingRecordCopy_1_bits_state_wLaneLastReport;
      chainingRecordCopy_1_bits_state_wTopLastReport <= recordEnq[1] ? initRecord_bits_state_wTopLastReport : topLastReport_6 | chainingRecordCopy_1_bits_state_wTopLastReport;
      chainingRecordCopy_1_bits_state_wLaneClear <= ~(recordEnq[1]) & (waitLaneClear_6 & ~dataInLaneCheck_6 | chainingRecordCopy_1_bits_state_wLaneClear);
      chainingRecordCopy_2_valid <= recordEnq[2] | ~stateClear_7 & chainingRecordCopy_2_valid;
      chainingRecordCopy_2_bits_elementMask <= recordEnq[2] ? initRecord_bits_elementMask : {256{elementUpdateValid_7}} & elementUpdate1H_7 | chainingRecordCopy_2_bits_elementMask;
      chainingRecordCopy_2_bits_state_stFinish <= recordEnq[2] ? initRecord_bits_state_stFinish : topLastReport_7 | chainingRecordCopy_2_bits_state_stFinish;
      chainingRecordCopy_2_bits_state_wWriteQueueClear <= recordEnq[2] ? initRecord_bits_state_wWriteQueueClear : chainingRecordCopy_2_bits_state_stFinish & ~dataInLsuQueue_7 | chainingRecordCopy_2_bits_state_wWriteQueueClear;
      chainingRecordCopy_2_bits_state_wLaneLastReport <= recordEnq[2] ? initRecord_bits_state_wLaneLastReport : laneLastReport_7 | chainingRecordCopy_2_bits_state_wLaneLastReport;
      chainingRecordCopy_2_bits_state_wTopLastReport <= recordEnq[2] ? initRecord_bits_state_wTopLastReport : topLastReport_7 | chainingRecordCopy_2_bits_state_wTopLastReport;
      chainingRecordCopy_2_bits_state_wLaneClear <= ~(recordEnq[2]) & (waitLaneClear_7 & ~dataInLaneCheck_7 | chainingRecordCopy_2_bits_state_wLaneClear);
      chainingRecordCopy_3_valid <= recordEnq[3] | ~stateClear_8 & chainingRecordCopy_3_valid;
      chainingRecordCopy_3_bits_elementMask <= recordEnq[3] ? initRecord_bits_elementMask : {256{elementUpdateValid_8}} & elementUpdate1H_8 | chainingRecordCopy_3_bits_elementMask;
      chainingRecordCopy_3_bits_state_stFinish <= recordEnq[3] ? initRecord_bits_state_stFinish : topLastReport_8 | chainingRecordCopy_3_bits_state_stFinish;
      chainingRecordCopy_3_bits_state_wWriteQueueClear <= recordEnq[3] ? initRecord_bits_state_wWriteQueueClear : chainingRecordCopy_3_bits_state_stFinish & ~dataInLsuQueue_8 | chainingRecordCopy_3_bits_state_wWriteQueueClear;
      chainingRecordCopy_3_bits_state_wLaneLastReport <= recordEnq[3] ? initRecord_bits_state_wLaneLastReport : laneLastReport_8 | chainingRecordCopy_3_bits_state_wLaneLastReport;
      chainingRecordCopy_3_bits_state_wTopLastReport <= recordEnq[3] ? initRecord_bits_state_wTopLastReport : topLastReport_8 | chainingRecordCopy_3_bits_state_wTopLastReport;
      chainingRecordCopy_3_bits_state_wLaneClear <= ~(recordEnq[3]) & (waitLaneClear_8 & ~dataInLaneCheck_8 | chainingRecordCopy_3_bits_state_wLaneClear);
      chainingRecordCopy_4_valid <= recordEnq[4] | ~stateClear_9 & chainingRecordCopy_4_valid;
      chainingRecordCopy_4_bits_elementMask <= recordEnq[4] ? initRecord_bits_elementMask : {256{elementUpdateValid_9}} & elementUpdate1H_9 | chainingRecordCopy_4_bits_elementMask;
      chainingRecordCopy_4_bits_state_stFinish <= recordEnq[4] ? initRecord_bits_state_stFinish : topLastReport_9 | chainingRecordCopy_4_bits_state_stFinish;
      chainingRecordCopy_4_bits_state_wWriteQueueClear <= recordEnq[4] ? initRecord_bits_state_wWriteQueueClear : chainingRecordCopy_4_bits_state_stFinish & ~dataInLsuQueue_9 | chainingRecordCopy_4_bits_state_wWriteQueueClear;
      chainingRecordCopy_4_bits_state_wLaneLastReport <= recordEnq[4] ? initRecord_bits_state_wLaneLastReport : laneLastReport_9 | chainingRecordCopy_4_bits_state_wLaneLastReport;
      chainingRecordCopy_4_bits_state_wTopLastReport <= recordEnq[4] ? initRecord_bits_state_wTopLastReport : topLastReport_9 | chainingRecordCopy_4_bits_state_wTopLastReport;
      chainingRecordCopy_4_bits_state_wLaneClear <= ~(recordEnq[4]) & (waitLaneClear_9 & ~dataInLaneCheck_9 | chainingRecordCopy_4_bits_state_wLaneClear);
      firstReadPipe_0_valid <= bankReadF_0 | bankReadF_1 | bankReadF_2 | bankReadF_3 | bankReadF_4 | bankReadF_5 | bankReadF_6 | bankReadF_7 | bankReadF_8 | bankReadF_9 | bankReadF_10 | bankReadF_11 | bankReadF_12 | bankReadF_13;
      firstReadPipe_0_bits_address <=
        (bankReadF_0 ? address : 10'h0) | (bankReadF_1 ? address_1 : 10'h0) | (bankReadF_2 ? address_2 : 10'h0) | (bankReadF_3 ? address_3 : 10'h0) | (bankReadF_4 ? address_4 : 10'h0) | (bankReadF_5 ? address_5 : 10'h0)
        | (bankReadF_6 ? address_6 : 10'h0) | (bankReadF_7 ? address_7 : 10'h0) | (bankReadF_8 ? address_8 : 10'h0) | (bankReadF_9 ? address_9 : 10'h0) | (bankReadF_10 ? address_10 : 10'h0) | (bankReadF_11 ? address_11 : 10'h0)
        | (bankReadF_12 ? address_12 : 10'h0) | (bankReadF_13 ? address_13 : 10'h0);
      writePipe_valid <= _writePipe_valid_T;
      if (_writePipe_valid_T) begin
        writePipe_bits_vd <= write_bits_vd_0;
        writePipe_bits_offset <= write_bits_offset_0;
        writePipe_bits_mask <= write_bits_mask_0;
        writePipe_bits_data <= write_bits_data_0;
        writePipe_bits_last <= write_bits_last_0;
        writePipe_bits_instructionIndex <= write_bits_instructionIndex_0;
      end
      pipeBank_pipe_v <= 1'h1;
      pipeBank_pipe_pipe_v <= pipeBank_pipe_v;
      pipeFirstUsed_pipe_v <= 1'h1;
      pipeFirstUsed_pipe_pipe_v <= pipeFirstUsed_pipe_v;
      pipeFire_pipe_v <= 1'h1;
      pipeFire_pipe_pipe_v <= pipeFire_pipe_v;
      pipeBank_pipe_v_1 <= 1'h1;
      pipeBank_pipe_pipe_v_1 <= pipeBank_pipe_v_1;
      pipeFirstUsed_pipe_v_1 <= 1'h1;
      pipeFirstUsed_pipe_pipe_v_1 <= pipeFirstUsed_pipe_v_1;
      pipeFire_pipe_v_1 <= 1'h1;
      pipeFire_pipe_pipe_v_1 <= pipeFire_pipe_v_1;
      pipeBank_pipe_v_2 <= 1'h1;
      pipeBank_pipe_pipe_v_2 <= pipeBank_pipe_v_2;
      pipeFirstUsed_pipe_v_2 <= 1'h1;
      pipeFirstUsed_pipe_pipe_v_2 <= pipeFirstUsed_pipe_v_2;
      pipeFire_pipe_v_2 <= 1'h1;
      pipeFire_pipe_pipe_v_2 <= pipeFire_pipe_v_2;
      pipeBank_pipe_v_3 <= 1'h1;
      pipeBank_pipe_pipe_v_3 <= pipeBank_pipe_v_3;
      pipeFirstUsed_pipe_v_3 <= 1'h1;
      pipeFirstUsed_pipe_pipe_v_3 <= pipeFirstUsed_pipe_v_3;
      pipeFire_pipe_v_3 <= 1'h1;
      pipeFire_pipe_pipe_v_3 <= pipeFire_pipe_v_3;
      pipeBank_pipe_v_4 <= 1'h1;
      pipeBank_pipe_pipe_v_4 <= pipeBank_pipe_v_4;
      pipeFirstUsed_pipe_v_4 <= 1'h1;
      pipeFirstUsed_pipe_pipe_v_4 <= pipeFirstUsed_pipe_v_4;
      pipeFire_pipe_v_4 <= 1'h1;
      pipeFire_pipe_pipe_v_4 <= pipeFire_pipe_v_4;
      pipeBank_pipe_v_5 <= 1'h1;
      pipeBank_pipe_pipe_v_5 <= pipeBank_pipe_v_5;
      pipeFirstUsed_pipe_v_5 <= 1'h1;
      pipeFirstUsed_pipe_pipe_v_5 <= pipeFirstUsed_pipe_v_5;
      pipeFire_pipe_v_5 <= 1'h1;
      pipeFire_pipe_pipe_v_5 <= pipeFire_pipe_v_5;
      pipeBank_pipe_v_6 <= 1'h1;
      pipeBank_pipe_pipe_v_6 <= pipeBank_pipe_v_6;
      pipeFirstUsed_pipe_v_6 <= 1'h1;
      pipeFirstUsed_pipe_pipe_v_6 <= pipeFirstUsed_pipe_v_6;
      pipeFire_pipe_v_6 <= 1'h1;
      pipeFire_pipe_pipe_v_6 <= pipeFire_pipe_v_6;
      pipeBank_pipe_v_7 <= 1'h1;
      pipeBank_pipe_pipe_v_7 <= pipeBank_pipe_v_7;
      pipeFirstUsed_pipe_v_7 <= 1'h1;
      pipeFirstUsed_pipe_pipe_v_7 <= pipeFirstUsed_pipe_v_7;
      pipeFire_pipe_v_7 <= 1'h1;
      pipeFire_pipe_pipe_v_7 <= pipeFire_pipe_v_7;
      pipeBank_pipe_v_8 <= 1'h1;
      pipeBank_pipe_pipe_v_8 <= pipeBank_pipe_v_8;
      pipeFirstUsed_pipe_v_8 <= 1'h1;
      pipeFirstUsed_pipe_pipe_v_8 <= pipeFirstUsed_pipe_v_8;
      pipeFire_pipe_v_8 <= 1'h1;
      pipeFire_pipe_pipe_v_8 <= pipeFire_pipe_v_8;
      pipeBank_pipe_v_9 <= 1'h1;
      pipeBank_pipe_pipe_v_9 <= pipeBank_pipe_v_9;
      pipeFirstUsed_pipe_v_9 <= 1'h1;
      pipeFirstUsed_pipe_pipe_v_9 <= pipeFirstUsed_pipe_v_9;
      pipeFire_pipe_v_9 <= 1'h1;
      pipeFire_pipe_pipe_v_9 <= pipeFire_pipe_v_9;
      pipeBank_pipe_v_10 <= 1'h1;
      pipeBank_pipe_pipe_v_10 <= pipeBank_pipe_v_10;
      pipeFirstUsed_pipe_v_10 <= 1'h1;
      pipeFirstUsed_pipe_pipe_v_10 <= pipeFirstUsed_pipe_v_10;
      pipeFire_pipe_v_10 <= 1'h1;
      pipeFire_pipe_pipe_v_10 <= pipeFire_pipe_v_10;
      pipeBank_pipe_v_11 <= 1'h1;
      pipeBank_pipe_pipe_v_11 <= pipeBank_pipe_v_11;
      pipeFirstUsed_pipe_v_11 <= 1'h1;
      pipeFirstUsed_pipe_pipe_v_11 <= pipeFirstUsed_pipe_v_11;
      pipeFire_pipe_v_11 <= 1'h1;
      pipeFire_pipe_pipe_v_11 <= pipeFire_pipe_v_11;
      pipeBank_pipe_v_12 <= 1'h1;
      pipeBank_pipe_pipe_v_12 <= pipeBank_pipe_v_12;
      pipeFirstUsed_pipe_v_12 <= 1'h1;
      pipeFirstUsed_pipe_pipe_v_12 <= pipeFirstUsed_pipe_v_12;
      pipeFire_pipe_v_12 <= 1'h1;
      pipeFire_pipe_pipe_v_12 <= pipeFire_pipe_v_12;
      pipeBank_pipe_v_13 <= 1'h1;
      pipeBank_pipe_pipe_v_13 <= pipeBank_pipe_v_13;
      pipeFirstUsed_pipe_v_13 <= 1'h1;
      pipeFirstUsed_pipe_pipe_v_13 <= pipeFirstUsed_pipe_v_13;
      pipeFire_pipe_v_13 <= 1'h1;
      pipeFire_pipe_pipe_v_13 <= pipeFire_pipe_v_13;
    end
    pipeBank_pipe_pipe_b <= 1'h1;
    pipeFirstUsed_pipe_pipe_b <= 1'h0;
    pipeFire_pipe_b <= _pipeFire_T;
    if (pipeFire_pipe_v)
      pipeFire_pipe_pipe_b <= pipeFire_pipe_b;
    pipeBank_pipe_pipe_b_1 <= 1'h1;
    pipeFirstUsed_pipe_b_1 <= firstUsed_1;
    if (pipeFirstUsed_pipe_v_1)
      pipeFirstUsed_pipe_pipe_b_1 <= pipeFirstUsed_pipe_b_1;
    pipeFire_pipe_b_1 <= _pipeFire_T_1;
    if (pipeFire_pipe_v_1)
      pipeFire_pipe_pipe_b_1 <= pipeFire_pipe_b_1;
    pipeBank_pipe_pipe_b_2 <= 1'h1;
    pipeFirstUsed_pipe_b_2 <= firstUsed_2;
    if (pipeFirstUsed_pipe_v_2)
      pipeFirstUsed_pipe_pipe_b_2 <= pipeFirstUsed_pipe_b_2;
    pipeFire_pipe_b_2 <= _pipeFire_T_2;
    if (pipeFire_pipe_v_2)
      pipeFire_pipe_pipe_b_2 <= pipeFire_pipe_b_2;
    pipeBank_pipe_pipe_b_3 <= 1'h1;
    pipeFirstUsed_pipe_b_3 <= firstUsed_3;
    if (pipeFirstUsed_pipe_v_3)
      pipeFirstUsed_pipe_pipe_b_3 <= pipeFirstUsed_pipe_b_3;
    pipeFire_pipe_b_3 <= _pipeFire_T_3;
    if (pipeFire_pipe_v_3)
      pipeFire_pipe_pipe_b_3 <= pipeFire_pipe_b_3;
    pipeBank_pipe_pipe_b_4 <= 1'h1;
    pipeFirstUsed_pipe_b_4 <= firstUsed_4;
    if (pipeFirstUsed_pipe_v_4)
      pipeFirstUsed_pipe_pipe_b_4 <= pipeFirstUsed_pipe_b_4;
    pipeFire_pipe_b_4 <= _pipeFire_T_4;
    if (pipeFire_pipe_v_4)
      pipeFire_pipe_pipe_b_4 <= pipeFire_pipe_b_4;
    pipeBank_pipe_pipe_b_5 <= 1'h1;
    pipeFirstUsed_pipe_b_5 <= firstUsed_5;
    if (pipeFirstUsed_pipe_v_5)
      pipeFirstUsed_pipe_pipe_b_5 <= pipeFirstUsed_pipe_b_5;
    pipeFire_pipe_b_5 <= _pipeFire_T_5;
    if (pipeFire_pipe_v_5)
      pipeFire_pipe_pipe_b_5 <= pipeFire_pipe_b_5;
    pipeBank_pipe_pipe_b_6 <= 1'h1;
    pipeFirstUsed_pipe_b_6 <= firstUsed_6;
    if (pipeFirstUsed_pipe_v_6)
      pipeFirstUsed_pipe_pipe_b_6 <= pipeFirstUsed_pipe_b_6;
    pipeFire_pipe_b_6 <= _pipeFire_T_6;
    if (pipeFire_pipe_v_6)
      pipeFire_pipe_pipe_b_6 <= pipeFire_pipe_b_6;
    pipeBank_pipe_pipe_b_7 <= 1'h1;
    pipeFirstUsed_pipe_b_7 <= firstUsed_7;
    if (pipeFirstUsed_pipe_v_7)
      pipeFirstUsed_pipe_pipe_b_7 <= pipeFirstUsed_pipe_b_7;
    pipeFire_pipe_b_7 <= _pipeFire_T_7;
    if (pipeFire_pipe_v_7)
      pipeFire_pipe_pipe_b_7 <= pipeFire_pipe_b_7;
    pipeBank_pipe_pipe_b_8 <= 1'h1;
    pipeFirstUsed_pipe_b_8 <= firstUsed_8;
    if (pipeFirstUsed_pipe_v_8)
      pipeFirstUsed_pipe_pipe_b_8 <= pipeFirstUsed_pipe_b_8;
    pipeFire_pipe_b_8 <= _pipeFire_T_8;
    if (pipeFire_pipe_v_8)
      pipeFire_pipe_pipe_b_8 <= pipeFire_pipe_b_8;
    pipeBank_pipe_pipe_b_9 <= 1'h1;
    pipeFirstUsed_pipe_b_9 <= firstUsed_9;
    if (pipeFirstUsed_pipe_v_9)
      pipeFirstUsed_pipe_pipe_b_9 <= pipeFirstUsed_pipe_b_9;
    pipeFire_pipe_b_9 <= _pipeFire_T_9;
    if (pipeFire_pipe_v_9)
      pipeFire_pipe_pipe_b_9 <= pipeFire_pipe_b_9;
    pipeBank_pipe_pipe_b_10 <= 1'h1;
    pipeFirstUsed_pipe_b_10 <= firstUsed_10;
    if (pipeFirstUsed_pipe_v_10)
      pipeFirstUsed_pipe_pipe_b_10 <= pipeFirstUsed_pipe_b_10;
    pipeFire_pipe_b_10 <= _pipeFire_T_10;
    if (pipeFire_pipe_v_10)
      pipeFire_pipe_pipe_b_10 <= pipeFire_pipe_b_10;
    pipeBank_pipe_pipe_b_11 <= 1'h1;
    pipeFirstUsed_pipe_b_11 <= firstUsed_11;
    if (pipeFirstUsed_pipe_v_11)
      pipeFirstUsed_pipe_pipe_b_11 <= pipeFirstUsed_pipe_b_11;
    pipeFire_pipe_b_11 <= _pipeFire_T_11;
    if (pipeFire_pipe_v_11)
      pipeFire_pipe_pipe_b_11 <= pipeFire_pipe_b_11;
    pipeBank_pipe_pipe_b_12 <= 1'h1;
    pipeFirstUsed_pipe_b_12 <= firstUsed_12;
    if (pipeFirstUsed_pipe_v_12)
      pipeFirstUsed_pipe_pipe_b_12 <= pipeFirstUsed_pipe_b_12;
    pipeFire_pipe_b_12 <= _pipeFire_T_12;
    if (pipeFire_pipe_v_12)
      pipeFire_pipe_pipe_b_12 <= pipeFire_pipe_b_12;
    pipeBank_pipe_pipe_b_13 <= 1'h1;
    pipeFirstUsed_pipe_b_13 <= firstUsed_13;
    if (pipeFirstUsed_pipe_v_13)
      pipeFirstUsed_pipe_pipe_b_13 <= pipeFirstUsed_pipe_b_13;
    pipeFire_pipe_b_13 <= _loadUpdateValidVec_T_27;
    if (pipeFire_pipe_v_13)
      pipeFire_pipe_pipe_b_13 <= pipeFire_pipe_b_13;
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:99];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [6:0] i = 7'h0; i < 7'h64; i += 7'h1) begin
          _RANDOM[i] = `RANDOM;
        end
        sramReady = _RANDOM[7'h0][0];
        sramResetCount = _RANDOM[7'h0][10:1];
        chainingRecord_0_valid = _RANDOM[7'h0][11];
        chainingRecord_0_bits_vd_valid = _RANDOM[7'h0][12];
        chainingRecord_0_bits_vd_bits = _RANDOM[7'h0][17:13];
        chainingRecord_0_bits_vs1_valid = _RANDOM[7'h0][18];
        chainingRecord_0_bits_vs1_bits = _RANDOM[7'h0][23:19];
        chainingRecord_0_bits_vs2 = _RANDOM[7'h0][28:24];
        chainingRecord_0_bits_instIndex = _RANDOM[7'h0][31:29];
        chainingRecord_0_bits_ls = _RANDOM[7'h1][0];
        chainingRecord_0_bits_st = _RANDOM[7'h1][1];
        chainingRecord_0_bits_gather = _RANDOM[7'h1][2];
        chainingRecord_0_bits_gather16 = _RANDOM[7'h1][3];
        chainingRecord_0_bits_crossWrite = _RANDOM[7'h1][4];
        chainingRecord_0_bits_crossRead = _RANDOM[7'h1][5];
        chainingRecord_0_bits_indexType = _RANDOM[7'h1][6];
        chainingRecord_0_bits_ma = _RANDOM[7'h1][7];
        chainingRecord_0_bits_onlyRead = _RANDOM[7'h1][8];
        chainingRecord_0_bits_slow = _RANDOM[7'h1][9];
        chainingRecord_0_bits_elementMask = {_RANDOM[7'h1][31:10], _RANDOM[7'h2], _RANDOM[7'h3], _RANDOM[7'h4], _RANDOM[7'h5], _RANDOM[7'h6], _RANDOM[7'h7], _RANDOM[7'h8], _RANDOM[7'h9][9:0]};
        chainingRecord_0_bits_state_stFinish = _RANDOM[7'h9][10];
        chainingRecord_0_bits_state_wWriteQueueClear = _RANDOM[7'h9][11];
        chainingRecord_0_bits_state_wLaneLastReport = _RANDOM[7'h9][12];
        chainingRecord_0_bits_state_wTopLastReport = _RANDOM[7'h9][13];
        chainingRecord_0_bits_state_wLaneClear = _RANDOM[7'h9][14];
        chainingRecord_1_valid = _RANDOM[7'h9][15];
        chainingRecord_1_bits_vd_valid = _RANDOM[7'h9][16];
        chainingRecord_1_bits_vd_bits = _RANDOM[7'h9][21:17];
        chainingRecord_1_bits_vs1_valid = _RANDOM[7'h9][22];
        chainingRecord_1_bits_vs1_bits = _RANDOM[7'h9][27:23];
        chainingRecord_1_bits_vs2 = {_RANDOM[7'h9][31:28], _RANDOM[7'hA][0]};
        chainingRecord_1_bits_instIndex = _RANDOM[7'hA][3:1];
        chainingRecord_1_bits_ls = _RANDOM[7'hA][4];
        chainingRecord_1_bits_st = _RANDOM[7'hA][5];
        chainingRecord_1_bits_gather = _RANDOM[7'hA][6];
        chainingRecord_1_bits_gather16 = _RANDOM[7'hA][7];
        chainingRecord_1_bits_crossWrite = _RANDOM[7'hA][8];
        chainingRecord_1_bits_crossRead = _RANDOM[7'hA][9];
        chainingRecord_1_bits_indexType = _RANDOM[7'hA][10];
        chainingRecord_1_bits_ma = _RANDOM[7'hA][11];
        chainingRecord_1_bits_onlyRead = _RANDOM[7'hA][12];
        chainingRecord_1_bits_slow = _RANDOM[7'hA][13];
        chainingRecord_1_bits_elementMask = {_RANDOM[7'hA][31:14], _RANDOM[7'hB], _RANDOM[7'hC], _RANDOM[7'hD], _RANDOM[7'hE], _RANDOM[7'hF], _RANDOM[7'h10], _RANDOM[7'h11], _RANDOM[7'h12][13:0]};
        chainingRecord_1_bits_state_stFinish = _RANDOM[7'h12][14];
        chainingRecord_1_bits_state_wWriteQueueClear = _RANDOM[7'h12][15];
        chainingRecord_1_bits_state_wLaneLastReport = _RANDOM[7'h12][16];
        chainingRecord_1_bits_state_wTopLastReport = _RANDOM[7'h12][17];
        chainingRecord_1_bits_state_wLaneClear = _RANDOM[7'h12][18];
        chainingRecord_2_valid = _RANDOM[7'h12][19];
        chainingRecord_2_bits_vd_valid = _RANDOM[7'h12][20];
        chainingRecord_2_bits_vd_bits = _RANDOM[7'h12][25:21];
        chainingRecord_2_bits_vs1_valid = _RANDOM[7'h12][26];
        chainingRecord_2_bits_vs1_bits = _RANDOM[7'h12][31:27];
        chainingRecord_2_bits_vs2 = _RANDOM[7'h13][4:0];
        chainingRecord_2_bits_instIndex = _RANDOM[7'h13][7:5];
        chainingRecord_2_bits_ls = _RANDOM[7'h13][8];
        chainingRecord_2_bits_st = _RANDOM[7'h13][9];
        chainingRecord_2_bits_gather = _RANDOM[7'h13][10];
        chainingRecord_2_bits_gather16 = _RANDOM[7'h13][11];
        chainingRecord_2_bits_crossWrite = _RANDOM[7'h13][12];
        chainingRecord_2_bits_crossRead = _RANDOM[7'h13][13];
        chainingRecord_2_bits_indexType = _RANDOM[7'h13][14];
        chainingRecord_2_bits_ma = _RANDOM[7'h13][15];
        chainingRecord_2_bits_onlyRead = _RANDOM[7'h13][16];
        chainingRecord_2_bits_slow = _RANDOM[7'h13][17];
        chainingRecord_2_bits_elementMask = {_RANDOM[7'h13][31:18], _RANDOM[7'h14], _RANDOM[7'h15], _RANDOM[7'h16], _RANDOM[7'h17], _RANDOM[7'h18], _RANDOM[7'h19], _RANDOM[7'h1A], _RANDOM[7'h1B][17:0]};
        chainingRecord_2_bits_state_stFinish = _RANDOM[7'h1B][18];
        chainingRecord_2_bits_state_wWriteQueueClear = _RANDOM[7'h1B][19];
        chainingRecord_2_bits_state_wLaneLastReport = _RANDOM[7'h1B][20];
        chainingRecord_2_bits_state_wTopLastReport = _RANDOM[7'h1B][21];
        chainingRecord_2_bits_state_wLaneClear = _RANDOM[7'h1B][22];
        chainingRecord_3_valid = _RANDOM[7'h1B][23];
        chainingRecord_3_bits_vd_valid = _RANDOM[7'h1B][24];
        chainingRecord_3_bits_vd_bits = _RANDOM[7'h1B][29:25];
        chainingRecord_3_bits_vs1_valid = _RANDOM[7'h1B][30];
        chainingRecord_3_bits_vs1_bits = {_RANDOM[7'h1B][31], _RANDOM[7'h1C][3:0]};
        chainingRecord_3_bits_vs2 = _RANDOM[7'h1C][8:4];
        chainingRecord_3_bits_instIndex = _RANDOM[7'h1C][11:9];
        chainingRecord_3_bits_ls = _RANDOM[7'h1C][12];
        chainingRecord_3_bits_st = _RANDOM[7'h1C][13];
        chainingRecord_3_bits_gather = _RANDOM[7'h1C][14];
        chainingRecord_3_bits_gather16 = _RANDOM[7'h1C][15];
        chainingRecord_3_bits_crossWrite = _RANDOM[7'h1C][16];
        chainingRecord_3_bits_crossRead = _RANDOM[7'h1C][17];
        chainingRecord_3_bits_indexType = _RANDOM[7'h1C][18];
        chainingRecord_3_bits_ma = _RANDOM[7'h1C][19];
        chainingRecord_3_bits_onlyRead = _RANDOM[7'h1C][20];
        chainingRecord_3_bits_slow = _RANDOM[7'h1C][21];
        chainingRecord_3_bits_elementMask = {_RANDOM[7'h1C][31:22], _RANDOM[7'h1D], _RANDOM[7'h1E], _RANDOM[7'h1F], _RANDOM[7'h20], _RANDOM[7'h21], _RANDOM[7'h22], _RANDOM[7'h23], _RANDOM[7'h24][21:0]};
        chainingRecord_3_bits_state_stFinish = _RANDOM[7'h24][22];
        chainingRecord_3_bits_state_wWriteQueueClear = _RANDOM[7'h24][23];
        chainingRecord_3_bits_state_wLaneLastReport = _RANDOM[7'h24][24];
        chainingRecord_3_bits_state_wTopLastReport = _RANDOM[7'h24][25];
        chainingRecord_3_bits_state_wLaneClear = _RANDOM[7'h24][26];
        chainingRecord_4_valid = _RANDOM[7'h24][27];
        chainingRecord_4_bits_vd_valid = _RANDOM[7'h24][28];
        chainingRecord_4_bits_vd_bits = {_RANDOM[7'h24][31:29], _RANDOM[7'h25][1:0]};
        chainingRecord_4_bits_vs1_valid = _RANDOM[7'h25][2];
        chainingRecord_4_bits_vs1_bits = _RANDOM[7'h25][7:3];
        chainingRecord_4_bits_vs2 = _RANDOM[7'h25][12:8];
        chainingRecord_4_bits_instIndex = _RANDOM[7'h25][15:13];
        chainingRecord_4_bits_ls = _RANDOM[7'h25][16];
        chainingRecord_4_bits_st = _RANDOM[7'h25][17];
        chainingRecord_4_bits_gather = _RANDOM[7'h25][18];
        chainingRecord_4_bits_gather16 = _RANDOM[7'h25][19];
        chainingRecord_4_bits_crossWrite = _RANDOM[7'h25][20];
        chainingRecord_4_bits_crossRead = _RANDOM[7'h25][21];
        chainingRecord_4_bits_indexType = _RANDOM[7'h25][22];
        chainingRecord_4_bits_ma = _RANDOM[7'h25][23];
        chainingRecord_4_bits_onlyRead = _RANDOM[7'h25][24];
        chainingRecord_4_bits_slow = _RANDOM[7'h25][25];
        chainingRecord_4_bits_elementMask = {_RANDOM[7'h25][31:26], _RANDOM[7'h26], _RANDOM[7'h27], _RANDOM[7'h28], _RANDOM[7'h29], _RANDOM[7'h2A], _RANDOM[7'h2B], _RANDOM[7'h2C], _RANDOM[7'h2D][25:0]};
        chainingRecord_4_bits_state_stFinish = _RANDOM[7'h2D][26];
        chainingRecord_4_bits_state_wWriteQueueClear = _RANDOM[7'h2D][27];
        chainingRecord_4_bits_state_wLaneLastReport = _RANDOM[7'h2D][28];
        chainingRecord_4_bits_state_wTopLastReport = _RANDOM[7'h2D][29];
        chainingRecord_4_bits_state_wLaneClear = _RANDOM[7'h2D][30];
        chainingRecordCopy_0_valid = _RANDOM[7'h2D][31];
        chainingRecordCopy_0_bits_vd_valid = _RANDOM[7'h2E][0];
        chainingRecordCopy_0_bits_vd_bits = _RANDOM[7'h2E][5:1];
        chainingRecordCopy_0_bits_vs1_valid = _RANDOM[7'h2E][6];
        chainingRecordCopy_0_bits_vs1_bits = _RANDOM[7'h2E][11:7];
        chainingRecordCopy_0_bits_vs2 = _RANDOM[7'h2E][16:12];
        chainingRecordCopy_0_bits_instIndex = _RANDOM[7'h2E][19:17];
        chainingRecordCopy_0_bits_ls = _RANDOM[7'h2E][20];
        chainingRecordCopy_0_bits_st = _RANDOM[7'h2E][21];
        chainingRecordCopy_0_bits_gather = _RANDOM[7'h2E][22];
        chainingRecordCopy_0_bits_gather16 = _RANDOM[7'h2E][23];
        chainingRecordCopy_0_bits_crossWrite = _RANDOM[7'h2E][24];
        chainingRecordCopy_0_bits_crossRead = _RANDOM[7'h2E][25];
        chainingRecordCopy_0_bits_indexType = _RANDOM[7'h2E][26];
        chainingRecordCopy_0_bits_ma = _RANDOM[7'h2E][27];
        chainingRecordCopy_0_bits_onlyRead = _RANDOM[7'h2E][28];
        chainingRecordCopy_0_bits_slow = _RANDOM[7'h2E][29];
        chainingRecordCopy_0_bits_elementMask = {_RANDOM[7'h2E][31:30], _RANDOM[7'h2F], _RANDOM[7'h30], _RANDOM[7'h31], _RANDOM[7'h32], _RANDOM[7'h33], _RANDOM[7'h34], _RANDOM[7'h35], _RANDOM[7'h36][29:0]};
        chainingRecordCopy_0_bits_state_stFinish = _RANDOM[7'h36][30];
        chainingRecordCopy_0_bits_state_wWriteQueueClear = _RANDOM[7'h36][31];
        chainingRecordCopy_0_bits_state_wLaneLastReport = _RANDOM[7'h37][0];
        chainingRecordCopy_0_bits_state_wTopLastReport = _RANDOM[7'h37][1];
        chainingRecordCopy_0_bits_state_wLaneClear = _RANDOM[7'h37][2];
        chainingRecordCopy_1_valid = _RANDOM[7'h37][3];
        chainingRecordCopy_1_bits_vd_valid = _RANDOM[7'h37][4];
        chainingRecordCopy_1_bits_vd_bits = _RANDOM[7'h37][9:5];
        chainingRecordCopy_1_bits_vs1_valid = _RANDOM[7'h37][10];
        chainingRecordCopy_1_bits_vs1_bits = _RANDOM[7'h37][15:11];
        chainingRecordCopy_1_bits_vs2 = _RANDOM[7'h37][20:16];
        chainingRecordCopy_1_bits_instIndex = _RANDOM[7'h37][23:21];
        chainingRecordCopy_1_bits_ls = _RANDOM[7'h37][24];
        chainingRecordCopy_1_bits_st = _RANDOM[7'h37][25];
        chainingRecordCopy_1_bits_gather = _RANDOM[7'h37][26];
        chainingRecordCopy_1_bits_gather16 = _RANDOM[7'h37][27];
        chainingRecordCopy_1_bits_crossWrite = _RANDOM[7'h37][28];
        chainingRecordCopy_1_bits_crossRead = _RANDOM[7'h37][29];
        chainingRecordCopy_1_bits_indexType = _RANDOM[7'h37][30];
        chainingRecordCopy_1_bits_ma = _RANDOM[7'h37][31];
        chainingRecordCopy_1_bits_onlyRead = _RANDOM[7'h38][0];
        chainingRecordCopy_1_bits_slow = _RANDOM[7'h38][1];
        chainingRecordCopy_1_bits_elementMask = {_RANDOM[7'h38][31:2], _RANDOM[7'h39], _RANDOM[7'h3A], _RANDOM[7'h3B], _RANDOM[7'h3C], _RANDOM[7'h3D], _RANDOM[7'h3E], _RANDOM[7'h3F], _RANDOM[7'h40][1:0]};
        chainingRecordCopy_1_bits_state_stFinish = _RANDOM[7'h40][2];
        chainingRecordCopy_1_bits_state_wWriteQueueClear = _RANDOM[7'h40][3];
        chainingRecordCopy_1_bits_state_wLaneLastReport = _RANDOM[7'h40][4];
        chainingRecordCopy_1_bits_state_wTopLastReport = _RANDOM[7'h40][5];
        chainingRecordCopy_1_bits_state_wLaneClear = _RANDOM[7'h40][6];
        chainingRecordCopy_2_valid = _RANDOM[7'h40][7];
        chainingRecordCopy_2_bits_vd_valid = _RANDOM[7'h40][8];
        chainingRecordCopy_2_bits_vd_bits = _RANDOM[7'h40][13:9];
        chainingRecordCopy_2_bits_vs1_valid = _RANDOM[7'h40][14];
        chainingRecordCopy_2_bits_vs1_bits = _RANDOM[7'h40][19:15];
        chainingRecordCopy_2_bits_vs2 = _RANDOM[7'h40][24:20];
        chainingRecordCopy_2_bits_instIndex = _RANDOM[7'h40][27:25];
        chainingRecordCopy_2_bits_ls = _RANDOM[7'h40][28];
        chainingRecordCopy_2_bits_st = _RANDOM[7'h40][29];
        chainingRecordCopy_2_bits_gather = _RANDOM[7'h40][30];
        chainingRecordCopy_2_bits_gather16 = _RANDOM[7'h40][31];
        chainingRecordCopy_2_bits_crossWrite = _RANDOM[7'h41][0];
        chainingRecordCopy_2_bits_crossRead = _RANDOM[7'h41][1];
        chainingRecordCopy_2_bits_indexType = _RANDOM[7'h41][2];
        chainingRecordCopy_2_bits_ma = _RANDOM[7'h41][3];
        chainingRecordCopy_2_bits_onlyRead = _RANDOM[7'h41][4];
        chainingRecordCopy_2_bits_slow = _RANDOM[7'h41][5];
        chainingRecordCopy_2_bits_elementMask = {_RANDOM[7'h41][31:6], _RANDOM[7'h42], _RANDOM[7'h43], _RANDOM[7'h44], _RANDOM[7'h45], _RANDOM[7'h46], _RANDOM[7'h47], _RANDOM[7'h48], _RANDOM[7'h49][5:0]};
        chainingRecordCopy_2_bits_state_stFinish = _RANDOM[7'h49][6];
        chainingRecordCopy_2_bits_state_wWriteQueueClear = _RANDOM[7'h49][7];
        chainingRecordCopy_2_bits_state_wLaneLastReport = _RANDOM[7'h49][8];
        chainingRecordCopy_2_bits_state_wTopLastReport = _RANDOM[7'h49][9];
        chainingRecordCopy_2_bits_state_wLaneClear = _RANDOM[7'h49][10];
        chainingRecordCopy_3_valid = _RANDOM[7'h49][11];
        chainingRecordCopy_3_bits_vd_valid = _RANDOM[7'h49][12];
        chainingRecordCopy_3_bits_vd_bits = _RANDOM[7'h49][17:13];
        chainingRecordCopy_3_bits_vs1_valid = _RANDOM[7'h49][18];
        chainingRecordCopy_3_bits_vs1_bits = _RANDOM[7'h49][23:19];
        chainingRecordCopy_3_bits_vs2 = _RANDOM[7'h49][28:24];
        chainingRecordCopy_3_bits_instIndex = _RANDOM[7'h49][31:29];
        chainingRecordCopy_3_bits_ls = _RANDOM[7'h4A][0];
        chainingRecordCopy_3_bits_st = _RANDOM[7'h4A][1];
        chainingRecordCopy_3_bits_gather = _RANDOM[7'h4A][2];
        chainingRecordCopy_3_bits_gather16 = _RANDOM[7'h4A][3];
        chainingRecordCopy_3_bits_crossWrite = _RANDOM[7'h4A][4];
        chainingRecordCopy_3_bits_crossRead = _RANDOM[7'h4A][5];
        chainingRecordCopy_3_bits_indexType = _RANDOM[7'h4A][6];
        chainingRecordCopy_3_bits_ma = _RANDOM[7'h4A][7];
        chainingRecordCopy_3_bits_onlyRead = _RANDOM[7'h4A][8];
        chainingRecordCopy_3_bits_slow = _RANDOM[7'h4A][9];
        chainingRecordCopy_3_bits_elementMask = {_RANDOM[7'h4A][31:10], _RANDOM[7'h4B], _RANDOM[7'h4C], _RANDOM[7'h4D], _RANDOM[7'h4E], _RANDOM[7'h4F], _RANDOM[7'h50], _RANDOM[7'h51], _RANDOM[7'h52][9:0]};
        chainingRecordCopy_3_bits_state_stFinish = _RANDOM[7'h52][10];
        chainingRecordCopy_3_bits_state_wWriteQueueClear = _RANDOM[7'h52][11];
        chainingRecordCopy_3_bits_state_wLaneLastReport = _RANDOM[7'h52][12];
        chainingRecordCopy_3_bits_state_wTopLastReport = _RANDOM[7'h52][13];
        chainingRecordCopy_3_bits_state_wLaneClear = _RANDOM[7'h52][14];
        chainingRecordCopy_4_valid = _RANDOM[7'h52][15];
        chainingRecordCopy_4_bits_vd_valid = _RANDOM[7'h52][16];
        chainingRecordCopy_4_bits_vd_bits = _RANDOM[7'h52][21:17];
        chainingRecordCopy_4_bits_vs1_valid = _RANDOM[7'h52][22];
        chainingRecordCopy_4_bits_vs1_bits = _RANDOM[7'h52][27:23];
        chainingRecordCopy_4_bits_vs2 = {_RANDOM[7'h52][31:28], _RANDOM[7'h53][0]};
        chainingRecordCopy_4_bits_instIndex = _RANDOM[7'h53][3:1];
        chainingRecordCopy_4_bits_ls = _RANDOM[7'h53][4];
        chainingRecordCopy_4_bits_st = _RANDOM[7'h53][5];
        chainingRecordCopy_4_bits_gather = _RANDOM[7'h53][6];
        chainingRecordCopy_4_bits_gather16 = _RANDOM[7'h53][7];
        chainingRecordCopy_4_bits_crossWrite = _RANDOM[7'h53][8];
        chainingRecordCopy_4_bits_crossRead = _RANDOM[7'h53][9];
        chainingRecordCopy_4_bits_indexType = _RANDOM[7'h53][10];
        chainingRecordCopy_4_bits_ma = _RANDOM[7'h53][11];
        chainingRecordCopy_4_bits_onlyRead = _RANDOM[7'h53][12];
        chainingRecordCopy_4_bits_slow = _RANDOM[7'h53][13];
        chainingRecordCopy_4_bits_elementMask = {_RANDOM[7'h53][31:14], _RANDOM[7'h54], _RANDOM[7'h55], _RANDOM[7'h56], _RANDOM[7'h57], _RANDOM[7'h58], _RANDOM[7'h59], _RANDOM[7'h5A], _RANDOM[7'h5B][13:0]};
        chainingRecordCopy_4_bits_state_stFinish = _RANDOM[7'h5B][14];
        chainingRecordCopy_4_bits_state_wWriteQueueClear = _RANDOM[7'h5B][15];
        chainingRecordCopy_4_bits_state_wLaneLastReport = _RANDOM[7'h5B][16];
        chainingRecordCopy_4_bits_state_wTopLastReport = _RANDOM[7'h5B][17];
        chainingRecordCopy_4_bits_state_wLaneClear = _RANDOM[7'h5B][18];
        firstReadPipe_0_valid = _RANDOM[7'h5B][19];
        firstReadPipe_0_bits_address = _RANDOM[7'h5B][29:20];
        writePipe_valid = _RANDOM[7'h5C][9];
        writePipe_bits_vd = _RANDOM[7'h5C][14:10];
        writePipe_bits_offset = _RANDOM[7'h5C][19:15];
        writePipe_bits_mask = _RANDOM[7'h5C][23:20];
        writePipe_bits_data = {_RANDOM[7'h5C][31:24], _RANDOM[7'h5D][23:0]};
        writePipe_bits_last = _RANDOM[7'h5D][24];
        writePipe_bits_instructionIndex = _RANDOM[7'h5D][27:25];
        pipeBank_pipe_v = _RANDOM[7'h5D][29];
        pipeBank_pipe_pipe_v = _RANDOM[7'h5D][31];
        pipeBank_pipe_pipe_b = _RANDOM[7'h5E][0];
        pipeFirstUsed_pipe_v = _RANDOM[7'h5E][1];
        pipeFirstUsed_pipe_pipe_v = _RANDOM[7'h5E][3];
        pipeFirstUsed_pipe_pipe_b = _RANDOM[7'h5E][4];
        pipeFire_pipe_v = _RANDOM[7'h5E][5];
        pipeFire_pipe_b = _RANDOM[7'h5E][6];
        pipeFire_pipe_pipe_v = _RANDOM[7'h5E][7];
        pipeFire_pipe_pipe_b = _RANDOM[7'h5E][8];
        pipeBank_pipe_v_1 = _RANDOM[7'h5E][9];
        pipeBank_pipe_pipe_v_1 = _RANDOM[7'h5E][11];
        pipeBank_pipe_pipe_b_1 = _RANDOM[7'h5E][12];
        pipeFirstUsed_pipe_v_1 = _RANDOM[7'h5E][13];
        pipeFirstUsed_pipe_b_1 = _RANDOM[7'h5E][14];
        pipeFirstUsed_pipe_pipe_v_1 = _RANDOM[7'h5E][15];
        pipeFirstUsed_pipe_pipe_b_1 = _RANDOM[7'h5E][16];
        pipeFire_pipe_v_1 = _RANDOM[7'h5E][17];
        pipeFire_pipe_b_1 = _RANDOM[7'h5E][18];
        pipeFire_pipe_pipe_v_1 = _RANDOM[7'h5E][19];
        pipeFire_pipe_pipe_b_1 = _RANDOM[7'h5E][20];
        pipeBank_pipe_v_2 = _RANDOM[7'h5E][21];
        pipeBank_pipe_pipe_v_2 = _RANDOM[7'h5E][23];
        pipeBank_pipe_pipe_b_2 = _RANDOM[7'h5E][24];
        pipeFirstUsed_pipe_v_2 = _RANDOM[7'h5E][25];
        pipeFirstUsed_pipe_b_2 = _RANDOM[7'h5E][26];
        pipeFirstUsed_pipe_pipe_v_2 = _RANDOM[7'h5E][27];
        pipeFirstUsed_pipe_pipe_b_2 = _RANDOM[7'h5E][28];
        pipeFire_pipe_v_2 = _RANDOM[7'h5E][29];
        pipeFire_pipe_b_2 = _RANDOM[7'h5E][30];
        pipeFire_pipe_pipe_v_2 = _RANDOM[7'h5E][31];
        pipeFire_pipe_pipe_b_2 = _RANDOM[7'h5F][0];
        pipeBank_pipe_v_3 = _RANDOM[7'h5F][1];
        pipeBank_pipe_pipe_v_3 = _RANDOM[7'h5F][3];
        pipeBank_pipe_pipe_b_3 = _RANDOM[7'h5F][4];
        pipeFirstUsed_pipe_v_3 = _RANDOM[7'h5F][5];
        pipeFirstUsed_pipe_b_3 = _RANDOM[7'h5F][6];
        pipeFirstUsed_pipe_pipe_v_3 = _RANDOM[7'h5F][7];
        pipeFirstUsed_pipe_pipe_b_3 = _RANDOM[7'h5F][8];
        pipeFire_pipe_v_3 = _RANDOM[7'h5F][9];
        pipeFire_pipe_b_3 = _RANDOM[7'h5F][10];
        pipeFire_pipe_pipe_v_3 = _RANDOM[7'h5F][11];
        pipeFire_pipe_pipe_b_3 = _RANDOM[7'h5F][12];
        pipeBank_pipe_v_4 = _RANDOM[7'h5F][13];
        pipeBank_pipe_pipe_v_4 = _RANDOM[7'h5F][15];
        pipeBank_pipe_pipe_b_4 = _RANDOM[7'h5F][16];
        pipeFirstUsed_pipe_v_4 = _RANDOM[7'h5F][17];
        pipeFirstUsed_pipe_b_4 = _RANDOM[7'h5F][18];
        pipeFirstUsed_pipe_pipe_v_4 = _RANDOM[7'h5F][19];
        pipeFirstUsed_pipe_pipe_b_4 = _RANDOM[7'h5F][20];
        pipeFire_pipe_v_4 = _RANDOM[7'h5F][21];
        pipeFire_pipe_b_4 = _RANDOM[7'h5F][22];
        pipeFire_pipe_pipe_v_4 = _RANDOM[7'h5F][23];
        pipeFire_pipe_pipe_b_4 = _RANDOM[7'h5F][24];
        pipeBank_pipe_v_5 = _RANDOM[7'h5F][25];
        pipeBank_pipe_pipe_v_5 = _RANDOM[7'h5F][27];
        pipeBank_pipe_pipe_b_5 = _RANDOM[7'h5F][28];
        pipeFirstUsed_pipe_v_5 = _RANDOM[7'h5F][29];
        pipeFirstUsed_pipe_b_5 = _RANDOM[7'h5F][30];
        pipeFirstUsed_pipe_pipe_v_5 = _RANDOM[7'h5F][31];
        pipeFirstUsed_pipe_pipe_b_5 = _RANDOM[7'h60][0];
        pipeFire_pipe_v_5 = _RANDOM[7'h60][1];
        pipeFire_pipe_b_5 = _RANDOM[7'h60][2];
        pipeFire_pipe_pipe_v_5 = _RANDOM[7'h60][3];
        pipeFire_pipe_pipe_b_5 = _RANDOM[7'h60][4];
        pipeBank_pipe_v_6 = _RANDOM[7'h60][5];
        pipeBank_pipe_pipe_v_6 = _RANDOM[7'h60][7];
        pipeBank_pipe_pipe_b_6 = _RANDOM[7'h60][8];
        pipeFirstUsed_pipe_v_6 = _RANDOM[7'h60][9];
        pipeFirstUsed_pipe_b_6 = _RANDOM[7'h60][10];
        pipeFirstUsed_pipe_pipe_v_6 = _RANDOM[7'h60][11];
        pipeFirstUsed_pipe_pipe_b_6 = _RANDOM[7'h60][12];
        pipeFire_pipe_v_6 = _RANDOM[7'h60][13];
        pipeFire_pipe_b_6 = _RANDOM[7'h60][14];
        pipeFire_pipe_pipe_v_6 = _RANDOM[7'h60][15];
        pipeFire_pipe_pipe_b_6 = _RANDOM[7'h60][16];
        pipeBank_pipe_v_7 = _RANDOM[7'h60][17];
        pipeBank_pipe_pipe_v_7 = _RANDOM[7'h60][19];
        pipeBank_pipe_pipe_b_7 = _RANDOM[7'h60][20];
        pipeFirstUsed_pipe_v_7 = _RANDOM[7'h60][21];
        pipeFirstUsed_pipe_b_7 = _RANDOM[7'h60][22];
        pipeFirstUsed_pipe_pipe_v_7 = _RANDOM[7'h60][23];
        pipeFirstUsed_pipe_pipe_b_7 = _RANDOM[7'h60][24];
        pipeFire_pipe_v_7 = _RANDOM[7'h60][25];
        pipeFire_pipe_b_7 = _RANDOM[7'h60][26];
        pipeFire_pipe_pipe_v_7 = _RANDOM[7'h60][27];
        pipeFire_pipe_pipe_b_7 = _RANDOM[7'h60][28];
        pipeBank_pipe_v_8 = _RANDOM[7'h60][29];
        pipeBank_pipe_pipe_v_8 = _RANDOM[7'h60][31];
        pipeBank_pipe_pipe_b_8 = _RANDOM[7'h61][0];
        pipeFirstUsed_pipe_v_8 = _RANDOM[7'h61][1];
        pipeFirstUsed_pipe_b_8 = _RANDOM[7'h61][2];
        pipeFirstUsed_pipe_pipe_v_8 = _RANDOM[7'h61][3];
        pipeFirstUsed_pipe_pipe_b_8 = _RANDOM[7'h61][4];
        pipeFire_pipe_v_8 = _RANDOM[7'h61][5];
        pipeFire_pipe_b_8 = _RANDOM[7'h61][6];
        pipeFire_pipe_pipe_v_8 = _RANDOM[7'h61][7];
        pipeFire_pipe_pipe_b_8 = _RANDOM[7'h61][8];
        pipeBank_pipe_v_9 = _RANDOM[7'h61][9];
        pipeBank_pipe_pipe_v_9 = _RANDOM[7'h61][11];
        pipeBank_pipe_pipe_b_9 = _RANDOM[7'h61][12];
        pipeFirstUsed_pipe_v_9 = _RANDOM[7'h61][13];
        pipeFirstUsed_pipe_b_9 = _RANDOM[7'h61][14];
        pipeFirstUsed_pipe_pipe_v_9 = _RANDOM[7'h61][15];
        pipeFirstUsed_pipe_pipe_b_9 = _RANDOM[7'h61][16];
        pipeFire_pipe_v_9 = _RANDOM[7'h61][17];
        pipeFire_pipe_b_9 = _RANDOM[7'h61][18];
        pipeFire_pipe_pipe_v_9 = _RANDOM[7'h61][19];
        pipeFire_pipe_pipe_b_9 = _RANDOM[7'h61][20];
        pipeBank_pipe_v_10 = _RANDOM[7'h61][21];
        pipeBank_pipe_pipe_v_10 = _RANDOM[7'h61][23];
        pipeBank_pipe_pipe_b_10 = _RANDOM[7'h61][24];
        pipeFirstUsed_pipe_v_10 = _RANDOM[7'h61][25];
        pipeFirstUsed_pipe_b_10 = _RANDOM[7'h61][26];
        pipeFirstUsed_pipe_pipe_v_10 = _RANDOM[7'h61][27];
        pipeFirstUsed_pipe_pipe_b_10 = _RANDOM[7'h61][28];
        pipeFire_pipe_v_10 = _RANDOM[7'h61][29];
        pipeFire_pipe_b_10 = _RANDOM[7'h61][30];
        pipeFire_pipe_pipe_v_10 = _RANDOM[7'h61][31];
        pipeFire_pipe_pipe_b_10 = _RANDOM[7'h62][0];
        pipeBank_pipe_v_11 = _RANDOM[7'h62][1];
        pipeBank_pipe_pipe_v_11 = _RANDOM[7'h62][3];
        pipeBank_pipe_pipe_b_11 = _RANDOM[7'h62][4];
        pipeFirstUsed_pipe_v_11 = _RANDOM[7'h62][5];
        pipeFirstUsed_pipe_b_11 = _RANDOM[7'h62][6];
        pipeFirstUsed_pipe_pipe_v_11 = _RANDOM[7'h62][7];
        pipeFirstUsed_pipe_pipe_b_11 = _RANDOM[7'h62][8];
        pipeFire_pipe_v_11 = _RANDOM[7'h62][9];
        pipeFire_pipe_b_11 = _RANDOM[7'h62][10];
        pipeFire_pipe_pipe_v_11 = _RANDOM[7'h62][11];
        pipeFire_pipe_pipe_b_11 = _RANDOM[7'h62][12];
        pipeBank_pipe_v_12 = _RANDOM[7'h62][13];
        pipeBank_pipe_pipe_v_12 = _RANDOM[7'h62][15];
        pipeBank_pipe_pipe_b_12 = _RANDOM[7'h62][16];
        pipeFirstUsed_pipe_v_12 = _RANDOM[7'h62][17];
        pipeFirstUsed_pipe_b_12 = _RANDOM[7'h62][18];
        pipeFirstUsed_pipe_pipe_v_12 = _RANDOM[7'h62][19];
        pipeFirstUsed_pipe_pipe_b_12 = _RANDOM[7'h62][20];
        pipeFire_pipe_v_12 = _RANDOM[7'h62][21];
        pipeFire_pipe_b_12 = _RANDOM[7'h62][22];
        pipeFire_pipe_pipe_v_12 = _RANDOM[7'h62][23];
        pipeFire_pipe_pipe_b_12 = _RANDOM[7'h62][24];
        pipeBank_pipe_v_13 = _RANDOM[7'h62][25];
        pipeBank_pipe_pipe_v_13 = _RANDOM[7'h62][27];
        pipeBank_pipe_pipe_b_13 = _RANDOM[7'h62][28];
        pipeFirstUsed_pipe_v_13 = _RANDOM[7'h62][29];
        pipeFirstUsed_pipe_b_13 = _RANDOM[7'h62][30];
        pipeFirstUsed_pipe_pipe_v_13 = _RANDOM[7'h62][31];
        pipeFirstUsed_pipe_pipe_b_13 = _RANDOM[7'h63][0];
        pipeFire_pipe_v_13 = _RANDOM[7'h63][1];
        pipeFire_pipe_b_13 = _RANDOM[7'h63][2];
        pipeFire_pipe_pipe_v_13 = _RANDOM[7'h63][3];
        pipeFire_pipe_pipe_b_13 = _RANDOM[7'h63][4];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  ChainingCheck readCheckResult_0_checkModule (
    .read_vs                 (readCheck_0_vs),
    .read_offset             (readCheck_0_offset),
    .read_instructionIndex   (readCheck_0_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_0_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_0_bits_instIndex),
    .record_bits_elementMask (chainingRecord_0_bits_elementMask),
    .recordValid             (recordValidVec_0),
    .checkResult             (_readCheckResult_0_checkModule_checkResult)
  );
  ChainingCheck readCheckResult_0_checkModule_1 (
    .read_vs                 (readCheck_0_vs),
    .read_offset             (readCheck_0_offset),
    .read_instructionIndex   (readCheck_0_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_1_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_1_bits_instIndex),
    .record_bits_elementMask (chainingRecord_1_bits_elementMask),
    .recordValid             (recordValidVec_1),
    .checkResult             (_readCheckResult_0_checkModule_1_checkResult)
  );
  ChainingCheck readCheckResult_0_checkModule_2 (
    .read_vs                 (readCheck_0_vs),
    .read_offset             (readCheck_0_offset),
    .read_instructionIndex   (readCheck_0_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_2_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_2_bits_instIndex),
    .record_bits_elementMask (chainingRecord_2_bits_elementMask),
    .recordValid             (recordValidVec_2),
    .checkResult             (_readCheckResult_0_checkModule_2_checkResult)
  );
  ChainingCheck readCheckResult_0_checkModule_3 (
    .read_vs                 (readCheck_0_vs),
    .read_offset             (readCheck_0_offset),
    .read_instructionIndex   (readCheck_0_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_3_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_3_bits_instIndex),
    .record_bits_elementMask (chainingRecord_3_bits_elementMask),
    .recordValid             (recordValidVec_3),
    .checkResult             (_readCheckResult_0_checkModule_3_checkResult)
  );
  ChainingCheck readCheckResult_0_checkModule_4 (
    .read_vs                 (readCheck_0_vs),
    .read_offset             (readCheck_0_offset),
    .read_instructionIndex   (readCheck_0_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_4_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_4_bits_instIndex),
    .record_bits_elementMask (chainingRecord_4_bits_elementMask),
    .recordValid             (recordValidVec_4),
    .checkResult             (_readCheckResult_0_checkModule_4_checkResult)
  );
  ChainingCheck readCheckResult_1_checkModule (
    .read_vs                 (readCheck_1_vs),
    .read_offset             (readCheck_1_offset),
    .read_instructionIndex   (readCheck_1_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_0_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_0_bits_instIndex),
    .record_bits_elementMask (chainingRecord_0_bits_elementMask),
    .recordValid             (recordValidVec_0),
    .checkResult             (_readCheckResult_1_checkModule_checkResult)
  );
  ChainingCheck readCheckResult_1_checkModule_1 (
    .read_vs                 (readCheck_1_vs),
    .read_offset             (readCheck_1_offset),
    .read_instructionIndex   (readCheck_1_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_1_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_1_bits_instIndex),
    .record_bits_elementMask (chainingRecord_1_bits_elementMask),
    .recordValid             (recordValidVec_1),
    .checkResult             (_readCheckResult_1_checkModule_1_checkResult)
  );
  ChainingCheck readCheckResult_1_checkModule_2 (
    .read_vs                 (readCheck_1_vs),
    .read_offset             (readCheck_1_offset),
    .read_instructionIndex   (readCheck_1_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_2_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_2_bits_instIndex),
    .record_bits_elementMask (chainingRecord_2_bits_elementMask),
    .recordValid             (recordValidVec_2),
    .checkResult             (_readCheckResult_1_checkModule_2_checkResult)
  );
  ChainingCheck readCheckResult_1_checkModule_3 (
    .read_vs                 (readCheck_1_vs),
    .read_offset             (readCheck_1_offset),
    .read_instructionIndex   (readCheck_1_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_3_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_3_bits_instIndex),
    .record_bits_elementMask (chainingRecord_3_bits_elementMask),
    .recordValid             (recordValidVec_3),
    .checkResult             (_readCheckResult_1_checkModule_3_checkResult)
  );
  ChainingCheck readCheckResult_1_checkModule_4 (
    .read_vs                 (readCheck_1_vs),
    .read_offset             (readCheck_1_offset),
    .read_instructionIndex   (readCheck_1_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_4_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_4_bits_instIndex),
    .record_bits_elementMask (chainingRecord_4_bits_elementMask),
    .recordValid             (recordValidVec_4),
    .checkResult             (_readCheckResult_1_checkModule_4_checkResult)
  );
  ChainingCheck readCheckResult_2_checkModule (
    .read_vs                 (readCheck_2_vs),
    .read_offset             (readCheck_2_offset),
    .read_instructionIndex   (readCheck_2_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_0_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_0_bits_instIndex),
    .record_bits_elementMask (chainingRecord_0_bits_elementMask),
    .recordValid             (recordValidVec_0),
    .checkResult             (_readCheckResult_2_checkModule_checkResult)
  );
  ChainingCheck readCheckResult_2_checkModule_1 (
    .read_vs                 (readCheck_2_vs),
    .read_offset             (readCheck_2_offset),
    .read_instructionIndex   (readCheck_2_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_1_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_1_bits_instIndex),
    .record_bits_elementMask (chainingRecord_1_bits_elementMask),
    .recordValid             (recordValidVec_1),
    .checkResult             (_readCheckResult_2_checkModule_1_checkResult)
  );
  ChainingCheck readCheckResult_2_checkModule_2 (
    .read_vs                 (readCheck_2_vs),
    .read_offset             (readCheck_2_offset),
    .read_instructionIndex   (readCheck_2_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_2_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_2_bits_instIndex),
    .record_bits_elementMask (chainingRecord_2_bits_elementMask),
    .recordValid             (recordValidVec_2),
    .checkResult             (_readCheckResult_2_checkModule_2_checkResult)
  );
  ChainingCheck readCheckResult_2_checkModule_3 (
    .read_vs                 (readCheck_2_vs),
    .read_offset             (readCheck_2_offset),
    .read_instructionIndex   (readCheck_2_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_3_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_3_bits_instIndex),
    .record_bits_elementMask (chainingRecord_3_bits_elementMask),
    .recordValid             (recordValidVec_3),
    .checkResult             (_readCheckResult_2_checkModule_3_checkResult)
  );
  ChainingCheck readCheckResult_2_checkModule_4 (
    .read_vs                 (readCheck_2_vs),
    .read_offset             (readCheck_2_offset),
    .read_instructionIndex   (readCheck_2_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_4_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_4_bits_instIndex),
    .record_bits_elementMask (chainingRecord_4_bits_elementMask),
    .recordValid             (recordValidVec_4),
    .checkResult             (_readCheckResult_2_checkModule_4_checkResult)
  );
  ChainingCheck readCheckResult_3_checkModule (
    .read_vs                 (readCheck_3_vs),
    .read_offset             (readCheck_3_offset),
    .read_instructionIndex   (readCheck_3_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_0_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_0_bits_instIndex),
    .record_bits_elementMask (chainingRecord_0_bits_elementMask),
    .recordValid             (recordValidVec_0),
    .checkResult             (_readCheckResult_3_checkModule_checkResult)
  );
  ChainingCheck readCheckResult_3_checkModule_1 (
    .read_vs                 (readCheck_3_vs),
    .read_offset             (readCheck_3_offset),
    .read_instructionIndex   (readCheck_3_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_1_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_1_bits_instIndex),
    .record_bits_elementMask (chainingRecord_1_bits_elementMask),
    .recordValid             (recordValidVec_1),
    .checkResult             (_readCheckResult_3_checkModule_1_checkResult)
  );
  ChainingCheck readCheckResult_3_checkModule_2 (
    .read_vs                 (readCheck_3_vs),
    .read_offset             (readCheck_3_offset),
    .read_instructionIndex   (readCheck_3_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_2_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_2_bits_instIndex),
    .record_bits_elementMask (chainingRecord_2_bits_elementMask),
    .recordValid             (recordValidVec_2),
    .checkResult             (_readCheckResult_3_checkModule_2_checkResult)
  );
  ChainingCheck readCheckResult_3_checkModule_3 (
    .read_vs                 (readCheck_3_vs),
    .read_offset             (readCheck_3_offset),
    .read_instructionIndex   (readCheck_3_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_3_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_3_bits_instIndex),
    .record_bits_elementMask (chainingRecord_3_bits_elementMask),
    .recordValid             (recordValidVec_3),
    .checkResult             (_readCheckResult_3_checkModule_3_checkResult)
  );
  ChainingCheck readCheckResult_3_checkModule_4 (
    .read_vs                 (readCheck_3_vs),
    .read_offset             (readCheck_3_offset),
    .read_instructionIndex   (readCheck_3_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_4_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_4_bits_instIndex),
    .record_bits_elementMask (chainingRecord_4_bits_elementMask),
    .recordValid             (recordValidVec_4),
    .checkResult             (_readCheckResult_3_checkModule_4_checkResult)
  );
  ChainingCheck readCheckResult_4_checkModule (
    .read_vs                 (readCheck_4_vs),
    .read_offset             (readCheck_4_offset),
    .read_instructionIndex   (readCheck_4_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_0_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_0_bits_instIndex),
    .record_bits_elementMask (chainingRecord_0_bits_elementMask),
    .recordValid             (recordValidVec_0),
    .checkResult             (_readCheckResult_4_checkModule_checkResult)
  );
  ChainingCheck readCheckResult_4_checkModule_1 (
    .read_vs                 (readCheck_4_vs),
    .read_offset             (readCheck_4_offset),
    .read_instructionIndex   (readCheck_4_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_1_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_1_bits_instIndex),
    .record_bits_elementMask (chainingRecord_1_bits_elementMask),
    .recordValid             (recordValidVec_1),
    .checkResult             (_readCheckResult_4_checkModule_1_checkResult)
  );
  ChainingCheck readCheckResult_4_checkModule_2 (
    .read_vs                 (readCheck_4_vs),
    .read_offset             (readCheck_4_offset),
    .read_instructionIndex   (readCheck_4_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_2_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_2_bits_instIndex),
    .record_bits_elementMask (chainingRecord_2_bits_elementMask),
    .recordValid             (recordValidVec_2),
    .checkResult             (_readCheckResult_4_checkModule_2_checkResult)
  );
  ChainingCheck readCheckResult_4_checkModule_3 (
    .read_vs                 (readCheck_4_vs),
    .read_offset             (readCheck_4_offset),
    .read_instructionIndex   (readCheck_4_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_3_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_3_bits_instIndex),
    .record_bits_elementMask (chainingRecord_3_bits_elementMask),
    .recordValid             (recordValidVec_3),
    .checkResult             (_readCheckResult_4_checkModule_3_checkResult)
  );
  ChainingCheck readCheckResult_4_checkModule_4 (
    .read_vs                 (readCheck_4_vs),
    .read_offset             (readCheck_4_offset),
    .read_instructionIndex   (readCheck_4_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_4_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_4_bits_instIndex),
    .record_bits_elementMask (chainingRecord_4_bits_elementMask),
    .recordValid             (recordValidVec_4),
    .checkResult             (_readCheckResult_4_checkModule_4_checkResult)
  );
  ChainingCheck readCheckResult_5_checkModule (
    .read_vs                 (readCheck_5_vs),
    .read_offset             (readCheck_5_offset),
    .read_instructionIndex   (readCheck_5_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_0_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_0_bits_instIndex),
    .record_bits_elementMask (chainingRecord_0_bits_elementMask),
    .recordValid             (recordValidVec_0),
    .checkResult             (_readCheckResult_5_checkModule_checkResult)
  );
  ChainingCheck readCheckResult_5_checkModule_1 (
    .read_vs                 (readCheck_5_vs),
    .read_offset             (readCheck_5_offset),
    .read_instructionIndex   (readCheck_5_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_1_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_1_bits_instIndex),
    .record_bits_elementMask (chainingRecord_1_bits_elementMask),
    .recordValid             (recordValidVec_1),
    .checkResult             (_readCheckResult_5_checkModule_1_checkResult)
  );
  ChainingCheck readCheckResult_5_checkModule_2 (
    .read_vs                 (readCheck_5_vs),
    .read_offset             (readCheck_5_offset),
    .read_instructionIndex   (readCheck_5_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_2_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_2_bits_instIndex),
    .record_bits_elementMask (chainingRecord_2_bits_elementMask),
    .recordValid             (recordValidVec_2),
    .checkResult             (_readCheckResult_5_checkModule_2_checkResult)
  );
  ChainingCheck readCheckResult_5_checkModule_3 (
    .read_vs                 (readCheck_5_vs),
    .read_offset             (readCheck_5_offset),
    .read_instructionIndex   (readCheck_5_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_3_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_3_bits_instIndex),
    .record_bits_elementMask (chainingRecord_3_bits_elementMask),
    .recordValid             (recordValidVec_3),
    .checkResult             (_readCheckResult_5_checkModule_3_checkResult)
  );
  ChainingCheck readCheckResult_5_checkModule_4 (
    .read_vs                 (readCheck_5_vs),
    .read_offset             (readCheck_5_offset),
    .read_instructionIndex   (readCheck_5_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_4_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_4_bits_instIndex),
    .record_bits_elementMask (chainingRecord_4_bits_elementMask),
    .recordValid             (recordValidVec_4),
    .checkResult             (_readCheckResult_5_checkModule_4_checkResult)
  );
  ChainingCheck readCheckResult_6_checkModule (
    .read_vs                 (readCheck_6_vs),
    .read_offset             (readCheck_6_offset),
    .read_instructionIndex   (readCheck_6_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_0_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_0_bits_instIndex),
    .record_bits_elementMask (chainingRecord_0_bits_elementMask),
    .recordValid             (recordValidVec_0),
    .checkResult             (_readCheckResult_6_checkModule_checkResult)
  );
  ChainingCheck readCheckResult_6_checkModule_1 (
    .read_vs                 (readCheck_6_vs),
    .read_offset             (readCheck_6_offset),
    .read_instructionIndex   (readCheck_6_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_1_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_1_bits_instIndex),
    .record_bits_elementMask (chainingRecord_1_bits_elementMask),
    .recordValid             (recordValidVec_1),
    .checkResult             (_readCheckResult_6_checkModule_1_checkResult)
  );
  ChainingCheck readCheckResult_6_checkModule_2 (
    .read_vs                 (readCheck_6_vs),
    .read_offset             (readCheck_6_offset),
    .read_instructionIndex   (readCheck_6_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_2_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_2_bits_instIndex),
    .record_bits_elementMask (chainingRecord_2_bits_elementMask),
    .recordValid             (recordValidVec_2),
    .checkResult             (_readCheckResult_6_checkModule_2_checkResult)
  );
  ChainingCheck readCheckResult_6_checkModule_3 (
    .read_vs                 (readCheck_6_vs),
    .read_offset             (readCheck_6_offset),
    .read_instructionIndex   (readCheck_6_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_3_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_3_bits_instIndex),
    .record_bits_elementMask (chainingRecord_3_bits_elementMask),
    .recordValid             (recordValidVec_3),
    .checkResult             (_readCheckResult_6_checkModule_3_checkResult)
  );
  ChainingCheck readCheckResult_6_checkModule_4 (
    .read_vs                 (readCheck_6_vs),
    .read_offset             (readCheck_6_offset),
    .read_instructionIndex   (readCheck_6_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_4_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_4_bits_instIndex),
    .record_bits_elementMask (chainingRecord_4_bits_elementMask),
    .recordValid             (recordValidVec_4),
    .checkResult             (_readCheckResult_6_checkModule_4_checkResult)
  );
  ChainingCheck readCheckResult_7_checkModule (
    .read_vs                 (readCheck_7_vs),
    .read_offset             (readCheck_7_offset),
    .read_instructionIndex   (readCheck_7_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_0_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_0_bits_instIndex),
    .record_bits_elementMask (chainingRecord_0_bits_elementMask),
    .recordValid             (recordValidVec_0),
    .checkResult             (_readCheckResult_7_checkModule_checkResult)
  );
  ChainingCheck readCheckResult_7_checkModule_1 (
    .read_vs                 (readCheck_7_vs),
    .read_offset             (readCheck_7_offset),
    .read_instructionIndex   (readCheck_7_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_1_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_1_bits_instIndex),
    .record_bits_elementMask (chainingRecord_1_bits_elementMask),
    .recordValid             (recordValidVec_1),
    .checkResult             (_readCheckResult_7_checkModule_1_checkResult)
  );
  ChainingCheck readCheckResult_7_checkModule_2 (
    .read_vs                 (readCheck_7_vs),
    .read_offset             (readCheck_7_offset),
    .read_instructionIndex   (readCheck_7_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_2_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_2_bits_instIndex),
    .record_bits_elementMask (chainingRecord_2_bits_elementMask),
    .recordValid             (recordValidVec_2),
    .checkResult             (_readCheckResult_7_checkModule_2_checkResult)
  );
  ChainingCheck readCheckResult_7_checkModule_3 (
    .read_vs                 (readCheck_7_vs),
    .read_offset             (readCheck_7_offset),
    .read_instructionIndex   (readCheck_7_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_3_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_3_bits_instIndex),
    .record_bits_elementMask (chainingRecord_3_bits_elementMask),
    .recordValid             (recordValidVec_3),
    .checkResult             (_readCheckResult_7_checkModule_3_checkResult)
  );
  ChainingCheck readCheckResult_7_checkModule_4 (
    .read_vs                 (readCheck_7_vs),
    .read_offset             (readCheck_7_offset),
    .read_instructionIndex   (readCheck_7_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_4_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_4_bits_instIndex),
    .record_bits_elementMask (chainingRecord_4_bits_elementMask),
    .recordValid             (recordValidVec_4),
    .checkResult             (_readCheckResult_7_checkModule_4_checkResult)
  );
  ChainingCheck readCheckResult_8_checkModule (
    .read_vs                 (readCheck_8_vs),
    .read_offset             (readCheck_8_offset),
    .read_instructionIndex   (readCheck_8_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_0_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_0_bits_instIndex),
    .record_bits_elementMask (chainingRecord_0_bits_elementMask),
    .recordValid             (recordValidVec_0),
    .checkResult             (_readCheckResult_8_checkModule_checkResult)
  );
  ChainingCheck readCheckResult_8_checkModule_1 (
    .read_vs                 (readCheck_8_vs),
    .read_offset             (readCheck_8_offset),
    .read_instructionIndex   (readCheck_8_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_1_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_1_bits_instIndex),
    .record_bits_elementMask (chainingRecord_1_bits_elementMask),
    .recordValid             (recordValidVec_1),
    .checkResult             (_readCheckResult_8_checkModule_1_checkResult)
  );
  ChainingCheck readCheckResult_8_checkModule_2 (
    .read_vs                 (readCheck_8_vs),
    .read_offset             (readCheck_8_offset),
    .read_instructionIndex   (readCheck_8_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_2_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_2_bits_instIndex),
    .record_bits_elementMask (chainingRecord_2_bits_elementMask),
    .recordValid             (recordValidVec_2),
    .checkResult             (_readCheckResult_8_checkModule_2_checkResult)
  );
  ChainingCheck readCheckResult_8_checkModule_3 (
    .read_vs                 (readCheck_8_vs),
    .read_offset             (readCheck_8_offset),
    .read_instructionIndex   (readCheck_8_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_3_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_3_bits_instIndex),
    .record_bits_elementMask (chainingRecord_3_bits_elementMask),
    .recordValid             (recordValidVec_3),
    .checkResult             (_readCheckResult_8_checkModule_3_checkResult)
  );
  ChainingCheck readCheckResult_8_checkModule_4 (
    .read_vs                 (readCheck_8_vs),
    .read_offset             (readCheck_8_offset),
    .read_instructionIndex   (readCheck_8_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_4_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_4_bits_instIndex),
    .record_bits_elementMask (chainingRecord_4_bits_elementMask),
    .recordValid             (recordValidVec_4),
    .checkResult             (_readCheckResult_8_checkModule_4_checkResult)
  );
  ChainingCheck readCheckResult_9_checkModule (
    .read_vs                 (readCheck_9_vs),
    .read_offset             (readCheck_9_offset),
    .read_instructionIndex   (readCheck_9_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_0_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_0_bits_instIndex),
    .record_bits_elementMask (chainingRecord_0_bits_elementMask),
    .recordValid             (recordValidVec_0),
    .checkResult             (_readCheckResult_9_checkModule_checkResult)
  );
  ChainingCheck readCheckResult_9_checkModule_1 (
    .read_vs                 (readCheck_9_vs),
    .read_offset             (readCheck_9_offset),
    .read_instructionIndex   (readCheck_9_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_1_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_1_bits_instIndex),
    .record_bits_elementMask (chainingRecord_1_bits_elementMask),
    .recordValid             (recordValidVec_1),
    .checkResult             (_readCheckResult_9_checkModule_1_checkResult)
  );
  ChainingCheck readCheckResult_9_checkModule_2 (
    .read_vs                 (readCheck_9_vs),
    .read_offset             (readCheck_9_offset),
    .read_instructionIndex   (readCheck_9_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_2_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_2_bits_instIndex),
    .record_bits_elementMask (chainingRecord_2_bits_elementMask),
    .recordValid             (recordValidVec_2),
    .checkResult             (_readCheckResult_9_checkModule_2_checkResult)
  );
  ChainingCheck readCheckResult_9_checkModule_3 (
    .read_vs                 (readCheck_9_vs),
    .read_offset             (readCheck_9_offset),
    .read_instructionIndex   (readCheck_9_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_3_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_3_bits_instIndex),
    .record_bits_elementMask (chainingRecord_3_bits_elementMask),
    .recordValid             (recordValidVec_3),
    .checkResult             (_readCheckResult_9_checkModule_3_checkResult)
  );
  ChainingCheck readCheckResult_9_checkModule_4 (
    .read_vs                 (readCheck_9_vs),
    .read_offset             (readCheck_9_offset),
    .read_instructionIndex   (readCheck_9_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_4_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_4_bits_instIndex),
    .record_bits_elementMask (chainingRecord_4_bits_elementMask),
    .recordValid             (recordValidVec_4),
    .checkResult             (_readCheckResult_9_checkModule_4_checkResult)
  );
  ChainingCheck readCheckResult_10_checkModule (
    .read_vs                 (readCheck_10_vs),
    .read_offset             (readCheck_10_offset),
    .read_instructionIndex   (readCheck_10_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_0_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_0_bits_instIndex),
    .record_bits_elementMask (chainingRecord_0_bits_elementMask),
    .recordValid             (recordValidVec_0),
    .checkResult             (_readCheckResult_10_checkModule_checkResult)
  );
  ChainingCheck readCheckResult_10_checkModule_1 (
    .read_vs                 (readCheck_10_vs),
    .read_offset             (readCheck_10_offset),
    .read_instructionIndex   (readCheck_10_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_1_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_1_bits_instIndex),
    .record_bits_elementMask (chainingRecord_1_bits_elementMask),
    .recordValid             (recordValidVec_1),
    .checkResult             (_readCheckResult_10_checkModule_1_checkResult)
  );
  ChainingCheck readCheckResult_10_checkModule_2 (
    .read_vs                 (readCheck_10_vs),
    .read_offset             (readCheck_10_offset),
    .read_instructionIndex   (readCheck_10_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_2_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_2_bits_instIndex),
    .record_bits_elementMask (chainingRecord_2_bits_elementMask),
    .recordValid             (recordValidVec_2),
    .checkResult             (_readCheckResult_10_checkModule_2_checkResult)
  );
  ChainingCheck readCheckResult_10_checkModule_3 (
    .read_vs                 (readCheck_10_vs),
    .read_offset             (readCheck_10_offset),
    .read_instructionIndex   (readCheck_10_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_3_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_3_bits_instIndex),
    .record_bits_elementMask (chainingRecord_3_bits_elementMask),
    .recordValid             (recordValidVec_3),
    .checkResult             (_readCheckResult_10_checkModule_3_checkResult)
  );
  ChainingCheck readCheckResult_10_checkModule_4 (
    .read_vs                 (readCheck_10_vs),
    .read_offset             (readCheck_10_offset),
    .read_instructionIndex   (readCheck_10_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_4_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_4_bits_instIndex),
    .record_bits_elementMask (chainingRecord_4_bits_elementMask),
    .recordValid             (recordValidVec_4),
    .checkResult             (_readCheckResult_10_checkModule_4_checkResult)
  );
  ChainingCheck readCheckResult_11_checkModule (
    .read_vs                 (readCheck_11_vs),
    .read_offset             (readCheck_11_offset),
    .read_instructionIndex   (readCheck_11_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_0_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_0_bits_instIndex),
    .record_bits_elementMask (chainingRecord_0_bits_elementMask),
    .recordValid             (recordValidVec_0),
    .checkResult             (_readCheckResult_11_checkModule_checkResult)
  );
  ChainingCheck readCheckResult_11_checkModule_1 (
    .read_vs                 (readCheck_11_vs),
    .read_offset             (readCheck_11_offset),
    .read_instructionIndex   (readCheck_11_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_1_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_1_bits_instIndex),
    .record_bits_elementMask (chainingRecord_1_bits_elementMask),
    .recordValid             (recordValidVec_1),
    .checkResult             (_readCheckResult_11_checkModule_1_checkResult)
  );
  ChainingCheck readCheckResult_11_checkModule_2 (
    .read_vs                 (readCheck_11_vs),
    .read_offset             (readCheck_11_offset),
    .read_instructionIndex   (readCheck_11_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_2_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_2_bits_instIndex),
    .record_bits_elementMask (chainingRecord_2_bits_elementMask),
    .recordValid             (recordValidVec_2),
    .checkResult             (_readCheckResult_11_checkModule_2_checkResult)
  );
  ChainingCheck readCheckResult_11_checkModule_3 (
    .read_vs                 (readCheck_11_vs),
    .read_offset             (readCheck_11_offset),
    .read_instructionIndex   (readCheck_11_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_3_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_3_bits_instIndex),
    .record_bits_elementMask (chainingRecord_3_bits_elementMask),
    .recordValid             (recordValidVec_3),
    .checkResult             (_readCheckResult_11_checkModule_3_checkResult)
  );
  ChainingCheck readCheckResult_11_checkModule_4 (
    .read_vs                 (readCheck_11_vs),
    .read_offset             (readCheck_11_offset),
    .read_instructionIndex   (readCheck_11_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_4_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_4_bits_instIndex),
    .record_bits_elementMask (chainingRecord_4_bits_elementMask),
    .recordValid             (recordValidVec_4),
    .checkResult             (_readCheckResult_11_checkModule_4_checkResult)
  );
  ChainingCheck readCheckResult_12_checkModule (
    .read_vs                 (readCheck_12_vs),
    .read_offset             (readCheck_12_offset),
    .read_instructionIndex   (readCheck_12_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_0_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_0_bits_instIndex),
    .record_bits_elementMask (chainingRecord_0_bits_elementMask),
    .recordValid             (recordValidVec_0),
    .checkResult             (_readCheckResult_12_checkModule_checkResult)
  );
  ChainingCheck readCheckResult_12_checkModule_1 (
    .read_vs                 (readCheck_12_vs),
    .read_offset             (readCheck_12_offset),
    .read_instructionIndex   (readCheck_12_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_1_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_1_bits_instIndex),
    .record_bits_elementMask (chainingRecord_1_bits_elementMask),
    .recordValid             (recordValidVec_1),
    .checkResult             (_readCheckResult_12_checkModule_1_checkResult)
  );
  ChainingCheck readCheckResult_12_checkModule_2 (
    .read_vs                 (readCheck_12_vs),
    .read_offset             (readCheck_12_offset),
    .read_instructionIndex   (readCheck_12_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_2_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_2_bits_instIndex),
    .record_bits_elementMask (chainingRecord_2_bits_elementMask),
    .recordValid             (recordValidVec_2),
    .checkResult             (_readCheckResult_12_checkModule_2_checkResult)
  );
  ChainingCheck readCheckResult_12_checkModule_3 (
    .read_vs                 (readCheck_12_vs),
    .read_offset             (readCheck_12_offset),
    .read_instructionIndex   (readCheck_12_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_3_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_3_bits_instIndex),
    .record_bits_elementMask (chainingRecord_3_bits_elementMask),
    .recordValid             (recordValidVec_3),
    .checkResult             (_readCheckResult_12_checkModule_3_checkResult)
  );
  ChainingCheck readCheckResult_12_checkModule_4 (
    .read_vs                 (readCheck_12_vs),
    .read_offset             (readCheck_12_offset),
    .read_instructionIndex   (readCheck_12_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_4_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_4_bits_instIndex),
    .record_bits_elementMask (chainingRecord_4_bits_elementMask),
    .recordValid             (recordValidVec_4),
    .checkResult             (_readCheckResult_12_checkModule_4_checkResult)
  );
  ChainingCheck readCheckResult_13_checkModule (
    .read_vs                 (readCheck_13_vs),
    .read_offset             (readCheck_13_offset),
    .read_instructionIndex   (readCheck_13_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_0_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_0_bits_instIndex),
    .record_bits_elementMask (chainingRecord_0_bits_elementMask),
    .recordValid             (recordValidVec_0),
    .checkResult             (_readCheckResult_13_checkModule_checkResult)
  );
  ChainingCheck readCheckResult_13_checkModule_1 (
    .read_vs                 (readCheck_13_vs),
    .read_offset             (readCheck_13_offset),
    .read_instructionIndex   (readCheck_13_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_1_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_1_bits_instIndex),
    .record_bits_elementMask (chainingRecord_1_bits_elementMask),
    .recordValid             (recordValidVec_1),
    .checkResult             (_readCheckResult_13_checkModule_1_checkResult)
  );
  ChainingCheck readCheckResult_13_checkModule_2 (
    .read_vs                 (readCheck_13_vs),
    .read_offset             (readCheck_13_offset),
    .read_instructionIndex   (readCheck_13_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_2_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_2_bits_instIndex),
    .record_bits_elementMask (chainingRecord_2_bits_elementMask),
    .recordValid             (recordValidVec_2),
    .checkResult             (_readCheckResult_13_checkModule_2_checkResult)
  );
  ChainingCheck readCheckResult_13_checkModule_3 (
    .read_vs                 (readCheck_13_vs),
    .read_offset             (readCheck_13_offset),
    .read_instructionIndex   (readCheck_13_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_3_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_3_bits_instIndex),
    .record_bits_elementMask (chainingRecord_3_bits_elementMask),
    .recordValid             (recordValidVec_3),
    .checkResult             (_readCheckResult_13_checkModule_3_checkResult)
  );
  ChainingCheck readCheckResult_13_checkModule_4 (
    .read_vs                 (readCheck_13_vs),
    .read_offset             (readCheck_13_offset),
    .read_instructionIndex   (readCheck_13_instructionIndex),
    .record_bits_vd_valid    (chainingRecord_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecord_4_bits_vd_bits),
    .record_bits_instIndex   (chainingRecord_4_bits_instIndex),
    .record_bits_elementMask (chainingRecord_4_bits_elementMask),
    .recordValid             (recordValidVec_4),
    .checkResult             (_readCheckResult_13_checkModule_4_checkResult)
  );
  ChainingCheck checkResult_ChainingCheck_readPort13_record0 (
    .read_vs                 (readRequests_13_bits_vs_0),
    .read_offset             (readRequests_13_bits_offset_0),
    .read_instructionIndex   (readRequests_13_bits_instructionIndex_0),
    .record_bits_vd_valid    (chainingRecordCopy_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_0_bits_vd_bits),
    .record_bits_instIndex   (chainingRecordCopy_0_bits_instIndex),
    .record_bits_elementMask (chainingRecordCopy_0_bits_elementMask),
    .recordValid             (recordValidVec_0),
    .checkResult             (_checkResult_ChainingCheck_readPort13_record0_checkResult)
  );
  ChainingCheck checkResult_ChainingCheck_readPort13_record1 (
    .read_vs                 (readRequests_13_bits_vs_0),
    .read_offset             (readRequests_13_bits_offset_0),
    .read_instructionIndex   (readRequests_13_bits_instructionIndex_0),
    .record_bits_vd_valid    (chainingRecordCopy_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_1_bits_vd_bits),
    .record_bits_instIndex   (chainingRecordCopy_1_bits_instIndex),
    .record_bits_elementMask (chainingRecordCopy_1_bits_elementMask),
    .recordValid             (recordValidVec_1),
    .checkResult             (_checkResult_ChainingCheck_readPort13_record1_checkResult)
  );
  ChainingCheck checkResult_ChainingCheck_readPort13_record2 (
    .read_vs                 (readRequests_13_bits_vs_0),
    .read_offset             (readRequests_13_bits_offset_0),
    .read_instructionIndex   (readRequests_13_bits_instructionIndex_0),
    .record_bits_vd_valid    (chainingRecordCopy_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_2_bits_vd_bits),
    .record_bits_instIndex   (chainingRecordCopy_2_bits_instIndex),
    .record_bits_elementMask (chainingRecordCopy_2_bits_elementMask),
    .recordValid             (recordValidVec_2),
    .checkResult             (_checkResult_ChainingCheck_readPort13_record2_checkResult)
  );
  ChainingCheck checkResult_ChainingCheck_readPort13_record3 (
    .read_vs                 (readRequests_13_bits_vs_0),
    .read_offset             (readRequests_13_bits_offset_0),
    .read_instructionIndex   (readRequests_13_bits_instructionIndex_0),
    .record_bits_vd_valid    (chainingRecordCopy_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_3_bits_vd_bits),
    .record_bits_instIndex   (chainingRecordCopy_3_bits_instIndex),
    .record_bits_elementMask (chainingRecordCopy_3_bits_elementMask),
    .recordValid             (recordValidVec_3),
    .checkResult             (_checkResult_ChainingCheck_readPort13_record3_checkResult)
  );
  ChainingCheck checkResult_ChainingCheck_readPort13_record4 (
    .read_vs                 (readRequests_13_bits_vs_0),
    .read_offset             (readRequests_13_bits_offset_0),
    .read_instructionIndex   (readRequests_13_bits_instructionIndex_0),
    .record_bits_vd_valid    (chainingRecordCopy_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_4_bits_vd_bits),
    .record_bits_instIndex   (chainingRecordCopy_4_bits_instIndex),
    .record_bits_elementMask (chainingRecordCopy_4_bits_elementMask),
    .recordValid             (recordValidVec_4),
    .checkResult             (_checkResult_ChainingCheck_readPort13_record4_checkResult)
  );
  sram_0R_0W_1RW_0M_1024x36 vrfSRAM_mem (
    .RW0_addr  (vrfSRAM_0_readwritePorts_0_address),
    .RW0_en    (vrfSRAM_0_readwritePorts_0_enable),
    .RW0_clk   (clock),
    .RW0_wmode (vrfSRAM_0_readwritePorts_0_isWrite),
    .RW0_wdata (vrfSRAM_0_readwritePorts_0_writeData),
    .RW0_rdata (vrfSRAM_0_readwritePorts_0_readData)
  );
  WriteCheck writeAllow_0_checkModule (
    .check_vd                (writeCheck_0_vd),
    .check_offset            (writeCheck_0_offset),
    .check_instructionIndex  (writeCheck_0_instructionIndex),
    .record_valid            (chainingRecordCopy_0_valid),
    .record_bits_vd_valid    (chainingRecordCopy_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_0_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_0_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_0_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_0_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_0_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_0_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_0_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_0_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_0_bits_elementMask),
    .checkResult             (_writeAllow_0_checkModule_checkResult)
  );
  WriteCheck writeAllow_0_checkModule_1 (
    .check_vd                (writeCheck_0_vd),
    .check_offset            (writeCheck_0_offset),
    .check_instructionIndex  (writeCheck_0_instructionIndex),
    .record_valid            (chainingRecordCopy_1_valid),
    .record_bits_vd_valid    (chainingRecordCopy_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_1_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_1_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_1_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_1_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_1_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_1_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_1_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_1_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_1_bits_elementMask),
    .checkResult             (_writeAllow_0_checkModule_1_checkResult)
  );
  WriteCheck writeAllow_0_checkModule_2 (
    .check_vd                (writeCheck_0_vd),
    .check_offset            (writeCheck_0_offset),
    .check_instructionIndex  (writeCheck_0_instructionIndex),
    .record_valid            (chainingRecordCopy_2_valid),
    .record_bits_vd_valid    (chainingRecordCopy_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_2_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_2_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_2_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_2_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_2_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_2_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_2_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_2_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_2_bits_elementMask),
    .checkResult             (_writeAllow_0_checkModule_2_checkResult)
  );
  WriteCheck writeAllow_0_checkModule_3 (
    .check_vd                (writeCheck_0_vd),
    .check_offset            (writeCheck_0_offset),
    .check_instructionIndex  (writeCheck_0_instructionIndex),
    .record_valid            (chainingRecordCopy_3_valid),
    .record_bits_vd_valid    (chainingRecordCopy_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_3_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_3_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_3_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_3_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_3_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_3_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_3_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_3_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_3_bits_elementMask),
    .checkResult             (_writeAllow_0_checkModule_3_checkResult)
  );
  WriteCheck writeAllow_0_checkModule_4 (
    .check_vd                (writeCheck_0_vd),
    .check_offset            (writeCheck_0_offset),
    .check_instructionIndex  (writeCheck_0_instructionIndex),
    .record_valid            (chainingRecordCopy_4_valid),
    .record_bits_vd_valid    (chainingRecordCopy_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_4_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_4_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_4_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_4_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_4_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_4_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_4_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_4_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_4_bits_elementMask),
    .checkResult             (_writeAllow_0_checkModule_4_checkResult)
  );
  WriteCheck writeAllow_1_checkModule (
    .check_vd                (writeCheck_1_vd),
    .check_offset            (writeCheck_1_offset),
    .check_instructionIndex  (writeCheck_1_instructionIndex),
    .record_valid            (chainingRecordCopy_0_valid),
    .record_bits_vd_valid    (chainingRecordCopy_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_0_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_0_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_0_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_0_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_0_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_0_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_0_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_0_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_0_bits_elementMask),
    .checkResult             (_writeAllow_1_checkModule_checkResult)
  );
  WriteCheck writeAllow_1_checkModule_1 (
    .check_vd                (writeCheck_1_vd),
    .check_offset            (writeCheck_1_offset),
    .check_instructionIndex  (writeCheck_1_instructionIndex),
    .record_valid            (chainingRecordCopy_1_valid),
    .record_bits_vd_valid    (chainingRecordCopy_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_1_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_1_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_1_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_1_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_1_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_1_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_1_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_1_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_1_bits_elementMask),
    .checkResult             (_writeAllow_1_checkModule_1_checkResult)
  );
  WriteCheck writeAllow_1_checkModule_2 (
    .check_vd                (writeCheck_1_vd),
    .check_offset            (writeCheck_1_offset),
    .check_instructionIndex  (writeCheck_1_instructionIndex),
    .record_valid            (chainingRecordCopy_2_valid),
    .record_bits_vd_valid    (chainingRecordCopy_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_2_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_2_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_2_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_2_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_2_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_2_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_2_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_2_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_2_bits_elementMask),
    .checkResult             (_writeAllow_1_checkModule_2_checkResult)
  );
  WriteCheck writeAllow_1_checkModule_3 (
    .check_vd                (writeCheck_1_vd),
    .check_offset            (writeCheck_1_offset),
    .check_instructionIndex  (writeCheck_1_instructionIndex),
    .record_valid            (chainingRecordCopy_3_valid),
    .record_bits_vd_valid    (chainingRecordCopy_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_3_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_3_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_3_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_3_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_3_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_3_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_3_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_3_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_3_bits_elementMask),
    .checkResult             (_writeAllow_1_checkModule_3_checkResult)
  );
  WriteCheck writeAllow_1_checkModule_4 (
    .check_vd                (writeCheck_1_vd),
    .check_offset            (writeCheck_1_offset),
    .check_instructionIndex  (writeCheck_1_instructionIndex),
    .record_valid            (chainingRecordCopy_4_valid),
    .record_bits_vd_valid    (chainingRecordCopy_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_4_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_4_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_4_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_4_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_4_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_4_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_4_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_4_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_4_bits_elementMask),
    .checkResult             (_writeAllow_1_checkModule_4_checkResult)
  );
  WriteCheck writeAllow_2_checkModule (
    .check_vd                (writeCheck_2_vd),
    .check_offset            (writeCheck_2_offset),
    .check_instructionIndex  (writeCheck_2_instructionIndex),
    .record_valid            (chainingRecordCopy_0_valid),
    .record_bits_vd_valid    (chainingRecordCopy_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_0_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_0_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_0_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_0_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_0_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_0_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_0_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_0_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_0_bits_elementMask),
    .checkResult             (_writeAllow_2_checkModule_checkResult)
  );
  WriteCheck writeAllow_2_checkModule_1 (
    .check_vd                (writeCheck_2_vd),
    .check_offset            (writeCheck_2_offset),
    .check_instructionIndex  (writeCheck_2_instructionIndex),
    .record_valid            (chainingRecordCopy_1_valid),
    .record_bits_vd_valid    (chainingRecordCopy_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_1_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_1_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_1_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_1_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_1_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_1_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_1_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_1_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_1_bits_elementMask),
    .checkResult             (_writeAllow_2_checkModule_1_checkResult)
  );
  WriteCheck writeAllow_2_checkModule_2 (
    .check_vd                (writeCheck_2_vd),
    .check_offset            (writeCheck_2_offset),
    .check_instructionIndex  (writeCheck_2_instructionIndex),
    .record_valid            (chainingRecordCopy_2_valid),
    .record_bits_vd_valid    (chainingRecordCopy_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_2_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_2_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_2_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_2_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_2_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_2_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_2_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_2_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_2_bits_elementMask),
    .checkResult             (_writeAllow_2_checkModule_2_checkResult)
  );
  WriteCheck writeAllow_2_checkModule_3 (
    .check_vd                (writeCheck_2_vd),
    .check_offset            (writeCheck_2_offset),
    .check_instructionIndex  (writeCheck_2_instructionIndex),
    .record_valid            (chainingRecordCopy_3_valid),
    .record_bits_vd_valid    (chainingRecordCopy_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_3_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_3_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_3_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_3_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_3_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_3_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_3_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_3_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_3_bits_elementMask),
    .checkResult             (_writeAllow_2_checkModule_3_checkResult)
  );
  WriteCheck writeAllow_2_checkModule_4 (
    .check_vd                (writeCheck_2_vd),
    .check_offset            (writeCheck_2_offset),
    .check_instructionIndex  (writeCheck_2_instructionIndex),
    .record_valid            (chainingRecordCopy_4_valid),
    .record_bits_vd_valid    (chainingRecordCopy_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_4_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_4_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_4_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_4_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_4_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_4_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_4_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_4_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_4_bits_elementMask),
    .checkResult             (_writeAllow_2_checkModule_4_checkResult)
  );
  WriteCheck writeAllow_3_checkModule (
    .check_vd                (writeCheck_3_vd),
    .check_offset            (writeCheck_3_offset),
    .check_instructionIndex  (writeCheck_3_instructionIndex),
    .record_valid            (chainingRecordCopy_0_valid),
    .record_bits_vd_valid    (chainingRecordCopy_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_0_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_0_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_0_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_0_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_0_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_0_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_0_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_0_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_0_bits_elementMask),
    .checkResult             (_writeAllow_3_checkModule_checkResult)
  );
  WriteCheck writeAllow_3_checkModule_1 (
    .check_vd                (writeCheck_3_vd),
    .check_offset            (writeCheck_3_offset),
    .check_instructionIndex  (writeCheck_3_instructionIndex),
    .record_valid            (chainingRecordCopy_1_valid),
    .record_bits_vd_valid    (chainingRecordCopy_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_1_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_1_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_1_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_1_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_1_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_1_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_1_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_1_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_1_bits_elementMask),
    .checkResult             (_writeAllow_3_checkModule_1_checkResult)
  );
  WriteCheck writeAllow_3_checkModule_2 (
    .check_vd                (writeCheck_3_vd),
    .check_offset            (writeCheck_3_offset),
    .check_instructionIndex  (writeCheck_3_instructionIndex),
    .record_valid            (chainingRecordCopy_2_valid),
    .record_bits_vd_valid    (chainingRecordCopy_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_2_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_2_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_2_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_2_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_2_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_2_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_2_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_2_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_2_bits_elementMask),
    .checkResult             (_writeAllow_3_checkModule_2_checkResult)
  );
  WriteCheck writeAllow_3_checkModule_3 (
    .check_vd                (writeCheck_3_vd),
    .check_offset            (writeCheck_3_offset),
    .check_instructionIndex  (writeCheck_3_instructionIndex),
    .record_valid            (chainingRecordCopy_3_valid),
    .record_bits_vd_valid    (chainingRecordCopy_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_3_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_3_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_3_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_3_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_3_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_3_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_3_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_3_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_3_bits_elementMask),
    .checkResult             (_writeAllow_3_checkModule_3_checkResult)
  );
  WriteCheck writeAllow_3_checkModule_4 (
    .check_vd                (writeCheck_3_vd),
    .check_offset            (writeCheck_3_offset),
    .check_instructionIndex  (writeCheck_3_instructionIndex),
    .record_valid            (chainingRecordCopy_4_valid),
    .record_bits_vd_valid    (chainingRecordCopy_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_4_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_4_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_4_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_4_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_4_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_4_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_4_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_4_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_4_bits_elementMask),
    .checkResult             (_writeAllow_3_checkModule_4_checkResult)
  );
  WriteCheck writeAllow_4_checkModule (
    .check_vd                (writeCheck_4_vd),
    .check_offset            (writeCheck_4_offset),
    .check_instructionIndex  (writeCheck_4_instructionIndex),
    .record_valid            (chainingRecordCopy_0_valid),
    .record_bits_vd_valid    (chainingRecordCopy_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_0_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_0_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_0_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_0_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_0_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_0_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_0_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_0_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_0_bits_elementMask),
    .checkResult             (_writeAllow_4_checkModule_checkResult)
  );
  WriteCheck writeAllow_4_checkModule_1 (
    .check_vd                (writeCheck_4_vd),
    .check_offset            (writeCheck_4_offset),
    .check_instructionIndex  (writeCheck_4_instructionIndex),
    .record_valid            (chainingRecordCopy_1_valid),
    .record_bits_vd_valid    (chainingRecordCopy_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_1_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_1_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_1_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_1_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_1_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_1_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_1_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_1_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_1_bits_elementMask),
    .checkResult             (_writeAllow_4_checkModule_1_checkResult)
  );
  WriteCheck writeAllow_4_checkModule_2 (
    .check_vd                (writeCheck_4_vd),
    .check_offset            (writeCheck_4_offset),
    .check_instructionIndex  (writeCheck_4_instructionIndex),
    .record_valid            (chainingRecordCopy_2_valid),
    .record_bits_vd_valid    (chainingRecordCopy_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_2_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_2_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_2_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_2_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_2_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_2_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_2_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_2_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_2_bits_elementMask),
    .checkResult             (_writeAllow_4_checkModule_2_checkResult)
  );
  WriteCheck writeAllow_4_checkModule_3 (
    .check_vd                (writeCheck_4_vd),
    .check_offset            (writeCheck_4_offset),
    .check_instructionIndex  (writeCheck_4_instructionIndex),
    .record_valid            (chainingRecordCopy_3_valid),
    .record_bits_vd_valid    (chainingRecordCopy_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_3_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_3_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_3_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_3_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_3_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_3_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_3_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_3_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_3_bits_elementMask),
    .checkResult             (_writeAllow_4_checkModule_3_checkResult)
  );
  WriteCheck writeAllow_4_checkModule_4 (
    .check_vd                (writeCheck_4_vd),
    .check_offset            (writeCheck_4_offset),
    .check_instructionIndex  (writeCheck_4_instructionIndex),
    .record_valid            (chainingRecordCopy_4_valid),
    .record_bits_vd_valid    (chainingRecordCopy_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_4_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_4_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_4_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_4_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_4_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_4_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_4_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_4_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_4_bits_elementMask),
    .checkResult             (_writeAllow_4_checkModule_4_checkResult)
  );
  WriteCheck writeAllow_5_checkModule (
    .check_vd                (writeCheck_5_vd),
    .check_offset            (writeCheck_5_offset),
    .check_instructionIndex  (writeCheck_5_instructionIndex),
    .record_valid            (chainingRecordCopy_0_valid),
    .record_bits_vd_valid    (chainingRecordCopy_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_0_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_0_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_0_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_0_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_0_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_0_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_0_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_0_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_0_bits_elementMask),
    .checkResult             (_writeAllow_5_checkModule_checkResult)
  );
  WriteCheck writeAllow_5_checkModule_1 (
    .check_vd                (writeCheck_5_vd),
    .check_offset            (writeCheck_5_offset),
    .check_instructionIndex  (writeCheck_5_instructionIndex),
    .record_valid            (chainingRecordCopy_1_valid),
    .record_bits_vd_valid    (chainingRecordCopy_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_1_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_1_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_1_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_1_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_1_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_1_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_1_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_1_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_1_bits_elementMask),
    .checkResult             (_writeAllow_5_checkModule_1_checkResult)
  );
  WriteCheck writeAllow_5_checkModule_2 (
    .check_vd                (writeCheck_5_vd),
    .check_offset            (writeCheck_5_offset),
    .check_instructionIndex  (writeCheck_5_instructionIndex),
    .record_valid            (chainingRecordCopy_2_valid),
    .record_bits_vd_valid    (chainingRecordCopy_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_2_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_2_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_2_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_2_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_2_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_2_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_2_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_2_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_2_bits_elementMask),
    .checkResult             (_writeAllow_5_checkModule_2_checkResult)
  );
  WriteCheck writeAllow_5_checkModule_3 (
    .check_vd                (writeCheck_5_vd),
    .check_offset            (writeCheck_5_offset),
    .check_instructionIndex  (writeCheck_5_instructionIndex),
    .record_valid            (chainingRecordCopy_3_valid),
    .record_bits_vd_valid    (chainingRecordCopy_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_3_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_3_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_3_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_3_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_3_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_3_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_3_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_3_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_3_bits_elementMask),
    .checkResult             (_writeAllow_5_checkModule_3_checkResult)
  );
  WriteCheck writeAllow_5_checkModule_4 (
    .check_vd                (writeCheck_5_vd),
    .check_offset            (writeCheck_5_offset),
    .check_instructionIndex  (writeCheck_5_instructionIndex),
    .record_valid            (chainingRecordCopy_4_valid),
    .record_bits_vd_valid    (chainingRecordCopy_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_4_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_4_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_4_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_4_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_4_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_4_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_4_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_4_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_4_bits_elementMask),
    .checkResult             (_writeAllow_5_checkModule_4_checkResult)
  );
  WriteCheck writeAllow_6_checkModule (
    .check_vd                (writeCheck_6_vd),
    .check_offset            (writeCheck_6_offset),
    .check_instructionIndex  (writeCheck_6_instructionIndex),
    .record_valid            (chainingRecordCopy_0_valid),
    .record_bits_vd_valid    (chainingRecordCopy_0_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_0_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_0_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_0_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_0_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_0_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_0_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_0_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_0_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_0_bits_elementMask),
    .checkResult             (_writeAllow_6_checkModule_checkResult)
  );
  WriteCheck writeAllow_6_checkModule_1 (
    .check_vd                (writeCheck_6_vd),
    .check_offset            (writeCheck_6_offset),
    .check_instructionIndex  (writeCheck_6_instructionIndex),
    .record_valid            (chainingRecordCopy_1_valid),
    .record_bits_vd_valid    (chainingRecordCopy_1_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_1_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_1_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_1_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_1_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_1_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_1_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_1_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_1_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_1_bits_elementMask),
    .checkResult             (_writeAllow_6_checkModule_1_checkResult)
  );
  WriteCheck writeAllow_6_checkModule_2 (
    .check_vd                (writeCheck_6_vd),
    .check_offset            (writeCheck_6_offset),
    .check_instructionIndex  (writeCheck_6_instructionIndex),
    .record_valid            (chainingRecordCopy_2_valid),
    .record_bits_vd_valid    (chainingRecordCopy_2_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_2_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_2_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_2_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_2_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_2_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_2_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_2_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_2_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_2_bits_elementMask),
    .checkResult             (_writeAllow_6_checkModule_2_checkResult)
  );
  WriteCheck writeAllow_6_checkModule_3 (
    .check_vd                (writeCheck_6_vd),
    .check_offset            (writeCheck_6_offset),
    .check_instructionIndex  (writeCheck_6_instructionIndex),
    .record_valid            (chainingRecordCopy_3_valid),
    .record_bits_vd_valid    (chainingRecordCopy_3_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_3_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_3_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_3_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_3_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_3_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_3_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_3_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_3_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_3_bits_elementMask),
    .checkResult             (_writeAllow_6_checkModule_3_checkResult)
  );
  WriteCheck writeAllow_6_checkModule_4 (
    .check_vd                (writeCheck_6_vd),
    .check_offset            (writeCheck_6_offset),
    .check_instructionIndex  (writeCheck_6_instructionIndex),
    .record_valid            (chainingRecordCopy_4_valid),
    .record_bits_vd_valid    (chainingRecordCopy_4_bits_vd_valid),
    .record_bits_vd_bits     (chainingRecordCopy_4_bits_vd_bits),
    .record_bits_vs1_valid   (chainingRecordCopy_4_bits_vs1_valid),
    .record_bits_vs1_bits    (chainingRecordCopy_4_bits_vs1_bits),
    .record_bits_vs2         (chainingRecordCopy_4_bits_vs2),
    .record_bits_instIndex   (chainingRecordCopy_4_bits_instIndex),
    .record_bits_gather      (chainingRecordCopy_4_bits_gather),
    .record_bits_gather16    (chainingRecordCopy_4_bits_gather16),
    .record_bits_onlyRead    (chainingRecordCopy_4_bits_onlyRead),
    .record_bits_elementMask (chainingRecordCopy_4_bits_elementMask),
    .checkResult             (_writeAllow_6_checkModule_4_checkResult)
  );
  assign readRequests_0_ready = readRequests_0_ready_0;
  assign readRequests_1_ready = readRequests_1_ready_0;
  assign readRequests_2_ready = readRequests_2_ready_0;
  assign readRequests_3_ready = readRequests_3_ready_0;
  assign readRequests_4_ready = readRequests_4_ready_0;
  assign readRequests_5_ready = readRequests_5_ready_0;
  assign readRequests_6_ready = readRequests_6_ready_0;
  assign readRequests_7_ready = readRequests_7_ready_0;
  assign readRequests_8_ready = readRequests_8_ready_0;
  assign readRequests_9_ready = readRequests_9_ready_0;
  assign readRequests_10_ready = readRequests_10_ready_0;
  assign readRequests_11_ready = readRequests_11_ready_0;
  assign readRequests_12_ready = readRequests_12_ready_0;
  assign readRequests_13_ready = readRequests_13_ready_0;
  assign readCheckResult_0 =
    _readCheckResult_0_checkModule_checkResult & _readCheckResult_0_checkModule_1_checkResult & _readCheckResult_0_checkModule_2_checkResult & _readCheckResult_0_checkModule_3_checkResult & _readCheckResult_0_checkModule_4_checkResult;
  assign readCheckResult_1 =
    _readCheckResult_1_checkModule_checkResult & _readCheckResult_1_checkModule_1_checkResult & _readCheckResult_1_checkModule_2_checkResult & _readCheckResult_1_checkModule_3_checkResult & _readCheckResult_1_checkModule_4_checkResult;
  assign readCheckResult_2 =
    _readCheckResult_2_checkModule_checkResult & _readCheckResult_2_checkModule_1_checkResult & _readCheckResult_2_checkModule_2_checkResult & _readCheckResult_2_checkModule_3_checkResult & _readCheckResult_2_checkModule_4_checkResult;
  assign readCheckResult_3 =
    _readCheckResult_3_checkModule_checkResult & _readCheckResult_3_checkModule_1_checkResult & _readCheckResult_3_checkModule_2_checkResult & _readCheckResult_3_checkModule_3_checkResult & _readCheckResult_3_checkModule_4_checkResult;
  assign readCheckResult_4 =
    _readCheckResult_4_checkModule_checkResult & _readCheckResult_4_checkModule_1_checkResult & _readCheckResult_4_checkModule_2_checkResult & _readCheckResult_4_checkModule_3_checkResult & _readCheckResult_4_checkModule_4_checkResult;
  assign readCheckResult_5 =
    _readCheckResult_5_checkModule_checkResult & _readCheckResult_5_checkModule_1_checkResult & _readCheckResult_5_checkModule_2_checkResult & _readCheckResult_5_checkModule_3_checkResult & _readCheckResult_5_checkModule_4_checkResult;
  assign readCheckResult_6 =
    _readCheckResult_6_checkModule_checkResult & _readCheckResult_6_checkModule_1_checkResult & _readCheckResult_6_checkModule_2_checkResult & _readCheckResult_6_checkModule_3_checkResult & _readCheckResult_6_checkModule_4_checkResult;
  assign readCheckResult_7 =
    _readCheckResult_7_checkModule_checkResult & _readCheckResult_7_checkModule_1_checkResult & _readCheckResult_7_checkModule_2_checkResult & _readCheckResult_7_checkModule_3_checkResult & _readCheckResult_7_checkModule_4_checkResult;
  assign readCheckResult_8 =
    _readCheckResult_8_checkModule_checkResult & _readCheckResult_8_checkModule_1_checkResult & _readCheckResult_8_checkModule_2_checkResult & _readCheckResult_8_checkModule_3_checkResult & _readCheckResult_8_checkModule_4_checkResult;
  assign readCheckResult_9 =
    _readCheckResult_9_checkModule_checkResult & _readCheckResult_9_checkModule_1_checkResult & _readCheckResult_9_checkModule_2_checkResult & _readCheckResult_9_checkModule_3_checkResult & _readCheckResult_9_checkModule_4_checkResult;
  assign readCheckResult_10 =
    _readCheckResult_10_checkModule_checkResult & _readCheckResult_10_checkModule_1_checkResult & _readCheckResult_10_checkModule_2_checkResult & _readCheckResult_10_checkModule_3_checkResult & _readCheckResult_10_checkModule_4_checkResult;
  assign readCheckResult_11 =
    _readCheckResult_11_checkModule_checkResult & _readCheckResult_11_checkModule_1_checkResult & _readCheckResult_11_checkModule_2_checkResult & _readCheckResult_11_checkModule_3_checkResult & _readCheckResult_11_checkModule_4_checkResult;
  assign readCheckResult_12 =
    _readCheckResult_12_checkModule_checkResult & _readCheckResult_12_checkModule_1_checkResult & _readCheckResult_12_checkModule_2_checkResult & _readCheckResult_12_checkModule_3_checkResult & _readCheckResult_12_checkModule_4_checkResult;
  assign readCheckResult_13 =
    _readCheckResult_13_checkModule_checkResult & _readCheckResult_13_checkModule_1_checkResult & _readCheckResult_13_checkModule_2_checkResult & _readCheckResult_13_checkModule_3_checkResult & _readCheckResult_13_checkModule_4_checkResult;
  assign readResults_0 = ~pipeFirstUsed_pipe_pipe_out_bits & pipeFire_pipe_pipe_out_bits ? readResultF_0 : 32'h0;
  assign readResults_1 = ~pipeFirstUsed_pipe_pipe_out_1_bits & pipeFire_pipe_pipe_out_1_bits ? readResultF_0 : 32'h0;
  assign readResults_2 = ~pipeFirstUsed_pipe_pipe_out_2_bits & pipeFire_pipe_pipe_out_2_bits ? readResultF_0 : 32'h0;
  assign readResults_3 = ~pipeFirstUsed_pipe_pipe_out_3_bits & pipeFire_pipe_pipe_out_3_bits ? readResultF_0 : 32'h0;
  assign readResults_4 = ~pipeFirstUsed_pipe_pipe_out_4_bits & pipeFire_pipe_pipe_out_4_bits ? readResultF_0 : 32'h0;
  assign readResults_5 = ~pipeFirstUsed_pipe_pipe_out_5_bits & pipeFire_pipe_pipe_out_5_bits ? readResultF_0 : 32'h0;
  assign readResults_6 = ~pipeFirstUsed_pipe_pipe_out_6_bits & pipeFire_pipe_pipe_out_6_bits ? readResultF_0 : 32'h0;
  assign readResults_7 = ~pipeFirstUsed_pipe_pipe_out_7_bits & pipeFire_pipe_pipe_out_7_bits ? readResultF_0 : 32'h0;
  assign readResults_8 = ~pipeFirstUsed_pipe_pipe_out_8_bits & pipeFire_pipe_pipe_out_8_bits ? readResultF_0 : 32'h0;
  assign readResults_9 = ~pipeFirstUsed_pipe_pipe_out_9_bits & pipeFire_pipe_pipe_out_9_bits ? readResultF_0 : 32'h0;
  assign readResults_10 = ~pipeFirstUsed_pipe_pipe_out_10_bits & pipeFire_pipe_pipe_out_10_bits ? readResultF_0 : 32'h0;
  assign readResults_11 = ~pipeFirstUsed_pipe_pipe_out_11_bits & pipeFire_pipe_pipe_out_11_bits ? readResultF_0 : 32'h0;
  assign readResults_12 = ~pipeFirstUsed_pipe_pipe_out_12_bits & pipeFire_pipe_pipe_out_12_bits ? readResultF_0 : 32'h0;
  assign readResults_13 = ~pipeFirstUsed_pipe_pipe_out_13_bits & pipeFire_pipe_pipe_out_13_bits ? readResultF_0 : 32'h0;
  assign write_ready = write_ready_0;
  assign writeAllow_0 = _writeAllow_0_checkModule_checkResult & _writeAllow_0_checkModule_1_checkResult & _writeAllow_0_checkModule_2_checkResult & _writeAllow_0_checkModule_3_checkResult & _writeAllow_0_checkModule_4_checkResult;
  assign writeAllow_1 = _writeAllow_1_checkModule_checkResult & _writeAllow_1_checkModule_1_checkResult & _writeAllow_1_checkModule_2_checkResult & _writeAllow_1_checkModule_3_checkResult & _writeAllow_1_checkModule_4_checkResult;
  assign writeAllow_2 = _writeAllow_2_checkModule_checkResult & _writeAllow_2_checkModule_1_checkResult & _writeAllow_2_checkModule_2_checkResult & _writeAllow_2_checkModule_3_checkResult & _writeAllow_2_checkModule_4_checkResult;
  assign writeAllow_3 = _writeAllow_3_checkModule_checkResult & _writeAllow_3_checkModule_1_checkResult & _writeAllow_3_checkModule_2_checkResult & _writeAllow_3_checkModule_3_checkResult & _writeAllow_3_checkModule_4_checkResult;
  assign writeAllow_4 = _writeAllow_4_checkModule_checkResult & _writeAllow_4_checkModule_1_checkResult & _writeAllow_4_checkModule_2_checkResult & _writeAllow_4_checkModule_3_checkResult & _writeAllow_4_checkModule_4_checkResult;
  assign writeAllow_5 = _writeAllow_5_checkModule_checkResult & _writeAllow_5_checkModule_1_checkResult & _writeAllow_5_checkModule_2_checkResult & _writeAllow_5_checkModule_3_checkResult & _writeAllow_5_checkModule_4_checkResult;
  assign writeAllow_6 = _writeAllow_6_checkModule_checkResult & _writeAllow_6_checkModule_1_checkResult & _writeAllow_6_checkModule_2_checkResult & _writeAllow_6_checkModule_3_checkResult & _writeAllow_6_checkModule_4_checkResult;
  assign vrfSlotRelease = recordRelease_0 | recordRelease_1 | recordRelease_2 | recordRelease_3 | recordRelease_4;
endmodule

