
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
  input         clock,
                reset,
  output        readRequests_0_ready,
  input         readRequests_0_valid,
  input  [4:0]  readRequests_0_bits_vs,
  input  [1:0]  readRequests_0_bits_readSource,
  input         readRequests_0_bits_offset,
  input  [2:0]  readRequests_0_bits_instructionIndex,
  output        readRequests_1_ready,
  input         readRequests_1_valid,
  input  [4:0]  readRequests_1_bits_vs,
  input  [1:0]  readRequests_1_bits_readSource,
  input         readRequests_1_bits_offset,
  input  [2:0]  readRequests_1_bits_instructionIndex,
  output        readRequests_2_ready,
  input         readRequests_2_valid,
  input  [4:0]  readRequests_2_bits_vs,
  input  [1:0]  readRequests_2_bits_readSource,
  input         readRequests_2_bits_offset,
  input  [2:0]  readRequests_2_bits_instructionIndex,
  output        readRequests_3_ready,
  input         readRequests_3_valid,
  input  [4:0]  readRequests_3_bits_vs,
  input  [1:0]  readRequests_3_bits_readSource,
  input         readRequests_3_bits_offset,
  input  [2:0]  readRequests_3_bits_instructionIndex,
  output        readRequests_4_ready,
  input         readRequests_4_valid,
  input  [4:0]  readRequests_4_bits_vs,
  input  [1:0]  readRequests_4_bits_readSource,
  input         readRequests_4_bits_offset,
  input  [2:0]  readRequests_4_bits_instructionIndex,
  output        readRequests_5_ready,
  input         readRequests_5_valid,
  input  [4:0]  readRequests_5_bits_vs,
  input  [1:0]  readRequests_5_bits_readSource,
  input         readRequests_5_bits_offset,
  input  [2:0]  readRequests_5_bits_instructionIndex,
  output        readRequests_6_ready,
  input         readRequests_6_valid,
  input  [4:0]  readRequests_6_bits_vs,
  input  [1:0]  readRequests_6_bits_readSource,
  input         readRequests_6_bits_offset,
  input  [2:0]  readRequests_6_bits_instructionIndex,
  output        readRequests_7_ready,
  input         readRequests_7_valid,
  input  [4:0]  readRequests_7_bits_vs,
  input  [1:0]  readRequests_7_bits_readSource,
  input         readRequests_7_bits_offset,
  input  [2:0]  readRequests_7_bits_instructionIndex,
  output        readRequests_8_ready,
  input         readRequests_8_valid,
  input  [4:0]  readRequests_8_bits_vs,
  input  [1:0]  readRequests_8_bits_readSource,
  input         readRequests_8_bits_offset,
  input  [2:0]  readRequests_8_bits_instructionIndex,
  output        readRequests_9_ready,
  input         readRequests_9_valid,
  input  [4:0]  readRequests_9_bits_vs,
  input  [1:0]  readRequests_9_bits_readSource,
  input         readRequests_9_bits_offset,
  input  [2:0]  readRequests_9_bits_instructionIndex,
  output        readRequests_10_ready,
  input         readRequests_10_valid,
  input  [4:0]  readRequests_10_bits_vs,
  input  [1:0]  readRequests_10_bits_readSource,
  input         readRequests_10_bits_offset,
  input  [2:0]  readRequests_10_bits_instructionIndex,
  output        readRequests_11_ready,
  input         readRequests_11_valid,
  input  [4:0]  readRequests_11_bits_vs,
  input  [1:0]  readRequests_11_bits_readSource,
  input         readRequests_11_bits_offset,
  input  [2:0]  readRequests_11_bits_instructionIndex,
  output        readRequests_12_ready,
  input         readRequests_12_valid,
  input  [4:0]  readRequests_12_bits_vs,
  input  [1:0]  readRequests_12_bits_readSource,
  input         readRequests_12_bits_offset,
  input  [2:0]  readRequests_12_bits_instructionIndex,
  output        readRequests_13_ready,
  input         readRequests_13_valid,
  input  [4:0]  readRequests_13_bits_vs,
  input  [1:0]  readRequests_13_bits_readSource,
  input         readRequests_13_bits_offset,
  input  [2:0]  readRequests_13_bits_instructionIndex,
  input  [4:0]  readCheck_0_vs,
  input         readCheck_0_offset,
  input  [2:0]  readCheck_0_instructionIndex,
  input  [4:0]  readCheck_1_vs,
  input         readCheck_1_offset,
  input  [2:0]  readCheck_1_instructionIndex,
  input  [4:0]  readCheck_2_vs,
  input         readCheck_2_offset,
  input  [2:0]  readCheck_2_instructionIndex,
  input  [4:0]  readCheck_3_vs,
  input         readCheck_3_offset,
  input  [2:0]  readCheck_3_instructionIndex,
  input  [4:0]  readCheck_4_vs,
  input         readCheck_4_offset,
  input  [2:0]  readCheck_4_instructionIndex,
  input  [4:0]  readCheck_5_vs,
  input         readCheck_5_offset,
  input  [2:0]  readCheck_5_instructionIndex,
  input  [4:0]  readCheck_6_vs,
  input         readCheck_6_offset,
  input  [2:0]  readCheck_6_instructionIndex,
  input  [4:0]  readCheck_7_vs,
  input         readCheck_7_offset,
  input  [2:0]  readCheck_7_instructionIndex,
  input  [4:0]  readCheck_8_vs,
  input         readCheck_8_offset,
  input  [2:0]  readCheck_8_instructionIndex,
  input  [4:0]  readCheck_9_vs,
  input         readCheck_9_offset,
  input  [2:0]  readCheck_9_instructionIndex,
  input  [4:0]  readCheck_10_vs,
  input         readCheck_10_offset,
  input  [2:0]  readCheck_10_instructionIndex,
  input  [4:0]  readCheck_11_vs,
  input         readCheck_11_offset,
  input  [2:0]  readCheck_11_instructionIndex,
  input  [4:0]  readCheck_12_vs,
  input         readCheck_12_offset,
  input  [2:0]  readCheck_12_instructionIndex,
  input  [4:0]  readCheck_13_vs,
  input         readCheck_13_offset,
  input  [2:0]  readCheck_13_instructionIndex,
  output        readCheckResult_0,
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
  output [31:0] readResults_0,
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
  output        write_ready,
  input         write_valid,
  input  [4:0]  write_bits_vd,
  input         write_bits_offset,
  input  [3:0]  write_bits_mask,
  input  [31:0] write_bits_data,
  input         write_bits_last,
  input  [2:0]  write_bits_instructionIndex,
  input  [4:0]  writeCheck_0_vd,
  input         writeCheck_0_offset,
  input  [2:0]  writeCheck_0_instructionIndex,
  input  [4:0]  writeCheck_1_vd,
  input         writeCheck_1_offset,
  input  [2:0]  writeCheck_1_instructionIndex,
  input  [4:0]  writeCheck_2_vd,
  input         writeCheck_2_offset,
  input  [2:0]  writeCheck_2_instructionIndex,
  input  [4:0]  writeCheck_3_vd,
  input         writeCheck_3_offset,
  input  [2:0]  writeCheck_3_instructionIndex,
  input  [4:0]  writeCheck_4_vd,
  input         writeCheck_4_offset,
  input  [2:0]  writeCheck_4_instructionIndex,
  input  [4:0]  writeCheck_5_vd,
  input         writeCheck_5_offset,
  input  [2:0]  writeCheck_5_instructionIndex,
  input  [4:0]  writeCheck_6_vd,
  input         writeCheck_6_offset,
  input  [2:0]  writeCheck_6_instructionIndex,
  output        writeAllow_0,
                writeAllow_1,
                writeAllow_2,
                writeAllow_3,
                writeAllow_4,
                writeAllow_5,
                writeAllow_6,
  input         instructionWriteReport_valid,
                instructionWriteReport_bits_vd_valid,
  input  [4:0]  instructionWriteReport_bits_vd_bits,
  input         instructionWriteReport_bits_vs1_valid,
  input  [4:0]  instructionWriteReport_bits_vs1_bits,
                instructionWriteReport_bits_vs2,
  input  [2:0]  instructionWriteReport_bits_instIndex,
  input         instructionWriteReport_bits_ls,
                instructionWriteReport_bits_st,
                instructionWriteReport_bits_gather,
                instructionWriteReport_bits_gather16,
                instructionWriteReport_bits_crossWrite,
                instructionWriteReport_bits_crossRead,
                instructionWriteReport_bits_indexType,
                instructionWriteReport_bits_ma,
                instructionWriteReport_bits_onlyRead,
                instructionWriteReport_bits_slow,
  input  [15:0] instructionWriteReport_bits_elementMask,
  input         instructionWriteReport_bits_state_stFinish,
                instructionWriteReport_bits_state_wWriteQueueClear,
                instructionWriteReport_bits_state_wLaneLastReport,
                instructionWriteReport_bits_state_wTopLastReport,
  input  [7:0]  instructionLastReport,
                lsuLastReport,
  output [7:0]  vrfSlotRelease,
  input  [7:0]  dataInLane,
                loadDataInLSUWriteQueue
);

  wire        _writeAllow_6_checkModule_4_checkResult;
  wire        _writeAllow_6_checkModule_3_checkResult;
  wire        _writeAllow_6_checkModule_2_checkResult;
  wire        _writeAllow_6_checkModule_1_checkResult;
  wire        _writeAllow_6_checkModule_checkResult;
  wire        _writeAllow_5_checkModule_4_checkResult;
  wire        _writeAllow_5_checkModule_3_checkResult;
  wire        _writeAllow_5_checkModule_2_checkResult;
  wire        _writeAllow_5_checkModule_1_checkResult;
  wire        _writeAllow_5_checkModule_checkResult;
  wire        _writeAllow_4_checkModule_4_checkResult;
  wire        _writeAllow_4_checkModule_3_checkResult;
  wire        _writeAllow_4_checkModule_2_checkResult;
  wire        _writeAllow_4_checkModule_1_checkResult;
  wire        _writeAllow_4_checkModule_checkResult;
  wire        _writeAllow_3_checkModule_4_checkResult;
  wire        _writeAllow_3_checkModule_3_checkResult;
  wire        _writeAllow_3_checkModule_2_checkResult;
  wire        _writeAllow_3_checkModule_1_checkResult;
  wire        _writeAllow_3_checkModule_checkResult;
  wire        _writeAllow_2_checkModule_4_checkResult;
  wire        _writeAllow_2_checkModule_3_checkResult;
  wire        _writeAllow_2_checkModule_2_checkResult;
  wire        _writeAllow_2_checkModule_1_checkResult;
  wire        _writeAllow_2_checkModule_checkResult;
  wire        _writeAllow_1_checkModule_4_checkResult;
  wire        _writeAllow_1_checkModule_3_checkResult;
  wire        _writeAllow_1_checkModule_2_checkResult;
  wire        _writeAllow_1_checkModule_1_checkResult;
  wire        _writeAllow_1_checkModule_checkResult;
  wire        _writeAllow_0_checkModule_4_checkResult;
  wire        _writeAllow_0_checkModule_3_checkResult;
  wire        _writeAllow_0_checkModule_2_checkResult;
  wire        _writeAllow_0_checkModule_1_checkResult;
  wire        _writeAllow_0_checkModule_checkResult;
  wire        _checkResult_ChainingCheck_readPort13_record4_checkResult;
  wire        _checkResult_ChainingCheck_readPort13_record3_checkResult;
  wire        _checkResult_ChainingCheck_readPort13_record2_checkResult;
  wire        _checkResult_ChainingCheck_readPort13_record1_checkResult;
  wire        _checkResult_ChainingCheck_readPort13_record0_checkResult;
  wire        _readCheckResult_13_checkModule_4_checkResult;
  wire        _readCheckResult_13_checkModule_3_checkResult;
  wire        _readCheckResult_13_checkModule_2_checkResult;
  wire        _readCheckResult_13_checkModule_1_checkResult;
  wire        _readCheckResult_13_checkModule_checkResult;
  wire        _readCheckResult_12_checkModule_4_checkResult;
  wire        _readCheckResult_12_checkModule_3_checkResult;
  wire        _readCheckResult_12_checkModule_2_checkResult;
  wire        _readCheckResult_12_checkModule_1_checkResult;
  wire        _readCheckResult_12_checkModule_checkResult;
  wire        _readCheckResult_11_checkModule_4_checkResult;
  wire        _readCheckResult_11_checkModule_3_checkResult;
  wire        _readCheckResult_11_checkModule_2_checkResult;
  wire        _readCheckResult_11_checkModule_1_checkResult;
  wire        _readCheckResult_11_checkModule_checkResult;
  wire        _readCheckResult_10_checkModule_4_checkResult;
  wire        _readCheckResult_10_checkModule_3_checkResult;
  wire        _readCheckResult_10_checkModule_2_checkResult;
  wire        _readCheckResult_10_checkModule_1_checkResult;
  wire        _readCheckResult_10_checkModule_checkResult;
  wire        _readCheckResult_9_checkModule_4_checkResult;
  wire        _readCheckResult_9_checkModule_3_checkResult;
  wire        _readCheckResult_9_checkModule_2_checkResult;
  wire        _readCheckResult_9_checkModule_1_checkResult;
  wire        _readCheckResult_9_checkModule_checkResult;
  wire        _readCheckResult_8_checkModule_4_checkResult;
  wire        _readCheckResult_8_checkModule_3_checkResult;
  wire        _readCheckResult_8_checkModule_2_checkResult;
  wire        _readCheckResult_8_checkModule_1_checkResult;
  wire        _readCheckResult_8_checkModule_checkResult;
  wire        _readCheckResult_7_checkModule_4_checkResult;
  wire        _readCheckResult_7_checkModule_3_checkResult;
  wire        _readCheckResult_7_checkModule_2_checkResult;
  wire        _readCheckResult_7_checkModule_1_checkResult;
  wire        _readCheckResult_7_checkModule_checkResult;
  wire        _readCheckResult_6_checkModule_4_checkResult;
  wire        _readCheckResult_6_checkModule_3_checkResult;
  wire        _readCheckResult_6_checkModule_2_checkResult;
  wire        _readCheckResult_6_checkModule_1_checkResult;
  wire        _readCheckResult_6_checkModule_checkResult;
  wire        _readCheckResult_5_checkModule_4_checkResult;
  wire        _readCheckResult_5_checkModule_3_checkResult;
  wire        _readCheckResult_5_checkModule_2_checkResult;
  wire        _readCheckResult_5_checkModule_1_checkResult;
  wire        _readCheckResult_5_checkModule_checkResult;
  wire        _readCheckResult_4_checkModule_4_checkResult;
  wire        _readCheckResult_4_checkModule_3_checkResult;
  wire        _readCheckResult_4_checkModule_2_checkResult;
  wire        _readCheckResult_4_checkModule_1_checkResult;
  wire        _readCheckResult_4_checkModule_checkResult;
  wire        _readCheckResult_3_checkModule_4_checkResult;
  wire        _readCheckResult_3_checkModule_3_checkResult;
  wire        _readCheckResult_3_checkModule_2_checkResult;
  wire        _readCheckResult_3_checkModule_1_checkResult;
  wire        _readCheckResult_3_checkModule_checkResult;
  wire        _readCheckResult_2_checkModule_4_checkResult;
  wire        _readCheckResult_2_checkModule_3_checkResult;
  wire        _readCheckResult_2_checkModule_2_checkResult;
  wire        _readCheckResult_2_checkModule_1_checkResult;
  wire        _readCheckResult_2_checkModule_checkResult;
  wire        _readCheckResult_1_checkModule_4_checkResult;
  wire        _readCheckResult_1_checkModule_3_checkResult;
  wire        _readCheckResult_1_checkModule_2_checkResult;
  wire        _readCheckResult_1_checkModule_1_checkResult;
  wire        _readCheckResult_1_checkModule_checkResult;
  wire        _readCheckResult_0_checkModule_4_checkResult;
  wire        _readCheckResult_0_checkModule_3_checkResult;
  wire        _readCheckResult_0_checkModule_2_checkResult;
  wire        _readCheckResult_0_checkModule_1_checkResult;
  wire        _readCheckResult_0_checkModule_checkResult;
  wire        readRequests_0_valid_0 = readRequests_0_valid;
  wire [4:0]  readRequests_0_bits_vs_0 = readRequests_0_bits_vs;
  wire [1:0]  readRequests_0_bits_readSource_0 = readRequests_0_bits_readSource;
  wire        readRequests_0_bits_offset_0 = readRequests_0_bits_offset;
  wire [2:0]  readRequests_0_bits_instructionIndex_0 = readRequests_0_bits_instructionIndex;
  wire        readRequests_1_valid_0 = readRequests_1_valid;
  wire [4:0]  readRequests_1_bits_vs_0 = readRequests_1_bits_vs;
  wire [1:0]  readRequests_1_bits_readSource_0 = readRequests_1_bits_readSource;
  wire        readRequests_1_bits_offset_0 = readRequests_1_bits_offset;
  wire [2:0]  readRequests_1_bits_instructionIndex_0 = readRequests_1_bits_instructionIndex;
  wire        readRequests_2_valid_0 = readRequests_2_valid;
  wire [4:0]  readRequests_2_bits_vs_0 = readRequests_2_bits_vs;
  wire [1:0]  readRequests_2_bits_readSource_0 = readRequests_2_bits_readSource;
  wire        readRequests_2_bits_offset_0 = readRequests_2_bits_offset;
  wire [2:0]  readRequests_2_bits_instructionIndex_0 = readRequests_2_bits_instructionIndex;
  wire        readRequests_3_valid_0 = readRequests_3_valid;
  wire [4:0]  readRequests_3_bits_vs_0 = readRequests_3_bits_vs;
  wire [1:0]  readRequests_3_bits_readSource_0 = readRequests_3_bits_readSource;
  wire        readRequests_3_bits_offset_0 = readRequests_3_bits_offset;
  wire [2:0]  readRequests_3_bits_instructionIndex_0 = readRequests_3_bits_instructionIndex;
  wire        readRequests_4_valid_0 = readRequests_4_valid;
  wire [4:0]  readRequests_4_bits_vs_0 = readRequests_4_bits_vs;
  wire [1:0]  readRequests_4_bits_readSource_0 = readRequests_4_bits_readSource;
  wire        readRequests_4_bits_offset_0 = readRequests_4_bits_offset;
  wire [2:0]  readRequests_4_bits_instructionIndex_0 = readRequests_4_bits_instructionIndex;
  wire        readRequests_5_valid_0 = readRequests_5_valid;
  wire [4:0]  readRequests_5_bits_vs_0 = readRequests_5_bits_vs;
  wire [1:0]  readRequests_5_bits_readSource_0 = readRequests_5_bits_readSource;
  wire        readRequests_5_bits_offset_0 = readRequests_5_bits_offset;
  wire [2:0]  readRequests_5_bits_instructionIndex_0 = readRequests_5_bits_instructionIndex;
  wire        readRequests_6_valid_0 = readRequests_6_valid;
  wire [4:0]  readRequests_6_bits_vs_0 = readRequests_6_bits_vs;
  wire [1:0]  readRequests_6_bits_readSource_0 = readRequests_6_bits_readSource;
  wire        readRequests_6_bits_offset_0 = readRequests_6_bits_offset;
  wire [2:0]  readRequests_6_bits_instructionIndex_0 = readRequests_6_bits_instructionIndex;
  wire        readRequests_7_valid_0 = readRequests_7_valid;
  wire [4:0]  readRequests_7_bits_vs_0 = readRequests_7_bits_vs;
  wire [1:0]  readRequests_7_bits_readSource_0 = readRequests_7_bits_readSource;
  wire        readRequests_7_bits_offset_0 = readRequests_7_bits_offset;
  wire [2:0]  readRequests_7_bits_instructionIndex_0 = readRequests_7_bits_instructionIndex;
  wire        readRequests_8_valid_0 = readRequests_8_valid;
  wire [4:0]  readRequests_8_bits_vs_0 = readRequests_8_bits_vs;
  wire [1:0]  readRequests_8_bits_readSource_0 = readRequests_8_bits_readSource;
  wire        readRequests_8_bits_offset_0 = readRequests_8_bits_offset;
  wire [2:0]  readRequests_8_bits_instructionIndex_0 = readRequests_8_bits_instructionIndex;
  wire        readRequests_9_valid_0 = readRequests_9_valid;
  wire [4:0]  readRequests_9_bits_vs_0 = readRequests_9_bits_vs;
  wire [1:0]  readRequests_9_bits_readSource_0 = readRequests_9_bits_readSource;
  wire        readRequests_9_bits_offset_0 = readRequests_9_bits_offset;
  wire [2:0]  readRequests_9_bits_instructionIndex_0 = readRequests_9_bits_instructionIndex;
  wire        readRequests_10_valid_0 = readRequests_10_valid;
  wire [4:0]  readRequests_10_bits_vs_0 = readRequests_10_bits_vs;
  wire [1:0]  readRequests_10_bits_readSource_0 = readRequests_10_bits_readSource;
  wire        readRequests_10_bits_offset_0 = readRequests_10_bits_offset;
  wire [2:0]  readRequests_10_bits_instructionIndex_0 = readRequests_10_bits_instructionIndex;
  wire        readRequests_11_valid_0 = readRequests_11_valid;
  wire [4:0]  readRequests_11_bits_vs_0 = readRequests_11_bits_vs;
  wire [1:0]  readRequests_11_bits_readSource_0 = readRequests_11_bits_readSource;
  wire        readRequests_11_bits_offset_0 = readRequests_11_bits_offset;
  wire [2:0]  readRequests_11_bits_instructionIndex_0 = readRequests_11_bits_instructionIndex;
  wire        readRequests_12_valid_0 = readRequests_12_valid;
  wire [4:0]  readRequests_12_bits_vs_0 = readRequests_12_bits_vs;
  wire [1:0]  readRequests_12_bits_readSource_0 = readRequests_12_bits_readSource;
  wire        readRequests_12_bits_offset_0 = readRequests_12_bits_offset;
  wire [2:0]  readRequests_12_bits_instructionIndex_0 = readRequests_12_bits_instructionIndex;
  wire        readRequests_13_valid_0 = readRequests_13_valid;
  wire [4:0]  readRequests_13_bits_vs_0 = readRequests_13_bits_vs;
  wire [1:0]  readRequests_13_bits_readSource_0 = readRequests_13_bits_readSource;
  wire        readRequests_13_bits_offset_0 = readRequests_13_bits_offset;
  wire [2:0]  readRequests_13_bits_instructionIndex_0 = readRequests_13_bits_instructionIndex;
  wire        write_valid_0 = write_valid;
  wire [4:0]  write_bits_vd_0 = write_bits_vd;
  wire        write_bits_offset_0 = write_bits_offset;
  wire [3:0]  write_bits_mask_0 = write_bits_mask;
  wire [31:0] write_bits_data_0 = write_bits_data;
  wire        write_bits_last_0 = write_bits_last;
  wire [2:0]  write_bits_instructionIndex_0 = write_bits_instructionIndex;
  wire        initRecord_bits_vd_valid = instructionWriteReport_bits_vd_valid;
  wire [4:0]  initRecord_bits_vd_bits = instructionWriteReport_bits_vd_bits;
  wire        initRecord_bits_vs1_valid = instructionWriteReport_bits_vs1_valid;
  wire [4:0]  initRecord_bits_vs1_bits = instructionWriteReport_bits_vs1_bits;
  wire [4:0]  initRecord_bits_vs2 = instructionWriteReport_bits_vs2;
  wire [2:0]  initRecord_bits_instIndex = instructionWriteReport_bits_instIndex;
  wire        initRecord_bits_ls = instructionWriteReport_bits_ls;
  wire        initRecord_bits_st = instructionWriteReport_bits_st;
  wire        initRecord_bits_gather = instructionWriteReport_bits_gather;
  wire        initRecord_bits_gather16 = instructionWriteReport_bits_gather16;
  wire        initRecord_bits_crossWrite = instructionWriteReport_bits_crossWrite;
  wire        initRecord_bits_crossRead = instructionWriteReport_bits_crossRead;
  wire        initRecord_bits_indexType = instructionWriteReport_bits_indexType;
  wire        initRecord_bits_ma = instructionWriteReport_bits_ma;
  wire        initRecord_bits_onlyRead = instructionWriteReport_bits_onlyRead;
  wire        initRecord_bits_slow = instructionWriteReport_bits_slow;
  wire [15:0] initRecord_bits_elementMask = instructionWriteReport_bits_elementMask;
  wire        initRecord_bits_state_stFinish = instructionWriteReport_bits_state_stFinish;
  wire        initRecord_bits_state_wWriteQueueClear = instructionWriteReport_bits_state_wWriteQueueClear;
  wire        initRecord_bits_state_wLaneLastReport = instructionWriteReport_bits_state_wLaneLastReport;
  wire        initRecord_bits_state_wTopLastReport = instructionWriteReport_bits_state_wTopLastReport;
  wire [1:0]  bankReadS_0 = 2'h0;
  wire [35:0] vrfSRAM_0_readwritePorts_0_writeData = 36'h0;
  wire [35:0] vrfSRAM_1_readwritePorts_0_writeData = 36'h0;
  wire        initRecord_valid = 1'h1;
  wire        firstUsed = 1'h0;
  wire        vrfSRAM_0_readwritePorts_0_isWrite = 1'h0;
  wire        vrfSRAM_1_readwritePorts_0_isWrite = 1'h0;
  wire        initRecord_bits_state_wLaneClear = 1'h0;
  reg         sramReady;
  reg  [4:0]  sramResetCount;
  wire        resetValid = ~sramReady;
  wire        readRequests_0_ready_0;
  wire        _pipeFire_T = readRequests_0_ready_0 & readRequests_0_valid_0;
  wire        readRequests_1_ready_0;
  wire        _pipeFire_T_1 = readRequests_1_ready_0 & readRequests_1_valid_0;
  wire        readRequests_2_ready_0;
  wire        _pipeFire_T_2 = readRequests_2_ready_0 & readRequests_2_valid_0;
  wire        readRequests_3_ready_0;
  wire        _pipeFire_T_3 = readRequests_3_ready_0 & readRequests_3_valid_0;
  wire        readRequests_4_ready_0;
  wire        _pipeFire_T_4 = readRequests_4_ready_0 & readRequests_4_valid_0;
  wire        readRequests_5_ready_0;
  wire        _pipeFire_T_5 = readRequests_5_ready_0 & readRequests_5_valid_0;
  wire        readRequests_6_ready_0;
  wire        _pipeFire_T_6 = readRequests_6_ready_0 & readRequests_6_valid_0;
  wire        readRequests_7_ready_0;
  wire        _pipeFire_T_7 = readRequests_7_ready_0 & readRequests_7_valid_0;
  wire        readRequests_8_ready_0;
  wire        _pipeFire_T_8 = readRequests_8_ready_0 & readRequests_8_valid_0;
  wire        readRequests_9_ready_0;
  wire        _pipeFire_T_9 = readRequests_9_ready_0 & readRequests_9_valid_0;
  wire        readRequests_10_ready_0;
  wire        _pipeFire_T_10 = readRequests_10_ready_0 & readRequests_10_valid_0;
  wire        readRequests_11_ready_0;
  wire        _pipeFire_T_11 = readRequests_11_ready_0 & readRequests_11_valid_0;
  wire        readRequests_12_ready_0;
  wire        _pipeFire_T_12 = readRequests_12_ready_0 & readRequests_12_valid_0;
  wire        readRequests_13_ready_0;
  wire        _loadUpdateValidVec_T_27 = readRequests_13_ready_0 & readRequests_13_valid_0;
  wire        write_ready_0;
  wire        _writePipe_valid_T = write_ready_0 & write_valid_0;
  wire [3:0]  portFireCount =
    {1'h0, {1'h0, {1'h0, _pipeFire_T} + {1'h0, _pipeFire_T_1} + {1'h0, _pipeFire_T_2}} + {1'h0, {1'h0, _pipeFire_T_3} + {1'h0, _pipeFire_T_4}} + {1'h0, {1'h0, _pipeFire_T_5} + {1'h0, _pipeFire_T_6}}}
    + {1'h0, {1'h0, {1'h0, _pipeFire_T_7} + {1'h0, _pipeFire_T_8}} + {1'h0, {1'h0, _pipeFire_T_9} + {1'h0, _pipeFire_T_10}}}
    + {1'h0, {1'h0, {1'h0, _pipeFire_T_11} + {1'h0, _pipeFire_T_12}} + {1'h0, {1'h0, _loadUpdateValidVec_T_27} + {1'h0, _writePipe_valid_T}}};
  wire [5:0]  writeIndex = {write_bits_vd_0, write_bits_offset_0};
  wire [1:0]  writeBank = 2'h1 << writeIndex[0];
  reg         chainingRecord_0_valid;
  reg         chainingRecord_0_bits_vd_valid;
  reg  [4:0]  chainingRecord_0_bits_vd_bits;
  reg         chainingRecord_0_bits_vs1_valid;
  reg  [4:0]  chainingRecord_0_bits_vs1_bits;
  reg  [4:0]  chainingRecord_0_bits_vs2;
  reg  [2:0]  chainingRecord_0_bits_instIndex;
  reg         chainingRecord_0_bits_ls;
  reg         chainingRecord_0_bits_st;
  reg         chainingRecord_0_bits_gather;
  reg         chainingRecord_0_bits_gather16;
  reg         chainingRecord_0_bits_crossWrite;
  reg         chainingRecord_0_bits_crossRead;
  reg         chainingRecord_0_bits_indexType;
  reg         chainingRecord_0_bits_ma;
  reg         chainingRecord_0_bits_onlyRead;
  reg         chainingRecord_0_bits_slow;
  reg  [15:0] chainingRecord_0_bits_elementMask;
  reg         chainingRecord_0_bits_state_stFinish;
  reg         chainingRecord_0_bits_state_wWriteQueueClear;
  reg         chainingRecord_0_bits_state_wLaneLastReport;
  reg         chainingRecord_0_bits_state_wTopLastReport;
  reg         chainingRecord_0_bits_state_wLaneClear;
  reg         chainingRecord_1_valid;
  reg         chainingRecord_1_bits_vd_valid;
  reg  [4:0]  chainingRecord_1_bits_vd_bits;
  reg         chainingRecord_1_bits_vs1_valid;
  reg  [4:0]  chainingRecord_1_bits_vs1_bits;
  reg  [4:0]  chainingRecord_1_bits_vs2;
  reg  [2:0]  chainingRecord_1_bits_instIndex;
  reg         chainingRecord_1_bits_ls;
  reg         chainingRecord_1_bits_st;
  reg         chainingRecord_1_bits_gather;
  reg         chainingRecord_1_bits_gather16;
  reg         chainingRecord_1_bits_crossWrite;
  reg         chainingRecord_1_bits_crossRead;
  reg         chainingRecord_1_bits_indexType;
  reg         chainingRecord_1_bits_ma;
  reg         chainingRecord_1_bits_onlyRead;
  reg         chainingRecord_1_bits_slow;
  reg  [15:0] chainingRecord_1_bits_elementMask;
  reg         chainingRecord_1_bits_state_stFinish;
  reg         chainingRecord_1_bits_state_wWriteQueueClear;
  reg         chainingRecord_1_bits_state_wLaneLastReport;
  reg         chainingRecord_1_bits_state_wTopLastReport;
  reg         chainingRecord_1_bits_state_wLaneClear;
  reg         chainingRecord_2_valid;
  reg         chainingRecord_2_bits_vd_valid;
  reg  [4:0]  chainingRecord_2_bits_vd_bits;
  reg         chainingRecord_2_bits_vs1_valid;
  reg  [4:0]  chainingRecord_2_bits_vs1_bits;
  reg  [4:0]  chainingRecord_2_bits_vs2;
  reg  [2:0]  chainingRecord_2_bits_instIndex;
  reg         chainingRecord_2_bits_ls;
  reg         chainingRecord_2_bits_st;
  reg         chainingRecord_2_bits_gather;
  reg         chainingRecord_2_bits_gather16;
  reg         chainingRecord_2_bits_crossWrite;
  reg         chainingRecord_2_bits_crossRead;
  reg         chainingRecord_2_bits_indexType;
  reg         chainingRecord_2_bits_ma;
  reg         chainingRecord_2_bits_onlyRead;
  reg         chainingRecord_2_bits_slow;
  reg  [15:0] chainingRecord_2_bits_elementMask;
  reg         chainingRecord_2_bits_state_stFinish;
  reg         chainingRecord_2_bits_state_wWriteQueueClear;
  reg         chainingRecord_2_bits_state_wLaneLastReport;
  reg         chainingRecord_2_bits_state_wTopLastReport;
  reg         chainingRecord_2_bits_state_wLaneClear;
  reg         chainingRecord_3_valid;
  reg         chainingRecord_3_bits_vd_valid;
  reg  [4:0]  chainingRecord_3_bits_vd_bits;
  reg         chainingRecord_3_bits_vs1_valid;
  reg  [4:0]  chainingRecord_3_bits_vs1_bits;
  reg  [4:0]  chainingRecord_3_bits_vs2;
  reg  [2:0]  chainingRecord_3_bits_instIndex;
  reg         chainingRecord_3_bits_ls;
  reg         chainingRecord_3_bits_st;
  reg         chainingRecord_3_bits_gather;
  reg         chainingRecord_3_bits_gather16;
  reg         chainingRecord_3_bits_crossWrite;
  reg         chainingRecord_3_bits_crossRead;
  reg         chainingRecord_3_bits_indexType;
  reg         chainingRecord_3_bits_ma;
  reg         chainingRecord_3_bits_onlyRead;
  reg         chainingRecord_3_bits_slow;
  reg  [15:0] chainingRecord_3_bits_elementMask;
  reg         chainingRecord_3_bits_state_stFinish;
  reg         chainingRecord_3_bits_state_wWriteQueueClear;
  reg         chainingRecord_3_bits_state_wLaneLastReport;
  reg         chainingRecord_3_bits_state_wTopLastReport;
  reg         chainingRecord_3_bits_state_wLaneClear;
  reg         chainingRecord_4_valid;
  reg         chainingRecord_4_bits_vd_valid;
  reg  [4:0]  chainingRecord_4_bits_vd_bits;
  reg         chainingRecord_4_bits_vs1_valid;
  reg  [4:0]  chainingRecord_4_bits_vs1_bits;
  reg  [4:0]  chainingRecord_4_bits_vs2;
  reg  [2:0]  chainingRecord_4_bits_instIndex;
  reg         chainingRecord_4_bits_ls;
  reg         chainingRecord_4_bits_st;
  reg         chainingRecord_4_bits_gather;
  reg         chainingRecord_4_bits_gather16;
  reg         chainingRecord_4_bits_crossWrite;
  reg         chainingRecord_4_bits_crossRead;
  reg         chainingRecord_4_bits_indexType;
  reg         chainingRecord_4_bits_ma;
  reg         chainingRecord_4_bits_onlyRead;
  reg         chainingRecord_4_bits_slow;
  reg  [15:0] chainingRecord_4_bits_elementMask;
  reg         chainingRecord_4_bits_state_stFinish;
  reg         chainingRecord_4_bits_state_wWriteQueueClear;
  reg         chainingRecord_4_bits_state_wLaneLastReport;
  reg         chainingRecord_4_bits_state_wTopLastReport;
  reg         chainingRecord_4_bits_state_wLaneClear;
  reg         chainingRecordCopy_0_valid;
  reg         chainingRecordCopy_0_bits_vd_valid;
  reg  [4:0]  chainingRecordCopy_0_bits_vd_bits;
  reg         chainingRecordCopy_0_bits_vs1_valid;
  reg  [4:0]  chainingRecordCopy_0_bits_vs1_bits;
  reg  [4:0]  chainingRecordCopy_0_bits_vs2;
  reg  [2:0]  chainingRecordCopy_0_bits_instIndex;
  reg         chainingRecordCopy_0_bits_ls;
  reg         chainingRecordCopy_0_bits_st;
  reg         chainingRecordCopy_0_bits_gather;
  reg         chainingRecordCopy_0_bits_gather16;
  reg         chainingRecordCopy_0_bits_crossWrite;
  reg         chainingRecordCopy_0_bits_crossRead;
  reg         chainingRecordCopy_0_bits_indexType;
  reg         chainingRecordCopy_0_bits_ma;
  reg         chainingRecordCopy_0_bits_onlyRead;
  reg         chainingRecordCopy_0_bits_slow;
  reg  [15:0] chainingRecordCopy_0_bits_elementMask;
  reg         chainingRecordCopy_0_bits_state_stFinish;
  reg         chainingRecordCopy_0_bits_state_wWriteQueueClear;
  reg         chainingRecordCopy_0_bits_state_wLaneLastReport;
  reg         chainingRecordCopy_0_bits_state_wTopLastReport;
  reg         chainingRecordCopy_0_bits_state_wLaneClear;
  reg         chainingRecordCopy_1_valid;
  reg         chainingRecordCopy_1_bits_vd_valid;
  reg  [4:0]  chainingRecordCopy_1_bits_vd_bits;
  reg         chainingRecordCopy_1_bits_vs1_valid;
  reg  [4:0]  chainingRecordCopy_1_bits_vs1_bits;
  reg  [4:0]  chainingRecordCopy_1_bits_vs2;
  reg  [2:0]  chainingRecordCopy_1_bits_instIndex;
  reg         chainingRecordCopy_1_bits_ls;
  reg         chainingRecordCopy_1_bits_st;
  reg         chainingRecordCopy_1_bits_gather;
  reg         chainingRecordCopy_1_bits_gather16;
  reg         chainingRecordCopy_1_bits_crossWrite;
  reg         chainingRecordCopy_1_bits_crossRead;
  reg         chainingRecordCopy_1_bits_indexType;
  reg         chainingRecordCopy_1_bits_ma;
  reg         chainingRecordCopy_1_bits_onlyRead;
  reg         chainingRecordCopy_1_bits_slow;
  reg  [15:0] chainingRecordCopy_1_bits_elementMask;
  reg         chainingRecordCopy_1_bits_state_stFinish;
  reg         chainingRecordCopy_1_bits_state_wWriteQueueClear;
  reg         chainingRecordCopy_1_bits_state_wLaneLastReport;
  reg         chainingRecordCopy_1_bits_state_wTopLastReport;
  reg         chainingRecordCopy_1_bits_state_wLaneClear;
  reg         chainingRecordCopy_2_valid;
  reg         chainingRecordCopy_2_bits_vd_valid;
  reg  [4:0]  chainingRecordCopy_2_bits_vd_bits;
  reg         chainingRecordCopy_2_bits_vs1_valid;
  reg  [4:0]  chainingRecordCopy_2_bits_vs1_bits;
  reg  [4:0]  chainingRecordCopy_2_bits_vs2;
  reg  [2:0]  chainingRecordCopy_2_bits_instIndex;
  reg         chainingRecordCopy_2_bits_ls;
  reg         chainingRecordCopy_2_bits_st;
  reg         chainingRecordCopy_2_bits_gather;
  reg         chainingRecordCopy_2_bits_gather16;
  reg         chainingRecordCopy_2_bits_crossWrite;
  reg         chainingRecordCopy_2_bits_crossRead;
  reg         chainingRecordCopy_2_bits_indexType;
  reg         chainingRecordCopy_2_bits_ma;
  reg         chainingRecordCopy_2_bits_onlyRead;
  reg         chainingRecordCopy_2_bits_slow;
  reg  [15:0] chainingRecordCopy_2_bits_elementMask;
  reg         chainingRecordCopy_2_bits_state_stFinish;
  reg         chainingRecordCopy_2_bits_state_wWriteQueueClear;
  reg         chainingRecordCopy_2_bits_state_wLaneLastReport;
  reg         chainingRecordCopy_2_bits_state_wTopLastReport;
  reg         chainingRecordCopy_2_bits_state_wLaneClear;
  reg         chainingRecordCopy_3_valid;
  reg         chainingRecordCopy_3_bits_vd_valid;
  reg  [4:0]  chainingRecordCopy_3_bits_vd_bits;
  reg         chainingRecordCopy_3_bits_vs1_valid;
  reg  [4:0]  chainingRecordCopy_3_bits_vs1_bits;
  reg  [4:0]  chainingRecordCopy_3_bits_vs2;
  reg  [2:0]  chainingRecordCopy_3_bits_instIndex;
  reg         chainingRecordCopy_3_bits_ls;
  reg         chainingRecordCopy_3_bits_st;
  reg         chainingRecordCopy_3_bits_gather;
  reg         chainingRecordCopy_3_bits_gather16;
  reg         chainingRecordCopy_3_bits_crossWrite;
  reg         chainingRecordCopy_3_bits_crossRead;
  reg         chainingRecordCopy_3_bits_indexType;
  reg         chainingRecordCopy_3_bits_ma;
  reg         chainingRecordCopy_3_bits_onlyRead;
  reg         chainingRecordCopy_3_bits_slow;
  reg  [15:0] chainingRecordCopy_3_bits_elementMask;
  reg         chainingRecordCopy_3_bits_state_stFinish;
  reg         chainingRecordCopy_3_bits_state_wWriteQueueClear;
  reg         chainingRecordCopy_3_bits_state_wLaneLastReport;
  reg         chainingRecordCopy_3_bits_state_wTopLastReport;
  reg         chainingRecordCopy_3_bits_state_wLaneClear;
  reg         chainingRecordCopy_4_valid;
  reg         chainingRecordCopy_4_bits_vd_valid;
  reg  [4:0]  chainingRecordCopy_4_bits_vd_bits;
  reg         chainingRecordCopy_4_bits_vs1_valid;
  reg  [4:0]  chainingRecordCopy_4_bits_vs1_bits;
  reg  [4:0]  chainingRecordCopy_4_bits_vs2;
  reg  [2:0]  chainingRecordCopy_4_bits_instIndex;
  reg         chainingRecordCopy_4_bits_ls;
  reg         chainingRecordCopy_4_bits_st;
  reg         chainingRecordCopy_4_bits_gather;
  reg         chainingRecordCopy_4_bits_gather16;
  reg         chainingRecordCopy_4_bits_crossWrite;
  reg         chainingRecordCopy_4_bits_crossRead;
  reg         chainingRecordCopy_4_bits_indexType;
  reg         chainingRecordCopy_4_bits_ma;
  reg         chainingRecordCopy_4_bits_onlyRead;
  reg         chainingRecordCopy_4_bits_slow;
  reg  [15:0] chainingRecordCopy_4_bits_elementMask;
  reg         chainingRecordCopy_4_bits_state_stFinish;
  reg         chainingRecordCopy_4_bits_state_wWriteQueueClear;
  reg         chainingRecordCopy_4_bits_state_wLaneLastReport;
  reg         chainingRecordCopy_4_bits_state_wTopLastReport;
  reg         chainingRecordCopy_4_bits_state_wLaneClear;
  wire        recordValidVec_0 = ~(&chainingRecord_0_bits_elementMask) & chainingRecord_0_valid;
  wire        recordValidVec_1 = ~(&chainingRecord_1_bits_elementMask) & chainingRecord_1_valid;
  wire        recordValidVec_2 = ~(&chainingRecord_2_bits_elementMask) & chainingRecord_2_valid;
  wire        recordValidVec_3 = ~(&chainingRecord_3_bits_elementMask) & chainingRecord_3_valid;
  wire        recordValidVec_4 = ~(&chainingRecord_4_bits_elementMask) & chainingRecord_4_valid;
  wire [1:0]  bankCorrect;
  reg         firstReadPipe_0_valid;
  wire        vrfSRAM_0_readwritePorts_0_enable = firstReadPipe_0_valid;
  reg  [4:0]  firstReadPipe_0_bits_address;
  wire [4:0]  vrfSRAM_0_readwritePorts_0_address = firstReadPipe_0_bits_address;
  reg         firstReadPipe_1_valid;
  wire        vrfSRAM_1_readwritePorts_0_enable = firstReadPipe_1_valid;
  reg  [4:0]  firstReadPipe_1_bits_address;
  wire [4:0]  vrfSRAM_1_readwritePorts_0_address = firstReadPipe_1_bits_address;
  reg         secondReadPipe_0_valid;
  reg  [4:0]  secondReadPipe_0_bits_address;
  reg         secondReadPipe_1_valid;
  reg  [4:0]  secondReadPipe_1_bits_address;
  reg         writePipe_valid;
  reg  [4:0]  writePipe_bits_vd;
  reg         writePipe_bits_offset;
  reg  [3:0]  writePipe_bits_mask;
  reg  [31:0] writePipe_bits_data;
  reg         writePipe_bits_last;
  reg  [2:0]  writePipe_bits_instructionIndex;
  reg  [1:0]  writeBankPipe;
  wire        _readRecord_T = chainingRecord_0_bits_instIndex == readCheck_0_instructionIndex;
  wire        _readRecord_T_1 = chainingRecord_1_bits_instIndex == readCheck_0_instructionIndex;
  wire        _readRecord_T_2 = chainingRecord_2_bits_instIndex == readCheck_0_instructionIndex;
  wire        _readRecord_T_3 = chainingRecord_3_bits_instIndex == readCheck_0_instructionIndex;
  wire        _readRecord_T_4 = chainingRecord_4_bits_instIndex == readCheck_0_instructionIndex;
  wire        readRecord_state_wLaneClear =
    _readRecord_T & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_1 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_2 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_3 & chainingRecord_3_bits_state_wLaneClear
    | _readRecord_T_4 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_state_wTopLastReport =
    _readRecord_T & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_1 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_2 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_3
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_4 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_state_wLaneLastReport =
    _readRecord_T & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_1 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_2 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_3
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_4 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_state_wWriteQueueClear =
    _readRecord_T & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_1 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_2 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_3
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_4 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_state_stFinish =
    _readRecord_T & chainingRecord_0_bits_state_stFinish | _readRecord_T_1 & chainingRecord_1_bits_state_stFinish | _readRecord_T_2 & chainingRecord_2_bits_state_stFinish | _readRecord_T_3 & chainingRecord_3_bits_state_stFinish
    | _readRecord_T_4 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_elementMask =
    (_readRecord_T ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_1 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_2 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_3 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_4 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_slow =
    _readRecord_T & chainingRecord_0_bits_slow | _readRecord_T_1 & chainingRecord_1_bits_slow | _readRecord_T_2 & chainingRecord_2_bits_slow | _readRecord_T_3 & chainingRecord_3_bits_slow | _readRecord_T_4 & chainingRecord_4_bits_slow;
  wire        readRecord_onlyRead =
    _readRecord_T & chainingRecord_0_bits_onlyRead | _readRecord_T_1 & chainingRecord_1_bits_onlyRead | _readRecord_T_2 & chainingRecord_2_bits_onlyRead | _readRecord_T_3 & chainingRecord_3_bits_onlyRead | _readRecord_T_4
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_ma =
    _readRecord_T & chainingRecord_0_bits_ma | _readRecord_T_1 & chainingRecord_1_bits_ma | _readRecord_T_2 & chainingRecord_2_bits_ma | _readRecord_T_3 & chainingRecord_3_bits_ma | _readRecord_T_4 & chainingRecord_4_bits_ma;
  wire        readRecord_indexType =
    _readRecord_T & chainingRecord_0_bits_indexType | _readRecord_T_1 & chainingRecord_1_bits_indexType | _readRecord_T_2 & chainingRecord_2_bits_indexType | _readRecord_T_3 & chainingRecord_3_bits_indexType | _readRecord_T_4
    & chainingRecord_4_bits_indexType;
  wire        readRecord_crossRead =
    _readRecord_T & chainingRecord_0_bits_crossRead | _readRecord_T_1 & chainingRecord_1_bits_crossRead | _readRecord_T_2 & chainingRecord_2_bits_crossRead | _readRecord_T_3 & chainingRecord_3_bits_crossRead | _readRecord_T_4
    & chainingRecord_4_bits_crossRead;
  wire        readRecord_crossWrite =
    _readRecord_T & chainingRecord_0_bits_crossWrite | _readRecord_T_1 & chainingRecord_1_bits_crossWrite | _readRecord_T_2 & chainingRecord_2_bits_crossWrite | _readRecord_T_3 & chainingRecord_3_bits_crossWrite | _readRecord_T_4
    & chainingRecord_4_bits_crossWrite;
  wire        readRecord_gather16 =
    _readRecord_T & chainingRecord_0_bits_gather16 | _readRecord_T_1 & chainingRecord_1_bits_gather16 | _readRecord_T_2 & chainingRecord_2_bits_gather16 | _readRecord_T_3 & chainingRecord_3_bits_gather16 | _readRecord_T_4
    & chainingRecord_4_bits_gather16;
  wire        readRecord_gather =
    _readRecord_T & chainingRecord_0_bits_gather | _readRecord_T_1 & chainingRecord_1_bits_gather | _readRecord_T_2 & chainingRecord_2_bits_gather | _readRecord_T_3 & chainingRecord_3_bits_gather | _readRecord_T_4
    & chainingRecord_4_bits_gather;
  wire        readRecord_st =
    _readRecord_T & chainingRecord_0_bits_st | _readRecord_T_1 & chainingRecord_1_bits_st | _readRecord_T_2 & chainingRecord_2_bits_st | _readRecord_T_3 & chainingRecord_3_bits_st | _readRecord_T_4 & chainingRecord_4_bits_st;
  wire        readRecord_ls =
    _readRecord_T & chainingRecord_0_bits_ls | _readRecord_T_1 & chainingRecord_1_bits_ls | _readRecord_T_2 & chainingRecord_2_bits_ls | _readRecord_T_3 & chainingRecord_3_bits_ls | _readRecord_T_4 & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_instIndex =
    (_readRecord_T ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_1 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_2 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_3 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_4 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_vs2 =
    (_readRecord_T ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_1 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_2 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_3 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_4 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_vs1_bits =
    (_readRecord_T ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_1 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_2 ? chainingRecord_2_bits_vs1_bits : 5'h0) | (_readRecord_T_3 ? chainingRecord_3_bits_vs1_bits : 5'h0)
    | (_readRecord_T_4 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_vs1_valid =
    _readRecord_T & chainingRecord_0_bits_vs1_valid | _readRecord_T_1 & chainingRecord_1_bits_vs1_valid | _readRecord_T_2 & chainingRecord_2_bits_vs1_valid | _readRecord_T_3 & chainingRecord_3_bits_vs1_valid | _readRecord_T_4
    & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_vd_bits =
    (_readRecord_T ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_1 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_2 ? chainingRecord_2_bits_vd_bits : 5'h0) | (_readRecord_T_3 ? chainingRecord_3_bits_vd_bits : 5'h0)
    | (_readRecord_T_4 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_vd_valid =
    _readRecord_T & chainingRecord_0_bits_vd_valid | _readRecord_T_1 & chainingRecord_1_bits_vd_valid | _readRecord_T_2 & chainingRecord_2_bits_vd_valid | _readRecord_T_3 & chainingRecord_3_bits_vd_valid | _readRecord_T_4
    & chainingRecord_4_bits_vd_valid;
  wire        _readRecord_T_220 = chainingRecord_0_bits_instIndex == readCheck_1_instructionIndex;
  wire        _readRecord_T_221 = chainingRecord_1_bits_instIndex == readCheck_1_instructionIndex;
  wire        _readRecord_T_222 = chainingRecord_2_bits_instIndex == readCheck_1_instructionIndex;
  wire        _readRecord_T_223 = chainingRecord_3_bits_instIndex == readCheck_1_instructionIndex;
  wire        _readRecord_T_224 = chainingRecord_4_bits_instIndex == readCheck_1_instructionIndex;
  wire        readRecord_1_state_wLaneClear =
    _readRecord_T_220 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_221 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_222 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_223
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_224 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_1_state_wTopLastReport =
    _readRecord_T_220 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_221 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_222 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_223
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_224 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_1_state_wLaneLastReport =
    _readRecord_T_220 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_221 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_222 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_223
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_224 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_1_state_wWriteQueueClear =
    _readRecord_T_220 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_221 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_222 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_223
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_224 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_1_state_stFinish =
    _readRecord_T_220 & chainingRecord_0_bits_state_stFinish | _readRecord_T_221 & chainingRecord_1_bits_state_stFinish | _readRecord_T_222 & chainingRecord_2_bits_state_stFinish | _readRecord_T_223 & chainingRecord_3_bits_state_stFinish
    | _readRecord_T_224 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_1_elementMask =
    (_readRecord_T_220 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_221 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_222 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_223 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_224 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_1_slow =
    _readRecord_T_220 & chainingRecord_0_bits_slow | _readRecord_T_221 & chainingRecord_1_bits_slow | _readRecord_T_222 & chainingRecord_2_bits_slow | _readRecord_T_223 & chainingRecord_3_bits_slow | _readRecord_T_224
    & chainingRecord_4_bits_slow;
  wire        readRecord_1_onlyRead =
    _readRecord_T_220 & chainingRecord_0_bits_onlyRead | _readRecord_T_221 & chainingRecord_1_bits_onlyRead | _readRecord_T_222 & chainingRecord_2_bits_onlyRead | _readRecord_T_223 & chainingRecord_3_bits_onlyRead | _readRecord_T_224
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_1_ma =
    _readRecord_T_220 & chainingRecord_0_bits_ma | _readRecord_T_221 & chainingRecord_1_bits_ma | _readRecord_T_222 & chainingRecord_2_bits_ma | _readRecord_T_223 & chainingRecord_3_bits_ma | _readRecord_T_224 & chainingRecord_4_bits_ma;
  wire        readRecord_1_indexType =
    _readRecord_T_220 & chainingRecord_0_bits_indexType | _readRecord_T_221 & chainingRecord_1_bits_indexType | _readRecord_T_222 & chainingRecord_2_bits_indexType | _readRecord_T_223 & chainingRecord_3_bits_indexType | _readRecord_T_224
    & chainingRecord_4_bits_indexType;
  wire        readRecord_1_crossRead =
    _readRecord_T_220 & chainingRecord_0_bits_crossRead | _readRecord_T_221 & chainingRecord_1_bits_crossRead | _readRecord_T_222 & chainingRecord_2_bits_crossRead | _readRecord_T_223 & chainingRecord_3_bits_crossRead | _readRecord_T_224
    & chainingRecord_4_bits_crossRead;
  wire        readRecord_1_crossWrite =
    _readRecord_T_220 & chainingRecord_0_bits_crossWrite | _readRecord_T_221 & chainingRecord_1_bits_crossWrite | _readRecord_T_222 & chainingRecord_2_bits_crossWrite | _readRecord_T_223 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_224 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_1_gather16 =
    _readRecord_T_220 & chainingRecord_0_bits_gather16 | _readRecord_T_221 & chainingRecord_1_bits_gather16 | _readRecord_T_222 & chainingRecord_2_bits_gather16 | _readRecord_T_223 & chainingRecord_3_bits_gather16 | _readRecord_T_224
    & chainingRecord_4_bits_gather16;
  wire        readRecord_1_gather =
    _readRecord_T_220 & chainingRecord_0_bits_gather | _readRecord_T_221 & chainingRecord_1_bits_gather | _readRecord_T_222 & chainingRecord_2_bits_gather | _readRecord_T_223 & chainingRecord_3_bits_gather | _readRecord_T_224
    & chainingRecord_4_bits_gather;
  wire        readRecord_1_st =
    _readRecord_T_220 & chainingRecord_0_bits_st | _readRecord_T_221 & chainingRecord_1_bits_st | _readRecord_T_222 & chainingRecord_2_bits_st | _readRecord_T_223 & chainingRecord_3_bits_st | _readRecord_T_224 & chainingRecord_4_bits_st;
  wire        readRecord_1_ls =
    _readRecord_T_220 & chainingRecord_0_bits_ls | _readRecord_T_221 & chainingRecord_1_bits_ls | _readRecord_T_222 & chainingRecord_2_bits_ls | _readRecord_T_223 & chainingRecord_3_bits_ls | _readRecord_T_224 & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_1_instIndex =
    (_readRecord_T_220 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_221 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_222 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_223 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_224 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_1_vs2 =
    (_readRecord_T_220 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_221 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_222 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_223 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_224 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_1_vs1_bits =
    (_readRecord_T_220 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_221 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_222 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_223 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_224 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_1_vs1_valid =
    _readRecord_T_220 & chainingRecord_0_bits_vs1_valid | _readRecord_T_221 & chainingRecord_1_bits_vs1_valid | _readRecord_T_222 & chainingRecord_2_bits_vs1_valid | _readRecord_T_223 & chainingRecord_3_bits_vs1_valid | _readRecord_T_224
    & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_1_vd_bits =
    (_readRecord_T_220 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_221 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_222 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_223 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_224 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_1_vd_valid =
    _readRecord_T_220 & chainingRecord_0_bits_vd_valid | _readRecord_T_221 & chainingRecord_1_bits_vd_valid | _readRecord_T_222 & chainingRecord_2_bits_vd_valid | _readRecord_T_223 & chainingRecord_3_bits_vd_valid | _readRecord_T_224
    & chainingRecord_4_bits_vd_valid;
  wire        _readRecord_T_440 = chainingRecord_0_bits_instIndex == readCheck_2_instructionIndex;
  wire        _readRecord_T_441 = chainingRecord_1_bits_instIndex == readCheck_2_instructionIndex;
  wire        _readRecord_T_442 = chainingRecord_2_bits_instIndex == readCheck_2_instructionIndex;
  wire        _readRecord_T_443 = chainingRecord_3_bits_instIndex == readCheck_2_instructionIndex;
  wire        _readRecord_T_444 = chainingRecord_4_bits_instIndex == readCheck_2_instructionIndex;
  wire        readRecord_2_state_wLaneClear =
    _readRecord_T_440 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_441 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_442 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_443
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_444 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_2_state_wTopLastReport =
    _readRecord_T_440 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_441 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_442 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_443
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_444 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_2_state_wLaneLastReport =
    _readRecord_T_440 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_441 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_442 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_443
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_444 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_2_state_wWriteQueueClear =
    _readRecord_T_440 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_441 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_442 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_443
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_444 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_2_state_stFinish =
    _readRecord_T_440 & chainingRecord_0_bits_state_stFinish | _readRecord_T_441 & chainingRecord_1_bits_state_stFinish | _readRecord_T_442 & chainingRecord_2_bits_state_stFinish | _readRecord_T_443 & chainingRecord_3_bits_state_stFinish
    | _readRecord_T_444 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_2_elementMask =
    (_readRecord_T_440 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_441 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_442 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_443 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_444 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_2_slow =
    _readRecord_T_440 & chainingRecord_0_bits_slow | _readRecord_T_441 & chainingRecord_1_bits_slow | _readRecord_T_442 & chainingRecord_2_bits_slow | _readRecord_T_443 & chainingRecord_3_bits_slow | _readRecord_T_444
    & chainingRecord_4_bits_slow;
  wire        readRecord_2_onlyRead =
    _readRecord_T_440 & chainingRecord_0_bits_onlyRead | _readRecord_T_441 & chainingRecord_1_bits_onlyRead | _readRecord_T_442 & chainingRecord_2_bits_onlyRead | _readRecord_T_443 & chainingRecord_3_bits_onlyRead | _readRecord_T_444
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_2_ma =
    _readRecord_T_440 & chainingRecord_0_bits_ma | _readRecord_T_441 & chainingRecord_1_bits_ma | _readRecord_T_442 & chainingRecord_2_bits_ma | _readRecord_T_443 & chainingRecord_3_bits_ma | _readRecord_T_444 & chainingRecord_4_bits_ma;
  wire        readRecord_2_indexType =
    _readRecord_T_440 & chainingRecord_0_bits_indexType | _readRecord_T_441 & chainingRecord_1_bits_indexType | _readRecord_T_442 & chainingRecord_2_bits_indexType | _readRecord_T_443 & chainingRecord_3_bits_indexType | _readRecord_T_444
    & chainingRecord_4_bits_indexType;
  wire        readRecord_2_crossRead =
    _readRecord_T_440 & chainingRecord_0_bits_crossRead | _readRecord_T_441 & chainingRecord_1_bits_crossRead | _readRecord_T_442 & chainingRecord_2_bits_crossRead | _readRecord_T_443 & chainingRecord_3_bits_crossRead | _readRecord_T_444
    & chainingRecord_4_bits_crossRead;
  wire        readRecord_2_crossWrite =
    _readRecord_T_440 & chainingRecord_0_bits_crossWrite | _readRecord_T_441 & chainingRecord_1_bits_crossWrite | _readRecord_T_442 & chainingRecord_2_bits_crossWrite | _readRecord_T_443 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_444 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_2_gather16 =
    _readRecord_T_440 & chainingRecord_0_bits_gather16 | _readRecord_T_441 & chainingRecord_1_bits_gather16 | _readRecord_T_442 & chainingRecord_2_bits_gather16 | _readRecord_T_443 & chainingRecord_3_bits_gather16 | _readRecord_T_444
    & chainingRecord_4_bits_gather16;
  wire        readRecord_2_gather =
    _readRecord_T_440 & chainingRecord_0_bits_gather | _readRecord_T_441 & chainingRecord_1_bits_gather | _readRecord_T_442 & chainingRecord_2_bits_gather | _readRecord_T_443 & chainingRecord_3_bits_gather | _readRecord_T_444
    & chainingRecord_4_bits_gather;
  wire        readRecord_2_st =
    _readRecord_T_440 & chainingRecord_0_bits_st | _readRecord_T_441 & chainingRecord_1_bits_st | _readRecord_T_442 & chainingRecord_2_bits_st | _readRecord_T_443 & chainingRecord_3_bits_st | _readRecord_T_444 & chainingRecord_4_bits_st;
  wire        readRecord_2_ls =
    _readRecord_T_440 & chainingRecord_0_bits_ls | _readRecord_T_441 & chainingRecord_1_bits_ls | _readRecord_T_442 & chainingRecord_2_bits_ls | _readRecord_T_443 & chainingRecord_3_bits_ls | _readRecord_T_444 & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_2_instIndex =
    (_readRecord_T_440 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_441 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_442 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_443 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_444 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_2_vs2 =
    (_readRecord_T_440 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_441 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_442 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_443 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_444 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_2_vs1_bits =
    (_readRecord_T_440 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_441 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_442 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_443 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_444 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_2_vs1_valid =
    _readRecord_T_440 & chainingRecord_0_bits_vs1_valid | _readRecord_T_441 & chainingRecord_1_bits_vs1_valid | _readRecord_T_442 & chainingRecord_2_bits_vs1_valid | _readRecord_T_443 & chainingRecord_3_bits_vs1_valid | _readRecord_T_444
    & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_2_vd_bits =
    (_readRecord_T_440 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_441 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_442 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_443 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_444 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_2_vd_valid =
    _readRecord_T_440 & chainingRecord_0_bits_vd_valid | _readRecord_T_441 & chainingRecord_1_bits_vd_valid | _readRecord_T_442 & chainingRecord_2_bits_vd_valid | _readRecord_T_443 & chainingRecord_3_bits_vd_valid | _readRecord_T_444
    & chainingRecord_4_bits_vd_valid;
  wire        _readRecord_T_660 = chainingRecord_0_bits_instIndex == readCheck_3_instructionIndex;
  wire        _readRecord_T_661 = chainingRecord_1_bits_instIndex == readCheck_3_instructionIndex;
  wire        _readRecord_T_662 = chainingRecord_2_bits_instIndex == readCheck_3_instructionIndex;
  wire        _readRecord_T_663 = chainingRecord_3_bits_instIndex == readCheck_3_instructionIndex;
  wire        _readRecord_T_664 = chainingRecord_4_bits_instIndex == readCheck_3_instructionIndex;
  wire        readRecord_3_state_wLaneClear =
    _readRecord_T_660 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_661 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_662 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_663
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_664 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_3_state_wTopLastReport =
    _readRecord_T_660 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_661 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_662 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_663
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_664 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_3_state_wLaneLastReport =
    _readRecord_T_660 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_661 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_662 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_663
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_664 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_3_state_wWriteQueueClear =
    _readRecord_T_660 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_661 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_662 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_663
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_664 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_3_state_stFinish =
    _readRecord_T_660 & chainingRecord_0_bits_state_stFinish | _readRecord_T_661 & chainingRecord_1_bits_state_stFinish | _readRecord_T_662 & chainingRecord_2_bits_state_stFinish | _readRecord_T_663 & chainingRecord_3_bits_state_stFinish
    | _readRecord_T_664 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_3_elementMask =
    (_readRecord_T_660 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_661 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_662 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_663 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_664 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_3_slow =
    _readRecord_T_660 & chainingRecord_0_bits_slow | _readRecord_T_661 & chainingRecord_1_bits_slow | _readRecord_T_662 & chainingRecord_2_bits_slow | _readRecord_T_663 & chainingRecord_3_bits_slow | _readRecord_T_664
    & chainingRecord_4_bits_slow;
  wire        readRecord_3_onlyRead =
    _readRecord_T_660 & chainingRecord_0_bits_onlyRead | _readRecord_T_661 & chainingRecord_1_bits_onlyRead | _readRecord_T_662 & chainingRecord_2_bits_onlyRead | _readRecord_T_663 & chainingRecord_3_bits_onlyRead | _readRecord_T_664
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_3_ma =
    _readRecord_T_660 & chainingRecord_0_bits_ma | _readRecord_T_661 & chainingRecord_1_bits_ma | _readRecord_T_662 & chainingRecord_2_bits_ma | _readRecord_T_663 & chainingRecord_3_bits_ma | _readRecord_T_664 & chainingRecord_4_bits_ma;
  wire        readRecord_3_indexType =
    _readRecord_T_660 & chainingRecord_0_bits_indexType | _readRecord_T_661 & chainingRecord_1_bits_indexType | _readRecord_T_662 & chainingRecord_2_bits_indexType | _readRecord_T_663 & chainingRecord_3_bits_indexType | _readRecord_T_664
    & chainingRecord_4_bits_indexType;
  wire        readRecord_3_crossRead =
    _readRecord_T_660 & chainingRecord_0_bits_crossRead | _readRecord_T_661 & chainingRecord_1_bits_crossRead | _readRecord_T_662 & chainingRecord_2_bits_crossRead | _readRecord_T_663 & chainingRecord_3_bits_crossRead | _readRecord_T_664
    & chainingRecord_4_bits_crossRead;
  wire        readRecord_3_crossWrite =
    _readRecord_T_660 & chainingRecord_0_bits_crossWrite | _readRecord_T_661 & chainingRecord_1_bits_crossWrite | _readRecord_T_662 & chainingRecord_2_bits_crossWrite | _readRecord_T_663 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_664 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_3_gather16 =
    _readRecord_T_660 & chainingRecord_0_bits_gather16 | _readRecord_T_661 & chainingRecord_1_bits_gather16 | _readRecord_T_662 & chainingRecord_2_bits_gather16 | _readRecord_T_663 & chainingRecord_3_bits_gather16 | _readRecord_T_664
    & chainingRecord_4_bits_gather16;
  wire        readRecord_3_gather =
    _readRecord_T_660 & chainingRecord_0_bits_gather | _readRecord_T_661 & chainingRecord_1_bits_gather | _readRecord_T_662 & chainingRecord_2_bits_gather | _readRecord_T_663 & chainingRecord_3_bits_gather | _readRecord_T_664
    & chainingRecord_4_bits_gather;
  wire        readRecord_3_st =
    _readRecord_T_660 & chainingRecord_0_bits_st | _readRecord_T_661 & chainingRecord_1_bits_st | _readRecord_T_662 & chainingRecord_2_bits_st | _readRecord_T_663 & chainingRecord_3_bits_st | _readRecord_T_664 & chainingRecord_4_bits_st;
  wire        readRecord_3_ls =
    _readRecord_T_660 & chainingRecord_0_bits_ls | _readRecord_T_661 & chainingRecord_1_bits_ls | _readRecord_T_662 & chainingRecord_2_bits_ls | _readRecord_T_663 & chainingRecord_3_bits_ls | _readRecord_T_664 & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_3_instIndex =
    (_readRecord_T_660 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_661 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_662 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_663 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_664 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_3_vs2 =
    (_readRecord_T_660 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_661 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_662 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_663 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_664 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_3_vs1_bits =
    (_readRecord_T_660 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_661 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_662 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_663 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_664 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_3_vs1_valid =
    _readRecord_T_660 & chainingRecord_0_bits_vs1_valid | _readRecord_T_661 & chainingRecord_1_bits_vs1_valid | _readRecord_T_662 & chainingRecord_2_bits_vs1_valid | _readRecord_T_663 & chainingRecord_3_bits_vs1_valid | _readRecord_T_664
    & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_3_vd_bits =
    (_readRecord_T_660 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_661 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_662 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_663 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_664 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_3_vd_valid =
    _readRecord_T_660 & chainingRecord_0_bits_vd_valid | _readRecord_T_661 & chainingRecord_1_bits_vd_valid | _readRecord_T_662 & chainingRecord_2_bits_vd_valid | _readRecord_T_663 & chainingRecord_3_bits_vd_valid | _readRecord_T_664
    & chainingRecord_4_bits_vd_valid;
  wire        _readRecord_T_880 = chainingRecord_0_bits_instIndex == readCheck_4_instructionIndex;
  wire        _readRecord_T_881 = chainingRecord_1_bits_instIndex == readCheck_4_instructionIndex;
  wire        _readRecord_T_882 = chainingRecord_2_bits_instIndex == readCheck_4_instructionIndex;
  wire        _readRecord_T_883 = chainingRecord_3_bits_instIndex == readCheck_4_instructionIndex;
  wire        _readRecord_T_884 = chainingRecord_4_bits_instIndex == readCheck_4_instructionIndex;
  wire        readRecord_4_state_wLaneClear =
    _readRecord_T_880 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_881 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_882 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_883
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_884 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_4_state_wTopLastReport =
    _readRecord_T_880 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_881 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_882 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_883
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_884 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_4_state_wLaneLastReport =
    _readRecord_T_880 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_881 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_882 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_883
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_884 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_4_state_wWriteQueueClear =
    _readRecord_T_880 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_881 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_882 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_883
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_884 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_4_state_stFinish =
    _readRecord_T_880 & chainingRecord_0_bits_state_stFinish | _readRecord_T_881 & chainingRecord_1_bits_state_stFinish | _readRecord_T_882 & chainingRecord_2_bits_state_stFinish | _readRecord_T_883 & chainingRecord_3_bits_state_stFinish
    | _readRecord_T_884 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_4_elementMask =
    (_readRecord_T_880 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_881 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_882 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_883 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_884 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_4_slow =
    _readRecord_T_880 & chainingRecord_0_bits_slow | _readRecord_T_881 & chainingRecord_1_bits_slow | _readRecord_T_882 & chainingRecord_2_bits_slow | _readRecord_T_883 & chainingRecord_3_bits_slow | _readRecord_T_884
    & chainingRecord_4_bits_slow;
  wire        readRecord_4_onlyRead =
    _readRecord_T_880 & chainingRecord_0_bits_onlyRead | _readRecord_T_881 & chainingRecord_1_bits_onlyRead | _readRecord_T_882 & chainingRecord_2_bits_onlyRead | _readRecord_T_883 & chainingRecord_3_bits_onlyRead | _readRecord_T_884
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_4_ma =
    _readRecord_T_880 & chainingRecord_0_bits_ma | _readRecord_T_881 & chainingRecord_1_bits_ma | _readRecord_T_882 & chainingRecord_2_bits_ma | _readRecord_T_883 & chainingRecord_3_bits_ma | _readRecord_T_884 & chainingRecord_4_bits_ma;
  wire        readRecord_4_indexType =
    _readRecord_T_880 & chainingRecord_0_bits_indexType | _readRecord_T_881 & chainingRecord_1_bits_indexType | _readRecord_T_882 & chainingRecord_2_bits_indexType | _readRecord_T_883 & chainingRecord_3_bits_indexType | _readRecord_T_884
    & chainingRecord_4_bits_indexType;
  wire        readRecord_4_crossRead =
    _readRecord_T_880 & chainingRecord_0_bits_crossRead | _readRecord_T_881 & chainingRecord_1_bits_crossRead | _readRecord_T_882 & chainingRecord_2_bits_crossRead | _readRecord_T_883 & chainingRecord_3_bits_crossRead | _readRecord_T_884
    & chainingRecord_4_bits_crossRead;
  wire        readRecord_4_crossWrite =
    _readRecord_T_880 & chainingRecord_0_bits_crossWrite | _readRecord_T_881 & chainingRecord_1_bits_crossWrite | _readRecord_T_882 & chainingRecord_2_bits_crossWrite | _readRecord_T_883 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_884 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_4_gather16 =
    _readRecord_T_880 & chainingRecord_0_bits_gather16 | _readRecord_T_881 & chainingRecord_1_bits_gather16 | _readRecord_T_882 & chainingRecord_2_bits_gather16 | _readRecord_T_883 & chainingRecord_3_bits_gather16 | _readRecord_T_884
    & chainingRecord_4_bits_gather16;
  wire        readRecord_4_gather =
    _readRecord_T_880 & chainingRecord_0_bits_gather | _readRecord_T_881 & chainingRecord_1_bits_gather | _readRecord_T_882 & chainingRecord_2_bits_gather | _readRecord_T_883 & chainingRecord_3_bits_gather | _readRecord_T_884
    & chainingRecord_4_bits_gather;
  wire        readRecord_4_st =
    _readRecord_T_880 & chainingRecord_0_bits_st | _readRecord_T_881 & chainingRecord_1_bits_st | _readRecord_T_882 & chainingRecord_2_bits_st | _readRecord_T_883 & chainingRecord_3_bits_st | _readRecord_T_884 & chainingRecord_4_bits_st;
  wire        readRecord_4_ls =
    _readRecord_T_880 & chainingRecord_0_bits_ls | _readRecord_T_881 & chainingRecord_1_bits_ls | _readRecord_T_882 & chainingRecord_2_bits_ls | _readRecord_T_883 & chainingRecord_3_bits_ls | _readRecord_T_884 & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_4_instIndex =
    (_readRecord_T_880 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_881 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_882 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_883 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_884 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_4_vs2 =
    (_readRecord_T_880 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_881 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_882 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_883 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_884 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_4_vs1_bits =
    (_readRecord_T_880 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_881 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_882 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_883 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_884 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_4_vs1_valid =
    _readRecord_T_880 & chainingRecord_0_bits_vs1_valid | _readRecord_T_881 & chainingRecord_1_bits_vs1_valid | _readRecord_T_882 & chainingRecord_2_bits_vs1_valid | _readRecord_T_883 & chainingRecord_3_bits_vs1_valid | _readRecord_T_884
    & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_4_vd_bits =
    (_readRecord_T_880 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_881 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_882 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_883 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_884 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_4_vd_valid =
    _readRecord_T_880 & chainingRecord_0_bits_vd_valid | _readRecord_T_881 & chainingRecord_1_bits_vd_valid | _readRecord_T_882 & chainingRecord_2_bits_vd_valid | _readRecord_T_883 & chainingRecord_3_bits_vd_valid | _readRecord_T_884
    & chainingRecord_4_bits_vd_valid;
  wire        _readRecord_T_1100 = chainingRecord_0_bits_instIndex == readCheck_5_instructionIndex;
  wire        _readRecord_T_1101 = chainingRecord_1_bits_instIndex == readCheck_5_instructionIndex;
  wire        _readRecord_T_1102 = chainingRecord_2_bits_instIndex == readCheck_5_instructionIndex;
  wire        _readRecord_T_1103 = chainingRecord_3_bits_instIndex == readCheck_5_instructionIndex;
  wire        _readRecord_T_1104 = chainingRecord_4_bits_instIndex == readCheck_5_instructionIndex;
  wire        readRecord_5_state_wLaneClear =
    _readRecord_T_1100 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_1101 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_1102 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_1103
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_1104 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_5_state_wTopLastReport =
    _readRecord_T_1100 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_1101 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_1102 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_1103
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_1104 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_5_state_wLaneLastReport =
    _readRecord_T_1100 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_1101 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_1102 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_1103
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_1104 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_5_state_wWriteQueueClear =
    _readRecord_T_1100 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_1101 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_1102 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_1103
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_1104 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_5_state_stFinish =
    _readRecord_T_1100 & chainingRecord_0_bits_state_stFinish | _readRecord_T_1101 & chainingRecord_1_bits_state_stFinish | _readRecord_T_1102 & chainingRecord_2_bits_state_stFinish | _readRecord_T_1103
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_1104 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_5_elementMask =
    (_readRecord_T_1100 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_1101 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_1102 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_1103 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_1104 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_5_slow =
    _readRecord_T_1100 & chainingRecord_0_bits_slow | _readRecord_T_1101 & chainingRecord_1_bits_slow | _readRecord_T_1102 & chainingRecord_2_bits_slow | _readRecord_T_1103 & chainingRecord_3_bits_slow | _readRecord_T_1104
    & chainingRecord_4_bits_slow;
  wire        readRecord_5_onlyRead =
    _readRecord_T_1100 & chainingRecord_0_bits_onlyRead | _readRecord_T_1101 & chainingRecord_1_bits_onlyRead | _readRecord_T_1102 & chainingRecord_2_bits_onlyRead | _readRecord_T_1103 & chainingRecord_3_bits_onlyRead | _readRecord_T_1104
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_5_ma =
    _readRecord_T_1100 & chainingRecord_0_bits_ma | _readRecord_T_1101 & chainingRecord_1_bits_ma | _readRecord_T_1102 & chainingRecord_2_bits_ma | _readRecord_T_1103 & chainingRecord_3_bits_ma | _readRecord_T_1104
    & chainingRecord_4_bits_ma;
  wire        readRecord_5_indexType =
    _readRecord_T_1100 & chainingRecord_0_bits_indexType | _readRecord_T_1101 & chainingRecord_1_bits_indexType | _readRecord_T_1102 & chainingRecord_2_bits_indexType | _readRecord_T_1103 & chainingRecord_3_bits_indexType
    | _readRecord_T_1104 & chainingRecord_4_bits_indexType;
  wire        readRecord_5_crossRead =
    _readRecord_T_1100 & chainingRecord_0_bits_crossRead | _readRecord_T_1101 & chainingRecord_1_bits_crossRead | _readRecord_T_1102 & chainingRecord_2_bits_crossRead | _readRecord_T_1103 & chainingRecord_3_bits_crossRead
    | _readRecord_T_1104 & chainingRecord_4_bits_crossRead;
  wire        readRecord_5_crossWrite =
    _readRecord_T_1100 & chainingRecord_0_bits_crossWrite | _readRecord_T_1101 & chainingRecord_1_bits_crossWrite | _readRecord_T_1102 & chainingRecord_2_bits_crossWrite | _readRecord_T_1103 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_1104 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_5_gather16 =
    _readRecord_T_1100 & chainingRecord_0_bits_gather16 | _readRecord_T_1101 & chainingRecord_1_bits_gather16 | _readRecord_T_1102 & chainingRecord_2_bits_gather16 | _readRecord_T_1103 & chainingRecord_3_bits_gather16 | _readRecord_T_1104
    & chainingRecord_4_bits_gather16;
  wire        readRecord_5_gather =
    _readRecord_T_1100 & chainingRecord_0_bits_gather | _readRecord_T_1101 & chainingRecord_1_bits_gather | _readRecord_T_1102 & chainingRecord_2_bits_gather | _readRecord_T_1103 & chainingRecord_3_bits_gather | _readRecord_T_1104
    & chainingRecord_4_bits_gather;
  wire        readRecord_5_st =
    _readRecord_T_1100 & chainingRecord_0_bits_st | _readRecord_T_1101 & chainingRecord_1_bits_st | _readRecord_T_1102 & chainingRecord_2_bits_st | _readRecord_T_1103 & chainingRecord_3_bits_st | _readRecord_T_1104
    & chainingRecord_4_bits_st;
  wire        readRecord_5_ls =
    _readRecord_T_1100 & chainingRecord_0_bits_ls | _readRecord_T_1101 & chainingRecord_1_bits_ls | _readRecord_T_1102 & chainingRecord_2_bits_ls | _readRecord_T_1103 & chainingRecord_3_bits_ls | _readRecord_T_1104
    & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_5_instIndex =
    (_readRecord_T_1100 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_1101 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_1102 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_1103 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_1104 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_5_vs2 =
    (_readRecord_T_1100 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_1101 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_1102 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_1103 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_1104 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_5_vs1_bits =
    (_readRecord_T_1100 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_1101 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_1102 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_1103 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_1104 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_5_vs1_valid =
    _readRecord_T_1100 & chainingRecord_0_bits_vs1_valid | _readRecord_T_1101 & chainingRecord_1_bits_vs1_valid | _readRecord_T_1102 & chainingRecord_2_bits_vs1_valid | _readRecord_T_1103 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_1104 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_5_vd_bits =
    (_readRecord_T_1100 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_1101 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_1102 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_1103 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_1104 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_5_vd_valid =
    _readRecord_T_1100 & chainingRecord_0_bits_vd_valid | _readRecord_T_1101 & chainingRecord_1_bits_vd_valid | _readRecord_T_1102 & chainingRecord_2_bits_vd_valid | _readRecord_T_1103 & chainingRecord_3_bits_vd_valid | _readRecord_T_1104
    & chainingRecord_4_bits_vd_valid;
  wire        _readRecord_T_1320 = chainingRecord_0_bits_instIndex == readCheck_6_instructionIndex;
  wire        _readRecord_T_1321 = chainingRecord_1_bits_instIndex == readCheck_6_instructionIndex;
  wire        _readRecord_T_1322 = chainingRecord_2_bits_instIndex == readCheck_6_instructionIndex;
  wire        _readRecord_T_1323 = chainingRecord_3_bits_instIndex == readCheck_6_instructionIndex;
  wire        _readRecord_T_1324 = chainingRecord_4_bits_instIndex == readCheck_6_instructionIndex;
  wire        readRecord_6_state_wLaneClear =
    _readRecord_T_1320 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_1321 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_1322 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_1323
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_1324 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_6_state_wTopLastReport =
    _readRecord_T_1320 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_1321 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_1322 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_1323
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_1324 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_6_state_wLaneLastReport =
    _readRecord_T_1320 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_1321 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_1322 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_1323
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_1324 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_6_state_wWriteQueueClear =
    _readRecord_T_1320 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_1321 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_1322 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_1323
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_1324 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_6_state_stFinish =
    _readRecord_T_1320 & chainingRecord_0_bits_state_stFinish | _readRecord_T_1321 & chainingRecord_1_bits_state_stFinish | _readRecord_T_1322 & chainingRecord_2_bits_state_stFinish | _readRecord_T_1323
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_1324 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_6_elementMask =
    (_readRecord_T_1320 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_1321 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_1322 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_1323 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_1324 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_6_slow =
    _readRecord_T_1320 & chainingRecord_0_bits_slow | _readRecord_T_1321 & chainingRecord_1_bits_slow | _readRecord_T_1322 & chainingRecord_2_bits_slow | _readRecord_T_1323 & chainingRecord_3_bits_slow | _readRecord_T_1324
    & chainingRecord_4_bits_slow;
  wire        readRecord_6_onlyRead =
    _readRecord_T_1320 & chainingRecord_0_bits_onlyRead | _readRecord_T_1321 & chainingRecord_1_bits_onlyRead | _readRecord_T_1322 & chainingRecord_2_bits_onlyRead | _readRecord_T_1323 & chainingRecord_3_bits_onlyRead | _readRecord_T_1324
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_6_ma =
    _readRecord_T_1320 & chainingRecord_0_bits_ma | _readRecord_T_1321 & chainingRecord_1_bits_ma | _readRecord_T_1322 & chainingRecord_2_bits_ma | _readRecord_T_1323 & chainingRecord_3_bits_ma | _readRecord_T_1324
    & chainingRecord_4_bits_ma;
  wire        readRecord_6_indexType =
    _readRecord_T_1320 & chainingRecord_0_bits_indexType | _readRecord_T_1321 & chainingRecord_1_bits_indexType | _readRecord_T_1322 & chainingRecord_2_bits_indexType | _readRecord_T_1323 & chainingRecord_3_bits_indexType
    | _readRecord_T_1324 & chainingRecord_4_bits_indexType;
  wire        readRecord_6_crossRead =
    _readRecord_T_1320 & chainingRecord_0_bits_crossRead | _readRecord_T_1321 & chainingRecord_1_bits_crossRead | _readRecord_T_1322 & chainingRecord_2_bits_crossRead | _readRecord_T_1323 & chainingRecord_3_bits_crossRead
    | _readRecord_T_1324 & chainingRecord_4_bits_crossRead;
  wire        readRecord_6_crossWrite =
    _readRecord_T_1320 & chainingRecord_0_bits_crossWrite | _readRecord_T_1321 & chainingRecord_1_bits_crossWrite | _readRecord_T_1322 & chainingRecord_2_bits_crossWrite | _readRecord_T_1323 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_1324 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_6_gather16 =
    _readRecord_T_1320 & chainingRecord_0_bits_gather16 | _readRecord_T_1321 & chainingRecord_1_bits_gather16 | _readRecord_T_1322 & chainingRecord_2_bits_gather16 | _readRecord_T_1323 & chainingRecord_3_bits_gather16 | _readRecord_T_1324
    & chainingRecord_4_bits_gather16;
  wire        readRecord_6_gather =
    _readRecord_T_1320 & chainingRecord_0_bits_gather | _readRecord_T_1321 & chainingRecord_1_bits_gather | _readRecord_T_1322 & chainingRecord_2_bits_gather | _readRecord_T_1323 & chainingRecord_3_bits_gather | _readRecord_T_1324
    & chainingRecord_4_bits_gather;
  wire        readRecord_6_st =
    _readRecord_T_1320 & chainingRecord_0_bits_st | _readRecord_T_1321 & chainingRecord_1_bits_st | _readRecord_T_1322 & chainingRecord_2_bits_st | _readRecord_T_1323 & chainingRecord_3_bits_st | _readRecord_T_1324
    & chainingRecord_4_bits_st;
  wire        readRecord_6_ls =
    _readRecord_T_1320 & chainingRecord_0_bits_ls | _readRecord_T_1321 & chainingRecord_1_bits_ls | _readRecord_T_1322 & chainingRecord_2_bits_ls | _readRecord_T_1323 & chainingRecord_3_bits_ls | _readRecord_T_1324
    & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_6_instIndex =
    (_readRecord_T_1320 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_1321 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_1322 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_1323 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_1324 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_6_vs2 =
    (_readRecord_T_1320 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_1321 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_1322 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_1323 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_1324 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_6_vs1_bits =
    (_readRecord_T_1320 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_1321 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_1322 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_1323 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_1324 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_6_vs1_valid =
    _readRecord_T_1320 & chainingRecord_0_bits_vs1_valid | _readRecord_T_1321 & chainingRecord_1_bits_vs1_valid | _readRecord_T_1322 & chainingRecord_2_bits_vs1_valid | _readRecord_T_1323 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_1324 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_6_vd_bits =
    (_readRecord_T_1320 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_1321 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_1322 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_1323 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_1324 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_6_vd_valid =
    _readRecord_T_1320 & chainingRecord_0_bits_vd_valid | _readRecord_T_1321 & chainingRecord_1_bits_vd_valid | _readRecord_T_1322 & chainingRecord_2_bits_vd_valid | _readRecord_T_1323 & chainingRecord_3_bits_vd_valid | _readRecord_T_1324
    & chainingRecord_4_bits_vd_valid;
  wire        _readRecord_T_1540 = chainingRecord_0_bits_instIndex == readCheck_7_instructionIndex;
  wire        _readRecord_T_1541 = chainingRecord_1_bits_instIndex == readCheck_7_instructionIndex;
  wire        _readRecord_T_1542 = chainingRecord_2_bits_instIndex == readCheck_7_instructionIndex;
  wire        _readRecord_T_1543 = chainingRecord_3_bits_instIndex == readCheck_7_instructionIndex;
  wire        _readRecord_T_1544 = chainingRecord_4_bits_instIndex == readCheck_7_instructionIndex;
  wire        readRecord_7_state_wLaneClear =
    _readRecord_T_1540 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_1541 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_1542 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_1543
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_1544 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_7_state_wTopLastReport =
    _readRecord_T_1540 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_1541 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_1542 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_1543
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_1544 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_7_state_wLaneLastReport =
    _readRecord_T_1540 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_1541 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_1542 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_1543
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_1544 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_7_state_wWriteQueueClear =
    _readRecord_T_1540 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_1541 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_1542 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_1543
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_1544 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_7_state_stFinish =
    _readRecord_T_1540 & chainingRecord_0_bits_state_stFinish | _readRecord_T_1541 & chainingRecord_1_bits_state_stFinish | _readRecord_T_1542 & chainingRecord_2_bits_state_stFinish | _readRecord_T_1543
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_1544 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_7_elementMask =
    (_readRecord_T_1540 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_1541 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_1542 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_1543 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_1544 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_7_slow =
    _readRecord_T_1540 & chainingRecord_0_bits_slow | _readRecord_T_1541 & chainingRecord_1_bits_slow | _readRecord_T_1542 & chainingRecord_2_bits_slow | _readRecord_T_1543 & chainingRecord_3_bits_slow | _readRecord_T_1544
    & chainingRecord_4_bits_slow;
  wire        readRecord_7_onlyRead =
    _readRecord_T_1540 & chainingRecord_0_bits_onlyRead | _readRecord_T_1541 & chainingRecord_1_bits_onlyRead | _readRecord_T_1542 & chainingRecord_2_bits_onlyRead | _readRecord_T_1543 & chainingRecord_3_bits_onlyRead | _readRecord_T_1544
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_7_ma =
    _readRecord_T_1540 & chainingRecord_0_bits_ma | _readRecord_T_1541 & chainingRecord_1_bits_ma | _readRecord_T_1542 & chainingRecord_2_bits_ma | _readRecord_T_1543 & chainingRecord_3_bits_ma | _readRecord_T_1544
    & chainingRecord_4_bits_ma;
  wire        readRecord_7_indexType =
    _readRecord_T_1540 & chainingRecord_0_bits_indexType | _readRecord_T_1541 & chainingRecord_1_bits_indexType | _readRecord_T_1542 & chainingRecord_2_bits_indexType | _readRecord_T_1543 & chainingRecord_3_bits_indexType
    | _readRecord_T_1544 & chainingRecord_4_bits_indexType;
  wire        readRecord_7_crossRead =
    _readRecord_T_1540 & chainingRecord_0_bits_crossRead | _readRecord_T_1541 & chainingRecord_1_bits_crossRead | _readRecord_T_1542 & chainingRecord_2_bits_crossRead | _readRecord_T_1543 & chainingRecord_3_bits_crossRead
    | _readRecord_T_1544 & chainingRecord_4_bits_crossRead;
  wire        readRecord_7_crossWrite =
    _readRecord_T_1540 & chainingRecord_0_bits_crossWrite | _readRecord_T_1541 & chainingRecord_1_bits_crossWrite | _readRecord_T_1542 & chainingRecord_2_bits_crossWrite | _readRecord_T_1543 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_1544 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_7_gather16 =
    _readRecord_T_1540 & chainingRecord_0_bits_gather16 | _readRecord_T_1541 & chainingRecord_1_bits_gather16 | _readRecord_T_1542 & chainingRecord_2_bits_gather16 | _readRecord_T_1543 & chainingRecord_3_bits_gather16 | _readRecord_T_1544
    & chainingRecord_4_bits_gather16;
  wire        readRecord_7_gather =
    _readRecord_T_1540 & chainingRecord_0_bits_gather | _readRecord_T_1541 & chainingRecord_1_bits_gather | _readRecord_T_1542 & chainingRecord_2_bits_gather | _readRecord_T_1543 & chainingRecord_3_bits_gather | _readRecord_T_1544
    & chainingRecord_4_bits_gather;
  wire        readRecord_7_st =
    _readRecord_T_1540 & chainingRecord_0_bits_st | _readRecord_T_1541 & chainingRecord_1_bits_st | _readRecord_T_1542 & chainingRecord_2_bits_st | _readRecord_T_1543 & chainingRecord_3_bits_st | _readRecord_T_1544
    & chainingRecord_4_bits_st;
  wire        readRecord_7_ls =
    _readRecord_T_1540 & chainingRecord_0_bits_ls | _readRecord_T_1541 & chainingRecord_1_bits_ls | _readRecord_T_1542 & chainingRecord_2_bits_ls | _readRecord_T_1543 & chainingRecord_3_bits_ls | _readRecord_T_1544
    & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_7_instIndex =
    (_readRecord_T_1540 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_1541 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_1542 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_1543 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_1544 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_7_vs2 =
    (_readRecord_T_1540 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_1541 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_1542 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_1543 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_1544 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_7_vs1_bits =
    (_readRecord_T_1540 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_1541 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_1542 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_1543 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_1544 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_7_vs1_valid =
    _readRecord_T_1540 & chainingRecord_0_bits_vs1_valid | _readRecord_T_1541 & chainingRecord_1_bits_vs1_valid | _readRecord_T_1542 & chainingRecord_2_bits_vs1_valid | _readRecord_T_1543 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_1544 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_7_vd_bits =
    (_readRecord_T_1540 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_1541 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_1542 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_1543 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_1544 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_7_vd_valid =
    _readRecord_T_1540 & chainingRecord_0_bits_vd_valid | _readRecord_T_1541 & chainingRecord_1_bits_vd_valid | _readRecord_T_1542 & chainingRecord_2_bits_vd_valid | _readRecord_T_1543 & chainingRecord_3_bits_vd_valid | _readRecord_T_1544
    & chainingRecord_4_bits_vd_valid;
  wire        _readRecord_T_1760 = chainingRecord_0_bits_instIndex == readCheck_8_instructionIndex;
  wire        _readRecord_T_1761 = chainingRecord_1_bits_instIndex == readCheck_8_instructionIndex;
  wire        _readRecord_T_1762 = chainingRecord_2_bits_instIndex == readCheck_8_instructionIndex;
  wire        _readRecord_T_1763 = chainingRecord_3_bits_instIndex == readCheck_8_instructionIndex;
  wire        _readRecord_T_1764 = chainingRecord_4_bits_instIndex == readCheck_8_instructionIndex;
  wire        readRecord_8_state_wLaneClear =
    _readRecord_T_1760 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_1761 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_1762 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_1763
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_1764 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_8_state_wTopLastReport =
    _readRecord_T_1760 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_1761 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_1762 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_1763
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_1764 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_8_state_wLaneLastReport =
    _readRecord_T_1760 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_1761 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_1762 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_1763
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_1764 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_8_state_wWriteQueueClear =
    _readRecord_T_1760 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_1761 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_1762 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_1763
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_1764 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_8_state_stFinish =
    _readRecord_T_1760 & chainingRecord_0_bits_state_stFinish | _readRecord_T_1761 & chainingRecord_1_bits_state_stFinish | _readRecord_T_1762 & chainingRecord_2_bits_state_stFinish | _readRecord_T_1763
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_1764 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_8_elementMask =
    (_readRecord_T_1760 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_1761 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_1762 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_1763 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_1764 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_8_slow =
    _readRecord_T_1760 & chainingRecord_0_bits_slow | _readRecord_T_1761 & chainingRecord_1_bits_slow | _readRecord_T_1762 & chainingRecord_2_bits_slow | _readRecord_T_1763 & chainingRecord_3_bits_slow | _readRecord_T_1764
    & chainingRecord_4_bits_slow;
  wire        readRecord_8_onlyRead =
    _readRecord_T_1760 & chainingRecord_0_bits_onlyRead | _readRecord_T_1761 & chainingRecord_1_bits_onlyRead | _readRecord_T_1762 & chainingRecord_2_bits_onlyRead | _readRecord_T_1763 & chainingRecord_3_bits_onlyRead | _readRecord_T_1764
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_8_ma =
    _readRecord_T_1760 & chainingRecord_0_bits_ma | _readRecord_T_1761 & chainingRecord_1_bits_ma | _readRecord_T_1762 & chainingRecord_2_bits_ma | _readRecord_T_1763 & chainingRecord_3_bits_ma | _readRecord_T_1764
    & chainingRecord_4_bits_ma;
  wire        readRecord_8_indexType =
    _readRecord_T_1760 & chainingRecord_0_bits_indexType | _readRecord_T_1761 & chainingRecord_1_bits_indexType | _readRecord_T_1762 & chainingRecord_2_bits_indexType | _readRecord_T_1763 & chainingRecord_3_bits_indexType
    | _readRecord_T_1764 & chainingRecord_4_bits_indexType;
  wire        readRecord_8_crossRead =
    _readRecord_T_1760 & chainingRecord_0_bits_crossRead | _readRecord_T_1761 & chainingRecord_1_bits_crossRead | _readRecord_T_1762 & chainingRecord_2_bits_crossRead | _readRecord_T_1763 & chainingRecord_3_bits_crossRead
    | _readRecord_T_1764 & chainingRecord_4_bits_crossRead;
  wire        readRecord_8_crossWrite =
    _readRecord_T_1760 & chainingRecord_0_bits_crossWrite | _readRecord_T_1761 & chainingRecord_1_bits_crossWrite | _readRecord_T_1762 & chainingRecord_2_bits_crossWrite | _readRecord_T_1763 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_1764 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_8_gather16 =
    _readRecord_T_1760 & chainingRecord_0_bits_gather16 | _readRecord_T_1761 & chainingRecord_1_bits_gather16 | _readRecord_T_1762 & chainingRecord_2_bits_gather16 | _readRecord_T_1763 & chainingRecord_3_bits_gather16 | _readRecord_T_1764
    & chainingRecord_4_bits_gather16;
  wire        readRecord_8_gather =
    _readRecord_T_1760 & chainingRecord_0_bits_gather | _readRecord_T_1761 & chainingRecord_1_bits_gather | _readRecord_T_1762 & chainingRecord_2_bits_gather | _readRecord_T_1763 & chainingRecord_3_bits_gather | _readRecord_T_1764
    & chainingRecord_4_bits_gather;
  wire        readRecord_8_st =
    _readRecord_T_1760 & chainingRecord_0_bits_st | _readRecord_T_1761 & chainingRecord_1_bits_st | _readRecord_T_1762 & chainingRecord_2_bits_st | _readRecord_T_1763 & chainingRecord_3_bits_st | _readRecord_T_1764
    & chainingRecord_4_bits_st;
  wire        readRecord_8_ls =
    _readRecord_T_1760 & chainingRecord_0_bits_ls | _readRecord_T_1761 & chainingRecord_1_bits_ls | _readRecord_T_1762 & chainingRecord_2_bits_ls | _readRecord_T_1763 & chainingRecord_3_bits_ls | _readRecord_T_1764
    & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_8_instIndex =
    (_readRecord_T_1760 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_1761 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_1762 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_1763 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_1764 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_8_vs2 =
    (_readRecord_T_1760 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_1761 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_1762 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_1763 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_1764 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_8_vs1_bits =
    (_readRecord_T_1760 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_1761 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_1762 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_1763 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_1764 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_8_vs1_valid =
    _readRecord_T_1760 & chainingRecord_0_bits_vs1_valid | _readRecord_T_1761 & chainingRecord_1_bits_vs1_valid | _readRecord_T_1762 & chainingRecord_2_bits_vs1_valid | _readRecord_T_1763 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_1764 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_8_vd_bits =
    (_readRecord_T_1760 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_1761 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_1762 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_1763 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_1764 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_8_vd_valid =
    _readRecord_T_1760 & chainingRecord_0_bits_vd_valid | _readRecord_T_1761 & chainingRecord_1_bits_vd_valid | _readRecord_T_1762 & chainingRecord_2_bits_vd_valid | _readRecord_T_1763 & chainingRecord_3_bits_vd_valid | _readRecord_T_1764
    & chainingRecord_4_bits_vd_valid;
  wire        _readRecord_T_1980 = chainingRecord_0_bits_instIndex == readCheck_9_instructionIndex;
  wire        _readRecord_T_1981 = chainingRecord_1_bits_instIndex == readCheck_9_instructionIndex;
  wire        _readRecord_T_1982 = chainingRecord_2_bits_instIndex == readCheck_9_instructionIndex;
  wire        _readRecord_T_1983 = chainingRecord_3_bits_instIndex == readCheck_9_instructionIndex;
  wire        _readRecord_T_1984 = chainingRecord_4_bits_instIndex == readCheck_9_instructionIndex;
  wire        readRecord_9_state_wLaneClear =
    _readRecord_T_1980 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_1981 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_1982 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_1983
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_1984 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_9_state_wTopLastReport =
    _readRecord_T_1980 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_1981 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_1982 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_1983
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_1984 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_9_state_wLaneLastReport =
    _readRecord_T_1980 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_1981 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_1982 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_1983
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_1984 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_9_state_wWriteQueueClear =
    _readRecord_T_1980 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_1981 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_1982 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_1983
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_1984 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_9_state_stFinish =
    _readRecord_T_1980 & chainingRecord_0_bits_state_stFinish | _readRecord_T_1981 & chainingRecord_1_bits_state_stFinish | _readRecord_T_1982 & chainingRecord_2_bits_state_stFinish | _readRecord_T_1983
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_1984 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_9_elementMask =
    (_readRecord_T_1980 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_1981 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_1982 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_1983 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_1984 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_9_slow =
    _readRecord_T_1980 & chainingRecord_0_bits_slow | _readRecord_T_1981 & chainingRecord_1_bits_slow | _readRecord_T_1982 & chainingRecord_2_bits_slow | _readRecord_T_1983 & chainingRecord_3_bits_slow | _readRecord_T_1984
    & chainingRecord_4_bits_slow;
  wire        readRecord_9_onlyRead =
    _readRecord_T_1980 & chainingRecord_0_bits_onlyRead | _readRecord_T_1981 & chainingRecord_1_bits_onlyRead | _readRecord_T_1982 & chainingRecord_2_bits_onlyRead | _readRecord_T_1983 & chainingRecord_3_bits_onlyRead | _readRecord_T_1984
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_9_ma =
    _readRecord_T_1980 & chainingRecord_0_bits_ma | _readRecord_T_1981 & chainingRecord_1_bits_ma | _readRecord_T_1982 & chainingRecord_2_bits_ma | _readRecord_T_1983 & chainingRecord_3_bits_ma | _readRecord_T_1984
    & chainingRecord_4_bits_ma;
  wire        readRecord_9_indexType =
    _readRecord_T_1980 & chainingRecord_0_bits_indexType | _readRecord_T_1981 & chainingRecord_1_bits_indexType | _readRecord_T_1982 & chainingRecord_2_bits_indexType | _readRecord_T_1983 & chainingRecord_3_bits_indexType
    | _readRecord_T_1984 & chainingRecord_4_bits_indexType;
  wire        readRecord_9_crossRead =
    _readRecord_T_1980 & chainingRecord_0_bits_crossRead | _readRecord_T_1981 & chainingRecord_1_bits_crossRead | _readRecord_T_1982 & chainingRecord_2_bits_crossRead | _readRecord_T_1983 & chainingRecord_3_bits_crossRead
    | _readRecord_T_1984 & chainingRecord_4_bits_crossRead;
  wire        readRecord_9_crossWrite =
    _readRecord_T_1980 & chainingRecord_0_bits_crossWrite | _readRecord_T_1981 & chainingRecord_1_bits_crossWrite | _readRecord_T_1982 & chainingRecord_2_bits_crossWrite | _readRecord_T_1983 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_1984 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_9_gather16 =
    _readRecord_T_1980 & chainingRecord_0_bits_gather16 | _readRecord_T_1981 & chainingRecord_1_bits_gather16 | _readRecord_T_1982 & chainingRecord_2_bits_gather16 | _readRecord_T_1983 & chainingRecord_3_bits_gather16 | _readRecord_T_1984
    & chainingRecord_4_bits_gather16;
  wire        readRecord_9_gather =
    _readRecord_T_1980 & chainingRecord_0_bits_gather | _readRecord_T_1981 & chainingRecord_1_bits_gather | _readRecord_T_1982 & chainingRecord_2_bits_gather | _readRecord_T_1983 & chainingRecord_3_bits_gather | _readRecord_T_1984
    & chainingRecord_4_bits_gather;
  wire        readRecord_9_st =
    _readRecord_T_1980 & chainingRecord_0_bits_st | _readRecord_T_1981 & chainingRecord_1_bits_st | _readRecord_T_1982 & chainingRecord_2_bits_st | _readRecord_T_1983 & chainingRecord_3_bits_st | _readRecord_T_1984
    & chainingRecord_4_bits_st;
  wire        readRecord_9_ls =
    _readRecord_T_1980 & chainingRecord_0_bits_ls | _readRecord_T_1981 & chainingRecord_1_bits_ls | _readRecord_T_1982 & chainingRecord_2_bits_ls | _readRecord_T_1983 & chainingRecord_3_bits_ls | _readRecord_T_1984
    & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_9_instIndex =
    (_readRecord_T_1980 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_1981 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_1982 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_1983 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_1984 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_9_vs2 =
    (_readRecord_T_1980 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_1981 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_1982 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_1983 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_1984 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_9_vs1_bits =
    (_readRecord_T_1980 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_1981 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_1982 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_1983 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_1984 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_9_vs1_valid =
    _readRecord_T_1980 & chainingRecord_0_bits_vs1_valid | _readRecord_T_1981 & chainingRecord_1_bits_vs1_valid | _readRecord_T_1982 & chainingRecord_2_bits_vs1_valid | _readRecord_T_1983 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_1984 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_9_vd_bits =
    (_readRecord_T_1980 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_1981 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_1982 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_1983 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_1984 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_9_vd_valid =
    _readRecord_T_1980 & chainingRecord_0_bits_vd_valid | _readRecord_T_1981 & chainingRecord_1_bits_vd_valid | _readRecord_T_1982 & chainingRecord_2_bits_vd_valid | _readRecord_T_1983 & chainingRecord_3_bits_vd_valid | _readRecord_T_1984
    & chainingRecord_4_bits_vd_valid;
  wire        _readRecord_T_2200 = chainingRecord_0_bits_instIndex == readCheck_10_instructionIndex;
  wire        _readRecord_T_2201 = chainingRecord_1_bits_instIndex == readCheck_10_instructionIndex;
  wire        _readRecord_T_2202 = chainingRecord_2_bits_instIndex == readCheck_10_instructionIndex;
  wire        _readRecord_T_2203 = chainingRecord_3_bits_instIndex == readCheck_10_instructionIndex;
  wire        _readRecord_T_2204 = chainingRecord_4_bits_instIndex == readCheck_10_instructionIndex;
  wire        readRecord_10_state_wLaneClear =
    _readRecord_T_2200 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_2201 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_2202 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_2203
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_2204 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_10_state_wTopLastReport =
    _readRecord_T_2200 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_2201 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_2202 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_2203
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_2204 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_10_state_wLaneLastReport =
    _readRecord_T_2200 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_2201 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_2202 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_2203
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_2204 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_10_state_wWriteQueueClear =
    _readRecord_T_2200 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_2201 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_2202 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_2203
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_2204 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_10_state_stFinish =
    _readRecord_T_2200 & chainingRecord_0_bits_state_stFinish | _readRecord_T_2201 & chainingRecord_1_bits_state_stFinish | _readRecord_T_2202 & chainingRecord_2_bits_state_stFinish | _readRecord_T_2203
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_2204 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_10_elementMask =
    (_readRecord_T_2200 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_2201 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_2202 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_2203 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_2204 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_10_slow =
    _readRecord_T_2200 & chainingRecord_0_bits_slow | _readRecord_T_2201 & chainingRecord_1_bits_slow | _readRecord_T_2202 & chainingRecord_2_bits_slow | _readRecord_T_2203 & chainingRecord_3_bits_slow | _readRecord_T_2204
    & chainingRecord_4_bits_slow;
  wire        readRecord_10_onlyRead =
    _readRecord_T_2200 & chainingRecord_0_bits_onlyRead | _readRecord_T_2201 & chainingRecord_1_bits_onlyRead | _readRecord_T_2202 & chainingRecord_2_bits_onlyRead | _readRecord_T_2203 & chainingRecord_3_bits_onlyRead | _readRecord_T_2204
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_10_ma =
    _readRecord_T_2200 & chainingRecord_0_bits_ma | _readRecord_T_2201 & chainingRecord_1_bits_ma | _readRecord_T_2202 & chainingRecord_2_bits_ma | _readRecord_T_2203 & chainingRecord_3_bits_ma | _readRecord_T_2204
    & chainingRecord_4_bits_ma;
  wire        readRecord_10_indexType =
    _readRecord_T_2200 & chainingRecord_0_bits_indexType | _readRecord_T_2201 & chainingRecord_1_bits_indexType | _readRecord_T_2202 & chainingRecord_2_bits_indexType | _readRecord_T_2203 & chainingRecord_3_bits_indexType
    | _readRecord_T_2204 & chainingRecord_4_bits_indexType;
  wire        readRecord_10_crossRead =
    _readRecord_T_2200 & chainingRecord_0_bits_crossRead | _readRecord_T_2201 & chainingRecord_1_bits_crossRead | _readRecord_T_2202 & chainingRecord_2_bits_crossRead | _readRecord_T_2203 & chainingRecord_3_bits_crossRead
    | _readRecord_T_2204 & chainingRecord_4_bits_crossRead;
  wire        readRecord_10_crossWrite =
    _readRecord_T_2200 & chainingRecord_0_bits_crossWrite | _readRecord_T_2201 & chainingRecord_1_bits_crossWrite | _readRecord_T_2202 & chainingRecord_2_bits_crossWrite | _readRecord_T_2203 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_2204 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_10_gather16 =
    _readRecord_T_2200 & chainingRecord_0_bits_gather16 | _readRecord_T_2201 & chainingRecord_1_bits_gather16 | _readRecord_T_2202 & chainingRecord_2_bits_gather16 | _readRecord_T_2203 & chainingRecord_3_bits_gather16 | _readRecord_T_2204
    & chainingRecord_4_bits_gather16;
  wire        readRecord_10_gather =
    _readRecord_T_2200 & chainingRecord_0_bits_gather | _readRecord_T_2201 & chainingRecord_1_bits_gather | _readRecord_T_2202 & chainingRecord_2_bits_gather | _readRecord_T_2203 & chainingRecord_3_bits_gather | _readRecord_T_2204
    & chainingRecord_4_bits_gather;
  wire        readRecord_10_st =
    _readRecord_T_2200 & chainingRecord_0_bits_st | _readRecord_T_2201 & chainingRecord_1_bits_st | _readRecord_T_2202 & chainingRecord_2_bits_st | _readRecord_T_2203 & chainingRecord_3_bits_st | _readRecord_T_2204
    & chainingRecord_4_bits_st;
  wire        readRecord_10_ls =
    _readRecord_T_2200 & chainingRecord_0_bits_ls | _readRecord_T_2201 & chainingRecord_1_bits_ls | _readRecord_T_2202 & chainingRecord_2_bits_ls | _readRecord_T_2203 & chainingRecord_3_bits_ls | _readRecord_T_2204
    & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_10_instIndex =
    (_readRecord_T_2200 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_2201 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_2202 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_2203 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_2204 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_10_vs2 =
    (_readRecord_T_2200 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_2201 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_2202 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_2203 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_2204 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_10_vs1_bits =
    (_readRecord_T_2200 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_2201 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_2202 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_2203 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_2204 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_10_vs1_valid =
    _readRecord_T_2200 & chainingRecord_0_bits_vs1_valid | _readRecord_T_2201 & chainingRecord_1_bits_vs1_valid | _readRecord_T_2202 & chainingRecord_2_bits_vs1_valid | _readRecord_T_2203 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_2204 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_10_vd_bits =
    (_readRecord_T_2200 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_2201 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_2202 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_2203 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_2204 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_10_vd_valid =
    _readRecord_T_2200 & chainingRecord_0_bits_vd_valid | _readRecord_T_2201 & chainingRecord_1_bits_vd_valid | _readRecord_T_2202 & chainingRecord_2_bits_vd_valid | _readRecord_T_2203 & chainingRecord_3_bits_vd_valid | _readRecord_T_2204
    & chainingRecord_4_bits_vd_valid;
  wire        _readRecord_T_2420 = chainingRecord_0_bits_instIndex == readCheck_11_instructionIndex;
  wire        _readRecord_T_2421 = chainingRecord_1_bits_instIndex == readCheck_11_instructionIndex;
  wire        _readRecord_T_2422 = chainingRecord_2_bits_instIndex == readCheck_11_instructionIndex;
  wire        _readRecord_T_2423 = chainingRecord_3_bits_instIndex == readCheck_11_instructionIndex;
  wire        _readRecord_T_2424 = chainingRecord_4_bits_instIndex == readCheck_11_instructionIndex;
  wire        readRecord_11_state_wLaneClear =
    _readRecord_T_2420 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_2421 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_2422 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_2423
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_2424 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_11_state_wTopLastReport =
    _readRecord_T_2420 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_2421 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_2422 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_2423
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_2424 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_11_state_wLaneLastReport =
    _readRecord_T_2420 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_2421 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_2422 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_2423
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_2424 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_11_state_wWriteQueueClear =
    _readRecord_T_2420 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_2421 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_2422 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_2423
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_2424 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_11_state_stFinish =
    _readRecord_T_2420 & chainingRecord_0_bits_state_stFinish | _readRecord_T_2421 & chainingRecord_1_bits_state_stFinish | _readRecord_T_2422 & chainingRecord_2_bits_state_stFinish | _readRecord_T_2423
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_2424 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_11_elementMask =
    (_readRecord_T_2420 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_2421 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_2422 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_2423 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_2424 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_11_slow =
    _readRecord_T_2420 & chainingRecord_0_bits_slow | _readRecord_T_2421 & chainingRecord_1_bits_slow | _readRecord_T_2422 & chainingRecord_2_bits_slow | _readRecord_T_2423 & chainingRecord_3_bits_slow | _readRecord_T_2424
    & chainingRecord_4_bits_slow;
  wire        readRecord_11_onlyRead =
    _readRecord_T_2420 & chainingRecord_0_bits_onlyRead | _readRecord_T_2421 & chainingRecord_1_bits_onlyRead | _readRecord_T_2422 & chainingRecord_2_bits_onlyRead | _readRecord_T_2423 & chainingRecord_3_bits_onlyRead | _readRecord_T_2424
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_11_ma =
    _readRecord_T_2420 & chainingRecord_0_bits_ma | _readRecord_T_2421 & chainingRecord_1_bits_ma | _readRecord_T_2422 & chainingRecord_2_bits_ma | _readRecord_T_2423 & chainingRecord_3_bits_ma | _readRecord_T_2424
    & chainingRecord_4_bits_ma;
  wire        readRecord_11_indexType =
    _readRecord_T_2420 & chainingRecord_0_bits_indexType | _readRecord_T_2421 & chainingRecord_1_bits_indexType | _readRecord_T_2422 & chainingRecord_2_bits_indexType | _readRecord_T_2423 & chainingRecord_3_bits_indexType
    | _readRecord_T_2424 & chainingRecord_4_bits_indexType;
  wire        readRecord_11_crossRead =
    _readRecord_T_2420 & chainingRecord_0_bits_crossRead | _readRecord_T_2421 & chainingRecord_1_bits_crossRead | _readRecord_T_2422 & chainingRecord_2_bits_crossRead | _readRecord_T_2423 & chainingRecord_3_bits_crossRead
    | _readRecord_T_2424 & chainingRecord_4_bits_crossRead;
  wire        readRecord_11_crossWrite =
    _readRecord_T_2420 & chainingRecord_0_bits_crossWrite | _readRecord_T_2421 & chainingRecord_1_bits_crossWrite | _readRecord_T_2422 & chainingRecord_2_bits_crossWrite | _readRecord_T_2423 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_2424 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_11_gather16 =
    _readRecord_T_2420 & chainingRecord_0_bits_gather16 | _readRecord_T_2421 & chainingRecord_1_bits_gather16 | _readRecord_T_2422 & chainingRecord_2_bits_gather16 | _readRecord_T_2423 & chainingRecord_3_bits_gather16 | _readRecord_T_2424
    & chainingRecord_4_bits_gather16;
  wire        readRecord_11_gather =
    _readRecord_T_2420 & chainingRecord_0_bits_gather | _readRecord_T_2421 & chainingRecord_1_bits_gather | _readRecord_T_2422 & chainingRecord_2_bits_gather | _readRecord_T_2423 & chainingRecord_3_bits_gather | _readRecord_T_2424
    & chainingRecord_4_bits_gather;
  wire        readRecord_11_st =
    _readRecord_T_2420 & chainingRecord_0_bits_st | _readRecord_T_2421 & chainingRecord_1_bits_st | _readRecord_T_2422 & chainingRecord_2_bits_st | _readRecord_T_2423 & chainingRecord_3_bits_st | _readRecord_T_2424
    & chainingRecord_4_bits_st;
  wire        readRecord_11_ls =
    _readRecord_T_2420 & chainingRecord_0_bits_ls | _readRecord_T_2421 & chainingRecord_1_bits_ls | _readRecord_T_2422 & chainingRecord_2_bits_ls | _readRecord_T_2423 & chainingRecord_3_bits_ls | _readRecord_T_2424
    & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_11_instIndex =
    (_readRecord_T_2420 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_2421 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_2422 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_2423 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_2424 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_11_vs2 =
    (_readRecord_T_2420 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_2421 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_2422 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_2423 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_2424 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_11_vs1_bits =
    (_readRecord_T_2420 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_2421 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_2422 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_2423 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_2424 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_11_vs1_valid =
    _readRecord_T_2420 & chainingRecord_0_bits_vs1_valid | _readRecord_T_2421 & chainingRecord_1_bits_vs1_valid | _readRecord_T_2422 & chainingRecord_2_bits_vs1_valid | _readRecord_T_2423 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_2424 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_11_vd_bits =
    (_readRecord_T_2420 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_2421 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_2422 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_2423 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_2424 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_11_vd_valid =
    _readRecord_T_2420 & chainingRecord_0_bits_vd_valid | _readRecord_T_2421 & chainingRecord_1_bits_vd_valid | _readRecord_T_2422 & chainingRecord_2_bits_vd_valid | _readRecord_T_2423 & chainingRecord_3_bits_vd_valid | _readRecord_T_2424
    & chainingRecord_4_bits_vd_valid;
  wire        _readRecord_T_2640 = chainingRecord_0_bits_instIndex == readCheck_12_instructionIndex;
  wire        _readRecord_T_2641 = chainingRecord_1_bits_instIndex == readCheck_12_instructionIndex;
  wire        _readRecord_T_2642 = chainingRecord_2_bits_instIndex == readCheck_12_instructionIndex;
  wire        _readRecord_T_2643 = chainingRecord_3_bits_instIndex == readCheck_12_instructionIndex;
  wire        _readRecord_T_2644 = chainingRecord_4_bits_instIndex == readCheck_12_instructionIndex;
  wire        readRecord_12_state_wLaneClear =
    _readRecord_T_2640 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_2641 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_2642 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_2643
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_2644 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_12_state_wTopLastReport =
    _readRecord_T_2640 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_2641 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_2642 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_2643
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_2644 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_12_state_wLaneLastReport =
    _readRecord_T_2640 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_2641 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_2642 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_2643
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_2644 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_12_state_wWriteQueueClear =
    _readRecord_T_2640 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_2641 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_2642 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_2643
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_2644 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_12_state_stFinish =
    _readRecord_T_2640 & chainingRecord_0_bits_state_stFinish | _readRecord_T_2641 & chainingRecord_1_bits_state_stFinish | _readRecord_T_2642 & chainingRecord_2_bits_state_stFinish | _readRecord_T_2643
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_2644 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_12_elementMask =
    (_readRecord_T_2640 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_2641 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_2642 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_2643 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_2644 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_12_slow =
    _readRecord_T_2640 & chainingRecord_0_bits_slow | _readRecord_T_2641 & chainingRecord_1_bits_slow | _readRecord_T_2642 & chainingRecord_2_bits_slow | _readRecord_T_2643 & chainingRecord_3_bits_slow | _readRecord_T_2644
    & chainingRecord_4_bits_slow;
  wire        readRecord_12_onlyRead =
    _readRecord_T_2640 & chainingRecord_0_bits_onlyRead | _readRecord_T_2641 & chainingRecord_1_bits_onlyRead | _readRecord_T_2642 & chainingRecord_2_bits_onlyRead | _readRecord_T_2643 & chainingRecord_3_bits_onlyRead | _readRecord_T_2644
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_12_ma =
    _readRecord_T_2640 & chainingRecord_0_bits_ma | _readRecord_T_2641 & chainingRecord_1_bits_ma | _readRecord_T_2642 & chainingRecord_2_bits_ma | _readRecord_T_2643 & chainingRecord_3_bits_ma | _readRecord_T_2644
    & chainingRecord_4_bits_ma;
  wire        readRecord_12_indexType =
    _readRecord_T_2640 & chainingRecord_0_bits_indexType | _readRecord_T_2641 & chainingRecord_1_bits_indexType | _readRecord_T_2642 & chainingRecord_2_bits_indexType | _readRecord_T_2643 & chainingRecord_3_bits_indexType
    | _readRecord_T_2644 & chainingRecord_4_bits_indexType;
  wire        readRecord_12_crossRead =
    _readRecord_T_2640 & chainingRecord_0_bits_crossRead | _readRecord_T_2641 & chainingRecord_1_bits_crossRead | _readRecord_T_2642 & chainingRecord_2_bits_crossRead | _readRecord_T_2643 & chainingRecord_3_bits_crossRead
    | _readRecord_T_2644 & chainingRecord_4_bits_crossRead;
  wire        readRecord_12_crossWrite =
    _readRecord_T_2640 & chainingRecord_0_bits_crossWrite | _readRecord_T_2641 & chainingRecord_1_bits_crossWrite | _readRecord_T_2642 & chainingRecord_2_bits_crossWrite | _readRecord_T_2643 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_2644 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_12_gather16 =
    _readRecord_T_2640 & chainingRecord_0_bits_gather16 | _readRecord_T_2641 & chainingRecord_1_bits_gather16 | _readRecord_T_2642 & chainingRecord_2_bits_gather16 | _readRecord_T_2643 & chainingRecord_3_bits_gather16 | _readRecord_T_2644
    & chainingRecord_4_bits_gather16;
  wire        readRecord_12_gather =
    _readRecord_T_2640 & chainingRecord_0_bits_gather | _readRecord_T_2641 & chainingRecord_1_bits_gather | _readRecord_T_2642 & chainingRecord_2_bits_gather | _readRecord_T_2643 & chainingRecord_3_bits_gather | _readRecord_T_2644
    & chainingRecord_4_bits_gather;
  wire        readRecord_12_st =
    _readRecord_T_2640 & chainingRecord_0_bits_st | _readRecord_T_2641 & chainingRecord_1_bits_st | _readRecord_T_2642 & chainingRecord_2_bits_st | _readRecord_T_2643 & chainingRecord_3_bits_st | _readRecord_T_2644
    & chainingRecord_4_bits_st;
  wire        readRecord_12_ls =
    _readRecord_T_2640 & chainingRecord_0_bits_ls | _readRecord_T_2641 & chainingRecord_1_bits_ls | _readRecord_T_2642 & chainingRecord_2_bits_ls | _readRecord_T_2643 & chainingRecord_3_bits_ls | _readRecord_T_2644
    & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_12_instIndex =
    (_readRecord_T_2640 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_2641 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_2642 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_2643 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_2644 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_12_vs2 =
    (_readRecord_T_2640 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_2641 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_2642 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_2643 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_2644 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_12_vs1_bits =
    (_readRecord_T_2640 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_2641 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_2642 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_2643 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_2644 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_12_vs1_valid =
    _readRecord_T_2640 & chainingRecord_0_bits_vs1_valid | _readRecord_T_2641 & chainingRecord_1_bits_vs1_valid | _readRecord_T_2642 & chainingRecord_2_bits_vs1_valid | _readRecord_T_2643 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_2644 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_12_vd_bits =
    (_readRecord_T_2640 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_2641 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_2642 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_2643 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_2644 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_12_vd_valid =
    _readRecord_T_2640 & chainingRecord_0_bits_vd_valid | _readRecord_T_2641 & chainingRecord_1_bits_vd_valid | _readRecord_T_2642 & chainingRecord_2_bits_vd_valid | _readRecord_T_2643 & chainingRecord_3_bits_vd_valid | _readRecord_T_2644
    & chainingRecord_4_bits_vd_valid;
  wire        _readRecord_T_2860 = chainingRecord_0_bits_instIndex == readCheck_13_instructionIndex;
  wire        _readRecord_T_2861 = chainingRecord_1_bits_instIndex == readCheck_13_instructionIndex;
  wire        _readRecord_T_2862 = chainingRecord_2_bits_instIndex == readCheck_13_instructionIndex;
  wire        _readRecord_T_2863 = chainingRecord_3_bits_instIndex == readCheck_13_instructionIndex;
  wire        _readRecord_T_2864 = chainingRecord_4_bits_instIndex == readCheck_13_instructionIndex;
  wire        readRecord_13_state_wLaneClear =
    _readRecord_T_2860 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_2861 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_2862 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_2863
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_2864 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_13_state_wTopLastReport =
    _readRecord_T_2860 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_2861 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_2862 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_2863
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_2864 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_13_state_wLaneLastReport =
    _readRecord_T_2860 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_2861 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_2862 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_2863
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_2864 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_13_state_wWriteQueueClear =
    _readRecord_T_2860 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_2861 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_2862 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_2863
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_2864 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_13_state_stFinish =
    _readRecord_T_2860 & chainingRecord_0_bits_state_stFinish | _readRecord_T_2861 & chainingRecord_1_bits_state_stFinish | _readRecord_T_2862 & chainingRecord_2_bits_state_stFinish | _readRecord_T_2863
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_2864 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_13_elementMask =
    (_readRecord_T_2860 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_2861 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_2862 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_2863 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_2864 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_13_slow =
    _readRecord_T_2860 & chainingRecord_0_bits_slow | _readRecord_T_2861 & chainingRecord_1_bits_slow | _readRecord_T_2862 & chainingRecord_2_bits_slow | _readRecord_T_2863 & chainingRecord_3_bits_slow | _readRecord_T_2864
    & chainingRecord_4_bits_slow;
  wire        readRecord_13_onlyRead =
    _readRecord_T_2860 & chainingRecord_0_bits_onlyRead | _readRecord_T_2861 & chainingRecord_1_bits_onlyRead | _readRecord_T_2862 & chainingRecord_2_bits_onlyRead | _readRecord_T_2863 & chainingRecord_3_bits_onlyRead | _readRecord_T_2864
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_13_ma =
    _readRecord_T_2860 & chainingRecord_0_bits_ma | _readRecord_T_2861 & chainingRecord_1_bits_ma | _readRecord_T_2862 & chainingRecord_2_bits_ma | _readRecord_T_2863 & chainingRecord_3_bits_ma | _readRecord_T_2864
    & chainingRecord_4_bits_ma;
  wire        readRecord_13_indexType =
    _readRecord_T_2860 & chainingRecord_0_bits_indexType | _readRecord_T_2861 & chainingRecord_1_bits_indexType | _readRecord_T_2862 & chainingRecord_2_bits_indexType | _readRecord_T_2863 & chainingRecord_3_bits_indexType
    | _readRecord_T_2864 & chainingRecord_4_bits_indexType;
  wire        readRecord_13_crossRead =
    _readRecord_T_2860 & chainingRecord_0_bits_crossRead | _readRecord_T_2861 & chainingRecord_1_bits_crossRead | _readRecord_T_2862 & chainingRecord_2_bits_crossRead | _readRecord_T_2863 & chainingRecord_3_bits_crossRead
    | _readRecord_T_2864 & chainingRecord_4_bits_crossRead;
  wire        readRecord_13_crossWrite =
    _readRecord_T_2860 & chainingRecord_0_bits_crossWrite | _readRecord_T_2861 & chainingRecord_1_bits_crossWrite | _readRecord_T_2862 & chainingRecord_2_bits_crossWrite | _readRecord_T_2863 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_2864 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_13_gather16 =
    _readRecord_T_2860 & chainingRecord_0_bits_gather16 | _readRecord_T_2861 & chainingRecord_1_bits_gather16 | _readRecord_T_2862 & chainingRecord_2_bits_gather16 | _readRecord_T_2863 & chainingRecord_3_bits_gather16 | _readRecord_T_2864
    & chainingRecord_4_bits_gather16;
  wire        readRecord_13_gather =
    _readRecord_T_2860 & chainingRecord_0_bits_gather | _readRecord_T_2861 & chainingRecord_1_bits_gather | _readRecord_T_2862 & chainingRecord_2_bits_gather | _readRecord_T_2863 & chainingRecord_3_bits_gather | _readRecord_T_2864
    & chainingRecord_4_bits_gather;
  wire        readRecord_13_st =
    _readRecord_T_2860 & chainingRecord_0_bits_st | _readRecord_T_2861 & chainingRecord_1_bits_st | _readRecord_T_2862 & chainingRecord_2_bits_st | _readRecord_T_2863 & chainingRecord_3_bits_st | _readRecord_T_2864
    & chainingRecord_4_bits_st;
  wire        readRecord_13_ls =
    _readRecord_T_2860 & chainingRecord_0_bits_ls | _readRecord_T_2861 & chainingRecord_1_bits_ls | _readRecord_T_2862 & chainingRecord_2_bits_ls | _readRecord_T_2863 & chainingRecord_3_bits_ls | _readRecord_T_2864
    & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_13_instIndex =
    (_readRecord_T_2860 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_2861 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_2862 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_2863 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_2864 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_13_vs2 =
    (_readRecord_T_2860 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_2861 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_2862 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_2863 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_2864 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_13_vs1_bits =
    (_readRecord_T_2860 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_2861 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_2862 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_2863 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_2864 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_13_vs1_valid =
    _readRecord_T_2860 & chainingRecord_0_bits_vs1_valid | _readRecord_T_2861 & chainingRecord_1_bits_vs1_valid | _readRecord_T_2862 & chainingRecord_2_bits_vs1_valid | _readRecord_T_2863 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_2864 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_13_vd_bits =
    (_readRecord_T_2860 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_2861 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_2862 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_2863 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_2864 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_13_vd_valid =
    _readRecord_T_2860 & chainingRecord_0_bits_vd_valid | _readRecord_T_2861 & chainingRecord_1_bits_vd_valid | _readRecord_T_2862 & chainingRecord_2_bits_vd_valid | _readRecord_T_2863 & chainingRecord_3_bits_vd_valid | _readRecord_T_2864
    & chainingRecord_4_bits_vd_valid;
  wire        _readRecord_T_3080 = chainingRecord_0_bits_instIndex == readRequests_0_bits_instructionIndex_0;
  wire        _readRecord_T_3081 = chainingRecord_1_bits_instIndex == readRequests_0_bits_instructionIndex_0;
  wire        _readRecord_T_3082 = chainingRecord_2_bits_instIndex == readRequests_0_bits_instructionIndex_0;
  wire        _readRecord_T_3083 = chainingRecord_3_bits_instIndex == readRequests_0_bits_instructionIndex_0;
  wire        _readRecord_T_3084 = chainingRecord_4_bits_instIndex == readRequests_0_bits_instructionIndex_0;
  wire        readRecord_14_state_wLaneClear =
    _readRecord_T_3080 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_3081 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_3082 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_3083
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_3084 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_14_state_wTopLastReport =
    _readRecord_T_3080 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_3081 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_3082 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_3083
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_3084 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_14_state_wLaneLastReport =
    _readRecord_T_3080 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_3081 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_3082 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_3083
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_3084 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_14_state_wWriteQueueClear =
    _readRecord_T_3080 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_3081 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_3082 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_3083
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_3084 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_14_state_stFinish =
    _readRecord_T_3080 & chainingRecord_0_bits_state_stFinish | _readRecord_T_3081 & chainingRecord_1_bits_state_stFinish | _readRecord_T_3082 & chainingRecord_2_bits_state_stFinish | _readRecord_T_3083
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_3084 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_14_elementMask =
    (_readRecord_T_3080 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_3081 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_3082 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_3083 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_3084 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_14_slow =
    _readRecord_T_3080 & chainingRecord_0_bits_slow | _readRecord_T_3081 & chainingRecord_1_bits_slow | _readRecord_T_3082 & chainingRecord_2_bits_slow | _readRecord_T_3083 & chainingRecord_3_bits_slow | _readRecord_T_3084
    & chainingRecord_4_bits_slow;
  wire        readRecord_14_onlyRead =
    _readRecord_T_3080 & chainingRecord_0_bits_onlyRead | _readRecord_T_3081 & chainingRecord_1_bits_onlyRead | _readRecord_T_3082 & chainingRecord_2_bits_onlyRead | _readRecord_T_3083 & chainingRecord_3_bits_onlyRead | _readRecord_T_3084
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_14_ma =
    _readRecord_T_3080 & chainingRecord_0_bits_ma | _readRecord_T_3081 & chainingRecord_1_bits_ma | _readRecord_T_3082 & chainingRecord_2_bits_ma | _readRecord_T_3083 & chainingRecord_3_bits_ma | _readRecord_T_3084
    & chainingRecord_4_bits_ma;
  wire        readRecord_14_indexType =
    _readRecord_T_3080 & chainingRecord_0_bits_indexType | _readRecord_T_3081 & chainingRecord_1_bits_indexType | _readRecord_T_3082 & chainingRecord_2_bits_indexType | _readRecord_T_3083 & chainingRecord_3_bits_indexType
    | _readRecord_T_3084 & chainingRecord_4_bits_indexType;
  wire        readRecord_14_crossRead =
    _readRecord_T_3080 & chainingRecord_0_bits_crossRead | _readRecord_T_3081 & chainingRecord_1_bits_crossRead | _readRecord_T_3082 & chainingRecord_2_bits_crossRead | _readRecord_T_3083 & chainingRecord_3_bits_crossRead
    | _readRecord_T_3084 & chainingRecord_4_bits_crossRead;
  wire        readRecord_14_crossWrite =
    _readRecord_T_3080 & chainingRecord_0_bits_crossWrite | _readRecord_T_3081 & chainingRecord_1_bits_crossWrite | _readRecord_T_3082 & chainingRecord_2_bits_crossWrite | _readRecord_T_3083 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_3084 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_14_gather16 =
    _readRecord_T_3080 & chainingRecord_0_bits_gather16 | _readRecord_T_3081 & chainingRecord_1_bits_gather16 | _readRecord_T_3082 & chainingRecord_2_bits_gather16 | _readRecord_T_3083 & chainingRecord_3_bits_gather16 | _readRecord_T_3084
    & chainingRecord_4_bits_gather16;
  wire        readRecord_14_gather =
    _readRecord_T_3080 & chainingRecord_0_bits_gather | _readRecord_T_3081 & chainingRecord_1_bits_gather | _readRecord_T_3082 & chainingRecord_2_bits_gather | _readRecord_T_3083 & chainingRecord_3_bits_gather | _readRecord_T_3084
    & chainingRecord_4_bits_gather;
  wire        readRecord_14_st =
    _readRecord_T_3080 & chainingRecord_0_bits_st | _readRecord_T_3081 & chainingRecord_1_bits_st | _readRecord_T_3082 & chainingRecord_2_bits_st | _readRecord_T_3083 & chainingRecord_3_bits_st | _readRecord_T_3084
    & chainingRecord_4_bits_st;
  wire        readRecord_14_ls =
    _readRecord_T_3080 & chainingRecord_0_bits_ls | _readRecord_T_3081 & chainingRecord_1_bits_ls | _readRecord_T_3082 & chainingRecord_2_bits_ls | _readRecord_T_3083 & chainingRecord_3_bits_ls | _readRecord_T_3084
    & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_14_instIndex =
    (_readRecord_T_3080 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_3081 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_3082 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_3083 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_3084 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_14_vs2 =
    (_readRecord_T_3080 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_3081 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_3082 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_3083 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_3084 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_14_vs1_bits =
    (_readRecord_T_3080 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_3081 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_3082 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_3083 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_3084 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_14_vs1_valid =
    _readRecord_T_3080 & chainingRecord_0_bits_vs1_valid | _readRecord_T_3081 & chainingRecord_1_bits_vs1_valid | _readRecord_T_3082 & chainingRecord_2_bits_vs1_valid | _readRecord_T_3083 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_3084 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_14_vd_bits =
    (_readRecord_T_3080 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_3081 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_3082 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_3083 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_3084 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_14_vd_valid =
    _readRecord_T_3080 & chainingRecord_0_bits_vd_valid | _readRecord_T_3081 & chainingRecord_1_bits_vd_valid | _readRecord_T_3082 & chainingRecord_2_bits_vd_valid | _readRecord_T_3083 & chainingRecord_3_bits_vd_valid | _readRecord_T_3084
    & chainingRecord_4_bits_vd_valid;
  wire [5:0]  address = {readRequests_0_bits_vs_0, readRequests_0_bits_offset_0};
  wire [1:0]  bank = 2'h1 << address[0];
  reg         pipeBank_pipe_v;
  reg  [1:0]  pipeBank_pipe_b;
  reg         pipeBank_pipe_pipe_v;
  wire        pipeBank_pipe_pipe_out_valid = pipeBank_pipe_pipe_v;
  reg  [1:0]  pipeBank_pipe_pipe_b;
  wire [1:0]  pipeBank_pipe_pipe_out_bits = pipeBank_pipe_pipe_b;
  assign bankCorrect = readRequests_0_valid_0 ? bank : 2'h0;
  wire [1:0]  bankReadF_0 = bankCorrect;
  wire        portConflictCheck =
    ~(write_valid_0 & bank == writeBank & write_bits_vd_0 == readRequests_0_bits_vs_0 & write_bits_offset_0 == readRequests_0_bits_offset_0 | writePipe_valid & bank == writeBankPipe & writePipe_bits_vd == readRequests_0_bits_vs_0
      & writePipe_bits_offset == readRequests_0_bits_offset_0);
  wire        portReady = |bank;
  assign readRequests_0_ready_0 = portReady & sramReady;
  reg         pipeFirstUsed_pipe_v;
  reg         pipeFirstUsed_pipe_pipe_v;
  wire        pipeFirstUsed_pipe_pipe_out_valid = pipeFirstUsed_pipe_pipe_v;
  reg         pipeFirstUsed_pipe_pipe_b;
  wire        pipeFirstUsed_pipe_pipe_out_bits = pipeFirstUsed_pipe_pipe_b;
  reg         pipeFire_pipe_v;
  reg         pipeFire_pipe_b;
  reg         pipeFire_pipe_pipe_v;
  wire        pipeFire_pipe_pipe_out_valid = pipeFire_pipe_pipe_v;
  reg         pipeFire_pipe_pipe_b;
  wire        pipeFire_pipe_pipe_out_bits = pipeFire_pipe_pipe_b;
  wire [31:0] readResultF_0;
  wire [31:0] readResultF_1;
  wire [31:0] readResultS_0;
  wire [31:0] readResultS_1;
  wire        _readRecord_T_3300 = chainingRecord_0_bits_instIndex == readRequests_1_bits_instructionIndex_0;
  wire        _readRecord_T_3301 = chainingRecord_1_bits_instIndex == readRequests_1_bits_instructionIndex_0;
  wire        _readRecord_T_3302 = chainingRecord_2_bits_instIndex == readRequests_1_bits_instructionIndex_0;
  wire        _readRecord_T_3303 = chainingRecord_3_bits_instIndex == readRequests_1_bits_instructionIndex_0;
  wire        _readRecord_T_3304 = chainingRecord_4_bits_instIndex == readRequests_1_bits_instructionIndex_0;
  wire        readRecord_15_state_wLaneClear =
    _readRecord_T_3300 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_3301 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_3302 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_3303
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_3304 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_15_state_wTopLastReport =
    _readRecord_T_3300 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_3301 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_3302 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_3303
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_3304 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_15_state_wLaneLastReport =
    _readRecord_T_3300 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_3301 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_3302 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_3303
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_3304 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_15_state_wWriteQueueClear =
    _readRecord_T_3300 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_3301 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_3302 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_3303
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_3304 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_15_state_stFinish =
    _readRecord_T_3300 & chainingRecord_0_bits_state_stFinish | _readRecord_T_3301 & chainingRecord_1_bits_state_stFinish | _readRecord_T_3302 & chainingRecord_2_bits_state_stFinish | _readRecord_T_3303
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_3304 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_15_elementMask =
    (_readRecord_T_3300 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_3301 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_3302 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_3303 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_3304 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_15_slow =
    _readRecord_T_3300 & chainingRecord_0_bits_slow | _readRecord_T_3301 & chainingRecord_1_bits_slow | _readRecord_T_3302 & chainingRecord_2_bits_slow | _readRecord_T_3303 & chainingRecord_3_bits_slow | _readRecord_T_3304
    & chainingRecord_4_bits_slow;
  wire        readRecord_15_onlyRead =
    _readRecord_T_3300 & chainingRecord_0_bits_onlyRead | _readRecord_T_3301 & chainingRecord_1_bits_onlyRead | _readRecord_T_3302 & chainingRecord_2_bits_onlyRead | _readRecord_T_3303 & chainingRecord_3_bits_onlyRead | _readRecord_T_3304
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_15_ma =
    _readRecord_T_3300 & chainingRecord_0_bits_ma | _readRecord_T_3301 & chainingRecord_1_bits_ma | _readRecord_T_3302 & chainingRecord_2_bits_ma | _readRecord_T_3303 & chainingRecord_3_bits_ma | _readRecord_T_3304
    & chainingRecord_4_bits_ma;
  wire        readRecord_15_indexType =
    _readRecord_T_3300 & chainingRecord_0_bits_indexType | _readRecord_T_3301 & chainingRecord_1_bits_indexType | _readRecord_T_3302 & chainingRecord_2_bits_indexType | _readRecord_T_3303 & chainingRecord_3_bits_indexType
    | _readRecord_T_3304 & chainingRecord_4_bits_indexType;
  wire        readRecord_15_crossRead =
    _readRecord_T_3300 & chainingRecord_0_bits_crossRead | _readRecord_T_3301 & chainingRecord_1_bits_crossRead | _readRecord_T_3302 & chainingRecord_2_bits_crossRead | _readRecord_T_3303 & chainingRecord_3_bits_crossRead
    | _readRecord_T_3304 & chainingRecord_4_bits_crossRead;
  wire        readRecord_15_crossWrite =
    _readRecord_T_3300 & chainingRecord_0_bits_crossWrite | _readRecord_T_3301 & chainingRecord_1_bits_crossWrite | _readRecord_T_3302 & chainingRecord_2_bits_crossWrite | _readRecord_T_3303 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_3304 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_15_gather16 =
    _readRecord_T_3300 & chainingRecord_0_bits_gather16 | _readRecord_T_3301 & chainingRecord_1_bits_gather16 | _readRecord_T_3302 & chainingRecord_2_bits_gather16 | _readRecord_T_3303 & chainingRecord_3_bits_gather16 | _readRecord_T_3304
    & chainingRecord_4_bits_gather16;
  wire        readRecord_15_gather =
    _readRecord_T_3300 & chainingRecord_0_bits_gather | _readRecord_T_3301 & chainingRecord_1_bits_gather | _readRecord_T_3302 & chainingRecord_2_bits_gather | _readRecord_T_3303 & chainingRecord_3_bits_gather | _readRecord_T_3304
    & chainingRecord_4_bits_gather;
  wire        readRecord_15_st =
    _readRecord_T_3300 & chainingRecord_0_bits_st | _readRecord_T_3301 & chainingRecord_1_bits_st | _readRecord_T_3302 & chainingRecord_2_bits_st | _readRecord_T_3303 & chainingRecord_3_bits_st | _readRecord_T_3304
    & chainingRecord_4_bits_st;
  wire        readRecord_15_ls =
    _readRecord_T_3300 & chainingRecord_0_bits_ls | _readRecord_T_3301 & chainingRecord_1_bits_ls | _readRecord_T_3302 & chainingRecord_2_bits_ls | _readRecord_T_3303 & chainingRecord_3_bits_ls | _readRecord_T_3304
    & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_15_instIndex =
    (_readRecord_T_3300 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_3301 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_3302 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_3303 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_3304 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_15_vs2 =
    (_readRecord_T_3300 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_3301 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_3302 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_3303 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_3304 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_15_vs1_bits =
    (_readRecord_T_3300 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_3301 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_3302 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_3303 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_3304 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_15_vs1_valid =
    _readRecord_T_3300 & chainingRecord_0_bits_vs1_valid | _readRecord_T_3301 & chainingRecord_1_bits_vs1_valid | _readRecord_T_3302 & chainingRecord_2_bits_vs1_valid | _readRecord_T_3303 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_3304 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_15_vd_bits =
    (_readRecord_T_3300 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_3301 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_3302 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_3303 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_3304 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_15_vd_valid =
    _readRecord_T_3300 & chainingRecord_0_bits_vd_valid | _readRecord_T_3301 & chainingRecord_1_bits_vd_valid | _readRecord_T_3302 & chainingRecord_2_bits_vd_valid | _readRecord_T_3303 & chainingRecord_3_bits_vd_valid | _readRecord_T_3304
    & chainingRecord_4_bits_vd_valid;
  wire [5:0]  address_1 = {readRequests_1_bits_vs_0, readRequests_1_bits_offset_0};
  wire [1:0]  bank_1 = 2'h1 << address_1[0];
  reg         pipeBank_pipe_v_1;
  reg  [1:0]  pipeBank_pipe_b_1;
  reg         pipeBank_pipe_pipe_v_1;
  wire        pipeBank_pipe_pipe_out_1_valid = pipeBank_pipe_pipe_v_1;
  reg  [1:0]  pipeBank_pipe_pipe_b_1;
  wire [1:0]  pipeBank_pipe_pipe_out_1_bits = pipeBank_pipe_pipe_b_1;
  wire [1:0]  bankCorrect_1 = readRequests_1_valid_0 ? bank_1 : 2'h0;
  wire        portConflictCheck_1 =
    ~(write_valid_0 & bank_1 == writeBank & write_bits_vd_0 == readRequests_1_bits_vs_0 & write_bits_offset_0 == readRequests_1_bits_offset_0 | writePipe_valid & bank_1 == writeBankPipe & writePipe_bits_vd == readRequests_1_bits_vs_0
      & writePipe_bits_offset == readRequests_1_bits_offset_0);
  wire        portReady_1 = |bank_1;
  assign readRequests_1_ready_0 = portReady_1 & sramReady;
  wire        firstUsed_1 = |(bank_1 & bankCorrect);
  wire [1:0]  bankReadF_1 = bankCorrect_1 & ~bankCorrect;
  wire [1:0]  bankReadS_1 = bankCorrect_1 & bankCorrect;
  reg         pipeFirstUsed_pipe_v_1;
  reg         pipeFirstUsed_pipe_b_1;
  reg         pipeFirstUsed_pipe_pipe_v_1;
  wire        pipeFirstUsed_pipe_pipe_out_1_valid = pipeFirstUsed_pipe_pipe_v_1;
  reg         pipeFirstUsed_pipe_pipe_b_1;
  wire        pipeFirstUsed_pipe_pipe_out_1_bits = pipeFirstUsed_pipe_pipe_b_1;
  reg         pipeFire_pipe_v_1;
  reg         pipeFire_pipe_b_1;
  reg         pipeFire_pipe_pipe_v_1;
  wire        pipeFire_pipe_pipe_out_1_valid = pipeFire_pipe_pipe_v_1;
  reg         pipeFire_pipe_pipe_b_1;
  wire        pipeFire_pipe_pipe_out_1_bits = pipeFire_pipe_pipe_b_1;
  wire [1:0]  _GEN = bankCorrect | bankCorrect_1;
  wire [1:0]  _GEN_0 = bankCorrect_1 & bankCorrect;
  wire        _readRecord_T_3520 = chainingRecord_0_bits_instIndex == readRequests_2_bits_instructionIndex_0;
  wire        _readRecord_T_3521 = chainingRecord_1_bits_instIndex == readRequests_2_bits_instructionIndex_0;
  wire        _readRecord_T_3522 = chainingRecord_2_bits_instIndex == readRequests_2_bits_instructionIndex_0;
  wire        _readRecord_T_3523 = chainingRecord_3_bits_instIndex == readRequests_2_bits_instructionIndex_0;
  wire        _readRecord_T_3524 = chainingRecord_4_bits_instIndex == readRequests_2_bits_instructionIndex_0;
  wire        readRecord_16_state_wLaneClear =
    _readRecord_T_3520 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_3521 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_3522 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_3523
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_3524 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_16_state_wTopLastReport =
    _readRecord_T_3520 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_3521 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_3522 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_3523
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_3524 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_16_state_wLaneLastReport =
    _readRecord_T_3520 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_3521 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_3522 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_3523
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_3524 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_16_state_wWriteQueueClear =
    _readRecord_T_3520 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_3521 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_3522 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_3523
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_3524 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_16_state_stFinish =
    _readRecord_T_3520 & chainingRecord_0_bits_state_stFinish | _readRecord_T_3521 & chainingRecord_1_bits_state_stFinish | _readRecord_T_3522 & chainingRecord_2_bits_state_stFinish | _readRecord_T_3523
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_3524 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_16_elementMask =
    (_readRecord_T_3520 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_3521 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_3522 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_3523 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_3524 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_16_slow =
    _readRecord_T_3520 & chainingRecord_0_bits_slow | _readRecord_T_3521 & chainingRecord_1_bits_slow | _readRecord_T_3522 & chainingRecord_2_bits_slow | _readRecord_T_3523 & chainingRecord_3_bits_slow | _readRecord_T_3524
    & chainingRecord_4_bits_slow;
  wire        readRecord_16_onlyRead =
    _readRecord_T_3520 & chainingRecord_0_bits_onlyRead | _readRecord_T_3521 & chainingRecord_1_bits_onlyRead | _readRecord_T_3522 & chainingRecord_2_bits_onlyRead | _readRecord_T_3523 & chainingRecord_3_bits_onlyRead | _readRecord_T_3524
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_16_ma =
    _readRecord_T_3520 & chainingRecord_0_bits_ma | _readRecord_T_3521 & chainingRecord_1_bits_ma | _readRecord_T_3522 & chainingRecord_2_bits_ma | _readRecord_T_3523 & chainingRecord_3_bits_ma | _readRecord_T_3524
    & chainingRecord_4_bits_ma;
  wire        readRecord_16_indexType =
    _readRecord_T_3520 & chainingRecord_0_bits_indexType | _readRecord_T_3521 & chainingRecord_1_bits_indexType | _readRecord_T_3522 & chainingRecord_2_bits_indexType | _readRecord_T_3523 & chainingRecord_3_bits_indexType
    | _readRecord_T_3524 & chainingRecord_4_bits_indexType;
  wire        readRecord_16_crossRead =
    _readRecord_T_3520 & chainingRecord_0_bits_crossRead | _readRecord_T_3521 & chainingRecord_1_bits_crossRead | _readRecord_T_3522 & chainingRecord_2_bits_crossRead | _readRecord_T_3523 & chainingRecord_3_bits_crossRead
    | _readRecord_T_3524 & chainingRecord_4_bits_crossRead;
  wire        readRecord_16_crossWrite =
    _readRecord_T_3520 & chainingRecord_0_bits_crossWrite | _readRecord_T_3521 & chainingRecord_1_bits_crossWrite | _readRecord_T_3522 & chainingRecord_2_bits_crossWrite | _readRecord_T_3523 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_3524 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_16_gather16 =
    _readRecord_T_3520 & chainingRecord_0_bits_gather16 | _readRecord_T_3521 & chainingRecord_1_bits_gather16 | _readRecord_T_3522 & chainingRecord_2_bits_gather16 | _readRecord_T_3523 & chainingRecord_3_bits_gather16 | _readRecord_T_3524
    & chainingRecord_4_bits_gather16;
  wire        readRecord_16_gather =
    _readRecord_T_3520 & chainingRecord_0_bits_gather | _readRecord_T_3521 & chainingRecord_1_bits_gather | _readRecord_T_3522 & chainingRecord_2_bits_gather | _readRecord_T_3523 & chainingRecord_3_bits_gather | _readRecord_T_3524
    & chainingRecord_4_bits_gather;
  wire        readRecord_16_st =
    _readRecord_T_3520 & chainingRecord_0_bits_st | _readRecord_T_3521 & chainingRecord_1_bits_st | _readRecord_T_3522 & chainingRecord_2_bits_st | _readRecord_T_3523 & chainingRecord_3_bits_st | _readRecord_T_3524
    & chainingRecord_4_bits_st;
  wire        readRecord_16_ls =
    _readRecord_T_3520 & chainingRecord_0_bits_ls | _readRecord_T_3521 & chainingRecord_1_bits_ls | _readRecord_T_3522 & chainingRecord_2_bits_ls | _readRecord_T_3523 & chainingRecord_3_bits_ls | _readRecord_T_3524
    & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_16_instIndex =
    (_readRecord_T_3520 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_3521 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_3522 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_3523 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_3524 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_16_vs2 =
    (_readRecord_T_3520 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_3521 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_3522 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_3523 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_3524 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_16_vs1_bits =
    (_readRecord_T_3520 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_3521 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_3522 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_3523 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_3524 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_16_vs1_valid =
    _readRecord_T_3520 & chainingRecord_0_bits_vs1_valid | _readRecord_T_3521 & chainingRecord_1_bits_vs1_valid | _readRecord_T_3522 & chainingRecord_2_bits_vs1_valid | _readRecord_T_3523 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_3524 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_16_vd_bits =
    (_readRecord_T_3520 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_3521 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_3522 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_3523 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_3524 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_16_vd_valid =
    _readRecord_T_3520 & chainingRecord_0_bits_vd_valid | _readRecord_T_3521 & chainingRecord_1_bits_vd_valid | _readRecord_T_3522 & chainingRecord_2_bits_vd_valid | _readRecord_T_3523 & chainingRecord_3_bits_vd_valid | _readRecord_T_3524
    & chainingRecord_4_bits_vd_valid;
  wire [5:0]  address_2 = {readRequests_2_bits_vs_0, readRequests_2_bits_offset_0};
  wire [1:0]  bank_2 = 2'h1 << address_2[0];
  reg         pipeBank_pipe_v_2;
  reg  [1:0]  pipeBank_pipe_b_2;
  reg         pipeBank_pipe_pipe_v_2;
  wire        pipeBank_pipe_pipe_out_2_valid = pipeBank_pipe_pipe_v_2;
  reg  [1:0]  pipeBank_pipe_pipe_b_2;
  wire [1:0]  pipeBank_pipe_pipe_out_2_bits = pipeBank_pipe_pipe_b_2;
  wire [1:0]  bankCorrect_2 = readRequests_2_valid_0 ? bank_2 : 2'h0;
  wire        portConflictCheck_2 =
    ~(write_valid_0 & bank_2 == writeBank & write_bits_vd_0 == readRequests_2_bits_vs_0 & write_bits_offset_0 == readRequests_2_bits_offset_0 | writePipe_valid & bank_2 == writeBankPipe & writePipe_bits_vd == readRequests_2_bits_vs_0
      & writePipe_bits_offset == readRequests_2_bits_offset_0);
  wire        portReady_2 = |(bank_2 & ~_GEN_0);
  assign readRequests_2_ready_0 = portReady_2 & sramReady;
  wire        firstUsed_2 = |(bank_2 & _GEN);
  wire [1:0]  bankReadF_2 = bankCorrect_2 & ~_GEN;
  wire [1:0]  bankReadS_2 = bankCorrect_2 & ~_GEN_0 & _GEN;
  reg         pipeFirstUsed_pipe_v_2;
  reg         pipeFirstUsed_pipe_b_2;
  reg         pipeFirstUsed_pipe_pipe_v_2;
  wire        pipeFirstUsed_pipe_pipe_out_2_valid = pipeFirstUsed_pipe_pipe_v_2;
  reg         pipeFirstUsed_pipe_pipe_b_2;
  wire        pipeFirstUsed_pipe_pipe_out_2_bits = pipeFirstUsed_pipe_pipe_b_2;
  reg         pipeFire_pipe_v_2;
  reg         pipeFire_pipe_b_2;
  reg         pipeFire_pipe_pipe_v_2;
  wire        pipeFire_pipe_pipe_out_2_valid = pipeFire_pipe_pipe_v_2;
  reg         pipeFire_pipe_pipe_b_2;
  wire        pipeFire_pipe_pipe_out_2_bits = pipeFire_pipe_pipe_b_2;
  wire [1:0]  _GEN_1 = _GEN | bankCorrect_2;
  wire [1:0]  _GEN_2 = bankCorrect_2 & _GEN | _GEN_0;
  wire        _readRecord_T_3740 = chainingRecord_0_bits_instIndex == readRequests_3_bits_instructionIndex_0;
  wire        _readRecord_T_3741 = chainingRecord_1_bits_instIndex == readRequests_3_bits_instructionIndex_0;
  wire        _readRecord_T_3742 = chainingRecord_2_bits_instIndex == readRequests_3_bits_instructionIndex_0;
  wire        _readRecord_T_3743 = chainingRecord_3_bits_instIndex == readRequests_3_bits_instructionIndex_0;
  wire        _readRecord_T_3744 = chainingRecord_4_bits_instIndex == readRequests_3_bits_instructionIndex_0;
  wire        readRecord_17_state_wLaneClear =
    _readRecord_T_3740 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_3741 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_3742 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_3743
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_3744 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_17_state_wTopLastReport =
    _readRecord_T_3740 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_3741 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_3742 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_3743
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_3744 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_17_state_wLaneLastReport =
    _readRecord_T_3740 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_3741 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_3742 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_3743
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_3744 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_17_state_wWriteQueueClear =
    _readRecord_T_3740 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_3741 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_3742 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_3743
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_3744 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_17_state_stFinish =
    _readRecord_T_3740 & chainingRecord_0_bits_state_stFinish | _readRecord_T_3741 & chainingRecord_1_bits_state_stFinish | _readRecord_T_3742 & chainingRecord_2_bits_state_stFinish | _readRecord_T_3743
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_3744 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_17_elementMask =
    (_readRecord_T_3740 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_3741 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_3742 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_3743 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_3744 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_17_slow =
    _readRecord_T_3740 & chainingRecord_0_bits_slow | _readRecord_T_3741 & chainingRecord_1_bits_slow | _readRecord_T_3742 & chainingRecord_2_bits_slow | _readRecord_T_3743 & chainingRecord_3_bits_slow | _readRecord_T_3744
    & chainingRecord_4_bits_slow;
  wire        readRecord_17_onlyRead =
    _readRecord_T_3740 & chainingRecord_0_bits_onlyRead | _readRecord_T_3741 & chainingRecord_1_bits_onlyRead | _readRecord_T_3742 & chainingRecord_2_bits_onlyRead | _readRecord_T_3743 & chainingRecord_3_bits_onlyRead | _readRecord_T_3744
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_17_ma =
    _readRecord_T_3740 & chainingRecord_0_bits_ma | _readRecord_T_3741 & chainingRecord_1_bits_ma | _readRecord_T_3742 & chainingRecord_2_bits_ma | _readRecord_T_3743 & chainingRecord_3_bits_ma | _readRecord_T_3744
    & chainingRecord_4_bits_ma;
  wire        readRecord_17_indexType =
    _readRecord_T_3740 & chainingRecord_0_bits_indexType | _readRecord_T_3741 & chainingRecord_1_bits_indexType | _readRecord_T_3742 & chainingRecord_2_bits_indexType | _readRecord_T_3743 & chainingRecord_3_bits_indexType
    | _readRecord_T_3744 & chainingRecord_4_bits_indexType;
  wire        readRecord_17_crossRead =
    _readRecord_T_3740 & chainingRecord_0_bits_crossRead | _readRecord_T_3741 & chainingRecord_1_bits_crossRead | _readRecord_T_3742 & chainingRecord_2_bits_crossRead | _readRecord_T_3743 & chainingRecord_3_bits_crossRead
    | _readRecord_T_3744 & chainingRecord_4_bits_crossRead;
  wire        readRecord_17_crossWrite =
    _readRecord_T_3740 & chainingRecord_0_bits_crossWrite | _readRecord_T_3741 & chainingRecord_1_bits_crossWrite | _readRecord_T_3742 & chainingRecord_2_bits_crossWrite | _readRecord_T_3743 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_3744 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_17_gather16 =
    _readRecord_T_3740 & chainingRecord_0_bits_gather16 | _readRecord_T_3741 & chainingRecord_1_bits_gather16 | _readRecord_T_3742 & chainingRecord_2_bits_gather16 | _readRecord_T_3743 & chainingRecord_3_bits_gather16 | _readRecord_T_3744
    & chainingRecord_4_bits_gather16;
  wire        readRecord_17_gather =
    _readRecord_T_3740 & chainingRecord_0_bits_gather | _readRecord_T_3741 & chainingRecord_1_bits_gather | _readRecord_T_3742 & chainingRecord_2_bits_gather | _readRecord_T_3743 & chainingRecord_3_bits_gather | _readRecord_T_3744
    & chainingRecord_4_bits_gather;
  wire        readRecord_17_st =
    _readRecord_T_3740 & chainingRecord_0_bits_st | _readRecord_T_3741 & chainingRecord_1_bits_st | _readRecord_T_3742 & chainingRecord_2_bits_st | _readRecord_T_3743 & chainingRecord_3_bits_st | _readRecord_T_3744
    & chainingRecord_4_bits_st;
  wire        readRecord_17_ls =
    _readRecord_T_3740 & chainingRecord_0_bits_ls | _readRecord_T_3741 & chainingRecord_1_bits_ls | _readRecord_T_3742 & chainingRecord_2_bits_ls | _readRecord_T_3743 & chainingRecord_3_bits_ls | _readRecord_T_3744
    & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_17_instIndex =
    (_readRecord_T_3740 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_3741 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_3742 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_3743 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_3744 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_17_vs2 =
    (_readRecord_T_3740 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_3741 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_3742 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_3743 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_3744 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_17_vs1_bits =
    (_readRecord_T_3740 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_3741 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_3742 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_3743 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_3744 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_17_vs1_valid =
    _readRecord_T_3740 & chainingRecord_0_bits_vs1_valid | _readRecord_T_3741 & chainingRecord_1_bits_vs1_valid | _readRecord_T_3742 & chainingRecord_2_bits_vs1_valid | _readRecord_T_3743 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_3744 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_17_vd_bits =
    (_readRecord_T_3740 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_3741 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_3742 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_3743 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_3744 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_17_vd_valid =
    _readRecord_T_3740 & chainingRecord_0_bits_vd_valid | _readRecord_T_3741 & chainingRecord_1_bits_vd_valid | _readRecord_T_3742 & chainingRecord_2_bits_vd_valid | _readRecord_T_3743 & chainingRecord_3_bits_vd_valid | _readRecord_T_3744
    & chainingRecord_4_bits_vd_valid;
  wire [5:0]  address_3 = {readRequests_3_bits_vs_0, readRequests_3_bits_offset_0};
  wire [1:0]  bank_3 = 2'h1 << address_3[0];
  reg         pipeBank_pipe_v_3;
  reg  [1:0]  pipeBank_pipe_b_3;
  reg         pipeBank_pipe_pipe_v_3;
  wire        pipeBank_pipe_pipe_out_3_valid = pipeBank_pipe_pipe_v_3;
  reg  [1:0]  pipeBank_pipe_pipe_b_3;
  wire [1:0]  pipeBank_pipe_pipe_out_3_bits = pipeBank_pipe_pipe_b_3;
  wire [1:0]  bankCorrect_3 = readRequests_3_valid_0 ? bank_3 : 2'h0;
  wire        portConflictCheck_3 =
    ~(write_valid_0 & bank_3 == writeBank & write_bits_vd_0 == readRequests_3_bits_vs_0 & write_bits_offset_0 == readRequests_3_bits_offset_0 | writePipe_valid & bank_3 == writeBankPipe & writePipe_bits_vd == readRequests_3_bits_vs_0
      & writePipe_bits_offset == readRequests_3_bits_offset_0);
  wire        portReady_3 = |(bank_3 & ~_GEN_2);
  assign readRequests_3_ready_0 = portReady_3 & sramReady;
  wire        firstUsed_3 = |(bank_3 & _GEN_1);
  wire [1:0]  bankReadF_3 = bankCorrect_3 & ~_GEN_1;
  wire [1:0]  bankReadS_3 = bankCorrect_3 & ~_GEN_2 & _GEN_1;
  reg         pipeFirstUsed_pipe_v_3;
  reg         pipeFirstUsed_pipe_b_3;
  reg         pipeFirstUsed_pipe_pipe_v_3;
  wire        pipeFirstUsed_pipe_pipe_out_3_valid = pipeFirstUsed_pipe_pipe_v_3;
  reg         pipeFirstUsed_pipe_pipe_b_3;
  wire        pipeFirstUsed_pipe_pipe_out_3_bits = pipeFirstUsed_pipe_pipe_b_3;
  reg         pipeFire_pipe_v_3;
  reg         pipeFire_pipe_b_3;
  reg         pipeFire_pipe_pipe_v_3;
  wire        pipeFire_pipe_pipe_out_3_valid = pipeFire_pipe_pipe_v_3;
  reg         pipeFire_pipe_pipe_b_3;
  wire        pipeFire_pipe_pipe_out_3_bits = pipeFire_pipe_pipe_b_3;
  wire [1:0]  _GEN_3 = _GEN_1 | bankCorrect_3;
  wire [1:0]  _GEN_4 = bankCorrect_3 & _GEN_1 | _GEN_2;
  wire        _readRecord_T_3960 = chainingRecord_0_bits_instIndex == readRequests_4_bits_instructionIndex_0;
  wire        _readRecord_T_3961 = chainingRecord_1_bits_instIndex == readRequests_4_bits_instructionIndex_0;
  wire        _readRecord_T_3962 = chainingRecord_2_bits_instIndex == readRequests_4_bits_instructionIndex_0;
  wire        _readRecord_T_3963 = chainingRecord_3_bits_instIndex == readRequests_4_bits_instructionIndex_0;
  wire        _readRecord_T_3964 = chainingRecord_4_bits_instIndex == readRequests_4_bits_instructionIndex_0;
  wire        readRecord_18_state_wLaneClear =
    _readRecord_T_3960 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_3961 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_3962 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_3963
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_3964 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_18_state_wTopLastReport =
    _readRecord_T_3960 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_3961 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_3962 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_3963
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_3964 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_18_state_wLaneLastReport =
    _readRecord_T_3960 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_3961 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_3962 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_3963
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_3964 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_18_state_wWriteQueueClear =
    _readRecord_T_3960 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_3961 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_3962 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_3963
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_3964 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_18_state_stFinish =
    _readRecord_T_3960 & chainingRecord_0_bits_state_stFinish | _readRecord_T_3961 & chainingRecord_1_bits_state_stFinish | _readRecord_T_3962 & chainingRecord_2_bits_state_stFinish | _readRecord_T_3963
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_3964 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_18_elementMask =
    (_readRecord_T_3960 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_3961 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_3962 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_3963 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_3964 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_18_slow =
    _readRecord_T_3960 & chainingRecord_0_bits_slow | _readRecord_T_3961 & chainingRecord_1_bits_slow | _readRecord_T_3962 & chainingRecord_2_bits_slow | _readRecord_T_3963 & chainingRecord_3_bits_slow | _readRecord_T_3964
    & chainingRecord_4_bits_slow;
  wire        readRecord_18_onlyRead =
    _readRecord_T_3960 & chainingRecord_0_bits_onlyRead | _readRecord_T_3961 & chainingRecord_1_bits_onlyRead | _readRecord_T_3962 & chainingRecord_2_bits_onlyRead | _readRecord_T_3963 & chainingRecord_3_bits_onlyRead | _readRecord_T_3964
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_18_ma =
    _readRecord_T_3960 & chainingRecord_0_bits_ma | _readRecord_T_3961 & chainingRecord_1_bits_ma | _readRecord_T_3962 & chainingRecord_2_bits_ma | _readRecord_T_3963 & chainingRecord_3_bits_ma | _readRecord_T_3964
    & chainingRecord_4_bits_ma;
  wire        readRecord_18_indexType =
    _readRecord_T_3960 & chainingRecord_0_bits_indexType | _readRecord_T_3961 & chainingRecord_1_bits_indexType | _readRecord_T_3962 & chainingRecord_2_bits_indexType | _readRecord_T_3963 & chainingRecord_3_bits_indexType
    | _readRecord_T_3964 & chainingRecord_4_bits_indexType;
  wire        readRecord_18_crossRead =
    _readRecord_T_3960 & chainingRecord_0_bits_crossRead | _readRecord_T_3961 & chainingRecord_1_bits_crossRead | _readRecord_T_3962 & chainingRecord_2_bits_crossRead | _readRecord_T_3963 & chainingRecord_3_bits_crossRead
    | _readRecord_T_3964 & chainingRecord_4_bits_crossRead;
  wire        readRecord_18_crossWrite =
    _readRecord_T_3960 & chainingRecord_0_bits_crossWrite | _readRecord_T_3961 & chainingRecord_1_bits_crossWrite | _readRecord_T_3962 & chainingRecord_2_bits_crossWrite | _readRecord_T_3963 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_3964 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_18_gather16 =
    _readRecord_T_3960 & chainingRecord_0_bits_gather16 | _readRecord_T_3961 & chainingRecord_1_bits_gather16 | _readRecord_T_3962 & chainingRecord_2_bits_gather16 | _readRecord_T_3963 & chainingRecord_3_bits_gather16 | _readRecord_T_3964
    & chainingRecord_4_bits_gather16;
  wire        readRecord_18_gather =
    _readRecord_T_3960 & chainingRecord_0_bits_gather | _readRecord_T_3961 & chainingRecord_1_bits_gather | _readRecord_T_3962 & chainingRecord_2_bits_gather | _readRecord_T_3963 & chainingRecord_3_bits_gather | _readRecord_T_3964
    & chainingRecord_4_bits_gather;
  wire        readRecord_18_st =
    _readRecord_T_3960 & chainingRecord_0_bits_st | _readRecord_T_3961 & chainingRecord_1_bits_st | _readRecord_T_3962 & chainingRecord_2_bits_st | _readRecord_T_3963 & chainingRecord_3_bits_st | _readRecord_T_3964
    & chainingRecord_4_bits_st;
  wire        readRecord_18_ls =
    _readRecord_T_3960 & chainingRecord_0_bits_ls | _readRecord_T_3961 & chainingRecord_1_bits_ls | _readRecord_T_3962 & chainingRecord_2_bits_ls | _readRecord_T_3963 & chainingRecord_3_bits_ls | _readRecord_T_3964
    & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_18_instIndex =
    (_readRecord_T_3960 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_3961 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_3962 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_3963 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_3964 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_18_vs2 =
    (_readRecord_T_3960 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_3961 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_3962 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_3963 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_3964 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_18_vs1_bits =
    (_readRecord_T_3960 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_3961 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_3962 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_3963 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_3964 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_18_vs1_valid =
    _readRecord_T_3960 & chainingRecord_0_bits_vs1_valid | _readRecord_T_3961 & chainingRecord_1_bits_vs1_valid | _readRecord_T_3962 & chainingRecord_2_bits_vs1_valid | _readRecord_T_3963 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_3964 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_18_vd_bits =
    (_readRecord_T_3960 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_3961 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_3962 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_3963 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_3964 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_18_vd_valid =
    _readRecord_T_3960 & chainingRecord_0_bits_vd_valid | _readRecord_T_3961 & chainingRecord_1_bits_vd_valid | _readRecord_T_3962 & chainingRecord_2_bits_vd_valid | _readRecord_T_3963 & chainingRecord_3_bits_vd_valid | _readRecord_T_3964
    & chainingRecord_4_bits_vd_valid;
  wire [5:0]  address_4 = {readRequests_4_bits_vs_0, readRequests_4_bits_offset_0};
  wire [1:0]  bank_4 = 2'h1 << address_4[0];
  reg         pipeBank_pipe_v_4;
  reg  [1:0]  pipeBank_pipe_b_4;
  reg         pipeBank_pipe_pipe_v_4;
  wire        pipeBank_pipe_pipe_out_4_valid = pipeBank_pipe_pipe_v_4;
  reg  [1:0]  pipeBank_pipe_pipe_b_4;
  wire [1:0]  pipeBank_pipe_pipe_out_4_bits = pipeBank_pipe_pipe_b_4;
  wire [1:0]  bankCorrect_4 = readRequests_4_valid_0 ? bank_4 : 2'h0;
  wire        portConflictCheck_4 =
    ~(write_valid_0 & bank_4 == writeBank & write_bits_vd_0 == readRequests_4_bits_vs_0 & write_bits_offset_0 == readRequests_4_bits_offset_0 | writePipe_valid & bank_4 == writeBankPipe & writePipe_bits_vd == readRequests_4_bits_vs_0
      & writePipe_bits_offset == readRequests_4_bits_offset_0);
  wire        portReady_4 = |(bank_4 & ~_GEN_4);
  assign readRequests_4_ready_0 = portReady_4 & sramReady;
  wire        firstUsed_4 = |(bank_4 & _GEN_3);
  wire [1:0]  bankReadF_4 = bankCorrect_4 & ~_GEN_3;
  wire [1:0]  bankReadS_4 = bankCorrect_4 & ~_GEN_4 & _GEN_3;
  reg         pipeFirstUsed_pipe_v_4;
  reg         pipeFirstUsed_pipe_b_4;
  reg         pipeFirstUsed_pipe_pipe_v_4;
  wire        pipeFirstUsed_pipe_pipe_out_4_valid = pipeFirstUsed_pipe_pipe_v_4;
  reg         pipeFirstUsed_pipe_pipe_b_4;
  wire        pipeFirstUsed_pipe_pipe_out_4_bits = pipeFirstUsed_pipe_pipe_b_4;
  reg         pipeFire_pipe_v_4;
  reg         pipeFire_pipe_b_4;
  reg         pipeFire_pipe_pipe_v_4;
  wire        pipeFire_pipe_pipe_out_4_valid = pipeFire_pipe_pipe_v_4;
  reg         pipeFire_pipe_pipe_b_4;
  wire        pipeFire_pipe_pipe_out_4_bits = pipeFire_pipe_pipe_b_4;
  wire [1:0]  _GEN_5 = _GEN_3 | bankCorrect_4;
  wire [1:0]  _GEN_6 = bankCorrect_4 & _GEN_3 | _GEN_4;
  wire        _readRecord_T_4180 = chainingRecord_0_bits_instIndex == readRequests_5_bits_instructionIndex_0;
  wire        _readRecord_T_4181 = chainingRecord_1_bits_instIndex == readRequests_5_bits_instructionIndex_0;
  wire        _readRecord_T_4182 = chainingRecord_2_bits_instIndex == readRequests_5_bits_instructionIndex_0;
  wire        _readRecord_T_4183 = chainingRecord_3_bits_instIndex == readRequests_5_bits_instructionIndex_0;
  wire        _readRecord_T_4184 = chainingRecord_4_bits_instIndex == readRequests_5_bits_instructionIndex_0;
  wire        readRecord_19_state_wLaneClear =
    _readRecord_T_4180 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_4181 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_4182 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_4183
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_4184 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_19_state_wTopLastReport =
    _readRecord_T_4180 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_4181 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_4182 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_4183
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_4184 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_19_state_wLaneLastReport =
    _readRecord_T_4180 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_4181 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_4182 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_4183
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_4184 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_19_state_wWriteQueueClear =
    _readRecord_T_4180 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_4181 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_4182 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_4183
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_4184 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_19_state_stFinish =
    _readRecord_T_4180 & chainingRecord_0_bits_state_stFinish | _readRecord_T_4181 & chainingRecord_1_bits_state_stFinish | _readRecord_T_4182 & chainingRecord_2_bits_state_stFinish | _readRecord_T_4183
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_4184 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_19_elementMask =
    (_readRecord_T_4180 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_4181 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_4182 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_4183 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_4184 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_19_slow =
    _readRecord_T_4180 & chainingRecord_0_bits_slow | _readRecord_T_4181 & chainingRecord_1_bits_slow | _readRecord_T_4182 & chainingRecord_2_bits_slow | _readRecord_T_4183 & chainingRecord_3_bits_slow | _readRecord_T_4184
    & chainingRecord_4_bits_slow;
  wire        readRecord_19_onlyRead =
    _readRecord_T_4180 & chainingRecord_0_bits_onlyRead | _readRecord_T_4181 & chainingRecord_1_bits_onlyRead | _readRecord_T_4182 & chainingRecord_2_bits_onlyRead | _readRecord_T_4183 & chainingRecord_3_bits_onlyRead | _readRecord_T_4184
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_19_ma =
    _readRecord_T_4180 & chainingRecord_0_bits_ma | _readRecord_T_4181 & chainingRecord_1_bits_ma | _readRecord_T_4182 & chainingRecord_2_bits_ma | _readRecord_T_4183 & chainingRecord_3_bits_ma | _readRecord_T_4184
    & chainingRecord_4_bits_ma;
  wire        readRecord_19_indexType =
    _readRecord_T_4180 & chainingRecord_0_bits_indexType | _readRecord_T_4181 & chainingRecord_1_bits_indexType | _readRecord_T_4182 & chainingRecord_2_bits_indexType | _readRecord_T_4183 & chainingRecord_3_bits_indexType
    | _readRecord_T_4184 & chainingRecord_4_bits_indexType;
  wire        readRecord_19_crossRead =
    _readRecord_T_4180 & chainingRecord_0_bits_crossRead | _readRecord_T_4181 & chainingRecord_1_bits_crossRead | _readRecord_T_4182 & chainingRecord_2_bits_crossRead | _readRecord_T_4183 & chainingRecord_3_bits_crossRead
    | _readRecord_T_4184 & chainingRecord_4_bits_crossRead;
  wire        readRecord_19_crossWrite =
    _readRecord_T_4180 & chainingRecord_0_bits_crossWrite | _readRecord_T_4181 & chainingRecord_1_bits_crossWrite | _readRecord_T_4182 & chainingRecord_2_bits_crossWrite | _readRecord_T_4183 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_4184 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_19_gather16 =
    _readRecord_T_4180 & chainingRecord_0_bits_gather16 | _readRecord_T_4181 & chainingRecord_1_bits_gather16 | _readRecord_T_4182 & chainingRecord_2_bits_gather16 | _readRecord_T_4183 & chainingRecord_3_bits_gather16 | _readRecord_T_4184
    & chainingRecord_4_bits_gather16;
  wire        readRecord_19_gather =
    _readRecord_T_4180 & chainingRecord_0_bits_gather | _readRecord_T_4181 & chainingRecord_1_bits_gather | _readRecord_T_4182 & chainingRecord_2_bits_gather | _readRecord_T_4183 & chainingRecord_3_bits_gather | _readRecord_T_4184
    & chainingRecord_4_bits_gather;
  wire        readRecord_19_st =
    _readRecord_T_4180 & chainingRecord_0_bits_st | _readRecord_T_4181 & chainingRecord_1_bits_st | _readRecord_T_4182 & chainingRecord_2_bits_st | _readRecord_T_4183 & chainingRecord_3_bits_st | _readRecord_T_4184
    & chainingRecord_4_bits_st;
  wire        readRecord_19_ls =
    _readRecord_T_4180 & chainingRecord_0_bits_ls | _readRecord_T_4181 & chainingRecord_1_bits_ls | _readRecord_T_4182 & chainingRecord_2_bits_ls | _readRecord_T_4183 & chainingRecord_3_bits_ls | _readRecord_T_4184
    & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_19_instIndex =
    (_readRecord_T_4180 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_4181 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_4182 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_4183 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_4184 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_19_vs2 =
    (_readRecord_T_4180 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_4181 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_4182 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_4183 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_4184 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_19_vs1_bits =
    (_readRecord_T_4180 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_4181 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_4182 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_4183 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_4184 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_19_vs1_valid =
    _readRecord_T_4180 & chainingRecord_0_bits_vs1_valid | _readRecord_T_4181 & chainingRecord_1_bits_vs1_valid | _readRecord_T_4182 & chainingRecord_2_bits_vs1_valid | _readRecord_T_4183 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_4184 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_19_vd_bits =
    (_readRecord_T_4180 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_4181 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_4182 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_4183 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_4184 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_19_vd_valid =
    _readRecord_T_4180 & chainingRecord_0_bits_vd_valid | _readRecord_T_4181 & chainingRecord_1_bits_vd_valid | _readRecord_T_4182 & chainingRecord_2_bits_vd_valid | _readRecord_T_4183 & chainingRecord_3_bits_vd_valid | _readRecord_T_4184
    & chainingRecord_4_bits_vd_valid;
  wire [5:0]  address_5 = {readRequests_5_bits_vs_0, readRequests_5_bits_offset_0};
  wire [1:0]  bank_5 = 2'h1 << address_5[0];
  reg         pipeBank_pipe_v_5;
  reg  [1:0]  pipeBank_pipe_b_5;
  reg         pipeBank_pipe_pipe_v_5;
  wire        pipeBank_pipe_pipe_out_5_valid = pipeBank_pipe_pipe_v_5;
  reg  [1:0]  pipeBank_pipe_pipe_b_5;
  wire [1:0]  pipeBank_pipe_pipe_out_5_bits = pipeBank_pipe_pipe_b_5;
  wire [1:0]  bankCorrect_5 = readRequests_5_valid_0 ? bank_5 : 2'h0;
  wire        portConflictCheck_5 =
    ~(write_valid_0 & bank_5 == writeBank & write_bits_vd_0 == readRequests_5_bits_vs_0 & write_bits_offset_0 == readRequests_5_bits_offset_0 | writePipe_valid & bank_5 == writeBankPipe & writePipe_bits_vd == readRequests_5_bits_vs_0
      & writePipe_bits_offset == readRequests_5_bits_offset_0);
  wire        portReady_5 = |(bank_5 & ~_GEN_6);
  assign readRequests_5_ready_0 = portReady_5 & sramReady;
  wire        firstUsed_5 = |(bank_5 & _GEN_5);
  wire [1:0]  bankReadF_5 = bankCorrect_5 & ~_GEN_5;
  wire [1:0]  bankReadS_5 = bankCorrect_5 & ~_GEN_6 & _GEN_5;
  reg         pipeFirstUsed_pipe_v_5;
  reg         pipeFirstUsed_pipe_b_5;
  reg         pipeFirstUsed_pipe_pipe_v_5;
  wire        pipeFirstUsed_pipe_pipe_out_5_valid = pipeFirstUsed_pipe_pipe_v_5;
  reg         pipeFirstUsed_pipe_pipe_b_5;
  wire        pipeFirstUsed_pipe_pipe_out_5_bits = pipeFirstUsed_pipe_pipe_b_5;
  reg         pipeFire_pipe_v_5;
  reg         pipeFire_pipe_b_5;
  reg         pipeFire_pipe_pipe_v_5;
  wire        pipeFire_pipe_pipe_out_5_valid = pipeFire_pipe_pipe_v_5;
  reg         pipeFire_pipe_pipe_b_5;
  wire        pipeFire_pipe_pipe_out_5_bits = pipeFire_pipe_pipe_b_5;
  wire [1:0]  _GEN_7 = _GEN_5 | bankCorrect_5;
  wire [1:0]  _GEN_8 = bankCorrect_5 & _GEN_5 | _GEN_6;
  wire        _readRecord_T_4400 = chainingRecord_0_bits_instIndex == readRequests_6_bits_instructionIndex_0;
  wire        _readRecord_T_4401 = chainingRecord_1_bits_instIndex == readRequests_6_bits_instructionIndex_0;
  wire        _readRecord_T_4402 = chainingRecord_2_bits_instIndex == readRequests_6_bits_instructionIndex_0;
  wire        _readRecord_T_4403 = chainingRecord_3_bits_instIndex == readRequests_6_bits_instructionIndex_0;
  wire        _readRecord_T_4404 = chainingRecord_4_bits_instIndex == readRequests_6_bits_instructionIndex_0;
  wire        readRecord_20_state_wLaneClear =
    _readRecord_T_4400 & chainingRecord_0_bits_state_wLaneClear | _readRecord_T_4401 & chainingRecord_1_bits_state_wLaneClear | _readRecord_T_4402 & chainingRecord_2_bits_state_wLaneClear | _readRecord_T_4403
    & chainingRecord_3_bits_state_wLaneClear | _readRecord_T_4404 & chainingRecord_4_bits_state_wLaneClear;
  wire        readRecord_20_state_wTopLastReport =
    _readRecord_T_4400 & chainingRecord_0_bits_state_wTopLastReport | _readRecord_T_4401 & chainingRecord_1_bits_state_wTopLastReport | _readRecord_T_4402 & chainingRecord_2_bits_state_wTopLastReport | _readRecord_T_4403
    & chainingRecord_3_bits_state_wTopLastReport | _readRecord_T_4404 & chainingRecord_4_bits_state_wTopLastReport;
  wire        readRecord_20_state_wLaneLastReport =
    _readRecord_T_4400 & chainingRecord_0_bits_state_wLaneLastReport | _readRecord_T_4401 & chainingRecord_1_bits_state_wLaneLastReport | _readRecord_T_4402 & chainingRecord_2_bits_state_wLaneLastReport | _readRecord_T_4403
    & chainingRecord_3_bits_state_wLaneLastReport | _readRecord_T_4404 & chainingRecord_4_bits_state_wLaneLastReport;
  wire        readRecord_20_state_wWriteQueueClear =
    _readRecord_T_4400 & chainingRecord_0_bits_state_wWriteQueueClear | _readRecord_T_4401 & chainingRecord_1_bits_state_wWriteQueueClear | _readRecord_T_4402 & chainingRecord_2_bits_state_wWriteQueueClear | _readRecord_T_4403
    & chainingRecord_3_bits_state_wWriteQueueClear | _readRecord_T_4404 & chainingRecord_4_bits_state_wWriteQueueClear;
  wire        readRecord_20_state_stFinish =
    _readRecord_T_4400 & chainingRecord_0_bits_state_stFinish | _readRecord_T_4401 & chainingRecord_1_bits_state_stFinish | _readRecord_T_4402 & chainingRecord_2_bits_state_stFinish | _readRecord_T_4403
    & chainingRecord_3_bits_state_stFinish | _readRecord_T_4404 & chainingRecord_4_bits_state_stFinish;
  wire [15:0] readRecord_20_elementMask =
    (_readRecord_T_4400 ? chainingRecord_0_bits_elementMask : 16'h0) | (_readRecord_T_4401 ? chainingRecord_1_bits_elementMask : 16'h0) | (_readRecord_T_4402 ? chainingRecord_2_bits_elementMask : 16'h0)
    | (_readRecord_T_4403 ? chainingRecord_3_bits_elementMask : 16'h0) | (_readRecord_T_4404 ? chainingRecord_4_bits_elementMask : 16'h0);
  wire        readRecord_20_slow =
    _readRecord_T_4400 & chainingRecord_0_bits_slow | _readRecord_T_4401 & chainingRecord_1_bits_slow | _readRecord_T_4402 & chainingRecord_2_bits_slow | _readRecord_T_4403 & chainingRecord_3_bits_slow | _readRecord_T_4404
    & chainingRecord_4_bits_slow;
  wire        readRecord_20_onlyRead =
    _readRecord_T_4400 & chainingRecord_0_bits_onlyRead | _readRecord_T_4401 & chainingRecord_1_bits_onlyRead | _readRecord_T_4402 & chainingRecord_2_bits_onlyRead | _readRecord_T_4403 & chainingRecord_3_bits_onlyRead | _readRecord_T_4404
    & chainingRecord_4_bits_onlyRead;
  wire        readRecord_20_ma =
    _readRecord_T_4400 & chainingRecord_0_bits_ma | _readRecord_T_4401 & chainingRecord_1_bits_ma | _readRecord_T_4402 & chainingRecord_2_bits_ma | _readRecord_T_4403 & chainingRecord_3_bits_ma | _readRecord_T_4404
    & chainingRecord_4_bits_ma;
  wire        readRecord_20_indexType =
    _readRecord_T_4400 & chainingRecord_0_bits_indexType | _readRecord_T_4401 & chainingRecord_1_bits_indexType | _readRecord_T_4402 & chainingRecord_2_bits_indexType | _readRecord_T_4403 & chainingRecord_3_bits_indexType
    | _readRecord_T_4404 & chainingRecord_4_bits_indexType;
  wire        readRecord_20_crossRead =
    _readRecord_T_4400 & chainingRecord_0_bits_crossRead | _readRecord_T_4401 & chainingRecord_1_bits_crossRead | _readRecord_T_4402 & chainingRecord_2_bits_crossRead | _readRecord_T_4403 & chainingRecord_3_bits_crossRead
    | _readRecord_T_4404 & chainingRecord_4_bits_crossRead;
  wire        readRecord_20_crossWrite =
    _readRecord_T_4400 & chainingRecord_0_bits_crossWrite | _readRecord_T_4401 & chainingRecord_1_bits_crossWrite | _readRecord_T_4402 & chainingRecord_2_bits_crossWrite | _readRecord_T_4403 & chainingRecord_3_bits_crossWrite
    | _readRecord_T_4404 & chainingRecord_4_bits_crossWrite;
  wire        readRecord_20_gather16 =
    _readRecord_T_4400 & chainingRecord_0_bits_gather16 | _readRecord_T_4401 & chainingRecord_1_bits_gather16 | _readRecord_T_4402 & chainingRecord_2_bits_gather16 | _readRecord_T_4403 & chainingRecord_3_bits_gather16 | _readRecord_T_4404
    & chainingRecord_4_bits_gather16;
  wire        readRecord_20_gather =
    _readRecord_T_4400 & chainingRecord_0_bits_gather | _readRecord_T_4401 & chainingRecord_1_bits_gather | _readRecord_T_4402 & chainingRecord_2_bits_gather | _readRecord_T_4403 & chainingRecord_3_bits_gather | _readRecord_T_4404
    & chainingRecord_4_bits_gather;
  wire        readRecord_20_st =
    _readRecord_T_4400 & chainingRecord_0_bits_st | _readRecord_T_4401 & chainingRecord_1_bits_st | _readRecord_T_4402 & chainingRecord_2_bits_st | _readRecord_T_4403 & chainingRecord_3_bits_st | _readRecord_T_4404
    & chainingRecord_4_bits_st;
  wire        readRecord_20_ls =
    _readRecord_T_4400 & chainingRecord_0_bits_ls | _readRecord_T_4401 & chainingRecord_1_bits_ls | _readRecord_T_4402 & chainingRecord_2_bits_ls | _readRecord_T_4403 & chainingRecord_3_bits_ls | _readRecord_T_4404
    & chainingRecord_4_bits_ls;
  wire [2:0]  readRecord_20_instIndex =
    (_readRecord_T_4400 ? chainingRecord_0_bits_instIndex : 3'h0) | (_readRecord_T_4401 ? chainingRecord_1_bits_instIndex : 3'h0) | (_readRecord_T_4402 ? chainingRecord_2_bits_instIndex : 3'h0)
    | (_readRecord_T_4403 ? chainingRecord_3_bits_instIndex : 3'h0) | (_readRecord_T_4404 ? chainingRecord_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_20_vs2 =
    (_readRecord_T_4400 ? chainingRecord_0_bits_vs2 : 5'h0) | (_readRecord_T_4401 ? chainingRecord_1_bits_vs2 : 5'h0) | (_readRecord_T_4402 ? chainingRecord_2_bits_vs2 : 5'h0) | (_readRecord_T_4403 ? chainingRecord_3_bits_vs2 : 5'h0)
    | (_readRecord_T_4404 ? chainingRecord_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_20_vs1_bits =
    (_readRecord_T_4400 ? chainingRecord_0_bits_vs1_bits : 5'h0) | (_readRecord_T_4401 ? chainingRecord_1_bits_vs1_bits : 5'h0) | (_readRecord_T_4402 ? chainingRecord_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_4403 ? chainingRecord_3_bits_vs1_bits : 5'h0) | (_readRecord_T_4404 ? chainingRecord_4_bits_vs1_bits : 5'h0);
  wire        readRecord_20_vs1_valid =
    _readRecord_T_4400 & chainingRecord_0_bits_vs1_valid | _readRecord_T_4401 & chainingRecord_1_bits_vs1_valid | _readRecord_T_4402 & chainingRecord_2_bits_vs1_valid | _readRecord_T_4403 & chainingRecord_3_bits_vs1_valid
    | _readRecord_T_4404 & chainingRecord_4_bits_vs1_valid;
  wire [4:0]  readRecord_20_vd_bits =
    (_readRecord_T_4400 ? chainingRecord_0_bits_vd_bits : 5'h0) | (_readRecord_T_4401 ? chainingRecord_1_bits_vd_bits : 5'h0) | (_readRecord_T_4402 ? chainingRecord_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_4403 ? chainingRecord_3_bits_vd_bits : 5'h0) | (_readRecord_T_4404 ? chainingRecord_4_bits_vd_bits : 5'h0);
  wire        readRecord_20_vd_valid =
    _readRecord_T_4400 & chainingRecord_0_bits_vd_valid | _readRecord_T_4401 & chainingRecord_1_bits_vd_valid | _readRecord_T_4402 & chainingRecord_2_bits_vd_valid | _readRecord_T_4403 & chainingRecord_3_bits_vd_valid | _readRecord_T_4404
    & chainingRecord_4_bits_vd_valid;
  wire [5:0]  address_6 = {readRequests_6_bits_vs_0, readRequests_6_bits_offset_0};
  wire [1:0]  bank_6 = 2'h1 << address_6[0];
  reg         pipeBank_pipe_v_6;
  reg  [1:0]  pipeBank_pipe_b_6;
  reg         pipeBank_pipe_pipe_v_6;
  wire        pipeBank_pipe_pipe_out_6_valid = pipeBank_pipe_pipe_v_6;
  reg  [1:0]  pipeBank_pipe_pipe_b_6;
  wire [1:0]  pipeBank_pipe_pipe_out_6_bits = pipeBank_pipe_pipe_b_6;
  wire [1:0]  bankCorrect_6 = readRequests_6_valid_0 ? bank_6 : 2'h0;
  wire        portConflictCheck_6 =
    ~(write_valid_0 & bank_6 == writeBank & write_bits_vd_0 == readRequests_6_bits_vs_0 & write_bits_offset_0 == readRequests_6_bits_offset_0 | writePipe_valid & bank_6 == writeBankPipe & writePipe_bits_vd == readRequests_6_bits_vs_0
      & writePipe_bits_offset == readRequests_6_bits_offset_0);
  wire        portReady_6 = |(bank_6 & ~_GEN_8);
  assign readRequests_6_ready_0 = portReady_6 & sramReady;
  wire        firstUsed_6 = |(bank_6 & _GEN_7);
  wire [1:0]  bankReadF_6 = bankCorrect_6 & ~_GEN_7;
  wire [1:0]  bankReadS_6 = bankCorrect_6 & ~_GEN_8 & _GEN_7;
  reg         pipeFirstUsed_pipe_v_6;
  reg         pipeFirstUsed_pipe_b_6;
  reg         pipeFirstUsed_pipe_pipe_v_6;
  wire        pipeFirstUsed_pipe_pipe_out_6_valid = pipeFirstUsed_pipe_pipe_v_6;
  reg         pipeFirstUsed_pipe_pipe_b_6;
  wire        pipeFirstUsed_pipe_pipe_out_6_bits = pipeFirstUsed_pipe_pipe_b_6;
  reg         pipeFire_pipe_v_6;
  reg         pipeFire_pipe_b_6;
  reg         pipeFire_pipe_pipe_v_6;
  wire        pipeFire_pipe_pipe_out_6_valid = pipeFire_pipe_pipe_v_6;
  reg         pipeFire_pipe_pipe_b_6;
  wire        pipeFire_pipe_pipe_out_6_bits = pipeFire_pipe_pipe_b_6;
  wire [1:0]  _GEN_9 = _GEN_7 | bankCorrect_6;
  wire [1:0]  _GEN_10 = bankCorrect_6 & _GEN_7 | _GEN_8;
  wire        _readRecord_T_4620 = chainingRecordCopy_0_bits_instIndex == readRequests_7_bits_instructionIndex_0;
  wire        _readRecord_T_4621 = chainingRecordCopy_1_bits_instIndex == readRequests_7_bits_instructionIndex_0;
  wire        _readRecord_T_4622 = chainingRecordCopy_2_bits_instIndex == readRequests_7_bits_instructionIndex_0;
  wire        _readRecord_T_4623 = chainingRecordCopy_3_bits_instIndex == readRequests_7_bits_instructionIndex_0;
  wire        _readRecord_T_4624 = chainingRecordCopy_4_bits_instIndex == readRequests_7_bits_instructionIndex_0;
  wire        readRecord_21_state_wLaneClear =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_state_wLaneClear | _readRecord_T_4621 & chainingRecordCopy_1_bits_state_wLaneClear | _readRecord_T_4622 & chainingRecordCopy_2_bits_state_wLaneClear | _readRecord_T_4623
    & chainingRecordCopy_3_bits_state_wLaneClear | _readRecord_T_4624 & chainingRecordCopy_4_bits_state_wLaneClear;
  wire        readRecord_21_state_wTopLastReport =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_state_wTopLastReport | _readRecord_T_4621 & chainingRecordCopy_1_bits_state_wTopLastReport | _readRecord_T_4622 & chainingRecordCopy_2_bits_state_wTopLastReport | _readRecord_T_4623
    & chainingRecordCopy_3_bits_state_wTopLastReport | _readRecord_T_4624 & chainingRecordCopy_4_bits_state_wTopLastReport;
  wire        readRecord_21_state_wLaneLastReport =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_state_wLaneLastReport | _readRecord_T_4621 & chainingRecordCopy_1_bits_state_wLaneLastReport | _readRecord_T_4622 & chainingRecordCopy_2_bits_state_wLaneLastReport | _readRecord_T_4623
    & chainingRecordCopy_3_bits_state_wLaneLastReport | _readRecord_T_4624 & chainingRecordCopy_4_bits_state_wLaneLastReport;
  wire        readRecord_21_state_wWriteQueueClear =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_state_wWriteQueueClear | _readRecord_T_4621 & chainingRecordCopy_1_bits_state_wWriteQueueClear | _readRecord_T_4622 & chainingRecordCopy_2_bits_state_wWriteQueueClear | _readRecord_T_4623
    & chainingRecordCopy_3_bits_state_wWriteQueueClear | _readRecord_T_4624 & chainingRecordCopy_4_bits_state_wWriteQueueClear;
  wire        readRecord_21_state_stFinish =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_state_stFinish | _readRecord_T_4621 & chainingRecordCopy_1_bits_state_stFinish | _readRecord_T_4622 & chainingRecordCopy_2_bits_state_stFinish | _readRecord_T_4623
    & chainingRecordCopy_3_bits_state_stFinish | _readRecord_T_4624 & chainingRecordCopy_4_bits_state_stFinish;
  wire [15:0] readRecord_21_elementMask =
    (_readRecord_T_4620 ? chainingRecordCopy_0_bits_elementMask : 16'h0) | (_readRecord_T_4621 ? chainingRecordCopy_1_bits_elementMask : 16'h0) | (_readRecord_T_4622 ? chainingRecordCopy_2_bits_elementMask : 16'h0)
    | (_readRecord_T_4623 ? chainingRecordCopy_3_bits_elementMask : 16'h0) | (_readRecord_T_4624 ? chainingRecordCopy_4_bits_elementMask : 16'h0);
  wire        readRecord_21_slow =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_slow | _readRecord_T_4621 & chainingRecordCopy_1_bits_slow | _readRecord_T_4622 & chainingRecordCopy_2_bits_slow | _readRecord_T_4623 & chainingRecordCopy_3_bits_slow | _readRecord_T_4624
    & chainingRecordCopy_4_bits_slow;
  wire        readRecord_21_onlyRead =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_onlyRead | _readRecord_T_4621 & chainingRecordCopy_1_bits_onlyRead | _readRecord_T_4622 & chainingRecordCopy_2_bits_onlyRead | _readRecord_T_4623 & chainingRecordCopy_3_bits_onlyRead
    | _readRecord_T_4624 & chainingRecordCopy_4_bits_onlyRead;
  wire        readRecord_21_ma =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_ma | _readRecord_T_4621 & chainingRecordCopy_1_bits_ma | _readRecord_T_4622 & chainingRecordCopy_2_bits_ma | _readRecord_T_4623 & chainingRecordCopy_3_bits_ma | _readRecord_T_4624
    & chainingRecordCopy_4_bits_ma;
  wire        readRecord_21_indexType =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_indexType | _readRecord_T_4621 & chainingRecordCopy_1_bits_indexType | _readRecord_T_4622 & chainingRecordCopy_2_bits_indexType | _readRecord_T_4623 & chainingRecordCopy_3_bits_indexType
    | _readRecord_T_4624 & chainingRecordCopy_4_bits_indexType;
  wire        readRecord_21_crossRead =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_crossRead | _readRecord_T_4621 & chainingRecordCopy_1_bits_crossRead | _readRecord_T_4622 & chainingRecordCopy_2_bits_crossRead | _readRecord_T_4623 & chainingRecordCopy_3_bits_crossRead
    | _readRecord_T_4624 & chainingRecordCopy_4_bits_crossRead;
  wire        readRecord_21_crossWrite =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_crossWrite | _readRecord_T_4621 & chainingRecordCopy_1_bits_crossWrite | _readRecord_T_4622 & chainingRecordCopy_2_bits_crossWrite | _readRecord_T_4623
    & chainingRecordCopy_3_bits_crossWrite | _readRecord_T_4624 & chainingRecordCopy_4_bits_crossWrite;
  wire        readRecord_21_gather16 =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_gather16 | _readRecord_T_4621 & chainingRecordCopy_1_bits_gather16 | _readRecord_T_4622 & chainingRecordCopy_2_bits_gather16 | _readRecord_T_4623 & chainingRecordCopy_3_bits_gather16
    | _readRecord_T_4624 & chainingRecordCopy_4_bits_gather16;
  wire        readRecord_21_gather =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_gather | _readRecord_T_4621 & chainingRecordCopy_1_bits_gather | _readRecord_T_4622 & chainingRecordCopy_2_bits_gather | _readRecord_T_4623 & chainingRecordCopy_3_bits_gather
    | _readRecord_T_4624 & chainingRecordCopy_4_bits_gather;
  wire        readRecord_21_st =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_st | _readRecord_T_4621 & chainingRecordCopy_1_bits_st | _readRecord_T_4622 & chainingRecordCopy_2_bits_st | _readRecord_T_4623 & chainingRecordCopy_3_bits_st | _readRecord_T_4624
    & chainingRecordCopy_4_bits_st;
  wire        readRecord_21_ls =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_ls | _readRecord_T_4621 & chainingRecordCopy_1_bits_ls | _readRecord_T_4622 & chainingRecordCopy_2_bits_ls | _readRecord_T_4623 & chainingRecordCopy_3_bits_ls | _readRecord_T_4624
    & chainingRecordCopy_4_bits_ls;
  wire [2:0]  readRecord_21_instIndex =
    (_readRecord_T_4620 ? chainingRecordCopy_0_bits_instIndex : 3'h0) | (_readRecord_T_4621 ? chainingRecordCopy_1_bits_instIndex : 3'h0) | (_readRecord_T_4622 ? chainingRecordCopy_2_bits_instIndex : 3'h0)
    | (_readRecord_T_4623 ? chainingRecordCopy_3_bits_instIndex : 3'h0) | (_readRecord_T_4624 ? chainingRecordCopy_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_21_vs2 =
    (_readRecord_T_4620 ? chainingRecordCopy_0_bits_vs2 : 5'h0) | (_readRecord_T_4621 ? chainingRecordCopy_1_bits_vs2 : 5'h0) | (_readRecord_T_4622 ? chainingRecordCopy_2_bits_vs2 : 5'h0)
    | (_readRecord_T_4623 ? chainingRecordCopy_3_bits_vs2 : 5'h0) | (_readRecord_T_4624 ? chainingRecordCopy_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_21_vs1_bits =
    (_readRecord_T_4620 ? chainingRecordCopy_0_bits_vs1_bits : 5'h0) | (_readRecord_T_4621 ? chainingRecordCopy_1_bits_vs1_bits : 5'h0) | (_readRecord_T_4622 ? chainingRecordCopy_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_4623 ? chainingRecordCopy_3_bits_vs1_bits : 5'h0) | (_readRecord_T_4624 ? chainingRecordCopy_4_bits_vs1_bits : 5'h0);
  wire        readRecord_21_vs1_valid =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_vs1_valid | _readRecord_T_4621 & chainingRecordCopy_1_bits_vs1_valid | _readRecord_T_4622 & chainingRecordCopy_2_bits_vs1_valid | _readRecord_T_4623 & chainingRecordCopy_3_bits_vs1_valid
    | _readRecord_T_4624 & chainingRecordCopy_4_bits_vs1_valid;
  wire [4:0]  readRecord_21_vd_bits =
    (_readRecord_T_4620 ? chainingRecordCopy_0_bits_vd_bits : 5'h0) | (_readRecord_T_4621 ? chainingRecordCopy_1_bits_vd_bits : 5'h0) | (_readRecord_T_4622 ? chainingRecordCopy_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_4623 ? chainingRecordCopy_3_bits_vd_bits : 5'h0) | (_readRecord_T_4624 ? chainingRecordCopy_4_bits_vd_bits : 5'h0);
  wire        readRecord_21_vd_valid =
    _readRecord_T_4620 & chainingRecordCopy_0_bits_vd_valid | _readRecord_T_4621 & chainingRecordCopy_1_bits_vd_valid | _readRecord_T_4622 & chainingRecordCopy_2_bits_vd_valid | _readRecord_T_4623 & chainingRecordCopy_3_bits_vd_valid
    | _readRecord_T_4624 & chainingRecordCopy_4_bits_vd_valid;
  wire [5:0]  address_7 = {readRequests_7_bits_vs_0, readRequests_7_bits_offset_0};
  wire [1:0]  bank_7 = 2'h1 << address_7[0];
  reg         pipeBank_pipe_v_7;
  reg  [1:0]  pipeBank_pipe_b_7;
  reg         pipeBank_pipe_pipe_v_7;
  wire        pipeBank_pipe_pipe_out_7_valid = pipeBank_pipe_pipe_v_7;
  reg  [1:0]  pipeBank_pipe_pipe_b_7;
  wire [1:0]  pipeBank_pipe_pipe_out_7_bits = pipeBank_pipe_pipe_b_7;
  wire [1:0]  bankCorrect_7 = readRequests_7_valid_0 ? bank_7 : 2'h0;
  wire        portConflictCheck_7 =
    ~(write_valid_0 & bank_7 == writeBank & write_bits_vd_0 == readRequests_7_bits_vs_0 & write_bits_offset_0 == readRequests_7_bits_offset_0 | writePipe_valid & bank_7 == writeBankPipe & writePipe_bits_vd == readRequests_7_bits_vs_0
      & writePipe_bits_offset == readRequests_7_bits_offset_0);
  wire        portReady_7 = |(bank_7 & ~_GEN_10);
  assign readRequests_7_ready_0 = portReady_7 & sramReady;
  wire        firstUsed_7 = |(bank_7 & _GEN_9);
  wire [1:0]  bankReadF_7 = bankCorrect_7 & ~_GEN_9;
  wire [1:0]  bankReadS_7 = bankCorrect_7 & ~_GEN_10 & _GEN_9;
  reg         pipeFirstUsed_pipe_v_7;
  reg         pipeFirstUsed_pipe_b_7;
  reg         pipeFirstUsed_pipe_pipe_v_7;
  wire        pipeFirstUsed_pipe_pipe_out_7_valid = pipeFirstUsed_pipe_pipe_v_7;
  reg         pipeFirstUsed_pipe_pipe_b_7;
  wire        pipeFirstUsed_pipe_pipe_out_7_bits = pipeFirstUsed_pipe_pipe_b_7;
  reg         pipeFire_pipe_v_7;
  reg         pipeFire_pipe_b_7;
  reg         pipeFire_pipe_pipe_v_7;
  wire        pipeFire_pipe_pipe_out_7_valid = pipeFire_pipe_pipe_v_7;
  reg         pipeFire_pipe_pipe_b_7;
  wire        pipeFire_pipe_pipe_out_7_bits = pipeFire_pipe_pipe_b_7;
  wire [1:0]  _GEN_11 = _GEN_9 | bankCorrect_7;
  wire [1:0]  _GEN_12 = bankCorrect_7 & _GEN_9 | _GEN_10;
  wire        _readRecord_T_4840 = chainingRecordCopy_0_bits_instIndex == readRequests_8_bits_instructionIndex_0;
  wire        _readRecord_T_4841 = chainingRecordCopy_1_bits_instIndex == readRequests_8_bits_instructionIndex_0;
  wire        _readRecord_T_4842 = chainingRecordCopy_2_bits_instIndex == readRequests_8_bits_instructionIndex_0;
  wire        _readRecord_T_4843 = chainingRecordCopy_3_bits_instIndex == readRequests_8_bits_instructionIndex_0;
  wire        _readRecord_T_4844 = chainingRecordCopy_4_bits_instIndex == readRequests_8_bits_instructionIndex_0;
  wire        readRecord_22_state_wLaneClear =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_state_wLaneClear | _readRecord_T_4841 & chainingRecordCopy_1_bits_state_wLaneClear | _readRecord_T_4842 & chainingRecordCopy_2_bits_state_wLaneClear | _readRecord_T_4843
    & chainingRecordCopy_3_bits_state_wLaneClear | _readRecord_T_4844 & chainingRecordCopy_4_bits_state_wLaneClear;
  wire        readRecord_22_state_wTopLastReport =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_state_wTopLastReport | _readRecord_T_4841 & chainingRecordCopy_1_bits_state_wTopLastReport | _readRecord_T_4842 & chainingRecordCopy_2_bits_state_wTopLastReport | _readRecord_T_4843
    & chainingRecordCopy_3_bits_state_wTopLastReport | _readRecord_T_4844 & chainingRecordCopy_4_bits_state_wTopLastReport;
  wire        readRecord_22_state_wLaneLastReport =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_state_wLaneLastReport | _readRecord_T_4841 & chainingRecordCopy_1_bits_state_wLaneLastReport | _readRecord_T_4842 & chainingRecordCopy_2_bits_state_wLaneLastReport | _readRecord_T_4843
    & chainingRecordCopy_3_bits_state_wLaneLastReport | _readRecord_T_4844 & chainingRecordCopy_4_bits_state_wLaneLastReport;
  wire        readRecord_22_state_wWriteQueueClear =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_state_wWriteQueueClear | _readRecord_T_4841 & chainingRecordCopy_1_bits_state_wWriteQueueClear | _readRecord_T_4842 & chainingRecordCopy_2_bits_state_wWriteQueueClear | _readRecord_T_4843
    & chainingRecordCopy_3_bits_state_wWriteQueueClear | _readRecord_T_4844 & chainingRecordCopy_4_bits_state_wWriteQueueClear;
  wire        readRecord_22_state_stFinish =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_state_stFinish | _readRecord_T_4841 & chainingRecordCopy_1_bits_state_stFinish | _readRecord_T_4842 & chainingRecordCopy_2_bits_state_stFinish | _readRecord_T_4843
    & chainingRecordCopy_3_bits_state_stFinish | _readRecord_T_4844 & chainingRecordCopy_4_bits_state_stFinish;
  wire [15:0] readRecord_22_elementMask =
    (_readRecord_T_4840 ? chainingRecordCopy_0_bits_elementMask : 16'h0) | (_readRecord_T_4841 ? chainingRecordCopy_1_bits_elementMask : 16'h0) | (_readRecord_T_4842 ? chainingRecordCopy_2_bits_elementMask : 16'h0)
    | (_readRecord_T_4843 ? chainingRecordCopy_3_bits_elementMask : 16'h0) | (_readRecord_T_4844 ? chainingRecordCopy_4_bits_elementMask : 16'h0);
  wire        readRecord_22_slow =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_slow | _readRecord_T_4841 & chainingRecordCopy_1_bits_slow | _readRecord_T_4842 & chainingRecordCopy_2_bits_slow | _readRecord_T_4843 & chainingRecordCopy_3_bits_slow | _readRecord_T_4844
    & chainingRecordCopy_4_bits_slow;
  wire        readRecord_22_onlyRead =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_onlyRead | _readRecord_T_4841 & chainingRecordCopy_1_bits_onlyRead | _readRecord_T_4842 & chainingRecordCopy_2_bits_onlyRead | _readRecord_T_4843 & chainingRecordCopy_3_bits_onlyRead
    | _readRecord_T_4844 & chainingRecordCopy_4_bits_onlyRead;
  wire        readRecord_22_ma =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_ma | _readRecord_T_4841 & chainingRecordCopy_1_bits_ma | _readRecord_T_4842 & chainingRecordCopy_2_bits_ma | _readRecord_T_4843 & chainingRecordCopy_3_bits_ma | _readRecord_T_4844
    & chainingRecordCopy_4_bits_ma;
  wire        readRecord_22_indexType =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_indexType | _readRecord_T_4841 & chainingRecordCopy_1_bits_indexType | _readRecord_T_4842 & chainingRecordCopy_2_bits_indexType | _readRecord_T_4843 & chainingRecordCopy_3_bits_indexType
    | _readRecord_T_4844 & chainingRecordCopy_4_bits_indexType;
  wire        readRecord_22_crossRead =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_crossRead | _readRecord_T_4841 & chainingRecordCopy_1_bits_crossRead | _readRecord_T_4842 & chainingRecordCopy_2_bits_crossRead | _readRecord_T_4843 & chainingRecordCopy_3_bits_crossRead
    | _readRecord_T_4844 & chainingRecordCopy_4_bits_crossRead;
  wire        readRecord_22_crossWrite =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_crossWrite | _readRecord_T_4841 & chainingRecordCopy_1_bits_crossWrite | _readRecord_T_4842 & chainingRecordCopy_2_bits_crossWrite | _readRecord_T_4843
    & chainingRecordCopy_3_bits_crossWrite | _readRecord_T_4844 & chainingRecordCopy_4_bits_crossWrite;
  wire        readRecord_22_gather16 =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_gather16 | _readRecord_T_4841 & chainingRecordCopy_1_bits_gather16 | _readRecord_T_4842 & chainingRecordCopy_2_bits_gather16 | _readRecord_T_4843 & chainingRecordCopy_3_bits_gather16
    | _readRecord_T_4844 & chainingRecordCopy_4_bits_gather16;
  wire        readRecord_22_gather =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_gather | _readRecord_T_4841 & chainingRecordCopy_1_bits_gather | _readRecord_T_4842 & chainingRecordCopy_2_bits_gather | _readRecord_T_4843 & chainingRecordCopy_3_bits_gather
    | _readRecord_T_4844 & chainingRecordCopy_4_bits_gather;
  wire        readRecord_22_st =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_st | _readRecord_T_4841 & chainingRecordCopy_1_bits_st | _readRecord_T_4842 & chainingRecordCopy_2_bits_st | _readRecord_T_4843 & chainingRecordCopy_3_bits_st | _readRecord_T_4844
    & chainingRecordCopy_4_bits_st;
  wire        readRecord_22_ls =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_ls | _readRecord_T_4841 & chainingRecordCopy_1_bits_ls | _readRecord_T_4842 & chainingRecordCopy_2_bits_ls | _readRecord_T_4843 & chainingRecordCopy_3_bits_ls | _readRecord_T_4844
    & chainingRecordCopy_4_bits_ls;
  wire [2:0]  readRecord_22_instIndex =
    (_readRecord_T_4840 ? chainingRecordCopy_0_bits_instIndex : 3'h0) | (_readRecord_T_4841 ? chainingRecordCopy_1_bits_instIndex : 3'h0) | (_readRecord_T_4842 ? chainingRecordCopy_2_bits_instIndex : 3'h0)
    | (_readRecord_T_4843 ? chainingRecordCopy_3_bits_instIndex : 3'h0) | (_readRecord_T_4844 ? chainingRecordCopy_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_22_vs2 =
    (_readRecord_T_4840 ? chainingRecordCopy_0_bits_vs2 : 5'h0) | (_readRecord_T_4841 ? chainingRecordCopy_1_bits_vs2 : 5'h0) | (_readRecord_T_4842 ? chainingRecordCopy_2_bits_vs2 : 5'h0)
    | (_readRecord_T_4843 ? chainingRecordCopy_3_bits_vs2 : 5'h0) | (_readRecord_T_4844 ? chainingRecordCopy_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_22_vs1_bits =
    (_readRecord_T_4840 ? chainingRecordCopy_0_bits_vs1_bits : 5'h0) | (_readRecord_T_4841 ? chainingRecordCopy_1_bits_vs1_bits : 5'h0) | (_readRecord_T_4842 ? chainingRecordCopy_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_4843 ? chainingRecordCopy_3_bits_vs1_bits : 5'h0) | (_readRecord_T_4844 ? chainingRecordCopy_4_bits_vs1_bits : 5'h0);
  wire        readRecord_22_vs1_valid =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_vs1_valid | _readRecord_T_4841 & chainingRecordCopy_1_bits_vs1_valid | _readRecord_T_4842 & chainingRecordCopy_2_bits_vs1_valid | _readRecord_T_4843 & chainingRecordCopy_3_bits_vs1_valid
    | _readRecord_T_4844 & chainingRecordCopy_4_bits_vs1_valid;
  wire [4:0]  readRecord_22_vd_bits =
    (_readRecord_T_4840 ? chainingRecordCopy_0_bits_vd_bits : 5'h0) | (_readRecord_T_4841 ? chainingRecordCopy_1_bits_vd_bits : 5'h0) | (_readRecord_T_4842 ? chainingRecordCopy_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_4843 ? chainingRecordCopy_3_bits_vd_bits : 5'h0) | (_readRecord_T_4844 ? chainingRecordCopy_4_bits_vd_bits : 5'h0);
  wire        readRecord_22_vd_valid =
    _readRecord_T_4840 & chainingRecordCopy_0_bits_vd_valid | _readRecord_T_4841 & chainingRecordCopy_1_bits_vd_valid | _readRecord_T_4842 & chainingRecordCopy_2_bits_vd_valid | _readRecord_T_4843 & chainingRecordCopy_3_bits_vd_valid
    | _readRecord_T_4844 & chainingRecordCopy_4_bits_vd_valid;
  wire [5:0]  address_8 = {readRequests_8_bits_vs_0, readRequests_8_bits_offset_0};
  wire [1:0]  bank_8 = 2'h1 << address_8[0];
  reg         pipeBank_pipe_v_8;
  reg  [1:0]  pipeBank_pipe_b_8;
  reg         pipeBank_pipe_pipe_v_8;
  wire        pipeBank_pipe_pipe_out_8_valid = pipeBank_pipe_pipe_v_8;
  reg  [1:0]  pipeBank_pipe_pipe_b_8;
  wire [1:0]  pipeBank_pipe_pipe_out_8_bits = pipeBank_pipe_pipe_b_8;
  wire [1:0]  bankCorrect_8 = readRequests_8_valid_0 ? bank_8 : 2'h0;
  wire        portConflictCheck_8 =
    ~(write_valid_0 & bank_8 == writeBank & write_bits_vd_0 == readRequests_8_bits_vs_0 & write_bits_offset_0 == readRequests_8_bits_offset_0 | writePipe_valid & bank_8 == writeBankPipe & writePipe_bits_vd == readRequests_8_bits_vs_0
      & writePipe_bits_offset == readRequests_8_bits_offset_0);
  wire        portReady_8 = |(bank_8 & ~_GEN_12);
  assign readRequests_8_ready_0 = portReady_8 & sramReady;
  wire        firstUsed_8 = |(bank_8 & _GEN_11);
  wire [1:0]  bankReadF_8 = bankCorrect_8 & ~_GEN_11;
  wire [1:0]  bankReadS_8 = bankCorrect_8 & ~_GEN_12 & _GEN_11;
  reg         pipeFirstUsed_pipe_v_8;
  reg         pipeFirstUsed_pipe_b_8;
  reg         pipeFirstUsed_pipe_pipe_v_8;
  wire        pipeFirstUsed_pipe_pipe_out_8_valid = pipeFirstUsed_pipe_pipe_v_8;
  reg         pipeFirstUsed_pipe_pipe_b_8;
  wire        pipeFirstUsed_pipe_pipe_out_8_bits = pipeFirstUsed_pipe_pipe_b_8;
  reg         pipeFire_pipe_v_8;
  reg         pipeFire_pipe_b_8;
  reg         pipeFire_pipe_pipe_v_8;
  wire        pipeFire_pipe_pipe_out_8_valid = pipeFire_pipe_pipe_v_8;
  reg         pipeFire_pipe_pipe_b_8;
  wire        pipeFire_pipe_pipe_out_8_bits = pipeFire_pipe_pipe_b_8;
  wire [1:0]  _GEN_13 = _GEN_11 | bankCorrect_8;
  wire [1:0]  _GEN_14 = bankCorrect_8 & _GEN_11 | _GEN_12;
  wire        _readRecord_T_5060 = chainingRecordCopy_0_bits_instIndex == readRequests_9_bits_instructionIndex_0;
  wire        _readRecord_T_5061 = chainingRecordCopy_1_bits_instIndex == readRequests_9_bits_instructionIndex_0;
  wire        _readRecord_T_5062 = chainingRecordCopy_2_bits_instIndex == readRequests_9_bits_instructionIndex_0;
  wire        _readRecord_T_5063 = chainingRecordCopy_3_bits_instIndex == readRequests_9_bits_instructionIndex_0;
  wire        _readRecord_T_5064 = chainingRecordCopy_4_bits_instIndex == readRequests_9_bits_instructionIndex_0;
  wire        readRecord_23_state_wLaneClear =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_state_wLaneClear | _readRecord_T_5061 & chainingRecordCopy_1_bits_state_wLaneClear | _readRecord_T_5062 & chainingRecordCopy_2_bits_state_wLaneClear | _readRecord_T_5063
    & chainingRecordCopy_3_bits_state_wLaneClear | _readRecord_T_5064 & chainingRecordCopy_4_bits_state_wLaneClear;
  wire        readRecord_23_state_wTopLastReport =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_state_wTopLastReport | _readRecord_T_5061 & chainingRecordCopy_1_bits_state_wTopLastReport | _readRecord_T_5062 & chainingRecordCopy_2_bits_state_wTopLastReport | _readRecord_T_5063
    & chainingRecordCopy_3_bits_state_wTopLastReport | _readRecord_T_5064 & chainingRecordCopy_4_bits_state_wTopLastReport;
  wire        readRecord_23_state_wLaneLastReport =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_state_wLaneLastReport | _readRecord_T_5061 & chainingRecordCopy_1_bits_state_wLaneLastReport | _readRecord_T_5062 & chainingRecordCopy_2_bits_state_wLaneLastReport | _readRecord_T_5063
    & chainingRecordCopy_3_bits_state_wLaneLastReport | _readRecord_T_5064 & chainingRecordCopy_4_bits_state_wLaneLastReport;
  wire        readRecord_23_state_wWriteQueueClear =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_state_wWriteQueueClear | _readRecord_T_5061 & chainingRecordCopy_1_bits_state_wWriteQueueClear | _readRecord_T_5062 & chainingRecordCopy_2_bits_state_wWriteQueueClear | _readRecord_T_5063
    & chainingRecordCopy_3_bits_state_wWriteQueueClear | _readRecord_T_5064 & chainingRecordCopy_4_bits_state_wWriteQueueClear;
  wire        readRecord_23_state_stFinish =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_state_stFinish | _readRecord_T_5061 & chainingRecordCopy_1_bits_state_stFinish | _readRecord_T_5062 & chainingRecordCopy_2_bits_state_stFinish | _readRecord_T_5063
    & chainingRecordCopy_3_bits_state_stFinish | _readRecord_T_5064 & chainingRecordCopy_4_bits_state_stFinish;
  wire [15:0] readRecord_23_elementMask =
    (_readRecord_T_5060 ? chainingRecordCopy_0_bits_elementMask : 16'h0) | (_readRecord_T_5061 ? chainingRecordCopy_1_bits_elementMask : 16'h0) | (_readRecord_T_5062 ? chainingRecordCopy_2_bits_elementMask : 16'h0)
    | (_readRecord_T_5063 ? chainingRecordCopy_3_bits_elementMask : 16'h0) | (_readRecord_T_5064 ? chainingRecordCopy_4_bits_elementMask : 16'h0);
  wire        readRecord_23_slow =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_slow | _readRecord_T_5061 & chainingRecordCopy_1_bits_slow | _readRecord_T_5062 & chainingRecordCopy_2_bits_slow | _readRecord_T_5063 & chainingRecordCopy_3_bits_slow | _readRecord_T_5064
    & chainingRecordCopy_4_bits_slow;
  wire        readRecord_23_onlyRead =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_onlyRead | _readRecord_T_5061 & chainingRecordCopy_1_bits_onlyRead | _readRecord_T_5062 & chainingRecordCopy_2_bits_onlyRead | _readRecord_T_5063 & chainingRecordCopy_3_bits_onlyRead
    | _readRecord_T_5064 & chainingRecordCopy_4_bits_onlyRead;
  wire        readRecord_23_ma =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_ma | _readRecord_T_5061 & chainingRecordCopy_1_bits_ma | _readRecord_T_5062 & chainingRecordCopy_2_bits_ma | _readRecord_T_5063 & chainingRecordCopy_3_bits_ma | _readRecord_T_5064
    & chainingRecordCopy_4_bits_ma;
  wire        readRecord_23_indexType =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_indexType | _readRecord_T_5061 & chainingRecordCopy_1_bits_indexType | _readRecord_T_5062 & chainingRecordCopy_2_bits_indexType | _readRecord_T_5063 & chainingRecordCopy_3_bits_indexType
    | _readRecord_T_5064 & chainingRecordCopy_4_bits_indexType;
  wire        readRecord_23_crossRead =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_crossRead | _readRecord_T_5061 & chainingRecordCopy_1_bits_crossRead | _readRecord_T_5062 & chainingRecordCopy_2_bits_crossRead | _readRecord_T_5063 & chainingRecordCopy_3_bits_crossRead
    | _readRecord_T_5064 & chainingRecordCopy_4_bits_crossRead;
  wire        readRecord_23_crossWrite =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_crossWrite | _readRecord_T_5061 & chainingRecordCopy_1_bits_crossWrite | _readRecord_T_5062 & chainingRecordCopy_2_bits_crossWrite | _readRecord_T_5063
    & chainingRecordCopy_3_bits_crossWrite | _readRecord_T_5064 & chainingRecordCopy_4_bits_crossWrite;
  wire        readRecord_23_gather16 =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_gather16 | _readRecord_T_5061 & chainingRecordCopy_1_bits_gather16 | _readRecord_T_5062 & chainingRecordCopy_2_bits_gather16 | _readRecord_T_5063 & chainingRecordCopy_3_bits_gather16
    | _readRecord_T_5064 & chainingRecordCopy_4_bits_gather16;
  wire        readRecord_23_gather =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_gather | _readRecord_T_5061 & chainingRecordCopy_1_bits_gather | _readRecord_T_5062 & chainingRecordCopy_2_bits_gather | _readRecord_T_5063 & chainingRecordCopy_3_bits_gather
    | _readRecord_T_5064 & chainingRecordCopy_4_bits_gather;
  wire        readRecord_23_st =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_st | _readRecord_T_5061 & chainingRecordCopy_1_bits_st | _readRecord_T_5062 & chainingRecordCopy_2_bits_st | _readRecord_T_5063 & chainingRecordCopy_3_bits_st | _readRecord_T_5064
    & chainingRecordCopy_4_bits_st;
  wire        readRecord_23_ls =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_ls | _readRecord_T_5061 & chainingRecordCopy_1_bits_ls | _readRecord_T_5062 & chainingRecordCopy_2_bits_ls | _readRecord_T_5063 & chainingRecordCopy_3_bits_ls | _readRecord_T_5064
    & chainingRecordCopy_4_bits_ls;
  wire [2:0]  readRecord_23_instIndex =
    (_readRecord_T_5060 ? chainingRecordCopy_0_bits_instIndex : 3'h0) | (_readRecord_T_5061 ? chainingRecordCopy_1_bits_instIndex : 3'h0) | (_readRecord_T_5062 ? chainingRecordCopy_2_bits_instIndex : 3'h0)
    | (_readRecord_T_5063 ? chainingRecordCopy_3_bits_instIndex : 3'h0) | (_readRecord_T_5064 ? chainingRecordCopy_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_23_vs2 =
    (_readRecord_T_5060 ? chainingRecordCopy_0_bits_vs2 : 5'h0) | (_readRecord_T_5061 ? chainingRecordCopy_1_bits_vs2 : 5'h0) | (_readRecord_T_5062 ? chainingRecordCopy_2_bits_vs2 : 5'h0)
    | (_readRecord_T_5063 ? chainingRecordCopy_3_bits_vs2 : 5'h0) | (_readRecord_T_5064 ? chainingRecordCopy_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_23_vs1_bits =
    (_readRecord_T_5060 ? chainingRecordCopy_0_bits_vs1_bits : 5'h0) | (_readRecord_T_5061 ? chainingRecordCopy_1_bits_vs1_bits : 5'h0) | (_readRecord_T_5062 ? chainingRecordCopy_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_5063 ? chainingRecordCopy_3_bits_vs1_bits : 5'h0) | (_readRecord_T_5064 ? chainingRecordCopy_4_bits_vs1_bits : 5'h0);
  wire        readRecord_23_vs1_valid =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_vs1_valid | _readRecord_T_5061 & chainingRecordCopy_1_bits_vs1_valid | _readRecord_T_5062 & chainingRecordCopy_2_bits_vs1_valid | _readRecord_T_5063 & chainingRecordCopy_3_bits_vs1_valid
    | _readRecord_T_5064 & chainingRecordCopy_4_bits_vs1_valid;
  wire [4:0]  readRecord_23_vd_bits =
    (_readRecord_T_5060 ? chainingRecordCopy_0_bits_vd_bits : 5'h0) | (_readRecord_T_5061 ? chainingRecordCopy_1_bits_vd_bits : 5'h0) | (_readRecord_T_5062 ? chainingRecordCopy_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_5063 ? chainingRecordCopy_3_bits_vd_bits : 5'h0) | (_readRecord_T_5064 ? chainingRecordCopy_4_bits_vd_bits : 5'h0);
  wire        readRecord_23_vd_valid =
    _readRecord_T_5060 & chainingRecordCopy_0_bits_vd_valid | _readRecord_T_5061 & chainingRecordCopy_1_bits_vd_valid | _readRecord_T_5062 & chainingRecordCopy_2_bits_vd_valid | _readRecord_T_5063 & chainingRecordCopy_3_bits_vd_valid
    | _readRecord_T_5064 & chainingRecordCopy_4_bits_vd_valid;
  wire [5:0]  address_9 = {readRequests_9_bits_vs_0, readRequests_9_bits_offset_0};
  wire [1:0]  bank_9 = 2'h1 << address_9[0];
  reg         pipeBank_pipe_v_9;
  reg  [1:0]  pipeBank_pipe_b_9;
  reg         pipeBank_pipe_pipe_v_9;
  wire        pipeBank_pipe_pipe_out_9_valid = pipeBank_pipe_pipe_v_9;
  reg  [1:0]  pipeBank_pipe_pipe_b_9;
  wire [1:0]  pipeBank_pipe_pipe_out_9_bits = pipeBank_pipe_pipe_b_9;
  wire [1:0]  bankCorrect_9 = readRequests_9_valid_0 ? bank_9 : 2'h0;
  wire        portConflictCheck_9 =
    ~(write_valid_0 & bank_9 == writeBank & write_bits_vd_0 == readRequests_9_bits_vs_0 & write_bits_offset_0 == readRequests_9_bits_offset_0 | writePipe_valid & bank_9 == writeBankPipe & writePipe_bits_vd == readRequests_9_bits_vs_0
      & writePipe_bits_offset == readRequests_9_bits_offset_0);
  wire        portReady_9 = |(bank_9 & ~_GEN_14);
  assign readRequests_9_ready_0 = portReady_9 & sramReady;
  wire        firstUsed_9 = |(bank_9 & _GEN_13);
  wire [1:0]  bankReadF_9 = bankCorrect_9 & ~_GEN_13;
  wire [1:0]  bankReadS_9 = bankCorrect_9 & ~_GEN_14 & _GEN_13;
  reg         pipeFirstUsed_pipe_v_9;
  reg         pipeFirstUsed_pipe_b_9;
  reg         pipeFirstUsed_pipe_pipe_v_9;
  wire        pipeFirstUsed_pipe_pipe_out_9_valid = pipeFirstUsed_pipe_pipe_v_9;
  reg         pipeFirstUsed_pipe_pipe_b_9;
  wire        pipeFirstUsed_pipe_pipe_out_9_bits = pipeFirstUsed_pipe_pipe_b_9;
  reg         pipeFire_pipe_v_9;
  reg         pipeFire_pipe_b_9;
  reg         pipeFire_pipe_pipe_v_9;
  wire        pipeFire_pipe_pipe_out_9_valid = pipeFire_pipe_pipe_v_9;
  reg         pipeFire_pipe_pipe_b_9;
  wire        pipeFire_pipe_pipe_out_9_bits = pipeFire_pipe_pipe_b_9;
  wire [1:0]  _GEN_15 = _GEN_13 | bankCorrect_9;
  wire [1:0]  _GEN_16 = bankCorrect_9 & _GEN_13 | _GEN_14;
  wire        _readRecord_T_5280 = chainingRecordCopy_0_bits_instIndex == readRequests_10_bits_instructionIndex_0;
  wire        _readRecord_T_5281 = chainingRecordCopy_1_bits_instIndex == readRequests_10_bits_instructionIndex_0;
  wire        _readRecord_T_5282 = chainingRecordCopy_2_bits_instIndex == readRequests_10_bits_instructionIndex_0;
  wire        _readRecord_T_5283 = chainingRecordCopy_3_bits_instIndex == readRequests_10_bits_instructionIndex_0;
  wire        _readRecord_T_5284 = chainingRecordCopy_4_bits_instIndex == readRequests_10_bits_instructionIndex_0;
  wire        readRecord_24_state_wLaneClear =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_state_wLaneClear | _readRecord_T_5281 & chainingRecordCopy_1_bits_state_wLaneClear | _readRecord_T_5282 & chainingRecordCopy_2_bits_state_wLaneClear | _readRecord_T_5283
    & chainingRecordCopy_3_bits_state_wLaneClear | _readRecord_T_5284 & chainingRecordCopy_4_bits_state_wLaneClear;
  wire        readRecord_24_state_wTopLastReport =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_state_wTopLastReport | _readRecord_T_5281 & chainingRecordCopy_1_bits_state_wTopLastReport | _readRecord_T_5282 & chainingRecordCopy_2_bits_state_wTopLastReport | _readRecord_T_5283
    & chainingRecordCopy_3_bits_state_wTopLastReport | _readRecord_T_5284 & chainingRecordCopy_4_bits_state_wTopLastReport;
  wire        readRecord_24_state_wLaneLastReport =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_state_wLaneLastReport | _readRecord_T_5281 & chainingRecordCopy_1_bits_state_wLaneLastReport | _readRecord_T_5282 & chainingRecordCopy_2_bits_state_wLaneLastReport | _readRecord_T_5283
    & chainingRecordCopy_3_bits_state_wLaneLastReport | _readRecord_T_5284 & chainingRecordCopy_4_bits_state_wLaneLastReport;
  wire        readRecord_24_state_wWriteQueueClear =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_state_wWriteQueueClear | _readRecord_T_5281 & chainingRecordCopy_1_bits_state_wWriteQueueClear | _readRecord_T_5282 & chainingRecordCopy_2_bits_state_wWriteQueueClear | _readRecord_T_5283
    & chainingRecordCopy_3_bits_state_wWriteQueueClear | _readRecord_T_5284 & chainingRecordCopy_4_bits_state_wWriteQueueClear;
  wire        readRecord_24_state_stFinish =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_state_stFinish | _readRecord_T_5281 & chainingRecordCopy_1_bits_state_stFinish | _readRecord_T_5282 & chainingRecordCopy_2_bits_state_stFinish | _readRecord_T_5283
    & chainingRecordCopy_3_bits_state_stFinish | _readRecord_T_5284 & chainingRecordCopy_4_bits_state_stFinish;
  wire [15:0] readRecord_24_elementMask =
    (_readRecord_T_5280 ? chainingRecordCopy_0_bits_elementMask : 16'h0) | (_readRecord_T_5281 ? chainingRecordCopy_1_bits_elementMask : 16'h0) | (_readRecord_T_5282 ? chainingRecordCopy_2_bits_elementMask : 16'h0)
    | (_readRecord_T_5283 ? chainingRecordCopy_3_bits_elementMask : 16'h0) | (_readRecord_T_5284 ? chainingRecordCopy_4_bits_elementMask : 16'h0);
  wire        readRecord_24_slow =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_slow | _readRecord_T_5281 & chainingRecordCopy_1_bits_slow | _readRecord_T_5282 & chainingRecordCopy_2_bits_slow | _readRecord_T_5283 & chainingRecordCopy_3_bits_slow | _readRecord_T_5284
    & chainingRecordCopy_4_bits_slow;
  wire        readRecord_24_onlyRead =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_onlyRead | _readRecord_T_5281 & chainingRecordCopy_1_bits_onlyRead | _readRecord_T_5282 & chainingRecordCopy_2_bits_onlyRead | _readRecord_T_5283 & chainingRecordCopy_3_bits_onlyRead
    | _readRecord_T_5284 & chainingRecordCopy_4_bits_onlyRead;
  wire        readRecord_24_ma =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_ma | _readRecord_T_5281 & chainingRecordCopy_1_bits_ma | _readRecord_T_5282 & chainingRecordCopy_2_bits_ma | _readRecord_T_5283 & chainingRecordCopy_3_bits_ma | _readRecord_T_5284
    & chainingRecordCopy_4_bits_ma;
  wire        readRecord_24_indexType =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_indexType | _readRecord_T_5281 & chainingRecordCopy_1_bits_indexType | _readRecord_T_5282 & chainingRecordCopy_2_bits_indexType | _readRecord_T_5283 & chainingRecordCopy_3_bits_indexType
    | _readRecord_T_5284 & chainingRecordCopy_4_bits_indexType;
  wire        readRecord_24_crossRead =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_crossRead | _readRecord_T_5281 & chainingRecordCopy_1_bits_crossRead | _readRecord_T_5282 & chainingRecordCopy_2_bits_crossRead | _readRecord_T_5283 & chainingRecordCopy_3_bits_crossRead
    | _readRecord_T_5284 & chainingRecordCopy_4_bits_crossRead;
  wire        readRecord_24_crossWrite =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_crossWrite | _readRecord_T_5281 & chainingRecordCopy_1_bits_crossWrite | _readRecord_T_5282 & chainingRecordCopy_2_bits_crossWrite | _readRecord_T_5283
    & chainingRecordCopy_3_bits_crossWrite | _readRecord_T_5284 & chainingRecordCopy_4_bits_crossWrite;
  wire        readRecord_24_gather16 =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_gather16 | _readRecord_T_5281 & chainingRecordCopy_1_bits_gather16 | _readRecord_T_5282 & chainingRecordCopy_2_bits_gather16 | _readRecord_T_5283 & chainingRecordCopy_3_bits_gather16
    | _readRecord_T_5284 & chainingRecordCopy_4_bits_gather16;
  wire        readRecord_24_gather =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_gather | _readRecord_T_5281 & chainingRecordCopy_1_bits_gather | _readRecord_T_5282 & chainingRecordCopy_2_bits_gather | _readRecord_T_5283 & chainingRecordCopy_3_bits_gather
    | _readRecord_T_5284 & chainingRecordCopy_4_bits_gather;
  wire        readRecord_24_st =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_st | _readRecord_T_5281 & chainingRecordCopy_1_bits_st | _readRecord_T_5282 & chainingRecordCopy_2_bits_st | _readRecord_T_5283 & chainingRecordCopy_3_bits_st | _readRecord_T_5284
    & chainingRecordCopy_4_bits_st;
  wire        readRecord_24_ls =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_ls | _readRecord_T_5281 & chainingRecordCopy_1_bits_ls | _readRecord_T_5282 & chainingRecordCopy_2_bits_ls | _readRecord_T_5283 & chainingRecordCopy_3_bits_ls | _readRecord_T_5284
    & chainingRecordCopy_4_bits_ls;
  wire [2:0]  readRecord_24_instIndex =
    (_readRecord_T_5280 ? chainingRecordCopy_0_bits_instIndex : 3'h0) | (_readRecord_T_5281 ? chainingRecordCopy_1_bits_instIndex : 3'h0) | (_readRecord_T_5282 ? chainingRecordCopy_2_bits_instIndex : 3'h0)
    | (_readRecord_T_5283 ? chainingRecordCopy_3_bits_instIndex : 3'h0) | (_readRecord_T_5284 ? chainingRecordCopy_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_24_vs2 =
    (_readRecord_T_5280 ? chainingRecordCopy_0_bits_vs2 : 5'h0) | (_readRecord_T_5281 ? chainingRecordCopy_1_bits_vs2 : 5'h0) | (_readRecord_T_5282 ? chainingRecordCopy_2_bits_vs2 : 5'h0)
    | (_readRecord_T_5283 ? chainingRecordCopy_3_bits_vs2 : 5'h0) | (_readRecord_T_5284 ? chainingRecordCopy_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_24_vs1_bits =
    (_readRecord_T_5280 ? chainingRecordCopy_0_bits_vs1_bits : 5'h0) | (_readRecord_T_5281 ? chainingRecordCopy_1_bits_vs1_bits : 5'h0) | (_readRecord_T_5282 ? chainingRecordCopy_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_5283 ? chainingRecordCopy_3_bits_vs1_bits : 5'h0) | (_readRecord_T_5284 ? chainingRecordCopy_4_bits_vs1_bits : 5'h0);
  wire        readRecord_24_vs1_valid =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_vs1_valid | _readRecord_T_5281 & chainingRecordCopy_1_bits_vs1_valid | _readRecord_T_5282 & chainingRecordCopy_2_bits_vs1_valid | _readRecord_T_5283 & chainingRecordCopy_3_bits_vs1_valid
    | _readRecord_T_5284 & chainingRecordCopy_4_bits_vs1_valid;
  wire [4:0]  readRecord_24_vd_bits =
    (_readRecord_T_5280 ? chainingRecordCopy_0_bits_vd_bits : 5'h0) | (_readRecord_T_5281 ? chainingRecordCopy_1_bits_vd_bits : 5'h0) | (_readRecord_T_5282 ? chainingRecordCopy_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_5283 ? chainingRecordCopy_3_bits_vd_bits : 5'h0) | (_readRecord_T_5284 ? chainingRecordCopy_4_bits_vd_bits : 5'h0);
  wire        readRecord_24_vd_valid =
    _readRecord_T_5280 & chainingRecordCopy_0_bits_vd_valid | _readRecord_T_5281 & chainingRecordCopy_1_bits_vd_valid | _readRecord_T_5282 & chainingRecordCopy_2_bits_vd_valid | _readRecord_T_5283 & chainingRecordCopy_3_bits_vd_valid
    | _readRecord_T_5284 & chainingRecordCopy_4_bits_vd_valid;
  wire [5:0]  address_10 = {readRequests_10_bits_vs_0, readRequests_10_bits_offset_0};
  wire [1:0]  bank_10 = 2'h1 << address_10[0];
  reg         pipeBank_pipe_v_10;
  reg  [1:0]  pipeBank_pipe_b_10;
  reg         pipeBank_pipe_pipe_v_10;
  wire        pipeBank_pipe_pipe_out_10_valid = pipeBank_pipe_pipe_v_10;
  reg  [1:0]  pipeBank_pipe_pipe_b_10;
  wire [1:0]  pipeBank_pipe_pipe_out_10_bits = pipeBank_pipe_pipe_b_10;
  wire [1:0]  bankCorrect_10 = readRequests_10_valid_0 ? bank_10 : 2'h0;
  wire        portConflictCheck_10 =
    ~(write_valid_0 & bank_10 == writeBank & write_bits_vd_0 == readRequests_10_bits_vs_0 & write_bits_offset_0 == readRequests_10_bits_offset_0 | writePipe_valid & bank_10 == writeBankPipe & writePipe_bits_vd == readRequests_10_bits_vs_0
      & writePipe_bits_offset == readRequests_10_bits_offset_0);
  wire        portReady_10 = |(bank_10 & ~_GEN_16);
  assign readRequests_10_ready_0 = portReady_10 & sramReady;
  wire        firstUsed_10 = |(bank_10 & _GEN_15);
  wire [1:0]  bankReadF_10 = bankCorrect_10 & ~_GEN_15;
  wire [1:0]  bankReadS_10 = bankCorrect_10 & ~_GEN_16 & _GEN_15;
  reg         pipeFirstUsed_pipe_v_10;
  reg         pipeFirstUsed_pipe_b_10;
  reg         pipeFirstUsed_pipe_pipe_v_10;
  wire        pipeFirstUsed_pipe_pipe_out_10_valid = pipeFirstUsed_pipe_pipe_v_10;
  reg         pipeFirstUsed_pipe_pipe_b_10;
  wire        pipeFirstUsed_pipe_pipe_out_10_bits = pipeFirstUsed_pipe_pipe_b_10;
  reg         pipeFire_pipe_v_10;
  reg         pipeFire_pipe_b_10;
  reg         pipeFire_pipe_pipe_v_10;
  wire        pipeFire_pipe_pipe_out_10_valid = pipeFire_pipe_pipe_v_10;
  reg         pipeFire_pipe_pipe_b_10;
  wire        pipeFire_pipe_pipe_out_10_bits = pipeFire_pipe_pipe_b_10;
  wire [1:0]  _GEN_17 = _GEN_15 | bankCorrect_10;
  wire [1:0]  _GEN_18 = bankCorrect_10 & _GEN_15 | _GEN_16;
  wire        _readRecord_T_5500 = chainingRecordCopy_0_bits_instIndex == readRequests_11_bits_instructionIndex_0;
  wire        _readRecord_T_5501 = chainingRecordCopy_1_bits_instIndex == readRequests_11_bits_instructionIndex_0;
  wire        _readRecord_T_5502 = chainingRecordCopy_2_bits_instIndex == readRequests_11_bits_instructionIndex_0;
  wire        _readRecord_T_5503 = chainingRecordCopy_3_bits_instIndex == readRequests_11_bits_instructionIndex_0;
  wire        _readRecord_T_5504 = chainingRecordCopy_4_bits_instIndex == readRequests_11_bits_instructionIndex_0;
  wire        readRecord_25_state_wLaneClear =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_state_wLaneClear | _readRecord_T_5501 & chainingRecordCopy_1_bits_state_wLaneClear | _readRecord_T_5502 & chainingRecordCopy_2_bits_state_wLaneClear | _readRecord_T_5503
    & chainingRecordCopy_3_bits_state_wLaneClear | _readRecord_T_5504 & chainingRecordCopy_4_bits_state_wLaneClear;
  wire        readRecord_25_state_wTopLastReport =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_state_wTopLastReport | _readRecord_T_5501 & chainingRecordCopy_1_bits_state_wTopLastReport | _readRecord_T_5502 & chainingRecordCopy_2_bits_state_wTopLastReport | _readRecord_T_5503
    & chainingRecordCopy_3_bits_state_wTopLastReport | _readRecord_T_5504 & chainingRecordCopy_4_bits_state_wTopLastReport;
  wire        readRecord_25_state_wLaneLastReport =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_state_wLaneLastReport | _readRecord_T_5501 & chainingRecordCopy_1_bits_state_wLaneLastReport | _readRecord_T_5502 & chainingRecordCopy_2_bits_state_wLaneLastReport | _readRecord_T_5503
    & chainingRecordCopy_3_bits_state_wLaneLastReport | _readRecord_T_5504 & chainingRecordCopy_4_bits_state_wLaneLastReport;
  wire        readRecord_25_state_wWriteQueueClear =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_state_wWriteQueueClear | _readRecord_T_5501 & chainingRecordCopy_1_bits_state_wWriteQueueClear | _readRecord_T_5502 & chainingRecordCopy_2_bits_state_wWriteQueueClear | _readRecord_T_5503
    & chainingRecordCopy_3_bits_state_wWriteQueueClear | _readRecord_T_5504 & chainingRecordCopy_4_bits_state_wWriteQueueClear;
  wire        readRecord_25_state_stFinish =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_state_stFinish | _readRecord_T_5501 & chainingRecordCopy_1_bits_state_stFinish | _readRecord_T_5502 & chainingRecordCopy_2_bits_state_stFinish | _readRecord_T_5503
    & chainingRecordCopy_3_bits_state_stFinish | _readRecord_T_5504 & chainingRecordCopy_4_bits_state_stFinish;
  wire [15:0] readRecord_25_elementMask =
    (_readRecord_T_5500 ? chainingRecordCopy_0_bits_elementMask : 16'h0) | (_readRecord_T_5501 ? chainingRecordCopy_1_bits_elementMask : 16'h0) | (_readRecord_T_5502 ? chainingRecordCopy_2_bits_elementMask : 16'h0)
    | (_readRecord_T_5503 ? chainingRecordCopy_3_bits_elementMask : 16'h0) | (_readRecord_T_5504 ? chainingRecordCopy_4_bits_elementMask : 16'h0);
  wire        readRecord_25_slow =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_slow | _readRecord_T_5501 & chainingRecordCopy_1_bits_slow | _readRecord_T_5502 & chainingRecordCopy_2_bits_slow | _readRecord_T_5503 & chainingRecordCopy_3_bits_slow | _readRecord_T_5504
    & chainingRecordCopy_4_bits_slow;
  wire        readRecord_25_onlyRead =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_onlyRead | _readRecord_T_5501 & chainingRecordCopy_1_bits_onlyRead | _readRecord_T_5502 & chainingRecordCopy_2_bits_onlyRead | _readRecord_T_5503 & chainingRecordCopy_3_bits_onlyRead
    | _readRecord_T_5504 & chainingRecordCopy_4_bits_onlyRead;
  wire        readRecord_25_ma =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_ma | _readRecord_T_5501 & chainingRecordCopy_1_bits_ma | _readRecord_T_5502 & chainingRecordCopy_2_bits_ma | _readRecord_T_5503 & chainingRecordCopy_3_bits_ma | _readRecord_T_5504
    & chainingRecordCopy_4_bits_ma;
  wire        readRecord_25_indexType =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_indexType | _readRecord_T_5501 & chainingRecordCopy_1_bits_indexType | _readRecord_T_5502 & chainingRecordCopy_2_bits_indexType | _readRecord_T_5503 & chainingRecordCopy_3_bits_indexType
    | _readRecord_T_5504 & chainingRecordCopy_4_bits_indexType;
  wire        readRecord_25_crossRead =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_crossRead | _readRecord_T_5501 & chainingRecordCopy_1_bits_crossRead | _readRecord_T_5502 & chainingRecordCopy_2_bits_crossRead | _readRecord_T_5503 & chainingRecordCopy_3_bits_crossRead
    | _readRecord_T_5504 & chainingRecordCopy_4_bits_crossRead;
  wire        readRecord_25_crossWrite =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_crossWrite | _readRecord_T_5501 & chainingRecordCopy_1_bits_crossWrite | _readRecord_T_5502 & chainingRecordCopy_2_bits_crossWrite | _readRecord_T_5503
    & chainingRecordCopy_3_bits_crossWrite | _readRecord_T_5504 & chainingRecordCopy_4_bits_crossWrite;
  wire        readRecord_25_gather16 =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_gather16 | _readRecord_T_5501 & chainingRecordCopy_1_bits_gather16 | _readRecord_T_5502 & chainingRecordCopy_2_bits_gather16 | _readRecord_T_5503 & chainingRecordCopy_3_bits_gather16
    | _readRecord_T_5504 & chainingRecordCopy_4_bits_gather16;
  wire        readRecord_25_gather =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_gather | _readRecord_T_5501 & chainingRecordCopy_1_bits_gather | _readRecord_T_5502 & chainingRecordCopy_2_bits_gather | _readRecord_T_5503 & chainingRecordCopy_3_bits_gather
    | _readRecord_T_5504 & chainingRecordCopy_4_bits_gather;
  wire        readRecord_25_st =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_st | _readRecord_T_5501 & chainingRecordCopy_1_bits_st | _readRecord_T_5502 & chainingRecordCopy_2_bits_st | _readRecord_T_5503 & chainingRecordCopy_3_bits_st | _readRecord_T_5504
    & chainingRecordCopy_4_bits_st;
  wire        readRecord_25_ls =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_ls | _readRecord_T_5501 & chainingRecordCopy_1_bits_ls | _readRecord_T_5502 & chainingRecordCopy_2_bits_ls | _readRecord_T_5503 & chainingRecordCopy_3_bits_ls | _readRecord_T_5504
    & chainingRecordCopy_4_bits_ls;
  wire [2:0]  readRecord_25_instIndex =
    (_readRecord_T_5500 ? chainingRecordCopy_0_bits_instIndex : 3'h0) | (_readRecord_T_5501 ? chainingRecordCopy_1_bits_instIndex : 3'h0) | (_readRecord_T_5502 ? chainingRecordCopy_2_bits_instIndex : 3'h0)
    | (_readRecord_T_5503 ? chainingRecordCopy_3_bits_instIndex : 3'h0) | (_readRecord_T_5504 ? chainingRecordCopy_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_25_vs2 =
    (_readRecord_T_5500 ? chainingRecordCopy_0_bits_vs2 : 5'h0) | (_readRecord_T_5501 ? chainingRecordCopy_1_bits_vs2 : 5'h0) | (_readRecord_T_5502 ? chainingRecordCopy_2_bits_vs2 : 5'h0)
    | (_readRecord_T_5503 ? chainingRecordCopy_3_bits_vs2 : 5'h0) | (_readRecord_T_5504 ? chainingRecordCopy_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_25_vs1_bits =
    (_readRecord_T_5500 ? chainingRecordCopy_0_bits_vs1_bits : 5'h0) | (_readRecord_T_5501 ? chainingRecordCopy_1_bits_vs1_bits : 5'h0) | (_readRecord_T_5502 ? chainingRecordCopy_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_5503 ? chainingRecordCopy_3_bits_vs1_bits : 5'h0) | (_readRecord_T_5504 ? chainingRecordCopy_4_bits_vs1_bits : 5'h0);
  wire        readRecord_25_vs1_valid =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_vs1_valid | _readRecord_T_5501 & chainingRecordCopy_1_bits_vs1_valid | _readRecord_T_5502 & chainingRecordCopy_2_bits_vs1_valid | _readRecord_T_5503 & chainingRecordCopy_3_bits_vs1_valid
    | _readRecord_T_5504 & chainingRecordCopy_4_bits_vs1_valid;
  wire [4:0]  readRecord_25_vd_bits =
    (_readRecord_T_5500 ? chainingRecordCopy_0_bits_vd_bits : 5'h0) | (_readRecord_T_5501 ? chainingRecordCopy_1_bits_vd_bits : 5'h0) | (_readRecord_T_5502 ? chainingRecordCopy_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_5503 ? chainingRecordCopy_3_bits_vd_bits : 5'h0) | (_readRecord_T_5504 ? chainingRecordCopy_4_bits_vd_bits : 5'h0);
  wire        readRecord_25_vd_valid =
    _readRecord_T_5500 & chainingRecordCopy_0_bits_vd_valid | _readRecord_T_5501 & chainingRecordCopy_1_bits_vd_valid | _readRecord_T_5502 & chainingRecordCopy_2_bits_vd_valid | _readRecord_T_5503 & chainingRecordCopy_3_bits_vd_valid
    | _readRecord_T_5504 & chainingRecordCopy_4_bits_vd_valid;
  wire [5:0]  address_11 = {readRequests_11_bits_vs_0, readRequests_11_bits_offset_0};
  wire [1:0]  bank_11 = 2'h1 << address_11[0];
  reg         pipeBank_pipe_v_11;
  reg  [1:0]  pipeBank_pipe_b_11;
  reg         pipeBank_pipe_pipe_v_11;
  wire        pipeBank_pipe_pipe_out_11_valid = pipeBank_pipe_pipe_v_11;
  reg  [1:0]  pipeBank_pipe_pipe_b_11;
  wire [1:0]  pipeBank_pipe_pipe_out_11_bits = pipeBank_pipe_pipe_b_11;
  wire [1:0]  bankCorrect_11 = readRequests_11_valid_0 ? bank_11 : 2'h0;
  wire        portConflictCheck_11 =
    ~(write_valid_0 & bank_11 == writeBank & write_bits_vd_0 == readRequests_11_bits_vs_0 & write_bits_offset_0 == readRequests_11_bits_offset_0 | writePipe_valid & bank_11 == writeBankPipe & writePipe_bits_vd == readRequests_11_bits_vs_0
      & writePipe_bits_offset == readRequests_11_bits_offset_0);
  wire        portReady_11 = |(bank_11 & ~_GEN_18);
  assign readRequests_11_ready_0 = portReady_11 & sramReady;
  wire        firstUsed_11 = |(bank_11 & _GEN_17);
  wire [1:0]  bankReadF_11 = bankCorrect_11 & ~_GEN_17;
  wire [1:0]  bankReadS_11 = bankCorrect_11 & ~_GEN_18 & _GEN_17;
  reg         pipeFirstUsed_pipe_v_11;
  reg         pipeFirstUsed_pipe_b_11;
  reg         pipeFirstUsed_pipe_pipe_v_11;
  wire        pipeFirstUsed_pipe_pipe_out_11_valid = pipeFirstUsed_pipe_pipe_v_11;
  reg         pipeFirstUsed_pipe_pipe_b_11;
  wire        pipeFirstUsed_pipe_pipe_out_11_bits = pipeFirstUsed_pipe_pipe_b_11;
  reg         pipeFire_pipe_v_11;
  reg         pipeFire_pipe_b_11;
  reg         pipeFire_pipe_pipe_v_11;
  wire        pipeFire_pipe_pipe_out_11_valid = pipeFire_pipe_pipe_v_11;
  reg         pipeFire_pipe_pipe_b_11;
  wire        pipeFire_pipe_pipe_out_11_bits = pipeFire_pipe_pipe_b_11;
  wire [1:0]  _GEN_19 = _GEN_17 | bankCorrect_11;
  wire [1:0]  _GEN_20 = bankCorrect_11 & _GEN_17 | _GEN_18;
  wire        _readRecord_T_5720 = chainingRecordCopy_0_bits_instIndex == readRequests_12_bits_instructionIndex_0;
  wire        _readRecord_T_5721 = chainingRecordCopy_1_bits_instIndex == readRequests_12_bits_instructionIndex_0;
  wire        _readRecord_T_5722 = chainingRecordCopy_2_bits_instIndex == readRequests_12_bits_instructionIndex_0;
  wire        _readRecord_T_5723 = chainingRecordCopy_3_bits_instIndex == readRequests_12_bits_instructionIndex_0;
  wire        _readRecord_T_5724 = chainingRecordCopy_4_bits_instIndex == readRequests_12_bits_instructionIndex_0;
  wire        readRecord_26_state_wLaneClear =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_state_wLaneClear | _readRecord_T_5721 & chainingRecordCopy_1_bits_state_wLaneClear | _readRecord_T_5722 & chainingRecordCopy_2_bits_state_wLaneClear | _readRecord_T_5723
    & chainingRecordCopy_3_bits_state_wLaneClear | _readRecord_T_5724 & chainingRecordCopy_4_bits_state_wLaneClear;
  wire        readRecord_26_state_wTopLastReport =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_state_wTopLastReport | _readRecord_T_5721 & chainingRecordCopy_1_bits_state_wTopLastReport | _readRecord_T_5722 & chainingRecordCopy_2_bits_state_wTopLastReport | _readRecord_T_5723
    & chainingRecordCopy_3_bits_state_wTopLastReport | _readRecord_T_5724 & chainingRecordCopy_4_bits_state_wTopLastReport;
  wire        readRecord_26_state_wLaneLastReport =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_state_wLaneLastReport | _readRecord_T_5721 & chainingRecordCopy_1_bits_state_wLaneLastReport | _readRecord_T_5722 & chainingRecordCopy_2_bits_state_wLaneLastReport | _readRecord_T_5723
    & chainingRecordCopy_3_bits_state_wLaneLastReport | _readRecord_T_5724 & chainingRecordCopy_4_bits_state_wLaneLastReport;
  wire        readRecord_26_state_wWriteQueueClear =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_state_wWriteQueueClear | _readRecord_T_5721 & chainingRecordCopy_1_bits_state_wWriteQueueClear | _readRecord_T_5722 & chainingRecordCopy_2_bits_state_wWriteQueueClear | _readRecord_T_5723
    & chainingRecordCopy_3_bits_state_wWriteQueueClear | _readRecord_T_5724 & chainingRecordCopy_4_bits_state_wWriteQueueClear;
  wire        readRecord_26_state_stFinish =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_state_stFinish | _readRecord_T_5721 & chainingRecordCopy_1_bits_state_stFinish | _readRecord_T_5722 & chainingRecordCopy_2_bits_state_stFinish | _readRecord_T_5723
    & chainingRecordCopy_3_bits_state_stFinish | _readRecord_T_5724 & chainingRecordCopy_4_bits_state_stFinish;
  wire [15:0] readRecord_26_elementMask =
    (_readRecord_T_5720 ? chainingRecordCopy_0_bits_elementMask : 16'h0) | (_readRecord_T_5721 ? chainingRecordCopy_1_bits_elementMask : 16'h0) | (_readRecord_T_5722 ? chainingRecordCopy_2_bits_elementMask : 16'h0)
    | (_readRecord_T_5723 ? chainingRecordCopy_3_bits_elementMask : 16'h0) | (_readRecord_T_5724 ? chainingRecordCopy_4_bits_elementMask : 16'h0);
  wire        readRecord_26_slow =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_slow | _readRecord_T_5721 & chainingRecordCopy_1_bits_slow | _readRecord_T_5722 & chainingRecordCopy_2_bits_slow | _readRecord_T_5723 & chainingRecordCopy_3_bits_slow | _readRecord_T_5724
    & chainingRecordCopy_4_bits_slow;
  wire        readRecord_26_onlyRead =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_onlyRead | _readRecord_T_5721 & chainingRecordCopy_1_bits_onlyRead | _readRecord_T_5722 & chainingRecordCopy_2_bits_onlyRead | _readRecord_T_5723 & chainingRecordCopy_3_bits_onlyRead
    | _readRecord_T_5724 & chainingRecordCopy_4_bits_onlyRead;
  wire        readRecord_26_ma =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_ma | _readRecord_T_5721 & chainingRecordCopy_1_bits_ma | _readRecord_T_5722 & chainingRecordCopy_2_bits_ma | _readRecord_T_5723 & chainingRecordCopy_3_bits_ma | _readRecord_T_5724
    & chainingRecordCopy_4_bits_ma;
  wire        readRecord_26_indexType =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_indexType | _readRecord_T_5721 & chainingRecordCopy_1_bits_indexType | _readRecord_T_5722 & chainingRecordCopy_2_bits_indexType | _readRecord_T_5723 & chainingRecordCopy_3_bits_indexType
    | _readRecord_T_5724 & chainingRecordCopy_4_bits_indexType;
  wire        readRecord_26_crossRead =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_crossRead | _readRecord_T_5721 & chainingRecordCopy_1_bits_crossRead | _readRecord_T_5722 & chainingRecordCopy_2_bits_crossRead | _readRecord_T_5723 & chainingRecordCopy_3_bits_crossRead
    | _readRecord_T_5724 & chainingRecordCopy_4_bits_crossRead;
  wire        readRecord_26_crossWrite =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_crossWrite | _readRecord_T_5721 & chainingRecordCopy_1_bits_crossWrite | _readRecord_T_5722 & chainingRecordCopy_2_bits_crossWrite | _readRecord_T_5723
    & chainingRecordCopy_3_bits_crossWrite | _readRecord_T_5724 & chainingRecordCopy_4_bits_crossWrite;
  wire        readRecord_26_gather16 =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_gather16 | _readRecord_T_5721 & chainingRecordCopy_1_bits_gather16 | _readRecord_T_5722 & chainingRecordCopy_2_bits_gather16 | _readRecord_T_5723 & chainingRecordCopy_3_bits_gather16
    | _readRecord_T_5724 & chainingRecordCopy_4_bits_gather16;
  wire        readRecord_26_gather =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_gather | _readRecord_T_5721 & chainingRecordCopy_1_bits_gather | _readRecord_T_5722 & chainingRecordCopy_2_bits_gather | _readRecord_T_5723 & chainingRecordCopy_3_bits_gather
    | _readRecord_T_5724 & chainingRecordCopy_4_bits_gather;
  wire        readRecord_26_st =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_st | _readRecord_T_5721 & chainingRecordCopy_1_bits_st | _readRecord_T_5722 & chainingRecordCopy_2_bits_st | _readRecord_T_5723 & chainingRecordCopy_3_bits_st | _readRecord_T_5724
    & chainingRecordCopy_4_bits_st;
  wire        readRecord_26_ls =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_ls | _readRecord_T_5721 & chainingRecordCopy_1_bits_ls | _readRecord_T_5722 & chainingRecordCopy_2_bits_ls | _readRecord_T_5723 & chainingRecordCopy_3_bits_ls | _readRecord_T_5724
    & chainingRecordCopy_4_bits_ls;
  wire [2:0]  readRecord_26_instIndex =
    (_readRecord_T_5720 ? chainingRecordCopy_0_bits_instIndex : 3'h0) | (_readRecord_T_5721 ? chainingRecordCopy_1_bits_instIndex : 3'h0) | (_readRecord_T_5722 ? chainingRecordCopy_2_bits_instIndex : 3'h0)
    | (_readRecord_T_5723 ? chainingRecordCopy_3_bits_instIndex : 3'h0) | (_readRecord_T_5724 ? chainingRecordCopy_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_26_vs2 =
    (_readRecord_T_5720 ? chainingRecordCopy_0_bits_vs2 : 5'h0) | (_readRecord_T_5721 ? chainingRecordCopy_1_bits_vs2 : 5'h0) | (_readRecord_T_5722 ? chainingRecordCopy_2_bits_vs2 : 5'h0)
    | (_readRecord_T_5723 ? chainingRecordCopy_3_bits_vs2 : 5'h0) | (_readRecord_T_5724 ? chainingRecordCopy_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_26_vs1_bits =
    (_readRecord_T_5720 ? chainingRecordCopy_0_bits_vs1_bits : 5'h0) | (_readRecord_T_5721 ? chainingRecordCopy_1_bits_vs1_bits : 5'h0) | (_readRecord_T_5722 ? chainingRecordCopy_2_bits_vs1_bits : 5'h0)
    | (_readRecord_T_5723 ? chainingRecordCopy_3_bits_vs1_bits : 5'h0) | (_readRecord_T_5724 ? chainingRecordCopy_4_bits_vs1_bits : 5'h0);
  wire        readRecord_26_vs1_valid =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_vs1_valid | _readRecord_T_5721 & chainingRecordCopy_1_bits_vs1_valid | _readRecord_T_5722 & chainingRecordCopy_2_bits_vs1_valid | _readRecord_T_5723 & chainingRecordCopy_3_bits_vs1_valid
    | _readRecord_T_5724 & chainingRecordCopy_4_bits_vs1_valid;
  wire [4:0]  readRecord_26_vd_bits =
    (_readRecord_T_5720 ? chainingRecordCopy_0_bits_vd_bits : 5'h0) | (_readRecord_T_5721 ? chainingRecordCopy_1_bits_vd_bits : 5'h0) | (_readRecord_T_5722 ? chainingRecordCopy_2_bits_vd_bits : 5'h0)
    | (_readRecord_T_5723 ? chainingRecordCopy_3_bits_vd_bits : 5'h0) | (_readRecord_T_5724 ? chainingRecordCopy_4_bits_vd_bits : 5'h0);
  wire        readRecord_26_vd_valid =
    _readRecord_T_5720 & chainingRecordCopy_0_bits_vd_valid | _readRecord_T_5721 & chainingRecordCopy_1_bits_vd_valid | _readRecord_T_5722 & chainingRecordCopy_2_bits_vd_valid | _readRecord_T_5723 & chainingRecordCopy_3_bits_vd_valid
    | _readRecord_T_5724 & chainingRecordCopy_4_bits_vd_valid;
  wire [5:0]  address_12 = {readRequests_12_bits_vs_0, readRequests_12_bits_offset_0};
  wire [1:0]  bank_12 = 2'h1 << address_12[0];
  reg         pipeBank_pipe_v_12;
  reg  [1:0]  pipeBank_pipe_b_12;
  reg         pipeBank_pipe_pipe_v_12;
  wire        pipeBank_pipe_pipe_out_12_valid = pipeBank_pipe_pipe_v_12;
  reg  [1:0]  pipeBank_pipe_pipe_b_12;
  wire [1:0]  pipeBank_pipe_pipe_out_12_bits = pipeBank_pipe_pipe_b_12;
  wire [1:0]  bankCorrect_12 = readRequests_12_valid_0 ? bank_12 : 2'h0;
  wire        portConflictCheck_12 =
    ~(write_valid_0 & bank_12 == writeBank & write_bits_vd_0 == readRequests_12_bits_vs_0 & write_bits_offset_0 == readRequests_12_bits_offset_0 | writePipe_valid & bank_12 == writeBankPipe & writePipe_bits_vd == readRequests_12_bits_vs_0
      & writePipe_bits_offset == readRequests_12_bits_offset_0);
  wire        portReady_12 = |(bank_12 & ~_GEN_20);
  assign readRequests_12_ready_0 = portReady_12 & sramReady;
  wire        firstUsed_12 = |(bank_12 & _GEN_19);
  wire [1:0]  bankReadF_12 = bankCorrect_12 & ~_GEN_19;
  wire [1:0]  bankReadS_12 = bankCorrect_12 & ~_GEN_20 & _GEN_19;
  reg         pipeFirstUsed_pipe_v_12;
  reg         pipeFirstUsed_pipe_b_12;
  reg         pipeFirstUsed_pipe_pipe_v_12;
  wire        pipeFirstUsed_pipe_pipe_out_12_valid = pipeFirstUsed_pipe_pipe_v_12;
  reg         pipeFirstUsed_pipe_pipe_b_12;
  wire        pipeFirstUsed_pipe_pipe_out_12_bits = pipeFirstUsed_pipe_pipe_b_12;
  reg         pipeFire_pipe_v_12;
  reg         pipeFire_pipe_b_12;
  reg         pipeFire_pipe_pipe_v_12;
  wire        pipeFire_pipe_pipe_out_12_valid = pipeFire_pipe_pipe_v_12;
  reg         pipeFire_pipe_pipe_b_12;
  wire        pipeFire_pipe_pipe_out_12_bits = pipeFire_pipe_pipe_b_12;
  wire [1:0]  _GEN_21 = _GEN_19 | bankCorrect_12;
  wire [1:0]  _GEN_22 = bankCorrect_12 & _GEN_19 | _GEN_20;
  wire        _loadUpdateValidVec_T_16 = chainingRecordCopy_0_bits_instIndex == readRequests_13_bits_instructionIndex_0;
  wire        _loadUpdateValidVec_T_19 = chainingRecordCopy_1_bits_instIndex == readRequests_13_bits_instructionIndex_0;
  wire        _loadUpdateValidVec_T_22 = chainingRecordCopy_2_bits_instIndex == readRequests_13_bits_instructionIndex_0;
  wire        _loadUpdateValidVec_T_25 = chainingRecordCopy_3_bits_instIndex == readRequests_13_bits_instructionIndex_0;
  wire        _loadUpdateValidVec_T_28 = chainingRecordCopy_4_bits_instIndex == readRequests_13_bits_instructionIndex_0;
  wire        readRecord_27_state_wLaneClear =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_state_wLaneClear | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_state_wLaneClear | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_state_wLaneClear
    | _loadUpdateValidVec_T_25 & chainingRecordCopy_3_bits_state_wLaneClear | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_state_wLaneClear;
  wire        readRecord_27_state_wTopLastReport =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_state_wTopLastReport | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_state_wTopLastReport | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_state_wTopLastReport
    | _loadUpdateValidVec_T_25 & chainingRecordCopy_3_bits_state_wTopLastReport | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_state_wTopLastReport;
  wire        readRecord_27_state_wLaneLastReport =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_state_wLaneLastReport | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_state_wLaneLastReport | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_state_wLaneLastReport
    | _loadUpdateValidVec_T_25 & chainingRecordCopy_3_bits_state_wLaneLastReport | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_state_wLaneLastReport;
  wire        readRecord_27_state_wWriteQueueClear =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_state_wWriteQueueClear | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_state_wWriteQueueClear | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_state_wWriteQueueClear
    | _loadUpdateValidVec_T_25 & chainingRecordCopy_3_bits_state_wWriteQueueClear | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_state_wWriteQueueClear;
  wire        readRecord_27_state_stFinish =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_state_stFinish | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_state_stFinish | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_state_stFinish | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_state_stFinish | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_state_stFinish;
  wire [15:0] readRecord_27_elementMask =
    (_loadUpdateValidVec_T_16 ? chainingRecordCopy_0_bits_elementMask : 16'h0) | (_loadUpdateValidVec_T_19 ? chainingRecordCopy_1_bits_elementMask : 16'h0) | (_loadUpdateValidVec_T_22 ? chainingRecordCopy_2_bits_elementMask : 16'h0)
    | (_loadUpdateValidVec_T_25 ? chainingRecordCopy_3_bits_elementMask : 16'h0) | (_loadUpdateValidVec_T_28 ? chainingRecordCopy_4_bits_elementMask : 16'h0);
  wire        readRecord_27_slow =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_slow | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_slow | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_slow | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_slow | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_slow;
  wire        readRecord_27_onlyRead =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_onlyRead | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_onlyRead | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_onlyRead | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_onlyRead | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_onlyRead;
  wire        readRecord_27_ma =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_ma | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_ma | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_ma | _loadUpdateValidVec_T_25 & chainingRecordCopy_3_bits_ma
    | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_ma;
  wire        readRecord_27_indexType =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_indexType | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_indexType | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_indexType | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_indexType | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_indexType;
  wire        readRecord_27_crossRead =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_crossRead | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_crossRead | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_crossRead | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_crossRead | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_crossRead;
  wire        readRecord_27_crossWrite =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_crossWrite | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_crossWrite | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_crossWrite | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_crossWrite | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_crossWrite;
  wire        readRecord_27_gather16 =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_gather16 | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_gather16 | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_gather16 | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_gather16 | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_gather16;
  wire        readRecord_27_gather =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_gather | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_gather | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_gather | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_gather | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_gather;
  wire        readRecord_27_st =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_st | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_st | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_st | _loadUpdateValidVec_T_25 & chainingRecordCopy_3_bits_st
    | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_st;
  wire        readRecord_27_ls =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_ls | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_ls | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_ls | _loadUpdateValidVec_T_25 & chainingRecordCopy_3_bits_ls
    | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_ls;
  wire [2:0]  readRecord_27_instIndex =
    (_loadUpdateValidVec_T_16 ? chainingRecordCopy_0_bits_instIndex : 3'h0) | (_loadUpdateValidVec_T_19 ? chainingRecordCopy_1_bits_instIndex : 3'h0) | (_loadUpdateValidVec_T_22 ? chainingRecordCopy_2_bits_instIndex : 3'h0)
    | (_loadUpdateValidVec_T_25 ? chainingRecordCopy_3_bits_instIndex : 3'h0) | (_loadUpdateValidVec_T_28 ? chainingRecordCopy_4_bits_instIndex : 3'h0);
  wire [4:0]  readRecord_27_vs2 =
    (_loadUpdateValidVec_T_16 ? chainingRecordCopy_0_bits_vs2 : 5'h0) | (_loadUpdateValidVec_T_19 ? chainingRecordCopy_1_bits_vs2 : 5'h0) | (_loadUpdateValidVec_T_22 ? chainingRecordCopy_2_bits_vs2 : 5'h0)
    | (_loadUpdateValidVec_T_25 ? chainingRecordCopy_3_bits_vs2 : 5'h0) | (_loadUpdateValidVec_T_28 ? chainingRecordCopy_4_bits_vs2 : 5'h0);
  wire [4:0]  readRecord_27_vs1_bits =
    (_loadUpdateValidVec_T_16 ? chainingRecordCopy_0_bits_vs1_bits : 5'h0) | (_loadUpdateValidVec_T_19 ? chainingRecordCopy_1_bits_vs1_bits : 5'h0) | (_loadUpdateValidVec_T_22 ? chainingRecordCopy_2_bits_vs1_bits : 5'h0)
    | (_loadUpdateValidVec_T_25 ? chainingRecordCopy_3_bits_vs1_bits : 5'h0) | (_loadUpdateValidVec_T_28 ? chainingRecordCopy_4_bits_vs1_bits : 5'h0);
  wire        readRecord_27_vs1_valid =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_vs1_valid | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_vs1_valid | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_vs1_valid | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_vs1_valid | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_vs1_valid;
  wire [4:0]  readRecord_27_vd_bits =
    (_loadUpdateValidVec_T_16 ? chainingRecordCopy_0_bits_vd_bits : 5'h0) | (_loadUpdateValidVec_T_19 ? chainingRecordCopy_1_bits_vd_bits : 5'h0) | (_loadUpdateValidVec_T_22 ? chainingRecordCopy_2_bits_vd_bits : 5'h0)
    | (_loadUpdateValidVec_T_25 ? chainingRecordCopy_3_bits_vd_bits : 5'h0) | (_loadUpdateValidVec_T_28 ? chainingRecordCopy_4_bits_vd_bits : 5'h0);
  wire        readRecord_27_vd_valid =
    _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_vd_valid | _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_vd_valid | _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_vd_valid | _loadUpdateValidVec_T_25
    & chainingRecordCopy_3_bits_vd_valid | _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_vd_valid;
  wire        portConflictCheck_13;
  wire        checkResult =
    _checkResult_ChainingCheck_readPort13_record0_checkResult & _checkResult_ChainingCheck_readPort13_record1_checkResult & _checkResult_ChainingCheck_readPort13_record2_checkResult
    & _checkResult_ChainingCheck_readPort13_record3_checkResult & _checkResult_ChainingCheck_readPort13_record4_checkResult & portConflictCheck_13;
  wire        validCorrect = readRequests_13_valid_0 & checkResult;
  wire [5:0]  address_13 = {readRequests_13_bits_vs_0, readRequests_13_bits_offset_0};
  wire [1:0]  bank_13 = 2'h1 << address_13[0];
  reg         pipeBank_pipe_v_13;
  reg  [1:0]  pipeBank_pipe_b_13;
  reg         pipeBank_pipe_pipe_v_13;
  wire        pipeBank_pipe_pipe_out_13_valid = pipeBank_pipe_pipe_v_13;
  reg  [1:0]  pipeBank_pipe_pipe_b_13;
  wire [1:0]  pipeBank_pipe_pipe_out_13_bits = pipeBank_pipe_pipe_b_13;
  wire [1:0]  bankCorrect_13 = validCorrect ? bank_13 : 2'h0;
  assign portConflictCheck_13 =
    ~(write_valid_0 & bank_13 == writeBank & write_bits_vd_0 == readRequests_13_bits_vs_0 & write_bits_offset_0 == readRequests_13_bits_offset_0 | writePipe_valid & bank_13 == writeBankPipe & writePipe_bits_vd == readRequests_13_bits_vs_0
      & writePipe_bits_offset == readRequests_13_bits_offset_0);
  wire        portReady_13 = (|(bank_13 & ~_GEN_22)) & checkResult;
  assign readRequests_13_ready_0 = portReady_13 & sramReady;
  wire        firstUsed_13 = |(bank_13 & _GEN_21);
  wire [1:0]  bankReadF_13 = bankCorrect_13 & ~_GEN_21;
  wire [1:0]  bankReadS_13 = bankCorrect_13 & ~_GEN_22 & _GEN_21;
  reg         pipeFirstUsed_pipe_v_13;
  reg         pipeFirstUsed_pipe_b_13;
  reg         pipeFirstUsed_pipe_pipe_v_13;
  wire        pipeFirstUsed_pipe_pipe_out_13_valid = pipeFirstUsed_pipe_pipe_v_13;
  reg         pipeFirstUsed_pipe_pipe_b_13;
  wire        pipeFirstUsed_pipe_pipe_out_13_bits = pipeFirstUsed_pipe_pipe_b_13;
  reg         pipeFire_pipe_v_13;
  reg         pipeFire_pipe_b_13;
  reg         pipeFire_pipe_pipe_v_13;
  wire        pipeFire_pipe_pipe_out_13_valid = pipeFire_pipe_pipe_v_13;
  reg         pipeFire_pipe_pipe_b_13;
  wire        pipeFire_pipe_pipe_out_13_bits = pipeFire_pipe_pipe_b_13;
  wire [1:0]  firstOccupied = _GEN_21 | bankCorrect_13;
  wire [1:0]  secondOccupied = bankCorrect_13 & _GEN_21 | _GEN_22;
  assign write_ready_0 = sramReady & (|(writeBank & ~secondOccupied));
  wire [31:0] writeData = resetValid ? 32'h0 : writePipe_bits_data;
  wire [4:0]  writeAddress = resetValid ? sramResetCount : writePipe_bits_vd;
  wire        ramWriteValid;
  wire        ramWriteValid_1;
  wire        writeValid = writePipe_valid & writeBankPipe[0];
  assign ramWriteValid = writeValid | resetValid;
  wire [35:0] vrfSRAM_0_readwritePorts_0_readData;
  wire        vrfSRAM_0_readwritePorts_1_isWrite = ramWriteValid;
  assign readResultF_0 = vrfSRAM_0_readwritePorts_0_readData[31:0];
  wire [35:0] vrfSRAM_0_readwritePorts_1_readData;
  assign readResultS_0 = vrfSRAM_0_readwritePorts_1_readData[31:0];
  wire [4:0]  vrfSRAM_0_readwritePorts_1_address = ramWriteValid ? writeAddress : secondReadPipe_0_bits_address;
  wire        vrfSRAM_0_readwritePorts_1_enable = ramWriteValid | secondReadPipe_0_valid;
  wire [35:0] _GEN_23 = {4'h0, writeData};
  wire [35:0] vrfSRAM_0_readwritePorts_1_writeData;
  assign vrfSRAM_0_readwritePorts_1_writeData = _GEN_23;
  wire [35:0] vrfSRAM_1_readwritePorts_1_writeData;
  assign vrfSRAM_1_readwritePorts_1_writeData = _GEN_23;
  wire        writeValid_1 = writePipe_valid & writeBankPipe[1];
  assign ramWriteValid_1 = writeValid_1 | resetValid;
  wire [35:0] vrfSRAM_1_readwritePorts_0_readData;
  wire        vrfSRAM_1_readwritePorts_1_isWrite = ramWriteValid_1;
  assign readResultF_1 = vrfSRAM_1_readwritePorts_0_readData[31:0];
  wire [35:0] vrfSRAM_1_readwritePorts_1_readData;
  assign readResultS_1 = vrfSRAM_1_readwritePorts_1_readData[31:0];
  wire [4:0]  vrfSRAM_1_readwritePorts_1_address = ramWriteValid_1 ? writeAddress : secondReadPipe_1_bits_address;
  wire        vrfSRAM_1_readwritePorts_1_enable = ramWriteValid_1 | secondReadPipe_1_valid;
  wire [1:0]  freeRecord_lo = {~chainingRecord_1_valid, ~chainingRecord_0_valid};
  wire [1:0]  freeRecord_hi_hi = {~chainingRecord_4_valid, ~chainingRecord_3_valid};
  wire [2:0]  freeRecord_hi = {freeRecord_hi_hi, ~chainingRecord_2_valid};
  wire [4:0]  freeRecord = {freeRecord_hi, freeRecord_lo};
  wire [3:0]  _recordFFO_T_2 = freeRecord[3:0] | {freeRecord[2:0], 1'h0};
  wire [4:0]  recordFFO = {~(_recordFFO_T_2 | {_recordFFO_T_2[1:0], 2'h0}), 1'h1} & freeRecord;
  wire [4:0]  recordEnq = instructionWriteReport_valid ? recordFFO : 5'h0;
  wire [15:0] writeOH_0 = 16'h1 << {12'h0, writePipe_bits_vd[2:0] - chainingRecord_0_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [15:0] loadReadOH_0 = 16'h1 << {12'h0, readRequests_13_bits_vs_0[2:0] - chainingRecord_0_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]  _GEN_24 = {5'h0, chainingRecord_0_bits_instIndex};
  wire        dataInLsuQueue = |(8'h1 << _GEN_24 & loadDataInLSUWriteQueue);
  wire        writeUpdateValidVec_0 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecord_0_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecord_0_bits_ls);
  wire [15:0] writeUpdate1HVec_0 = writeUpdateValidVec_0 ? writeOH_0 : 16'h0;
  wire        loadUpdateValidVec_0 = _loadUpdateValidVec_T_27 & readRequests_13_bits_instructionIndex_0 == chainingRecord_0_bits_instIndex & chainingRecord_0_bits_st;
  wire [15:0] loadUpdate1HVec_0 = loadUpdateValidVec_0 ? loadReadOH_0 : 16'h0;
  wire        elementUpdateValid = writeUpdateValidVec_0 | loadUpdateValidVec_0;
  wire [15:0] elementUpdate1H = writeUpdate1HVec_0 | loadUpdate1HVec_0;
  wire        dataInLaneCheck = |(8'h1 << _GEN_24 & dataInLane);
  wire        laneLastReport = |(8'h1 << _GEN_24 & instructionLastReport);
  wire        topLastReport = |(8'h1 << _GEN_24 & lsuLastReport);
  wire        waitLaneClear = chainingRecord_0_bits_state_stFinish & chainingRecord_0_bits_state_wWriteQueueClear & chainingRecord_0_bits_state_wLaneLastReport & chainingRecord_0_bits_state_wTopLastReport;
  wire        stateClear = waitLaneClear & chainingRecord_0_bits_state_wLaneClear | (&chainingRecord_0_bits_elementMask) & ~chainingRecord_0_bits_onlyRead;
  wire [15:0] writeOH_0_1 = 16'h1 << {12'h0, writePipe_bits_vd[2:0] - chainingRecord_1_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [15:0] loadReadOH_0_1 = 16'h1 << {12'h0, readRequests_13_bits_vs_0[2:0] - chainingRecord_1_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]  _GEN_25 = {5'h0, chainingRecord_1_bits_instIndex};
  wire        dataInLsuQueue_1 = |(8'h1 << _GEN_25 & loadDataInLSUWriteQueue);
  wire        writeUpdateValidVec_0_1 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecord_1_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecord_1_bits_ls);
  wire [15:0] writeUpdate1HVec_0_1 = writeUpdateValidVec_0_1 ? writeOH_0_1 : 16'h0;
  wire        loadUpdateValidVec_0_1 = _loadUpdateValidVec_T_27 & readRequests_13_bits_instructionIndex_0 == chainingRecord_1_bits_instIndex & chainingRecord_1_bits_st;
  wire [15:0] loadUpdate1HVec_0_1 = loadUpdateValidVec_0_1 ? loadReadOH_0_1 : 16'h0;
  wire        elementUpdateValid_1 = writeUpdateValidVec_0_1 | loadUpdateValidVec_0_1;
  wire [15:0] elementUpdate1H_1 = writeUpdate1HVec_0_1 | loadUpdate1HVec_0_1;
  wire        dataInLaneCheck_1 = |(8'h1 << _GEN_25 & dataInLane);
  wire        laneLastReport_1 = |(8'h1 << _GEN_25 & instructionLastReport);
  wire        topLastReport_1 = |(8'h1 << _GEN_25 & lsuLastReport);
  wire        waitLaneClear_1 = chainingRecord_1_bits_state_stFinish & chainingRecord_1_bits_state_wWriteQueueClear & chainingRecord_1_bits_state_wLaneLastReport & chainingRecord_1_bits_state_wTopLastReport;
  wire        stateClear_1 = waitLaneClear_1 & chainingRecord_1_bits_state_wLaneClear | (&chainingRecord_1_bits_elementMask) & ~chainingRecord_1_bits_onlyRead;
  wire [15:0] writeOH_0_2 = 16'h1 << {12'h0, writePipe_bits_vd[2:0] - chainingRecord_2_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [15:0] loadReadOH_0_2 = 16'h1 << {12'h0, readRequests_13_bits_vs_0[2:0] - chainingRecord_2_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]  _GEN_26 = {5'h0, chainingRecord_2_bits_instIndex};
  wire        dataInLsuQueue_2 = |(8'h1 << _GEN_26 & loadDataInLSUWriteQueue);
  wire        writeUpdateValidVec_0_2 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecord_2_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecord_2_bits_ls);
  wire [15:0] writeUpdate1HVec_0_2 = writeUpdateValidVec_0_2 ? writeOH_0_2 : 16'h0;
  wire        loadUpdateValidVec_0_2 = _loadUpdateValidVec_T_27 & readRequests_13_bits_instructionIndex_0 == chainingRecord_2_bits_instIndex & chainingRecord_2_bits_st;
  wire [15:0] loadUpdate1HVec_0_2 = loadUpdateValidVec_0_2 ? loadReadOH_0_2 : 16'h0;
  wire        elementUpdateValid_2 = writeUpdateValidVec_0_2 | loadUpdateValidVec_0_2;
  wire [15:0] elementUpdate1H_2 = writeUpdate1HVec_0_2 | loadUpdate1HVec_0_2;
  wire        dataInLaneCheck_2 = |(8'h1 << _GEN_26 & dataInLane);
  wire        laneLastReport_2 = |(8'h1 << _GEN_26 & instructionLastReport);
  wire        topLastReport_2 = |(8'h1 << _GEN_26 & lsuLastReport);
  wire        waitLaneClear_2 = chainingRecord_2_bits_state_stFinish & chainingRecord_2_bits_state_wWriteQueueClear & chainingRecord_2_bits_state_wLaneLastReport & chainingRecord_2_bits_state_wTopLastReport;
  wire        stateClear_2 = waitLaneClear_2 & chainingRecord_2_bits_state_wLaneClear | (&chainingRecord_2_bits_elementMask) & ~chainingRecord_2_bits_onlyRead;
  wire [15:0] writeOH_0_3 = 16'h1 << {12'h0, writePipe_bits_vd[2:0] - chainingRecord_3_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [15:0] loadReadOH_0_3 = 16'h1 << {12'h0, readRequests_13_bits_vs_0[2:0] - chainingRecord_3_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]  _GEN_27 = {5'h0, chainingRecord_3_bits_instIndex};
  wire        dataInLsuQueue_3 = |(8'h1 << _GEN_27 & loadDataInLSUWriteQueue);
  wire        writeUpdateValidVec_0_3 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecord_3_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecord_3_bits_ls);
  wire [15:0] writeUpdate1HVec_0_3 = writeUpdateValidVec_0_3 ? writeOH_0_3 : 16'h0;
  wire        loadUpdateValidVec_0_3 = _loadUpdateValidVec_T_27 & readRequests_13_bits_instructionIndex_0 == chainingRecord_3_bits_instIndex & chainingRecord_3_bits_st;
  wire [15:0] loadUpdate1HVec_0_3 = loadUpdateValidVec_0_3 ? loadReadOH_0_3 : 16'h0;
  wire        elementUpdateValid_3 = writeUpdateValidVec_0_3 | loadUpdateValidVec_0_3;
  wire [15:0] elementUpdate1H_3 = writeUpdate1HVec_0_3 | loadUpdate1HVec_0_3;
  wire        dataInLaneCheck_3 = |(8'h1 << _GEN_27 & dataInLane);
  wire        laneLastReport_3 = |(8'h1 << _GEN_27 & instructionLastReport);
  wire        topLastReport_3 = |(8'h1 << _GEN_27 & lsuLastReport);
  wire        waitLaneClear_3 = chainingRecord_3_bits_state_stFinish & chainingRecord_3_bits_state_wWriteQueueClear & chainingRecord_3_bits_state_wLaneLastReport & chainingRecord_3_bits_state_wTopLastReport;
  wire        stateClear_3 = waitLaneClear_3 & chainingRecord_3_bits_state_wLaneClear | (&chainingRecord_3_bits_elementMask) & ~chainingRecord_3_bits_onlyRead;
  wire [15:0] writeOH_0_4 = 16'h1 << {12'h0, writePipe_bits_vd[2:0] - chainingRecord_4_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [15:0] loadReadOH_0_4 = 16'h1 << {12'h0, readRequests_13_bits_vs_0[2:0] - chainingRecord_4_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]  _GEN_28 = {5'h0, chainingRecord_4_bits_instIndex};
  wire        dataInLsuQueue_4 = |(8'h1 << _GEN_28 & loadDataInLSUWriteQueue);
  wire        writeUpdateValidVec_0_4 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecord_4_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecord_4_bits_ls);
  wire [15:0] writeUpdate1HVec_0_4 = writeUpdateValidVec_0_4 ? writeOH_0_4 : 16'h0;
  wire        loadUpdateValidVec_0_4 = _loadUpdateValidVec_T_27 & readRequests_13_bits_instructionIndex_0 == chainingRecord_4_bits_instIndex & chainingRecord_4_bits_st;
  wire [15:0] loadUpdate1HVec_0_4 = loadUpdateValidVec_0_4 ? loadReadOH_0_4 : 16'h0;
  wire        elementUpdateValid_4 = writeUpdateValidVec_0_4 | loadUpdateValidVec_0_4;
  wire [15:0] elementUpdate1H_4 = writeUpdate1HVec_0_4 | loadUpdate1HVec_0_4;
  wire        dataInLaneCheck_4 = |(8'h1 << _GEN_28 & dataInLane);
  wire        laneLastReport_4 = |(8'h1 << _GEN_28 & instructionLastReport);
  wire        topLastReport_4 = |(8'h1 << _GEN_28 & lsuLastReport);
  wire        waitLaneClear_4 = chainingRecord_4_bits_state_stFinish & chainingRecord_4_bits_state_wWriteQueueClear & chainingRecord_4_bits_state_wLaneLastReport & chainingRecord_4_bits_state_wTopLastReport;
  wire        stateClear_4 = waitLaneClear_4 & chainingRecord_4_bits_state_wLaneClear | (&chainingRecord_4_bits_elementMask) & ~chainingRecord_4_bits_onlyRead;
  wire [15:0] writeOH_0_5 = 16'h1 << {12'h0, writePipe_bits_vd[2:0] - chainingRecordCopy_0_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [15:0] loadReadOH_0_5 = 16'h1 << {12'h0, readRequests_13_bits_vs_0[2:0] - chainingRecordCopy_0_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]  _GEN_29 = {5'h0, chainingRecordCopy_0_bits_instIndex};
  wire        dataInLsuQueue_5 = |(8'h1 << _GEN_29 & loadDataInLSUWriteQueue);
  wire        writeUpdateValidVec_0_5 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecordCopy_0_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecordCopy_0_bits_ls);
  wire [15:0] writeUpdate1HVec_0_5 = writeUpdateValidVec_0_5 ? writeOH_0_5 : 16'h0;
  wire        loadUpdateValidVec_0_5 = _loadUpdateValidVec_T_27 & _loadUpdateValidVec_T_16 & chainingRecordCopy_0_bits_st;
  wire [15:0] loadUpdate1HVec_0_5 = loadUpdateValidVec_0_5 ? loadReadOH_0_5 : 16'h0;
  wire        elementUpdateValid_5 = writeUpdateValidVec_0_5 | loadUpdateValidVec_0_5;
  wire [15:0] elementUpdate1H_5 = writeUpdate1HVec_0_5 | loadUpdate1HVec_0_5;
  wire        dataInLaneCheck_5 = |(8'h1 << _GEN_29 & dataInLane);
  wire        laneLastReport_5 = |(8'h1 << _GEN_29 & instructionLastReport);
  wire        topLastReport_5 = |(8'h1 << _GEN_29 & lsuLastReport);
  wire        waitLaneClear_5 = chainingRecordCopy_0_bits_state_stFinish & chainingRecordCopy_0_bits_state_wWriteQueueClear & chainingRecordCopy_0_bits_state_wLaneLastReport & chainingRecordCopy_0_bits_state_wTopLastReport;
  wire        stateClear_5 = waitLaneClear_5 & chainingRecordCopy_0_bits_state_wLaneClear | (&chainingRecordCopy_0_bits_elementMask) & ~chainingRecordCopy_0_bits_onlyRead;
  wire [7:0]  recordRelease_0 = stateClear_5 & chainingRecordCopy_0_valid ? 8'h1 << _GEN_29 : stateClear & chainingRecord_0_valid ? 8'h1 << _GEN_24 : 8'h0;
  wire [15:0] writeOH_0_6 = 16'h1 << {12'h0, writePipe_bits_vd[2:0] - chainingRecordCopy_1_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [15:0] loadReadOH_0_6 = 16'h1 << {12'h0, readRequests_13_bits_vs_0[2:0] - chainingRecordCopy_1_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]  _GEN_30 = {5'h0, chainingRecordCopy_1_bits_instIndex};
  wire        dataInLsuQueue_6 = |(8'h1 << _GEN_30 & loadDataInLSUWriteQueue);
  wire        writeUpdateValidVec_0_6 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecordCopy_1_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecordCopy_1_bits_ls);
  wire [15:0] writeUpdate1HVec_0_6 = writeUpdateValidVec_0_6 ? writeOH_0_6 : 16'h0;
  wire        loadUpdateValidVec_0_6 = _loadUpdateValidVec_T_27 & _loadUpdateValidVec_T_19 & chainingRecordCopy_1_bits_st;
  wire [15:0] loadUpdate1HVec_0_6 = loadUpdateValidVec_0_6 ? loadReadOH_0_6 : 16'h0;
  wire        elementUpdateValid_6 = writeUpdateValidVec_0_6 | loadUpdateValidVec_0_6;
  wire [15:0] elementUpdate1H_6 = writeUpdate1HVec_0_6 | loadUpdate1HVec_0_6;
  wire        dataInLaneCheck_6 = |(8'h1 << _GEN_30 & dataInLane);
  wire        laneLastReport_6 = |(8'h1 << _GEN_30 & instructionLastReport);
  wire        topLastReport_6 = |(8'h1 << _GEN_30 & lsuLastReport);
  wire        waitLaneClear_6 = chainingRecordCopy_1_bits_state_stFinish & chainingRecordCopy_1_bits_state_wWriteQueueClear & chainingRecordCopy_1_bits_state_wLaneLastReport & chainingRecordCopy_1_bits_state_wTopLastReport;
  wire        stateClear_6 = waitLaneClear_6 & chainingRecordCopy_1_bits_state_wLaneClear | (&chainingRecordCopy_1_bits_elementMask) & ~chainingRecordCopy_1_bits_onlyRead;
  wire [7:0]  recordRelease_1 = stateClear_6 & chainingRecordCopy_1_valid ? 8'h1 << _GEN_30 : stateClear_1 & chainingRecord_1_valid ? 8'h1 << _GEN_25 : 8'h0;
  wire [15:0] writeOH_0_7 = 16'h1 << {12'h0, writePipe_bits_vd[2:0] - chainingRecordCopy_2_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [15:0] loadReadOH_0_7 = 16'h1 << {12'h0, readRequests_13_bits_vs_0[2:0] - chainingRecordCopy_2_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]  _GEN_31 = {5'h0, chainingRecordCopy_2_bits_instIndex};
  wire        dataInLsuQueue_7 = |(8'h1 << _GEN_31 & loadDataInLSUWriteQueue);
  wire        writeUpdateValidVec_0_7 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecordCopy_2_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecordCopy_2_bits_ls);
  wire [15:0] writeUpdate1HVec_0_7 = writeUpdateValidVec_0_7 ? writeOH_0_7 : 16'h0;
  wire        loadUpdateValidVec_0_7 = _loadUpdateValidVec_T_27 & _loadUpdateValidVec_T_22 & chainingRecordCopy_2_bits_st;
  wire [15:0] loadUpdate1HVec_0_7 = loadUpdateValidVec_0_7 ? loadReadOH_0_7 : 16'h0;
  wire        elementUpdateValid_7 = writeUpdateValidVec_0_7 | loadUpdateValidVec_0_7;
  wire [15:0] elementUpdate1H_7 = writeUpdate1HVec_0_7 | loadUpdate1HVec_0_7;
  wire        dataInLaneCheck_7 = |(8'h1 << _GEN_31 & dataInLane);
  wire        laneLastReport_7 = |(8'h1 << _GEN_31 & instructionLastReport);
  wire        topLastReport_7 = |(8'h1 << _GEN_31 & lsuLastReport);
  wire        waitLaneClear_7 = chainingRecordCopy_2_bits_state_stFinish & chainingRecordCopy_2_bits_state_wWriteQueueClear & chainingRecordCopy_2_bits_state_wLaneLastReport & chainingRecordCopy_2_bits_state_wTopLastReport;
  wire        stateClear_7 = waitLaneClear_7 & chainingRecordCopy_2_bits_state_wLaneClear | (&chainingRecordCopy_2_bits_elementMask) & ~chainingRecordCopy_2_bits_onlyRead;
  wire [7:0]  recordRelease_2 = stateClear_7 & chainingRecordCopy_2_valid ? 8'h1 << _GEN_31 : stateClear_2 & chainingRecord_2_valid ? 8'h1 << _GEN_26 : 8'h0;
  wire [15:0] writeOH_0_8 = 16'h1 << {12'h0, writePipe_bits_vd[2:0] - chainingRecordCopy_3_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [15:0] loadReadOH_0_8 = 16'h1 << {12'h0, readRequests_13_bits_vs_0[2:0] - chainingRecordCopy_3_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]  _GEN_32 = {5'h0, chainingRecordCopy_3_bits_instIndex};
  wire        dataInLsuQueue_8 = |(8'h1 << _GEN_32 & loadDataInLSUWriteQueue);
  wire        writeUpdateValidVec_0_8 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecordCopy_3_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecordCopy_3_bits_ls);
  wire [15:0] writeUpdate1HVec_0_8 = writeUpdateValidVec_0_8 ? writeOH_0_8 : 16'h0;
  wire        loadUpdateValidVec_0_8 = _loadUpdateValidVec_T_27 & _loadUpdateValidVec_T_25 & chainingRecordCopy_3_bits_st;
  wire [15:0] loadUpdate1HVec_0_8 = loadUpdateValidVec_0_8 ? loadReadOH_0_8 : 16'h0;
  wire        elementUpdateValid_8 = writeUpdateValidVec_0_8 | loadUpdateValidVec_0_8;
  wire [15:0] elementUpdate1H_8 = writeUpdate1HVec_0_8 | loadUpdate1HVec_0_8;
  wire        dataInLaneCheck_8 = |(8'h1 << _GEN_32 & dataInLane);
  wire        laneLastReport_8 = |(8'h1 << _GEN_32 & instructionLastReport);
  wire        topLastReport_8 = |(8'h1 << _GEN_32 & lsuLastReport);
  wire        waitLaneClear_8 = chainingRecordCopy_3_bits_state_stFinish & chainingRecordCopy_3_bits_state_wWriteQueueClear & chainingRecordCopy_3_bits_state_wLaneLastReport & chainingRecordCopy_3_bits_state_wTopLastReport;
  wire        stateClear_8 = waitLaneClear_8 & chainingRecordCopy_3_bits_state_wLaneClear | (&chainingRecordCopy_3_bits_elementMask) & ~chainingRecordCopy_3_bits_onlyRead;
  wire [7:0]  recordRelease_3 = stateClear_8 & chainingRecordCopy_3_valid ? 8'h1 << _GEN_32 : stateClear_3 & chainingRecord_3_valid ? 8'h1 << _GEN_27 : 8'h0;
  wire [15:0] writeOH_0_9 = 16'h1 << {12'h0, writePipe_bits_vd[2:0] - chainingRecordCopy_4_bits_vd_bits[2:0], writePipe_bits_offset};
  wire [15:0] loadReadOH_0_9 = 16'h1 << {12'h0, readRequests_13_bits_vs_0[2:0] - chainingRecordCopy_4_bits_vs2[2:0], readRequests_13_bits_offset_0};
  wire [7:0]  _GEN_33 = {5'h0, chainingRecordCopy_4_bits_instIndex};
  wire        dataInLsuQueue_9 = |(8'h1 << _GEN_33 & loadDataInLSUWriteQueue);
  wire        writeUpdateValidVec_0_9 = writePipe_valid & writePipe_bits_instructionIndex == chainingRecordCopy_4_bits_instIndex & (writePipe_bits_mask[3] | ~chainingRecordCopy_4_bits_ls);
  wire [15:0] writeUpdate1HVec_0_9 = writeUpdateValidVec_0_9 ? writeOH_0_9 : 16'h0;
  wire        loadUpdateValidVec_0_9 = _loadUpdateValidVec_T_27 & _loadUpdateValidVec_T_28 & chainingRecordCopy_4_bits_st;
  wire [15:0] loadUpdate1HVec_0_9 = loadUpdateValidVec_0_9 ? loadReadOH_0_9 : 16'h0;
  wire        elementUpdateValid_9 = writeUpdateValidVec_0_9 | loadUpdateValidVec_0_9;
  wire [15:0] elementUpdate1H_9 = writeUpdate1HVec_0_9 | loadUpdate1HVec_0_9;
  wire        dataInLaneCheck_9 = |(8'h1 << _GEN_33 & dataInLane);
  wire        laneLastReport_9 = |(8'h1 << _GEN_33 & instructionLastReport);
  wire        topLastReport_9 = |(8'h1 << _GEN_33 & lsuLastReport);
  wire        waitLaneClear_9 = chainingRecordCopy_4_bits_state_stFinish & chainingRecordCopy_4_bits_state_wWriteQueueClear & chainingRecordCopy_4_bits_state_wLaneLastReport & chainingRecordCopy_4_bits_state_wTopLastReport;
  wire        stateClear_9 = waitLaneClear_9 & chainingRecordCopy_4_bits_state_wLaneClear | (&chainingRecordCopy_4_bits_elementMask) & ~chainingRecordCopy_4_bits_onlyRead;
  wire [7:0]  recordRelease_4 = stateClear_9 & chainingRecordCopy_4_valid ? 8'h1 << _GEN_33 : stateClear_4 & chainingRecord_4_valid ? 8'h1 << _GEN_28 : 8'h0;
  wire        _hazardVec_isStore_T_6 = chainingRecordCopy_0_valid & chainingRecordCopy_0_bits_ls;
  wire        _GEN_34 = _hazardVec_isStore_T_6 & ~chainingRecordCopy_0_bits_st;
  wire        hazardVec_isLoad_0;
  assign hazardVec_isLoad_0 = _GEN_34;
  wire        hazardVec_isLoad_0_1;
  assign hazardVec_isLoad_0_1 = _GEN_34;
  wire        hazardVec_isLoad_0_2;
  assign hazardVec_isLoad_0_2 = _GEN_34;
  wire        hazardVec_isLoad_0_3;
  assign hazardVec_isLoad_0_3 = _GEN_34;
  wire        _hazardVec_isStore_T_12 = chainingRecordCopy_1_valid & chainingRecordCopy_1_bits_ls;
  wire        _GEN_35 = _hazardVec_isStore_T_12 & ~chainingRecordCopy_1_bits_st;
  wire        hazardVec_isLoad_1;
  assign hazardVec_isLoad_1 = _GEN_35;
  wire        hazardVec_isLoad_0_4;
  assign hazardVec_isLoad_0_4 = _GEN_35;
  wire        hazardVec_isLoad_0_5;
  assign hazardVec_isLoad_0_5 = _GEN_35;
  wire        hazardVec_isLoad_0_6;
  assign hazardVec_isLoad_0_6 = _GEN_35;
  wire        _GEN_36 = _hazardVec_isStore_T_6 & chainingRecordCopy_0_bits_st;
  wire        hazardVec_isStore_0;
  assign hazardVec_isStore_0 = _GEN_36;
  wire        hazardVec_isStore_0_1;
  assign hazardVec_isStore_0_1 = _GEN_36;
  wire        hazardVec_isStore_0_2;
  assign hazardVec_isStore_0_2 = _GEN_36;
  wire        hazardVec_isStore_0_3;
  assign hazardVec_isStore_0_3 = _GEN_36;
  wire        _GEN_37 = _hazardVec_isStore_T_12 & chainingRecordCopy_1_bits_st;
  wire        hazardVec_isStore_1;
  assign hazardVec_isStore_1 = _GEN_37;
  wire        hazardVec_isStore_0_4;
  assign hazardVec_isStore_0_4 = _GEN_37;
  wire        hazardVec_isStore_0_5;
  assign hazardVec_isStore_0_5 = _GEN_37;
  wire        hazardVec_isStore_0_6;
  assign hazardVec_isStore_0_6 = _GEN_37;
  wire        _GEN_38 = chainingRecordCopy_0_valid & chainingRecordCopy_0_bits_slow;
  wire        hazardVec_isSlow_0;
  assign hazardVec_isSlow_0 = _GEN_38;
  wire        hazardVec_isSlow_0_1;
  assign hazardVec_isSlow_0_1 = _GEN_38;
  wire        hazardVec_isSlow_0_2;
  assign hazardVec_isSlow_0_2 = _GEN_38;
  wire        hazardVec_isSlow_0_3;
  assign hazardVec_isSlow_0_3 = _GEN_38;
  wire        _GEN_39 = chainingRecordCopy_1_valid & chainingRecordCopy_1_bits_slow;
  wire        hazardVec_isSlow_1;
  assign hazardVec_isSlow_1 = _GEN_39;
  wire        hazardVec_isSlow_0_4;
  assign hazardVec_isSlow_0_4 = _GEN_39;
  wire        hazardVec_isSlow_0_5;
  assign hazardVec_isSlow_0_5 = _GEN_39;
  wire        hazardVec_isSlow_0_6;
  assign hazardVec_isSlow_0_6 = _GEN_39;
  wire        hazardVec_samVd =
    chainingRecordCopy_0_bits_vd_valid & chainingRecordCopy_1_bits_vd_valid & chainingRecordCopy_0_bits_vd_bits[4:3] == chainingRecordCopy_1_bits_vd_bits[4:3]
    & (chainingRecordCopy_0_bits_elementMask | chainingRecordCopy_1_bits_elementMask) != 16'hFFFF;
  wire        hazardVec_sourceVdEqSinkVs =
    chainingRecordCopy_0_bits_vd_valid & (chainingRecordCopy_0_bits_vd_bits == chainingRecordCopy_1_bits_vs2 | chainingRecordCopy_0_bits_vd_bits == chainingRecordCopy_1_bits_vs1_bits & chainingRecordCopy_1_bits_vs1_valid);
  wire        hazardVec_sinkVdEqSourceVs =
    chainingRecordCopy_1_bits_vd_valid & (chainingRecordCopy_1_bits_vd_bits == chainingRecordCopy_0_bits_vs2 | chainingRecordCopy_1_bits_vd_bits == chainingRecordCopy_0_bits_vs1_bits & chainingRecordCopy_0_bits_vs1_valid);
  wire        hazardVec_older =
    chainingRecordCopy_1_bits_instIndex == chainingRecordCopy_0_bits_instIndex | chainingRecordCopy_1_bits_instIndex[1:0] < chainingRecordCopy_0_bits_instIndex[1:0] ^ chainingRecordCopy_1_bits_instIndex[2]
    ^ chainingRecordCopy_0_bits_instIndex[2];
  wire        hazardVec_hazardForeLoad =
    (hazardVec_older ? hazardVec_isLoad_0 & hazardVec_isSlow_1 : hazardVec_isLoad_1 & hazardVec_isSlow_0) & (hazardVec_samVd | (hazardVec_older ? hazardVec_sourceVdEqSinkVs : hazardVec_sinkVdEqSourceVs));
  wire        hazardVec_rawForeStore = (hazardVec_older ? hazardVec_isStore_0 & hazardVec_isSlow_1 : hazardVec_isStore_1 & hazardVec_isSlow_0) & hazardVec_samVd;
  wire        _hazardVec_isStore_T_16 = chainingRecordCopy_2_valid & chainingRecordCopy_2_bits_ls;
  wire        _GEN_40 = _hazardVec_isStore_T_16 & ~chainingRecordCopy_2_bits_st;
  wire        hazardVec_isLoad_1_1;
  assign hazardVec_isLoad_1_1 = _GEN_40;
  wire        hazardVec_isLoad_1_4;
  assign hazardVec_isLoad_1_4 = _GEN_40;
  wire        hazardVec_isLoad_0_7;
  assign hazardVec_isLoad_0_7 = _GEN_40;
  wire        hazardVec_isLoad_0_8;
  assign hazardVec_isLoad_0_8 = _GEN_40;
  wire        _GEN_41 = _hazardVec_isStore_T_16 & chainingRecordCopy_2_bits_st;
  wire        hazardVec_isStore_1_1;
  assign hazardVec_isStore_1_1 = _GEN_41;
  wire        hazardVec_isStore_1_4;
  assign hazardVec_isStore_1_4 = _GEN_41;
  wire        hazardVec_isStore_0_7;
  assign hazardVec_isStore_0_7 = _GEN_41;
  wire        hazardVec_isStore_0_8;
  assign hazardVec_isStore_0_8 = _GEN_41;
  wire        _GEN_42 = chainingRecordCopy_2_valid & chainingRecordCopy_2_bits_slow;
  wire        hazardVec_isSlow_1_1;
  assign hazardVec_isSlow_1_1 = _GEN_42;
  wire        hazardVec_isSlow_1_4;
  assign hazardVec_isSlow_1_4 = _GEN_42;
  wire        hazardVec_isSlow_0_7;
  assign hazardVec_isSlow_0_7 = _GEN_42;
  wire        hazardVec_isSlow_0_8;
  assign hazardVec_isSlow_0_8 = _GEN_42;
  wire        hazardVec_samVd_1 =
    chainingRecordCopy_0_bits_vd_valid & chainingRecordCopy_2_bits_vd_valid & chainingRecordCopy_0_bits_vd_bits[4:3] == chainingRecordCopy_2_bits_vd_bits[4:3]
    & (chainingRecordCopy_0_bits_elementMask | chainingRecordCopy_2_bits_elementMask) != 16'hFFFF;
  wire        hazardVec_sourceVdEqSinkVs_1 =
    chainingRecordCopy_0_bits_vd_valid & (chainingRecordCopy_0_bits_vd_bits == chainingRecordCopy_2_bits_vs2 | chainingRecordCopy_0_bits_vd_bits == chainingRecordCopy_2_bits_vs1_bits & chainingRecordCopy_2_bits_vs1_valid);
  wire        hazardVec_sinkVdEqSourceVs_1 =
    chainingRecordCopy_2_bits_vd_valid & (chainingRecordCopy_2_bits_vd_bits == chainingRecordCopy_0_bits_vs2 | chainingRecordCopy_2_bits_vd_bits == chainingRecordCopy_0_bits_vs1_bits & chainingRecordCopy_0_bits_vs1_valid);
  wire        hazardVec_older_1 =
    chainingRecordCopy_2_bits_instIndex == chainingRecordCopy_0_bits_instIndex | chainingRecordCopy_2_bits_instIndex[1:0] < chainingRecordCopy_0_bits_instIndex[1:0] ^ chainingRecordCopy_2_bits_instIndex[2]
    ^ chainingRecordCopy_0_bits_instIndex[2];
  wire        hazardVec_hazardForeLoad_1 =
    (hazardVec_older_1 ? hazardVec_isLoad_0_1 & hazardVec_isSlow_1_1 : hazardVec_isLoad_1_1 & hazardVec_isSlow_0_1) & (hazardVec_samVd_1 | (hazardVec_older_1 ? hazardVec_sourceVdEqSinkVs_1 : hazardVec_sinkVdEqSourceVs_1));
  wire        hazardVec_rawForeStore_1 = (hazardVec_older_1 ? hazardVec_isStore_0_1 & hazardVec_isSlow_1_1 : hazardVec_isStore_1_1 & hazardVec_isSlow_0_1) & hazardVec_samVd_1;
  wire        _hazardVec_isStore_T_18 = chainingRecordCopy_3_valid & chainingRecordCopy_3_bits_ls;
  wire        _GEN_43 = _hazardVec_isStore_T_18 & ~chainingRecordCopy_3_bits_st;
  wire        hazardVec_isLoad_1_2;
  assign hazardVec_isLoad_1_2 = _GEN_43;
  wire        hazardVec_isLoad_1_5;
  assign hazardVec_isLoad_1_5 = _GEN_43;
  wire        hazardVec_isLoad_1_7;
  assign hazardVec_isLoad_1_7 = _GEN_43;
  wire        hazardVec_isLoad_0_9;
  assign hazardVec_isLoad_0_9 = _GEN_43;
  wire        _GEN_44 = _hazardVec_isStore_T_18 & chainingRecordCopy_3_bits_st;
  wire        hazardVec_isStore_1_2;
  assign hazardVec_isStore_1_2 = _GEN_44;
  wire        hazardVec_isStore_1_5;
  assign hazardVec_isStore_1_5 = _GEN_44;
  wire        hazardVec_isStore_1_7;
  assign hazardVec_isStore_1_7 = _GEN_44;
  wire        hazardVec_isStore_0_9;
  assign hazardVec_isStore_0_9 = _GEN_44;
  wire        _GEN_45 = chainingRecordCopy_3_valid & chainingRecordCopy_3_bits_slow;
  wire        hazardVec_isSlow_1_2;
  assign hazardVec_isSlow_1_2 = _GEN_45;
  wire        hazardVec_isSlow_1_5;
  assign hazardVec_isSlow_1_5 = _GEN_45;
  wire        hazardVec_isSlow_1_7;
  assign hazardVec_isSlow_1_7 = _GEN_45;
  wire        hazardVec_isSlow_0_9;
  assign hazardVec_isSlow_0_9 = _GEN_45;
  wire        hazardVec_samVd_2 =
    chainingRecordCopy_0_bits_vd_valid & chainingRecordCopy_3_bits_vd_valid & chainingRecordCopy_0_bits_vd_bits[4:3] == chainingRecordCopy_3_bits_vd_bits[4:3]
    & (chainingRecordCopy_0_bits_elementMask | chainingRecordCopy_3_bits_elementMask) != 16'hFFFF;
  wire        hazardVec_sourceVdEqSinkVs_2 =
    chainingRecordCopy_0_bits_vd_valid & (chainingRecordCopy_0_bits_vd_bits == chainingRecordCopy_3_bits_vs2 | chainingRecordCopy_0_bits_vd_bits == chainingRecordCopy_3_bits_vs1_bits & chainingRecordCopy_3_bits_vs1_valid);
  wire        hazardVec_sinkVdEqSourceVs_2 =
    chainingRecordCopy_3_bits_vd_valid & (chainingRecordCopy_3_bits_vd_bits == chainingRecordCopy_0_bits_vs2 | chainingRecordCopy_3_bits_vd_bits == chainingRecordCopy_0_bits_vs1_bits & chainingRecordCopy_0_bits_vs1_valid);
  wire        hazardVec_older_2 =
    chainingRecordCopy_3_bits_instIndex == chainingRecordCopy_0_bits_instIndex | chainingRecordCopy_3_bits_instIndex[1:0] < chainingRecordCopy_0_bits_instIndex[1:0] ^ chainingRecordCopy_3_bits_instIndex[2]
    ^ chainingRecordCopy_0_bits_instIndex[2];
  wire        hazardVec_hazardForeLoad_2 =
    (hazardVec_older_2 ? hazardVec_isLoad_0_2 & hazardVec_isSlow_1_2 : hazardVec_isLoad_1_2 & hazardVec_isSlow_0_2) & (hazardVec_samVd_2 | (hazardVec_older_2 ? hazardVec_sourceVdEqSinkVs_2 : hazardVec_sinkVdEqSourceVs_2));
  wire        hazardVec_rawForeStore_2 = (hazardVec_older_2 ? hazardVec_isStore_0_2 & hazardVec_isSlow_1_2 : hazardVec_isStore_1_2 & hazardVec_isSlow_0_2) & hazardVec_samVd_2;
  wire        _hazardVec_isStore_T_19 = chainingRecordCopy_4_valid & chainingRecordCopy_4_bits_ls;
  wire        _GEN_46 = _hazardVec_isStore_T_19 & ~chainingRecordCopy_4_bits_st;
  wire        hazardVec_isLoad_1_3;
  assign hazardVec_isLoad_1_3 = _GEN_46;
  wire        hazardVec_isLoad_1_6;
  assign hazardVec_isLoad_1_6 = _GEN_46;
  wire        hazardVec_isLoad_1_8;
  assign hazardVec_isLoad_1_8 = _GEN_46;
  wire        hazardVec_isLoad_1_9;
  assign hazardVec_isLoad_1_9 = _GEN_46;
  wire        _GEN_47 = _hazardVec_isStore_T_19 & chainingRecordCopy_4_bits_st;
  wire        hazardVec_isStore_1_3;
  assign hazardVec_isStore_1_3 = _GEN_47;
  wire        hazardVec_isStore_1_6;
  assign hazardVec_isStore_1_6 = _GEN_47;
  wire        hazardVec_isStore_1_8;
  assign hazardVec_isStore_1_8 = _GEN_47;
  wire        hazardVec_isStore_1_9;
  assign hazardVec_isStore_1_9 = _GEN_47;
  wire        _GEN_48 = chainingRecordCopy_4_valid & chainingRecordCopy_4_bits_slow;
  wire        hazardVec_isSlow_1_3;
  assign hazardVec_isSlow_1_3 = _GEN_48;
  wire        hazardVec_isSlow_1_6;
  assign hazardVec_isSlow_1_6 = _GEN_48;
  wire        hazardVec_isSlow_1_8;
  assign hazardVec_isSlow_1_8 = _GEN_48;
  wire        hazardVec_isSlow_1_9;
  assign hazardVec_isSlow_1_9 = _GEN_48;
  wire        hazardVec_samVd_3 =
    chainingRecordCopy_0_bits_vd_valid & chainingRecordCopy_4_bits_vd_valid & chainingRecordCopy_0_bits_vd_bits[4:3] == chainingRecordCopy_4_bits_vd_bits[4:3]
    & (chainingRecordCopy_0_bits_elementMask | chainingRecordCopy_4_bits_elementMask) != 16'hFFFF;
  wire        hazardVec_sourceVdEqSinkVs_3 =
    chainingRecordCopy_0_bits_vd_valid & (chainingRecordCopy_0_bits_vd_bits == chainingRecordCopy_4_bits_vs2 | chainingRecordCopy_0_bits_vd_bits == chainingRecordCopy_4_bits_vs1_bits & chainingRecordCopy_4_bits_vs1_valid);
  wire        hazardVec_sinkVdEqSourceVs_3 =
    chainingRecordCopy_4_bits_vd_valid & (chainingRecordCopy_4_bits_vd_bits == chainingRecordCopy_0_bits_vs2 | chainingRecordCopy_4_bits_vd_bits == chainingRecordCopy_0_bits_vs1_bits & chainingRecordCopy_0_bits_vs1_valid);
  wire        hazardVec_older_3 =
    chainingRecordCopy_4_bits_instIndex == chainingRecordCopy_0_bits_instIndex | chainingRecordCopy_4_bits_instIndex[1:0] < chainingRecordCopy_0_bits_instIndex[1:0] ^ chainingRecordCopy_4_bits_instIndex[2]
    ^ chainingRecordCopy_0_bits_instIndex[2];
  wire        hazardVec_hazardForeLoad_3 =
    (hazardVec_older_3 ? hazardVec_isLoad_0_3 & hazardVec_isSlow_1_3 : hazardVec_isLoad_1_3 & hazardVec_isSlow_0_3) & (hazardVec_samVd_3 | (hazardVec_older_3 ? hazardVec_sourceVdEqSinkVs_3 : hazardVec_sinkVdEqSourceVs_3));
  wire        hazardVec_rawForeStore_3 = (hazardVec_older_3 ? hazardVec_isStore_0_3 & hazardVec_isSlow_1_3 : hazardVec_isStore_1_3 & hazardVec_isSlow_0_3) & hazardVec_samVd_3;
  wire        hazardVec_samVd_4 =
    chainingRecordCopy_1_bits_vd_valid & chainingRecordCopy_2_bits_vd_valid & chainingRecordCopy_1_bits_vd_bits[4:3] == chainingRecordCopy_2_bits_vd_bits[4:3]
    & (chainingRecordCopy_1_bits_elementMask | chainingRecordCopy_2_bits_elementMask) != 16'hFFFF;
  wire        hazardVec_sourceVdEqSinkVs_4 =
    chainingRecordCopy_1_bits_vd_valid & (chainingRecordCopy_1_bits_vd_bits == chainingRecordCopy_2_bits_vs2 | chainingRecordCopy_1_bits_vd_bits == chainingRecordCopy_2_bits_vs1_bits & chainingRecordCopy_2_bits_vs1_valid);
  wire        hazardVec_sinkVdEqSourceVs_4 =
    chainingRecordCopy_2_bits_vd_valid & (chainingRecordCopy_2_bits_vd_bits == chainingRecordCopy_1_bits_vs2 | chainingRecordCopy_2_bits_vd_bits == chainingRecordCopy_1_bits_vs1_bits & chainingRecordCopy_1_bits_vs1_valid);
  wire        hazardVec_older_4 =
    chainingRecordCopy_2_bits_instIndex == chainingRecordCopy_1_bits_instIndex | chainingRecordCopy_2_bits_instIndex[1:0] < chainingRecordCopy_1_bits_instIndex[1:0] ^ chainingRecordCopy_2_bits_instIndex[2]
    ^ chainingRecordCopy_1_bits_instIndex[2];
  wire        hazardVec_hazardForeLoad_4 =
    (hazardVec_older_4 ? hazardVec_isLoad_0_4 & hazardVec_isSlow_1_4 : hazardVec_isLoad_1_4 & hazardVec_isSlow_0_4) & (hazardVec_samVd_4 | (hazardVec_older_4 ? hazardVec_sourceVdEqSinkVs_4 : hazardVec_sinkVdEqSourceVs_4));
  wire        hazardVec_rawForeStore_4 = (hazardVec_older_4 ? hazardVec_isStore_0_4 & hazardVec_isSlow_1_4 : hazardVec_isStore_1_4 & hazardVec_isSlow_0_4) & hazardVec_samVd_4;
  wire        hazardVec_samVd_5 =
    chainingRecordCopy_1_bits_vd_valid & chainingRecordCopy_3_bits_vd_valid & chainingRecordCopy_1_bits_vd_bits[4:3] == chainingRecordCopy_3_bits_vd_bits[4:3]
    & (chainingRecordCopy_1_bits_elementMask | chainingRecordCopy_3_bits_elementMask) != 16'hFFFF;
  wire        hazardVec_sourceVdEqSinkVs_5 =
    chainingRecordCopy_1_bits_vd_valid & (chainingRecordCopy_1_bits_vd_bits == chainingRecordCopy_3_bits_vs2 | chainingRecordCopy_1_bits_vd_bits == chainingRecordCopy_3_bits_vs1_bits & chainingRecordCopy_3_bits_vs1_valid);
  wire        hazardVec_sinkVdEqSourceVs_5 =
    chainingRecordCopy_3_bits_vd_valid & (chainingRecordCopy_3_bits_vd_bits == chainingRecordCopy_1_bits_vs2 | chainingRecordCopy_3_bits_vd_bits == chainingRecordCopy_1_bits_vs1_bits & chainingRecordCopy_1_bits_vs1_valid);
  wire        hazardVec_older_5 =
    chainingRecordCopy_3_bits_instIndex == chainingRecordCopy_1_bits_instIndex | chainingRecordCopy_3_bits_instIndex[1:0] < chainingRecordCopy_1_bits_instIndex[1:0] ^ chainingRecordCopy_3_bits_instIndex[2]
    ^ chainingRecordCopy_1_bits_instIndex[2];
  wire        hazardVec_hazardForeLoad_5 =
    (hazardVec_older_5 ? hazardVec_isLoad_0_5 & hazardVec_isSlow_1_5 : hazardVec_isLoad_1_5 & hazardVec_isSlow_0_5) & (hazardVec_samVd_5 | (hazardVec_older_5 ? hazardVec_sourceVdEqSinkVs_5 : hazardVec_sinkVdEqSourceVs_5));
  wire        hazardVec_rawForeStore_5 = (hazardVec_older_5 ? hazardVec_isStore_0_5 & hazardVec_isSlow_1_5 : hazardVec_isStore_1_5 & hazardVec_isSlow_0_5) & hazardVec_samVd_5;
  wire        hazardVec_samVd_6 =
    chainingRecordCopy_1_bits_vd_valid & chainingRecordCopy_4_bits_vd_valid & chainingRecordCopy_1_bits_vd_bits[4:3] == chainingRecordCopy_4_bits_vd_bits[4:3]
    & (chainingRecordCopy_1_bits_elementMask | chainingRecordCopy_4_bits_elementMask) != 16'hFFFF;
  wire        hazardVec_sourceVdEqSinkVs_6 =
    chainingRecordCopy_1_bits_vd_valid & (chainingRecordCopy_1_bits_vd_bits == chainingRecordCopy_4_bits_vs2 | chainingRecordCopy_1_bits_vd_bits == chainingRecordCopy_4_bits_vs1_bits & chainingRecordCopy_4_bits_vs1_valid);
  wire        hazardVec_sinkVdEqSourceVs_6 =
    chainingRecordCopy_4_bits_vd_valid & (chainingRecordCopy_4_bits_vd_bits == chainingRecordCopy_1_bits_vs2 | chainingRecordCopy_4_bits_vd_bits == chainingRecordCopy_1_bits_vs1_bits & chainingRecordCopy_1_bits_vs1_valid);
  wire        hazardVec_older_6 =
    chainingRecordCopy_4_bits_instIndex == chainingRecordCopy_1_bits_instIndex | chainingRecordCopy_4_bits_instIndex[1:0] < chainingRecordCopy_1_bits_instIndex[1:0] ^ chainingRecordCopy_4_bits_instIndex[2]
    ^ chainingRecordCopy_1_bits_instIndex[2];
  wire        hazardVec_hazardForeLoad_6 =
    (hazardVec_older_6 ? hazardVec_isLoad_0_6 & hazardVec_isSlow_1_6 : hazardVec_isLoad_1_6 & hazardVec_isSlow_0_6) & (hazardVec_samVd_6 | (hazardVec_older_6 ? hazardVec_sourceVdEqSinkVs_6 : hazardVec_sinkVdEqSourceVs_6));
  wire        hazardVec_rawForeStore_6 = (hazardVec_older_6 ? hazardVec_isStore_0_6 & hazardVec_isSlow_1_6 : hazardVec_isStore_1_6 & hazardVec_isSlow_0_6) & hazardVec_samVd_6;
  wire        hazardVec_samVd_7 =
    chainingRecordCopy_2_bits_vd_valid & chainingRecordCopy_3_bits_vd_valid & chainingRecordCopy_2_bits_vd_bits[4:3] == chainingRecordCopy_3_bits_vd_bits[4:3]
    & (chainingRecordCopy_2_bits_elementMask | chainingRecordCopy_3_bits_elementMask) != 16'hFFFF;
  wire        hazardVec_sourceVdEqSinkVs_7 =
    chainingRecordCopy_2_bits_vd_valid & (chainingRecordCopy_2_bits_vd_bits == chainingRecordCopy_3_bits_vs2 | chainingRecordCopy_2_bits_vd_bits == chainingRecordCopy_3_bits_vs1_bits & chainingRecordCopy_3_bits_vs1_valid);
  wire        hazardVec_sinkVdEqSourceVs_7 =
    chainingRecordCopy_3_bits_vd_valid & (chainingRecordCopy_3_bits_vd_bits == chainingRecordCopy_2_bits_vs2 | chainingRecordCopy_3_bits_vd_bits == chainingRecordCopy_2_bits_vs1_bits & chainingRecordCopy_2_bits_vs1_valid);
  wire        hazardVec_older_7 =
    chainingRecordCopy_3_bits_instIndex == chainingRecordCopy_2_bits_instIndex | chainingRecordCopy_3_bits_instIndex[1:0] < chainingRecordCopy_2_bits_instIndex[1:0] ^ chainingRecordCopy_3_bits_instIndex[2]
    ^ chainingRecordCopy_2_bits_instIndex[2];
  wire        hazardVec_hazardForeLoad_7 =
    (hazardVec_older_7 ? hazardVec_isLoad_0_7 & hazardVec_isSlow_1_7 : hazardVec_isLoad_1_7 & hazardVec_isSlow_0_7) & (hazardVec_samVd_7 | (hazardVec_older_7 ? hazardVec_sourceVdEqSinkVs_7 : hazardVec_sinkVdEqSourceVs_7));
  wire        hazardVec_rawForeStore_7 = (hazardVec_older_7 ? hazardVec_isStore_0_7 & hazardVec_isSlow_1_7 : hazardVec_isStore_1_7 & hazardVec_isSlow_0_7) & hazardVec_samVd_7;
  wire        hazardVec_samVd_8 =
    chainingRecordCopy_2_bits_vd_valid & chainingRecordCopy_4_bits_vd_valid & chainingRecordCopy_2_bits_vd_bits[4:3] == chainingRecordCopy_4_bits_vd_bits[4:3]
    & (chainingRecordCopy_2_bits_elementMask | chainingRecordCopy_4_bits_elementMask) != 16'hFFFF;
  wire        hazardVec_sourceVdEqSinkVs_8 =
    chainingRecordCopy_2_bits_vd_valid & (chainingRecordCopy_2_bits_vd_bits == chainingRecordCopy_4_bits_vs2 | chainingRecordCopy_2_bits_vd_bits == chainingRecordCopy_4_bits_vs1_bits & chainingRecordCopy_4_bits_vs1_valid);
  wire        hazardVec_sinkVdEqSourceVs_8 =
    chainingRecordCopy_4_bits_vd_valid & (chainingRecordCopy_4_bits_vd_bits == chainingRecordCopy_2_bits_vs2 | chainingRecordCopy_4_bits_vd_bits == chainingRecordCopy_2_bits_vs1_bits & chainingRecordCopy_2_bits_vs1_valid);
  wire        hazardVec_older_8 =
    chainingRecordCopy_4_bits_instIndex == chainingRecordCopy_2_bits_instIndex | chainingRecordCopy_4_bits_instIndex[1:0] < chainingRecordCopy_2_bits_instIndex[1:0] ^ chainingRecordCopy_4_bits_instIndex[2]
    ^ chainingRecordCopy_2_bits_instIndex[2];
  wire        hazardVec_hazardForeLoad_8 =
    (hazardVec_older_8 ? hazardVec_isLoad_0_8 & hazardVec_isSlow_1_8 : hazardVec_isLoad_1_8 & hazardVec_isSlow_0_8) & (hazardVec_samVd_8 | (hazardVec_older_8 ? hazardVec_sourceVdEqSinkVs_8 : hazardVec_sinkVdEqSourceVs_8));
  wire        hazardVec_rawForeStore_8 = (hazardVec_older_8 ? hazardVec_isStore_0_8 & hazardVec_isSlow_1_8 : hazardVec_isStore_1_8 & hazardVec_isSlow_0_8) & hazardVec_samVd_8;
  wire        hazardVec_samVd_9 =
    chainingRecordCopy_3_bits_vd_valid & chainingRecordCopy_4_bits_vd_valid & chainingRecordCopy_3_bits_vd_bits[4:3] == chainingRecordCopy_4_bits_vd_bits[4:3]
    & (chainingRecordCopy_3_bits_elementMask | chainingRecordCopy_4_bits_elementMask) != 16'hFFFF;
  wire        hazardVec_sourceVdEqSinkVs_9 =
    chainingRecordCopy_3_bits_vd_valid & (chainingRecordCopy_3_bits_vd_bits == chainingRecordCopy_4_bits_vs2 | chainingRecordCopy_3_bits_vd_bits == chainingRecordCopy_4_bits_vs1_bits & chainingRecordCopy_4_bits_vs1_valid);
  wire        hazardVec_sinkVdEqSourceVs_9 =
    chainingRecordCopy_4_bits_vd_valid & (chainingRecordCopy_4_bits_vd_bits == chainingRecordCopy_3_bits_vs2 | chainingRecordCopy_4_bits_vd_bits == chainingRecordCopy_3_bits_vs1_bits & chainingRecordCopy_3_bits_vs1_valid);
  wire        hazardVec_older_9 =
    chainingRecordCopy_4_bits_instIndex == chainingRecordCopy_3_bits_instIndex | chainingRecordCopy_4_bits_instIndex[1:0] < chainingRecordCopy_3_bits_instIndex[1:0] ^ chainingRecordCopy_4_bits_instIndex[2]
    ^ chainingRecordCopy_3_bits_instIndex[2];
  wire        hazardVec_hazardForeLoad_9 =
    (hazardVec_older_9 ? hazardVec_isLoad_0_9 & hazardVec_isSlow_1_9 : hazardVec_isLoad_1_9 & hazardVec_isSlow_0_9) & (hazardVec_samVd_9 | (hazardVec_older_9 ? hazardVec_sourceVdEqSinkVs_9 : hazardVec_sinkVdEqSourceVs_9));
  wire        hazardVec_rawForeStore_9 = (hazardVec_older_9 ? hazardVec_isStore_0_9 & hazardVec_isSlow_1_9 : hazardVec_isStore_1_9 & hazardVec_isSlow_0_9) & hazardVec_samVd_9;
  always @(posedge clock) begin
    if (reset) begin
      sramReady <= 1'h0;
      sramResetCount <= 5'h0;
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
      chainingRecord_0_bits_elementMask <= 16'h0;
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
      chainingRecord_1_bits_elementMask <= 16'h0;
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
      chainingRecord_2_bits_elementMask <= 16'h0;
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
      chainingRecord_3_bits_elementMask <= 16'h0;
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
      chainingRecord_4_bits_elementMask <= 16'h0;
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
      chainingRecordCopy_0_bits_elementMask <= 16'h0;
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
      chainingRecordCopy_1_bits_elementMask <= 16'h0;
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
      chainingRecordCopy_2_bits_elementMask <= 16'h0;
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
      chainingRecordCopy_3_bits_elementMask <= 16'h0;
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
      chainingRecordCopy_4_bits_elementMask <= 16'h0;
      chainingRecordCopy_4_bits_state_stFinish <= 1'h0;
      chainingRecordCopy_4_bits_state_wWriteQueueClear <= 1'h0;
      chainingRecordCopy_4_bits_state_wLaneLastReport <= 1'h0;
      chainingRecordCopy_4_bits_state_wTopLastReport <= 1'h0;
      chainingRecordCopy_4_bits_state_wLaneClear <= 1'h0;
      firstReadPipe_0_valid <= 1'h0;
      firstReadPipe_0_bits_address <= 5'h0;
      firstReadPipe_1_valid <= 1'h0;
      firstReadPipe_1_bits_address <= 5'h0;
      secondReadPipe_0_valid <= 1'h0;
      secondReadPipe_0_bits_address <= 5'h0;
      secondReadPipe_1_valid <= 1'h0;
      secondReadPipe_1_bits_address <= 5'h0;
      writePipe_valid <= 1'h0;
      writePipe_bits_vd <= 5'h0;
      writePipe_bits_offset <= 1'h0;
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
        sramResetCount <= sramResetCount + 5'h1;
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
      chainingRecord_0_bits_elementMask <= recordEnq[0] ? initRecord_bits_elementMask : {16{elementUpdateValid}} & elementUpdate1H | chainingRecord_0_bits_elementMask;
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
      chainingRecord_1_bits_elementMask <= recordEnq[1] ? initRecord_bits_elementMask : {16{elementUpdateValid_1}} & elementUpdate1H_1 | chainingRecord_1_bits_elementMask;
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
      chainingRecord_2_bits_elementMask <= recordEnq[2] ? initRecord_bits_elementMask : {16{elementUpdateValid_2}} & elementUpdate1H_2 | chainingRecord_2_bits_elementMask;
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
      chainingRecord_3_bits_elementMask <= recordEnq[3] ? initRecord_bits_elementMask : {16{elementUpdateValid_3}} & elementUpdate1H_3 | chainingRecord_3_bits_elementMask;
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
      chainingRecord_4_bits_elementMask <= recordEnq[4] ? initRecord_bits_elementMask : {16{elementUpdateValid_4}} & elementUpdate1H_4 | chainingRecord_4_bits_elementMask;
      chainingRecord_4_bits_state_stFinish <= recordEnq[4] ? initRecord_bits_state_stFinish : topLastReport_4 | chainingRecord_4_bits_state_stFinish;
      chainingRecord_4_bits_state_wWriteQueueClear <= recordEnq[4] ? initRecord_bits_state_wWriteQueueClear : chainingRecord_4_bits_state_stFinish & ~dataInLsuQueue_4 | chainingRecord_4_bits_state_wWriteQueueClear;
      chainingRecord_4_bits_state_wLaneLastReport <= recordEnq[4] ? initRecord_bits_state_wLaneLastReport : laneLastReport_4 | chainingRecord_4_bits_state_wLaneLastReport;
      chainingRecord_4_bits_state_wTopLastReport <= recordEnq[4] ? initRecord_bits_state_wTopLastReport : topLastReport_4 | chainingRecord_4_bits_state_wTopLastReport;
      chainingRecord_4_bits_state_wLaneClear <= ~(recordEnq[4]) & (waitLaneClear_4 & ~dataInLaneCheck_4 | chainingRecord_4_bits_state_wLaneClear);
      chainingRecordCopy_0_valid <= recordEnq[0] | ~stateClear_5 & chainingRecordCopy_0_valid;
      chainingRecordCopy_0_bits_elementMask <= recordEnq[0] ? initRecord_bits_elementMask : {16{elementUpdateValid_5}} & elementUpdate1H_5 | chainingRecordCopy_0_bits_elementMask;
      chainingRecordCopy_0_bits_state_stFinish <= recordEnq[0] ? initRecord_bits_state_stFinish : topLastReport_5 | chainingRecordCopy_0_bits_state_stFinish;
      chainingRecordCopy_0_bits_state_wWriteQueueClear <= recordEnq[0] ? initRecord_bits_state_wWriteQueueClear : chainingRecordCopy_0_bits_state_stFinish & ~dataInLsuQueue_5 | chainingRecordCopy_0_bits_state_wWriteQueueClear;
      chainingRecordCopy_0_bits_state_wLaneLastReport <= recordEnq[0] ? initRecord_bits_state_wLaneLastReport : laneLastReport_5 | chainingRecordCopy_0_bits_state_wLaneLastReport;
      chainingRecordCopy_0_bits_state_wTopLastReport <= recordEnq[0] ? initRecord_bits_state_wTopLastReport : topLastReport_5 | chainingRecordCopy_0_bits_state_wTopLastReport;
      chainingRecordCopy_0_bits_state_wLaneClear <= ~(recordEnq[0]) & (waitLaneClear_5 & ~dataInLaneCheck_5 | chainingRecordCopy_0_bits_state_wLaneClear);
      chainingRecordCopy_1_valid <= recordEnq[1] | ~stateClear_6 & chainingRecordCopy_1_valid;
      chainingRecordCopy_1_bits_elementMask <= recordEnq[1] ? initRecord_bits_elementMask : {16{elementUpdateValid_6}} & elementUpdate1H_6 | chainingRecordCopy_1_bits_elementMask;
      chainingRecordCopy_1_bits_state_stFinish <= recordEnq[1] ? initRecord_bits_state_stFinish : topLastReport_6 | chainingRecordCopy_1_bits_state_stFinish;
      chainingRecordCopy_1_bits_state_wWriteQueueClear <= recordEnq[1] ? initRecord_bits_state_wWriteQueueClear : chainingRecordCopy_1_bits_state_stFinish & ~dataInLsuQueue_6 | chainingRecordCopy_1_bits_state_wWriteQueueClear;
      chainingRecordCopy_1_bits_state_wLaneLastReport <= recordEnq[1] ? initRecord_bits_state_wLaneLastReport : laneLastReport_6 | chainingRecordCopy_1_bits_state_wLaneLastReport;
      chainingRecordCopy_1_bits_state_wTopLastReport <= recordEnq[1] ? initRecord_bits_state_wTopLastReport : topLastReport_6 | chainingRecordCopy_1_bits_state_wTopLastReport;
      chainingRecordCopy_1_bits_state_wLaneClear <= ~(recordEnq[1]) & (waitLaneClear_6 & ~dataInLaneCheck_6 | chainingRecordCopy_1_bits_state_wLaneClear);
      chainingRecordCopy_2_valid <= recordEnq[2] | ~stateClear_7 & chainingRecordCopy_2_valid;
      chainingRecordCopy_2_bits_elementMask <= recordEnq[2] ? initRecord_bits_elementMask : {16{elementUpdateValid_7}} & elementUpdate1H_7 | chainingRecordCopy_2_bits_elementMask;
      chainingRecordCopy_2_bits_state_stFinish <= recordEnq[2] ? initRecord_bits_state_stFinish : topLastReport_7 | chainingRecordCopy_2_bits_state_stFinish;
      chainingRecordCopy_2_bits_state_wWriteQueueClear <= recordEnq[2] ? initRecord_bits_state_wWriteQueueClear : chainingRecordCopy_2_bits_state_stFinish & ~dataInLsuQueue_7 | chainingRecordCopy_2_bits_state_wWriteQueueClear;
      chainingRecordCopy_2_bits_state_wLaneLastReport <= recordEnq[2] ? initRecord_bits_state_wLaneLastReport : laneLastReport_7 | chainingRecordCopy_2_bits_state_wLaneLastReport;
      chainingRecordCopy_2_bits_state_wTopLastReport <= recordEnq[2] ? initRecord_bits_state_wTopLastReport : topLastReport_7 | chainingRecordCopy_2_bits_state_wTopLastReport;
      chainingRecordCopy_2_bits_state_wLaneClear <= ~(recordEnq[2]) & (waitLaneClear_7 & ~dataInLaneCheck_7 | chainingRecordCopy_2_bits_state_wLaneClear);
      chainingRecordCopy_3_valid <= recordEnq[3] | ~stateClear_8 & chainingRecordCopy_3_valid;
      chainingRecordCopy_3_bits_elementMask <= recordEnq[3] ? initRecord_bits_elementMask : {16{elementUpdateValid_8}} & elementUpdate1H_8 | chainingRecordCopy_3_bits_elementMask;
      chainingRecordCopy_3_bits_state_stFinish <= recordEnq[3] ? initRecord_bits_state_stFinish : topLastReport_8 | chainingRecordCopy_3_bits_state_stFinish;
      chainingRecordCopy_3_bits_state_wWriteQueueClear <= recordEnq[3] ? initRecord_bits_state_wWriteQueueClear : chainingRecordCopy_3_bits_state_stFinish & ~dataInLsuQueue_8 | chainingRecordCopy_3_bits_state_wWriteQueueClear;
      chainingRecordCopy_3_bits_state_wLaneLastReport <= recordEnq[3] ? initRecord_bits_state_wLaneLastReport : laneLastReport_8 | chainingRecordCopy_3_bits_state_wLaneLastReport;
      chainingRecordCopy_3_bits_state_wTopLastReport <= recordEnq[3] ? initRecord_bits_state_wTopLastReport : topLastReport_8 | chainingRecordCopy_3_bits_state_wTopLastReport;
      chainingRecordCopy_3_bits_state_wLaneClear <= ~(recordEnq[3]) & (waitLaneClear_8 & ~dataInLaneCheck_8 | chainingRecordCopy_3_bits_state_wLaneClear);
      chainingRecordCopy_4_valid <= recordEnq[4] | ~stateClear_9 & chainingRecordCopy_4_valid;
      chainingRecordCopy_4_bits_elementMask <= recordEnq[4] ? initRecord_bits_elementMask : {16{elementUpdateValid_9}} & elementUpdate1H_9 | chainingRecordCopy_4_bits_elementMask;
      chainingRecordCopy_4_bits_state_stFinish <= recordEnq[4] ? initRecord_bits_state_stFinish : topLastReport_9 | chainingRecordCopy_4_bits_state_stFinish;
      chainingRecordCopy_4_bits_state_wWriteQueueClear <= recordEnq[4] ? initRecord_bits_state_wWriteQueueClear : chainingRecordCopy_4_bits_state_stFinish & ~dataInLsuQueue_9 | chainingRecordCopy_4_bits_state_wWriteQueueClear;
      chainingRecordCopy_4_bits_state_wLaneLastReport <= recordEnq[4] ? initRecord_bits_state_wLaneLastReport : laneLastReport_9 | chainingRecordCopy_4_bits_state_wLaneLastReport;
      chainingRecordCopy_4_bits_state_wTopLastReport <= recordEnq[4] ? initRecord_bits_state_wTopLastReport : topLastReport_9 | chainingRecordCopy_4_bits_state_wTopLastReport;
      chainingRecordCopy_4_bits_state_wLaneClear <= ~(recordEnq[4]) & (waitLaneClear_9 & ~dataInLaneCheck_9 | chainingRecordCopy_4_bits_state_wLaneClear);
      firstReadPipe_0_valid <=
        bankReadF_0[0] | bankReadF_1[0] | bankReadF_2[0] | bankReadF_3[0] | bankReadF_4[0] | bankReadF_5[0] | bankReadF_6[0] | bankReadF_7[0] | bankReadF_8[0] | bankReadF_9[0] | bankReadF_10[0] | bankReadF_11[0] | bankReadF_12[0]
        | bankReadF_13[0];
      firstReadPipe_0_bits_address <=
        (bankReadF_0[0] ? readRequests_0_bits_vs_0 : 5'h0) | (bankReadF_1[0] ? readRequests_1_bits_vs_0 : 5'h0) | (bankReadF_2[0] ? readRequests_2_bits_vs_0 : 5'h0) | (bankReadF_3[0] ? readRequests_3_bits_vs_0 : 5'h0)
        | (bankReadF_4[0] ? readRequests_4_bits_vs_0 : 5'h0) | (bankReadF_5[0] ? readRequests_5_bits_vs_0 : 5'h0) | (bankReadF_6[0] ? readRequests_6_bits_vs_0 : 5'h0) | (bankReadF_7[0] ? readRequests_7_bits_vs_0 : 5'h0)
        | (bankReadF_8[0] ? readRequests_8_bits_vs_0 : 5'h0) | (bankReadF_9[0] ? readRequests_9_bits_vs_0 : 5'h0) | (bankReadF_10[0] ? readRequests_10_bits_vs_0 : 5'h0) | (bankReadF_11[0] ? readRequests_11_bits_vs_0 : 5'h0)
        | (bankReadF_12[0] ? readRequests_12_bits_vs_0 : 5'h0) | (bankReadF_13[0] ? readRequests_13_bits_vs_0 : 5'h0);
      firstReadPipe_1_valid <=
        bankReadF_0[1] | bankReadF_1[1] | bankReadF_2[1] | bankReadF_3[1] | bankReadF_4[1] | bankReadF_5[1] | bankReadF_6[1] | bankReadF_7[1] | bankReadF_8[1] | bankReadF_9[1] | bankReadF_10[1] | bankReadF_11[1] | bankReadF_12[1]
        | bankReadF_13[1];
      firstReadPipe_1_bits_address <=
        (bankReadF_0[1] ? readRequests_0_bits_vs_0 : 5'h0) | (bankReadF_1[1] ? readRequests_1_bits_vs_0 : 5'h0) | (bankReadF_2[1] ? readRequests_2_bits_vs_0 : 5'h0) | (bankReadF_3[1] ? readRequests_3_bits_vs_0 : 5'h0)
        | (bankReadF_4[1] ? readRequests_4_bits_vs_0 : 5'h0) | (bankReadF_5[1] ? readRequests_5_bits_vs_0 : 5'h0) | (bankReadF_6[1] ? readRequests_6_bits_vs_0 : 5'h0) | (bankReadF_7[1] ? readRequests_7_bits_vs_0 : 5'h0)
        | (bankReadF_8[1] ? readRequests_8_bits_vs_0 : 5'h0) | (bankReadF_9[1] ? readRequests_9_bits_vs_0 : 5'h0) | (bankReadF_10[1] ? readRequests_10_bits_vs_0 : 5'h0) | (bankReadF_11[1] ? readRequests_11_bits_vs_0 : 5'h0)
        | (bankReadF_12[1] ? readRequests_12_bits_vs_0 : 5'h0) | (bankReadF_13[1] ? readRequests_13_bits_vs_0 : 5'h0);
      secondReadPipe_0_valid <=
        bankReadS_1[0] | bankReadS_2[0] | bankReadS_3[0] | bankReadS_4[0] | bankReadS_5[0] | bankReadS_6[0] | bankReadS_7[0] | bankReadS_8[0] | bankReadS_9[0] | bankReadS_10[0] | bankReadS_11[0] | bankReadS_12[0] | bankReadS_13[0];
      secondReadPipe_0_bits_address <=
        (bankReadS_1[0] ? readRequests_1_bits_vs_0 : 5'h0) | (bankReadS_2[0] ? readRequests_2_bits_vs_0 : 5'h0) | (bankReadS_3[0] ? readRequests_3_bits_vs_0 : 5'h0) | (bankReadS_4[0] ? readRequests_4_bits_vs_0 : 5'h0)
        | (bankReadS_5[0] ? readRequests_5_bits_vs_0 : 5'h0) | (bankReadS_6[0] ? readRequests_6_bits_vs_0 : 5'h0) | (bankReadS_7[0] ? readRequests_7_bits_vs_0 : 5'h0) | (bankReadS_8[0] ? readRequests_8_bits_vs_0 : 5'h0)
        | (bankReadS_9[0] ? readRequests_9_bits_vs_0 : 5'h0) | (bankReadS_10[0] ? readRequests_10_bits_vs_0 : 5'h0) | (bankReadS_11[0] ? readRequests_11_bits_vs_0 : 5'h0) | (bankReadS_12[0] ? readRequests_12_bits_vs_0 : 5'h0)
        | (bankReadS_13[0] ? readRequests_13_bits_vs_0 : 5'h0);
      secondReadPipe_1_valid <=
        bankReadS_1[1] | bankReadS_2[1] | bankReadS_3[1] | bankReadS_4[1] | bankReadS_5[1] | bankReadS_6[1] | bankReadS_7[1] | bankReadS_8[1] | bankReadS_9[1] | bankReadS_10[1] | bankReadS_11[1] | bankReadS_12[1] | bankReadS_13[1];
      secondReadPipe_1_bits_address <=
        (bankReadS_1[1] ? readRequests_1_bits_vs_0 : 5'h0) | (bankReadS_2[1] ? readRequests_2_bits_vs_0 : 5'h0) | (bankReadS_3[1] ? readRequests_3_bits_vs_0 : 5'h0) | (bankReadS_4[1] ? readRequests_4_bits_vs_0 : 5'h0)
        | (bankReadS_5[1] ? readRequests_5_bits_vs_0 : 5'h0) | (bankReadS_6[1] ? readRequests_6_bits_vs_0 : 5'h0) | (bankReadS_7[1] ? readRequests_7_bits_vs_0 : 5'h0) | (bankReadS_8[1] ? readRequests_8_bits_vs_0 : 5'h0)
        | (bankReadS_9[1] ? readRequests_9_bits_vs_0 : 5'h0) | (bankReadS_10[1] ? readRequests_10_bits_vs_0 : 5'h0) | (bankReadS_11[1] ? readRequests_11_bits_vs_0 : 5'h0) | (bankReadS_12[1] ? readRequests_12_bits_vs_0 : 5'h0)
        | (bankReadS_13[1] ? readRequests_13_bits_vs_0 : 5'h0);
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
    writeBankPipe <= writeBank;
    pipeBank_pipe_b <= bank;
    if (pipeBank_pipe_v)
      pipeBank_pipe_pipe_b <= pipeBank_pipe_b;
    pipeFirstUsed_pipe_pipe_b <= 1'h0;
    pipeFire_pipe_b <= _pipeFire_T;
    if (pipeFire_pipe_v)
      pipeFire_pipe_pipe_b <= pipeFire_pipe_b;
    pipeBank_pipe_b_1 <= bank_1;
    if (pipeBank_pipe_v_1)
      pipeBank_pipe_pipe_b_1 <= pipeBank_pipe_b_1;
    pipeFirstUsed_pipe_b_1 <= firstUsed_1;
    if (pipeFirstUsed_pipe_v_1)
      pipeFirstUsed_pipe_pipe_b_1 <= pipeFirstUsed_pipe_b_1;
    pipeFire_pipe_b_1 <= _pipeFire_T_1;
    if (pipeFire_pipe_v_1)
      pipeFire_pipe_pipe_b_1 <= pipeFire_pipe_b_1;
    pipeBank_pipe_b_2 <= bank_2;
    if (pipeBank_pipe_v_2)
      pipeBank_pipe_pipe_b_2 <= pipeBank_pipe_b_2;
    pipeFirstUsed_pipe_b_2 <= firstUsed_2;
    if (pipeFirstUsed_pipe_v_2)
      pipeFirstUsed_pipe_pipe_b_2 <= pipeFirstUsed_pipe_b_2;
    pipeFire_pipe_b_2 <= _pipeFire_T_2;
    if (pipeFire_pipe_v_2)
      pipeFire_pipe_pipe_b_2 <= pipeFire_pipe_b_2;
    pipeBank_pipe_b_3 <= bank_3;
    if (pipeBank_pipe_v_3)
      pipeBank_pipe_pipe_b_3 <= pipeBank_pipe_b_3;
    pipeFirstUsed_pipe_b_3 <= firstUsed_3;
    if (pipeFirstUsed_pipe_v_3)
      pipeFirstUsed_pipe_pipe_b_3 <= pipeFirstUsed_pipe_b_3;
    pipeFire_pipe_b_3 <= _pipeFire_T_3;
    if (pipeFire_pipe_v_3)
      pipeFire_pipe_pipe_b_3 <= pipeFire_pipe_b_3;
    pipeBank_pipe_b_4 <= bank_4;
    if (pipeBank_pipe_v_4)
      pipeBank_pipe_pipe_b_4 <= pipeBank_pipe_b_4;
    pipeFirstUsed_pipe_b_4 <= firstUsed_4;
    if (pipeFirstUsed_pipe_v_4)
      pipeFirstUsed_pipe_pipe_b_4 <= pipeFirstUsed_pipe_b_4;
    pipeFire_pipe_b_4 <= _pipeFire_T_4;
    if (pipeFire_pipe_v_4)
      pipeFire_pipe_pipe_b_4 <= pipeFire_pipe_b_4;
    pipeBank_pipe_b_5 <= bank_5;
    if (pipeBank_pipe_v_5)
      pipeBank_pipe_pipe_b_5 <= pipeBank_pipe_b_5;
    pipeFirstUsed_pipe_b_5 <= firstUsed_5;
    if (pipeFirstUsed_pipe_v_5)
      pipeFirstUsed_pipe_pipe_b_5 <= pipeFirstUsed_pipe_b_5;
    pipeFire_pipe_b_5 <= _pipeFire_T_5;
    if (pipeFire_pipe_v_5)
      pipeFire_pipe_pipe_b_5 <= pipeFire_pipe_b_5;
    pipeBank_pipe_b_6 <= bank_6;
    if (pipeBank_pipe_v_6)
      pipeBank_pipe_pipe_b_6 <= pipeBank_pipe_b_6;
    pipeFirstUsed_pipe_b_6 <= firstUsed_6;
    if (pipeFirstUsed_pipe_v_6)
      pipeFirstUsed_pipe_pipe_b_6 <= pipeFirstUsed_pipe_b_6;
    pipeFire_pipe_b_6 <= _pipeFire_T_6;
    if (pipeFire_pipe_v_6)
      pipeFire_pipe_pipe_b_6 <= pipeFire_pipe_b_6;
    pipeBank_pipe_b_7 <= bank_7;
    if (pipeBank_pipe_v_7)
      pipeBank_pipe_pipe_b_7 <= pipeBank_pipe_b_7;
    pipeFirstUsed_pipe_b_7 <= firstUsed_7;
    if (pipeFirstUsed_pipe_v_7)
      pipeFirstUsed_pipe_pipe_b_7 <= pipeFirstUsed_pipe_b_7;
    pipeFire_pipe_b_7 <= _pipeFire_T_7;
    if (pipeFire_pipe_v_7)
      pipeFire_pipe_pipe_b_7 <= pipeFire_pipe_b_7;
    pipeBank_pipe_b_8 <= bank_8;
    if (pipeBank_pipe_v_8)
      pipeBank_pipe_pipe_b_8 <= pipeBank_pipe_b_8;
    pipeFirstUsed_pipe_b_8 <= firstUsed_8;
    if (pipeFirstUsed_pipe_v_8)
      pipeFirstUsed_pipe_pipe_b_8 <= pipeFirstUsed_pipe_b_8;
    pipeFire_pipe_b_8 <= _pipeFire_T_8;
    if (pipeFire_pipe_v_8)
      pipeFire_pipe_pipe_b_8 <= pipeFire_pipe_b_8;
    pipeBank_pipe_b_9 <= bank_9;
    if (pipeBank_pipe_v_9)
      pipeBank_pipe_pipe_b_9 <= pipeBank_pipe_b_9;
    pipeFirstUsed_pipe_b_9 <= firstUsed_9;
    if (pipeFirstUsed_pipe_v_9)
      pipeFirstUsed_pipe_pipe_b_9 <= pipeFirstUsed_pipe_b_9;
    pipeFire_pipe_b_9 <= _pipeFire_T_9;
    if (pipeFire_pipe_v_9)
      pipeFire_pipe_pipe_b_9 <= pipeFire_pipe_b_9;
    pipeBank_pipe_b_10 <= bank_10;
    if (pipeBank_pipe_v_10)
      pipeBank_pipe_pipe_b_10 <= pipeBank_pipe_b_10;
    pipeFirstUsed_pipe_b_10 <= firstUsed_10;
    if (pipeFirstUsed_pipe_v_10)
      pipeFirstUsed_pipe_pipe_b_10 <= pipeFirstUsed_pipe_b_10;
    pipeFire_pipe_b_10 <= _pipeFire_T_10;
    if (pipeFire_pipe_v_10)
      pipeFire_pipe_pipe_b_10 <= pipeFire_pipe_b_10;
    pipeBank_pipe_b_11 <= bank_11;
    if (pipeBank_pipe_v_11)
      pipeBank_pipe_pipe_b_11 <= pipeBank_pipe_b_11;
    pipeFirstUsed_pipe_b_11 <= firstUsed_11;
    if (pipeFirstUsed_pipe_v_11)
      pipeFirstUsed_pipe_pipe_b_11 <= pipeFirstUsed_pipe_b_11;
    pipeFire_pipe_b_11 <= _pipeFire_T_11;
    if (pipeFire_pipe_v_11)
      pipeFire_pipe_pipe_b_11 <= pipeFire_pipe_b_11;
    pipeBank_pipe_b_12 <= bank_12;
    if (pipeBank_pipe_v_12)
      pipeBank_pipe_pipe_b_12 <= pipeBank_pipe_b_12;
    pipeFirstUsed_pipe_b_12 <= firstUsed_12;
    if (pipeFirstUsed_pipe_v_12)
      pipeFirstUsed_pipe_pipe_b_12 <= pipeFirstUsed_pipe_b_12;
    pipeFire_pipe_b_12 <= _pipeFire_T_12;
    if (pipeFire_pipe_v_12)
      pipeFire_pipe_pipe_b_12 <= pipeFire_pipe_b_12;
    pipeBank_pipe_b_13 <= bank_13;
    if (pipeBank_pipe_v_13)
      pipeBank_pipe_pipe_b_13 <= pipeBank_pipe_b_13;
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
      automatic logic [31:0] _RANDOM[0:24];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [4:0] i = 5'h0; i < 5'h19; i += 5'h1) begin
          _RANDOM[i] = `RANDOM;
        end
        sramReady = _RANDOM[5'h0][0];
        sramResetCount = _RANDOM[5'h0][5:1];
        chainingRecord_0_valid = _RANDOM[5'h0][6];
        chainingRecord_0_bits_vd_valid = _RANDOM[5'h0][7];
        chainingRecord_0_bits_vd_bits = _RANDOM[5'h0][12:8];
        chainingRecord_0_bits_vs1_valid = _RANDOM[5'h0][13];
        chainingRecord_0_bits_vs1_bits = _RANDOM[5'h0][18:14];
        chainingRecord_0_bits_vs2 = _RANDOM[5'h0][23:19];
        chainingRecord_0_bits_instIndex = _RANDOM[5'h0][26:24];
        chainingRecord_0_bits_ls = _RANDOM[5'h0][27];
        chainingRecord_0_bits_st = _RANDOM[5'h0][28];
        chainingRecord_0_bits_gather = _RANDOM[5'h0][29];
        chainingRecord_0_bits_gather16 = _RANDOM[5'h0][30];
        chainingRecord_0_bits_crossWrite = _RANDOM[5'h0][31];
        chainingRecord_0_bits_crossRead = _RANDOM[5'h1][0];
        chainingRecord_0_bits_indexType = _RANDOM[5'h1][1];
        chainingRecord_0_bits_ma = _RANDOM[5'h1][2];
        chainingRecord_0_bits_onlyRead = _RANDOM[5'h1][3];
        chainingRecord_0_bits_slow = _RANDOM[5'h1][4];
        chainingRecord_0_bits_elementMask = _RANDOM[5'h1][20:5];
        chainingRecord_0_bits_state_stFinish = _RANDOM[5'h1][21];
        chainingRecord_0_bits_state_wWriteQueueClear = _RANDOM[5'h1][22];
        chainingRecord_0_bits_state_wLaneLastReport = _RANDOM[5'h1][23];
        chainingRecord_0_bits_state_wTopLastReport = _RANDOM[5'h1][24];
        chainingRecord_0_bits_state_wLaneClear = _RANDOM[5'h1][25];
        chainingRecord_1_valid = _RANDOM[5'h1][26];
        chainingRecord_1_bits_vd_valid = _RANDOM[5'h1][27];
        chainingRecord_1_bits_vd_bits = {_RANDOM[5'h1][31:28], _RANDOM[5'h2][0]};
        chainingRecord_1_bits_vs1_valid = _RANDOM[5'h2][1];
        chainingRecord_1_bits_vs1_bits = _RANDOM[5'h2][6:2];
        chainingRecord_1_bits_vs2 = _RANDOM[5'h2][11:7];
        chainingRecord_1_bits_instIndex = _RANDOM[5'h2][14:12];
        chainingRecord_1_bits_ls = _RANDOM[5'h2][15];
        chainingRecord_1_bits_st = _RANDOM[5'h2][16];
        chainingRecord_1_bits_gather = _RANDOM[5'h2][17];
        chainingRecord_1_bits_gather16 = _RANDOM[5'h2][18];
        chainingRecord_1_bits_crossWrite = _RANDOM[5'h2][19];
        chainingRecord_1_bits_crossRead = _RANDOM[5'h2][20];
        chainingRecord_1_bits_indexType = _RANDOM[5'h2][21];
        chainingRecord_1_bits_ma = _RANDOM[5'h2][22];
        chainingRecord_1_bits_onlyRead = _RANDOM[5'h2][23];
        chainingRecord_1_bits_slow = _RANDOM[5'h2][24];
        chainingRecord_1_bits_elementMask = {_RANDOM[5'h2][31:25], _RANDOM[5'h3][8:0]};
        chainingRecord_1_bits_state_stFinish = _RANDOM[5'h3][9];
        chainingRecord_1_bits_state_wWriteQueueClear = _RANDOM[5'h3][10];
        chainingRecord_1_bits_state_wLaneLastReport = _RANDOM[5'h3][11];
        chainingRecord_1_bits_state_wTopLastReport = _RANDOM[5'h3][12];
        chainingRecord_1_bits_state_wLaneClear = _RANDOM[5'h3][13];
        chainingRecord_2_valid = _RANDOM[5'h3][14];
        chainingRecord_2_bits_vd_valid = _RANDOM[5'h3][15];
        chainingRecord_2_bits_vd_bits = _RANDOM[5'h3][20:16];
        chainingRecord_2_bits_vs1_valid = _RANDOM[5'h3][21];
        chainingRecord_2_bits_vs1_bits = _RANDOM[5'h3][26:22];
        chainingRecord_2_bits_vs2 = _RANDOM[5'h3][31:27];
        chainingRecord_2_bits_instIndex = _RANDOM[5'h4][2:0];
        chainingRecord_2_bits_ls = _RANDOM[5'h4][3];
        chainingRecord_2_bits_st = _RANDOM[5'h4][4];
        chainingRecord_2_bits_gather = _RANDOM[5'h4][5];
        chainingRecord_2_bits_gather16 = _RANDOM[5'h4][6];
        chainingRecord_2_bits_crossWrite = _RANDOM[5'h4][7];
        chainingRecord_2_bits_crossRead = _RANDOM[5'h4][8];
        chainingRecord_2_bits_indexType = _RANDOM[5'h4][9];
        chainingRecord_2_bits_ma = _RANDOM[5'h4][10];
        chainingRecord_2_bits_onlyRead = _RANDOM[5'h4][11];
        chainingRecord_2_bits_slow = _RANDOM[5'h4][12];
        chainingRecord_2_bits_elementMask = _RANDOM[5'h4][28:13];
        chainingRecord_2_bits_state_stFinish = _RANDOM[5'h4][29];
        chainingRecord_2_bits_state_wWriteQueueClear = _RANDOM[5'h4][30];
        chainingRecord_2_bits_state_wLaneLastReport = _RANDOM[5'h4][31];
        chainingRecord_2_bits_state_wTopLastReport = _RANDOM[5'h5][0];
        chainingRecord_2_bits_state_wLaneClear = _RANDOM[5'h5][1];
        chainingRecord_3_valid = _RANDOM[5'h5][2];
        chainingRecord_3_bits_vd_valid = _RANDOM[5'h5][3];
        chainingRecord_3_bits_vd_bits = _RANDOM[5'h5][8:4];
        chainingRecord_3_bits_vs1_valid = _RANDOM[5'h5][9];
        chainingRecord_3_bits_vs1_bits = _RANDOM[5'h5][14:10];
        chainingRecord_3_bits_vs2 = _RANDOM[5'h5][19:15];
        chainingRecord_3_bits_instIndex = _RANDOM[5'h5][22:20];
        chainingRecord_3_bits_ls = _RANDOM[5'h5][23];
        chainingRecord_3_bits_st = _RANDOM[5'h5][24];
        chainingRecord_3_bits_gather = _RANDOM[5'h5][25];
        chainingRecord_3_bits_gather16 = _RANDOM[5'h5][26];
        chainingRecord_3_bits_crossWrite = _RANDOM[5'h5][27];
        chainingRecord_3_bits_crossRead = _RANDOM[5'h5][28];
        chainingRecord_3_bits_indexType = _RANDOM[5'h5][29];
        chainingRecord_3_bits_ma = _RANDOM[5'h5][30];
        chainingRecord_3_bits_onlyRead = _RANDOM[5'h5][31];
        chainingRecord_3_bits_slow = _RANDOM[5'h6][0];
        chainingRecord_3_bits_elementMask = _RANDOM[5'h6][16:1];
        chainingRecord_3_bits_state_stFinish = _RANDOM[5'h6][17];
        chainingRecord_3_bits_state_wWriteQueueClear = _RANDOM[5'h6][18];
        chainingRecord_3_bits_state_wLaneLastReport = _RANDOM[5'h6][19];
        chainingRecord_3_bits_state_wTopLastReport = _RANDOM[5'h6][20];
        chainingRecord_3_bits_state_wLaneClear = _RANDOM[5'h6][21];
        chainingRecord_4_valid = _RANDOM[5'h6][22];
        chainingRecord_4_bits_vd_valid = _RANDOM[5'h6][23];
        chainingRecord_4_bits_vd_bits = _RANDOM[5'h6][28:24];
        chainingRecord_4_bits_vs1_valid = _RANDOM[5'h6][29];
        chainingRecord_4_bits_vs1_bits = {_RANDOM[5'h6][31:30], _RANDOM[5'h7][2:0]};
        chainingRecord_4_bits_vs2 = _RANDOM[5'h7][7:3];
        chainingRecord_4_bits_instIndex = _RANDOM[5'h7][10:8];
        chainingRecord_4_bits_ls = _RANDOM[5'h7][11];
        chainingRecord_4_bits_st = _RANDOM[5'h7][12];
        chainingRecord_4_bits_gather = _RANDOM[5'h7][13];
        chainingRecord_4_bits_gather16 = _RANDOM[5'h7][14];
        chainingRecord_4_bits_crossWrite = _RANDOM[5'h7][15];
        chainingRecord_4_bits_crossRead = _RANDOM[5'h7][16];
        chainingRecord_4_bits_indexType = _RANDOM[5'h7][17];
        chainingRecord_4_bits_ma = _RANDOM[5'h7][18];
        chainingRecord_4_bits_onlyRead = _RANDOM[5'h7][19];
        chainingRecord_4_bits_slow = _RANDOM[5'h7][20];
        chainingRecord_4_bits_elementMask = {_RANDOM[5'h7][31:21], _RANDOM[5'h8][4:0]};
        chainingRecord_4_bits_state_stFinish = _RANDOM[5'h8][5];
        chainingRecord_4_bits_state_wWriteQueueClear = _RANDOM[5'h8][6];
        chainingRecord_4_bits_state_wLaneLastReport = _RANDOM[5'h8][7];
        chainingRecord_4_bits_state_wTopLastReport = _RANDOM[5'h8][8];
        chainingRecord_4_bits_state_wLaneClear = _RANDOM[5'h8][9];
        chainingRecordCopy_0_valid = _RANDOM[5'h8][10];
        chainingRecordCopy_0_bits_vd_valid = _RANDOM[5'h8][11];
        chainingRecordCopy_0_bits_vd_bits = _RANDOM[5'h8][16:12];
        chainingRecordCopy_0_bits_vs1_valid = _RANDOM[5'h8][17];
        chainingRecordCopy_0_bits_vs1_bits = _RANDOM[5'h8][22:18];
        chainingRecordCopy_0_bits_vs2 = _RANDOM[5'h8][27:23];
        chainingRecordCopy_0_bits_instIndex = _RANDOM[5'h8][30:28];
        chainingRecordCopy_0_bits_ls = _RANDOM[5'h8][31];
        chainingRecordCopy_0_bits_st = _RANDOM[5'h9][0];
        chainingRecordCopy_0_bits_gather = _RANDOM[5'h9][1];
        chainingRecordCopy_0_bits_gather16 = _RANDOM[5'h9][2];
        chainingRecordCopy_0_bits_crossWrite = _RANDOM[5'h9][3];
        chainingRecordCopy_0_bits_crossRead = _RANDOM[5'h9][4];
        chainingRecordCopy_0_bits_indexType = _RANDOM[5'h9][5];
        chainingRecordCopy_0_bits_ma = _RANDOM[5'h9][6];
        chainingRecordCopy_0_bits_onlyRead = _RANDOM[5'h9][7];
        chainingRecordCopy_0_bits_slow = _RANDOM[5'h9][8];
        chainingRecordCopy_0_bits_elementMask = _RANDOM[5'h9][24:9];
        chainingRecordCopy_0_bits_state_stFinish = _RANDOM[5'h9][25];
        chainingRecordCopy_0_bits_state_wWriteQueueClear = _RANDOM[5'h9][26];
        chainingRecordCopy_0_bits_state_wLaneLastReport = _RANDOM[5'h9][27];
        chainingRecordCopy_0_bits_state_wTopLastReport = _RANDOM[5'h9][28];
        chainingRecordCopy_0_bits_state_wLaneClear = _RANDOM[5'h9][29];
        chainingRecordCopy_1_valid = _RANDOM[5'h9][30];
        chainingRecordCopy_1_bits_vd_valid = _RANDOM[5'h9][31];
        chainingRecordCopy_1_bits_vd_bits = _RANDOM[5'hA][4:0];
        chainingRecordCopy_1_bits_vs1_valid = _RANDOM[5'hA][5];
        chainingRecordCopy_1_bits_vs1_bits = _RANDOM[5'hA][10:6];
        chainingRecordCopy_1_bits_vs2 = _RANDOM[5'hA][15:11];
        chainingRecordCopy_1_bits_instIndex = _RANDOM[5'hA][18:16];
        chainingRecordCopy_1_bits_ls = _RANDOM[5'hA][19];
        chainingRecordCopy_1_bits_st = _RANDOM[5'hA][20];
        chainingRecordCopy_1_bits_gather = _RANDOM[5'hA][21];
        chainingRecordCopy_1_bits_gather16 = _RANDOM[5'hA][22];
        chainingRecordCopy_1_bits_crossWrite = _RANDOM[5'hA][23];
        chainingRecordCopy_1_bits_crossRead = _RANDOM[5'hA][24];
        chainingRecordCopy_1_bits_indexType = _RANDOM[5'hA][25];
        chainingRecordCopy_1_bits_ma = _RANDOM[5'hA][26];
        chainingRecordCopy_1_bits_onlyRead = _RANDOM[5'hA][27];
        chainingRecordCopy_1_bits_slow = _RANDOM[5'hA][28];
        chainingRecordCopy_1_bits_elementMask = {_RANDOM[5'hA][31:29], _RANDOM[5'hB][12:0]};
        chainingRecordCopy_1_bits_state_stFinish = _RANDOM[5'hB][13];
        chainingRecordCopy_1_bits_state_wWriteQueueClear = _RANDOM[5'hB][14];
        chainingRecordCopy_1_bits_state_wLaneLastReport = _RANDOM[5'hB][15];
        chainingRecordCopy_1_bits_state_wTopLastReport = _RANDOM[5'hB][16];
        chainingRecordCopy_1_bits_state_wLaneClear = _RANDOM[5'hB][17];
        chainingRecordCopy_2_valid = _RANDOM[5'hB][18];
        chainingRecordCopy_2_bits_vd_valid = _RANDOM[5'hB][19];
        chainingRecordCopy_2_bits_vd_bits = _RANDOM[5'hB][24:20];
        chainingRecordCopy_2_bits_vs1_valid = _RANDOM[5'hB][25];
        chainingRecordCopy_2_bits_vs1_bits = _RANDOM[5'hB][30:26];
        chainingRecordCopy_2_bits_vs2 = {_RANDOM[5'hB][31], _RANDOM[5'hC][3:0]};
        chainingRecordCopy_2_bits_instIndex = _RANDOM[5'hC][6:4];
        chainingRecordCopy_2_bits_ls = _RANDOM[5'hC][7];
        chainingRecordCopy_2_bits_st = _RANDOM[5'hC][8];
        chainingRecordCopy_2_bits_gather = _RANDOM[5'hC][9];
        chainingRecordCopy_2_bits_gather16 = _RANDOM[5'hC][10];
        chainingRecordCopy_2_bits_crossWrite = _RANDOM[5'hC][11];
        chainingRecordCopy_2_bits_crossRead = _RANDOM[5'hC][12];
        chainingRecordCopy_2_bits_indexType = _RANDOM[5'hC][13];
        chainingRecordCopy_2_bits_ma = _RANDOM[5'hC][14];
        chainingRecordCopy_2_bits_onlyRead = _RANDOM[5'hC][15];
        chainingRecordCopy_2_bits_slow = _RANDOM[5'hC][16];
        chainingRecordCopy_2_bits_elementMask = {_RANDOM[5'hC][31:17], _RANDOM[5'hD][0]};
        chainingRecordCopy_2_bits_state_stFinish = _RANDOM[5'hD][1];
        chainingRecordCopy_2_bits_state_wWriteQueueClear = _RANDOM[5'hD][2];
        chainingRecordCopy_2_bits_state_wLaneLastReport = _RANDOM[5'hD][3];
        chainingRecordCopy_2_bits_state_wTopLastReport = _RANDOM[5'hD][4];
        chainingRecordCopy_2_bits_state_wLaneClear = _RANDOM[5'hD][5];
        chainingRecordCopy_3_valid = _RANDOM[5'hD][6];
        chainingRecordCopy_3_bits_vd_valid = _RANDOM[5'hD][7];
        chainingRecordCopy_3_bits_vd_bits = _RANDOM[5'hD][12:8];
        chainingRecordCopy_3_bits_vs1_valid = _RANDOM[5'hD][13];
        chainingRecordCopy_3_bits_vs1_bits = _RANDOM[5'hD][18:14];
        chainingRecordCopy_3_bits_vs2 = _RANDOM[5'hD][23:19];
        chainingRecordCopy_3_bits_instIndex = _RANDOM[5'hD][26:24];
        chainingRecordCopy_3_bits_ls = _RANDOM[5'hD][27];
        chainingRecordCopy_3_bits_st = _RANDOM[5'hD][28];
        chainingRecordCopy_3_bits_gather = _RANDOM[5'hD][29];
        chainingRecordCopy_3_bits_gather16 = _RANDOM[5'hD][30];
        chainingRecordCopy_3_bits_crossWrite = _RANDOM[5'hD][31];
        chainingRecordCopy_3_bits_crossRead = _RANDOM[5'hE][0];
        chainingRecordCopy_3_bits_indexType = _RANDOM[5'hE][1];
        chainingRecordCopy_3_bits_ma = _RANDOM[5'hE][2];
        chainingRecordCopy_3_bits_onlyRead = _RANDOM[5'hE][3];
        chainingRecordCopy_3_bits_slow = _RANDOM[5'hE][4];
        chainingRecordCopy_3_bits_elementMask = _RANDOM[5'hE][20:5];
        chainingRecordCopy_3_bits_state_stFinish = _RANDOM[5'hE][21];
        chainingRecordCopy_3_bits_state_wWriteQueueClear = _RANDOM[5'hE][22];
        chainingRecordCopy_3_bits_state_wLaneLastReport = _RANDOM[5'hE][23];
        chainingRecordCopy_3_bits_state_wTopLastReport = _RANDOM[5'hE][24];
        chainingRecordCopy_3_bits_state_wLaneClear = _RANDOM[5'hE][25];
        chainingRecordCopy_4_valid = _RANDOM[5'hE][26];
        chainingRecordCopy_4_bits_vd_valid = _RANDOM[5'hE][27];
        chainingRecordCopy_4_bits_vd_bits = {_RANDOM[5'hE][31:28], _RANDOM[5'hF][0]};
        chainingRecordCopy_4_bits_vs1_valid = _RANDOM[5'hF][1];
        chainingRecordCopy_4_bits_vs1_bits = _RANDOM[5'hF][6:2];
        chainingRecordCopy_4_bits_vs2 = _RANDOM[5'hF][11:7];
        chainingRecordCopy_4_bits_instIndex = _RANDOM[5'hF][14:12];
        chainingRecordCopy_4_bits_ls = _RANDOM[5'hF][15];
        chainingRecordCopy_4_bits_st = _RANDOM[5'hF][16];
        chainingRecordCopy_4_bits_gather = _RANDOM[5'hF][17];
        chainingRecordCopy_4_bits_gather16 = _RANDOM[5'hF][18];
        chainingRecordCopy_4_bits_crossWrite = _RANDOM[5'hF][19];
        chainingRecordCopy_4_bits_crossRead = _RANDOM[5'hF][20];
        chainingRecordCopy_4_bits_indexType = _RANDOM[5'hF][21];
        chainingRecordCopy_4_bits_ma = _RANDOM[5'hF][22];
        chainingRecordCopy_4_bits_onlyRead = _RANDOM[5'hF][23];
        chainingRecordCopy_4_bits_slow = _RANDOM[5'hF][24];
        chainingRecordCopy_4_bits_elementMask = {_RANDOM[5'hF][31:25], _RANDOM[5'h10][8:0]};
        chainingRecordCopy_4_bits_state_stFinish = _RANDOM[5'h10][9];
        chainingRecordCopy_4_bits_state_wWriteQueueClear = _RANDOM[5'h10][10];
        chainingRecordCopy_4_bits_state_wLaneLastReport = _RANDOM[5'h10][11];
        chainingRecordCopy_4_bits_state_wTopLastReport = _RANDOM[5'h10][12];
        chainingRecordCopy_4_bits_state_wLaneClear = _RANDOM[5'h10][13];
        firstReadPipe_0_valid = _RANDOM[5'h10][14];
        firstReadPipe_0_bits_address = _RANDOM[5'h10][19:15];
        firstReadPipe_1_valid = _RANDOM[5'h10][20];
        firstReadPipe_1_bits_address = _RANDOM[5'h10][25:21];
        secondReadPipe_0_valid = _RANDOM[5'h10][26];
        secondReadPipe_0_bits_address = _RANDOM[5'h10][31:27];
        secondReadPipe_1_valid = _RANDOM[5'h11][0];
        secondReadPipe_1_bits_address = _RANDOM[5'h11][5:1];
        writePipe_valid = _RANDOM[5'h11][6];
        writePipe_bits_vd = _RANDOM[5'h11][11:7];
        writePipe_bits_offset = _RANDOM[5'h11][12];
        writePipe_bits_mask = _RANDOM[5'h11][16:13];
        writePipe_bits_data = {_RANDOM[5'h11][31:17], _RANDOM[5'h12][16:0]};
        writePipe_bits_last = _RANDOM[5'h12][17];
        writePipe_bits_instructionIndex = _RANDOM[5'h12][20:18];
        writeBankPipe = _RANDOM[5'h12][22:21];
        pipeBank_pipe_v = _RANDOM[5'h12][23];
        pipeBank_pipe_b = _RANDOM[5'h12][25:24];
        pipeBank_pipe_pipe_v = _RANDOM[5'h12][26];
        pipeBank_pipe_pipe_b = _RANDOM[5'h12][28:27];
        pipeFirstUsed_pipe_v = _RANDOM[5'h12][29];
        pipeFirstUsed_pipe_pipe_v = _RANDOM[5'h12][31];
        pipeFirstUsed_pipe_pipe_b = _RANDOM[5'h13][0];
        pipeFire_pipe_v = _RANDOM[5'h13][1];
        pipeFire_pipe_b = _RANDOM[5'h13][2];
        pipeFire_pipe_pipe_v = _RANDOM[5'h13][3];
        pipeFire_pipe_pipe_b = _RANDOM[5'h13][4];
        pipeBank_pipe_v_1 = _RANDOM[5'h13][5];
        pipeBank_pipe_b_1 = _RANDOM[5'h13][7:6];
        pipeBank_pipe_pipe_v_1 = _RANDOM[5'h13][8];
        pipeBank_pipe_pipe_b_1 = _RANDOM[5'h13][10:9];
        pipeFirstUsed_pipe_v_1 = _RANDOM[5'h13][11];
        pipeFirstUsed_pipe_b_1 = _RANDOM[5'h13][12];
        pipeFirstUsed_pipe_pipe_v_1 = _RANDOM[5'h13][13];
        pipeFirstUsed_pipe_pipe_b_1 = _RANDOM[5'h13][14];
        pipeFire_pipe_v_1 = _RANDOM[5'h13][15];
        pipeFire_pipe_b_1 = _RANDOM[5'h13][16];
        pipeFire_pipe_pipe_v_1 = _RANDOM[5'h13][17];
        pipeFire_pipe_pipe_b_1 = _RANDOM[5'h13][18];
        pipeBank_pipe_v_2 = _RANDOM[5'h13][19];
        pipeBank_pipe_b_2 = _RANDOM[5'h13][21:20];
        pipeBank_pipe_pipe_v_2 = _RANDOM[5'h13][22];
        pipeBank_pipe_pipe_b_2 = _RANDOM[5'h13][24:23];
        pipeFirstUsed_pipe_v_2 = _RANDOM[5'h13][25];
        pipeFirstUsed_pipe_b_2 = _RANDOM[5'h13][26];
        pipeFirstUsed_pipe_pipe_v_2 = _RANDOM[5'h13][27];
        pipeFirstUsed_pipe_pipe_b_2 = _RANDOM[5'h13][28];
        pipeFire_pipe_v_2 = _RANDOM[5'h13][29];
        pipeFire_pipe_b_2 = _RANDOM[5'h13][30];
        pipeFire_pipe_pipe_v_2 = _RANDOM[5'h13][31];
        pipeFire_pipe_pipe_b_2 = _RANDOM[5'h14][0];
        pipeBank_pipe_v_3 = _RANDOM[5'h14][1];
        pipeBank_pipe_b_3 = _RANDOM[5'h14][3:2];
        pipeBank_pipe_pipe_v_3 = _RANDOM[5'h14][4];
        pipeBank_pipe_pipe_b_3 = _RANDOM[5'h14][6:5];
        pipeFirstUsed_pipe_v_3 = _RANDOM[5'h14][7];
        pipeFirstUsed_pipe_b_3 = _RANDOM[5'h14][8];
        pipeFirstUsed_pipe_pipe_v_3 = _RANDOM[5'h14][9];
        pipeFirstUsed_pipe_pipe_b_3 = _RANDOM[5'h14][10];
        pipeFire_pipe_v_3 = _RANDOM[5'h14][11];
        pipeFire_pipe_b_3 = _RANDOM[5'h14][12];
        pipeFire_pipe_pipe_v_3 = _RANDOM[5'h14][13];
        pipeFire_pipe_pipe_b_3 = _RANDOM[5'h14][14];
        pipeBank_pipe_v_4 = _RANDOM[5'h14][15];
        pipeBank_pipe_b_4 = _RANDOM[5'h14][17:16];
        pipeBank_pipe_pipe_v_4 = _RANDOM[5'h14][18];
        pipeBank_pipe_pipe_b_4 = _RANDOM[5'h14][20:19];
        pipeFirstUsed_pipe_v_4 = _RANDOM[5'h14][21];
        pipeFirstUsed_pipe_b_4 = _RANDOM[5'h14][22];
        pipeFirstUsed_pipe_pipe_v_4 = _RANDOM[5'h14][23];
        pipeFirstUsed_pipe_pipe_b_4 = _RANDOM[5'h14][24];
        pipeFire_pipe_v_4 = _RANDOM[5'h14][25];
        pipeFire_pipe_b_4 = _RANDOM[5'h14][26];
        pipeFire_pipe_pipe_v_4 = _RANDOM[5'h14][27];
        pipeFire_pipe_pipe_b_4 = _RANDOM[5'h14][28];
        pipeBank_pipe_v_5 = _RANDOM[5'h14][29];
        pipeBank_pipe_b_5 = _RANDOM[5'h14][31:30];
        pipeBank_pipe_pipe_v_5 = _RANDOM[5'h15][0];
        pipeBank_pipe_pipe_b_5 = _RANDOM[5'h15][2:1];
        pipeFirstUsed_pipe_v_5 = _RANDOM[5'h15][3];
        pipeFirstUsed_pipe_b_5 = _RANDOM[5'h15][4];
        pipeFirstUsed_pipe_pipe_v_5 = _RANDOM[5'h15][5];
        pipeFirstUsed_pipe_pipe_b_5 = _RANDOM[5'h15][6];
        pipeFire_pipe_v_5 = _RANDOM[5'h15][7];
        pipeFire_pipe_b_5 = _RANDOM[5'h15][8];
        pipeFire_pipe_pipe_v_5 = _RANDOM[5'h15][9];
        pipeFire_pipe_pipe_b_5 = _RANDOM[5'h15][10];
        pipeBank_pipe_v_6 = _RANDOM[5'h15][11];
        pipeBank_pipe_b_6 = _RANDOM[5'h15][13:12];
        pipeBank_pipe_pipe_v_6 = _RANDOM[5'h15][14];
        pipeBank_pipe_pipe_b_6 = _RANDOM[5'h15][16:15];
        pipeFirstUsed_pipe_v_6 = _RANDOM[5'h15][17];
        pipeFirstUsed_pipe_b_6 = _RANDOM[5'h15][18];
        pipeFirstUsed_pipe_pipe_v_6 = _RANDOM[5'h15][19];
        pipeFirstUsed_pipe_pipe_b_6 = _RANDOM[5'h15][20];
        pipeFire_pipe_v_6 = _RANDOM[5'h15][21];
        pipeFire_pipe_b_6 = _RANDOM[5'h15][22];
        pipeFire_pipe_pipe_v_6 = _RANDOM[5'h15][23];
        pipeFire_pipe_pipe_b_6 = _RANDOM[5'h15][24];
        pipeBank_pipe_v_7 = _RANDOM[5'h15][25];
        pipeBank_pipe_b_7 = _RANDOM[5'h15][27:26];
        pipeBank_pipe_pipe_v_7 = _RANDOM[5'h15][28];
        pipeBank_pipe_pipe_b_7 = _RANDOM[5'h15][30:29];
        pipeFirstUsed_pipe_v_7 = _RANDOM[5'h15][31];
        pipeFirstUsed_pipe_b_7 = _RANDOM[5'h16][0];
        pipeFirstUsed_pipe_pipe_v_7 = _RANDOM[5'h16][1];
        pipeFirstUsed_pipe_pipe_b_7 = _RANDOM[5'h16][2];
        pipeFire_pipe_v_7 = _RANDOM[5'h16][3];
        pipeFire_pipe_b_7 = _RANDOM[5'h16][4];
        pipeFire_pipe_pipe_v_7 = _RANDOM[5'h16][5];
        pipeFire_pipe_pipe_b_7 = _RANDOM[5'h16][6];
        pipeBank_pipe_v_8 = _RANDOM[5'h16][7];
        pipeBank_pipe_b_8 = _RANDOM[5'h16][9:8];
        pipeBank_pipe_pipe_v_8 = _RANDOM[5'h16][10];
        pipeBank_pipe_pipe_b_8 = _RANDOM[5'h16][12:11];
        pipeFirstUsed_pipe_v_8 = _RANDOM[5'h16][13];
        pipeFirstUsed_pipe_b_8 = _RANDOM[5'h16][14];
        pipeFirstUsed_pipe_pipe_v_8 = _RANDOM[5'h16][15];
        pipeFirstUsed_pipe_pipe_b_8 = _RANDOM[5'h16][16];
        pipeFire_pipe_v_8 = _RANDOM[5'h16][17];
        pipeFire_pipe_b_8 = _RANDOM[5'h16][18];
        pipeFire_pipe_pipe_v_8 = _RANDOM[5'h16][19];
        pipeFire_pipe_pipe_b_8 = _RANDOM[5'h16][20];
        pipeBank_pipe_v_9 = _RANDOM[5'h16][21];
        pipeBank_pipe_b_9 = _RANDOM[5'h16][23:22];
        pipeBank_pipe_pipe_v_9 = _RANDOM[5'h16][24];
        pipeBank_pipe_pipe_b_9 = _RANDOM[5'h16][26:25];
        pipeFirstUsed_pipe_v_9 = _RANDOM[5'h16][27];
        pipeFirstUsed_pipe_b_9 = _RANDOM[5'h16][28];
        pipeFirstUsed_pipe_pipe_v_9 = _RANDOM[5'h16][29];
        pipeFirstUsed_pipe_pipe_b_9 = _RANDOM[5'h16][30];
        pipeFire_pipe_v_9 = _RANDOM[5'h16][31];
        pipeFire_pipe_b_9 = _RANDOM[5'h17][0];
        pipeFire_pipe_pipe_v_9 = _RANDOM[5'h17][1];
        pipeFire_pipe_pipe_b_9 = _RANDOM[5'h17][2];
        pipeBank_pipe_v_10 = _RANDOM[5'h17][3];
        pipeBank_pipe_b_10 = _RANDOM[5'h17][5:4];
        pipeBank_pipe_pipe_v_10 = _RANDOM[5'h17][6];
        pipeBank_pipe_pipe_b_10 = _RANDOM[5'h17][8:7];
        pipeFirstUsed_pipe_v_10 = _RANDOM[5'h17][9];
        pipeFirstUsed_pipe_b_10 = _RANDOM[5'h17][10];
        pipeFirstUsed_pipe_pipe_v_10 = _RANDOM[5'h17][11];
        pipeFirstUsed_pipe_pipe_b_10 = _RANDOM[5'h17][12];
        pipeFire_pipe_v_10 = _RANDOM[5'h17][13];
        pipeFire_pipe_b_10 = _RANDOM[5'h17][14];
        pipeFire_pipe_pipe_v_10 = _RANDOM[5'h17][15];
        pipeFire_pipe_pipe_b_10 = _RANDOM[5'h17][16];
        pipeBank_pipe_v_11 = _RANDOM[5'h17][17];
        pipeBank_pipe_b_11 = _RANDOM[5'h17][19:18];
        pipeBank_pipe_pipe_v_11 = _RANDOM[5'h17][20];
        pipeBank_pipe_pipe_b_11 = _RANDOM[5'h17][22:21];
        pipeFirstUsed_pipe_v_11 = _RANDOM[5'h17][23];
        pipeFirstUsed_pipe_b_11 = _RANDOM[5'h17][24];
        pipeFirstUsed_pipe_pipe_v_11 = _RANDOM[5'h17][25];
        pipeFirstUsed_pipe_pipe_b_11 = _RANDOM[5'h17][26];
        pipeFire_pipe_v_11 = _RANDOM[5'h17][27];
        pipeFire_pipe_b_11 = _RANDOM[5'h17][28];
        pipeFire_pipe_pipe_v_11 = _RANDOM[5'h17][29];
        pipeFire_pipe_pipe_b_11 = _RANDOM[5'h17][30];
        pipeBank_pipe_v_12 = _RANDOM[5'h17][31];
        pipeBank_pipe_b_12 = _RANDOM[5'h18][1:0];
        pipeBank_pipe_pipe_v_12 = _RANDOM[5'h18][2];
        pipeBank_pipe_pipe_b_12 = _RANDOM[5'h18][4:3];
        pipeFirstUsed_pipe_v_12 = _RANDOM[5'h18][5];
        pipeFirstUsed_pipe_b_12 = _RANDOM[5'h18][6];
        pipeFirstUsed_pipe_pipe_v_12 = _RANDOM[5'h18][7];
        pipeFirstUsed_pipe_pipe_b_12 = _RANDOM[5'h18][8];
        pipeFire_pipe_v_12 = _RANDOM[5'h18][9];
        pipeFire_pipe_b_12 = _RANDOM[5'h18][10];
        pipeFire_pipe_pipe_v_12 = _RANDOM[5'h18][11];
        pipeFire_pipe_pipe_b_12 = _RANDOM[5'h18][12];
        pipeBank_pipe_v_13 = _RANDOM[5'h18][13];
        pipeBank_pipe_b_13 = _RANDOM[5'h18][15:14];
        pipeBank_pipe_pipe_v_13 = _RANDOM[5'h18][16];
        pipeBank_pipe_pipe_b_13 = _RANDOM[5'h18][18:17];
        pipeFirstUsed_pipe_v_13 = _RANDOM[5'h18][19];
        pipeFirstUsed_pipe_b_13 = _RANDOM[5'h18][20];
        pipeFirstUsed_pipe_pipe_v_13 = _RANDOM[5'h18][21];
        pipeFirstUsed_pipe_pipe_b_13 = _RANDOM[5'h18][22];
        pipeFire_pipe_v_13 = _RANDOM[5'h18][23];
        pipeFire_pipe_b_13 = _RANDOM[5'h18][24];
        pipeFire_pipe_pipe_v_13 = _RANDOM[5'h18][25];
        pipeFire_pipe_pipe_b_13 = _RANDOM[5'h18][26];
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
  sram_0R_0W_2RW_0M_32x36 vrfSRAM_mem (
    .RW0_addr  (vrfSRAM_0_readwritePorts_0_address),
    .RW0_en    (vrfSRAM_0_readwritePorts_0_enable),
    .RW0_clk   (clock),
    .RW0_wmode (1'h0),
    .RW0_wdata (36'h0),
    .RW0_rdata (vrfSRAM_0_readwritePorts_0_readData),
    .RW1_addr  (vrfSRAM_0_readwritePorts_1_address),
    .RW1_en    (vrfSRAM_0_readwritePorts_1_enable),
    .RW1_clk   (clock),
    .RW1_wmode (vrfSRAM_0_readwritePorts_1_isWrite),
    .RW1_wdata (vrfSRAM_0_readwritePorts_1_writeData),
    .RW1_rdata (vrfSRAM_0_readwritePorts_1_readData)
  );
  sram_0R_0W_2RW_0M_32x36 vrfSRAM_mem_1 (
    .RW0_addr  (vrfSRAM_1_readwritePorts_0_address),
    .RW0_en    (vrfSRAM_1_readwritePorts_0_enable),
    .RW0_clk   (clock),
    .RW0_wmode (1'h0),
    .RW0_wdata (36'h0),
    .RW0_rdata (vrfSRAM_1_readwritePorts_0_readData),
    .RW1_addr  (vrfSRAM_1_readwritePorts_1_address),
    .RW1_en    (vrfSRAM_1_readwritePorts_1_enable),
    .RW1_clk   (clock),
    .RW1_wmode (vrfSRAM_1_readwritePorts_1_isWrite),
    .RW1_wdata (vrfSRAM_1_readwritePorts_1_writeData),
    .RW1_rdata (vrfSRAM_1_readwritePorts_1_readData)
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
  assign readResults_0 =
    (~pipeFirstUsed_pipe_pipe_out_bits & pipeFire_pipe_pipe_out_bits ? (pipeBank_pipe_pipe_out_bits[0] ? readResultF_0 : 32'h0) | (pipeBank_pipe_pipe_out_bits[1] ? readResultF_1 : 32'h0) : 32'h0)
    | (pipeFirstUsed_pipe_pipe_out_bits & pipeFire_pipe_pipe_out_bits ? (pipeBank_pipe_pipe_out_bits[0] ? readResultS_0 : 32'h0) | (pipeBank_pipe_pipe_out_bits[1] ? readResultS_1 : 32'h0) : 32'h0);
  assign readResults_1 =
    (~pipeFirstUsed_pipe_pipe_out_1_bits & pipeFire_pipe_pipe_out_1_bits ? (pipeBank_pipe_pipe_out_1_bits[0] ? readResultF_0 : 32'h0) | (pipeBank_pipe_pipe_out_1_bits[1] ? readResultF_1 : 32'h0) : 32'h0)
    | (pipeFirstUsed_pipe_pipe_out_1_bits & pipeFire_pipe_pipe_out_1_bits ? (pipeBank_pipe_pipe_out_1_bits[0] ? readResultS_0 : 32'h0) | (pipeBank_pipe_pipe_out_1_bits[1] ? readResultS_1 : 32'h0) : 32'h0);
  assign readResults_2 =
    (~pipeFirstUsed_pipe_pipe_out_2_bits & pipeFire_pipe_pipe_out_2_bits ? (pipeBank_pipe_pipe_out_2_bits[0] ? readResultF_0 : 32'h0) | (pipeBank_pipe_pipe_out_2_bits[1] ? readResultF_1 : 32'h0) : 32'h0)
    | (pipeFirstUsed_pipe_pipe_out_2_bits & pipeFire_pipe_pipe_out_2_bits ? (pipeBank_pipe_pipe_out_2_bits[0] ? readResultS_0 : 32'h0) | (pipeBank_pipe_pipe_out_2_bits[1] ? readResultS_1 : 32'h0) : 32'h0);
  assign readResults_3 =
    (~pipeFirstUsed_pipe_pipe_out_3_bits & pipeFire_pipe_pipe_out_3_bits ? (pipeBank_pipe_pipe_out_3_bits[0] ? readResultF_0 : 32'h0) | (pipeBank_pipe_pipe_out_3_bits[1] ? readResultF_1 : 32'h0) : 32'h0)
    | (pipeFirstUsed_pipe_pipe_out_3_bits & pipeFire_pipe_pipe_out_3_bits ? (pipeBank_pipe_pipe_out_3_bits[0] ? readResultS_0 : 32'h0) | (pipeBank_pipe_pipe_out_3_bits[1] ? readResultS_1 : 32'h0) : 32'h0);
  assign readResults_4 =
    (~pipeFirstUsed_pipe_pipe_out_4_bits & pipeFire_pipe_pipe_out_4_bits ? (pipeBank_pipe_pipe_out_4_bits[0] ? readResultF_0 : 32'h0) | (pipeBank_pipe_pipe_out_4_bits[1] ? readResultF_1 : 32'h0) : 32'h0)
    | (pipeFirstUsed_pipe_pipe_out_4_bits & pipeFire_pipe_pipe_out_4_bits ? (pipeBank_pipe_pipe_out_4_bits[0] ? readResultS_0 : 32'h0) | (pipeBank_pipe_pipe_out_4_bits[1] ? readResultS_1 : 32'h0) : 32'h0);
  assign readResults_5 =
    (~pipeFirstUsed_pipe_pipe_out_5_bits & pipeFire_pipe_pipe_out_5_bits ? (pipeBank_pipe_pipe_out_5_bits[0] ? readResultF_0 : 32'h0) | (pipeBank_pipe_pipe_out_5_bits[1] ? readResultF_1 : 32'h0) : 32'h0)
    | (pipeFirstUsed_pipe_pipe_out_5_bits & pipeFire_pipe_pipe_out_5_bits ? (pipeBank_pipe_pipe_out_5_bits[0] ? readResultS_0 : 32'h0) | (pipeBank_pipe_pipe_out_5_bits[1] ? readResultS_1 : 32'h0) : 32'h0);
  assign readResults_6 =
    (~pipeFirstUsed_pipe_pipe_out_6_bits & pipeFire_pipe_pipe_out_6_bits ? (pipeBank_pipe_pipe_out_6_bits[0] ? readResultF_0 : 32'h0) | (pipeBank_pipe_pipe_out_6_bits[1] ? readResultF_1 : 32'h0) : 32'h0)
    | (pipeFirstUsed_pipe_pipe_out_6_bits & pipeFire_pipe_pipe_out_6_bits ? (pipeBank_pipe_pipe_out_6_bits[0] ? readResultS_0 : 32'h0) | (pipeBank_pipe_pipe_out_6_bits[1] ? readResultS_1 : 32'h0) : 32'h0);
  assign readResults_7 =
    (~pipeFirstUsed_pipe_pipe_out_7_bits & pipeFire_pipe_pipe_out_7_bits ? (pipeBank_pipe_pipe_out_7_bits[0] ? readResultF_0 : 32'h0) | (pipeBank_pipe_pipe_out_7_bits[1] ? readResultF_1 : 32'h0) : 32'h0)
    | (pipeFirstUsed_pipe_pipe_out_7_bits & pipeFire_pipe_pipe_out_7_bits ? (pipeBank_pipe_pipe_out_7_bits[0] ? readResultS_0 : 32'h0) | (pipeBank_pipe_pipe_out_7_bits[1] ? readResultS_1 : 32'h0) : 32'h0);
  assign readResults_8 =
    (~pipeFirstUsed_pipe_pipe_out_8_bits & pipeFire_pipe_pipe_out_8_bits ? (pipeBank_pipe_pipe_out_8_bits[0] ? readResultF_0 : 32'h0) | (pipeBank_pipe_pipe_out_8_bits[1] ? readResultF_1 : 32'h0) : 32'h0)
    | (pipeFirstUsed_pipe_pipe_out_8_bits & pipeFire_pipe_pipe_out_8_bits ? (pipeBank_pipe_pipe_out_8_bits[0] ? readResultS_0 : 32'h0) | (pipeBank_pipe_pipe_out_8_bits[1] ? readResultS_1 : 32'h0) : 32'h0);
  assign readResults_9 =
    (~pipeFirstUsed_pipe_pipe_out_9_bits & pipeFire_pipe_pipe_out_9_bits ? (pipeBank_pipe_pipe_out_9_bits[0] ? readResultF_0 : 32'h0) | (pipeBank_pipe_pipe_out_9_bits[1] ? readResultF_1 : 32'h0) : 32'h0)
    | (pipeFirstUsed_pipe_pipe_out_9_bits & pipeFire_pipe_pipe_out_9_bits ? (pipeBank_pipe_pipe_out_9_bits[0] ? readResultS_0 : 32'h0) | (pipeBank_pipe_pipe_out_9_bits[1] ? readResultS_1 : 32'h0) : 32'h0);
  assign readResults_10 =
    (~pipeFirstUsed_pipe_pipe_out_10_bits & pipeFire_pipe_pipe_out_10_bits ? (pipeBank_pipe_pipe_out_10_bits[0] ? readResultF_0 : 32'h0) | (pipeBank_pipe_pipe_out_10_bits[1] ? readResultF_1 : 32'h0) : 32'h0)
    | (pipeFirstUsed_pipe_pipe_out_10_bits & pipeFire_pipe_pipe_out_10_bits ? (pipeBank_pipe_pipe_out_10_bits[0] ? readResultS_0 : 32'h0) | (pipeBank_pipe_pipe_out_10_bits[1] ? readResultS_1 : 32'h0) : 32'h0);
  assign readResults_11 =
    (~pipeFirstUsed_pipe_pipe_out_11_bits & pipeFire_pipe_pipe_out_11_bits ? (pipeBank_pipe_pipe_out_11_bits[0] ? readResultF_0 : 32'h0) | (pipeBank_pipe_pipe_out_11_bits[1] ? readResultF_1 : 32'h0) : 32'h0)
    | (pipeFirstUsed_pipe_pipe_out_11_bits & pipeFire_pipe_pipe_out_11_bits ? (pipeBank_pipe_pipe_out_11_bits[0] ? readResultS_0 : 32'h0) | (pipeBank_pipe_pipe_out_11_bits[1] ? readResultS_1 : 32'h0) : 32'h0);
  assign readResults_12 =
    (~pipeFirstUsed_pipe_pipe_out_12_bits & pipeFire_pipe_pipe_out_12_bits ? (pipeBank_pipe_pipe_out_12_bits[0] ? readResultF_0 : 32'h0) | (pipeBank_pipe_pipe_out_12_bits[1] ? readResultF_1 : 32'h0) : 32'h0)
    | (pipeFirstUsed_pipe_pipe_out_12_bits & pipeFire_pipe_pipe_out_12_bits ? (pipeBank_pipe_pipe_out_12_bits[0] ? readResultS_0 : 32'h0) | (pipeBank_pipe_pipe_out_12_bits[1] ? readResultS_1 : 32'h0) : 32'h0);
  assign readResults_13 =
    (~pipeFirstUsed_pipe_pipe_out_13_bits & pipeFire_pipe_pipe_out_13_bits ? (pipeBank_pipe_pipe_out_13_bits[0] ? readResultF_0 : 32'h0) | (pipeBank_pipe_pipe_out_13_bits[1] ? readResultF_1 : 32'h0) : 32'h0)
    | (pipeFirstUsed_pipe_pipe_out_13_bits & pipeFire_pipe_pipe_out_13_bits ? (pipeBank_pipe_pipe_out_13_bits[0] ? readResultS_0 : 32'h0) | (pipeBank_pipe_pipe_out_13_bits[1] ? readResultS_1 : 32'h0) : 32'h0);
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

