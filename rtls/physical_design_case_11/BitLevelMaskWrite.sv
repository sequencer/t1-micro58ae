
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
module BitLevelMaskWrite(
  input         clock,
                reset,
                needWAR,
  input  [4:0]  vd,
  output        in_0_ready,
  input         in_0_valid,
  input  [31:0] in_0_bits_data,
                in_0_bits_bitMask,
  input  [3:0]  in_0_bits_mask,
  input  [6:0]  in_0_bits_groupCounter,
  input         in_0_bits_ffoByOther,
  output        in_1_ready,
  input         in_1_valid,
  input  [31:0] in_1_bits_data,
                in_1_bits_bitMask,
  input  [3:0]  in_1_bits_mask,
  input  [6:0]  in_1_bits_groupCounter,
  input         in_1_bits_ffoByOther,
  output        in_2_ready,
  input         in_2_valid,
  input  [31:0] in_2_bits_data,
                in_2_bits_bitMask,
  input  [3:0]  in_2_bits_mask,
  input  [6:0]  in_2_bits_groupCounter,
  input         in_2_bits_ffoByOther,
  output        in_3_ready,
  input         in_3_valid,
  input  [31:0] in_3_bits_data,
                in_3_bits_bitMask,
  input  [3:0]  in_3_bits_mask,
  input  [6:0]  in_3_bits_groupCounter,
  input         in_3_bits_ffoByOther,
  output        in_4_ready,
  input         in_4_valid,
  input  [31:0] in_4_bits_data,
                in_4_bits_bitMask,
  input  [3:0]  in_4_bits_mask,
  input  [6:0]  in_4_bits_groupCounter,
  input         in_4_bits_ffoByOther,
  output        in_5_ready,
  input         in_5_valid,
  input  [31:0] in_5_bits_data,
                in_5_bits_bitMask,
  input  [3:0]  in_5_bits_mask,
  input  [6:0]  in_5_bits_groupCounter,
  input         in_5_bits_ffoByOther,
  output        in_6_ready,
  input         in_6_valid,
  input  [31:0] in_6_bits_data,
                in_6_bits_bitMask,
  input  [3:0]  in_6_bits_mask,
  input  [6:0]  in_6_bits_groupCounter,
  input         in_6_bits_ffoByOther,
  output        in_7_ready,
  input         in_7_valid,
  input  [31:0] in_7_bits_data,
                in_7_bits_bitMask,
  input  [3:0]  in_7_bits_mask,
  input  [6:0]  in_7_bits_groupCounter,
  input         in_7_bits_ffoByOther,
                out_0_ready,
  output        out_0_valid,
                out_0_bits_ffoByOther,
  output [31:0] out_0_bits_writeData_data,
  output [3:0]  out_0_bits_writeData_mask,
  output [6:0]  out_0_bits_writeData_groupCounter,
  input         out_1_ready,
  output        out_1_valid,
                out_1_bits_ffoByOther,
  output [31:0] out_1_bits_writeData_data,
  output [3:0]  out_1_bits_writeData_mask,
  output [6:0]  out_1_bits_writeData_groupCounter,
  input         out_2_ready,
  output        out_2_valid,
                out_2_bits_ffoByOther,
  output [31:0] out_2_bits_writeData_data,
  output [3:0]  out_2_bits_writeData_mask,
  output [6:0]  out_2_bits_writeData_groupCounter,
  input         out_3_ready,
  output        out_3_valid,
                out_3_bits_ffoByOther,
  output [31:0] out_3_bits_writeData_data,
  output [3:0]  out_3_bits_writeData_mask,
  output [6:0]  out_3_bits_writeData_groupCounter,
  input         out_4_ready,
  output        out_4_valid,
                out_4_bits_ffoByOther,
  output [31:0] out_4_bits_writeData_data,
  output [3:0]  out_4_bits_writeData_mask,
  output [6:0]  out_4_bits_writeData_groupCounter,
  input         out_5_ready,
  output        out_5_valid,
                out_5_bits_ffoByOther,
  output [31:0] out_5_bits_writeData_data,
  output [3:0]  out_5_bits_writeData_mask,
  output [6:0]  out_5_bits_writeData_groupCounter,
  input         out_6_ready,
  output        out_6_valid,
                out_6_bits_ffoByOther,
  output [31:0] out_6_bits_writeData_data,
  output [3:0]  out_6_bits_writeData_mask,
  output [6:0]  out_6_bits_writeData_groupCounter,
  input         out_7_ready,
  output        out_7_valid,
                out_7_bits_ffoByOther,
  output [31:0] out_7_bits_writeData_data,
  output [3:0]  out_7_bits_writeData_mask,
  output [6:0]  out_7_bits_writeData_groupCounter,
  input         readChannel_0_ready,
  output        readChannel_0_valid,
  output [4:0]  readChannel_0_bits_vs,
  output [2:0]  readChannel_0_bits_offset,
  input         readChannel_1_ready,
  output        readChannel_1_valid,
  output [4:0]  readChannel_1_bits_vs,
  output [2:0]  readChannel_1_bits_offset,
  input         readChannel_2_ready,
  output        readChannel_2_valid,
  output [4:0]  readChannel_2_bits_vs,
  output [2:0]  readChannel_2_bits_offset,
  input         readChannel_3_ready,
  output        readChannel_3_valid,
  output [4:0]  readChannel_3_bits_vs,
  output [2:0]  readChannel_3_bits_offset,
  input         readChannel_4_ready,
  output        readChannel_4_valid,
  output [4:0]  readChannel_4_bits_vs,
  output [2:0]  readChannel_4_bits_offset,
  input         readChannel_5_ready,
  output        readChannel_5_valid,
  output [4:0]  readChannel_5_bits_vs,
  output [2:0]  readChannel_5_bits_offset,
  input         readChannel_6_ready,
  output        readChannel_6_valid,
  output [4:0]  readChannel_6_bits_vs,
  output [2:0]  readChannel_6_bits_offset,
  input         readChannel_7_ready,
  output        readChannel_7_valid,
  output [4:0]  readChannel_7_bits_vs,
  output [2:0]  readChannel_7_bits_offset,
  input         readResult_0_valid,
  input  [31:0] readResult_0_bits,
  input         readResult_1_valid,
  input  [31:0] readResult_1_bits,
  input         readResult_2_valid,
  input  [31:0] readResult_2_bits,
  input         readResult_3_valid,
  input  [31:0] readResult_3_bits,
  input         readResult_4_valid,
  input  [31:0] readResult_4_bits,
  input         readResult_5_valid,
  input  [31:0] readResult_5_bits,
  input         readResult_6_valid,
  input  [31:0] readResult_6_bits,
  input         readResult_7_valid,
  input  [31:0] readResult_7_bits,
  output        stageClear
);

  wire        _stageClearVec_WaitReadQueue_fifo_7_empty;
  wire        _stageClearVec_WaitReadQueue_fifo_7_full;
  wire        _stageClearVec_WaitReadQueue_fifo_7_error;
  wire [75:0] _stageClearVec_WaitReadQueue_fifo_7_data_out;
  wire        _stageClearVec_reqQueue_fifo_7_empty;
  wire        _stageClearVec_reqQueue_fifo_7_full;
  wire        _stageClearVec_reqQueue_fifo_7_error;
  wire [75:0] _stageClearVec_reqQueue_fifo_7_data_out;
  wire        _stageClearVec_WaitReadQueue_fifo_6_empty;
  wire        _stageClearVec_WaitReadQueue_fifo_6_full;
  wire        _stageClearVec_WaitReadQueue_fifo_6_error;
  wire [75:0] _stageClearVec_WaitReadQueue_fifo_6_data_out;
  wire        _stageClearVec_reqQueue_fifo_6_empty;
  wire        _stageClearVec_reqQueue_fifo_6_full;
  wire        _stageClearVec_reqQueue_fifo_6_error;
  wire [75:0] _stageClearVec_reqQueue_fifo_6_data_out;
  wire        _stageClearVec_WaitReadQueue_fifo_5_empty;
  wire        _stageClearVec_WaitReadQueue_fifo_5_full;
  wire        _stageClearVec_WaitReadQueue_fifo_5_error;
  wire [75:0] _stageClearVec_WaitReadQueue_fifo_5_data_out;
  wire        _stageClearVec_reqQueue_fifo_5_empty;
  wire        _stageClearVec_reqQueue_fifo_5_full;
  wire        _stageClearVec_reqQueue_fifo_5_error;
  wire [75:0] _stageClearVec_reqQueue_fifo_5_data_out;
  wire        _stageClearVec_WaitReadQueue_fifo_4_empty;
  wire        _stageClearVec_WaitReadQueue_fifo_4_full;
  wire        _stageClearVec_WaitReadQueue_fifo_4_error;
  wire [75:0] _stageClearVec_WaitReadQueue_fifo_4_data_out;
  wire        _stageClearVec_reqQueue_fifo_4_empty;
  wire        _stageClearVec_reqQueue_fifo_4_full;
  wire        _stageClearVec_reqQueue_fifo_4_error;
  wire [75:0] _stageClearVec_reqQueue_fifo_4_data_out;
  wire        _stageClearVec_WaitReadQueue_fifo_3_empty;
  wire        _stageClearVec_WaitReadQueue_fifo_3_full;
  wire        _stageClearVec_WaitReadQueue_fifo_3_error;
  wire [75:0] _stageClearVec_WaitReadQueue_fifo_3_data_out;
  wire        _stageClearVec_reqQueue_fifo_3_empty;
  wire        _stageClearVec_reqQueue_fifo_3_full;
  wire        _stageClearVec_reqQueue_fifo_3_error;
  wire [75:0] _stageClearVec_reqQueue_fifo_3_data_out;
  wire        _stageClearVec_WaitReadQueue_fifo_2_empty;
  wire        _stageClearVec_WaitReadQueue_fifo_2_full;
  wire        _stageClearVec_WaitReadQueue_fifo_2_error;
  wire [75:0] _stageClearVec_WaitReadQueue_fifo_2_data_out;
  wire        _stageClearVec_reqQueue_fifo_2_empty;
  wire        _stageClearVec_reqQueue_fifo_2_full;
  wire        _stageClearVec_reqQueue_fifo_2_error;
  wire [75:0] _stageClearVec_reqQueue_fifo_2_data_out;
  wire        _stageClearVec_WaitReadQueue_fifo_1_empty;
  wire        _stageClearVec_WaitReadQueue_fifo_1_full;
  wire        _stageClearVec_WaitReadQueue_fifo_1_error;
  wire [75:0] _stageClearVec_WaitReadQueue_fifo_1_data_out;
  wire        _stageClearVec_reqQueue_fifo_1_empty;
  wire        _stageClearVec_reqQueue_fifo_1_full;
  wire        _stageClearVec_reqQueue_fifo_1_error;
  wire [75:0] _stageClearVec_reqQueue_fifo_1_data_out;
  wire        _stageClearVec_WaitReadQueue_fifo_empty;
  wire        _stageClearVec_WaitReadQueue_fifo_full;
  wire        _stageClearVec_WaitReadQueue_fifo_error;
  wire [75:0] _stageClearVec_WaitReadQueue_fifo_data_out;
  wire        _stageClearVec_reqQueue_fifo_empty;
  wire        _stageClearVec_reqQueue_fifo_full;
  wire        _stageClearVec_reqQueue_fifo_error;
  wire [75:0] _stageClearVec_reqQueue_fifo_data_out;
  wire        stageClearVec_WaitReadQueue_7_almostFull;
  wire        stageClearVec_WaitReadQueue_7_almostEmpty;
  wire        stageClearVec_reqQueue_7_almostFull;
  wire        stageClearVec_reqQueue_7_almostEmpty;
  wire        stageClearVec_WaitReadQueue_6_almostFull;
  wire        stageClearVec_WaitReadQueue_6_almostEmpty;
  wire        stageClearVec_reqQueue_6_almostFull;
  wire        stageClearVec_reqQueue_6_almostEmpty;
  wire        stageClearVec_WaitReadQueue_5_almostFull;
  wire        stageClearVec_WaitReadQueue_5_almostEmpty;
  wire        stageClearVec_reqQueue_5_almostFull;
  wire        stageClearVec_reqQueue_5_almostEmpty;
  wire        stageClearVec_WaitReadQueue_4_almostFull;
  wire        stageClearVec_WaitReadQueue_4_almostEmpty;
  wire        stageClearVec_reqQueue_4_almostFull;
  wire        stageClearVec_reqQueue_4_almostEmpty;
  wire        stageClearVec_WaitReadQueue_3_almostFull;
  wire        stageClearVec_WaitReadQueue_3_almostEmpty;
  wire        stageClearVec_reqQueue_3_almostFull;
  wire        stageClearVec_reqQueue_3_almostEmpty;
  wire        stageClearVec_WaitReadQueue_2_almostFull;
  wire        stageClearVec_WaitReadQueue_2_almostEmpty;
  wire        stageClearVec_reqQueue_2_almostFull;
  wire        stageClearVec_reqQueue_2_almostEmpty;
  wire        stageClearVec_WaitReadQueue_1_almostFull;
  wire        stageClearVec_WaitReadQueue_1_almostEmpty;
  wire        stageClearVec_reqQueue_1_almostFull;
  wire        stageClearVec_reqQueue_1_almostEmpty;
  wire        stageClearVec_WaitReadQueue_almostFull;
  wire        stageClearVec_WaitReadQueue_almostEmpty;
  wire        stageClearVec_reqQueue_almostFull;
  wire        stageClearVec_reqQueue_almostEmpty;
  wire        stageClearVec_reqQueue_7_deq_bits_ffoByOther;
  wire [6:0]  stageClearVec_reqQueue_7_deq_bits_groupCounter;
  wire [3:0]  stageClearVec_reqQueue_7_deq_bits_mask;
  wire [31:0] stageClearVec_reqQueue_7_deq_bits_bitMask;
  wire [31:0] stageClearVec_reqQueue_7_deq_bits_data;
  wire        stageClearVec_reqQueue_6_deq_bits_ffoByOther;
  wire [6:0]  stageClearVec_reqQueue_6_deq_bits_groupCounter;
  wire [3:0]  stageClearVec_reqQueue_6_deq_bits_mask;
  wire [31:0] stageClearVec_reqQueue_6_deq_bits_bitMask;
  wire [31:0] stageClearVec_reqQueue_6_deq_bits_data;
  wire        stageClearVec_reqQueue_5_deq_bits_ffoByOther;
  wire [6:0]  stageClearVec_reqQueue_5_deq_bits_groupCounter;
  wire [3:0]  stageClearVec_reqQueue_5_deq_bits_mask;
  wire [31:0] stageClearVec_reqQueue_5_deq_bits_bitMask;
  wire [31:0] stageClearVec_reqQueue_5_deq_bits_data;
  wire        stageClearVec_reqQueue_4_deq_bits_ffoByOther;
  wire [6:0]  stageClearVec_reqQueue_4_deq_bits_groupCounter;
  wire [3:0]  stageClearVec_reqQueue_4_deq_bits_mask;
  wire [31:0] stageClearVec_reqQueue_4_deq_bits_bitMask;
  wire [31:0] stageClearVec_reqQueue_4_deq_bits_data;
  wire        stageClearVec_reqQueue_3_deq_bits_ffoByOther;
  wire [6:0]  stageClearVec_reqQueue_3_deq_bits_groupCounter;
  wire [3:0]  stageClearVec_reqQueue_3_deq_bits_mask;
  wire [31:0] stageClearVec_reqQueue_3_deq_bits_bitMask;
  wire [31:0] stageClearVec_reqQueue_3_deq_bits_data;
  wire        stageClearVec_reqQueue_2_deq_bits_ffoByOther;
  wire [6:0]  stageClearVec_reqQueue_2_deq_bits_groupCounter;
  wire [3:0]  stageClearVec_reqQueue_2_deq_bits_mask;
  wire [31:0] stageClearVec_reqQueue_2_deq_bits_bitMask;
  wire [31:0] stageClearVec_reqQueue_2_deq_bits_data;
  wire        stageClearVec_reqQueue_1_deq_bits_ffoByOther;
  wire [6:0]  stageClearVec_reqQueue_1_deq_bits_groupCounter;
  wire [3:0]  stageClearVec_reqQueue_1_deq_bits_mask;
  wire [31:0] stageClearVec_reqQueue_1_deq_bits_bitMask;
  wire [31:0] stageClearVec_reqQueue_1_deq_bits_data;
  wire        stageClearVec_reqQueue_deq_bits_ffoByOther;
  wire [6:0]  stageClearVec_reqQueue_deq_bits_groupCounter;
  wire [3:0]  stageClearVec_reqQueue_deq_bits_mask;
  wire [31:0] stageClearVec_reqQueue_deq_bits_bitMask;
  wire [31:0] stageClearVec_reqQueue_deq_bits_data;
  wire        in_0_valid_0 = in_0_valid;
  wire [31:0] in_0_bits_data_0 = in_0_bits_data;
  wire [31:0] in_0_bits_bitMask_0 = in_0_bits_bitMask;
  wire [3:0]  in_0_bits_mask_0 = in_0_bits_mask;
  wire [6:0]  in_0_bits_groupCounter_0 = in_0_bits_groupCounter;
  wire        in_0_bits_ffoByOther_0 = in_0_bits_ffoByOther;
  wire        in_1_valid_0 = in_1_valid;
  wire [31:0] in_1_bits_data_0 = in_1_bits_data;
  wire [31:0] in_1_bits_bitMask_0 = in_1_bits_bitMask;
  wire [3:0]  in_1_bits_mask_0 = in_1_bits_mask;
  wire [6:0]  in_1_bits_groupCounter_0 = in_1_bits_groupCounter;
  wire        in_1_bits_ffoByOther_0 = in_1_bits_ffoByOther;
  wire        in_2_valid_0 = in_2_valid;
  wire [31:0] in_2_bits_data_0 = in_2_bits_data;
  wire [31:0] in_2_bits_bitMask_0 = in_2_bits_bitMask;
  wire [3:0]  in_2_bits_mask_0 = in_2_bits_mask;
  wire [6:0]  in_2_bits_groupCounter_0 = in_2_bits_groupCounter;
  wire        in_2_bits_ffoByOther_0 = in_2_bits_ffoByOther;
  wire        in_3_valid_0 = in_3_valid;
  wire [31:0] in_3_bits_data_0 = in_3_bits_data;
  wire [31:0] in_3_bits_bitMask_0 = in_3_bits_bitMask;
  wire [3:0]  in_3_bits_mask_0 = in_3_bits_mask;
  wire [6:0]  in_3_bits_groupCounter_0 = in_3_bits_groupCounter;
  wire        in_3_bits_ffoByOther_0 = in_3_bits_ffoByOther;
  wire        in_4_valid_0 = in_4_valid;
  wire [31:0] in_4_bits_data_0 = in_4_bits_data;
  wire [31:0] in_4_bits_bitMask_0 = in_4_bits_bitMask;
  wire [3:0]  in_4_bits_mask_0 = in_4_bits_mask;
  wire [6:0]  in_4_bits_groupCounter_0 = in_4_bits_groupCounter;
  wire        in_4_bits_ffoByOther_0 = in_4_bits_ffoByOther;
  wire        in_5_valid_0 = in_5_valid;
  wire [31:0] in_5_bits_data_0 = in_5_bits_data;
  wire [31:0] in_5_bits_bitMask_0 = in_5_bits_bitMask;
  wire [3:0]  in_5_bits_mask_0 = in_5_bits_mask;
  wire [6:0]  in_5_bits_groupCounter_0 = in_5_bits_groupCounter;
  wire        in_5_bits_ffoByOther_0 = in_5_bits_ffoByOther;
  wire        in_6_valid_0 = in_6_valid;
  wire [31:0] in_6_bits_data_0 = in_6_bits_data;
  wire [31:0] in_6_bits_bitMask_0 = in_6_bits_bitMask;
  wire [3:0]  in_6_bits_mask_0 = in_6_bits_mask;
  wire [6:0]  in_6_bits_groupCounter_0 = in_6_bits_groupCounter;
  wire        in_6_bits_ffoByOther_0 = in_6_bits_ffoByOther;
  wire        in_7_valid_0 = in_7_valid;
  wire [31:0] in_7_bits_data_0 = in_7_bits_data;
  wire [31:0] in_7_bits_bitMask_0 = in_7_bits_bitMask;
  wire [3:0]  in_7_bits_mask_0 = in_7_bits_mask;
  wire [6:0]  in_7_bits_groupCounter_0 = in_7_bits_groupCounter;
  wire        in_7_bits_ffoByOther_0 = in_7_bits_ffoByOther;
  wire        out_0_ready_0 = out_0_ready;
  wire        out_1_ready_0 = out_1_ready;
  wire        out_2_ready_0 = out_2_ready;
  wire        out_3_ready_0 = out_3_ready;
  wire        out_4_ready_0 = out_4_ready;
  wire        out_5_ready_0 = out_5_ready;
  wire        out_6_ready_0 = out_6_ready;
  wire        out_7_ready_0 = out_7_ready;
  wire        readChannel_0_ready_0 = readChannel_0_ready;
  wire        readChannel_1_ready_0 = readChannel_1_ready;
  wire        readChannel_2_ready_0 = readChannel_2_ready;
  wire        readChannel_3_ready_0 = readChannel_3_ready;
  wire        readChannel_4_ready_0 = readChannel_4_ready;
  wire        readChannel_5_ready_0 = readChannel_5_ready;
  wire        readChannel_6_ready_0 = readChannel_6_ready;
  wire        readChannel_7_ready_0 = readChannel_7_ready;
  wire [1:0]  readChannel_0_bits_readSource = 2'h0;
  wire [1:0]  readChannel_1_bits_readSource = 2'h0;
  wire [1:0]  readChannel_2_bits_readSource = 2'h0;
  wire [1:0]  readChannel_3_bits_readSource = 2'h0;
  wire [1:0]  readChannel_4_bits_readSource = 2'h0;
  wire [1:0]  readChannel_5_bits_readSource = 2'h0;
  wire [1:0]  readChannel_6_bits_readSource = 2'h0;
  wire [1:0]  readChannel_7_bits_readSource = 2'h0;
  wire [2:0]  out_0_bits_index = 3'h0;
  wire [2:0]  out_1_bits_index = 3'h0;
  wire [2:0]  out_2_bits_index = 3'h0;
  wire [2:0]  out_3_bits_index = 3'h0;
  wire [2:0]  out_4_bits_index = 3'h0;
  wire [2:0]  out_5_bits_index = 3'h0;
  wire [2:0]  out_6_bits_index = 3'h0;
  wire [2:0]  out_7_bits_index = 3'h0;
  wire [2:0]  readChannel_0_bits_instructionIndex = 3'h0;
  wire [2:0]  readChannel_1_bits_instructionIndex = 3'h0;
  wire [2:0]  readChannel_2_bits_instructionIndex = 3'h0;
  wire [2:0]  readChannel_3_bits_instructionIndex = 3'h0;
  wire [2:0]  readChannel_4_bits_instructionIndex = 3'h0;
  wire [2:0]  readChannel_5_bits_instructionIndex = 3'h0;
  wire [2:0]  readChannel_6_bits_instructionIndex = 3'h0;
  wire [2:0]  readChannel_7_bits_instructionIndex = 3'h0;
  wire [4:0]  out_0_bits_writeData_vd = 5'h0;
  wire [4:0]  out_1_bits_writeData_vd = 5'h0;
  wire [4:0]  out_2_bits_writeData_vd = 5'h0;
  wire [4:0]  out_3_bits_writeData_vd = 5'h0;
  wire [4:0]  out_4_bits_writeData_vd = 5'h0;
  wire [4:0]  out_5_bits_writeData_vd = 5'h0;
  wire [4:0]  out_6_bits_writeData_vd = 5'h0;
  wire [4:0]  out_7_bits_writeData_vd = 5'h0;
  wire        stageClearVec_reqQueue_enq_ready;
  wire        stageClearVec_reqQueue_enq_valid = in_0_valid_0;
  wire [31:0] stageClearVec_reqQueue_enq_bits_data = in_0_bits_data_0;
  wire [31:0] stageClearVec_reqQueue_enq_bits_bitMask = in_0_bits_bitMask_0;
  wire [3:0]  stageClearVec_reqQueue_enq_bits_mask = in_0_bits_mask_0;
  wire [6:0]  stageClearVec_reqQueue_enq_bits_groupCounter = in_0_bits_groupCounter_0;
  wire        stageClearVec_reqQueue_enq_bits_ffoByOther = in_0_bits_ffoByOther_0;
  wire        stageClearVec_reqQueue_1_enq_ready;
  wire        stageClearVec_reqQueue_1_enq_valid = in_1_valid_0;
  wire [31:0] stageClearVec_reqQueue_1_enq_bits_data = in_1_bits_data_0;
  wire [31:0] stageClearVec_reqQueue_1_enq_bits_bitMask = in_1_bits_bitMask_0;
  wire [3:0]  stageClearVec_reqQueue_1_enq_bits_mask = in_1_bits_mask_0;
  wire [6:0]  stageClearVec_reqQueue_1_enq_bits_groupCounter = in_1_bits_groupCounter_0;
  wire        stageClearVec_reqQueue_1_enq_bits_ffoByOther = in_1_bits_ffoByOther_0;
  wire        stageClearVec_reqQueue_2_enq_ready;
  wire        stageClearVec_reqQueue_2_enq_valid = in_2_valid_0;
  wire [31:0] stageClearVec_reqQueue_2_enq_bits_data = in_2_bits_data_0;
  wire [31:0] stageClearVec_reqQueue_2_enq_bits_bitMask = in_2_bits_bitMask_0;
  wire [3:0]  stageClearVec_reqQueue_2_enq_bits_mask = in_2_bits_mask_0;
  wire [6:0]  stageClearVec_reqQueue_2_enq_bits_groupCounter = in_2_bits_groupCounter_0;
  wire        stageClearVec_reqQueue_2_enq_bits_ffoByOther = in_2_bits_ffoByOther_0;
  wire        stageClearVec_reqQueue_3_enq_ready;
  wire        stageClearVec_reqQueue_3_enq_valid = in_3_valid_0;
  wire [31:0] stageClearVec_reqQueue_3_enq_bits_data = in_3_bits_data_0;
  wire [31:0] stageClearVec_reqQueue_3_enq_bits_bitMask = in_3_bits_bitMask_0;
  wire [3:0]  stageClearVec_reqQueue_3_enq_bits_mask = in_3_bits_mask_0;
  wire [6:0]  stageClearVec_reqQueue_3_enq_bits_groupCounter = in_3_bits_groupCounter_0;
  wire        stageClearVec_reqQueue_3_enq_bits_ffoByOther = in_3_bits_ffoByOther_0;
  wire        stageClearVec_reqQueue_4_enq_ready;
  wire        stageClearVec_reqQueue_4_enq_valid = in_4_valid_0;
  wire [31:0] stageClearVec_reqQueue_4_enq_bits_data = in_4_bits_data_0;
  wire [31:0] stageClearVec_reqQueue_4_enq_bits_bitMask = in_4_bits_bitMask_0;
  wire [3:0]  stageClearVec_reqQueue_4_enq_bits_mask = in_4_bits_mask_0;
  wire [6:0]  stageClearVec_reqQueue_4_enq_bits_groupCounter = in_4_bits_groupCounter_0;
  wire        stageClearVec_reqQueue_4_enq_bits_ffoByOther = in_4_bits_ffoByOther_0;
  wire        stageClearVec_reqQueue_5_enq_ready;
  wire        stageClearVec_reqQueue_5_enq_valid = in_5_valid_0;
  wire [31:0] stageClearVec_reqQueue_5_enq_bits_data = in_5_bits_data_0;
  wire [31:0] stageClearVec_reqQueue_5_enq_bits_bitMask = in_5_bits_bitMask_0;
  wire [3:0]  stageClearVec_reqQueue_5_enq_bits_mask = in_5_bits_mask_0;
  wire [6:0]  stageClearVec_reqQueue_5_enq_bits_groupCounter = in_5_bits_groupCounter_0;
  wire        stageClearVec_reqQueue_5_enq_bits_ffoByOther = in_5_bits_ffoByOther_0;
  wire        stageClearVec_reqQueue_6_enq_ready;
  wire        stageClearVec_reqQueue_6_enq_valid = in_6_valid_0;
  wire [31:0] stageClearVec_reqQueue_6_enq_bits_data = in_6_bits_data_0;
  wire [31:0] stageClearVec_reqQueue_6_enq_bits_bitMask = in_6_bits_bitMask_0;
  wire [3:0]  stageClearVec_reqQueue_6_enq_bits_mask = in_6_bits_mask_0;
  wire [6:0]  stageClearVec_reqQueue_6_enq_bits_groupCounter = in_6_bits_groupCounter_0;
  wire        stageClearVec_reqQueue_6_enq_bits_ffoByOther = in_6_bits_ffoByOther_0;
  wire        stageClearVec_reqQueue_7_enq_ready;
  wire        stageClearVec_reqQueue_7_enq_valid = in_7_valid_0;
  wire [31:0] stageClearVec_reqQueue_7_enq_bits_data = in_7_bits_data_0;
  wire [31:0] stageClearVec_reqQueue_7_enq_bits_bitMask = in_7_bits_bitMask_0;
  wire [3:0]  stageClearVec_reqQueue_7_enq_bits_mask = in_7_bits_mask_0;
  wire [6:0]  stageClearVec_reqQueue_7_enq_bits_groupCounter = in_7_bits_groupCounter_0;
  wire        stageClearVec_reqQueue_7_enq_bits_ffoByOther = in_7_bits_ffoByOther_0;
  wire        stageClearVec_WaitReadQueue_deq_bits_ffoByOther;
  wire [6:0]  stageClearVec_WaitReadQueue_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_1_deq_bits_ffoByOther;
  wire [6:0]  stageClearVec_WaitReadQueue_1_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_2_deq_bits_ffoByOther;
  wire [6:0]  stageClearVec_WaitReadQueue_2_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_3_deq_bits_ffoByOther;
  wire [6:0]  stageClearVec_WaitReadQueue_3_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_4_deq_bits_ffoByOther;
  wire [6:0]  stageClearVec_WaitReadQueue_4_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_5_deq_bits_ffoByOther;
  wire [6:0]  stageClearVec_WaitReadQueue_5_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_6_deq_bits_ffoByOther;
  wire [6:0]  stageClearVec_WaitReadQueue_6_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_7_deq_bits_ffoByOther;
  wire [6:0]  stageClearVec_WaitReadQueue_7_deq_bits_groupCounter;
  wire        in_0_ready_0 = stageClearVec_reqQueue_enq_ready;
  wire        stageClearVec_reqQueue_deq_valid;
  assign stageClearVec_reqQueue_deq_valid = ~_stageClearVec_reqQueue_fifo_empty;
  wire [31:0] stageClearVec_reqQueue_dataOut_data;
  wire [31:0] stageClearVec_reqQueue_dataOut_bitMask;
  wire [31:0] stageClearVec_WaitReadQueue_enq_bits_data = stageClearVec_reqQueue_deq_bits_data;
  wire [3:0]  stageClearVec_reqQueue_dataOut_mask;
  wire [31:0] stageClearVec_WaitReadQueue_enq_bits_bitMask = stageClearVec_reqQueue_deq_bits_bitMask;
  wire [6:0]  stageClearVec_reqQueue_dataOut_groupCounter;
  wire [3:0]  stageClearVec_WaitReadQueue_enq_bits_mask = stageClearVec_reqQueue_deq_bits_mask;
  wire        stageClearVec_reqQueue_dataOut_ffoByOther;
  wire [6:0]  stageClearVec_WaitReadQueue_enq_bits_groupCounter = stageClearVec_reqQueue_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_enq_bits_ffoByOther = stageClearVec_reqQueue_deq_bits_ffoByOther;
  wire [7:0]  stageClearVec_reqQueue_dataIn_lo = {stageClearVec_reqQueue_enq_bits_groupCounter, stageClearVec_reqQueue_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_reqQueue_dataIn_hi_hi = {stageClearVec_reqQueue_enq_bits_data, stageClearVec_reqQueue_enq_bits_bitMask};
  wire [67:0] stageClearVec_reqQueue_dataIn_hi = {stageClearVec_reqQueue_dataIn_hi_hi, stageClearVec_reqQueue_enq_bits_mask};
  wire [75:0] stageClearVec_reqQueue_dataIn = {stageClearVec_reqQueue_dataIn_hi, stageClearVec_reqQueue_dataIn_lo};
  assign stageClearVec_reqQueue_dataOut_ffoByOther = _stageClearVec_reqQueue_fifo_data_out[0];
  assign stageClearVec_reqQueue_dataOut_groupCounter = _stageClearVec_reqQueue_fifo_data_out[7:1];
  assign stageClearVec_reqQueue_dataOut_mask = _stageClearVec_reqQueue_fifo_data_out[11:8];
  assign stageClearVec_reqQueue_dataOut_bitMask = _stageClearVec_reqQueue_fifo_data_out[43:12];
  assign stageClearVec_reqQueue_dataOut_data = _stageClearVec_reqQueue_fifo_data_out[75:44];
  assign stageClearVec_reqQueue_deq_bits_data = stageClearVec_reqQueue_dataOut_data;
  assign stageClearVec_reqQueue_deq_bits_bitMask = stageClearVec_reqQueue_dataOut_bitMask;
  assign stageClearVec_reqQueue_deq_bits_mask = stageClearVec_reqQueue_dataOut_mask;
  assign stageClearVec_reqQueue_deq_bits_groupCounter = stageClearVec_reqQueue_dataOut_groupCounter;
  assign stageClearVec_reqQueue_deq_bits_ffoByOther = stageClearVec_reqQueue_dataOut_ffoByOther;
  assign stageClearVec_reqQueue_enq_ready = ~_stageClearVec_reqQueue_fifo_full;
  wire        stageClearVec_reqQueue_deq_ready;
  wire        stageClearVec_WaitReadQueue_deq_valid;
  assign stageClearVec_WaitReadQueue_deq_valid = ~_stageClearVec_WaitReadQueue_fifo_empty;
  wire [31:0] stageClearVec_WaitReadQueue_dataOut_data;
  wire [31:0] stageClearVec_WaitReadQueue_dataOut_bitMask;
  wire [3:0]  stageClearVec_WaitReadQueue_dataOut_mask;
  wire [6:0]  stageClearVec_WaitReadQueue_dataOut_groupCounter;
  wire [6:0]  out_0_bits_writeData_groupCounter_0 = stageClearVec_WaitReadQueue_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_dataOut_ffoByOther;
  wire        out_0_bits_ffoByOther_0 = stageClearVec_WaitReadQueue_deq_bits_ffoByOther;
  wire [7:0]  stageClearVec_WaitReadQueue_dataIn_lo = {stageClearVec_WaitReadQueue_enq_bits_groupCounter, stageClearVec_WaitReadQueue_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_WaitReadQueue_dataIn_hi_hi = {stageClearVec_WaitReadQueue_enq_bits_data, stageClearVec_WaitReadQueue_enq_bits_bitMask};
  wire [67:0] stageClearVec_WaitReadQueue_dataIn_hi = {stageClearVec_WaitReadQueue_dataIn_hi_hi, stageClearVec_WaitReadQueue_enq_bits_mask};
  wire [75:0] stageClearVec_WaitReadQueue_dataIn = {stageClearVec_WaitReadQueue_dataIn_hi, stageClearVec_WaitReadQueue_dataIn_lo};
  assign stageClearVec_WaitReadQueue_dataOut_ffoByOther = _stageClearVec_WaitReadQueue_fifo_data_out[0];
  assign stageClearVec_WaitReadQueue_dataOut_groupCounter = _stageClearVec_WaitReadQueue_fifo_data_out[7:1];
  assign stageClearVec_WaitReadQueue_dataOut_mask = _stageClearVec_WaitReadQueue_fifo_data_out[11:8];
  assign stageClearVec_WaitReadQueue_dataOut_bitMask = _stageClearVec_WaitReadQueue_fifo_data_out[43:12];
  assign stageClearVec_WaitReadQueue_dataOut_data = _stageClearVec_WaitReadQueue_fifo_data_out[75:44];
  wire [31:0] stageClearVec_WaitReadQueue_deq_bits_data = stageClearVec_WaitReadQueue_dataOut_data;
  wire [31:0] stageClearVec_WaitReadQueue_deq_bits_bitMask = stageClearVec_WaitReadQueue_dataOut_bitMask;
  wire [3:0]  stageClearVec_WaitReadQueue_deq_bits_mask = stageClearVec_WaitReadQueue_dataOut_mask;
  assign stageClearVec_WaitReadQueue_deq_bits_groupCounter = stageClearVec_WaitReadQueue_dataOut_groupCounter;
  assign stageClearVec_WaitReadQueue_deq_bits_ffoByOther = stageClearVec_WaitReadQueue_dataOut_ffoByOther;
  wire        stageClearVec_WaitReadQueue_enq_ready = ~_stageClearVec_WaitReadQueue_fifo_full;
  wire        stageClearVec_WaitReadQueue_enq_valid;
  wire        stageClearVec_WaitReadQueue_deq_ready;
  wire        stageClearVec_readReady = ~needWAR | readChannel_0_ready_0;
  assign stageClearVec_WaitReadQueue_enq_valid = stageClearVec_reqQueue_deq_valid & stageClearVec_readReady;
  assign stageClearVec_reqQueue_deq_ready = stageClearVec_WaitReadQueue_enq_ready & stageClearVec_readReady;
  wire        readChannel_0_valid_0 = stageClearVec_reqQueue_deq_valid & needWAR & stageClearVec_WaitReadQueue_enq_ready;
  wire [4:0]  readChannel_0_bits_vs_0 = vd + {1'h0, stageClearVec_reqQueue_deq_bits_groupCounter[6:3]};
  wire [2:0]  readChannel_0_bits_offset_0 = stageClearVec_reqQueue_deq_bits_groupCounter[2:0];
  wire        stageClearVec_readResultValid = ~needWAR | readResult_0_valid;
  wire [31:0] stageClearVec_WARData = stageClearVec_WaitReadQueue_deq_bits_data & stageClearVec_WaitReadQueue_deq_bits_bitMask | readResult_0_bits & ~stageClearVec_WaitReadQueue_deq_bits_bitMask;
  wire        out_0_valid_0 = stageClearVec_WaitReadQueue_deq_valid & stageClearVec_readResultValid;
  assign stageClearVec_WaitReadQueue_deq_ready = out_0_ready_0 & stageClearVec_readResultValid;
  wire [31:0] out_0_bits_writeData_data_0 = needWAR ? stageClearVec_WARData : stageClearVec_WaitReadQueue_deq_bits_data;
  wire [3:0]  out_0_bits_writeData_mask_0 = needWAR ? 4'hF : stageClearVec_WaitReadQueue_deq_bits_mask;
  reg  [2:0]  stageClearVec_counter;
  wire        _stageClearVec_T = in_0_ready_0 & in_0_valid_0;
  wire [2:0]  stageClearVec_counterChange = _stageClearVec_T ? 3'h1 : 3'h7;
  wire        stageClearVec_0 = stageClearVec_counter == 3'h0;
  wire        in_1_ready_0 = stageClearVec_reqQueue_1_enq_ready;
  wire        stageClearVec_reqQueue_1_deq_valid;
  assign stageClearVec_reqQueue_1_deq_valid = ~_stageClearVec_reqQueue_fifo_1_empty;
  wire [31:0] stageClearVec_reqQueue_dataOut_1_data;
  wire [31:0] stageClearVec_reqQueue_dataOut_1_bitMask;
  wire [31:0] stageClearVec_WaitReadQueue_1_enq_bits_data = stageClearVec_reqQueue_1_deq_bits_data;
  wire [3:0]  stageClearVec_reqQueue_dataOut_1_mask;
  wire [31:0] stageClearVec_WaitReadQueue_1_enq_bits_bitMask = stageClearVec_reqQueue_1_deq_bits_bitMask;
  wire [6:0]  stageClearVec_reqQueue_dataOut_1_groupCounter;
  wire [3:0]  stageClearVec_WaitReadQueue_1_enq_bits_mask = stageClearVec_reqQueue_1_deq_bits_mask;
  wire        stageClearVec_reqQueue_dataOut_1_ffoByOther;
  wire [6:0]  stageClearVec_WaitReadQueue_1_enq_bits_groupCounter = stageClearVec_reqQueue_1_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_1_enq_bits_ffoByOther = stageClearVec_reqQueue_1_deq_bits_ffoByOther;
  wire [7:0]  stageClearVec_reqQueue_dataIn_lo_1 = {stageClearVec_reqQueue_1_enq_bits_groupCounter, stageClearVec_reqQueue_1_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_reqQueue_dataIn_hi_hi_1 = {stageClearVec_reqQueue_1_enq_bits_data, stageClearVec_reqQueue_1_enq_bits_bitMask};
  wire [67:0] stageClearVec_reqQueue_dataIn_hi_1 = {stageClearVec_reqQueue_dataIn_hi_hi_1, stageClearVec_reqQueue_1_enq_bits_mask};
  wire [75:0] stageClearVec_reqQueue_dataIn_1 = {stageClearVec_reqQueue_dataIn_hi_1, stageClearVec_reqQueue_dataIn_lo_1};
  assign stageClearVec_reqQueue_dataOut_1_ffoByOther = _stageClearVec_reqQueue_fifo_1_data_out[0];
  assign stageClearVec_reqQueue_dataOut_1_groupCounter = _stageClearVec_reqQueue_fifo_1_data_out[7:1];
  assign stageClearVec_reqQueue_dataOut_1_mask = _stageClearVec_reqQueue_fifo_1_data_out[11:8];
  assign stageClearVec_reqQueue_dataOut_1_bitMask = _stageClearVec_reqQueue_fifo_1_data_out[43:12];
  assign stageClearVec_reqQueue_dataOut_1_data = _stageClearVec_reqQueue_fifo_1_data_out[75:44];
  assign stageClearVec_reqQueue_1_deq_bits_data = stageClearVec_reqQueue_dataOut_1_data;
  assign stageClearVec_reqQueue_1_deq_bits_bitMask = stageClearVec_reqQueue_dataOut_1_bitMask;
  assign stageClearVec_reqQueue_1_deq_bits_mask = stageClearVec_reqQueue_dataOut_1_mask;
  assign stageClearVec_reqQueue_1_deq_bits_groupCounter = stageClearVec_reqQueue_dataOut_1_groupCounter;
  assign stageClearVec_reqQueue_1_deq_bits_ffoByOther = stageClearVec_reqQueue_dataOut_1_ffoByOther;
  assign stageClearVec_reqQueue_1_enq_ready = ~_stageClearVec_reqQueue_fifo_1_full;
  wire        stageClearVec_reqQueue_1_deq_ready;
  wire        stageClearVec_WaitReadQueue_1_deq_valid;
  assign stageClearVec_WaitReadQueue_1_deq_valid = ~_stageClearVec_WaitReadQueue_fifo_1_empty;
  wire [31:0] stageClearVec_WaitReadQueue_dataOut_1_data;
  wire [31:0] stageClearVec_WaitReadQueue_dataOut_1_bitMask;
  wire [3:0]  stageClearVec_WaitReadQueue_dataOut_1_mask;
  wire [6:0]  stageClearVec_WaitReadQueue_dataOut_1_groupCounter;
  wire [6:0]  out_1_bits_writeData_groupCounter_0 = stageClearVec_WaitReadQueue_1_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_dataOut_1_ffoByOther;
  wire        out_1_bits_ffoByOther_0 = stageClearVec_WaitReadQueue_1_deq_bits_ffoByOther;
  wire [7:0]  stageClearVec_WaitReadQueue_dataIn_lo_1 = {stageClearVec_WaitReadQueue_1_enq_bits_groupCounter, stageClearVec_WaitReadQueue_1_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_WaitReadQueue_dataIn_hi_hi_1 = {stageClearVec_WaitReadQueue_1_enq_bits_data, stageClearVec_WaitReadQueue_1_enq_bits_bitMask};
  wire [67:0] stageClearVec_WaitReadQueue_dataIn_hi_1 = {stageClearVec_WaitReadQueue_dataIn_hi_hi_1, stageClearVec_WaitReadQueue_1_enq_bits_mask};
  wire [75:0] stageClearVec_WaitReadQueue_dataIn_1 = {stageClearVec_WaitReadQueue_dataIn_hi_1, stageClearVec_WaitReadQueue_dataIn_lo_1};
  assign stageClearVec_WaitReadQueue_dataOut_1_ffoByOther = _stageClearVec_WaitReadQueue_fifo_1_data_out[0];
  assign stageClearVec_WaitReadQueue_dataOut_1_groupCounter = _stageClearVec_WaitReadQueue_fifo_1_data_out[7:1];
  assign stageClearVec_WaitReadQueue_dataOut_1_mask = _stageClearVec_WaitReadQueue_fifo_1_data_out[11:8];
  assign stageClearVec_WaitReadQueue_dataOut_1_bitMask = _stageClearVec_WaitReadQueue_fifo_1_data_out[43:12];
  assign stageClearVec_WaitReadQueue_dataOut_1_data = _stageClearVec_WaitReadQueue_fifo_1_data_out[75:44];
  wire [31:0] stageClearVec_WaitReadQueue_1_deq_bits_data = stageClearVec_WaitReadQueue_dataOut_1_data;
  wire [31:0] stageClearVec_WaitReadQueue_1_deq_bits_bitMask = stageClearVec_WaitReadQueue_dataOut_1_bitMask;
  wire [3:0]  stageClearVec_WaitReadQueue_1_deq_bits_mask = stageClearVec_WaitReadQueue_dataOut_1_mask;
  assign stageClearVec_WaitReadQueue_1_deq_bits_groupCounter = stageClearVec_WaitReadQueue_dataOut_1_groupCounter;
  assign stageClearVec_WaitReadQueue_1_deq_bits_ffoByOther = stageClearVec_WaitReadQueue_dataOut_1_ffoByOther;
  wire        stageClearVec_WaitReadQueue_1_enq_ready = ~_stageClearVec_WaitReadQueue_fifo_1_full;
  wire        stageClearVec_WaitReadQueue_1_enq_valid;
  wire        stageClearVec_WaitReadQueue_1_deq_ready;
  wire        stageClearVec_readReady_1 = ~needWAR | readChannel_1_ready_0;
  assign stageClearVec_WaitReadQueue_1_enq_valid = stageClearVec_reqQueue_1_deq_valid & stageClearVec_readReady_1;
  assign stageClearVec_reqQueue_1_deq_ready = stageClearVec_WaitReadQueue_1_enq_ready & stageClearVec_readReady_1;
  wire        readChannel_1_valid_0 = stageClearVec_reqQueue_1_deq_valid & needWAR & stageClearVec_WaitReadQueue_1_enq_ready;
  wire [4:0]  readChannel_1_bits_vs_0 = vd + {1'h0, stageClearVec_reqQueue_1_deq_bits_groupCounter[6:3]};
  wire [2:0]  readChannel_1_bits_offset_0 = stageClearVec_reqQueue_1_deq_bits_groupCounter[2:0];
  wire        stageClearVec_readResultValid_1 = ~needWAR | readResult_1_valid;
  wire [31:0] stageClearVec_WARData_1 = stageClearVec_WaitReadQueue_1_deq_bits_data & stageClearVec_WaitReadQueue_1_deq_bits_bitMask | readResult_1_bits & ~stageClearVec_WaitReadQueue_1_deq_bits_bitMask;
  wire        out_1_valid_0 = stageClearVec_WaitReadQueue_1_deq_valid & stageClearVec_readResultValid_1;
  assign stageClearVec_WaitReadQueue_1_deq_ready = out_1_ready_0 & stageClearVec_readResultValid_1;
  wire [31:0] out_1_bits_writeData_data_0 = needWAR ? stageClearVec_WARData_1 : stageClearVec_WaitReadQueue_1_deq_bits_data;
  wire [3:0]  out_1_bits_writeData_mask_0 = needWAR ? 4'hF : stageClearVec_WaitReadQueue_1_deq_bits_mask;
  reg  [2:0]  stageClearVec_counter_1;
  wire        _stageClearVec_T_3 = in_1_ready_0 & in_1_valid_0;
  wire [2:0]  stageClearVec_counterChange_1 = _stageClearVec_T_3 ? 3'h1 : 3'h7;
  wire        stageClearVec_1 = stageClearVec_counter_1 == 3'h0;
  wire        in_2_ready_0 = stageClearVec_reqQueue_2_enq_ready;
  wire        stageClearVec_reqQueue_2_deq_valid;
  assign stageClearVec_reqQueue_2_deq_valid = ~_stageClearVec_reqQueue_fifo_2_empty;
  wire [31:0] stageClearVec_reqQueue_dataOut_2_data;
  wire [31:0] stageClearVec_reqQueue_dataOut_2_bitMask;
  wire [31:0] stageClearVec_WaitReadQueue_2_enq_bits_data = stageClearVec_reqQueue_2_deq_bits_data;
  wire [3:0]  stageClearVec_reqQueue_dataOut_2_mask;
  wire [31:0] stageClearVec_WaitReadQueue_2_enq_bits_bitMask = stageClearVec_reqQueue_2_deq_bits_bitMask;
  wire [6:0]  stageClearVec_reqQueue_dataOut_2_groupCounter;
  wire [3:0]  stageClearVec_WaitReadQueue_2_enq_bits_mask = stageClearVec_reqQueue_2_deq_bits_mask;
  wire        stageClearVec_reqQueue_dataOut_2_ffoByOther;
  wire [6:0]  stageClearVec_WaitReadQueue_2_enq_bits_groupCounter = stageClearVec_reqQueue_2_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_2_enq_bits_ffoByOther = stageClearVec_reqQueue_2_deq_bits_ffoByOther;
  wire [7:0]  stageClearVec_reqQueue_dataIn_lo_2 = {stageClearVec_reqQueue_2_enq_bits_groupCounter, stageClearVec_reqQueue_2_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_reqQueue_dataIn_hi_hi_2 = {stageClearVec_reqQueue_2_enq_bits_data, stageClearVec_reqQueue_2_enq_bits_bitMask};
  wire [67:0] stageClearVec_reqQueue_dataIn_hi_2 = {stageClearVec_reqQueue_dataIn_hi_hi_2, stageClearVec_reqQueue_2_enq_bits_mask};
  wire [75:0] stageClearVec_reqQueue_dataIn_2 = {stageClearVec_reqQueue_dataIn_hi_2, stageClearVec_reqQueue_dataIn_lo_2};
  assign stageClearVec_reqQueue_dataOut_2_ffoByOther = _stageClearVec_reqQueue_fifo_2_data_out[0];
  assign stageClearVec_reqQueue_dataOut_2_groupCounter = _stageClearVec_reqQueue_fifo_2_data_out[7:1];
  assign stageClearVec_reqQueue_dataOut_2_mask = _stageClearVec_reqQueue_fifo_2_data_out[11:8];
  assign stageClearVec_reqQueue_dataOut_2_bitMask = _stageClearVec_reqQueue_fifo_2_data_out[43:12];
  assign stageClearVec_reqQueue_dataOut_2_data = _stageClearVec_reqQueue_fifo_2_data_out[75:44];
  assign stageClearVec_reqQueue_2_deq_bits_data = stageClearVec_reqQueue_dataOut_2_data;
  assign stageClearVec_reqQueue_2_deq_bits_bitMask = stageClearVec_reqQueue_dataOut_2_bitMask;
  assign stageClearVec_reqQueue_2_deq_bits_mask = stageClearVec_reqQueue_dataOut_2_mask;
  assign stageClearVec_reqQueue_2_deq_bits_groupCounter = stageClearVec_reqQueue_dataOut_2_groupCounter;
  assign stageClearVec_reqQueue_2_deq_bits_ffoByOther = stageClearVec_reqQueue_dataOut_2_ffoByOther;
  assign stageClearVec_reqQueue_2_enq_ready = ~_stageClearVec_reqQueue_fifo_2_full;
  wire        stageClearVec_reqQueue_2_deq_ready;
  wire        stageClearVec_WaitReadQueue_2_deq_valid;
  assign stageClearVec_WaitReadQueue_2_deq_valid = ~_stageClearVec_WaitReadQueue_fifo_2_empty;
  wire [31:0] stageClearVec_WaitReadQueue_dataOut_2_data;
  wire [31:0] stageClearVec_WaitReadQueue_dataOut_2_bitMask;
  wire [3:0]  stageClearVec_WaitReadQueue_dataOut_2_mask;
  wire [6:0]  stageClearVec_WaitReadQueue_dataOut_2_groupCounter;
  wire [6:0]  out_2_bits_writeData_groupCounter_0 = stageClearVec_WaitReadQueue_2_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_dataOut_2_ffoByOther;
  wire        out_2_bits_ffoByOther_0 = stageClearVec_WaitReadQueue_2_deq_bits_ffoByOther;
  wire [7:0]  stageClearVec_WaitReadQueue_dataIn_lo_2 = {stageClearVec_WaitReadQueue_2_enq_bits_groupCounter, stageClearVec_WaitReadQueue_2_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_WaitReadQueue_dataIn_hi_hi_2 = {stageClearVec_WaitReadQueue_2_enq_bits_data, stageClearVec_WaitReadQueue_2_enq_bits_bitMask};
  wire [67:0] stageClearVec_WaitReadQueue_dataIn_hi_2 = {stageClearVec_WaitReadQueue_dataIn_hi_hi_2, stageClearVec_WaitReadQueue_2_enq_bits_mask};
  wire [75:0] stageClearVec_WaitReadQueue_dataIn_2 = {stageClearVec_WaitReadQueue_dataIn_hi_2, stageClearVec_WaitReadQueue_dataIn_lo_2};
  assign stageClearVec_WaitReadQueue_dataOut_2_ffoByOther = _stageClearVec_WaitReadQueue_fifo_2_data_out[0];
  assign stageClearVec_WaitReadQueue_dataOut_2_groupCounter = _stageClearVec_WaitReadQueue_fifo_2_data_out[7:1];
  assign stageClearVec_WaitReadQueue_dataOut_2_mask = _stageClearVec_WaitReadQueue_fifo_2_data_out[11:8];
  assign stageClearVec_WaitReadQueue_dataOut_2_bitMask = _stageClearVec_WaitReadQueue_fifo_2_data_out[43:12];
  assign stageClearVec_WaitReadQueue_dataOut_2_data = _stageClearVec_WaitReadQueue_fifo_2_data_out[75:44];
  wire [31:0] stageClearVec_WaitReadQueue_2_deq_bits_data = stageClearVec_WaitReadQueue_dataOut_2_data;
  wire [31:0] stageClearVec_WaitReadQueue_2_deq_bits_bitMask = stageClearVec_WaitReadQueue_dataOut_2_bitMask;
  wire [3:0]  stageClearVec_WaitReadQueue_2_deq_bits_mask = stageClearVec_WaitReadQueue_dataOut_2_mask;
  assign stageClearVec_WaitReadQueue_2_deq_bits_groupCounter = stageClearVec_WaitReadQueue_dataOut_2_groupCounter;
  assign stageClearVec_WaitReadQueue_2_deq_bits_ffoByOther = stageClearVec_WaitReadQueue_dataOut_2_ffoByOther;
  wire        stageClearVec_WaitReadQueue_2_enq_ready = ~_stageClearVec_WaitReadQueue_fifo_2_full;
  wire        stageClearVec_WaitReadQueue_2_enq_valid;
  wire        stageClearVec_WaitReadQueue_2_deq_ready;
  wire        stageClearVec_readReady_2 = ~needWAR | readChannel_2_ready_0;
  assign stageClearVec_WaitReadQueue_2_enq_valid = stageClearVec_reqQueue_2_deq_valid & stageClearVec_readReady_2;
  assign stageClearVec_reqQueue_2_deq_ready = stageClearVec_WaitReadQueue_2_enq_ready & stageClearVec_readReady_2;
  wire        readChannel_2_valid_0 = stageClearVec_reqQueue_2_deq_valid & needWAR & stageClearVec_WaitReadQueue_2_enq_ready;
  wire [4:0]  readChannel_2_bits_vs_0 = vd + {1'h0, stageClearVec_reqQueue_2_deq_bits_groupCounter[6:3]};
  wire [2:0]  readChannel_2_bits_offset_0 = stageClearVec_reqQueue_2_deq_bits_groupCounter[2:0];
  wire        stageClearVec_readResultValid_2 = ~needWAR | readResult_2_valid;
  wire [31:0] stageClearVec_WARData_2 = stageClearVec_WaitReadQueue_2_deq_bits_data & stageClearVec_WaitReadQueue_2_deq_bits_bitMask | readResult_2_bits & ~stageClearVec_WaitReadQueue_2_deq_bits_bitMask;
  wire        out_2_valid_0 = stageClearVec_WaitReadQueue_2_deq_valid & stageClearVec_readResultValid_2;
  assign stageClearVec_WaitReadQueue_2_deq_ready = out_2_ready_0 & stageClearVec_readResultValid_2;
  wire [31:0] out_2_bits_writeData_data_0 = needWAR ? stageClearVec_WARData_2 : stageClearVec_WaitReadQueue_2_deq_bits_data;
  wire [3:0]  out_2_bits_writeData_mask_0 = needWAR ? 4'hF : stageClearVec_WaitReadQueue_2_deq_bits_mask;
  reg  [2:0]  stageClearVec_counter_2;
  wire        _stageClearVec_T_6 = in_2_ready_0 & in_2_valid_0;
  wire [2:0]  stageClearVec_counterChange_2 = _stageClearVec_T_6 ? 3'h1 : 3'h7;
  wire        stageClearVec_2 = stageClearVec_counter_2 == 3'h0;
  wire        in_3_ready_0 = stageClearVec_reqQueue_3_enq_ready;
  wire        stageClearVec_reqQueue_3_deq_valid;
  assign stageClearVec_reqQueue_3_deq_valid = ~_stageClearVec_reqQueue_fifo_3_empty;
  wire [31:0] stageClearVec_reqQueue_dataOut_3_data;
  wire [31:0] stageClearVec_reqQueue_dataOut_3_bitMask;
  wire [31:0] stageClearVec_WaitReadQueue_3_enq_bits_data = stageClearVec_reqQueue_3_deq_bits_data;
  wire [3:0]  stageClearVec_reqQueue_dataOut_3_mask;
  wire [31:0] stageClearVec_WaitReadQueue_3_enq_bits_bitMask = stageClearVec_reqQueue_3_deq_bits_bitMask;
  wire [6:0]  stageClearVec_reqQueue_dataOut_3_groupCounter;
  wire [3:0]  stageClearVec_WaitReadQueue_3_enq_bits_mask = stageClearVec_reqQueue_3_deq_bits_mask;
  wire        stageClearVec_reqQueue_dataOut_3_ffoByOther;
  wire [6:0]  stageClearVec_WaitReadQueue_3_enq_bits_groupCounter = stageClearVec_reqQueue_3_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_3_enq_bits_ffoByOther = stageClearVec_reqQueue_3_deq_bits_ffoByOther;
  wire [7:0]  stageClearVec_reqQueue_dataIn_lo_3 = {stageClearVec_reqQueue_3_enq_bits_groupCounter, stageClearVec_reqQueue_3_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_reqQueue_dataIn_hi_hi_3 = {stageClearVec_reqQueue_3_enq_bits_data, stageClearVec_reqQueue_3_enq_bits_bitMask};
  wire [67:0] stageClearVec_reqQueue_dataIn_hi_3 = {stageClearVec_reqQueue_dataIn_hi_hi_3, stageClearVec_reqQueue_3_enq_bits_mask};
  wire [75:0] stageClearVec_reqQueue_dataIn_3 = {stageClearVec_reqQueue_dataIn_hi_3, stageClearVec_reqQueue_dataIn_lo_3};
  assign stageClearVec_reqQueue_dataOut_3_ffoByOther = _stageClearVec_reqQueue_fifo_3_data_out[0];
  assign stageClearVec_reqQueue_dataOut_3_groupCounter = _stageClearVec_reqQueue_fifo_3_data_out[7:1];
  assign stageClearVec_reqQueue_dataOut_3_mask = _stageClearVec_reqQueue_fifo_3_data_out[11:8];
  assign stageClearVec_reqQueue_dataOut_3_bitMask = _stageClearVec_reqQueue_fifo_3_data_out[43:12];
  assign stageClearVec_reqQueue_dataOut_3_data = _stageClearVec_reqQueue_fifo_3_data_out[75:44];
  assign stageClearVec_reqQueue_3_deq_bits_data = stageClearVec_reqQueue_dataOut_3_data;
  assign stageClearVec_reqQueue_3_deq_bits_bitMask = stageClearVec_reqQueue_dataOut_3_bitMask;
  assign stageClearVec_reqQueue_3_deq_bits_mask = stageClearVec_reqQueue_dataOut_3_mask;
  assign stageClearVec_reqQueue_3_deq_bits_groupCounter = stageClearVec_reqQueue_dataOut_3_groupCounter;
  assign stageClearVec_reqQueue_3_deq_bits_ffoByOther = stageClearVec_reqQueue_dataOut_3_ffoByOther;
  assign stageClearVec_reqQueue_3_enq_ready = ~_stageClearVec_reqQueue_fifo_3_full;
  wire        stageClearVec_reqQueue_3_deq_ready;
  wire        stageClearVec_WaitReadQueue_3_deq_valid;
  assign stageClearVec_WaitReadQueue_3_deq_valid = ~_stageClearVec_WaitReadQueue_fifo_3_empty;
  wire [31:0] stageClearVec_WaitReadQueue_dataOut_3_data;
  wire [31:0] stageClearVec_WaitReadQueue_dataOut_3_bitMask;
  wire [3:0]  stageClearVec_WaitReadQueue_dataOut_3_mask;
  wire [6:0]  stageClearVec_WaitReadQueue_dataOut_3_groupCounter;
  wire [6:0]  out_3_bits_writeData_groupCounter_0 = stageClearVec_WaitReadQueue_3_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_dataOut_3_ffoByOther;
  wire        out_3_bits_ffoByOther_0 = stageClearVec_WaitReadQueue_3_deq_bits_ffoByOther;
  wire [7:0]  stageClearVec_WaitReadQueue_dataIn_lo_3 = {stageClearVec_WaitReadQueue_3_enq_bits_groupCounter, stageClearVec_WaitReadQueue_3_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_WaitReadQueue_dataIn_hi_hi_3 = {stageClearVec_WaitReadQueue_3_enq_bits_data, stageClearVec_WaitReadQueue_3_enq_bits_bitMask};
  wire [67:0] stageClearVec_WaitReadQueue_dataIn_hi_3 = {stageClearVec_WaitReadQueue_dataIn_hi_hi_3, stageClearVec_WaitReadQueue_3_enq_bits_mask};
  wire [75:0] stageClearVec_WaitReadQueue_dataIn_3 = {stageClearVec_WaitReadQueue_dataIn_hi_3, stageClearVec_WaitReadQueue_dataIn_lo_3};
  assign stageClearVec_WaitReadQueue_dataOut_3_ffoByOther = _stageClearVec_WaitReadQueue_fifo_3_data_out[0];
  assign stageClearVec_WaitReadQueue_dataOut_3_groupCounter = _stageClearVec_WaitReadQueue_fifo_3_data_out[7:1];
  assign stageClearVec_WaitReadQueue_dataOut_3_mask = _stageClearVec_WaitReadQueue_fifo_3_data_out[11:8];
  assign stageClearVec_WaitReadQueue_dataOut_3_bitMask = _stageClearVec_WaitReadQueue_fifo_3_data_out[43:12];
  assign stageClearVec_WaitReadQueue_dataOut_3_data = _stageClearVec_WaitReadQueue_fifo_3_data_out[75:44];
  wire [31:0] stageClearVec_WaitReadQueue_3_deq_bits_data = stageClearVec_WaitReadQueue_dataOut_3_data;
  wire [31:0] stageClearVec_WaitReadQueue_3_deq_bits_bitMask = stageClearVec_WaitReadQueue_dataOut_3_bitMask;
  wire [3:0]  stageClearVec_WaitReadQueue_3_deq_bits_mask = stageClearVec_WaitReadQueue_dataOut_3_mask;
  assign stageClearVec_WaitReadQueue_3_deq_bits_groupCounter = stageClearVec_WaitReadQueue_dataOut_3_groupCounter;
  assign stageClearVec_WaitReadQueue_3_deq_bits_ffoByOther = stageClearVec_WaitReadQueue_dataOut_3_ffoByOther;
  wire        stageClearVec_WaitReadQueue_3_enq_ready = ~_stageClearVec_WaitReadQueue_fifo_3_full;
  wire        stageClearVec_WaitReadQueue_3_enq_valid;
  wire        stageClearVec_WaitReadQueue_3_deq_ready;
  wire        stageClearVec_readReady_3 = ~needWAR | readChannel_3_ready_0;
  assign stageClearVec_WaitReadQueue_3_enq_valid = stageClearVec_reqQueue_3_deq_valid & stageClearVec_readReady_3;
  assign stageClearVec_reqQueue_3_deq_ready = stageClearVec_WaitReadQueue_3_enq_ready & stageClearVec_readReady_3;
  wire        readChannel_3_valid_0 = stageClearVec_reqQueue_3_deq_valid & needWAR & stageClearVec_WaitReadQueue_3_enq_ready;
  wire [4:0]  readChannel_3_bits_vs_0 = vd + {1'h0, stageClearVec_reqQueue_3_deq_bits_groupCounter[6:3]};
  wire [2:0]  readChannel_3_bits_offset_0 = stageClearVec_reqQueue_3_deq_bits_groupCounter[2:0];
  wire        stageClearVec_readResultValid_3 = ~needWAR | readResult_3_valid;
  wire [31:0] stageClearVec_WARData_3 = stageClearVec_WaitReadQueue_3_deq_bits_data & stageClearVec_WaitReadQueue_3_deq_bits_bitMask | readResult_3_bits & ~stageClearVec_WaitReadQueue_3_deq_bits_bitMask;
  wire        out_3_valid_0 = stageClearVec_WaitReadQueue_3_deq_valid & stageClearVec_readResultValid_3;
  assign stageClearVec_WaitReadQueue_3_deq_ready = out_3_ready_0 & stageClearVec_readResultValid_3;
  wire [31:0] out_3_bits_writeData_data_0 = needWAR ? stageClearVec_WARData_3 : stageClearVec_WaitReadQueue_3_deq_bits_data;
  wire [3:0]  out_3_bits_writeData_mask_0 = needWAR ? 4'hF : stageClearVec_WaitReadQueue_3_deq_bits_mask;
  reg  [2:0]  stageClearVec_counter_3;
  wire        _stageClearVec_T_9 = in_3_ready_0 & in_3_valid_0;
  wire [2:0]  stageClearVec_counterChange_3 = _stageClearVec_T_9 ? 3'h1 : 3'h7;
  wire        stageClearVec_3 = stageClearVec_counter_3 == 3'h0;
  wire        in_4_ready_0 = stageClearVec_reqQueue_4_enq_ready;
  wire        stageClearVec_reqQueue_4_deq_valid;
  assign stageClearVec_reqQueue_4_deq_valid = ~_stageClearVec_reqQueue_fifo_4_empty;
  wire [31:0] stageClearVec_reqQueue_dataOut_4_data;
  wire [31:0] stageClearVec_reqQueue_dataOut_4_bitMask;
  wire [31:0] stageClearVec_WaitReadQueue_4_enq_bits_data = stageClearVec_reqQueue_4_deq_bits_data;
  wire [3:0]  stageClearVec_reqQueue_dataOut_4_mask;
  wire [31:0] stageClearVec_WaitReadQueue_4_enq_bits_bitMask = stageClearVec_reqQueue_4_deq_bits_bitMask;
  wire [6:0]  stageClearVec_reqQueue_dataOut_4_groupCounter;
  wire [3:0]  stageClearVec_WaitReadQueue_4_enq_bits_mask = stageClearVec_reqQueue_4_deq_bits_mask;
  wire        stageClearVec_reqQueue_dataOut_4_ffoByOther;
  wire [6:0]  stageClearVec_WaitReadQueue_4_enq_bits_groupCounter = stageClearVec_reqQueue_4_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_4_enq_bits_ffoByOther = stageClearVec_reqQueue_4_deq_bits_ffoByOther;
  wire [7:0]  stageClearVec_reqQueue_dataIn_lo_4 = {stageClearVec_reqQueue_4_enq_bits_groupCounter, stageClearVec_reqQueue_4_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_reqQueue_dataIn_hi_hi_4 = {stageClearVec_reqQueue_4_enq_bits_data, stageClearVec_reqQueue_4_enq_bits_bitMask};
  wire [67:0] stageClearVec_reqQueue_dataIn_hi_4 = {stageClearVec_reqQueue_dataIn_hi_hi_4, stageClearVec_reqQueue_4_enq_bits_mask};
  wire [75:0] stageClearVec_reqQueue_dataIn_4 = {stageClearVec_reqQueue_dataIn_hi_4, stageClearVec_reqQueue_dataIn_lo_4};
  assign stageClearVec_reqQueue_dataOut_4_ffoByOther = _stageClearVec_reqQueue_fifo_4_data_out[0];
  assign stageClearVec_reqQueue_dataOut_4_groupCounter = _stageClearVec_reqQueue_fifo_4_data_out[7:1];
  assign stageClearVec_reqQueue_dataOut_4_mask = _stageClearVec_reqQueue_fifo_4_data_out[11:8];
  assign stageClearVec_reqQueue_dataOut_4_bitMask = _stageClearVec_reqQueue_fifo_4_data_out[43:12];
  assign stageClearVec_reqQueue_dataOut_4_data = _stageClearVec_reqQueue_fifo_4_data_out[75:44];
  assign stageClearVec_reqQueue_4_deq_bits_data = stageClearVec_reqQueue_dataOut_4_data;
  assign stageClearVec_reqQueue_4_deq_bits_bitMask = stageClearVec_reqQueue_dataOut_4_bitMask;
  assign stageClearVec_reqQueue_4_deq_bits_mask = stageClearVec_reqQueue_dataOut_4_mask;
  assign stageClearVec_reqQueue_4_deq_bits_groupCounter = stageClearVec_reqQueue_dataOut_4_groupCounter;
  assign stageClearVec_reqQueue_4_deq_bits_ffoByOther = stageClearVec_reqQueue_dataOut_4_ffoByOther;
  assign stageClearVec_reqQueue_4_enq_ready = ~_stageClearVec_reqQueue_fifo_4_full;
  wire        stageClearVec_reqQueue_4_deq_ready;
  wire        stageClearVec_WaitReadQueue_4_deq_valid;
  assign stageClearVec_WaitReadQueue_4_deq_valid = ~_stageClearVec_WaitReadQueue_fifo_4_empty;
  wire [31:0] stageClearVec_WaitReadQueue_dataOut_4_data;
  wire [31:0] stageClearVec_WaitReadQueue_dataOut_4_bitMask;
  wire [3:0]  stageClearVec_WaitReadQueue_dataOut_4_mask;
  wire [6:0]  stageClearVec_WaitReadQueue_dataOut_4_groupCounter;
  wire [6:0]  out_4_bits_writeData_groupCounter_0 = stageClearVec_WaitReadQueue_4_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_dataOut_4_ffoByOther;
  wire        out_4_bits_ffoByOther_0 = stageClearVec_WaitReadQueue_4_deq_bits_ffoByOther;
  wire [7:0]  stageClearVec_WaitReadQueue_dataIn_lo_4 = {stageClearVec_WaitReadQueue_4_enq_bits_groupCounter, stageClearVec_WaitReadQueue_4_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_WaitReadQueue_dataIn_hi_hi_4 = {stageClearVec_WaitReadQueue_4_enq_bits_data, stageClearVec_WaitReadQueue_4_enq_bits_bitMask};
  wire [67:0] stageClearVec_WaitReadQueue_dataIn_hi_4 = {stageClearVec_WaitReadQueue_dataIn_hi_hi_4, stageClearVec_WaitReadQueue_4_enq_bits_mask};
  wire [75:0] stageClearVec_WaitReadQueue_dataIn_4 = {stageClearVec_WaitReadQueue_dataIn_hi_4, stageClearVec_WaitReadQueue_dataIn_lo_4};
  assign stageClearVec_WaitReadQueue_dataOut_4_ffoByOther = _stageClearVec_WaitReadQueue_fifo_4_data_out[0];
  assign stageClearVec_WaitReadQueue_dataOut_4_groupCounter = _stageClearVec_WaitReadQueue_fifo_4_data_out[7:1];
  assign stageClearVec_WaitReadQueue_dataOut_4_mask = _stageClearVec_WaitReadQueue_fifo_4_data_out[11:8];
  assign stageClearVec_WaitReadQueue_dataOut_4_bitMask = _stageClearVec_WaitReadQueue_fifo_4_data_out[43:12];
  assign stageClearVec_WaitReadQueue_dataOut_4_data = _stageClearVec_WaitReadQueue_fifo_4_data_out[75:44];
  wire [31:0] stageClearVec_WaitReadQueue_4_deq_bits_data = stageClearVec_WaitReadQueue_dataOut_4_data;
  wire [31:0] stageClearVec_WaitReadQueue_4_deq_bits_bitMask = stageClearVec_WaitReadQueue_dataOut_4_bitMask;
  wire [3:0]  stageClearVec_WaitReadQueue_4_deq_bits_mask = stageClearVec_WaitReadQueue_dataOut_4_mask;
  assign stageClearVec_WaitReadQueue_4_deq_bits_groupCounter = stageClearVec_WaitReadQueue_dataOut_4_groupCounter;
  assign stageClearVec_WaitReadQueue_4_deq_bits_ffoByOther = stageClearVec_WaitReadQueue_dataOut_4_ffoByOther;
  wire        stageClearVec_WaitReadQueue_4_enq_ready = ~_stageClearVec_WaitReadQueue_fifo_4_full;
  wire        stageClearVec_WaitReadQueue_4_enq_valid;
  wire        stageClearVec_WaitReadQueue_4_deq_ready;
  wire        stageClearVec_readReady_4 = ~needWAR | readChannel_4_ready_0;
  assign stageClearVec_WaitReadQueue_4_enq_valid = stageClearVec_reqQueue_4_deq_valid & stageClearVec_readReady_4;
  assign stageClearVec_reqQueue_4_deq_ready = stageClearVec_WaitReadQueue_4_enq_ready & stageClearVec_readReady_4;
  wire        readChannel_4_valid_0 = stageClearVec_reqQueue_4_deq_valid & needWAR & stageClearVec_WaitReadQueue_4_enq_ready;
  wire [4:0]  readChannel_4_bits_vs_0 = vd + {1'h0, stageClearVec_reqQueue_4_deq_bits_groupCounter[6:3]};
  wire [2:0]  readChannel_4_bits_offset_0 = stageClearVec_reqQueue_4_deq_bits_groupCounter[2:0];
  wire        stageClearVec_readResultValid_4 = ~needWAR | readResult_4_valid;
  wire [31:0] stageClearVec_WARData_4 = stageClearVec_WaitReadQueue_4_deq_bits_data & stageClearVec_WaitReadQueue_4_deq_bits_bitMask | readResult_4_bits & ~stageClearVec_WaitReadQueue_4_deq_bits_bitMask;
  wire        out_4_valid_0 = stageClearVec_WaitReadQueue_4_deq_valid & stageClearVec_readResultValid_4;
  assign stageClearVec_WaitReadQueue_4_deq_ready = out_4_ready_0 & stageClearVec_readResultValid_4;
  wire [31:0] out_4_bits_writeData_data_0 = needWAR ? stageClearVec_WARData_4 : stageClearVec_WaitReadQueue_4_deq_bits_data;
  wire [3:0]  out_4_bits_writeData_mask_0 = needWAR ? 4'hF : stageClearVec_WaitReadQueue_4_deq_bits_mask;
  reg  [2:0]  stageClearVec_counter_4;
  wire        _stageClearVec_T_12 = in_4_ready_0 & in_4_valid_0;
  wire [2:0]  stageClearVec_counterChange_4 = _stageClearVec_T_12 ? 3'h1 : 3'h7;
  wire        stageClearVec_4 = stageClearVec_counter_4 == 3'h0;
  wire        in_5_ready_0 = stageClearVec_reqQueue_5_enq_ready;
  wire        stageClearVec_reqQueue_5_deq_valid;
  assign stageClearVec_reqQueue_5_deq_valid = ~_stageClearVec_reqQueue_fifo_5_empty;
  wire [31:0] stageClearVec_reqQueue_dataOut_5_data;
  wire [31:0] stageClearVec_reqQueue_dataOut_5_bitMask;
  wire [31:0] stageClearVec_WaitReadQueue_5_enq_bits_data = stageClearVec_reqQueue_5_deq_bits_data;
  wire [3:0]  stageClearVec_reqQueue_dataOut_5_mask;
  wire [31:0] stageClearVec_WaitReadQueue_5_enq_bits_bitMask = stageClearVec_reqQueue_5_deq_bits_bitMask;
  wire [6:0]  stageClearVec_reqQueue_dataOut_5_groupCounter;
  wire [3:0]  stageClearVec_WaitReadQueue_5_enq_bits_mask = stageClearVec_reqQueue_5_deq_bits_mask;
  wire        stageClearVec_reqQueue_dataOut_5_ffoByOther;
  wire [6:0]  stageClearVec_WaitReadQueue_5_enq_bits_groupCounter = stageClearVec_reqQueue_5_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_5_enq_bits_ffoByOther = stageClearVec_reqQueue_5_deq_bits_ffoByOther;
  wire [7:0]  stageClearVec_reqQueue_dataIn_lo_5 = {stageClearVec_reqQueue_5_enq_bits_groupCounter, stageClearVec_reqQueue_5_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_reqQueue_dataIn_hi_hi_5 = {stageClearVec_reqQueue_5_enq_bits_data, stageClearVec_reqQueue_5_enq_bits_bitMask};
  wire [67:0] stageClearVec_reqQueue_dataIn_hi_5 = {stageClearVec_reqQueue_dataIn_hi_hi_5, stageClearVec_reqQueue_5_enq_bits_mask};
  wire [75:0] stageClearVec_reqQueue_dataIn_5 = {stageClearVec_reqQueue_dataIn_hi_5, stageClearVec_reqQueue_dataIn_lo_5};
  assign stageClearVec_reqQueue_dataOut_5_ffoByOther = _stageClearVec_reqQueue_fifo_5_data_out[0];
  assign stageClearVec_reqQueue_dataOut_5_groupCounter = _stageClearVec_reqQueue_fifo_5_data_out[7:1];
  assign stageClearVec_reqQueue_dataOut_5_mask = _stageClearVec_reqQueue_fifo_5_data_out[11:8];
  assign stageClearVec_reqQueue_dataOut_5_bitMask = _stageClearVec_reqQueue_fifo_5_data_out[43:12];
  assign stageClearVec_reqQueue_dataOut_5_data = _stageClearVec_reqQueue_fifo_5_data_out[75:44];
  assign stageClearVec_reqQueue_5_deq_bits_data = stageClearVec_reqQueue_dataOut_5_data;
  assign stageClearVec_reqQueue_5_deq_bits_bitMask = stageClearVec_reqQueue_dataOut_5_bitMask;
  assign stageClearVec_reqQueue_5_deq_bits_mask = stageClearVec_reqQueue_dataOut_5_mask;
  assign stageClearVec_reqQueue_5_deq_bits_groupCounter = stageClearVec_reqQueue_dataOut_5_groupCounter;
  assign stageClearVec_reqQueue_5_deq_bits_ffoByOther = stageClearVec_reqQueue_dataOut_5_ffoByOther;
  assign stageClearVec_reqQueue_5_enq_ready = ~_stageClearVec_reqQueue_fifo_5_full;
  wire        stageClearVec_reqQueue_5_deq_ready;
  wire        stageClearVec_WaitReadQueue_5_deq_valid;
  assign stageClearVec_WaitReadQueue_5_deq_valid = ~_stageClearVec_WaitReadQueue_fifo_5_empty;
  wire [31:0] stageClearVec_WaitReadQueue_dataOut_5_data;
  wire [31:0] stageClearVec_WaitReadQueue_dataOut_5_bitMask;
  wire [3:0]  stageClearVec_WaitReadQueue_dataOut_5_mask;
  wire [6:0]  stageClearVec_WaitReadQueue_dataOut_5_groupCounter;
  wire [6:0]  out_5_bits_writeData_groupCounter_0 = stageClearVec_WaitReadQueue_5_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_dataOut_5_ffoByOther;
  wire        out_5_bits_ffoByOther_0 = stageClearVec_WaitReadQueue_5_deq_bits_ffoByOther;
  wire [7:0]  stageClearVec_WaitReadQueue_dataIn_lo_5 = {stageClearVec_WaitReadQueue_5_enq_bits_groupCounter, stageClearVec_WaitReadQueue_5_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_WaitReadQueue_dataIn_hi_hi_5 = {stageClearVec_WaitReadQueue_5_enq_bits_data, stageClearVec_WaitReadQueue_5_enq_bits_bitMask};
  wire [67:0] stageClearVec_WaitReadQueue_dataIn_hi_5 = {stageClearVec_WaitReadQueue_dataIn_hi_hi_5, stageClearVec_WaitReadQueue_5_enq_bits_mask};
  wire [75:0] stageClearVec_WaitReadQueue_dataIn_5 = {stageClearVec_WaitReadQueue_dataIn_hi_5, stageClearVec_WaitReadQueue_dataIn_lo_5};
  assign stageClearVec_WaitReadQueue_dataOut_5_ffoByOther = _stageClearVec_WaitReadQueue_fifo_5_data_out[0];
  assign stageClearVec_WaitReadQueue_dataOut_5_groupCounter = _stageClearVec_WaitReadQueue_fifo_5_data_out[7:1];
  assign stageClearVec_WaitReadQueue_dataOut_5_mask = _stageClearVec_WaitReadQueue_fifo_5_data_out[11:8];
  assign stageClearVec_WaitReadQueue_dataOut_5_bitMask = _stageClearVec_WaitReadQueue_fifo_5_data_out[43:12];
  assign stageClearVec_WaitReadQueue_dataOut_5_data = _stageClearVec_WaitReadQueue_fifo_5_data_out[75:44];
  wire [31:0] stageClearVec_WaitReadQueue_5_deq_bits_data = stageClearVec_WaitReadQueue_dataOut_5_data;
  wire [31:0] stageClearVec_WaitReadQueue_5_deq_bits_bitMask = stageClearVec_WaitReadQueue_dataOut_5_bitMask;
  wire [3:0]  stageClearVec_WaitReadQueue_5_deq_bits_mask = stageClearVec_WaitReadQueue_dataOut_5_mask;
  assign stageClearVec_WaitReadQueue_5_deq_bits_groupCounter = stageClearVec_WaitReadQueue_dataOut_5_groupCounter;
  assign stageClearVec_WaitReadQueue_5_deq_bits_ffoByOther = stageClearVec_WaitReadQueue_dataOut_5_ffoByOther;
  wire        stageClearVec_WaitReadQueue_5_enq_ready = ~_stageClearVec_WaitReadQueue_fifo_5_full;
  wire        stageClearVec_WaitReadQueue_5_enq_valid;
  wire        stageClearVec_WaitReadQueue_5_deq_ready;
  wire        stageClearVec_readReady_5 = ~needWAR | readChannel_5_ready_0;
  assign stageClearVec_WaitReadQueue_5_enq_valid = stageClearVec_reqQueue_5_deq_valid & stageClearVec_readReady_5;
  assign stageClearVec_reqQueue_5_deq_ready = stageClearVec_WaitReadQueue_5_enq_ready & stageClearVec_readReady_5;
  wire        readChannel_5_valid_0 = stageClearVec_reqQueue_5_deq_valid & needWAR & stageClearVec_WaitReadQueue_5_enq_ready;
  wire [4:0]  readChannel_5_bits_vs_0 = vd + {1'h0, stageClearVec_reqQueue_5_deq_bits_groupCounter[6:3]};
  wire [2:0]  readChannel_5_bits_offset_0 = stageClearVec_reqQueue_5_deq_bits_groupCounter[2:0];
  wire        stageClearVec_readResultValid_5 = ~needWAR | readResult_5_valid;
  wire [31:0] stageClearVec_WARData_5 = stageClearVec_WaitReadQueue_5_deq_bits_data & stageClearVec_WaitReadQueue_5_deq_bits_bitMask | readResult_5_bits & ~stageClearVec_WaitReadQueue_5_deq_bits_bitMask;
  wire        out_5_valid_0 = stageClearVec_WaitReadQueue_5_deq_valid & stageClearVec_readResultValid_5;
  assign stageClearVec_WaitReadQueue_5_deq_ready = out_5_ready_0 & stageClearVec_readResultValid_5;
  wire [31:0] out_5_bits_writeData_data_0 = needWAR ? stageClearVec_WARData_5 : stageClearVec_WaitReadQueue_5_deq_bits_data;
  wire [3:0]  out_5_bits_writeData_mask_0 = needWAR ? 4'hF : stageClearVec_WaitReadQueue_5_deq_bits_mask;
  reg  [2:0]  stageClearVec_counter_5;
  wire        _stageClearVec_T_15 = in_5_ready_0 & in_5_valid_0;
  wire [2:0]  stageClearVec_counterChange_5 = _stageClearVec_T_15 ? 3'h1 : 3'h7;
  wire        stageClearVec_5 = stageClearVec_counter_5 == 3'h0;
  wire        in_6_ready_0 = stageClearVec_reqQueue_6_enq_ready;
  wire        stageClearVec_reqQueue_6_deq_valid;
  assign stageClearVec_reqQueue_6_deq_valid = ~_stageClearVec_reqQueue_fifo_6_empty;
  wire [31:0] stageClearVec_reqQueue_dataOut_6_data;
  wire [31:0] stageClearVec_reqQueue_dataOut_6_bitMask;
  wire [31:0] stageClearVec_WaitReadQueue_6_enq_bits_data = stageClearVec_reqQueue_6_deq_bits_data;
  wire [3:0]  stageClearVec_reqQueue_dataOut_6_mask;
  wire [31:0] stageClearVec_WaitReadQueue_6_enq_bits_bitMask = stageClearVec_reqQueue_6_deq_bits_bitMask;
  wire [6:0]  stageClearVec_reqQueue_dataOut_6_groupCounter;
  wire [3:0]  stageClearVec_WaitReadQueue_6_enq_bits_mask = stageClearVec_reqQueue_6_deq_bits_mask;
  wire        stageClearVec_reqQueue_dataOut_6_ffoByOther;
  wire [6:0]  stageClearVec_WaitReadQueue_6_enq_bits_groupCounter = stageClearVec_reqQueue_6_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_6_enq_bits_ffoByOther = stageClearVec_reqQueue_6_deq_bits_ffoByOther;
  wire [7:0]  stageClearVec_reqQueue_dataIn_lo_6 = {stageClearVec_reqQueue_6_enq_bits_groupCounter, stageClearVec_reqQueue_6_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_reqQueue_dataIn_hi_hi_6 = {stageClearVec_reqQueue_6_enq_bits_data, stageClearVec_reqQueue_6_enq_bits_bitMask};
  wire [67:0] stageClearVec_reqQueue_dataIn_hi_6 = {stageClearVec_reqQueue_dataIn_hi_hi_6, stageClearVec_reqQueue_6_enq_bits_mask};
  wire [75:0] stageClearVec_reqQueue_dataIn_6 = {stageClearVec_reqQueue_dataIn_hi_6, stageClearVec_reqQueue_dataIn_lo_6};
  assign stageClearVec_reqQueue_dataOut_6_ffoByOther = _stageClearVec_reqQueue_fifo_6_data_out[0];
  assign stageClearVec_reqQueue_dataOut_6_groupCounter = _stageClearVec_reqQueue_fifo_6_data_out[7:1];
  assign stageClearVec_reqQueue_dataOut_6_mask = _stageClearVec_reqQueue_fifo_6_data_out[11:8];
  assign stageClearVec_reqQueue_dataOut_6_bitMask = _stageClearVec_reqQueue_fifo_6_data_out[43:12];
  assign stageClearVec_reqQueue_dataOut_6_data = _stageClearVec_reqQueue_fifo_6_data_out[75:44];
  assign stageClearVec_reqQueue_6_deq_bits_data = stageClearVec_reqQueue_dataOut_6_data;
  assign stageClearVec_reqQueue_6_deq_bits_bitMask = stageClearVec_reqQueue_dataOut_6_bitMask;
  assign stageClearVec_reqQueue_6_deq_bits_mask = stageClearVec_reqQueue_dataOut_6_mask;
  assign stageClearVec_reqQueue_6_deq_bits_groupCounter = stageClearVec_reqQueue_dataOut_6_groupCounter;
  assign stageClearVec_reqQueue_6_deq_bits_ffoByOther = stageClearVec_reqQueue_dataOut_6_ffoByOther;
  assign stageClearVec_reqQueue_6_enq_ready = ~_stageClearVec_reqQueue_fifo_6_full;
  wire        stageClearVec_reqQueue_6_deq_ready;
  wire        stageClearVec_WaitReadQueue_6_deq_valid;
  assign stageClearVec_WaitReadQueue_6_deq_valid = ~_stageClearVec_WaitReadQueue_fifo_6_empty;
  wire [31:0] stageClearVec_WaitReadQueue_dataOut_6_data;
  wire [31:0] stageClearVec_WaitReadQueue_dataOut_6_bitMask;
  wire [3:0]  stageClearVec_WaitReadQueue_dataOut_6_mask;
  wire [6:0]  stageClearVec_WaitReadQueue_dataOut_6_groupCounter;
  wire [6:0]  out_6_bits_writeData_groupCounter_0 = stageClearVec_WaitReadQueue_6_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_dataOut_6_ffoByOther;
  wire        out_6_bits_ffoByOther_0 = stageClearVec_WaitReadQueue_6_deq_bits_ffoByOther;
  wire [7:0]  stageClearVec_WaitReadQueue_dataIn_lo_6 = {stageClearVec_WaitReadQueue_6_enq_bits_groupCounter, stageClearVec_WaitReadQueue_6_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_WaitReadQueue_dataIn_hi_hi_6 = {stageClearVec_WaitReadQueue_6_enq_bits_data, stageClearVec_WaitReadQueue_6_enq_bits_bitMask};
  wire [67:0] stageClearVec_WaitReadQueue_dataIn_hi_6 = {stageClearVec_WaitReadQueue_dataIn_hi_hi_6, stageClearVec_WaitReadQueue_6_enq_bits_mask};
  wire [75:0] stageClearVec_WaitReadQueue_dataIn_6 = {stageClearVec_WaitReadQueue_dataIn_hi_6, stageClearVec_WaitReadQueue_dataIn_lo_6};
  assign stageClearVec_WaitReadQueue_dataOut_6_ffoByOther = _stageClearVec_WaitReadQueue_fifo_6_data_out[0];
  assign stageClearVec_WaitReadQueue_dataOut_6_groupCounter = _stageClearVec_WaitReadQueue_fifo_6_data_out[7:1];
  assign stageClearVec_WaitReadQueue_dataOut_6_mask = _stageClearVec_WaitReadQueue_fifo_6_data_out[11:8];
  assign stageClearVec_WaitReadQueue_dataOut_6_bitMask = _stageClearVec_WaitReadQueue_fifo_6_data_out[43:12];
  assign stageClearVec_WaitReadQueue_dataOut_6_data = _stageClearVec_WaitReadQueue_fifo_6_data_out[75:44];
  wire [31:0] stageClearVec_WaitReadQueue_6_deq_bits_data = stageClearVec_WaitReadQueue_dataOut_6_data;
  wire [31:0] stageClearVec_WaitReadQueue_6_deq_bits_bitMask = stageClearVec_WaitReadQueue_dataOut_6_bitMask;
  wire [3:0]  stageClearVec_WaitReadQueue_6_deq_bits_mask = stageClearVec_WaitReadQueue_dataOut_6_mask;
  assign stageClearVec_WaitReadQueue_6_deq_bits_groupCounter = stageClearVec_WaitReadQueue_dataOut_6_groupCounter;
  assign stageClearVec_WaitReadQueue_6_deq_bits_ffoByOther = stageClearVec_WaitReadQueue_dataOut_6_ffoByOther;
  wire        stageClearVec_WaitReadQueue_6_enq_ready = ~_stageClearVec_WaitReadQueue_fifo_6_full;
  wire        stageClearVec_WaitReadQueue_6_enq_valid;
  wire        stageClearVec_WaitReadQueue_6_deq_ready;
  wire        stageClearVec_readReady_6 = ~needWAR | readChannel_6_ready_0;
  assign stageClearVec_WaitReadQueue_6_enq_valid = stageClearVec_reqQueue_6_deq_valid & stageClearVec_readReady_6;
  assign stageClearVec_reqQueue_6_deq_ready = stageClearVec_WaitReadQueue_6_enq_ready & stageClearVec_readReady_6;
  wire        readChannel_6_valid_0 = stageClearVec_reqQueue_6_deq_valid & needWAR & stageClearVec_WaitReadQueue_6_enq_ready;
  wire [4:0]  readChannel_6_bits_vs_0 = vd + {1'h0, stageClearVec_reqQueue_6_deq_bits_groupCounter[6:3]};
  wire [2:0]  readChannel_6_bits_offset_0 = stageClearVec_reqQueue_6_deq_bits_groupCounter[2:0];
  wire        stageClearVec_readResultValid_6 = ~needWAR | readResult_6_valid;
  wire [31:0] stageClearVec_WARData_6 = stageClearVec_WaitReadQueue_6_deq_bits_data & stageClearVec_WaitReadQueue_6_deq_bits_bitMask | readResult_6_bits & ~stageClearVec_WaitReadQueue_6_deq_bits_bitMask;
  wire        out_6_valid_0 = stageClearVec_WaitReadQueue_6_deq_valid & stageClearVec_readResultValid_6;
  assign stageClearVec_WaitReadQueue_6_deq_ready = out_6_ready_0 & stageClearVec_readResultValid_6;
  wire [31:0] out_6_bits_writeData_data_0 = needWAR ? stageClearVec_WARData_6 : stageClearVec_WaitReadQueue_6_deq_bits_data;
  wire [3:0]  out_6_bits_writeData_mask_0 = needWAR ? 4'hF : stageClearVec_WaitReadQueue_6_deq_bits_mask;
  reg  [2:0]  stageClearVec_counter_6;
  wire        _stageClearVec_T_18 = in_6_ready_0 & in_6_valid_0;
  wire [2:0]  stageClearVec_counterChange_6 = _stageClearVec_T_18 ? 3'h1 : 3'h7;
  wire        stageClearVec_6 = stageClearVec_counter_6 == 3'h0;
  wire        in_7_ready_0 = stageClearVec_reqQueue_7_enq_ready;
  wire        stageClearVec_reqQueue_7_deq_valid;
  assign stageClearVec_reqQueue_7_deq_valid = ~_stageClearVec_reqQueue_fifo_7_empty;
  wire [31:0] stageClearVec_reqQueue_dataOut_7_data;
  wire [31:0] stageClearVec_reqQueue_dataOut_7_bitMask;
  wire [31:0] stageClearVec_WaitReadQueue_7_enq_bits_data = stageClearVec_reqQueue_7_deq_bits_data;
  wire [3:0]  stageClearVec_reqQueue_dataOut_7_mask;
  wire [31:0] stageClearVec_WaitReadQueue_7_enq_bits_bitMask = stageClearVec_reqQueue_7_deq_bits_bitMask;
  wire [6:0]  stageClearVec_reqQueue_dataOut_7_groupCounter;
  wire [3:0]  stageClearVec_WaitReadQueue_7_enq_bits_mask = stageClearVec_reqQueue_7_deq_bits_mask;
  wire        stageClearVec_reqQueue_dataOut_7_ffoByOther;
  wire [6:0]  stageClearVec_WaitReadQueue_7_enq_bits_groupCounter = stageClearVec_reqQueue_7_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_7_enq_bits_ffoByOther = stageClearVec_reqQueue_7_deq_bits_ffoByOther;
  wire [7:0]  stageClearVec_reqQueue_dataIn_lo_7 = {stageClearVec_reqQueue_7_enq_bits_groupCounter, stageClearVec_reqQueue_7_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_reqQueue_dataIn_hi_hi_7 = {stageClearVec_reqQueue_7_enq_bits_data, stageClearVec_reqQueue_7_enq_bits_bitMask};
  wire [67:0] stageClearVec_reqQueue_dataIn_hi_7 = {stageClearVec_reqQueue_dataIn_hi_hi_7, stageClearVec_reqQueue_7_enq_bits_mask};
  wire [75:0] stageClearVec_reqQueue_dataIn_7 = {stageClearVec_reqQueue_dataIn_hi_7, stageClearVec_reqQueue_dataIn_lo_7};
  assign stageClearVec_reqQueue_dataOut_7_ffoByOther = _stageClearVec_reqQueue_fifo_7_data_out[0];
  assign stageClearVec_reqQueue_dataOut_7_groupCounter = _stageClearVec_reqQueue_fifo_7_data_out[7:1];
  assign stageClearVec_reqQueue_dataOut_7_mask = _stageClearVec_reqQueue_fifo_7_data_out[11:8];
  assign stageClearVec_reqQueue_dataOut_7_bitMask = _stageClearVec_reqQueue_fifo_7_data_out[43:12];
  assign stageClearVec_reqQueue_dataOut_7_data = _stageClearVec_reqQueue_fifo_7_data_out[75:44];
  assign stageClearVec_reqQueue_7_deq_bits_data = stageClearVec_reqQueue_dataOut_7_data;
  assign stageClearVec_reqQueue_7_deq_bits_bitMask = stageClearVec_reqQueue_dataOut_7_bitMask;
  assign stageClearVec_reqQueue_7_deq_bits_mask = stageClearVec_reqQueue_dataOut_7_mask;
  assign stageClearVec_reqQueue_7_deq_bits_groupCounter = stageClearVec_reqQueue_dataOut_7_groupCounter;
  assign stageClearVec_reqQueue_7_deq_bits_ffoByOther = stageClearVec_reqQueue_dataOut_7_ffoByOther;
  assign stageClearVec_reqQueue_7_enq_ready = ~_stageClearVec_reqQueue_fifo_7_full;
  wire        stageClearVec_reqQueue_7_deq_ready;
  wire        stageClearVec_WaitReadQueue_7_deq_valid;
  assign stageClearVec_WaitReadQueue_7_deq_valid = ~_stageClearVec_WaitReadQueue_fifo_7_empty;
  wire [31:0] stageClearVec_WaitReadQueue_dataOut_7_data;
  wire [31:0] stageClearVec_WaitReadQueue_dataOut_7_bitMask;
  wire [3:0]  stageClearVec_WaitReadQueue_dataOut_7_mask;
  wire [6:0]  stageClearVec_WaitReadQueue_dataOut_7_groupCounter;
  wire [6:0]  out_7_bits_writeData_groupCounter_0 = stageClearVec_WaitReadQueue_7_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_dataOut_7_ffoByOther;
  wire        out_7_bits_ffoByOther_0 = stageClearVec_WaitReadQueue_7_deq_bits_ffoByOther;
  wire [7:0]  stageClearVec_WaitReadQueue_dataIn_lo_7 = {stageClearVec_WaitReadQueue_7_enq_bits_groupCounter, stageClearVec_WaitReadQueue_7_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_WaitReadQueue_dataIn_hi_hi_7 = {stageClearVec_WaitReadQueue_7_enq_bits_data, stageClearVec_WaitReadQueue_7_enq_bits_bitMask};
  wire [67:0] stageClearVec_WaitReadQueue_dataIn_hi_7 = {stageClearVec_WaitReadQueue_dataIn_hi_hi_7, stageClearVec_WaitReadQueue_7_enq_bits_mask};
  wire [75:0] stageClearVec_WaitReadQueue_dataIn_7 = {stageClearVec_WaitReadQueue_dataIn_hi_7, stageClearVec_WaitReadQueue_dataIn_lo_7};
  assign stageClearVec_WaitReadQueue_dataOut_7_ffoByOther = _stageClearVec_WaitReadQueue_fifo_7_data_out[0];
  assign stageClearVec_WaitReadQueue_dataOut_7_groupCounter = _stageClearVec_WaitReadQueue_fifo_7_data_out[7:1];
  assign stageClearVec_WaitReadQueue_dataOut_7_mask = _stageClearVec_WaitReadQueue_fifo_7_data_out[11:8];
  assign stageClearVec_WaitReadQueue_dataOut_7_bitMask = _stageClearVec_WaitReadQueue_fifo_7_data_out[43:12];
  assign stageClearVec_WaitReadQueue_dataOut_7_data = _stageClearVec_WaitReadQueue_fifo_7_data_out[75:44];
  wire [31:0] stageClearVec_WaitReadQueue_7_deq_bits_data = stageClearVec_WaitReadQueue_dataOut_7_data;
  wire [31:0] stageClearVec_WaitReadQueue_7_deq_bits_bitMask = stageClearVec_WaitReadQueue_dataOut_7_bitMask;
  wire [3:0]  stageClearVec_WaitReadQueue_7_deq_bits_mask = stageClearVec_WaitReadQueue_dataOut_7_mask;
  assign stageClearVec_WaitReadQueue_7_deq_bits_groupCounter = stageClearVec_WaitReadQueue_dataOut_7_groupCounter;
  assign stageClearVec_WaitReadQueue_7_deq_bits_ffoByOther = stageClearVec_WaitReadQueue_dataOut_7_ffoByOther;
  wire        stageClearVec_WaitReadQueue_7_enq_ready = ~_stageClearVec_WaitReadQueue_fifo_7_full;
  wire        stageClearVec_WaitReadQueue_7_enq_valid;
  wire        stageClearVec_WaitReadQueue_7_deq_ready;
  wire        stageClearVec_readReady_7 = ~needWAR | readChannel_7_ready_0;
  assign stageClearVec_WaitReadQueue_7_enq_valid = stageClearVec_reqQueue_7_deq_valid & stageClearVec_readReady_7;
  assign stageClearVec_reqQueue_7_deq_ready = stageClearVec_WaitReadQueue_7_enq_ready & stageClearVec_readReady_7;
  wire        readChannel_7_valid_0 = stageClearVec_reqQueue_7_deq_valid & needWAR & stageClearVec_WaitReadQueue_7_enq_ready;
  wire [4:0]  readChannel_7_bits_vs_0 = vd + {1'h0, stageClearVec_reqQueue_7_deq_bits_groupCounter[6:3]};
  wire [2:0]  readChannel_7_bits_offset_0 = stageClearVec_reqQueue_7_deq_bits_groupCounter[2:0];
  wire        stageClearVec_readResultValid_7 = ~needWAR | readResult_7_valid;
  wire [31:0] stageClearVec_WARData_7 = stageClearVec_WaitReadQueue_7_deq_bits_data & stageClearVec_WaitReadQueue_7_deq_bits_bitMask | readResult_7_bits & ~stageClearVec_WaitReadQueue_7_deq_bits_bitMask;
  wire        out_7_valid_0 = stageClearVec_WaitReadQueue_7_deq_valid & stageClearVec_readResultValid_7;
  assign stageClearVec_WaitReadQueue_7_deq_ready = out_7_ready_0 & stageClearVec_readResultValid_7;
  wire [31:0] out_7_bits_writeData_data_0 = needWAR ? stageClearVec_WARData_7 : stageClearVec_WaitReadQueue_7_deq_bits_data;
  wire [3:0]  out_7_bits_writeData_mask_0 = needWAR ? 4'hF : stageClearVec_WaitReadQueue_7_deq_bits_mask;
  reg  [2:0]  stageClearVec_counter_7;
  wire        _stageClearVec_T_21 = in_7_ready_0 & in_7_valid_0;
  wire [2:0]  stageClearVec_counterChange_7 = _stageClearVec_T_21 ? 3'h1 : 3'h7;
  wire        stageClearVec_7 = stageClearVec_counter_7 == 3'h0;
  always @(posedge clock) begin
    if (reset) begin
      stageClearVec_counter <= 3'h0;
      stageClearVec_counter_1 <= 3'h0;
      stageClearVec_counter_2 <= 3'h0;
      stageClearVec_counter_3 <= 3'h0;
      stageClearVec_counter_4 <= 3'h0;
      stageClearVec_counter_5 <= 3'h0;
      stageClearVec_counter_6 <= 3'h0;
      stageClearVec_counter_7 <= 3'h0;
    end
    else begin
      if (_stageClearVec_T ^ out_0_ready_0 & out_0_valid_0)
        stageClearVec_counter <= stageClearVec_counter + stageClearVec_counterChange;
      if (_stageClearVec_T_3 ^ out_1_ready_0 & out_1_valid_0)
        stageClearVec_counter_1 <= stageClearVec_counter_1 + stageClearVec_counterChange_1;
      if (_stageClearVec_T_6 ^ out_2_ready_0 & out_2_valid_0)
        stageClearVec_counter_2 <= stageClearVec_counter_2 + stageClearVec_counterChange_2;
      if (_stageClearVec_T_9 ^ out_3_ready_0 & out_3_valid_0)
        stageClearVec_counter_3 <= stageClearVec_counter_3 + stageClearVec_counterChange_3;
      if (_stageClearVec_T_12 ^ out_4_ready_0 & out_4_valid_0)
        stageClearVec_counter_4 <= stageClearVec_counter_4 + stageClearVec_counterChange_4;
      if (_stageClearVec_T_15 ^ out_5_ready_0 & out_5_valid_0)
        stageClearVec_counter_5 <= stageClearVec_counter_5 + stageClearVec_counterChange_5;
      if (_stageClearVec_T_18 ^ out_6_ready_0 & out_6_valid_0)
        stageClearVec_counter_6 <= stageClearVec_counter_6 + stageClearVec_counterChange_6;
      if (_stageClearVec_T_21 ^ out_7_ready_0 & out_7_valid_0)
        stageClearVec_counter_7 <= stageClearVec_counter_7 + stageClearVec_counterChange_7;
    end
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:0];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        _RANDOM[/*Zero width*/ 1'b0] = `RANDOM;
        stageClearVec_counter = _RANDOM[/*Zero width*/ 1'b0][2:0];
        stageClearVec_counter_1 = _RANDOM[/*Zero width*/ 1'b0][5:3];
        stageClearVec_counter_2 = _RANDOM[/*Zero width*/ 1'b0][8:6];
        stageClearVec_counter_3 = _RANDOM[/*Zero width*/ 1'b0][11:9];
        stageClearVec_counter_4 = _RANDOM[/*Zero width*/ 1'b0][14:12];
        stageClearVec_counter_5 = _RANDOM[/*Zero width*/ 1'b0][17:15];
        stageClearVec_counter_6 = _RANDOM[/*Zero width*/ 1'b0][20:18];
        stageClearVec_counter_7 = _RANDOM[/*Zero width*/ 1'b0][23:21];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  wire        stageClearVec_reqQueue_empty;
  assign stageClearVec_reqQueue_empty = _stageClearVec_reqQueue_fifo_empty;
  wire        stageClearVec_reqQueue_full;
  assign stageClearVec_reqQueue_full = _stageClearVec_reqQueue_fifo_full;
  wire        stageClearVec_WaitReadQueue_empty;
  assign stageClearVec_WaitReadQueue_empty = _stageClearVec_WaitReadQueue_fifo_empty;
  wire        stageClearVec_WaitReadQueue_full;
  assign stageClearVec_WaitReadQueue_full = _stageClearVec_WaitReadQueue_fifo_full;
  wire        stageClearVec_reqQueue_1_empty;
  assign stageClearVec_reqQueue_1_empty = _stageClearVec_reqQueue_fifo_1_empty;
  wire        stageClearVec_reqQueue_1_full;
  assign stageClearVec_reqQueue_1_full = _stageClearVec_reqQueue_fifo_1_full;
  wire        stageClearVec_WaitReadQueue_1_empty;
  assign stageClearVec_WaitReadQueue_1_empty = _stageClearVec_WaitReadQueue_fifo_1_empty;
  wire        stageClearVec_WaitReadQueue_1_full;
  assign stageClearVec_WaitReadQueue_1_full = _stageClearVec_WaitReadQueue_fifo_1_full;
  wire        stageClearVec_reqQueue_2_empty;
  assign stageClearVec_reqQueue_2_empty = _stageClearVec_reqQueue_fifo_2_empty;
  wire        stageClearVec_reqQueue_2_full;
  assign stageClearVec_reqQueue_2_full = _stageClearVec_reqQueue_fifo_2_full;
  wire        stageClearVec_WaitReadQueue_2_empty;
  assign stageClearVec_WaitReadQueue_2_empty = _stageClearVec_WaitReadQueue_fifo_2_empty;
  wire        stageClearVec_WaitReadQueue_2_full;
  assign stageClearVec_WaitReadQueue_2_full = _stageClearVec_WaitReadQueue_fifo_2_full;
  wire        stageClearVec_reqQueue_3_empty;
  assign stageClearVec_reqQueue_3_empty = _stageClearVec_reqQueue_fifo_3_empty;
  wire        stageClearVec_reqQueue_3_full;
  assign stageClearVec_reqQueue_3_full = _stageClearVec_reqQueue_fifo_3_full;
  wire        stageClearVec_WaitReadQueue_3_empty;
  assign stageClearVec_WaitReadQueue_3_empty = _stageClearVec_WaitReadQueue_fifo_3_empty;
  wire        stageClearVec_WaitReadQueue_3_full;
  assign stageClearVec_WaitReadQueue_3_full = _stageClearVec_WaitReadQueue_fifo_3_full;
  wire        stageClearVec_reqQueue_4_empty;
  assign stageClearVec_reqQueue_4_empty = _stageClearVec_reqQueue_fifo_4_empty;
  wire        stageClearVec_reqQueue_4_full;
  assign stageClearVec_reqQueue_4_full = _stageClearVec_reqQueue_fifo_4_full;
  wire        stageClearVec_WaitReadQueue_4_empty;
  assign stageClearVec_WaitReadQueue_4_empty = _stageClearVec_WaitReadQueue_fifo_4_empty;
  wire        stageClearVec_WaitReadQueue_4_full;
  assign stageClearVec_WaitReadQueue_4_full = _stageClearVec_WaitReadQueue_fifo_4_full;
  wire        stageClearVec_reqQueue_5_empty;
  assign stageClearVec_reqQueue_5_empty = _stageClearVec_reqQueue_fifo_5_empty;
  wire        stageClearVec_reqQueue_5_full;
  assign stageClearVec_reqQueue_5_full = _stageClearVec_reqQueue_fifo_5_full;
  wire        stageClearVec_WaitReadQueue_5_empty;
  assign stageClearVec_WaitReadQueue_5_empty = _stageClearVec_WaitReadQueue_fifo_5_empty;
  wire        stageClearVec_WaitReadQueue_5_full;
  assign stageClearVec_WaitReadQueue_5_full = _stageClearVec_WaitReadQueue_fifo_5_full;
  wire        stageClearVec_reqQueue_6_empty;
  assign stageClearVec_reqQueue_6_empty = _stageClearVec_reqQueue_fifo_6_empty;
  wire        stageClearVec_reqQueue_6_full;
  assign stageClearVec_reqQueue_6_full = _stageClearVec_reqQueue_fifo_6_full;
  wire        stageClearVec_WaitReadQueue_6_empty;
  assign stageClearVec_WaitReadQueue_6_empty = _stageClearVec_WaitReadQueue_fifo_6_empty;
  wire        stageClearVec_WaitReadQueue_6_full;
  assign stageClearVec_WaitReadQueue_6_full = _stageClearVec_WaitReadQueue_fifo_6_full;
  wire        stageClearVec_reqQueue_7_empty;
  assign stageClearVec_reqQueue_7_empty = _stageClearVec_reqQueue_fifo_7_empty;
  wire        stageClearVec_reqQueue_7_full;
  assign stageClearVec_reqQueue_7_full = _stageClearVec_reqQueue_fifo_7_full;
  wire        stageClearVec_WaitReadQueue_7_empty;
  assign stageClearVec_WaitReadQueue_7_empty = _stageClearVec_WaitReadQueue_fifo_7_empty;
  wire        stageClearVec_WaitReadQueue_7_full;
  assign stageClearVec_WaitReadQueue_7_full = _stageClearVec_WaitReadQueue_fifo_7_full;
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(76)
  ) stageClearVec_reqQueue_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(stageClearVec_reqQueue_enq_ready & stageClearVec_reqQueue_enq_valid)),
    .pop_req_n    (~(stageClearVec_reqQueue_deq_ready & ~_stageClearVec_reqQueue_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (stageClearVec_reqQueue_dataIn),
    .empty        (_stageClearVec_reqQueue_fifo_empty),
    .almost_empty (stageClearVec_reqQueue_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (stageClearVec_reqQueue_almostFull),
    .full         (_stageClearVec_reqQueue_fifo_full),
    .error        (_stageClearVec_reqQueue_fifo_error),
    .data_out     (_stageClearVec_reqQueue_fifo_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(76)
  ) stageClearVec_WaitReadQueue_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(stageClearVec_WaitReadQueue_enq_ready & stageClearVec_WaitReadQueue_enq_valid)),
    .pop_req_n    (~(stageClearVec_WaitReadQueue_deq_ready & ~_stageClearVec_WaitReadQueue_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (stageClearVec_WaitReadQueue_dataIn),
    .empty        (_stageClearVec_WaitReadQueue_fifo_empty),
    .almost_empty (stageClearVec_WaitReadQueue_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (stageClearVec_WaitReadQueue_almostFull),
    .full         (_stageClearVec_WaitReadQueue_fifo_full),
    .error        (_stageClearVec_WaitReadQueue_fifo_error),
    .data_out     (_stageClearVec_WaitReadQueue_fifo_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(76)
  ) stageClearVec_reqQueue_fifo_1 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(stageClearVec_reqQueue_1_enq_ready & stageClearVec_reqQueue_1_enq_valid)),
    .pop_req_n    (~(stageClearVec_reqQueue_1_deq_ready & ~_stageClearVec_reqQueue_fifo_1_empty)),
    .diag_n       (1'h1),
    .data_in      (stageClearVec_reqQueue_dataIn_1),
    .empty        (_stageClearVec_reqQueue_fifo_1_empty),
    .almost_empty (stageClearVec_reqQueue_1_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (stageClearVec_reqQueue_1_almostFull),
    .full         (_stageClearVec_reqQueue_fifo_1_full),
    .error        (_stageClearVec_reqQueue_fifo_1_error),
    .data_out     (_stageClearVec_reqQueue_fifo_1_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(76)
  ) stageClearVec_WaitReadQueue_fifo_1 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(stageClearVec_WaitReadQueue_1_enq_ready & stageClearVec_WaitReadQueue_1_enq_valid)),
    .pop_req_n    (~(stageClearVec_WaitReadQueue_1_deq_ready & ~_stageClearVec_WaitReadQueue_fifo_1_empty)),
    .diag_n       (1'h1),
    .data_in      (stageClearVec_WaitReadQueue_dataIn_1),
    .empty        (_stageClearVec_WaitReadQueue_fifo_1_empty),
    .almost_empty (stageClearVec_WaitReadQueue_1_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (stageClearVec_WaitReadQueue_1_almostFull),
    .full         (_stageClearVec_WaitReadQueue_fifo_1_full),
    .error        (_stageClearVec_WaitReadQueue_fifo_1_error),
    .data_out     (_stageClearVec_WaitReadQueue_fifo_1_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(76)
  ) stageClearVec_reqQueue_fifo_2 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(stageClearVec_reqQueue_2_enq_ready & stageClearVec_reqQueue_2_enq_valid)),
    .pop_req_n    (~(stageClearVec_reqQueue_2_deq_ready & ~_stageClearVec_reqQueue_fifo_2_empty)),
    .diag_n       (1'h1),
    .data_in      (stageClearVec_reqQueue_dataIn_2),
    .empty        (_stageClearVec_reqQueue_fifo_2_empty),
    .almost_empty (stageClearVec_reqQueue_2_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (stageClearVec_reqQueue_2_almostFull),
    .full         (_stageClearVec_reqQueue_fifo_2_full),
    .error        (_stageClearVec_reqQueue_fifo_2_error),
    .data_out     (_stageClearVec_reqQueue_fifo_2_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(76)
  ) stageClearVec_WaitReadQueue_fifo_2 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(stageClearVec_WaitReadQueue_2_enq_ready & stageClearVec_WaitReadQueue_2_enq_valid)),
    .pop_req_n    (~(stageClearVec_WaitReadQueue_2_deq_ready & ~_stageClearVec_WaitReadQueue_fifo_2_empty)),
    .diag_n       (1'h1),
    .data_in      (stageClearVec_WaitReadQueue_dataIn_2),
    .empty        (_stageClearVec_WaitReadQueue_fifo_2_empty),
    .almost_empty (stageClearVec_WaitReadQueue_2_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (stageClearVec_WaitReadQueue_2_almostFull),
    .full         (_stageClearVec_WaitReadQueue_fifo_2_full),
    .error        (_stageClearVec_WaitReadQueue_fifo_2_error),
    .data_out     (_stageClearVec_WaitReadQueue_fifo_2_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(76)
  ) stageClearVec_reqQueue_fifo_3 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(stageClearVec_reqQueue_3_enq_ready & stageClearVec_reqQueue_3_enq_valid)),
    .pop_req_n    (~(stageClearVec_reqQueue_3_deq_ready & ~_stageClearVec_reqQueue_fifo_3_empty)),
    .diag_n       (1'h1),
    .data_in      (stageClearVec_reqQueue_dataIn_3),
    .empty        (_stageClearVec_reqQueue_fifo_3_empty),
    .almost_empty (stageClearVec_reqQueue_3_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (stageClearVec_reqQueue_3_almostFull),
    .full         (_stageClearVec_reqQueue_fifo_3_full),
    .error        (_stageClearVec_reqQueue_fifo_3_error),
    .data_out     (_stageClearVec_reqQueue_fifo_3_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(76)
  ) stageClearVec_WaitReadQueue_fifo_3 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(stageClearVec_WaitReadQueue_3_enq_ready & stageClearVec_WaitReadQueue_3_enq_valid)),
    .pop_req_n    (~(stageClearVec_WaitReadQueue_3_deq_ready & ~_stageClearVec_WaitReadQueue_fifo_3_empty)),
    .diag_n       (1'h1),
    .data_in      (stageClearVec_WaitReadQueue_dataIn_3),
    .empty        (_stageClearVec_WaitReadQueue_fifo_3_empty),
    .almost_empty (stageClearVec_WaitReadQueue_3_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (stageClearVec_WaitReadQueue_3_almostFull),
    .full         (_stageClearVec_WaitReadQueue_fifo_3_full),
    .error        (_stageClearVec_WaitReadQueue_fifo_3_error),
    .data_out     (_stageClearVec_WaitReadQueue_fifo_3_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(76)
  ) stageClearVec_reqQueue_fifo_4 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(stageClearVec_reqQueue_4_enq_ready & stageClearVec_reqQueue_4_enq_valid)),
    .pop_req_n    (~(stageClearVec_reqQueue_4_deq_ready & ~_stageClearVec_reqQueue_fifo_4_empty)),
    .diag_n       (1'h1),
    .data_in      (stageClearVec_reqQueue_dataIn_4),
    .empty        (_stageClearVec_reqQueue_fifo_4_empty),
    .almost_empty (stageClearVec_reqQueue_4_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (stageClearVec_reqQueue_4_almostFull),
    .full         (_stageClearVec_reqQueue_fifo_4_full),
    .error        (_stageClearVec_reqQueue_fifo_4_error),
    .data_out     (_stageClearVec_reqQueue_fifo_4_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(76)
  ) stageClearVec_WaitReadQueue_fifo_4 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(stageClearVec_WaitReadQueue_4_enq_ready & stageClearVec_WaitReadQueue_4_enq_valid)),
    .pop_req_n    (~(stageClearVec_WaitReadQueue_4_deq_ready & ~_stageClearVec_WaitReadQueue_fifo_4_empty)),
    .diag_n       (1'h1),
    .data_in      (stageClearVec_WaitReadQueue_dataIn_4),
    .empty        (_stageClearVec_WaitReadQueue_fifo_4_empty),
    .almost_empty (stageClearVec_WaitReadQueue_4_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (stageClearVec_WaitReadQueue_4_almostFull),
    .full         (_stageClearVec_WaitReadQueue_fifo_4_full),
    .error        (_stageClearVec_WaitReadQueue_fifo_4_error),
    .data_out     (_stageClearVec_WaitReadQueue_fifo_4_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(76)
  ) stageClearVec_reqQueue_fifo_5 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(stageClearVec_reqQueue_5_enq_ready & stageClearVec_reqQueue_5_enq_valid)),
    .pop_req_n    (~(stageClearVec_reqQueue_5_deq_ready & ~_stageClearVec_reqQueue_fifo_5_empty)),
    .diag_n       (1'h1),
    .data_in      (stageClearVec_reqQueue_dataIn_5),
    .empty        (_stageClearVec_reqQueue_fifo_5_empty),
    .almost_empty (stageClearVec_reqQueue_5_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (stageClearVec_reqQueue_5_almostFull),
    .full         (_stageClearVec_reqQueue_fifo_5_full),
    .error        (_stageClearVec_reqQueue_fifo_5_error),
    .data_out     (_stageClearVec_reqQueue_fifo_5_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(76)
  ) stageClearVec_WaitReadQueue_fifo_5 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(stageClearVec_WaitReadQueue_5_enq_ready & stageClearVec_WaitReadQueue_5_enq_valid)),
    .pop_req_n    (~(stageClearVec_WaitReadQueue_5_deq_ready & ~_stageClearVec_WaitReadQueue_fifo_5_empty)),
    .diag_n       (1'h1),
    .data_in      (stageClearVec_WaitReadQueue_dataIn_5),
    .empty        (_stageClearVec_WaitReadQueue_fifo_5_empty),
    .almost_empty (stageClearVec_WaitReadQueue_5_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (stageClearVec_WaitReadQueue_5_almostFull),
    .full         (_stageClearVec_WaitReadQueue_fifo_5_full),
    .error        (_stageClearVec_WaitReadQueue_fifo_5_error),
    .data_out     (_stageClearVec_WaitReadQueue_fifo_5_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(76)
  ) stageClearVec_reqQueue_fifo_6 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(stageClearVec_reqQueue_6_enq_ready & stageClearVec_reqQueue_6_enq_valid)),
    .pop_req_n    (~(stageClearVec_reqQueue_6_deq_ready & ~_stageClearVec_reqQueue_fifo_6_empty)),
    .diag_n       (1'h1),
    .data_in      (stageClearVec_reqQueue_dataIn_6),
    .empty        (_stageClearVec_reqQueue_fifo_6_empty),
    .almost_empty (stageClearVec_reqQueue_6_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (stageClearVec_reqQueue_6_almostFull),
    .full         (_stageClearVec_reqQueue_fifo_6_full),
    .error        (_stageClearVec_reqQueue_fifo_6_error),
    .data_out     (_stageClearVec_reqQueue_fifo_6_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(76)
  ) stageClearVec_WaitReadQueue_fifo_6 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(stageClearVec_WaitReadQueue_6_enq_ready & stageClearVec_WaitReadQueue_6_enq_valid)),
    .pop_req_n    (~(stageClearVec_WaitReadQueue_6_deq_ready & ~_stageClearVec_WaitReadQueue_fifo_6_empty)),
    .diag_n       (1'h1),
    .data_in      (stageClearVec_WaitReadQueue_dataIn_6),
    .empty        (_stageClearVec_WaitReadQueue_fifo_6_empty),
    .almost_empty (stageClearVec_WaitReadQueue_6_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (stageClearVec_WaitReadQueue_6_almostFull),
    .full         (_stageClearVec_WaitReadQueue_fifo_6_full),
    .error        (_stageClearVec_WaitReadQueue_fifo_6_error),
    .data_out     (_stageClearVec_WaitReadQueue_fifo_6_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(76)
  ) stageClearVec_reqQueue_fifo_7 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(stageClearVec_reqQueue_7_enq_ready & stageClearVec_reqQueue_7_enq_valid)),
    .pop_req_n    (~(stageClearVec_reqQueue_7_deq_ready & ~_stageClearVec_reqQueue_fifo_7_empty)),
    .diag_n       (1'h1),
    .data_in      (stageClearVec_reqQueue_dataIn_7),
    .empty        (_stageClearVec_reqQueue_fifo_7_empty),
    .almost_empty (stageClearVec_reqQueue_7_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (stageClearVec_reqQueue_7_almostFull),
    .full         (_stageClearVec_reqQueue_fifo_7_full),
    .error        (_stageClearVec_reqQueue_fifo_7_error),
    .data_out     (_stageClearVec_reqQueue_fifo_7_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(76)
  ) stageClearVec_WaitReadQueue_fifo_7 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(stageClearVec_WaitReadQueue_7_enq_ready & stageClearVec_WaitReadQueue_7_enq_valid)),
    .pop_req_n    (~(stageClearVec_WaitReadQueue_7_deq_ready & ~_stageClearVec_WaitReadQueue_fifo_7_empty)),
    .diag_n       (1'h1),
    .data_in      (stageClearVec_WaitReadQueue_dataIn_7),
    .empty        (_stageClearVec_WaitReadQueue_fifo_7_empty),
    .almost_empty (stageClearVec_WaitReadQueue_7_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (stageClearVec_WaitReadQueue_7_almostFull),
    .full         (_stageClearVec_WaitReadQueue_fifo_7_full),
    .error        (_stageClearVec_WaitReadQueue_fifo_7_error),
    .data_out     (_stageClearVec_WaitReadQueue_fifo_7_data_out)
  );
  assign in_0_ready = in_0_ready_0;
  assign in_1_ready = in_1_ready_0;
  assign in_2_ready = in_2_ready_0;
  assign in_3_ready = in_3_ready_0;
  assign in_4_ready = in_4_ready_0;
  assign in_5_ready = in_5_ready_0;
  assign in_6_ready = in_6_ready_0;
  assign in_7_ready = in_7_ready_0;
  assign out_0_valid = out_0_valid_0;
  assign out_0_bits_ffoByOther = out_0_bits_ffoByOther_0;
  assign out_0_bits_writeData_data = out_0_bits_writeData_data_0;
  assign out_0_bits_writeData_mask = out_0_bits_writeData_mask_0;
  assign out_0_bits_writeData_groupCounter = out_0_bits_writeData_groupCounter_0;
  assign out_1_valid = out_1_valid_0;
  assign out_1_bits_ffoByOther = out_1_bits_ffoByOther_0;
  assign out_1_bits_writeData_data = out_1_bits_writeData_data_0;
  assign out_1_bits_writeData_mask = out_1_bits_writeData_mask_0;
  assign out_1_bits_writeData_groupCounter = out_1_bits_writeData_groupCounter_0;
  assign out_2_valid = out_2_valid_0;
  assign out_2_bits_ffoByOther = out_2_bits_ffoByOther_0;
  assign out_2_bits_writeData_data = out_2_bits_writeData_data_0;
  assign out_2_bits_writeData_mask = out_2_bits_writeData_mask_0;
  assign out_2_bits_writeData_groupCounter = out_2_bits_writeData_groupCounter_0;
  assign out_3_valid = out_3_valid_0;
  assign out_3_bits_ffoByOther = out_3_bits_ffoByOther_0;
  assign out_3_bits_writeData_data = out_3_bits_writeData_data_0;
  assign out_3_bits_writeData_mask = out_3_bits_writeData_mask_0;
  assign out_3_bits_writeData_groupCounter = out_3_bits_writeData_groupCounter_0;
  assign out_4_valid = out_4_valid_0;
  assign out_4_bits_ffoByOther = out_4_bits_ffoByOther_0;
  assign out_4_bits_writeData_data = out_4_bits_writeData_data_0;
  assign out_4_bits_writeData_mask = out_4_bits_writeData_mask_0;
  assign out_4_bits_writeData_groupCounter = out_4_bits_writeData_groupCounter_0;
  assign out_5_valid = out_5_valid_0;
  assign out_5_bits_ffoByOther = out_5_bits_ffoByOther_0;
  assign out_5_bits_writeData_data = out_5_bits_writeData_data_0;
  assign out_5_bits_writeData_mask = out_5_bits_writeData_mask_0;
  assign out_5_bits_writeData_groupCounter = out_5_bits_writeData_groupCounter_0;
  assign out_6_valid = out_6_valid_0;
  assign out_6_bits_ffoByOther = out_6_bits_ffoByOther_0;
  assign out_6_bits_writeData_data = out_6_bits_writeData_data_0;
  assign out_6_bits_writeData_mask = out_6_bits_writeData_mask_0;
  assign out_6_bits_writeData_groupCounter = out_6_bits_writeData_groupCounter_0;
  assign out_7_valid = out_7_valid_0;
  assign out_7_bits_ffoByOther = out_7_bits_ffoByOther_0;
  assign out_7_bits_writeData_data = out_7_bits_writeData_data_0;
  assign out_7_bits_writeData_mask = out_7_bits_writeData_mask_0;
  assign out_7_bits_writeData_groupCounter = out_7_bits_writeData_groupCounter_0;
  assign readChannel_0_valid = readChannel_0_valid_0;
  assign readChannel_0_bits_vs = readChannel_0_bits_vs_0;
  assign readChannel_0_bits_offset = readChannel_0_bits_offset_0;
  assign readChannel_1_valid = readChannel_1_valid_0;
  assign readChannel_1_bits_vs = readChannel_1_bits_vs_0;
  assign readChannel_1_bits_offset = readChannel_1_bits_offset_0;
  assign readChannel_2_valid = readChannel_2_valid_0;
  assign readChannel_2_bits_vs = readChannel_2_bits_vs_0;
  assign readChannel_2_bits_offset = readChannel_2_bits_offset_0;
  assign readChannel_3_valid = readChannel_3_valid_0;
  assign readChannel_3_bits_vs = readChannel_3_bits_vs_0;
  assign readChannel_3_bits_offset = readChannel_3_bits_offset_0;
  assign readChannel_4_valid = readChannel_4_valid_0;
  assign readChannel_4_bits_vs = readChannel_4_bits_vs_0;
  assign readChannel_4_bits_offset = readChannel_4_bits_offset_0;
  assign readChannel_5_valid = readChannel_5_valid_0;
  assign readChannel_5_bits_vs = readChannel_5_bits_vs_0;
  assign readChannel_5_bits_offset = readChannel_5_bits_offset_0;
  assign readChannel_6_valid = readChannel_6_valid_0;
  assign readChannel_6_bits_vs = readChannel_6_bits_vs_0;
  assign readChannel_6_bits_offset = readChannel_6_bits_offset_0;
  assign readChannel_7_valid = readChannel_7_valid_0;
  assign readChannel_7_bits_vs = readChannel_7_bits_vs_0;
  assign readChannel_7_bits_offset = readChannel_7_bits_offset_0;
  assign stageClear = stageClearVec_0 & stageClearVec_1 & stageClearVec_2 & stageClearVec_3 & stageClearVec_4 & stageClearVec_5 & stageClearVec_6 & stageClearVec_7;
endmodule

