
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
  input  [10:0] in_0_bits_groupCounter,
  input         in_0_bits_ffoByOther,
  output        in_1_ready,
  input         in_1_valid,
  input  [31:0] in_1_bits_data,
                in_1_bits_bitMask,
  input  [3:0]  in_1_bits_mask,
  input  [10:0] in_1_bits_groupCounter,
  input         in_1_bits_ffoByOther,
  output        in_2_ready,
  input         in_2_valid,
  input  [31:0] in_2_bits_data,
                in_2_bits_bitMask,
  input  [3:0]  in_2_bits_mask,
  input  [10:0] in_2_bits_groupCounter,
  input         in_2_bits_ffoByOther,
  output        in_3_ready,
  input         in_3_valid,
  input  [31:0] in_3_bits_data,
                in_3_bits_bitMask,
  input  [3:0]  in_3_bits_mask,
  input  [10:0] in_3_bits_groupCounter,
  input         in_3_bits_ffoByOther,
                out_0_ready,
  output        out_0_valid,
                out_0_bits_ffoByOther,
  output [31:0] out_0_bits_writeData_data,
  output [3:0]  out_0_bits_writeData_mask,
  output [10:0] out_0_bits_writeData_groupCounter,
  input         out_1_ready,
  output        out_1_valid,
                out_1_bits_ffoByOther,
  output [31:0] out_1_bits_writeData_data,
  output [3:0]  out_1_bits_writeData_mask,
  output [10:0] out_1_bits_writeData_groupCounter,
  input         out_2_ready,
  output        out_2_valid,
                out_2_bits_ffoByOther,
  output [31:0] out_2_bits_writeData_data,
  output [3:0]  out_2_bits_writeData_mask,
  output [10:0] out_2_bits_writeData_groupCounter,
  input         out_3_ready,
  output        out_3_valid,
                out_3_bits_ffoByOther,
  output [31:0] out_3_bits_writeData_data,
  output [3:0]  out_3_bits_writeData_mask,
  output [10:0] out_3_bits_writeData_groupCounter,
  input         readChannel_0_ready,
  output        readChannel_0_valid,
  output [4:0]  readChannel_0_bits_vs,
  output [6:0]  readChannel_0_bits_offset,
  input         readChannel_1_ready,
  output        readChannel_1_valid,
  output [4:0]  readChannel_1_bits_vs,
  output [6:0]  readChannel_1_bits_offset,
  input         readChannel_2_ready,
  output        readChannel_2_valid,
  output [4:0]  readChannel_2_bits_vs,
  output [6:0]  readChannel_2_bits_offset,
  input         readChannel_3_ready,
  output        readChannel_3_valid,
  output [4:0]  readChannel_3_bits_vs,
  output [6:0]  readChannel_3_bits_offset,
  input         readResult_0_valid,
  input  [31:0] readResult_0_bits,
  input         readResult_1_valid,
  input  [31:0] readResult_1_bits,
  input         readResult_2_valid,
  input  [31:0] readResult_2_bits,
  input         readResult_3_valid,
  input  [31:0] readResult_3_bits,
  output        stageClear
);

  wire        _stageClearVec_WaitReadQueue_fifo_3_empty;
  wire        _stageClearVec_WaitReadQueue_fifo_3_full;
  wire        _stageClearVec_WaitReadQueue_fifo_3_error;
  wire [79:0] _stageClearVec_WaitReadQueue_fifo_3_data_out;
  wire        _stageClearVec_reqQueue_fifo_3_empty;
  wire        _stageClearVec_reqQueue_fifo_3_full;
  wire        _stageClearVec_reqQueue_fifo_3_error;
  wire [79:0] _stageClearVec_reqQueue_fifo_3_data_out;
  wire        _stageClearVec_WaitReadQueue_fifo_2_empty;
  wire        _stageClearVec_WaitReadQueue_fifo_2_full;
  wire        _stageClearVec_WaitReadQueue_fifo_2_error;
  wire [79:0] _stageClearVec_WaitReadQueue_fifo_2_data_out;
  wire        _stageClearVec_reqQueue_fifo_2_empty;
  wire        _stageClearVec_reqQueue_fifo_2_full;
  wire        _stageClearVec_reqQueue_fifo_2_error;
  wire [79:0] _stageClearVec_reqQueue_fifo_2_data_out;
  wire        _stageClearVec_WaitReadQueue_fifo_1_empty;
  wire        _stageClearVec_WaitReadQueue_fifo_1_full;
  wire        _stageClearVec_WaitReadQueue_fifo_1_error;
  wire [79:0] _stageClearVec_WaitReadQueue_fifo_1_data_out;
  wire        _stageClearVec_reqQueue_fifo_1_empty;
  wire        _stageClearVec_reqQueue_fifo_1_full;
  wire        _stageClearVec_reqQueue_fifo_1_error;
  wire [79:0] _stageClearVec_reqQueue_fifo_1_data_out;
  wire        _stageClearVec_WaitReadQueue_fifo_empty;
  wire        _stageClearVec_WaitReadQueue_fifo_full;
  wire        _stageClearVec_WaitReadQueue_fifo_error;
  wire [79:0] _stageClearVec_WaitReadQueue_fifo_data_out;
  wire        _stageClearVec_reqQueue_fifo_empty;
  wire        _stageClearVec_reqQueue_fifo_full;
  wire        _stageClearVec_reqQueue_fifo_error;
  wire [79:0] _stageClearVec_reqQueue_fifo_data_out;
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
  wire        stageClearVec_reqQueue_3_deq_bits_ffoByOther;
  wire [10:0] stageClearVec_reqQueue_3_deq_bits_groupCounter;
  wire [3:0]  stageClearVec_reqQueue_3_deq_bits_mask;
  wire [31:0] stageClearVec_reqQueue_3_deq_bits_bitMask;
  wire [31:0] stageClearVec_reqQueue_3_deq_bits_data;
  wire        stageClearVec_reqQueue_2_deq_bits_ffoByOther;
  wire [10:0] stageClearVec_reqQueue_2_deq_bits_groupCounter;
  wire [3:0]  stageClearVec_reqQueue_2_deq_bits_mask;
  wire [31:0] stageClearVec_reqQueue_2_deq_bits_bitMask;
  wire [31:0] stageClearVec_reqQueue_2_deq_bits_data;
  wire        stageClearVec_reqQueue_1_deq_bits_ffoByOther;
  wire [10:0] stageClearVec_reqQueue_1_deq_bits_groupCounter;
  wire [3:0]  stageClearVec_reqQueue_1_deq_bits_mask;
  wire [31:0] stageClearVec_reqQueue_1_deq_bits_bitMask;
  wire [31:0] stageClearVec_reqQueue_1_deq_bits_data;
  wire        stageClearVec_reqQueue_deq_bits_ffoByOther;
  wire [10:0] stageClearVec_reqQueue_deq_bits_groupCounter;
  wire [3:0]  stageClearVec_reqQueue_deq_bits_mask;
  wire [31:0] stageClearVec_reqQueue_deq_bits_bitMask;
  wire [31:0] stageClearVec_reqQueue_deq_bits_data;
  wire        in_0_valid_0 = in_0_valid;
  wire [31:0] in_0_bits_data_0 = in_0_bits_data;
  wire [31:0] in_0_bits_bitMask_0 = in_0_bits_bitMask;
  wire [3:0]  in_0_bits_mask_0 = in_0_bits_mask;
  wire [10:0] in_0_bits_groupCounter_0 = in_0_bits_groupCounter;
  wire        in_0_bits_ffoByOther_0 = in_0_bits_ffoByOther;
  wire        in_1_valid_0 = in_1_valid;
  wire [31:0] in_1_bits_data_0 = in_1_bits_data;
  wire [31:0] in_1_bits_bitMask_0 = in_1_bits_bitMask;
  wire [3:0]  in_1_bits_mask_0 = in_1_bits_mask;
  wire [10:0] in_1_bits_groupCounter_0 = in_1_bits_groupCounter;
  wire        in_1_bits_ffoByOther_0 = in_1_bits_ffoByOther;
  wire        in_2_valid_0 = in_2_valid;
  wire [31:0] in_2_bits_data_0 = in_2_bits_data;
  wire [31:0] in_2_bits_bitMask_0 = in_2_bits_bitMask;
  wire [3:0]  in_2_bits_mask_0 = in_2_bits_mask;
  wire [10:0] in_2_bits_groupCounter_0 = in_2_bits_groupCounter;
  wire        in_2_bits_ffoByOther_0 = in_2_bits_ffoByOther;
  wire        in_3_valid_0 = in_3_valid;
  wire [31:0] in_3_bits_data_0 = in_3_bits_data;
  wire [31:0] in_3_bits_bitMask_0 = in_3_bits_bitMask;
  wire [3:0]  in_3_bits_mask_0 = in_3_bits_mask;
  wire [10:0] in_3_bits_groupCounter_0 = in_3_bits_groupCounter;
  wire        in_3_bits_ffoByOther_0 = in_3_bits_ffoByOther;
  wire        out_0_ready_0 = out_0_ready;
  wire        out_1_ready_0 = out_1_ready;
  wire        out_2_ready_0 = out_2_ready;
  wire        out_3_ready_0 = out_3_ready;
  wire        readChannel_0_ready_0 = readChannel_0_ready;
  wire        readChannel_1_ready_0 = readChannel_1_ready;
  wire        readChannel_2_ready_0 = readChannel_2_ready;
  wire        readChannel_3_ready_0 = readChannel_3_ready;
  wire [1:0]  readChannel_0_bits_readSource = 2'h0;
  wire [1:0]  readChannel_1_bits_readSource = 2'h0;
  wire [1:0]  readChannel_2_bits_readSource = 2'h0;
  wire [1:0]  readChannel_3_bits_readSource = 2'h0;
  wire [2:0]  out_0_bits_index = 3'h0;
  wire [2:0]  out_1_bits_index = 3'h0;
  wire [2:0]  out_2_bits_index = 3'h0;
  wire [2:0]  out_3_bits_index = 3'h0;
  wire [2:0]  readChannel_0_bits_instructionIndex = 3'h0;
  wire [2:0]  readChannel_1_bits_instructionIndex = 3'h0;
  wire [2:0]  readChannel_2_bits_instructionIndex = 3'h0;
  wire [2:0]  readChannel_3_bits_instructionIndex = 3'h0;
  wire [4:0]  out_0_bits_writeData_vd = 5'h0;
  wire [4:0]  out_1_bits_writeData_vd = 5'h0;
  wire [4:0]  out_2_bits_writeData_vd = 5'h0;
  wire [4:0]  out_3_bits_writeData_vd = 5'h0;
  wire        stageClearVec_reqQueue_enq_ready;
  wire        stageClearVec_reqQueue_enq_valid = in_0_valid_0;
  wire [31:0] stageClearVec_reqQueue_enq_bits_data = in_0_bits_data_0;
  wire [31:0] stageClearVec_reqQueue_enq_bits_bitMask = in_0_bits_bitMask_0;
  wire [3:0]  stageClearVec_reqQueue_enq_bits_mask = in_0_bits_mask_0;
  wire [10:0] stageClearVec_reqQueue_enq_bits_groupCounter = in_0_bits_groupCounter_0;
  wire        stageClearVec_reqQueue_enq_bits_ffoByOther = in_0_bits_ffoByOther_0;
  wire        stageClearVec_reqQueue_1_enq_ready;
  wire        stageClearVec_reqQueue_1_enq_valid = in_1_valid_0;
  wire [31:0] stageClearVec_reqQueue_1_enq_bits_data = in_1_bits_data_0;
  wire [31:0] stageClearVec_reqQueue_1_enq_bits_bitMask = in_1_bits_bitMask_0;
  wire [3:0]  stageClearVec_reqQueue_1_enq_bits_mask = in_1_bits_mask_0;
  wire [10:0] stageClearVec_reqQueue_1_enq_bits_groupCounter = in_1_bits_groupCounter_0;
  wire        stageClearVec_reqQueue_1_enq_bits_ffoByOther = in_1_bits_ffoByOther_0;
  wire        stageClearVec_reqQueue_2_enq_ready;
  wire        stageClearVec_reqQueue_2_enq_valid = in_2_valid_0;
  wire [31:0] stageClearVec_reqQueue_2_enq_bits_data = in_2_bits_data_0;
  wire [31:0] stageClearVec_reqQueue_2_enq_bits_bitMask = in_2_bits_bitMask_0;
  wire [3:0]  stageClearVec_reqQueue_2_enq_bits_mask = in_2_bits_mask_0;
  wire [10:0] stageClearVec_reqQueue_2_enq_bits_groupCounter = in_2_bits_groupCounter_0;
  wire        stageClearVec_reqQueue_2_enq_bits_ffoByOther = in_2_bits_ffoByOther_0;
  wire        stageClearVec_reqQueue_3_enq_ready;
  wire        stageClearVec_reqQueue_3_enq_valid = in_3_valid_0;
  wire [31:0] stageClearVec_reqQueue_3_enq_bits_data = in_3_bits_data_0;
  wire [31:0] stageClearVec_reqQueue_3_enq_bits_bitMask = in_3_bits_bitMask_0;
  wire [3:0]  stageClearVec_reqQueue_3_enq_bits_mask = in_3_bits_mask_0;
  wire [10:0] stageClearVec_reqQueue_3_enq_bits_groupCounter = in_3_bits_groupCounter_0;
  wire        stageClearVec_reqQueue_3_enq_bits_ffoByOther = in_3_bits_ffoByOther_0;
  wire        stageClearVec_WaitReadQueue_deq_bits_ffoByOther;
  wire [10:0] stageClearVec_WaitReadQueue_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_1_deq_bits_ffoByOther;
  wire [10:0] stageClearVec_WaitReadQueue_1_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_2_deq_bits_ffoByOther;
  wire [10:0] stageClearVec_WaitReadQueue_2_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_3_deq_bits_ffoByOther;
  wire [10:0] stageClearVec_WaitReadQueue_3_deq_bits_groupCounter;
  wire        in_0_ready_0 = stageClearVec_reqQueue_enq_ready;
  wire        stageClearVec_reqQueue_deq_valid;
  assign stageClearVec_reqQueue_deq_valid = ~_stageClearVec_reqQueue_fifo_empty;
  wire [31:0] stageClearVec_reqQueue_dataOut_data;
  wire [31:0] stageClearVec_reqQueue_dataOut_bitMask;
  wire [31:0] stageClearVec_WaitReadQueue_enq_bits_data = stageClearVec_reqQueue_deq_bits_data;
  wire [3:0]  stageClearVec_reqQueue_dataOut_mask;
  wire [31:0] stageClearVec_WaitReadQueue_enq_bits_bitMask = stageClearVec_reqQueue_deq_bits_bitMask;
  wire [10:0] stageClearVec_reqQueue_dataOut_groupCounter;
  wire [3:0]  stageClearVec_WaitReadQueue_enq_bits_mask = stageClearVec_reqQueue_deq_bits_mask;
  wire        stageClearVec_reqQueue_dataOut_ffoByOther;
  wire [10:0] stageClearVec_WaitReadQueue_enq_bits_groupCounter = stageClearVec_reqQueue_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_enq_bits_ffoByOther = stageClearVec_reqQueue_deq_bits_ffoByOther;
  wire [11:0] stageClearVec_reqQueue_dataIn_lo = {stageClearVec_reqQueue_enq_bits_groupCounter, stageClearVec_reqQueue_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_reqQueue_dataIn_hi_hi = {stageClearVec_reqQueue_enq_bits_data, stageClearVec_reqQueue_enq_bits_bitMask};
  wire [67:0] stageClearVec_reqQueue_dataIn_hi = {stageClearVec_reqQueue_dataIn_hi_hi, stageClearVec_reqQueue_enq_bits_mask};
  wire [79:0] stageClearVec_reqQueue_dataIn = {stageClearVec_reqQueue_dataIn_hi, stageClearVec_reqQueue_dataIn_lo};
  assign stageClearVec_reqQueue_dataOut_ffoByOther = _stageClearVec_reqQueue_fifo_data_out[0];
  assign stageClearVec_reqQueue_dataOut_groupCounter = _stageClearVec_reqQueue_fifo_data_out[11:1];
  assign stageClearVec_reqQueue_dataOut_mask = _stageClearVec_reqQueue_fifo_data_out[15:12];
  assign stageClearVec_reqQueue_dataOut_bitMask = _stageClearVec_reqQueue_fifo_data_out[47:16];
  assign stageClearVec_reqQueue_dataOut_data = _stageClearVec_reqQueue_fifo_data_out[79:48];
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
  wire [10:0] stageClearVec_WaitReadQueue_dataOut_groupCounter;
  wire [10:0] out_0_bits_writeData_groupCounter_0 = stageClearVec_WaitReadQueue_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_dataOut_ffoByOther;
  wire        out_0_bits_ffoByOther_0 = stageClearVec_WaitReadQueue_deq_bits_ffoByOther;
  wire [11:0] stageClearVec_WaitReadQueue_dataIn_lo = {stageClearVec_WaitReadQueue_enq_bits_groupCounter, stageClearVec_WaitReadQueue_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_WaitReadQueue_dataIn_hi_hi = {stageClearVec_WaitReadQueue_enq_bits_data, stageClearVec_WaitReadQueue_enq_bits_bitMask};
  wire [67:0] stageClearVec_WaitReadQueue_dataIn_hi = {stageClearVec_WaitReadQueue_dataIn_hi_hi, stageClearVec_WaitReadQueue_enq_bits_mask};
  wire [79:0] stageClearVec_WaitReadQueue_dataIn = {stageClearVec_WaitReadQueue_dataIn_hi, stageClearVec_WaitReadQueue_dataIn_lo};
  assign stageClearVec_WaitReadQueue_dataOut_ffoByOther = _stageClearVec_WaitReadQueue_fifo_data_out[0];
  assign stageClearVec_WaitReadQueue_dataOut_groupCounter = _stageClearVec_WaitReadQueue_fifo_data_out[11:1];
  assign stageClearVec_WaitReadQueue_dataOut_mask = _stageClearVec_WaitReadQueue_fifo_data_out[15:12];
  assign stageClearVec_WaitReadQueue_dataOut_bitMask = _stageClearVec_WaitReadQueue_fifo_data_out[47:16];
  assign stageClearVec_WaitReadQueue_dataOut_data = _stageClearVec_WaitReadQueue_fifo_data_out[79:48];
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
  wire [4:0]  readChannel_0_bits_vs_0 = vd + {1'h0, stageClearVec_reqQueue_deq_bits_groupCounter[10:7]};
  wire [6:0]  readChannel_0_bits_offset_0 = stageClearVec_reqQueue_deq_bits_groupCounter[6:0];
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
  wire [10:0] stageClearVec_reqQueue_dataOut_1_groupCounter;
  wire [3:0]  stageClearVec_WaitReadQueue_1_enq_bits_mask = stageClearVec_reqQueue_1_deq_bits_mask;
  wire        stageClearVec_reqQueue_dataOut_1_ffoByOther;
  wire [10:0] stageClearVec_WaitReadQueue_1_enq_bits_groupCounter = stageClearVec_reqQueue_1_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_1_enq_bits_ffoByOther = stageClearVec_reqQueue_1_deq_bits_ffoByOther;
  wire [11:0] stageClearVec_reqQueue_dataIn_lo_1 = {stageClearVec_reqQueue_1_enq_bits_groupCounter, stageClearVec_reqQueue_1_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_reqQueue_dataIn_hi_hi_1 = {stageClearVec_reqQueue_1_enq_bits_data, stageClearVec_reqQueue_1_enq_bits_bitMask};
  wire [67:0] stageClearVec_reqQueue_dataIn_hi_1 = {stageClearVec_reqQueue_dataIn_hi_hi_1, stageClearVec_reqQueue_1_enq_bits_mask};
  wire [79:0] stageClearVec_reqQueue_dataIn_1 = {stageClearVec_reqQueue_dataIn_hi_1, stageClearVec_reqQueue_dataIn_lo_1};
  assign stageClearVec_reqQueue_dataOut_1_ffoByOther = _stageClearVec_reqQueue_fifo_1_data_out[0];
  assign stageClearVec_reqQueue_dataOut_1_groupCounter = _stageClearVec_reqQueue_fifo_1_data_out[11:1];
  assign stageClearVec_reqQueue_dataOut_1_mask = _stageClearVec_reqQueue_fifo_1_data_out[15:12];
  assign stageClearVec_reqQueue_dataOut_1_bitMask = _stageClearVec_reqQueue_fifo_1_data_out[47:16];
  assign stageClearVec_reqQueue_dataOut_1_data = _stageClearVec_reqQueue_fifo_1_data_out[79:48];
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
  wire [10:0] stageClearVec_WaitReadQueue_dataOut_1_groupCounter;
  wire [10:0] out_1_bits_writeData_groupCounter_0 = stageClearVec_WaitReadQueue_1_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_dataOut_1_ffoByOther;
  wire        out_1_bits_ffoByOther_0 = stageClearVec_WaitReadQueue_1_deq_bits_ffoByOther;
  wire [11:0] stageClearVec_WaitReadQueue_dataIn_lo_1 = {stageClearVec_WaitReadQueue_1_enq_bits_groupCounter, stageClearVec_WaitReadQueue_1_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_WaitReadQueue_dataIn_hi_hi_1 = {stageClearVec_WaitReadQueue_1_enq_bits_data, stageClearVec_WaitReadQueue_1_enq_bits_bitMask};
  wire [67:0] stageClearVec_WaitReadQueue_dataIn_hi_1 = {stageClearVec_WaitReadQueue_dataIn_hi_hi_1, stageClearVec_WaitReadQueue_1_enq_bits_mask};
  wire [79:0] stageClearVec_WaitReadQueue_dataIn_1 = {stageClearVec_WaitReadQueue_dataIn_hi_1, stageClearVec_WaitReadQueue_dataIn_lo_1};
  assign stageClearVec_WaitReadQueue_dataOut_1_ffoByOther = _stageClearVec_WaitReadQueue_fifo_1_data_out[0];
  assign stageClearVec_WaitReadQueue_dataOut_1_groupCounter = _stageClearVec_WaitReadQueue_fifo_1_data_out[11:1];
  assign stageClearVec_WaitReadQueue_dataOut_1_mask = _stageClearVec_WaitReadQueue_fifo_1_data_out[15:12];
  assign stageClearVec_WaitReadQueue_dataOut_1_bitMask = _stageClearVec_WaitReadQueue_fifo_1_data_out[47:16];
  assign stageClearVec_WaitReadQueue_dataOut_1_data = _stageClearVec_WaitReadQueue_fifo_1_data_out[79:48];
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
  wire [4:0]  readChannel_1_bits_vs_0 = vd + {1'h0, stageClearVec_reqQueue_1_deq_bits_groupCounter[10:7]};
  wire [6:0]  readChannel_1_bits_offset_0 = stageClearVec_reqQueue_1_deq_bits_groupCounter[6:0];
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
  wire [10:0] stageClearVec_reqQueue_dataOut_2_groupCounter;
  wire [3:0]  stageClearVec_WaitReadQueue_2_enq_bits_mask = stageClearVec_reqQueue_2_deq_bits_mask;
  wire        stageClearVec_reqQueue_dataOut_2_ffoByOther;
  wire [10:0] stageClearVec_WaitReadQueue_2_enq_bits_groupCounter = stageClearVec_reqQueue_2_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_2_enq_bits_ffoByOther = stageClearVec_reqQueue_2_deq_bits_ffoByOther;
  wire [11:0] stageClearVec_reqQueue_dataIn_lo_2 = {stageClearVec_reqQueue_2_enq_bits_groupCounter, stageClearVec_reqQueue_2_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_reqQueue_dataIn_hi_hi_2 = {stageClearVec_reqQueue_2_enq_bits_data, stageClearVec_reqQueue_2_enq_bits_bitMask};
  wire [67:0] stageClearVec_reqQueue_dataIn_hi_2 = {stageClearVec_reqQueue_dataIn_hi_hi_2, stageClearVec_reqQueue_2_enq_bits_mask};
  wire [79:0] stageClearVec_reqQueue_dataIn_2 = {stageClearVec_reqQueue_dataIn_hi_2, stageClearVec_reqQueue_dataIn_lo_2};
  assign stageClearVec_reqQueue_dataOut_2_ffoByOther = _stageClearVec_reqQueue_fifo_2_data_out[0];
  assign stageClearVec_reqQueue_dataOut_2_groupCounter = _stageClearVec_reqQueue_fifo_2_data_out[11:1];
  assign stageClearVec_reqQueue_dataOut_2_mask = _stageClearVec_reqQueue_fifo_2_data_out[15:12];
  assign stageClearVec_reqQueue_dataOut_2_bitMask = _stageClearVec_reqQueue_fifo_2_data_out[47:16];
  assign stageClearVec_reqQueue_dataOut_2_data = _stageClearVec_reqQueue_fifo_2_data_out[79:48];
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
  wire [10:0] stageClearVec_WaitReadQueue_dataOut_2_groupCounter;
  wire [10:0] out_2_bits_writeData_groupCounter_0 = stageClearVec_WaitReadQueue_2_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_dataOut_2_ffoByOther;
  wire        out_2_bits_ffoByOther_0 = stageClearVec_WaitReadQueue_2_deq_bits_ffoByOther;
  wire [11:0] stageClearVec_WaitReadQueue_dataIn_lo_2 = {stageClearVec_WaitReadQueue_2_enq_bits_groupCounter, stageClearVec_WaitReadQueue_2_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_WaitReadQueue_dataIn_hi_hi_2 = {stageClearVec_WaitReadQueue_2_enq_bits_data, stageClearVec_WaitReadQueue_2_enq_bits_bitMask};
  wire [67:0] stageClearVec_WaitReadQueue_dataIn_hi_2 = {stageClearVec_WaitReadQueue_dataIn_hi_hi_2, stageClearVec_WaitReadQueue_2_enq_bits_mask};
  wire [79:0] stageClearVec_WaitReadQueue_dataIn_2 = {stageClearVec_WaitReadQueue_dataIn_hi_2, stageClearVec_WaitReadQueue_dataIn_lo_2};
  assign stageClearVec_WaitReadQueue_dataOut_2_ffoByOther = _stageClearVec_WaitReadQueue_fifo_2_data_out[0];
  assign stageClearVec_WaitReadQueue_dataOut_2_groupCounter = _stageClearVec_WaitReadQueue_fifo_2_data_out[11:1];
  assign stageClearVec_WaitReadQueue_dataOut_2_mask = _stageClearVec_WaitReadQueue_fifo_2_data_out[15:12];
  assign stageClearVec_WaitReadQueue_dataOut_2_bitMask = _stageClearVec_WaitReadQueue_fifo_2_data_out[47:16];
  assign stageClearVec_WaitReadQueue_dataOut_2_data = _stageClearVec_WaitReadQueue_fifo_2_data_out[79:48];
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
  wire [4:0]  readChannel_2_bits_vs_0 = vd + {1'h0, stageClearVec_reqQueue_2_deq_bits_groupCounter[10:7]};
  wire [6:0]  readChannel_2_bits_offset_0 = stageClearVec_reqQueue_2_deq_bits_groupCounter[6:0];
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
  wire [10:0] stageClearVec_reqQueue_dataOut_3_groupCounter;
  wire [3:0]  stageClearVec_WaitReadQueue_3_enq_bits_mask = stageClearVec_reqQueue_3_deq_bits_mask;
  wire        stageClearVec_reqQueue_dataOut_3_ffoByOther;
  wire [10:0] stageClearVec_WaitReadQueue_3_enq_bits_groupCounter = stageClearVec_reqQueue_3_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_3_enq_bits_ffoByOther = stageClearVec_reqQueue_3_deq_bits_ffoByOther;
  wire [11:0] stageClearVec_reqQueue_dataIn_lo_3 = {stageClearVec_reqQueue_3_enq_bits_groupCounter, stageClearVec_reqQueue_3_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_reqQueue_dataIn_hi_hi_3 = {stageClearVec_reqQueue_3_enq_bits_data, stageClearVec_reqQueue_3_enq_bits_bitMask};
  wire [67:0] stageClearVec_reqQueue_dataIn_hi_3 = {stageClearVec_reqQueue_dataIn_hi_hi_3, stageClearVec_reqQueue_3_enq_bits_mask};
  wire [79:0] stageClearVec_reqQueue_dataIn_3 = {stageClearVec_reqQueue_dataIn_hi_3, stageClearVec_reqQueue_dataIn_lo_3};
  assign stageClearVec_reqQueue_dataOut_3_ffoByOther = _stageClearVec_reqQueue_fifo_3_data_out[0];
  assign stageClearVec_reqQueue_dataOut_3_groupCounter = _stageClearVec_reqQueue_fifo_3_data_out[11:1];
  assign stageClearVec_reqQueue_dataOut_3_mask = _stageClearVec_reqQueue_fifo_3_data_out[15:12];
  assign stageClearVec_reqQueue_dataOut_3_bitMask = _stageClearVec_reqQueue_fifo_3_data_out[47:16];
  assign stageClearVec_reqQueue_dataOut_3_data = _stageClearVec_reqQueue_fifo_3_data_out[79:48];
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
  wire [10:0] stageClearVec_WaitReadQueue_dataOut_3_groupCounter;
  wire [10:0] out_3_bits_writeData_groupCounter_0 = stageClearVec_WaitReadQueue_3_deq_bits_groupCounter;
  wire        stageClearVec_WaitReadQueue_dataOut_3_ffoByOther;
  wire        out_3_bits_ffoByOther_0 = stageClearVec_WaitReadQueue_3_deq_bits_ffoByOther;
  wire [11:0] stageClearVec_WaitReadQueue_dataIn_lo_3 = {stageClearVec_WaitReadQueue_3_enq_bits_groupCounter, stageClearVec_WaitReadQueue_3_enq_bits_ffoByOther};
  wire [63:0] stageClearVec_WaitReadQueue_dataIn_hi_hi_3 = {stageClearVec_WaitReadQueue_3_enq_bits_data, stageClearVec_WaitReadQueue_3_enq_bits_bitMask};
  wire [67:0] stageClearVec_WaitReadQueue_dataIn_hi_3 = {stageClearVec_WaitReadQueue_dataIn_hi_hi_3, stageClearVec_WaitReadQueue_3_enq_bits_mask};
  wire [79:0] stageClearVec_WaitReadQueue_dataIn_3 = {stageClearVec_WaitReadQueue_dataIn_hi_3, stageClearVec_WaitReadQueue_dataIn_lo_3};
  assign stageClearVec_WaitReadQueue_dataOut_3_ffoByOther = _stageClearVec_WaitReadQueue_fifo_3_data_out[0];
  assign stageClearVec_WaitReadQueue_dataOut_3_groupCounter = _stageClearVec_WaitReadQueue_fifo_3_data_out[11:1];
  assign stageClearVec_WaitReadQueue_dataOut_3_mask = _stageClearVec_WaitReadQueue_fifo_3_data_out[15:12];
  assign stageClearVec_WaitReadQueue_dataOut_3_bitMask = _stageClearVec_WaitReadQueue_fifo_3_data_out[47:16];
  assign stageClearVec_WaitReadQueue_dataOut_3_data = _stageClearVec_WaitReadQueue_fifo_3_data_out[79:48];
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
  wire [4:0]  readChannel_3_bits_vs_0 = vd + {1'h0, stageClearVec_reqQueue_3_deq_bits_groupCounter[10:7]};
  wire [6:0]  readChannel_3_bits_offset_0 = stageClearVec_reqQueue_3_deq_bits_groupCounter[6:0];
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
  always @(posedge clock) begin
    if (reset) begin
      stageClearVec_counter <= 3'h0;
      stageClearVec_counter_1 <= 3'h0;
      stageClearVec_counter_2 <= 3'h0;
      stageClearVec_counter_3 <= 3'h0;
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
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(80)
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
    .width(80)
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
    .width(80)
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
    .width(80)
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
    .width(80)
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
    .width(80)
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
    .width(80)
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
    .width(80)
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
  assign in_0_ready = in_0_ready_0;
  assign in_1_ready = in_1_ready_0;
  assign in_2_ready = in_2_ready_0;
  assign in_3_ready = in_3_ready_0;
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
  assign stageClear = stageClearVec_0 & stageClearVec_1 & stageClearVec_2 & stageClearVec_3;
endmodule

