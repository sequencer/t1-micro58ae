
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
module ReadStageRRArbiter_1(
  input        clock,
               reset,
  output       io_in_0_ready,
  input        io_in_0_valid,
  input  [4:0] io_in_0_bits_vs,
  input  [6:0] io_in_0_bits_offset,
  input  [3:0] io_in_0_bits_groupIndex,
               io_in_0_bits_readSource,
  input  [2:0] io_in_0_bits_instructionIndex,
  output       io_in_1_ready,
  input        io_in_1_valid,
  input  [4:0] io_in_1_bits_vs,
  input  [6:0] io_in_1_bits_offset,
  input  [3:0] io_in_1_bits_groupIndex,
               io_in_1_bits_readSource,
  input  [2:0] io_in_1_bits_instructionIndex,
  input        io_out_ready,
  output       io_out_valid,
  output [4:0] io_out_bits_vs,
  output [6:0] io_out_bits_offset,
  output [3:0] io_out_bits_readSource,
  output [2:0] io_out_bits_instructionIndex
);

  wire       io_in_0_valid_0 = io_in_0_valid;
  wire [4:0] io_in_0_bits_vs_0 = io_in_0_bits_vs;
  wire [6:0] io_in_0_bits_offset_0 = io_in_0_bits_offset;
  wire [3:0] io_in_0_bits_groupIndex_0 = io_in_0_bits_groupIndex;
  wire [3:0] io_in_0_bits_readSource_0 = io_in_0_bits_readSource;
  wire [2:0] io_in_0_bits_instructionIndex_0 = io_in_0_bits_instructionIndex;
  wire       io_in_1_valid_0 = io_in_1_valid;
  wire [4:0] io_in_1_bits_vs_0 = io_in_1_bits_vs;
  wire [6:0] io_in_1_bits_offset_0 = io_in_1_bits_offset;
  wire [3:0] io_in_1_bits_groupIndex_0 = io_in_1_bits_groupIndex;
  wire [3:0] io_in_1_bits_readSource_0 = io_in_1_bits_readSource;
  wire [2:0] io_in_1_bits_instructionIndex_0 = io_in_1_bits_instructionIndex;
  wire       io_out_ready_0 = io_out_ready;
  reg        choseMask;
  wire       select_0 = io_in_0_valid_0 & (choseMask | ~io_in_1_valid_0);
  wire       select_1 = io_in_1_valid_0 & (~choseMask | ~io_in_0_valid_0);
  wire       io_out_valid_0 = io_in_0_valid_0 | io_in_1_valid_0;
  wire [2:0] io_out_bits_instructionIndex_0 = (select_0 ? io_in_0_bits_instructionIndex_0 : 3'h0) | (select_1 ? io_in_1_bits_instructionIndex_0 : 3'h0);
  wire [3:0] io_out_bits_readSource_0 = (select_0 ? io_in_0_bits_readSource_0 : 4'h0) | (select_1 ? io_in_1_bits_readSource_0 : 4'h0);
  wire [3:0] io_out_bits_groupIndex = (select_0 ? io_in_0_bits_groupIndex_0 : 4'h0) | (select_1 ? io_in_1_bits_groupIndex_0 : 4'h0);
  wire [6:0] io_out_bits_offset_0 = (select_0 ? io_in_0_bits_offset_0 : 7'h0) | (select_1 ? io_in_1_bits_offset_0 : 7'h0);
  wire [4:0] io_out_bits_vs_0 = (select_0 ? io_in_0_bits_vs_0 : 5'h0) | (select_1 ? io_in_1_bits_vs_0 : 5'h0);
  wire       io_in_0_ready_0 = select_0 & io_out_ready_0;
  wire       io_in_1_ready_0 = select_1 & io_out_ready_0;
  always @(posedge clock) begin
    if (reset)
      choseMask <= 1'h1;
    else if (io_out_ready_0 & io_out_valid_0)
      choseMask <= io_in_1_ready_0 & io_in_1_valid_0;
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
        choseMask = _RANDOM[/*Zero width*/ 1'b0][0];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  assign io_in_0_ready = io_in_0_ready_0;
  assign io_in_1_ready = io_in_1_ready_0;
  assign io_out_valid = io_out_valid_0;
  assign io_out_bits_vs = io_out_bits_vs_0;
  assign io_out_bits_offset = io_out_bits_offset_0;
  assign io_out_bits_readSource = io_out_bits_readSource_0;
  assign io_out_bits_instructionIndex = io_out_bits_instructionIndex_0;
endmodule

