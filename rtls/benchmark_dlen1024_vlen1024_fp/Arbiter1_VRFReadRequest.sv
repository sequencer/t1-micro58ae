module Arbiter1_VRFReadRequest(
  output       io_in_0_ready,
  input        io_in_0_valid,
  input  [4:0] io_in_0_bits_vs,
  input  [1:0] io_in_0_bits_readSource,
  input  [2:0] io_in_0_bits_instructionIndex,
  input        io_out_ready,
  output       io_out_valid,
  output [4:0] io_out_bits_vs,
  output [1:0] io_out_bits_readSource,
  output [2:0] io_out_bits_instructionIndex
);

  wire       io_in_0_valid_0 = io_in_0_valid;
  wire [4:0] io_in_0_bits_vs_0 = io_in_0_bits_vs;
  wire [1:0] io_in_0_bits_readSource_0 = io_in_0_bits_readSource;
  wire [2:0] io_in_0_bits_instructionIndex_0 = io_in_0_bits_instructionIndex;
  wire       io_out_ready_0 = io_out_ready;
  wire       io_out_valid_0 = io_in_0_valid_0;
  wire [4:0] io_out_bits_vs_0 = io_in_0_bits_vs_0;
  wire [1:0] io_out_bits_readSource_0 = io_in_0_bits_readSource_0;
  wire [2:0] io_out_bits_instructionIndex_0 = io_in_0_bits_instructionIndex_0;
  wire       io_in_0_ready_0 = io_out_ready_0;
  assign io_in_0_ready = io_in_0_ready_0;
  assign io_out_valid = io_out_valid_0;
  assign io_out_bits_vs = io_out_bits_vs_0;
  assign io_out_bits_readSource = io_out_bits_readSource_0;
  assign io_out_bits_instructionIndex = io_out_bits_instructionIndex_0;
endmodule

