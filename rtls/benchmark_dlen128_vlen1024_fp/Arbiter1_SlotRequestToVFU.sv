module Arbiter1_SlotRequestToVFU(
  output        io_in_0_ready,
  input         io_in_0_valid,
  input  [32:0] io_in_0_bits_src_0,
                io_in_0_bits_src_1,
                io_in_0_bits_src_2,
                io_in_0_bits_src_3,
  input  [3:0]  io_in_0_bits_opcode,
                io_in_0_bits_mask,
                io_in_0_bits_executeMask,
  input         io_in_0_bits_sign0,
                io_in_0_bits_sign,
                io_in_0_bits_reverse,
                io_in_0_bits_average,
                io_in_0_bits_saturate,
  input  [1:0]  io_in_0_bits_vxrm,
                io_in_0_bits_vSew,
  input  [19:0] io_in_0_bits_shifterSize,
  input         io_in_0_bits_rem,
  input  [1:0]  io_in_0_bits_executeIndex,
  input  [10:0] io_in_0_bits_popInit,
  input  [6:0]  io_in_0_bits_groupIndex,
  input  [1:0]  io_in_0_bits_laneIndex,
  input         io_in_0_bits_maskType,
                io_in_0_bits_narrow,
  input  [1:0]  io_in_0_bits_unitSelet,
  input         io_in_0_bits_floatMul,
  input  [2:0]  io_in_0_bits_roundingMode,
  input  [1:0]  io_in_0_bits_tag,
  input         io_out_ready,
  output        io_out_valid,
  output [32:0] io_out_bits_src_0,
                io_out_bits_src_1,
                io_out_bits_src_2,
                io_out_bits_src_3,
  output [3:0]  io_out_bits_opcode,
                io_out_bits_mask,
                io_out_bits_executeMask,
  output        io_out_bits_sign0,
                io_out_bits_sign,
                io_out_bits_reverse,
                io_out_bits_average,
                io_out_bits_saturate,
  output [1:0]  io_out_bits_vxrm,
                io_out_bits_vSew,
  output [19:0] io_out_bits_shifterSize,
  output        io_out_bits_rem,
  output [1:0]  io_out_bits_executeIndex,
  output [10:0] io_out_bits_popInit,
  output [6:0]  io_out_bits_groupIndex,
  output [1:0]  io_out_bits_laneIndex,
  output        io_out_bits_maskType,
                io_out_bits_narrow,
  output [1:0]  io_out_bits_unitSelet,
  output        io_out_bits_floatMul,
  output [2:0]  io_out_bits_roundingMode,
  output [1:0]  io_out_bits_tag
);

  wire        io_in_0_valid_0 = io_in_0_valid;
  wire [32:0] io_in_0_bits_src_0_0 = io_in_0_bits_src_0;
  wire [32:0] io_in_0_bits_src_1_0 = io_in_0_bits_src_1;
  wire [32:0] io_in_0_bits_src_2_0 = io_in_0_bits_src_2;
  wire [32:0] io_in_0_bits_src_3_0 = io_in_0_bits_src_3;
  wire [3:0]  io_in_0_bits_opcode_0 = io_in_0_bits_opcode;
  wire [3:0]  io_in_0_bits_mask_0 = io_in_0_bits_mask;
  wire [3:0]  io_in_0_bits_executeMask_0 = io_in_0_bits_executeMask;
  wire        io_in_0_bits_sign0_0 = io_in_0_bits_sign0;
  wire        io_in_0_bits_sign_0 = io_in_0_bits_sign;
  wire        io_in_0_bits_reverse_0 = io_in_0_bits_reverse;
  wire        io_in_0_bits_average_0 = io_in_0_bits_average;
  wire        io_in_0_bits_saturate_0 = io_in_0_bits_saturate;
  wire [1:0]  io_in_0_bits_vxrm_0 = io_in_0_bits_vxrm;
  wire [1:0]  io_in_0_bits_vSew_0 = io_in_0_bits_vSew;
  wire [19:0] io_in_0_bits_shifterSize_0 = io_in_0_bits_shifterSize;
  wire        io_in_0_bits_rem_0 = io_in_0_bits_rem;
  wire [1:0]  io_in_0_bits_executeIndex_0 = io_in_0_bits_executeIndex;
  wire [10:0] io_in_0_bits_popInit_0 = io_in_0_bits_popInit;
  wire [6:0]  io_in_0_bits_groupIndex_0 = io_in_0_bits_groupIndex;
  wire [1:0]  io_in_0_bits_laneIndex_0 = io_in_0_bits_laneIndex;
  wire        io_in_0_bits_maskType_0 = io_in_0_bits_maskType;
  wire        io_in_0_bits_narrow_0 = io_in_0_bits_narrow;
  wire [1:0]  io_in_0_bits_unitSelet_0 = io_in_0_bits_unitSelet;
  wire        io_in_0_bits_floatMul_0 = io_in_0_bits_floatMul;
  wire [2:0]  io_in_0_bits_roundingMode_0 = io_in_0_bits_roundingMode;
  wire [1:0]  io_in_0_bits_tag_0 = io_in_0_bits_tag;
  wire        io_out_ready_0 = io_out_ready;
  wire        io_in_0_bits_complete = 1'h0;
  wire        io_out_bits_complete = 1'h0;
  wire        io_out_valid_0 = io_in_0_valid_0;
  wire [32:0] io_out_bits_src_0_0 = io_in_0_bits_src_0_0;
  wire [32:0] io_out_bits_src_1_0 = io_in_0_bits_src_1_0;
  wire [32:0] io_out_bits_src_2_0 = io_in_0_bits_src_2_0;
  wire [32:0] io_out_bits_src_3_0 = io_in_0_bits_src_3_0;
  wire [3:0]  io_out_bits_opcode_0 = io_in_0_bits_opcode_0;
  wire [3:0]  io_out_bits_mask_0 = io_in_0_bits_mask_0;
  wire [3:0]  io_out_bits_executeMask_0 = io_in_0_bits_executeMask_0;
  wire        io_out_bits_sign0_0 = io_in_0_bits_sign0_0;
  wire        io_out_bits_sign_0 = io_in_0_bits_sign_0;
  wire        io_out_bits_reverse_0 = io_in_0_bits_reverse_0;
  wire        io_out_bits_average_0 = io_in_0_bits_average_0;
  wire        io_out_bits_saturate_0 = io_in_0_bits_saturate_0;
  wire [1:0]  io_out_bits_vxrm_0 = io_in_0_bits_vxrm_0;
  wire [1:0]  io_out_bits_vSew_0 = io_in_0_bits_vSew_0;
  wire [19:0] io_out_bits_shifterSize_0 = io_in_0_bits_shifterSize_0;
  wire        io_out_bits_rem_0 = io_in_0_bits_rem_0;
  wire [1:0]  io_out_bits_executeIndex_0 = io_in_0_bits_executeIndex_0;
  wire [10:0] io_out_bits_popInit_0 = io_in_0_bits_popInit_0;
  wire [6:0]  io_out_bits_groupIndex_0 = io_in_0_bits_groupIndex_0;
  wire [1:0]  io_out_bits_laneIndex_0 = io_in_0_bits_laneIndex_0;
  wire        io_out_bits_maskType_0 = io_in_0_bits_maskType_0;
  wire        io_out_bits_narrow_0 = io_in_0_bits_narrow_0;
  wire [1:0]  io_out_bits_unitSelet_0 = io_in_0_bits_unitSelet_0;
  wire        io_out_bits_floatMul_0 = io_in_0_bits_floatMul_0;
  wire [2:0]  io_out_bits_roundingMode_0 = io_in_0_bits_roundingMode_0;
  wire [1:0]  io_out_bits_tag_0 = io_in_0_bits_tag_0;
  wire        io_in_0_ready_0 = io_out_ready_0;
  assign io_in_0_ready = io_in_0_ready_0;
  assign io_out_valid = io_out_valid_0;
  assign io_out_bits_src_0 = io_out_bits_src_0_0;
  assign io_out_bits_src_1 = io_out_bits_src_1_0;
  assign io_out_bits_src_2 = io_out_bits_src_2_0;
  assign io_out_bits_src_3 = io_out_bits_src_3_0;
  assign io_out_bits_opcode = io_out_bits_opcode_0;
  assign io_out_bits_mask = io_out_bits_mask_0;
  assign io_out_bits_executeMask = io_out_bits_executeMask_0;
  assign io_out_bits_sign0 = io_out_bits_sign0_0;
  assign io_out_bits_sign = io_out_bits_sign_0;
  assign io_out_bits_reverse = io_out_bits_reverse_0;
  assign io_out_bits_average = io_out_bits_average_0;
  assign io_out_bits_saturate = io_out_bits_saturate_0;
  assign io_out_bits_vxrm = io_out_bits_vxrm_0;
  assign io_out_bits_vSew = io_out_bits_vSew_0;
  assign io_out_bits_shifterSize = io_out_bits_shifterSize_0;
  assign io_out_bits_rem = io_out_bits_rem_0;
  assign io_out_bits_executeIndex = io_out_bits_executeIndex_0;
  assign io_out_bits_popInit = io_out_bits_popInit_0;
  assign io_out_bits_groupIndex = io_out_bits_groupIndex_0;
  assign io_out_bits_laneIndex = io_out_bits_laneIndex_0;
  assign io_out_bits_maskType = io_out_bits_maskType_0;
  assign io_out_bits_narrow = io_out_bits_narrow_0;
  assign io_out_bits_unitSelet = io_out_bits_unitSelet_0;
  assign io_out_bits_floatMul = io_out_bits_floatMul_0;
  assign io_out_bits_roundingMode = io_out_bits_roundingMode_0;
  assign io_out_bits_tag = io_out_bits_tag_0;
endmodule

