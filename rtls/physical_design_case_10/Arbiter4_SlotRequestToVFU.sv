module Arbiter4_SlotRequestToVFU(
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
  input  [11:0] io_in_0_bits_popInit,
  input  [7:0]  io_in_0_bits_groupIndex,
  input  [1:0]  io_in_0_bits_laneIndex,
  input         io_in_0_bits_maskType,
                io_in_0_bits_narrow,
  output        io_in_1_ready,
  input         io_in_1_valid,
  input  [32:0] io_in_1_bits_src_0,
                io_in_1_bits_src_1,
                io_in_1_bits_src_2,
                io_in_1_bits_src_3,
  input  [3:0]  io_in_1_bits_opcode,
                io_in_1_bits_mask,
                io_in_1_bits_executeMask,
  input         io_in_1_bits_sign0,
                io_in_1_bits_sign,
                io_in_1_bits_reverse,
                io_in_1_bits_average,
                io_in_1_bits_saturate,
  input  [1:0]  io_in_1_bits_vxrm,
                io_in_1_bits_vSew,
  input  [19:0] io_in_1_bits_shifterSize,
  input         io_in_1_bits_rem,
  input  [7:0]  io_in_1_bits_groupIndex,
  input  [1:0]  io_in_1_bits_laneIndex,
  input         io_in_1_bits_maskType,
                io_in_1_bits_narrow,
  output        io_in_2_ready,
  input         io_in_2_valid,
  input  [32:0] io_in_2_bits_src_0,
                io_in_2_bits_src_1,
                io_in_2_bits_src_2,
                io_in_2_bits_src_3,
  input  [3:0]  io_in_2_bits_opcode,
                io_in_2_bits_mask,
                io_in_2_bits_executeMask,
  input         io_in_2_bits_sign0,
                io_in_2_bits_sign,
                io_in_2_bits_reverse,
                io_in_2_bits_average,
                io_in_2_bits_saturate,
  input  [1:0]  io_in_2_bits_vxrm,
                io_in_2_bits_vSew,
  input  [19:0] io_in_2_bits_shifterSize,
  input         io_in_2_bits_rem,
  input  [7:0]  io_in_2_bits_groupIndex,
  input  [1:0]  io_in_2_bits_laneIndex,
  input         io_in_2_bits_maskType,
                io_in_2_bits_narrow,
  output        io_in_3_ready,
  input         io_in_3_valid,
  input  [32:0] io_in_3_bits_src_0,
                io_in_3_bits_src_1,
                io_in_3_bits_src_2,
                io_in_3_bits_src_3,
  input  [3:0]  io_in_3_bits_opcode,
                io_in_3_bits_mask,
                io_in_3_bits_executeMask,
  input         io_in_3_bits_sign0,
                io_in_3_bits_sign,
                io_in_3_bits_reverse,
                io_in_3_bits_average,
                io_in_3_bits_saturate,
  input  [1:0]  io_in_3_bits_vxrm,
                io_in_3_bits_vSew,
  input  [19:0] io_in_3_bits_shifterSize,
  input         io_in_3_bits_rem,
  input  [7:0]  io_in_3_bits_groupIndex,
  input  [1:0]  io_in_3_bits_laneIndex,
  input         io_in_3_bits_maskType,
                io_in_3_bits_narrow,
                io_out_ready,
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
  output [11:0] io_out_bits_popInit,
  output [7:0]  io_out_bits_groupIndex,
  output [1:0]  io_out_bits_laneIndex,
  output        io_out_bits_maskType,
                io_out_bits_narrow,
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
  wire [11:0] io_in_0_bits_popInit_0 = io_in_0_bits_popInit;
  wire [7:0]  io_in_0_bits_groupIndex_0 = io_in_0_bits_groupIndex;
  wire [1:0]  io_in_0_bits_laneIndex_0 = io_in_0_bits_laneIndex;
  wire        io_in_0_bits_maskType_0 = io_in_0_bits_maskType;
  wire        io_in_0_bits_narrow_0 = io_in_0_bits_narrow;
  wire        io_in_1_valid_0 = io_in_1_valid;
  wire [32:0] io_in_1_bits_src_0_0 = io_in_1_bits_src_0;
  wire [32:0] io_in_1_bits_src_1_0 = io_in_1_bits_src_1;
  wire [32:0] io_in_1_bits_src_2_0 = io_in_1_bits_src_2;
  wire [32:0] io_in_1_bits_src_3_0 = io_in_1_bits_src_3;
  wire [3:0]  io_in_1_bits_opcode_0 = io_in_1_bits_opcode;
  wire [3:0]  io_in_1_bits_mask_0 = io_in_1_bits_mask;
  wire [3:0]  io_in_1_bits_executeMask_0 = io_in_1_bits_executeMask;
  wire        io_in_1_bits_sign0_0 = io_in_1_bits_sign0;
  wire        io_in_1_bits_sign_0 = io_in_1_bits_sign;
  wire        io_in_1_bits_reverse_0 = io_in_1_bits_reverse;
  wire        io_in_1_bits_average_0 = io_in_1_bits_average;
  wire        io_in_1_bits_saturate_0 = io_in_1_bits_saturate;
  wire [1:0]  io_in_1_bits_vxrm_0 = io_in_1_bits_vxrm;
  wire [1:0]  io_in_1_bits_vSew_0 = io_in_1_bits_vSew;
  wire [19:0] io_in_1_bits_shifterSize_0 = io_in_1_bits_shifterSize;
  wire        io_in_1_bits_rem_0 = io_in_1_bits_rem;
  wire [7:0]  io_in_1_bits_groupIndex_0 = io_in_1_bits_groupIndex;
  wire [1:0]  io_in_1_bits_laneIndex_0 = io_in_1_bits_laneIndex;
  wire        io_in_1_bits_maskType_0 = io_in_1_bits_maskType;
  wire        io_in_1_bits_narrow_0 = io_in_1_bits_narrow;
  wire        io_in_2_valid_0 = io_in_2_valid;
  wire [32:0] io_in_2_bits_src_0_0 = io_in_2_bits_src_0;
  wire [32:0] io_in_2_bits_src_1_0 = io_in_2_bits_src_1;
  wire [32:0] io_in_2_bits_src_2_0 = io_in_2_bits_src_2;
  wire [32:0] io_in_2_bits_src_3_0 = io_in_2_bits_src_3;
  wire [3:0]  io_in_2_bits_opcode_0 = io_in_2_bits_opcode;
  wire [3:0]  io_in_2_bits_mask_0 = io_in_2_bits_mask;
  wire [3:0]  io_in_2_bits_executeMask_0 = io_in_2_bits_executeMask;
  wire        io_in_2_bits_sign0_0 = io_in_2_bits_sign0;
  wire        io_in_2_bits_sign_0 = io_in_2_bits_sign;
  wire        io_in_2_bits_reverse_0 = io_in_2_bits_reverse;
  wire        io_in_2_bits_average_0 = io_in_2_bits_average;
  wire        io_in_2_bits_saturate_0 = io_in_2_bits_saturate;
  wire [1:0]  io_in_2_bits_vxrm_0 = io_in_2_bits_vxrm;
  wire [1:0]  io_in_2_bits_vSew_0 = io_in_2_bits_vSew;
  wire [19:0] io_in_2_bits_shifterSize_0 = io_in_2_bits_shifterSize;
  wire        io_in_2_bits_rem_0 = io_in_2_bits_rem;
  wire [7:0]  io_in_2_bits_groupIndex_0 = io_in_2_bits_groupIndex;
  wire [1:0]  io_in_2_bits_laneIndex_0 = io_in_2_bits_laneIndex;
  wire        io_in_2_bits_maskType_0 = io_in_2_bits_maskType;
  wire        io_in_2_bits_narrow_0 = io_in_2_bits_narrow;
  wire        io_in_3_valid_0 = io_in_3_valid;
  wire [32:0] io_in_3_bits_src_0_0 = io_in_3_bits_src_0;
  wire [32:0] io_in_3_bits_src_1_0 = io_in_3_bits_src_1;
  wire [32:0] io_in_3_bits_src_2_0 = io_in_3_bits_src_2;
  wire [32:0] io_in_3_bits_src_3_0 = io_in_3_bits_src_3;
  wire [3:0]  io_in_3_bits_opcode_0 = io_in_3_bits_opcode;
  wire [3:0]  io_in_3_bits_mask_0 = io_in_3_bits_mask;
  wire [3:0]  io_in_3_bits_executeMask_0 = io_in_3_bits_executeMask;
  wire        io_in_3_bits_sign0_0 = io_in_3_bits_sign0;
  wire        io_in_3_bits_sign_0 = io_in_3_bits_sign;
  wire        io_in_3_bits_reverse_0 = io_in_3_bits_reverse;
  wire        io_in_3_bits_average_0 = io_in_3_bits_average;
  wire        io_in_3_bits_saturate_0 = io_in_3_bits_saturate;
  wire [1:0]  io_in_3_bits_vxrm_0 = io_in_3_bits_vxrm;
  wire [1:0]  io_in_3_bits_vSew_0 = io_in_3_bits_vSew;
  wire [19:0] io_in_3_bits_shifterSize_0 = io_in_3_bits_shifterSize;
  wire        io_in_3_bits_rem_0 = io_in_3_bits_rem;
  wire [7:0]  io_in_3_bits_groupIndex_0 = io_in_3_bits_groupIndex;
  wire [1:0]  io_in_3_bits_laneIndex_0 = io_in_3_bits_laneIndex;
  wire        io_in_3_bits_maskType_0 = io_in_3_bits_maskType;
  wire        io_in_3_bits_narrow_0 = io_in_3_bits_narrow;
  wire        io_out_ready_0 = io_out_ready;
  wire [1:0]  io_in_3_bits_tag = 2'h3;
  wire [1:0]  io_in_2_bits_tag = 2'h2;
  wire [1:0]  io_in_1_bits_tag = 2'h1;
  wire [11:0] io_in_1_bits_popInit = 12'h0;
  wire [11:0] io_in_2_bits_popInit = 12'h0;
  wire [11:0] io_in_3_bits_popInit = 12'h0;
  wire        io_in_0_bits_complete = 1'h0;
  wire        io_in_1_bits_complete = 1'h0;
  wire        io_in_2_bits_complete = 1'h0;
  wire        io_in_3_bits_complete = 1'h0;
  wire        io_out_bits_complete = 1'h0;
  wire [1:0]  io_in_0_bits_tag = 2'h0;
  wire [1:0]  io_in_1_bits_executeIndex = 2'h0;
  wire [1:0]  io_in_2_bits_executeIndex = 2'h0;
  wire [1:0]  io_in_3_bits_executeIndex = 2'h0;
  wire        io_in_0_ready_0 = io_out_ready_0;
  wire [1:0]  _GEN = io_in_0_valid_0 ? 2'h0 : io_in_1_valid_0 ? 2'h1 : {1'h1, ~io_in_2_valid_0};
  wire [1:0]  io_out_bits_tag_0;
  assign io_out_bits_tag_0 = _GEN;
  wire [1:0]  io_chosen;
  assign io_chosen = _GEN;
  wire [32:0] io_out_bits_src_0_0 = io_in_0_valid_0 ? io_in_0_bits_src_0_0 : io_in_1_valid_0 ? io_in_1_bits_src_0_0 : io_in_2_valid_0 ? io_in_2_bits_src_0_0 : io_in_3_bits_src_0_0;
  wire [32:0] io_out_bits_src_1_0 = io_in_0_valid_0 ? io_in_0_bits_src_1_0 : io_in_1_valid_0 ? io_in_1_bits_src_1_0 : io_in_2_valid_0 ? io_in_2_bits_src_1_0 : io_in_3_bits_src_1_0;
  wire [32:0] io_out_bits_src_2_0 = io_in_0_valid_0 ? io_in_0_bits_src_2_0 : io_in_1_valid_0 ? io_in_1_bits_src_2_0 : io_in_2_valid_0 ? io_in_2_bits_src_2_0 : io_in_3_bits_src_2_0;
  wire [32:0] io_out_bits_src_3_0 = io_in_0_valid_0 ? io_in_0_bits_src_3_0 : io_in_1_valid_0 ? io_in_1_bits_src_3_0 : io_in_2_valid_0 ? io_in_2_bits_src_3_0 : io_in_3_bits_src_3_0;
  wire [3:0]  io_out_bits_opcode_0 = io_in_0_valid_0 ? io_in_0_bits_opcode_0 : io_in_1_valid_0 ? io_in_1_bits_opcode_0 : io_in_2_valid_0 ? io_in_2_bits_opcode_0 : io_in_3_bits_opcode_0;
  wire [3:0]  io_out_bits_mask_0 = io_in_0_valid_0 ? io_in_0_bits_mask_0 : io_in_1_valid_0 ? io_in_1_bits_mask_0 : io_in_2_valid_0 ? io_in_2_bits_mask_0 : io_in_3_bits_mask_0;
  wire [3:0]  io_out_bits_executeMask_0 = io_in_0_valid_0 ? io_in_0_bits_executeMask_0 : io_in_1_valid_0 ? io_in_1_bits_executeMask_0 : io_in_2_valid_0 ? io_in_2_bits_executeMask_0 : io_in_3_bits_executeMask_0;
  wire        io_out_bits_sign0_0 = io_in_0_valid_0 ? io_in_0_bits_sign0_0 : io_in_1_valid_0 ? io_in_1_bits_sign0_0 : io_in_2_valid_0 ? io_in_2_bits_sign0_0 : io_in_3_bits_sign0_0;
  wire        io_out_bits_sign_0 = io_in_0_valid_0 ? io_in_0_bits_sign_0 : io_in_1_valid_0 ? io_in_1_bits_sign_0 : io_in_2_valid_0 ? io_in_2_bits_sign_0 : io_in_3_bits_sign_0;
  wire        io_out_bits_reverse_0 = io_in_0_valid_0 ? io_in_0_bits_reverse_0 : io_in_1_valid_0 ? io_in_1_bits_reverse_0 : io_in_2_valid_0 ? io_in_2_bits_reverse_0 : io_in_3_bits_reverse_0;
  wire        io_out_bits_average_0 = io_in_0_valid_0 ? io_in_0_bits_average_0 : io_in_1_valid_0 ? io_in_1_bits_average_0 : io_in_2_valid_0 ? io_in_2_bits_average_0 : io_in_3_bits_average_0;
  wire        io_out_bits_saturate_0 = io_in_0_valid_0 ? io_in_0_bits_saturate_0 : io_in_1_valid_0 ? io_in_1_bits_saturate_0 : io_in_2_valid_0 ? io_in_2_bits_saturate_0 : io_in_3_bits_saturate_0;
  wire [1:0]  io_out_bits_vxrm_0 = io_in_0_valid_0 ? io_in_0_bits_vxrm_0 : io_in_1_valid_0 ? io_in_1_bits_vxrm_0 : io_in_2_valid_0 ? io_in_2_bits_vxrm_0 : io_in_3_bits_vxrm_0;
  wire [1:0]  io_out_bits_vSew_0 = io_in_0_valid_0 ? io_in_0_bits_vSew_0 : io_in_1_valid_0 ? io_in_1_bits_vSew_0 : io_in_2_valid_0 ? io_in_2_bits_vSew_0 : io_in_3_bits_vSew_0;
  wire [19:0] io_out_bits_shifterSize_0 = io_in_0_valid_0 ? io_in_0_bits_shifterSize_0 : io_in_1_valid_0 ? io_in_1_bits_shifterSize_0 : io_in_2_valid_0 ? io_in_2_bits_shifterSize_0 : io_in_3_bits_shifterSize_0;
  wire        io_out_bits_rem_0 = io_in_0_valid_0 ? io_in_0_bits_rem_0 : io_in_1_valid_0 ? io_in_1_bits_rem_0 : io_in_2_valid_0 ? io_in_2_bits_rem_0 : io_in_3_bits_rem_0;
  wire [1:0]  io_out_bits_executeIndex_0 = io_in_0_valid_0 ? io_in_0_bits_executeIndex_0 : 2'h0;
  wire [11:0] io_out_bits_popInit_0 = io_in_0_valid_0 ? io_in_0_bits_popInit_0 : 12'h0;
  wire [7:0]  io_out_bits_groupIndex_0 = io_in_0_valid_0 ? io_in_0_bits_groupIndex_0 : io_in_1_valid_0 ? io_in_1_bits_groupIndex_0 : io_in_2_valid_0 ? io_in_2_bits_groupIndex_0 : io_in_3_bits_groupIndex_0;
  wire [1:0]  io_out_bits_laneIndex_0 = io_in_0_valid_0 ? io_in_0_bits_laneIndex_0 : io_in_1_valid_0 ? io_in_1_bits_laneIndex_0 : io_in_2_valid_0 ? io_in_2_bits_laneIndex_0 : io_in_3_bits_laneIndex_0;
  wire        io_out_bits_maskType_0 = io_in_0_valid_0 ? io_in_0_bits_maskType_0 : io_in_1_valid_0 ? io_in_1_bits_maskType_0 : io_in_2_valid_0 ? io_in_2_bits_maskType_0 : io_in_3_bits_maskType_0;
  wire        io_out_bits_narrow_0 = io_in_0_valid_0 ? io_in_0_bits_narrow_0 : io_in_1_valid_0 ? io_in_1_bits_narrow_0 : io_in_2_valid_0 ? io_in_2_bits_narrow_0 : io_in_3_bits_narrow_0;
  wire        _grant_T = io_in_0_valid_0 | io_in_1_valid_0;
  wire        grant_1 = ~io_in_0_valid_0;
  wire        grant_2 = ~_grant_T;
  wire        grant_3 = ~(_grant_T | io_in_2_valid_0);
  wire        io_in_1_ready_0 = grant_1 & io_out_ready_0;
  wire        io_in_2_ready_0 = grant_2 & io_out_ready_0;
  wire        io_in_3_ready_0 = grant_3 & io_out_ready_0;
  wire        io_out_valid_0 = ~grant_3 | io_in_3_valid_0;
  assign io_in_0_ready = io_in_0_ready_0;
  assign io_in_1_ready = io_in_1_ready_0;
  assign io_in_2_ready = io_in_2_ready_0;
  assign io_in_3_ready = io_in_3_ready_0;
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
  assign io_out_bits_tag = io_out_bits_tag_0;
endmodule

