
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
module Distributor(
  input         clock,
                reset,
                requestToVfu_ready,
  output        requestToVfu_valid,
  output [32:0] requestToVfu_bits_src_0,
                requestToVfu_bits_src_1,
                requestToVfu_bits_src_2,
                requestToVfu_bits_src_3,
  output [3:0]  requestToVfu_bits_opcode,
                requestToVfu_bits_mask,
  output        requestToVfu_bits_sign,
  output [1:0]  requestToVfu_bits_vxrm,
                requestToVfu_bits_vSew,
  output [19:0] requestToVfu_bits_shifterSize,
  output [1:0]  requestToVfu_bits_executeIndex,
  output [11:0] requestToVfu_bits_popInit,
  output [7:0]  requestToVfu_bits_groupIndex,
  output [1:0]  requestToVfu_bits_laneIndex,
  output        requestToVfu_bits_maskType,
                requestToVfu_bits_narrow,
  output [1:0]  requestToVfu_bits_tag,
  input         responseFromVfu_valid,
  input  [31:0] responseFromVfu_bits_data,
  input         responseFromVfu_bits_clipFail,
                responseFromVfu_bits_ffoSuccess,
  output        requestFromSlot_ready,
  input         requestFromSlot_valid,
  input  [32:0] requestFromSlot_bits_src_0,
                requestFromSlot_bits_src_1,
                requestFromSlot_bits_src_2,
                requestFromSlot_bits_src_3,
  input  [3:0]  requestFromSlot_bits_opcode,
                requestFromSlot_bits_mask,
                requestFromSlot_bits_executeMask,
  input         requestFromSlot_bits_sign0,
                requestFromSlot_bits_sign,
                requestFromSlot_bits_reverse,
                requestFromSlot_bits_average,
                requestFromSlot_bits_saturate,
  input  [1:0]  requestFromSlot_bits_vxrm,
                requestFromSlot_bits_vSew,
  input  [19:0] requestFromSlot_bits_shifterSize,
  input         requestFromSlot_bits_rem,
  input  [1:0]  requestFromSlot_bits_executeIndex,
  input  [11:0] requestFromSlot_bits_popInit,
  input  [7:0]  requestFromSlot_bits_groupIndex,
  input  [1:0]  requestFromSlot_bits_laneIndex,
  input         requestFromSlot_bits_maskType,
                requestFromSlot_bits_narrow,
  input  [1:0]  requestFromSlot_bits_tag,
  output        responseToSlot_valid,
  output [31:0] responseToSlot_bits_data,
  output        responseToSlot_bits_ffoSuccess,
  output [3:0]  responseToSlot_bits_vxsat,
  output [1:0]  responseToSlot_bits_tag
);

  wire         _executeQueue_fifo_empty;
  wire         _executeQueue_fifo_full;
  wire         _executeQueue_fifo_error;
  wire         executeQueue_almostFull;
  wire         executeQueue_almostEmpty;
  wire         requestToVfu_ready_0 = requestToVfu_ready;
  wire         requestFromSlot_valid_0 = requestFromSlot_valid;
  wire [32:0]  requestFromSlot_bits_src_0_0 = requestFromSlot_bits_src_0;
  wire [32:0]  requestFromSlot_bits_src_1_0 = requestFromSlot_bits_src_1;
  wire [32:0]  requestFromSlot_bits_src_2_0 = requestFromSlot_bits_src_2;
  wire [32:0]  requestFromSlot_bits_src_3_0 = requestFromSlot_bits_src_3;
  wire [3:0]   requestFromSlot_bits_opcode_0 = requestFromSlot_bits_opcode;
  wire [3:0]   requestFromSlot_bits_mask_0 = requestFromSlot_bits_mask;
  wire [3:0]   requestFromSlot_bits_executeMask_0 = requestFromSlot_bits_executeMask;
  wire         requestFromSlot_bits_sign0_0 = requestFromSlot_bits_sign0;
  wire         requestFromSlot_bits_sign_0 = requestFromSlot_bits_sign;
  wire         requestFromSlot_bits_reverse_0 = requestFromSlot_bits_reverse;
  wire         requestFromSlot_bits_average_0 = requestFromSlot_bits_average;
  wire         requestFromSlot_bits_saturate_0 = requestFromSlot_bits_saturate;
  wire [1:0]   requestFromSlot_bits_vxrm_0 = requestFromSlot_bits_vxrm;
  wire [1:0]   requestFromSlot_bits_vSew_0 = requestFromSlot_bits_vSew;
  wire [19:0]  requestFromSlot_bits_shifterSize_0 = requestFromSlot_bits_shifterSize;
  wire         requestFromSlot_bits_rem_0 = requestFromSlot_bits_rem;
  wire [1:0]   requestFromSlot_bits_executeIndex_0 = requestFromSlot_bits_executeIndex;
  wire [11:0]  requestFromSlot_bits_popInit_0 = requestFromSlot_bits_popInit;
  wire [7:0]   requestFromSlot_bits_groupIndex_0 = requestFromSlot_bits_groupIndex;
  wire [1:0]   requestFromSlot_bits_laneIndex_0 = requestFromSlot_bits_laneIndex;
  wire         requestFromSlot_bits_maskType_0 = requestFromSlot_bits_maskType;
  wire         requestFromSlot_bits_narrow_0 = requestFromSlot_bits_narrow;
  wire [1:0]   requestFromSlot_bits_tag_0 = requestFromSlot_bits_tag;
  wire         executeQueue_deq_ready = responseFromVfu_valid;
  wire [3:0]   responseWire_bits_adderMaskResp = 4'h0;
  wire [1:0]   responseWire_bits_executeIndex = 2'h0;
  wire [4:0]   responseWire_bits_exceptionFlags = 5'h0;
  wire         requestToVfu_bits_complete = 1'h0;
  wire         requestFromSlot_bits_complete = 1'h0;
  wire         responseWire_bits_clipFail = 1'h0;
  wire         responseWire_bits_divBusy = 1'h0;
  wire         updateFFO;
  reg          requestReg_valid;
  reg  [32:0]  requestReg_bits_src_0;
  reg  [32:0]  requestReg_bits_src_1;
  reg  [32:0]  requestReg_bits_src_2;
  reg  [32:0]  requestReg_bits_src_3;
  reg  [3:0]   requestReg_bits_opcode;
  wire [3:0]   requestToVfu_bits_opcode_0 = requestReg_bits_opcode;
  reg  [3:0]   requestReg_bits_mask;
  reg  [3:0]   requestReg_bits_executeMask;
  wire [3:0]   requestToVfu_bits_executeMask = requestReg_bits_executeMask;
  reg          requestReg_bits_sign0;
  wire         requestToVfu_bits_sign0 = requestReg_bits_sign0;
  reg          requestReg_bits_sign;
  wire         requestToVfu_bits_sign_0 = requestReg_bits_sign;
  reg          requestReg_bits_reverse;
  wire         requestToVfu_bits_reverse = requestReg_bits_reverse;
  reg          requestReg_bits_average;
  wire         requestToVfu_bits_average = requestReg_bits_average;
  reg          requestReg_bits_saturate;
  wire         requestToVfu_bits_saturate = requestReg_bits_saturate;
  reg  [1:0]   requestReg_bits_vxrm;
  wire [1:0]   requestToVfu_bits_vxrm_0 = requestReg_bits_vxrm;
  reg  [1:0]   requestReg_bits_vSew;
  wire [1:0]   requestToVfu_bits_vSew_0 = requestReg_bits_vSew;
  reg  [19:0]  requestReg_bits_shifterSize;
  reg          requestReg_bits_rem;
  wire         requestToVfu_bits_rem = requestReg_bits_rem;
  reg  [1:0]   requestReg_bits_executeIndex;
  reg  [11:0]  requestReg_bits_popInit;
  wire [11:0]  requestToVfu_bits_popInit_0 = requestReg_bits_popInit;
  reg  [7:0]   requestReg_bits_groupIndex;
  wire [7:0]   requestToVfu_bits_groupIndex_0 = requestReg_bits_groupIndex;
  reg  [1:0]   requestReg_bits_laneIndex;
  wire [1:0]   requestToVfu_bits_laneIndex_0 = requestReg_bits_laneIndex;
  reg          requestReg_bits_maskType;
  wire         requestToVfu_bits_maskType_0 = requestReg_bits_maskType;
  reg          requestReg_bits_narrow;
  wire         requestToVfu_bits_narrow_0 = requestReg_bits_narrow;
  reg  [1:0]   requestReg_bits_tag;
  wire [1:0]   requestToVfu_bits_tag_0 = requestReg_bits_tag;
  wire [1:0]   responseWire_bits_tag = requestReg_bits_tag;
  reg          sendRequestValid;
  wire         requestToVfu_valid_0 = sendRequestValid;
  reg          ffoSuccess;
  reg          vxsatResult;
  reg  [32:0]  responseData;
  reg  [1:0]   executeIndex;
  wire [1:0]   requestToVfu_bits_executeIndex_0 = executeIndex;
  wire [3:0]   _vSew1HReq_T = 4'h1 << requestFromSlot_bits_vSew_0;
  wire [2:0]   vSew1HReq = _vSew1HReq_T[2:0];
  wire [3:0]   _vSew1H_T = 4'h1 << requestReg_bits_vSew;
  wire [2:0]   vSew1H = _vSew1H_T[2:0];
  wire [2:0]   _nextExecuteIndexForNextGroup_T_3 = requestFromSlot_bits_executeMask_0[2:0] | {requestFromSlot_bits_executeMask_0[1:0], 1'h0};
  wire [2:0]   _GEN = ~(_nextExecuteIndexForNextGroup_T_3 | {_nextExecuteIndexForNextGroup_T_3[0], 2'h0});
  wire [1:0]   nextExecuteIndexForNextGroup_hi = _GEN[2:1] & requestFromSlot_bits_executeMask_0[3:2];
  wire [1:0]   nextExecuteIndexForNextGroup_lo = {_GEN[0], 1'h1} & requestFromSlot_bits_executeMask_0[1:0];
  wire [1:0]   nextExecuteIndexForNextGroup =
    (vSew1HReq[0] ? {|nextExecuteIndexForNextGroup_hi, nextExecuteIndexForNextGroup_hi[1] | nextExecuteIndexForNextGroup_lo[1]} : 2'h0) | (vSew1HReq[1] ? {~(requestFromSlot_bits_executeMask_0[0]), 1'h0} : 2'h0);
  wire [3:0]   currentOHForExecuteGroup = 4'h1 << executeIndex;
  wire [2:0]   _GEN_0 = currentOHForExecuteGroup[2:0] | currentOHForExecuteGroup[3:1];
  wire [3:0]   remainder = requestReg_bits_executeMask & ~{currentOHForExecuteGroup[3], _GEN_0[2], _GEN_0[1:0] | {currentOHForExecuteGroup[3], _GEN_0[2]}};
  wire [2:0]   _nextIndex1H_T_2 = remainder[2:0] | {remainder[1:0], 1'h0};
  wire [3:0]   nextIndex1H = {~(_nextIndex1H_T_2 | {_nextIndex1H_T_2[0], 2'h0}), 1'h1} & remainder;
  wire [1:0]   nextExecuteIndex_hi = nextIndex1H[3:2];
  wire [1:0]   nextExecuteIndex_lo = nextIndex1H[1:0];
  wire [1:0]   nextExecuteIndex = (vSew1H[0] ? {|nextExecuteIndex_hi, nextExecuteIndex_hi[1] | nextExecuteIndex_lo[1]} : 2'h0) | {vSew1H[1], 1'h0};
  wire         executeQueue_enq_valid = requestToVfu_ready_0 & requestToVfu_valid_0;
  wire [3:0]   byteMaskForExecution = (vSew1H[0] ? currentOHForExecuteGroup : 4'h0) | (vSew1H[1] ? {{2{executeIndex[1]}}, ~(executeIndex[1]), ~(executeIndex[1])} : 4'h0) | {4{vSew1H[2]}};
  wire [15:0]  bitMaskForExecution_lo = {{8{byteMaskForExecution[1]}}, {8{byteMaskForExecution[0]}}};
  wire [15:0]  bitMaskForExecution_hi = {{8{byteMaskForExecution[3]}}, {8{byteMaskForExecution[2]}}};
  wire [31:0]  bitMaskForExecution = {bitMaskForExecution_hi, bitMaskForExecution_lo};
  wire [32:0]  dataMasked = {1'h0, requestReg_bits_src_0[31:0] & bitMaskForExecution};
  wire [7:0]   collapse0 = dataMasked[7:0] | dataMasked[15:8] | dataMasked[23:16] | dataMasked[31:24];
  wire [15:0]  collapse1 = dataMasked[15:0] | dataMasked[31:16];
  wire [32:0]  requestToVfu_bits_src_0_0 =
    (vSew1H[0] ? {{25{requestReg_bits_sign0 & collapse0[7]}}, collapse0} : 33'h0) | (vSew1H[1] ? {{17{requestReg_bits_sign0 & collapse1[15]}}, collapse1} : 33'h0) | (vSew1H[2] ? requestReg_bits_src_0 : 33'h0);
  wire [32:0]  dataMasked_1 = {1'h0, requestReg_bits_src_1[31:0] & bitMaskForExecution};
  wire [7:0]   collapse0_1 = dataMasked_1[7:0] | dataMasked_1[15:8] | dataMasked_1[23:16] | dataMasked_1[31:24];
  wire [15:0]  collapse1_1 = dataMasked_1[15:0] | dataMasked_1[31:16];
  wire [32:0]  requestToVfu_bits_src_1_0 =
    (vSew1H[0] ? {{25{requestReg_bits_sign & collapse0_1[7]}}, collapse0_1} : 33'h0) | (vSew1H[1] ? {{17{requestReg_bits_sign & collapse1_1[15]}}, collapse1_1} : 33'h0) | (vSew1H[2] ? requestReg_bits_src_1 : 33'h0);
  wire [32:0]  dataMasked_2 = {1'h0, requestReg_bits_src_2[31:0] & bitMaskForExecution};
  wire [7:0]   collapse0_2 = dataMasked_2[7:0] | dataMasked_2[15:8] | dataMasked_2[23:16] | dataMasked_2[31:24];
  wire [15:0]  collapse1_2 = dataMasked_2[15:0] | dataMasked_2[31:16];
  wire [32:0]  requestToVfu_bits_src_2_0 = (vSew1H[0] ? {25'h0, collapse0_2} : 33'h0) | (vSew1H[1] ? {17'h0, collapse1_2} : 33'h0) | (vSew1H[2] ? requestReg_bits_src_2 : 33'h0);
  wire [32:0]  dataMasked_3 = {1'h0, requestReg_bits_src_3[31:0] & bitMaskForExecution};
  wire [7:0]   collapse0_3 = dataMasked_3[7:0] | dataMasked_3[15:8] | dataMasked_3[23:16] | dataMasked_3[31:24];
  wire [15:0]  collapse1_3 = dataMasked_3[15:0] | dataMasked_3[31:16];
  wire [32:0]  requestToVfu_bits_src_3_0 = (vSew1H[0] ? {25'h0, collapse0_3} : 33'h0) | (vSew1H[1] ? {17'h0, collapse1_3} : 33'h0) | (vSew1H[2] ? requestReg_bits_src_3 : 33'h0);
  wire [3:0]   requestToVfu_bits_mask_0 = {3'h0, |(requestReg_bits_mask & currentOHForExecuteGroup)};
  wire [19:0]  requestToVfu_bits_shifterSize_0 =
    {15'h0,
     (currentOHForExecuteGroup[0] ? requestReg_bits_shifterSize[4:0] : 5'h0) | (currentOHForExecuteGroup[1] ? requestReg_bits_shifterSize[9:5] : 5'h0) | (currentOHForExecuteGroup[2] ? requestReg_bits_shifterSize[14:10] : 5'h0)
       | (currentOHForExecuteGroup[3] ? requestReg_bits_shifterSize[19:15] : 5'h0)};
  wire         isLastRequest = vSew1H[0] & remainder == 4'h0 | vSew1H[1] & remainder[1:0] == 2'h0 | vSew1H[2];
  wire         lastRequestFire = isLastRequest & executeQueue_enq_valid;
  wire         executeQueue_deq_valid;
  assign executeQueue_deq_valid = ~_executeQueue_fifo_empty;
  wire         executeQueue_enq_ready = ~_executeQueue_fifo_full;
  wire [2:0]   executeQueue_enq_bits = {isLastRequest, executeIndex};
  wire [2:0]   executeQueue_deq_bits;
  wire [3:0]   writeIndex = {2'h0, executeQueue_deq_bits[1:0]};
  wire         isLastResponse = executeQueue_deq_bits[2] & responseFromVfu_valid & executeQueue_deq_valid;
  wire [15:0]  writeIndex1H = 16'h1 << writeIndex;
  wire [15:0]  _writeMaskInByte_T_13 = vSew1H[0] ? writeIndex1H : 16'h0;
  wire [15:0]  writeMaskInByte = {_writeMaskInByte_T_13[15:4], _writeMaskInByte_T_13[3:0] | (vSew1H[1] ? {{2{writeIndex[1]}}, ~(writeIndex[1]), ~(writeIndex[1])} : 4'h0) | {4{vSew1H[2]}}};
  wire [15:0]  writeMaskInBit_lo_lo_lo = {{8{writeMaskInByte[1]}}, {8{writeMaskInByte[0]}}};
  wire [15:0]  writeMaskInBit_lo_lo_hi = {{8{writeMaskInByte[3]}}, {8{writeMaskInByte[2]}}};
  wire [31:0]  writeMaskInBit_lo_lo = {writeMaskInBit_lo_lo_hi, writeMaskInBit_lo_lo_lo};
  wire [15:0]  writeMaskInBit_lo_hi_lo = {{8{writeMaskInByte[5]}}, {8{writeMaskInByte[4]}}};
  wire [15:0]  writeMaskInBit_lo_hi_hi = {{8{writeMaskInByte[7]}}, {8{writeMaskInByte[6]}}};
  wire [31:0]  writeMaskInBit_lo_hi = {writeMaskInBit_lo_hi_hi, writeMaskInBit_lo_hi_lo};
  wire [63:0]  writeMaskInBit_lo = {writeMaskInBit_lo_hi, writeMaskInBit_lo_lo};
  wire [15:0]  writeMaskInBit_hi_lo_lo = {{8{writeMaskInByte[9]}}, {8{writeMaskInByte[8]}}};
  wire [15:0]  writeMaskInBit_hi_lo_hi = {{8{writeMaskInByte[11]}}, {8{writeMaskInByte[10]}}};
  wire [31:0]  writeMaskInBit_hi_lo = {writeMaskInBit_hi_lo_hi, writeMaskInBit_hi_lo_lo};
  wire [15:0]  writeMaskInBit_hi_hi_lo = {{8{writeMaskInByte[13]}}, {8{writeMaskInByte[12]}}};
  wire [15:0]  writeMaskInBit_hi_hi_hi = {{8{writeMaskInByte[15]}}, {8{writeMaskInByte[14]}}};
  wire [31:0]  writeMaskInBit_hi_hi = {writeMaskInBit_hi_hi_hi, writeMaskInBit_hi_hi_lo};
  wire [63:0]  writeMaskInBit_hi = {writeMaskInBit_hi_hi, writeMaskInBit_hi_lo};
  wire [127:0] writeMaskInBit = {writeMaskInBit_hi, writeMaskInBit_lo};
  wire [6:0]   dataOffset = {writeIndex, 3'h0};
  wire [158:0] _executeResult_T = {127'h0, responseFromVfu_bits_data} << dataOffset;
  wire [32:0]  executeResult = _executeResult_T[32:0];
  wire [127:0] resultUpdate = {95'h0, writeMaskInBit[32:0] & executeResult | ~(writeMaskInBit[32:0]) & responseData};
  assign updateFFO = responseFromVfu_bits_ffoSuccess | ffoSuccess;
  wire         responseWire_bits_ffoSuccess = updateFFO;
  wire         updateVxsat = responseFromVfu_bits_clipFail | vxsatResult;
  wire         requestFromSlot_ready_0 = ~requestReg_valid | isLastResponse;
  wire         responseWire_valid = isLastResponse & requestReg_valid;
  wire [31:0]  responseWire_bits_data = resultUpdate[31:0];
  wire [3:0]   responseWire_bits_vxsat = {3'h0, updateVxsat};
  reg          pipeResponse_valid;
  reg  [31:0]  pipeResponse_bits_data;
  reg          pipeResponse_bits_ffoSuccess;
  reg  [3:0]   pipeResponse_bits_vxsat;
  reg  [1:0]   pipeResponse_bits_tag;
  always @(posedge clock) begin
    if (reset) begin
      requestReg_valid <= 1'h0;
      requestReg_bits_src_0 <= 33'h0;
      requestReg_bits_src_1 <= 33'h0;
      requestReg_bits_src_2 <= 33'h0;
      requestReg_bits_src_3 <= 33'h0;
      requestReg_bits_opcode <= 4'h0;
      requestReg_bits_mask <= 4'h0;
      requestReg_bits_executeMask <= 4'h0;
      requestReg_bits_sign0 <= 1'h0;
      requestReg_bits_sign <= 1'h0;
      requestReg_bits_reverse <= 1'h0;
      requestReg_bits_average <= 1'h0;
      requestReg_bits_saturate <= 1'h0;
      requestReg_bits_vxrm <= 2'h0;
      requestReg_bits_vSew <= 2'h0;
      requestReg_bits_shifterSize <= 20'h0;
      requestReg_bits_rem <= 1'h0;
      requestReg_bits_executeIndex <= 2'h0;
      requestReg_bits_popInit <= 12'h0;
      requestReg_bits_groupIndex <= 8'h0;
      requestReg_bits_laneIndex <= 2'h0;
      requestReg_bits_maskType <= 1'h0;
      requestReg_bits_narrow <= 1'h0;
      requestReg_bits_tag <= 2'h0;
      sendRequestValid <= 1'h0;
      ffoSuccess <= 1'h0;
      vxsatResult <= 1'h0;
      responseData <= 33'h0;
      executeIndex <= 2'h0;
      pipeResponse_valid <= 1'h0;
      pipeResponse_bits_data <= 32'h0;
      pipeResponse_bits_ffoSuccess <= 1'h0;
      pipeResponse_bits_vxsat <= 4'h0;
      pipeResponse_bits_tag <= 2'h0;
    end
    else begin
      automatic logic _vxsatResult_T;
      _vxsatResult_T = requestFromSlot_ready_0 & requestFromSlot_valid_0;
      if (_vxsatResult_T ^ responseWire_valid)
        requestReg_valid <= _vxsatResult_T;
      if (_vxsatResult_T) begin
        requestReg_bits_src_0 <= requestFromSlot_bits_src_0_0;
        requestReg_bits_src_1 <= requestFromSlot_bits_src_1_0;
        requestReg_bits_src_2 <= requestFromSlot_bits_src_2_0;
        requestReg_bits_src_3 <= requestFromSlot_bits_src_3_0;
        requestReg_bits_opcode <= requestFromSlot_bits_opcode_0;
        requestReg_bits_mask <= requestFromSlot_bits_mask_0;
        requestReg_bits_executeMask <= requestFromSlot_bits_executeMask_0;
        requestReg_bits_sign0 <= requestFromSlot_bits_sign0_0;
        requestReg_bits_sign <= requestFromSlot_bits_sign_0;
        requestReg_bits_reverse <= requestFromSlot_bits_reverse_0;
        requestReg_bits_average <= requestFromSlot_bits_average_0;
        requestReg_bits_saturate <= requestFromSlot_bits_saturate_0;
        requestReg_bits_vxrm <= requestFromSlot_bits_vxrm_0;
        requestReg_bits_vSew <= requestFromSlot_bits_vSew_0;
        requestReg_bits_shifterSize <= requestFromSlot_bits_shifterSize_0;
        requestReg_bits_rem <= requestFromSlot_bits_rem_0;
        requestReg_bits_executeIndex <= requestFromSlot_bits_executeIndex_0;
        requestReg_bits_popInit <= requestFromSlot_bits_popInit_0;
        requestReg_bits_groupIndex <= requestFromSlot_bits_groupIndex_0;
        requestReg_bits_laneIndex <= requestFromSlot_bits_laneIndex_0;
        requestReg_bits_maskType <= requestFromSlot_bits_maskType_0;
        requestReg_bits_narrow <= requestFromSlot_bits_narrow_0;
        requestReg_bits_tag <= requestFromSlot_bits_tag_0;
      end
      if (_vxsatResult_T | lastRequestFire)
        sendRequestValid <= _vxsatResult_T;
      if (responseFromVfu_valid | _vxsatResult_T) begin
        ffoSuccess <= updateFFO & ~_vxsatResult_T;
        vxsatResult <= updateVxsat & ~_vxsatResult_T;
      end
      if (responseFromVfu_valid)
        responseData <= resultUpdate[32:0];
      if (executeQueue_enq_valid | _vxsatResult_T)
        executeIndex <= _vxsatResult_T ? nextExecuteIndexForNextGroup : nextExecuteIndex;
      pipeResponse_valid <= responseWire_valid;
      pipeResponse_bits_data <= responseWire_bits_data;
      pipeResponse_bits_ffoSuccess <= responseWire_bits_ffoSuccess;
      pipeResponse_bits_vxsat <= responseWire_bits_vxsat;
      pipeResponse_bits_tag <= responseWire_bits_tag;
    end
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:9];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [3:0] i = 4'h0; i < 4'hA; i += 4'h1) begin
          _RANDOM[i] = `RANDOM;
        end
        requestReg_valid = _RANDOM[4'h0][0];
        requestReg_bits_src_0 = {_RANDOM[4'h0][31:1], _RANDOM[4'h1][1:0]};
        requestReg_bits_src_1 = {_RANDOM[4'h1][31:2], _RANDOM[4'h2][2:0]};
        requestReg_bits_src_2 = {_RANDOM[4'h2][31:3], _RANDOM[4'h3][3:0]};
        requestReg_bits_src_3 = {_RANDOM[4'h3][31:4], _RANDOM[4'h4][4:0]};
        requestReg_bits_opcode = _RANDOM[4'h4][8:5];
        requestReg_bits_mask = _RANDOM[4'h4][12:9];
        requestReg_bits_executeMask = _RANDOM[4'h4][16:13];
        requestReg_bits_sign0 = _RANDOM[4'h4][17];
        requestReg_bits_sign = _RANDOM[4'h4][18];
        requestReg_bits_reverse = _RANDOM[4'h4][19];
        requestReg_bits_average = _RANDOM[4'h4][20];
        requestReg_bits_saturate = _RANDOM[4'h4][21];
        requestReg_bits_vxrm = _RANDOM[4'h4][23:22];
        requestReg_bits_vSew = _RANDOM[4'h4][25:24];
        requestReg_bits_shifterSize = {_RANDOM[4'h4][31:26], _RANDOM[4'h5][13:0]};
        requestReg_bits_rem = _RANDOM[4'h5][14];
        requestReg_bits_executeIndex = _RANDOM[4'h5][16:15];
        requestReg_bits_popInit = _RANDOM[4'h5][28:17];
        requestReg_bits_groupIndex = {_RANDOM[4'h5][31:29], _RANDOM[4'h6][4:0]};
        requestReg_bits_laneIndex = _RANDOM[4'h6][6:5];
        requestReg_bits_maskType = _RANDOM[4'h6][8];
        requestReg_bits_narrow = _RANDOM[4'h6][9];
        requestReg_bits_tag = _RANDOM[4'h6][11:10];
        sendRequestValid = _RANDOM[4'h6][12];
        ffoSuccess = _RANDOM[4'h6][13];
        vxsatResult = _RANDOM[4'h6][14];
        responseData = {_RANDOM[4'h6][31:15], _RANDOM[4'h7][15:0]};
        executeIndex = _RANDOM[4'h7][17:16];
        pipeResponse_valid = _RANDOM[4'h7][18];
        pipeResponse_bits_data = {_RANDOM[4'h7][31:19], _RANDOM[4'h8][18:0]};
        pipeResponse_bits_ffoSuccess = _RANDOM[4'h8][22];
        pipeResponse_bits_vxsat = _RANDOM[4'h8][31:28];
        pipeResponse_bits_tag = _RANDOM[4'h9][6:5];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  wire         executeQueue_empty;
  assign executeQueue_empty = _executeQueue_fifo_empty;
  wire         executeQueue_full;
  assign executeQueue_full = _executeQueue_fifo_full;
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(3)
  ) executeQueue_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(executeQueue_enq_ready & executeQueue_enq_valid)),
    .pop_req_n    (~(executeQueue_deq_ready & ~_executeQueue_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (executeQueue_enq_bits),
    .empty        (_executeQueue_fifo_empty),
    .almost_empty (executeQueue_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (executeQueue_almostFull),
    .full         (_executeQueue_fifo_full),
    .error        (_executeQueue_fifo_error),
    .data_out     (executeQueue_deq_bits)
  );
  assign requestToVfu_valid = requestToVfu_valid_0;
  assign requestToVfu_bits_src_0 = requestToVfu_bits_src_0_0;
  assign requestToVfu_bits_src_1 = requestToVfu_bits_src_1_0;
  assign requestToVfu_bits_src_2 = requestToVfu_bits_src_2_0;
  assign requestToVfu_bits_src_3 = requestToVfu_bits_src_3_0;
  assign requestToVfu_bits_opcode = requestToVfu_bits_opcode_0;
  assign requestToVfu_bits_mask = requestToVfu_bits_mask_0;
  assign requestToVfu_bits_sign = requestToVfu_bits_sign_0;
  assign requestToVfu_bits_vxrm = requestToVfu_bits_vxrm_0;
  assign requestToVfu_bits_vSew = requestToVfu_bits_vSew_0;
  assign requestToVfu_bits_shifterSize = requestToVfu_bits_shifterSize_0;
  assign requestToVfu_bits_executeIndex = requestToVfu_bits_executeIndex_0;
  assign requestToVfu_bits_popInit = requestToVfu_bits_popInit_0;
  assign requestToVfu_bits_groupIndex = requestToVfu_bits_groupIndex_0;
  assign requestToVfu_bits_laneIndex = requestToVfu_bits_laneIndex_0;
  assign requestToVfu_bits_maskType = requestToVfu_bits_maskType_0;
  assign requestToVfu_bits_narrow = requestToVfu_bits_narrow_0;
  assign requestToVfu_bits_tag = requestToVfu_bits_tag_0;
  assign requestFromSlot_ready = requestFromSlot_ready_0;
  assign responseToSlot_valid = pipeResponse_valid;
  assign responseToSlot_bits_data = pipeResponse_bits_data;
  assign responseToSlot_bits_ffoSuccess = pipeResponse_bits_ffoSuccess;
  assign responseToSlot_bits_vxsat = pipeResponse_bits_vxsat;
  assign responseToSlot_bits_tag = pipeResponse_bits_tag;
endmodule

