
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
module LaneStage3(
  input         clock,
                reset,
  output        enqueue_ready,
  input         enqueue_valid,
  input  [8:0]  enqueue_bits_groupCounter,
  input  [31:0] enqueue_bits_data,
                enqueue_bits_pipeData,
  input  [3:0]  enqueue_bits_mask,
  input  [9:0]  enqueue_bits_ffoIndex,
  input  [31:0] enqueue_bits_crossWriteData_0,
                enqueue_bits_crossWriteData_1,
  input         enqueue_bits_sSendResponse,
                enqueue_bits_ffoSuccess,
                enqueue_bits_decodeResult_specialSlot,
  input  [4:0]  enqueue_bits_decodeResult_topUop,
  input         enqueue_bits_decodeResult_popCount,
                enqueue_bits_decodeResult_ffo,
                enqueue_bits_decodeResult_average,
                enqueue_bits_decodeResult_reverse,
                enqueue_bits_decodeResult_dontNeedExecuteInLane,
                enqueue_bits_decodeResult_scheduler,
                enqueue_bits_decodeResult_sReadVD,
                enqueue_bits_decodeResult_vtype,
                enqueue_bits_decodeResult_sWrite,
                enqueue_bits_decodeResult_crossRead,
                enqueue_bits_decodeResult_crossWrite,
                enqueue_bits_decodeResult_maskUnit,
                enqueue_bits_decodeResult_special,
                enqueue_bits_decodeResult_saturate,
                enqueue_bits_decodeResult_vwmacc,
                enqueue_bits_decodeResult_readOnly,
                enqueue_bits_decodeResult_maskSource,
                enqueue_bits_decodeResult_maskDestination,
                enqueue_bits_decodeResult_maskLogic,
  input  [3:0]  enqueue_bits_decodeResult_uop,
  input         enqueue_bits_decodeResult_iota,
                enqueue_bits_decodeResult_mv,
                enqueue_bits_decodeResult_extend,
                enqueue_bits_decodeResult_unOrderWrite,
                enqueue_bits_decodeResult_compress,
                enqueue_bits_decodeResult_gather16,
                enqueue_bits_decodeResult_gather,
                enqueue_bits_decodeResult_slid,
                enqueue_bits_decodeResult_targetRd,
                enqueue_bits_decodeResult_widenReduce,
                enqueue_bits_decodeResult_red,
                enqueue_bits_decodeResult_nr,
                enqueue_bits_decodeResult_itype,
                enqueue_bits_decodeResult_unsigned1,
                enqueue_bits_decodeResult_unsigned0,
                enqueue_bits_decodeResult_other,
                enqueue_bits_decodeResult_multiCycle,
                enqueue_bits_decodeResult_divider,
                enqueue_bits_decodeResult_multiplier,
                enqueue_bits_decodeResult_shift,
                enqueue_bits_decodeResult_adder,
                enqueue_bits_decodeResult_logic,
  input  [2:0]  enqueue_bits_instructionIndex,
  input         enqueue_bits_loadStore,
  input  [4:0]  enqueue_bits_vd,
  input         vrfWriteRequest_ready,
  output        vrfWriteRequest_valid,
  output [4:0]  vrfWriteRequest_bits_vd,
                vrfWriteRequest_bits_offset,
  output [3:0]  vrfWriteRequest_bits_mask,
  output [31:0] vrfWriteRequest_bits_data,
  output        vrfWriteRequest_bits_last,
  output [2:0]  vrfWriteRequest_bits_instructionIndex,
  input         crossWritePort_0_ready,
  output        crossWritePort_0_valid,
  output [31:0] crossWritePort_0_bits_data,
  output [1:0]  crossWritePort_0_bits_mask,
  output [2:0]  crossWritePort_0_bits_instructionIndex,
  output [8:0]  crossWritePort_0_bits_counter,
  input         crossWritePort_1_ready,
  output        crossWritePort_1_valid,
  output [31:0] crossWritePort_1_bits_data,
  output [1:0]  crossWritePort_1_bits_mask,
  output [2:0]  crossWritePort_1_bits_instructionIndex,
  output [8:0]  crossWritePort_1_bits_counter
);

  wire        _vrfPtrReplica_fifo_empty;
  wire        _vrfPtrReplica_fifo_full;
  wire        _vrfPtrReplica_fifo_error;
  wire        _vrfWriteQueue_fifo_empty;
  wire        _vrfWriteQueue_fifo_full;
  wire        _vrfWriteQueue_fifo_error;
  wire [49:0] _vrfWriteQueue_fifo_data_out;
  wire        vrfPtrReplica_almostFull;
  wire        vrfPtrReplica_almostEmpty;
  wire        vrfWriteQueue_almostFull;
  wire        vrfWriteQueue_almostEmpty;
  wire        vrfWriteQueue_enq_valid;
  wire [4:0]  vrfWriteQueue_enq_bits_offset;
  wire        enqueue_valid_0 = enqueue_valid;
  wire [8:0]  enqueue_bits_groupCounter_0 = enqueue_bits_groupCounter;
  wire [31:0] enqueue_bits_data_0 = enqueue_bits_data;
  wire [31:0] enqueue_bits_pipeData_0 = enqueue_bits_pipeData;
  wire [3:0]  enqueue_bits_mask_0 = enqueue_bits_mask;
  wire [9:0]  enqueue_bits_ffoIndex_0 = enqueue_bits_ffoIndex;
  wire [31:0] enqueue_bits_crossWriteData_0_0 = enqueue_bits_crossWriteData_0;
  wire [31:0] enqueue_bits_crossWriteData_1_0 = enqueue_bits_crossWriteData_1;
  wire        enqueue_bits_sSendResponse_0 = enqueue_bits_sSendResponse;
  wire        enqueue_bits_ffoSuccess_0 = enqueue_bits_ffoSuccess;
  wire        enqueue_bits_decodeResult_specialSlot_0 = enqueue_bits_decodeResult_specialSlot;
  wire [4:0]  enqueue_bits_decodeResult_topUop_0 = enqueue_bits_decodeResult_topUop;
  wire        enqueue_bits_decodeResult_popCount_0 = enqueue_bits_decodeResult_popCount;
  wire        enqueue_bits_decodeResult_ffo_0 = enqueue_bits_decodeResult_ffo;
  wire        enqueue_bits_decodeResult_average_0 = enqueue_bits_decodeResult_average;
  wire        enqueue_bits_decodeResult_reverse_0 = enqueue_bits_decodeResult_reverse;
  wire        enqueue_bits_decodeResult_dontNeedExecuteInLane_0 = enqueue_bits_decodeResult_dontNeedExecuteInLane;
  wire        enqueue_bits_decodeResult_scheduler_0 = enqueue_bits_decodeResult_scheduler;
  wire        enqueue_bits_decodeResult_sReadVD_0 = enqueue_bits_decodeResult_sReadVD;
  wire        enqueue_bits_decodeResult_vtype_0 = enqueue_bits_decodeResult_vtype;
  wire        enqueue_bits_decodeResult_sWrite_0 = enqueue_bits_decodeResult_sWrite;
  wire        enqueue_bits_decodeResult_crossRead_0 = enqueue_bits_decodeResult_crossRead;
  wire        enqueue_bits_decodeResult_crossWrite_0 = enqueue_bits_decodeResult_crossWrite;
  wire        enqueue_bits_decodeResult_maskUnit_0 = enqueue_bits_decodeResult_maskUnit;
  wire        enqueue_bits_decodeResult_special_0 = enqueue_bits_decodeResult_special;
  wire        enqueue_bits_decodeResult_saturate_0 = enqueue_bits_decodeResult_saturate;
  wire        enqueue_bits_decodeResult_vwmacc_0 = enqueue_bits_decodeResult_vwmacc;
  wire        enqueue_bits_decodeResult_readOnly_0 = enqueue_bits_decodeResult_readOnly;
  wire        enqueue_bits_decodeResult_maskSource_0 = enqueue_bits_decodeResult_maskSource;
  wire        enqueue_bits_decodeResult_maskDestination_0 = enqueue_bits_decodeResult_maskDestination;
  wire        enqueue_bits_decodeResult_maskLogic_0 = enqueue_bits_decodeResult_maskLogic;
  wire [3:0]  enqueue_bits_decodeResult_uop_0 = enqueue_bits_decodeResult_uop;
  wire        enqueue_bits_decodeResult_iota_0 = enqueue_bits_decodeResult_iota;
  wire        enqueue_bits_decodeResult_mv_0 = enqueue_bits_decodeResult_mv;
  wire        enqueue_bits_decodeResult_extend_0 = enqueue_bits_decodeResult_extend;
  wire        enqueue_bits_decodeResult_unOrderWrite_0 = enqueue_bits_decodeResult_unOrderWrite;
  wire        enqueue_bits_decodeResult_compress_0 = enqueue_bits_decodeResult_compress;
  wire        enqueue_bits_decodeResult_gather16_0 = enqueue_bits_decodeResult_gather16;
  wire        enqueue_bits_decodeResult_gather_0 = enqueue_bits_decodeResult_gather;
  wire        enqueue_bits_decodeResult_slid_0 = enqueue_bits_decodeResult_slid;
  wire        enqueue_bits_decodeResult_targetRd_0 = enqueue_bits_decodeResult_targetRd;
  wire        enqueue_bits_decodeResult_widenReduce_0 = enqueue_bits_decodeResult_widenReduce;
  wire        enqueue_bits_decodeResult_red_0 = enqueue_bits_decodeResult_red;
  wire        enqueue_bits_decodeResult_nr_0 = enqueue_bits_decodeResult_nr;
  wire        enqueue_bits_decodeResult_itype_0 = enqueue_bits_decodeResult_itype;
  wire        enqueue_bits_decodeResult_unsigned1_0 = enqueue_bits_decodeResult_unsigned1;
  wire        enqueue_bits_decodeResult_unsigned0_0 = enqueue_bits_decodeResult_unsigned0;
  wire        enqueue_bits_decodeResult_other_0 = enqueue_bits_decodeResult_other;
  wire        enqueue_bits_decodeResult_multiCycle_0 = enqueue_bits_decodeResult_multiCycle;
  wire        enqueue_bits_decodeResult_divider_0 = enqueue_bits_decodeResult_divider;
  wire        enqueue_bits_decodeResult_multiplier_0 = enqueue_bits_decodeResult_multiplier;
  wire        enqueue_bits_decodeResult_shift_0 = enqueue_bits_decodeResult_shift;
  wire        enqueue_bits_decodeResult_adder_0 = enqueue_bits_decodeResult_adder;
  wire        enqueue_bits_decodeResult_logic_0 = enqueue_bits_decodeResult_logic;
  wire [2:0]  enqueue_bits_instructionIndex_0 = enqueue_bits_instructionIndex;
  wire        enqueue_bits_loadStore_0 = enqueue_bits_loadStore;
  wire [4:0]  enqueue_bits_vd_0 = enqueue_bits_vd;
  wire        vrfWriteRequest_ready_0 = vrfWriteRequest_ready;
  wire        crossWritePort_0_ready_0 = crossWritePort_0_ready;
  wire        crossWritePort_1_ready_0 = crossWritePort_1_ready;
  wire        enqueue_bits_ffoByOtherLanes = 1'h0;
  wire        vrfWriteQueue_enq_bits_last = 1'h0;
  wire        vrfWriteQueue_deq_ready = vrfWriteRequest_ready_0;
  wire        vrfPtrReplica_deq_valid;
  wire [4:0]  vrfWriteQueue_deq_bits_vd;
  wire [4:0]  vrfPtrReplica_deq_bits;
  wire [3:0]  vrfWriteQueue_deq_bits_mask;
  wire [31:0] vrfWriteQueue_deq_bits_data;
  wire        vrfWriteQueue_deq_bits_last;
  wire [2:0]  vrfWriteQueue_deq_bits_instructionIndex;
  reg  [8:0]  pipeEnqueue_groupCounter;
  wire [8:0]  crossWritePort_0_bits_counter_0 = pipeEnqueue_groupCounter;
  wire [8:0]  crossWritePort_1_bits_counter_0 = pipeEnqueue_groupCounter;
  reg  [31:0] pipeEnqueue_data;
  reg  [31:0] pipeEnqueue_pipeData;
  reg  [3:0]  pipeEnqueue_mask;
  wire [3:0]  vrfWriteQueue_enq_bits_mask = pipeEnqueue_mask;
  reg  [9:0]  pipeEnqueue_ffoIndex;
  reg  [31:0] pipeEnqueue_crossWriteData_0;
  wire [31:0] crossWritePort_0_bits_data_0 = pipeEnqueue_crossWriteData_0;
  reg  [31:0] pipeEnqueue_crossWriteData_1;
  wire [31:0] crossWritePort_1_bits_data_0 = pipeEnqueue_crossWriteData_1;
  reg         pipeEnqueue_sSendResponse;
  reg         pipeEnqueue_ffoSuccess;
  reg         pipeEnqueue_decodeResult_specialSlot;
  reg  [4:0]  pipeEnqueue_decodeResult_topUop;
  reg         pipeEnqueue_decodeResult_popCount;
  reg         pipeEnqueue_decodeResult_ffo;
  reg         pipeEnqueue_decodeResult_average;
  reg         pipeEnqueue_decodeResult_reverse;
  reg         pipeEnqueue_decodeResult_dontNeedExecuteInLane;
  reg         pipeEnqueue_decodeResult_scheduler;
  reg         pipeEnqueue_decodeResult_sReadVD;
  reg         pipeEnqueue_decodeResult_vtype;
  reg         pipeEnqueue_decodeResult_sWrite;
  reg         pipeEnqueue_decodeResult_crossRead;
  reg         pipeEnqueue_decodeResult_crossWrite;
  reg         pipeEnqueue_decodeResult_maskUnit;
  reg         pipeEnqueue_decodeResult_special;
  reg         pipeEnqueue_decodeResult_saturate;
  reg         pipeEnqueue_decodeResult_vwmacc;
  reg         pipeEnqueue_decodeResult_readOnly;
  reg         pipeEnqueue_decodeResult_maskSource;
  reg         pipeEnqueue_decodeResult_maskDestination;
  reg         pipeEnqueue_decodeResult_maskLogic;
  reg  [3:0]  pipeEnqueue_decodeResult_uop;
  reg         pipeEnqueue_decodeResult_iota;
  reg         pipeEnqueue_decodeResult_mv;
  reg         pipeEnqueue_decodeResult_extend;
  reg         pipeEnqueue_decodeResult_unOrderWrite;
  reg         pipeEnqueue_decodeResult_compress;
  reg         pipeEnqueue_decodeResult_gather16;
  reg         pipeEnqueue_decodeResult_gather;
  reg         pipeEnqueue_decodeResult_slid;
  reg         pipeEnqueue_decodeResult_targetRd;
  reg         pipeEnqueue_decodeResult_widenReduce;
  reg         pipeEnqueue_decodeResult_red;
  reg         pipeEnqueue_decodeResult_nr;
  reg         pipeEnqueue_decodeResult_itype;
  reg         pipeEnqueue_decodeResult_unsigned1;
  reg         pipeEnqueue_decodeResult_unsigned0;
  reg         pipeEnqueue_decodeResult_other;
  reg         pipeEnqueue_decodeResult_multiCycle;
  reg         pipeEnqueue_decodeResult_divider;
  reg         pipeEnqueue_decodeResult_multiplier;
  reg         pipeEnqueue_decodeResult_shift;
  reg         pipeEnqueue_decodeResult_adder;
  reg         pipeEnqueue_decodeResult_logic;
  reg  [2:0]  pipeEnqueue_instructionIndex;
  wire [2:0]  crossWritePort_0_bits_instructionIndex_0 = pipeEnqueue_instructionIndex;
  wire [2:0]  crossWritePort_1_bits_instructionIndex_0 = pipeEnqueue_instructionIndex;
  wire [2:0]  vrfWriteQueue_enq_bits_instructionIndex = pipeEnqueue_instructionIndex;
  reg         pipeEnqueue_loadStore;
  reg  [4:0]  pipeEnqueue_vd;
  reg         stageValidReg;
  reg         sCrossWriteLSB;
  reg         sCrossWriteMSB;
  wire        vrfPtrReplica_enq_valid = vrfWriteQueue_enq_valid;
  wire [4:0]  vrfPtrReplica_enq_bits = vrfWriteQueue_enq_bits_offset;
  wire [31:0] dataSelect;
  wire        vrfPtrReplica_deq_ready = vrfWriteQueue_deq_ready;
  wire        vrfWriteQueue_deq_valid;
  assign vrfWriteQueue_deq_valid = ~_vrfWriteQueue_fifo_empty;
  wire [4:0]  vrfWriteQueue_dataOut_vd;
  wire [4:0]  vrfWriteRequest_bits_vd_0 = vrfWriteQueue_deq_bits_vd;
  wire [4:0]  vrfWriteQueue_dataOut_offset;
  wire [3:0]  vrfWriteQueue_dataOut_mask;
  wire [3:0]  vrfWriteRequest_bits_mask_0 = vrfWriteQueue_deq_bits_mask;
  wire [31:0] vrfWriteQueue_dataOut_data;
  wire [31:0] vrfWriteRequest_bits_data_0 = vrfWriteQueue_deq_bits_data;
  wire        vrfWriteQueue_dataOut_last;
  wire        vrfWriteRequest_bits_last_0 = vrfWriteQueue_deq_bits_last;
  wire [2:0]  vrfWriteQueue_dataOut_instructionIndex;
  wire [2:0]  vrfWriteRequest_bits_instructionIndex_0 = vrfWriteQueue_deq_bits_instructionIndex;
  wire [31:0] vrfWriteQueue_enq_bits_data;
  wire [32:0] vrfWriteQueue_dataIn_lo_hi = {vrfWriteQueue_enq_bits_data, 1'h0};
  wire [35:0] vrfWriteQueue_dataIn_lo = {vrfWriteQueue_dataIn_lo_hi, vrfWriteQueue_enq_bits_instructionIndex};
  wire [4:0]  vrfWriteQueue_enq_bits_vd;
  wire [9:0]  vrfWriteQueue_dataIn_hi_hi = {vrfWriteQueue_enq_bits_vd, vrfWriteQueue_enq_bits_offset};
  wire [13:0] vrfWriteQueue_dataIn_hi = {vrfWriteQueue_dataIn_hi_hi, vrfWriteQueue_enq_bits_mask};
  wire [49:0] vrfWriteQueue_dataIn = {vrfWriteQueue_dataIn_hi, vrfWriteQueue_dataIn_lo};
  assign vrfWriteQueue_dataOut_instructionIndex = _vrfWriteQueue_fifo_data_out[2:0];
  assign vrfWriteQueue_dataOut_last = _vrfWriteQueue_fifo_data_out[3];
  assign vrfWriteQueue_dataOut_data = _vrfWriteQueue_fifo_data_out[35:4];
  assign vrfWriteQueue_dataOut_mask = _vrfWriteQueue_fifo_data_out[39:36];
  assign vrfWriteQueue_dataOut_offset = _vrfWriteQueue_fifo_data_out[44:40];
  assign vrfWriteQueue_dataOut_vd = _vrfWriteQueue_fifo_data_out[49:45];
  assign vrfWriteQueue_deq_bits_vd = vrfWriteQueue_dataOut_vd;
  wire [4:0]  vrfWriteQueue_deq_bits_offset = vrfWriteQueue_dataOut_offset;
  assign vrfWriteQueue_deq_bits_mask = vrfWriteQueue_dataOut_mask;
  assign vrfWriteQueue_deq_bits_data = vrfWriteQueue_dataOut_data;
  assign vrfWriteQueue_deq_bits_last = vrfWriteQueue_dataOut_last;
  assign vrfWriteQueue_deq_bits_instructionIndex = vrfWriteQueue_dataOut_instructionIndex;
  wire        vrfWriteQueue_enq_ready = ~_vrfWriteQueue_fifo_full;
  assign vrfPtrReplica_deq_valid = ~_vrfPtrReplica_fifo_empty;
  wire        vrfWriteRequest_valid_0 = vrfPtrReplica_deq_valid;
  wire [4:0]  vrfWriteRequest_bits_offset_0 = vrfPtrReplica_deq_bits;
  wire        vrfPtrReplica_enq_ready = ~_vrfPtrReplica_fifo_full;
  wire        vrfWriteReady = vrfWriteQueue_enq_ready | pipeEnqueue_decodeResult_sWrite;
  wire        crossWritePort_0_valid_0 = stageValidReg & ~sCrossWriteLSB;
  wire [1:0]  crossWritePort_0_bits_mask_0 = pipeEnqueue_mask[1:0];
  wire        crossWritePort_1_valid_0 = stageValidReg & ~sCrossWriteMSB;
  wire [1:0]  crossWritePort_1_bits_mask_0 = pipeEnqueue_mask[3:2];
  assign dataSelect = pipeEnqueue_decodeResult_nr ? pipeEnqueue_pipeData : pipeEnqueue_data;
  assign vrfWriteQueue_enq_bits_data = dataSelect;
  assign vrfWriteQueue_enq_valid = stageValidReg & ~pipeEnqueue_decodeResult_sWrite;
  assign vrfWriteQueue_enq_bits_vd = pipeEnqueue_vd + {1'h0, pipeEnqueue_groupCounter[8:5]};
  assign vrfWriteQueue_enq_bits_offset = pipeEnqueue_groupCounter[4:0];
  wire        CrossLaneWriteOver = sCrossWriteLSB & sCrossWriteMSB;
  wire        enqueue_ready_0 = ~stageValidReg | CrossLaneWriteOver & vrfWriteReady;
  wire        dequeueFire = stageValidReg & CrossLaneWriteOver & vrfWriteReady;
  always @(posedge clock) begin
    if (reset) begin
      pipeEnqueue_groupCounter <= 9'h0;
      pipeEnqueue_data <= 32'h0;
      pipeEnqueue_pipeData <= 32'h0;
      pipeEnqueue_mask <= 4'h0;
      pipeEnqueue_ffoIndex <= 10'h0;
      pipeEnqueue_crossWriteData_0 <= 32'h0;
      pipeEnqueue_crossWriteData_1 <= 32'h0;
      pipeEnqueue_sSendResponse <= 1'h0;
      pipeEnqueue_ffoSuccess <= 1'h0;
      pipeEnqueue_decodeResult_specialSlot <= 1'h0;
      pipeEnqueue_decodeResult_topUop <= 5'h0;
      pipeEnqueue_decodeResult_popCount <= 1'h0;
      pipeEnqueue_decodeResult_ffo <= 1'h0;
      pipeEnqueue_decodeResult_average <= 1'h0;
      pipeEnqueue_decodeResult_reverse <= 1'h0;
      pipeEnqueue_decodeResult_dontNeedExecuteInLane <= 1'h0;
      pipeEnqueue_decodeResult_scheduler <= 1'h0;
      pipeEnqueue_decodeResult_sReadVD <= 1'h0;
      pipeEnqueue_decodeResult_vtype <= 1'h0;
      pipeEnqueue_decodeResult_sWrite <= 1'h0;
      pipeEnqueue_decodeResult_crossRead <= 1'h0;
      pipeEnqueue_decodeResult_crossWrite <= 1'h0;
      pipeEnqueue_decodeResult_maskUnit <= 1'h0;
      pipeEnqueue_decodeResult_special <= 1'h0;
      pipeEnqueue_decodeResult_saturate <= 1'h0;
      pipeEnqueue_decodeResult_vwmacc <= 1'h0;
      pipeEnqueue_decodeResult_readOnly <= 1'h0;
      pipeEnqueue_decodeResult_maskSource <= 1'h0;
      pipeEnqueue_decodeResult_maskDestination <= 1'h0;
      pipeEnqueue_decodeResult_maskLogic <= 1'h0;
      pipeEnqueue_decodeResult_uop <= 4'h0;
      pipeEnqueue_decodeResult_iota <= 1'h0;
      pipeEnqueue_decodeResult_mv <= 1'h0;
      pipeEnqueue_decodeResult_extend <= 1'h0;
      pipeEnqueue_decodeResult_unOrderWrite <= 1'h0;
      pipeEnqueue_decodeResult_compress <= 1'h0;
      pipeEnqueue_decodeResult_gather16 <= 1'h0;
      pipeEnqueue_decodeResult_gather <= 1'h0;
      pipeEnqueue_decodeResult_slid <= 1'h0;
      pipeEnqueue_decodeResult_targetRd <= 1'h0;
      pipeEnqueue_decodeResult_widenReduce <= 1'h0;
      pipeEnqueue_decodeResult_red <= 1'h0;
      pipeEnqueue_decodeResult_nr <= 1'h0;
      pipeEnqueue_decodeResult_itype <= 1'h0;
      pipeEnqueue_decodeResult_unsigned1 <= 1'h0;
      pipeEnqueue_decodeResult_unsigned0 <= 1'h0;
      pipeEnqueue_decodeResult_other <= 1'h0;
      pipeEnqueue_decodeResult_multiCycle <= 1'h0;
      pipeEnqueue_decodeResult_divider <= 1'h0;
      pipeEnqueue_decodeResult_multiplier <= 1'h0;
      pipeEnqueue_decodeResult_shift <= 1'h0;
      pipeEnqueue_decodeResult_adder <= 1'h0;
      pipeEnqueue_decodeResult_logic <= 1'h0;
      pipeEnqueue_instructionIndex <= 3'h0;
      pipeEnqueue_loadStore <= 1'h0;
      pipeEnqueue_vd <= 5'h0;
      stageValidReg <= 1'h0;
      sCrossWriteLSB <= 1'h1;
      sCrossWriteMSB <= 1'h1;
    end
    else begin
      automatic logic _stageValidReg_T;
      _stageValidReg_T = enqueue_ready_0 & enqueue_valid_0;
      if (_stageValidReg_T) begin
        pipeEnqueue_groupCounter <= enqueue_bits_groupCounter_0;
        pipeEnqueue_data <= enqueue_bits_data_0;
        pipeEnqueue_pipeData <= enqueue_bits_pipeData_0;
        pipeEnqueue_mask <= enqueue_bits_mask_0;
        pipeEnqueue_ffoIndex <= enqueue_bits_ffoIndex_0;
        pipeEnqueue_crossWriteData_0 <= enqueue_bits_crossWriteData_0_0;
        pipeEnqueue_crossWriteData_1 <= enqueue_bits_crossWriteData_1_0;
        pipeEnqueue_sSendResponse <= enqueue_bits_sSendResponse_0;
        pipeEnqueue_ffoSuccess <= enqueue_bits_ffoSuccess_0;
        pipeEnqueue_decodeResult_specialSlot <= enqueue_bits_decodeResult_specialSlot_0;
        pipeEnqueue_decodeResult_topUop <= enqueue_bits_decodeResult_topUop_0;
        pipeEnqueue_decodeResult_popCount <= enqueue_bits_decodeResult_popCount_0;
        pipeEnqueue_decodeResult_ffo <= enqueue_bits_decodeResult_ffo_0;
        pipeEnqueue_decodeResult_average <= enqueue_bits_decodeResult_average_0;
        pipeEnqueue_decodeResult_reverse <= enqueue_bits_decodeResult_reverse_0;
        pipeEnqueue_decodeResult_dontNeedExecuteInLane <= enqueue_bits_decodeResult_dontNeedExecuteInLane_0;
        pipeEnqueue_decodeResult_scheduler <= enqueue_bits_decodeResult_scheduler_0;
        pipeEnqueue_decodeResult_sReadVD <= enqueue_bits_decodeResult_sReadVD_0;
        pipeEnqueue_decodeResult_vtype <= enqueue_bits_decodeResult_vtype_0;
        pipeEnqueue_decodeResult_sWrite <= enqueue_bits_decodeResult_sWrite_0;
        pipeEnqueue_decodeResult_crossRead <= enqueue_bits_decodeResult_crossRead_0;
        pipeEnqueue_decodeResult_crossWrite <= enqueue_bits_decodeResult_crossWrite_0;
        pipeEnqueue_decodeResult_maskUnit <= enqueue_bits_decodeResult_maskUnit_0;
        pipeEnqueue_decodeResult_special <= enqueue_bits_decodeResult_special_0;
        pipeEnqueue_decodeResult_saturate <= enqueue_bits_decodeResult_saturate_0;
        pipeEnqueue_decodeResult_vwmacc <= enqueue_bits_decodeResult_vwmacc_0;
        pipeEnqueue_decodeResult_readOnly <= enqueue_bits_decodeResult_readOnly_0;
        pipeEnqueue_decodeResult_maskSource <= enqueue_bits_decodeResult_maskSource_0;
        pipeEnqueue_decodeResult_maskDestination <= enqueue_bits_decodeResult_maskDestination_0;
        pipeEnqueue_decodeResult_maskLogic <= enqueue_bits_decodeResult_maskLogic_0;
        pipeEnqueue_decodeResult_uop <= enqueue_bits_decodeResult_uop_0;
        pipeEnqueue_decodeResult_iota <= enqueue_bits_decodeResult_iota_0;
        pipeEnqueue_decodeResult_mv <= enqueue_bits_decodeResult_mv_0;
        pipeEnqueue_decodeResult_extend <= enqueue_bits_decodeResult_extend_0;
        pipeEnqueue_decodeResult_unOrderWrite <= enqueue_bits_decodeResult_unOrderWrite_0;
        pipeEnqueue_decodeResult_compress <= enqueue_bits_decodeResult_compress_0;
        pipeEnqueue_decodeResult_gather16 <= enqueue_bits_decodeResult_gather16_0;
        pipeEnqueue_decodeResult_gather <= enqueue_bits_decodeResult_gather_0;
        pipeEnqueue_decodeResult_slid <= enqueue_bits_decodeResult_slid_0;
        pipeEnqueue_decodeResult_targetRd <= enqueue_bits_decodeResult_targetRd_0;
        pipeEnqueue_decodeResult_widenReduce <= enqueue_bits_decodeResult_widenReduce_0;
        pipeEnqueue_decodeResult_red <= enqueue_bits_decodeResult_red_0;
        pipeEnqueue_decodeResult_nr <= enqueue_bits_decodeResult_nr_0;
        pipeEnqueue_decodeResult_itype <= enqueue_bits_decodeResult_itype_0;
        pipeEnqueue_decodeResult_unsigned1 <= enqueue_bits_decodeResult_unsigned1_0;
        pipeEnqueue_decodeResult_unsigned0 <= enqueue_bits_decodeResult_unsigned0_0;
        pipeEnqueue_decodeResult_other <= enqueue_bits_decodeResult_other_0;
        pipeEnqueue_decodeResult_multiCycle <= enqueue_bits_decodeResult_multiCycle_0;
        pipeEnqueue_decodeResult_divider <= enqueue_bits_decodeResult_divider_0;
        pipeEnqueue_decodeResult_multiplier <= enqueue_bits_decodeResult_multiplier_0;
        pipeEnqueue_decodeResult_shift <= enqueue_bits_decodeResult_shift_0;
        pipeEnqueue_decodeResult_adder <= enqueue_bits_decodeResult_adder_0;
        pipeEnqueue_decodeResult_logic <= enqueue_bits_decodeResult_logic_0;
        pipeEnqueue_instructionIndex <= enqueue_bits_instructionIndex_0;
        pipeEnqueue_loadStore <= enqueue_bits_loadStore_0;
        pipeEnqueue_vd <= enqueue_bits_vd_0;
      end
      if (dequeueFire ^ _stageValidReg_T)
        stageValidReg <= _stageValidReg_T;
      sCrossWriteLSB <= crossWritePort_0_ready_0 & crossWritePort_0_valid_0 | (_stageValidReg_T ? ~enqueue_bits_decodeResult_crossWrite_0 : sCrossWriteLSB);
      sCrossWriteMSB <= crossWritePort_1_ready_0 & crossWritePort_1_valid_0 | (_stageValidReg_T ? ~enqueue_bits_decodeResult_crossWrite_0 : sCrossWriteMSB);
    end
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:6];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [2:0] i = 3'h0; i < 3'h7; i += 3'h1) begin
          _RANDOM[i] = `RANDOM;
        end
        pipeEnqueue_groupCounter = _RANDOM[3'h0][8:0];
        pipeEnqueue_data = {_RANDOM[3'h0][31:9], _RANDOM[3'h1][8:0]};
        pipeEnqueue_pipeData = {_RANDOM[3'h1][31:9], _RANDOM[3'h2][8:0]};
        pipeEnqueue_mask = _RANDOM[3'h2][12:9];
        pipeEnqueue_ffoIndex = _RANDOM[3'h2][22:13];
        pipeEnqueue_crossWriteData_0 = {_RANDOM[3'h2][31:23], _RANDOM[3'h3][22:0]};
        pipeEnqueue_crossWriteData_1 = {_RANDOM[3'h3][31:23], _RANDOM[3'h4][22:0]};
        pipeEnqueue_sSendResponse = _RANDOM[3'h4][23];
        pipeEnqueue_ffoSuccess = _RANDOM[3'h4][24];
        pipeEnqueue_decodeResult_specialSlot = _RANDOM[3'h4][25];
        pipeEnqueue_decodeResult_topUop = _RANDOM[3'h4][30:26];
        pipeEnqueue_decodeResult_popCount = _RANDOM[3'h4][31];
        pipeEnqueue_decodeResult_ffo = _RANDOM[3'h5][0];
        pipeEnqueue_decodeResult_average = _RANDOM[3'h5][1];
        pipeEnqueue_decodeResult_reverse = _RANDOM[3'h5][2];
        pipeEnqueue_decodeResult_dontNeedExecuteInLane = _RANDOM[3'h5][3];
        pipeEnqueue_decodeResult_scheduler = _RANDOM[3'h5][4];
        pipeEnqueue_decodeResult_sReadVD = _RANDOM[3'h5][5];
        pipeEnqueue_decodeResult_vtype = _RANDOM[3'h5][6];
        pipeEnqueue_decodeResult_sWrite = _RANDOM[3'h5][7];
        pipeEnqueue_decodeResult_crossRead = _RANDOM[3'h5][8];
        pipeEnqueue_decodeResult_crossWrite = _RANDOM[3'h5][9];
        pipeEnqueue_decodeResult_maskUnit = _RANDOM[3'h5][10];
        pipeEnqueue_decodeResult_special = _RANDOM[3'h5][11];
        pipeEnqueue_decodeResult_saturate = _RANDOM[3'h5][12];
        pipeEnqueue_decodeResult_vwmacc = _RANDOM[3'h5][13];
        pipeEnqueue_decodeResult_readOnly = _RANDOM[3'h5][14];
        pipeEnqueue_decodeResult_maskSource = _RANDOM[3'h5][15];
        pipeEnqueue_decodeResult_maskDestination = _RANDOM[3'h5][16];
        pipeEnqueue_decodeResult_maskLogic = _RANDOM[3'h5][17];
        pipeEnqueue_decodeResult_uop = _RANDOM[3'h5][21:18];
        pipeEnqueue_decodeResult_iota = _RANDOM[3'h5][22];
        pipeEnqueue_decodeResult_mv = _RANDOM[3'h5][23];
        pipeEnqueue_decodeResult_extend = _RANDOM[3'h5][24];
        pipeEnqueue_decodeResult_unOrderWrite = _RANDOM[3'h5][25];
        pipeEnqueue_decodeResult_compress = _RANDOM[3'h5][26];
        pipeEnqueue_decodeResult_gather16 = _RANDOM[3'h5][27];
        pipeEnqueue_decodeResult_gather = _RANDOM[3'h5][28];
        pipeEnqueue_decodeResult_slid = _RANDOM[3'h5][29];
        pipeEnqueue_decodeResult_targetRd = _RANDOM[3'h5][30];
        pipeEnqueue_decodeResult_widenReduce = _RANDOM[3'h5][31];
        pipeEnqueue_decodeResult_red = _RANDOM[3'h6][0];
        pipeEnqueue_decodeResult_nr = _RANDOM[3'h6][1];
        pipeEnqueue_decodeResult_itype = _RANDOM[3'h6][2];
        pipeEnqueue_decodeResult_unsigned1 = _RANDOM[3'h6][3];
        pipeEnqueue_decodeResult_unsigned0 = _RANDOM[3'h6][4];
        pipeEnqueue_decodeResult_other = _RANDOM[3'h6][5];
        pipeEnqueue_decodeResult_multiCycle = _RANDOM[3'h6][6];
        pipeEnqueue_decodeResult_divider = _RANDOM[3'h6][7];
        pipeEnqueue_decodeResult_multiplier = _RANDOM[3'h6][8];
        pipeEnqueue_decodeResult_shift = _RANDOM[3'h6][9];
        pipeEnqueue_decodeResult_adder = _RANDOM[3'h6][10];
        pipeEnqueue_decodeResult_logic = _RANDOM[3'h6][11];
        pipeEnqueue_instructionIndex = _RANDOM[3'h6][14:12];
        pipeEnqueue_loadStore = _RANDOM[3'h6][16];
        pipeEnqueue_vd = _RANDOM[3'h6][21:17];
        stageValidReg = _RANDOM[3'h6][22];
        sCrossWriteLSB = _RANDOM[3'h6][23];
        sCrossWriteMSB = _RANDOM[3'h6][24];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  wire        vrfWriteQueue_empty;
  assign vrfWriteQueue_empty = _vrfWriteQueue_fifo_empty;
  wire        vrfWriteQueue_full;
  assign vrfWriteQueue_full = _vrfWriteQueue_fifo_full;
  wire        vrfPtrReplica_empty;
  assign vrfPtrReplica_empty = _vrfPtrReplica_fifo_empty;
  wire        vrfPtrReplica_full;
  assign vrfPtrReplica_full = _vrfPtrReplica_fifo_full;
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(50)
  ) vrfWriteQueue_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(vrfWriteQueue_enq_ready & vrfWriteQueue_enq_valid)),
    .pop_req_n    (~(vrfWriteQueue_deq_ready & ~_vrfWriteQueue_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (vrfWriteQueue_dataIn),
    .empty        (_vrfWriteQueue_fifo_empty),
    .almost_empty (vrfWriteQueue_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (vrfWriteQueue_almostFull),
    .full         (_vrfWriteQueue_fifo_full),
    .error        (_vrfWriteQueue_fifo_error),
    .data_out     (_vrfWriteQueue_fifo_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(5)
  ) vrfPtrReplica_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(vrfPtrReplica_enq_ready & vrfPtrReplica_enq_valid)),
    .pop_req_n    (~(vrfPtrReplica_deq_ready & ~_vrfPtrReplica_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (vrfPtrReplica_enq_bits),
    .empty        (_vrfPtrReplica_fifo_empty),
    .almost_empty (vrfPtrReplica_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (vrfPtrReplica_almostFull),
    .full         (_vrfPtrReplica_fifo_full),
    .error        (_vrfPtrReplica_fifo_error),
    .data_out     (vrfPtrReplica_deq_bits)
  );
  assign enqueue_ready = enqueue_ready_0;
  assign vrfWriteRequest_valid = vrfWriteRequest_valid_0;
  assign vrfWriteRequest_bits_vd = vrfWriteRequest_bits_vd_0;
  assign vrfWriteRequest_bits_offset = vrfWriteRequest_bits_offset_0;
  assign vrfWriteRequest_bits_mask = vrfWriteRequest_bits_mask_0;
  assign vrfWriteRequest_bits_data = vrfWriteRequest_bits_data_0;
  assign vrfWriteRequest_bits_last = vrfWriteRequest_bits_last_0;
  assign vrfWriteRequest_bits_instructionIndex = vrfWriteRequest_bits_instructionIndex_0;
  assign crossWritePort_0_valid = crossWritePort_0_valid_0;
  assign crossWritePort_0_bits_data = crossWritePort_0_bits_data_0;
  assign crossWritePort_0_bits_mask = crossWritePort_0_bits_mask_0;
  assign crossWritePort_0_bits_instructionIndex = crossWritePort_0_bits_instructionIndex_0;
  assign crossWritePort_0_bits_counter = crossWritePort_0_bits_counter_0;
  assign crossWritePort_1_valid = crossWritePort_1_valid_0;
  assign crossWritePort_1_bits_data = crossWritePort_1_bits_data_0;
  assign crossWritePort_1_bits_mask = crossWritePort_1_bits_mask_0;
  assign crossWritePort_1_bits_instructionIndex = crossWritePort_1_bits_instructionIndex_0;
  assign crossWritePort_1_bits_counter = crossWritePort_1_bits_counter_0;
endmodule

