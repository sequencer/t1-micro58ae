module LaneStage3_1(
  input         clock,
                reset,
  output        enqueue_ready,
  input         enqueue_valid,
  input  [9:0]  enqueue_bits_groupCounter,
  input  [31:0] enqueue_bits_data,
  input  [3:0]  enqueue_bits_mask,
  input  [10:0] enqueue_bits_ffoIndex,
  input         enqueue_bits_decodeResult_specialSlot,
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
  output [5:0]  vrfWriteRequest_bits_offset,
  output [3:0]  vrfWriteRequest_bits_mask,
  output [31:0] vrfWriteRequest_bits_data,
  output        vrfWriteRequest_bits_last,
  output [2:0]  vrfWriteRequest_bits_instructionIndex
);

  wire        _vrfPtrReplica_fifo_empty;
  wire        _vrfPtrReplica_fifo_full;
  wire        _vrfPtrReplica_fifo_error;
  wire        _vrfWriteQueue_fifo_empty;
  wire        _vrfWriteQueue_fifo_full;
  wire        _vrfWriteQueue_fifo_error;
  wire [50:0] _vrfWriteQueue_fifo_data_out;
  wire        vrfPtrReplica_almostFull;
  wire        vrfPtrReplica_almostEmpty;
  wire        vrfWriteQueue_almostFull;
  wire        vrfWriteQueue_almostEmpty;
  wire [5:0]  vrfWriteQueue_enq_bits_offset;
  wire        enqueue_valid_0 = enqueue_valid;
  wire [9:0]  enqueue_bits_groupCounter_0 = enqueue_bits_groupCounter;
  wire [31:0] enqueue_bits_data_0 = enqueue_bits_data;
  wire [3:0]  enqueue_bits_mask_0 = enqueue_bits_mask;
  wire [10:0] enqueue_bits_ffoIndex_0 = enqueue_bits_ffoIndex;
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
  wire        enqueue_bits_sSendResponse = 1'h0;
  wire        enqueue_bits_ffoSuccess = 1'h0;
  wire        enqueue_bits_ffoByOtherLanes = 1'h0;
  wire        vrfWriteQueue_enq_bits_last = 1'h0;
  wire [31:0] enqueue_bits_pipeData = 32'h0;
  wire [31:0] enqueue_bits_crossWriteData_0 = 32'h0;
  wire [31:0] enqueue_bits_crossWriteData_1 = 32'h0;
  wire        vrfWriteQueue_enq_ready;
  wire        vrfWriteQueue_enq_valid = enqueue_valid_0;
  wire [31:0] vrfWriteQueue_enq_bits_data = enqueue_bits_data_0;
  wire [3:0]  vrfWriteQueue_enq_bits_mask = enqueue_bits_mask_0;
  wire [2:0]  vrfWriteQueue_enq_bits_instructionIndex = enqueue_bits_instructionIndex_0;
  wire        vrfWriteQueue_deq_ready = vrfWriteRequest_ready_0;
  wire        vrfPtrReplica_deq_valid;
  wire [4:0]  vrfWriteQueue_deq_bits_vd;
  wire [5:0]  vrfPtrReplica_deq_bits;
  wire [3:0]  vrfWriteQueue_deq_bits_mask;
  wire [31:0] vrfWriteQueue_deq_bits_data;
  wire        vrfWriteQueue_deq_bits_last;
  wire [2:0]  vrfWriteQueue_deq_bits_instructionIndex;
  wire        enqueue_ready_0 = vrfWriteQueue_enq_ready;
  wire        vrfPtrReplica_enq_valid = vrfWriteQueue_enq_valid;
  wire [5:0]  vrfPtrReplica_enq_bits = vrfWriteQueue_enq_bits_offset;
  wire        vrfPtrReplica_deq_ready = vrfWriteQueue_deq_ready;
  wire        vrfWriteQueue_deq_valid;
  assign vrfWriteQueue_deq_valid = ~_vrfWriteQueue_fifo_empty;
  wire [4:0]  vrfWriteQueue_dataOut_vd;
  wire [4:0]  vrfWriteRequest_bits_vd_0 = vrfWriteQueue_deq_bits_vd;
  wire [5:0]  vrfWriteQueue_dataOut_offset;
  wire [3:0]  vrfWriteQueue_dataOut_mask;
  wire [3:0]  vrfWriteRequest_bits_mask_0 = vrfWriteQueue_deq_bits_mask;
  wire [31:0] vrfWriteQueue_dataOut_data;
  wire [31:0] vrfWriteRequest_bits_data_0 = vrfWriteQueue_deq_bits_data;
  wire        vrfWriteQueue_dataOut_last;
  wire        vrfWriteRequest_bits_last_0 = vrfWriteQueue_deq_bits_last;
  wire [2:0]  vrfWriteQueue_dataOut_instructionIndex;
  wire [2:0]  vrfWriteRequest_bits_instructionIndex_0 = vrfWriteQueue_deq_bits_instructionIndex;
  wire [32:0] vrfWriteQueue_dataIn_lo_hi = {vrfWriteQueue_enq_bits_data, 1'h0};
  wire [35:0] vrfWriteQueue_dataIn_lo = {vrfWriteQueue_dataIn_lo_hi, vrfWriteQueue_enq_bits_instructionIndex};
  wire [4:0]  vrfWriteQueue_enq_bits_vd;
  wire [10:0] vrfWriteQueue_dataIn_hi_hi = {vrfWriteQueue_enq_bits_vd, vrfWriteQueue_enq_bits_offset};
  wire [14:0] vrfWriteQueue_dataIn_hi = {vrfWriteQueue_dataIn_hi_hi, vrfWriteQueue_enq_bits_mask};
  wire [50:0] vrfWriteQueue_dataIn = {vrfWriteQueue_dataIn_hi, vrfWriteQueue_dataIn_lo};
  assign vrfWriteQueue_dataOut_instructionIndex = _vrfWriteQueue_fifo_data_out[2:0];
  assign vrfWriteQueue_dataOut_last = _vrfWriteQueue_fifo_data_out[3];
  assign vrfWriteQueue_dataOut_data = _vrfWriteQueue_fifo_data_out[35:4];
  assign vrfWriteQueue_dataOut_mask = _vrfWriteQueue_fifo_data_out[39:36];
  assign vrfWriteQueue_dataOut_offset = _vrfWriteQueue_fifo_data_out[45:40];
  assign vrfWriteQueue_dataOut_vd = _vrfWriteQueue_fifo_data_out[50:46];
  assign vrfWriteQueue_deq_bits_vd = vrfWriteQueue_dataOut_vd;
  wire [5:0]  vrfWriteQueue_deq_bits_offset = vrfWriteQueue_dataOut_offset;
  assign vrfWriteQueue_deq_bits_mask = vrfWriteQueue_dataOut_mask;
  assign vrfWriteQueue_deq_bits_data = vrfWriteQueue_dataOut_data;
  assign vrfWriteQueue_deq_bits_last = vrfWriteQueue_dataOut_last;
  assign vrfWriteQueue_deq_bits_instructionIndex = vrfWriteQueue_dataOut_instructionIndex;
  assign vrfWriteQueue_enq_ready = ~_vrfWriteQueue_fifo_full;
  assign vrfPtrReplica_deq_valid = ~_vrfPtrReplica_fifo_empty;
  wire        vrfWriteRequest_valid_0 = vrfPtrReplica_deq_valid;
  wire [5:0]  vrfWriteRequest_bits_offset_0 = vrfPtrReplica_deq_bits;
  wire        vrfPtrReplica_enq_ready = ~_vrfPtrReplica_fifo_full;
  assign vrfWriteQueue_enq_bits_vd = enqueue_bits_vd_0 + {1'h0, enqueue_bits_groupCounter_0[9:6]};
  assign vrfWriteQueue_enq_bits_offset = enqueue_bits_groupCounter_0[5:0];
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
    .width(51)
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
    .width(6)
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
endmodule

