
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
module MaskExchangeUnit(
  input         clock,
                reset,
  output        enqueue_ready,
  input         enqueue_valid,
  input  [12:0] enqueue_bits_groupCounter,
  input  [31:0] enqueue_bits_data,
                enqueue_bits_pipeData,
  input  [3:0]  enqueue_bits_mask,
  input  [13:0] enqueue_bits_ffoIndex,
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
  input         dequeue_ready,
  output        dequeue_valid,
  output [12:0] dequeue_bits_groupCounter,
  output [31:0] dequeue_bits_data,
                dequeue_bits_pipeData,
  output [3:0]  dequeue_bits_mask,
  output [13:0] dequeue_bits_ffoIndex,
  output [31:0] dequeue_bits_crossWriteData_0,
                dequeue_bits_crossWriteData_1,
  output        dequeue_bits_sSendResponse,
                dequeue_bits_ffoSuccess,
                dequeue_bits_decodeResult_specialSlot,
  output [4:0]  dequeue_bits_decodeResult_topUop,
  output        dequeue_bits_decodeResult_popCount,
                dequeue_bits_decodeResult_ffo,
                dequeue_bits_decodeResult_average,
                dequeue_bits_decodeResult_reverse,
                dequeue_bits_decodeResult_dontNeedExecuteInLane,
                dequeue_bits_decodeResult_scheduler,
                dequeue_bits_decodeResult_sReadVD,
                dequeue_bits_decodeResult_vtype,
                dequeue_bits_decodeResult_sWrite,
                dequeue_bits_decodeResult_crossRead,
                dequeue_bits_decodeResult_crossWrite,
                dequeue_bits_decodeResult_maskUnit,
                dequeue_bits_decodeResult_special,
                dequeue_bits_decodeResult_saturate,
                dequeue_bits_decodeResult_vwmacc,
                dequeue_bits_decodeResult_readOnly,
                dequeue_bits_decodeResult_maskSource,
                dequeue_bits_decodeResult_maskDestination,
                dequeue_bits_decodeResult_maskLogic,
  output [3:0]  dequeue_bits_decodeResult_uop,
  output        dequeue_bits_decodeResult_iota,
                dequeue_bits_decodeResult_mv,
                dequeue_bits_decodeResult_extend,
                dequeue_bits_decodeResult_unOrderWrite,
                dequeue_bits_decodeResult_compress,
                dequeue_bits_decodeResult_gather16,
                dequeue_bits_decodeResult_gather,
                dequeue_bits_decodeResult_slid,
                dequeue_bits_decodeResult_targetRd,
                dequeue_bits_decodeResult_widenReduce,
                dequeue_bits_decodeResult_red,
                dequeue_bits_decodeResult_nr,
                dequeue_bits_decodeResult_itype,
                dequeue_bits_decodeResult_unsigned1,
                dequeue_bits_decodeResult_unsigned0,
                dequeue_bits_decodeResult_other,
                dequeue_bits_decodeResult_multiCycle,
                dequeue_bits_decodeResult_divider,
                dequeue_bits_decodeResult_multiplier,
                dequeue_bits_decodeResult_shift,
                dequeue_bits_decodeResult_adder,
                dequeue_bits_decodeResult_logic,
  output [2:0]  dequeue_bits_instructionIndex,
  output        dequeue_bits_loadStore,
  output [4:0]  dequeue_bits_vd,
  output        maskReq_valid,
  output [31:0] maskReq_bits_source1,
                maskReq_bits_source2,
  output [2:0]  maskReq_bits_index,
  output        maskReq_bits_ffo,
                maskRequestToLSU,
  input         tokenIO_maskRequestRelease
);

  wire        _maskReq_valid_output;
  wire        enqueue_valid_0 = enqueue_valid;
  wire [12:0] enqueue_bits_groupCounter_0 = enqueue_bits_groupCounter;
  wire [31:0] enqueue_bits_data_0 = enqueue_bits_data;
  wire [31:0] enqueue_bits_pipeData_0 = enqueue_bits_pipeData;
  wire [3:0]  enqueue_bits_mask_0 = enqueue_bits_mask;
  wire [13:0] enqueue_bits_ffoIndex_0 = enqueue_bits_ffoIndex;
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
  wire        dequeue_ready_0 = dequeue_ready;
  wire        tokenIO_maskRequestRelease_0 = tokenIO_maskRequestRelease;
  wire        enqueue_bits_ffoByOtherLanes = 1'h0;
  wire        dequeue_bits_ffoByOtherLanes = 1'h0;
  wire [12:0] dequeue_bits_groupCounter_0 = enqueue_bits_groupCounter_0;
  wire [31:0] dequeue_bits_data_0 = enqueue_bits_data_0;
  wire [31:0] dequeue_bits_pipeData_0 = enqueue_bits_pipeData_0;
  wire [3:0]  dequeue_bits_mask_0 = enqueue_bits_mask_0;
  wire [13:0] dequeue_bits_ffoIndex_0 = enqueue_bits_ffoIndex_0;
  wire [31:0] dequeue_bits_crossWriteData_0_0 = enqueue_bits_crossWriteData_0_0;
  wire [31:0] dequeue_bits_crossWriteData_1_0 = enqueue_bits_crossWriteData_1_0;
  wire        dequeue_bits_sSendResponse_0 = enqueue_bits_sSendResponse_0;
  wire        dequeue_bits_ffoSuccess_0 = enqueue_bits_ffoSuccess_0;
  wire        dequeue_bits_decodeResult_specialSlot_0 = enqueue_bits_decodeResult_specialSlot_0;
  wire [4:0]  dequeue_bits_decodeResult_topUop_0 = enqueue_bits_decodeResult_topUop_0;
  wire        dequeue_bits_decodeResult_popCount_0 = enqueue_bits_decodeResult_popCount_0;
  wire        dequeue_bits_decodeResult_ffo_0 = enqueue_bits_decodeResult_ffo_0;
  wire        dequeue_bits_decodeResult_average_0 = enqueue_bits_decodeResult_average_0;
  wire        dequeue_bits_decodeResult_reverse_0 = enqueue_bits_decodeResult_reverse_0;
  wire        dequeue_bits_decodeResult_dontNeedExecuteInLane_0 = enqueue_bits_decodeResult_dontNeedExecuteInLane_0;
  wire        dequeue_bits_decodeResult_scheduler_0 = enqueue_bits_decodeResult_scheduler_0;
  wire        dequeue_bits_decodeResult_sReadVD_0 = enqueue_bits_decodeResult_sReadVD_0;
  wire        dequeue_bits_decodeResult_vtype_0 = enqueue_bits_decodeResult_vtype_0;
  wire        dequeue_bits_decodeResult_sWrite_0 = enqueue_bits_decodeResult_sWrite_0;
  wire        dequeue_bits_decodeResult_crossRead_0 = enqueue_bits_decodeResult_crossRead_0;
  wire        dequeue_bits_decodeResult_crossWrite_0 = enqueue_bits_decodeResult_crossWrite_0;
  wire        dequeue_bits_decodeResult_maskUnit_0 = enqueue_bits_decodeResult_maskUnit_0;
  wire        dequeue_bits_decodeResult_special_0 = enqueue_bits_decodeResult_special_0;
  wire        dequeue_bits_decodeResult_saturate_0 = enqueue_bits_decodeResult_saturate_0;
  wire        dequeue_bits_decodeResult_vwmacc_0 = enqueue_bits_decodeResult_vwmacc_0;
  wire        dequeue_bits_decodeResult_readOnly_0 = enqueue_bits_decodeResult_readOnly_0;
  wire        dequeue_bits_decodeResult_maskSource_0 = enqueue_bits_decodeResult_maskSource_0;
  wire        dequeue_bits_decodeResult_maskDestination_0 = enqueue_bits_decodeResult_maskDestination_0;
  wire        dequeue_bits_decodeResult_maskLogic_0 = enqueue_bits_decodeResult_maskLogic_0;
  wire [3:0]  dequeue_bits_decodeResult_uop_0 = enqueue_bits_decodeResult_uop_0;
  wire        dequeue_bits_decodeResult_iota_0 = enqueue_bits_decodeResult_iota_0;
  wire        dequeue_bits_decodeResult_mv_0 = enqueue_bits_decodeResult_mv_0;
  wire        dequeue_bits_decodeResult_extend_0 = enqueue_bits_decodeResult_extend_0;
  wire        dequeue_bits_decodeResult_unOrderWrite_0 = enqueue_bits_decodeResult_unOrderWrite_0;
  wire        dequeue_bits_decodeResult_compress_0 = enqueue_bits_decodeResult_compress_0;
  wire        dequeue_bits_decodeResult_gather16_0 = enqueue_bits_decodeResult_gather16_0;
  wire        dequeue_bits_decodeResult_gather_0 = enqueue_bits_decodeResult_gather_0;
  wire        dequeue_bits_decodeResult_slid_0 = enqueue_bits_decodeResult_slid_0;
  wire        dequeue_bits_decodeResult_targetRd_0 = enqueue_bits_decodeResult_targetRd_0;
  wire        dequeue_bits_decodeResult_widenReduce_0 = enqueue_bits_decodeResult_widenReduce_0;
  wire        dequeue_bits_decodeResult_red_0 = enqueue_bits_decodeResult_red_0;
  wire        dequeue_bits_decodeResult_nr_0 = enqueue_bits_decodeResult_nr_0;
  wire        dequeue_bits_decodeResult_itype_0 = enqueue_bits_decodeResult_itype_0;
  wire        dequeue_bits_decodeResult_unsigned1_0 = enqueue_bits_decodeResult_unsigned1_0;
  wire        dequeue_bits_decodeResult_unsigned0_0 = enqueue_bits_decodeResult_unsigned0_0;
  wire        dequeue_bits_decodeResult_other_0 = enqueue_bits_decodeResult_other_0;
  wire        dequeue_bits_decodeResult_multiCycle_0 = enqueue_bits_decodeResult_multiCycle_0;
  wire        dequeue_bits_decodeResult_divider_0 = enqueue_bits_decodeResult_divider_0;
  wire        dequeue_bits_decodeResult_multiplier_0 = enqueue_bits_decodeResult_multiplier_0;
  wire        dequeue_bits_decodeResult_shift_0 = enqueue_bits_decodeResult_shift_0;
  wire        dequeue_bits_decodeResult_adder_0 = enqueue_bits_decodeResult_adder_0;
  wire        dequeue_bits_decodeResult_logic_0 = enqueue_bits_decodeResult_logic_0;
  wire [2:0]  dequeue_bits_instructionIndex_0 = enqueue_bits_instructionIndex_0;
  wire        dequeue_bits_loadStore_0 = enqueue_bits_loadStore_0;
  wire [4:0]  dequeue_bits_vd_0 = enqueue_bits_vd_0;
  wire        enqIsMaskRequest = ~enqueue_bits_sSendResponse_0;
  wire        enqSendToDeq = ~enqueue_bits_decodeResult_maskUnit_0 & enqueue_bits_sSendResponse_0;
  wire        enqFFoIndex = enqueue_bits_decodeResult_ffo_0 & enqueue_bits_decodeResult_targetRd_0;
  reg  [3:0]  maskRequestAllow_counter;
  wire [3:0]  maskRequestAllow_counterChange = _maskReq_valid_output ? 4'h1 : 4'hF;
  wire        maskRequestAllow = ~(maskRequestAllow_counter[3]);
  assign _maskReq_valid_output = enqIsMaskRequest & enqueue_valid_0 & maskRequestAllow;
  wire        maskRequestEnqReady = ~enqIsMaskRequest | maskRequestAllow;
  wire        dequeue_valid_0 = enqueue_valid_0 & enqSendToDeq;
  wire        enqueue_ready_0 = enqSendToDeq ? dequeue_ready_0 : maskRequestEnqReady;
  always @(posedge clock) begin
    if (reset)
      maskRequestAllow_counter <= 4'h0;
    else if (_maskReq_valid_output ^ tokenIO_maskRequestRelease_0)
      maskRequestAllow_counter <= maskRequestAllow_counter + maskRequestAllow_counterChange;
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
        maskRequestAllow_counter = _RANDOM[/*Zero width*/ 1'b0][3:0];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  assign enqueue_ready = enqueue_ready_0;
  assign dequeue_valid = dequeue_valid_0;
  assign dequeue_bits_groupCounter = dequeue_bits_groupCounter_0;
  assign dequeue_bits_data = dequeue_bits_data_0;
  assign dequeue_bits_pipeData = dequeue_bits_pipeData_0;
  assign dequeue_bits_mask = dequeue_bits_mask_0;
  assign dequeue_bits_ffoIndex = dequeue_bits_ffoIndex_0;
  assign dequeue_bits_crossWriteData_0 = dequeue_bits_crossWriteData_0_0;
  assign dequeue_bits_crossWriteData_1 = dequeue_bits_crossWriteData_1_0;
  assign dequeue_bits_sSendResponse = dequeue_bits_sSendResponse_0;
  assign dequeue_bits_ffoSuccess = dequeue_bits_ffoSuccess_0;
  assign dequeue_bits_decodeResult_specialSlot = dequeue_bits_decodeResult_specialSlot_0;
  assign dequeue_bits_decodeResult_topUop = dequeue_bits_decodeResult_topUop_0;
  assign dequeue_bits_decodeResult_popCount = dequeue_bits_decodeResult_popCount_0;
  assign dequeue_bits_decodeResult_ffo = dequeue_bits_decodeResult_ffo_0;
  assign dequeue_bits_decodeResult_average = dequeue_bits_decodeResult_average_0;
  assign dequeue_bits_decodeResult_reverse = dequeue_bits_decodeResult_reverse_0;
  assign dequeue_bits_decodeResult_dontNeedExecuteInLane = dequeue_bits_decodeResult_dontNeedExecuteInLane_0;
  assign dequeue_bits_decodeResult_scheduler = dequeue_bits_decodeResult_scheduler_0;
  assign dequeue_bits_decodeResult_sReadVD = dequeue_bits_decodeResult_sReadVD_0;
  assign dequeue_bits_decodeResult_vtype = dequeue_bits_decodeResult_vtype_0;
  assign dequeue_bits_decodeResult_sWrite = dequeue_bits_decodeResult_sWrite_0;
  assign dequeue_bits_decodeResult_crossRead = dequeue_bits_decodeResult_crossRead_0;
  assign dequeue_bits_decodeResult_crossWrite = dequeue_bits_decodeResult_crossWrite_0;
  assign dequeue_bits_decodeResult_maskUnit = dequeue_bits_decodeResult_maskUnit_0;
  assign dequeue_bits_decodeResult_special = dequeue_bits_decodeResult_special_0;
  assign dequeue_bits_decodeResult_saturate = dequeue_bits_decodeResult_saturate_0;
  assign dequeue_bits_decodeResult_vwmacc = dequeue_bits_decodeResult_vwmacc_0;
  assign dequeue_bits_decodeResult_readOnly = dequeue_bits_decodeResult_readOnly_0;
  assign dequeue_bits_decodeResult_maskSource = dequeue_bits_decodeResult_maskSource_0;
  assign dequeue_bits_decodeResult_maskDestination = dequeue_bits_decodeResult_maskDestination_0;
  assign dequeue_bits_decodeResult_maskLogic = dequeue_bits_decodeResult_maskLogic_0;
  assign dequeue_bits_decodeResult_uop = dequeue_bits_decodeResult_uop_0;
  assign dequeue_bits_decodeResult_iota = dequeue_bits_decodeResult_iota_0;
  assign dequeue_bits_decodeResult_mv = dequeue_bits_decodeResult_mv_0;
  assign dequeue_bits_decodeResult_extend = dequeue_bits_decodeResult_extend_0;
  assign dequeue_bits_decodeResult_unOrderWrite = dequeue_bits_decodeResult_unOrderWrite_0;
  assign dequeue_bits_decodeResult_compress = dequeue_bits_decodeResult_compress_0;
  assign dequeue_bits_decodeResult_gather16 = dequeue_bits_decodeResult_gather16_0;
  assign dequeue_bits_decodeResult_gather = dequeue_bits_decodeResult_gather_0;
  assign dequeue_bits_decodeResult_slid = dequeue_bits_decodeResult_slid_0;
  assign dequeue_bits_decodeResult_targetRd = dequeue_bits_decodeResult_targetRd_0;
  assign dequeue_bits_decodeResult_widenReduce = dequeue_bits_decodeResult_widenReduce_0;
  assign dequeue_bits_decodeResult_red = dequeue_bits_decodeResult_red_0;
  assign dequeue_bits_decodeResult_nr = dequeue_bits_decodeResult_nr_0;
  assign dequeue_bits_decodeResult_itype = dequeue_bits_decodeResult_itype_0;
  assign dequeue_bits_decodeResult_unsigned1 = dequeue_bits_decodeResult_unsigned1_0;
  assign dequeue_bits_decodeResult_unsigned0 = dequeue_bits_decodeResult_unsigned0_0;
  assign dequeue_bits_decodeResult_other = dequeue_bits_decodeResult_other_0;
  assign dequeue_bits_decodeResult_multiCycle = dequeue_bits_decodeResult_multiCycle_0;
  assign dequeue_bits_decodeResult_divider = dequeue_bits_decodeResult_divider_0;
  assign dequeue_bits_decodeResult_multiplier = dequeue_bits_decodeResult_multiplier_0;
  assign dequeue_bits_decodeResult_shift = dequeue_bits_decodeResult_shift_0;
  assign dequeue_bits_decodeResult_adder = dequeue_bits_decodeResult_adder_0;
  assign dequeue_bits_decodeResult_logic = dequeue_bits_decodeResult_logic_0;
  assign dequeue_bits_instructionIndex = dequeue_bits_instructionIndex_0;
  assign dequeue_bits_loadStore = dequeue_bits_loadStore_0;
  assign dequeue_bits_vd = dequeue_bits_vd_0;
  assign maskReq_valid = _maskReq_valid_output;
  assign maskReq_bits_source1 = enqueue_bits_pipeData_0;
  assign maskReq_bits_source2 = enqFFoIndex ? {18'h0, enqueue_bits_ffoIndex_0} : enqueue_bits_data_0;
  assign maskReq_bits_index = enqueue_bits_instructionIndex_0;
  assign maskReq_bits_ffo = enqueue_bits_ffoSuccess_0;
  assign maskRequestToLSU = enqueue_bits_loadStore_0;
endmodule

