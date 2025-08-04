
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
module LaneExecutionBridge_1(
  input         clock,
                reset,
  output        enqueue_ready,
  input         enqueue_valid,
  input  [31:0] enqueue_bits_src_0,
                enqueue_bits_src_1,
                enqueue_bits_src_2,
  input         enqueue_bits_bordersForMaskLogic,
  input  [3:0]  enqueue_bits_mask,
                enqueue_bits_maskForFilter,
  input  [12:0] enqueue_bits_groupCounter,
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
  input  [2:0]  enqueue_bits_vSew1H,
  input  [16:0] enqueue_bits_csr_vl,
                enqueue_bits_csr_vStart,
  input  [2:0]  enqueue_bits_csr_vlmul,
  input  [1:0]  enqueue_bits_csr_vSew,
                enqueue_bits_csr_vxrm,
  input         enqueue_bits_csr_vta,
                enqueue_bits_csr_vma,
                enqueue_bits_maskType,
  input  [1:0]  enqueue_bits_laneIndex,
  input  [2:0]  enqueue_bits_instructionIndex,
  input         dequeue_ready,
  output        dequeue_valid,
  output [31:0] dequeue_bits_data,
  output [13:0] dequeue_bits_ffoIndex,
  input         vfuRequest_ready,
  output        vfuRequest_valid,
  output [32:0] vfuRequest_bits_src_0,
                vfuRequest_bits_src_1,
                vfuRequest_bits_src_2,
                vfuRequest_bits_src_3,
  output [3:0]  vfuRequest_bits_opcode,
                vfuRequest_bits_mask,
                vfuRequest_bits_executeMask,
  output        vfuRequest_bits_sign0,
                vfuRequest_bits_sign,
                vfuRequest_bits_reverse,
                vfuRequest_bits_average,
                vfuRequest_bits_saturate,
  output [1:0]  vfuRequest_bits_vxrm,
                vfuRequest_bits_vSew,
  output [19:0] vfuRequest_bits_shifterSize,
  output        vfuRequest_bits_rem,
  output [12:0] vfuRequest_bits_groupIndex,
  output [1:0]  vfuRequest_bits_laneIndex,
  output        vfuRequest_bits_maskType,
                vfuRequest_bits_narrow,
  input         dataResponse_valid,
  input  [31:0] dataResponse_bits_data,
  output        executeDecode_specialSlot,
  output [4:0]  executeDecode_topUop,
  output        executeDecode_popCount,
                executeDecode_ffo,
                executeDecode_average,
                executeDecode_reverse,
                executeDecode_dontNeedExecuteInLane,
                executeDecode_scheduler,
                executeDecode_sReadVD,
                executeDecode_vtype,
                executeDecode_sWrite,
                executeDecode_crossRead,
                executeDecode_crossWrite,
                executeDecode_maskUnit,
                executeDecode_special,
                executeDecode_saturate,
                executeDecode_vwmacc,
                executeDecode_readOnly,
                executeDecode_maskSource,
                executeDecode_maskDestination,
                executeDecode_maskLogic,
  output [3:0]  executeDecode_uop,
  output        executeDecode_iota,
                executeDecode_mv,
                executeDecode_extend,
                executeDecode_unOrderWrite,
                executeDecode_compress,
                executeDecode_gather16,
                executeDecode_gather,
                executeDecode_slid,
                executeDecode_targetRd,
                executeDecode_widenReduce,
                executeDecode_red,
                executeDecode_nr,
                executeDecode_itype,
                executeDecode_unsigned1,
                executeDecode_unsigned0,
                executeDecode_other,
                executeDecode_multiCycle,
                executeDecode_divider,
                executeDecode_multiplier,
                executeDecode_shift,
                executeDecode_adder,
                executeDecode_logic,
                responseDecode_specialSlot,
  output [4:0]  responseDecode_topUop,
  output        responseDecode_popCount,
                responseDecode_ffo,
                responseDecode_average,
                responseDecode_reverse,
                responseDecode_dontNeedExecuteInLane,
                responseDecode_scheduler,
                responseDecode_sReadVD,
                responseDecode_vtype,
                responseDecode_sWrite,
                responseDecode_crossRead,
                responseDecode_crossWrite,
                responseDecode_maskUnit,
                responseDecode_special,
                responseDecode_saturate,
                responseDecode_vwmacc,
                responseDecode_readOnly,
                responseDecode_maskSource,
                responseDecode_maskDestination,
                responseDecode_maskLogic,
  output [3:0]  responseDecode_uop,
  output        responseDecode_iota,
                responseDecode_mv,
                responseDecode_extend,
                responseDecode_unOrderWrite,
                responseDecode_compress,
                responseDecode_gather16,
                responseDecode_gather,
                responseDecode_slid,
                responseDecode_targetRd,
                responseDecode_widenReduce,
                responseDecode_red,
                responseDecode_nr,
                responseDecode_itype,
                responseDecode_unsigned1,
                responseDecode_unsigned0,
                responseDecode_other,
                responseDecode_multiCycle,
                responseDecode_divider,
                responseDecode_multiplier,
                responseDecode_shift,
                responseDecode_adder,
                responseDecode_logic,
  output [2:0]  responseIndex
);

  wire         _queue_fifo_empty;
  wire         _queue_fifo_full;
  wire         _queue_fifo_error;
  wire [45:0]  _queue_fifo_data_out;
  wire         _recordQueue_fifo_empty;
  wire         _recordQueue_fifo_full;
  wire         _recordQueue_fifo_error;
  wire [107:0] _recordQueue_fifo_data_out;
  wire         queue_almostFull;
  wire         queue_almostEmpty;
  wire         recordQueue_almostFull;
  wire         recordQueue_almostEmpty;
  reg          executionRecord_decodeResult_unsigned0;
  reg          executionRecord_decodeResult_unsigned1;
  wire         enqueue_valid_0 = enqueue_valid;
  wire [31:0]  enqueue_bits_src_0_0 = enqueue_bits_src_0;
  wire [31:0]  enqueue_bits_src_1_0 = enqueue_bits_src_1;
  wire [31:0]  enqueue_bits_src_2_0 = enqueue_bits_src_2;
  wire         enqueue_bits_bordersForMaskLogic_0 = enqueue_bits_bordersForMaskLogic;
  wire [3:0]   enqueue_bits_mask_0 = enqueue_bits_mask;
  wire [3:0]   enqueue_bits_maskForFilter_0 = enqueue_bits_maskForFilter;
  wire [12:0]  enqueue_bits_groupCounter_0 = enqueue_bits_groupCounter;
  wire         enqueue_bits_decodeResult_specialSlot_0 = enqueue_bits_decodeResult_specialSlot;
  wire [4:0]   enqueue_bits_decodeResult_topUop_0 = enqueue_bits_decodeResult_topUop;
  wire         enqueue_bits_decodeResult_popCount_0 = enqueue_bits_decodeResult_popCount;
  wire         enqueue_bits_decodeResult_ffo_0 = enqueue_bits_decodeResult_ffo;
  wire         enqueue_bits_decodeResult_average_0 = enqueue_bits_decodeResult_average;
  wire         enqueue_bits_decodeResult_reverse_0 = enqueue_bits_decodeResult_reverse;
  wire         enqueue_bits_decodeResult_dontNeedExecuteInLane_0 = enqueue_bits_decodeResult_dontNeedExecuteInLane;
  wire         enqueue_bits_decodeResult_scheduler_0 = enqueue_bits_decodeResult_scheduler;
  wire         enqueue_bits_decodeResult_sReadVD_0 = enqueue_bits_decodeResult_sReadVD;
  wire         enqueue_bits_decodeResult_vtype_0 = enqueue_bits_decodeResult_vtype;
  wire         enqueue_bits_decodeResult_sWrite_0 = enqueue_bits_decodeResult_sWrite;
  wire         enqueue_bits_decodeResult_crossRead_0 = enqueue_bits_decodeResult_crossRead;
  wire         enqueue_bits_decodeResult_crossWrite_0 = enqueue_bits_decodeResult_crossWrite;
  wire         enqueue_bits_decodeResult_maskUnit_0 = enqueue_bits_decodeResult_maskUnit;
  wire         enqueue_bits_decodeResult_special_0 = enqueue_bits_decodeResult_special;
  wire         enqueue_bits_decodeResult_saturate_0 = enqueue_bits_decodeResult_saturate;
  wire         enqueue_bits_decodeResult_vwmacc_0 = enqueue_bits_decodeResult_vwmacc;
  wire         enqueue_bits_decodeResult_readOnly_0 = enqueue_bits_decodeResult_readOnly;
  wire         enqueue_bits_decodeResult_maskSource_0 = enqueue_bits_decodeResult_maskSource;
  wire         enqueue_bits_decodeResult_maskDestination_0 = enqueue_bits_decodeResult_maskDestination;
  wire         enqueue_bits_decodeResult_maskLogic_0 = enqueue_bits_decodeResult_maskLogic;
  wire [3:0]   enqueue_bits_decodeResult_uop_0 = enqueue_bits_decodeResult_uop;
  wire         enqueue_bits_decodeResult_iota_0 = enqueue_bits_decodeResult_iota;
  wire         enqueue_bits_decodeResult_mv_0 = enqueue_bits_decodeResult_mv;
  wire         enqueue_bits_decodeResult_extend_0 = enqueue_bits_decodeResult_extend;
  wire         enqueue_bits_decodeResult_unOrderWrite_0 = enqueue_bits_decodeResult_unOrderWrite;
  wire         enqueue_bits_decodeResult_compress_0 = enqueue_bits_decodeResult_compress;
  wire         enqueue_bits_decodeResult_gather16_0 = enqueue_bits_decodeResult_gather16;
  wire         enqueue_bits_decodeResult_gather_0 = enqueue_bits_decodeResult_gather;
  wire         enqueue_bits_decodeResult_slid_0 = enqueue_bits_decodeResult_slid;
  wire         enqueue_bits_decodeResult_targetRd_0 = enqueue_bits_decodeResult_targetRd;
  wire         enqueue_bits_decodeResult_widenReduce_0 = enqueue_bits_decodeResult_widenReduce;
  wire         enqueue_bits_decodeResult_red_0 = enqueue_bits_decodeResult_red;
  wire         enqueue_bits_decodeResult_nr_0 = enqueue_bits_decodeResult_nr;
  wire         enqueue_bits_decodeResult_itype_0 = enqueue_bits_decodeResult_itype;
  wire         enqueue_bits_decodeResult_unsigned1_0 = enqueue_bits_decodeResult_unsigned1;
  wire         enqueue_bits_decodeResult_unsigned0_0 = enqueue_bits_decodeResult_unsigned0;
  wire         enqueue_bits_decodeResult_other_0 = enqueue_bits_decodeResult_other;
  wire         enqueue_bits_decodeResult_multiCycle_0 = enqueue_bits_decodeResult_multiCycle;
  wire         enqueue_bits_decodeResult_divider_0 = enqueue_bits_decodeResult_divider;
  wire         enqueue_bits_decodeResult_multiplier_0 = enqueue_bits_decodeResult_multiplier;
  wire         enqueue_bits_decodeResult_shift_0 = enqueue_bits_decodeResult_shift;
  wire         enqueue_bits_decodeResult_adder_0 = enqueue_bits_decodeResult_adder;
  wire         enqueue_bits_decodeResult_logic_0 = enqueue_bits_decodeResult_logic;
  wire [2:0]   enqueue_bits_vSew1H_0 = enqueue_bits_vSew1H;
  wire [16:0]  enqueue_bits_csr_vl_0 = enqueue_bits_csr_vl;
  wire [16:0]  enqueue_bits_csr_vStart_0 = enqueue_bits_csr_vStart;
  wire [2:0]   enqueue_bits_csr_vlmul_0 = enqueue_bits_csr_vlmul;
  wire [1:0]   enqueue_bits_csr_vSew_0 = enqueue_bits_csr_vSew;
  wire [1:0]   enqueue_bits_csr_vxrm_0 = enqueue_bits_csr_vxrm;
  wire         enqueue_bits_csr_vta_0 = enqueue_bits_csr_vta;
  wire         enqueue_bits_csr_vma_0 = enqueue_bits_csr_vma;
  wire         enqueue_bits_maskType_0 = enqueue_bits_maskType;
  wire [1:0]   enqueue_bits_laneIndex_0 = enqueue_bits_laneIndex;
  wire [2:0]   enqueue_bits_instructionIndex_0 = enqueue_bits_instructionIndex;
  wire         dequeue_ready_0 = dequeue_ready;
  wire         vfuRequest_ready_0 = vfuRequest_ready;
  wire         reduceReady = 1'h1;
  wire [1:0]   vfuRequest_bits_tag = 2'h1;
  wire [16:0]  vfuRequest_bits_popInit = 17'h0;
  wire         vfuRequest_bits_complete = 1'h0;
  wire         recordQueue_enq_bits_executeIndex = 1'h0;
  wire         reduceLastResponse = 1'h0;
  wire [1:0]   vfuRequest_bits_executeIndex = 2'h0;
  wire         queue_deq_ready = dequeue_ready_0;
  wire         queue_deq_valid;
  wire [31:0]  queue_deq_bits_data;
  wire [13:0]  queue_deq_bits_ffoIndex;
  wire         vfuRequest_bits_sign0_0;
  assign vfuRequest_bits_sign0_0 = ~executionRecord_decodeResult_unsigned0;
  wire         vfuRequest_bits_sign_0;
  assign vfuRequest_bits_sign_0 = ~executionRecord_decodeResult_unsigned1;
  wire         narrowInRecord;
  reg          executionRecord_crossReadVS2;
  reg          executionRecord_bordersForMaskLogic;
  wire         recordQueue_enq_bits_bordersForMaskLogic = executionRecord_bordersForMaskLogic;
  reg  [3:0]   executionRecord_maskForMaskInput;
  reg  [3:0]   executionRecord_maskForFilter;
  wire [3:0]   vfuRequest_bits_executeMask_0 = executionRecord_maskForFilter;
  wire [3:0]   recordQueue_enq_bits_maskForFilter = executionRecord_maskForFilter;
  reg  [31:0]  executionRecord_source_0;
  reg  [31:0]  executionRecord_source_1;
  wire [31:0]  recordQueue_enq_bits_source2 = executionRecord_source_1;
  reg  [31:0]  executionRecord_source_2;
  reg  [12:0]  executionRecord_groupCounter;
  wire [12:0]  vfuRequest_bits_groupIndex_0 = executionRecord_groupCounter;
  wire [12:0]  recordQueue_enq_bits_groupCounter = executionRecord_groupCounter;
  reg  [2:0]   executionRecord_vSew1H;
  wire [2:0]   recordQueue_enq_bits_vSew1H = executionRecord_vSew1H;
  reg  [16:0]  executionRecord_csr_vl;
  reg  [16:0]  executionRecord_csr_vStart;
  reg  [2:0]   executionRecord_csr_vlmul;
  reg  [1:0]   executionRecord_csr_vSew;
  reg  [1:0]   executionRecord_csr_vxrm;
  wire [1:0]   vfuRequest_bits_vxrm_0 = executionRecord_csr_vxrm;
  reg          executionRecord_csr_vta;
  reg          executionRecord_csr_vma;
  reg          executionRecord_maskType;
  wire         vfuRequest_bits_maskType_0 = executionRecord_maskType;
  reg  [1:0]   executionRecord_laneIndex;
  wire [1:0]   vfuRequest_bits_laneIndex_0 = executionRecord_laneIndex;
  reg          executionRecord_decodeResult_specialSlot;
  wire         recordQueue_enq_bits_decodeResult_specialSlot = executionRecord_decodeResult_specialSlot;
  reg  [4:0]   executionRecord_decodeResult_topUop;
  wire [4:0]   recordQueue_enq_bits_decodeResult_topUop = executionRecord_decodeResult_topUop;
  reg          executionRecord_decodeResult_popCount;
  wire         recordQueue_enq_bits_decodeResult_popCount = executionRecord_decodeResult_popCount;
  reg          executionRecord_decodeResult_ffo;
  wire         recordQueue_enq_bits_decodeResult_ffo = executionRecord_decodeResult_ffo;
  reg          executionRecord_decodeResult_average;
  wire         vfuRequest_bits_average_0 = executionRecord_decodeResult_average;
  wire         recordQueue_enq_bits_decodeResult_average = executionRecord_decodeResult_average;
  reg          executionRecord_decodeResult_reverse;
  wire         vfuRequest_bits_reverse_0 = executionRecord_decodeResult_reverse;
  wire         recordQueue_enq_bits_decodeResult_reverse = executionRecord_decodeResult_reverse;
  reg          executionRecord_decodeResult_dontNeedExecuteInLane;
  wire         recordQueue_enq_bits_decodeResult_dontNeedExecuteInLane = executionRecord_decodeResult_dontNeedExecuteInLane;
  reg          executionRecord_decodeResult_scheduler;
  wire         recordQueue_enq_bits_decodeResult_scheduler = executionRecord_decodeResult_scheduler;
  reg          executionRecord_decodeResult_sReadVD;
  wire         recordQueue_enq_bits_decodeResult_sReadVD = executionRecord_decodeResult_sReadVD;
  reg          executionRecord_decodeResult_vtype;
  wire         recordQueue_enq_bits_decodeResult_vtype = executionRecord_decodeResult_vtype;
  reg          executionRecord_decodeResult_sWrite;
  wire         recordQueue_enq_bits_decodeResult_sWrite = executionRecord_decodeResult_sWrite;
  reg          executionRecord_decodeResult_crossRead;
  wire         recordQueue_enq_bits_decodeResult_crossRead = executionRecord_decodeResult_crossRead;
  reg          executionRecord_decodeResult_crossWrite;
  wire         recordQueue_enq_bits_decodeResult_crossWrite = executionRecord_decodeResult_crossWrite;
  reg          executionRecord_decodeResult_maskUnit;
  wire         recordQueue_enq_bits_decodeResult_maskUnit = executionRecord_decodeResult_maskUnit;
  reg          executionRecord_decodeResult_special;
  wire         recordQueue_enq_bits_decodeResult_special = executionRecord_decodeResult_special;
  reg          executionRecord_decodeResult_saturate;
  wire         vfuRequest_bits_saturate_0 = executionRecord_decodeResult_saturate;
  wire         recordQueue_enq_bits_decodeResult_saturate = executionRecord_decodeResult_saturate;
  reg          executionRecord_decodeResult_vwmacc;
  wire         recordQueue_enq_bits_decodeResult_vwmacc = executionRecord_decodeResult_vwmacc;
  reg          executionRecord_decodeResult_readOnly;
  wire         recordQueue_enq_bits_decodeResult_readOnly = executionRecord_decodeResult_readOnly;
  reg          executionRecord_decodeResult_maskSource;
  wire         recordQueue_enq_bits_decodeResult_maskSource = executionRecord_decodeResult_maskSource;
  reg          executionRecord_decodeResult_maskDestination;
  wire         recordQueue_enq_bits_decodeResult_maskDestination = executionRecord_decodeResult_maskDestination;
  reg          executionRecord_decodeResult_maskLogic;
  wire         recordQueue_enq_bits_decodeResult_maskLogic = executionRecord_decodeResult_maskLogic;
  reg  [3:0]   executionRecord_decodeResult_uop;
  wire [3:0]   vfuRequest_bits_opcode_0 = executionRecord_decodeResult_uop;
  wire [3:0]   recordQueue_enq_bits_decodeResult_uop = executionRecord_decodeResult_uop;
  reg          executionRecord_decodeResult_iota;
  wire         recordQueue_enq_bits_decodeResult_iota = executionRecord_decodeResult_iota;
  reg          executionRecord_decodeResult_mv;
  wire         recordQueue_enq_bits_decodeResult_mv = executionRecord_decodeResult_mv;
  reg          executionRecord_decodeResult_extend;
  wire         recordQueue_enq_bits_decodeResult_extend = executionRecord_decodeResult_extend;
  reg          executionRecord_decodeResult_unOrderWrite;
  wire         recordQueue_enq_bits_decodeResult_unOrderWrite = executionRecord_decodeResult_unOrderWrite;
  reg          executionRecord_decodeResult_compress;
  wire         recordQueue_enq_bits_decodeResult_compress = executionRecord_decodeResult_compress;
  reg          executionRecord_decodeResult_gather16;
  wire         recordQueue_enq_bits_decodeResult_gather16 = executionRecord_decodeResult_gather16;
  reg          executionRecord_decodeResult_gather;
  wire         recordQueue_enq_bits_decodeResult_gather = executionRecord_decodeResult_gather;
  reg          executionRecord_decodeResult_slid;
  wire         recordQueue_enq_bits_decodeResult_slid = executionRecord_decodeResult_slid;
  reg          executionRecord_decodeResult_targetRd;
  wire         recordQueue_enq_bits_decodeResult_targetRd = executionRecord_decodeResult_targetRd;
  reg          executionRecord_decodeResult_widenReduce;
  wire         recordQueue_enq_bits_decodeResult_widenReduce = executionRecord_decodeResult_widenReduce;
  reg          executionRecord_decodeResult_red;
  wire         recordQueue_enq_bits_decodeResult_red = executionRecord_decodeResult_red;
  reg          executionRecord_decodeResult_nr;
  wire         recordQueue_enq_bits_decodeResult_nr = executionRecord_decodeResult_nr;
  reg          executionRecord_decodeResult_itype;
  wire         recordQueue_enq_bits_decodeResult_itype = executionRecord_decodeResult_itype;
  wire         recordQueue_enq_bits_decodeResult_unsigned1 = executionRecord_decodeResult_unsigned1;
  wire         recordQueue_enq_bits_decodeResult_unsigned0 = executionRecord_decodeResult_unsigned0;
  reg          executionRecord_decodeResult_other;
  wire         recordQueue_enq_bits_decodeResult_other = executionRecord_decodeResult_other;
  reg          executionRecord_decodeResult_multiCycle;
  wire         recordQueue_enq_bits_decodeResult_multiCycle = executionRecord_decodeResult_multiCycle;
  reg          executionRecord_decodeResult_divider;
  wire         recordQueue_enq_bits_decodeResult_divider = executionRecord_decodeResult_divider;
  reg          executionRecord_decodeResult_multiplier;
  wire         recordQueue_enq_bits_decodeResult_multiplier = executionRecord_decodeResult_multiplier;
  reg          executionRecord_decodeResult_shift;
  wire         recordQueue_enq_bits_decodeResult_shift = executionRecord_decodeResult_shift;
  reg          executionRecord_decodeResult_adder;
  wire         recordQueue_enq_bits_decodeResult_adder = executionRecord_decodeResult_adder;
  reg          executionRecord_decodeResult_logic;
  wire         recordQueue_enq_bits_decodeResult_logic = executionRecord_decodeResult_logic;
  reg  [2:0]   executionRecord_instructionIndex;
  wire [2:0]   recordQueue_enq_bits_instructionIndex = executionRecord_instructionIndex;
  reg          executionRecordValid;
  reg  [31:0]  executionResult;
  reg  [2:0]   outStanding;
  wire         vfuRequest_valid_0;
  wire         _outStandingUpdate_T = vfuRequest_ready_0 & vfuRequest_valid_0;
  wire [2:0]   outStandingUpdate = _outStandingUpdate_T ? 3'h1 : 3'h7;
  wire         responseFinish = outStanding == 3'h0;
  wire         doubleExecutionInRecord = executionRecord_decodeResult_crossWrite | executionRecord_decodeResult_crossRead | executionRecord_decodeResult_widenReduce;
  assign narrowInRecord = ~executionRecord_decodeResult_crossWrite & executionRecord_decodeResult_crossRead;
  wire         vfuRequest_bits_narrow_0 = narrowInRecord;
  wire         recordQueueReadyForNoExecute;
  wire         recordDequeueReady = vfuRequest_ready_0 | recordQueueReadyForNoExecute;
  wire         recordDequeueFire = executionRecordValid & recordDequeueReady;
  wire [7:0]   extendSource1_dataVec_0 = executionRecord_source_0[7:0];
  wire [7:0]   extendSource1_dataVec_1 = executionRecord_source_0[15:8];
  wire [7:0]   extendSource1_dataVec_2 = executionRecord_source_0[23:16];
  wire [7:0]   extendSource1_dataVec_3 = executionRecord_source_0[31:24];
  wire [7:0]   extendSource1_signVec_0 = {8{extendSource1_dataVec_0[7] & ~executionRecord_decodeResult_unsigned0}};
  wire [7:0]   extendSource1_signVec_1 = {8{extendSource1_dataVec_1[7] & ~executionRecord_decodeResult_unsigned0}};
  wire [7:0]   extendSource1_signVec_2 = {8{extendSource1_dataVec_2[7] & ~executionRecord_decodeResult_unsigned0}};
  wire [7:0]   extendSource1_signVec_3 = {8{extendSource1_dataVec_3[7] & ~executionRecord_decodeResult_unsigned0}};
  wire         extendSource1_sewIsZero = executionRecord_vSew1H[0];
  wire         extendSource2_sewIsZero = executionRecord_vSew1H[0];
  wire [63:0]  extendSource1_sourceExtend =
    {extendSource1_signVec_3,
     extendSource1_sewIsZero ? extendSource1_dataVec_3 : extendSource1_signVec_3,
     extendSource1_sewIsZero ? extendSource1_signVec_2 : extendSource1_dataVec_3,
     extendSource1_dataVec_2,
     extendSource1_signVec_1,
     extendSource1_sewIsZero ? extendSource1_dataVec_1 : extendSource1_signVec_1,
     extendSource1_sewIsZero ? extendSource1_signVec_0 : extendSource1_dataVec_1,
     extendSource1_dataVec_0};
  wire [31:0]  extendSource1 = extendSource1_sourceExtend[31:0];
  wire [7:0]   extendSource2_dataVec_0 = executionRecord_source_1[7:0];
  wire [7:0]   extendSource2_dataVec_1 = executionRecord_source_1[15:8];
  wire [7:0]   extendSource2_dataVec_2 = executionRecord_source_1[23:16];
  wire [7:0]   extendSource2_dataVec_3 = executionRecord_source_1[31:24];
  wire [7:0]   extendSource2_signVec_0 = {8{extendSource2_dataVec_0[7] & ~executionRecord_decodeResult_unsigned1}};
  wire [7:0]   extendSource2_signVec_1 = {8{extendSource2_dataVec_1[7] & ~executionRecord_decodeResult_unsigned1}};
  wire [7:0]   extendSource2_signVec_2 = {8{extendSource2_dataVec_2[7] & ~executionRecord_decodeResult_unsigned1}};
  wire [7:0]   extendSource2_signVec_3 = {8{extendSource2_dataVec_3[7] & ~executionRecord_decodeResult_unsigned1}};
  wire [63:0]  extendSource2_sourceExtend =
    {extendSource2_signVec_3,
     extendSource2_sewIsZero ? extendSource2_dataVec_3 : extendSource2_signVec_3,
     extendSource2_sewIsZero ? extendSource2_signVec_2 : extendSource2_dataVec_3,
     extendSource2_dataVec_2,
     extendSource2_signVec_1,
     extendSource2_sewIsZero ? extendSource2_dataVec_1 : extendSource2_signVec_1,
     extendSource2_sewIsZero ? extendSource2_signVec_0 : extendSource2_dataVec_1,
     extendSource2_dataVec_0};
  wire [31:0]  extendSource2 = extendSource2_sourceExtend[31:0];
  wire [31:0]  normalSource1 = executionRecord_decodeResult_red & ~executionRecord_decodeResult_maskLogic ? 32'h0 : executionRecord_source_0;
  wire [31:0]  _lastGroupMask_T_1 = 32'h1 << executionRecord_csr_vl[4:0];
  wire [29:0]  _GEN = _lastGroupMask_T_1[30:1] | _lastGroupMask_T_1[31:2];
  wire [28:0]  _GEN_0 = _GEN[28:0] | {_lastGroupMask_T_1[31], _GEN[29:2]};
  wire [26:0]  _GEN_1 = _GEN_0[26:0] | {_lastGroupMask_T_1[31], _GEN[29], _GEN_0[28:4]};
  wire [22:0]  _GEN_2 = _GEN_1[22:0] | {_lastGroupMask_T_1[31], _GEN[29], _GEN_0[28:27], _GEN_1[26:8]};
  wire [30:0]  lastGroupMask = {_lastGroupMask_T_1[31], _GEN[29], _GEN_0[28:27], _GEN_1[26:23], _GEN_2[22:15], _GEN_2[14:0] | {_lastGroupMask_T_1[31], _GEN[29], _GEN_0[28:27], _GEN_1[26:23], _GEN_2[22:16]}};
  wire [31:0]  maskCorrect = executionRecord_bordersForMaskLogic ? {1'h0, lastGroupMask} : 32'hFFFFFFFF;
  wire [3:0]   maskExtend = executionRecord_vSew1H[1] ? {{2{executionRecord_maskForMaskInput[1]}}, {2{executionRecord_maskForMaskInput[0]}}} : executionRecord_maskForMaskInput;
  wire [32:0]  vfuRequest_bits_src_0_0 = {1'h0, normalSource1};
  wire [32:0]  vfuRequest_bits_src_1_0 = {1'h0, executionRecord_source_1};
  wire [32:0]  vfuRequest_bits_src_2_0 = {1'h0, executionRecord_source_2};
  wire [32:0]  vfuRequest_bits_src_3_0 = {1'h0, maskCorrect};
  wire [3:0]   vfuRequest_bits_mask_0 = executionRecord_decodeResult_adder ? (executionRecord_decodeResult_maskSource ? executionRecord_maskForMaskInput : 4'h0) : maskExtend | {4{~executionRecord_maskType}};
  wire [1:0]   vfuRequest_bits_vSew_0 = doubleExecutionInRecord ? executionRecord_csr_vSew + 2'h1 : executionRecord_csr_vSew;
  wire [1:0]   shifterSizeBit = executionRecord_crossReadVS2 ? executionRecord_vSew1H[1:0] : executionRecord_vSew1H[2:1];
  wire [9:0]   vfuRequest_bits_shifterSize_lo =
    {(shifterSizeBit[0] ? {1'h0, normalSource1[11]} : 2'h0) | (shifterSizeBit[1] ? normalSource1[12:11] : 2'h0),
     normalSource1[10:8],
     (shifterSizeBit[0] ? {1'h0, normalSource1[3]} : 2'h0) | (shifterSizeBit[1] ? normalSource1[4:3] : 2'h0),
     normalSource1[2:0]};
  wire [9:0]   vfuRequest_bits_shifterSize_hi =
    {(shifterSizeBit[0] ? {1'h0, normalSource1[27]} : 2'h0) | (shifterSizeBit[1] ? normalSource1[28:27] : 2'h0),
     normalSource1[26:24],
     (shifterSizeBit[0] ? {1'h0, normalSource1[19]} : 2'h0) | (shifterSizeBit[1] ? normalSource1[20:19] : 2'h0),
     normalSource1[18:16]};
  wire [19:0]  vfuRequest_bits_shifterSize_0 = {vfuRequest_bits_shifterSize_hi, vfuRequest_bits_shifterSize_lo};
  wire         vfuRequest_bits_rem_0 = executionRecord_decodeResult_uop[0];
  assign vfuRequest_valid_0 = executionRecordValid & ~executionRecord_decodeResult_dontNeedExecuteInLane;
  wire [1:0]   recordQueue_dataIn_lo_lo_lo_lo = {recordQueue_enq_bits_decodeResult_adder, recordQueue_enq_bits_decodeResult_logic};
  wire [1:0]   recordQueue_dataIn_lo_lo_lo_hi_hi = {recordQueue_enq_bits_decodeResult_divider, recordQueue_enq_bits_decodeResult_multiplier};
  wire [2:0]   recordQueue_dataIn_lo_lo_lo_hi = {recordQueue_dataIn_lo_lo_lo_hi_hi, recordQueue_enq_bits_decodeResult_shift};
  wire [4:0]   recordQueue_dataIn_lo_lo_lo = {recordQueue_dataIn_lo_lo_lo_hi, recordQueue_dataIn_lo_lo_lo_lo};
  wire [1:0]   recordQueue_dataIn_lo_lo_hi_lo_hi = {recordQueue_enq_bits_decodeResult_unsigned0, recordQueue_enq_bits_decodeResult_other};
  wire [2:0]   recordQueue_dataIn_lo_lo_hi_lo = {recordQueue_dataIn_lo_lo_hi_lo_hi, recordQueue_enq_bits_decodeResult_multiCycle};
  wire [1:0]   recordQueue_dataIn_lo_lo_hi_hi_hi = {recordQueue_enq_bits_decodeResult_nr, recordQueue_enq_bits_decodeResult_itype};
  wire [2:0]   recordQueue_dataIn_lo_lo_hi_hi = {recordQueue_dataIn_lo_lo_hi_hi_hi, recordQueue_enq_bits_decodeResult_unsigned1};
  wire [5:0]   recordQueue_dataIn_lo_lo_hi = {recordQueue_dataIn_lo_lo_hi_hi, recordQueue_dataIn_lo_lo_hi_lo};
  wire [10:0]  recordQueue_dataIn_lo_lo = {recordQueue_dataIn_lo_lo_hi, recordQueue_dataIn_lo_lo_lo};
  wire [1:0]   recordQueue_dataIn_lo_hi_lo_lo = {recordQueue_enq_bits_decodeResult_widenReduce, recordQueue_enq_bits_decodeResult_red};
  wire [1:0]   recordQueue_dataIn_lo_hi_lo_hi_hi = {recordQueue_enq_bits_decodeResult_gather, recordQueue_enq_bits_decodeResult_slid};
  wire [2:0]   recordQueue_dataIn_lo_hi_lo_hi = {recordQueue_dataIn_lo_hi_lo_hi_hi, recordQueue_enq_bits_decodeResult_targetRd};
  wire [4:0]   recordQueue_dataIn_lo_hi_lo = {recordQueue_dataIn_lo_hi_lo_hi, recordQueue_dataIn_lo_hi_lo_lo};
  wire [1:0]   recordQueue_dataIn_lo_hi_hi_lo_hi = {recordQueue_enq_bits_decodeResult_unOrderWrite, recordQueue_enq_bits_decodeResult_compress};
  wire [2:0]   recordQueue_dataIn_lo_hi_hi_lo = {recordQueue_dataIn_lo_hi_hi_lo_hi, recordQueue_enq_bits_decodeResult_gather16};
  wire [1:0]   recordQueue_dataIn_lo_hi_hi_hi_hi = {recordQueue_enq_bits_decodeResult_iota, recordQueue_enq_bits_decodeResult_mv};
  wire [2:0]   recordQueue_dataIn_lo_hi_hi_hi = {recordQueue_dataIn_lo_hi_hi_hi_hi, recordQueue_enq_bits_decodeResult_extend};
  wire [5:0]   recordQueue_dataIn_lo_hi_hi = {recordQueue_dataIn_lo_hi_hi_hi, recordQueue_dataIn_lo_hi_hi_lo};
  wire [10:0]  recordQueue_dataIn_lo_hi = {recordQueue_dataIn_lo_hi_hi, recordQueue_dataIn_lo_hi_lo};
  wire [21:0]  recordQueue_dataIn_lo = {recordQueue_dataIn_lo_hi, recordQueue_dataIn_lo_lo};
  wire [4:0]   recordQueue_dataIn_hi_lo_lo_lo = {recordQueue_enq_bits_decodeResult_maskLogic, recordQueue_enq_bits_decodeResult_uop};
  wire [1:0]   recordQueue_dataIn_hi_lo_lo_hi_hi = {recordQueue_enq_bits_decodeResult_readOnly, recordQueue_enq_bits_decodeResult_maskSource};
  wire [2:0]   recordQueue_dataIn_hi_lo_lo_hi = {recordQueue_dataIn_hi_lo_lo_hi_hi, recordQueue_enq_bits_decodeResult_maskDestination};
  wire [7:0]   recordQueue_dataIn_hi_lo_lo = {recordQueue_dataIn_hi_lo_lo_hi, recordQueue_dataIn_hi_lo_lo_lo};
  wire [1:0]   recordQueue_dataIn_hi_lo_hi_lo_hi = {recordQueue_enq_bits_decodeResult_special, recordQueue_enq_bits_decodeResult_saturate};
  wire [2:0]   recordQueue_dataIn_hi_lo_hi_lo = {recordQueue_dataIn_hi_lo_hi_lo_hi, recordQueue_enq_bits_decodeResult_vwmacc};
  wire [1:0]   recordQueue_dataIn_hi_lo_hi_hi_hi = {recordQueue_enq_bits_decodeResult_crossRead, recordQueue_enq_bits_decodeResult_crossWrite};
  wire [2:0]   recordQueue_dataIn_hi_lo_hi_hi = {recordQueue_dataIn_hi_lo_hi_hi_hi, recordQueue_enq_bits_decodeResult_maskUnit};
  wire [5:0]   recordQueue_dataIn_hi_lo_hi = {recordQueue_dataIn_hi_lo_hi_hi, recordQueue_dataIn_hi_lo_hi_lo};
  wire [13:0]  recordQueue_dataIn_hi_lo = {recordQueue_dataIn_hi_lo_hi, recordQueue_dataIn_hi_lo_lo};
  wire [1:0]   recordQueue_dataIn_hi_hi_lo_lo = {recordQueue_enq_bits_decodeResult_vtype, recordQueue_enq_bits_decodeResult_sWrite};
  wire [1:0]   recordQueue_dataIn_hi_hi_lo_hi_hi = {recordQueue_enq_bits_decodeResult_dontNeedExecuteInLane, recordQueue_enq_bits_decodeResult_scheduler};
  wire [2:0]   recordQueue_dataIn_hi_hi_lo_hi = {recordQueue_dataIn_hi_hi_lo_hi_hi, recordQueue_enq_bits_decodeResult_sReadVD};
  wire [4:0]   recordQueue_dataIn_hi_hi_lo = {recordQueue_dataIn_hi_hi_lo_hi, recordQueue_dataIn_hi_hi_lo_lo};
  wire [1:0]   recordQueue_dataIn_hi_hi_hi_lo_hi = {recordQueue_enq_bits_decodeResult_ffo, recordQueue_enq_bits_decodeResult_average};
  wire [2:0]   recordQueue_dataIn_hi_hi_hi_lo = {recordQueue_dataIn_hi_hi_hi_lo_hi, recordQueue_enq_bits_decodeResult_reverse};
  wire [5:0]   recordQueue_dataIn_hi_hi_hi_hi_hi = {recordQueue_enq_bits_decodeResult_specialSlot, recordQueue_enq_bits_decodeResult_topUop};
  wire [6:0]   recordQueue_dataIn_hi_hi_hi_hi = {recordQueue_dataIn_hi_hi_hi_hi_hi, recordQueue_enq_bits_decodeResult_popCount};
  wire [9:0]   recordQueue_dataIn_hi_hi_hi = {recordQueue_dataIn_hi_hi_hi_hi, recordQueue_dataIn_hi_hi_hi_lo};
  wire [14:0]  recordQueue_dataIn_hi_hi = {recordQueue_dataIn_hi_hi_hi, recordQueue_dataIn_hi_hi_lo};
  wire [28:0]  recordQueue_dataIn_hi = {recordQueue_dataIn_hi_hi, recordQueue_dataIn_hi_lo};
  wire [5:0]   recordQueue_dataIn_lo_lo_1 = {recordQueue_enq_bits_vSew1H, recordQueue_enq_bits_instructionIndex};
  wire [82:0]  recordQueue_dataIn_lo_hi_1 = {recordQueue_enq_bits_source2, recordQueue_dataIn_hi, recordQueue_dataIn_lo};
  wire [88:0]  recordQueue_dataIn_lo_1 = {recordQueue_dataIn_lo_hi_1, recordQueue_dataIn_lo_lo_1};
  wire [13:0]  recordQueue_dataIn_hi_lo_1 = {recordQueue_enq_bits_groupCounter, 1'h0};
  wire [4:0]   recordQueue_dataIn_hi_hi_1 = {recordQueue_enq_bits_bordersForMaskLogic, recordQueue_enq_bits_maskForFilter};
  wire [18:0]  recordQueue_dataIn_hi_1 = {recordQueue_dataIn_hi_hi_1, recordQueue_dataIn_hi_lo_1};
  wire [107:0] recordQueue_dataIn = {recordQueue_dataIn_hi_1, recordQueue_dataIn_lo_1};
  wire [2:0]   recordQueue_dataOut_instructionIndex = _recordQueue_fifo_data_out[2:0];
  wire [2:0]   recordQueue_dataOut_vSew1H = _recordQueue_fifo_data_out[5:3];
  wire         recordQueue_dataOut_decodeResult_logic = _recordQueue_fifo_data_out[6];
  wire         recordQueue_dataOut_decodeResult_adder = _recordQueue_fifo_data_out[7];
  wire         recordQueue_dataOut_decodeResult_shift = _recordQueue_fifo_data_out[8];
  wire         recordQueue_dataOut_decodeResult_multiplier = _recordQueue_fifo_data_out[9];
  wire         recordQueue_dataOut_decodeResult_divider = _recordQueue_fifo_data_out[10];
  wire         recordQueue_dataOut_decodeResult_multiCycle = _recordQueue_fifo_data_out[11];
  wire         recordQueue_dataOut_decodeResult_other = _recordQueue_fifo_data_out[12];
  wire         recordQueue_dataOut_decodeResult_unsigned0 = _recordQueue_fifo_data_out[13];
  wire         recordQueue_dataOut_decodeResult_unsigned1 = _recordQueue_fifo_data_out[14];
  wire         recordQueue_dataOut_decodeResult_itype = _recordQueue_fifo_data_out[15];
  wire         recordQueue_dataOut_decodeResult_nr = _recordQueue_fifo_data_out[16];
  wire         recordQueue_dataOut_decodeResult_red = _recordQueue_fifo_data_out[17];
  wire         recordQueue_dataOut_decodeResult_widenReduce = _recordQueue_fifo_data_out[18];
  wire         recordQueue_dataOut_decodeResult_targetRd = _recordQueue_fifo_data_out[19];
  wire         recordQueue_dataOut_decodeResult_slid = _recordQueue_fifo_data_out[20];
  wire         recordQueue_dataOut_decodeResult_gather = _recordQueue_fifo_data_out[21];
  wire         recordQueue_dataOut_decodeResult_gather16 = _recordQueue_fifo_data_out[22];
  wire         recordQueue_dataOut_decodeResult_compress = _recordQueue_fifo_data_out[23];
  wire         recordQueue_dataOut_decodeResult_unOrderWrite = _recordQueue_fifo_data_out[24];
  wire         recordQueue_dataOut_decodeResult_extend = _recordQueue_fifo_data_out[25];
  wire         recordQueue_dataOut_decodeResult_mv = _recordQueue_fifo_data_out[26];
  wire         recordQueue_dataOut_decodeResult_iota = _recordQueue_fifo_data_out[27];
  wire [3:0]   recordQueue_dataOut_decodeResult_uop = _recordQueue_fifo_data_out[31:28];
  wire         recordQueue_dataOut_decodeResult_maskLogic = _recordQueue_fifo_data_out[32];
  wire         recordQueue_dataOut_decodeResult_maskDestination = _recordQueue_fifo_data_out[33];
  wire         recordQueue_dataOut_decodeResult_maskSource = _recordQueue_fifo_data_out[34];
  wire         recordQueue_dataOut_decodeResult_readOnly = _recordQueue_fifo_data_out[35];
  wire         recordQueue_dataOut_decodeResult_vwmacc = _recordQueue_fifo_data_out[36];
  wire         recordQueue_dataOut_decodeResult_saturate = _recordQueue_fifo_data_out[37];
  wire         recordQueue_dataOut_decodeResult_special = _recordQueue_fifo_data_out[38];
  wire         recordQueue_dataOut_decodeResult_maskUnit = _recordQueue_fifo_data_out[39];
  wire         recordQueue_dataOut_decodeResult_crossWrite = _recordQueue_fifo_data_out[40];
  wire         recordQueue_dataOut_decodeResult_crossRead = _recordQueue_fifo_data_out[41];
  wire         recordQueue_dataOut_decodeResult_sWrite = _recordQueue_fifo_data_out[42];
  wire         recordQueue_dataOut_decodeResult_vtype = _recordQueue_fifo_data_out[43];
  wire         recordQueue_dataOut_decodeResult_sReadVD = _recordQueue_fifo_data_out[44];
  wire         recordQueue_dataOut_decodeResult_scheduler = _recordQueue_fifo_data_out[45];
  wire         recordQueue_dataOut_decodeResult_dontNeedExecuteInLane = _recordQueue_fifo_data_out[46];
  wire         recordQueue_dataOut_decodeResult_reverse = _recordQueue_fifo_data_out[47];
  wire         recordQueue_dataOut_decodeResult_average = _recordQueue_fifo_data_out[48];
  wire         recordQueue_dataOut_decodeResult_ffo = _recordQueue_fifo_data_out[49];
  wire         recordQueue_dataOut_decodeResult_popCount = _recordQueue_fifo_data_out[50];
  wire [4:0]   recordQueue_dataOut_decodeResult_topUop = _recordQueue_fifo_data_out[55:51];
  wire         recordQueue_dataOut_decodeResult_specialSlot = _recordQueue_fifo_data_out[56];
  wire [31:0]  recordQueue_dataOut_source2 = _recordQueue_fifo_data_out[88:57];
  wire         recordQueue_dataOut_executeIndex = _recordQueue_fifo_data_out[89];
  wire [12:0]  recordQueue_dataOut_groupCounter = _recordQueue_fifo_data_out[102:90];
  wire [3:0]   recordQueue_dataOut_maskForFilter = _recordQueue_fifo_data_out[106:103];
  wire         recordQueue_dataOut_bordersForMaskLogic = _recordQueue_fifo_data_out[107];
  wire         recordQueue_enq_ready = ~_recordQueue_fifo_full;
  wire         recordQueue_deq_ready;
  wire         recordQueue_enq_valid;
  wire         recordQueue_deq_valid = ~_recordQueue_fifo_empty | recordQueue_enq_valid;
  wire         recordQueue_deq_bits_bordersForMaskLogic = _recordQueue_fifo_empty ? recordQueue_enq_bits_bordersForMaskLogic : recordQueue_dataOut_bordersForMaskLogic;
  wire [3:0]   recordQueue_deq_bits_maskForFilter = _recordQueue_fifo_empty ? recordQueue_enq_bits_maskForFilter : recordQueue_dataOut_maskForFilter;
  wire [12:0]  recordQueue_deq_bits_groupCounter = _recordQueue_fifo_empty ? recordQueue_enq_bits_groupCounter : recordQueue_dataOut_groupCounter;
  wire         recordQueue_deq_bits_executeIndex = ~_recordQueue_fifo_empty & recordQueue_dataOut_executeIndex;
  wire [31:0]  recordQueue_deq_bits_source2 = _recordQueue_fifo_empty ? recordQueue_enq_bits_source2 : recordQueue_dataOut_source2;
  wire         recordQueue_deq_bits_decodeResult_specialSlot = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_specialSlot : recordQueue_dataOut_decodeResult_specialSlot;
  wire [4:0]   recordQueue_deq_bits_decodeResult_topUop = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_topUop : recordQueue_dataOut_decodeResult_topUop;
  wire         recordQueue_deq_bits_decodeResult_popCount = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_popCount : recordQueue_dataOut_decodeResult_popCount;
  wire         recordQueue_deq_bits_decodeResult_ffo = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_ffo : recordQueue_dataOut_decodeResult_ffo;
  wire         recordQueue_deq_bits_decodeResult_average = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_average : recordQueue_dataOut_decodeResult_average;
  wire         recordQueue_deq_bits_decodeResult_reverse = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_reverse : recordQueue_dataOut_decodeResult_reverse;
  wire         recordQueue_deq_bits_decodeResult_dontNeedExecuteInLane = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_dontNeedExecuteInLane : recordQueue_dataOut_decodeResult_dontNeedExecuteInLane;
  wire         recordQueue_deq_bits_decodeResult_scheduler = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_scheduler : recordQueue_dataOut_decodeResult_scheduler;
  wire         recordQueue_deq_bits_decodeResult_sReadVD = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_sReadVD : recordQueue_dataOut_decodeResult_sReadVD;
  wire         recordQueue_deq_bits_decodeResult_vtype = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_vtype : recordQueue_dataOut_decodeResult_vtype;
  wire         recordQueue_deq_bits_decodeResult_sWrite = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_sWrite : recordQueue_dataOut_decodeResult_sWrite;
  wire         recordQueue_deq_bits_decodeResult_crossRead = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_crossRead : recordQueue_dataOut_decodeResult_crossRead;
  wire         recordQueue_deq_bits_decodeResult_crossWrite = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_crossWrite : recordQueue_dataOut_decodeResult_crossWrite;
  wire         recordQueue_deq_bits_decodeResult_maskUnit = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_maskUnit : recordQueue_dataOut_decodeResult_maskUnit;
  wire         recordQueue_deq_bits_decodeResult_special = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_special : recordQueue_dataOut_decodeResult_special;
  wire         recordQueue_deq_bits_decodeResult_saturate = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_saturate : recordQueue_dataOut_decodeResult_saturate;
  wire         recordQueue_deq_bits_decodeResult_vwmacc = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_vwmacc : recordQueue_dataOut_decodeResult_vwmacc;
  wire         recordQueue_deq_bits_decodeResult_readOnly = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_readOnly : recordQueue_dataOut_decodeResult_readOnly;
  wire         recordQueue_deq_bits_decodeResult_maskSource = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_maskSource : recordQueue_dataOut_decodeResult_maskSource;
  wire         recordQueue_deq_bits_decodeResult_maskDestination = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_maskDestination : recordQueue_dataOut_decodeResult_maskDestination;
  wire         recordQueue_deq_bits_decodeResult_maskLogic = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_maskLogic : recordQueue_dataOut_decodeResult_maskLogic;
  wire [3:0]   recordQueue_deq_bits_decodeResult_uop = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_uop : recordQueue_dataOut_decodeResult_uop;
  wire         recordQueue_deq_bits_decodeResult_iota = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_iota : recordQueue_dataOut_decodeResult_iota;
  wire         recordQueue_deq_bits_decodeResult_mv = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_mv : recordQueue_dataOut_decodeResult_mv;
  wire         recordQueue_deq_bits_decodeResult_extend = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_extend : recordQueue_dataOut_decodeResult_extend;
  wire         recordQueue_deq_bits_decodeResult_unOrderWrite = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_unOrderWrite : recordQueue_dataOut_decodeResult_unOrderWrite;
  wire         recordQueue_deq_bits_decodeResult_compress = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_compress : recordQueue_dataOut_decodeResult_compress;
  wire         recordQueue_deq_bits_decodeResult_gather16 = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_gather16 : recordQueue_dataOut_decodeResult_gather16;
  wire         recordQueue_deq_bits_decodeResult_gather = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_gather : recordQueue_dataOut_decodeResult_gather;
  wire         recordQueue_deq_bits_decodeResult_slid = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_slid : recordQueue_dataOut_decodeResult_slid;
  wire         recordQueue_deq_bits_decodeResult_targetRd = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_targetRd : recordQueue_dataOut_decodeResult_targetRd;
  wire         recordQueue_deq_bits_decodeResult_widenReduce = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_widenReduce : recordQueue_dataOut_decodeResult_widenReduce;
  wire         recordQueue_deq_bits_decodeResult_red = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_red : recordQueue_dataOut_decodeResult_red;
  wire         recordQueue_deq_bits_decodeResult_nr = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_nr : recordQueue_dataOut_decodeResult_nr;
  wire         recordQueue_deq_bits_decodeResult_itype = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_itype : recordQueue_dataOut_decodeResult_itype;
  wire         recordQueue_deq_bits_decodeResult_unsigned1 = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_unsigned1 : recordQueue_dataOut_decodeResult_unsigned1;
  wire         recordQueue_deq_bits_decodeResult_unsigned0 = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_unsigned0 : recordQueue_dataOut_decodeResult_unsigned0;
  wire         recordQueue_deq_bits_decodeResult_other = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_other : recordQueue_dataOut_decodeResult_other;
  wire         recordQueue_deq_bits_decodeResult_multiCycle = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_multiCycle : recordQueue_dataOut_decodeResult_multiCycle;
  wire         recordQueue_deq_bits_decodeResult_divider = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_divider : recordQueue_dataOut_decodeResult_divider;
  wire         recordQueue_deq_bits_decodeResult_multiplier = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_multiplier : recordQueue_dataOut_decodeResult_multiplier;
  wire         recordQueue_deq_bits_decodeResult_shift = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_shift : recordQueue_dataOut_decodeResult_shift;
  wire         recordQueue_deq_bits_decodeResult_adder = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_adder : recordQueue_dataOut_decodeResult_adder;
  wire         recordQueue_deq_bits_decodeResult_logic = _recordQueue_fifo_empty ? recordQueue_enq_bits_decodeResult_logic : recordQueue_dataOut_decodeResult_logic;
  wire [2:0]   recordQueue_deq_bits_vSew1H = _recordQueue_fifo_empty ? recordQueue_enq_bits_vSew1H : recordQueue_dataOut_vSew1H;
  wire [2:0]   recordQueue_deq_bits_instructionIndex = _recordQueue_fifo_empty ? recordQueue_enq_bits_instructionIndex : recordQueue_dataOut_instructionIndex;
  assign recordQueueReadyForNoExecute = executionRecord_decodeResult_dontNeedExecuteInLane & recordQueue_enq_ready;
  assign recordQueue_enq_valid = executionRecordValid & (vfuRequest_ready_0 | executionRecord_decodeResult_dontNeedExecuteInLane);
  wire         doubleExecutionInQueue = recordQueue_deq_bits_decodeResult_crossWrite | recordQueue_deq_bits_decodeResult_crossRead | recordQueue_deq_bits_decodeResult_widenReduce;
  wire [31:0]  dataDequeue = recordQueue_deq_bits_decodeResult_dontNeedExecuteInLane ? recordQueue_deq_bits_source2 : dataResponse_bits_data;
  wire [31:0]  queue_enq_bits_data = dataDequeue;
  wire [7:0]   deqVec_0 = dataDequeue[7:0];
  wire [7:0]   deqVec_1 = dataDequeue[15:8];
  wire [7:0]   deqVec_2 = dataDequeue[23:16];
  wire [7:0]   deqVec_3 = dataDequeue[31:24];
  wire [31:0]  narrowUpdate = {recordQueue_deq_bits_vSew1H[0] ? deqVec_2 : deqVec_1, deqVec_0, executionResult[31:16]};
  wire         narrowInQueue = ~recordQueue_deq_bits_decodeResult_crossWrite & recordQueue_deq_bits_decodeResult_crossRead;
  wire [31:0]  lastSlotDataUpdate = narrowInQueue ? narrowUpdate : dataDequeue;
  assign queue_deq_valid = ~_queue_fifo_empty;
  wire         dequeue_valid_0 = queue_deq_valid;
  wire [31:0]  queue_dataOut_data;
  wire [31:0]  dequeue_bits_data_0 = queue_deq_bits_data;
  wire [13:0]  queue_dataOut_ffoIndex;
  wire [13:0]  dequeue_bits_ffoIndex_0 = queue_deq_bits_ffoIndex;
  wire [13:0]  queue_enq_bits_ffoIndex;
  wire [45:0]  queue_dataIn = {queue_enq_bits_data, queue_enq_bits_ffoIndex};
  assign queue_dataOut_ffoIndex = _queue_fifo_data_out[13:0];
  assign queue_dataOut_data = _queue_fifo_data_out[45:14];
  assign queue_deq_bits_data = queue_dataOut_data;
  assign queue_deq_bits_ffoIndex = queue_dataOut_ffoIndex;
  wire         queue_enq_ready = ~_queue_fifo_full;
  wire         queue_enq_valid;
  assign queue_enq_bits_ffoIndex = {recordQueue_deq_bits_groupCounter[8:0], dataResponse_bits_data[4:0]};
  assign recordQueue_deq_ready = dataResponse_valid | recordQueue_deq_bits_decodeResult_dontNeedExecuteInLane & queue_enq_ready;
  assign queue_enq_valid = recordQueue_deq_valid & (dataResponse_valid & (~doubleExecutionInQueue | recordQueue_deq_bits_executeIndex) | recordQueue_deq_bits_decodeResult_dontNeedExecuteInLane);
  wire [1:0]   executionTypeInRecord_lo_hi = {executionRecord_decodeResult_shift, executionRecord_decodeResult_adder};
  wire [2:0]   executionTypeInRecord_lo = {executionTypeInRecord_lo_hi, executionRecord_decodeResult_logic};
  wire [1:0]   executionTypeInRecord_hi_hi = {executionRecord_decodeResult_other, executionRecord_decodeResult_divider};
  wire [2:0]   executionTypeInRecord_hi = {executionTypeInRecord_hi_hi, executionRecord_decodeResult_multiplier};
  wire [5:0]   executionTypeInRecord = {executionTypeInRecord_hi, executionTypeInRecord_lo};
  wire [1:0]   enqType_lo_hi = {enqueue_bits_decodeResult_shift_0, enqueue_bits_decodeResult_adder_0};
  wire [2:0]   enqType_lo = {enqType_lo_hi, enqueue_bits_decodeResult_logic_0};
  wire [1:0]   enqType_hi_hi = {enqueue_bits_decodeResult_other_0, enqueue_bits_decodeResult_divider_0};
  wire [2:0]   enqType_hi = {enqType_hi_hi, enqueue_bits_decodeResult_multiplier_0};
  wire [5:0]   enqType = {enqType_hi, enqType_lo};
  wire         typeCheck = executionTypeInRecord == enqType | ~(executionRecordValid | recordQueue_deq_valid);
  wire         enqueue_ready_0 = (~executionRecordValid | recordDequeueReady) & typeCheck;
  always @(posedge clock) begin
    if (reset) begin
      executionRecord_crossReadVS2 <= 1'h0;
      executionRecord_bordersForMaskLogic <= 1'h0;
      executionRecord_maskForMaskInput <= 4'h0;
      executionRecord_maskForFilter <= 4'h0;
      executionRecord_source_0 <= 32'h0;
      executionRecord_source_1 <= 32'h0;
      executionRecord_source_2 <= 32'h0;
      executionRecord_groupCounter <= 13'h0;
      executionRecord_vSew1H <= 3'h0;
      executionRecord_csr_vl <= 17'h0;
      executionRecord_csr_vStart <= 17'h0;
      executionRecord_csr_vlmul <= 3'h0;
      executionRecord_csr_vSew <= 2'h0;
      executionRecord_csr_vxrm <= 2'h0;
      executionRecord_csr_vta <= 1'h0;
      executionRecord_csr_vma <= 1'h0;
      executionRecord_maskType <= 1'h0;
      executionRecord_laneIndex <= 2'h0;
      executionRecord_decodeResult_specialSlot <= 1'h0;
      executionRecord_decodeResult_topUop <= 5'h0;
      executionRecord_decodeResult_popCount <= 1'h0;
      executionRecord_decodeResult_ffo <= 1'h0;
      executionRecord_decodeResult_average <= 1'h0;
      executionRecord_decodeResult_reverse <= 1'h0;
      executionRecord_decodeResult_dontNeedExecuteInLane <= 1'h0;
      executionRecord_decodeResult_scheduler <= 1'h0;
      executionRecord_decodeResult_sReadVD <= 1'h0;
      executionRecord_decodeResult_vtype <= 1'h0;
      executionRecord_decodeResult_sWrite <= 1'h0;
      executionRecord_decodeResult_crossRead <= 1'h0;
      executionRecord_decodeResult_crossWrite <= 1'h0;
      executionRecord_decodeResult_maskUnit <= 1'h0;
      executionRecord_decodeResult_special <= 1'h0;
      executionRecord_decodeResult_saturate <= 1'h0;
      executionRecord_decodeResult_vwmacc <= 1'h0;
      executionRecord_decodeResult_readOnly <= 1'h0;
      executionRecord_decodeResult_maskSource <= 1'h0;
      executionRecord_decodeResult_maskDestination <= 1'h0;
      executionRecord_decodeResult_maskLogic <= 1'h0;
      executionRecord_decodeResult_uop <= 4'h0;
      executionRecord_decodeResult_iota <= 1'h0;
      executionRecord_decodeResult_mv <= 1'h0;
      executionRecord_decodeResult_extend <= 1'h0;
      executionRecord_decodeResult_unOrderWrite <= 1'h0;
      executionRecord_decodeResult_compress <= 1'h0;
      executionRecord_decodeResult_gather16 <= 1'h0;
      executionRecord_decodeResult_gather <= 1'h0;
      executionRecord_decodeResult_slid <= 1'h0;
      executionRecord_decodeResult_targetRd <= 1'h0;
      executionRecord_decodeResult_widenReduce <= 1'h0;
      executionRecord_decodeResult_red <= 1'h0;
      executionRecord_decodeResult_nr <= 1'h0;
      executionRecord_decodeResult_itype <= 1'h0;
      executionRecord_decodeResult_unsigned1 <= 1'h0;
      executionRecord_decodeResult_unsigned0 <= 1'h0;
      executionRecord_decodeResult_other <= 1'h0;
      executionRecord_decodeResult_multiCycle <= 1'h0;
      executionRecord_decodeResult_divider <= 1'h0;
      executionRecord_decodeResult_multiplier <= 1'h0;
      executionRecord_decodeResult_shift <= 1'h0;
      executionRecord_decodeResult_adder <= 1'h0;
      executionRecord_decodeResult_logic <= 1'h0;
      executionRecord_instructionIndex <= 3'h0;
      executionRecordValid <= 1'h0;
      executionResult <= 32'h0;
      outStanding <= 3'h0;
    end
    else begin
      automatic logic _executionRecordValid_T;
      _executionRecordValid_T = enqueue_ready_0 & enqueue_valid_0;
      if (_executionRecordValid_T) begin
        executionRecord_crossReadVS2 <= enqueue_bits_decodeResult_crossRead_0 & ~enqueue_bits_decodeResult_vwmacc_0;
        executionRecord_bordersForMaskLogic <= enqueue_bits_bordersForMaskLogic_0 & enqueue_bits_decodeResult_maskLogic_0;
        executionRecord_maskForMaskInput <= enqueue_bits_mask_0;
        executionRecord_maskForFilter <= enqueue_bits_maskForFilter_0;
        executionRecord_source_0 <= enqueue_bits_src_0_0;
        executionRecord_source_1 <= enqueue_bits_src_1_0;
        executionRecord_source_2 <= enqueue_bits_src_2_0;
        executionRecord_groupCounter <= enqueue_bits_groupCounter_0;
        executionRecord_vSew1H <= enqueue_bits_vSew1H_0;
        executionRecord_csr_vl <= enqueue_bits_csr_vl_0;
        executionRecord_csr_vStart <= enqueue_bits_csr_vStart_0;
        executionRecord_csr_vlmul <= enqueue_bits_csr_vlmul_0;
        executionRecord_csr_vSew <= enqueue_bits_csr_vSew_0;
        executionRecord_csr_vxrm <= enqueue_bits_csr_vxrm_0;
        executionRecord_csr_vta <= enqueue_bits_csr_vta_0;
        executionRecord_csr_vma <= enqueue_bits_csr_vma_0;
        executionRecord_maskType <= enqueue_bits_maskType_0;
        executionRecord_laneIndex <= enqueue_bits_laneIndex_0;
        executionRecord_decodeResult_specialSlot <= enqueue_bits_decodeResult_specialSlot_0;
        executionRecord_decodeResult_topUop <= enqueue_bits_decodeResult_topUop_0;
        executionRecord_decodeResult_popCount <= enqueue_bits_decodeResult_popCount_0;
        executionRecord_decodeResult_ffo <= enqueue_bits_decodeResult_ffo_0;
        executionRecord_decodeResult_average <= enqueue_bits_decodeResult_average_0;
        executionRecord_decodeResult_reverse <= enqueue_bits_decodeResult_reverse_0;
        executionRecord_decodeResult_dontNeedExecuteInLane <= enqueue_bits_decodeResult_dontNeedExecuteInLane_0;
        executionRecord_decodeResult_scheduler <= enqueue_bits_decodeResult_scheduler_0;
        executionRecord_decodeResult_sReadVD <= enqueue_bits_decodeResult_sReadVD_0;
        executionRecord_decodeResult_vtype <= enqueue_bits_decodeResult_vtype_0;
        executionRecord_decodeResult_sWrite <= enqueue_bits_decodeResult_sWrite_0;
        executionRecord_decodeResult_crossRead <= enqueue_bits_decodeResult_crossRead_0;
        executionRecord_decodeResult_crossWrite <= enqueue_bits_decodeResult_crossWrite_0;
        executionRecord_decodeResult_maskUnit <= enqueue_bits_decodeResult_maskUnit_0;
        executionRecord_decodeResult_special <= enqueue_bits_decodeResult_special_0;
        executionRecord_decodeResult_saturate <= enqueue_bits_decodeResult_saturate_0;
        executionRecord_decodeResult_vwmacc <= enqueue_bits_decodeResult_vwmacc_0;
        executionRecord_decodeResult_readOnly <= enqueue_bits_decodeResult_readOnly_0;
        executionRecord_decodeResult_maskSource <= enqueue_bits_decodeResult_maskSource_0;
        executionRecord_decodeResult_maskDestination <= enqueue_bits_decodeResult_maskDestination_0;
        executionRecord_decodeResult_maskLogic <= enqueue_bits_decodeResult_maskLogic_0;
        executionRecord_decodeResult_uop <= enqueue_bits_decodeResult_uop_0;
        executionRecord_decodeResult_iota <= enqueue_bits_decodeResult_iota_0;
        executionRecord_decodeResult_mv <= enqueue_bits_decodeResult_mv_0;
        executionRecord_decodeResult_extend <= enqueue_bits_decodeResult_extend_0;
        executionRecord_decodeResult_unOrderWrite <= enqueue_bits_decodeResult_unOrderWrite_0;
        executionRecord_decodeResult_compress <= enqueue_bits_decodeResult_compress_0;
        executionRecord_decodeResult_gather16 <= enqueue_bits_decodeResult_gather16_0;
        executionRecord_decodeResult_gather <= enqueue_bits_decodeResult_gather_0;
        executionRecord_decodeResult_slid <= enqueue_bits_decodeResult_slid_0;
        executionRecord_decodeResult_targetRd <= enqueue_bits_decodeResult_targetRd_0;
        executionRecord_decodeResult_widenReduce <= enqueue_bits_decodeResult_widenReduce_0;
        executionRecord_decodeResult_red <= enqueue_bits_decodeResult_red_0;
        executionRecord_decodeResult_nr <= enqueue_bits_decodeResult_nr_0;
        executionRecord_decodeResult_itype <= enqueue_bits_decodeResult_itype_0;
        executionRecord_decodeResult_unsigned1 <= enqueue_bits_decodeResult_unsigned1_0;
        executionRecord_decodeResult_unsigned0 <= enqueue_bits_decodeResult_unsigned0_0;
        executionRecord_decodeResult_other <= enqueue_bits_decodeResult_other_0;
        executionRecord_decodeResult_multiCycle <= enqueue_bits_decodeResult_multiCycle_0;
        executionRecord_decodeResult_divider <= enqueue_bits_decodeResult_divider_0;
        executionRecord_decodeResult_multiplier <= enqueue_bits_decodeResult_multiplier_0;
        executionRecord_decodeResult_shift <= enqueue_bits_decodeResult_shift_0;
        executionRecord_decodeResult_adder <= enqueue_bits_decodeResult_adder_0;
        executionRecord_decodeResult_logic <= enqueue_bits_decodeResult_logic_0;
        executionRecord_instructionIndex <= enqueue_bits_instructionIndex_0;
      end
      if (_executionRecordValid_T ^ recordDequeueFire)
        executionRecordValid <= _executionRecordValid_T;
      if (dataResponse_valid)
        executionResult <= lastSlotDataUpdate;
      if (_outStandingUpdate_T ^ dataResponse_valid)
        outStanding <= outStanding + outStandingUpdate;
    end
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:8];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [3:0] i = 4'h0; i < 4'h9; i += 4'h1) begin
          _RANDOM[i] = `RANDOM;
        end
        executionRecord_crossReadVS2 = _RANDOM[4'h0][0];
        executionRecord_bordersForMaskLogic = _RANDOM[4'h0][1];
        executionRecord_maskForMaskInput = _RANDOM[4'h0][5:2];
        executionRecord_maskForFilter = _RANDOM[4'h0][9:6];
        executionRecord_source_0 = {_RANDOM[4'h0][31:11], _RANDOM[4'h1][10:0]};
        executionRecord_source_1 = {_RANDOM[4'h1][31:11], _RANDOM[4'h2][10:0]};
        executionRecord_source_2 = {_RANDOM[4'h2][31:11], _RANDOM[4'h3][10:0]};
        executionRecord_groupCounter = _RANDOM[4'h3][23:11];
        executionRecord_vSew1H = _RANDOM[4'h3][26:24];
        executionRecord_csr_vl = {_RANDOM[4'h3][31:27], _RANDOM[4'h4][11:0]};
        executionRecord_csr_vStart = _RANDOM[4'h4][28:12];
        executionRecord_csr_vlmul = _RANDOM[4'h4][31:29];
        executionRecord_csr_vSew = _RANDOM[4'h5][1:0];
        executionRecord_csr_vxrm = _RANDOM[4'h5][3:2];
        executionRecord_csr_vta = _RANDOM[4'h5][4];
        executionRecord_csr_vma = _RANDOM[4'h5][5];
        executionRecord_maskType = _RANDOM[4'h5][6];
        executionRecord_laneIndex = _RANDOM[4'h5][8:7];
        executionRecord_decodeResult_specialSlot = _RANDOM[4'h5][9];
        executionRecord_decodeResult_topUop = _RANDOM[4'h5][14:10];
        executionRecord_decodeResult_popCount = _RANDOM[4'h5][15];
        executionRecord_decodeResult_ffo = _RANDOM[4'h5][16];
        executionRecord_decodeResult_average = _RANDOM[4'h5][17];
        executionRecord_decodeResult_reverse = _RANDOM[4'h5][18];
        executionRecord_decodeResult_dontNeedExecuteInLane = _RANDOM[4'h5][19];
        executionRecord_decodeResult_scheduler = _RANDOM[4'h5][20];
        executionRecord_decodeResult_sReadVD = _RANDOM[4'h5][21];
        executionRecord_decodeResult_vtype = _RANDOM[4'h5][22];
        executionRecord_decodeResult_sWrite = _RANDOM[4'h5][23];
        executionRecord_decodeResult_crossRead = _RANDOM[4'h5][24];
        executionRecord_decodeResult_crossWrite = _RANDOM[4'h5][25];
        executionRecord_decodeResult_maskUnit = _RANDOM[4'h5][26];
        executionRecord_decodeResult_special = _RANDOM[4'h5][27];
        executionRecord_decodeResult_saturate = _RANDOM[4'h5][28];
        executionRecord_decodeResult_vwmacc = _RANDOM[4'h5][29];
        executionRecord_decodeResult_readOnly = _RANDOM[4'h5][30];
        executionRecord_decodeResult_maskSource = _RANDOM[4'h5][31];
        executionRecord_decodeResult_maskDestination = _RANDOM[4'h6][0];
        executionRecord_decodeResult_maskLogic = _RANDOM[4'h6][1];
        executionRecord_decodeResult_uop = _RANDOM[4'h6][5:2];
        executionRecord_decodeResult_iota = _RANDOM[4'h6][6];
        executionRecord_decodeResult_mv = _RANDOM[4'h6][7];
        executionRecord_decodeResult_extend = _RANDOM[4'h6][8];
        executionRecord_decodeResult_unOrderWrite = _RANDOM[4'h6][9];
        executionRecord_decodeResult_compress = _RANDOM[4'h6][10];
        executionRecord_decodeResult_gather16 = _RANDOM[4'h6][11];
        executionRecord_decodeResult_gather = _RANDOM[4'h6][12];
        executionRecord_decodeResult_slid = _RANDOM[4'h6][13];
        executionRecord_decodeResult_targetRd = _RANDOM[4'h6][14];
        executionRecord_decodeResult_widenReduce = _RANDOM[4'h6][15];
        executionRecord_decodeResult_red = _RANDOM[4'h6][16];
        executionRecord_decodeResult_nr = _RANDOM[4'h6][17];
        executionRecord_decodeResult_itype = _RANDOM[4'h6][18];
        executionRecord_decodeResult_unsigned1 = _RANDOM[4'h6][19];
        executionRecord_decodeResult_unsigned0 = _RANDOM[4'h6][20];
        executionRecord_decodeResult_other = _RANDOM[4'h6][21];
        executionRecord_decodeResult_multiCycle = _RANDOM[4'h6][22];
        executionRecord_decodeResult_divider = _RANDOM[4'h6][23];
        executionRecord_decodeResult_multiplier = _RANDOM[4'h6][24];
        executionRecord_decodeResult_shift = _RANDOM[4'h6][25];
        executionRecord_decodeResult_adder = _RANDOM[4'h6][26];
        executionRecord_decodeResult_logic = _RANDOM[4'h6][27];
        executionRecord_instructionIndex = _RANDOM[4'h6][30:28];
        executionRecordValid = _RANDOM[4'h6][31];
        executionResult = _RANDOM[4'h7];
        outStanding = _RANDOM[4'h8][2:0];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  wire         recordQueue_empty;
  assign recordQueue_empty = _recordQueue_fifo_empty;
  wire         recordQueue_full;
  assign recordQueue_full = _recordQueue_fifo_full;
  wire         queue_empty;
  assign queue_empty = _queue_fifo_empty;
  wire         queue_full;
  assign queue_full = _queue_fifo_full;
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(5),
    .err_mode(2),
    .rst_mode(3),
    .width(108)
  ) recordQueue_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(recordQueue_enq_ready & recordQueue_enq_valid & ~(_recordQueue_fifo_empty & recordQueue_deq_ready))),
    .pop_req_n    (~(recordQueue_deq_ready & ~_recordQueue_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (recordQueue_dataIn),
    .empty        (_recordQueue_fifo_empty),
    .almost_empty (recordQueue_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (recordQueue_almostFull),
    .full         (_recordQueue_fifo_full),
    .error        (_recordQueue_fifo_error),
    .data_out     (_recordQueue_fifo_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(46)
  ) queue_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(queue_enq_ready & queue_enq_valid)),
    .pop_req_n    (~(queue_deq_ready & ~_queue_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (queue_dataIn),
    .empty        (_queue_fifo_empty),
    .almost_empty (queue_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (queue_almostFull),
    .full         (_queue_fifo_full),
    .error        (_queue_fifo_error),
    .data_out     (_queue_fifo_data_out)
  );
  assign enqueue_ready = enqueue_ready_0;
  assign dequeue_valid = dequeue_valid_0;
  assign dequeue_bits_data = dequeue_bits_data_0;
  assign dequeue_bits_ffoIndex = dequeue_bits_ffoIndex_0;
  assign vfuRequest_valid = vfuRequest_valid_0;
  assign vfuRequest_bits_src_0 = vfuRequest_bits_src_0_0;
  assign vfuRequest_bits_src_1 = vfuRequest_bits_src_1_0;
  assign vfuRequest_bits_src_2 = vfuRequest_bits_src_2_0;
  assign vfuRequest_bits_src_3 = vfuRequest_bits_src_3_0;
  assign vfuRequest_bits_opcode = vfuRequest_bits_opcode_0;
  assign vfuRequest_bits_mask = vfuRequest_bits_mask_0;
  assign vfuRequest_bits_executeMask = vfuRequest_bits_executeMask_0;
  assign vfuRequest_bits_sign0 = vfuRequest_bits_sign0_0;
  assign vfuRequest_bits_sign = vfuRequest_bits_sign_0;
  assign vfuRequest_bits_reverse = vfuRequest_bits_reverse_0;
  assign vfuRequest_bits_average = vfuRequest_bits_average_0;
  assign vfuRequest_bits_saturate = vfuRequest_bits_saturate_0;
  assign vfuRequest_bits_vxrm = vfuRequest_bits_vxrm_0;
  assign vfuRequest_bits_vSew = vfuRequest_bits_vSew_0;
  assign vfuRequest_bits_shifterSize = vfuRequest_bits_shifterSize_0;
  assign vfuRequest_bits_rem = vfuRequest_bits_rem_0;
  assign vfuRequest_bits_groupIndex = vfuRequest_bits_groupIndex_0;
  assign vfuRequest_bits_laneIndex = vfuRequest_bits_laneIndex_0;
  assign vfuRequest_bits_maskType = vfuRequest_bits_maskType_0;
  assign vfuRequest_bits_narrow = vfuRequest_bits_narrow_0;
  assign executeDecode_specialSlot = executionRecord_decodeResult_specialSlot;
  assign executeDecode_topUop = executionRecord_decodeResult_topUop;
  assign executeDecode_popCount = executionRecord_decodeResult_popCount;
  assign executeDecode_ffo = executionRecord_decodeResult_ffo;
  assign executeDecode_average = executionRecord_decodeResult_average;
  assign executeDecode_reverse = executionRecord_decodeResult_reverse;
  assign executeDecode_dontNeedExecuteInLane = executionRecord_decodeResult_dontNeedExecuteInLane;
  assign executeDecode_scheduler = executionRecord_decodeResult_scheduler;
  assign executeDecode_sReadVD = executionRecord_decodeResult_sReadVD;
  assign executeDecode_vtype = executionRecord_decodeResult_vtype;
  assign executeDecode_sWrite = executionRecord_decodeResult_sWrite;
  assign executeDecode_crossRead = executionRecord_decodeResult_crossRead;
  assign executeDecode_crossWrite = executionRecord_decodeResult_crossWrite;
  assign executeDecode_maskUnit = executionRecord_decodeResult_maskUnit;
  assign executeDecode_special = executionRecord_decodeResult_special;
  assign executeDecode_saturate = executionRecord_decodeResult_saturate;
  assign executeDecode_vwmacc = executionRecord_decodeResult_vwmacc;
  assign executeDecode_readOnly = executionRecord_decodeResult_readOnly;
  assign executeDecode_maskSource = executionRecord_decodeResult_maskSource;
  assign executeDecode_maskDestination = executionRecord_decodeResult_maskDestination;
  assign executeDecode_maskLogic = executionRecord_decodeResult_maskLogic;
  assign executeDecode_uop = executionRecord_decodeResult_uop;
  assign executeDecode_iota = executionRecord_decodeResult_iota;
  assign executeDecode_mv = executionRecord_decodeResult_mv;
  assign executeDecode_extend = executionRecord_decodeResult_extend;
  assign executeDecode_unOrderWrite = executionRecord_decodeResult_unOrderWrite;
  assign executeDecode_compress = executionRecord_decodeResult_compress;
  assign executeDecode_gather16 = executionRecord_decodeResult_gather16;
  assign executeDecode_gather = executionRecord_decodeResult_gather;
  assign executeDecode_slid = executionRecord_decodeResult_slid;
  assign executeDecode_targetRd = executionRecord_decodeResult_targetRd;
  assign executeDecode_widenReduce = executionRecord_decodeResult_widenReduce;
  assign executeDecode_red = executionRecord_decodeResult_red;
  assign executeDecode_nr = executionRecord_decodeResult_nr;
  assign executeDecode_itype = executionRecord_decodeResult_itype;
  assign executeDecode_unsigned1 = executionRecord_decodeResult_unsigned1;
  assign executeDecode_unsigned0 = executionRecord_decodeResult_unsigned0;
  assign executeDecode_other = executionRecord_decodeResult_other;
  assign executeDecode_multiCycle = executionRecord_decodeResult_multiCycle;
  assign executeDecode_divider = executionRecord_decodeResult_divider;
  assign executeDecode_multiplier = executionRecord_decodeResult_multiplier;
  assign executeDecode_shift = executionRecord_decodeResult_shift;
  assign executeDecode_adder = executionRecord_decodeResult_adder;
  assign executeDecode_logic = executionRecord_decodeResult_logic;
  assign responseDecode_specialSlot = recordQueue_deq_bits_decodeResult_specialSlot;
  assign responseDecode_topUop = recordQueue_deq_bits_decodeResult_topUop;
  assign responseDecode_popCount = recordQueue_deq_bits_decodeResult_popCount;
  assign responseDecode_ffo = recordQueue_deq_bits_decodeResult_ffo;
  assign responseDecode_average = recordQueue_deq_bits_decodeResult_average;
  assign responseDecode_reverse = recordQueue_deq_bits_decodeResult_reverse;
  assign responseDecode_dontNeedExecuteInLane = recordQueue_deq_bits_decodeResult_dontNeedExecuteInLane;
  assign responseDecode_scheduler = recordQueue_deq_bits_decodeResult_scheduler;
  assign responseDecode_sReadVD = recordQueue_deq_bits_decodeResult_sReadVD;
  assign responseDecode_vtype = recordQueue_deq_bits_decodeResult_vtype;
  assign responseDecode_sWrite = recordQueue_deq_bits_decodeResult_sWrite;
  assign responseDecode_crossRead = recordQueue_deq_bits_decodeResult_crossRead;
  assign responseDecode_crossWrite = recordQueue_deq_bits_decodeResult_crossWrite;
  assign responseDecode_maskUnit = recordQueue_deq_bits_decodeResult_maskUnit;
  assign responseDecode_special = recordQueue_deq_bits_decodeResult_special;
  assign responseDecode_saturate = recordQueue_deq_bits_decodeResult_saturate;
  assign responseDecode_vwmacc = recordQueue_deq_bits_decodeResult_vwmacc;
  assign responseDecode_readOnly = recordQueue_deq_bits_decodeResult_readOnly;
  assign responseDecode_maskSource = recordQueue_deq_bits_decodeResult_maskSource;
  assign responseDecode_maskDestination = recordQueue_deq_bits_decodeResult_maskDestination;
  assign responseDecode_maskLogic = recordQueue_deq_bits_decodeResult_maskLogic;
  assign responseDecode_uop = recordQueue_deq_bits_decodeResult_uop;
  assign responseDecode_iota = recordQueue_deq_bits_decodeResult_iota;
  assign responseDecode_mv = recordQueue_deq_bits_decodeResult_mv;
  assign responseDecode_extend = recordQueue_deq_bits_decodeResult_extend;
  assign responseDecode_unOrderWrite = recordQueue_deq_bits_decodeResult_unOrderWrite;
  assign responseDecode_compress = recordQueue_deq_bits_decodeResult_compress;
  assign responseDecode_gather16 = recordQueue_deq_bits_decodeResult_gather16;
  assign responseDecode_gather = recordQueue_deq_bits_decodeResult_gather;
  assign responseDecode_slid = recordQueue_deq_bits_decodeResult_slid;
  assign responseDecode_targetRd = recordQueue_deq_bits_decodeResult_targetRd;
  assign responseDecode_widenReduce = recordQueue_deq_bits_decodeResult_widenReduce;
  assign responseDecode_red = recordQueue_deq_bits_decodeResult_red;
  assign responseDecode_nr = recordQueue_deq_bits_decodeResult_nr;
  assign responseDecode_itype = recordQueue_deq_bits_decodeResult_itype;
  assign responseDecode_unsigned1 = recordQueue_deq_bits_decodeResult_unsigned1;
  assign responseDecode_unsigned0 = recordQueue_deq_bits_decodeResult_unsigned0;
  assign responseDecode_other = recordQueue_deq_bits_decodeResult_other;
  assign responseDecode_multiCycle = recordQueue_deq_bits_decodeResult_multiCycle;
  assign responseDecode_divider = recordQueue_deq_bits_decodeResult_divider;
  assign responseDecode_multiplier = recordQueue_deq_bits_decodeResult_multiplier;
  assign responseDecode_shift = recordQueue_deq_bits_decodeResult_shift;
  assign responseDecode_adder = recordQueue_deq_bits_decodeResult_adder;
  assign responseDecode_logic = recordQueue_deq_bits_decodeResult_logic;
  assign responseIndex = recordQueue_deq_bits_instructionIndex;
endmodule

