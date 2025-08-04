
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
module LaneStage1_1(
  input         clock,
                reset,
  output        enqueue_ready,
  input         enqueue_valid,
  input  [4:0]  enqueue_bits_groupCounter,
  input  [3:0]  enqueue_bits_maskForMaskInput,
                enqueue_bits_boundaryMaskCorrection,
  input  [2:0]  enqueue_bits_instructionIndex,
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
  input  [1:0]  enqueue_bits_laneIndex,
  input         enqueue_bits_skipRead,
  input  [4:0]  enqueue_bits_vs1,
                enqueue_bits_vs2,
                enqueue_bits_vd,
  input  [2:0]  enqueue_bits_vSew1H,
  input         enqueue_bits_maskNotMaskedElement,
  input  [8:0]  enqueue_bits_csr_vl,
                enqueue_bits_csr_vStart,
  input  [2:0]  enqueue_bits_csr_vlmul,
  input  [1:0]  enqueue_bits_csr_vSew,
                enqueue_bits_csr_vxrm,
  input         enqueue_bits_csr_vta,
                enqueue_bits_csr_vma,
                enqueue_bits_maskType,
                enqueue_bits_loadStore,
  input  [31:0] enqueue_bits_readFromScalar,
  input         enqueue_bits_bordersForMaskLogic,
                dequeue_ready,
  output        dequeue_valid,
  output [3:0]  dequeue_bits_maskForFilter,
                dequeue_bits_mask,
  output [4:0]  dequeue_bits_groupCounter,
  output [31:0] dequeue_bits_src_0,
                dequeue_bits_src_1,
                dequeue_bits_src_2,
  output        dequeue_bits_decodeResult_specialSlot,
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
  output [2:0]  dequeue_bits_vSew1H,
  output [8:0]  dequeue_bits_csr_vl,
                dequeue_bits_csr_vStart,
  output [2:0]  dequeue_bits_csr_vlmul,
  output [1:0]  dequeue_bits_csr_vSew,
                dequeue_bits_csr_vxrm,
  output        dequeue_bits_csr_vta,
                dequeue_bits_csr_vma,
                dequeue_bits_maskType,
  output [1:0]  dequeue_bits_laneIndex,
  output [2:0]  dequeue_bits_instructionIndex,
  output        dequeue_bits_loadStore,
  output [4:0]  dequeue_bits_vd,
  output        dequeue_bits_bordersForMaskLogic,
  input         vrfReadRequest_0_ready,
  output        vrfReadRequest_0_valid,
  output [4:0]  vrfReadRequest_0_bits_vs,
  output [1:0]  vrfReadRequest_0_bits_readSource,
  output        vrfReadRequest_0_bits_offset,
  output [2:0]  vrfReadRequest_0_bits_instructionIndex,
  input         vrfReadRequest_1_ready,
  output        vrfReadRequest_1_valid,
  output [4:0]  vrfReadRequest_1_bits_vs,
  output [1:0]  vrfReadRequest_1_bits_readSource,
  output        vrfReadRequest_1_bits_offset,
  output [2:0]  vrfReadRequest_1_bits_instructionIndex,
  input         vrfReadRequest_2_ready,
  output        vrfReadRequest_2_valid,
  output [4:0]  vrfReadRequest_2_bits_vs,
  output [1:0]  vrfReadRequest_2_bits_readSource,
  output        vrfReadRequest_2_bits_offset,
  output [2:0]  vrfReadRequest_2_bits_instructionIndex,
  output [4:0]  vrfCheckRequest_0_vs,
  output [1:0]  vrfCheckRequest_0_readSource,
  output        vrfCheckRequest_0_offset,
  output [2:0]  vrfCheckRequest_0_instructionIndex,
  output [4:0]  vrfCheckRequest_1_vs,
  output [1:0]  vrfCheckRequest_1_readSource,
  output        vrfCheckRequest_1_offset,
  output [2:0]  vrfCheckRequest_1_instructionIndex,
  output [4:0]  vrfCheckRequest_2_vs,
  output [1:0]  vrfCheckRequest_2_readSource,
  output        vrfCheckRequest_2_offset,
  output [2:0]  vrfCheckRequest_2_instructionIndex,
  input         checkResult_0,
                checkResult_1,
                checkResult_2,
  input  [31:0] vrfReadResult_0,
                vrfReadResult_1,
                vrfReadResult_2
);

  wire         _dataQueueVd_fifo_empty;
  wire         _dataQueueVd_fifo_full;
  wire         _dataQueueVd_fifo_error;
  wire         _dataQueueVs2_fifo_empty;
  wire         _dataQueueVs2_fifo_full;
  wire         _dataQueueVs2_fifo_error;
  wire         _dataQueueVs1_io_fifo_empty;
  wire         _dataQueueVs1_io_fifo_full;
  wire         _dataQueueVs1_io_fifo_error;
  wire         _readPipe2_enqueue_ready;
  wire         _readPipe1_enqueue_ready;
  wire         _pipeQueue_fifo_empty;
  wire         _pipeQueue_fifo_full;
  wire         _pipeQueue_fifo_error;
  wire [150:0] _pipeQueue_fifo_data_out;
  wire         _queueBeforeCheckVd_fifo_empty;
  wire         _queueBeforeCheckVd_fifo_full;
  wire         _queueBeforeCheckVd_fifo_error;
  wire [16:0]  _queueBeforeCheckVd_fifo_data_out;
  wire         _queueBeforeCheck2_fifo_empty;
  wire         _queueBeforeCheck2_fifo_full;
  wire         _queueBeforeCheck2_fifo_error;
  wire [16:0]  _queueBeforeCheck2_fifo_data_out;
  wire         _queueBeforeCheck1_fifo_empty;
  wire         _queueBeforeCheck1_fifo_full;
  wire         _queueBeforeCheck1_fifo_error;
  wire [16:0]  _queueBeforeCheck1_fifo_data_out;
  wire         _queueAfterCheckVd_fifo_empty;
  wire         _queueAfterCheckVd_fifo_full;
  wire         _queueAfterCheckVd_fifo_error;
  wire [16:0]  _queueAfterCheckVd_fifo_data_out;
  wire         _queueAfterCheck2_fifo_empty;
  wire         _queueAfterCheck2_fifo_full;
  wire         _queueAfterCheck2_fifo_error;
  wire [16:0]  _queueAfterCheck2_fifo_data_out;
  wire         _queueAfterCheck1_fifo_empty;
  wire         _queueAfterCheck1_fifo_full;
  wire         _queueAfterCheck1_fifo_error;
  wire [16:0]  _queueAfterCheck1_fifo_data_out;
  wire         dataQueueVd_almostFull;
  wire         dataQueueVd_almostEmpty;
  wire         dataQueueVs2_almostFull;
  wire         dataQueueVs2_almostEmpty;
  wire         dataQueueVs1_io_almostFull;
  wire         dataQueueVs1_io_almostEmpty;
  wire [31:0]  dataQueueVd_enq_bits;
  wire [2:0]   vrfReadRequest_2_bits_instructionIndex_0;
  wire         vrfReadRequest_2_bits_offset_0;
  wire [1:0]   vrfReadRequest_2_bits_readSource_0;
  wire [4:0]   vrfReadRequest_2_bits_vs_0;
  wire         vrfReadRequest_2_valid_0;
  wire [31:0]  dataQueueVs2_enq_bits;
  wire [2:0]   vrfReadRequest_1_bits_instructionIndex_0;
  wire         vrfReadRequest_1_bits_offset_0;
  wire [1:0]   vrfReadRequest_1_bits_readSource_0;
  wire [4:0]   vrfReadRequest_1_bits_vs_0;
  wire         vrfReadRequest_1_valid_0;
  wire [31:0]  dataQueueVs1_io_enq_bits;
  wire [2:0]   vrfReadRequest_0_bits_instructionIndex_0;
  wire         vrfReadRequest_0_bits_offset_0;
  wire [1:0]   vrfReadRequest_0_bits_readSource_0;
  wire [4:0]   vrfReadRequest_0_bits_vs_0;
  wire         vrfReadRequest_0_valid_0;
  wire         pipeQueue_almostFull;
  wire         pipeQueue_almostEmpty;
  wire         queueBeforeCheckVd_almostFull;
  wire         queueBeforeCheckVd_almostEmpty;
  wire         queueBeforeCheck2_almostFull;
  wire         queueBeforeCheck2_almostEmpty;
  wire         queueBeforeCheck1_almostFull;
  wire         queueBeforeCheck1_almostEmpty;
  wire         queueAfterCheckVd_almostFull;
  wire         queueAfterCheckVd_almostEmpty;
  wire         queueAfterCheck2_almostFull;
  wire         queueAfterCheck2_almostEmpty;
  wire         queueAfterCheck1_almostFull;
  wire         queueAfterCheck1_almostEmpty;
  wire         enqueue_valid_0 = enqueue_valid;
  wire [4:0]   enqueue_bits_groupCounter_0 = enqueue_bits_groupCounter;
  wire [3:0]   enqueue_bits_maskForMaskInput_0 = enqueue_bits_maskForMaskInput;
  wire [3:0]   enqueue_bits_boundaryMaskCorrection_0 = enqueue_bits_boundaryMaskCorrection;
  wire [2:0]   enqueue_bits_instructionIndex_0 = enqueue_bits_instructionIndex;
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
  wire [1:0]   enqueue_bits_laneIndex_0 = enqueue_bits_laneIndex;
  wire         enqueue_bits_skipRead_0 = enqueue_bits_skipRead;
  wire [4:0]   enqueue_bits_vs1_0 = enqueue_bits_vs1;
  wire [4:0]   enqueue_bits_vs2_0 = enqueue_bits_vs2;
  wire [4:0]   enqueue_bits_vd_0 = enqueue_bits_vd;
  wire [2:0]   enqueue_bits_vSew1H_0 = enqueue_bits_vSew1H;
  wire         enqueue_bits_maskNotMaskedElement_0 = enqueue_bits_maskNotMaskedElement;
  wire [8:0]   enqueue_bits_csr_vl_0 = enqueue_bits_csr_vl;
  wire [8:0]   enqueue_bits_csr_vStart_0 = enqueue_bits_csr_vStart;
  wire [2:0]   enqueue_bits_csr_vlmul_0 = enqueue_bits_csr_vlmul;
  wire [1:0]   enqueue_bits_csr_vSew_0 = enqueue_bits_csr_vSew;
  wire [1:0]   enqueue_bits_csr_vxrm_0 = enqueue_bits_csr_vxrm;
  wire         enqueue_bits_csr_vta_0 = enqueue_bits_csr_vta;
  wire         enqueue_bits_csr_vma_0 = enqueue_bits_csr_vma;
  wire         enqueue_bits_maskType_0 = enqueue_bits_maskType;
  wire         enqueue_bits_loadStore_0 = enqueue_bits_loadStore;
  wire [31:0]  enqueue_bits_readFromScalar_0 = enqueue_bits_readFromScalar;
  wire         enqueue_bits_bordersForMaskLogic_0 = enqueue_bits_bordersForMaskLogic;
  wire         dequeue_ready_0 = dequeue_ready;
  wire         vrfReadRequest_0_ready_0 = vrfReadRequest_0_ready;
  wire         vrfReadRequest_1_ready_0 = vrfReadRequest_1_ready;
  wire         vrfReadRequest_2_ready_0 = vrfReadRequest_2_ready;
  wire [3:0]   queueBeforeCheckVd_enq_bits_readSource = 4'h2;
  wire [3:0]   queueBeforeCheck2_enq_bits_readSource = 4'h1;
  wire [4:0]   pipeQueue_enq_bits_groupCounter = enqueue_bits_groupCounter_0;
  wire [3:0]   pipeQueue_enq_bits_maskForMaskInput = enqueue_bits_maskForMaskInput_0;
  wire [3:0]   pipeQueue_enq_bits_boundaryMaskCorrection = enqueue_bits_boundaryMaskCorrection_0;
  wire [2:0]   queueBeforeCheck1_enq_bits_instructionIndex = enqueue_bits_instructionIndex_0;
  wire [2:0]   queueBeforeCheck2_enq_bits_instructionIndex = enqueue_bits_instructionIndex_0;
  wire [2:0]   queueBeforeCheckVd_enq_bits_instructionIndex = enqueue_bits_instructionIndex_0;
  wire [2:0]   pipeQueue_enq_bits_instructionIndex = enqueue_bits_instructionIndex_0;
  wire         pipeQueue_enq_bits_decodeResult_specialSlot = enqueue_bits_decodeResult_specialSlot_0;
  wire [4:0]   pipeQueue_enq_bits_decodeResult_topUop = enqueue_bits_decodeResult_topUop_0;
  wire         pipeQueue_enq_bits_decodeResult_popCount = enqueue_bits_decodeResult_popCount_0;
  wire         pipeQueue_enq_bits_decodeResult_ffo = enqueue_bits_decodeResult_ffo_0;
  wire         pipeQueue_enq_bits_decodeResult_average = enqueue_bits_decodeResult_average_0;
  wire         pipeQueue_enq_bits_decodeResult_reverse = enqueue_bits_decodeResult_reverse_0;
  wire         pipeQueue_enq_bits_decodeResult_dontNeedExecuteInLane = enqueue_bits_decodeResult_dontNeedExecuteInLane_0;
  wire         pipeQueue_enq_bits_decodeResult_scheduler = enqueue_bits_decodeResult_scheduler_0;
  wire         pipeQueue_enq_bits_decodeResult_sReadVD = enqueue_bits_decodeResult_sReadVD_0;
  wire         pipeQueue_enq_bits_decodeResult_vtype = enqueue_bits_decodeResult_vtype_0;
  wire         pipeQueue_enq_bits_decodeResult_sWrite = enqueue_bits_decodeResult_sWrite_0;
  wire         pipeQueue_enq_bits_decodeResult_crossRead = enqueue_bits_decodeResult_crossRead_0;
  wire         pipeQueue_enq_bits_decodeResult_crossWrite = enqueue_bits_decodeResult_crossWrite_0;
  wire         pipeQueue_enq_bits_decodeResult_maskUnit = enqueue_bits_decodeResult_maskUnit_0;
  wire         pipeQueue_enq_bits_decodeResult_special = enqueue_bits_decodeResult_special_0;
  wire         pipeQueue_enq_bits_decodeResult_saturate = enqueue_bits_decodeResult_saturate_0;
  wire         pipeQueue_enq_bits_decodeResult_vwmacc = enqueue_bits_decodeResult_vwmacc_0;
  wire         pipeQueue_enq_bits_decodeResult_readOnly = enqueue_bits_decodeResult_readOnly_0;
  wire         pipeQueue_enq_bits_decodeResult_maskSource = enqueue_bits_decodeResult_maskSource_0;
  wire         pipeQueue_enq_bits_decodeResult_maskDestination = enqueue_bits_decodeResult_maskDestination_0;
  wire         pipeQueue_enq_bits_decodeResult_maskLogic = enqueue_bits_decodeResult_maskLogic_0;
  wire [3:0]   pipeQueue_enq_bits_decodeResult_uop = enqueue_bits_decodeResult_uop_0;
  wire         pipeQueue_enq_bits_decodeResult_iota = enqueue_bits_decodeResult_iota_0;
  wire         pipeQueue_enq_bits_decodeResult_mv = enqueue_bits_decodeResult_mv_0;
  wire         pipeQueue_enq_bits_decodeResult_extend = enqueue_bits_decodeResult_extend_0;
  wire         pipeQueue_enq_bits_decodeResult_unOrderWrite = enqueue_bits_decodeResult_unOrderWrite_0;
  wire         pipeQueue_enq_bits_decodeResult_compress = enqueue_bits_decodeResult_compress_0;
  wire         pipeQueue_enq_bits_decodeResult_gather16 = enqueue_bits_decodeResult_gather16_0;
  wire         pipeQueue_enq_bits_decodeResult_gather = enqueue_bits_decodeResult_gather_0;
  wire         pipeQueue_enq_bits_decodeResult_slid = enqueue_bits_decodeResult_slid_0;
  wire         pipeQueue_enq_bits_decodeResult_targetRd = enqueue_bits_decodeResult_targetRd_0;
  wire         pipeQueue_enq_bits_decodeResult_widenReduce = enqueue_bits_decodeResult_widenReduce_0;
  wire         pipeQueue_enq_bits_decodeResult_red = enqueue_bits_decodeResult_red_0;
  wire         pipeQueue_enq_bits_decodeResult_nr = enqueue_bits_decodeResult_nr_0;
  wire         pipeQueue_enq_bits_decodeResult_itype = enqueue_bits_decodeResult_itype_0;
  wire         pipeQueue_enq_bits_decodeResult_unsigned1 = enqueue_bits_decodeResult_unsigned1_0;
  wire         pipeQueue_enq_bits_decodeResult_unsigned0 = enqueue_bits_decodeResult_unsigned0_0;
  wire         pipeQueue_enq_bits_decodeResult_other = enqueue_bits_decodeResult_other_0;
  wire         pipeQueue_enq_bits_decodeResult_multiCycle = enqueue_bits_decodeResult_multiCycle_0;
  wire         pipeQueue_enq_bits_decodeResult_divider = enqueue_bits_decodeResult_divider_0;
  wire         pipeQueue_enq_bits_decodeResult_multiplier = enqueue_bits_decodeResult_multiplier_0;
  wire         pipeQueue_enq_bits_decodeResult_shift = enqueue_bits_decodeResult_shift_0;
  wire         pipeQueue_enq_bits_decodeResult_adder = enqueue_bits_decodeResult_adder_0;
  wire         pipeQueue_enq_bits_decodeResult_logic = enqueue_bits_decodeResult_logic_0;
  wire [1:0]   pipeQueue_enq_bits_laneIndex = enqueue_bits_laneIndex_0;
  wire         pipeQueue_enq_bits_skipRead = enqueue_bits_skipRead_0;
  wire [4:0]   pipeQueue_enq_bits_vs1 = enqueue_bits_vs1_0;
  wire [4:0]   pipeQueue_enq_bits_vs2 = enqueue_bits_vs2_0;
  wire [4:0]   pipeQueue_enq_bits_vd = enqueue_bits_vd_0;
  wire [2:0]   pipeQueue_enq_bits_vSew1H = enqueue_bits_vSew1H_0;
  wire         pipeQueue_enq_bits_maskNotMaskedElement = enqueue_bits_maskNotMaskedElement_0;
  wire [8:0]   pipeQueue_enq_bits_csr_vl = enqueue_bits_csr_vl_0;
  wire [8:0]   pipeQueue_enq_bits_csr_vStart = enqueue_bits_csr_vStart_0;
  wire [2:0]   pipeQueue_enq_bits_csr_vlmul = enqueue_bits_csr_vlmul_0;
  wire [1:0]   pipeQueue_enq_bits_csr_vSew = enqueue_bits_csr_vSew_0;
  wire [1:0]   pipeQueue_enq_bits_csr_vxrm = enqueue_bits_csr_vxrm_0;
  wire         pipeQueue_enq_bits_csr_vta = enqueue_bits_csr_vta_0;
  wire         pipeQueue_enq_bits_csr_vma = enqueue_bits_csr_vma_0;
  wire         pipeQueue_enq_bits_maskType = enqueue_bits_maskType_0;
  wire         pipeQueue_enq_bits_loadStore = enqueue_bits_loadStore_0;
  wire [31:0]  pipeQueue_enq_bits_readFromScalar = enqueue_bits_readFromScalar_0;
  wire         pipeQueue_enq_bits_bordersForMaskLogic = enqueue_bits_bordersForMaskLogic_0;
  wire [3:0]   pipeQueue_deq_bits_maskForMaskInput;
  wire [4:0]   pipeQueue_deq_bits_groupCounter;
  wire [31:0]  source1Select;
  wire [31:0]  dataQueueVs2_deq_bits;
  wire [31:0]  dataQueueVd_deq_bits;
  wire         pipeQueue_deq_bits_decodeResult_specialSlot;
  wire [4:0]   pipeQueue_deq_bits_decodeResult_topUop;
  wire         pipeQueue_deq_bits_decodeResult_popCount;
  wire         pipeQueue_deq_bits_decodeResult_ffo;
  wire         pipeQueue_deq_bits_decodeResult_average;
  wire         pipeQueue_deq_bits_decodeResult_reverse;
  wire         pipeQueue_deq_bits_decodeResult_dontNeedExecuteInLane;
  wire         pipeQueue_deq_bits_decodeResult_scheduler;
  wire         pipeQueue_deq_bits_decodeResult_sReadVD;
  wire         pipeQueue_deq_bits_decodeResult_vtype;
  wire         pipeQueue_deq_bits_decodeResult_sWrite;
  wire         pipeQueue_deq_bits_decodeResult_crossRead;
  wire         pipeQueue_deq_bits_decodeResult_crossWrite;
  wire         pipeQueue_deq_bits_decodeResult_maskUnit;
  wire         pipeQueue_deq_bits_decodeResult_special;
  wire         pipeQueue_deq_bits_decodeResult_saturate;
  wire         pipeQueue_deq_bits_decodeResult_vwmacc;
  wire         pipeQueue_deq_bits_decodeResult_readOnly;
  wire         pipeQueue_deq_bits_decodeResult_maskSource;
  wire         pipeQueue_deq_bits_decodeResult_maskDestination;
  wire         pipeQueue_deq_bits_decodeResult_maskLogic;
  wire [3:0]   pipeQueue_deq_bits_decodeResult_uop;
  wire         pipeQueue_deq_bits_decodeResult_iota;
  wire         pipeQueue_deq_bits_decodeResult_mv;
  wire         pipeQueue_deq_bits_decodeResult_extend;
  wire         pipeQueue_deq_bits_decodeResult_unOrderWrite;
  wire         pipeQueue_deq_bits_decodeResult_compress;
  wire         pipeQueue_deq_bits_decodeResult_gather16;
  wire         pipeQueue_deq_bits_decodeResult_gather;
  wire         pipeQueue_deq_bits_decodeResult_slid;
  wire         pipeQueue_deq_bits_decodeResult_targetRd;
  wire         pipeQueue_deq_bits_decodeResult_widenReduce;
  wire         pipeQueue_deq_bits_decodeResult_red;
  wire         pipeQueue_deq_bits_decodeResult_nr;
  wire         pipeQueue_deq_bits_decodeResult_itype;
  wire         pipeQueue_deq_bits_decodeResult_unsigned1;
  wire         pipeQueue_deq_bits_decodeResult_unsigned0;
  wire         pipeQueue_deq_bits_decodeResult_other;
  wire         pipeQueue_deq_bits_decodeResult_multiCycle;
  wire         pipeQueue_deq_bits_decodeResult_divider;
  wire         pipeQueue_deq_bits_decodeResult_multiplier;
  wire         pipeQueue_deq_bits_decodeResult_shift;
  wire         pipeQueue_deq_bits_decodeResult_adder;
  wire         pipeQueue_deq_bits_decodeResult_logic;
  wire [2:0]   pipeQueue_deq_bits_vSew1H;
  wire [8:0]   pipeQueue_deq_bits_csr_vl;
  wire [8:0]   pipeQueue_deq_bits_csr_vStart;
  wire [2:0]   pipeQueue_deq_bits_csr_vlmul;
  wire [1:0]   pipeQueue_deq_bits_csr_vSew;
  wire [1:0]   pipeQueue_deq_bits_csr_vxrm;
  wire         pipeQueue_deq_bits_csr_vta;
  wire         pipeQueue_deq_bits_csr_vma;
  wire         pipeQueue_deq_bits_maskType;
  wire [1:0]   pipeQueue_deq_bits_laneIndex;
  wire [2:0]   pipeQueue_deq_bits_instructionIndex;
  wire         pipeQueue_deq_bits_loadStore;
  wire [4:0]   pipeQueue_deq_bits_vd;
  wire         pipeQueue_deq_bits_bordersForMaskLogic;
  wire [4:0]   queueBeforeCheck1_deq_bits_vs;
  wire         queueBeforeCheck1_deq_bits_offset;
  wire [3:0]   queueBeforeCheck1_deq_bits_groupIndex;
  wire [3:0]   queueBeforeCheck1_deq_bits_readSource;
  wire [2:0]   queueBeforeCheck1_deq_bits_instructionIndex;
  wire         queueAfterCheck1_deq_valid;
  assign queueAfterCheck1_deq_valid = ~_queueAfterCheck1_fifo_empty;
  wire [4:0]   queueAfterCheck1_dataOut_vs;
  wire         queueAfterCheck1_dataOut_offset;
  wire [3:0]   queueAfterCheck1_dataOut_groupIndex;
  wire [3:0]   queueAfterCheck1_dataOut_readSource;
  wire [2:0]   queueAfterCheck1_dataOut_instructionIndex;
  wire [3:0]   queueAfterCheck1_enq_bits_readSource;
  wire [2:0]   queueAfterCheck1_enq_bits_instructionIndex;
  wire [6:0]   queueAfterCheck1_dataIn_lo = {queueAfterCheck1_enq_bits_readSource, queueAfterCheck1_enq_bits_instructionIndex};
  wire [4:0]   queueAfterCheck1_enq_bits_vs;
  wire         queueAfterCheck1_enq_bits_offset;
  wire [5:0]   queueAfterCheck1_dataIn_hi_hi = {queueAfterCheck1_enq_bits_vs, queueAfterCheck1_enq_bits_offset};
  wire [3:0]   queueAfterCheck1_enq_bits_groupIndex;
  wire [9:0]   queueAfterCheck1_dataIn_hi = {queueAfterCheck1_dataIn_hi_hi, queueAfterCheck1_enq_bits_groupIndex};
  wire [16:0]  queueAfterCheck1_dataIn = {queueAfterCheck1_dataIn_hi, queueAfterCheck1_dataIn_lo};
  assign queueAfterCheck1_dataOut_instructionIndex = _queueAfterCheck1_fifo_data_out[2:0];
  assign queueAfterCheck1_dataOut_readSource = _queueAfterCheck1_fifo_data_out[6:3];
  assign queueAfterCheck1_dataOut_groupIndex = _queueAfterCheck1_fifo_data_out[10:7];
  assign queueAfterCheck1_dataOut_offset = _queueAfterCheck1_fifo_data_out[11];
  assign queueAfterCheck1_dataOut_vs = _queueAfterCheck1_fifo_data_out[16:12];
  wire [4:0]   queueAfterCheck1_deq_bits_vs = queueAfterCheck1_dataOut_vs;
  wire         queueAfterCheck1_deq_bits_offset = queueAfterCheck1_dataOut_offset;
  wire [3:0]   queueAfterCheck1_deq_bits_groupIndex = queueAfterCheck1_dataOut_groupIndex;
  wire [3:0]   queueAfterCheck1_deq_bits_readSource = queueAfterCheck1_dataOut_readSource;
  wire [2:0]   queueAfterCheck1_deq_bits_instructionIndex = queueAfterCheck1_dataOut_instructionIndex;
  wire         queueAfterCheck1_enq_ready = ~_queueAfterCheck1_fifo_full;
  wire         queueAfterCheck1_enq_valid;
  wire         queueAfterCheck1_deq_ready;
  wire [4:0]   queueBeforeCheck2_deq_bits_vs;
  wire         queueBeforeCheck2_deq_bits_offset;
  wire [3:0]   queueBeforeCheck2_deq_bits_groupIndex;
  wire [3:0]   queueBeforeCheck2_deq_bits_readSource;
  wire [2:0]   queueBeforeCheck2_deq_bits_instructionIndex;
  wire         queueAfterCheck2_deq_valid;
  assign queueAfterCheck2_deq_valid = ~_queueAfterCheck2_fifo_empty;
  wire [4:0]   queueAfterCheck2_dataOut_vs;
  wire         queueAfterCheck2_dataOut_offset;
  wire [3:0]   queueAfterCheck2_dataOut_groupIndex;
  wire [3:0]   queueAfterCheck2_dataOut_readSource;
  wire [2:0]   queueAfterCheck2_dataOut_instructionIndex;
  wire [3:0]   queueAfterCheck2_enq_bits_readSource;
  wire [2:0]   queueAfterCheck2_enq_bits_instructionIndex;
  wire [6:0]   queueAfterCheck2_dataIn_lo = {queueAfterCheck2_enq_bits_readSource, queueAfterCheck2_enq_bits_instructionIndex};
  wire [4:0]   queueAfterCheck2_enq_bits_vs;
  wire         queueAfterCheck2_enq_bits_offset;
  wire [5:0]   queueAfterCheck2_dataIn_hi_hi = {queueAfterCheck2_enq_bits_vs, queueAfterCheck2_enq_bits_offset};
  wire [3:0]   queueAfterCheck2_enq_bits_groupIndex;
  wire [9:0]   queueAfterCheck2_dataIn_hi = {queueAfterCheck2_dataIn_hi_hi, queueAfterCheck2_enq_bits_groupIndex};
  wire [16:0]  queueAfterCheck2_dataIn = {queueAfterCheck2_dataIn_hi, queueAfterCheck2_dataIn_lo};
  assign queueAfterCheck2_dataOut_instructionIndex = _queueAfterCheck2_fifo_data_out[2:0];
  assign queueAfterCheck2_dataOut_readSource = _queueAfterCheck2_fifo_data_out[6:3];
  assign queueAfterCheck2_dataOut_groupIndex = _queueAfterCheck2_fifo_data_out[10:7];
  assign queueAfterCheck2_dataOut_offset = _queueAfterCheck2_fifo_data_out[11];
  assign queueAfterCheck2_dataOut_vs = _queueAfterCheck2_fifo_data_out[16:12];
  wire [4:0]   queueAfterCheck2_deq_bits_vs = queueAfterCheck2_dataOut_vs;
  wire         queueAfterCheck2_deq_bits_offset = queueAfterCheck2_dataOut_offset;
  wire [3:0]   queueAfterCheck2_deq_bits_groupIndex = queueAfterCheck2_dataOut_groupIndex;
  wire [3:0]   queueAfterCheck2_deq_bits_readSource = queueAfterCheck2_dataOut_readSource;
  wire [2:0]   queueAfterCheck2_deq_bits_instructionIndex = queueAfterCheck2_dataOut_instructionIndex;
  wire         queueAfterCheck2_enq_ready = ~_queueAfterCheck2_fifo_full;
  wire         queueAfterCheck2_enq_valid;
  wire         queueAfterCheck2_deq_ready;
  wire [4:0]   queueBeforeCheckVd_deq_bits_vs;
  wire         queueBeforeCheckVd_deq_bits_offset;
  wire [3:0]   queueBeforeCheckVd_deq_bits_groupIndex;
  wire [3:0]   queueBeforeCheckVd_deq_bits_readSource;
  wire [2:0]   queueBeforeCheckVd_deq_bits_instructionIndex;
  wire         queueAfterCheckVd_deq_valid;
  assign queueAfterCheckVd_deq_valid = ~_queueAfterCheckVd_fifo_empty;
  wire [4:0]   queueAfterCheckVd_dataOut_vs;
  wire         queueAfterCheckVd_dataOut_offset;
  wire [3:0]   queueAfterCheckVd_dataOut_groupIndex;
  wire [3:0]   queueAfterCheckVd_dataOut_readSource;
  wire [2:0]   queueAfterCheckVd_dataOut_instructionIndex;
  wire [3:0]   queueAfterCheckVd_enq_bits_readSource;
  wire [2:0]   queueAfterCheckVd_enq_bits_instructionIndex;
  wire [6:0]   queueAfterCheckVd_dataIn_lo = {queueAfterCheckVd_enq_bits_readSource, queueAfterCheckVd_enq_bits_instructionIndex};
  wire [4:0]   queueAfterCheckVd_enq_bits_vs;
  wire         queueAfterCheckVd_enq_bits_offset;
  wire [5:0]   queueAfterCheckVd_dataIn_hi_hi = {queueAfterCheckVd_enq_bits_vs, queueAfterCheckVd_enq_bits_offset};
  wire [3:0]   queueAfterCheckVd_enq_bits_groupIndex;
  wire [9:0]   queueAfterCheckVd_dataIn_hi = {queueAfterCheckVd_dataIn_hi_hi, queueAfterCheckVd_enq_bits_groupIndex};
  wire [16:0]  queueAfterCheckVd_dataIn = {queueAfterCheckVd_dataIn_hi, queueAfterCheckVd_dataIn_lo};
  assign queueAfterCheckVd_dataOut_instructionIndex = _queueAfterCheckVd_fifo_data_out[2:0];
  assign queueAfterCheckVd_dataOut_readSource = _queueAfterCheckVd_fifo_data_out[6:3];
  assign queueAfterCheckVd_dataOut_groupIndex = _queueAfterCheckVd_fifo_data_out[10:7];
  assign queueAfterCheckVd_dataOut_offset = _queueAfterCheckVd_fifo_data_out[11];
  assign queueAfterCheckVd_dataOut_vs = _queueAfterCheckVd_fifo_data_out[16:12];
  wire [4:0]   queueAfterCheckVd_deq_bits_vs = queueAfterCheckVd_dataOut_vs;
  wire         queueAfterCheckVd_deq_bits_offset = queueAfterCheckVd_dataOut_offset;
  wire [3:0]   queueAfterCheckVd_deq_bits_groupIndex = queueAfterCheckVd_dataOut_groupIndex;
  wire [3:0]   queueAfterCheckVd_deq_bits_readSource = queueAfterCheckVd_dataOut_readSource;
  wire [2:0]   queueAfterCheckVd_deq_bits_instructionIndex = queueAfterCheckVd_dataOut_instructionIndex;
  wire         queueAfterCheckVd_enq_ready = ~_queueAfterCheckVd_fifo_full;
  wire         queueAfterCheckVd_enq_valid;
  wire         queueAfterCheckVd_deq_ready;
  wire         queueBeforeCheck1_deq_valid;
  assign queueBeforeCheck1_deq_valid = ~_queueBeforeCheck1_fifo_empty;
  wire [4:0]   queueBeforeCheck1_dataOut_vs;
  assign queueAfterCheck1_enq_bits_vs = queueBeforeCheck1_deq_bits_vs;
  wire         queueBeforeCheck1_dataOut_offset;
  assign queueAfterCheck1_enq_bits_offset = queueBeforeCheck1_deq_bits_offset;
  wire [3:0]   queueBeforeCheck1_dataOut_groupIndex;
  assign queueAfterCheck1_enq_bits_groupIndex = queueBeforeCheck1_deq_bits_groupIndex;
  wire [3:0]   queueBeforeCheck1_dataOut_readSource;
  assign queueAfterCheck1_enq_bits_readSource = queueBeforeCheck1_deq_bits_readSource;
  wire [2:0]   queueBeforeCheck1_dataOut_instructionIndex;
  assign queueAfterCheck1_enq_bits_instructionIndex = queueBeforeCheck1_deq_bits_instructionIndex;
  wire [3:0]   queueBeforeCheck1_enq_bits_readSource;
  wire [6:0]   queueBeforeCheck1_dataIn_lo = {queueBeforeCheck1_enq_bits_readSource, queueBeforeCheck1_enq_bits_instructionIndex};
  wire [4:0]   queueBeforeCheck1_enq_bits_vs;
  wire         queueBeforeCheck1_enq_bits_offset;
  wire [5:0]   queueBeforeCheck1_dataIn_hi_hi = {queueBeforeCheck1_enq_bits_vs, queueBeforeCheck1_enq_bits_offset};
  wire [3:0]   queueBeforeCheck1_enq_bits_groupIndex;
  wire [9:0]   queueBeforeCheck1_dataIn_hi = {queueBeforeCheck1_dataIn_hi_hi, queueBeforeCheck1_enq_bits_groupIndex};
  wire [16:0]  queueBeforeCheck1_dataIn = {queueBeforeCheck1_dataIn_hi, queueBeforeCheck1_dataIn_lo};
  assign queueBeforeCheck1_dataOut_instructionIndex = _queueBeforeCheck1_fifo_data_out[2:0];
  assign queueBeforeCheck1_dataOut_readSource = _queueBeforeCheck1_fifo_data_out[6:3];
  assign queueBeforeCheck1_dataOut_groupIndex = _queueBeforeCheck1_fifo_data_out[10:7];
  assign queueBeforeCheck1_dataOut_offset = _queueBeforeCheck1_fifo_data_out[11];
  assign queueBeforeCheck1_dataOut_vs = _queueBeforeCheck1_fifo_data_out[16:12];
  assign queueBeforeCheck1_deq_bits_vs = queueBeforeCheck1_dataOut_vs;
  assign queueBeforeCheck1_deq_bits_offset = queueBeforeCheck1_dataOut_offset;
  assign queueBeforeCheck1_deq_bits_groupIndex = queueBeforeCheck1_dataOut_groupIndex;
  assign queueBeforeCheck1_deq_bits_readSource = queueBeforeCheck1_dataOut_readSource;
  assign queueBeforeCheck1_deq_bits_instructionIndex = queueBeforeCheck1_dataOut_instructionIndex;
  wire         queueBeforeCheck1_enq_ready = ~_queueBeforeCheck1_fifo_full;
  wire         queueBeforeCheck1_enq_valid;
  wire         queueBeforeCheck1_deq_ready;
  wire         queueBeforeCheck2_deq_valid;
  assign queueBeforeCheck2_deq_valid = ~_queueBeforeCheck2_fifo_empty;
  wire [4:0]   queueBeforeCheck2_dataOut_vs;
  assign queueAfterCheck2_enq_bits_vs = queueBeforeCheck2_deq_bits_vs;
  wire         queueBeforeCheck2_dataOut_offset;
  assign queueAfterCheck2_enq_bits_offset = queueBeforeCheck2_deq_bits_offset;
  wire [3:0]   queueBeforeCheck2_dataOut_groupIndex;
  assign queueAfterCheck2_enq_bits_groupIndex = queueBeforeCheck2_deq_bits_groupIndex;
  wire [3:0]   queueBeforeCheck2_dataOut_readSource;
  assign queueAfterCheck2_enq_bits_readSource = queueBeforeCheck2_deq_bits_readSource;
  wire [2:0]   queueBeforeCheck2_dataOut_instructionIndex;
  assign queueAfterCheck2_enq_bits_instructionIndex = queueBeforeCheck2_deq_bits_instructionIndex;
  wire [6:0]   queueBeforeCheck2_dataIn_lo = {4'h1, queueBeforeCheck2_enq_bits_instructionIndex};
  wire [4:0]   queueBeforeCheck2_enq_bits_vs;
  wire         queueBeforeCheck2_enq_bits_offset;
  wire [5:0]   queueBeforeCheck2_dataIn_hi_hi = {queueBeforeCheck2_enq_bits_vs, queueBeforeCheck2_enq_bits_offset};
  wire [3:0]   queueBeforeCheck2_enq_bits_groupIndex;
  wire [9:0]   queueBeforeCheck2_dataIn_hi = {queueBeforeCheck2_dataIn_hi_hi, queueBeforeCheck2_enq_bits_groupIndex};
  wire [16:0]  queueBeforeCheck2_dataIn = {queueBeforeCheck2_dataIn_hi, queueBeforeCheck2_dataIn_lo};
  assign queueBeforeCheck2_dataOut_instructionIndex = _queueBeforeCheck2_fifo_data_out[2:0];
  assign queueBeforeCheck2_dataOut_readSource = _queueBeforeCheck2_fifo_data_out[6:3];
  assign queueBeforeCheck2_dataOut_groupIndex = _queueBeforeCheck2_fifo_data_out[10:7];
  assign queueBeforeCheck2_dataOut_offset = _queueBeforeCheck2_fifo_data_out[11];
  assign queueBeforeCheck2_dataOut_vs = _queueBeforeCheck2_fifo_data_out[16:12];
  assign queueBeforeCheck2_deq_bits_vs = queueBeforeCheck2_dataOut_vs;
  assign queueBeforeCheck2_deq_bits_offset = queueBeforeCheck2_dataOut_offset;
  assign queueBeforeCheck2_deq_bits_groupIndex = queueBeforeCheck2_dataOut_groupIndex;
  assign queueBeforeCheck2_deq_bits_readSource = queueBeforeCheck2_dataOut_readSource;
  assign queueBeforeCheck2_deq_bits_instructionIndex = queueBeforeCheck2_dataOut_instructionIndex;
  wire         queueBeforeCheck2_enq_ready = ~_queueBeforeCheck2_fifo_full;
  wire         queueBeforeCheck2_enq_valid;
  wire         queueBeforeCheck2_deq_ready;
  wire         queueBeforeCheckVd_deq_valid;
  assign queueBeforeCheckVd_deq_valid = ~_queueBeforeCheckVd_fifo_empty;
  wire [4:0]   queueBeforeCheckVd_dataOut_vs;
  assign queueAfterCheckVd_enq_bits_vs = queueBeforeCheckVd_deq_bits_vs;
  wire         queueBeforeCheckVd_dataOut_offset;
  assign queueAfterCheckVd_enq_bits_offset = queueBeforeCheckVd_deq_bits_offset;
  wire [3:0]   queueBeforeCheckVd_dataOut_groupIndex;
  assign queueAfterCheckVd_enq_bits_groupIndex = queueBeforeCheckVd_deq_bits_groupIndex;
  wire [3:0]   queueBeforeCheckVd_dataOut_readSource;
  assign queueAfterCheckVd_enq_bits_readSource = queueBeforeCheckVd_deq_bits_readSource;
  wire [2:0]   queueBeforeCheckVd_dataOut_instructionIndex;
  assign queueAfterCheckVd_enq_bits_instructionIndex = queueBeforeCheckVd_deq_bits_instructionIndex;
  wire [6:0]   queueBeforeCheckVd_dataIn_lo = {4'h2, queueBeforeCheckVd_enq_bits_instructionIndex};
  wire [4:0]   queueBeforeCheckVd_enq_bits_vs;
  wire         queueBeforeCheckVd_enq_bits_offset;
  wire [5:0]   queueBeforeCheckVd_dataIn_hi_hi = {queueBeforeCheckVd_enq_bits_vs, queueBeforeCheckVd_enq_bits_offset};
  wire [3:0]   queueBeforeCheckVd_enq_bits_groupIndex;
  wire [9:0]   queueBeforeCheckVd_dataIn_hi = {queueBeforeCheckVd_dataIn_hi_hi, queueBeforeCheckVd_enq_bits_groupIndex};
  wire [16:0]  queueBeforeCheckVd_dataIn = {queueBeforeCheckVd_dataIn_hi, queueBeforeCheckVd_dataIn_lo};
  assign queueBeforeCheckVd_dataOut_instructionIndex = _queueBeforeCheckVd_fifo_data_out[2:0];
  assign queueBeforeCheckVd_dataOut_readSource = _queueBeforeCheckVd_fifo_data_out[6:3];
  assign queueBeforeCheckVd_dataOut_groupIndex = _queueBeforeCheckVd_fifo_data_out[10:7];
  assign queueBeforeCheckVd_dataOut_offset = _queueBeforeCheckVd_fifo_data_out[11];
  assign queueBeforeCheckVd_dataOut_vs = _queueBeforeCheckVd_fifo_data_out[16:12];
  assign queueBeforeCheckVd_deq_bits_vs = queueBeforeCheckVd_dataOut_vs;
  assign queueBeforeCheckVd_deq_bits_offset = queueBeforeCheckVd_dataOut_offset;
  assign queueBeforeCheckVd_deq_bits_groupIndex = queueBeforeCheckVd_dataOut_groupIndex;
  assign queueBeforeCheckVd_deq_bits_readSource = queueBeforeCheckVd_dataOut_readSource;
  assign queueBeforeCheckVd_deq_bits_instructionIndex = queueBeforeCheckVd_dataOut_instructionIndex;
  wire         queueBeforeCheckVd_enq_ready = ~_queueBeforeCheckVd_fifo_full;
  wire         queueBeforeCheckVd_enq_valid;
  wire         queueBeforeCheckVd_deq_ready;
  wire         pipeQueue_deq_valid;
  assign pipeQueue_deq_valid = ~_pipeQueue_fifo_empty;
  wire [4:0]   pipeQueue_dataOut_groupCounter;
  wire [4:0]   dequeue_bits_groupCounter_0 = pipeQueue_deq_bits_groupCounter;
  wire [3:0]   pipeQueue_dataOut_maskForMaskInput;
  wire [3:0]   dequeue_bits_mask_0 = pipeQueue_deq_bits_maskForMaskInput;
  wire [3:0]   pipeQueue_dataOut_boundaryMaskCorrection;
  wire [2:0]   pipeQueue_dataOut_instructionIndex;
  wire [2:0]   dequeue_bits_instructionIndex_0 = pipeQueue_deq_bits_instructionIndex;
  wire         pipeQueue_dataOut_decodeResult_specialSlot;
  wire         dequeue_bits_decodeResult_specialSlot_0 = pipeQueue_deq_bits_decodeResult_specialSlot;
  wire [4:0]   pipeQueue_dataOut_decodeResult_topUop;
  wire [4:0]   dequeue_bits_decodeResult_topUop_0 = pipeQueue_deq_bits_decodeResult_topUop;
  wire         pipeQueue_dataOut_decodeResult_popCount;
  wire         dequeue_bits_decodeResult_popCount_0 = pipeQueue_deq_bits_decodeResult_popCount;
  wire         pipeQueue_dataOut_decodeResult_ffo;
  wire         dequeue_bits_decodeResult_ffo_0 = pipeQueue_deq_bits_decodeResult_ffo;
  wire         pipeQueue_dataOut_decodeResult_average;
  wire         dequeue_bits_decodeResult_average_0 = pipeQueue_deq_bits_decodeResult_average;
  wire         pipeQueue_dataOut_decodeResult_reverse;
  wire         dequeue_bits_decodeResult_reverse_0 = pipeQueue_deq_bits_decodeResult_reverse;
  wire         pipeQueue_dataOut_decodeResult_dontNeedExecuteInLane;
  wire         dequeue_bits_decodeResult_dontNeedExecuteInLane_0 = pipeQueue_deq_bits_decodeResult_dontNeedExecuteInLane;
  wire         pipeQueue_dataOut_decodeResult_scheduler;
  wire         dequeue_bits_decodeResult_scheduler_0 = pipeQueue_deq_bits_decodeResult_scheduler;
  wire         pipeQueue_dataOut_decodeResult_sReadVD;
  wire         dequeue_bits_decodeResult_sReadVD_0 = pipeQueue_deq_bits_decodeResult_sReadVD;
  wire         pipeQueue_dataOut_decodeResult_vtype;
  wire         dequeue_bits_decodeResult_vtype_0 = pipeQueue_deq_bits_decodeResult_vtype;
  wire         pipeQueue_dataOut_decodeResult_sWrite;
  wire         dequeue_bits_decodeResult_sWrite_0 = pipeQueue_deq_bits_decodeResult_sWrite;
  wire         pipeQueue_dataOut_decodeResult_crossRead;
  wire         dequeue_bits_decodeResult_crossRead_0 = pipeQueue_deq_bits_decodeResult_crossRead;
  wire         pipeQueue_dataOut_decodeResult_crossWrite;
  wire         dequeue_bits_decodeResult_crossWrite_0 = pipeQueue_deq_bits_decodeResult_crossWrite;
  wire         pipeQueue_dataOut_decodeResult_maskUnit;
  wire         dequeue_bits_decodeResult_maskUnit_0 = pipeQueue_deq_bits_decodeResult_maskUnit;
  wire         pipeQueue_dataOut_decodeResult_special;
  wire         dequeue_bits_decodeResult_special_0 = pipeQueue_deq_bits_decodeResult_special;
  wire         pipeQueue_dataOut_decodeResult_saturate;
  wire         dequeue_bits_decodeResult_saturate_0 = pipeQueue_deq_bits_decodeResult_saturate;
  wire         pipeQueue_dataOut_decodeResult_vwmacc;
  wire         dequeue_bits_decodeResult_vwmacc_0 = pipeQueue_deq_bits_decodeResult_vwmacc;
  wire         pipeQueue_dataOut_decodeResult_readOnly;
  wire         dequeue_bits_decodeResult_readOnly_0 = pipeQueue_deq_bits_decodeResult_readOnly;
  wire         pipeQueue_dataOut_decodeResult_maskSource;
  wire         dequeue_bits_decodeResult_maskSource_0 = pipeQueue_deq_bits_decodeResult_maskSource;
  wire         pipeQueue_dataOut_decodeResult_maskDestination;
  wire         dequeue_bits_decodeResult_maskDestination_0 = pipeQueue_deq_bits_decodeResult_maskDestination;
  wire         pipeQueue_dataOut_decodeResult_maskLogic;
  wire         dequeue_bits_decodeResult_maskLogic_0 = pipeQueue_deq_bits_decodeResult_maskLogic;
  wire [3:0]   pipeQueue_dataOut_decodeResult_uop;
  wire [3:0]   dequeue_bits_decodeResult_uop_0 = pipeQueue_deq_bits_decodeResult_uop;
  wire         pipeQueue_dataOut_decodeResult_iota;
  wire         dequeue_bits_decodeResult_iota_0 = pipeQueue_deq_bits_decodeResult_iota;
  wire         pipeQueue_dataOut_decodeResult_mv;
  wire         dequeue_bits_decodeResult_mv_0 = pipeQueue_deq_bits_decodeResult_mv;
  wire         pipeQueue_dataOut_decodeResult_extend;
  wire         dequeue_bits_decodeResult_extend_0 = pipeQueue_deq_bits_decodeResult_extend;
  wire         pipeQueue_dataOut_decodeResult_unOrderWrite;
  wire         dequeue_bits_decodeResult_unOrderWrite_0 = pipeQueue_deq_bits_decodeResult_unOrderWrite;
  wire         pipeQueue_dataOut_decodeResult_compress;
  wire         dequeue_bits_decodeResult_compress_0 = pipeQueue_deq_bits_decodeResult_compress;
  wire         pipeQueue_dataOut_decodeResult_gather16;
  wire         dequeue_bits_decodeResult_gather16_0 = pipeQueue_deq_bits_decodeResult_gather16;
  wire         pipeQueue_dataOut_decodeResult_gather;
  wire         dequeue_bits_decodeResult_gather_0 = pipeQueue_deq_bits_decodeResult_gather;
  wire         pipeQueue_dataOut_decodeResult_slid;
  wire         dequeue_bits_decodeResult_slid_0 = pipeQueue_deq_bits_decodeResult_slid;
  wire         pipeQueue_dataOut_decodeResult_targetRd;
  wire         dequeue_bits_decodeResult_targetRd_0 = pipeQueue_deq_bits_decodeResult_targetRd;
  wire         pipeQueue_dataOut_decodeResult_widenReduce;
  wire         dequeue_bits_decodeResult_widenReduce_0 = pipeQueue_deq_bits_decodeResult_widenReduce;
  wire         pipeQueue_dataOut_decodeResult_red;
  wire         dequeue_bits_decodeResult_red_0 = pipeQueue_deq_bits_decodeResult_red;
  wire         pipeQueue_dataOut_decodeResult_nr;
  wire         dequeue_bits_decodeResult_nr_0 = pipeQueue_deq_bits_decodeResult_nr;
  wire         pipeQueue_dataOut_decodeResult_itype;
  wire         dequeue_bits_decodeResult_itype_0 = pipeQueue_deq_bits_decodeResult_itype;
  wire         pipeQueue_dataOut_decodeResult_unsigned1;
  wire         dequeue_bits_decodeResult_unsigned1_0 = pipeQueue_deq_bits_decodeResult_unsigned1;
  wire         pipeQueue_dataOut_decodeResult_unsigned0;
  wire         dequeue_bits_decodeResult_unsigned0_0 = pipeQueue_deq_bits_decodeResult_unsigned0;
  wire         pipeQueue_dataOut_decodeResult_other;
  wire         dequeue_bits_decodeResult_other_0 = pipeQueue_deq_bits_decodeResult_other;
  wire         pipeQueue_dataOut_decodeResult_multiCycle;
  wire         dequeue_bits_decodeResult_multiCycle_0 = pipeQueue_deq_bits_decodeResult_multiCycle;
  wire         pipeQueue_dataOut_decodeResult_divider;
  wire         dequeue_bits_decodeResult_divider_0 = pipeQueue_deq_bits_decodeResult_divider;
  wire         pipeQueue_dataOut_decodeResult_multiplier;
  wire         dequeue_bits_decodeResult_multiplier_0 = pipeQueue_deq_bits_decodeResult_multiplier;
  wire         pipeQueue_dataOut_decodeResult_shift;
  wire         dequeue_bits_decodeResult_shift_0 = pipeQueue_deq_bits_decodeResult_shift;
  wire         pipeQueue_dataOut_decodeResult_adder;
  wire         dequeue_bits_decodeResult_adder_0 = pipeQueue_deq_bits_decodeResult_adder;
  wire         pipeQueue_dataOut_decodeResult_logic;
  wire         dequeue_bits_decodeResult_logic_0 = pipeQueue_deq_bits_decodeResult_logic;
  wire [1:0]   pipeQueue_dataOut_laneIndex;
  wire [1:0]   dequeue_bits_laneIndex_0 = pipeQueue_deq_bits_laneIndex;
  wire         pipeQueue_dataOut_skipRead;
  wire [4:0]   pipeQueue_dataOut_vs1;
  wire [4:0]   pipeQueue_dataOut_vs2;
  wire [4:0]   pipeQueue_dataOut_vd;
  wire [4:0]   dequeue_bits_vd_0 = pipeQueue_deq_bits_vd;
  wire [2:0]   pipeQueue_dataOut_vSew1H;
  wire [2:0]   dequeue_bits_vSew1H_0 = pipeQueue_deq_bits_vSew1H;
  wire         pipeQueue_dataOut_maskNotMaskedElement;
  wire [8:0]   pipeQueue_dataOut_csr_vl;
  wire [8:0]   dequeue_bits_csr_vl_0 = pipeQueue_deq_bits_csr_vl;
  wire [8:0]   pipeQueue_dataOut_csr_vStart;
  wire [8:0]   dequeue_bits_csr_vStart_0 = pipeQueue_deq_bits_csr_vStart;
  wire [2:0]   pipeQueue_dataOut_csr_vlmul;
  wire [2:0]   dequeue_bits_csr_vlmul_0 = pipeQueue_deq_bits_csr_vlmul;
  wire [1:0]   pipeQueue_dataOut_csr_vSew;
  wire [1:0]   dequeue_bits_csr_vSew_0 = pipeQueue_deq_bits_csr_vSew;
  wire [1:0]   pipeQueue_dataOut_csr_vxrm;
  wire [1:0]   dequeue_bits_csr_vxrm_0 = pipeQueue_deq_bits_csr_vxrm;
  wire         pipeQueue_dataOut_csr_vta;
  wire         dequeue_bits_csr_vta_0 = pipeQueue_deq_bits_csr_vta;
  wire         pipeQueue_dataOut_csr_vma;
  wire         dequeue_bits_csr_vma_0 = pipeQueue_deq_bits_csr_vma;
  wire         pipeQueue_dataOut_maskType;
  wire         dequeue_bits_maskType_0 = pipeQueue_deq_bits_maskType;
  wire         pipeQueue_dataOut_loadStore;
  wire         dequeue_bits_loadStore_0 = pipeQueue_deq_bits_loadStore;
  wire [31:0]  pipeQueue_dataOut_readFromScalar;
  wire         pipeQueue_dataOut_bordersForMaskLogic;
  wire         dequeue_bits_bordersForMaskLogic_0 = pipeQueue_deq_bits_bordersForMaskLogic;
  wire [2:0]   pipeQueue_dataIn_lo_hi = {pipeQueue_enq_bits_csr_vxrm, pipeQueue_enq_bits_csr_vta};
  wire [3:0]   pipeQueue_dataIn_lo = {pipeQueue_dataIn_lo_hi, pipeQueue_enq_bits_csr_vma};
  wire [4:0]   pipeQueue_dataIn_hi_lo = {pipeQueue_enq_bits_csr_vlmul, pipeQueue_enq_bits_csr_vSew};
  wire [17:0]  pipeQueue_dataIn_hi_hi = {pipeQueue_enq_bits_csr_vl, pipeQueue_enq_bits_csr_vStart};
  wire [22:0]  pipeQueue_dataIn_hi = {pipeQueue_dataIn_hi_hi, pipeQueue_dataIn_hi_lo};
  wire [1:0]   pipeQueue_dataIn_lo_lo_lo_lo = {pipeQueue_enq_bits_decodeResult_adder, pipeQueue_enq_bits_decodeResult_logic};
  wire [1:0]   pipeQueue_dataIn_lo_lo_lo_hi_hi = {pipeQueue_enq_bits_decodeResult_divider, pipeQueue_enq_bits_decodeResult_multiplier};
  wire [2:0]   pipeQueue_dataIn_lo_lo_lo_hi = {pipeQueue_dataIn_lo_lo_lo_hi_hi, pipeQueue_enq_bits_decodeResult_shift};
  wire [4:0]   pipeQueue_dataIn_lo_lo_lo = {pipeQueue_dataIn_lo_lo_lo_hi, pipeQueue_dataIn_lo_lo_lo_lo};
  wire [1:0]   pipeQueue_dataIn_lo_lo_hi_lo_hi = {pipeQueue_enq_bits_decodeResult_unsigned0, pipeQueue_enq_bits_decodeResult_other};
  wire [2:0]   pipeQueue_dataIn_lo_lo_hi_lo = {pipeQueue_dataIn_lo_lo_hi_lo_hi, pipeQueue_enq_bits_decodeResult_multiCycle};
  wire [1:0]   pipeQueue_dataIn_lo_lo_hi_hi_hi = {pipeQueue_enq_bits_decodeResult_nr, pipeQueue_enq_bits_decodeResult_itype};
  wire [2:0]   pipeQueue_dataIn_lo_lo_hi_hi = {pipeQueue_dataIn_lo_lo_hi_hi_hi, pipeQueue_enq_bits_decodeResult_unsigned1};
  wire [5:0]   pipeQueue_dataIn_lo_lo_hi = {pipeQueue_dataIn_lo_lo_hi_hi, pipeQueue_dataIn_lo_lo_hi_lo};
  wire [10:0]  pipeQueue_dataIn_lo_lo = {pipeQueue_dataIn_lo_lo_hi, pipeQueue_dataIn_lo_lo_lo};
  wire [1:0]   pipeQueue_dataIn_lo_hi_lo_lo = {pipeQueue_enq_bits_decodeResult_widenReduce, pipeQueue_enq_bits_decodeResult_red};
  wire [1:0]   pipeQueue_dataIn_lo_hi_lo_hi_hi = {pipeQueue_enq_bits_decodeResult_gather, pipeQueue_enq_bits_decodeResult_slid};
  wire [2:0]   pipeQueue_dataIn_lo_hi_lo_hi = {pipeQueue_dataIn_lo_hi_lo_hi_hi, pipeQueue_enq_bits_decodeResult_targetRd};
  wire [4:0]   pipeQueue_dataIn_lo_hi_lo = {pipeQueue_dataIn_lo_hi_lo_hi, pipeQueue_dataIn_lo_hi_lo_lo};
  wire [1:0]   pipeQueue_dataIn_lo_hi_hi_lo_hi = {pipeQueue_enq_bits_decodeResult_unOrderWrite, pipeQueue_enq_bits_decodeResult_compress};
  wire [2:0]   pipeQueue_dataIn_lo_hi_hi_lo = {pipeQueue_dataIn_lo_hi_hi_lo_hi, pipeQueue_enq_bits_decodeResult_gather16};
  wire [1:0]   pipeQueue_dataIn_lo_hi_hi_hi_hi = {pipeQueue_enq_bits_decodeResult_iota, pipeQueue_enq_bits_decodeResult_mv};
  wire [2:0]   pipeQueue_dataIn_lo_hi_hi_hi = {pipeQueue_dataIn_lo_hi_hi_hi_hi, pipeQueue_enq_bits_decodeResult_extend};
  wire [5:0]   pipeQueue_dataIn_lo_hi_hi = {pipeQueue_dataIn_lo_hi_hi_hi, pipeQueue_dataIn_lo_hi_hi_lo};
  wire [10:0]  pipeQueue_dataIn_lo_hi_1 = {pipeQueue_dataIn_lo_hi_hi, pipeQueue_dataIn_lo_hi_lo};
  wire [21:0]  pipeQueue_dataIn_lo_1 = {pipeQueue_dataIn_lo_hi_1, pipeQueue_dataIn_lo_lo};
  wire [4:0]   pipeQueue_dataIn_hi_lo_lo_lo = {pipeQueue_enq_bits_decodeResult_maskLogic, pipeQueue_enq_bits_decodeResult_uop};
  wire [1:0]   pipeQueue_dataIn_hi_lo_lo_hi_hi = {pipeQueue_enq_bits_decodeResult_readOnly, pipeQueue_enq_bits_decodeResult_maskSource};
  wire [2:0]   pipeQueue_dataIn_hi_lo_lo_hi = {pipeQueue_dataIn_hi_lo_lo_hi_hi, pipeQueue_enq_bits_decodeResult_maskDestination};
  wire [7:0]   pipeQueue_dataIn_hi_lo_lo = {pipeQueue_dataIn_hi_lo_lo_hi, pipeQueue_dataIn_hi_lo_lo_lo};
  wire [1:0]   pipeQueue_dataIn_hi_lo_hi_lo_hi = {pipeQueue_enq_bits_decodeResult_special, pipeQueue_enq_bits_decodeResult_saturate};
  wire [2:0]   pipeQueue_dataIn_hi_lo_hi_lo = {pipeQueue_dataIn_hi_lo_hi_lo_hi, pipeQueue_enq_bits_decodeResult_vwmacc};
  wire [1:0]   pipeQueue_dataIn_hi_lo_hi_hi_hi = {pipeQueue_enq_bits_decodeResult_crossRead, pipeQueue_enq_bits_decodeResult_crossWrite};
  wire [2:0]   pipeQueue_dataIn_hi_lo_hi_hi = {pipeQueue_dataIn_hi_lo_hi_hi_hi, pipeQueue_enq_bits_decodeResult_maskUnit};
  wire [5:0]   pipeQueue_dataIn_hi_lo_hi = {pipeQueue_dataIn_hi_lo_hi_hi, pipeQueue_dataIn_hi_lo_hi_lo};
  wire [13:0]  pipeQueue_dataIn_hi_lo_1 = {pipeQueue_dataIn_hi_lo_hi, pipeQueue_dataIn_hi_lo_lo};
  wire [1:0]   pipeQueue_dataIn_hi_hi_lo_lo = {pipeQueue_enq_bits_decodeResult_vtype, pipeQueue_enq_bits_decodeResult_sWrite};
  wire [1:0]   pipeQueue_dataIn_hi_hi_lo_hi_hi = {pipeQueue_enq_bits_decodeResult_dontNeedExecuteInLane, pipeQueue_enq_bits_decodeResult_scheduler};
  wire [2:0]   pipeQueue_dataIn_hi_hi_lo_hi = {pipeQueue_dataIn_hi_hi_lo_hi_hi, pipeQueue_enq_bits_decodeResult_sReadVD};
  wire [4:0]   pipeQueue_dataIn_hi_hi_lo = {pipeQueue_dataIn_hi_hi_lo_hi, pipeQueue_dataIn_hi_hi_lo_lo};
  wire [1:0]   pipeQueue_dataIn_hi_hi_hi_lo_hi = {pipeQueue_enq_bits_decodeResult_ffo, pipeQueue_enq_bits_decodeResult_average};
  wire [2:0]   pipeQueue_dataIn_hi_hi_hi_lo = {pipeQueue_dataIn_hi_hi_hi_lo_hi, pipeQueue_enq_bits_decodeResult_reverse};
  wire [5:0]   pipeQueue_dataIn_hi_hi_hi_hi_hi = {pipeQueue_enq_bits_decodeResult_specialSlot, pipeQueue_enq_bits_decodeResult_topUop};
  wire [6:0]   pipeQueue_dataIn_hi_hi_hi_hi = {pipeQueue_dataIn_hi_hi_hi_hi_hi, pipeQueue_enq_bits_decodeResult_popCount};
  wire [9:0]   pipeQueue_dataIn_hi_hi_hi = {pipeQueue_dataIn_hi_hi_hi_hi, pipeQueue_dataIn_hi_hi_hi_lo};
  wire [14:0]  pipeQueue_dataIn_hi_hi_1 = {pipeQueue_dataIn_hi_hi_hi, pipeQueue_dataIn_hi_hi_lo};
  wire [28:0]  pipeQueue_dataIn_hi_1 = {pipeQueue_dataIn_hi_hi_1, pipeQueue_dataIn_hi_lo_1};
  wire [32:0]  pipeQueue_dataIn_lo_lo_lo_1 = {pipeQueue_enq_bits_readFromScalar, pipeQueue_enq_bits_bordersForMaskLogic};
  wire [1:0]   pipeQueue_dataIn_lo_lo_hi_1 = {pipeQueue_enq_bits_maskType, pipeQueue_enq_bits_loadStore};
  wire [34:0]  pipeQueue_dataIn_lo_lo_1 = {pipeQueue_dataIn_lo_lo_hi_1, pipeQueue_dataIn_lo_lo_lo_1};
  wire [27:0]  pipeQueue_dataIn_lo_hi_lo_1 = {pipeQueue_enq_bits_maskNotMaskedElement, pipeQueue_dataIn_hi, pipeQueue_dataIn_lo};
  wire [7:0]   pipeQueue_dataIn_lo_hi_hi_1 = {pipeQueue_enq_bits_vd, pipeQueue_enq_bits_vSew1H};
  wire [35:0]  pipeQueue_dataIn_lo_hi_2 = {pipeQueue_dataIn_lo_hi_hi_1, pipeQueue_dataIn_lo_hi_lo_1};
  wire [70:0]  pipeQueue_dataIn_lo_2 = {pipeQueue_dataIn_lo_hi_2, pipeQueue_dataIn_lo_lo_1};
  wire [9:0]   pipeQueue_dataIn_hi_lo_lo_1 = {pipeQueue_enq_bits_vs1, pipeQueue_enq_bits_vs2};
  wire [2:0]   pipeQueue_dataIn_hi_lo_hi_1 = {pipeQueue_enq_bits_laneIndex, pipeQueue_enq_bits_skipRead};
  wire [12:0]  pipeQueue_dataIn_hi_lo_2 = {pipeQueue_dataIn_hi_lo_hi_1, pipeQueue_dataIn_hi_lo_lo_1};
  wire [53:0]  pipeQueue_dataIn_hi_hi_lo_1 = {pipeQueue_enq_bits_instructionIndex, pipeQueue_dataIn_hi_1, pipeQueue_dataIn_lo_1};
  wire [8:0]   pipeQueue_dataIn_hi_hi_hi_hi_1 = {pipeQueue_enq_bits_groupCounter, pipeQueue_enq_bits_maskForMaskInput};
  wire [12:0]  pipeQueue_dataIn_hi_hi_hi_1 = {pipeQueue_dataIn_hi_hi_hi_hi_1, pipeQueue_enq_bits_boundaryMaskCorrection};
  wire [66:0]  pipeQueue_dataIn_hi_hi_2 = {pipeQueue_dataIn_hi_hi_hi_1, pipeQueue_dataIn_hi_hi_lo_1};
  wire [79:0]  pipeQueue_dataIn_hi_2 = {pipeQueue_dataIn_hi_hi_2, pipeQueue_dataIn_hi_lo_2};
  wire [150:0] pipeQueue_dataIn = {pipeQueue_dataIn_hi_2, pipeQueue_dataIn_lo_2};
  assign pipeQueue_dataOut_bordersForMaskLogic = _pipeQueue_fifo_data_out[0];
  assign pipeQueue_dataOut_readFromScalar = _pipeQueue_fifo_data_out[32:1];
  assign pipeQueue_dataOut_loadStore = _pipeQueue_fifo_data_out[33];
  assign pipeQueue_dataOut_maskType = _pipeQueue_fifo_data_out[34];
  assign pipeQueue_dataOut_csr_vma = _pipeQueue_fifo_data_out[35];
  assign pipeQueue_dataOut_csr_vta = _pipeQueue_fifo_data_out[36];
  assign pipeQueue_dataOut_csr_vxrm = _pipeQueue_fifo_data_out[38:37];
  assign pipeQueue_dataOut_csr_vSew = _pipeQueue_fifo_data_out[40:39];
  assign pipeQueue_dataOut_csr_vlmul = _pipeQueue_fifo_data_out[43:41];
  assign pipeQueue_dataOut_csr_vStart = _pipeQueue_fifo_data_out[52:44];
  assign pipeQueue_dataOut_csr_vl = _pipeQueue_fifo_data_out[61:53];
  assign pipeQueue_dataOut_maskNotMaskedElement = _pipeQueue_fifo_data_out[62];
  assign pipeQueue_dataOut_vSew1H = _pipeQueue_fifo_data_out[65:63];
  assign pipeQueue_dataOut_vd = _pipeQueue_fifo_data_out[70:66];
  assign pipeQueue_dataOut_vs2 = _pipeQueue_fifo_data_out[75:71];
  assign pipeQueue_dataOut_vs1 = _pipeQueue_fifo_data_out[80:76];
  assign pipeQueue_dataOut_skipRead = _pipeQueue_fifo_data_out[81];
  assign pipeQueue_dataOut_laneIndex = _pipeQueue_fifo_data_out[83:82];
  assign pipeQueue_dataOut_decodeResult_logic = _pipeQueue_fifo_data_out[84];
  assign pipeQueue_dataOut_decodeResult_adder = _pipeQueue_fifo_data_out[85];
  assign pipeQueue_dataOut_decodeResult_shift = _pipeQueue_fifo_data_out[86];
  assign pipeQueue_dataOut_decodeResult_multiplier = _pipeQueue_fifo_data_out[87];
  assign pipeQueue_dataOut_decodeResult_divider = _pipeQueue_fifo_data_out[88];
  assign pipeQueue_dataOut_decodeResult_multiCycle = _pipeQueue_fifo_data_out[89];
  assign pipeQueue_dataOut_decodeResult_other = _pipeQueue_fifo_data_out[90];
  assign pipeQueue_dataOut_decodeResult_unsigned0 = _pipeQueue_fifo_data_out[91];
  assign pipeQueue_dataOut_decodeResult_unsigned1 = _pipeQueue_fifo_data_out[92];
  assign pipeQueue_dataOut_decodeResult_itype = _pipeQueue_fifo_data_out[93];
  assign pipeQueue_dataOut_decodeResult_nr = _pipeQueue_fifo_data_out[94];
  assign pipeQueue_dataOut_decodeResult_red = _pipeQueue_fifo_data_out[95];
  assign pipeQueue_dataOut_decodeResult_widenReduce = _pipeQueue_fifo_data_out[96];
  assign pipeQueue_dataOut_decodeResult_targetRd = _pipeQueue_fifo_data_out[97];
  assign pipeQueue_dataOut_decodeResult_slid = _pipeQueue_fifo_data_out[98];
  assign pipeQueue_dataOut_decodeResult_gather = _pipeQueue_fifo_data_out[99];
  assign pipeQueue_dataOut_decodeResult_gather16 = _pipeQueue_fifo_data_out[100];
  assign pipeQueue_dataOut_decodeResult_compress = _pipeQueue_fifo_data_out[101];
  assign pipeQueue_dataOut_decodeResult_unOrderWrite = _pipeQueue_fifo_data_out[102];
  assign pipeQueue_dataOut_decodeResult_extend = _pipeQueue_fifo_data_out[103];
  assign pipeQueue_dataOut_decodeResult_mv = _pipeQueue_fifo_data_out[104];
  assign pipeQueue_dataOut_decodeResult_iota = _pipeQueue_fifo_data_out[105];
  assign pipeQueue_dataOut_decodeResult_uop = _pipeQueue_fifo_data_out[109:106];
  assign pipeQueue_dataOut_decodeResult_maskLogic = _pipeQueue_fifo_data_out[110];
  assign pipeQueue_dataOut_decodeResult_maskDestination = _pipeQueue_fifo_data_out[111];
  assign pipeQueue_dataOut_decodeResult_maskSource = _pipeQueue_fifo_data_out[112];
  assign pipeQueue_dataOut_decodeResult_readOnly = _pipeQueue_fifo_data_out[113];
  assign pipeQueue_dataOut_decodeResult_vwmacc = _pipeQueue_fifo_data_out[114];
  assign pipeQueue_dataOut_decodeResult_saturate = _pipeQueue_fifo_data_out[115];
  assign pipeQueue_dataOut_decodeResult_special = _pipeQueue_fifo_data_out[116];
  assign pipeQueue_dataOut_decodeResult_maskUnit = _pipeQueue_fifo_data_out[117];
  assign pipeQueue_dataOut_decodeResult_crossWrite = _pipeQueue_fifo_data_out[118];
  assign pipeQueue_dataOut_decodeResult_crossRead = _pipeQueue_fifo_data_out[119];
  assign pipeQueue_dataOut_decodeResult_sWrite = _pipeQueue_fifo_data_out[120];
  assign pipeQueue_dataOut_decodeResult_vtype = _pipeQueue_fifo_data_out[121];
  assign pipeQueue_dataOut_decodeResult_sReadVD = _pipeQueue_fifo_data_out[122];
  assign pipeQueue_dataOut_decodeResult_scheduler = _pipeQueue_fifo_data_out[123];
  assign pipeQueue_dataOut_decodeResult_dontNeedExecuteInLane = _pipeQueue_fifo_data_out[124];
  assign pipeQueue_dataOut_decodeResult_reverse = _pipeQueue_fifo_data_out[125];
  assign pipeQueue_dataOut_decodeResult_average = _pipeQueue_fifo_data_out[126];
  assign pipeQueue_dataOut_decodeResult_ffo = _pipeQueue_fifo_data_out[127];
  assign pipeQueue_dataOut_decodeResult_popCount = _pipeQueue_fifo_data_out[128];
  assign pipeQueue_dataOut_decodeResult_topUop = _pipeQueue_fifo_data_out[133:129];
  assign pipeQueue_dataOut_decodeResult_specialSlot = _pipeQueue_fifo_data_out[134];
  assign pipeQueue_dataOut_instructionIndex = _pipeQueue_fifo_data_out[137:135];
  assign pipeQueue_dataOut_boundaryMaskCorrection = _pipeQueue_fifo_data_out[141:138];
  assign pipeQueue_dataOut_maskForMaskInput = _pipeQueue_fifo_data_out[145:142];
  assign pipeQueue_dataOut_groupCounter = _pipeQueue_fifo_data_out[150:146];
  assign pipeQueue_deq_bits_groupCounter = pipeQueue_dataOut_groupCounter;
  assign pipeQueue_deq_bits_maskForMaskInput = pipeQueue_dataOut_maskForMaskInput;
  wire [3:0]   pipeQueue_deq_bits_boundaryMaskCorrection = pipeQueue_dataOut_boundaryMaskCorrection;
  assign pipeQueue_deq_bits_instructionIndex = pipeQueue_dataOut_instructionIndex;
  assign pipeQueue_deq_bits_decodeResult_specialSlot = pipeQueue_dataOut_decodeResult_specialSlot;
  assign pipeQueue_deq_bits_decodeResult_topUop = pipeQueue_dataOut_decodeResult_topUop;
  assign pipeQueue_deq_bits_decodeResult_popCount = pipeQueue_dataOut_decodeResult_popCount;
  assign pipeQueue_deq_bits_decodeResult_ffo = pipeQueue_dataOut_decodeResult_ffo;
  assign pipeQueue_deq_bits_decodeResult_average = pipeQueue_dataOut_decodeResult_average;
  assign pipeQueue_deq_bits_decodeResult_reverse = pipeQueue_dataOut_decodeResult_reverse;
  assign pipeQueue_deq_bits_decodeResult_dontNeedExecuteInLane = pipeQueue_dataOut_decodeResult_dontNeedExecuteInLane;
  assign pipeQueue_deq_bits_decodeResult_scheduler = pipeQueue_dataOut_decodeResult_scheduler;
  assign pipeQueue_deq_bits_decodeResult_sReadVD = pipeQueue_dataOut_decodeResult_sReadVD;
  assign pipeQueue_deq_bits_decodeResult_vtype = pipeQueue_dataOut_decodeResult_vtype;
  assign pipeQueue_deq_bits_decodeResult_sWrite = pipeQueue_dataOut_decodeResult_sWrite;
  assign pipeQueue_deq_bits_decodeResult_crossRead = pipeQueue_dataOut_decodeResult_crossRead;
  assign pipeQueue_deq_bits_decodeResult_crossWrite = pipeQueue_dataOut_decodeResult_crossWrite;
  assign pipeQueue_deq_bits_decodeResult_maskUnit = pipeQueue_dataOut_decodeResult_maskUnit;
  assign pipeQueue_deq_bits_decodeResult_special = pipeQueue_dataOut_decodeResult_special;
  assign pipeQueue_deq_bits_decodeResult_saturate = pipeQueue_dataOut_decodeResult_saturate;
  assign pipeQueue_deq_bits_decodeResult_vwmacc = pipeQueue_dataOut_decodeResult_vwmacc;
  assign pipeQueue_deq_bits_decodeResult_readOnly = pipeQueue_dataOut_decodeResult_readOnly;
  assign pipeQueue_deq_bits_decodeResult_maskSource = pipeQueue_dataOut_decodeResult_maskSource;
  assign pipeQueue_deq_bits_decodeResult_maskDestination = pipeQueue_dataOut_decodeResult_maskDestination;
  assign pipeQueue_deq_bits_decodeResult_maskLogic = pipeQueue_dataOut_decodeResult_maskLogic;
  assign pipeQueue_deq_bits_decodeResult_uop = pipeQueue_dataOut_decodeResult_uop;
  assign pipeQueue_deq_bits_decodeResult_iota = pipeQueue_dataOut_decodeResult_iota;
  assign pipeQueue_deq_bits_decodeResult_mv = pipeQueue_dataOut_decodeResult_mv;
  assign pipeQueue_deq_bits_decodeResult_extend = pipeQueue_dataOut_decodeResult_extend;
  assign pipeQueue_deq_bits_decodeResult_unOrderWrite = pipeQueue_dataOut_decodeResult_unOrderWrite;
  assign pipeQueue_deq_bits_decodeResult_compress = pipeQueue_dataOut_decodeResult_compress;
  assign pipeQueue_deq_bits_decodeResult_gather16 = pipeQueue_dataOut_decodeResult_gather16;
  assign pipeQueue_deq_bits_decodeResult_gather = pipeQueue_dataOut_decodeResult_gather;
  assign pipeQueue_deq_bits_decodeResult_slid = pipeQueue_dataOut_decodeResult_slid;
  assign pipeQueue_deq_bits_decodeResult_targetRd = pipeQueue_dataOut_decodeResult_targetRd;
  assign pipeQueue_deq_bits_decodeResult_widenReduce = pipeQueue_dataOut_decodeResult_widenReduce;
  assign pipeQueue_deq_bits_decodeResult_red = pipeQueue_dataOut_decodeResult_red;
  assign pipeQueue_deq_bits_decodeResult_nr = pipeQueue_dataOut_decodeResult_nr;
  assign pipeQueue_deq_bits_decodeResult_itype = pipeQueue_dataOut_decodeResult_itype;
  assign pipeQueue_deq_bits_decodeResult_unsigned1 = pipeQueue_dataOut_decodeResult_unsigned1;
  assign pipeQueue_deq_bits_decodeResult_unsigned0 = pipeQueue_dataOut_decodeResult_unsigned0;
  assign pipeQueue_deq_bits_decodeResult_other = pipeQueue_dataOut_decodeResult_other;
  assign pipeQueue_deq_bits_decodeResult_multiCycle = pipeQueue_dataOut_decodeResult_multiCycle;
  assign pipeQueue_deq_bits_decodeResult_divider = pipeQueue_dataOut_decodeResult_divider;
  assign pipeQueue_deq_bits_decodeResult_multiplier = pipeQueue_dataOut_decodeResult_multiplier;
  assign pipeQueue_deq_bits_decodeResult_shift = pipeQueue_dataOut_decodeResult_shift;
  assign pipeQueue_deq_bits_decodeResult_adder = pipeQueue_dataOut_decodeResult_adder;
  assign pipeQueue_deq_bits_decodeResult_logic = pipeQueue_dataOut_decodeResult_logic;
  assign pipeQueue_deq_bits_laneIndex = pipeQueue_dataOut_laneIndex;
  wire         pipeQueue_deq_bits_skipRead = pipeQueue_dataOut_skipRead;
  wire [4:0]   pipeQueue_deq_bits_vs1 = pipeQueue_dataOut_vs1;
  wire [4:0]   pipeQueue_deq_bits_vs2 = pipeQueue_dataOut_vs2;
  assign pipeQueue_deq_bits_vd = pipeQueue_dataOut_vd;
  assign pipeQueue_deq_bits_vSew1H = pipeQueue_dataOut_vSew1H;
  wire         pipeQueue_deq_bits_maskNotMaskedElement = pipeQueue_dataOut_maskNotMaskedElement;
  assign pipeQueue_deq_bits_csr_vl = pipeQueue_dataOut_csr_vl;
  assign pipeQueue_deq_bits_csr_vStart = pipeQueue_dataOut_csr_vStart;
  assign pipeQueue_deq_bits_csr_vlmul = pipeQueue_dataOut_csr_vlmul;
  assign pipeQueue_deq_bits_csr_vSew = pipeQueue_dataOut_csr_vSew;
  assign pipeQueue_deq_bits_csr_vxrm = pipeQueue_dataOut_csr_vxrm;
  assign pipeQueue_deq_bits_csr_vta = pipeQueue_dataOut_csr_vta;
  assign pipeQueue_deq_bits_csr_vma = pipeQueue_dataOut_csr_vma;
  assign pipeQueue_deq_bits_maskType = pipeQueue_dataOut_maskType;
  assign pipeQueue_deq_bits_loadStore = pipeQueue_dataOut_loadStore;
  wire [31:0]  pipeQueue_deq_bits_readFromScalar = pipeQueue_dataOut_readFromScalar;
  assign pipeQueue_deq_bits_bordersForMaskLogic = pipeQueue_dataOut_bordersForMaskLogic;
  wire         pipeQueue_enq_ready = ~_pipeQueue_fifo_full;
  wire         pipeQueue_enq_valid;
  wire         pipeQueue_deq_ready;
  wire         enqueue_ready_0;
  wire         dequeue_valid_0;
  assign pipeQueue_enq_valid = enqueue_ready_0 & enqueue_valid_0;
  assign pipeQueue_deq_ready = dequeue_ready_0 & dequeue_valid_0;
  wire         allReadQueueReady = queueBeforeCheck1_enq_ready & queueBeforeCheck2_enq_ready & queueBeforeCheckVd_enq_ready;
  assign queueBeforeCheck1_enq_bits_groupIndex = enqueue_bits_groupCounter_0[3:0];
  assign queueBeforeCheck2_enq_bits_groupIndex = enqueue_bits_groupCounter_0[3:0];
  assign queueBeforeCheckVd_enq_bits_groupIndex = enqueue_bits_groupCounter_0[3:0];
  assign enqueue_ready_0 = allReadQueueReady & pipeQueue_enq_ready;
  assign queueBeforeCheck1_deq_ready = queueAfterCheck1_enq_ready & checkResult_0;
  assign queueAfterCheck1_enq_valid = queueBeforeCheck1_deq_valid & checkResult_0;
  assign queueBeforeCheck2_deq_ready = queueAfterCheck2_enq_ready & checkResult_1;
  assign queueAfterCheck2_enq_valid = queueBeforeCheck2_deq_valid & checkResult_1;
  assign queueBeforeCheckVd_deq_ready = queueAfterCheckVd_enq_ready & checkResult_2;
  assign queueAfterCheckVd_enq_valid = queueBeforeCheckVd_deq_valid & checkResult_2;
  assign queueBeforeCheck1_enq_valid = pipeQueue_enq_valid & enqueue_bits_decodeResult_vtype_0 & ~enqueue_bits_skipRead_0;
  assign queueBeforeCheck2_enq_valid = pipeQueue_enq_valid & ~enqueue_bits_skipRead_0;
  assign queueBeforeCheckVd_enq_valid = pipeQueue_enq_valid & ~enqueue_bits_decodeResult_sReadVD_0;
  wire [4:0]   _GEN = {1'h0, enqueue_bits_groupCounter_0[4:1]};
  assign queueBeforeCheck1_enq_bits_vs = enqueue_bits_decodeResult_maskLogic_0 & ~enqueue_bits_decodeResult_logic_0 ? 5'h0 : enqueue_bits_vs1_0 + _GEN;
  assign queueBeforeCheck1_enq_bits_readSource = {2'h0, {2{enqueue_bits_decodeResult_maskLogic_0 & ~enqueue_bits_decodeResult_logic_0}}};
  assign queueBeforeCheck2_enq_bits_vs = enqueue_bits_vs2_0 + _GEN;
  assign queueBeforeCheckVd_enq_bits_vs = enqueue_bits_vd_0 + _GEN;
  assign queueBeforeCheck1_enq_bits_offset = enqueue_bits_groupCounter_0[0];
  assign queueBeforeCheck2_enq_bits_offset = enqueue_bits_groupCounter_0[0];
  assign queueBeforeCheckVd_enq_bits_offset = enqueue_bits_groupCounter_0[0];
  wire         dataQueueVs1_io_deq_valid;
  assign dataQueueVs1_io_deq_valid = ~_dataQueueVs1_io_fifo_empty;
  wire         dataQueueVs1_io_enq_ready = ~_dataQueueVs1_io_fifo_full;
  wire         dataQueueVs1_io_enq_valid;
  wire         dataQueueVs1_io_deq_ready;
  wire         dataQueueVs2_deq_valid;
  assign dataQueueVs2_deq_valid = ~_dataQueueVs2_fifo_empty;
  wire [31:0]  dequeue_bits_src_1_0 = dataQueueVs2_deq_bits;
  wire         dataQueueVs2_enq_ready = ~_dataQueueVs2_fifo_full;
  wire         dataQueueVs2_enq_valid;
  wire         dataQueueVs2_deq_ready;
  wire         dataQueueVd_deq_valid;
  assign dataQueueVd_deq_valid = ~_dataQueueVd_fifo_empty;
  wire [31:0]  dequeue_bits_src_2_0 = dataQueueVd_deq_bits;
  wire         dataQueueVd_enq_ready = ~_dataQueueVd_fifo_full;
  wire         dataQueueVd_enq_valid;
  wire         dataQueueVd_deq_ready;
  reg  [2:0]   dataQueueNotFull2_counterReg;
  wire         dataQueueNotFull2_doEnq = queueAfterCheck2_deq_ready & queueAfterCheck2_deq_valid;
  wire         dataQueueNotFull2_doDeq = dataQueueVs2_deq_ready & dataQueueVs2_deq_valid;
  wire [2:0]   dataQueueNotFull2_countChange = dataQueueNotFull2_doEnq ? 3'h1 : 3'h7;
  wire         dataQueueNotFull2 = ~(dataQueueNotFull2_counterReg[2]);
  reg  [2:0]   dataQueueNotFullVd_counterReg;
  wire         dataQueueNotFullVd_doEnq = queueAfterCheckVd_deq_ready & queueAfterCheckVd_deq_valid;
  wire         dataQueueNotFullVd_doDeq = dataQueueVd_deq_ready & dataQueueVd_deq_valid;
  wire [2:0]   dataQueueNotFullVd_countChange = dataQueueNotFullVd_doEnq ? 3'h1 : 3'h7;
  wire         dataQueueNotFullVd = ~(dataQueueNotFullVd_counterReg[2]);
  assign queueAfterCheck2_deq_ready = _readPipe1_enqueue_ready & dataQueueNotFull2;
  assign queueAfterCheckVd_deq_ready = _readPipe2_enqueue_ready & dataQueueNotFullVd;
  wire [31:0]  dataQueueVs1_io_deq_bits;
  assign source1Select = pipeQueue_deq_bits_decodeResult_vtype ? dataQueueVs1_io_deq_bits : pipeQueue_deq_bits_readFromScalar;
  wire [31:0]  dequeue_bits_src_0_0 = source1Select;
  wire [3:0]   dequeue_bits_maskForFilter_0 = ({4{pipeQueue_deq_bits_maskNotMaskedElement}} | pipeQueue_deq_bits_maskForMaskInput) & pipeQueue_deq_bits_boundaryMaskCorrection;
  wire         dataQueueValidVec_0 = dataQueueVs1_io_deq_valid | ~pipeQueue_deq_bits_decodeResult_vtype | pipeQueue_deq_bits_skipRead;
  wire         dataQueueValidVec_1 = dataQueueVs2_deq_valid | pipeQueue_deq_bits_skipRead;
  wire         dataQueueValidVec_2 = dataQueueVd_deq_valid | pipeQueue_deq_bits_decodeResult_sReadVD;
  wire [1:0]   allDataQueueValid_hi = {dataQueueValidVec_2, dataQueueValidVec_1};
  wire         allDataQueueValid = &{allDataQueueValid_hi, dataQueueValidVec_0};
  assign dequeue_valid_0 = allDataQueueValid & pipeQueue_deq_valid;
  wire         _dataQueueVd_deq_ready_T = allDataQueueValid & dequeue_ready_0;
  assign dataQueueVs1_io_deq_ready = _dataQueueVd_deq_ready_T & pipeQueue_deq_bits_decodeResult_vtype;
  assign dataQueueVs2_deq_ready = _dataQueueVd_deq_ready_T & ~pipeQueue_deq_bits_skipRead;
  assign dataQueueVd_deq_ready = _dataQueueVd_deq_ready_T & ~pipeQueue_deq_bits_decodeResult_sReadVD;
  wire         stageFinish = ~pipeQueue_deq_valid;
  always @(posedge clock) begin
    if (reset) begin
      dataQueueNotFull2_counterReg <= 3'h0;
      dataQueueNotFullVd_counterReg <= 3'h0;
    end
    else begin
      if (dataQueueNotFull2_doEnq ^ dataQueueNotFull2_doDeq)
        dataQueueNotFull2_counterReg <= dataQueueNotFull2_counterReg + dataQueueNotFull2_countChange;
      if (dataQueueNotFullVd_doEnq ^ dataQueueNotFullVd_doDeq)
        dataQueueNotFullVd_counterReg <= dataQueueNotFullVd_counterReg + dataQueueNotFullVd_countChange;
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
        dataQueueNotFull2_counterReg = _RANDOM[/*Zero width*/ 1'b0][2:0];
        dataQueueNotFullVd_counterReg = _RANDOM[/*Zero width*/ 1'b0][5:3];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  wire         queueAfterCheck1_empty;
  assign queueAfterCheck1_empty = _queueAfterCheck1_fifo_empty;
  wire         queueAfterCheck1_full;
  assign queueAfterCheck1_full = _queueAfterCheck1_fifo_full;
  wire         queueAfterCheck2_empty;
  assign queueAfterCheck2_empty = _queueAfterCheck2_fifo_empty;
  wire         queueAfterCheck2_full;
  assign queueAfterCheck2_full = _queueAfterCheck2_fifo_full;
  wire         queueAfterCheckVd_empty;
  assign queueAfterCheckVd_empty = _queueAfterCheckVd_fifo_empty;
  wire         queueAfterCheckVd_full;
  assign queueAfterCheckVd_full = _queueAfterCheckVd_fifo_full;
  wire         queueBeforeCheck1_empty;
  assign queueBeforeCheck1_empty = _queueBeforeCheck1_fifo_empty;
  wire         queueBeforeCheck1_full;
  assign queueBeforeCheck1_full = _queueBeforeCheck1_fifo_full;
  wire         queueBeforeCheck2_empty;
  assign queueBeforeCheck2_empty = _queueBeforeCheck2_fifo_empty;
  wire         queueBeforeCheck2_full;
  assign queueBeforeCheck2_full = _queueBeforeCheck2_fifo_full;
  wire         queueBeforeCheckVd_empty;
  assign queueBeforeCheckVd_empty = _queueBeforeCheckVd_fifo_empty;
  wire         queueBeforeCheckVd_full;
  assign queueBeforeCheckVd_full = _queueBeforeCheckVd_fifo_full;
  wire         pipeQueue_empty;
  assign pipeQueue_empty = _pipeQueue_fifo_empty;
  wire         pipeQueue_full;
  assign pipeQueue_full = _pipeQueue_fifo_full;
  wire         dataQueueVs1_io_empty;
  assign dataQueueVs1_io_empty = _dataQueueVs1_io_fifo_empty;
  wire         dataQueueVs1_io_full;
  assign dataQueueVs1_io_full = _dataQueueVs1_io_fifo_full;
  wire         dataQueueVs2_empty;
  assign dataQueueVs2_empty = _dataQueueVs2_fifo_empty;
  wire         dataQueueVs2_full;
  assign dataQueueVs2_full = _dataQueueVs2_fifo_full;
  wire         dataQueueVd_empty;
  assign dataQueueVd_empty = _dataQueueVd_fifo_empty;
  wire         dataQueueVd_full;
  assign dataQueueVd_full = _dataQueueVd_fifo_full;
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(17)
  ) queueAfterCheck1_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(queueAfterCheck1_enq_ready & queueAfterCheck1_enq_valid)),
    .pop_req_n    (~(queueAfterCheck1_deq_ready & ~_queueAfterCheck1_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (queueAfterCheck1_dataIn),
    .empty        (_queueAfterCheck1_fifo_empty),
    .almost_empty (queueAfterCheck1_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (queueAfterCheck1_almostFull),
    .full         (_queueAfterCheck1_fifo_full),
    .error        (_queueAfterCheck1_fifo_error),
    .data_out     (_queueAfterCheck1_fifo_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(17)
  ) queueAfterCheck2_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(queueAfterCheck2_enq_ready & queueAfterCheck2_enq_valid)),
    .pop_req_n    (~(queueAfterCheck2_deq_ready & ~_queueAfterCheck2_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (queueAfterCheck2_dataIn),
    .empty        (_queueAfterCheck2_fifo_empty),
    .almost_empty (queueAfterCheck2_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (queueAfterCheck2_almostFull),
    .full         (_queueAfterCheck2_fifo_full),
    .error        (_queueAfterCheck2_fifo_error),
    .data_out     (_queueAfterCheck2_fifo_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(17)
  ) queueAfterCheckVd_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(queueAfterCheckVd_enq_ready & queueAfterCheckVd_enq_valid)),
    .pop_req_n    (~(queueAfterCheckVd_deq_ready & ~_queueAfterCheckVd_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (queueAfterCheckVd_dataIn),
    .empty        (_queueAfterCheckVd_fifo_empty),
    .almost_empty (queueAfterCheckVd_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (queueAfterCheckVd_almostFull),
    .full         (_queueAfterCheckVd_fifo_full),
    .error        (_queueAfterCheckVd_fifo_error),
    .data_out     (_queueAfterCheckVd_fifo_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(17)
  ) queueBeforeCheck1_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(queueBeforeCheck1_enq_ready & queueBeforeCheck1_enq_valid)),
    .pop_req_n    (~(queueBeforeCheck1_deq_ready & ~_queueBeforeCheck1_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (queueBeforeCheck1_dataIn),
    .empty        (_queueBeforeCheck1_fifo_empty),
    .almost_empty (queueBeforeCheck1_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (queueBeforeCheck1_almostFull),
    .full         (_queueBeforeCheck1_fifo_full),
    .error        (_queueBeforeCheck1_fifo_error),
    .data_out     (_queueBeforeCheck1_fifo_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(17)
  ) queueBeforeCheck2_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(queueBeforeCheck2_enq_ready & queueBeforeCheck2_enq_valid)),
    .pop_req_n    (~(queueBeforeCheck2_deq_ready & ~_queueBeforeCheck2_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (queueBeforeCheck2_dataIn),
    .empty        (_queueBeforeCheck2_fifo_empty),
    .almost_empty (queueBeforeCheck2_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (queueBeforeCheck2_almostFull),
    .full         (_queueBeforeCheck2_fifo_full),
    .error        (_queueBeforeCheck2_fifo_error),
    .data_out     (_queueBeforeCheck2_fifo_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(17)
  ) queueBeforeCheckVd_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(queueBeforeCheckVd_enq_ready & queueBeforeCheckVd_enq_valid)),
    .pop_req_n    (~(queueBeforeCheckVd_deq_ready & ~_queueBeforeCheckVd_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (queueBeforeCheckVd_dataIn),
    .empty        (_queueBeforeCheckVd_fifo_empty),
    .almost_empty (queueBeforeCheckVd_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (queueBeforeCheckVd_almostFull),
    .full         (_queueBeforeCheckVd_fifo_full),
    .error        (_queueBeforeCheckVd_fifo_error),
    .data_out     (_queueBeforeCheckVd_fifo_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(14),
    .err_mode(2),
    .rst_mode(3),
    .width(151)
  ) pipeQueue_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(pipeQueue_enq_ready & pipeQueue_enq_valid)),
    .pop_req_n    (~(pipeQueue_deq_ready & ~_pipeQueue_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (pipeQueue_dataIn),
    .empty        (_pipeQueue_fifo_empty),
    .almost_empty (pipeQueue_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (pipeQueue_almostFull),
    .full         (_pipeQueue_fifo_full),
    .error        (_pipeQueue_fifo_error),
    .data_out     (_pipeQueue_fifo_data_out)
  );
  VrfReadPipe readPipe0 (
    .clock                                (clock),
    .reset                                (reset),
    .enqueue_ready                        (queueAfterCheck1_deq_ready),
    .enqueue_valid                        (queueAfterCheck1_deq_valid),
    .enqueue_bits_vs                      (queueAfterCheck1_deq_bits_vs),
    .enqueue_bits_offset                  (queueAfterCheck1_deq_bits_offset),
    .enqueue_bits_groupIndex              (queueAfterCheck1_deq_bits_groupIndex),
    .enqueue_bits_readSource              (queueAfterCheck1_deq_bits_readSource),
    .enqueue_bits_instructionIndex        (queueAfterCheck1_deq_bits_instructionIndex),
    .vrfReadRequest_ready                 (vrfReadRequest_0_ready_0),
    .vrfReadRequest_valid                 (vrfReadRequest_0_valid_0),
    .vrfReadRequest_bits_vs               (vrfReadRequest_0_bits_vs_0),
    .vrfReadRequest_bits_readSource       (vrfReadRequest_0_bits_readSource_0),
    .vrfReadRequest_bits_offset           (vrfReadRequest_0_bits_offset_0),
    .vrfReadRequest_bits_instructionIndex (vrfReadRequest_0_bits_instructionIndex_0),
    .vrfReadResult                        (vrfReadResult_0),
    .dequeue_ready                        (dataQueueVs1_io_enq_ready),
    .dequeue_valid                        (dataQueueVs1_io_enq_valid),
    .dequeue_bits                         (dataQueueVs1_io_enq_bits)
  );
  VrfReadPipe readPipe1 (
    .clock                                (clock),
    .reset                                (reset),
    .enqueue_ready                        (_readPipe1_enqueue_ready),
    .enqueue_valid                        (queueAfterCheck2_deq_valid & dataQueueNotFull2),
    .enqueue_bits_vs                      (queueAfterCheck2_deq_bits_vs),
    .enqueue_bits_offset                  (queueAfterCheck2_deq_bits_offset),
    .enqueue_bits_groupIndex              (queueAfterCheck2_deq_bits_groupIndex),
    .enqueue_bits_readSource              (queueAfterCheck2_deq_bits_readSource),
    .enqueue_bits_instructionIndex        (queueAfterCheck2_deq_bits_instructionIndex),
    .vrfReadRequest_ready                 (vrfReadRequest_1_ready_0),
    .vrfReadRequest_valid                 (vrfReadRequest_1_valid_0),
    .vrfReadRequest_bits_vs               (vrfReadRequest_1_bits_vs_0),
    .vrfReadRequest_bits_readSource       (vrfReadRequest_1_bits_readSource_0),
    .vrfReadRequest_bits_offset           (vrfReadRequest_1_bits_offset_0),
    .vrfReadRequest_bits_instructionIndex (vrfReadRequest_1_bits_instructionIndex_0),
    .vrfReadResult                        (vrfReadResult_1),
    .dequeue_ready                        (dataQueueVs2_enq_ready),
    .dequeue_valid                        (dataQueueVs2_enq_valid),
    .dequeue_bits                         (dataQueueVs2_enq_bits)
  );
  VrfReadPipe readPipe2 (
    .clock                                (clock),
    .reset                                (reset),
    .enqueue_ready                        (_readPipe2_enqueue_ready),
    .enqueue_valid                        (queueAfterCheckVd_deq_valid & dataQueueNotFullVd),
    .enqueue_bits_vs                      (queueAfterCheckVd_deq_bits_vs),
    .enqueue_bits_offset                  (queueAfterCheckVd_deq_bits_offset),
    .enqueue_bits_groupIndex              (queueAfterCheckVd_deq_bits_groupIndex),
    .enqueue_bits_readSource              (queueAfterCheckVd_deq_bits_readSource),
    .enqueue_bits_instructionIndex        (queueAfterCheckVd_deq_bits_instructionIndex),
    .vrfReadRequest_ready                 (vrfReadRequest_2_ready_0),
    .vrfReadRequest_valid                 (vrfReadRequest_2_valid_0),
    .vrfReadRequest_bits_vs               (vrfReadRequest_2_bits_vs_0),
    .vrfReadRequest_bits_readSource       (vrfReadRequest_2_bits_readSource_0),
    .vrfReadRequest_bits_offset           (vrfReadRequest_2_bits_offset_0),
    .vrfReadRequest_bits_instructionIndex (vrfReadRequest_2_bits_instructionIndex_0),
    .vrfReadResult                        (vrfReadResult_2),
    .dequeue_ready                        (dataQueueVd_enq_ready),
    .dequeue_valid                        (dataQueueVd_enq_valid),
    .dequeue_bits                         (dataQueueVd_enq_bits)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(32)
  ) dataQueueVs1_io_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(dataQueueVs1_io_enq_ready & dataQueueVs1_io_enq_valid)),
    .pop_req_n    (~(dataQueueVs1_io_deq_ready & ~_dataQueueVs1_io_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (dataQueueVs1_io_enq_bits),
    .empty        (_dataQueueVs1_io_fifo_empty),
    .almost_empty (dataQueueVs1_io_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (dataQueueVs1_io_almostFull),
    .full         (_dataQueueVs1_io_fifo_full),
    .error        (_dataQueueVs1_io_fifo_error),
    .data_out     (dataQueueVs1_io_deq_bits)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(32)
  ) dataQueueVs2_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(dataQueueVs2_enq_ready & dataQueueVs2_enq_valid)),
    .pop_req_n    (~(dataQueueVs2_deq_ready & ~_dataQueueVs2_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (dataQueueVs2_enq_bits),
    .empty        (_dataQueueVs2_fifo_empty),
    .almost_empty (dataQueueVs2_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (dataQueueVs2_almostFull),
    .full         (_dataQueueVs2_fifo_full),
    .error        (_dataQueueVs2_fifo_error),
    .data_out     (dataQueueVs2_deq_bits)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(32)
  ) dataQueueVd_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(dataQueueVd_enq_ready & dataQueueVd_enq_valid)),
    .pop_req_n    (~(dataQueueVd_deq_ready & ~_dataQueueVd_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (dataQueueVd_enq_bits),
    .empty        (_dataQueueVd_fifo_empty),
    .almost_empty (dataQueueVd_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (dataQueueVd_almostFull),
    .full         (_dataQueueVd_fifo_full),
    .error        (_dataQueueVd_fifo_error),
    .data_out     (dataQueueVd_deq_bits)
  );
  assign enqueue_ready = enqueue_ready_0;
  assign dequeue_valid = dequeue_valid_0;
  assign dequeue_bits_maskForFilter = dequeue_bits_maskForFilter_0;
  assign dequeue_bits_mask = dequeue_bits_mask_0;
  assign dequeue_bits_groupCounter = dequeue_bits_groupCounter_0;
  assign dequeue_bits_src_0 = dequeue_bits_src_0_0;
  assign dequeue_bits_src_1 = dequeue_bits_src_1_0;
  assign dequeue_bits_src_2 = dequeue_bits_src_2_0;
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
  assign dequeue_bits_vSew1H = dequeue_bits_vSew1H_0;
  assign dequeue_bits_csr_vl = dequeue_bits_csr_vl_0;
  assign dequeue_bits_csr_vStart = dequeue_bits_csr_vStart_0;
  assign dequeue_bits_csr_vlmul = dequeue_bits_csr_vlmul_0;
  assign dequeue_bits_csr_vSew = dequeue_bits_csr_vSew_0;
  assign dequeue_bits_csr_vxrm = dequeue_bits_csr_vxrm_0;
  assign dequeue_bits_csr_vta = dequeue_bits_csr_vta_0;
  assign dequeue_bits_csr_vma = dequeue_bits_csr_vma_0;
  assign dequeue_bits_maskType = dequeue_bits_maskType_0;
  assign dequeue_bits_laneIndex = dequeue_bits_laneIndex_0;
  assign dequeue_bits_instructionIndex = dequeue_bits_instructionIndex_0;
  assign dequeue_bits_loadStore = dequeue_bits_loadStore_0;
  assign dequeue_bits_vd = dequeue_bits_vd_0;
  assign dequeue_bits_bordersForMaskLogic = dequeue_bits_bordersForMaskLogic_0;
  assign vrfReadRequest_0_valid = vrfReadRequest_0_valid_0;
  assign vrfReadRequest_0_bits_vs = vrfReadRequest_0_bits_vs_0;
  assign vrfReadRequest_0_bits_readSource = vrfReadRequest_0_bits_readSource_0;
  assign vrfReadRequest_0_bits_offset = vrfReadRequest_0_bits_offset_0;
  assign vrfReadRequest_0_bits_instructionIndex = vrfReadRequest_0_bits_instructionIndex_0;
  assign vrfReadRequest_1_valid = vrfReadRequest_1_valid_0;
  assign vrfReadRequest_1_bits_vs = vrfReadRequest_1_bits_vs_0;
  assign vrfReadRequest_1_bits_readSource = vrfReadRequest_1_bits_readSource_0;
  assign vrfReadRequest_1_bits_offset = vrfReadRequest_1_bits_offset_0;
  assign vrfReadRequest_1_bits_instructionIndex = vrfReadRequest_1_bits_instructionIndex_0;
  assign vrfReadRequest_2_valid = vrfReadRequest_2_valid_0;
  assign vrfReadRequest_2_bits_vs = vrfReadRequest_2_bits_vs_0;
  assign vrfReadRequest_2_bits_readSource = vrfReadRequest_2_bits_readSource_0;
  assign vrfReadRequest_2_bits_offset = vrfReadRequest_2_bits_offset_0;
  assign vrfReadRequest_2_bits_instructionIndex = vrfReadRequest_2_bits_instructionIndex_0;
  assign vrfCheckRequest_0_vs = queueBeforeCheck1_deq_bits_vs;
  assign vrfCheckRequest_0_readSource = queueBeforeCheck1_deq_bits_readSource[1:0];
  assign vrfCheckRequest_0_offset = queueBeforeCheck1_deq_bits_offset;
  assign vrfCheckRequest_0_instructionIndex = queueBeforeCheck1_deq_bits_instructionIndex;
  assign vrfCheckRequest_1_vs = queueBeforeCheck2_deq_bits_vs;
  assign vrfCheckRequest_1_readSource = queueBeforeCheck2_deq_bits_readSource[1:0];
  assign vrfCheckRequest_1_offset = queueBeforeCheck2_deq_bits_offset;
  assign vrfCheckRequest_1_instructionIndex = queueBeforeCheck2_deq_bits_instructionIndex;
  assign vrfCheckRequest_2_vs = queueBeforeCheckVd_deq_bits_vs;
  assign vrfCheckRequest_2_readSource = queueBeforeCheckVd_deq_bits_readSource[1:0];
  assign vrfCheckRequest_2_offset = queueBeforeCheckVd_deq_bits_offset;
  assign vrfCheckRequest_2_instructionIndex = queueBeforeCheckVd_deq_bits_instructionIndex;
endmodule

