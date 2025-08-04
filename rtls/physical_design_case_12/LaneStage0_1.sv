
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
module LaneStage0_1(
  input         clock,
                reset,
  output        enqueue_ready,
  input         enqueue_valid,
  input  [4:0]  enqueue_bits_maskIndex,
  input  [31:0] enqueue_bits_maskForMaskGroup,
  input  [5:0]  enqueue_bits_maskGroupCount,
  input  [31:0] enqueue_bits_readFromScalar,
  input  [2:0]  enqueue_bits_vSew1H,
  input         enqueue_bits_loadStore,
  input  [3:0]  enqueue_bits_laneIndex,
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
  input  [5:0]  enqueue_bits_lastGroupForInstruction,
  input         enqueue_bits_isLastLaneForInstruction,
                enqueue_bits_instructionFinished,
  input  [11:0] enqueue_bits_csr_vl,
                enqueue_bits_csr_vStart,
  input  [2:0]  enqueue_bits_csr_vlmul,
  input  [1:0]  enqueue_bits_csr_vSew,
                enqueue_bits_csr_vxrm,
  input         enqueue_bits_csr_vta,
                enqueue_bits_csr_vma,
                enqueue_bits_maskType,
                enqueue_bits_maskNotMaskedElement,
  input  [4:0]  enqueue_bits_vs1,
                enqueue_bits_vs2,
                enqueue_bits_vd,
  input  [2:0]  enqueue_bits_instructionIndex,
  input         enqueue_bits_additionalRW,
                enqueue_bits_skipRead,
                enqueue_bits_skipEnable,
                dequeue_ready,
  output        dequeue_valid,
  output [3:0]  dequeue_bits_maskForMaskInput,
                dequeue_bits_boundaryMaskCorrection,
  output [5:0]  dequeue_bits_groupCounter,
  output [31:0] dequeue_bits_readFromScalar,
  output [2:0]  dequeue_bits_instructionIndex,
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
  output [3:0]  dequeue_bits_laneIndex,
  output        dequeue_bits_skipRead,
  output [4:0]  dequeue_bits_vs1,
                dequeue_bits_vs2,
                dequeue_bits_vd,
  output [2:0]  dequeue_bits_vSew1H,
  output        dequeue_bits_maskNotMaskedElement,
  output [11:0] dequeue_bits_csr_vl,
                dequeue_bits_csr_vStart,
  output [2:0]  dequeue_bits_csr_vlmul,
  output [1:0]  dequeue_bits_csr_vSew,
                dequeue_bits_csr_vxrm,
  output        dequeue_bits_csr_vta,
                dequeue_bits_csr_vma,
                dequeue_bits_maskType,
                dequeue_bits_loadStore,
                dequeue_bits_bordersForMaskLogic,
  output [5:0]  updateLaneState_maskGroupCount,
  output [4:0]  updateLaneState_maskIndex,
  output        updateLaneState_outOfExecutionRange,
                updateLaneState_maskExhausted,
                tokenReport_valid,
                tokenReport_bits_decodeResult_sWrite,
                tokenReport_bits_decodeResult_maskUnit,
  output [2:0]  tokenReport_bits_instructionIndex
);

  wire        _updateLaneState_outOfExecutionRange_output;
  wire        enqueue_valid_0 = enqueue_valid;
  wire [4:0]  enqueue_bits_maskIndex_0 = enqueue_bits_maskIndex;
  wire [31:0] enqueue_bits_maskForMaskGroup_0 = enqueue_bits_maskForMaskGroup;
  wire [5:0]  enqueue_bits_maskGroupCount_0 = enqueue_bits_maskGroupCount;
  wire [31:0] enqueue_bits_readFromScalar_0 = enqueue_bits_readFromScalar;
  wire [2:0]  enqueue_bits_vSew1H_0 = enqueue_bits_vSew1H;
  wire        enqueue_bits_loadStore_0 = enqueue_bits_loadStore;
  wire [3:0]  enqueue_bits_laneIndex_0 = enqueue_bits_laneIndex;
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
  wire [5:0]  enqueue_bits_lastGroupForInstruction_0 = enqueue_bits_lastGroupForInstruction;
  wire        enqueue_bits_isLastLaneForInstruction_0 = enqueue_bits_isLastLaneForInstruction;
  wire        enqueue_bits_instructionFinished_0 = enqueue_bits_instructionFinished;
  wire [11:0] enqueue_bits_csr_vl_0 = enqueue_bits_csr_vl;
  wire [11:0] enqueue_bits_csr_vStart_0 = enqueue_bits_csr_vStart;
  wire [2:0]  enqueue_bits_csr_vlmul_0 = enqueue_bits_csr_vlmul;
  wire [1:0]  enqueue_bits_csr_vSew_0 = enqueue_bits_csr_vSew;
  wire [1:0]  enqueue_bits_csr_vxrm_0 = enqueue_bits_csr_vxrm;
  wire        enqueue_bits_csr_vta_0 = enqueue_bits_csr_vta;
  wire        enqueue_bits_csr_vma_0 = enqueue_bits_csr_vma;
  wire        enqueue_bits_maskType_0 = enqueue_bits_maskType;
  wire        enqueue_bits_maskNotMaskedElement_0 = enqueue_bits_maskNotMaskedElement;
  wire [4:0]  enqueue_bits_vs1_0 = enqueue_bits_vs1;
  wire [4:0]  enqueue_bits_vs2_0 = enqueue_bits_vs2;
  wire [4:0]  enqueue_bits_vd_0 = enqueue_bits_vd;
  wire [2:0]  enqueue_bits_instructionIndex_0 = enqueue_bits_instructionIndex;
  wire        enqueue_bits_additionalRW_0 = enqueue_bits_additionalRW;
  wire        enqueue_bits_skipRead_0 = enqueue_bits_skipRead;
  wire        enqueue_bits_skipEnable_0 = enqueue_bits_skipEnable;
  wire        dequeue_ready_0 = dequeue_ready;
  wire        stageFinish = 1'h1;
  wire [4:0]  filterVec_groupIndex_2 = enqueue_bits_maskIndex_0;
  wire [2:0]  stageWire_vSew1H = enqueue_bits_vSew1H_0;
  wire        stageWire_loadStore = enqueue_bits_loadStore_0;
  wire [3:0]  stageWire_laneIndex = enqueue_bits_laneIndex_0;
  wire        stageWire_decodeResult_specialSlot = enqueue_bits_decodeResult_specialSlot_0;
  wire [4:0]  stageWire_decodeResult_topUop = enqueue_bits_decodeResult_topUop_0;
  wire        stageWire_decodeResult_popCount = enqueue_bits_decodeResult_popCount_0;
  wire        stageWire_decodeResult_ffo = enqueue_bits_decodeResult_ffo_0;
  wire        stageWire_decodeResult_average = enqueue_bits_decodeResult_average_0;
  wire        stageWire_decodeResult_reverse = enqueue_bits_decodeResult_reverse_0;
  wire        stageWire_decodeResult_dontNeedExecuteInLane = enqueue_bits_decodeResult_dontNeedExecuteInLane_0;
  wire        stageWire_decodeResult_scheduler = enqueue_bits_decodeResult_scheduler_0;
  wire        stageWire_decodeResult_sReadVD = enqueue_bits_decodeResult_sReadVD_0;
  wire        stageWire_decodeResult_vtype = enqueue_bits_decodeResult_vtype_0;
  wire        stageWire_decodeResult_sWrite = enqueue_bits_decodeResult_sWrite_0;
  wire        stageWire_decodeResult_crossRead = enqueue_bits_decodeResult_crossRead_0;
  wire        stageWire_decodeResult_crossWrite = enqueue_bits_decodeResult_crossWrite_0;
  wire        stageWire_decodeResult_maskUnit = enqueue_bits_decodeResult_maskUnit_0;
  wire        stageWire_decodeResult_special = enqueue_bits_decodeResult_special_0;
  wire        stageWire_decodeResult_saturate = enqueue_bits_decodeResult_saturate_0;
  wire        stageWire_decodeResult_vwmacc = enqueue_bits_decodeResult_vwmacc_0;
  wire        stageWire_decodeResult_readOnly = enqueue_bits_decodeResult_readOnly_0;
  wire        stageWire_decodeResult_maskSource = enqueue_bits_decodeResult_maskSource_0;
  wire        stageWire_decodeResult_maskDestination = enqueue_bits_decodeResult_maskDestination_0;
  wire        stageWire_decodeResult_maskLogic = enqueue_bits_decodeResult_maskLogic_0;
  wire [3:0]  stageWire_decodeResult_uop = enqueue_bits_decodeResult_uop_0;
  wire        stageWire_decodeResult_iota = enqueue_bits_decodeResult_iota_0;
  wire        stageWire_decodeResult_mv = enqueue_bits_decodeResult_mv_0;
  wire        stageWire_decodeResult_extend = enqueue_bits_decodeResult_extend_0;
  wire        stageWire_decodeResult_unOrderWrite = enqueue_bits_decodeResult_unOrderWrite_0;
  wire        stageWire_decodeResult_compress = enqueue_bits_decodeResult_compress_0;
  wire        stageWire_decodeResult_gather16 = enqueue_bits_decodeResult_gather16_0;
  wire        stageWire_decodeResult_gather = enqueue_bits_decodeResult_gather_0;
  wire        stageWire_decodeResult_slid = enqueue_bits_decodeResult_slid_0;
  wire        stageWire_decodeResult_targetRd = enqueue_bits_decodeResult_targetRd_0;
  wire        stageWire_decodeResult_widenReduce = enqueue_bits_decodeResult_widenReduce_0;
  wire        stageWire_decodeResult_red = enqueue_bits_decodeResult_red_0;
  wire        stageWire_decodeResult_nr = enqueue_bits_decodeResult_nr_0;
  wire        stageWire_decodeResult_itype = enqueue_bits_decodeResult_itype_0;
  wire        stageWire_decodeResult_unsigned1 = enqueue_bits_decodeResult_unsigned1_0;
  wire        stageWire_decodeResult_unsigned0 = enqueue_bits_decodeResult_unsigned0_0;
  wire        stageWire_decodeResult_other = enqueue_bits_decodeResult_other_0;
  wire        stageWire_decodeResult_multiCycle = enqueue_bits_decodeResult_multiCycle_0;
  wire        stageWire_decodeResult_divider = enqueue_bits_decodeResult_divider_0;
  wire        stageWire_decodeResult_multiplier = enqueue_bits_decodeResult_multiplier_0;
  wire        stageWire_decodeResult_shift = enqueue_bits_decodeResult_shift_0;
  wire        stageWire_decodeResult_adder = enqueue_bits_decodeResult_adder_0;
  wire        stageWire_decodeResult_logic = enqueue_bits_decodeResult_logic_0;
  wire [11:0] stageWire_csr_vl = enqueue_bits_csr_vl_0;
  wire [11:0] stageWire_csr_vStart = enqueue_bits_csr_vStart_0;
  wire [2:0]  stageWire_csr_vlmul = enqueue_bits_csr_vlmul_0;
  wire [1:0]  stageWire_csr_vSew = enqueue_bits_csr_vSew_0;
  wire [1:0]  stageWire_csr_vxrm = enqueue_bits_csr_vxrm_0;
  wire        stageWire_csr_vta = enqueue_bits_csr_vta_0;
  wire        stageWire_csr_vma = enqueue_bits_csr_vma_0;
  wire        stageWire_maskType = enqueue_bits_maskType_0;
  wire        stageWire_maskNotMaskedElement = enqueue_bits_maskNotMaskedElement_0;
  wire [4:0]  stageWire_vs1 = enqueue_bits_vs1_0;
  wire [4:0]  stageWire_vs2 = enqueue_bits_vs2_0;
  wire [4:0]  stageWire_vd = enqueue_bits_vd_0;
  wire [2:0]  stageWire_instructionIndex = enqueue_bits_instructionIndex_0;
  wire        stageWire_skipRead = enqueue_bits_skipRead_0;
  reg         stageValidReg;
  wire        dequeue_valid_0 = stageValidReg;
  wire        enqueue_ready_0 = ~stageValidReg | dequeue_ready_0;
  wire [3:0]  stageWire_maskForMaskInput;
  wire        notMaskedAllElement =
    enqueue_bits_vSew1H_0[0] & (|stageWire_maskForMaskInput) | enqueue_bits_vSew1H_0[1] & (|(stageWire_maskForMaskInput[1:0])) | enqueue_bits_vSew1H_0[2] & stageWire_maskForMaskInput[0] | enqueue_bits_maskNotMaskedElement_0
    | enqueue_bits_decodeResult_maskDestination_0 | enqueue_bits_decodeResult_red_0 | enqueue_bits_decodeResult_readOnly_0 | enqueue_bits_loadStore_0 | enqueue_bits_decodeResult_gather_0 | enqueue_bits_decodeResult_crossRead_0
    | enqueue_bits_decodeResult_crossWrite_0;
  wire        enqFire = enqueue_ready_0 & enqueue_valid_0 & (~_updateLaneState_outOfExecutionRange_output | enqueue_bits_additionalRW_0) & notMaskedAllElement;
  reg  [3:0]  stageDataReg_maskForMaskInput;
  wire [3:0]  dequeue_bits_maskForMaskInput_0 = stageDataReg_maskForMaskInput;
  reg  [3:0]  stageDataReg_boundaryMaskCorrection;
  wire [3:0]  dequeue_bits_boundaryMaskCorrection_0 = stageDataReg_boundaryMaskCorrection;
  reg  [5:0]  stageDataReg_groupCounter;
  wire [5:0]  dequeue_bits_groupCounter_0 = stageDataReg_groupCounter;
  reg  [31:0] stageDataReg_readFromScalar;
  wire [31:0] dequeue_bits_readFromScalar_0 = stageDataReg_readFromScalar;
  reg  [2:0]  stageDataReg_instructionIndex;
  wire [2:0]  dequeue_bits_instructionIndex_0 = stageDataReg_instructionIndex;
  reg         stageDataReg_decodeResult_specialSlot;
  wire        dequeue_bits_decodeResult_specialSlot_0 = stageDataReg_decodeResult_specialSlot;
  reg  [4:0]  stageDataReg_decodeResult_topUop;
  wire [4:0]  dequeue_bits_decodeResult_topUop_0 = stageDataReg_decodeResult_topUop;
  reg         stageDataReg_decodeResult_popCount;
  wire        dequeue_bits_decodeResult_popCount_0 = stageDataReg_decodeResult_popCount;
  reg         stageDataReg_decodeResult_ffo;
  wire        dequeue_bits_decodeResult_ffo_0 = stageDataReg_decodeResult_ffo;
  reg         stageDataReg_decodeResult_average;
  wire        dequeue_bits_decodeResult_average_0 = stageDataReg_decodeResult_average;
  reg         stageDataReg_decodeResult_reverse;
  wire        dequeue_bits_decodeResult_reverse_0 = stageDataReg_decodeResult_reverse;
  reg         stageDataReg_decodeResult_dontNeedExecuteInLane;
  wire        dequeue_bits_decodeResult_dontNeedExecuteInLane_0 = stageDataReg_decodeResult_dontNeedExecuteInLane;
  reg         stageDataReg_decodeResult_scheduler;
  wire        dequeue_bits_decodeResult_scheduler_0 = stageDataReg_decodeResult_scheduler;
  reg         stageDataReg_decodeResult_sReadVD;
  wire        dequeue_bits_decodeResult_sReadVD_0 = stageDataReg_decodeResult_sReadVD;
  reg         stageDataReg_decodeResult_vtype;
  wire        dequeue_bits_decodeResult_vtype_0 = stageDataReg_decodeResult_vtype;
  reg         stageDataReg_decodeResult_sWrite;
  wire        dequeue_bits_decodeResult_sWrite_0 = stageDataReg_decodeResult_sWrite;
  reg         stageDataReg_decodeResult_crossRead;
  wire        dequeue_bits_decodeResult_crossRead_0 = stageDataReg_decodeResult_crossRead;
  reg         stageDataReg_decodeResult_crossWrite;
  wire        dequeue_bits_decodeResult_crossWrite_0 = stageDataReg_decodeResult_crossWrite;
  reg         stageDataReg_decodeResult_maskUnit;
  wire        dequeue_bits_decodeResult_maskUnit_0 = stageDataReg_decodeResult_maskUnit;
  reg         stageDataReg_decodeResult_special;
  wire        dequeue_bits_decodeResult_special_0 = stageDataReg_decodeResult_special;
  reg         stageDataReg_decodeResult_saturate;
  wire        dequeue_bits_decodeResult_saturate_0 = stageDataReg_decodeResult_saturate;
  reg         stageDataReg_decodeResult_vwmacc;
  wire        dequeue_bits_decodeResult_vwmacc_0 = stageDataReg_decodeResult_vwmacc;
  reg         stageDataReg_decodeResult_readOnly;
  wire        dequeue_bits_decodeResult_readOnly_0 = stageDataReg_decodeResult_readOnly;
  reg         stageDataReg_decodeResult_maskSource;
  wire        dequeue_bits_decodeResult_maskSource_0 = stageDataReg_decodeResult_maskSource;
  reg         stageDataReg_decodeResult_maskDestination;
  wire        dequeue_bits_decodeResult_maskDestination_0 = stageDataReg_decodeResult_maskDestination;
  reg         stageDataReg_decodeResult_maskLogic;
  wire        dequeue_bits_decodeResult_maskLogic_0 = stageDataReg_decodeResult_maskLogic;
  reg  [3:0]  stageDataReg_decodeResult_uop;
  wire [3:0]  dequeue_bits_decodeResult_uop_0 = stageDataReg_decodeResult_uop;
  reg         stageDataReg_decodeResult_iota;
  wire        dequeue_bits_decodeResult_iota_0 = stageDataReg_decodeResult_iota;
  reg         stageDataReg_decodeResult_mv;
  wire        dequeue_bits_decodeResult_mv_0 = stageDataReg_decodeResult_mv;
  reg         stageDataReg_decodeResult_extend;
  wire        dequeue_bits_decodeResult_extend_0 = stageDataReg_decodeResult_extend;
  reg         stageDataReg_decodeResult_unOrderWrite;
  wire        dequeue_bits_decodeResult_unOrderWrite_0 = stageDataReg_decodeResult_unOrderWrite;
  reg         stageDataReg_decodeResult_compress;
  wire        dequeue_bits_decodeResult_compress_0 = stageDataReg_decodeResult_compress;
  reg         stageDataReg_decodeResult_gather16;
  wire        dequeue_bits_decodeResult_gather16_0 = stageDataReg_decodeResult_gather16;
  reg         stageDataReg_decodeResult_gather;
  wire        dequeue_bits_decodeResult_gather_0 = stageDataReg_decodeResult_gather;
  reg         stageDataReg_decodeResult_slid;
  wire        dequeue_bits_decodeResult_slid_0 = stageDataReg_decodeResult_slid;
  reg         stageDataReg_decodeResult_targetRd;
  wire        dequeue_bits_decodeResult_targetRd_0 = stageDataReg_decodeResult_targetRd;
  reg         stageDataReg_decodeResult_widenReduce;
  wire        dequeue_bits_decodeResult_widenReduce_0 = stageDataReg_decodeResult_widenReduce;
  reg         stageDataReg_decodeResult_red;
  wire        dequeue_bits_decodeResult_red_0 = stageDataReg_decodeResult_red;
  reg         stageDataReg_decodeResult_nr;
  wire        dequeue_bits_decodeResult_nr_0 = stageDataReg_decodeResult_nr;
  reg         stageDataReg_decodeResult_itype;
  wire        dequeue_bits_decodeResult_itype_0 = stageDataReg_decodeResult_itype;
  reg         stageDataReg_decodeResult_unsigned1;
  wire        dequeue_bits_decodeResult_unsigned1_0 = stageDataReg_decodeResult_unsigned1;
  reg         stageDataReg_decodeResult_unsigned0;
  wire        dequeue_bits_decodeResult_unsigned0_0 = stageDataReg_decodeResult_unsigned0;
  reg         stageDataReg_decodeResult_other;
  wire        dequeue_bits_decodeResult_other_0 = stageDataReg_decodeResult_other;
  reg         stageDataReg_decodeResult_multiCycle;
  wire        dequeue_bits_decodeResult_multiCycle_0 = stageDataReg_decodeResult_multiCycle;
  reg         stageDataReg_decodeResult_divider;
  wire        dequeue_bits_decodeResult_divider_0 = stageDataReg_decodeResult_divider;
  reg         stageDataReg_decodeResult_multiplier;
  wire        dequeue_bits_decodeResult_multiplier_0 = stageDataReg_decodeResult_multiplier;
  reg         stageDataReg_decodeResult_shift;
  wire        dequeue_bits_decodeResult_shift_0 = stageDataReg_decodeResult_shift;
  reg         stageDataReg_decodeResult_adder;
  wire        dequeue_bits_decodeResult_adder_0 = stageDataReg_decodeResult_adder;
  reg         stageDataReg_decodeResult_logic;
  wire        dequeue_bits_decodeResult_logic_0 = stageDataReg_decodeResult_logic;
  reg  [3:0]  stageDataReg_laneIndex;
  wire [3:0]  dequeue_bits_laneIndex_0 = stageDataReg_laneIndex;
  reg         stageDataReg_skipRead;
  wire        dequeue_bits_skipRead_0 = stageDataReg_skipRead;
  reg  [4:0]  stageDataReg_vs1;
  wire [4:0]  dequeue_bits_vs1_0 = stageDataReg_vs1;
  reg  [4:0]  stageDataReg_vs2;
  wire [4:0]  dequeue_bits_vs2_0 = stageDataReg_vs2;
  reg  [4:0]  stageDataReg_vd;
  wire [4:0]  dequeue_bits_vd_0 = stageDataReg_vd;
  reg  [2:0]  stageDataReg_vSew1H;
  wire [2:0]  dequeue_bits_vSew1H_0 = stageDataReg_vSew1H;
  reg         stageDataReg_maskNotMaskedElement;
  wire        dequeue_bits_maskNotMaskedElement_0 = stageDataReg_maskNotMaskedElement;
  reg  [11:0] stageDataReg_csr_vl;
  wire [11:0] dequeue_bits_csr_vl_0 = stageDataReg_csr_vl;
  reg  [11:0] stageDataReg_csr_vStart;
  wire [11:0] dequeue_bits_csr_vStart_0 = stageDataReg_csr_vStart;
  reg  [2:0]  stageDataReg_csr_vlmul;
  wire [2:0]  dequeue_bits_csr_vlmul_0 = stageDataReg_csr_vlmul;
  reg  [1:0]  stageDataReg_csr_vSew;
  wire [1:0]  dequeue_bits_csr_vSew_0 = stageDataReg_csr_vSew;
  reg  [1:0]  stageDataReg_csr_vxrm;
  wire [1:0]  dequeue_bits_csr_vxrm_0 = stageDataReg_csr_vxrm;
  reg         stageDataReg_csr_vta;
  wire        dequeue_bits_csr_vta_0 = stageDataReg_csr_vta;
  reg         stageDataReg_csr_vma;
  wire        dequeue_bits_csr_vma_0 = stageDataReg_csr_vma;
  reg         stageDataReg_maskType;
  wire        dequeue_bits_maskType_0 = stageDataReg_maskType;
  reg         stageDataReg_loadStore;
  wire        dequeue_bits_loadStore_0 = stageDataReg_loadStore;
  reg         stageDataReg_bordersForMaskLogic;
  wire        dequeue_bits_bordersForMaskLogic_0 = stageDataReg_bordersForMaskLogic;
  wire [2:0]  filterVec_groupIndex = enqueue_bits_maskIndex_0[4:2];
  wire [7:0]  _filterVec_groupFilter_T = 8'h1 << filterVec_groupIndex;
  wire [7:0]  _filterVec_groupFilter_T_3 = _filterVec_groupFilter_T | {_filterVec_groupFilter_T[6:0], 1'h0};
  wire [7:0]  _filterVec_groupFilter_T_6 = _filterVec_groupFilter_T_3 | {_filterVec_groupFilter_T_3[5:0], 2'h0};
  wire [8:0]  filterVec_groupFilter = {_filterVec_groupFilter_T_6 | {_filterVec_groupFilter_T_6[3:0], 4'h0}, 1'h0};
  wire [31:0] _GEN = enqueue_bits_skipEnable_0 ? enqueue_bits_maskForMaskGroup_0 : 32'hFFFFFFFF;
  wire [31:0] filterVec_maskCorrection;
  assign filterVec_maskCorrection = _GEN;
  wire [31:0] filterVec_maskCorrection_1;
  assign filterVec_maskCorrection_1 = _GEN;
  wire [31:0] filterVec_maskCorrection_2;
  assign filterVec_maskCorrection_2 = _GEN;
  wire [1:0]  filterVec_maskForDataGroup_lo_lo =
    {filterVec_maskCorrection[4] | filterVec_maskCorrection[5] | filterVec_maskCorrection[6] | filterVec_maskCorrection[7],
     filterVec_maskCorrection[0] | filterVec_maskCorrection[1] | filterVec_maskCorrection[2] | filterVec_maskCorrection[3]};
  wire [1:0]  filterVec_maskForDataGroup_lo_hi =
    {filterVec_maskCorrection[12] | filterVec_maskCorrection[13] | filterVec_maskCorrection[14] | filterVec_maskCorrection[15],
     filterVec_maskCorrection[8] | filterVec_maskCorrection[9] | filterVec_maskCorrection[10] | filterVec_maskCorrection[11]};
  wire [3:0]  filterVec_maskForDataGroup_lo = {filterVec_maskForDataGroup_lo_hi, filterVec_maskForDataGroup_lo_lo};
  wire [1:0]  filterVec_maskForDataGroup_hi_lo =
    {filterVec_maskCorrection[20] | filterVec_maskCorrection[21] | filterVec_maskCorrection[22] | filterVec_maskCorrection[23],
     filterVec_maskCorrection[16] | filterVec_maskCorrection[17] | filterVec_maskCorrection[18] | filterVec_maskCorrection[19]};
  wire [1:0]  filterVec_maskForDataGroup_hi_hi =
    {filterVec_maskCorrection[28] | filterVec_maskCorrection[29] | filterVec_maskCorrection[30] | filterVec_maskCorrection[31],
     filterVec_maskCorrection[24] | filterVec_maskCorrection[25] | filterVec_maskCorrection[26] | filterVec_maskCorrection[27]};
  wire [3:0]  filterVec_maskForDataGroup_hi = {filterVec_maskForDataGroup_hi_hi, filterVec_maskForDataGroup_hi_lo};
  wire [7:0]  filterVec_maskForDataGroup = {filterVec_maskForDataGroup_hi, filterVec_maskForDataGroup_lo};
  wire [8:0]  filterVec_groupFilterByMask = {1'h0, filterVec_groupFilter[7:0] & filterVec_maskForDataGroup};
  wire [7:0]  _filterVec_nextDataGroupOH_T_2 = filterVec_groupFilterByMask[7:0] | {filterVec_groupFilterByMask[6:0], 1'h0};
  wire [7:0]  _filterVec_nextDataGroupOH_T_5 = _filterVec_nextDataGroupOH_T_2 | {_filterVec_nextDataGroupOH_T_2[5:0], 2'h0};
  wire [8:0]  filterVec_nextDataGroupOH = {~(_filterVec_nextDataGroupOH_T_5 | {_filterVec_nextDataGroupOH_T_5[3:0], 4'h0}), 1'h1} & filterVec_groupFilterByMask;
  wire        filterVec_0_1 = |filterVec_nextDataGroupOH;
  wire        filterVec_nextElementBaseIndex_hi = filterVec_nextDataGroupOH[8];
  wire [7:0]  filterVec_nextElementBaseIndex_lo = filterVec_nextDataGroupOH[7:0];
  wire [7:0]  _filterVec_nextElementBaseIndex_T_1 = {7'h0, filterVec_nextElementBaseIndex_hi} | filterVec_nextElementBaseIndex_lo;
  wire [3:0]  filterVec_nextElementBaseIndex_hi_1 = _filterVec_nextElementBaseIndex_T_1[7:4];
  wire [3:0]  filterVec_nextElementBaseIndex_lo_1 = _filterVec_nextElementBaseIndex_T_1[3:0];
  wire [3:0]  _filterVec_nextElementBaseIndex_T_3 = filterVec_nextElementBaseIndex_hi_1 | filterVec_nextElementBaseIndex_lo_1;
  wire [1:0]  filterVec_nextElementBaseIndex_hi_2 = _filterVec_nextElementBaseIndex_T_3[3:2];
  wire [1:0]  filterVec_nextElementBaseIndex_lo_2 = _filterVec_nextElementBaseIndex_T_3[1:0];
  wire [5:0]  filterVec_0_2 = {filterVec_nextElementBaseIndex_hi, |filterVec_nextElementBaseIndex_hi_1, |filterVec_nextElementBaseIndex_hi_2, filterVec_nextElementBaseIndex_hi_2[1] | filterVec_nextElementBaseIndex_lo_2[1], 2'h0};
  wire [3:0]  filterVec_groupIndex_1 = enqueue_bits_maskIndex_0[4:1];
  wire [15:0] _filterVec_groupFilter_T_11 = 16'h1 << filterVec_groupIndex_1;
  wire [15:0] _filterVec_groupFilter_T_14 = _filterVec_groupFilter_T_11 | {_filterVec_groupFilter_T_11[14:0], 1'h0};
  wire [15:0] _filterVec_groupFilter_T_17 = _filterVec_groupFilter_T_14 | {_filterVec_groupFilter_T_14[13:0], 2'h0};
  wire [15:0] _filterVec_groupFilter_T_20 = _filterVec_groupFilter_T_17 | {_filterVec_groupFilter_T_17[11:0], 4'h0};
  wire [16:0] filterVec_groupFilter_1 = {_filterVec_groupFilter_T_20 | {_filterVec_groupFilter_T_20[7:0], 8'h0}, 1'h0};
  wire [1:0]  filterVec_maskForDataGroup_lo_lo_lo = {filterVec_maskCorrection_1[2] | filterVec_maskCorrection_1[3], filterVec_maskCorrection_1[0] | filterVec_maskCorrection_1[1]};
  wire [1:0]  filterVec_maskForDataGroup_lo_lo_hi = {filterVec_maskCorrection_1[6] | filterVec_maskCorrection_1[7], filterVec_maskCorrection_1[4] | filterVec_maskCorrection_1[5]};
  wire [3:0]  filterVec_maskForDataGroup_lo_lo_1 = {filterVec_maskForDataGroup_lo_lo_hi, filterVec_maskForDataGroup_lo_lo_lo};
  wire [1:0]  filterVec_maskForDataGroup_lo_hi_lo = {filterVec_maskCorrection_1[10] | filterVec_maskCorrection_1[11], filterVec_maskCorrection_1[8] | filterVec_maskCorrection_1[9]};
  wire [1:0]  filterVec_maskForDataGroup_lo_hi_hi = {filterVec_maskCorrection_1[14] | filterVec_maskCorrection_1[15], filterVec_maskCorrection_1[12] | filterVec_maskCorrection_1[13]};
  wire [3:0]  filterVec_maskForDataGroup_lo_hi_1 = {filterVec_maskForDataGroup_lo_hi_hi, filterVec_maskForDataGroup_lo_hi_lo};
  wire [7:0]  filterVec_maskForDataGroup_lo_1 = {filterVec_maskForDataGroup_lo_hi_1, filterVec_maskForDataGroup_lo_lo_1};
  wire [1:0]  filterVec_maskForDataGroup_hi_lo_lo = {filterVec_maskCorrection_1[18] | filterVec_maskCorrection_1[19], filterVec_maskCorrection_1[16] | filterVec_maskCorrection_1[17]};
  wire [1:0]  filterVec_maskForDataGroup_hi_lo_hi = {filterVec_maskCorrection_1[22] | filterVec_maskCorrection_1[23], filterVec_maskCorrection_1[20] | filterVec_maskCorrection_1[21]};
  wire [3:0]  filterVec_maskForDataGroup_hi_lo_1 = {filterVec_maskForDataGroup_hi_lo_hi, filterVec_maskForDataGroup_hi_lo_lo};
  wire [1:0]  filterVec_maskForDataGroup_hi_hi_lo = {filterVec_maskCorrection_1[26] | filterVec_maskCorrection_1[27], filterVec_maskCorrection_1[24] | filterVec_maskCorrection_1[25]};
  wire [1:0]  filterVec_maskForDataGroup_hi_hi_hi = {filterVec_maskCorrection_1[30] | filterVec_maskCorrection_1[31], filterVec_maskCorrection_1[28] | filterVec_maskCorrection_1[29]};
  wire [3:0]  filterVec_maskForDataGroup_hi_hi_1 = {filterVec_maskForDataGroup_hi_hi_hi, filterVec_maskForDataGroup_hi_hi_lo};
  wire [7:0]  filterVec_maskForDataGroup_hi_1 = {filterVec_maskForDataGroup_hi_hi_1, filterVec_maskForDataGroup_hi_lo_1};
  wire [15:0] filterVec_maskForDataGroup_1 = {filterVec_maskForDataGroup_hi_1, filterVec_maskForDataGroup_lo_1};
  wire [16:0] filterVec_groupFilterByMask_1 = {1'h0, filterVec_groupFilter_1[15:0] & filterVec_maskForDataGroup_1};
  wire [15:0] _filterVec_nextDataGroupOH_T_18 = filterVec_groupFilterByMask_1[15:0] | {filterVec_groupFilterByMask_1[14:0], 1'h0};
  wire [15:0] _filterVec_nextDataGroupOH_T_21 = _filterVec_nextDataGroupOH_T_18 | {_filterVec_nextDataGroupOH_T_18[13:0], 2'h0};
  wire [15:0] _filterVec_nextDataGroupOH_T_24 = _filterVec_nextDataGroupOH_T_21 | {_filterVec_nextDataGroupOH_T_21[11:0], 4'h0};
  wire [16:0] filterVec_nextDataGroupOH_1 = {~(_filterVec_nextDataGroupOH_T_24 | {_filterVec_nextDataGroupOH_T_24[7:0], 8'h0}), 1'h1} & filterVec_groupFilterByMask_1;
  wire        filterVec_1_1 = |filterVec_nextDataGroupOH_1;
  wire        filterVec_nextElementBaseIndex_hi_3 = filterVec_nextDataGroupOH_1[16];
  wire [15:0] filterVec_nextElementBaseIndex_lo_3 = filterVec_nextDataGroupOH_1[15:0];
  wire [15:0] _filterVec_nextElementBaseIndex_T_11 = {15'h0, filterVec_nextElementBaseIndex_hi_3} | filterVec_nextElementBaseIndex_lo_3;
  wire [7:0]  filterVec_nextElementBaseIndex_hi_4 = _filterVec_nextElementBaseIndex_T_11[15:8];
  wire [7:0]  filterVec_nextElementBaseIndex_lo_4 = _filterVec_nextElementBaseIndex_T_11[7:0];
  wire [7:0]  _filterVec_nextElementBaseIndex_T_13 = filterVec_nextElementBaseIndex_hi_4 | filterVec_nextElementBaseIndex_lo_4;
  wire [3:0]  filterVec_nextElementBaseIndex_hi_5 = _filterVec_nextElementBaseIndex_T_13[7:4];
  wire [3:0]  filterVec_nextElementBaseIndex_lo_5 = _filterVec_nextElementBaseIndex_T_13[3:0];
  wire [3:0]  _filterVec_nextElementBaseIndex_T_15 = filterVec_nextElementBaseIndex_hi_5 | filterVec_nextElementBaseIndex_lo_5;
  wire [1:0]  filterVec_nextElementBaseIndex_hi_6 = _filterVec_nextElementBaseIndex_T_15[3:2];
  wire [1:0]  filterVec_nextElementBaseIndex_lo_6 = _filterVec_nextElementBaseIndex_T_15[1:0];
  wire [5:0]  filterVec_1_2 =
    {filterVec_nextElementBaseIndex_hi_3,
     |filterVec_nextElementBaseIndex_hi_4,
     |filterVec_nextElementBaseIndex_hi_5,
     |filterVec_nextElementBaseIndex_hi_6,
     filterVec_nextElementBaseIndex_hi_6[1] | filterVec_nextElementBaseIndex_lo_6[1],
     1'h0};
  wire [31:0] _filterVec_groupFilter_T_25 = 32'h1 << filterVec_groupIndex_2;
  wire [31:0] _filterVec_groupFilter_T_28 = _filterVec_groupFilter_T_25 | {_filterVec_groupFilter_T_25[30:0], 1'h0};
  wire [31:0] _filterVec_groupFilter_T_31 = _filterVec_groupFilter_T_28 | {_filterVec_groupFilter_T_28[29:0], 2'h0};
  wire [31:0] _filterVec_groupFilter_T_34 = _filterVec_groupFilter_T_31 | {_filterVec_groupFilter_T_31[27:0], 4'h0};
  wire [31:0] _filterVec_groupFilter_T_37 = _filterVec_groupFilter_T_34 | {_filterVec_groupFilter_T_34[23:0], 8'h0};
  wire [32:0] filterVec_groupFilter_2 = {_filterVec_groupFilter_T_37 | {_filterVec_groupFilter_T_37[15:0], 16'h0}, 1'h0};
  wire [1:0]  filterVec_maskForDataGroup_lo_lo_lo_lo = filterVec_maskCorrection_2[1:0];
  wire [1:0]  filterVec_maskForDataGroup_lo_lo_lo_hi = filterVec_maskCorrection_2[3:2];
  wire [3:0]  filterVec_maskForDataGroup_lo_lo_lo_1 = {filterVec_maskForDataGroup_lo_lo_lo_hi, filterVec_maskForDataGroup_lo_lo_lo_lo};
  wire [1:0]  filterVec_maskForDataGroup_lo_lo_hi_lo = filterVec_maskCorrection_2[5:4];
  wire [1:0]  filterVec_maskForDataGroup_lo_lo_hi_hi = filterVec_maskCorrection_2[7:6];
  wire [3:0]  filterVec_maskForDataGroup_lo_lo_hi_1 = {filterVec_maskForDataGroup_lo_lo_hi_hi, filterVec_maskForDataGroup_lo_lo_hi_lo};
  wire [7:0]  filterVec_maskForDataGroup_lo_lo_2 = {filterVec_maskForDataGroup_lo_lo_hi_1, filterVec_maskForDataGroup_lo_lo_lo_1};
  wire [1:0]  filterVec_maskForDataGroup_lo_hi_lo_lo = filterVec_maskCorrection_2[9:8];
  wire [1:0]  filterVec_maskForDataGroup_lo_hi_lo_hi = filterVec_maskCorrection_2[11:10];
  wire [3:0]  filterVec_maskForDataGroup_lo_hi_lo_1 = {filterVec_maskForDataGroup_lo_hi_lo_hi, filterVec_maskForDataGroup_lo_hi_lo_lo};
  wire [1:0]  filterVec_maskForDataGroup_lo_hi_hi_lo = filterVec_maskCorrection_2[13:12];
  wire [1:0]  filterVec_maskForDataGroup_lo_hi_hi_hi = filterVec_maskCorrection_2[15:14];
  wire [3:0]  filterVec_maskForDataGroup_lo_hi_hi_1 = {filterVec_maskForDataGroup_lo_hi_hi_hi, filterVec_maskForDataGroup_lo_hi_hi_lo};
  wire [7:0]  filterVec_maskForDataGroup_lo_hi_2 = {filterVec_maskForDataGroup_lo_hi_hi_1, filterVec_maskForDataGroup_lo_hi_lo_1};
  wire [15:0] filterVec_maskForDataGroup_lo_2 = {filterVec_maskForDataGroup_lo_hi_2, filterVec_maskForDataGroup_lo_lo_2};
  wire [1:0]  filterVec_maskForDataGroup_hi_lo_lo_lo = filterVec_maskCorrection_2[17:16];
  wire [1:0]  filterVec_maskForDataGroup_hi_lo_lo_hi = filterVec_maskCorrection_2[19:18];
  wire [3:0]  filterVec_maskForDataGroup_hi_lo_lo_1 = {filterVec_maskForDataGroup_hi_lo_lo_hi, filterVec_maskForDataGroup_hi_lo_lo_lo};
  wire [1:0]  filterVec_maskForDataGroup_hi_lo_hi_lo = filterVec_maskCorrection_2[21:20];
  wire [1:0]  filterVec_maskForDataGroup_hi_lo_hi_hi = filterVec_maskCorrection_2[23:22];
  wire [3:0]  filterVec_maskForDataGroup_hi_lo_hi_1 = {filterVec_maskForDataGroup_hi_lo_hi_hi, filterVec_maskForDataGroup_hi_lo_hi_lo};
  wire [7:0]  filterVec_maskForDataGroup_hi_lo_2 = {filterVec_maskForDataGroup_hi_lo_hi_1, filterVec_maskForDataGroup_hi_lo_lo_1};
  wire [1:0]  filterVec_maskForDataGroup_hi_hi_lo_lo = filterVec_maskCorrection_2[25:24];
  wire [1:0]  filterVec_maskForDataGroup_hi_hi_lo_hi = filterVec_maskCorrection_2[27:26];
  wire [3:0]  filterVec_maskForDataGroup_hi_hi_lo_1 = {filterVec_maskForDataGroup_hi_hi_lo_hi, filterVec_maskForDataGroup_hi_hi_lo_lo};
  wire [1:0]  filterVec_maskForDataGroup_hi_hi_hi_lo = filterVec_maskCorrection_2[29:28];
  wire [1:0]  filterVec_maskForDataGroup_hi_hi_hi_hi = filterVec_maskCorrection_2[31:30];
  wire [3:0]  filterVec_maskForDataGroup_hi_hi_hi_1 = {filterVec_maskForDataGroup_hi_hi_hi_hi, filterVec_maskForDataGroup_hi_hi_hi_lo};
  wire [7:0]  filterVec_maskForDataGroup_hi_hi_2 = {filterVec_maskForDataGroup_hi_hi_hi_1, filterVec_maskForDataGroup_hi_hi_lo_1};
  wire [15:0] filterVec_maskForDataGroup_hi_2 = {filterVec_maskForDataGroup_hi_hi_2, filterVec_maskForDataGroup_hi_lo_2};
  wire [31:0] filterVec_maskForDataGroup_2 = {filterVec_maskForDataGroup_hi_2, filterVec_maskForDataGroup_lo_2};
  wire [32:0] filterVec_groupFilterByMask_2 = {1'h0, filterVec_groupFilter_2[31:0] & filterVec_maskForDataGroup_2};
  wire [31:0] _filterVec_nextDataGroupOH_T_37 = filterVec_groupFilterByMask_2[31:0] | {filterVec_groupFilterByMask_2[30:0], 1'h0};
  wire [31:0] _filterVec_nextDataGroupOH_T_40 = _filterVec_nextDataGroupOH_T_37 | {_filterVec_nextDataGroupOH_T_37[29:0], 2'h0};
  wire [31:0] _filterVec_nextDataGroupOH_T_43 = _filterVec_nextDataGroupOH_T_40 | {_filterVec_nextDataGroupOH_T_40[27:0], 4'h0};
  wire [31:0] _filterVec_nextDataGroupOH_T_46 = _filterVec_nextDataGroupOH_T_43 | {_filterVec_nextDataGroupOH_T_43[23:0], 8'h0};
  wire [32:0] filterVec_nextDataGroupOH_2 = {~(_filterVec_nextDataGroupOH_T_46 | {_filterVec_nextDataGroupOH_T_46[15:0], 16'h0}), 1'h1} & filterVec_groupFilterByMask_2;
  wire        filterVec_2_1 = |filterVec_nextDataGroupOH_2;
  wire        filterVec_nextElementBaseIndex_hi_7 = filterVec_nextDataGroupOH_2[32];
  wire [31:0] filterVec_nextElementBaseIndex_lo_7 = filterVec_nextDataGroupOH_2[31:0];
  wire [31:0] _filterVec_nextElementBaseIndex_T_24 = {31'h0, filterVec_nextElementBaseIndex_hi_7} | filterVec_nextElementBaseIndex_lo_7;
  wire [15:0] filterVec_nextElementBaseIndex_hi_8 = _filterVec_nextElementBaseIndex_T_24[31:16];
  wire [15:0] filterVec_nextElementBaseIndex_lo_8 = _filterVec_nextElementBaseIndex_T_24[15:0];
  wire [15:0] _filterVec_nextElementBaseIndex_T_26 = filterVec_nextElementBaseIndex_hi_8 | filterVec_nextElementBaseIndex_lo_8;
  wire [7:0]  filterVec_nextElementBaseIndex_hi_9 = _filterVec_nextElementBaseIndex_T_26[15:8];
  wire [7:0]  filterVec_nextElementBaseIndex_lo_9 = _filterVec_nextElementBaseIndex_T_26[7:0];
  wire [7:0]  _filterVec_nextElementBaseIndex_T_28 = filterVec_nextElementBaseIndex_hi_9 | filterVec_nextElementBaseIndex_lo_9;
  wire [3:0]  filterVec_nextElementBaseIndex_hi_10 = _filterVec_nextElementBaseIndex_T_28[7:4];
  wire [3:0]  filterVec_nextElementBaseIndex_lo_10 = _filterVec_nextElementBaseIndex_T_28[3:0];
  wire [3:0]  _filterVec_nextElementBaseIndex_T_30 = filterVec_nextElementBaseIndex_hi_10 | filterVec_nextElementBaseIndex_lo_10;
  wire [1:0]  filterVec_nextElementBaseIndex_hi_11 = _filterVec_nextElementBaseIndex_T_30[3:2];
  wire [1:0]  filterVec_nextElementBaseIndex_lo_11 = _filterVec_nextElementBaseIndex_T_30[1:0];
  wire [5:0]  filterVec_2_2 =
    {filterVec_nextElementBaseIndex_hi_7,
     |filterVec_nextElementBaseIndex_hi_8,
     |filterVec_nextElementBaseIndex_hi_9,
     |filterVec_nextElementBaseIndex_hi_10,
     |filterVec_nextElementBaseIndex_hi_11,
     filterVec_nextElementBaseIndex_hi_11[1] | filterVec_nextElementBaseIndex_lo_11[1]};
  wire        nextOrR = enqueue_bits_vSew1H_0[0] & filterVec_0_1 | enqueue_bits_vSew1H_0[1] & filterVec_1_1 | enqueue_bits_vSew1H_0[2] & filterVec_2_1;
  wire        maskGroupWillUpdate = enqueue_bits_decodeResult_maskLogic_0 | ~nextOrR;
  wire [3:0]  elementLengthOH = enqueue_bits_decodeResult_maskLogic_0 ? 4'h1 : {enqueue_bits_vSew1H_0, 1'h0};
  wire [10:0] dataGroupIndex =
    {1'h0,
     {1'h0, {3'h0, elementLengthOH[0] ? enqueue_bits_maskGroupCount_0 : 6'h0} | (elementLengthOH[1] ? {enqueue_bits_maskGroupCount_0, enqueue_bits_maskIndex_0[4:2]} : 9'h0)}
       | (elementLengthOH[2] ? {enqueue_bits_maskGroupCount_0, enqueue_bits_maskIndex_0[4:1]} : 10'h0)} | (elementLengthOH[3] ? {enqueue_bits_maskGroupCount_0, enqueue_bits_maskIndex_0} : 11'h0);
  wire [10:0] _GEN_0 = {5'h0, enqueue_bits_lastGroupForInstruction_0};
  assign _updateLaneState_outOfExecutionRange_output = dataGroupIndex > _GEN_0 | enqueue_bits_instructionFinished_0;
  wire        isTheLastGroup = dataGroupIndex == _GEN_0;
  wire        vlNeedCorrect = enqueue_bits_vSew1H_0[0] & (|(enqueue_bits_csr_vl_0[1:0])) | enqueue_bits_vSew1H_0[1] & enqueue_bits_csr_vl_0[0];
  wire [3:0]  _correctMask_T_2 = 4'h1 << enqueue_bits_csr_vl_0[1:0];
  wire [1:0]  _GEN_1 = _correctMask_T_2[2:1] | _correctMask_T_2[3:2];
  wire [3:0]  correctMask = {1'h0, enqueue_bits_vSew1H_0[0] ? {_correctMask_T_2[3], _GEN_1[1], _GEN_1[0] | _correctMask_T_2[3]} : 3'h0} | {3'h0, enqueue_bits_vSew1H_0[1]};
  wire        needCorrect = isTheLastGroup & enqueue_bits_isLastLaneForInstruction_0 & vlNeedCorrect;
  wire [3:0]  maskCorrect = needCorrect ? correctMask : 4'hF;
  wire [3:0]  crossReadOnlyMask = {4{~_updateLaneState_outOfExecutionRange_output}};
  wire [31:0] _stageWire_maskForMaskInput_T = enqueue_bits_maskForMaskGroup_0 >> enqueue_bits_maskIndex_0;
  assign stageWire_maskForMaskInput = enqueue_bits_maskType_0 ? _stageWire_maskForMaskInput_T[3:0] : 4'h0;
  wire [3:0]  stageWire_boundaryMaskCorrection = maskCorrect & crossReadOnlyMask;
  wire [5:0]  stageWire_groupCounter = dataGroupIndex[5:0];
  wire [31:0] stageWire_readFromScalar =
    (enqueue_bits_vSew1H_0[0] ? {2{{2{enqueue_bits_readFromScalar_0[7:0]}}}} : 32'h0) | (enqueue_bits_vSew1H_0[1] ? {2{enqueue_bits_readFromScalar_0[15:0]}} : 32'h0) | (enqueue_bits_vSew1H_0[2] ? enqueue_bits_readFromScalar_0 : 32'h0);
  wire        stageWire_bordersForMaskLogic = stageWire_groupCounter == enqueue_bits_lastGroupForInstruction_0 & enqueue_bits_isLastLaneForInstruction_0;
  always @(posedge clock) begin
    if (reset) begin
      stageValidReg <= 1'h0;
      stageDataReg_maskForMaskInput <= 4'h0;
      stageDataReg_boundaryMaskCorrection <= 4'h0;
      stageDataReg_groupCounter <= 6'h0;
      stageDataReg_readFromScalar <= 32'h0;
      stageDataReg_instructionIndex <= 3'h0;
      stageDataReg_decodeResult_specialSlot <= 1'h0;
      stageDataReg_decodeResult_topUop <= 5'h0;
      stageDataReg_decodeResult_popCount <= 1'h0;
      stageDataReg_decodeResult_ffo <= 1'h0;
      stageDataReg_decodeResult_average <= 1'h0;
      stageDataReg_decodeResult_reverse <= 1'h0;
      stageDataReg_decodeResult_dontNeedExecuteInLane <= 1'h0;
      stageDataReg_decodeResult_scheduler <= 1'h0;
      stageDataReg_decodeResult_sReadVD <= 1'h0;
      stageDataReg_decodeResult_vtype <= 1'h0;
      stageDataReg_decodeResult_sWrite <= 1'h0;
      stageDataReg_decodeResult_crossRead <= 1'h0;
      stageDataReg_decodeResult_crossWrite <= 1'h0;
      stageDataReg_decodeResult_maskUnit <= 1'h0;
      stageDataReg_decodeResult_special <= 1'h0;
      stageDataReg_decodeResult_saturate <= 1'h0;
      stageDataReg_decodeResult_vwmacc <= 1'h0;
      stageDataReg_decodeResult_readOnly <= 1'h0;
      stageDataReg_decodeResult_maskSource <= 1'h0;
      stageDataReg_decodeResult_maskDestination <= 1'h0;
      stageDataReg_decodeResult_maskLogic <= 1'h0;
      stageDataReg_decodeResult_uop <= 4'h0;
      stageDataReg_decodeResult_iota <= 1'h0;
      stageDataReg_decodeResult_mv <= 1'h0;
      stageDataReg_decodeResult_extend <= 1'h0;
      stageDataReg_decodeResult_unOrderWrite <= 1'h0;
      stageDataReg_decodeResult_compress <= 1'h0;
      stageDataReg_decodeResult_gather16 <= 1'h0;
      stageDataReg_decodeResult_gather <= 1'h0;
      stageDataReg_decodeResult_slid <= 1'h0;
      stageDataReg_decodeResult_targetRd <= 1'h0;
      stageDataReg_decodeResult_widenReduce <= 1'h0;
      stageDataReg_decodeResult_red <= 1'h0;
      stageDataReg_decodeResult_nr <= 1'h0;
      stageDataReg_decodeResult_itype <= 1'h0;
      stageDataReg_decodeResult_unsigned1 <= 1'h0;
      stageDataReg_decodeResult_unsigned0 <= 1'h0;
      stageDataReg_decodeResult_other <= 1'h0;
      stageDataReg_decodeResult_multiCycle <= 1'h0;
      stageDataReg_decodeResult_divider <= 1'h0;
      stageDataReg_decodeResult_multiplier <= 1'h0;
      stageDataReg_decodeResult_shift <= 1'h0;
      stageDataReg_decodeResult_adder <= 1'h0;
      stageDataReg_decodeResult_logic <= 1'h0;
      stageDataReg_laneIndex <= 4'h0;
      stageDataReg_skipRead <= 1'h0;
      stageDataReg_vs1 <= 5'h0;
      stageDataReg_vs2 <= 5'h0;
      stageDataReg_vd <= 5'h0;
      stageDataReg_vSew1H <= 3'h0;
      stageDataReg_maskNotMaskedElement <= 1'h0;
      stageDataReg_csr_vl <= 12'h0;
      stageDataReg_csr_vStart <= 12'h0;
      stageDataReg_csr_vlmul <= 3'h0;
      stageDataReg_csr_vSew <= 2'h0;
      stageDataReg_csr_vxrm <= 2'h0;
      stageDataReg_csr_vta <= 1'h0;
      stageDataReg_csr_vma <= 1'h0;
      stageDataReg_maskType <= 1'h0;
      stageDataReg_loadStore <= 1'h0;
      stageDataReg_bordersForMaskLogic <= 1'h0;
    end
    else begin
      if (enqFire ^ dequeue_ready_0 & dequeue_valid_0)
        stageValidReg <= enqFire;
      if (enqFire) begin
        stageDataReg_maskForMaskInput <= stageWire_maskForMaskInput;
        stageDataReg_boundaryMaskCorrection <= stageWire_boundaryMaskCorrection;
        stageDataReg_groupCounter <= stageWire_groupCounter;
        stageDataReg_readFromScalar <= stageWire_readFromScalar;
        stageDataReg_instructionIndex <= stageWire_instructionIndex;
        stageDataReg_decodeResult_specialSlot <= stageWire_decodeResult_specialSlot;
        stageDataReg_decodeResult_topUop <= stageWire_decodeResult_topUop;
        stageDataReg_decodeResult_popCount <= stageWire_decodeResult_popCount;
        stageDataReg_decodeResult_ffo <= stageWire_decodeResult_ffo;
        stageDataReg_decodeResult_average <= stageWire_decodeResult_average;
        stageDataReg_decodeResult_reverse <= stageWire_decodeResult_reverse;
        stageDataReg_decodeResult_dontNeedExecuteInLane <= stageWire_decodeResult_dontNeedExecuteInLane;
        stageDataReg_decodeResult_scheduler <= stageWire_decodeResult_scheduler;
        stageDataReg_decodeResult_sReadVD <= stageWire_decodeResult_sReadVD;
        stageDataReg_decodeResult_vtype <= stageWire_decodeResult_vtype;
        stageDataReg_decodeResult_sWrite <= stageWire_decodeResult_sWrite;
        stageDataReg_decodeResult_crossRead <= stageWire_decodeResult_crossRead;
        stageDataReg_decodeResult_crossWrite <= stageWire_decodeResult_crossWrite;
        stageDataReg_decodeResult_maskUnit <= stageWire_decodeResult_maskUnit;
        stageDataReg_decodeResult_special <= stageWire_decodeResult_special;
        stageDataReg_decodeResult_saturate <= stageWire_decodeResult_saturate;
        stageDataReg_decodeResult_vwmacc <= stageWire_decodeResult_vwmacc;
        stageDataReg_decodeResult_readOnly <= stageWire_decodeResult_readOnly;
        stageDataReg_decodeResult_maskSource <= stageWire_decodeResult_maskSource;
        stageDataReg_decodeResult_maskDestination <= stageWire_decodeResult_maskDestination;
        stageDataReg_decodeResult_maskLogic <= stageWire_decodeResult_maskLogic;
        stageDataReg_decodeResult_uop <= stageWire_decodeResult_uop;
        stageDataReg_decodeResult_iota <= stageWire_decodeResult_iota;
        stageDataReg_decodeResult_mv <= stageWire_decodeResult_mv;
        stageDataReg_decodeResult_extend <= stageWire_decodeResult_extend;
        stageDataReg_decodeResult_unOrderWrite <= stageWire_decodeResult_unOrderWrite;
        stageDataReg_decodeResult_compress <= stageWire_decodeResult_compress;
        stageDataReg_decodeResult_gather16 <= stageWire_decodeResult_gather16;
        stageDataReg_decodeResult_gather <= stageWire_decodeResult_gather;
        stageDataReg_decodeResult_slid <= stageWire_decodeResult_slid;
        stageDataReg_decodeResult_targetRd <= stageWire_decodeResult_targetRd;
        stageDataReg_decodeResult_widenReduce <= stageWire_decodeResult_widenReduce;
        stageDataReg_decodeResult_red <= stageWire_decodeResult_red;
        stageDataReg_decodeResult_nr <= stageWire_decodeResult_nr;
        stageDataReg_decodeResult_itype <= stageWire_decodeResult_itype;
        stageDataReg_decodeResult_unsigned1 <= stageWire_decodeResult_unsigned1;
        stageDataReg_decodeResult_unsigned0 <= stageWire_decodeResult_unsigned0;
        stageDataReg_decodeResult_other <= stageWire_decodeResult_other;
        stageDataReg_decodeResult_multiCycle <= stageWire_decodeResult_multiCycle;
        stageDataReg_decodeResult_divider <= stageWire_decodeResult_divider;
        stageDataReg_decodeResult_multiplier <= stageWire_decodeResult_multiplier;
        stageDataReg_decodeResult_shift <= stageWire_decodeResult_shift;
        stageDataReg_decodeResult_adder <= stageWire_decodeResult_adder;
        stageDataReg_decodeResult_logic <= stageWire_decodeResult_logic;
        stageDataReg_laneIndex <= stageWire_laneIndex;
        stageDataReg_skipRead <= stageWire_skipRead;
        stageDataReg_vs1 <= stageWire_vs1;
        stageDataReg_vs2 <= stageWire_vs2;
        stageDataReg_vd <= stageWire_vd;
        stageDataReg_vSew1H <= stageWire_vSew1H;
        stageDataReg_maskNotMaskedElement <= stageWire_maskNotMaskedElement;
        stageDataReg_csr_vl <= stageWire_csr_vl;
        stageDataReg_csr_vStart <= stageWire_csr_vStart;
        stageDataReg_csr_vlmul <= stageWire_csr_vlmul;
        stageDataReg_csr_vSew <= stageWire_csr_vSew;
        stageDataReg_csr_vxrm <= stageWire_csr_vxrm;
        stageDataReg_csr_vta <= stageWire_csr_vta;
        stageDataReg_csr_vma <= stageWire_csr_vma;
        stageDataReg_maskType <= stageWire_maskType;
        stageDataReg_loadStore <= stageWire_loadStore;
        stageDataReg_bordersForMaskLogic <= stageWire_bordersForMaskLogic;
      end
    end
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:5];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [2:0] i = 3'h0; i < 3'h6; i += 3'h1) begin
          _RANDOM[i] = `RANDOM;
        end
        stageValidReg = _RANDOM[3'h0][0];
        stageDataReg_maskForMaskInput = _RANDOM[3'h0][4:1];
        stageDataReg_boundaryMaskCorrection = _RANDOM[3'h0][8:5];
        stageDataReg_groupCounter = _RANDOM[3'h0][14:9];
        stageDataReg_readFromScalar = {_RANDOM[3'h0][31:15], _RANDOM[3'h1][14:0]};
        stageDataReg_instructionIndex = _RANDOM[3'h1][17:15];
        stageDataReg_decodeResult_specialSlot = _RANDOM[3'h1][18];
        stageDataReg_decodeResult_topUop = _RANDOM[3'h1][23:19];
        stageDataReg_decodeResult_popCount = _RANDOM[3'h1][24];
        stageDataReg_decodeResult_ffo = _RANDOM[3'h1][25];
        stageDataReg_decodeResult_average = _RANDOM[3'h1][26];
        stageDataReg_decodeResult_reverse = _RANDOM[3'h1][27];
        stageDataReg_decodeResult_dontNeedExecuteInLane = _RANDOM[3'h1][28];
        stageDataReg_decodeResult_scheduler = _RANDOM[3'h1][29];
        stageDataReg_decodeResult_sReadVD = _RANDOM[3'h1][30];
        stageDataReg_decodeResult_vtype = _RANDOM[3'h1][31];
        stageDataReg_decodeResult_sWrite = _RANDOM[3'h2][0];
        stageDataReg_decodeResult_crossRead = _RANDOM[3'h2][1];
        stageDataReg_decodeResult_crossWrite = _RANDOM[3'h2][2];
        stageDataReg_decodeResult_maskUnit = _RANDOM[3'h2][3];
        stageDataReg_decodeResult_special = _RANDOM[3'h2][4];
        stageDataReg_decodeResult_saturate = _RANDOM[3'h2][5];
        stageDataReg_decodeResult_vwmacc = _RANDOM[3'h2][6];
        stageDataReg_decodeResult_readOnly = _RANDOM[3'h2][7];
        stageDataReg_decodeResult_maskSource = _RANDOM[3'h2][8];
        stageDataReg_decodeResult_maskDestination = _RANDOM[3'h2][9];
        stageDataReg_decodeResult_maskLogic = _RANDOM[3'h2][10];
        stageDataReg_decodeResult_uop = _RANDOM[3'h2][14:11];
        stageDataReg_decodeResult_iota = _RANDOM[3'h2][15];
        stageDataReg_decodeResult_mv = _RANDOM[3'h2][16];
        stageDataReg_decodeResult_extend = _RANDOM[3'h2][17];
        stageDataReg_decodeResult_unOrderWrite = _RANDOM[3'h2][18];
        stageDataReg_decodeResult_compress = _RANDOM[3'h2][19];
        stageDataReg_decodeResult_gather16 = _RANDOM[3'h2][20];
        stageDataReg_decodeResult_gather = _RANDOM[3'h2][21];
        stageDataReg_decodeResult_slid = _RANDOM[3'h2][22];
        stageDataReg_decodeResult_targetRd = _RANDOM[3'h2][23];
        stageDataReg_decodeResult_widenReduce = _RANDOM[3'h2][24];
        stageDataReg_decodeResult_red = _RANDOM[3'h2][25];
        stageDataReg_decodeResult_nr = _RANDOM[3'h2][26];
        stageDataReg_decodeResult_itype = _RANDOM[3'h2][27];
        stageDataReg_decodeResult_unsigned1 = _RANDOM[3'h2][28];
        stageDataReg_decodeResult_unsigned0 = _RANDOM[3'h2][29];
        stageDataReg_decodeResult_other = _RANDOM[3'h2][30];
        stageDataReg_decodeResult_multiCycle = _RANDOM[3'h2][31];
        stageDataReg_decodeResult_divider = _RANDOM[3'h3][0];
        stageDataReg_decodeResult_multiplier = _RANDOM[3'h3][1];
        stageDataReg_decodeResult_shift = _RANDOM[3'h3][2];
        stageDataReg_decodeResult_adder = _RANDOM[3'h3][3];
        stageDataReg_decodeResult_logic = _RANDOM[3'h3][4];
        stageDataReg_laneIndex = _RANDOM[3'h3][8:5];
        stageDataReg_skipRead = _RANDOM[3'h3][9];
        stageDataReg_vs1 = _RANDOM[3'h3][14:10];
        stageDataReg_vs2 = _RANDOM[3'h3][19:15];
        stageDataReg_vd = _RANDOM[3'h3][24:20];
        stageDataReg_vSew1H = _RANDOM[3'h3][27:25];
        stageDataReg_maskNotMaskedElement = _RANDOM[3'h3][28];
        stageDataReg_csr_vl = {_RANDOM[3'h3][31:29], _RANDOM[3'h4][8:0]};
        stageDataReg_csr_vStart = _RANDOM[3'h4][20:9];
        stageDataReg_csr_vlmul = _RANDOM[3'h4][23:21];
        stageDataReg_csr_vSew = _RANDOM[3'h4][25:24];
        stageDataReg_csr_vxrm = _RANDOM[3'h4][27:26];
        stageDataReg_csr_vta = _RANDOM[3'h4][28];
        stageDataReg_csr_vma = _RANDOM[3'h4][29];
        stageDataReg_maskType = _RANDOM[3'h4][30];
        stageDataReg_loadStore = _RANDOM[3'h4][31];
        stageDataReg_bordersForMaskLogic = _RANDOM[3'h5][0];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  assign enqueue_ready = enqueue_ready_0;
  assign dequeue_valid = dequeue_valid_0;
  assign dequeue_bits_maskForMaskInput = dequeue_bits_maskForMaskInput_0;
  assign dequeue_bits_boundaryMaskCorrection = dequeue_bits_boundaryMaskCorrection_0;
  assign dequeue_bits_groupCounter = dequeue_bits_groupCounter_0;
  assign dequeue_bits_readFromScalar = dequeue_bits_readFromScalar_0;
  assign dequeue_bits_instructionIndex = dequeue_bits_instructionIndex_0;
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
  assign dequeue_bits_laneIndex = dequeue_bits_laneIndex_0;
  assign dequeue_bits_skipRead = dequeue_bits_skipRead_0;
  assign dequeue_bits_vs1 = dequeue_bits_vs1_0;
  assign dequeue_bits_vs2 = dequeue_bits_vs2_0;
  assign dequeue_bits_vd = dequeue_bits_vd_0;
  assign dequeue_bits_vSew1H = dequeue_bits_vSew1H_0;
  assign dequeue_bits_maskNotMaskedElement = dequeue_bits_maskNotMaskedElement_0;
  assign dequeue_bits_csr_vl = dequeue_bits_csr_vl_0;
  assign dequeue_bits_csr_vStart = dequeue_bits_csr_vStart_0;
  assign dequeue_bits_csr_vlmul = dequeue_bits_csr_vlmul_0;
  assign dequeue_bits_csr_vSew = dequeue_bits_csr_vSew_0;
  assign dequeue_bits_csr_vxrm = dequeue_bits_csr_vxrm_0;
  assign dequeue_bits_csr_vta = dequeue_bits_csr_vta_0;
  assign dequeue_bits_csr_vma = dequeue_bits_csr_vma_0;
  assign dequeue_bits_maskType = dequeue_bits_maskType_0;
  assign dequeue_bits_loadStore = dequeue_bits_loadStore_0;
  assign dequeue_bits_bordersForMaskLogic = dequeue_bits_bordersForMaskLogic_0;
  assign updateLaneState_maskGroupCount = enqueue_bits_maskGroupCount_0 + {5'h0, maskGroupWillUpdate};
  assign updateLaneState_maskIndex =
    enqueue_bits_decodeResult_maskLogic_0 ? 5'h0 : (enqueue_bits_vSew1H_0[0] ? filterVec_0_2[4:0] : 5'h0) | (enqueue_bits_vSew1H_0[1] ? filterVec_1_2[4:0] : 5'h0) | (enqueue_bits_vSew1H_0[2] ? filterVec_2_2[4:0] : 5'h0);
  assign updateLaneState_outOfExecutionRange = _updateLaneState_outOfExecutionRange_output;
  assign updateLaneState_maskExhausted = ~nextOrR;
  assign tokenReport_valid = enqFire;
  assign tokenReport_bits_decodeResult_sWrite = enqueue_bits_decodeResult_sWrite_0;
  assign tokenReport_bits_decodeResult_maskUnit = enqueue_bits_decodeResult_maskUnit_0;
  assign tokenReport_bits_instructionIndex = enqueue_bits_instructionIndex_0;
endmodule

