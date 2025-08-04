module LaneStage2_1(
  input         clock,
                reset,
  output        enqueue_ready,
  input         enqueue_valid,
  input  [31:0] enqueue_bits_src_0,
                enqueue_bits_src_1,
                enqueue_bits_src_2,
  input  [4:0]  enqueue_bits_groupCounter,
  input  [3:0]  enqueue_bits_maskForFilter,
                enqueue_bits_mask,
  input         enqueue_bits_bordersForMaskLogic,
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
  input  [11:0] enqueue_bits_csr_vl,
                enqueue_bits_csr_vStart,
  input  [2:0]  enqueue_bits_csr_vlmul,
  input  [1:0]  enqueue_bits_csr_vSew,
                enqueue_bits_csr_vxrm,
  input         enqueue_bits_csr_vta,
                enqueue_bits_csr_vma,
  input  [2:0]  enqueue_bits_vSew1H,
  input         enqueue_bits_maskType,
                dequeue_ready,
  output        dequeue_valid,
  output [4:0]  dequeue_bits_groupCounter,
  output [3:0]  dequeue_bits_mask,
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
  output [2:0]  dequeue_bits_instructionIndex,
  output        dequeue_bits_loadStore,
  output [4:0]  dequeue_bits_vd
);

  wire        _executionQueue_fifo_empty;
  wire        _executionQueue_fifo_full;
  wire        _executionQueue_fifo_error;
  wire [68:0] _executionQueue_fifo_data_out;
  wire        executionQueue_almostFull;
  wire        executionQueue_almostEmpty;
  wire        enqueue_valid_0 = enqueue_valid;
  wire [31:0] enqueue_bits_src_0_0 = enqueue_bits_src_0;
  wire [31:0] enqueue_bits_src_1_0 = enqueue_bits_src_1;
  wire [31:0] enqueue_bits_src_2_0 = enqueue_bits_src_2;
  wire [4:0]  enqueue_bits_groupCounter_0 = enqueue_bits_groupCounter;
  wire [3:0]  enqueue_bits_maskForFilter_0 = enqueue_bits_maskForFilter;
  wire [3:0]  enqueue_bits_mask_0 = enqueue_bits_mask;
  wire        enqueue_bits_bordersForMaskLogic_0 = enqueue_bits_bordersForMaskLogic;
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
  wire [11:0] enqueue_bits_csr_vl_0 = enqueue_bits_csr_vl;
  wire [11:0] enqueue_bits_csr_vStart_0 = enqueue_bits_csr_vStart;
  wire [2:0]  enqueue_bits_csr_vlmul_0 = enqueue_bits_csr_vlmul;
  wire [1:0]  enqueue_bits_csr_vSew_0 = enqueue_bits_csr_vSew;
  wire [1:0]  enqueue_bits_csr_vxrm_0 = enqueue_bits_csr_vxrm;
  wire        enqueue_bits_csr_vta_0 = enqueue_bits_csr_vta;
  wire        enqueue_bits_csr_vma_0 = enqueue_bits_csr_vma;
  wire [2:0]  enqueue_bits_vSew1H_0 = enqueue_bits_vSew1H;
  wire        enqueue_bits_maskType_0 = enqueue_bits_maskType;
  wire        dequeue_ready_0 = dequeue_ready;
  wire        stageFinish = 1'h1;
  wire        executionQueue_enq_ready;
  wire        executionQueue_enq_valid = enqueue_valid_0;
  wire [4:0]  executionQueue_enq_bits_groupCounter = enqueue_bits_groupCounter_0;
  wire        executionQueue_enq_bits_decodeResult_specialSlot = enqueue_bits_decodeResult_specialSlot_0;
  wire [4:0]  executionQueue_enq_bits_decodeResult_topUop = enqueue_bits_decodeResult_topUop_0;
  wire        executionQueue_enq_bits_decodeResult_popCount = enqueue_bits_decodeResult_popCount_0;
  wire        executionQueue_enq_bits_decodeResult_ffo = enqueue_bits_decodeResult_ffo_0;
  wire        executionQueue_enq_bits_decodeResult_average = enqueue_bits_decodeResult_average_0;
  wire        executionQueue_enq_bits_decodeResult_reverse = enqueue_bits_decodeResult_reverse_0;
  wire        executionQueue_enq_bits_decodeResult_dontNeedExecuteInLane = enqueue_bits_decodeResult_dontNeedExecuteInLane_0;
  wire        executionQueue_enq_bits_decodeResult_scheduler = enqueue_bits_decodeResult_scheduler_0;
  wire        executionQueue_enq_bits_decodeResult_sReadVD = enqueue_bits_decodeResult_sReadVD_0;
  wire        executionQueue_enq_bits_decodeResult_vtype = enqueue_bits_decodeResult_vtype_0;
  wire        executionQueue_enq_bits_decodeResult_sWrite = enqueue_bits_decodeResult_sWrite_0;
  wire        executionQueue_enq_bits_decodeResult_crossRead = enqueue_bits_decodeResult_crossRead_0;
  wire        executionQueue_enq_bits_decodeResult_crossWrite = enqueue_bits_decodeResult_crossWrite_0;
  wire        executionQueue_enq_bits_decodeResult_maskUnit = enqueue_bits_decodeResult_maskUnit_0;
  wire        executionQueue_enq_bits_decodeResult_special = enqueue_bits_decodeResult_special_0;
  wire        executionQueue_enq_bits_decodeResult_saturate = enqueue_bits_decodeResult_saturate_0;
  wire        executionQueue_enq_bits_decodeResult_vwmacc = enqueue_bits_decodeResult_vwmacc_0;
  wire        executionQueue_enq_bits_decodeResult_readOnly = enqueue_bits_decodeResult_readOnly_0;
  wire        executionQueue_enq_bits_decodeResult_maskSource = enqueue_bits_decodeResult_maskSource_0;
  wire        executionQueue_enq_bits_decodeResult_maskDestination = enqueue_bits_decodeResult_maskDestination_0;
  wire        executionQueue_enq_bits_decodeResult_maskLogic = enqueue_bits_decodeResult_maskLogic_0;
  wire [3:0]  executionQueue_enq_bits_decodeResult_uop = enqueue_bits_decodeResult_uop_0;
  wire        executionQueue_enq_bits_decodeResult_iota = enqueue_bits_decodeResult_iota_0;
  wire        executionQueue_enq_bits_decodeResult_mv = enqueue_bits_decodeResult_mv_0;
  wire        executionQueue_enq_bits_decodeResult_extend = enqueue_bits_decodeResult_extend_0;
  wire        executionQueue_enq_bits_decodeResult_unOrderWrite = enqueue_bits_decodeResult_unOrderWrite_0;
  wire        executionQueue_enq_bits_decodeResult_compress = enqueue_bits_decodeResult_compress_0;
  wire        executionQueue_enq_bits_decodeResult_gather16 = enqueue_bits_decodeResult_gather16_0;
  wire        executionQueue_enq_bits_decodeResult_gather = enqueue_bits_decodeResult_gather_0;
  wire        executionQueue_enq_bits_decodeResult_slid = enqueue_bits_decodeResult_slid_0;
  wire        executionQueue_enq_bits_decodeResult_targetRd = enqueue_bits_decodeResult_targetRd_0;
  wire        executionQueue_enq_bits_decodeResult_widenReduce = enqueue_bits_decodeResult_widenReduce_0;
  wire        executionQueue_enq_bits_decodeResult_red = enqueue_bits_decodeResult_red_0;
  wire        executionQueue_enq_bits_decodeResult_nr = enqueue_bits_decodeResult_nr_0;
  wire        executionQueue_enq_bits_decodeResult_itype = enqueue_bits_decodeResult_itype_0;
  wire        executionQueue_enq_bits_decodeResult_unsigned1 = enqueue_bits_decodeResult_unsigned1_0;
  wire        executionQueue_enq_bits_decodeResult_unsigned0 = enqueue_bits_decodeResult_unsigned0_0;
  wire        executionQueue_enq_bits_decodeResult_other = enqueue_bits_decodeResult_other_0;
  wire        executionQueue_enq_bits_decodeResult_multiCycle = enqueue_bits_decodeResult_multiCycle_0;
  wire        executionQueue_enq_bits_decodeResult_divider = enqueue_bits_decodeResult_divider_0;
  wire        executionQueue_enq_bits_decodeResult_multiplier = enqueue_bits_decodeResult_multiplier_0;
  wire        executionQueue_enq_bits_decodeResult_shift = enqueue_bits_decodeResult_shift_0;
  wire        executionQueue_enq_bits_decodeResult_adder = enqueue_bits_decodeResult_adder_0;
  wire        executionQueue_enq_bits_decodeResult_logic = enqueue_bits_decodeResult_logic_0;
  wire [2:0]  executionQueue_enq_bits_instructionIndex = enqueue_bits_instructionIndex_0;
  wire        executionQueue_enq_bits_loadStore = enqueue_bits_loadStore_0;
  wire [4:0]  executionQueue_enq_bits_vd = enqueue_bits_vd_0;
  wire        executionQueue_deq_ready = dequeue_ready_0;
  wire        executionQueue_deq_valid;
  wire [4:0]  executionQueue_deq_bits_groupCounter;
  wire [3:0]  executionQueue_deq_bits_mask;
  wire        executionQueue_deq_bits_decodeResult_specialSlot;
  wire [4:0]  executionQueue_deq_bits_decodeResult_topUop;
  wire        executionQueue_deq_bits_decodeResult_popCount;
  wire        executionQueue_deq_bits_decodeResult_ffo;
  wire        executionQueue_deq_bits_decodeResult_average;
  wire        executionQueue_deq_bits_decodeResult_reverse;
  wire        executionQueue_deq_bits_decodeResult_dontNeedExecuteInLane;
  wire        executionQueue_deq_bits_decodeResult_scheduler;
  wire        executionQueue_deq_bits_decodeResult_sReadVD;
  wire        executionQueue_deq_bits_decodeResult_vtype;
  wire        executionQueue_deq_bits_decodeResult_sWrite;
  wire        executionQueue_deq_bits_decodeResult_crossRead;
  wire        executionQueue_deq_bits_decodeResult_crossWrite;
  wire        executionQueue_deq_bits_decodeResult_maskUnit;
  wire        executionQueue_deq_bits_decodeResult_special;
  wire        executionQueue_deq_bits_decodeResult_saturate;
  wire        executionQueue_deq_bits_decodeResult_vwmacc;
  wire        executionQueue_deq_bits_decodeResult_readOnly;
  wire        executionQueue_deq_bits_decodeResult_maskSource;
  wire        executionQueue_deq_bits_decodeResult_maskDestination;
  wire        executionQueue_deq_bits_decodeResult_maskLogic;
  wire [3:0]  executionQueue_deq_bits_decodeResult_uop;
  wire        executionQueue_deq_bits_decodeResult_iota;
  wire        executionQueue_deq_bits_decodeResult_mv;
  wire        executionQueue_deq_bits_decodeResult_extend;
  wire        executionQueue_deq_bits_decodeResult_unOrderWrite;
  wire        executionQueue_deq_bits_decodeResult_compress;
  wire        executionQueue_deq_bits_decodeResult_gather16;
  wire        executionQueue_deq_bits_decodeResult_gather;
  wire        executionQueue_deq_bits_decodeResult_slid;
  wire        executionQueue_deq_bits_decodeResult_targetRd;
  wire        executionQueue_deq_bits_decodeResult_widenReduce;
  wire        executionQueue_deq_bits_decodeResult_red;
  wire        executionQueue_deq_bits_decodeResult_nr;
  wire        executionQueue_deq_bits_decodeResult_itype;
  wire        executionQueue_deq_bits_decodeResult_unsigned1;
  wire        executionQueue_deq_bits_decodeResult_unsigned0;
  wire        executionQueue_deq_bits_decodeResult_other;
  wire        executionQueue_deq_bits_decodeResult_multiCycle;
  wire        executionQueue_deq_bits_decodeResult_divider;
  wire        executionQueue_deq_bits_decodeResult_multiplier;
  wire        executionQueue_deq_bits_decodeResult_shift;
  wire        executionQueue_deq_bits_decodeResult_adder;
  wire        executionQueue_deq_bits_decodeResult_logic;
  wire [2:0]  executionQueue_deq_bits_instructionIndex;
  wire        executionQueue_deq_bits_loadStore;
  wire [4:0]  executionQueue_deq_bits_vd;
  wire        enqueue_ready_0 = executionQueue_enq_ready;
  assign executionQueue_deq_valid = ~_executionQueue_fifo_empty;
  wire        dequeue_valid_0 = executionQueue_deq_valid;
  wire [4:0]  executionQueue_dataOut_groupCounter;
  wire [4:0]  dequeue_bits_groupCounter_0 = executionQueue_deq_bits_groupCounter;
  wire [3:0]  executionQueue_dataOut_mask;
  wire [3:0]  dequeue_bits_mask_0 = executionQueue_deq_bits_mask;
  wire        executionQueue_dataOut_decodeResult_specialSlot;
  wire        dequeue_bits_decodeResult_specialSlot_0 = executionQueue_deq_bits_decodeResult_specialSlot;
  wire [4:0]  executionQueue_dataOut_decodeResult_topUop;
  wire [4:0]  dequeue_bits_decodeResult_topUop_0 = executionQueue_deq_bits_decodeResult_topUop;
  wire        executionQueue_dataOut_decodeResult_popCount;
  wire        dequeue_bits_decodeResult_popCount_0 = executionQueue_deq_bits_decodeResult_popCount;
  wire        executionQueue_dataOut_decodeResult_ffo;
  wire        dequeue_bits_decodeResult_ffo_0 = executionQueue_deq_bits_decodeResult_ffo;
  wire        executionQueue_dataOut_decodeResult_average;
  wire        dequeue_bits_decodeResult_average_0 = executionQueue_deq_bits_decodeResult_average;
  wire        executionQueue_dataOut_decodeResult_reverse;
  wire        dequeue_bits_decodeResult_reverse_0 = executionQueue_deq_bits_decodeResult_reverse;
  wire        executionQueue_dataOut_decodeResult_dontNeedExecuteInLane;
  wire        dequeue_bits_decodeResult_dontNeedExecuteInLane_0 = executionQueue_deq_bits_decodeResult_dontNeedExecuteInLane;
  wire        executionQueue_dataOut_decodeResult_scheduler;
  wire        dequeue_bits_decodeResult_scheduler_0 = executionQueue_deq_bits_decodeResult_scheduler;
  wire        executionQueue_dataOut_decodeResult_sReadVD;
  wire        dequeue_bits_decodeResult_sReadVD_0 = executionQueue_deq_bits_decodeResult_sReadVD;
  wire        executionQueue_dataOut_decodeResult_vtype;
  wire        dequeue_bits_decodeResult_vtype_0 = executionQueue_deq_bits_decodeResult_vtype;
  wire        executionQueue_dataOut_decodeResult_sWrite;
  wire        dequeue_bits_decodeResult_sWrite_0 = executionQueue_deq_bits_decodeResult_sWrite;
  wire        executionQueue_dataOut_decodeResult_crossRead;
  wire        dequeue_bits_decodeResult_crossRead_0 = executionQueue_deq_bits_decodeResult_crossRead;
  wire        executionQueue_dataOut_decodeResult_crossWrite;
  wire        dequeue_bits_decodeResult_crossWrite_0 = executionQueue_deq_bits_decodeResult_crossWrite;
  wire        executionQueue_dataOut_decodeResult_maskUnit;
  wire        dequeue_bits_decodeResult_maskUnit_0 = executionQueue_deq_bits_decodeResult_maskUnit;
  wire        executionQueue_dataOut_decodeResult_special;
  wire        dequeue_bits_decodeResult_special_0 = executionQueue_deq_bits_decodeResult_special;
  wire        executionQueue_dataOut_decodeResult_saturate;
  wire        dequeue_bits_decodeResult_saturate_0 = executionQueue_deq_bits_decodeResult_saturate;
  wire        executionQueue_dataOut_decodeResult_vwmacc;
  wire        dequeue_bits_decodeResult_vwmacc_0 = executionQueue_deq_bits_decodeResult_vwmacc;
  wire        executionQueue_dataOut_decodeResult_readOnly;
  wire        dequeue_bits_decodeResult_readOnly_0 = executionQueue_deq_bits_decodeResult_readOnly;
  wire        executionQueue_dataOut_decodeResult_maskSource;
  wire        dequeue_bits_decodeResult_maskSource_0 = executionQueue_deq_bits_decodeResult_maskSource;
  wire        executionQueue_dataOut_decodeResult_maskDestination;
  wire        dequeue_bits_decodeResult_maskDestination_0 = executionQueue_deq_bits_decodeResult_maskDestination;
  wire        executionQueue_dataOut_decodeResult_maskLogic;
  wire        dequeue_bits_decodeResult_maskLogic_0 = executionQueue_deq_bits_decodeResult_maskLogic;
  wire [3:0]  executionQueue_dataOut_decodeResult_uop;
  wire [3:0]  dequeue_bits_decodeResult_uop_0 = executionQueue_deq_bits_decodeResult_uop;
  wire        executionQueue_dataOut_decodeResult_iota;
  wire        dequeue_bits_decodeResult_iota_0 = executionQueue_deq_bits_decodeResult_iota;
  wire        executionQueue_dataOut_decodeResult_mv;
  wire        dequeue_bits_decodeResult_mv_0 = executionQueue_deq_bits_decodeResult_mv;
  wire        executionQueue_dataOut_decodeResult_extend;
  wire        dequeue_bits_decodeResult_extend_0 = executionQueue_deq_bits_decodeResult_extend;
  wire        executionQueue_dataOut_decodeResult_unOrderWrite;
  wire        dequeue_bits_decodeResult_unOrderWrite_0 = executionQueue_deq_bits_decodeResult_unOrderWrite;
  wire        executionQueue_dataOut_decodeResult_compress;
  wire        dequeue_bits_decodeResult_compress_0 = executionQueue_deq_bits_decodeResult_compress;
  wire        executionQueue_dataOut_decodeResult_gather16;
  wire        dequeue_bits_decodeResult_gather16_0 = executionQueue_deq_bits_decodeResult_gather16;
  wire        executionQueue_dataOut_decodeResult_gather;
  wire        dequeue_bits_decodeResult_gather_0 = executionQueue_deq_bits_decodeResult_gather;
  wire        executionQueue_dataOut_decodeResult_slid;
  wire        dequeue_bits_decodeResult_slid_0 = executionQueue_deq_bits_decodeResult_slid;
  wire        executionQueue_dataOut_decodeResult_targetRd;
  wire        dequeue_bits_decodeResult_targetRd_0 = executionQueue_deq_bits_decodeResult_targetRd;
  wire        executionQueue_dataOut_decodeResult_widenReduce;
  wire        dequeue_bits_decodeResult_widenReduce_0 = executionQueue_deq_bits_decodeResult_widenReduce;
  wire        executionQueue_dataOut_decodeResult_red;
  wire        dequeue_bits_decodeResult_red_0 = executionQueue_deq_bits_decodeResult_red;
  wire        executionQueue_dataOut_decodeResult_nr;
  wire        dequeue_bits_decodeResult_nr_0 = executionQueue_deq_bits_decodeResult_nr;
  wire        executionQueue_dataOut_decodeResult_itype;
  wire        dequeue_bits_decodeResult_itype_0 = executionQueue_deq_bits_decodeResult_itype;
  wire        executionQueue_dataOut_decodeResult_unsigned1;
  wire        dequeue_bits_decodeResult_unsigned1_0 = executionQueue_deq_bits_decodeResult_unsigned1;
  wire        executionQueue_dataOut_decodeResult_unsigned0;
  wire        dequeue_bits_decodeResult_unsigned0_0 = executionQueue_deq_bits_decodeResult_unsigned0;
  wire        executionQueue_dataOut_decodeResult_other;
  wire        dequeue_bits_decodeResult_other_0 = executionQueue_deq_bits_decodeResult_other;
  wire        executionQueue_dataOut_decodeResult_multiCycle;
  wire        dequeue_bits_decodeResult_multiCycle_0 = executionQueue_deq_bits_decodeResult_multiCycle;
  wire        executionQueue_dataOut_decodeResult_divider;
  wire        dequeue_bits_decodeResult_divider_0 = executionQueue_deq_bits_decodeResult_divider;
  wire        executionQueue_dataOut_decodeResult_multiplier;
  wire        dequeue_bits_decodeResult_multiplier_0 = executionQueue_deq_bits_decodeResult_multiplier;
  wire        executionQueue_dataOut_decodeResult_shift;
  wire        dequeue_bits_decodeResult_shift_0 = executionQueue_deq_bits_decodeResult_shift;
  wire        executionQueue_dataOut_decodeResult_adder;
  wire        dequeue_bits_decodeResult_adder_0 = executionQueue_deq_bits_decodeResult_adder;
  wire        executionQueue_dataOut_decodeResult_logic;
  wire        dequeue_bits_decodeResult_logic_0 = executionQueue_deq_bits_decodeResult_logic;
  wire [2:0]  executionQueue_dataOut_instructionIndex;
  wire [2:0]  dequeue_bits_instructionIndex_0 = executionQueue_deq_bits_instructionIndex;
  wire        executionQueue_dataOut_loadStore;
  wire        dequeue_bits_loadStore_0 = executionQueue_deq_bits_loadStore;
  wire [4:0]  executionQueue_dataOut_vd;
  wire [4:0]  dequeue_bits_vd_0 = executionQueue_deq_bits_vd;
  wire [1:0]  executionQueue_dataIn_lo_lo_lo_lo = {executionQueue_enq_bits_decodeResult_adder, executionQueue_enq_bits_decodeResult_logic};
  wire [1:0]  executionQueue_dataIn_lo_lo_lo_hi_hi = {executionQueue_enq_bits_decodeResult_divider, executionQueue_enq_bits_decodeResult_multiplier};
  wire [2:0]  executionQueue_dataIn_lo_lo_lo_hi = {executionQueue_dataIn_lo_lo_lo_hi_hi, executionQueue_enq_bits_decodeResult_shift};
  wire [4:0]  executionQueue_dataIn_lo_lo_lo = {executionQueue_dataIn_lo_lo_lo_hi, executionQueue_dataIn_lo_lo_lo_lo};
  wire [1:0]  executionQueue_dataIn_lo_lo_hi_lo_hi = {executionQueue_enq_bits_decodeResult_unsigned0, executionQueue_enq_bits_decodeResult_other};
  wire [2:0]  executionQueue_dataIn_lo_lo_hi_lo = {executionQueue_dataIn_lo_lo_hi_lo_hi, executionQueue_enq_bits_decodeResult_multiCycle};
  wire [1:0]  executionQueue_dataIn_lo_lo_hi_hi_hi = {executionQueue_enq_bits_decodeResult_nr, executionQueue_enq_bits_decodeResult_itype};
  wire [2:0]  executionQueue_dataIn_lo_lo_hi_hi = {executionQueue_dataIn_lo_lo_hi_hi_hi, executionQueue_enq_bits_decodeResult_unsigned1};
  wire [5:0]  executionQueue_dataIn_lo_lo_hi = {executionQueue_dataIn_lo_lo_hi_hi, executionQueue_dataIn_lo_lo_hi_lo};
  wire [10:0] executionQueue_dataIn_lo_lo = {executionQueue_dataIn_lo_lo_hi, executionQueue_dataIn_lo_lo_lo};
  wire [1:0]  executionQueue_dataIn_lo_hi_lo_lo = {executionQueue_enq_bits_decodeResult_widenReduce, executionQueue_enq_bits_decodeResult_red};
  wire [1:0]  executionQueue_dataIn_lo_hi_lo_hi_hi = {executionQueue_enq_bits_decodeResult_gather, executionQueue_enq_bits_decodeResult_slid};
  wire [2:0]  executionQueue_dataIn_lo_hi_lo_hi = {executionQueue_dataIn_lo_hi_lo_hi_hi, executionQueue_enq_bits_decodeResult_targetRd};
  wire [4:0]  executionQueue_dataIn_lo_hi_lo = {executionQueue_dataIn_lo_hi_lo_hi, executionQueue_dataIn_lo_hi_lo_lo};
  wire [1:0]  executionQueue_dataIn_lo_hi_hi_lo_hi = {executionQueue_enq_bits_decodeResult_unOrderWrite, executionQueue_enq_bits_decodeResult_compress};
  wire [2:0]  executionQueue_dataIn_lo_hi_hi_lo = {executionQueue_dataIn_lo_hi_hi_lo_hi, executionQueue_enq_bits_decodeResult_gather16};
  wire [1:0]  executionQueue_dataIn_lo_hi_hi_hi_hi = {executionQueue_enq_bits_decodeResult_iota, executionQueue_enq_bits_decodeResult_mv};
  wire [2:0]  executionQueue_dataIn_lo_hi_hi_hi = {executionQueue_dataIn_lo_hi_hi_hi_hi, executionQueue_enq_bits_decodeResult_extend};
  wire [5:0]  executionQueue_dataIn_lo_hi_hi = {executionQueue_dataIn_lo_hi_hi_hi, executionQueue_dataIn_lo_hi_hi_lo};
  wire [10:0] executionQueue_dataIn_lo_hi = {executionQueue_dataIn_lo_hi_hi, executionQueue_dataIn_lo_hi_lo};
  wire [21:0] executionQueue_dataIn_lo = {executionQueue_dataIn_lo_hi, executionQueue_dataIn_lo_lo};
  wire [4:0]  executionQueue_dataIn_hi_lo_lo_lo = {executionQueue_enq_bits_decodeResult_maskLogic, executionQueue_enq_bits_decodeResult_uop};
  wire [1:0]  executionQueue_dataIn_hi_lo_lo_hi_hi = {executionQueue_enq_bits_decodeResult_readOnly, executionQueue_enq_bits_decodeResult_maskSource};
  wire [2:0]  executionQueue_dataIn_hi_lo_lo_hi = {executionQueue_dataIn_hi_lo_lo_hi_hi, executionQueue_enq_bits_decodeResult_maskDestination};
  wire [7:0]  executionQueue_dataIn_hi_lo_lo = {executionQueue_dataIn_hi_lo_lo_hi, executionQueue_dataIn_hi_lo_lo_lo};
  wire [1:0]  executionQueue_dataIn_hi_lo_hi_lo_hi = {executionQueue_enq_bits_decodeResult_special, executionQueue_enq_bits_decodeResult_saturate};
  wire [2:0]  executionQueue_dataIn_hi_lo_hi_lo = {executionQueue_dataIn_hi_lo_hi_lo_hi, executionQueue_enq_bits_decodeResult_vwmacc};
  wire [1:0]  executionQueue_dataIn_hi_lo_hi_hi_hi = {executionQueue_enq_bits_decodeResult_crossRead, executionQueue_enq_bits_decodeResult_crossWrite};
  wire [2:0]  executionQueue_dataIn_hi_lo_hi_hi = {executionQueue_dataIn_hi_lo_hi_hi_hi, executionQueue_enq_bits_decodeResult_maskUnit};
  wire [5:0]  executionQueue_dataIn_hi_lo_hi = {executionQueue_dataIn_hi_lo_hi_hi, executionQueue_dataIn_hi_lo_hi_lo};
  wire [13:0] executionQueue_dataIn_hi_lo = {executionQueue_dataIn_hi_lo_hi, executionQueue_dataIn_hi_lo_lo};
  wire [1:0]  executionQueue_dataIn_hi_hi_lo_lo = {executionQueue_enq_bits_decodeResult_vtype, executionQueue_enq_bits_decodeResult_sWrite};
  wire [1:0]  executionQueue_dataIn_hi_hi_lo_hi_hi = {executionQueue_enq_bits_decodeResult_dontNeedExecuteInLane, executionQueue_enq_bits_decodeResult_scheduler};
  wire [2:0]  executionQueue_dataIn_hi_hi_lo_hi = {executionQueue_dataIn_hi_hi_lo_hi_hi, executionQueue_enq_bits_decodeResult_sReadVD};
  wire [4:0]  executionQueue_dataIn_hi_hi_lo = {executionQueue_dataIn_hi_hi_lo_hi, executionQueue_dataIn_hi_hi_lo_lo};
  wire [1:0]  executionQueue_dataIn_hi_hi_hi_lo_hi = {executionQueue_enq_bits_decodeResult_ffo, executionQueue_enq_bits_decodeResult_average};
  wire [2:0]  executionQueue_dataIn_hi_hi_hi_lo = {executionQueue_dataIn_hi_hi_hi_lo_hi, executionQueue_enq_bits_decodeResult_reverse};
  wire [5:0]  executionQueue_dataIn_hi_hi_hi_hi_hi = {executionQueue_enq_bits_decodeResult_specialSlot, executionQueue_enq_bits_decodeResult_topUop};
  wire [6:0]  executionQueue_dataIn_hi_hi_hi_hi = {executionQueue_dataIn_hi_hi_hi_hi_hi, executionQueue_enq_bits_decodeResult_popCount};
  wire [9:0]  executionQueue_dataIn_hi_hi_hi = {executionQueue_dataIn_hi_hi_hi_hi, executionQueue_dataIn_hi_hi_hi_lo};
  wire [14:0] executionQueue_dataIn_hi_hi = {executionQueue_dataIn_hi_hi_hi, executionQueue_dataIn_hi_hi_lo};
  wire [28:0] executionQueue_dataIn_hi = {executionQueue_dataIn_hi_hi, executionQueue_dataIn_hi_lo};
  wire [3:0]  executionQueue_dataIn_lo_hi_1 = {executionQueue_enq_bits_instructionIndex, executionQueue_enq_bits_loadStore};
  wire [8:0]  executionQueue_dataIn_lo_1 = {executionQueue_dataIn_lo_hi_1, executionQueue_enq_bits_vd};
  wire [3:0]  executionQueue_enq_bits_mask;
  wire [8:0]  executionQueue_dataIn_hi_hi_1 = {executionQueue_enq_bits_groupCounter, executionQueue_enq_bits_mask};
  wire [59:0] executionQueue_dataIn_hi_1 = {executionQueue_dataIn_hi_hi_1, executionQueue_dataIn_hi, executionQueue_dataIn_lo};
  wire [68:0] executionQueue_dataIn = {executionQueue_dataIn_hi_1, executionQueue_dataIn_lo_1};
  assign executionQueue_dataOut_vd = _executionQueue_fifo_data_out[4:0];
  assign executionQueue_dataOut_loadStore = _executionQueue_fifo_data_out[5];
  assign executionQueue_dataOut_instructionIndex = _executionQueue_fifo_data_out[8:6];
  assign executionQueue_dataOut_decodeResult_logic = _executionQueue_fifo_data_out[9];
  assign executionQueue_dataOut_decodeResult_adder = _executionQueue_fifo_data_out[10];
  assign executionQueue_dataOut_decodeResult_shift = _executionQueue_fifo_data_out[11];
  assign executionQueue_dataOut_decodeResult_multiplier = _executionQueue_fifo_data_out[12];
  assign executionQueue_dataOut_decodeResult_divider = _executionQueue_fifo_data_out[13];
  assign executionQueue_dataOut_decodeResult_multiCycle = _executionQueue_fifo_data_out[14];
  assign executionQueue_dataOut_decodeResult_other = _executionQueue_fifo_data_out[15];
  assign executionQueue_dataOut_decodeResult_unsigned0 = _executionQueue_fifo_data_out[16];
  assign executionQueue_dataOut_decodeResult_unsigned1 = _executionQueue_fifo_data_out[17];
  assign executionQueue_dataOut_decodeResult_itype = _executionQueue_fifo_data_out[18];
  assign executionQueue_dataOut_decodeResult_nr = _executionQueue_fifo_data_out[19];
  assign executionQueue_dataOut_decodeResult_red = _executionQueue_fifo_data_out[20];
  assign executionQueue_dataOut_decodeResult_widenReduce = _executionQueue_fifo_data_out[21];
  assign executionQueue_dataOut_decodeResult_targetRd = _executionQueue_fifo_data_out[22];
  assign executionQueue_dataOut_decodeResult_slid = _executionQueue_fifo_data_out[23];
  assign executionQueue_dataOut_decodeResult_gather = _executionQueue_fifo_data_out[24];
  assign executionQueue_dataOut_decodeResult_gather16 = _executionQueue_fifo_data_out[25];
  assign executionQueue_dataOut_decodeResult_compress = _executionQueue_fifo_data_out[26];
  assign executionQueue_dataOut_decodeResult_unOrderWrite = _executionQueue_fifo_data_out[27];
  assign executionQueue_dataOut_decodeResult_extend = _executionQueue_fifo_data_out[28];
  assign executionQueue_dataOut_decodeResult_mv = _executionQueue_fifo_data_out[29];
  assign executionQueue_dataOut_decodeResult_iota = _executionQueue_fifo_data_out[30];
  assign executionQueue_dataOut_decodeResult_uop = _executionQueue_fifo_data_out[34:31];
  assign executionQueue_dataOut_decodeResult_maskLogic = _executionQueue_fifo_data_out[35];
  assign executionQueue_dataOut_decodeResult_maskDestination = _executionQueue_fifo_data_out[36];
  assign executionQueue_dataOut_decodeResult_maskSource = _executionQueue_fifo_data_out[37];
  assign executionQueue_dataOut_decodeResult_readOnly = _executionQueue_fifo_data_out[38];
  assign executionQueue_dataOut_decodeResult_vwmacc = _executionQueue_fifo_data_out[39];
  assign executionQueue_dataOut_decodeResult_saturate = _executionQueue_fifo_data_out[40];
  assign executionQueue_dataOut_decodeResult_special = _executionQueue_fifo_data_out[41];
  assign executionQueue_dataOut_decodeResult_maskUnit = _executionQueue_fifo_data_out[42];
  assign executionQueue_dataOut_decodeResult_crossWrite = _executionQueue_fifo_data_out[43];
  assign executionQueue_dataOut_decodeResult_crossRead = _executionQueue_fifo_data_out[44];
  assign executionQueue_dataOut_decodeResult_sWrite = _executionQueue_fifo_data_out[45];
  assign executionQueue_dataOut_decodeResult_vtype = _executionQueue_fifo_data_out[46];
  assign executionQueue_dataOut_decodeResult_sReadVD = _executionQueue_fifo_data_out[47];
  assign executionQueue_dataOut_decodeResult_scheduler = _executionQueue_fifo_data_out[48];
  assign executionQueue_dataOut_decodeResult_dontNeedExecuteInLane = _executionQueue_fifo_data_out[49];
  assign executionQueue_dataOut_decodeResult_reverse = _executionQueue_fifo_data_out[50];
  assign executionQueue_dataOut_decodeResult_average = _executionQueue_fifo_data_out[51];
  assign executionQueue_dataOut_decodeResult_ffo = _executionQueue_fifo_data_out[52];
  assign executionQueue_dataOut_decodeResult_popCount = _executionQueue_fifo_data_out[53];
  assign executionQueue_dataOut_decodeResult_topUop = _executionQueue_fifo_data_out[58:54];
  assign executionQueue_dataOut_decodeResult_specialSlot = _executionQueue_fifo_data_out[59];
  assign executionQueue_dataOut_mask = _executionQueue_fifo_data_out[63:60];
  assign executionQueue_dataOut_groupCounter = _executionQueue_fifo_data_out[68:64];
  assign executionQueue_deq_bits_groupCounter = executionQueue_dataOut_groupCounter;
  assign executionQueue_deq_bits_mask = executionQueue_dataOut_mask;
  assign executionQueue_deq_bits_decodeResult_specialSlot = executionQueue_dataOut_decodeResult_specialSlot;
  assign executionQueue_deq_bits_decodeResult_topUop = executionQueue_dataOut_decodeResult_topUop;
  assign executionQueue_deq_bits_decodeResult_popCount = executionQueue_dataOut_decodeResult_popCount;
  assign executionQueue_deq_bits_decodeResult_ffo = executionQueue_dataOut_decodeResult_ffo;
  assign executionQueue_deq_bits_decodeResult_average = executionQueue_dataOut_decodeResult_average;
  assign executionQueue_deq_bits_decodeResult_reverse = executionQueue_dataOut_decodeResult_reverse;
  assign executionQueue_deq_bits_decodeResult_dontNeedExecuteInLane = executionQueue_dataOut_decodeResult_dontNeedExecuteInLane;
  assign executionQueue_deq_bits_decodeResult_scheduler = executionQueue_dataOut_decodeResult_scheduler;
  assign executionQueue_deq_bits_decodeResult_sReadVD = executionQueue_dataOut_decodeResult_sReadVD;
  assign executionQueue_deq_bits_decodeResult_vtype = executionQueue_dataOut_decodeResult_vtype;
  assign executionQueue_deq_bits_decodeResult_sWrite = executionQueue_dataOut_decodeResult_sWrite;
  assign executionQueue_deq_bits_decodeResult_crossRead = executionQueue_dataOut_decodeResult_crossRead;
  assign executionQueue_deq_bits_decodeResult_crossWrite = executionQueue_dataOut_decodeResult_crossWrite;
  assign executionQueue_deq_bits_decodeResult_maskUnit = executionQueue_dataOut_decodeResult_maskUnit;
  assign executionQueue_deq_bits_decodeResult_special = executionQueue_dataOut_decodeResult_special;
  assign executionQueue_deq_bits_decodeResult_saturate = executionQueue_dataOut_decodeResult_saturate;
  assign executionQueue_deq_bits_decodeResult_vwmacc = executionQueue_dataOut_decodeResult_vwmacc;
  assign executionQueue_deq_bits_decodeResult_readOnly = executionQueue_dataOut_decodeResult_readOnly;
  assign executionQueue_deq_bits_decodeResult_maskSource = executionQueue_dataOut_decodeResult_maskSource;
  assign executionQueue_deq_bits_decodeResult_maskDestination = executionQueue_dataOut_decodeResult_maskDestination;
  assign executionQueue_deq_bits_decodeResult_maskLogic = executionQueue_dataOut_decodeResult_maskLogic;
  assign executionQueue_deq_bits_decodeResult_uop = executionQueue_dataOut_decodeResult_uop;
  assign executionQueue_deq_bits_decodeResult_iota = executionQueue_dataOut_decodeResult_iota;
  assign executionQueue_deq_bits_decodeResult_mv = executionQueue_dataOut_decodeResult_mv;
  assign executionQueue_deq_bits_decodeResult_extend = executionQueue_dataOut_decodeResult_extend;
  assign executionQueue_deq_bits_decodeResult_unOrderWrite = executionQueue_dataOut_decodeResult_unOrderWrite;
  assign executionQueue_deq_bits_decodeResult_compress = executionQueue_dataOut_decodeResult_compress;
  assign executionQueue_deq_bits_decodeResult_gather16 = executionQueue_dataOut_decodeResult_gather16;
  assign executionQueue_deq_bits_decodeResult_gather = executionQueue_dataOut_decodeResult_gather;
  assign executionQueue_deq_bits_decodeResult_slid = executionQueue_dataOut_decodeResult_slid;
  assign executionQueue_deq_bits_decodeResult_targetRd = executionQueue_dataOut_decodeResult_targetRd;
  assign executionQueue_deq_bits_decodeResult_widenReduce = executionQueue_dataOut_decodeResult_widenReduce;
  assign executionQueue_deq_bits_decodeResult_red = executionQueue_dataOut_decodeResult_red;
  assign executionQueue_deq_bits_decodeResult_nr = executionQueue_dataOut_decodeResult_nr;
  assign executionQueue_deq_bits_decodeResult_itype = executionQueue_dataOut_decodeResult_itype;
  assign executionQueue_deq_bits_decodeResult_unsigned1 = executionQueue_dataOut_decodeResult_unsigned1;
  assign executionQueue_deq_bits_decodeResult_unsigned0 = executionQueue_dataOut_decodeResult_unsigned0;
  assign executionQueue_deq_bits_decodeResult_other = executionQueue_dataOut_decodeResult_other;
  assign executionQueue_deq_bits_decodeResult_multiCycle = executionQueue_dataOut_decodeResult_multiCycle;
  assign executionQueue_deq_bits_decodeResult_divider = executionQueue_dataOut_decodeResult_divider;
  assign executionQueue_deq_bits_decodeResult_multiplier = executionQueue_dataOut_decodeResult_multiplier;
  assign executionQueue_deq_bits_decodeResult_shift = executionQueue_dataOut_decodeResult_shift;
  assign executionQueue_deq_bits_decodeResult_adder = executionQueue_dataOut_decodeResult_adder;
  assign executionQueue_deq_bits_decodeResult_logic = executionQueue_dataOut_decodeResult_logic;
  assign executionQueue_deq_bits_instructionIndex = executionQueue_dataOut_instructionIndex;
  assign executionQueue_deq_bits_loadStore = executionQueue_dataOut_loadStore;
  assign executionQueue_deq_bits_vd = executionQueue_dataOut_vd;
  assign executionQueue_enq_ready = ~_executionQueue_fifo_full;
  wire [31:0] _bordersCorrectMask_T_1 = 32'h1 << enqueue_bits_csr_vl_0[4:0];
  wire [29:0] _GEN = _bordersCorrectMask_T_1[30:1] | _bordersCorrectMask_T_1[31:2];
  wire [28:0] _GEN_0 = _GEN[28:0] | {_bordersCorrectMask_T_1[31], _GEN[29:2]};
  wire [26:0] _GEN_1 = _GEN_0[26:0] | {_bordersCorrectMask_T_1[31], _GEN[29], _GEN_0[28:4]};
  wire [22:0] _GEN_2 = _GEN_1[22:0] | {_bordersCorrectMask_T_1[31], _GEN[29], _GEN_0[28:27], _GEN_1[26:8]};
  wire [31:0] bordersCorrectMask =
    enqueue_bits_bordersForMaskLogic_0
      ? {1'h0, _bordersCorrectMask_T_1[31], _GEN[29], _GEN_0[28:27], _GEN_1[26:23], _GEN_2[22:15], _GEN_2[14:0] | {_bordersCorrectMask_T_1[31], _GEN[29], _GEN_0[28:27], _GEN_1[26:23], _GEN_2[22:16]}}
      : 32'hFFFFFFFF;
  wire [31:0] maskTypeMask = enqueue_bits_maskType_0 ? enqueue_bits_src_0_0 : 32'hFFFFFFFF;
  wire [31:0] complexMask = bordersCorrectMask & maskTypeMask;
  wire [31:0] ffoCompleteWrite = enqueue_bits_maskType_0 | enqueue_bits_bordersForMaskLogic_0 ? ~complexMask & enqueue_bits_src_2_0 : 32'h0;
  assign executionQueue_enq_bits_mask =
    (enqueue_bits_vSew1H_0[0] ? enqueue_bits_maskForFilter_0 : 4'h0) | (enqueue_bits_vSew1H_0[1] ? {{2{enqueue_bits_maskForFilter_0[1]}}, {2{enqueue_bits_maskForFilter_0[0]}}} : 4'h0)
    | {4{enqueue_bits_vSew1H_0[2] & enqueue_bits_maskForFilter_0[0]}};
  wire        executionQueue_empty;
  assign executionQueue_empty = _executionQueue_fifo_empty;
  wire        executionQueue_full;
  assign executionQueue_full = _executionQueue_fifo_full;
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(69)
  ) executionQueue_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(executionQueue_enq_ready & executionQueue_enq_valid)),
    .pop_req_n    (~(executionQueue_deq_ready & ~_executionQueue_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (executionQueue_dataIn),
    .empty        (_executionQueue_fifo_empty),
    .almost_empty (executionQueue_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (executionQueue_almostFull),
    .full         (_executionQueue_fifo_full),
    .error        (_executionQueue_fifo_error),
    .data_out     (_executionQueue_fifo_data_out)
  );
  assign enqueue_ready = enqueue_ready_0;
  assign dequeue_valid = dequeue_valid_0;
  assign dequeue_bits_groupCounter = dequeue_bits_groupCounter_0;
  assign dequeue_bits_mask = dequeue_bits_mask_0;
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
endmodule

