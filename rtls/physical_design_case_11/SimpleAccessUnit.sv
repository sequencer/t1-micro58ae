
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
module SimpleAccessUnit(
  input         clock,
                reset,
                lsuRequest_valid,
  input  [2:0]  lsuRequest_bits_instructionInformation_nf,
  input         lsuRequest_bits_instructionInformation_mew,
  input  [1:0]  lsuRequest_bits_instructionInformation_mop,
  input  [4:0]  lsuRequest_bits_instructionInformation_lumop,
  input  [1:0]  lsuRequest_bits_instructionInformation_eew,
  input  [4:0]  lsuRequest_bits_instructionInformation_vs3,
  input         lsuRequest_bits_instructionInformation_isStore,
                lsuRequest_bits_instructionInformation_maskedLoadStore,
  input  [31:0] lsuRequest_bits_rs1Data,
                lsuRequest_bits_rs2Data,
  input  [2:0]  lsuRequest_bits_instructionIndex,
  input         vrfReadDataPorts_ready,
  output        vrfReadDataPorts_valid,
  output [4:0]  vrfReadDataPorts_bits_vs,
  output [2:0]  vrfReadDataPorts_bits_offset,
                vrfReadDataPorts_bits_instructionIndex,
  input         vrfReadResults_valid,
  input  [31:0] vrfReadResults_bits,
  input         offsetReadResult_0_valid,
  input  [31:0] offsetReadResult_0_bits,
  input         offsetReadResult_1_valid,
  input  [31:0] offsetReadResult_1_bits,
  input         offsetReadResult_2_valid,
  input  [31:0] offsetReadResult_2_bits,
  input         offsetReadResult_3_valid,
  input  [31:0] offsetReadResult_3_bits,
  input         offsetReadResult_4_valid,
  input  [31:0] offsetReadResult_4_bits,
  input         offsetReadResult_5_valid,
  input  [31:0] offsetReadResult_5_bits,
  input         offsetReadResult_6_valid,
  input  [31:0] offsetReadResult_6_bits,
  input         offsetReadResult_7_valid,
  input  [31:0] offsetReadResult_7_bits,
                maskInput,
  output        maskSelect_valid,
  output [5:0]  maskSelect_bits,
  input         memReadRequest_ready,
  output        memReadRequest_valid,
  output [31:0] memReadRequest_bits_address,
  output [7:0]  memReadRequest_bits_source,
  output        memReadResponse_ready,
  input         memReadResponse_valid,
  input  [31:0] memReadResponse_bits_data,
  input  [7:0]  memReadResponse_bits_source,
  input         memWriteRequest_ready,
  output        memWriteRequest_valid,
  output [31:0] memWriteRequest_bits_data,
  output [3:0]  memWriteRequest_bits_mask,
  output [7:0]  memWriteRequest_bits_source,
  output [31:0] memWriteRequest_bits_address,
  output [1:0]  memWriteRequest_bits_size,
  input         vrfWritePort_ready,
  output        vrfWritePort_valid,
  output [4:0]  vrfWritePort_bits_vd,
  output [2:0]  vrfWritePort_bits_offset,
  output [3:0]  vrfWritePort_bits_mask,
  output [31:0] vrfWritePort_bits_data,
  output        vrfWritePort_bits_last,
  output [2:0]  vrfWritePort_bits_instructionIndex,
  input  [11:0] csrInterface_vl,
                csrInterface_vStart,
  input  [2:0]  csrInterface_vlmul,
  input  [1:0]  csrInterface_vSew,
                csrInterface_vxrm,
  input         csrInterface_vta,
                csrInterface_vma,
  output        status_idle,
                status_last,
  output [2:0]  status_instructionIndex,
  output [7:0]  status_targetLane,
  output        status_isStore,
                offsetRelease_0,
                offsetRelease_1,
                offsetRelease_2,
                offsetRelease_3,
                offsetRelease_4,
                offsetRelease_5,
                offsetRelease_6,
                offsetRelease_7
);

  wire             _s1EnqDataQueue_fifo_empty;
  wire             _s1EnqDataQueue_fifo_full;
  wire             _s1EnqDataQueue_fifo_error;
  wire             _s1EnqQueue_fifo_empty;
  wire             _s1EnqQueue_fifo_full;
  wire             _s1EnqQueue_fifo_error;
  wire [71:0]      _s1EnqQueue_fifo_data_out;
  wire             _offsetQueueVec_queue_fifo_7_empty;
  wire             _offsetQueueVec_queue_fifo_7_full;
  wire             _offsetQueueVec_queue_fifo_7_error;
  wire             _offsetQueueVec_queue_fifo_6_empty;
  wire             _offsetQueueVec_queue_fifo_6_full;
  wire             _offsetQueueVec_queue_fifo_6_error;
  wire             _offsetQueueVec_queue_fifo_5_empty;
  wire             _offsetQueueVec_queue_fifo_5_full;
  wire             _offsetQueueVec_queue_fifo_5_error;
  wire             _offsetQueueVec_queue_fifo_4_empty;
  wire             _offsetQueueVec_queue_fifo_4_full;
  wire             _offsetQueueVec_queue_fifo_4_error;
  wire             _offsetQueueVec_queue_fifo_3_empty;
  wire             _offsetQueueVec_queue_fifo_3_full;
  wire             _offsetQueueVec_queue_fifo_3_error;
  wire             _offsetQueueVec_queue_fifo_2_empty;
  wire             _offsetQueueVec_queue_fifo_2_full;
  wire             _offsetQueueVec_queue_fifo_2_error;
  wire             _offsetQueueVec_queue_fifo_1_empty;
  wire             _offsetQueueVec_queue_fifo_1_full;
  wire             _offsetQueueVec_queue_fifo_1_error;
  wire             _offsetQueueVec_queue_fifo_empty;
  wire             _offsetQueueVec_queue_fifo_full;
  wire             _offsetQueueVec_queue_fifo_error;
  wire             s1EnqDataQueue_almostFull;
  wire             s1EnqDataQueue_almostEmpty;
  wire             s1EnqQueue_almostFull;
  wire             s1EnqQueue_almostEmpty;
  wire             offsetQueueVec_7_almostFull;
  wire             offsetQueueVec_7_almostEmpty;
  wire             offsetQueueVec_6_almostFull;
  wire             offsetQueueVec_6_almostEmpty;
  wire             offsetQueueVec_5_almostFull;
  wire             offsetQueueVec_5_almostEmpty;
  wire             offsetQueueVec_4_almostFull;
  wire             offsetQueueVec_4_almostEmpty;
  wire             offsetQueueVec_3_almostFull;
  wire             offsetQueueVec_3_almostEmpty;
  wire             offsetQueueVec_2_almostFull;
  wire             offsetQueueVec_2_almostEmpty;
  wire             offsetQueueVec_1_almostFull;
  wire             offsetQueueVec_1_almostEmpty;
  wire             offsetQueueVec_0_almostFull;
  wire             offsetQueueVec_0_almostEmpty;
  wire [31:0]      s1EnqDataQueue_deq_bits;
  wire             s1EnqueueReady;
  wire [31:0]      s1EnqQueue_deq_bits_address;
  wire [2:0]       s1EnqQueue_deq_bits_segmentIndex;
  wire [4:0]       s1EnqQueue_deq_bits_indexInMaskGroup;
  wire             vrfReadDataPorts_ready_0 = vrfReadDataPorts_ready;
  wire             memReadRequest_ready_0 = memReadRequest_ready;
  wire             memReadResponse_valid_0 = memReadResponse_valid;
  wire [31:0]      memReadResponse_bits_data_0 = memReadResponse_bits_data;
  wire [7:0]       memReadResponse_bits_source_0 = memReadResponse_bits_source;
  wire             memWriteRequest_ready_0 = memWriteRequest_ready;
  wire             vrfWritePort_ready_0 = vrfWritePort_ready;
  wire             offsetQueueVec_0_enq_valid = offsetReadResult_0_valid;
  wire [31:0]      offsetQueueVec_0_enq_bits = offsetReadResult_0_bits;
  wire             offsetQueueVec_1_enq_valid = offsetReadResult_1_valid;
  wire [31:0]      offsetQueueVec_1_enq_bits = offsetReadResult_1_bits;
  wire             offsetQueueVec_2_enq_valid = offsetReadResult_2_valid;
  wire [31:0]      offsetQueueVec_2_enq_bits = offsetReadResult_2_bits;
  wire             offsetQueueVec_3_enq_valid = offsetReadResult_3_valid;
  wire [31:0]      offsetQueueVec_3_enq_bits = offsetReadResult_3_bits;
  wire             offsetQueueVec_4_enq_valid = offsetReadResult_4_valid;
  wire [31:0]      offsetQueueVec_4_enq_bits = offsetReadResult_4_bits;
  wire             offsetQueueVec_5_enq_valid = offsetReadResult_5_valid;
  wire [31:0]      offsetQueueVec_5_enq_bits = offsetReadResult_5_bits;
  wire             offsetQueueVec_6_enq_valid = offsetReadResult_6_valid;
  wire [31:0]      offsetQueueVec_6_enq_bits = offsetReadResult_6_bits;
  wire             offsetQueueVec_7_enq_valid = offsetReadResult_7_valid;
  wire [31:0]      offsetQueueVec_7_enq_bits = offsetReadResult_7_bits;
  wire             s1EnqDataQueue_enq_valid = vrfReadResults_valid;
  wire [31:0]      s1EnqDataQueue_enq_bits = vrfReadResults_bits;
  wire [1:0]       vrfReadDataPorts_bits_readSource = 2'h2;
  wire [1:0]       memReadRequest_bits_size = 2'h2;
  wire [31:0]      s1EnqQueue_enq_bits_readData = 32'h0;
  wire [7:0]       memoryRequestSource;
  wire             vrfWritePort_valid_0 = memReadResponse_valid_0;
  wire [3:0]       storeMask;
  wire [1:0]       dataEEW;
  wire             memReadResponse_ready_0 = vrfWritePort_ready_0;
  wire [2:0]       writeOffset;
  wire             last;
  wire             offsetQueueVec_0_deq_valid;
  assign offsetQueueVec_0_deq_valid = ~_offsetQueueVec_queue_fifo_empty;
  wire             offsetQueueVec_0_enq_ready = ~_offsetQueueVec_queue_fifo_full;
  wire             offsetQueueVec_0_deq_ready;
  reg              offsetQueueVec_deqLock;
  wire             waitQueueDeq_0 = offsetQueueVec_deqLock;
  wire             _allElementsMasked_T_1 = offsetQueueVec_0_deq_ready & offsetQueueVec_0_deq_valid;
  wire             stateIdle;
  assign offsetQueueVec_0_deq_ready = ~offsetQueueVec_deqLock | stateIdle;
  wire             offsetQueueVec_1_deq_valid;
  assign offsetQueueVec_1_deq_valid = ~_offsetQueueVec_queue_fifo_1_empty;
  wire             offsetQueueVec_1_enq_ready = ~_offsetQueueVec_queue_fifo_1_full;
  wire             offsetQueueVec_1_deq_ready;
  reg              offsetQueueVec_deqLock_1;
  wire             waitQueueDeq_1 = offsetQueueVec_deqLock_1;
  wire             _allElementsMasked_T_2 = offsetQueueVec_1_deq_ready & offsetQueueVec_1_deq_valid;
  assign offsetQueueVec_1_deq_ready = ~offsetQueueVec_deqLock_1 | stateIdle;
  wire             offsetQueueVec_2_deq_valid;
  assign offsetQueueVec_2_deq_valid = ~_offsetQueueVec_queue_fifo_2_empty;
  wire             offsetQueueVec_2_enq_ready = ~_offsetQueueVec_queue_fifo_2_full;
  wire             offsetQueueVec_2_deq_ready;
  reg              offsetQueueVec_deqLock_2;
  wire             waitQueueDeq_2 = offsetQueueVec_deqLock_2;
  wire             _allElementsMasked_T_3 = offsetQueueVec_2_deq_ready & offsetQueueVec_2_deq_valid;
  assign offsetQueueVec_2_deq_ready = ~offsetQueueVec_deqLock_2 | stateIdle;
  wire             offsetQueueVec_3_deq_valid;
  assign offsetQueueVec_3_deq_valid = ~_offsetQueueVec_queue_fifo_3_empty;
  wire             offsetQueueVec_3_enq_ready = ~_offsetQueueVec_queue_fifo_3_full;
  wire             offsetQueueVec_3_deq_ready;
  reg              offsetQueueVec_deqLock_3;
  wire             waitQueueDeq_3 = offsetQueueVec_deqLock_3;
  wire             _allElementsMasked_T_4 = offsetQueueVec_3_deq_ready & offsetQueueVec_3_deq_valid;
  assign offsetQueueVec_3_deq_ready = ~offsetQueueVec_deqLock_3 | stateIdle;
  wire             offsetQueueVec_4_deq_valid;
  assign offsetQueueVec_4_deq_valid = ~_offsetQueueVec_queue_fifo_4_empty;
  wire             offsetQueueVec_4_enq_ready = ~_offsetQueueVec_queue_fifo_4_full;
  wire             offsetQueueVec_4_deq_ready;
  reg              offsetQueueVec_deqLock_4;
  wire             waitQueueDeq_4 = offsetQueueVec_deqLock_4;
  wire             _allElementsMasked_T_5 = offsetQueueVec_4_deq_ready & offsetQueueVec_4_deq_valid;
  assign offsetQueueVec_4_deq_ready = ~offsetQueueVec_deqLock_4 | stateIdle;
  wire             offsetQueueVec_5_deq_valid;
  assign offsetQueueVec_5_deq_valid = ~_offsetQueueVec_queue_fifo_5_empty;
  wire             offsetQueueVec_5_enq_ready = ~_offsetQueueVec_queue_fifo_5_full;
  wire             offsetQueueVec_5_deq_ready;
  reg              offsetQueueVec_deqLock_5;
  wire             waitQueueDeq_5 = offsetQueueVec_deqLock_5;
  wire             _allElementsMasked_T_6 = offsetQueueVec_5_deq_ready & offsetQueueVec_5_deq_valid;
  assign offsetQueueVec_5_deq_ready = ~offsetQueueVec_deqLock_5 | stateIdle;
  wire             offsetQueueVec_6_deq_valid;
  assign offsetQueueVec_6_deq_valid = ~_offsetQueueVec_queue_fifo_6_empty;
  wire             offsetQueueVec_6_enq_ready = ~_offsetQueueVec_queue_fifo_6_full;
  wire             offsetQueueVec_6_deq_ready;
  reg              offsetQueueVec_deqLock_6;
  wire             waitQueueDeq_6 = offsetQueueVec_deqLock_6;
  wire             _allElementsMasked_T_7 = offsetQueueVec_6_deq_ready & offsetQueueVec_6_deq_valid;
  assign offsetQueueVec_6_deq_ready = ~offsetQueueVec_deqLock_6 | stateIdle;
  wire             offsetQueueVec_7_deq_valid;
  assign offsetQueueVec_7_deq_valid = ~_offsetQueueVec_queue_fifo_7_empty;
  wire             offsetQueueVec_7_enq_ready = ~_offsetQueueVec_queue_fifo_7_full;
  wire             offsetQueueVec_7_deq_ready;
  reg              offsetQueueVec_deqLock_7;
  wire             waitQueueDeq_7 = offsetQueueVec_deqLock_7;
  wire             _allElementsMasked_T_8 = offsetQueueVec_7_deq_ready & offsetQueueVec_7_deq_valid;
  assign offsetQueueVec_7_deq_ready = ~offsetQueueVec_deqLock_7 | stateIdle;
  wire             memReadRequest_valid_0;
  wire             _outstandingTLDMessages_T_4 = memReadRequest_ready_0 & memReadRequest_valid_0;
  wire             memWriteRequest_valid_0;
  wire             _probeWire_valid_T = memWriteRequest_ready_0 & memWriteRequest_valid_0;
  wire             memRequestFire = _outstandingTLDMessages_T_4 | _probeWire_valid_T;
  reg  [2:0]       lsuRequestReg_instructionInformation_nf;
  reg              lsuRequestReg_instructionInformation_mew;
  reg  [1:0]       lsuRequestReg_instructionInformation_mop;
  reg  [4:0]       lsuRequestReg_instructionInformation_lumop;
  reg  [1:0]       lsuRequestReg_instructionInformation_eew;
  reg  [4:0]       lsuRequestReg_instructionInformation_vs3;
  reg              lsuRequestReg_instructionInformation_isStore;
  reg              lsuRequestReg_instructionInformation_maskedLoadStore;
  reg  [31:0]      lsuRequestReg_rs1Data;
  reg  [31:0]      lsuRequestReg_rs2Data;
  reg  [2:0]       lsuRequestReg_instructionIndex;
  wire [2:0]       vrfReadDataPorts_bits_instructionIndex_0 = lsuRequestReg_instructionIndex;
  wire [2:0]       vrfWritePort_bits_instructionIndex_0 = lsuRequestReg_instructionIndex;
  reg  [11:0]      csrInterfaceReg_vl;
  reg  [11:0]      csrInterfaceReg_vStart;
  reg  [2:0]       csrInterfaceReg_vlmul;
  reg  [1:0]       csrInterfaceReg_vSew;
  reg  [1:0]       csrInterfaceReg_vxrm;
  reg              csrInterfaceReg_vta;
  reg              csrInterfaceReg_vma;
  wire             _isMaskLoadStore_T = lsuRequestReg_instructionInformation_mop == 2'h0;
  wire             isWholeRegisterLoadStore = _isMaskLoadStore_T & lsuRequestReg_instructionInformation_lumop == 5'h8;
  wire             isSegmentLoadStore = (|lsuRequestReg_instructionInformation_nf) & ~isWholeRegisterLoadStore;
  wire             isMaskLoadStore = _isMaskLoadStore_T & lsuRequestReg_instructionInformation_lumop[0];
  wire             isIndexedLoadStore = lsuRequestReg_instructionInformation_mop[0];
  wire             _waitFirstMemoryResponseForFaultOnlyFirst_T = lsuRequest_bits_instructionInformation_mop == 2'h0;
  wire             requestIsWholeRegisterLoadStore = _waitFirstMemoryResponseForFaultOnlyFirst_T & lsuRequest_bits_instructionInformation_lumop == 5'h8;
  wire             requestIsMaskLoadStore = _waitFirstMemoryResponseForFaultOnlyFirst_T & lsuRequest_bits_instructionInformation_lumop[0];
  wire [1:0]       requestEEW = lsuRequest_bits_instructionInformation_mop[0] ? csrInterface_vSew : requestIsMaskLoadStore ? 2'h0 : requestIsWholeRegisterLoadStore ? 2'h2 : lsuRequest_bits_instructionInformation_eew;
  wire [2:0]       requestNF = requestIsWholeRegisterLoadStore ? 3'h0 : lsuRequest_bits_instructionInformation_nf;
  reg  [6:0]       dataWidthForSegmentLoadStore;
  reg  [2:0]       elementByteWidth;
  reg  [3:0]       segmentInstructionIndexInterval;
  reg  [31:0]      outstandingTLDMessages;
  wire             noOutstandingMessages = outstandingTLDMessages == 32'h0;
  reg              indexedInstructionOffsets_0_valid;
  reg  [31:0]      indexedInstructionOffsets_0_bits;
  reg              indexedInstructionOffsets_1_valid;
  reg  [31:0]      indexedInstructionOffsets_1_bits;
  reg              indexedInstructionOffsets_2_valid;
  reg  [31:0]      indexedInstructionOffsets_2_bits;
  reg              indexedInstructionOffsets_3_valid;
  reg  [31:0]      indexedInstructionOffsets_3_bits;
  reg              indexedInstructionOffsets_4_valid;
  reg  [31:0]      indexedInstructionOffsets_4_bits;
  reg              indexedInstructionOffsets_5_valid;
  reg  [31:0]      indexedInstructionOffsets_5_bits;
  reg              indexedInstructionOffsets_6_valid;
  reg  [31:0]      indexedInstructionOffsets_6_bits;
  reg              indexedInstructionOffsets_7_valid;
  reg  [31:0]      indexedInstructionOffsets_7_bits;
  reg  [6:0]       groupIndex;
  wire [6:0]       nextGroupIndex = lsuRequest_valid ? 7'h0 : groupIndex + 7'h1;
  reg  [1:0]       indexOfIndexedInstructionOffsets;
  wire [1:0]       indexOfIndexedInstructionOffsetsNext = lsuRequest_valid ? 2'h3 : indexOfIndexedInstructionOffsets + 2'h1;
  reg  [31:0]      maskReg;
  reg  [2:0]       segmentIndex;
  wire [2:0]       s0Wire_segmentIndex = segmentIndex;
  wire [2:0]       segmentIndexNext = segmentIndex + 3'h1;
  wire             segmentEnd = segmentIndex == lsuRequestReg_instructionInformation_nf;
  wire             lastElementForSegment = ~isSegmentLoadStore | segmentEnd;
  wire             s0Fire;
  wire             segmentEndWithHandshake = s0Fire & lastElementForSegment;
  reg  [1:0]       state;
  reg  [31:0]      sentMemoryRequests;
  wire [31:0]      unsentMemoryRequests = ~sentMemoryRequests;
  wire [31:0]      maskedUnsentMemoryRequests = maskReg & unsentMemoryRequests;
  wire [30:0]      _findFirstMaskedUnsentMemoryRequests_T_2 = maskedUnsentMemoryRequests[30:0] | {maskedUnsentMemoryRequests[29:0], 1'h0};
  wire [30:0]      _findFirstMaskedUnsentMemoryRequests_T_5 = _findFirstMaskedUnsentMemoryRequests_T_2 | {_findFirstMaskedUnsentMemoryRequests_T_2[28:0], 2'h0};
  wire [30:0]      _findFirstMaskedUnsentMemoryRequests_T_8 = _findFirstMaskedUnsentMemoryRequests_T_5 | {_findFirstMaskedUnsentMemoryRequests_T_5[26:0], 4'h0};
  wire [30:0]      _findFirstMaskedUnsentMemoryRequests_T_11 = _findFirstMaskedUnsentMemoryRequests_T_8 | {_findFirstMaskedUnsentMemoryRequests_T_8[22:0], 8'h0};
  wire [31:0]      findFirstMaskedUnsentMemoryRequests = {~(_findFirstMaskedUnsentMemoryRequests_T_11 | {_findFirstMaskedUnsentMemoryRequests_T_11[14:0], 16'h0}), 1'h1} & maskedUnsentMemoryRequests;
  wire [32:0]      findFirstUnsentMemoryRequestsNext = {1'h0, {sentMemoryRequests[30:0], 1'h1} & unsentMemoryRequests};
  wire [31:0]      nextElementForMemoryRequest = lsuRequestReg_instructionInformation_maskedLoadStore ? findFirstMaskedUnsentMemoryRequests : findFirstUnsentMemoryRequestsNext[31:0];
  wire [15:0]      nextElementForMemoryRequestIndex_hi = nextElementForMemoryRequest[31:16];
  wire [15:0]      nextElementForMemoryRequestIndex_lo = nextElementForMemoryRequest[15:0];
  wire [15:0]      _nextElementForMemoryRequestIndex_T_1 = nextElementForMemoryRequestIndex_hi | nextElementForMemoryRequestIndex_lo;
  wire [7:0]       nextElementForMemoryRequestIndex_hi_1 = _nextElementForMemoryRequestIndex_T_1[15:8];
  wire [7:0]       nextElementForMemoryRequestIndex_lo_1 = _nextElementForMemoryRequestIndex_T_1[7:0];
  wire [7:0]       _nextElementForMemoryRequestIndex_T_3 = nextElementForMemoryRequestIndex_hi_1 | nextElementForMemoryRequestIndex_lo_1;
  wire [3:0]       nextElementForMemoryRequestIndex_hi_2 = _nextElementForMemoryRequestIndex_T_3[7:4];
  wire [3:0]       nextElementForMemoryRequestIndex_lo_2 = _nextElementForMemoryRequestIndex_T_3[3:0];
  wire [3:0]       _nextElementForMemoryRequestIndex_T_5 = nextElementForMemoryRequestIndex_hi_2 | nextElementForMemoryRequestIndex_lo_2;
  wire [1:0]       nextElementForMemoryRequestIndex_hi_3 = _nextElementForMemoryRequestIndex_T_5[3:2];
  wire [1:0]       nextElementForMemoryRequestIndex_lo_3 = _nextElementForMemoryRequestIndex_T_5[1:0];
  wire [4:0]       nextElementForMemoryRequestIndex =
    {|nextElementForMemoryRequestIndex_hi,
     |nextElementForMemoryRequestIndex_hi_1,
     |nextElementForMemoryRequestIndex_hi_2,
     |nextElementForMemoryRequestIndex_hi_3,
     nextElementForMemoryRequestIndex_hi_3[1] | nextElementForMemoryRequestIndex_lo_3[1]};
  wire [4:0]       s0Wire_indexInGroup = nextElementForMemoryRequestIndex;
  assign dataEEW = isIndexedLoadStore ? csrInterfaceReg_vSew : isMaskLoadStore ? 2'h0 : isWholeRegisterLoadStore ? 2'h2 : lsuRequestReg_instructionInformation_eew;
  wire [1:0]       memWriteRequest_bits_size_0 = dataEEW;
  wire [3:0]       dataEEWOH = 4'h1 << dataEEW;
  wire             noMoreMaskedUnsentMemoryRequests = maskedUnsentMemoryRequests == 32'h0;
  wire             maskGroupEndAndRequestNewMask = (noMoreMaskedUnsentMemoryRequests | nextElementForMemoryRequest[31] & segmentEndWithHandshake) & lsuRequestReg_instructionInformation_maskedLoadStore;
  wire             maskGroupEnd = maskGroupEndAndRequestNewMask | nextElementForMemoryRequest[31] & segmentEndWithHandshake;
  wire [3:0]       _offsetEEWOH_T = 4'h1 << lsuRequestReg_instructionInformation_eew;
  wire [2:0]       offsetEEWOH = _offsetEEWOH_T[2:0];
  wire [11:0]      _offsetForStride_T = {groupIndex, nextElementForMemoryRequestIndex};
  wire [11:0]      offsetForUnitStride;
  assign offsetForUnitStride = _offsetForStride_T;
  wire [11:0]      s0ElementIndex;
  assign s0ElementIndex = _offsetForStride_T;
  wire [14:0]      _globalOffsetOfIndexedInstructionOffsets_T_1 = {3'h0, groupIndex, nextElementForMemoryRequestIndex} << lsuRequestReg_instructionInformation_eew;
  wire [6:0]       globalOffsetOfIndexedInstructionOffsets = _globalOffsetOfIndexedInstructionOffsets_T_1[6:0];
  wire [1:0]       offsetGroupIndexOfMemoryRequest = globalOffsetOfIndexedInstructionOffsets[6:5];
  wire [4:0]       offsetOfOffsetGroup = globalOffsetOfIndexedInstructionOffsets[4:0];
  wire [63:0]      offsetOfCurrentMemoryRequest_lo_lo = {indexedInstructionOffsets_1_bits, indexedInstructionOffsets_0_bits};
  wire [63:0]      offsetOfCurrentMemoryRequest_lo_hi = {indexedInstructionOffsets_3_bits, indexedInstructionOffsets_2_bits};
  wire [127:0]     offsetOfCurrentMemoryRequest_lo = {offsetOfCurrentMemoryRequest_lo_hi, offsetOfCurrentMemoryRequest_lo_lo};
  wire [63:0]      offsetOfCurrentMemoryRequest_hi_lo = {indexedInstructionOffsets_5_bits, indexedInstructionOffsets_4_bits};
  wire [63:0]      offsetOfCurrentMemoryRequest_hi_hi = {indexedInstructionOffsets_7_bits, indexedInstructionOffsets_6_bits};
  wire [127:0]     offsetOfCurrentMemoryRequest_hi = {offsetOfCurrentMemoryRequest_hi_hi, offsetOfCurrentMemoryRequest_hi_lo};
  wire [255:0]     _offsetOfCurrentMemoryRequest_T_2 = {offsetOfCurrentMemoryRequest_hi, offsetOfCurrentMemoryRequest_lo} >> {248'h0, offsetOfOffsetGroup, 3'h0};
  wire [15:0]      offsetOfCurrentMemoryRequest_lo_1 = {{8{~(offsetEEWOH[0])}}, 8'hFF};
  wire [15:0]      offsetOfCurrentMemoryRequest_hi_1 = {16{offsetEEWOH[2]}};
  wire [31:0]      offsetOfCurrentMemoryRequest = _offsetOfCurrentMemoryRequest_T_2[31:0] & {offsetOfCurrentMemoryRequest_hi_1, offsetOfCurrentMemoryRequest_lo_1};
  wire [1:0]       offsetValidCheck_lo_lo = {indexedInstructionOffsets_1_valid, indexedInstructionOffsets_0_valid};
  wire [1:0]       offsetValidCheck_lo_hi = {indexedInstructionOffsets_3_valid, indexedInstructionOffsets_2_valid};
  wire [3:0]       offsetValidCheck_lo = {offsetValidCheck_lo_hi, offsetValidCheck_lo_lo};
  wire [1:0]       offsetValidCheck_hi_lo = {indexedInstructionOffsets_5_valid, indexedInstructionOffsets_4_valid};
  wire [1:0]       offsetValidCheck_hi_hi = {indexedInstructionOffsets_7_valid, indexedInstructionOffsets_6_valid};
  wire [3:0]       offsetValidCheck_hi = {offsetValidCheck_hi_hi, offsetValidCheck_hi_lo};
  wire [7:0]       _offsetValidCheck_T_2 = {offsetValidCheck_hi, offsetValidCheck_lo} >> offsetOfOffsetGroup[4:2];
  wire             offsetValidCheck = _offsetValidCheck_T_2[0];
  wire [1:0]       offsetGroupMatch = offsetGroupIndexOfMemoryRequest ^ indexOfIndexedInstructionOffsets;
  wire             offsetGroupCheck = (~(lsuRequestReg_instructionInformation_eew[0]) | ~(offsetGroupMatch[0])) & (~(lsuRequestReg_instructionInformation_eew[1]) | offsetGroupMatch == 2'h0);
  wire [43:0]      offsetForStride = {32'h0, groupIndex, nextElementForMemoryRequestIndex} * {12'h0, lsuRequestReg_rs2Data};
  wire [43:0]      baseOffsetForElement =
    isIndexedLoadStore ? {12'h0, offsetOfCurrentMemoryRequest} : lsuRequestReg_instructionInformation_mop[1] ? offsetForStride : {25'h0, {7'h0, offsetForUnitStride} * {12'h0, dataWidthForSegmentLoadStore}};
  wire [7:0]       laneOfOffsetOfOffsetGroup = 8'h1 << offsetOfOffsetGroup[4:2];
  wire             indexedInstructionOffsetExhausted = offsetEEWOH[0] & (&(offsetOfOffsetGroup[1:0])) | offsetEEWOH[1] & offsetOfOffsetGroup[1] | offsetEEWOH[2];
  wire             _GEN = segmentEndWithHandshake & indexedInstructionOffsetExhausted;
  wire [1:0]       _GEN_0 = {waitQueueDeq_1, waitQueueDeq_0};
  wire [1:0]       usedIndexedInstructionOffsets_0_lo_lo;
  assign usedIndexedInstructionOffsets_0_lo_lo = _GEN_0;
  wire [1:0]       usedIndexedInstructionOffsets_1_lo_lo;
  assign usedIndexedInstructionOffsets_1_lo_lo = _GEN_0;
  wire [1:0]       usedIndexedInstructionOffsets_2_lo_lo;
  assign usedIndexedInstructionOffsets_2_lo_lo = _GEN_0;
  wire [1:0]       usedIndexedInstructionOffsets_3_lo_lo;
  assign usedIndexedInstructionOffsets_3_lo_lo = _GEN_0;
  wire [1:0]       usedIndexedInstructionOffsets_4_lo_lo;
  assign usedIndexedInstructionOffsets_4_lo_lo = _GEN_0;
  wire [1:0]       usedIndexedInstructionOffsets_5_lo_lo;
  assign usedIndexedInstructionOffsets_5_lo_lo = _GEN_0;
  wire [1:0]       usedIndexedInstructionOffsets_6_lo_lo;
  assign usedIndexedInstructionOffsets_6_lo_lo = _GEN_0;
  wire [1:0]       usedIndexedInstructionOffsets_7_lo_lo;
  assign usedIndexedInstructionOffsets_7_lo_lo = _GEN_0;
  wire [1:0]       _GEN_1 = {waitQueueDeq_3, waitQueueDeq_2};
  wire [1:0]       usedIndexedInstructionOffsets_0_lo_hi;
  assign usedIndexedInstructionOffsets_0_lo_hi = _GEN_1;
  wire [1:0]       usedIndexedInstructionOffsets_1_lo_hi;
  assign usedIndexedInstructionOffsets_1_lo_hi = _GEN_1;
  wire [1:0]       usedIndexedInstructionOffsets_2_lo_hi;
  assign usedIndexedInstructionOffsets_2_lo_hi = _GEN_1;
  wire [1:0]       usedIndexedInstructionOffsets_3_lo_hi;
  assign usedIndexedInstructionOffsets_3_lo_hi = _GEN_1;
  wire [1:0]       usedIndexedInstructionOffsets_4_lo_hi;
  assign usedIndexedInstructionOffsets_4_lo_hi = _GEN_1;
  wire [1:0]       usedIndexedInstructionOffsets_5_lo_hi;
  assign usedIndexedInstructionOffsets_5_lo_hi = _GEN_1;
  wire [1:0]       usedIndexedInstructionOffsets_6_lo_hi;
  assign usedIndexedInstructionOffsets_6_lo_hi = _GEN_1;
  wire [1:0]       usedIndexedInstructionOffsets_7_lo_hi;
  assign usedIndexedInstructionOffsets_7_lo_hi = _GEN_1;
  wire [3:0]       usedIndexedInstructionOffsets_0_lo = {usedIndexedInstructionOffsets_0_lo_hi, usedIndexedInstructionOffsets_0_lo_lo};
  wire [1:0]       _GEN_2 = {waitQueueDeq_5, waitQueueDeq_4};
  wire [1:0]       usedIndexedInstructionOffsets_0_hi_lo;
  assign usedIndexedInstructionOffsets_0_hi_lo = _GEN_2;
  wire [1:0]       usedIndexedInstructionOffsets_1_hi_lo;
  assign usedIndexedInstructionOffsets_1_hi_lo = _GEN_2;
  wire [1:0]       usedIndexedInstructionOffsets_2_hi_lo;
  assign usedIndexedInstructionOffsets_2_hi_lo = _GEN_2;
  wire [1:0]       usedIndexedInstructionOffsets_3_hi_lo;
  assign usedIndexedInstructionOffsets_3_hi_lo = _GEN_2;
  wire [1:0]       usedIndexedInstructionOffsets_4_hi_lo;
  assign usedIndexedInstructionOffsets_4_hi_lo = _GEN_2;
  wire [1:0]       usedIndexedInstructionOffsets_5_hi_lo;
  assign usedIndexedInstructionOffsets_5_hi_lo = _GEN_2;
  wire [1:0]       usedIndexedInstructionOffsets_6_hi_lo;
  assign usedIndexedInstructionOffsets_6_hi_lo = _GEN_2;
  wire [1:0]       usedIndexedInstructionOffsets_7_hi_lo;
  assign usedIndexedInstructionOffsets_7_hi_lo = _GEN_2;
  wire [1:0]       _GEN_3 = {waitQueueDeq_7, waitQueueDeq_6};
  wire [1:0]       usedIndexedInstructionOffsets_0_hi_hi;
  assign usedIndexedInstructionOffsets_0_hi_hi = _GEN_3;
  wire [1:0]       usedIndexedInstructionOffsets_1_hi_hi;
  assign usedIndexedInstructionOffsets_1_hi_hi = _GEN_3;
  wire [1:0]       usedIndexedInstructionOffsets_2_hi_hi;
  assign usedIndexedInstructionOffsets_2_hi_hi = _GEN_3;
  wire [1:0]       usedIndexedInstructionOffsets_3_hi_hi;
  assign usedIndexedInstructionOffsets_3_hi_hi = _GEN_3;
  wire [1:0]       usedIndexedInstructionOffsets_4_hi_hi;
  assign usedIndexedInstructionOffsets_4_hi_hi = _GEN_3;
  wire [1:0]       usedIndexedInstructionOffsets_5_hi_hi;
  assign usedIndexedInstructionOffsets_5_hi_hi = _GEN_3;
  wire [1:0]       usedIndexedInstructionOffsets_6_hi_hi;
  assign usedIndexedInstructionOffsets_6_hi_hi = _GEN_3;
  wire [1:0]       usedIndexedInstructionOffsets_7_hi_hi;
  assign usedIndexedInstructionOffsets_7_hi_hi = _GEN_3;
  wire [3:0]       usedIndexedInstructionOffsets_0_hi = {usedIndexedInstructionOffsets_0_hi_hi, usedIndexedInstructionOffsets_0_hi_lo};
  wire             requestOffset;
  wire             usedIndexedInstructionOffsets_0 = laneOfOffsetOfOffsetGroup[0] & _GEN | requestOffset & (&{usedIndexedInstructionOffsets_0_hi, usedIndexedInstructionOffsets_0_lo}) | maskGroupEndAndRequestNewMask;
  wire [3:0]       usedIndexedInstructionOffsets_1_lo = {usedIndexedInstructionOffsets_1_lo_hi, usedIndexedInstructionOffsets_1_lo_lo};
  wire [3:0]       usedIndexedInstructionOffsets_1_hi = {usedIndexedInstructionOffsets_1_hi_hi, usedIndexedInstructionOffsets_1_hi_lo};
  wire             usedIndexedInstructionOffsets_1 = laneOfOffsetOfOffsetGroup[1] & _GEN | requestOffset & (&{usedIndexedInstructionOffsets_1_hi, usedIndexedInstructionOffsets_1_lo}) | maskGroupEndAndRequestNewMask;
  wire [3:0]       usedIndexedInstructionOffsets_2_lo = {usedIndexedInstructionOffsets_2_lo_hi, usedIndexedInstructionOffsets_2_lo_lo};
  wire [3:0]       usedIndexedInstructionOffsets_2_hi = {usedIndexedInstructionOffsets_2_hi_hi, usedIndexedInstructionOffsets_2_hi_lo};
  wire             usedIndexedInstructionOffsets_2 = laneOfOffsetOfOffsetGroup[2] & _GEN | requestOffset & (&{usedIndexedInstructionOffsets_2_hi, usedIndexedInstructionOffsets_2_lo}) | maskGroupEndAndRequestNewMask;
  wire [3:0]       usedIndexedInstructionOffsets_3_lo = {usedIndexedInstructionOffsets_3_lo_hi, usedIndexedInstructionOffsets_3_lo_lo};
  wire [3:0]       usedIndexedInstructionOffsets_3_hi = {usedIndexedInstructionOffsets_3_hi_hi, usedIndexedInstructionOffsets_3_hi_lo};
  wire             usedIndexedInstructionOffsets_3 = laneOfOffsetOfOffsetGroup[3] & _GEN | requestOffset & (&{usedIndexedInstructionOffsets_3_hi, usedIndexedInstructionOffsets_3_lo}) | maskGroupEndAndRequestNewMask;
  wire [3:0]       usedIndexedInstructionOffsets_4_lo = {usedIndexedInstructionOffsets_4_lo_hi, usedIndexedInstructionOffsets_4_lo_lo};
  wire [3:0]       usedIndexedInstructionOffsets_4_hi = {usedIndexedInstructionOffsets_4_hi_hi, usedIndexedInstructionOffsets_4_hi_lo};
  wire             usedIndexedInstructionOffsets_4 = laneOfOffsetOfOffsetGroup[4] & _GEN | requestOffset & (&{usedIndexedInstructionOffsets_4_hi, usedIndexedInstructionOffsets_4_lo}) | maskGroupEndAndRequestNewMask;
  wire [3:0]       usedIndexedInstructionOffsets_5_lo = {usedIndexedInstructionOffsets_5_lo_hi, usedIndexedInstructionOffsets_5_lo_lo};
  wire [3:0]       usedIndexedInstructionOffsets_5_hi = {usedIndexedInstructionOffsets_5_hi_hi, usedIndexedInstructionOffsets_5_hi_lo};
  wire             usedIndexedInstructionOffsets_5 = laneOfOffsetOfOffsetGroup[5] & _GEN | requestOffset & (&{usedIndexedInstructionOffsets_5_hi, usedIndexedInstructionOffsets_5_lo}) | maskGroupEndAndRequestNewMask;
  wire [3:0]       usedIndexedInstructionOffsets_6_lo = {usedIndexedInstructionOffsets_6_lo_hi, usedIndexedInstructionOffsets_6_lo_lo};
  wire [3:0]       usedIndexedInstructionOffsets_6_hi = {usedIndexedInstructionOffsets_6_hi_hi, usedIndexedInstructionOffsets_6_hi_lo};
  wire             usedIndexedInstructionOffsets_6 = laneOfOffsetOfOffsetGroup[6] & _GEN | requestOffset & (&{usedIndexedInstructionOffsets_6_hi, usedIndexedInstructionOffsets_6_lo}) | maskGroupEndAndRequestNewMask;
  wire [3:0]       usedIndexedInstructionOffsets_7_lo = {usedIndexedInstructionOffsets_7_lo_hi, usedIndexedInstructionOffsets_7_lo_lo};
  wire [3:0]       usedIndexedInstructionOffsets_7_hi = {usedIndexedInstructionOffsets_7_hi_hi, usedIndexedInstructionOffsets_7_hi_lo};
  wire             usedIndexedInstructionOffsets_7 = laneOfOffsetOfOffsetGroup[7] & _GEN | requestOffset & (&{usedIndexedInstructionOffsets_7_hi, usedIndexedInstructionOffsets_7_lo}) | maskGroupEndAndRequestNewMask;
  wire [31:0]      memoryRequestSourceOH;
  wire             sourceFree = (memoryRequestSourceOH & outstandingTLDMessages) == 32'h0;
  wire             stateIsRequest = state == 2'h1;
  wire [11:0]      nextElementIndex = {stateIsRequest ? groupIndex : nextGroupIndex, stateIsRequest ? nextElementForMemoryRequestIndex : 5'h0};
  wire [9:0]       wholeEvl = {{1'h0, lsuRequestReg_instructionInformation_nf} + 4'h1, 6'h0};
  wire [11:0]      evl = isWholeRegisterLoadStore ? {2'h0, wholeEvl} : isMaskLoadStore ? {3'h0, csrInterfaceReg_vl[11:3] + {8'h0, |(csrInterfaceReg_vl[2:0])}} : csrInterfaceReg_vl;
  assign last = nextElementIndex >= evl;
  wire             vrfWritePort_bits_last_0 = last;
  wire             maskCheck = ~lsuRequestReg_instructionInformation_maskedLoadStore | ~noMoreMaskedUnsentMemoryRequests;
  wire             indexCheck = ~isIndexedLoadStore | offsetValidCheck & offsetGroupCheck;
  reg              firstMemoryRequestOfInstruction;
  reg              waitFirstMemoryResponseForFaultOnlyFirst;
  wire             fofCheck = firstMemoryRequestOfInstruction | ~waitFirstMemoryResponseForFaultOnlyFirst;
  wire             _requestOffset_T = stateIsRequest & maskCheck;
  wire             stateReady = _requestOffset_T & indexCheck & fofCheck;
  assign requestOffset = _requestOffset_T & ~indexCheck & fofCheck;
  wire             s0EnqueueValid = stateReady & ~last;
  reg              s0Valid;
  reg  [4:0]       s0Reg_readVS;
  wire [4:0]       vrfReadDataPorts_bits_vs_0 = s0Reg_readVS;
  reg  [2:0]       s0Reg_offsetForVSInLane;
  wire [2:0]       vrfReadDataPorts_bits_offset_0 = s0Reg_offsetForVSInLane;
  reg  [31:0]      s0Reg_addressOffset;
  reg  [2:0]       s0Reg_segmentIndex;
  wire [2:0]       s1EnqQueue_enq_bits_segmentIndex = s0Reg_segmentIndex;
  reg  [2:0]       s0Reg_offsetForLane;
  reg  [4:0]       s0Reg_indexInGroup;
  wire [4:0]       s1EnqQueue_enq_bits_indexInMaskGroup = s0Reg_indexInGroup;
  wire             s1EnqDataQueue_deq_ready = s1EnqueueReady;
  wire             s1EnqQueue_deq_valid;
  assign s1EnqQueue_deq_valid = ~_s1EnqQueue_fifo_empty;
  wire [4:0]       s1EnqQueue_dataOut_indexInMaskGroup;
  wire [2:0]       s1EnqQueue_dataOut_segmentIndex;
  wire [4:0]       s1Wire_indexInMaskGroup = s1EnqQueue_deq_bits_indexInMaskGroup;
  wire [31:0]      s1EnqQueue_dataOut_address;
  wire [2:0]       s1Wire_segmentIndex = s1EnqQueue_deq_bits_segmentIndex;
  wire [31:0]      s1EnqQueue_dataOut_readData;
  wire [31:0]      s1Wire_address = s1EnqQueue_deq_bits_address;
  wire [31:0]      s1EnqQueue_enq_bits_address;
  wire [63:0]      s1EnqQueue_dataIn_lo = {s1EnqQueue_enq_bits_address, 32'h0};
  wire [7:0]       s1EnqQueue_dataIn_hi = {s1EnqQueue_enq_bits_indexInMaskGroup, s1EnqQueue_enq_bits_segmentIndex};
  wire [71:0]      s1EnqQueue_dataIn = {s1EnqQueue_dataIn_hi, s1EnqQueue_dataIn_lo};
  assign s1EnqQueue_dataOut_readData = _s1EnqQueue_fifo_data_out[31:0];
  assign s1EnqQueue_dataOut_address = _s1EnqQueue_fifo_data_out[63:32];
  assign s1EnqQueue_dataOut_segmentIndex = _s1EnqQueue_fifo_data_out[66:64];
  assign s1EnqQueue_dataOut_indexInMaskGroup = _s1EnqQueue_fifo_data_out[71:67];
  assign s1EnqQueue_deq_bits_indexInMaskGroup = s1EnqQueue_dataOut_indexInMaskGroup;
  assign s1EnqQueue_deq_bits_segmentIndex = s1EnqQueue_dataOut_segmentIndex;
  assign s1EnqQueue_deq_bits_address = s1EnqQueue_dataOut_address;
  wire [31:0]      s1EnqQueue_deq_bits_readData = s1EnqQueue_dataOut_readData;
  wire             s1EnqQueue_enq_ready = ~_s1EnqQueue_fifo_full;
  wire             s1EnqQueue_enq_valid;
  wire             s0DequeueFire = s1EnqQueue_enq_ready & s1EnqQueue_enq_valid;
  wire             s1EnqQueue_deq_ready;
  wire             s1EnqDataQueue_deq_valid;
  assign s1EnqDataQueue_deq_valid = ~_s1EnqDataQueue_fifo_empty;
  wire [31:0]      s1Wire_readData = s1EnqDataQueue_deq_bits;
  wire             s1EnqDataQueue_enq_ready = ~_s1EnqDataQueue_fifo_full;
  wire [14:0]      _GEN_4 = {13'h0, dataEEW};
  wire [14:0]      _storeBaseByteOffset_T = {3'h0, s0ElementIndex} << _GEN_4;
  wire [10:0]      storeBaseByteOffset = _storeBaseByteOffset_T[10:0];
  wire [4:0]       _GEN_5 = {1'h0, segmentInstructionIndexInterval};
  wire [4:0]       s0Wire_readVS = lsuRequestReg_instructionInformation_vs3 + (isSegmentLoadStore ? {2'h0, segmentIndex} * _GEN_5 : 5'h0) + {2'h0, storeBaseByteOffset[10:8]};
  wire [2:0]       s0Wire_offsetForVSInLane = storeBaseByteOffset[7:5];
  wire [31:0]      s0Wire_addressOffset = baseOffsetForElement[31:0] + {26'h0, {3'h0, elementByteWidth} * {3'h0, segmentIndex}};
  wire [2:0]       s0Wire_offsetForLane = storeBaseByteOffset[4:2];
  wire             vrfReadDataPorts_valid_0 = s0Valid & lsuRequestReg_instructionInformation_isStore & s1EnqQueue_enq_ready;
  wire             readReady = ~lsuRequestReg_instructionInformation_isStore | vrfReadDataPorts_ready_0;
  reg              s1Valid;
  reg  [4:0]       s1Reg_indexInMaskGroup;
  reg  [2:0]       s1Reg_segmentIndex;
  reg  [31:0]      s1Reg_address;
  wire [31:0]      memWriteRequest_bits_address_0 = s1Reg_address;
  reg  [31:0]      s1Reg_readData;
  wire             memRequestReady = lsuRequestReg_instructionInformation_isStore ? memWriteRequest_ready_0 : memReadRequest_ready_0;
  wire             s2EnqueueReady = memRequestReady & sourceFree;
  assign s1EnqueueReady = s2EnqueueReady | ~s1Valid;
  wire             s0EnqueueReady = s1EnqQueue_enq_ready & readReady | ~s0Valid;
  assign s0Fire = s0EnqueueReady & s0EnqueueValid;
  wire             pipelineClear = ~s0Valid & ~s1Valid & ~s1EnqQueue_deq_valid;
  assign s1EnqQueue_enq_valid = s0Valid & readReady;
  assign s1EnqQueue_enq_bits_address = lsuRequestReg_rs1Data + s0Reg_addressOffset;
  wire             s1DataEnqValid = s1EnqDataQueue_deq_valid | ~lsuRequestReg_instructionInformation_isStore;
  wire             s1EnqValid = s1DataEnqValid & s1EnqQueue_deq_valid;
  wire             s1Fire = s1EnqValid & s1EnqueueReady;
  assign s1EnqQueue_deq_ready = s1EnqueueReady & s1DataEnqValid;
  wire [1:0]       addressInBeatByte = s1Reg_address[1:0];
  wire [3:0]       baseMask = {{2{dataEEWOH[2]}}, ~(dataEEWOH[0]), 1'h1};
  wire [6:0]       _storeMask_T = {3'h0, baseMask} << addressInBeatByte;
  assign storeMask = _storeMask_T[3:0];
  wire [3:0]       memWriteRequest_bits_mask_0 = storeMask;
  wire [4:0]       _storeOffsetByIndex_T_1 = {3'h0, s1Reg_indexInMaskGroup[1:0]} << dataEEW;
  wire [1:0]       storeOffsetByIndex = _storeOffsetByIndex_T_1[1:0];
  wire [62:0]      storeData = {31'h0, s1Reg_readData} << {58'h0, addressInBeatByte, 3'h0} >> {58'h0, storeOffsetByIndex, 3'h0};
  assign memoryRequestSource = isSegmentLoadStore ? {s1Reg_indexInMaskGroup, s1Reg_segmentIndex} : {3'h0, s1Reg_indexInMaskGroup};
  wire [7:0]       memReadRequest_bits_source_0 = memoryRequestSource;
  wire [7:0]       memWriteRequest_bits_source_0 = memoryRequestSource;
  assign memoryRequestSourceOH = 32'h1 << memoryRequestSource[4:0];
  wire [31:0]      memReadRequest_bits_address_0 = {s1Reg_address[31:2], 2'h0};
  wire             _memWriteRequest_valid_T = s1Valid & sourceFree;
  assign memReadRequest_valid_0 = _memWriteRequest_valid_T & ~lsuRequestReg_instructionInformation_isStore;
  assign memWriteRequest_valid_0 = _memWriteRequest_valid_T & lsuRequestReg_instructionInformation_isStore;
  wire [31:0]      memWriteRequest_bits_data_0 = storeData[31:0];
  reg  [1:0]       offsetRecord_0;
  reg  [1:0]       offsetRecord_1;
  reg  [1:0]       offsetRecord_2;
  reg  [1:0]       offsetRecord_3;
  reg  [1:0]       offsetRecord_4;
  reg  [1:0]       offsetRecord_5;
  reg  [1:0]       offsetRecord_6;
  reg  [1:0]       offsetRecord_7;
  reg  [1:0]       offsetRecord_8;
  reg  [1:0]       offsetRecord_9;
  reg  [1:0]       offsetRecord_10;
  reg  [1:0]       offsetRecord_11;
  reg  [1:0]       offsetRecord_12;
  reg  [1:0]       offsetRecord_13;
  reg  [1:0]       offsetRecord_14;
  reg  [1:0]       offsetRecord_15;
  reg  [1:0]       offsetRecord_16;
  reg  [1:0]       offsetRecord_17;
  reg  [1:0]       offsetRecord_18;
  reg  [1:0]       offsetRecord_19;
  reg  [1:0]       offsetRecord_20;
  reg  [1:0]       offsetRecord_21;
  reg  [1:0]       offsetRecord_22;
  reg  [1:0]       offsetRecord_23;
  reg  [1:0]       offsetRecord_24;
  reg  [1:0]       offsetRecord_25;
  reg  [1:0]       offsetRecord_26;
  reg  [1:0]       offsetRecord_27;
  reg  [1:0]       offsetRecord_28;
  reg  [1:0]       offsetRecord_29;
  reg  [1:0]       offsetRecord_30;
  reg  [1:0]       offsetRecord_31;
  wire [4:0]       indexInMaskGroupResponse = isSegmentLoadStore ? memReadResponse_bits_source_0[7:3] : memReadResponse_bits_source_0[4:0];
  wire [31:0]      responseSourceLSBOH = 32'h1 << memReadResponse_bits_source_0[4:0];
  wire [14:0]      loadBaseByteOffset = {3'h0, groupIndex, indexInMaskGroupResponse} << _GEN_4;
  wire [31:0][1:0] _GEN_6 =
    {{offsetRecord_31},
     {offsetRecord_30},
     {offsetRecord_29},
     {offsetRecord_28},
     {offsetRecord_27},
     {offsetRecord_26},
     {offsetRecord_25},
     {offsetRecord_24},
     {offsetRecord_23},
     {offsetRecord_22},
     {offsetRecord_21},
     {offsetRecord_20},
     {offsetRecord_19},
     {offsetRecord_18},
     {offsetRecord_17},
     {offsetRecord_16},
     {offsetRecord_15},
     {offsetRecord_14},
     {offsetRecord_13},
     {offsetRecord_12},
     {offsetRecord_11},
     {offsetRecord_10},
     {offsetRecord_9},
     {offsetRecord_8},
     {offsetRecord_7},
     {offsetRecord_6},
     {offsetRecord_5},
     {offsetRecord_4},
     {offsetRecord_3},
     {offsetRecord_2},
     {offsetRecord_1},
     {offsetRecord_0}};
  wire [4:0]       addressOffset = {_GEN_6[memReadResponse_bits_source_0[4:0]], 3'h0};
  wire [62:0]      _vrfWritePort_bits_data_T_3 = {31'h0, memReadResponse_bits_data_0 >> addressOffset} << {58'h0, loadBaseByteOffset[1:0], 3'h0};
  wire [31:0]      vrfWritePort_bits_data_0 = _vrfWritePort_bits_data_T_3[31:0];
  wire [3:0]       vrfWritePort_bits_mask_0 = (dataEEWOH[0] ? 4'h1 << loadBaseByteOffset[1:0] : 4'h0) | (dataEEWOH[1] ? {{2{loadBaseByteOffset[1]}}, ~(loadBaseByteOffset[1]), ~(loadBaseByteOffset[1])} : 4'h0) | {4{dataEEWOH[2]}};
  wire [4:0]       vrfWritePort_bits_vd_0 = lsuRequestReg_instructionInformation_vs3 + (isSegmentLoadStore ? {2'h0, memReadResponse_bits_source_0[2:0]} * _GEN_5 : 5'h0) + {2'h0, loadBaseByteOffset[10:8]};
  assign writeOffset = loadBaseByteOffset[7:5];
  wire [2:0]       vrfWritePort_bits_offset_0 = writeOffset;
  wire             _GEN_7 = state == 2'h2 & noOutstandingMessages & pipelineClear & ~memReadResponse_valid_0;
  wire             invalidInstruction = csrInterface_vl == 12'h0 & ~requestIsWholeRegisterLoadStore & lsuRequest_valid;
  reg              invalidInstructionNext;
  assign stateIdle = state == 2'h0;
  wire             allElementsMasked =
    stateIdle & (_allElementsMasked_T_1 | _allElementsMasked_T_2 | _allElementsMasked_T_3 | _allElementsMasked_T_4 | _allElementsMasked_T_5 | _allElementsMasked_T_6 | _allElementsMasked_T_7 | _allElementsMasked_T_8);
  wire             _GEN_12 = lsuRequest_valid & ~invalidInstruction;
  wire             updateOffsetGroupEnable = _GEN_12 | _GEN_7 & ~last;
  reg              status_last_REG;
  wire             _status_last_output = ~status_last_REG & stateIdle | invalidInstructionNext | allElementsMasked;
  wire [7:0]       _dataOffset_T = {3'h0, s1EnqQueue_deq_bits_indexInMaskGroup} << dataEEW;
  wire [4:0]       dataOffset = {_dataOffset_T[1:0], 3'h0};
  wire [31:0]      offsetQueueVec_0_deq_bits;
  wire [31:0]      offsetQueueVec_1_deq_bits;
  wire [31:0]      offsetQueueVec_2_deq_bits;
  wire [31:0]      offsetQueueVec_3_deq_bits;
  wire [31:0]      offsetQueueVec_4_deq_bits;
  wire [31:0]      offsetQueueVec_5_deq_bits;
  wire [31:0]      offsetQueueVec_6_deq_bits;
  wire [31:0]      offsetQueueVec_7_deq_bits;
  always @(posedge clock) begin
    if (reset) begin
      offsetQueueVec_deqLock <= 1'h0;
      offsetQueueVec_deqLock_1 <= 1'h0;
      offsetQueueVec_deqLock_2 <= 1'h0;
      offsetQueueVec_deqLock_3 <= 1'h0;
      offsetQueueVec_deqLock_4 <= 1'h0;
      offsetQueueVec_deqLock_5 <= 1'h0;
      offsetQueueVec_deqLock_6 <= 1'h0;
      offsetQueueVec_deqLock_7 <= 1'h0;
      lsuRequestReg_instructionInformation_nf <= 3'h0;
      lsuRequestReg_instructionInformation_mew <= 1'h0;
      lsuRequestReg_instructionInformation_mop <= 2'h0;
      lsuRequestReg_instructionInformation_lumop <= 5'h0;
      lsuRequestReg_instructionInformation_eew <= 2'h0;
      lsuRequestReg_instructionInformation_vs3 <= 5'h0;
      lsuRequestReg_instructionInformation_isStore <= 1'h0;
      lsuRequestReg_instructionInformation_maskedLoadStore <= 1'h0;
      lsuRequestReg_rs1Data <= 32'h0;
      lsuRequestReg_rs2Data <= 32'h0;
      lsuRequestReg_instructionIndex <= 3'h0;
      csrInterfaceReg_vl <= 12'h0;
      csrInterfaceReg_vStart <= 12'h0;
      csrInterfaceReg_vlmul <= 3'h0;
      csrInterfaceReg_vSew <= 2'h0;
      csrInterfaceReg_vxrm <= 2'h0;
      csrInterfaceReg_vta <= 1'h0;
      csrInterfaceReg_vma <= 1'h0;
      dataWidthForSegmentLoadStore <= 7'h0;
      elementByteWidth <= 3'h0;
      segmentInstructionIndexInterval <= 4'h0;
      outstandingTLDMessages <= 32'h0;
      indexedInstructionOffsets_0_valid <= 1'h0;
      indexedInstructionOffsets_0_bits <= 32'h0;
      indexedInstructionOffsets_1_valid <= 1'h0;
      indexedInstructionOffsets_1_bits <= 32'h0;
      indexedInstructionOffsets_2_valid <= 1'h0;
      indexedInstructionOffsets_2_bits <= 32'h0;
      indexedInstructionOffsets_3_valid <= 1'h0;
      indexedInstructionOffsets_3_bits <= 32'h0;
      indexedInstructionOffsets_4_valid <= 1'h0;
      indexedInstructionOffsets_4_bits <= 32'h0;
      indexedInstructionOffsets_5_valid <= 1'h0;
      indexedInstructionOffsets_5_bits <= 32'h0;
      indexedInstructionOffsets_6_valid <= 1'h0;
      indexedInstructionOffsets_6_bits <= 32'h0;
      indexedInstructionOffsets_7_valid <= 1'h0;
      indexedInstructionOffsets_7_bits <= 32'h0;
      groupIndex <= 7'h0;
      maskReg <= 32'h0;
      segmentIndex <= 3'h0;
      state <= 2'h0;
      sentMemoryRequests <= 32'h0;
      firstMemoryRequestOfInstruction <= 1'h0;
      waitFirstMemoryResponseForFaultOnlyFirst <= 1'h0;
      s0Valid <= 1'h0;
      s0Reg_readVS <= 5'h0;
      s0Reg_offsetForVSInLane <= 3'h0;
      s0Reg_addressOffset <= 32'h0;
      s0Reg_segmentIndex <= 3'h0;
      s0Reg_offsetForLane <= 3'h0;
      s0Reg_indexInGroup <= 5'h0;
      s1Valid <= 1'h0;
      s1Reg_indexInMaskGroup <= 5'h0;
      s1Reg_segmentIndex <= 3'h0;
      s1Reg_address <= 32'h0;
      s1Reg_readData <= 32'h0;
      offsetRecord_0 <= 2'h0;
      offsetRecord_1 <= 2'h0;
      offsetRecord_2 <= 2'h0;
      offsetRecord_3 <= 2'h0;
      offsetRecord_4 <= 2'h0;
      offsetRecord_5 <= 2'h0;
      offsetRecord_6 <= 2'h0;
      offsetRecord_7 <= 2'h0;
      offsetRecord_8 <= 2'h0;
      offsetRecord_9 <= 2'h0;
      offsetRecord_10 <= 2'h0;
      offsetRecord_11 <= 2'h0;
      offsetRecord_12 <= 2'h0;
      offsetRecord_13 <= 2'h0;
      offsetRecord_14 <= 2'h0;
      offsetRecord_15 <= 2'h0;
      offsetRecord_16 <= 2'h0;
      offsetRecord_17 <= 2'h0;
      offsetRecord_18 <= 2'h0;
      offsetRecord_19 <= 2'h0;
      offsetRecord_20 <= 2'h0;
      offsetRecord_21 <= 2'h0;
      offsetRecord_22 <= 2'h0;
      offsetRecord_23 <= 2'h0;
      offsetRecord_24 <= 2'h0;
      offsetRecord_25 <= 2'h0;
      offsetRecord_26 <= 2'h0;
      offsetRecord_27 <= 2'h0;
      offsetRecord_28 <= 2'h0;
      offsetRecord_29 <= 2'h0;
      offsetRecord_30 <= 2'h0;
      offsetRecord_31 <= 2'h0;
    end
    else begin
      automatic logic _offsetQueueVec_T_21 = lsuRequest_valid | requestOffset;
      automatic logic _outstandingTLDMessages_T;
      _outstandingTLDMessages_T = memReadResponse_ready_0 & memReadResponse_valid_0;
      if (_offsetQueueVec_T_21 | _allElementsMasked_T_1)
        offsetQueueVec_deqLock <= _allElementsMasked_T_1;
      if (_offsetQueueVec_T_21 | _allElementsMasked_T_2)
        offsetQueueVec_deqLock_1 <= _allElementsMasked_T_2;
      if (_offsetQueueVec_T_21 | _allElementsMasked_T_3)
        offsetQueueVec_deqLock_2 <= _allElementsMasked_T_3;
      if (_offsetQueueVec_T_21 | _allElementsMasked_T_4)
        offsetQueueVec_deqLock_3 <= _allElementsMasked_T_4;
      if (_offsetQueueVec_T_21 | _allElementsMasked_T_5)
        offsetQueueVec_deqLock_4 <= _allElementsMasked_T_5;
      if (_offsetQueueVec_T_21 | _allElementsMasked_T_6)
        offsetQueueVec_deqLock_5 <= _allElementsMasked_T_6;
      if (_offsetQueueVec_T_21 | _allElementsMasked_T_7)
        offsetQueueVec_deqLock_6 <= _allElementsMasked_T_7;
      if (_offsetQueueVec_T_21 | _allElementsMasked_T_8)
        offsetQueueVec_deqLock_7 <= _allElementsMasked_T_8;
      if (lsuRequest_valid) begin
        automatic logic [3:0] _elementByteWidth_T = 4'h1 << requestEEW;
        lsuRequestReg_instructionInformation_nf <= lsuRequest_bits_instructionInformation_nf;
        lsuRequestReg_instructionInformation_mew <= lsuRequest_bits_instructionInformation_mew;
        lsuRequestReg_instructionInformation_mop <= lsuRequest_bits_instructionInformation_mop;
        lsuRequestReg_instructionInformation_lumop <= lsuRequest_bits_instructionInformation_lumop;
        lsuRequestReg_instructionInformation_eew <= lsuRequest_bits_instructionInformation_eew;
        lsuRequestReg_instructionInformation_vs3 <= lsuRequest_bits_instructionInformation_vs3;
        lsuRequestReg_instructionInformation_isStore <= lsuRequest_bits_instructionInformation_isStore;
        lsuRequestReg_instructionInformation_maskedLoadStore <= lsuRequest_bits_instructionInformation_maskedLoadStore;
        lsuRequestReg_rs1Data <= lsuRequest_bits_rs1Data;
        lsuRequestReg_rs2Data <= lsuRequest_bits_rs2Data;
        lsuRequestReg_instructionIndex <= lsuRequest_bits_instructionIndex;
        csrInterfaceReg_vl <= csrInterface_vl;
        csrInterfaceReg_vStart <= csrInterface_vStart;
        csrInterfaceReg_vlmul <= csrInterface_vlmul;
        csrInterfaceReg_vSew <= csrInterface_vSew;
        csrInterfaceReg_vxrm <= csrInterface_vxrm;
        csrInterfaceReg_vta <= csrInterface_vta;
        csrInterfaceReg_vma <= csrInterface_vma;
        dataWidthForSegmentLoadStore <= {3'h0, {1'h0, requestNF} + 4'h1} * {4'h0, _elementByteWidth_T[2:0]};
        elementByteWidth <= _elementByteWidth_T[2:0];
        segmentInstructionIndexInterval <= csrInterface_vlmul[2] ? 4'h1 : 4'h1 << csrInterface_vlmul[1:0];
      end
      if ((_outstandingTLDMessages_T | _outstandingTLDMessages_T_4) & ~lsuRequestReg_instructionInformation_isStore)
        outstandingTLDMessages <= outstandingTLDMessages & ~(_outstandingTLDMessages_T ? responseSourceLSBOH : 32'h0) | (_outstandingTLDMessages_T_4 ? memoryRequestSourceOH : 32'h0);
      indexedInstructionOffsets_0_valid <= _allElementsMasked_T_1 | indexedInstructionOffsets_0_valid & ~usedIndexedInstructionOffsets_0 & ~_status_last_output;
      if (_allElementsMasked_T_1)
        indexedInstructionOffsets_0_bits <= offsetQueueVec_0_deq_bits;
      indexedInstructionOffsets_1_valid <= _allElementsMasked_T_2 | indexedInstructionOffsets_1_valid & ~usedIndexedInstructionOffsets_1 & ~_status_last_output;
      if (_allElementsMasked_T_2)
        indexedInstructionOffsets_1_bits <= offsetQueueVec_1_deq_bits;
      indexedInstructionOffsets_2_valid <= _allElementsMasked_T_3 | indexedInstructionOffsets_2_valid & ~usedIndexedInstructionOffsets_2 & ~_status_last_output;
      if (_allElementsMasked_T_3)
        indexedInstructionOffsets_2_bits <= offsetQueueVec_2_deq_bits;
      indexedInstructionOffsets_3_valid <= _allElementsMasked_T_4 | indexedInstructionOffsets_3_valid & ~usedIndexedInstructionOffsets_3 & ~_status_last_output;
      if (_allElementsMasked_T_4)
        indexedInstructionOffsets_3_bits <= offsetQueueVec_3_deq_bits;
      indexedInstructionOffsets_4_valid <= _allElementsMasked_T_5 | indexedInstructionOffsets_4_valid & ~usedIndexedInstructionOffsets_4 & ~_status_last_output;
      if (_allElementsMasked_T_5)
        indexedInstructionOffsets_4_bits <= offsetQueueVec_4_deq_bits;
      indexedInstructionOffsets_5_valid <= _allElementsMasked_T_6 | indexedInstructionOffsets_5_valid & ~usedIndexedInstructionOffsets_5 & ~_status_last_output;
      if (_allElementsMasked_T_6)
        indexedInstructionOffsets_5_bits <= offsetQueueVec_5_deq_bits;
      indexedInstructionOffsets_6_valid <= _allElementsMasked_T_7 | indexedInstructionOffsets_6_valid & ~usedIndexedInstructionOffsets_6 & ~_status_last_output;
      if (_allElementsMasked_T_7)
        indexedInstructionOffsets_6_bits <= offsetQueueVec_6_deq_bits;
      indexedInstructionOffsets_7_valid <= _allElementsMasked_T_8 | indexedInstructionOffsets_7_valid & ~usedIndexedInstructionOffsets_7 & ~_status_last_output;
      if (_allElementsMasked_T_8)
        indexedInstructionOffsets_7_bits <= offsetQueueVec_7_deq_bits;
      if (updateOffsetGroupEnable)
        groupIndex <= nextGroupIndex;
      if (maskGroupEndAndRequestNewMask | lsuRequest_valid)
        maskReg <= maskInput;
      if (isSegmentLoadStore & s0Fire | lsuRequest_valid)
        segmentIndex <= segmentEnd | lsuRequest_valid ? 3'h0 : segmentIndexNext;
      if (_GEN_12)
        state <= 2'h1;
      else if (_GEN_7)
        state <= {1'h0, ~last};
      else if (stateIsRequest & (last | maskGroupEnd))
        state <= 2'h2;
      if (segmentEndWithHandshake | updateOffsetGroupEnable) begin
        automatic logic [30:0] _GEN_8 = nextElementForMemoryRequest[30:0] | nextElementForMemoryRequest[31:1];
        automatic logic [29:0] _GEN_9 = _GEN_8[29:0] | {nextElementForMemoryRequest[31], _GEN_8[30:2]};
        automatic logic [27:0] _GEN_10 = _GEN_9[27:0] | {nextElementForMemoryRequest[31], _GEN_8[30], _GEN_9[29:4]};
        automatic logic [23:0] _GEN_11 = _GEN_10[23:0] | {nextElementForMemoryRequest[31], _GEN_8[30], _GEN_9[29:28], _GEN_10[27:8]};
        sentMemoryRequests <=
          updateOffsetGroupEnable
            ? 32'h0
            : {nextElementForMemoryRequest[31], _GEN_8[30], _GEN_9[29:28], _GEN_10[27:24], _GEN_11[23:16], _GEN_11[15:0] | {nextElementForMemoryRequest[31], _GEN_8[30], _GEN_9[29:28], _GEN_10[27:24], _GEN_11[23:16]}};
      end
      if (lsuRequest_valid | _outstandingTLDMessages_T_4)
        firstMemoryRequestOfInstruction <= lsuRequest_valid;
      if (lsuRequest_valid | _outstandingTLDMessages_T)
        waitFirstMemoryResponseForFaultOnlyFirst <= lsuRequest_valid & _waitFirstMemoryResponseForFaultOnlyFirst_T & lsuRequest_bits_instructionInformation_lumop[4] & ~lsuRequest_bits_instructionInformation_isStore;
      if (s0Fire ^ s0DequeueFire)
        s0Valid <= s0Fire;
      if (s0Fire) begin
        s0Reg_readVS <= s0Wire_readVS;
        s0Reg_offsetForVSInLane <= s0Wire_offsetForVSInLane;
        s0Reg_addressOffset <= s0Wire_addressOffset;
        s0Reg_segmentIndex <= s0Wire_segmentIndex;
        s0Reg_offsetForLane <= s0Wire_offsetForLane;
        s0Reg_indexInGroup <= s0Wire_indexInGroup;
      end
      if (s1Fire ^ memRequestFire)
        s1Valid <= s1Fire;
      if (s1Fire) begin
        s1Reg_indexInMaskGroup <= s1Wire_indexInMaskGroup;
        s1Reg_segmentIndex <= s1Wire_segmentIndex;
        s1Reg_address <= s1Wire_address;
        s1Reg_readData <= s1Wire_readData;
      end
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[0])
        offsetRecord_0 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[1])
        offsetRecord_1 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[2])
        offsetRecord_2 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[3])
        offsetRecord_3 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[4])
        offsetRecord_4 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[5])
        offsetRecord_5 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[6])
        offsetRecord_6 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[7])
        offsetRecord_7 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[8])
        offsetRecord_8 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[9])
        offsetRecord_9 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[10])
        offsetRecord_10 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[11])
        offsetRecord_11 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[12])
        offsetRecord_12 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[13])
        offsetRecord_13 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[14])
        offsetRecord_14 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[15])
        offsetRecord_15 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[16])
        offsetRecord_16 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[17])
        offsetRecord_17 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[18])
        offsetRecord_18 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[19])
        offsetRecord_19 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[20])
        offsetRecord_20 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[21])
        offsetRecord_21 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[22])
        offsetRecord_22 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[23])
        offsetRecord_23 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[24])
        offsetRecord_24 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[25])
        offsetRecord_25 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[26])
        offsetRecord_26 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[27])
        offsetRecord_27 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[28])
        offsetRecord_28 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[29])
        offsetRecord_29 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[30])
        offsetRecord_30 <= addressInBeatByte;
      if (_outstandingTLDMessages_T_4 & memoryRequestSourceOH[31])
        offsetRecord_31 <= addressInBeatByte;
    end
    if (lsuRequest_valid | _allElementsMasked_T_1)
      indexOfIndexedInstructionOffsets <= indexOfIndexedInstructionOffsetsNext;
    invalidInstructionNext <= invalidInstruction;
    status_last_REG <= stateIdle;
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:22];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [4:0] i = 5'h0; i < 5'h17; i += 5'h1) begin
          _RANDOM[i] = `RANDOM;
        end
        offsetQueueVec_deqLock = _RANDOM[5'h0][0];
        offsetQueueVec_deqLock_1 = _RANDOM[5'h0][1];
        offsetQueueVec_deqLock_2 = _RANDOM[5'h0][2];
        offsetQueueVec_deqLock_3 = _RANDOM[5'h0][3];
        offsetQueueVec_deqLock_4 = _RANDOM[5'h0][4];
        offsetQueueVec_deqLock_5 = _RANDOM[5'h0][5];
        offsetQueueVec_deqLock_6 = _RANDOM[5'h0][6];
        offsetQueueVec_deqLock_7 = _RANDOM[5'h0][7];
        lsuRequestReg_instructionInformation_nf = _RANDOM[5'h0][10:8];
        lsuRequestReg_instructionInformation_mew = _RANDOM[5'h0][11];
        lsuRequestReg_instructionInformation_mop = _RANDOM[5'h0][13:12];
        lsuRequestReg_instructionInformation_lumop = _RANDOM[5'h0][18:14];
        lsuRequestReg_instructionInformation_eew = _RANDOM[5'h0][20:19];
        lsuRequestReg_instructionInformation_vs3 = _RANDOM[5'h0][25:21];
        lsuRequestReg_instructionInformation_isStore = _RANDOM[5'h0][26];
        lsuRequestReg_instructionInformation_maskedLoadStore = _RANDOM[5'h0][27];
        lsuRequestReg_rs1Data = {_RANDOM[5'h0][31:28], _RANDOM[5'h1][27:0]};
        lsuRequestReg_rs2Data = {_RANDOM[5'h1][31:28], _RANDOM[5'h2][27:0]};
        lsuRequestReg_instructionIndex = _RANDOM[5'h2][30:28];
        csrInterfaceReg_vl = {_RANDOM[5'h2][31], _RANDOM[5'h3][10:0]};
        csrInterfaceReg_vStart = _RANDOM[5'h3][22:11];
        csrInterfaceReg_vlmul = _RANDOM[5'h3][25:23];
        csrInterfaceReg_vSew = _RANDOM[5'h3][27:26];
        csrInterfaceReg_vxrm = _RANDOM[5'h3][29:28];
        csrInterfaceReg_vta = _RANDOM[5'h3][30];
        csrInterfaceReg_vma = _RANDOM[5'h3][31];
        dataWidthForSegmentLoadStore = _RANDOM[5'h4][6:0];
        elementByteWidth = _RANDOM[5'h4][9:7];
        segmentInstructionIndexInterval = _RANDOM[5'h4][13:10];
        outstandingTLDMessages = {_RANDOM[5'h4][31:14], _RANDOM[5'h5][13:0]};
        indexedInstructionOffsets_0_valid = _RANDOM[5'h5][14];
        indexedInstructionOffsets_0_bits = {_RANDOM[5'h5][31:15], _RANDOM[5'h6][14:0]};
        indexedInstructionOffsets_1_valid = _RANDOM[5'h6][15];
        indexedInstructionOffsets_1_bits = {_RANDOM[5'h6][31:16], _RANDOM[5'h7][15:0]};
        indexedInstructionOffsets_2_valid = _RANDOM[5'h7][16];
        indexedInstructionOffsets_2_bits = {_RANDOM[5'h7][31:17], _RANDOM[5'h8][16:0]};
        indexedInstructionOffsets_3_valid = _RANDOM[5'h8][17];
        indexedInstructionOffsets_3_bits = {_RANDOM[5'h8][31:18], _RANDOM[5'h9][17:0]};
        indexedInstructionOffsets_4_valid = _RANDOM[5'h9][18];
        indexedInstructionOffsets_4_bits = {_RANDOM[5'h9][31:19], _RANDOM[5'hA][18:0]};
        indexedInstructionOffsets_5_valid = _RANDOM[5'hA][19];
        indexedInstructionOffsets_5_bits = {_RANDOM[5'hA][31:20], _RANDOM[5'hB][19:0]};
        indexedInstructionOffsets_6_valid = _RANDOM[5'hB][20];
        indexedInstructionOffsets_6_bits = {_RANDOM[5'hB][31:21], _RANDOM[5'hC][20:0]};
        indexedInstructionOffsets_7_valid = _RANDOM[5'hC][21];
        indexedInstructionOffsets_7_bits = {_RANDOM[5'hC][31:22], _RANDOM[5'hD][21:0]};
        groupIndex = _RANDOM[5'hD][28:22];
        indexOfIndexedInstructionOffsets = _RANDOM[5'hD][30:29];
        maskReg = {_RANDOM[5'hD][31], _RANDOM[5'hE][30:0]};
        segmentIndex = {_RANDOM[5'hE][31], _RANDOM[5'hF][1:0]};
        state = _RANDOM[5'hF][3:2];
        sentMemoryRequests = {_RANDOM[5'hF][31:4], _RANDOM[5'h10][3:0]};
        firstMemoryRequestOfInstruction = _RANDOM[5'h10][4];
        waitFirstMemoryResponseForFaultOnlyFirst = _RANDOM[5'h10][5];
        s0Valid = _RANDOM[5'h10][6];
        s0Reg_readVS = _RANDOM[5'h10][11:7];
        s0Reg_offsetForVSInLane = _RANDOM[5'h10][14:12];
        s0Reg_addressOffset = {_RANDOM[5'h10][31:15], _RANDOM[5'h11][14:0]};
        s0Reg_segmentIndex = _RANDOM[5'h11][17:15];
        s0Reg_offsetForLane = _RANDOM[5'h11][20:18];
        s0Reg_indexInGroup = _RANDOM[5'h11][25:21];
        s1Valid = _RANDOM[5'h11][26];
        s1Reg_indexInMaskGroup = _RANDOM[5'h11][31:27];
        s1Reg_segmentIndex = _RANDOM[5'h12][2:0];
        s1Reg_address = {_RANDOM[5'h12][31:3], _RANDOM[5'h13][2:0]};
        s1Reg_readData = {_RANDOM[5'h13][31:3], _RANDOM[5'h14][2:0]};
        offsetRecord_0 = _RANDOM[5'h14][4:3];
        offsetRecord_1 = _RANDOM[5'h14][6:5];
        offsetRecord_2 = _RANDOM[5'h14][8:7];
        offsetRecord_3 = _RANDOM[5'h14][10:9];
        offsetRecord_4 = _RANDOM[5'h14][12:11];
        offsetRecord_5 = _RANDOM[5'h14][14:13];
        offsetRecord_6 = _RANDOM[5'h14][16:15];
        offsetRecord_7 = _RANDOM[5'h14][18:17];
        offsetRecord_8 = _RANDOM[5'h14][20:19];
        offsetRecord_9 = _RANDOM[5'h14][22:21];
        offsetRecord_10 = _RANDOM[5'h14][24:23];
        offsetRecord_11 = _RANDOM[5'h14][26:25];
        offsetRecord_12 = _RANDOM[5'h14][28:27];
        offsetRecord_13 = _RANDOM[5'h14][30:29];
        offsetRecord_14 = {_RANDOM[5'h14][31], _RANDOM[5'h15][0]};
        offsetRecord_15 = _RANDOM[5'h15][2:1];
        offsetRecord_16 = _RANDOM[5'h15][4:3];
        offsetRecord_17 = _RANDOM[5'h15][6:5];
        offsetRecord_18 = _RANDOM[5'h15][8:7];
        offsetRecord_19 = _RANDOM[5'h15][10:9];
        offsetRecord_20 = _RANDOM[5'h15][12:11];
        offsetRecord_21 = _RANDOM[5'h15][14:13];
        offsetRecord_22 = _RANDOM[5'h15][16:15];
        offsetRecord_23 = _RANDOM[5'h15][18:17];
        offsetRecord_24 = _RANDOM[5'h15][20:19];
        offsetRecord_25 = _RANDOM[5'h15][22:21];
        offsetRecord_26 = _RANDOM[5'h15][24:23];
        offsetRecord_27 = _RANDOM[5'h15][26:25];
        offsetRecord_28 = _RANDOM[5'h15][28:27];
        offsetRecord_29 = _RANDOM[5'h15][30:29];
        offsetRecord_30 = {_RANDOM[5'h15][31], _RANDOM[5'h16][0]};
        offsetRecord_31 = _RANDOM[5'h16][2:1];
        invalidInstructionNext = _RANDOM[5'h16][3];
        status_last_REG = _RANDOM[5'h16][4];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  wire             offsetQueueVec_0_empty;
  assign offsetQueueVec_0_empty = _offsetQueueVec_queue_fifo_empty;
  wire             offsetQueueVec_0_full;
  assign offsetQueueVec_0_full = _offsetQueueVec_queue_fifo_full;
  wire             offsetQueueVec_1_empty;
  assign offsetQueueVec_1_empty = _offsetQueueVec_queue_fifo_1_empty;
  wire             offsetQueueVec_1_full;
  assign offsetQueueVec_1_full = _offsetQueueVec_queue_fifo_1_full;
  wire             offsetQueueVec_2_empty;
  assign offsetQueueVec_2_empty = _offsetQueueVec_queue_fifo_2_empty;
  wire             offsetQueueVec_2_full;
  assign offsetQueueVec_2_full = _offsetQueueVec_queue_fifo_2_full;
  wire             offsetQueueVec_3_empty;
  assign offsetQueueVec_3_empty = _offsetQueueVec_queue_fifo_3_empty;
  wire             offsetQueueVec_3_full;
  assign offsetQueueVec_3_full = _offsetQueueVec_queue_fifo_3_full;
  wire             offsetQueueVec_4_empty;
  assign offsetQueueVec_4_empty = _offsetQueueVec_queue_fifo_4_empty;
  wire             offsetQueueVec_4_full;
  assign offsetQueueVec_4_full = _offsetQueueVec_queue_fifo_4_full;
  wire             offsetQueueVec_5_empty;
  assign offsetQueueVec_5_empty = _offsetQueueVec_queue_fifo_5_empty;
  wire             offsetQueueVec_5_full;
  assign offsetQueueVec_5_full = _offsetQueueVec_queue_fifo_5_full;
  wire             offsetQueueVec_6_empty;
  assign offsetQueueVec_6_empty = _offsetQueueVec_queue_fifo_6_empty;
  wire             offsetQueueVec_6_full;
  assign offsetQueueVec_6_full = _offsetQueueVec_queue_fifo_6_full;
  wire             offsetQueueVec_7_empty;
  assign offsetQueueVec_7_empty = _offsetQueueVec_queue_fifo_7_empty;
  wire             offsetQueueVec_7_full;
  assign offsetQueueVec_7_full = _offsetQueueVec_queue_fifo_7_full;
  wire             s1EnqQueue_empty;
  assign s1EnqQueue_empty = _s1EnqQueue_fifo_empty;
  wire             s1EnqQueue_full;
  assign s1EnqQueue_full = _s1EnqQueue_fifo_full;
  wire             s1EnqDataQueue_empty;
  assign s1EnqDataQueue_empty = _s1EnqDataQueue_fifo_empty;
  wire             s1EnqDataQueue_full;
  assign s1EnqDataQueue_full = _s1EnqDataQueue_fifo_full;
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(8),
    .err_mode(2),
    .rst_mode(3),
    .width(32)
  ) offsetQueueVec_queue_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(offsetQueueVec_0_enq_ready & offsetQueueVec_0_enq_valid)),
    .pop_req_n    (~(offsetQueueVec_0_deq_ready & ~_offsetQueueVec_queue_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (offsetQueueVec_0_enq_bits),
    .empty        (_offsetQueueVec_queue_fifo_empty),
    .almost_empty (offsetQueueVec_0_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (offsetQueueVec_0_almostFull),
    .full         (_offsetQueueVec_queue_fifo_full),
    .error        (_offsetQueueVec_queue_fifo_error),
    .data_out     (offsetQueueVec_0_deq_bits)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(8),
    .err_mode(2),
    .rst_mode(3),
    .width(32)
  ) offsetQueueVec_queue_fifo_1 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(offsetQueueVec_1_enq_ready & offsetQueueVec_1_enq_valid)),
    .pop_req_n    (~(offsetQueueVec_1_deq_ready & ~_offsetQueueVec_queue_fifo_1_empty)),
    .diag_n       (1'h1),
    .data_in      (offsetQueueVec_1_enq_bits),
    .empty        (_offsetQueueVec_queue_fifo_1_empty),
    .almost_empty (offsetQueueVec_1_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (offsetQueueVec_1_almostFull),
    .full         (_offsetQueueVec_queue_fifo_1_full),
    .error        (_offsetQueueVec_queue_fifo_1_error),
    .data_out     (offsetQueueVec_1_deq_bits)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(8),
    .err_mode(2),
    .rst_mode(3),
    .width(32)
  ) offsetQueueVec_queue_fifo_2 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(offsetQueueVec_2_enq_ready & offsetQueueVec_2_enq_valid)),
    .pop_req_n    (~(offsetQueueVec_2_deq_ready & ~_offsetQueueVec_queue_fifo_2_empty)),
    .diag_n       (1'h1),
    .data_in      (offsetQueueVec_2_enq_bits),
    .empty        (_offsetQueueVec_queue_fifo_2_empty),
    .almost_empty (offsetQueueVec_2_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (offsetQueueVec_2_almostFull),
    .full         (_offsetQueueVec_queue_fifo_2_full),
    .error        (_offsetQueueVec_queue_fifo_2_error),
    .data_out     (offsetQueueVec_2_deq_bits)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(8),
    .err_mode(2),
    .rst_mode(3),
    .width(32)
  ) offsetQueueVec_queue_fifo_3 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(offsetQueueVec_3_enq_ready & offsetQueueVec_3_enq_valid)),
    .pop_req_n    (~(offsetQueueVec_3_deq_ready & ~_offsetQueueVec_queue_fifo_3_empty)),
    .diag_n       (1'h1),
    .data_in      (offsetQueueVec_3_enq_bits),
    .empty        (_offsetQueueVec_queue_fifo_3_empty),
    .almost_empty (offsetQueueVec_3_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (offsetQueueVec_3_almostFull),
    .full         (_offsetQueueVec_queue_fifo_3_full),
    .error        (_offsetQueueVec_queue_fifo_3_error),
    .data_out     (offsetQueueVec_3_deq_bits)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(8),
    .err_mode(2),
    .rst_mode(3),
    .width(32)
  ) offsetQueueVec_queue_fifo_4 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(offsetQueueVec_4_enq_ready & offsetQueueVec_4_enq_valid)),
    .pop_req_n    (~(offsetQueueVec_4_deq_ready & ~_offsetQueueVec_queue_fifo_4_empty)),
    .diag_n       (1'h1),
    .data_in      (offsetQueueVec_4_enq_bits),
    .empty        (_offsetQueueVec_queue_fifo_4_empty),
    .almost_empty (offsetQueueVec_4_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (offsetQueueVec_4_almostFull),
    .full         (_offsetQueueVec_queue_fifo_4_full),
    .error        (_offsetQueueVec_queue_fifo_4_error),
    .data_out     (offsetQueueVec_4_deq_bits)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(8),
    .err_mode(2),
    .rst_mode(3),
    .width(32)
  ) offsetQueueVec_queue_fifo_5 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(offsetQueueVec_5_enq_ready & offsetQueueVec_5_enq_valid)),
    .pop_req_n    (~(offsetQueueVec_5_deq_ready & ~_offsetQueueVec_queue_fifo_5_empty)),
    .diag_n       (1'h1),
    .data_in      (offsetQueueVec_5_enq_bits),
    .empty        (_offsetQueueVec_queue_fifo_5_empty),
    .almost_empty (offsetQueueVec_5_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (offsetQueueVec_5_almostFull),
    .full         (_offsetQueueVec_queue_fifo_5_full),
    .error        (_offsetQueueVec_queue_fifo_5_error),
    .data_out     (offsetQueueVec_5_deq_bits)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(8),
    .err_mode(2),
    .rst_mode(3),
    .width(32)
  ) offsetQueueVec_queue_fifo_6 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(offsetQueueVec_6_enq_ready & offsetQueueVec_6_enq_valid)),
    .pop_req_n    (~(offsetQueueVec_6_deq_ready & ~_offsetQueueVec_queue_fifo_6_empty)),
    .diag_n       (1'h1),
    .data_in      (offsetQueueVec_6_enq_bits),
    .empty        (_offsetQueueVec_queue_fifo_6_empty),
    .almost_empty (offsetQueueVec_6_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (offsetQueueVec_6_almostFull),
    .full         (_offsetQueueVec_queue_fifo_6_full),
    .error        (_offsetQueueVec_queue_fifo_6_error),
    .data_out     (offsetQueueVec_6_deq_bits)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(8),
    .err_mode(2),
    .rst_mode(3),
    .width(32)
  ) offsetQueueVec_queue_fifo_7 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(offsetQueueVec_7_enq_ready & offsetQueueVec_7_enq_valid)),
    .pop_req_n    (~(offsetQueueVec_7_deq_ready & ~_offsetQueueVec_queue_fifo_7_empty)),
    .diag_n       (1'h1),
    .data_in      (offsetQueueVec_7_enq_bits),
    .empty        (_offsetQueueVec_queue_fifo_7_empty),
    .almost_empty (offsetQueueVec_7_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (offsetQueueVec_7_almostFull),
    .full         (_offsetQueueVec_queue_fifo_7_full),
    .error        (_offsetQueueVec_queue_fifo_7_error),
    .data_out     (offsetQueueVec_7_deq_bits)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(6),
    .err_mode(2),
    .rst_mode(3),
    .width(72)
  ) s1EnqQueue_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~s0DequeueFire),
    .pop_req_n    (~(s1EnqQueue_deq_ready & ~_s1EnqQueue_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (s1EnqQueue_dataIn),
    .empty        (_s1EnqQueue_fifo_empty),
    .almost_empty (s1EnqQueue_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (s1EnqQueue_almostFull),
    .full         (_s1EnqQueue_fifo_full),
    .error        (_s1EnqQueue_fifo_error),
    .data_out     (_s1EnqQueue_fifo_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(6),
    .err_mode(2),
    .rst_mode(3),
    .width(32)
  ) s1EnqDataQueue_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(s1EnqDataQueue_enq_ready & s1EnqDataQueue_enq_valid)),
    .pop_req_n    (~(s1EnqDataQueue_deq_ready & ~_s1EnqDataQueue_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (s1EnqDataQueue_enq_bits),
    .empty        (_s1EnqDataQueue_fifo_empty),
    .almost_empty (s1EnqDataQueue_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (s1EnqDataQueue_almostFull),
    .full         (_s1EnqDataQueue_fifo_full),
    .error        (_s1EnqDataQueue_fifo_error),
    .data_out     (s1EnqDataQueue_deq_bits)
  );
  assign vrfReadDataPorts_valid = vrfReadDataPorts_valid_0;
  assign vrfReadDataPorts_bits_vs = vrfReadDataPorts_bits_vs_0;
  assign vrfReadDataPorts_bits_offset = vrfReadDataPorts_bits_offset_0;
  assign vrfReadDataPorts_bits_instructionIndex = vrfReadDataPorts_bits_instructionIndex_0;
  assign maskSelect_valid = maskGroupEndAndRequestNewMask;
  assign maskSelect_bits = nextGroupIndex[5:0];
  assign memReadRequest_valid = memReadRequest_valid_0;
  assign memReadRequest_bits_address = memReadRequest_bits_address_0;
  assign memReadRequest_bits_source = memReadRequest_bits_source_0;
  assign memReadResponse_ready = memReadResponse_ready_0;
  assign memWriteRequest_valid = memWriteRequest_valid_0;
  assign memWriteRequest_bits_data = memWriteRequest_bits_data_0;
  assign memWriteRequest_bits_mask = memWriteRequest_bits_mask_0;
  assign memWriteRequest_bits_source = memWriteRequest_bits_source_0;
  assign memWriteRequest_bits_address = memWriteRequest_bits_address_0;
  assign memWriteRequest_bits_size = memWriteRequest_bits_size_0;
  assign vrfWritePort_valid = vrfWritePort_valid_0;
  assign vrfWritePort_bits_vd = vrfWritePort_bits_vd_0;
  assign vrfWritePort_bits_offset = vrfWritePort_bits_offset_0;
  assign vrfWritePort_bits_mask = vrfWritePort_bits_mask_0;
  assign vrfWritePort_bits_data = vrfWritePort_bits_data_0;
  assign vrfWritePort_bits_last = vrfWritePort_bits_last_0;
  assign vrfWritePort_bits_instructionIndex = vrfWritePort_bits_instructionIndex_0;
  assign status_idle = stateIdle;
  assign status_last = _status_last_output;
  assign status_instructionIndex = lsuRequestReg_instructionIndex;
  assign status_targetLane = 8'h1 << (lsuRequestReg_instructionInformation_isStore ? s0Reg_offsetForLane : loadBaseByteOffset[4:2]);
  assign status_isStore = lsuRequestReg_instructionInformation_isStore;
  assign offsetRelease_0 = _allElementsMasked_T_1;
  assign offsetRelease_1 = _allElementsMasked_T_2;
  assign offsetRelease_2 = _allElementsMasked_T_3;
  assign offsetRelease_3 = _allElementsMasked_T_4;
  assign offsetRelease_4 = _allElementsMasked_T_5;
  assign offsetRelease_5 = _allElementsMasked_T_6;
  assign offsetRelease_6 = _allElementsMasked_T_7;
  assign offsetRelease_7 = _allElementsMasked_T_8;
endmodule

