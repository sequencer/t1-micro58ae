
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
module LSU(
  input          clock,
                 reset,
  output         request_ready,
  input          request_valid,
  input  [2:0]   request_bits_instructionInformation_nf,
  input          request_bits_instructionInformation_mew,
  input  [1:0]   request_bits_instructionInformation_mop,
  input  [4:0]   request_bits_instructionInformation_lumop,
  input  [1:0]   request_bits_instructionInformation_eew,
  input  [4:0]   request_bits_instructionInformation_vs3,
  input          request_bits_instructionInformation_isStore,
                 request_bits_instructionInformation_maskedLoadStore,
  input  [31:0]  request_bits_rs1Data,
                 request_bits_rs2Data,
  input  [2:0]   request_bits_instructionIndex,
  input          v0UpdateVec_0_valid,
  input  [31:0]  v0UpdateVec_0_bits_data,
  input  [3:0]   v0UpdateVec_0_bits_mask,
  input          v0UpdateVec_1_valid,
  input  [31:0]  v0UpdateVec_1_bits_data,
  input  [3:0]   v0UpdateVec_1_bits_mask,
  input          v0UpdateVec_2_valid,
  input  [31:0]  v0UpdateVec_2_bits_data,
  input  [3:0]   v0UpdateVec_2_bits_mask,
  input          v0UpdateVec_3_valid,
  input  [31:0]  v0UpdateVec_3_bits_data,
  input  [3:0]   v0UpdateVec_3_bits_mask,
  input          axi4Port_aw_ready,
  output         axi4Port_aw_valid,
  output [1:0]   axi4Port_aw_bits_id,
  output [31:0]  axi4Port_aw_bits_addr,
  input          axi4Port_w_ready,
  output         axi4Port_w_valid,
  output [127:0] axi4Port_w_bits_data,
  output [15:0]  axi4Port_w_bits_strb,
  input          axi4Port_b_valid,
  input  [1:0]   axi4Port_b_bits_id,
                 axi4Port_b_bits_resp,
  input          axi4Port_ar_ready,
  output         axi4Port_ar_valid,
  output [31:0]  axi4Port_ar_bits_addr,
  output         axi4Port_r_ready,
  input          axi4Port_r_valid,
  input  [1:0]   axi4Port_r_bits_id,
  input  [127:0] axi4Port_r_bits_data,
  input  [1:0]   axi4Port_r_bits_resp,
  input          axi4Port_r_bits_last,
                 simpleAccessPorts_aw_ready,
  output         simpleAccessPorts_aw_valid,
  output [1:0]   simpleAccessPorts_aw_bits_id,
  output [31:0]  simpleAccessPorts_aw_bits_addr,
  output [2:0]   simpleAccessPorts_aw_bits_size,
  input          simpleAccessPorts_w_ready,
  output         simpleAccessPorts_w_valid,
  output [31:0]  simpleAccessPorts_w_bits_data,
  output [3:0]   simpleAccessPorts_w_bits_strb,
  input          simpleAccessPorts_b_valid,
  input  [1:0]   simpleAccessPorts_b_bits_id,
                 simpleAccessPorts_b_bits_resp,
  input          simpleAccessPorts_ar_ready,
  output         simpleAccessPorts_ar_valid,
  output [31:0]  simpleAccessPorts_ar_bits_addr,
  output         simpleAccessPorts_r_ready,
  input          simpleAccessPorts_r_valid,
  input  [1:0]   simpleAccessPorts_r_bits_id,
  input  [31:0]  simpleAccessPorts_r_bits_data,
  input  [1:0]   simpleAccessPorts_r_bits_resp,
  input          simpleAccessPorts_r_bits_last,
                 vrfReadDataPorts_0_ready,
  output         vrfReadDataPorts_0_valid,
  output [4:0]   vrfReadDataPorts_0_bits_vs,
  output [2:0]   vrfReadDataPorts_0_bits_instructionIndex,
  input          vrfReadDataPorts_1_ready,
  output         vrfReadDataPorts_1_valid,
  output [4:0]   vrfReadDataPorts_1_bits_vs,
  output [2:0]   vrfReadDataPorts_1_bits_instructionIndex,
  input          vrfReadDataPorts_2_ready,
  output         vrfReadDataPorts_2_valid,
  output [4:0]   vrfReadDataPorts_2_bits_vs,
  output [2:0]   vrfReadDataPorts_2_bits_instructionIndex,
  input          vrfReadDataPorts_3_ready,
  output         vrfReadDataPorts_3_valid,
  output [4:0]   vrfReadDataPorts_3_bits_vs,
  output [2:0]   vrfReadDataPorts_3_bits_instructionIndex,
  input          vrfReadResults_0_valid,
  input  [31:0]  vrfReadResults_0_bits,
  input          vrfReadResults_1_valid,
  input  [31:0]  vrfReadResults_1_bits,
  input          vrfReadResults_2_valid,
  input  [31:0]  vrfReadResults_2_bits,
  input          vrfReadResults_3_valid,
  input  [31:0]  vrfReadResults_3_bits,
  input          vrfWritePort_0_ready,
  output         vrfWritePort_0_valid,
  output [4:0]   vrfWritePort_0_bits_vd,
  output [3:0]   vrfWritePort_0_bits_mask,
  output [31:0]  vrfWritePort_0_bits_data,
  output         vrfWritePort_0_bits_last,
  output [2:0]   vrfWritePort_0_bits_instructionIndex,
  input          vrfWritePort_1_ready,
  output         vrfWritePort_1_valid,
  output [4:0]   vrfWritePort_1_bits_vd,
  output [3:0]   vrfWritePort_1_bits_mask,
  output [31:0]  vrfWritePort_1_bits_data,
  output         vrfWritePort_1_bits_last,
  output [2:0]   vrfWritePort_1_bits_instructionIndex,
  input          vrfWritePort_2_ready,
  output         vrfWritePort_2_valid,
  output [4:0]   vrfWritePort_2_bits_vd,
  output [3:0]   vrfWritePort_2_bits_mask,
  output [31:0]  vrfWritePort_2_bits_data,
  output         vrfWritePort_2_bits_last,
  output [2:0]   vrfWritePort_2_bits_instructionIndex,
  input          vrfWritePort_3_ready,
  output         vrfWritePort_3_valid,
  output [4:0]   vrfWritePort_3_bits_vd,
  output [3:0]   vrfWritePort_3_bits_mask,
  output [31:0]  vrfWritePort_3_bits_data,
  output         vrfWritePort_3_bits_last,
  output [2:0]   vrfWritePort_3_bits_instructionIndex,
  input          writeRelease_0,
                 writeRelease_1,
                 writeRelease_2,
                 writeRelease_3,
  output [7:0]   dataInWriteQueue_0,
                 dataInWriteQueue_1,
                 dataInWriteQueue_2,
                 dataInWriteQueue_3,
  input  [7:0]   csrInterface_vl,
                 csrInterface_vStart,
  input  [2:0]   csrInterface_vlmul,
  input  [1:0]   csrInterface_vSew,
                 csrInterface_vxrm,
  input          csrInterface_vta,
                 csrInterface_vma,
                 offsetReadResult_0_valid,
  input  [31:0]  offsetReadResult_0_bits,
  input          offsetReadResult_1_valid,
  input  [31:0]  offsetReadResult_1_bits,
  input          offsetReadResult_2_valid,
  input  [31:0]  offsetReadResult_2_bits,
  input          offsetReadResult_3_valid,
  input  [31:0]  offsetReadResult_3_bits,
  output [7:0]   lastReport,
  output [3:0]   tokenIO_offsetGroupRelease
);

  wire             _simpleDataQueue_fifo_empty;
  wire             _simpleDataQueue_fifo_full;
  wire             _simpleDataQueue_fifo_error;
  wire [77:0]      _simpleDataQueue_fifo_data_out;
  wire             _simpleSourceQueue_fifo_empty;
  wire             _simpleSourceQueue_fifo_full;
  wire             _simpleSourceQueue_fifo_error;
  wire             _dataQueue_fifo_empty;
  wire             _dataQueue_fifo_full;
  wire             _dataQueue_fifo_error;
  wire [179:0]     _dataQueue_fifo_data_out;
  wire             _sourceQueue_fifo_empty;
  wire             _sourceQueue_fifo_full;
  wire             _sourceQueue_fifo_error;
  wire             _writeIndexQueue_fifo_3_empty;
  wire             _writeIndexQueue_fifo_3_full;
  wire             _writeIndexQueue_fifo_3_error;
  wire             _writeIndexQueue_fifo_2_empty;
  wire             _writeIndexQueue_fifo_2_full;
  wire             _writeIndexQueue_fifo_2_error;
  wire             _writeIndexQueue_fifo_1_empty;
  wire             _writeIndexQueue_fifo_1_full;
  wire             _writeIndexQueue_fifo_1_error;
  wire             _writeIndexQueue_fifo_empty;
  wire             _writeIndexQueue_fifo_full;
  wire             _writeIndexQueue_fifo_error;
  wire             _otherUnitDataQueueVec_fifo_3_empty;
  wire             _otherUnitDataQueueVec_fifo_3_full;
  wire             _otherUnitDataQueueVec_fifo_3_error;
  wire [31:0]      _otherUnitDataQueueVec_fifo_3_data_out;
  wire             _otherUnitDataQueueVec_fifo_2_empty;
  wire             _otherUnitDataQueueVec_fifo_2_full;
  wire             _otherUnitDataQueueVec_fifo_2_error;
  wire [31:0]      _otherUnitDataQueueVec_fifo_2_data_out;
  wire             _otherUnitDataQueueVec_fifo_1_empty;
  wire             _otherUnitDataQueueVec_fifo_1_full;
  wire             _otherUnitDataQueueVec_fifo_1_error;
  wire [31:0]      _otherUnitDataQueueVec_fifo_1_data_out;
  wire             _otherUnitDataQueueVec_fifo_empty;
  wire             _otherUnitDataQueueVec_fifo_full;
  wire             _otherUnitDataQueueVec_fifo_error;
  wire [31:0]      _otherUnitDataQueueVec_fifo_data_out;
  wire             _otherUnitTargetQueue_fifo_empty;
  wire             _otherUnitTargetQueue_fifo_full;
  wire             _otherUnitTargetQueue_fifo_error;
  wire             _writeQueueVec_fifo_3_empty;
  wire             _writeQueueVec_fifo_3_full;
  wire             _writeQueueVec_fifo_3_error;
  wire [48:0]      _writeQueueVec_fifo_3_data_out;
  wire             _writeQueueVec_fifo_2_empty;
  wire             _writeQueueVec_fifo_2_full;
  wire             _writeQueueVec_fifo_2_error;
  wire [48:0]      _writeQueueVec_fifo_2_data_out;
  wire             _writeQueueVec_fifo_1_empty;
  wire             _writeQueueVec_fifo_1_full;
  wire             _writeQueueVec_fifo_1_error;
  wire [48:0]      _writeQueueVec_fifo_1_data_out;
  wire             _writeQueueVec_fifo_empty;
  wire             _writeQueueVec_fifo_full;
  wire             _writeQueueVec_fifo_error;
  wire [48:0]      _writeQueueVec_fifo_data_out;
  wire             _otherUnit_vrfReadDataPorts_valid;
  wire [4:0]       _otherUnit_vrfReadDataPorts_bits_vs;
  wire [2:0]       _otherUnit_vrfReadDataPorts_bits_instructionIndex;
  wire             _otherUnit_maskSelect_valid;
  wire [2:0]       _otherUnit_maskSelect_bits;
  wire             _otherUnit_memReadRequest_valid;
  wire             _otherUnit_memWriteRequest_valid;
  wire [7:0]       _otherUnit_memWriteRequest_bits_source;
  wire [31:0]      _otherUnit_memWriteRequest_bits_address;
  wire [1:0]       _otherUnit_memWriteRequest_bits_size;
  wire             _otherUnit_vrfWritePort_valid;
  wire [4:0]       _otherUnit_vrfWritePort_bits_vd;
  wire [3:0]       _otherUnit_vrfWritePort_bits_mask;
  wire [31:0]      _otherUnit_vrfWritePort_bits_data;
  wire             _otherUnit_vrfWritePort_bits_last;
  wire [2:0]       _otherUnit_vrfWritePort_bits_instructionIndex;
  wire             _otherUnit_status_idle;
  wire             _otherUnit_status_last;
  wire [2:0]       _otherUnit_status_instructionIndex;
  wire [3:0]       _otherUnit_status_targetLane;
  wire             _otherUnit_status_isStore;
  wire             _otherUnit_offsetRelease_0;
  wire             _otherUnit_offsetRelease_1;
  wire             _otherUnit_offsetRelease_2;
  wire             _otherUnit_offsetRelease_3;
  wire             _storeUnit_maskSelect_valid;
  wire [2:0]       _storeUnit_maskSelect_bits;
  wire             _storeUnit_memRequest_valid;
  wire [3:0]       _storeUnit_memRequest_bits_index;
  wire [31:0]      _storeUnit_memRequest_bits_address;
  wire             _storeUnit_status_idle;
  wire             _storeUnit_status_last;
  wire [2:0]       _storeUnit_status_instructionIndex;
  wire [31:0]      _storeUnit_status_startAddress;
  wire [31:0]      _storeUnit_status_endAddress;
  wire             _storeUnit_vrfReadDataPorts_0_valid;
  wire [4:0]       _storeUnit_vrfReadDataPorts_0_bits_vs;
  wire [2:0]       _storeUnit_vrfReadDataPorts_0_bits_instructionIndex;
  wire             _storeUnit_vrfReadDataPorts_1_valid;
  wire [4:0]       _storeUnit_vrfReadDataPorts_1_bits_vs;
  wire [2:0]       _storeUnit_vrfReadDataPorts_1_bits_instructionIndex;
  wire             _storeUnit_vrfReadDataPorts_2_valid;
  wire [4:0]       _storeUnit_vrfReadDataPorts_2_bits_vs;
  wire [2:0]       _storeUnit_vrfReadDataPorts_2_bits_instructionIndex;
  wire             _storeUnit_vrfReadDataPorts_3_valid;
  wire [4:0]       _storeUnit_vrfReadDataPorts_3_bits_vs;
  wire [2:0]       _storeUnit_vrfReadDataPorts_3_bits_instructionIndex;
  wire             _loadUnit_maskSelect_valid;
  wire [2:0]       _loadUnit_maskSelect_bits;
  wire             _loadUnit_memRequest_valid;
  wire             _loadUnit_status_idle;
  wire             _loadUnit_status_last;
  wire [2:0]       _loadUnit_status_instructionIndex;
  wire [31:0]      _loadUnit_status_startAddress;
  wire [31:0]      _loadUnit_status_endAddress;
  wire             _loadUnit_vrfWritePort_0_valid;
  wire [4:0]       _loadUnit_vrfWritePort_0_bits_vd;
  wire [3:0]       _loadUnit_vrfWritePort_0_bits_mask;
  wire [31:0]      _loadUnit_vrfWritePort_0_bits_data;
  wire [2:0]       _loadUnit_vrfWritePort_0_bits_instructionIndex;
  wire             _loadUnit_vrfWritePort_1_valid;
  wire [4:0]       _loadUnit_vrfWritePort_1_bits_vd;
  wire [3:0]       _loadUnit_vrfWritePort_1_bits_mask;
  wire [31:0]      _loadUnit_vrfWritePort_1_bits_data;
  wire [2:0]       _loadUnit_vrfWritePort_1_bits_instructionIndex;
  wire             _loadUnit_vrfWritePort_2_valid;
  wire [4:0]       _loadUnit_vrfWritePort_2_bits_vd;
  wire [3:0]       _loadUnit_vrfWritePort_2_bits_mask;
  wire [31:0]      _loadUnit_vrfWritePort_2_bits_data;
  wire [2:0]       _loadUnit_vrfWritePort_2_bits_instructionIndex;
  wire             _loadUnit_vrfWritePort_3_valid;
  wire [4:0]       _loadUnit_vrfWritePort_3_bits_vd;
  wire [3:0]       _loadUnit_vrfWritePort_3_bits_mask;
  wire [31:0]      _loadUnit_vrfWritePort_3_bits_data;
  wire [2:0]       _loadUnit_vrfWritePort_3_bits_instructionIndex;
  wire             simpleDataQueue_almostFull;
  wire             simpleDataQueue_almostEmpty;
  wire             simpleSourceQueue_almostFull;
  wire             simpleSourceQueue_almostEmpty;
  wire             dataQueue_almostFull;
  wire             dataQueue_almostEmpty;
  wire             sourceQueue_almostFull;
  wire             sourceQueue_almostEmpty;
  wire             writeIndexQueue_3_almostFull;
  wire             writeIndexQueue_3_almostEmpty;
  wire             writeIndexQueue_2_almostFull;
  wire             writeIndexQueue_2_almostEmpty;
  wire             writeIndexQueue_1_almostFull;
  wire             writeIndexQueue_1_almostEmpty;
  wire             writeIndexQueue_almostFull;
  wire             writeIndexQueue_almostEmpty;
  wire             otherUnitDataQueueVec_3_almostFull;
  wire             otherUnitDataQueueVec_3_almostEmpty;
  wire             otherUnitDataQueueVec_2_almostFull;
  wire             otherUnitDataQueueVec_2_almostEmpty;
  wire             otherUnitDataQueueVec_1_almostFull;
  wire             otherUnitDataQueueVec_1_almostEmpty;
  wire             otherUnitDataQueueVec_0_almostFull;
  wire             otherUnitDataQueueVec_0_almostEmpty;
  wire             otherUnitTargetQueue_almostFull;
  wire             otherUnitTargetQueue_almostEmpty;
  wire             writeQueueVec_3_almostFull;
  wire             writeQueueVec_3_almostEmpty;
  wire             writeQueueVec_2_almostFull;
  wire             writeQueueVec_2_almostEmpty;
  wire             writeQueueVec_1_almostFull;
  wire             writeQueueVec_1_almostEmpty;
  wire             writeQueueVec_0_almostFull;
  wire             writeQueueVec_0_almostEmpty;
  wire [6:0]       simpleSourceQueue_enq_bits;
  wire [31:0]      simpleAccessPorts_ar_bits_addr_0;
  wire [3:0]       sourceQueue_enq_bits;
  wire [31:0]      axi4Port_ar_bits_addr_0;
  wire [4:0]       writeQueueVec_3_enq_bits_data_vd;
  wire [4:0]       writeQueueVec_2_enq_bits_data_vd;
  wire [4:0]       writeQueueVec_1_enq_bits_data_vd;
  wire [4:0]       writeQueueVec_0_enq_bits_data_vd;
  wire             request_valid_0 = request_valid;
  wire [2:0]       request_bits_instructionInformation_nf_0 = request_bits_instructionInformation_nf;
  wire             request_bits_instructionInformation_mew_0 = request_bits_instructionInformation_mew;
  wire [1:0]       request_bits_instructionInformation_mop_0 = request_bits_instructionInformation_mop;
  wire [4:0]       request_bits_instructionInformation_lumop_0 = request_bits_instructionInformation_lumop;
  wire [1:0]       request_bits_instructionInformation_eew_0 = request_bits_instructionInformation_eew;
  wire [4:0]       request_bits_instructionInformation_vs3_0 = request_bits_instructionInformation_vs3;
  wire             request_bits_instructionInformation_isStore_0 = request_bits_instructionInformation_isStore;
  wire             request_bits_instructionInformation_maskedLoadStore_0 = request_bits_instructionInformation_maskedLoadStore;
  wire [31:0]      request_bits_rs1Data_0 = request_bits_rs1Data;
  wire [31:0]      request_bits_rs2Data_0 = request_bits_rs2Data;
  wire [2:0]       request_bits_instructionIndex_0 = request_bits_instructionIndex;
  wire             axi4Port_aw_ready_0 = axi4Port_aw_ready;
  wire             axi4Port_w_ready_0 = axi4Port_w_ready;
  wire             axi4Port_b_valid_0 = axi4Port_b_valid;
  wire [1:0]       axi4Port_b_bits_id_0 = axi4Port_b_bits_id;
  wire [1:0]       axi4Port_b_bits_resp_0 = axi4Port_b_bits_resp;
  wire             axi4Port_ar_ready_0 = axi4Port_ar_ready;
  wire             axi4Port_r_valid_0 = axi4Port_r_valid;
  wire [1:0]       axi4Port_r_bits_id_0 = axi4Port_r_bits_id;
  wire [127:0]     axi4Port_r_bits_data_0 = axi4Port_r_bits_data;
  wire [1:0]       axi4Port_r_bits_resp_0 = axi4Port_r_bits_resp;
  wire             axi4Port_r_bits_last_0 = axi4Port_r_bits_last;
  wire             simpleAccessPorts_aw_ready_0 = simpleAccessPorts_aw_ready;
  wire             simpleAccessPorts_w_ready_0 = simpleAccessPorts_w_ready;
  wire             simpleAccessPorts_b_valid_0 = simpleAccessPorts_b_valid;
  wire [1:0]       simpleAccessPorts_b_bits_id_0 = simpleAccessPorts_b_bits_id;
  wire [1:0]       simpleAccessPorts_b_bits_resp_0 = simpleAccessPorts_b_bits_resp;
  wire             simpleAccessPorts_ar_ready_0 = simpleAccessPorts_ar_ready;
  wire             simpleAccessPorts_r_valid_0 = simpleAccessPorts_r_valid;
  wire [1:0]       simpleAccessPorts_r_bits_id_0 = simpleAccessPorts_r_bits_id;
  wire [31:0]      simpleAccessPorts_r_bits_data_0 = simpleAccessPorts_r_bits_data;
  wire [1:0]       simpleAccessPorts_r_bits_resp_0 = simpleAccessPorts_r_bits_resp;
  wire             simpleAccessPorts_r_bits_last_0 = simpleAccessPorts_r_bits_last;
  wire             vrfReadDataPorts_0_ready_0 = vrfReadDataPorts_0_ready;
  wire             vrfReadDataPorts_1_ready_0 = vrfReadDataPorts_1_ready;
  wire             vrfReadDataPorts_2_ready_0 = vrfReadDataPorts_2_ready;
  wire             vrfReadDataPorts_3_ready_0 = vrfReadDataPorts_3_ready;
  wire             vrfWritePort_0_ready_0 = vrfWritePort_0_ready;
  wire             vrfWritePort_1_ready_0 = vrfWritePort_1_ready;
  wire             vrfWritePort_2_ready_0 = vrfWritePort_2_ready;
  wire             vrfWritePort_3_ready_0 = vrfWritePort_3_ready;
  wire [31:0]      otherUnitDataQueueVec_0_enq_bits = vrfReadResults_0_bits;
  wire [31:0]      otherUnitDataQueueVec_1_enq_bits = vrfReadResults_1_bits;
  wire [31:0]      otherUnitDataQueueVec_2_enq_bits = vrfReadResults_2_bits;
  wire [31:0]      otherUnitDataQueueVec_3_enq_bits = vrfReadResults_3_bits;
  wire             writeIndexQueue_deq_ready = writeRelease_0;
  wire             writeIndexQueue_1_deq_ready = writeRelease_1;
  wire             writeIndexQueue_2_deq_ready = writeRelease_2;
  wire             writeIndexQueue_3_deq_ready = writeRelease_3;
  wire [3:0]       writeQueueVec_3_enq_bits_targetLane = 4'h8;
  wire [3:0]       writeQueueVec_2_enq_bits_targetLane = 4'h4;
  wire [3:0]       writeQueueVec_1_enq_bits_targetLane = 4'h2;
  wire [3:0]       writeQueueVec_0_enq_bits_targetLane = 4'h1;
  wire [3:0]       axi4Port_aw_bits_cache = 4'h0;
  wire [3:0]       axi4Port_aw_bits_qos = 4'h0;
  wire [3:0]       axi4Port_aw_bits_region = 4'h0;
  wire [3:0]       axi4Port_ar_bits_cache = 4'h0;
  wire [3:0]       axi4Port_ar_bits_qos = 4'h0;
  wire [3:0]       axi4Port_ar_bits_region = 4'h0;
  wire [3:0]       simpleAccessPorts_aw_bits_cache = 4'h0;
  wire [3:0]       simpleAccessPorts_aw_bits_qos = 4'h0;
  wire [3:0]       simpleAccessPorts_aw_bits_region = 4'h0;
  wire [3:0]       simpleAccessPorts_ar_bits_cache = 4'h0;
  wire [3:0]       simpleAccessPorts_ar_bits_qos = 4'h0;
  wire [3:0]       simpleAccessPorts_ar_bits_region = 4'h0;
  wire [1:0]       vrfReadDataPorts_0_bits_readSource = 2'h2;
  wire [1:0]       vrfReadDataPorts_1_bits_readSource = 2'h2;
  wire [1:0]       vrfReadDataPorts_2_bits_readSource = 2'h2;
  wire [1:0]       vrfReadDataPorts_3_bits_readSource = 2'h2;
  wire [1:0]       axi4Port_ar_bits_id = 2'h0;
  wire [1:0]       simpleAccessPorts_ar_bits_id = 2'h0;
  wire [1:0]       axi4Port_aw_bits_burst = 2'h1;
  wire [1:0]       axi4Port_ar_bits_burst = 2'h1;
  wire [1:0]       simpleAccessPorts_aw_bits_burst = 2'h1;
  wire [1:0]       simpleAccessPorts_ar_bits_burst = 2'h1;
  wire [7:0]       axi4Port_aw_bits_len = 8'h0;
  wire [7:0]       axi4Port_ar_bits_len = 8'h0;
  wire [7:0]       simpleAccessPorts_aw_bits_len = 8'h0;
  wire [7:0]       simpleAccessPorts_ar_bits_len = 8'h0;
  wire [2:0]       axi4Port_aw_bits_size = 3'h4;
  wire [2:0]       axi4Port_ar_bits_size = 3'h4;
  wire             axi4Port_aw_bits_lock = 1'h0;
  wire             axi4Port_ar_bits_lock = 1'h0;
  wire             simpleAccessPorts_aw_bits_lock = 1'h0;
  wire             simpleAccessPorts_ar_bits_lock = 1'h0;
  wire [2:0]       axi4Port_aw_bits_prot = 3'h0;
  wire [2:0]       axi4Port_ar_bits_prot = 3'h0;
  wire [2:0]       simpleAccessPorts_aw_bits_prot = 3'h0;
  wire [2:0]       simpleAccessPorts_ar_bits_prot = 3'h0;
  wire             axi4Port_w_bits_last = 1'h1;
  wire             axi4Port_b_ready = 1'h1;
  wire             simpleAccessPorts_w_bits_last = 1'h1;
  wire             simpleAccessPorts_b_ready = 1'h1;
  wire [2:0]       simpleAccessPorts_ar_bits_size = 3'h2;
  wire             dataQueue_deq_ready = axi4Port_w_ready_0;
  wire             dataQueue_deq_valid;
  wire [127:0]     dataQueue_deq_bits_data;
  wire [15:0]      dataQueue_deq_bits_mask;
  wire             simpleDataQueue_deq_ready = simpleAccessPorts_w_ready_0;
  wire             simpleDataQueue_deq_valid;
  wire [31:0]      simpleDataQueue_deq_bits_data;
  wire [3:0]       simpleDataQueue_deq_bits_mask;
  wire             writeQueueVec_0_deq_ready = vrfWritePort_0_ready_0;
  wire             writeQueueVec_0_deq_valid;
  wire [4:0]       writeQueueVec_0_deq_bits_data_vd;
  wire [3:0]       writeQueueVec_0_deq_bits_data_mask;
  wire [31:0]      writeQueueVec_0_deq_bits_data_data;
  wire             writeQueueVec_0_deq_bits_data_last;
  wire [2:0]       writeQueueVec_0_deq_bits_data_instructionIndex;
  wire             writeQueueVec_1_deq_ready = vrfWritePort_1_ready_0;
  wire             writeQueueVec_1_deq_valid;
  wire [4:0]       writeQueueVec_1_deq_bits_data_vd;
  wire [3:0]       writeQueueVec_1_deq_bits_data_mask;
  wire [31:0]      writeQueueVec_1_deq_bits_data_data;
  wire             writeQueueVec_1_deq_bits_data_last;
  wire [2:0]       writeQueueVec_1_deq_bits_data_instructionIndex;
  wire             writeQueueVec_2_deq_ready = vrfWritePort_2_ready_0;
  wire             writeQueueVec_2_deq_valid;
  wire [4:0]       writeQueueVec_2_deq_bits_data_vd;
  wire [3:0]       writeQueueVec_2_deq_bits_data_mask;
  wire [31:0]      writeQueueVec_2_deq_bits_data_data;
  wire             writeQueueVec_2_deq_bits_data_last;
  wire [2:0]       writeQueueVec_2_deq_bits_data_instructionIndex;
  wire             writeQueueVec_3_deq_ready = vrfWritePort_3_ready_0;
  wire             writeQueueVec_3_deq_valid;
  wire [4:0]       writeQueueVec_3_deq_bits_data_vd;
  wire [3:0]       writeQueueVec_3_deq_bits_data_mask;
  wire [31:0]      writeQueueVec_3_deq_bits_data_data;
  wire             writeQueueVec_3_deq_bits_data_last;
  wire [2:0]       writeQueueVec_3_deq_bits_data_instructionIndex;
  reg  [31:0]      v0_0;
  reg  [31:0]      v0_1;
  reg  [31:0]      v0_2;
  reg  [31:0]      v0_3;
  wire [15:0]      maskExt_lo = {{8{v0UpdateVec_0_bits_mask[1]}}, {8{v0UpdateVec_0_bits_mask[0]}}};
  wire [15:0]      maskExt_hi = {{8{v0UpdateVec_0_bits_mask[3]}}, {8{v0UpdateVec_0_bits_mask[2]}}};
  wire [31:0]      maskExt = {maskExt_hi, maskExt_lo};
  wire [15:0]      maskExt_lo_1 = {{8{v0UpdateVec_1_bits_mask[1]}}, {8{v0UpdateVec_1_bits_mask[0]}}};
  wire [15:0]      maskExt_hi_1 = {{8{v0UpdateVec_1_bits_mask[3]}}, {8{v0UpdateVec_1_bits_mask[2]}}};
  wire [31:0]      maskExt_1 = {maskExt_hi_1, maskExt_lo_1};
  wire [15:0]      maskExt_lo_2 = {{8{v0UpdateVec_2_bits_mask[1]}}, {8{v0UpdateVec_2_bits_mask[0]}}};
  wire [15:0]      maskExt_hi_2 = {{8{v0UpdateVec_2_bits_mask[3]}}, {8{v0UpdateVec_2_bits_mask[2]}}};
  wire [31:0]      maskExt_2 = {maskExt_hi_2, maskExt_lo_2};
  wire [15:0]      maskExt_lo_3 = {{8{v0UpdateVec_3_bits_mask[1]}}, {8{v0UpdateVec_3_bits_mask[0]}}};
  wire [15:0]      maskExt_hi_3 = {{8{v0UpdateVec_3_bits_mask[3]}}, {8{v0UpdateVec_3_bits_mask[2]}}};
  wire [31:0]      maskExt_3 = {maskExt_hi_3, maskExt_lo_3};
  wire             alwaysMerge = {request_bits_instructionInformation_mop_0, request_bits_instructionInformation_lumop_0[2:0], request_bits_instructionInformation_lumop_0[4]} == 6'h0;
  wire             useLoadUnit = alwaysMerge & ~request_bits_instructionInformation_isStore_0;
  wire             useStoreUnit = alwaysMerge & request_bits_instructionInformation_isStore_0;
  wire             useOtherUnit = ~alwaysMerge;
  wire             addressCheck = _otherUnit_status_idle & (~useOtherUnit | _loadUnit_status_idle & _storeUnit_status_idle);
  wire             unitReady = useLoadUnit & _loadUnit_status_idle | useStoreUnit & _storeUnit_status_idle | useOtherUnit & _otherUnit_status_idle;
  wire             request_ready_0 = unitReady & addressCheck;
  wire             requestFire = request_ready_0 & request_valid_0;
  wire             reqEnq_0 = useLoadUnit & requestFire;
  wire             reqEnq_1 = useStoreUnit & requestFire;
  wire             reqEnq_2 = useOtherUnit & requestFire;
  wire [2:0]       maskSelect = _loadUnit_maskSelect_valid ? _loadUnit_maskSelect_bits : 3'h0;
  wire [63:0]      _GEN = {v0_1, v0_0};
  wire [63:0]      loadUnit_maskInput_lo;
  assign loadUnit_maskInput_lo = _GEN;
  wire [63:0]      storeUnit_maskInput_lo;
  assign storeUnit_maskInput_lo = _GEN;
  wire [63:0]      otherUnit_maskInput_lo;
  assign otherUnit_maskInput_lo = _GEN;
  wire [63:0]      _GEN_0 = {v0_3, v0_2};
  wire [63:0]      loadUnit_maskInput_hi;
  assign loadUnit_maskInput_hi = _GEN_0;
  wire [63:0]      storeUnit_maskInput_hi;
  assign storeUnit_maskInput_hi = _GEN_0;
  wire [63:0]      otherUnit_maskInput_hi;
  assign otherUnit_maskInput_hi = _GEN_0;
  wire [7:0][15:0] _GEN_1 =
    {{loadUnit_maskInput_hi[63:48]},
     {loadUnit_maskInput_hi[47:32]},
     {loadUnit_maskInput_hi[31:16]},
     {loadUnit_maskInput_hi[15:0]},
     {loadUnit_maskInput_lo[63:48]},
     {loadUnit_maskInput_lo[47:32]},
     {loadUnit_maskInput_lo[31:16]},
     {loadUnit_maskInput_lo[15:0]}};
  wire [2:0]       maskSelect_1 = _storeUnit_maskSelect_valid ? _storeUnit_maskSelect_bits : 3'h0;
  wire [7:0][15:0] _GEN_2 =
    {{storeUnit_maskInput_hi[63:48]},
     {storeUnit_maskInput_hi[47:32]},
     {storeUnit_maskInput_hi[31:16]},
     {storeUnit_maskInput_hi[15:0]},
     {storeUnit_maskInput_lo[63:48]},
     {storeUnit_maskInput_lo[47:32]},
     {storeUnit_maskInput_lo[31:16]},
     {storeUnit_maskInput_lo[15:0]}};
  wire [2:0]       maskSelect_2 = _otherUnit_maskSelect_valid ? _otherUnit_maskSelect_bits : 3'h0;
  wire [7:0][15:0] _GEN_3 =
    {{otherUnit_maskInput_hi[63:48]},
     {otherUnit_maskInput_hi[47:32]},
     {otherUnit_maskInput_hi[31:16]},
     {otherUnit_maskInput_hi[15:0]},
     {otherUnit_maskInput_lo[63:48]},
     {otherUnit_maskInput_lo[47:32]},
     {otherUnit_maskInput_lo[31:16]},
     {otherUnit_maskInput_lo[15:0]}};
  wire [4:0]       writeQueueVec_dataIn_hi_hi = writeQueueVec_0_enq_bits_data_vd;
  wire             vrfWritePort_0_valid_0 = writeQueueVec_0_deq_valid;
  wire [4:0]       vrfWritePort_0_bits_vd_0 = writeQueueVec_0_deq_bits_data_vd;
  wire [3:0]       vrfWritePort_0_bits_mask_0 = writeQueueVec_0_deq_bits_data_mask;
  wire [31:0]      vrfWritePort_0_bits_data_0 = writeQueueVec_0_deq_bits_data_data;
  wire             vrfWritePort_0_bits_last_0 = writeQueueVec_0_deq_bits_data_last;
  wire [2:0]       vrfWritePort_0_bits_instructionIndex_0 = writeQueueVec_0_deq_bits_data_instructionIndex;
  wire [2:0]       writeIndexQueue_enq_bits = writeQueueVec_0_deq_bits_data_instructionIndex;
  wire [31:0]      writeQueueVec_0_enq_bits_data_data;
  wire             writeQueueVec_0_enq_bits_data_last;
  wire [32:0]      writeQueueVec_dataIn_lo_hi = {writeQueueVec_0_enq_bits_data_data, writeQueueVec_0_enq_bits_data_last};
  wire [2:0]       writeQueueVec_0_enq_bits_data_instructionIndex;
  wire [35:0]      writeQueueVec_dataIn_lo = {writeQueueVec_dataIn_lo_hi, writeQueueVec_0_enq_bits_data_instructionIndex};
  wire [3:0]       writeQueueVec_0_enq_bits_data_mask;
  wire [8:0]       writeQueueVec_dataIn_hi = {writeQueueVec_dataIn_hi_hi, writeQueueVec_0_enq_bits_data_mask};
  wire [48:0]      writeQueueVec_dataIn = {writeQueueVec_dataIn_hi, writeQueueVec_dataIn_lo, 4'h1};
  wire [3:0]       writeQueueVec_dataOut_targetLane = _writeQueueVec_fifo_data_out[3:0];
  wire [2:0]       writeQueueVec_dataOut_data_instructionIndex = _writeQueueVec_fifo_data_out[6:4];
  wire             writeQueueVec_dataOut_data_last = _writeQueueVec_fifo_data_out[7];
  wire [31:0]      writeQueueVec_dataOut_data_data = _writeQueueVec_fifo_data_out[39:8];
  wire [3:0]       writeQueueVec_dataOut_data_mask = _writeQueueVec_fifo_data_out[43:40];
  wire [4:0]       writeQueueVec_dataOut_data_vd = _writeQueueVec_fifo_data_out[48:44];
  wire             writeQueueVec_0_enq_ready = ~_writeQueueVec_fifo_full;
  wire             writeQueueVec_0_enq_valid;
  wire             _probeWire_slots_0_writeValid_T = writeQueueVec_0_enq_ready & writeQueueVec_0_enq_valid;
  assign writeQueueVec_0_deq_valid = ~_writeQueueVec_fifo_empty | writeQueueVec_0_enq_valid;
  assign writeQueueVec_0_deq_bits_data_vd = _writeQueueVec_fifo_empty ? writeQueueVec_0_enq_bits_data_vd : writeQueueVec_dataOut_data_vd;
  assign writeQueueVec_0_deq_bits_data_mask = _writeQueueVec_fifo_empty ? writeQueueVec_0_enq_bits_data_mask : writeQueueVec_dataOut_data_mask;
  assign writeQueueVec_0_deq_bits_data_data = _writeQueueVec_fifo_empty ? writeQueueVec_0_enq_bits_data_data : writeQueueVec_dataOut_data_data;
  assign writeQueueVec_0_deq_bits_data_last = _writeQueueVec_fifo_empty ? writeQueueVec_0_enq_bits_data_last : writeQueueVec_dataOut_data_last;
  assign writeQueueVec_0_deq_bits_data_instructionIndex = _writeQueueVec_fifo_empty ? writeQueueVec_0_enq_bits_data_instructionIndex : writeQueueVec_dataOut_data_instructionIndex;
  wire [3:0]       writeQueueVec_0_deq_bits_targetLane = _writeQueueVec_fifo_empty ? 4'h1 : writeQueueVec_dataOut_targetLane;
  wire [4:0]       writeQueueVec_dataIn_hi_hi_1 = writeQueueVec_1_enq_bits_data_vd;
  wire             vrfWritePort_1_valid_0 = writeQueueVec_1_deq_valid;
  wire [4:0]       vrfWritePort_1_bits_vd_0 = writeQueueVec_1_deq_bits_data_vd;
  wire [3:0]       vrfWritePort_1_bits_mask_0 = writeQueueVec_1_deq_bits_data_mask;
  wire [31:0]      vrfWritePort_1_bits_data_0 = writeQueueVec_1_deq_bits_data_data;
  wire             vrfWritePort_1_bits_last_0 = writeQueueVec_1_deq_bits_data_last;
  wire [2:0]       vrfWritePort_1_bits_instructionIndex_0 = writeQueueVec_1_deq_bits_data_instructionIndex;
  wire [2:0]       writeIndexQueue_1_enq_bits = writeQueueVec_1_deq_bits_data_instructionIndex;
  wire [31:0]      writeQueueVec_1_enq_bits_data_data;
  wire             writeQueueVec_1_enq_bits_data_last;
  wire [32:0]      writeQueueVec_dataIn_lo_hi_1 = {writeQueueVec_1_enq_bits_data_data, writeQueueVec_1_enq_bits_data_last};
  wire [2:0]       writeQueueVec_1_enq_bits_data_instructionIndex;
  wire [35:0]      writeQueueVec_dataIn_lo_1 = {writeQueueVec_dataIn_lo_hi_1, writeQueueVec_1_enq_bits_data_instructionIndex};
  wire [3:0]       writeQueueVec_1_enq_bits_data_mask;
  wire [8:0]       writeQueueVec_dataIn_hi_1 = {writeQueueVec_dataIn_hi_hi_1, writeQueueVec_1_enq_bits_data_mask};
  wire [48:0]      writeQueueVec_dataIn_1 = {writeQueueVec_dataIn_hi_1, writeQueueVec_dataIn_lo_1, 4'h2};
  wire [3:0]       writeQueueVec_dataOut_1_targetLane = _writeQueueVec_fifo_1_data_out[3:0];
  wire [2:0]       writeQueueVec_dataOut_1_data_instructionIndex = _writeQueueVec_fifo_1_data_out[6:4];
  wire             writeQueueVec_dataOut_1_data_last = _writeQueueVec_fifo_1_data_out[7];
  wire [31:0]      writeQueueVec_dataOut_1_data_data = _writeQueueVec_fifo_1_data_out[39:8];
  wire [3:0]       writeQueueVec_dataOut_1_data_mask = _writeQueueVec_fifo_1_data_out[43:40];
  wire [4:0]       writeQueueVec_dataOut_1_data_vd = _writeQueueVec_fifo_1_data_out[48:44];
  wire             writeQueueVec_1_enq_ready = ~_writeQueueVec_fifo_1_full;
  wire             writeQueueVec_1_enq_valid;
  wire             _probeWire_slots_1_writeValid_T = writeQueueVec_1_enq_ready & writeQueueVec_1_enq_valid;
  assign writeQueueVec_1_deq_valid = ~_writeQueueVec_fifo_1_empty | writeQueueVec_1_enq_valid;
  assign writeQueueVec_1_deq_bits_data_vd = _writeQueueVec_fifo_1_empty ? writeQueueVec_1_enq_bits_data_vd : writeQueueVec_dataOut_1_data_vd;
  assign writeQueueVec_1_deq_bits_data_mask = _writeQueueVec_fifo_1_empty ? writeQueueVec_1_enq_bits_data_mask : writeQueueVec_dataOut_1_data_mask;
  assign writeQueueVec_1_deq_bits_data_data = _writeQueueVec_fifo_1_empty ? writeQueueVec_1_enq_bits_data_data : writeQueueVec_dataOut_1_data_data;
  assign writeQueueVec_1_deq_bits_data_last = _writeQueueVec_fifo_1_empty ? writeQueueVec_1_enq_bits_data_last : writeQueueVec_dataOut_1_data_last;
  assign writeQueueVec_1_deq_bits_data_instructionIndex = _writeQueueVec_fifo_1_empty ? writeQueueVec_1_enq_bits_data_instructionIndex : writeQueueVec_dataOut_1_data_instructionIndex;
  wire [3:0]       writeQueueVec_1_deq_bits_targetLane = _writeQueueVec_fifo_1_empty ? 4'h2 : writeQueueVec_dataOut_1_targetLane;
  wire [4:0]       writeQueueVec_dataIn_hi_hi_2 = writeQueueVec_2_enq_bits_data_vd;
  wire             vrfWritePort_2_valid_0 = writeQueueVec_2_deq_valid;
  wire [4:0]       vrfWritePort_2_bits_vd_0 = writeQueueVec_2_deq_bits_data_vd;
  wire [3:0]       vrfWritePort_2_bits_mask_0 = writeQueueVec_2_deq_bits_data_mask;
  wire [31:0]      vrfWritePort_2_bits_data_0 = writeQueueVec_2_deq_bits_data_data;
  wire             vrfWritePort_2_bits_last_0 = writeQueueVec_2_deq_bits_data_last;
  wire [2:0]       vrfWritePort_2_bits_instructionIndex_0 = writeQueueVec_2_deq_bits_data_instructionIndex;
  wire [2:0]       writeIndexQueue_2_enq_bits = writeQueueVec_2_deq_bits_data_instructionIndex;
  wire [31:0]      writeQueueVec_2_enq_bits_data_data;
  wire             writeQueueVec_2_enq_bits_data_last;
  wire [32:0]      writeQueueVec_dataIn_lo_hi_2 = {writeQueueVec_2_enq_bits_data_data, writeQueueVec_2_enq_bits_data_last};
  wire [2:0]       writeQueueVec_2_enq_bits_data_instructionIndex;
  wire [35:0]      writeQueueVec_dataIn_lo_2 = {writeQueueVec_dataIn_lo_hi_2, writeQueueVec_2_enq_bits_data_instructionIndex};
  wire [3:0]       writeQueueVec_2_enq_bits_data_mask;
  wire [8:0]       writeQueueVec_dataIn_hi_2 = {writeQueueVec_dataIn_hi_hi_2, writeQueueVec_2_enq_bits_data_mask};
  wire [48:0]      writeQueueVec_dataIn_2 = {writeQueueVec_dataIn_hi_2, writeQueueVec_dataIn_lo_2, 4'h4};
  wire [3:0]       writeQueueVec_dataOut_2_targetLane = _writeQueueVec_fifo_2_data_out[3:0];
  wire [2:0]       writeQueueVec_dataOut_2_data_instructionIndex = _writeQueueVec_fifo_2_data_out[6:4];
  wire             writeQueueVec_dataOut_2_data_last = _writeQueueVec_fifo_2_data_out[7];
  wire [31:0]      writeQueueVec_dataOut_2_data_data = _writeQueueVec_fifo_2_data_out[39:8];
  wire [3:0]       writeQueueVec_dataOut_2_data_mask = _writeQueueVec_fifo_2_data_out[43:40];
  wire [4:0]       writeQueueVec_dataOut_2_data_vd = _writeQueueVec_fifo_2_data_out[48:44];
  wire             writeQueueVec_2_enq_ready = ~_writeQueueVec_fifo_2_full;
  wire             writeQueueVec_2_enq_valid;
  wire             _probeWire_slots_2_writeValid_T = writeQueueVec_2_enq_ready & writeQueueVec_2_enq_valid;
  assign writeQueueVec_2_deq_valid = ~_writeQueueVec_fifo_2_empty | writeQueueVec_2_enq_valid;
  assign writeQueueVec_2_deq_bits_data_vd = _writeQueueVec_fifo_2_empty ? writeQueueVec_2_enq_bits_data_vd : writeQueueVec_dataOut_2_data_vd;
  assign writeQueueVec_2_deq_bits_data_mask = _writeQueueVec_fifo_2_empty ? writeQueueVec_2_enq_bits_data_mask : writeQueueVec_dataOut_2_data_mask;
  assign writeQueueVec_2_deq_bits_data_data = _writeQueueVec_fifo_2_empty ? writeQueueVec_2_enq_bits_data_data : writeQueueVec_dataOut_2_data_data;
  assign writeQueueVec_2_deq_bits_data_last = _writeQueueVec_fifo_2_empty ? writeQueueVec_2_enq_bits_data_last : writeQueueVec_dataOut_2_data_last;
  assign writeQueueVec_2_deq_bits_data_instructionIndex = _writeQueueVec_fifo_2_empty ? writeQueueVec_2_enq_bits_data_instructionIndex : writeQueueVec_dataOut_2_data_instructionIndex;
  wire [3:0]       writeQueueVec_2_deq_bits_targetLane = _writeQueueVec_fifo_2_empty ? 4'h4 : writeQueueVec_dataOut_2_targetLane;
  wire [4:0]       writeQueueVec_dataIn_hi_hi_3 = writeQueueVec_3_enq_bits_data_vd;
  wire             vrfWritePort_3_valid_0 = writeQueueVec_3_deq_valid;
  wire [4:0]       vrfWritePort_3_bits_vd_0 = writeQueueVec_3_deq_bits_data_vd;
  wire [3:0]       vrfWritePort_3_bits_mask_0 = writeQueueVec_3_deq_bits_data_mask;
  wire [31:0]      vrfWritePort_3_bits_data_0 = writeQueueVec_3_deq_bits_data_data;
  wire             vrfWritePort_3_bits_last_0 = writeQueueVec_3_deq_bits_data_last;
  wire [2:0]       vrfWritePort_3_bits_instructionIndex_0 = writeQueueVec_3_deq_bits_data_instructionIndex;
  wire [2:0]       writeIndexQueue_3_enq_bits = writeQueueVec_3_deq_bits_data_instructionIndex;
  wire [31:0]      writeQueueVec_3_enq_bits_data_data;
  wire             writeQueueVec_3_enq_bits_data_last;
  wire [32:0]      writeQueueVec_dataIn_lo_hi_3 = {writeQueueVec_3_enq_bits_data_data, writeQueueVec_3_enq_bits_data_last};
  wire [2:0]       writeQueueVec_3_enq_bits_data_instructionIndex;
  wire [35:0]      writeQueueVec_dataIn_lo_3 = {writeQueueVec_dataIn_lo_hi_3, writeQueueVec_3_enq_bits_data_instructionIndex};
  wire [3:0]       writeQueueVec_3_enq_bits_data_mask;
  wire [8:0]       writeQueueVec_dataIn_hi_3 = {writeQueueVec_dataIn_hi_hi_3, writeQueueVec_3_enq_bits_data_mask};
  wire [48:0]      writeQueueVec_dataIn_3 = {writeQueueVec_dataIn_hi_3, writeQueueVec_dataIn_lo_3, 4'h8};
  wire [3:0]       writeQueueVec_dataOut_3_targetLane = _writeQueueVec_fifo_3_data_out[3:0];
  wire [2:0]       writeQueueVec_dataOut_3_data_instructionIndex = _writeQueueVec_fifo_3_data_out[6:4];
  wire             writeQueueVec_dataOut_3_data_last = _writeQueueVec_fifo_3_data_out[7];
  wire [31:0]      writeQueueVec_dataOut_3_data_data = _writeQueueVec_fifo_3_data_out[39:8];
  wire [3:0]       writeQueueVec_dataOut_3_data_mask = _writeQueueVec_fifo_3_data_out[43:40];
  wire [4:0]       writeQueueVec_dataOut_3_data_vd = _writeQueueVec_fifo_3_data_out[48:44];
  wire             writeQueueVec_3_enq_ready = ~_writeQueueVec_fifo_3_full;
  wire             writeQueueVec_3_enq_valid;
  wire             _probeWire_slots_3_writeValid_T = writeQueueVec_3_enq_ready & writeQueueVec_3_enq_valid;
  assign writeQueueVec_3_deq_valid = ~_writeQueueVec_fifo_3_empty | writeQueueVec_3_enq_valid;
  assign writeQueueVec_3_deq_bits_data_vd = _writeQueueVec_fifo_3_empty ? writeQueueVec_3_enq_bits_data_vd : writeQueueVec_dataOut_3_data_vd;
  assign writeQueueVec_3_deq_bits_data_mask = _writeQueueVec_fifo_3_empty ? writeQueueVec_3_enq_bits_data_mask : writeQueueVec_dataOut_3_data_mask;
  assign writeQueueVec_3_deq_bits_data_data = _writeQueueVec_fifo_3_empty ? writeQueueVec_3_enq_bits_data_data : writeQueueVec_dataOut_3_data_data;
  assign writeQueueVec_3_deq_bits_data_last = _writeQueueVec_fifo_3_empty ? writeQueueVec_3_enq_bits_data_last : writeQueueVec_dataOut_3_data_last;
  assign writeQueueVec_3_deq_bits_data_instructionIndex = _writeQueueVec_fifo_3_empty ? writeQueueVec_3_enq_bits_data_instructionIndex : writeQueueVec_dataOut_3_data_instructionIndex;
  wire [3:0]       writeQueueVec_3_deq_bits_targetLane = _writeQueueVec_fifo_3_empty ? 4'h8 : writeQueueVec_dataOut_3_targetLane;
  wire             otherUnitTargetQueue_deq_valid;
  assign otherUnitTargetQueue_deq_valid = ~_otherUnitTargetQueue_fifo_empty;
  wire             otherUnitTargetQueue_deq_ready;
  wire             otherUnitTargetQueue_enq_ready = ~_otherUnitTargetQueue_fifo_full | otherUnitTargetQueue_deq_ready;
  wire             otherUnitTargetQueue_enq_valid;
  wire             otherUnitDataQueueVec_0_enq_ready = ~_otherUnitDataQueueVec_fifo_full;
  wire             otherUnitDataQueueVec_0_deq_ready;
  wire             otherUnitDataQueueVec_0_enq_valid;
  wire             otherUnitDataQueueVec_0_deq_valid = ~_otherUnitDataQueueVec_fifo_empty | otherUnitDataQueueVec_0_enq_valid;
  wire [31:0]      otherUnitDataQueueVec_0_deq_bits = _otherUnitDataQueueVec_fifo_empty ? otherUnitDataQueueVec_0_enq_bits : _otherUnitDataQueueVec_fifo_data_out;
  wire             otherUnitDataQueueVec_1_enq_ready = ~_otherUnitDataQueueVec_fifo_1_full;
  wire             otherUnitDataQueueVec_1_deq_ready;
  wire             otherUnitDataQueueVec_1_enq_valid;
  wire             otherUnitDataQueueVec_1_deq_valid = ~_otherUnitDataQueueVec_fifo_1_empty | otherUnitDataQueueVec_1_enq_valid;
  wire [31:0]      otherUnitDataQueueVec_1_deq_bits = _otherUnitDataQueueVec_fifo_1_empty ? otherUnitDataQueueVec_1_enq_bits : _otherUnitDataQueueVec_fifo_1_data_out;
  wire             otherUnitDataQueueVec_2_enq_ready = ~_otherUnitDataQueueVec_fifo_2_full;
  wire             otherUnitDataQueueVec_2_deq_ready;
  wire             otherUnitDataQueueVec_2_enq_valid;
  wire             otherUnitDataQueueVec_2_deq_valid = ~_otherUnitDataQueueVec_fifo_2_empty | otherUnitDataQueueVec_2_enq_valid;
  wire [31:0]      otherUnitDataQueueVec_2_deq_bits = _otherUnitDataQueueVec_fifo_2_empty ? otherUnitDataQueueVec_2_enq_bits : _otherUnitDataQueueVec_fifo_2_data_out;
  wire             otherUnitDataQueueVec_3_enq_ready = ~_otherUnitDataQueueVec_fifo_3_full;
  wire             otherUnitDataQueueVec_3_deq_ready;
  wire             otherUnitDataQueueVec_3_enq_valid;
  wire             otherUnitDataQueueVec_3_deq_valid = ~_otherUnitDataQueueVec_fifo_3_empty | otherUnitDataQueueVec_3_enq_valid;
  wire [31:0]      otherUnitDataQueueVec_3_deq_bits = _otherUnitDataQueueVec_fifo_3_empty ? otherUnitDataQueueVec_3_enq_bits : _otherUnitDataQueueVec_fifo_3_data_out;
  wire [3:0]       otherTryReadVrf = _otherUnit_vrfReadDataPorts_valid ? _otherUnit_status_targetLane : 4'h0;
  wire             vrfReadDataPorts_0_valid_0 = otherTryReadVrf[0] | _storeUnit_vrfReadDataPorts_0_valid;
  wire [4:0]       vrfReadDataPorts_0_bits_vs_0 = otherTryReadVrf[0] ? _otherUnit_vrfReadDataPorts_bits_vs : _storeUnit_vrfReadDataPorts_0_bits_vs;
  wire [2:0]       vrfReadDataPorts_0_bits_instructionIndex_0 = otherTryReadVrf[0] ? _otherUnit_vrfReadDataPorts_bits_instructionIndex : _storeUnit_vrfReadDataPorts_0_bits_instructionIndex;
  wire             otherUnitTargetQueue_empty;
  assign otherUnitDataQueueVec_0_enq_valid = vrfReadResults_0_valid & ~otherUnitTargetQueue_empty;
  wire [3:0]       dataDeqFire;
  assign otherUnitDataQueueVec_0_deq_ready = dataDeqFire[0];
  wire             vrfReadDataPorts_1_valid_0 = otherTryReadVrf[1] | _storeUnit_vrfReadDataPorts_1_valid;
  wire [4:0]       vrfReadDataPorts_1_bits_vs_0 = otherTryReadVrf[1] ? _otherUnit_vrfReadDataPorts_bits_vs : _storeUnit_vrfReadDataPorts_1_bits_vs;
  wire [2:0]       vrfReadDataPorts_1_bits_instructionIndex_0 = otherTryReadVrf[1] ? _otherUnit_vrfReadDataPorts_bits_instructionIndex : _storeUnit_vrfReadDataPorts_1_bits_instructionIndex;
  assign otherUnitDataQueueVec_1_enq_valid = vrfReadResults_1_valid & ~otherUnitTargetQueue_empty;
  assign otherUnitDataQueueVec_1_deq_ready = dataDeqFire[1];
  wire             vrfReadDataPorts_2_valid_0 = otherTryReadVrf[2] | _storeUnit_vrfReadDataPorts_2_valid;
  wire [4:0]       vrfReadDataPorts_2_bits_vs_0 = otherTryReadVrf[2] ? _otherUnit_vrfReadDataPorts_bits_vs : _storeUnit_vrfReadDataPorts_2_bits_vs;
  wire [2:0]       vrfReadDataPorts_2_bits_instructionIndex_0 = otherTryReadVrf[2] ? _otherUnit_vrfReadDataPorts_bits_instructionIndex : _storeUnit_vrfReadDataPorts_2_bits_instructionIndex;
  assign otherUnitDataQueueVec_2_enq_valid = vrfReadResults_2_valid & ~otherUnitTargetQueue_empty;
  assign otherUnitDataQueueVec_2_deq_ready = dataDeqFire[2];
  wire             vrfReadDataPorts_3_valid_0 = otherTryReadVrf[3] | _storeUnit_vrfReadDataPorts_3_valid;
  wire [4:0]       vrfReadDataPorts_3_bits_vs_0 = otherTryReadVrf[3] ? _otherUnit_vrfReadDataPorts_bits_vs : _storeUnit_vrfReadDataPorts_3_bits_vs;
  wire [2:0]       vrfReadDataPorts_3_bits_instructionIndex_0 = otherTryReadVrf[3] ? _otherUnit_vrfReadDataPorts_bits_instructionIndex : _storeUnit_vrfReadDataPorts_3_bits_instructionIndex;
  assign otherUnitDataQueueVec_3_enq_valid = vrfReadResults_3_valid & ~otherUnitTargetQueue_empty;
  assign otherUnitDataQueueVec_3_deq_ready = dataDeqFire[3];
  wire [1:0]       otherUnit_vrfReadDataPorts_ready_lo = {vrfReadDataPorts_1_ready_0, vrfReadDataPorts_0_ready_0};
  wire [1:0]       otherUnit_vrfReadDataPorts_ready_hi = {vrfReadDataPorts_3_ready_0, vrfReadDataPorts_2_ready_0};
  wire             otherUnit_vrfReadDataPorts_ready = (|(otherTryReadVrf & {otherUnit_vrfReadDataPorts_ready_hi, otherUnit_vrfReadDataPorts_ready_lo})) & otherUnitTargetQueue_enq_ready;
  assign otherUnitTargetQueue_enq_valid = otherUnit_vrfReadDataPorts_ready & _otherUnit_vrfReadDataPorts_valid;
  wire [3:0]       otherUnitTargetQueue_deq_bits;
  wire [1:0]       otherUnit_vrfReadResults_valid_lo = {otherUnitDataQueueVec_1_deq_valid, otherUnitDataQueueVec_0_deq_valid};
  wire [1:0]       otherUnit_vrfReadResults_valid_hi = {otherUnitDataQueueVec_3_deq_valid, otherUnitDataQueueVec_2_deq_valid};
  assign otherUnitTargetQueue_deq_ready = otherUnitTargetQueue_deq_valid & (|(otherUnitTargetQueue_deq_bits & {otherUnit_vrfReadResults_valid_hi, otherUnit_vrfReadResults_valid_lo}));
  assign dataDeqFire = otherUnitTargetQueue_deq_ready ? otherUnitTargetQueue_deq_bits : 4'h0;
  wire [3:0]       otherTryToWrite = _otherUnit_vrfWritePort_valid ? _otherUnit_status_targetLane : 4'h0;
  wire [1:0]       otherUnit_vrfWritePort_ready_lo = {writeQueueVec_1_enq_ready, writeQueueVec_0_enq_ready};
  wire [1:0]       otherUnit_vrfWritePort_ready_hi = {writeQueueVec_3_enq_ready, writeQueueVec_2_enq_ready};
  assign writeQueueVec_0_enq_valid = otherTryToWrite[0] | _loadUnit_vrfWritePort_0_valid;
  assign writeQueueVec_0_enq_bits_data_vd = otherTryToWrite[0] ? _otherUnit_vrfWritePort_bits_vd : _loadUnit_vrfWritePort_0_bits_vd;
  assign writeQueueVec_0_enq_bits_data_mask = otherTryToWrite[0] ? _otherUnit_vrfWritePort_bits_mask : _loadUnit_vrfWritePort_0_bits_mask;
  assign writeQueueVec_0_enq_bits_data_data = otherTryToWrite[0] ? _otherUnit_vrfWritePort_bits_data : _loadUnit_vrfWritePort_0_bits_data;
  assign writeQueueVec_0_enq_bits_data_last = otherTryToWrite[0] & _otherUnit_vrfWritePort_bits_last;
  assign writeQueueVec_0_enq_bits_data_instructionIndex = otherTryToWrite[0] ? _otherUnit_vrfWritePort_bits_instructionIndex : _loadUnit_vrfWritePort_0_bits_instructionIndex;
  assign writeQueueVec_1_enq_valid = otherTryToWrite[1] | _loadUnit_vrfWritePort_1_valid;
  assign writeQueueVec_1_enq_bits_data_vd = otherTryToWrite[1] ? _otherUnit_vrfWritePort_bits_vd : _loadUnit_vrfWritePort_1_bits_vd;
  assign writeQueueVec_1_enq_bits_data_mask = otherTryToWrite[1] ? _otherUnit_vrfWritePort_bits_mask : _loadUnit_vrfWritePort_1_bits_mask;
  assign writeQueueVec_1_enq_bits_data_data = otherTryToWrite[1] ? _otherUnit_vrfWritePort_bits_data : _loadUnit_vrfWritePort_1_bits_data;
  assign writeQueueVec_1_enq_bits_data_last = otherTryToWrite[1] & _otherUnit_vrfWritePort_bits_last;
  assign writeQueueVec_1_enq_bits_data_instructionIndex = otherTryToWrite[1] ? _otherUnit_vrfWritePort_bits_instructionIndex : _loadUnit_vrfWritePort_1_bits_instructionIndex;
  assign writeQueueVec_2_enq_valid = otherTryToWrite[2] | _loadUnit_vrfWritePort_2_valid;
  assign writeQueueVec_2_enq_bits_data_vd = otherTryToWrite[2] ? _otherUnit_vrfWritePort_bits_vd : _loadUnit_vrfWritePort_2_bits_vd;
  assign writeQueueVec_2_enq_bits_data_mask = otherTryToWrite[2] ? _otherUnit_vrfWritePort_bits_mask : _loadUnit_vrfWritePort_2_bits_mask;
  assign writeQueueVec_2_enq_bits_data_data = otherTryToWrite[2] ? _otherUnit_vrfWritePort_bits_data : _loadUnit_vrfWritePort_2_bits_data;
  assign writeQueueVec_2_enq_bits_data_last = otherTryToWrite[2] & _otherUnit_vrfWritePort_bits_last;
  assign writeQueueVec_2_enq_bits_data_instructionIndex = otherTryToWrite[2] ? _otherUnit_vrfWritePort_bits_instructionIndex : _loadUnit_vrfWritePort_2_bits_instructionIndex;
  assign writeQueueVec_3_enq_valid = otherTryToWrite[3] | _loadUnit_vrfWritePort_3_valid;
  assign writeQueueVec_3_enq_bits_data_vd = otherTryToWrite[3] ? _otherUnit_vrfWritePort_bits_vd : _loadUnit_vrfWritePort_3_bits_vd;
  assign writeQueueVec_3_enq_bits_data_mask = otherTryToWrite[3] ? _otherUnit_vrfWritePort_bits_mask : _loadUnit_vrfWritePort_3_bits_mask;
  assign writeQueueVec_3_enq_bits_data_data = otherTryToWrite[3] ? _otherUnit_vrfWritePort_bits_data : _loadUnit_vrfWritePort_3_bits_data;
  assign writeQueueVec_3_enq_bits_data_last = otherTryToWrite[3] & _otherUnit_vrfWritePort_bits_last;
  assign writeQueueVec_3_enq_bits_data_instructionIndex = otherTryToWrite[3] ? _otherUnit_vrfWritePort_bits_instructionIndex : _loadUnit_vrfWritePort_3_bits_instructionIndex;
  wire [7:0]       _GEN_4 = {5'h0, _loadUnit_status_instructionIndex};
  wire [7:0]       _GEN_5 = {5'h0, _otherUnit_status_instructionIndex};
  wire [7:0]       dataInMSHR = (_loadUnit_status_idle ? 8'h0 : 8'h1 << _GEN_4) | (_otherUnit_status_idle | _otherUnit_status_isStore ? 8'h0 : 8'h1 << _GEN_5);
  reg  [6:0]       queueCount_0;
  reg  [6:0]       queueCount_1;
  reg  [6:0]       queueCount_2;
  reg  [6:0]       queueCount_3;
  reg  [6:0]       queueCount_4;
  reg  [6:0]       queueCount_5;
  reg  [6:0]       queueCount_6;
  reg  [6:0]       queueCount_7;
  wire [7:0]       enqOH = 8'h1 << writeQueueVec_0_enq_bits_data_instructionIndex;
  wire [7:0]       queueEnq = _probeWire_slots_0_writeValid_T ? enqOH : 8'h0;
  wire             writeIndexQueue_deq_valid;
  assign writeIndexQueue_deq_valid = ~_writeIndexQueue_fifo_empty;
  wire             writeIndexQueue_enq_ready = ~_writeIndexQueue_fifo_full;
  wire             writeIndexQueue_enq_valid;
  assign writeIndexQueue_enq_valid = writeQueueVec_0_deq_ready & writeQueueVec_0_deq_valid;
  wire [2:0]       writeIndexQueue_deq_bits;
  wire [7:0]       queueDeq = writeIndexQueue_deq_ready & writeIndexQueue_deq_valid ? 8'h1 << writeIndexQueue_deq_bits : 8'h0;
  wire [6:0]       counterUpdate = queueEnq[0] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_1 = queueEnq[1] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_2 = queueEnq[2] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_3 = queueEnq[3] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_4 = queueEnq[4] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_5 = queueEnq[5] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_6 = queueEnq[6] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_7 = queueEnq[7] ? 7'h1 : 7'h7F;
  wire [1:0]       dataInWriteQueue_0_lo_lo = {|queueCount_1, |queueCount_0};
  wire [1:0]       dataInWriteQueue_0_lo_hi = {|queueCount_3, |queueCount_2};
  wire [3:0]       dataInWriteQueue_0_lo = {dataInWriteQueue_0_lo_hi, dataInWriteQueue_0_lo_lo};
  wire [1:0]       dataInWriteQueue_0_hi_lo = {|queueCount_5, |queueCount_4};
  wire [1:0]       dataInWriteQueue_0_hi_hi = {|queueCount_7, |queueCount_6};
  wire [3:0]       dataInWriteQueue_0_hi = {dataInWriteQueue_0_hi_hi, dataInWriteQueue_0_hi_lo};
  reg  [6:0]       queueCount_0_1;
  reg  [6:0]       queueCount_1_1;
  reg  [6:0]       queueCount_2_1;
  reg  [6:0]       queueCount_3_1;
  reg  [6:0]       queueCount_4_1;
  reg  [6:0]       queueCount_5_1;
  reg  [6:0]       queueCount_6_1;
  reg  [6:0]       queueCount_7_1;
  wire [7:0]       enqOH_1 = 8'h1 << writeQueueVec_1_enq_bits_data_instructionIndex;
  wire [7:0]       queueEnq_1 = _probeWire_slots_1_writeValid_T ? enqOH_1 : 8'h0;
  wire             writeIndexQueue_1_deq_valid;
  assign writeIndexQueue_1_deq_valid = ~_writeIndexQueue_fifo_1_empty;
  wire             writeIndexQueue_1_enq_ready = ~_writeIndexQueue_fifo_1_full;
  wire             writeIndexQueue_1_enq_valid;
  assign writeIndexQueue_1_enq_valid = writeQueueVec_1_deq_ready & writeQueueVec_1_deq_valid;
  wire [2:0]       writeIndexQueue_1_deq_bits;
  wire [7:0]       queueDeq_1 = writeIndexQueue_1_deq_ready & writeIndexQueue_1_deq_valid ? 8'h1 << writeIndexQueue_1_deq_bits : 8'h0;
  wire [6:0]       counterUpdate_8 = queueEnq_1[0] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_9 = queueEnq_1[1] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_10 = queueEnq_1[2] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_11 = queueEnq_1[3] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_12 = queueEnq_1[4] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_13 = queueEnq_1[5] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_14 = queueEnq_1[6] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_15 = queueEnq_1[7] ? 7'h1 : 7'h7F;
  wire [1:0]       dataInWriteQueue_1_lo_lo = {|queueCount_1_1, |queueCount_0_1};
  wire [1:0]       dataInWriteQueue_1_lo_hi = {|queueCount_3_1, |queueCount_2_1};
  wire [3:0]       dataInWriteQueue_1_lo = {dataInWriteQueue_1_lo_hi, dataInWriteQueue_1_lo_lo};
  wire [1:0]       dataInWriteQueue_1_hi_lo = {|queueCount_5_1, |queueCount_4_1};
  wire [1:0]       dataInWriteQueue_1_hi_hi = {|queueCount_7_1, |queueCount_6_1};
  wire [3:0]       dataInWriteQueue_1_hi = {dataInWriteQueue_1_hi_hi, dataInWriteQueue_1_hi_lo};
  reg  [6:0]       queueCount_0_2;
  reg  [6:0]       queueCount_1_2;
  reg  [6:0]       queueCount_2_2;
  reg  [6:0]       queueCount_3_2;
  reg  [6:0]       queueCount_4_2;
  reg  [6:0]       queueCount_5_2;
  reg  [6:0]       queueCount_6_2;
  reg  [6:0]       queueCount_7_2;
  wire [7:0]       enqOH_2 = 8'h1 << writeQueueVec_2_enq_bits_data_instructionIndex;
  wire [7:0]       queueEnq_2 = _probeWire_slots_2_writeValid_T ? enqOH_2 : 8'h0;
  wire             writeIndexQueue_2_deq_valid;
  assign writeIndexQueue_2_deq_valid = ~_writeIndexQueue_fifo_2_empty;
  wire             writeIndexQueue_2_enq_ready = ~_writeIndexQueue_fifo_2_full;
  wire             writeIndexQueue_2_enq_valid;
  assign writeIndexQueue_2_enq_valid = writeQueueVec_2_deq_ready & writeQueueVec_2_deq_valid;
  wire [2:0]       writeIndexQueue_2_deq_bits;
  wire [7:0]       queueDeq_2 = writeIndexQueue_2_deq_ready & writeIndexQueue_2_deq_valid ? 8'h1 << writeIndexQueue_2_deq_bits : 8'h0;
  wire [6:0]       counterUpdate_16 = queueEnq_2[0] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_17 = queueEnq_2[1] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_18 = queueEnq_2[2] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_19 = queueEnq_2[3] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_20 = queueEnq_2[4] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_21 = queueEnq_2[5] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_22 = queueEnq_2[6] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_23 = queueEnq_2[7] ? 7'h1 : 7'h7F;
  wire [1:0]       dataInWriteQueue_2_lo_lo = {|queueCount_1_2, |queueCount_0_2};
  wire [1:0]       dataInWriteQueue_2_lo_hi = {|queueCount_3_2, |queueCount_2_2};
  wire [3:0]       dataInWriteQueue_2_lo = {dataInWriteQueue_2_lo_hi, dataInWriteQueue_2_lo_lo};
  wire [1:0]       dataInWriteQueue_2_hi_lo = {|queueCount_5_2, |queueCount_4_2};
  wire [1:0]       dataInWriteQueue_2_hi_hi = {|queueCount_7_2, |queueCount_6_2};
  wire [3:0]       dataInWriteQueue_2_hi = {dataInWriteQueue_2_hi_hi, dataInWriteQueue_2_hi_lo};
  reg  [6:0]       queueCount_0_3;
  reg  [6:0]       queueCount_1_3;
  reg  [6:0]       queueCount_2_3;
  reg  [6:0]       queueCount_3_3;
  reg  [6:0]       queueCount_4_3;
  reg  [6:0]       queueCount_5_3;
  reg  [6:0]       queueCount_6_3;
  reg  [6:0]       queueCount_7_3;
  wire [7:0]       enqOH_3 = 8'h1 << writeQueueVec_3_enq_bits_data_instructionIndex;
  wire [7:0]       queueEnq_3 = _probeWire_slots_3_writeValid_T ? enqOH_3 : 8'h0;
  wire             writeIndexQueue_3_deq_valid;
  assign writeIndexQueue_3_deq_valid = ~_writeIndexQueue_fifo_3_empty;
  wire             writeIndexQueue_3_enq_ready = ~_writeIndexQueue_fifo_3_full;
  wire             writeIndexQueue_3_enq_valid;
  assign writeIndexQueue_3_enq_valid = writeQueueVec_3_deq_ready & writeQueueVec_3_deq_valid;
  wire [2:0]       writeIndexQueue_3_deq_bits;
  wire [7:0]       queueDeq_3 = writeIndexQueue_3_deq_ready & writeIndexQueue_3_deq_valid ? 8'h1 << writeIndexQueue_3_deq_bits : 8'h0;
  wire [6:0]       counterUpdate_24 = queueEnq_3[0] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_25 = queueEnq_3[1] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_26 = queueEnq_3[2] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_27 = queueEnq_3[3] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_28 = queueEnq_3[4] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_29 = queueEnq_3[5] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_30 = queueEnq_3[6] ? 7'h1 : 7'h7F;
  wire [6:0]       counterUpdate_31 = queueEnq_3[7] ? 7'h1 : 7'h7F;
  wire [1:0]       dataInWriteQueue_3_lo_lo = {|queueCount_1_3, |queueCount_0_3};
  wire [1:0]       dataInWriteQueue_3_lo_hi = {|queueCount_3_3, |queueCount_2_3};
  wire [3:0]       dataInWriteQueue_3_lo = {dataInWriteQueue_3_lo_hi, dataInWriteQueue_3_lo_lo};
  wire [1:0]       dataInWriteQueue_3_hi_lo = {|queueCount_5_3, |queueCount_4_3};
  wire [1:0]       dataInWriteQueue_3_hi_hi = {|queueCount_7_3, |queueCount_6_3};
  wire [3:0]       dataInWriteQueue_3_hi = {dataInWriteQueue_3_hi_hi, dataInWriteQueue_3_hi_lo};
  wire             sourceQueue_deq_valid;
  assign sourceQueue_deq_valid = ~_sourceQueue_fifo_empty;
  wire             sourceQueue_enq_ready = ~_sourceQueue_fifo_full;
  wire             sourceQueue_enq_valid;
  wire             sourceQueue_deq_ready;
  wire             axi4Port_ar_valid_0 = _loadUnit_memRequest_valid & sourceQueue_enq_ready;
  wire             axi4Port_r_ready_0;
  assign sourceQueue_enq_valid = _loadUnit_memRequest_valid & axi4Port_ar_ready_0;
  assign sourceQueue_deq_ready = axi4Port_r_ready_0 & axi4Port_r_valid_0;
  assign dataQueue_deq_valid = ~_dataQueue_fifo_empty;
  wire             axi4Port_w_valid_0 = dataQueue_deq_valid;
  wire [127:0]     dataQueue_dataOut_data;
  wire [127:0]     axi4Port_w_bits_data_0 = dataQueue_deq_bits_data;
  wire [15:0]      dataQueue_dataOut_mask;
  wire [15:0]      axi4Port_w_bits_strb_0 = dataQueue_deq_bits_mask;
  wire [3:0]       dataQueue_dataOut_index;
  wire [31:0]      dataQueue_dataOut_address;
  wire [3:0]       dataQueue_enq_bits_index;
  wire [31:0]      dataQueue_enq_bits_address;
  wire [35:0]      dataQueue_dataIn_lo = {dataQueue_enq_bits_index, dataQueue_enq_bits_address};
  wire [127:0]     dataQueue_enq_bits_data;
  wire [15:0]      dataQueue_enq_bits_mask;
  wire [143:0]     dataQueue_dataIn_hi = {dataQueue_enq_bits_data, dataQueue_enq_bits_mask};
  wire [179:0]     dataQueue_dataIn = {dataQueue_dataIn_hi, dataQueue_dataIn_lo};
  assign dataQueue_dataOut_address = _dataQueue_fifo_data_out[31:0];
  assign dataQueue_dataOut_index = _dataQueue_fifo_data_out[35:32];
  assign dataQueue_dataOut_mask = _dataQueue_fifo_data_out[51:36];
  assign dataQueue_dataOut_data = _dataQueue_fifo_data_out[179:52];
  assign dataQueue_deq_bits_data = dataQueue_dataOut_data;
  assign dataQueue_deq_bits_mask = dataQueue_dataOut_mask;
  wire [3:0]       dataQueue_deq_bits_index = dataQueue_dataOut_index;
  wire [31:0]      dataQueue_deq_bits_address = dataQueue_dataOut_address;
  wire             dataQueue_enq_ready = ~_dataQueue_fifo_full;
  wire             dataQueue_enq_valid;
  wire             axi4Port_aw_valid_0 = _storeUnit_memRequest_valid & dataQueue_enq_ready;
  wire [1:0]       axi4Port_aw_bits_id_0 = _storeUnit_memRequest_bits_index[1:0];
  assign dataQueue_enq_valid = _storeUnit_memRequest_valid & axi4Port_aw_ready_0;
  wire             simpleSourceQueue_deq_valid;
  assign simpleSourceQueue_deq_valid = ~_simpleSourceQueue_fifo_empty;
  wire             simpleSourceQueue_enq_ready = ~_simpleSourceQueue_fifo_full;
  wire             simpleSourceQueue_enq_valid;
  wire             simpleSourceQueue_deq_ready;
  wire             simpleAccessPorts_ar_valid_0 = _otherUnit_memReadRequest_valid & simpleSourceQueue_enq_ready;
  wire             simpleAccessPorts_r_ready_0;
  assign simpleSourceQueue_enq_valid = _otherUnit_memReadRequest_valid & simpleAccessPorts_ar_ready_0;
  assign simpleSourceQueue_deq_ready = simpleAccessPorts_r_ready_0 & simpleAccessPorts_r_valid_0;
  assign simpleDataQueue_deq_valid = ~_simpleDataQueue_fifo_empty;
  wire             simpleAccessPorts_w_valid_0 = simpleDataQueue_deq_valid;
  wire [31:0]      simpleDataQueue_dataOut_data;
  wire [31:0]      simpleAccessPorts_w_bits_data_0 = simpleDataQueue_deq_bits_data;
  wire [3:0]       simpleDataQueue_dataOut_mask;
  wire [3:0]       simpleAccessPorts_w_bits_strb_0 = simpleDataQueue_deq_bits_mask;
  wire [7:0]       simpleDataQueue_dataOut_source;
  wire [31:0]      simpleDataQueue_dataOut_address;
  wire [1:0]       simpleDataQueue_dataOut_size;
  wire [31:0]      simpleDataQueue_enq_bits_address;
  wire [1:0]       simpleDataQueue_enq_bits_size;
  wire [33:0]      simpleDataQueue_dataIn_lo = {simpleDataQueue_enq_bits_address, simpleDataQueue_enq_bits_size};
  wire [31:0]      simpleDataQueue_enq_bits_data;
  wire [3:0]       simpleDataQueue_enq_bits_mask;
  wire [35:0]      simpleDataQueue_dataIn_hi_hi = {simpleDataQueue_enq_bits_data, simpleDataQueue_enq_bits_mask};
  wire [7:0]       simpleDataQueue_enq_bits_source;
  wire [43:0]      simpleDataQueue_dataIn_hi = {simpleDataQueue_dataIn_hi_hi, simpleDataQueue_enq_bits_source};
  wire [77:0]      simpleDataQueue_dataIn = {simpleDataQueue_dataIn_hi, simpleDataQueue_dataIn_lo};
  assign simpleDataQueue_dataOut_size = _simpleDataQueue_fifo_data_out[1:0];
  assign simpleDataQueue_dataOut_address = _simpleDataQueue_fifo_data_out[33:2];
  assign simpleDataQueue_dataOut_source = _simpleDataQueue_fifo_data_out[41:34];
  assign simpleDataQueue_dataOut_mask = _simpleDataQueue_fifo_data_out[45:42];
  assign simpleDataQueue_dataOut_data = _simpleDataQueue_fifo_data_out[77:46];
  assign simpleDataQueue_deq_bits_data = simpleDataQueue_dataOut_data;
  assign simpleDataQueue_deq_bits_mask = simpleDataQueue_dataOut_mask;
  wire [7:0]       simpleDataQueue_deq_bits_source = simpleDataQueue_dataOut_source;
  wire [31:0]      simpleDataQueue_deq_bits_address = simpleDataQueue_dataOut_address;
  wire [1:0]       simpleDataQueue_deq_bits_size = simpleDataQueue_dataOut_size;
  wire             simpleDataQueue_enq_ready = ~_simpleDataQueue_fifo_full;
  wire             simpleDataQueue_enq_valid;
  wire             simpleAccessPorts_aw_valid_0 = _otherUnit_memWriteRequest_valid & dataQueue_enq_ready;
  wire [2:0]       simpleAccessPorts_aw_bits_size_0 = {1'h0, _otherUnit_memWriteRequest_bits_size};
  wire [1:0]       simpleAccessPorts_aw_bits_id_0 = _otherUnit_memWriteRequest_bits_source[1:0];
  assign simpleDataQueue_enq_valid = _otherUnit_memWriteRequest_valid & simpleAccessPorts_aw_ready_0;
  wire [1:0]       tokenIO_offsetGroupRelease_lo = {_otherUnit_offsetRelease_1, _otherUnit_offsetRelease_0};
  wire [1:0]       tokenIO_offsetGroupRelease_hi = {_otherUnit_offsetRelease_3, _otherUnit_offsetRelease_2};
  wire             unitOrder =
    _loadUnit_status_instructionIndex == _storeUnit_status_instructionIndex | _loadUnit_status_instructionIndex[1:0] < _storeUnit_status_instructionIndex[1:0] ^ _loadUnit_status_instructionIndex[2] ^ _storeUnit_status_instructionIndex[2];
  wire             loadAddressConflict = _loadUnit_status_startAddress >= _storeUnit_status_startAddress & _loadUnit_status_startAddress <= _storeUnit_status_endAddress;
  wire             storeAddressConflict = _storeUnit_status_startAddress >= _loadUnit_status_startAddress & _storeUnit_status_startAddress <= _loadUnit_status_endAddress;
  wire             stallLoad = ~unitOrder & loadAddressConflict & ~_storeUnit_status_idle;
  wire             stallStore = unitOrder & storeAddressConflict & ~_loadUnit_status_idle;
  always @(posedge clock) begin
    if (reset) begin
      v0_0 <= 32'h0;
      v0_1 <= 32'h0;
      v0_2 <= 32'h0;
      v0_3 <= 32'h0;
      queueCount_0 <= 7'h0;
      queueCount_1 <= 7'h0;
      queueCount_2 <= 7'h0;
      queueCount_3 <= 7'h0;
      queueCount_4 <= 7'h0;
      queueCount_5 <= 7'h0;
      queueCount_6 <= 7'h0;
      queueCount_7 <= 7'h0;
      queueCount_0_1 <= 7'h0;
      queueCount_1_1 <= 7'h0;
      queueCount_2_1 <= 7'h0;
      queueCount_3_1 <= 7'h0;
      queueCount_4_1 <= 7'h0;
      queueCount_5_1 <= 7'h0;
      queueCount_6_1 <= 7'h0;
      queueCount_7_1 <= 7'h0;
      queueCount_0_2 <= 7'h0;
      queueCount_1_2 <= 7'h0;
      queueCount_2_2 <= 7'h0;
      queueCount_3_2 <= 7'h0;
      queueCount_4_2 <= 7'h0;
      queueCount_5_2 <= 7'h0;
      queueCount_6_2 <= 7'h0;
      queueCount_7_2 <= 7'h0;
      queueCount_0_3 <= 7'h0;
      queueCount_1_3 <= 7'h0;
      queueCount_2_3 <= 7'h0;
      queueCount_3_3 <= 7'h0;
      queueCount_4_3 <= 7'h0;
      queueCount_5_3 <= 7'h0;
      queueCount_6_3 <= 7'h0;
      queueCount_7_3 <= 7'h0;
    end
    else begin
      if (v0UpdateVec_0_valid)
        v0_0 <= v0_0 & ~maskExt | maskExt & v0UpdateVec_0_bits_data;
      if (v0UpdateVec_1_valid)
        v0_1 <= v0_1 & ~maskExt_1 | maskExt_1 & v0UpdateVec_1_bits_data;
      if (v0UpdateVec_2_valid)
        v0_2 <= v0_2 & ~maskExt_2 | maskExt_2 & v0UpdateVec_2_bits_data;
      if (v0UpdateVec_3_valid)
        v0_3 <= v0_3 & ~maskExt_3 | maskExt_3 & v0UpdateVec_3_bits_data;
      if (queueEnq[0] ^ queueDeq[0])
        queueCount_0 <= queueCount_0 + counterUpdate;
      if (queueEnq[1] ^ queueDeq[1])
        queueCount_1 <= queueCount_1 + counterUpdate_1;
      if (queueEnq[2] ^ queueDeq[2])
        queueCount_2 <= queueCount_2 + counterUpdate_2;
      if (queueEnq[3] ^ queueDeq[3])
        queueCount_3 <= queueCount_3 + counterUpdate_3;
      if (queueEnq[4] ^ queueDeq[4])
        queueCount_4 <= queueCount_4 + counterUpdate_4;
      if (queueEnq[5] ^ queueDeq[5])
        queueCount_5 <= queueCount_5 + counterUpdate_5;
      if (queueEnq[6] ^ queueDeq[6])
        queueCount_6 <= queueCount_6 + counterUpdate_6;
      if (queueEnq[7] ^ queueDeq[7])
        queueCount_7 <= queueCount_7 + counterUpdate_7;
      if (queueEnq_1[0] ^ queueDeq_1[0])
        queueCount_0_1 <= queueCount_0_1 + counterUpdate_8;
      if (queueEnq_1[1] ^ queueDeq_1[1])
        queueCount_1_1 <= queueCount_1_1 + counterUpdate_9;
      if (queueEnq_1[2] ^ queueDeq_1[2])
        queueCount_2_1 <= queueCount_2_1 + counterUpdate_10;
      if (queueEnq_1[3] ^ queueDeq_1[3])
        queueCount_3_1 <= queueCount_3_1 + counterUpdate_11;
      if (queueEnq_1[4] ^ queueDeq_1[4])
        queueCount_4_1 <= queueCount_4_1 + counterUpdate_12;
      if (queueEnq_1[5] ^ queueDeq_1[5])
        queueCount_5_1 <= queueCount_5_1 + counterUpdate_13;
      if (queueEnq_1[6] ^ queueDeq_1[6])
        queueCount_6_1 <= queueCount_6_1 + counterUpdate_14;
      if (queueEnq_1[7] ^ queueDeq_1[7])
        queueCount_7_1 <= queueCount_7_1 + counterUpdate_15;
      if (queueEnq_2[0] ^ queueDeq_2[0])
        queueCount_0_2 <= queueCount_0_2 + counterUpdate_16;
      if (queueEnq_2[1] ^ queueDeq_2[1])
        queueCount_1_2 <= queueCount_1_2 + counterUpdate_17;
      if (queueEnq_2[2] ^ queueDeq_2[2])
        queueCount_2_2 <= queueCount_2_2 + counterUpdate_18;
      if (queueEnq_2[3] ^ queueDeq_2[3])
        queueCount_3_2 <= queueCount_3_2 + counterUpdate_19;
      if (queueEnq_2[4] ^ queueDeq_2[4])
        queueCount_4_2 <= queueCount_4_2 + counterUpdate_20;
      if (queueEnq_2[5] ^ queueDeq_2[5])
        queueCount_5_2 <= queueCount_5_2 + counterUpdate_21;
      if (queueEnq_2[6] ^ queueDeq_2[6])
        queueCount_6_2 <= queueCount_6_2 + counterUpdate_22;
      if (queueEnq_2[7] ^ queueDeq_2[7])
        queueCount_7_2 <= queueCount_7_2 + counterUpdate_23;
      if (queueEnq_3[0] ^ queueDeq_3[0])
        queueCount_0_3 <= queueCount_0_3 + counterUpdate_24;
      if (queueEnq_3[1] ^ queueDeq_3[1])
        queueCount_1_3 <= queueCount_1_3 + counterUpdate_25;
      if (queueEnq_3[2] ^ queueDeq_3[2])
        queueCount_2_3 <= queueCount_2_3 + counterUpdate_26;
      if (queueEnq_3[3] ^ queueDeq_3[3])
        queueCount_3_3 <= queueCount_3_3 + counterUpdate_27;
      if (queueEnq_3[4] ^ queueDeq_3[4])
        queueCount_4_3 <= queueCount_4_3 + counterUpdate_28;
      if (queueEnq_3[5] ^ queueDeq_3[5])
        queueCount_5_3 <= queueCount_5_3 + counterUpdate_29;
      if (queueEnq_3[6] ^ queueDeq_3[6])
        queueCount_6_3 <= queueCount_6_3 + counterUpdate_30;
      if (queueEnq_3[7] ^ queueDeq_3[7])
        queueCount_7_3 <= queueCount_7_3 + counterUpdate_31;
    end
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:10];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [3:0] i = 4'h0; i < 4'hB; i += 4'h1) begin
          _RANDOM[i] = `RANDOM;
        end
        v0_0 = _RANDOM[4'h0];
        v0_1 = _RANDOM[4'h1];
        v0_2 = _RANDOM[4'h2];
        v0_3 = _RANDOM[4'h3];
        queueCount_0 = _RANDOM[4'h4][6:0];
        queueCount_1 = _RANDOM[4'h4][13:7];
        queueCount_2 = _RANDOM[4'h4][20:14];
        queueCount_3 = _RANDOM[4'h4][27:21];
        queueCount_4 = {_RANDOM[4'h4][31:28], _RANDOM[4'h5][2:0]};
        queueCount_5 = _RANDOM[4'h5][9:3];
        queueCount_6 = _RANDOM[4'h5][16:10];
        queueCount_7 = _RANDOM[4'h5][23:17];
        queueCount_0_1 = _RANDOM[4'h5][30:24];
        queueCount_1_1 = {_RANDOM[4'h5][31], _RANDOM[4'h6][5:0]};
        queueCount_2_1 = _RANDOM[4'h6][12:6];
        queueCount_3_1 = _RANDOM[4'h6][19:13];
        queueCount_4_1 = _RANDOM[4'h6][26:20];
        queueCount_5_1 = {_RANDOM[4'h6][31:27], _RANDOM[4'h7][1:0]};
        queueCount_6_1 = _RANDOM[4'h7][8:2];
        queueCount_7_1 = _RANDOM[4'h7][15:9];
        queueCount_0_2 = _RANDOM[4'h7][22:16];
        queueCount_1_2 = _RANDOM[4'h7][29:23];
        queueCount_2_2 = {_RANDOM[4'h7][31:30], _RANDOM[4'h8][4:0]};
        queueCount_3_2 = _RANDOM[4'h8][11:5];
        queueCount_4_2 = _RANDOM[4'h8][18:12];
        queueCount_5_2 = _RANDOM[4'h8][25:19];
        queueCount_6_2 = {_RANDOM[4'h8][31:26], _RANDOM[4'h9][0]};
        queueCount_7_2 = _RANDOM[4'h9][7:1];
        queueCount_0_3 = _RANDOM[4'h9][14:8];
        queueCount_1_3 = _RANDOM[4'h9][21:15];
        queueCount_2_3 = _RANDOM[4'h9][28:22];
        queueCount_3_3 = {_RANDOM[4'h9][31:29], _RANDOM[4'hA][3:0]};
        queueCount_4_3 = _RANDOM[4'hA][10:4];
        queueCount_5_3 = _RANDOM[4'hA][17:11];
        queueCount_6_3 = _RANDOM[4'hA][24:18];
        queueCount_7_3 = _RANDOM[4'hA][31:25];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  wire [3:0]       sourceQueue_deq_bits;
  wire [31:0]      axi4Port_aw_bits_addr_0;
  assign axi4Port_aw_bits_addr_0 = _storeUnit_memRequest_bits_address;
  assign dataQueue_enq_bits_index = _storeUnit_memRequest_bits_index;
  assign dataQueue_enq_bits_address = _storeUnit_memRequest_bits_address;
  wire [6:0]       simpleSourceQueue_deq_bits;
  wire [31:0]      simpleAccessPorts_aw_bits_addr_0;
  assign simpleAccessPorts_aw_bits_addr_0 = _otherUnit_memWriteRequest_bits_address;
  wire [3:0]       otherUnitTargetQueue_enq_bits;
  assign otherUnitTargetQueue_enq_bits = _otherUnit_status_targetLane;
  assign simpleDataQueue_enq_bits_source = _otherUnit_memWriteRequest_bits_source;
  assign simpleDataQueue_enq_bits_address = _otherUnit_memWriteRequest_bits_address;
  assign simpleDataQueue_enq_bits_size = _otherUnit_memWriteRequest_bits_size;
  wire             writeQueueVec_0_empty;
  assign writeQueueVec_0_empty = _writeQueueVec_fifo_empty;
  wire             writeQueueVec_0_full;
  assign writeQueueVec_0_full = _writeQueueVec_fifo_full;
  wire             writeQueueVec_1_empty;
  assign writeQueueVec_1_empty = _writeQueueVec_fifo_1_empty;
  wire             writeQueueVec_1_full;
  assign writeQueueVec_1_full = _writeQueueVec_fifo_1_full;
  wire             writeQueueVec_2_empty;
  assign writeQueueVec_2_empty = _writeQueueVec_fifo_2_empty;
  wire             writeQueueVec_2_full;
  assign writeQueueVec_2_full = _writeQueueVec_fifo_2_full;
  wire             writeQueueVec_3_empty;
  assign writeQueueVec_3_empty = _writeQueueVec_fifo_3_empty;
  wire             writeQueueVec_3_full;
  assign writeQueueVec_3_full = _writeQueueVec_fifo_3_full;
  assign otherUnitTargetQueue_empty = _otherUnitTargetQueue_fifo_empty;
  wire             otherUnitTargetQueue_full;
  assign otherUnitTargetQueue_full = _otherUnitTargetQueue_fifo_full;
  wire             otherUnitDataQueueVec_0_empty;
  assign otherUnitDataQueueVec_0_empty = _otherUnitDataQueueVec_fifo_empty;
  wire             otherUnitDataQueueVec_0_full;
  assign otherUnitDataQueueVec_0_full = _otherUnitDataQueueVec_fifo_full;
  wire             otherUnitDataQueueVec_1_empty;
  assign otherUnitDataQueueVec_1_empty = _otherUnitDataQueueVec_fifo_1_empty;
  wire             otherUnitDataQueueVec_1_full;
  assign otherUnitDataQueueVec_1_full = _otherUnitDataQueueVec_fifo_1_full;
  wire             otherUnitDataQueueVec_2_empty;
  assign otherUnitDataQueueVec_2_empty = _otherUnitDataQueueVec_fifo_2_empty;
  wire             otherUnitDataQueueVec_2_full;
  assign otherUnitDataQueueVec_2_full = _otherUnitDataQueueVec_fifo_2_full;
  wire             otherUnitDataQueueVec_3_empty;
  assign otherUnitDataQueueVec_3_empty = _otherUnitDataQueueVec_fifo_3_empty;
  wire             otherUnitDataQueueVec_3_full;
  assign otherUnitDataQueueVec_3_full = _otherUnitDataQueueVec_fifo_3_full;
  wire             writeIndexQueue_empty;
  assign writeIndexQueue_empty = _writeIndexQueue_fifo_empty;
  wire             writeIndexQueue_full;
  assign writeIndexQueue_full = _writeIndexQueue_fifo_full;
  wire             writeIndexQueue_1_empty;
  assign writeIndexQueue_1_empty = _writeIndexQueue_fifo_1_empty;
  wire             writeIndexQueue_1_full;
  assign writeIndexQueue_1_full = _writeIndexQueue_fifo_1_full;
  wire             writeIndexQueue_2_empty;
  assign writeIndexQueue_2_empty = _writeIndexQueue_fifo_2_empty;
  wire             writeIndexQueue_2_full;
  assign writeIndexQueue_2_full = _writeIndexQueue_fifo_2_full;
  wire             writeIndexQueue_3_empty;
  assign writeIndexQueue_3_empty = _writeIndexQueue_fifo_3_empty;
  wire             writeIndexQueue_3_full;
  assign writeIndexQueue_3_full = _writeIndexQueue_fifo_3_full;
  wire             sourceQueue_empty;
  assign sourceQueue_empty = _sourceQueue_fifo_empty;
  wire             sourceQueue_full;
  assign sourceQueue_full = _sourceQueue_fifo_full;
  wire             dataQueue_empty;
  assign dataQueue_empty = _dataQueue_fifo_empty;
  wire             dataQueue_full;
  assign dataQueue_full = _dataQueue_fifo_full;
  wire             simpleSourceQueue_empty;
  assign simpleSourceQueue_empty = _simpleSourceQueue_fifo_empty;
  wire             simpleSourceQueue_full;
  assign simpleSourceQueue_full = _simpleSourceQueue_fifo_full;
  wire             simpleDataQueue_empty;
  assign simpleDataQueue_empty = _simpleDataQueue_fifo_empty;
  wire             simpleDataQueue_full;
  assign simpleDataQueue_full = _simpleDataQueue_fifo_full;
  LoadUnit loadUnit (
    .clock                                                  (clock),
    .reset                                                  (reset),
    .lsuRequest_valid                                       (reqEnq_0),
    .lsuRequest_bits_instructionInformation_nf              (request_bits_instructionInformation_nf_0),
    .lsuRequest_bits_instructionInformation_mew             (request_bits_instructionInformation_mew_0),
    .lsuRequest_bits_instructionInformation_mop             (request_bits_instructionInformation_mop_0),
    .lsuRequest_bits_instructionInformation_lumop           (request_bits_instructionInformation_lumop_0),
    .lsuRequest_bits_instructionInformation_eew             (request_bits_instructionInformation_eew_0),
    .lsuRequest_bits_instructionInformation_vs3             (request_bits_instructionInformation_vs3_0),
    .lsuRequest_bits_instructionInformation_isStore         (request_bits_instructionInformation_isStore_0),
    .lsuRequest_bits_instructionInformation_maskedLoadStore (request_bits_instructionInformation_maskedLoadStore_0),
    .lsuRequest_bits_rs1Data                                (request_bits_rs1Data_0),
    .lsuRequest_bits_rs2Data                                (request_bits_rs2Data_0),
    .lsuRequest_bits_instructionIndex                       (request_bits_instructionIndex_0),
    .csrInterface_vl                                        (csrInterface_vl),
    .csrInterface_vStart                                    (csrInterface_vStart),
    .csrInterface_vlmul                                     (csrInterface_vlmul),
    .csrInterface_vSew                                      (csrInterface_vSew),
    .csrInterface_vxrm                                      (csrInterface_vxrm),
    .csrInterface_vta                                       (csrInterface_vta),
    .csrInterface_vma                                       (csrInterface_vma),
    .maskInput                                              (_GEN_1[maskSelect]),
    .maskSelect_valid                                       (_loadUnit_maskSelect_valid),
    .maskSelect_bits                                        (_loadUnit_maskSelect_bits),
    .addressConflict                                        (stallLoad),
    .memRequest_ready                                       (sourceQueue_enq_ready & axi4Port_ar_ready_0),
    .memRequest_valid                                       (_loadUnit_memRequest_valid),
    .memRequest_bits_src                                    (sourceQueue_enq_bits),
    .memRequest_bits_address                                (axi4Port_ar_bits_addr_0),
    .memResponse_ready                                      (axi4Port_r_ready_0),
    .memResponse_valid                                      (axi4Port_r_valid_0),
    .memResponse_bits_data                                  (axi4Port_r_bits_data_0),
    .memResponse_bits_index                                 (sourceQueue_deq_bits),
    .status_idle                                            (_loadUnit_status_idle),
    .status_last                                            (_loadUnit_status_last),
    .status_instructionIndex                                (_loadUnit_status_instructionIndex),
    .status_changeMaskGroup                                 (/* unused */),
    .status_startAddress                                    (_loadUnit_status_startAddress),
    .status_endAddress                                      (_loadUnit_status_endAddress),
    .vrfWritePort_0_ready                                   (writeQueueVec_0_enq_ready & ~(otherTryToWrite[0])),
    .vrfWritePort_0_valid                                   (_loadUnit_vrfWritePort_0_valid),
    .vrfWritePort_0_bits_vd                                 (_loadUnit_vrfWritePort_0_bits_vd),
    .vrfWritePort_0_bits_mask                               (_loadUnit_vrfWritePort_0_bits_mask),
    .vrfWritePort_0_bits_data                               (_loadUnit_vrfWritePort_0_bits_data),
    .vrfWritePort_0_bits_instructionIndex                   (_loadUnit_vrfWritePort_0_bits_instructionIndex),
    .vrfWritePort_1_ready                                   (writeQueueVec_1_enq_ready & ~(otherTryToWrite[1])),
    .vrfWritePort_1_valid                                   (_loadUnit_vrfWritePort_1_valid),
    .vrfWritePort_1_bits_vd                                 (_loadUnit_vrfWritePort_1_bits_vd),
    .vrfWritePort_1_bits_mask                               (_loadUnit_vrfWritePort_1_bits_mask),
    .vrfWritePort_1_bits_data                               (_loadUnit_vrfWritePort_1_bits_data),
    .vrfWritePort_1_bits_instructionIndex                   (_loadUnit_vrfWritePort_1_bits_instructionIndex),
    .vrfWritePort_2_ready                                   (writeQueueVec_2_enq_ready & ~(otherTryToWrite[2])),
    .vrfWritePort_2_valid                                   (_loadUnit_vrfWritePort_2_valid),
    .vrfWritePort_2_bits_vd                                 (_loadUnit_vrfWritePort_2_bits_vd),
    .vrfWritePort_2_bits_mask                               (_loadUnit_vrfWritePort_2_bits_mask),
    .vrfWritePort_2_bits_data                               (_loadUnit_vrfWritePort_2_bits_data),
    .vrfWritePort_2_bits_instructionIndex                   (_loadUnit_vrfWritePort_2_bits_instructionIndex),
    .vrfWritePort_3_ready                                   (writeQueueVec_3_enq_ready & ~(otherTryToWrite[3])),
    .vrfWritePort_3_valid                                   (_loadUnit_vrfWritePort_3_valid),
    .vrfWritePort_3_bits_vd                                 (_loadUnit_vrfWritePort_3_bits_vd),
    .vrfWritePort_3_bits_mask                               (_loadUnit_vrfWritePort_3_bits_mask),
    .vrfWritePort_3_bits_data                               (_loadUnit_vrfWritePort_3_bits_data),
    .vrfWritePort_3_bits_instructionIndex                   (_loadUnit_vrfWritePort_3_bits_instructionIndex)
  );
  StoreUnit storeUnit (
    .clock                                                  (clock),
    .reset                                                  (reset),
    .lsuRequest_valid                                       (reqEnq_1),
    .lsuRequest_bits_instructionInformation_nf              (request_bits_instructionInformation_nf_0),
    .lsuRequest_bits_instructionInformation_mew             (request_bits_instructionInformation_mew_0),
    .lsuRequest_bits_instructionInformation_mop             (request_bits_instructionInformation_mop_0),
    .lsuRequest_bits_instructionInformation_lumop           (request_bits_instructionInformation_lumop_0),
    .lsuRequest_bits_instructionInformation_eew             (request_bits_instructionInformation_eew_0),
    .lsuRequest_bits_instructionInformation_vs3             (request_bits_instructionInformation_vs3_0),
    .lsuRequest_bits_instructionInformation_isStore         (request_bits_instructionInformation_isStore_0),
    .lsuRequest_bits_instructionInformation_maskedLoadStore (request_bits_instructionInformation_maskedLoadStore_0),
    .lsuRequest_bits_rs1Data                                (request_bits_rs1Data_0),
    .lsuRequest_bits_rs2Data                                (request_bits_rs2Data_0),
    .lsuRequest_bits_instructionIndex                       (request_bits_instructionIndex_0),
    .csrInterface_vl                                        (csrInterface_vl),
    .csrInterface_vStart                                    (csrInterface_vStart),
    .csrInterface_vlmul                                     (csrInterface_vlmul),
    .csrInterface_vSew                                      (csrInterface_vSew),
    .csrInterface_vxrm                                      (csrInterface_vxrm),
    .csrInterface_vta                                       (csrInterface_vta),
    .csrInterface_vma                                       (csrInterface_vma),
    .maskInput                                              (_GEN_2[maskSelect_1]),
    .maskSelect_valid                                       (_storeUnit_maskSelect_valid),
    .maskSelect_bits                                        (_storeUnit_maskSelect_bits),
    .memRequest_ready                                       (axi4Port_aw_ready_0 & dataQueue_enq_ready),
    .memRequest_valid                                       (_storeUnit_memRequest_valid),
    .memRequest_bits_data                                   (dataQueue_enq_bits_data),
    .memRequest_bits_mask                                   (dataQueue_enq_bits_mask),
    .memRequest_bits_index                                  (_storeUnit_memRequest_bits_index),
    .memRequest_bits_address                                (_storeUnit_memRequest_bits_address),
    .status_idle                                            (_storeUnit_status_idle),
    .status_last                                            (_storeUnit_status_last),
    .status_instructionIndex                                (_storeUnit_status_instructionIndex),
    .status_changeMaskGroup                                 (/* unused */),
    .status_startAddress                                    (_storeUnit_status_startAddress),
    .status_endAddress                                      (_storeUnit_status_endAddress),
    .vrfReadDataPorts_0_ready                               (vrfReadDataPorts_0_ready_0 & ~(otherTryReadVrf[0])),
    .vrfReadDataPorts_0_valid                               (_storeUnit_vrfReadDataPorts_0_valid),
    .vrfReadDataPorts_0_bits_vs                             (_storeUnit_vrfReadDataPorts_0_bits_vs),
    .vrfReadDataPorts_0_bits_instructionIndex               (_storeUnit_vrfReadDataPorts_0_bits_instructionIndex),
    .vrfReadDataPorts_1_ready                               (vrfReadDataPorts_1_ready_0 & ~(otherTryReadVrf[1])),
    .vrfReadDataPorts_1_valid                               (_storeUnit_vrfReadDataPorts_1_valid),
    .vrfReadDataPorts_1_bits_vs                             (_storeUnit_vrfReadDataPorts_1_bits_vs),
    .vrfReadDataPorts_1_bits_instructionIndex               (_storeUnit_vrfReadDataPorts_1_bits_instructionIndex),
    .vrfReadDataPorts_2_ready                               (vrfReadDataPorts_2_ready_0 & ~(otherTryReadVrf[2])),
    .vrfReadDataPorts_2_valid                               (_storeUnit_vrfReadDataPorts_2_valid),
    .vrfReadDataPorts_2_bits_vs                             (_storeUnit_vrfReadDataPorts_2_bits_vs),
    .vrfReadDataPorts_2_bits_instructionIndex               (_storeUnit_vrfReadDataPorts_2_bits_instructionIndex),
    .vrfReadDataPorts_3_ready                               (vrfReadDataPorts_3_ready_0 & ~(otherTryReadVrf[3])),
    .vrfReadDataPorts_3_valid                               (_storeUnit_vrfReadDataPorts_3_valid),
    .vrfReadDataPorts_3_bits_vs                             (_storeUnit_vrfReadDataPorts_3_bits_vs),
    .vrfReadDataPorts_3_bits_instructionIndex               (_storeUnit_vrfReadDataPorts_3_bits_instructionIndex),
    .vrfReadResults_0_valid                                 (vrfReadResults_0_valid & otherUnitTargetQueue_empty),
    .vrfReadResults_0_bits                                  (vrfReadResults_0_bits),
    .vrfReadResults_1_valid                                 (vrfReadResults_1_valid & otherUnitTargetQueue_empty),
    .vrfReadResults_1_bits                                  (vrfReadResults_1_bits),
    .vrfReadResults_2_valid                                 (vrfReadResults_2_valid & otherUnitTargetQueue_empty),
    .vrfReadResults_2_bits                                  (vrfReadResults_2_bits),
    .vrfReadResults_3_valid                                 (vrfReadResults_3_valid & otherUnitTargetQueue_empty),
    .vrfReadResults_3_bits                                  (vrfReadResults_3_bits),
    .storeResponse                                          (axi4Port_b_valid_0)
  );
  SimpleAccessUnit otherUnit (
    .clock                                                  (clock),
    .reset                                                  (reset),
    .lsuRequest_valid                                       (reqEnq_2),
    .lsuRequest_bits_instructionInformation_nf              (request_bits_instructionInformation_nf_0),
    .lsuRequest_bits_instructionInformation_mew             (request_bits_instructionInformation_mew_0),
    .lsuRequest_bits_instructionInformation_mop             (request_bits_instructionInformation_mop_0),
    .lsuRequest_bits_instructionInformation_lumop           (request_bits_instructionInformation_lumop_0),
    .lsuRequest_bits_instructionInformation_eew             (request_bits_instructionInformation_eew_0),
    .lsuRequest_bits_instructionInformation_vs3             (request_bits_instructionInformation_vs3_0),
    .lsuRequest_bits_instructionInformation_isStore         (request_bits_instructionInformation_isStore_0),
    .lsuRequest_bits_instructionInformation_maskedLoadStore (request_bits_instructionInformation_maskedLoadStore_0),
    .lsuRequest_bits_rs1Data                                (request_bits_rs1Data_0),
    .lsuRequest_bits_rs2Data                                (request_bits_rs2Data_0),
    .lsuRequest_bits_instructionIndex                       (request_bits_instructionIndex_0),
    .vrfReadDataPorts_ready                                 (otherUnit_vrfReadDataPorts_ready),
    .vrfReadDataPorts_valid                                 (_otherUnit_vrfReadDataPorts_valid),
    .vrfReadDataPorts_bits_vs                               (_otherUnit_vrfReadDataPorts_bits_vs),
    .vrfReadDataPorts_bits_instructionIndex                 (_otherUnit_vrfReadDataPorts_bits_instructionIndex),
    .vrfReadResults_valid                                   (otherUnitTargetQueue_deq_ready),
    .vrfReadResults_bits
      ((otherUnitTargetQueue_deq_bits[0] ? otherUnitDataQueueVec_0_deq_bits : 32'h0) | (otherUnitTargetQueue_deq_bits[1] ? otherUnitDataQueueVec_1_deq_bits : 32'h0)
       | (otherUnitTargetQueue_deq_bits[2] ? otherUnitDataQueueVec_2_deq_bits : 32'h0) | (otherUnitTargetQueue_deq_bits[3] ? otherUnitDataQueueVec_3_deq_bits : 32'h0)),
    .offsetReadResult_0_valid                               (offsetReadResult_0_valid),
    .offsetReadResult_0_bits                                (offsetReadResult_0_bits),
    .offsetReadResult_1_valid                               (offsetReadResult_1_valid),
    .offsetReadResult_1_bits                                (offsetReadResult_1_bits),
    .offsetReadResult_2_valid                               (offsetReadResult_2_valid),
    .offsetReadResult_2_bits                                (offsetReadResult_2_bits),
    .offsetReadResult_3_valid                               (offsetReadResult_3_valid),
    .offsetReadResult_3_bits                                (offsetReadResult_3_bits),
    .maskInput                                              (_GEN_3[maskSelect_2]),
    .maskSelect_valid                                       (_otherUnit_maskSelect_valid),
    .maskSelect_bits                                        (_otherUnit_maskSelect_bits),
    .memReadRequest_ready                                   (simpleSourceQueue_enq_ready & simpleAccessPorts_ar_ready_0),
    .memReadRequest_valid                                   (_otherUnit_memReadRequest_valid),
    .memReadRequest_bits_address                            (simpleAccessPorts_ar_bits_addr_0),
    .memReadRequest_bits_source                             (simpleSourceQueue_enq_bits),
    .memReadResponse_ready                                  (simpleAccessPorts_r_ready_0),
    .memReadResponse_valid                                  (simpleAccessPorts_r_valid_0),
    .memReadResponse_bits_data                              (simpleAccessPorts_r_bits_data_0),
    .memReadResponse_bits_source                            (simpleSourceQueue_deq_bits),
    .memWriteRequest_ready                                  (simpleAccessPorts_aw_ready_0 & simpleDataQueue_enq_ready),
    .memWriteRequest_valid                                  (_otherUnit_memWriteRequest_valid),
    .memWriteRequest_bits_data                              (simpleDataQueue_enq_bits_data),
    .memWriteRequest_bits_mask                              (simpleDataQueue_enq_bits_mask),
    .memWriteRequest_bits_source                            (_otherUnit_memWriteRequest_bits_source),
    .memWriteRequest_bits_address                           (_otherUnit_memWriteRequest_bits_address),
    .memWriteRequest_bits_size                              (_otherUnit_memWriteRequest_bits_size),
    .vrfWritePort_ready                                     (|(_otherUnit_status_targetLane & {otherUnit_vrfWritePort_ready_hi, otherUnit_vrfWritePort_ready_lo})),
    .vrfWritePort_valid                                     (_otherUnit_vrfWritePort_valid),
    .vrfWritePort_bits_vd                                   (_otherUnit_vrfWritePort_bits_vd),
    .vrfWritePort_bits_mask                                 (_otherUnit_vrfWritePort_bits_mask),
    .vrfWritePort_bits_data                                 (_otherUnit_vrfWritePort_bits_data),
    .vrfWritePort_bits_last                                 (_otherUnit_vrfWritePort_bits_last),
    .vrfWritePort_bits_instructionIndex                     (_otherUnit_vrfWritePort_bits_instructionIndex),
    .csrInterface_vl                                        (csrInterface_vl),
    .csrInterface_vStart                                    (csrInterface_vStart),
    .csrInterface_vlmul                                     (csrInterface_vlmul),
    .csrInterface_vSew                                      (csrInterface_vSew),
    .csrInterface_vxrm                                      (csrInterface_vxrm),
    .csrInterface_vta                                       (csrInterface_vta),
    .csrInterface_vma                                       (csrInterface_vma),
    .status_idle                                            (_otherUnit_status_idle),
    .status_last                                            (_otherUnit_status_last),
    .status_instructionIndex                                (_otherUnit_status_instructionIndex),
    .status_targetLane                                      (_otherUnit_status_targetLane),
    .status_isStore                                         (_otherUnit_status_isStore),
    .offsetRelease_0                                        (_otherUnit_offsetRelease_0),
    .offsetRelease_1                                        (_otherUnit_offsetRelease_1),
    .offsetRelease_2                                        (_otherUnit_offsetRelease_2),
    .offsetRelease_3                                        (_otherUnit_offsetRelease_3)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(96),
    .err_mode(2),
    .rst_mode(3),
    .width(49)
  ) writeQueueVec_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(_probeWire_slots_0_writeValid_T & ~(_writeQueueVec_fifo_empty & writeQueueVec_0_deq_ready))),
    .pop_req_n    (~(writeQueueVec_0_deq_ready & ~_writeQueueVec_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (writeQueueVec_dataIn),
    .empty        (_writeQueueVec_fifo_empty),
    .almost_empty (writeQueueVec_0_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (writeQueueVec_0_almostFull),
    .full         (_writeQueueVec_fifo_full),
    .error        (_writeQueueVec_fifo_error),
    .data_out     (_writeQueueVec_fifo_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(96),
    .err_mode(2),
    .rst_mode(3),
    .width(49)
  ) writeQueueVec_fifo_1 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(_probeWire_slots_1_writeValid_T & ~(_writeQueueVec_fifo_1_empty & writeQueueVec_1_deq_ready))),
    .pop_req_n    (~(writeQueueVec_1_deq_ready & ~_writeQueueVec_fifo_1_empty)),
    .diag_n       (1'h1),
    .data_in      (writeQueueVec_dataIn_1),
    .empty        (_writeQueueVec_fifo_1_empty),
    .almost_empty (writeQueueVec_1_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (writeQueueVec_1_almostFull),
    .full         (_writeQueueVec_fifo_1_full),
    .error        (_writeQueueVec_fifo_1_error),
    .data_out     (_writeQueueVec_fifo_1_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(96),
    .err_mode(2),
    .rst_mode(3),
    .width(49)
  ) writeQueueVec_fifo_2 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(_probeWire_slots_2_writeValid_T & ~(_writeQueueVec_fifo_2_empty & writeQueueVec_2_deq_ready))),
    .pop_req_n    (~(writeQueueVec_2_deq_ready & ~_writeQueueVec_fifo_2_empty)),
    .diag_n       (1'h1),
    .data_in      (writeQueueVec_dataIn_2),
    .empty        (_writeQueueVec_fifo_2_empty),
    .almost_empty (writeQueueVec_2_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (writeQueueVec_2_almostFull),
    .full         (_writeQueueVec_fifo_2_full),
    .error        (_writeQueueVec_fifo_2_error),
    .data_out     (_writeQueueVec_fifo_2_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(96),
    .err_mode(2),
    .rst_mode(3),
    .width(49)
  ) writeQueueVec_fifo_3 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(_probeWire_slots_3_writeValid_T & ~(_writeQueueVec_fifo_3_empty & writeQueueVec_3_deq_ready))),
    .pop_req_n    (~(writeQueueVec_3_deq_ready & ~_writeQueueVec_fifo_3_empty)),
    .diag_n       (1'h1),
    .data_in      (writeQueueVec_dataIn_3),
    .empty        (_writeQueueVec_fifo_3_empty),
    .almost_empty (writeQueueVec_3_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (writeQueueVec_3_almostFull),
    .full         (_writeQueueVec_fifo_3_full),
    .error        (_writeQueueVec_fifo_3_error),
    .data_out     (_writeQueueVec_fifo_3_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(8),
    .err_mode(2),
    .rst_mode(3),
    .width(4)
  ) otherUnitTargetQueue_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(otherUnitTargetQueue_enq_ready & otherUnitTargetQueue_enq_valid)),
    .pop_req_n    (~(otherUnitTargetQueue_deq_ready & ~_otherUnitTargetQueue_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (otherUnitTargetQueue_enq_bits),
    .empty        (_otherUnitTargetQueue_fifo_empty),
    .almost_empty (otherUnitTargetQueue_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (otherUnitTargetQueue_almostFull),
    .full         (_otherUnitTargetQueue_fifo_full),
    .error        (_otherUnitTargetQueue_fifo_error),
    .data_out     (otherUnitTargetQueue_deq_bits)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(32)
  ) otherUnitDataQueueVec_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(otherUnitDataQueueVec_0_enq_ready & otherUnitDataQueueVec_0_enq_valid & ~(_otherUnitDataQueueVec_fifo_empty & otherUnitDataQueueVec_0_deq_ready))),
    .pop_req_n    (~(otherUnitDataQueueVec_0_deq_ready & ~_otherUnitDataQueueVec_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (otherUnitDataQueueVec_0_enq_bits),
    .empty        (_otherUnitDataQueueVec_fifo_empty),
    .almost_empty (otherUnitDataQueueVec_0_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (otherUnitDataQueueVec_0_almostFull),
    .full         (_otherUnitDataQueueVec_fifo_full),
    .error        (_otherUnitDataQueueVec_fifo_error),
    .data_out     (_otherUnitDataQueueVec_fifo_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(32)
  ) otherUnitDataQueueVec_fifo_1 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(otherUnitDataQueueVec_1_enq_ready & otherUnitDataQueueVec_1_enq_valid & ~(_otherUnitDataQueueVec_fifo_1_empty & otherUnitDataQueueVec_1_deq_ready))),
    .pop_req_n    (~(otherUnitDataQueueVec_1_deq_ready & ~_otherUnitDataQueueVec_fifo_1_empty)),
    .diag_n       (1'h1),
    .data_in      (otherUnitDataQueueVec_1_enq_bits),
    .empty        (_otherUnitDataQueueVec_fifo_1_empty),
    .almost_empty (otherUnitDataQueueVec_1_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (otherUnitDataQueueVec_1_almostFull),
    .full         (_otherUnitDataQueueVec_fifo_1_full),
    .error        (_otherUnitDataQueueVec_fifo_1_error),
    .data_out     (_otherUnitDataQueueVec_fifo_1_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(32)
  ) otherUnitDataQueueVec_fifo_2 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(otherUnitDataQueueVec_2_enq_ready & otherUnitDataQueueVec_2_enq_valid & ~(_otherUnitDataQueueVec_fifo_2_empty & otherUnitDataQueueVec_2_deq_ready))),
    .pop_req_n    (~(otherUnitDataQueueVec_2_deq_ready & ~_otherUnitDataQueueVec_fifo_2_empty)),
    .diag_n       (1'h1),
    .data_in      (otherUnitDataQueueVec_2_enq_bits),
    .empty        (_otherUnitDataQueueVec_fifo_2_empty),
    .almost_empty (otherUnitDataQueueVec_2_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (otherUnitDataQueueVec_2_almostFull),
    .full         (_otherUnitDataQueueVec_fifo_2_full),
    .error        (_otherUnitDataQueueVec_fifo_2_error),
    .data_out     (_otherUnitDataQueueVec_fifo_2_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(32)
  ) otherUnitDataQueueVec_fifo_3 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(otherUnitDataQueueVec_3_enq_ready & otherUnitDataQueueVec_3_enq_valid & ~(_otherUnitDataQueueVec_fifo_3_empty & otherUnitDataQueueVec_3_deq_ready))),
    .pop_req_n    (~(otherUnitDataQueueVec_3_deq_ready & ~_otherUnitDataQueueVec_fifo_3_empty)),
    .diag_n       (1'h1),
    .data_in      (otherUnitDataQueueVec_3_enq_bits),
    .empty        (_otherUnitDataQueueVec_fifo_3_empty),
    .almost_empty (otherUnitDataQueueVec_3_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (otherUnitDataQueueVec_3_almostFull),
    .full         (_otherUnitDataQueueVec_fifo_3_full),
    .error        (_otherUnitDataQueueVec_fifo_3_error),
    .data_out     (_otherUnitDataQueueVec_fifo_3_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(8),
    .err_mode(2),
    .rst_mode(3),
    .width(3)
  ) writeIndexQueue_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(writeIndexQueue_enq_ready & writeIndexQueue_enq_valid)),
    .pop_req_n    (~(writeIndexQueue_deq_ready & ~_writeIndexQueue_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (writeIndexQueue_enq_bits),
    .empty        (_writeIndexQueue_fifo_empty),
    .almost_empty (writeIndexQueue_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (writeIndexQueue_almostFull),
    .full         (_writeIndexQueue_fifo_full),
    .error        (_writeIndexQueue_fifo_error),
    .data_out     (writeIndexQueue_deq_bits)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(8),
    .err_mode(2),
    .rst_mode(3),
    .width(3)
  ) writeIndexQueue_fifo_1 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(writeIndexQueue_1_enq_ready & writeIndexQueue_1_enq_valid)),
    .pop_req_n    (~(writeIndexQueue_1_deq_ready & ~_writeIndexQueue_fifo_1_empty)),
    .diag_n       (1'h1),
    .data_in      (writeIndexQueue_1_enq_bits),
    .empty        (_writeIndexQueue_fifo_1_empty),
    .almost_empty (writeIndexQueue_1_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (writeIndexQueue_1_almostFull),
    .full         (_writeIndexQueue_fifo_1_full),
    .error        (_writeIndexQueue_fifo_1_error),
    .data_out     (writeIndexQueue_1_deq_bits)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(8),
    .err_mode(2),
    .rst_mode(3),
    .width(3)
  ) writeIndexQueue_fifo_2 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(writeIndexQueue_2_enq_ready & writeIndexQueue_2_enq_valid)),
    .pop_req_n    (~(writeIndexQueue_2_deq_ready & ~_writeIndexQueue_fifo_2_empty)),
    .diag_n       (1'h1),
    .data_in      (writeIndexQueue_2_enq_bits),
    .empty        (_writeIndexQueue_fifo_2_empty),
    .almost_empty (writeIndexQueue_2_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (writeIndexQueue_2_almostFull),
    .full         (_writeIndexQueue_fifo_2_full),
    .error        (_writeIndexQueue_fifo_2_error),
    .data_out     (writeIndexQueue_2_deq_bits)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(8),
    .err_mode(2),
    .rst_mode(3),
    .width(3)
  ) writeIndexQueue_fifo_3 (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(writeIndexQueue_3_enq_ready & writeIndexQueue_3_enq_valid)),
    .pop_req_n    (~(writeIndexQueue_3_deq_ready & ~_writeIndexQueue_fifo_3_empty)),
    .diag_n       (1'h1),
    .data_in      (writeIndexQueue_3_enq_bits),
    .empty        (_writeIndexQueue_fifo_3_empty),
    .almost_empty (writeIndexQueue_3_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (writeIndexQueue_3_almostFull),
    .full         (_writeIndexQueue_fifo_3_full),
    .error        (_writeIndexQueue_fifo_3_error),
    .data_out     (writeIndexQueue_3_deq_bits)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(8),
    .err_mode(2),
    .rst_mode(3),
    .width(4)
  ) sourceQueue_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(sourceQueue_enq_ready & sourceQueue_enq_valid)),
    .pop_req_n    (~(sourceQueue_deq_ready & ~_sourceQueue_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (sourceQueue_enq_bits),
    .empty        (_sourceQueue_fifo_empty),
    .almost_empty (sourceQueue_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (sourceQueue_almostFull),
    .full         (_sourceQueue_fifo_full),
    .error        (_sourceQueue_fifo_error),
    .data_out     (sourceQueue_deq_bits)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(2),
    .err_mode(2),
    .rst_mode(3),
    .width(180)
  ) dataQueue_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(dataQueue_enq_ready & dataQueue_enq_valid)),
    .pop_req_n    (~(dataQueue_deq_ready & ~_dataQueue_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (dataQueue_dataIn),
    .empty        (_dataQueue_fifo_empty),
    .almost_empty (dataQueue_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (dataQueue_almostFull),
    .full         (_dataQueue_fifo_full),
    .error        (_dataQueue_fifo_error),
    .data_out     (_dataQueue_fifo_data_out)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(8),
    .err_mode(2),
    .rst_mode(3),
    .width(7)
  ) simpleSourceQueue_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(simpleSourceQueue_enq_ready & simpleSourceQueue_enq_valid)),
    .pop_req_n    (~(simpleSourceQueue_deq_ready & ~_simpleSourceQueue_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (simpleSourceQueue_enq_bits),
    .empty        (_simpleSourceQueue_fifo_empty),
    .almost_empty (simpleSourceQueue_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (simpleSourceQueue_almostFull),
    .full         (_simpleSourceQueue_fifo_full),
    .error        (_simpleSourceQueue_fifo_error),
    .data_out     (simpleSourceQueue_deq_bits)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(2),
    .err_mode(2),
    .rst_mode(3),
    .width(78)
  ) simpleDataQueue_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(simpleDataQueue_enq_ready & simpleDataQueue_enq_valid)),
    .pop_req_n    (~(simpleDataQueue_deq_ready & ~_simpleDataQueue_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (simpleDataQueue_dataIn),
    .empty        (_simpleDataQueue_fifo_empty),
    .almost_empty (simpleDataQueue_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (simpleDataQueue_almostFull),
    .full         (_simpleDataQueue_fifo_full),
    .error        (_simpleDataQueue_fifo_error),
    .data_out     (_simpleDataQueue_fifo_data_out)
  );
  assign request_ready = request_ready_0;
  assign axi4Port_aw_valid = axi4Port_aw_valid_0;
  assign axi4Port_aw_bits_id = axi4Port_aw_bits_id_0;
  assign axi4Port_aw_bits_addr = axi4Port_aw_bits_addr_0;
  assign axi4Port_w_valid = axi4Port_w_valid_0;
  assign axi4Port_w_bits_data = axi4Port_w_bits_data_0;
  assign axi4Port_w_bits_strb = axi4Port_w_bits_strb_0;
  assign axi4Port_ar_valid = axi4Port_ar_valid_0;
  assign axi4Port_ar_bits_addr = axi4Port_ar_bits_addr_0;
  assign axi4Port_r_ready = axi4Port_r_ready_0;
  assign simpleAccessPorts_aw_valid = simpleAccessPorts_aw_valid_0;
  assign simpleAccessPorts_aw_bits_id = simpleAccessPorts_aw_bits_id_0;
  assign simpleAccessPorts_aw_bits_addr = simpleAccessPorts_aw_bits_addr_0;
  assign simpleAccessPorts_aw_bits_size = simpleAccessPorts_aw_bits_size_0;
  assign simpleAccessPorts_w_valid = simpleAccessPorts_w_valid_0;
  assign simpleAccessPorts_w_bits_data = simpleAccessPorts_w_bits_data_0;
  assign simpleAccessPorts_w_bits_strb = simpleAccessPorts_w_bits_strb_0;
  assign simpleAccessPorts_ar_valid = simpleAccessPorts_ar_valid_0;
  assign simpleAccessPorts_ar_bits_addr = simpleAccessPorts_ar_bits_addr_0;
  assign simpleAccessPorts_r_ready = simpleAccessPorts_r_ready_0;
  assign vrfReadDataPorts_0_valid = vrfReadDataPorts_0_valid_0;
  assign vrfReadDataPorts_0_bits_vs = vrfReadDataPorts_0_bits_vs_0;
  assign vrfReadDataPorts_0_bits_instructionIndex = vrfReadDataPorts_0_bits_instructionIndex_0;
  assign vrfReadDataPorts_1_valid = vrfReadDataPorts_1_valid_0;
  assign vrfReadDataPorts_1_bits_vs = vrfReadDataPorts_1_bits_vs_0;
  assign vrfReadDataPorts_1_bits_instructionIndex = vrfReadDataPorts_1_bits_instructionIndex_0;
  assign vrfReadDataPorts_2_valid = vrfReadDataPorts_2_valid_0;
  assign vrfReadDataPorts_2_bits_vs = vrfReadDataPorts_2_bits_vs_0;
  assign vrfReadDataPorts_2_bits_instructionIndex = vrfReadDataPorts_2_bits_instructionIndex_0;
  assign vrfReadDataPorts_3_valid = vrfReadDataPorts_3_valid_0;
  assign vrfReadDataPorts_3_bits_vs = vrfReadDataPorts_3_bits_vs_0;
  assign vrfReadDataPorts_3_bits_instructionIndex = vrfReadDataPorts_3_bits_instructionIndex_0;
  assign vrfWritePort_0_valid = vrfWritePort_0_valid_0;
  assign vrfWritePort_0_bits_vd = vrfWritePort_0_bits_vd_0;
  assign vrfWritePort_0_bits_mask = vrfWritePort_0_bits_mask_0;
  assign vrfWritePort_0_bits_data = vrfWritePort_0_bits_data_0;
  assign vrfWritePort_0_bits_last = vrfWritePort_0_bits_last_0;
  assign vrfWritePort_0_bits_instructionIndex = vrfWritePort_0_bits_instructionIndex_0;
  assign vrfWritePort_1_valid = vrfWritePort_1_valid_0;
  assign vrfWritePort_1_bits_vd = vrfWritePort_1_bits_vd_0;
  assign vrfWritePort_1_bits_mask = vrfWritePort_1_bits_mask_0;
  assign vrfWritePort_1_bits_data = vrfWritePort_1_bits_data_0;
  assign vrfWritePort_1_bits_last = vrfWritePort_1_bits_last_0;
  assign vrfWritePort_1_bits_instructionIndex = vrfWritePort_1_bits_instructionIndex_0;
  assign vrfWritePort_2_valid = vrfWritePort_2_valid_0;
  assign vrfWritePort_2_bits_vd = vrfWritePort_2_bits_vd_0;
  assign vrfWritePort_2_bits_mask = vrfWritePort_2_bits_mask_0;
  assign vrfWritePort_2_bits_data = vrfWritePort_2_bits_data_0;
  assign vrfWritePort_2_bits_last = vrfWritePort_2_bits_last_0;
  assign vrfWritePort_2_bits_instructionIndex = vrfWritePort_2_bits_instructionIndex_0;
  assign vrfWritePort_3_valid = vrfWritePort_3_valid_0;
  assign vrfWritePort_3_bits_vd = vrfWritePort_3_bits_vd_0;
  assign vrfWritePort_3_bits_mask = vrfWritePort_3_bits_mask_0;
  assign vrfWritePort_3_bits_data = vrfWritePort_3_bits_data_0;
  assign vrfWritePort_3_bits_last = vrfWritePort_3_bits_last_0;
  assign vrfWritePort_3_bits_instructionIndex = vrfWritePort_3_bits_instructionIndex_0;
  assign dataInWriteQueue_0 = {dataInWriteQueue_0_hi, dataInWriteQueue_0_lo} | dataInMSHR;
  assign dataInWriteQueue_1 = {dataInWriteQueue_1_hi, dataInWriteQueue_1_lo} | dataInMSHR;
  assign dataInWriteQueue_2 = {dataInWriteQueue_2_hi, dataInWriteQueue_2_lo} | dataInMSHR;
  assign dataInWriteQueue_3 = {dataInWriteQueue_3_hi, dataInWriteQueue_3_lo} | dataInMSHR;
  assign lastReport = (_loadUnit_status_last ? 8'h1 << _GEN_4 : 8'h0) | (_storeUnit_status_last ? 8'h1 << _storeUnit_status_instructionIndex : 8'h0) | (_otherUnit_status_last ? 8'h1 << _GEN_5 : 8'h0);
  assign tokenIO_offsetGroupRelease = {tokenIO_offsetGroupRelease_hi, tokenIO_offsetGroupRelease_lo};
endmodule

