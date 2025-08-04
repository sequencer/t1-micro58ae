
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
module VrfReadPipe_1(
  input         clock,
                reset,
  output        enqueue_ready,
  input         enqueue_valid,
  input  [4:0]  enqueue_bits_vs,
  input  [3:0]  enqueue_bits_groupIndex,
                enqueue_bits_readSource,
  input  [2:0]  enqueue_bits_instructionIndex,
  output        contender_ready,
  input         contender_valid,
  input  [4:0]  contender_bits_vs,
  input  [3:0]  contender_bits_groupIndex,
                contender_bits_readSource,
  input  [2:0]  contender_bits_instructionIndex,
  input         vrfReadRequest_ready,
  output        vrfReadRequest_valid,
  output [4:0]  vrfReadRequest_bits_vs,
  output [1:0]  vrfReadRequest_bits_readSource,
  output [2:0]  vrfReadRequest_bits_instructionIndex,
  input  [31:0] vrfReadResult,
  input         dequeue_ready,
  output        dequeue_valid,
  output [31:0] dequeue_bits,
  input         contenderDequeue_ready,
  output        contenderDequeue_valid,
  output [31:0] contenderDequeue_bits
);

  wire        _contenderDataQueue_fifo_empty;
  wire        _contenderDataQueue_fifo_full;
  wire        _contenderDataQueue_fifo_error;
  wire        _dataQueue_fifo_empty;
  wire        _dataQueue_fifo_full;
  wire        _dataQueue_fifo_error;
  wire        _reqArbitrate_io_in_0_ready;
  wire        _reqArbitrate_io_in_1_ready;
  wire [3:0]  _reqArbitrate_io_out_bits_readSource;
  wire        contenderDataQueue_almostFull;
  wire        contenderDataQueue_almostEmpty;
  wire        dataQueue_almostFull;
  wire        dataQueue_almostEmpty;
  wire [2:0]  vrfReadRequest_bits_instructionIndex_0;
  wire [4:0]  vrfReadRequest_bits_vs_0;
  wire        enqueue_valid_0 = enqueue_valid;
  wire [4:0]  enqueue_bits_vs_0 = enqueue_bits_vs;
  wire [3:0]  enqueue_bits_groupIndex_0 = enqueue_bits_groupIndex;
  wire [3:0]  enqueue_bits_readSource_0 = enqueue_bits_readSource;
  wire [2:0]  enqueue_bits_instructionIndex_0 = enqueue_bits_instructionIndex;
  wire        contender_valid_0 = contender_valid;
  wire [4:0]  contender_bits_vs_0 = contender_bits_vs;
  wire [3:0]  contender_bits_groupIndex_0 = contender_bits_groupIndex;
  wire [3:0]  contender_bits_readSource_0 = contender_bits_readSource;
  wire [2:0]  contender_bits_instructionIndex_0 = contender_bits_instructionIndex;
  wire        vrfReadRequest_ready_0 = vrfReadRequest_ready;
  wire        dequeue_ready_0 = dequeue_ready;
  wire        contenderDequeue_ready_0 = contenderDequeue_ready;
  wire [31:0] dataQueue_enq_bits = vrfReadResult;
  wire [31:0] contenderDataQueue_enq_bits = vrfReadResult;
  wire        dataQueue_deq_ready = dequeue_ready_0;
  wire        dataQueue_deq_valid;
  wire [31:0] dataQueue_deq_bits;
  wire        contenderDataQueue_deq_ready = contenderDequeue_ready_0;
  wire        contenderDataQueue_deq_valid;
  wire [31:0] contenderDataQueue_deq_bits;
  wire        enqueue_ready_0 = _reqArbitrate_io_in_0_ready & dequeue_ready_0;
  wire        contender_ready_0 = _reqArbitrate_io_in_1_ready & contenderDequeue_ready_0;
  wire [1:0]  vrfReadRequest_bits_readSource_0 = _reqArbitrate_io_out_bits_readSource[1:0];
  assign dataQueue_deq_valid = ~_dataQueue_fifo_empty;
  wire        dequeue_valid_0 = dataQueue_deq_valid;
  wire [31:0] dequeue_bits_0 = dataQueue_deq_bits;
  wire        dataQueue_enq_ready = ~_dataQueue_fifo_full;
  wire        dataQueue_enq_valid;
  assign contenderDataQueue_deq_valid = ~_contenderDataQueue_fifo_empty;
  wire        contenderDequeue_valid_0 = contenderDataQueue_deq_valid;
  wire [31:0] contenderDequeue_bits_0 = contenderDataQueue_deq_bits;
  wire        contenderDataQueue_enq_ready = ~_contenderDataQueue_fifo_full;
  wire        contenderDataQueue_enq_valid;
  reg         enqFirePipe_pipe_v;
  reg         enqFirePipe_pipe_b;
  reg         enqFirePipe_pipe_pipe_v;
  wire        enqFirePipe_valid = enqFirePipe_pipe_pipe_v;
  reg         enqFirePipe_pipe_pipe_b;
  wire        enqFirePipe_bits = enqFirePipe_pipe_pipe_b;
  assign dataQueue_enq_valid = enqFirePipe_valid & enqFirePipe_bits;
  wire        vrfReadRequest_valid_0;
  assign contenderDataQueue_enq_valid = enqFirePipe_valid & ~enqFirePipe_bits;
  always @(posedge clock) begin
    automatic logic _enqFirePipe_T;
    _enqFirePipe_T = vrfReadRequest_ready_0 & vrfReadRequest_valid_0;
    if (reset) begin
      enqFirePipe_pipe_v <= 1'h0;
      enqFirePipe_pipe_pipe_v <= 1'h0;
    end
    else begin
      enqFirePipe_pipe_v <= _enqFirePipe_T;
      enqFirePipe_pipe_pipe_v <= enqFirePipe_pipe_v;
    end
    if (_enqFirePipe_T)
      enqFirePipe_pipe_b <= enqueue_ready_0 & enqueue_valid_0;
    if (enqFirePipe_pipe_v)
      enqFirePipe_pipe_pipe_b <= enqFirePipe_pipe_b;
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
        enqFirePipe_pipe_v = _RANDOM[/*Zero width*/ 1'b0][0];
        enqFirePipe_pipe_b = _RANDOM[/*Zero width*/ 1'b0][1];
        enqFirePipe_pipe_pipe_v = _RANDOM[/*Zero width*/ 1'b0][2];
        enqFirePipe_pipe_pipe_b = _RANDOM[/*Zero width*/ 1'b0][3];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  wire        dataQueue_empty;
  assign dataQueue_empty = _dataQueue_fifo_empty;
  wire        dataQueue_full;
  assign dataQueue_full = _dataQueue_fifo_full;
  wire        contenderDataQueue_empty;
  assign contenderDataQueue_empty = _contenderDataQueue_fifo_empty;
  wire        contenderDataQueue_full;
  assign contenderDataQueue_full = _contenderDataQueue_fifo_full;
  ReadStageRRArbiter_1 reqArbitrate (
    .clock                         (clock),
    .reset                         (reset),
    .io_in_0_ready                 (_reqArbitrate_io_in_0_ready),
    .io_in_0_valid                 (enqueue_valid_0 & dequeue_ready_0),
    .io_in_0_bits_vs               (enqueue_bits_vs_0),
    .io_in_0_bits_groupIndex       (enqueue_bits_groupIndex_0),
    .io_in_0_bits_readSource       (enqueue_bits_readSource_0),
    .io_in_0_bits_instructionIndex (enqueue_bits_instructionIndex_0),
    .io_in_1_ready                 (_reqArbitrate_io_in_1_ready),
    .io_in_1_valid                 (contender_valid_0 & contenderDequeue_ready_0),
    .io_in_1_bits_vs               (contender_bits_vs_0),
    .io_in_1_bits_groupIndex       (contender_bits_groupIndex_0),
    .io_in_1_bits_readSource       (contender_bits_readSource_0),
    .io_in_1_bits_instructionIndex (contender_bits_instructionIndex_0),
    .io_out_ready                  (vrfReadRequest_ready_0),
    .io_out_valid                  (vrfReadRequest_valid_0),
    .io_out_bits_vs                (vrfReadRequest_bits_vs_0),
    .io_out_bits_readSource        (_reqArbitrate_io_out_bits_readSource),
    .io_out_bits_instructionIndex  (vrfReadRequest_bits_instructionIndex_0)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(32)
  ) dataQueue_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(dataQueue_enq_ready & dataQueue_enq_valid)),
    .pop_req_n    (~(dataQueue_deq_ready & ~_dataQueue_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (dataQueue_enq_bits),
    .empty        (_dataQueue_fifo_empty),
    .almost_empty (dataQueue_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (dataQueue_almostFull),
    .full         (_dataQueue_fifo_full),
    .error        (_dataQueue_fifo_error),
    .data_out     (dataQueue_deq_bits)
  );
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(32)
  ) contenderDataQueue_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(contenderDataQueue_enq_ready & contenderDataQueue_enq_valid)),
    .pop_req_n    (~(contenderDataQueue_deq_ready & ~_contenderDataQueue_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (contenderDataQueue_enq_bits),
    .empty        (_contenderDataQueue_fifo_empty),
    .almost_empty (contenderDataQueue_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (contenderDataQueue_almostFull),
    .full         (_contenderDataQueue_fifo_full),
    .error        (_contenderDataQueue_fifo_error),
    .data_out     (contenderDataQueue_deq_bits)
  );
  assign enqueue_ready = enqueue_ready_0;
  assign contender_ready = contender_ready_0;
  assign vrfReadRequest_valid = vrfReadRequest_valid_0;
  assign vrfReadRequest_bits_vs = vrfReadRequest_bits_vs_0;
  assign vrfReadRequest_bits_readSource = vrfReadRequest_bits_readSource_0;
  assign vrfReadRequest_bits_instructionIndex = vrfReadRequest_bits_instructionIndex_0;
  assign dequeue_valid = dequeue_valid_0;
  assign dequeue_bits = dequeue_bits_0;
  assign contenderDequeue_valid = contenderDequeue_valid_0;
  assign contenderDequeue_bits = contenderDequeue_bits_0;
endmodule

