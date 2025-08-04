
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
module MaskedWrite(
  input         clock,
                reset,
  output        enqueue_ready,
  input         enqueue_valid,
  input  [4:0]  enqueue_bits_vd,
                enqueue_bits_offset,
  input  [3:0]  enqueue_bits_mask,
  input  [31:0] enqueue_bits_data,
  input         enqueue_bits_last,
  input  [2:0]  enqueue_bits_instructionIndex,
  input         dequeue_ready,
  output        dequeue_valid,
  output [4:0]  dequeue_bits_vd,
                dequeue_bits_offset,
  output [3:0]  dequeue_bits_mask,
  output [31:0] dequeue_bits_data,
  output        dequeue_bits_last,
  output [2:0]  dequeue_bits_instructionIndex,
  input         vrfReadRequest_ready,
  output        vrfReadRequest_valid,
  output [4:0]  vrfReadRequest_bits_vs,
                vrfReadRequest_bits_offset,
  output [2:0]  vrfReadRequest_bits_instructionIndex,
  input  [31:0] vrfReadResult
);

  wire        _vrfReadPipe_fifo_empty;
  wire        _vrfReadPipe_fifo_full;
  wire        _vrfReadPipe_fifo_error;
  wire        vrfReadPipe_almostFull;
  wire        vrfReadPipe_almostEmpty;
  wire [2:0]  dequeueWire_bits_instructionIndex;
  wire        dequeueWire_bits_last;
  wire [31:0] dequeueWire_bits_data;
  wire [3:0]  dequeueWire_bits_mask;
  wire [4:0]  dequeueWire_bits_offset;
  wire [4:0]  dequeueWire_bits_vd;
  wire        dequeueWire_valid;
  wire        enqueue_valid_0 = enqueue_valid;
  wire [4:0]  enqueue_bits_vd_0 = enqueue_bits_vd;
  wire [4:0]  enqueue_bits_offset_0 = enqueue_bits_offset;
  wire [3:0]  enqueue_bits_mask_0 = enqueue_bits_mask;
  wire [31:0] enqueue_bits_data_0 = enqueue_bits_data;
  wire        enqueue_bits_last_0 = enqueue_bits_last;
  wire [2:0]  enqueue_bits_instructionIndex_0 = enqueue_bits_instructionIndex;
  wire        dequeue_ready_0 = dequeue_ready;
  wire        vrfReadRequest_ready_0 = vrfReadRequest_ready;
  wire [31:0] vrfReadPipe_enq_bits = vrfReadResult;
  wire [1:0]  vrfReadRequest_bits_readSource = 2'h2;
  wire        s1EnqReady;
  wire [4:0]  vrfReadRequest_bits_vs_0 = enqueue_bits_vd_0;
  wire [4:0]  vrfReadRequest_bits_offset_0 = enqueue_bits_offset_0;
  wire [2:0]  vrfReadRequest_bits_instructionIndex_0 = enqueue_bits_instructionIndex_0;
  wire        dequeueQueue_deq_ready = dequeue_ready_0;
  wire        dequeueQueue_deq_valid;
  wire [4:0]  dequeueQueue_deq_bits_vd;
  wire [4:0]  dequeueQueue_deq_bits_offset;
  wire [3:0]  dequeueQueue_deq_bits_mask;
  wire [31:0] dequeueQueue_deq_bits_data;
  wire        dequeueQueue_deq_bits_last;
  wire [2:0]  dequeueQueue_deq_bits_instructionIndex;
  wire        readBeforeWrite;
  wire        dequeueQueue_enq_ready;
  wire        dequeueQueue_enq_valid = dequeueWire_valid;
  wire [4:0]  dequeueQueue_enq_bits_vd = dequeueWire_bits_vd;
  wire [4:0]  dequeueQueue_enq_bits_offset = dequeueWire_bits_offset;
  wire [3:0]  dequeueQueue_enq_bits_mask = dequeueWire_bits_mask;
  wire [31:0] dequeueQueue_enq_bits_data = dequeueWire_bits_data;
  wire        dequeueQueue_enq_bits_last = dequeueWire_bits_last;
  wire [2:0]  dequeueQueue_enq_bits_instructionIndex = dequeueWire_bits_instructionIndex;
  wire        dequeueWire_ready = dequeueQueue_enq_ready;
  wire        dequeue_valid_0 = dequeueQueue_deq_valid;
  wire [4:0]  dequeue_bits_vd_0 = dequeueQueue_deq_bits_vd;
  wire [4:0]  dequeue_bits_offset_0 = dequeueQueue_deq_bits_offset;
  wire [3:0]  dequeue_bits_mask_0 = dequeueQueue_deq_bits_mask;
  wire [31:0] dequeue_bits_data_0 = dequeueQueue_deq_bits_data;
  wire        dequeue_bits_last_0 = dequeueQueue_deq_bits_last;
  wire [2:0]  dequeue_bits_instructionIndex_0 = dequeueQueue_deq_bits_instructionIndex;
  wire        dequeueQueue_full;
  reg  [4:0]  dequeueQueue_data_vd;
  reg  [4:0]  dequeueQueue_data_offset;
  reg  [3:0]  dequeueQueue_data_mask;
  reg  [31:0] dequeueQueue_data_data;
  reg         dequeueQueue_data_last;
  reg  [2:0]  dequeueQueue_data_instructionIndex;
  reg         dequeueQueue_empty;
  assign dequeueQueue_enq_ready = dequeueQueue_empty;
  wire        dequeueQueue_empty_0 = dequeueQueue_empty;
  assign dequeueQueue_full = ~dequeueQueue_empty;
  wire        dequeueQueue_full_0 = dequeueQueue_full;
  wire        dequeueQueue_push = dequeueQueue_enq_ready & dequeueQueue_enq_valid & ~(dequeueQueue_empty & dequeueQueue_deq_ready);
  wire        dequeueQueue_pop = dequeueQueue_deq_ready & dequeueQueue_full;
  assign dequeueQueue_deq_valid = dequeueQueue_full | dequeueQueue_enq_valid;
  assign dequeueQueue_deq_bits_vd = dequeueQueue_empty ? dequeueQueue_enq_bits_vd : dequeueQueue_data_vd;
  assign dequeueQueue_deq_bits_offset = dequeueQueue_empty ? dequeueQueue_enq_bits_offset : dequeueQueue_data_offset;
  assign dequeueQueue_deq_bits_mask = dequeueQueue_empty ? dequeueQueue_enq_bits_mask : dequeueQueue_data_mask;
  assign dequeueQueue_deq_bits_data = dequeueQueue_empty ? dequeueQueue_enq_bits_data : dequeueQueue_data_data;
  assign dequeueQueue_deq_bits_last = dequeueQueue_empty ? dequeueQueue_enq_bits_last : dequeueQueue_data_last;
  assign dequeueQueue_deq_bits_instructionIndex = dequeueQueue_empty ? dequeueQueue_enq_bits_instructionIndex : dequeueQueue_data_instructionIndex;
  reg         s3Valid;
  assign dequeueWire_valid = s3Valid;
  reg  [4:0]  s3Pipe_vd;
  assign dequeueWire_bits_vd = s3Pipe_vd;
  reg  [4:0]  s3Pipe_offset;
  assign dequeueWire_bits_offset = s3Pipe_offset;
  reg  [3:0]  s3Pipe_mask;
  assign dequeueWire_bits_mask = s3Pipe_mask;
  reg  [31:0] s3Pipe_data;
  reg         s3Pipe_last;
  assign dequeueWire_bits_last = s3Pipe_last;
  reg  [2:0]  s3Pipe_instructionIndex;
  assign dequeueWire_bits_instructionIndex = s3Pipe_instructionIndex;
  reg  [31:0] s3BypassData;
  wire [7:0]  dataInS3 = s3Valid ? 8'h1 << s3Pipe_instructionIndex : 8'h0;
  reg         fwd3;
  reg         s2Valid;
  reg  [4:0]  s2Pipe_vd;
  reg  [4:0]  s2Pipe_offset;
  reg  [3:0]  s2Pipe_mask;
  reg  [31:0] s2Pipe_data;
  reg         s2Pipe_last;
  reg  [2:0]  s2Pipe_instructionIndex;
  reg  [31:0] s2BypassData;
  reg         s2EnqHitS1;
  wire [7:0]  dataInS2 = s2Valid ? 8'h1 << s2Pipe_instructionIndex : 8'h0;
  reg         fwd2;
  reg         s1Valid;
  reg  [4:0]  s1Pipe_vd;
  reg  [4:0]  s1Pipe_offset;
  reg  [3:0]  s1Pipe_mask;
  reg  [31:0] s1Pipe_data;
  reg         s1Pipe_last;
  reg  [2:0]  s1Pipe_instructionIndex;
  reg  [31:0] s1BypassData;
  reg         s1EnqHitS1;
  reg         s1EnqHitS2;
  wire [7:0]  dataInS1 = s1Valid ? 8'h1 << s1Pipe_instructionIndex : 8'h0;
  reg         fwd1;
  wire        s3EnqReady = dequeueWire_ready | ~s3Valid;
  wire        s3Fire = s3EnqReady & s2Valid;
  wire        s2EnqReady = s3EnqReady | ~s2Valid;
  wire        s2Fire = s2EnqReady & s1Valid;
  wire        enqueue_ready_0 = s1EnqReady;
  wire        s1Fire = enqueue_ready_0 & enqueue_valid_0;
  wire [9:0]  _hitQueue_T_1 = {enqueue_bits_vd_0, enqueue_bits_offset_0};
  wire        enqHitS1 = s1Valid & _hitQueue_T_1 == {s1Pipe_vd, s1Pipe_offset};
  wire        enqHitS2 = s2Valid & _hitQueue_T_1 == {s2Pipe_vd, s2Pipe_offset};
  wire        enqHitS3 = s3Valid & _hitQueue_T_1 == {s3Pipe_vd, s3Pipe_offset};
  wire        hitQueue = ~dequeueQueue_empty_0 & _hitQueue_T_1 == {dequeueQueue_deq_bits_vd, dequeueQueue_deq_bits_offset};
  wire        fwd = enqHitS1 | enqHitS2 | enqHitS3;
  assign s1EnqReady = (s2EnqReady | ~s1Valid) & ~hitQueue;
  wire [7:0]  dataInQueue = dequeueQueue_empty_0 ? 8'h0 : 8'h1 << dequeueQueue_deq_bits_instructionIndex;
  wire        enqNeedRead = enqueue_bits_mask_0 != 4'hF & ~fwd;
  assign readBeforeWrite = s1Fire & enqNeedRead;
  wire        vrfReadRequest_valid_0 = readBeforeWrite;
  wire        readDataValid_pipe_pipe_out_valid;
  wire        vrfReadPipe_deq_valid;
  assign vrfReadPipe_deq_valid = ~_vrfReadPipe_fifo_empty;
  wire        vrfReadPipe_enq_ready = ~_vrfReadPipe_fifo_full;
  wire        vrfReadPipe_enq_valid;
  wire        vrfReadPipe_deq_ready;
  reg         readDataValid_pipe_v;
  reg         readDataValid_pipe_pipe_v;
  assign readDataValid_pipe_pipe_out_valid = readDataValid_pipe_pipe_v;
  reg         readDataValid_pipe_pipe_b;
  wire        readDataValid_pipe_pipe_out_bits = readDataValid_pipe_pipe_b;
  assign vrfReadPipe_enq_valid = readDataValid_pipe_pipe_out_valid;
  wire [7:0]  maskedWrite1H = dataInS3 | dataInS2 | dataInS1 | dataInQueue;
  wire [15:0] maskFill_lo = {{8{s3Pipe_mask[1]}}, {8{s3Pipe_mask[0]}}};
  wire [15:0] maskFill_hi = {{8{s3Pipe_mask[3]}}, {8{s3Pipe_mask[2]}}};
  wire [31:0] maskFill = {maskFill_hi, maskFill_lo};
  wire [31:0] vrfReadPipe_deq_bits;
  wire [31:0] readDataSelect = fwd3 ? s3BypassData : vrfReadPipe_deq_bits;
  wire        s3ReadFromVrf = s3Pipe_mask != 4'hF & ~fwd3;
  assign dequeueWire_bits_data = s3Pipe_data & maskFill | readDataSelect & ~maskFill;
  wire        _vrfReadPipe_deq_ready_T = dequeueWire_ready & dequeueWire_valid;
  assign vrfReadPipe_deq_ready = _vrfReadPipe_deq_ready_T & s3ReadFromVrf;
  always @(posedge clock) begin
    if (dequeueQueue_push) begin
      dequeueQueue_data_vd <= dequeueQueue_enq_bits_vd;
      dequeueQueue_data_offset <= dequeueQueue_enq_bits_offset;
      dequeueQueue_data_mask <= dequeueQueue_enq_bits_mask;
      dequeueQueue_data_data <= dequeueQueue_enq_bits_data;
      dequeueQueue_data_last <= dequeueQueue_enq_bits_last;
      dequeueQueue_data_instructionIndex <= dequeueQueue_enq_bits_instructionIndex;
    end
    readDataValid_pipe_pipe_b <= 1'h0;
    if (reset) begin
      dequeueQueue_empty <= 1'h1;
      s3Valid <= 1'h0;
      s3Pipe_vd <= 5'h0;
      s3Pipe_offset <= 5'h0;
      s3Pipe_mask <= 4'h0;
      s3Pipe_data <= 32'h0;
      s3Pipe_last <= 1'h0;
      s3Pipe_instructionIndex <= 3'h0;
      s3BypassData <= 32'h0;
      fwd3 <= 1'h0;
      s2Valid <= 1'h0;
      s2Pipe_vd <= 5'h0;
      s2Pipe_offset <= 5'h0;
      s2Pipe_mask <= 4'h0;
      s2Pipe_data <= 32'h0;
      s2Pipe_last <= 1'h0;
      s2Pipe_instructionIndex <= 3'h0;
      s2BypassData <= 32'h0;
      s2EnqHitS1 <= 1'h0;
      fwd2 <= 1'h0;
      s1Valid <= 1'h0;
      s1Pipe_vd <= 5'h0;
      s1Pipe_offset <= 5'h0;
      s1Pipe_mask <= 4'h0;
      s1Pipe_data <= 32'h0;
      s1Pipe_last <= 1'h0;
      s1Pipe_instructionIndex <= 3'h0;
      s1BypassData <= 32'h0;
      s1EnqHitS1 <= 1'h0;
      s1EnqHitS2 <= 1'h0;
      fwd1 <= 1'h0;
      readDataValid_pipe_v <= 1'h0;
      readDataValid_pipe_pipe_v <= 1'h0;
    end
    else begin
      if (~(dequeueQueue_push == dequeueQueue_pop))
        dequeueQueue_empty <= dequeueQueue_pop;
      if (s3Fire ^ _vrfReadPipe_deq_ready_T)
        s3Valid <= s3Fire;
      if (s3Fire) begin
        s3Pipe_vd <= s2Pipe_vd;
        s3Pipe_offset <= s2Pipe_offset;
        s3Pipe_mask <= s2Pipe_mask;
        s3Pipe_data <= s2Pipe_data;
        s3Pipe_last <= s2Pipe_last;
        s3Pipe_instructionIndex <= s2Pipe_instructionIndex;
        s3BypassData <= s2EnqHitS1 ? dequeueWire_bits_data : s2BypassData;
        fwd3 <= fwd2;
      end
      if (s2Fire ^ s3Fire)
        s2Valid <= s2Fire;
      if (s2Fire) begin
        s2Pipe_vd <= s1Pipe_vd;
        s2Pipe_offset <= s1Pipe_offset;
        s2Pipe_mask <= s1Pipe_mask;
        s2Pipe_data <= s1Pipe_data;
        s2Pipe_last <= s1Pipe_last;
        s2Pipe_instructionIndex <= s1Pipe_instructionIndex;
        s2BypassData <= s1EnqHitS2 ? dequeueWire_bits_data : s1BypassData;
        s2EnqHitS1 <= s1EnqHitS1;
        fwd2 <= fwd1;
      end
      if (s1Fire ^ s2Fire)
        s1Valid <= s1Fire;
      if (s1Fire) begin
        s1Pipe_vd <= enqueue_bits_vd_0;
        s1Pipe_offset <= enqueue_bits_offset_0;
        s1Pipe_mask <= enqueue_bits_mask_0;
        s1Pipe_data <= enqueue_bits_data_0;
        s1Pipe_last <= enqueue_bits_last_0;
        s1Pipe_instructionIndex <= enqueue_bits_instructionIndex_0;
        s1BypassData <= dequeueWire_bits_data;
        s1EnqHitS1 <= enqHitS1;
        s1EnqHitS2 <= enqHitS2;
        fwd1 <= fwd;
      end
      readDataValid_pipe_v <= readBeforeWrite;
      readDataValid_pipe_pipe_v <= readDataValid_pipe_v;
    end
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:9];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [3:0] i = 4'h0; i < 4'hA; i += 4'h1) begin
          _RANDOM[i] = `RANDOM;
        end
        dequeueQueue_data_vd = _RANDOM[4'h0][4:0];
        dequeueQueue_data_offset = _RANDOM[4'h0][9:5];
        dequeueQueue_data_mask = _RANDOM[4'h0][13:10];
        dequeueQueue_data_data = {_RANDOM[4'h0][31:14], _RANDOM[4'h1][13:0]};
        dequeueQueue_data_last = _RANDOM[4'h1][14];
        dequeueQueue_data_instructionIndex = _RANDOM[4'h1][17:15];
        dequeueQueue_empty = _RANDOM[4'h1][18];
        s3Valid = _RANDOM[4'h1][19];
        s3Pipe_vd = _RANDOM[4'h1][24:20];
        s3Pipe_offset = _RANDOM[4'h1][29:25];
        s3Pipe_mask = {_RANDOM[4'h1][31:30], _RANDOM[4'h2][1:0]};
        s3Pipe_data = {_RANDOM[4'h2][31:2], _RANDOM[4'h3][1:0]};
        s3Pipe_last = _RANDOM[4'h3][2];
        s3Pipe_instructionIndex = _RANDOM[4'h3][5:3];
        s3BypassData = {_RANDOM[4'h3][31:6], _RANDOM[4'h4][5:0]};
        fwd3 = _RANDOM[4'h4][6];
        s2Valid = _RANDOM[4'h4][7];
        s2Pipe_vd = _RANDOM[4'h4][12:8];
        s2Pipe_offset = _RANDOM[4'h4][17:13];
        s2Pipe_mask = _RANDOM[4'h4][21:18];
        s2Pipe_data = {_RANDOM[4'h4][31:22], _RANDOM[4'h5][21:0]};
        s2Pipe_last = _RANDOM[4'h5][22];
        s2Pipe_instructionIndex = _RANDOM[4'h5][25:23];
        s2BypassData = {_RANDOM[4'h5][31:26], _RANDOM[4'h6][25:0]};
        s2EnqHitS1 = _RANDOM[4'h6][26];
        fwd2 = _RANDOM[4'h6][27];
        s1Valid = _RANDOM[4'h6][28];
        s1Pipe_vd = {_RANDOM[4'h6][31:29], _RANDOM[4'h7][1:0]};
        s1Pipe_offset = _RANDOM[4'h7][6:2];
        s1Pipe_mask = _RANDOM[4'h7][10:7];
        s1Pipe_data = {_RANDOM[4'h7][31:11], _RANDOM[4'h8][10:0]};
        s1Pipe_last = _RANDOM[4'h8][11];
        s1Pipe_instructionIndex = _RANDOM[4'h8][14:12];
        s1BypassData = {_RANDOM[4'h8][31:15], _RANDOM[4'h9][14:0]};
        s1EnqHitS1 = _RANDOM[4'h9][15];
        s1EnqHitS2 = _RANDOM[4'h9][16];
        fwd1 = _RANDOM[4'h9][17];
        readDataValid_pipe_v = _RANDOM[4'h9][18];
        readDataValid_pipe_pipe_v = _RANDOM[4'h9][20];
        readDataValid_pipe_pipe_b = _RANDOM[4'h9][21];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  wire        vrfReadPipe_empty;
  assign vrfReadPipe_empty = _vrfReadPipe_fifo_empty;
  wire        vrfReadPipe_full;
  assign vrfReadPipe_full = _vrfReadPipe_fifo_full;
  DW_fifo_s1_sf #(
    .ae_level(1),
    .af_level(1),
    .depth(4),
    .err_mode(2),
    .rst_mode(3),
    .width(32)
  ) vrfReadPipe_fifo (
    .clk          (clock),
    .rst_n        (~reset),
    .push_req_n   (~(vrfReadPipe_enq_ready & vrfReadPipe_enq_valid)),
    .pop_req_n    (~(vrfReadPipe_deq_ready & ~_vrfReadPipe_fifo_empty)),
    .diag_n       (1'h1),
    .data_in      (vrfReadPipe_enq_bits),
    .empty        (_vrfReadPipe_fifo_empty),
    .almost_empty (vrfReadPipe_almostEmpty),
    .half_full    (/* unused */),
    .almost_full  (vrfReadPipe_almostFull),
    .full         (_vrfReadPipe_fifo_full),
    .error        (_vrfReadPipe_fifo_error),
    .data_out     (vrfReadPipe_deq_bits)
  );
  assign enqueue_ready = enqueue_ready_0;
  assign dequeue_valid = dequeue_valid_0;
  assign dequeue_bits_vd = dequeue_bits_vd_0;
  assign dequeue_bits_offset = dequeue_bits_offset_0;
  assign dequeue_bits_mask = dequeue_bits_mask_0;
  assign dequeue_bits_data = dequeue_bits_data_0;
  assign dequeue_bits_last = dequeue_bits_last_0;
  assign dequeue_bits_instructionIndex = dequeue_bits_instructionIndex_0;
  assign vrfReadRequest_valid = vrfReadRequest_valid_0;
  assign vrfReadRequest_bits_vs = vrfReadRequest_bits_vs_0;
  assign vrfReadRequest_bits_offset = vrfReadRequest_bits_offset_0;
  assign vrfReadRequest_bits_instructionIndex = vrfReadRequest_bits_instructionIndex_0;
endmodule

