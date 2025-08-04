
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
module LaneDivFP(
  input         clock,
                reset,
  output        requestIO_ready,
  input         requestIO_valid,
  input  [1:0]  requestIO_bits_tag,
  input  [31:0] requestIO_bits_src_0,
                requestIO_bits_src_1,
  input  [3:0]  requestIO_bits_opcode,
  input         requestIO_bits_sign,
  input  [1:0]  requestIO_bits_executeIndex,
  input  [2:0]  requestIO_bits_roundingMode,
  output        responseIO_valid,
  output [1:0]  responseIO_bits_tag,
  output [31:0] responseIO_bits_data,
  output [4:0]  responseIO_bits_exceptionFlags,
  output [1:0]  responseIO_bits_executeIndex,
  output        responseIO_bits_busy
);

  wire        responseValid;
  wire        requestIO_valid_0 = requestIO_valid;
  wire [1:0]  requestIO_bits_tag_0 = requestIO_bits_tag;
  wire [31:0] requestIO_bits_src_0_0 = requestIO_bits_src_0;
  wire [31:0] requestIO_bits_src_1_0 = requestIO_bits_src_1;
  wire [3:0]  requestIO_bits_opcode_0 = requestIO_bits_opcode;
  wire        requestIO_bits_sign_0 = requestIO_bits_sign;
  wire [1:0]  requestIO_bits_executeIndex_0 = requestIO_bits_executeIndex;
  wire [2:0]  requestIO_bits_roundingMode_0 = requestIO_bits_roundingMode;
  wire [1:0]  response_tag = 2'h0;
  wire        responseIO_ready = 1'h1;
  wire        request_pipeResponse_valid;
  wire [1:0]  request_pipeResponse_bits_tag;
  wire [31:0] request_pipeResponse_bits_data;
  wire [4:0]  request_pipeResponse_bits_exceptionFlags;
  wire [1:0]  request_pipeResponse_bits_executeIndex;
  wire        request_pipeResponse_bits_busy;
  reg  [1:0]  requestReg_tag;
  reg  [31:0] requestReg_src_0;
  reg  [31:0] requestReg_src_1;
  reg  [3:0]  requestReg_opcode;
  reg         requestReg_sign;
  reg  [1:0]  requestReg_executeIndex;
  reg  [2:0]  requestReg_roundingMode;
  reg         requestRegValid;
  wire        vfuRequestReady;
  wire        vfuRequestFire = vfuRequestReady & requestRegValid;
  wire        requestIO_ready_0 = ~requestRegValid | vfuRequestReady;
  wire [1:0]  response_executeIndex;
  wire        response_busy;
  wire        request_responseValidWire = responseValid;
  wire [2:0]  request_responseWire_lo = {response_executeIndex, response_busy};
  wire [31:0] response_data;
  wire [33:0] request_responseWire_hi_hi = {2'h0, response_data};
  wire [4:0]  response_exceptionFlags;
  wire [38:0] request_responseWire_hi = {request_responseWire_hi_hi, response_exceptionFlags};
  wire        request_responseWire_busy = request_responseWire_lo[0];
  wire [1:0]  request_responseWire_executeIndex = request_responseWire_lo[2:1];
  wire [4:0]  request_responseWire_exceptionFlags = request_responseWire_hi[4:0];
  wire [31:0] request_responseWire_data = request_responseWire_hi[36:5];
  reg  [1:0]  request_responseWire_tag_r;
  wire [1:0]  request_responseWire_tag = request_responseWire_tag_r;
  reg         request_pipeResponse_pipe_v;
  assign request_pipeResponse_valid = request_pipeResponse_pipe_v;
  reg  [1:0]  request_pipeResponse_pipe_b_tag;
  assign request_pipeResponse_bits_tag = request_pipeResponse_pipe_b_tag;
  reg  [31:0] request_pipeResponse_pipe_b_data;
  assign request_pipeResponse_bits_data = request_pipeResponse_pipe_b_data;
  reg  [4:0]  request_pipeResponse_pipe_b_exceptionFlags;
  assign request_pipeResponse_bits_exceptionFlags = request_pipeResponse_pipe_b_exceptionFlags;
  reg  [1:0]  request_pipeResponse_pipe_b_executeIndex;
  assign request_pipeResponse_bits_executeIndex = request_pipeResponse_pipe_b_executeIndex;
  reg         request_pipeResponse_pipe_b_busy;
  assign request_pipeResponse_bits_busy = request_pipeResponse_pipe_b_busy;
  wire        responseIO_valid_0 = request_pipeResponse_valid;
  wire [1:0]  responseIO_bits_tag_0 = request_pipeResponse_bits_tag;
  wire [31:0] responseIO_bits_data_0 = request_pipeResponse_bits_data;
  wire [4:0]  responseIO_bits_exceptionFlags_0 = request_pipeResponse_bits_exceptionFlags;
  wire [1:0]  responseIO_bits_executeIndex_0 = request_pipeResponse_bits_executeIndex;
  wire        responseIO_bits_busy_0 = request_pipeResponse_bits_busy;
  wire [2:0]  request_lo_hi = {requestReg_sign, requestReg_executeIndex};
  wire [5:0]  request_lo = {request_lo_hi, requestReg_roundingMode};
  wire [65:0] request_hi_hi = {requestReg_tag, requestReg_src_1, requestReg_src_0};
  wire [69:0] request_hi = {request_hi_hi, requestReg_opcode};
  wire [2:0]  request_roundingMode = request_lo[2:0];
  wire [1:0]  request_executeIndex = request_lo[4:3];
  wire        request_sign = request_lo[5];
  wire [3:0]  request_opcode = request_hi[3:0];
  wire [31:0] request_src_0 = request_hi[35:4];
  wire [31:0] request_src_1 = request_hi[67:36];
  wire [1:0]  request_tag = request_hi[69:68];
  wire        integerEn = ~(request_opcode[3]);
  wire        isRem = request_opcode == 4'h1;
  wire        fractEn = ~integerEn;
  wire        rdiv = request_opcode == 4'hA;
  wire        sqrt = request_opcode == 4'h9;
  wire [31:0] divIn0 = rdiv ? request_src_0 : request_src_1;
  wire [31:0] divIn1 = rdiv ? request_src_1 : request_src_0;
  reg  [1:0]  indexReg;
  assign response_executeIndex = indexReg;
  reg         response_busy_r;
  assign response_busy = response_busy_r;
  always @(posedge clock) begin
    if (reset) begin
      requestReg_tag <= 2'h0;
      requestReg_src_0 <= 32'h0;
      requestReg_src_1 <= 32'h0;
      requestReg_opcode <= 4'h0;
      requestReg_sign <= 1'h0;
      requestReg_executeIndex <= 2'h0;
      requestReg_roundingMode <= 3'h0;
      requestRegValid <= 1'h0;
      request_responseWire_tag_r <= 2'h0;
      request_pipeResponse_pipe_v <= 1'h0;
      indexReg <= 2'h0;
      response_busy_r <= 1'h0;
    end
    else begin
      automatic logic _requestRegValid_T;
      _requestRegValid_T = requestIO_ready_0 & requestIO_valid_0;
      if (_requestRegValid_T) begin
        requestReg_tag <= requestIO_bits_tag_0;
        requestReg_src_0 <= requestIO_bits_src_0_0;
        requestReg_src_1 <= requestIO_bits_src_1_0;
        requestReg_opcode <= requestIO_bits_opcode_0;
        requestReg_sign <= requestIO_bits_sign_0;
        requestReg_executeIndex <= requestIO_bits_executeIndex_0;
        requestReg_roundingMode <= requestIO_bits_roundingMode_0;
      end
      if (vfuRequestFire ^ _requestRegValid_T)
        requestRegValid <= _requestRegValid_T;
      if (vfuRequestFire) begin
        request_responseWire_tag_r <= requestReg_tag;
        indexReg <= request_executeIndex;
      end
      request_pipeResponse_pipe_v <= request_responseValidWire;
      if (vfuRequestFire ^ responseIO_valid_0)
        response_busy_r <= vfuRequestFire;
    end
    if (request_responseValidWire) begin
      request_pipeResponse_pipe_b_tag <= request_responseWire_tag;
      request_pipeResponse_pipe_b_data <= request_responseWire_data;
      request_pipeResponse_pipe_b_exceptionFlags <= request_responseWire_exceptionFlags;
      request_pipeResponse_pipe_b_executeIndex <= request_responseWire_executeIndex;
      request_pipeResponse_pipe_b_busy <= request_responseWire_busy;
    end
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:3];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [2:0] i = 3'h0; i < 3'h4; i += 3'h1) begin
          _RANDOM[i[1:0]] = `RANDOM;
        end
        requestReg_tag = _RANDOM[2'h0][1:0];
        requestReg_src_0 = {_RANDOM[2'h0][31:2], _RANDOM[2'h1][1:0]};
        requestReg_src_1 = {_RANDOM[2'h1][31:2], _RANDOM[2'h2][1:0]};
        requestReg_opcode = _RANDOM[2'h2][5:2];
        requestReg_sign = _RANDOM[2'h2][6];
        requestReg_executeIndex = _RANDOM[2'h2][8:7];
        requestReg_roundingMode = _RANDOM[2'h2][11:9];
        requestRegValid = _RANDOM[2'h2][12];
        request_responseWire_tag_r = _RANDOM[2'h2][14:13];
        request_pipeResponse_pipe_v = _RANDOM[2'h2][15];
        request_pipeResponse_pipe_b_tag = _RANDOM[2'h2][17:16];
        request_pipeResponse_pipe_b_data = {_RANDOM[2'h2][31:18], _RANDOM[2'h3][17:0]};
        request_pipeResponse_pipe_b_exceptionFlags = _RANDOM[2'h3][22:18];
        request_pipeResponse_pipe_b_executeIndex = _RANDOM[2'h3][24:23];
        request_pipeResponse_pipe_b_busy = _RANDOM[2'h3][25];
        indexReg = _RANDOM[2'h3][27:26];
        response_busy_r = _RANDOM[2'h3][28];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  SRTFPWrapper wrapper (
    .clock                      (clock),
    .reset                      (reset),
    .input_ready                (vfuRequestReady),
    .input_valid                (requestRegValid),
    .input_bits_a               (fractEn ? divIn0 : request_src_1),
    .input_bits_b               (fractEn ? divIn1 : request_src_0),
    .input_bits_signIn          (request_sign),
    .input_bits_opFloat         (fractEn),
    .input_bits_opSqrt          (sqrt),
    .input_bits_opRem           (isRem),
    .input_bits_roundingMode    (request_roundingMode),
    .output_valid               (responseValid),
    .output_bits_result         (response_data),
    .output_bits_exceptionFlags (response_exceptionFlags)
  );
  assign requestIO_ready = requestIO_ready_0;
  assign responseIO_valid = responseIO_valid_0;
  assign responseIO_bits_tag = responseIO_bits_tag_0;
  assign responseIO_bits_data = responseIO_bits_data_0;
  assign responseIO_bits_exceptionFlags = responseIO_bits_exceptionFlags_0;
  assign responseIO_bits_executeIndex = responseIO_bits_executeIndex_0;
  assign responseIO_bits_busy = responseIO_bits_busy_0;
endmodule

