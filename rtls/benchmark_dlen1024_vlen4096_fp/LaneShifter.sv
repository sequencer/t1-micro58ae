
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
module LaneShifter(
  input         clock,
                reset,
                requestIO_valid,
  input  [1:0]  requestIO_bits_tag,
  input  [31:0] requestIO_bits_src_0,
                requestIO_bits_src_1,
  input  [4:0]  requestIO_bits_shifterSize,
  input  [2:0]  requestIO_bits_opcode,
  input  [1:0]  requestIO_bits_vxrm,
  output        responseIO_valid,
  output [1:0]  responseIO_bits_tag,
  output [31:0] responseIO_bits_data
);

  wire [31:0] response_data;
  wire        requestIO_valid_0 = requestIO_valid;
  wire [1:0]  requestIO_bits_tag_0 = requestIO_bits_tag;
  wire [31:0] requestIO_bits_src_0_0 = requestIO_bits_src_0;
  wire [31:0] requestIO_bits_src_1_0 = requestIO_bits_src_1;
  wire [4:0]  requestIO_bits_shifterSize_0 = requestIO_bits_shifterSize;
  wire [2:0]  requestIO_bits_opcode_0 = requestIO_bits_opcode;
  wire [1:0]  requestIO_bits_vxrm_0 = requestIO_bits_vxrm;
  wire [1:0]  response_tag = 2'h0;
  wire        requestIO_ready = 1'h1;
  wire        responseIO_ready = 1'h1;
  wire        request_pipeResponse_valid;
  wire [1:0]  request_pipeResponse_bits_tag;
  wire [31:0] request_pipeResponse_bits_data;
  reg  [1:0]  requestReg_tag;
  wire [1:0]  request_responseWire_tag = requestReg_tag;
  reg  [31:0] requestReg_src_0;
  reg  [31:0] requestReg_src_1;
  reg  [4:0]  requestReg_shifterSize;
  reg  [2:0]  requestReg_opcode;
  reg  [1:0]  requestReg_vxrm;
  reg         requestRegValid;
  wire        vfuRequestFire = requestRegValid;
  wire        request_responseValidWire = requestRegValid;
  wire [31:0] request_responseWire_data = response_data;
  reg         request_pipeResponse_pipe_v;
  assign request_pipeResponse_valid = request_pipeResponse_pipe_v;
  reg  [1:0]  request_pipeResponse_pipe_b_tag;
  assign request_pipeResponse_bits_tag = request_pipeResponse_pipe_b_tag;
  reg  [31:0] request_pipeResponse_pipe_b_data;
  assign request_pipeResponse_bits_data = request_pipeResponse_pipe_b_data;
  wire        responseIO_valid_0 = request_pipeResponse_valid;
  wire [1:0]  responseIO_bits_tag_0 = request_pipeResponse_bits_tag;
  wire [31:0] responseIO_bits_data_0 = request_pipeResponse_bits_data;
  wire [4:0]  request_lo = {requestReg_opcode, requestReg_vxrm};
  wire [65:0] request_hi_hi = {requestReg_tag, requestReg_src_1, requestReg_src_0};
  wire [70:0] request_hi = {request_hi_hi, requestReg_shifterSize};
  wire [1:0]  request_vxrm = request_lo[1:0];
  wire [2:0]  request_opcode = request_lo[4:2];
  wire [4:0]  request_shifterSize = request_hi[4:0];
  wire [31:0] request_src_0 = request_hi[36:5];
  wire [31:0] request_src_1 = request_hi[68:37];
  wire [1:0]  request_tag = request_hi[70:69];
  wire [31:0] extend = {32{request_opcode[1] & request_src_1[31]}};
  wire [63:0] extendData = {extend, request_src_1};
  wire [31:0] roundTail = 32'h1 << request_shifterSize;
  wire [30:0] lostMSB = roundTail[31:1];
  wire [31:0] roundMask = roundTail - 32'h1;
  wire        vds1 = |(request_src_1[30:0] & lostMSB);
  wire        vLostLSB = |(request_src_1[30:0] & roundMask[31:1]);
  wire        vd = |(roundTail & request_src_1);
  wire [3:0]  _roundR_T = 4'h1 << request_vxrm;
  wire        roundR = (_roundR_T[0] & vds1 | _roundR_T[1] & vds1 & (vLostLSB | vd) | _roundR_T[3] & ~vd & (vds1 | vLostLSB)) & request_opcode[2];
  wire [94:0] _response_data_T_1 = {31'h0, extendData} << request_shifterSize;
  wire [63:0] _response_data_T_2 = extendData >> request_shifterSize;
  assign response_data = (request_opcode[0] ? _response_data_T_1[31:0] : _response_data_T_2[31:0]) + {31'h0, roundR};
  always @(posedge clock) begin
    if (reset) begin
      requestReg_tag <= 2'h0;
      requestReg_src_0 <= 32'h0;
      requestReg_src_1 <= 32'h0;
      requestReg_shifterSize <= 5'h0;
      requestReg_opcode <= 3'h0;
      requestReg_vxrm <= 2'h0;
      requestRegValid <= 1'h0;
      request_pipeResponse_pipe_v <= 1'h0;
    end
    else begin
      if (requestIO_valid_0) begin
        requestReg_tag <= requestIO_bits_tag_0;
        requestReg_src_0 <= requestIO_bits_src_0_0;
        requestReg_src_1 <= requestIO_bits_src_1_0;
        requestReg_shifterSize <= requestIO_bits_shifterSize_0;
        requestReg_opcode <= requestIO_bits_opcode_0;
        requestReg_vxrm <= requestIO_bits_vxrm_0;
      end
      requestRegValid <= requestIO_valid_0;
      request_pipeResponse_pipe_v <= request_responseValidWire;
    end
    if (request_responseValidWire) begin
      request_pipeResponse_pipe_b_tag <= request_responseWire_tag;
      request_pipeResponse_pipe_b_data <= request_responseWire_data;
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
        requestReg_shifterSize = _RANDOM[2'h2][6:2];
        requestReg_opcode = _RANDOM[2'h2][9:7];
        requestReg_vxrm = _RANDOM[2'h2][11:10];
        requestRegValid = _RANDOM[2'h2][12];
        request_pipeResponse_pipe_v = _RANDOM[2'h2][13];
        request_pipeResponse_pipe_b_tag = _RANDOM[2'h2][15:14];
        request_pipeResponse_pipe_b_data = {_RANDOM[2'h2][31:16], _RANDOM[2'h3][15:0]};
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  assign responseIO_valid = responseIO_valid_0;
  assign responseIO_bits_tag = responseIO_bits_tag_0;
  assign responseIO_bits_data = responseIO_bits_data_0;
endmodule

