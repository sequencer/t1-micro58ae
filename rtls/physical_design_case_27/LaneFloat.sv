
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
module LaneFloat(
  input         clock,
                reset,
                requestIO_valid,
  input  [1:0]  requestIO_bits_tag,
  input         requestIO_bits_sign,
  input  [31:0] requestIO_bits_src_0,
                requestIO_bits_src_1,
                requestIO_bits_src_2,
  input  [3:0]  requestIO_bits_opcode,
  input  [1:0]  requestIO_bits_unitSelet,
  input         requestIO_bits_floatMul,
  input  [2:0]  requestIO_bits_roundingMode,
  input  [1:0]  requestIO_bits_executeIndex,
  output        responseIO_valid,
  output [1:0]  responseIO_bits_tag,
  output [31:0] responseIO_bits_data,
  output        responseIO_bits_adderMaskResp,
  output [4:0]  responseIO_bits_exceptionFlags,
  output [1:0]  responseIO_bits_executeIndex
);

  wire [31:0]  _rsqrt7Module_out_data;
  wire [4:0]   _rsqrt7Module_out_exceptionFlags;
  wire [31:0]  _rec7Module_out_data;
  wire [4:0]   _rec7Module_out_exceptionFlags;
  wire [31:0]  _fnToInt_io_out;
  wire [2:0]   _fnToInt_io_intExceptionFlags;
  wire [32:0]  _intToFn_io_out;
  wire [4:0]   _intToFn_io_exceptionFlags;
  wire         _compareModule_io_lt;
  wire         _compareModule_io_eq;
  wire         _compareModule_io_gt;
  wire [32:0]  _mulAddRecFN_io_out;
  wire [4:0]   _mulAddRecFN_io_exceptionFlags;
  wire         requestIO_valid_0 = requestIO_valid;
  wire [1:0]   requestIO_bits_tag_0 = requestIO_bits_tag;
  wire         requestIO_bits_sign_0 = requestIO_bits_sign;
  wire [31:0]  requestIO_bits_src_0_0 = requestIO_bits_src_0;
  wire [31:0]  requestIO_bits_src_1_0 = requestIO_bits_src_1;
  wire [31:0]  requestIO_bits_src_2_0 = requestIO_bits_src_2;
  wire [3:0]   requestIO_bits_opcode_0 = requestIO_bits_opcode;
  wire [1:0]   requestIO_bits_unitSelet_0 = requestIO_bits_unitSelet;
  wire         requestIO_bits_floatMul_0 = requestIO_bits_floatMul;
  wire [2:0]   requestIO_bits_roundingMode_0 = requestIO_bits_roundingMode;
  wire [1:0]   requestIO_bits_executeIndex_0 = requestIO_bits_executeIndex;
  wire [1:0]   response_tag = 2'h0;
  wire         requestIO_ready = 1'h1;
  wire         responseIO_ready = 1'h1;
  wire         request_pipeResponse_valid;
  wire [1:0]   request_pipeResponse_bits_tag;
  wire [31:0]  request_pipeResponse_bits_data;
  wire         request_pipeResponse_bits_adderMaskResp;
  wire [4:0]   request_pipeResponse_bits_exceptionFlags;
  wire [1:0]   request_pipeResponse_bits_executeIndex;
  reg  [1:0]   requestReg_tag;
  wire [1:0]   request_responseWire_tag = requestReg_tag;
  reg          requestReg_sign;
  reg  [31:0]  requestReg_src_0;
  reg  [31:0]  requestReg_src_1;
  reg  [31:0]  requestReg_src_2;
  reg  [3:0]   requestReg_opcode;
  reg  [1:0]   requestReg_unitSelet;
  reg          requestReg_floatMul;
  reg  [2:0]   requestReg_roundingMode;
  reg  [1:0]   requestReg_executeIndex;
  reg          requestRegValid;
  wire         vfuRequestFire = requestRegValid;
  wire         request_responseValidWire = requestRegValid;
  wire [31:0]  result;
  wire [4:0]   flags;
  wire [1:0]   request_executeIndex;
  wire [4:0]   response_exceptionFlags;
  wire [1:0]   response_executeIndex;
  wire [6:0]   request_responseWire_lo = {response_exceptionFlags, response_executeIndex};
  wire [31:0]  response_data;
  wire [33:0]  request_responseWire_hi_hi = {2'h0, response_data};
  wire         response_adderMaskResp;
  wire [34:0]  request_responseWire_hi = {request_responseWire_hi_hi, response_adderMaskResp};
  wire [1:0]   request_responseWire_executeIndex = request_responseWire_lo[1:0];
  wire [4:0]   request_responseWire_exceptionFlags = request_responseWire_lo[6:2];
  wire         request_responseWire_adderMaskResp = request_responseWire_hi[0];
  wire [31:0]  request_responseWire_data = request_responseWire_hi[32:1];
  reg          request_pipeResponse_pipe_v;
  reg  [1:0]   request_pipeResponse_pipe_b_tag;
  reg  [31:0]  request_pipeResponse_pipe_b_data;
  reg          request_pipeResponse_pipe_b_adderMaskResp;
  reg  [4:0]   request_pipeResponse_pipe_b_exceptionFlags;
  reg  [1:0]   request_pipeResponse_pipe_b_executeIndex;
  reg          request_pipeResponse_pipe_pipe_v;
  reg  [1:0]   request_pipeResponse_pipe_pipe_b_tag;
  reg  [31:0]  request_pipeResponse_pipe_pipe_b_data;
  reg          request_pipeResponse_pipe_pipe_b_adderMaskResp;
  reg  [4:0]   request_pipeResponse_pipe_pipe_b_exceptionFlags;
  reg  [1:0]   request_pipeResponse_pipe_pipe_b_executeIndex;
  reg          request_pipeResponse_pipe_pipe_pipe_v;
  assign request_pipeResponse_valid = request_pipeResponse_pipe_pipe_pipe_v;
  reg  [1:0]   request_pipeResponse_pipe_pipe_pipe_b_tag;
  assign request_pipeResponse_bits_tag = request_pipeResponse_pipe_pipe_pipe_b_tag;
  reg  [31:0]  request_pipeResponse_pipe_pipe_pipe_b_data;
  assign request_pipeResponse_bits_data = request_pipeResponse_pipe_pipe_pipe_b_data;
  reg          request_pipeResponse_pipe_pipe_pipe_b_adderMaskResp;
  assign request_pipeResponse_bits_adderMaskResp = request_pipeResponse_pipe_pipe_pipe_b_adderMaskResp;
  reg  [4:0]   request_pipeResponse_pipe_pipe_pipe_b_exceptionFlags;
  assign request_pipeResponse_bits_exceptionFlags = request_pipeResponse_pipe_pipe_pipe_b_exceptionFlags;
  reg  [1:0]   request_pipeResponse_pipe_pipe_pipe_b_executeIndex;
  assign request_pipeResponse_bits_executeIndex = request_pipeResponse_pipe_pipe_pipe_b_executeIndex;
  wire         responseIO_valid_0 = request_pipeResponse_valid;
  wire [1:0]   responseIO_bits_tag_0 = request_pipeResponse_bits_tag;
  wire [31:0]  responseIO_bits_data_0 = request_pipeResponse_bits_data;
  wire         responseIO_bits_adderMaskResp_0 = request_pipeResponse_bits_adderMaskResp;
  wire [4:0]   responseIO_bits_exceptionFlags_0 = request_pipeResponse_bits_exceptionFlags;
  wire [1:0]   responseIO_bits_executeIndex_0 = request_pipeResponse_bits_executeIndex;
  wire [63:0]  request_hi = {requestReg_src_2, requestReg_src_1};
  wire [4:0]   request_lo_lo = {requestReg_roundingMode, requestReg_executeIndex};
  wire [2:0]   request_lo_hi = {requestReg_unitSelet, requestReg_floatMul};
  wire [7:0]   request_lo = {request_lo_hi, request_lo_lo};
  wire [99:0]  request_hi_lo = {request_hi, requestReg_src_0, requestReg_opcode};
  wire [2:0]   request_hi_hi = {requestReg_tag, requestReg_sign};
  wire [102:0] request_hi_1 = {request_hi_hi, request_hi_lo};
  assign request_executeIndex = request_lo[1:0];
  wire [2:0]   request_roundingMode = request_lo[4:2];
  wire         request_floatMul = request_lo[5];
  wire [1:0]   request_unitSelet = request_lo[7:6];
  wire [3:0]   request_opcode = request_hi_1[3:0];
  wire [31:0]  request_src_0 = request_hi_1[35:4];
  wire [31:0]  request_src_1 = request_hi_1[67:36];
  wire [31:0]  request_src_2 = request_hi_1[99:68];
  wire         request_sign = request_hi_1[100];
  wire [1:0]   request_tag = request_hi_1[102:101];
  assign response_executeIndex = request_executeIndex;
  wire         recIn0_rawIn_sign = request_src_0[31];
  wire         recIn0_rawIn_sign_0 = recIn0_rawIn_sign;
  wire [7:0]   recIn0_rawIn_expIn = request_src_0[30:23];
  wire [22:0]  recIn0_rawIn_fractIn = request_src_0[22:0];
  wire         recIn0_rawIn_isZeroExpIn = recIn0_rawIn_expIn == 8'h0;
  wire         recIn0_rawIn_isZeroFractIn = recIn0_rawIn_fractIn == 23'h0;
  wire [4:0]   recIn0_rawIn_normDist =
    recIn0_rawIn_fractIn[22]
      ? 5'h0
      : recIn0_rawIn_fractIn[21]
          ? 5'h1
          : recIn0_rawIn_fractIn[20]
              ? 5'h2
              : recIn0_rawIn_fractIn[19]
                  ? 5'h3
                  : recIn0_rawIn_fractIn[18]
                      ? 5'h4
                      : recIn0_rawIn_fractIn[17]
                          ? 5'h5
                          : recIn0_rawIn_fractIn[16]
                              ? 5'h6
                              : recIn0_rawIn_fractIn[15]
                                  ? 5'h7
                                  : recIn0_rawIn_fractIn[14]
                                      ? 5'h8
                                      : recIn0_rawIn_fractIn[13]
                                          ? 5'h9
                                          : recIn0_rawIn_fractIn[12]
                                              ? 5'hA
                                              : recIn0_rawIn_fractIn[11]
                                                  ? 5'hB
                                                  : recIn0_rawIn_fractIn[10]
                                                      ? 5'hC
                                                      : recIn0_rawIn_fractIn[9]
                                                          ? 5'hD
                                                          : recIn0_rawIn_fractIn[8]
                                                              ? 5'hE
                                                              : recIn0_rawIn_fractIn[7]
                                                                  ? 5'hF
                                                                  : recIn0_rawIn_fractIn[6]
                                                                      ? 5'h10
                                                                      : recIn0_rawIn_fractIn[5]
                                                                          ? 5'h11
                                                                          : recIn0_rawIn_fractIn[4] ? 5'h12 : recIn0_rawIn_fractIn[3] ? 5'h13 : recIn0_rawIn_fractIn[2] ? 5'h14 : recIn0_rawIn_fractIn[1] ? 5'h15 : 5'h16;
  wire [53:0]  _recIn0_rawIn_subnormFract_T = {31'h0, recIn0_rawIn_fractIn} << recIn0_rawIn_normDist;
  wire [22:0]  recIn0_rawIn_subnormFract = {_recIn0_rawIn_subnormFract_T[21:0], 1'h0};
  wire [8:0]   recIn0_rawIn_adjustedExp = (recIn0_rawIn_isZeroExpIn ? {4'hF, ~recIn0_rawIn_normDist} : {1'h0, recIn0_rawIn_expIn}) + {7'h20, recIn0_rawIn_isZeroExpIn ? 2'h2 : 2'h1};
  wire         recIn0_rawIn_isZero = recIn0_rawIn_isZeroExpIn & recIn0_rawIn_isZeroFractIn;
  wire         recIn0_rawIn_isZero_0 = recIn0_rawIn_isZero;
  wire         recIn0_rawIn_isSpecial = &(recIn0_rawIn_adjustedExp[8:7]);
  wire         recIn0_rawIn_isNaN = recIn0_rawIn_isSpecial & ~recIn0_rawIn_isZeroFractIn;
  wire         recIn0_rawIn_isInf = recIn0_rawIn_isSpecial & recIn0_rawIn_isZeroFractIn;
  wire [9:0]   recIn0_rawIn_sExp = {1'h0, recIn0_rawIn_adjustedExp};
  wire [24:0]  recIn0_rawIn_sig = {1'h0, ~recIn0_rawIn_isZero, recIn0_rawIn_isZeroExpIn ? recIn0_rawIn_subnormFract : recIn0_rawIn_fractIn};
  wire [2:0]   _recIn0_T_1 = recIn0_rawIn_isZero_0 ? 3'h0 : recIn0_rawIn_sExp[8:6];
  wire [32:0]  recIn0 = {recIn0_rawIn_sign_0, _recIn0_T_1[2:1], _recIn0_T_1[0] | recIn0_rawIn_isNaN, recIn0_rawIn_sExp[5:0], recIn0_rawIn_sig[22:0]};
  wire         recIn1_rawIn_sign = request_src_1[31];
  wire         recIn1_rawIn_sign_0 = recIn1_rawIn_sign;
  wire [7:0]   recIn1_rawIn_expIn = request_src_1[30:23];
  wire [22:0]  recIn1_rawIn_fractIn = request_src_1[22:0];
  wire         recIn1_rawIn_isZeroExpIn = recIn1_rawIn_expIn == 8'h0;
  wire         recIn1_rawIn_isZeroFractIn = recIn1_rawIn_fractIn == 23'h0;
  wire [4:0]   recIn1_rawIn_normDist =
    recIn1_rawIn_fractIn[22]
      ? 5'h0
      : recIn1_rawIn_fractIn[21]
          ? 5'h1
          : recIn1_rawIn_fractIn[20]
              ? 5'h2
              : recIn1_rawIn_fractIn[19]
                  ? 5'h3
                  : recIn1_rawIn_fractIn[18]
                      ? 5'h4
                      : recIn1_rawIn_fractIn[17]
                          ? 5'h5
                          : recIn1_rawIn_fractIn[16]
                              ? 5'h6
                              : recIn1_rawIn_fractIn[15]
                                  ? 5'h7
                                  : recIn1_rawIn_fractIn[14]
                                      ? 5'h8
                                      : recIn1_rawIn_fractIn[13]
                                          ? 5'h9
                                          : recIn1_rawIn_fractIn[12]
                                              ? 5'hA
                                              : recIn1_rawIn_fractIn[11]
                                                  ? 5'hB
                                                  : recIn1_rawIn_fractIn[10]
                                                      ? 5'hC
                                                      : recIn1_rawIn_fractIn[9]
                                                          ? 5'hD
                                                          : recIn1_rawIn_fractIn[8]
                                                              ? 5'hE
                                                              : recIn1_rawIn_fractIn[7]
                                                                  ? 5'hF
                                                                  : recIn1_rawIn_fractIn[6]
                                                                      ? 5'h10
                                                                      : recIn1_rawIn_fractIn[5]
                                                                          ? 5'h11
                                                                          : recIn1_rawIn_fractIn[4] ? 5'h12 : recIn1_rawIn_fractIn[3] ? 5'h13 : recIn1_rawIn_fractIn[2] ? 5'h14 : recIn1_rawIn_fractIn[1] ? 5'h15 : 5'h16;
  wire [53:0]  _recIn1_rawIn_subnormFract_T = {31'h0, recIn1_rawIn_fractIn} << recIn1_rawIn_normDist;
  wire [22:0]  recIn1_rawIn_subnormFract = {_recIn1_rawIn_subnormFract_T[21:0], 1'h0};
  wire [8:0]   recIn1_rawIn_adjustedExp = (recIn1_rawIn_isZeroExpIn ? {4'hF, ~recIn1_rawIn_normDist} : {1'h0, recIn1_rawIn_expIn}) + {7'h20, recIn1_rawIn_isZeroExpIn ? 2'h2 : 2'h1};
  wire         recIn1_rawIn_isZero = recIn1_rawIn_isZeroExpIn & recIn1_rawIn_isZeroFractIn;
  wire         recIn1_rawIn_isZero_0 = recIn1_rawIn_isZero;
  wire         recIn1_rawIn_isSpecial = &(recIn1_rawIn_adjustedExp[8:7]);
  wire         recIn1_rawIn_isNaN = recIn1_rawIn_isSpecial & ~recIn1_rawIn_isZeroFractIn;
  wire         recIn1_rawIn_isInf = recIn1_rawIn_isSpecial & recIn1_rawIn_isZeroFractIn;
  wire [9:0]   recIn1_rawIn_sExp = {1'h0, recIn1_rawIn_adjustedExp};
  wire [24:0]  recIn1_rawIn_sig = {1'h0, ~recIn1_rawIn_isZero, recIn1_rawIn_isZeroExpIn ? recIn1_rawIn_subnormFract : recIn1_rawIn_fractIn};
  wire [2:0]   _recIn1_T_1 = recIn1_rawIn_isZero_0 ? 3'h0 : recIn1_rawIn_sExp[8:6];
  wire [32:0]  recIn1 = {recIn1_rawIn_sign_0, _recIn1_T_1[2:1], _recIn1_T_1[0] | recIn1_rawIn_isNaN, recIn1_rawIn_sExp[5:0], recIn1_rawIn_sig[22:0]};
  wire         recIn2_rawIn_sign = request_src_2[31];
  wire         recIn2_rawIn_sign_0 = recIn2_rawIn_sign;
  wire [7:0]   recIn2_rawIn_expIn = request_src_2[30:23];
  wire [22:0]  recIn2_rawIn_fractIn = request_src_2[22:0];
  wire         recIn2_rawIn_isZeroExpIn = recIn2_rawIn_expIn == 8'h0;
  wire         recIn2_rawIn_isZeroFractIn = recIn2_rawIn_fractIn == 23'h0;
  wire [4:0]   recIn2_rawIn_normDist =
    recIn2_rawIn_fractIn[22]
      ? 5'h0
      : recIn2_rawIn_fractIn[21]
          ? 5'h1
          : recIn2_rawIn_fractIn[20]
              ? 5'h2
              : recIn2_rawIn_fractIn[19]
                  ? 5'h3
                  : recIn2_rawIn_fractIn[18]
                      ? 5'h4
                      : recIn2_rawIn_fractIn[17]
                          ? 5'h5
                          : recIn2_rawIn_fractIn[16]
                              ? 5'h6
                              : recIn2_rawIn_fractIn[15]
                                  ? 5'h7
                                  : recIn2_rawIn_fractIn[14]
                                      ? 5'h8
                                      : recIn2_rawIn_fractIn[13]
                                          ? 5'h9
                                          : recIn2_rawIn_fractIn[12]
                                              ? 5'hA
                                              : recIn2_rawIn_fractIn[11]
                                                  ? 5'hB
                                                  : recIn2_rawIn_fractIn[10]
                                                      ? 5'hC
                                                      : recIn2_rawIn_fractIn[9]
                                                          ? 5'hD
                                                          : recIn2_rawIn_fractIn[8]
                                                              ? 5'hE
                                                              : recIn2_rawIn_fractIn[7]
                                                                  ? 5'hF
                                                                  : recIn2_rawIn_fractIn[6]
                                                                      ? 5'h10
                                                                      : recIn2_rawIn_fractIn[5]
                                                                          ? 5'h11
                                                                          : recIn2_rawIn_fractIn[4] ? 5'h12 : recIn2_rawIn_fractIn[3] ? 5'h13 : recIn2_rawIn_fractIn[2] ? 5'h14 : recIn2_rawIn_fractIn[1] ? 5'h15 : 5'h16;
  wire [53:0]  _recIn2_rawIn_subnormFract_T = {31'h0, recIn2_rawIn_fractIn} << recIn2_rawIn_normDist;
  wire [22:0]  recIn2_rawIn_subnormFract = {_recIn2_rawIn_subnormFract_T[21:0], 1'h0};
  wire [8:0]   recIn2_rawIn_adjustedExp = (recIn2_rawIn_isZeroExpIn ? {4'hF, ~recIn2_rawIn_normDist} : {1'h0, recIn2_rawIn_expIn}) + {7'h20, recIn2_rawIn_isZeroExpIn ? 2'h2 : 2'h1};
  wire         recIn2_rawIn_isZero = recIn2_rawIn_isZeroExpIn & recIn2_rawIn_isZeroFractIn;
  wire         recIn2_rawIn_isZero_0 = recIn2_rawIn_isZero;
  wire         recIn2_rawIn_isSpecial = &(recIn2_rawIn_adjustedExp[8:7]);
  wire         recIn2_rawIn_isNaN = recIn2_rawIn_isSpecial & ~recIn2_rawIn_isZeroFractIn;
  wire         recIn2_rawIn_isInf = recIn2_rawIn_isSpecial & recIn2_rawIn_isZeroFractIn;
  wire [9:0]   recIn2_rawIn_sExp = {1'h0, recIn2_rawIn_adjustedExp};
  wire [24:0]  recIn2_rawIn_sig = {1'h0, ~recIn2_rawIn_isZero, recIn2_rawIn_isZeroExpIn ? recIn2_rawIn_subnormFract : recIn2_rawIn_fractIn};
  wire [2:0]   _recIn2_T_1 = recIn2_rawIn_isZero_0 ? 3'h0 : recIn2_rawIn_sExp[8:6];
  wire [32:0]  recIn2 = {recIn2_rawIn_sign_0, _recIn2_T_1[2:1], _recIn2_T_1[0] | recIn2_rawIn_isNaN, recIn2_rawIn_sExp[5:0], recIn2_rawIn_sig[22:0]};
  wire [8:0]   raw0_exp = recIn0[31:23];
  wire         raw0_isZero = raw0_exp[8:6] == 3'h0;
  wire         raw0_isZero_0 = raw0_isZero;
  wire         raw0_isSpecial = &(raw0_exp[8:7]);
  wire         raw0_isNaN = raw0_isSpecial & raw0_exp[6];
  wire         raw0_isInf = raw0_isSpecial & ~(raw0_exp[6]);
  wire         raw0_sign = recIn0[32];
  wire [9:0]   raw0_sExp = {1'h0, raw0_exp};
  wire [24:0]  raw0_sig = {1'h0, ~raw0_isZero, recIn0[22:0]};
  wire [8:0]   raw1_exp = recIn1[31:23];
  wire [8:0]   in1classify_rawIn_exp = recIn1[31:23];
  wire         raw1_isZero = raw1_exp[8:6] == 3'h0;
  wire         raw1_isZero_0 = raw1_isZero;
  wire         raw1_isSpecial = &(raw1_exp[8:7]);
  wire         raw1_isNaN = raw1_isSpecial & raw1_exp[6];
  wire         raw1_isInf = raw1_isSpecial & ~(raw1_exp[6]);
  wire         raw1_sign = recIn1[32];
  wire         in1classify_rawIn_sign = recIn1[32];
  wire [9:0]   raw1_sExp = {1'h0, raw1_exp};
  wire [24:0]  raw1_sig = {1'h0, ~raw1_isZero, recIn1[22:0]};
  wire [3:0]   unitSeleOH = 4'h1 << request_unitSelet;
  wire         fmaEn = unitSeleOH[0];
  wire         compareEn = unitSeleOH[2];
  wire         otherEn = unitSeleOH[3];
  assign response_data = result;
  assign response_exceptionFlags = flags;
  wire         sub = fmaEn & request_opcode == 4'h9;
  wire         addsub = fmaEn & request_opcode[3];
  wire         maf = fmaEn & ~(|(request_opcode[3:2]));
  wire         rmaf = fmaEn & request_opcode[3:2] == 2'h1;
  wire [32:0]  fmaIn0 = sub ? recIn1 : recIn0;
  wire [32:0]  fmaIn1 = addsub ? 33'h80000000 : rmaf ? recIn2 : recIn1;
  wire [32:0]  fmaIn2 = sub ? recIn0 : maf & ~request_floatMul ? recIn2 : maf & request_floatMul ? {(request_src_0 ^ request_src_1) & 32'h80000000, 1'h0} : recIn1;
  wire [8:0]   fmaResult_rawIn_exp = _mulAddRecFN_io_out[31:23];
  wire         fmaResult_rawIn_isZero = fmaResult_rawIn_exp[8:6] == 3'h0;
  wire         fmaResult_rawIn_isZero_0 = fmaResult_rawIn_isZero;
  wire         fmaResult_rawIn_isSpecial = &(fmaResult_rawIn_exp[8:7]);
  wire         fmaResult_rawIn_isNaN = fmaResult_rawIn_isSpecial & fmaResult_rawIn_exp[6];
  wire         fmaResult_rawIn_isInf = fmaResult_rawIn_isSpecial & ~(fmaResult_rawIn_exp[6]);
  wire         fmaResult_rawIn_sign = _mulAddRecFN_io_out[32];
  wire [9:0]   fmaResult_rawIn_sExp = {1'h0, fmaResult_rawIn_exp};
  wire [24:0]  fmaResult_rawIn_sig = {1'h0, ~fmaResult_rawIn_isZero, _mulAddRecFN_io_out[22:0]};
  wire         fmaResult_isSubnormal = $signed(fmaResult_rawIn_sExp) < 10'sh82;
  wire [4:0]   fmaResult_denormShiftDist = 5'h1 - fmaResult_rawIn_sExp[4:0];
  wire [23:0]  _fmaResult_denormFract_T_1 = fmaResult_rawIn_sig[24:1] >> fmaResult_denormShiftDist;
  wire [22:0]  fmaResult_denormFract = _fmaResult_denormFract_T_1[22:0];
  wire [7:0]   fmaResult_expOut = (fmaResult_isSubnormal ? 8'h0 : fmaResult_rawIn_sExp[7:0] + 8'h7F) | {8{fmaResult_rawIn_isNaN | fmaResult_rawIn_isInf}};
  wire [22:0]  fmaResult_fractOut = fmaResult_isSubnormal ? fmaResult_denormFract : fmaResult_rawIn_isInf ? 23'h0 : fmaResult_rawIn_sig[22:0];
  wire [8:0]   fmaResult_hi = {fmaResult_rawIn_sign, fmaResult_expOut};
  wire [31:0]  fmaResult = {fmaResult_hi, fmaResult_fractOut};
  wire         oneNaN = raw0_isNaN ^ raw1_isNaN;
  wire [31:0]  compareNaN = oneNaN ? (raw0_isNaN ? request_src_1 : request_src_0) : 32'h7FC00000;
  wire         hasNaN = raw0_isNaN | raw1_isNaN;
  wire         differentZeros = _compareModule_io_eq & (recIn1_rawIn_sign ^ recIn0_rawIn_sign);
  wire [2:0]   _GEN = {request_opcode[3], request_opcode[1:0]};
  wire         _sgnjSign_T_7 = request_opcode == 4'h3;
  wire         _sgnjSign_T_3 = request_opcode == 4'h2;
  wire         _otherResult_T_3 = request_opcode == 4'h4;
  wire [31:0]  compareResult =
    _GEN == 3'h4 & hasNaN
      ? compareNaN
      : _GEN == 3'h4
          ? (~(request_opcode[2]) & _compareModule_io_lt | request_opcode[2] & _compareModule_io_gt | differentZeros & (request_opcode[2] ^ recIn1_rawIn_sign) ? request_src_1 : request_src_0)
          : {31'h0,
             _sgnjSign_T_7
               ? _compareModule_io_lt | _compareModule_io_eq
               : request_opcode == 4'h5 ? _compareModule_io_gt | _compareModule_io_eq : _sgnjSign_T_3 ? _compareModule_io_lt : _otherResult_T_3 ? _compareModule_io_gt : request_opcode == 4'h0 ^ _compareModule_io_eq};
  wire         convertRtz = &(request_opcode[3:2]);
  wire         _convertFlags_T = request_opcode == 4'h8;
  wire [8:0]   convertResult_rawIn_exp = _intToFn_io_out[31:23];
  wire         convertResult_rawIn_isZero = convertResult_rawIn_exp[8:6] == 3'h0;
  wire         convertResult_rawIn_isZero_0 = convertResult_rawIn_isZero;
  wire         convertResult_rawIn_isSpecial = &(convertResult_rawIn_exp[8:7]);
  wire         convertResult_rawIn_isNaN = convertResult_rawIn_isSpecial & convertResult_rawIn_exp[6];
  wire         convertResult_rawIn_isInf = convertResult_rawIn_isSpecial & ~(convertResult_rawIn_exp[6]);
  wire         convertResult_rawIn_sign = _intToFn_io_out[32];
  wire [9:0]   convertResult_rawIn_sExp = {1'h0, convertResult_rawIn_exp};
  wire [24:0]  convertResult_rawIn_sig = {1'h0, ~convertResult_rawIn_isZero, _intToFn_io_out[22:0]};
  wire         convertResult_isSubnormal = $signed(convertResult_rawIn_sExp) < 10'sh82;
  wire [4:0]   convertResult_denormShiftDist = 5'h1 - convertResult_rawIn_sExp[4:0];
  wire [23:0]  _convertResult_denormFract_T_1 = convertResult_rawIn_sig[24:1] >> convertResult_denormShiftDist;
  wire [22:0]  convertResult_denormFract = _convertResult_denormFract_T_1[22:0];
  wire [7:0]   convertResult_expOut = (convertResult_isSubnormal ? 8'h0 : convertResult_rawIn_sExp[7:0] + 8'h7F) | {8{convertResult_rawIn_isNaN | convertResult_rawIn_isInf}};
  wire [22:0]  convertResult_fractOut = convertResult_isSubnormal ? convertResult_denormFract : convertResult_rawIn_isInf ? 23'h0 : convertResult_rawIn_sig[22:0];
  wire [8:0]   convertResult_hi = {convertResult_rawIn_sign, convertResult_expOut};
  wire [31:0]  convertResult = _convertFlags_T ? {convertResult_hi, convertResult_fractOut} : _fnToInt_io_out;
  wire [4:0]   convertFlags = _convertFlags_T ? _intToFn_io_exceptionFlags : {2'h0, _fnToInt_io_intExceptionFlags};
  wire         sgnjSign = otherEn & request_opcode == 4'h1 ? recIn0_rawIn_sign : otherEn & _sgnjSign_T_3 ? ~recIn0_rawIn_sign : otherEn & _sgnjSign_T_7 & (recIn0_rawIn_sign ^ recIn1_rawIn_sign);
  wire [31:0]  sgnjresult = {sgnjSign, request_src_1[30:0]};
  wire         in1classify_rawIn_isZero = in1classify_rawIn_exp[8:6] == 3'h0;
  wire         in1classify_rawIn_isZero_0 = in1classify_rawIn_isZero;
  wire         in1classify_rawIn_isSpecial = &(in1classify_rawIn_exp[8:7]);
  wire         in1classify_rawIn_isNaN = in1classify_rawIn_isSpecial & in1classify_rawIn_exp[6];
  wire         in1classify_rawIn_isInf = in1classify_rawIn_isSpecial & ~(in1classify_rawIn_exp[6]);
  wire [9:0]   in1classify_rawIn_sExp = {1'h0, in1classify_rawIn_exp};
  wire [24:0]  in1classify_rawIn_sig = {1'h0, ~in1classify_rawIn_isZero, recIn1[22:0]};
  wire         in1classify_isSigNaN = in1classify_rawIn_isNaN & ~(in1classify_rawIn_sig[22]);
  wire         in1classify_isFiniteNonzero = ~in1classify_rawIn_isNaN & ~in1classify_rawIn_isInf & ~in1classify_rawIn_isZero_0;
  wire         in1classify_isSubnormal = $signed(in1classify_rawIn_sExp) < 10'sh82;
  wire         _in1classify_T_23 = in1classify_rawIn_sign & in1classify_isFiniteNonzero;
  wire [9:0]   in1classify =
    {in1classify_rawIn_isNaN & ~in1classify_isSigNaN,
     in1classify_isSigNaN,
     ~in1classify_rawIn_sign & in1classify_rawIn_isInf,
     ~in1classify_rawIn_sign & in1classify_isFiniteNonzero & ~in1classify_isSubnormal,
     ~in1classify_rawIn_sign & in1classify_isFiniteNonzero & in1classify_isSubnormal,
     ~in1classify_rawIn_sign & in1classify_rawIn_isZero_0,
     in1classify_rawIn_sign & in1classify_rawIn_isZero_0,
     _in1classify_T_23 & in1classify_isSubnormal,
     _in1classify_T_23 & ~in1classify_isSubnormal,
     in1classify_rawIn_sign & in1classify_rawIn_isInf};
  wire         _otherResult_T_4 = request_opcode == 4'h6;
  wire         rec7En = _otherResult_T_4 & otherEn;
  wire         _otherResult_T_5 = request_opcode == 4'h7;
  wire         rsqrt7En = _otherResult_T_5 & otherEn;
  wire [31:0]  otherResult =
    request_opcode[3] ? convertResult : (|(request_opcode[3:2])) ? (_otherResult_T_3 ? {22'h0, in1classify} : _otherResult_T_4 ? _rec7Module_out_data : _otherResult_T_5 ? _rsqrt7Module_out_data : 32'h0) : sgnjresult;
  wire [4:0]   otherFlags = rec7En ? _rec7Module_out_exceptionFlags : rsqrt7En ? _rsqrt7Module_out_exceptionFlags : convertFlags;
  assign result = (fmaEn ? fmaResult : 32'h0) | (compareEn ? compareResult : 32'h0) | (otherEn ? otherResult : 32'h0);
  wire [4:0]   compareFlags;
  assign flags = (fmaEn ? _mulAddRecFN_io_exceptionFlags : 5'h0) | (compareEn ? compareFlags : 5'h0) | (otherEn ? otherFlags : 5'h0);
  assign response_adderMaskResp = result[0];
  always @(posedge clock) begin
    if (reset) begin
      requestReg_tag <= 2'h0;
      requestReg_sign <= 1'h0;
      requestReg_src_0 <= 32'h0;
      requestReg_src_1 <= 32'h0;
      requestReg_src_2 <= 32'h0;
      requestReg_opcode <= 4'h0;
      requestReg_unitSelet <= 2'h0;
      requestReg_floatMul <= 1'h0;
      requestReg_roundingMode <= 3'h0;
      requestReg_executeIndex <= 2'h0;
      requestRegValid <= 1'h0;
      request_pipeResponse_pipe_v <= 1'h0;
      request_pipeResponse_pipe_pipe_v <= 1'h0;
      request_pipeResponse_pipe_pipe_pipe_v <= 1'h0;
    end
    else begin
      if (requestIO_valid_0) begin
        requestReg_tag <= requestIO_bits_tag_0;
        requestReg_sign <= requestIO_bits_sign_0;
        requestReg_src_0 <= requestIO_bits_src_0_0;
        requestReg_src_1 <= requestIO_bits_src_1_0;
        requestReg_src_2 <= requestIO_bits_src_2_0;
        requestReg_opcode <= requestIO_bits_opcode_0;
        requestReg_unitSelet <= requestIO_bits_unitSelet_0;
        requestReg_floatMul <= requestIO_bits_floatMul_0;
        requestReg_roundingMode <= requestIO_bits_roundingMode_0;
        requestReg_executeIndex <= requestIO_bits_executeIndex_0;
      end
      requestRegValid <= requestIO_valid_0;
      request_pipeResponse_pipe_v <= request_responseValidWire;
      request_pipeResponse_pipe_pipe_v <= request_pipeResponse_pipe_v;
      request_pipeResponse_pipe_pipe_pipe_v <= request_pipeResponse_pipe_pipe_v;
    end
    if (request_responseValidWire) begin
      request_pipeResponse_pipe_b_tag <= request_responseWire_tag;
      request_pipeResponse_pipe_b_data <= request_responseWire_data;
      request_pipeResponse_pipe_b_adderMaskResp <= request_responseWire_adderMaskResp;
      request_pipeResponse_pipe_b_exceptionFlags <= request_responseWire_exceptionFlags;
      request_pipeResponse_pipe_b_executeIndex <= request_responseWire_executeIndex;
    end
    if (request_pipeResponse_pipe_v) begin
      request_pipeResponse_pipe_pipe_b_tag <= request_pipeResponse_pipe_b_tag;
      request_pipeResponse_pipe_pipe_b_data <= request_pipeResponse_pipe_b_data;
      request_pipeResponse_pipe_pipe_b_adderMaskResp <= request_pipeResponse_pipe_b_adderMaskResp;
      request_pipeResponse_pipe_pipe_b_exceptionFlags <= request_pipeResponse_pipe_b_exceptionFlags;
      request_pipeResponse_pipe_pipe_b_executeIndex <= request_pipeResponse_pipe_b_executeIndex;
    end
    if (request_pipeResponse_pipe_pipe_v) begin
      request_pipeResponse_pipe_pipe_pipe_b_tag <= request_pipeResponse_pipe_pipe_b_tag;
      request_pipeResponse_pipe_pipe_pipe_b_data <= request_pipeResponse_pipe_pipe_b_data;
      request_pipeResponse_pipe_pipe_pipe_b_adderMaskResp <= request_pipeResponse_pipe_pipe_b_adderMaskResp;
      request_pipeResponse_pipe_pipe_pipe_b_exceptionFlags <= request_pipeResponse_pipe_pipe_b_exceptionFlags;
      request_pipeResponse_pipe_pipe_pipe_b_executeIndex <= request_pipeResponse_pipe_pipe_b_executeIndex;
    end
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:7];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [3:0] i = 4'h0; i < 4'h8; i += 4'h1) begin
          _RANDOM[i[2:0]] = `RANDOM;
        end
        requestReg_tag = _RANDOM[3'h0][1:0];
        requestReg_sign = _RANDOM[3'h0][2];
        requestReg_src_0 = {_RANDOM[3'h0][31:3], _RANDOM[3'h1][2:0]};
        requestReg_src_1 = {_RANDOM[3'h1][31:3], _RANDOM[3'h2][2:0]};
        requestReg_src_2 = {_RANDOM[3'h2][31:3], _RANDOM[3'h3][2:0]};
        requestReg_opcode = _RANDOM[3'h3][6:3];
        requestReg_unitSelet = _RANDOM[3'h3][8:7];
        requestReg_floatMul = _RANDOM[3'h3][9];
        requestReg_roundingMode = _RANDOM[3'h3][12:10];
        requestReg_executeIndex = _RANDOM[3'h3][14:13];
        requestRegValid = _RANDOM[3'h3][15];
        request_pipeResponse_pipe_v = _RANDOM[3'h3][16];
        request_pipeResponse_pipe_b_tag = _RANDOM[3'h3][18:17];
        request_pipeResponse_pipe_b_data = {_RANDOM[3'h3][31:19], _RANDOM[3'h4][18:0]};
        request_pipeResponse_pipe_b_adderMaskResp = _RANDOM[3'h4][19];
        request_pipeResponse_pipe_b_exceptionFlags = _RANDOM[3'h4][24:20];
        request_pipeResponse_pipe_b_executeIndex = _RANDOM[3'h4][26:25];
        request_pipeResponse_pipe_pipe_v = _RANDOM[3'h4][27];
        request_pipeResponse_pipe_pipe_b_tag = _RANDOM[3'h4][29:28];
        request_pipeResponse_pipe_pipe_b_data = {_RANDOM[3'h4][31:30], _RANDOM[3'h5][29:0]};
        request_pipeResponse_pipe_pipe_b_adderMaskResp = _RANDOM[3'h5][30];
        request_pipeResponse_pipe_pipe_b_exceptionFlags = {_RANDOM[3'h5][31], _RANDOM[3'h6][3:0]};
        request_pipeResponse_pipe_pipe_b_executeIndex = _RANDOM[3'h6][5:4];
        request_pipeResponse_pipe_pipe_pipe_v = _RANDOM[3'h6][6];
        request_pipeResponse_pipe_pipe_pipe_b_tag = _RANDOM[3'h6][8:7];
        request_pipeResponse_pipe_pipe_pipe_b_data = {_RANDOM[3'h6][31:9], _RANDOM[3'h7][8:0]};
        request_pipeResponse_pipe_pipe_pipe_b_adderMaskResp = _RANDOM[3'h7][9];
        request_pipeResponse_pipe_pipe_pipe_b_exceptionFlags = _RANDOM[3'h7][14:10];
        request_pipeResponse_pipe_pipe_pipe_b_executeIndex = _RANDOM[3'h7][16:15];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  MulAddRecFN_e8_s24 mulAddRecFN (
    .io_op             (request_opcode[1:0]),
    .io_a              (fmaIn0),
    .io_b              (fmaIn1),
    .io_c              (fmaIn2),
    .io_roundingMode   (request_roundingMode),
    .io_out            (_mulAddRecFN_io_out),
    .io_exceptionFlags (_mulAddRecFN_io_exceptionFlags)
  );
  CompareRecFN compareModule (
    .io_a              (recIn1),
    .io_b              (recIn0),
    .io_signaling      (request_opcode[3:1] == 3'h1 | request_opcode[3:1] == 3'h2),
    .io_lt             (_compareModule_io_lt),
    .io_eq             (_compareModule_io_eq),
    .io_gt             (_compareModule_io_gt),
    .io_exceptionFlags (compareFlags)
  );
  INToRecFN_i32_e8_s24 intToFn (
    .io_signedIn       (request_sign),
    .io_in             (request_src_1),
    .io_roundingMode   (request_roundingMode),
    .io_out            (_intToFn_io_out),
    .io_exceptionFlags (_intToFn_io_exceptionFlags)
  );
  RecFNToIN_e8_s24_i32 fnToInt (
    .io_in                (recIn1),
    .io_roundingMode      (convertRtz ? 3'h1 : request_roundingMode),
    .io_signedOut         (~(request_opcode[3] & request_opcode[0])),
    .io_out               (_fnToInt_io_out),
    .io_intExceptionFlags (_fnToInt_io_intExceptionFlags)
  );
  Rec7Fn rec7Module (
    .in_data            (request_src_1),
    .in_classifyIn      (in1classify),
    .in_roundingMode    (request_roundingMode),
    .out_data           (_rec7Module_out_data),
    .out_exceptionFlags (_rec7Module_out_exceptionFlags)
  );
  Rsqrt7Fn rsqrt7Module (
    .in_data            (request_src_1),
    .in_classifyIn      (in1classify),
    .out_data           (_rsqrt7Module_out_data),
    .out_exceptionFlags (_rsqrt7Module_out_exceptionFlags)
  );
  assign responseIO_valid = responseIO_valid_0;
  assign responseIO_bits_tag = responseIO_bits_tag_0;
  assign responseIO_bits_data = responseIO_bits_data_0;
  assign responseIO_bits_adderMaskResp = responseIO_bits_adderMaskResp_0;
  assign responseIO_bits_exceptionFlags = responseIO_bits_exceptionFlags_0;
  assign responseIO_bits_executeIndex = responseIO_bits_executeIndex_0;
endmodule

