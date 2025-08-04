
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
module LaneAdder(
  input         clock,
                reset,
                requestIO_valid,
  input  [1:0]  requestIO_bits_tag,
  input  [31:0] requestIO_bits_src_0,
                requestIO_bits_src_1,
  input  [3:0]  requestIO_bits_mask,
                requestIO_bits_opcode,
  input         requestIO_bits_sign,
                requestIO_bits_reverse,
                requestIO_bits_average,
                requestIO_bits_saturate,
  input  [1:0]  requestIO_bits_vxrm,
                requestIO_bits_vSew,
                requestIO_bits_executeIndex,
  output        responseIO_valid,
  output [1:0]  responseIO_bits_tag,
  output [31:0] responseIO_bits_data,
  output [3:0]  responseIO_bits_adderMaskResp,
                responseIO_bits_vxsat,
  output [1:0]  responseIO_bits_executeIndex
);

  wire [31:0] _adder_z;
  wire [3:0]  _adder_cout;
  wire        attributeSelect_3_6;
  wire        attributeSelect_3_5;
  wire        attributeSelect_3_3;
  wire        attributeSelect_3_2;
  wire        attributeSelect_2_6;
  wire        attributeSelect_2_5;
  wire        attributeSelect_2_3;
  wire        attributeSelect_2_2;
  wire        attributeSelect_1_6;
  wire        attributeSelect_1_5;
  wire        attributeSelect_1_3;
  wire        attributeSelect_1_2;
  wire        attributeSelect_0_6;
  wire        attributeSelect_0_5;
  wire        attributeSelect_0_3;
  wire        attributeSelect_0_2;
  wire        requestIO_valid_0 = requestIO_valid;
  wire [1:0]  requestIO_bits_tag_0 = requestIO_bits_tag;
  wire [31:0] requestIO_bits_src_0_0 = requestIO_bits_src_0;
  wire [31:0] requestIO_bits_src_1_0 = requestIO_bits_src_1;
  wire [3:0]  requestIO_bits_mask_0 = requestIO_bits_mask;
  wire [3:0]  requestIO_bits_opcode_0 = requestIO_bits_opcode;
  wire        requestIO_bits_sign_0 = requestIO_bits_sign;
  wire        requestIO_bits_reverse_0 = requestIO_bits_reverse;
  wire        requestIO_bits_average_0 = requestIO_bits_average;
  wire        requestIO_bits_saturate_0 = requestIO_bits_saturate;
  wire [1:0]  requestIO_bits_vxrm_0 = requestIO_bits_vxrm;
  wire [1:0]  requestIO_bits_vSew_0 = requestIO_bits_vSew;
  wire [1:0]  requestIO_bits_executeIndex_0 = requestIO_bits_executeIndex;
  wire [1:0]  response_tag = 2'h0;
  wire        requestIO_ready = 1'h1;
  wire        responseIO_ready = 1'h1;
  wire        request_pipeResponse_valid;
  wire [1:0]  request_pipeResponse_bits_tag;
  wire [31:0] request_pipeResponse_bits_data;
  wire [3:0]  request_pipeResponse_bits_adderMaskResp;
  wire [3:0]  request_pipeResponse_bits_vxsat;
  wire [1:0]  request_pipeResponse_bits_executeIndex;
  reg  [1:0]  requestReg_tag;
  wire [1:0]  request_responseWire_tag = requestReg_tag;
  reg  [31:0] requestReg_src_0;
  reg  [31:0] requestReg_src_1;
  reg  [3:0]  requestReg_mask;
  reg  [3:0]  requestReg_opcode;
  reg         requestReg_sign;
  reg         requestReg_reverse;
  reg         requestReg_average;
  reg         requestReg_saturate;
  reg  [1:0]  requestReg_vxrm;
  reg  [1:0]  requestReg_vSew;
  reg  [1:0]  requestReg_executeIndex;
  reg         requestRegValid;
  wire        vfuRequestFire = requestRegValid;
  wire        request_responseValidWire = requestRegValid;
  wire [1:0]  request_executeIndex;
  wire [3:0]  response_vxsat;
  wire [1:0]  response_executeIndex;
  wire [5:0]  request_responseWire_lo = {response_vxsat, response_executeIndex};
  wire [31:0] response_data;
  wire [33:0] request_responseWire_hi_hi = {2'h0, response_data};
  wire [3:0]  response_adderMaskResp;
  wire [37:0] request_responseWire_hi = {request_responseWire_hi_hi, response_adderMaskResp};
  wire [1:0]  request_responseWire_executeIndex = request_responseWire_lo[1:0];
  wire [3:0]  request_responseWire_vxsat = request_responseWire_lo[5:2];
  wire [3:0]  request_responseWire_adderMaskResp = request_responseWire_hi[3:0];
  wire [31:0] request_responseWire_data = request_responseWire_hi[35:4];
  reg         request_pipeResponse_pipe_v;
  assign request_pipeResponse_valid = request_pipeResponse_pipe_v;
  reg  [1:0]  request_pipeResponse_pipe_b_tag;
  assign request_pipeResponse_bits_tag = request_pipeResponse_pipe_b_tag;
  reg  [31:0] request_pipeResponse_pipe_b_data;
  assign request_pipeResponse_bits_data = request_pipeResponse_pipe_b_data;
  reg  [3:0]  request_pipeResponse_pipe_b_adderMaskResp;
  assign request_pipeResponse_bits_adderMaskResp = request_pipeResponse_pipe_b_adderMaskResp;
  reg  [3:0]  request_pipeResponse_pipe_b_vxsat;
  assign request_pipeResponse_bits_vxsat = request_pipeResponse_pipe_b_vxsat;
  reg  [1:0]  request_pipeResponse_pipe_b_executeIndex;
  assign request_pipeResponse_bits_executeIndex = request_pipeResponse_pipe_b_executeIndex;
  wire        responseIO_valid_0 = request_pipeResponse_valid;
  wire [1:0]  responseIO_bits_tag_0 = request_pipeResponse_bits_tag;
  wire [31:0] responseIO_bits_data_0 = request_pipeResponse_bits_data;
  wire [3:0]  responseIO_bits_adderMaskResp_0 = request_pipeResponse_bits_adderMaskResp;
  wire [3:0]  responseIO_bits_vxsat_0 = request_pipeResponse_bits_vxsat;
  wire [1:0]  responseIO_bits_executeIndex_0 = request_pipeResponse_bits_executeIndex;
  wire [3:0]  request_lo_lo = {requestReg_vSew, requestReg_executeIndex};
  wire [1:0]  request_lo_hi_hi = {requestReg_average, requestReg_saturate};
  wire [3:0]  request_lo_hi = {request_lo_hi_hi, requestReg_vxrm};
  wire [7:0]  request_lo = {request_lo_hi, request_lo_lo};
  wire [4:0]  request_hi_lo_hi = {requestReg_opcode, requestReg_sign};
  wire [5:0]  request_hi_lo = {request_hi_lo_hi, requestReg_reverse};
  wire [65:0] request_hi_hi_hi = {requestReg_tag, requestReg_src_1, requestReg_src_0};
  wire [69:0] request_hi_hi = {request_hi_hi_hi, requestReg_mask};
  wire [75:0] request_hi = {request_hi_hi, request_hi_lo};
  assign request_executeIndex = request_lo[1:0];
  wire [1:0]  request_vSew = request_lo[3:2];
  wire [1:0]  request_vxrm = request_lo[5:4];
  wire        request_saturate = request_lo[6];
  wire        request_average = request_lo[7];
  wire        request_reverse = request_hi[0];
  wire        request_sign = request_hi[1];
  wire [3:0]  request_opcode = request_hi[5:2];
  wire [3:0]  request_mask = request_hi[9:6];
  wire [31:0] request_src_0 = request_hi[41:10];
  wire [31:0] request_src_1 = request_hi[73:42];
  wire [1:0]  request_tag = request_hi[75:74];
  assign response_executeIndex = request_executeIndex;
  wire [15:0] _uopOH_T = 16'h1 << request_opcode;
  wire [11:0] uopOH = _uopOH_T[11:0];
  wire        isSub = ~(uopOH[0] | uopOH[10]);
  wire [31:0] subOperation0 = {32{isSub & ~request_reverse}} ^ request_src_0;
  wire        _attributeVec_operation1Sign_T_7 = isSub & request_reverse;
  wire [31:0] subOperation1 = {32{_attributeVec_operation1Sign_T_7}} ^ request_src_1;
  wire [3:0]  operation2 = {4{isSub}} ^ request_mask;
  wire        vSewOrR = |request_vSew;
  wire [3:0]  _view__sew_T = 4'h1 << request_vSew;
  wire [2:0]  vSew1H = _view__sew_T[2:0];
  wire [1:0]  roundingBitsVec_roundingTail = subOperation0[1:0] + subOperation1[1:0] + {1'h0, operation2[0]};
  wire [3:0]  _GEN = 4'h1 << request_vxrm;
  wire [3:0]  roundingBitsVec_vxrmOH;
  assign roundingBitsVec_vxrmOH = _GEN;
  wire [3:0]  roundingBitsVec_vxrmOH_1;
  assign roundingBitsVec_vxrmOH_1 = _GEN;
  wire [3:0]  roundingBitsVec_vxrmOH_2;
  assign roundingBitsVec_vxrmOH_2 = _GEN;
  wire [3:0]  roundingBitsVec_vxrmOH_3;
  assign roundingBitsVec_vxrmOH_3 = _GEN;
  wire        roundingBitsVec_0 =
    roundingBitsVec_vxrmOH[0] & roundingBitsVec_roundingTail[0] | roundingBitsVec_vxrmOH[1] & (&roundingBitsVec_roundingTail) | roundingBitsVec_vxrmOH[3] & roundingBitsVec_roundingTail[0] & ~(roundingBitsVec_roundingTail[1]);
  wire [1:0]  roundingBitsVec_roundingTail_1 = subOperation0[9:8] + subOperation1[9:8] + {1'h0, operation2[1]};
  wire        roundingBitsVec_1 =
    roundingBitsVec_vxrmOH_1[0] & roundingBitsVec_roundingTail_1[0] | roundingBitsVec_vxrmOH_1[1] & (&roundingBitsVec_roundingTail_1) | roundingBitsVec_vxrmOH_1[3] & roundingBitsVec_roundingTail_1[0] & ~(roundingBitsVec_roundingTail_1[1]);
  wire [1:0]  roundingBitsVec_roundingTail_2 = subOperation0[17:16] + subOperation1[17:16] + {1'h0, operation2[2]};
  wire        roundingBitsVec_2 =
    roundingBitsVec_vxrmOH_2[0] & roundingBitsVec_roundingTail_2[0] | roundingBitsVec_vxrmOH_2[1] & (&roundingBitsVec_roundingTail_2) | roundingBitsVec_vxrmOH_2[3] & roundingBitsVec_roundingTail_2[0] & ~(roundingBitsVec_roundingTail_2[1]);
  wire [1:0]  roundingBitsVec_roundingTail_3 = subOperation0[25:24] + subOperation1[25:24] + {1'h0, operation2[3]};
  wire        roundingBitsVec_3 =
    roundingBitsVec_vxrmOH_3[0] & roundingBitsVec_roundingTail_3[0] | roundingBitsVec_vxrmOH_3[1] & (&roundingBitsVec_roundingTail_3) | roundingBitsVec_vxrmOH_3[3] & roundingBitsVec_roundingTail_3[0] & ~(roundingBitsVec_roundingTail_3[1]);
  wire [31:0] operation2ForAverage =
    {6'h0, vSew1H[0] ? {roundingBitsVec_3, operation2[3]} : 2'h0, 6'h0, vSew1H[2] ? 2'h0 : {roundingBitsVec_2, operation2[2]}, 6'h0, vSew1H[0] ? {roundingBitsVec_1, operation2[1]} : 2'h0, 6'h0, roundingBitsVec_0, operation2[0]};
  wire [31:0] xor_0 = subOperation0 ^ subOperation1;
  wire [31:0] s = xor_0 ^ operation2ForAverage;
  wire [31:0] c = xor_0 & operation2ForAverage | subOperation0 & subOperation1;
  wire [30:0] averageSum = s[31:1];
  wire [30:0] operation1ForAverage = {averageSum[30:24], ~(vSew1H[0]) & averageSum[23], averageSum[22:16], vSew1H[2] & averageSum[15], averageSum[14:8], ~(vSew1H[0]) & averageSum[7], averageSum[6:0]};
  wire [3:0]  adderCarryOut = {vSew1H[0] & _adder_cout[3] | vSew1H[1] & _adder_cout[1] | vSew1H[2] & _adder_cout[0], _adder_cout[2], vSew1H[0] ? _adder_cout[1] : _adder_cout[0], _adder_cout[0]};
  wire [3:0]  isFirstBlock = {1'h1, vSew1H[0], ~(vSew1H[2]), vSew1H[0]};
  wire [7:0]  adderResultVec_0 = _adder_z[7:0];
  wire [7:0]  adderResultVec_1 = _adder_z[15:8];
  wire [7:0]  adderResultVec_2 = _adder_z[23:16];
  wire [7:0]  adderResultVec_3 = _adder_z[31:24];
  wire        isZero_0 = adderResultVec_0 == 8'h0;
  wire        isZero_1 = adderResultVec_1 == 8'h0;
  wire        isZero_2 = adderResultVec_2 == 8'h0;
  wire        isZero_3 = adderResultVec_3 == 8'h0;
  wire        isZero01 = isZero_0 & isZero_1;
  wire        isZero23 = isZero_2 & isZero_3;
  wire        allIsZero = isZero01 & isZero23;
  wire        equalVec_0 = vSew1H[0] & isZero_0 | vSew1H[1] & isZero01 | vSew1H[2] & allIsZero;
  wire        equalVec_1 = vSew1H[0] & isZero_1 | vSew1H[1] & isZero01 | vSew1H[2] & allIsZero;
  wire        equalVec_2 = vSew1H[0] & isZero_2 | vSew1H[1] & isZero23 | vSew1H[2] & allIsZero;
  wire        equalVec_3 = vSew1H[0] & isZero_3 | vSew1H[1] & isZero23 | vSew1H[2] & allIsZero;
  wire        attributeVec_sourceSign0 = request_src_0[7];
  wire        attributeVec_sourceSign1 = request_src_1[7];
  wire        attributeVec_operation0Sign = attributeVec_sourceSign0 & request_sign ^ isSub & ~request_reverse;
  wire        attributeVec_operation1Sign = attributeVec_sourceSign1 & request_sign ^ _attributeVec_operation1Sign_T_7;
  wire        attributeVec_resultSign = adderResultVec_0[7];
  wire        attributeVec_uIntLess = attributeVec_sourceSign0 ^ attributeVec_sourceSign1 ? attributeVec_sourceSign0 : attributeVec_resultSign;
  wire        attributeVec_allNegative = attributeVec_operation0Sign & attributeVec_operation1Sign;
  wire        attributeVec_lowerOverflow = attributeVec_allNegative & ~attributeVec_resultSign | isSub & ~request_sign & attributeVec_uIntLess;
  wire        attributeVec_0_0 = attributeVec_lowerOverflow;
  wire        attributeVec_upperOverflow = request_sign ? ~attributeVec_operation0Sign & ~attributeVec_operation1Sign & attributeVec_resultSign : adderCarryOut[0] & ~isSub;
  wire        attributeVec_0_1 = attributeVec_upperOverflow;
  wire        attributeVec_less = request_sign ? attributeVec_resultSign & ~attributeVec_upperOverflow | attributeVec_lowerOverflow : attributeVec_uIntLess;
  wire        attributeVec_0_3 = attributeVec_less;
  wire        attributeVec_overflow = (attributeVec_upperOverflow | attributeVec_lowerOverflow) & request_saturate;
  wire        attributeVec_0_2 = attributeVec_overflow;
  wire        _attributeVec_maskSelect_T_4 = attributeVec_less | equalVec_0;
  wire        attributeVec_maskSelect =
    uopOH[2] & attributeVec_less | uopOH[3] & _attributeVec_maskSelect_T_4 | uopOH[4] & ~_attributeVec_maskSelect_T_4 | uopOH[5] & ~attributeVec_less | uopOH[8] & equalVec_0 | uopOH[9] & ~equalVec_0 | uopOH[10] & attributeVec_upperOverflow
    | uopOH[11] & attributeVec_lowerOverflow;
  wire        attributeVec_0_4 = attributeVec_maskSelect;
  wire        attributeVec_anyNegative = attributeVec_operation0Sign | attributeVec_operation1Sign;
  wire        attributeVec_0_5 = attributeVec_anyNegative;
  wire        attributeVec_0_6 = attributeVec_allNegative;
  wire        attributeVec_sourceSign0_1 = request_src_0[15];
  wire        attributeVec_sourceSign1_1 = request_src_1[15];
  wire        attributeVec_operation0Sign_1 = attributeVec_sourceSign0_1 & request_sign ^ isSub & ~request_reverse;
  wire        attributeVec_operation1Sign_1 = attributeVec_sourceSign1_1 & request_sign ^ _attributeVec_operation1Sign_T_7;
  wire        attributeVec_resultSign_1 = adderResultVec_1[7];
  wire        attributeVec_uIntLess_1 = attributeVec_sourceSign0_1 ^ attributeVec_sourceSign1_1 ? attributeVec_sourceSign0_1 : attributeVec_resultSign_1;
  wire        attributeVec_allNegative_1 = attributeVec_operation0Sign_1 & attributeVec_operation1Sign_1;
  wire        attributeVec_lowerOverflow_1 = attributeVec_allNegative_1 & ~attributeVec_resultSign_1 | isSub & ~request_sign & attributeVec_uIntLess_1;
  wire        attributeVec_1_0 = attributeVec_lowerOverflow_1;
  wire        attributeVec_upperOverflow_1 = request_sign ? ~attributeVec_operation0Sign_1 & ~attributeVec_operation1Sign_1 & attributeVec_resultSign_1 : adderCarryOut[1] & ~isSub;
  wire        attributeVec_1_1 = attributeVec_upperOverflow_1;
  wire        attributeVec_less_1 = request_sign ? attributeVec_resultSign_1 & ~attributeVec_upperOverflow_1 | attributeVec_lowerOverflow_1 : attributeVec_uIntLess_1;
  wire        attributeVec_1_3 = attributeVec_less_1;
  wire        attributeVec_overflow_1 = (attributeVec_upperOverflow_1 | attributeVec_lowerOverflow_1) & request_saturate;
  wire        attributeVec_1_2 = attributeVec_overflow_1;
  wire        _attributeVec_maskSelect_T_35 = attributeVec_less_1 | equalVec_1;
  wire        attributeVec_maskSelect_1 =
    uopOH[2] & attributeVec_less_1 | uopOH[3] & _attributeVec_maskSelect_T_35 | uopOH[4] & ~_attributeVec_maskSelect_T_35 | uopOH[5] & ~attributeVec_less_1 | uopOH[8] & equalVec_1 | uopOH[9] & ~equalVec_1 | uopOH[10]
    & attributeVec_upperOverflow_1 | uopOH[11] & attributeVec_lowerOverflow_1;
  wire        attributeVec_1_4 = attributeVec_maskSelect_1;
  wire        attributeVec_anyNegative_1 = attributeVec_operation0Sign_1 | attributeVec_operation1Sign_1;
  wire        attributeVec_1_5 = attributeVec_anyNegative_1;
  wire        attributeVec_1_6 = attributeVec_allNegative_1;
  wire        attributeVec_sourceSign0_2 = request_src_0[23];
  wire        attributeVec_sourceSign1_2 = request_src_1[23];
  wire        attributeVec_operation0Sign_2 = attributeVec_sourceSign0_2 & request_sign ^ isSub & ~request_reverse;
  wire        attributeVec_operation1Sign_2 = attributeVec_sourceSign1_2 & request_sign ^ _attributeVec_operation1Sign_T_7;
  wire        attributeVec_resultSign_2 = adderResultVec_2[7];
  wire        attributeVec_uIntLess_2 = attributeVec_sourceSign0_2 ^ attributeVec_sourceSign1_2 ? attributeVec_sourceSign0_2 : attributeVec_resultSign_2;
  wire        attributeVec_allNegative_2 = attributeVec_operation0Sign_2 & attributeVec_operation1Sign_2;
  wire        attributeVec_lowerOverflow_2 = attributeVec_allNegative_2 & ~attributeVec_resultSign_2 | isSub & ~request_sign & attributeVec_uIntLess_2;
  wire        attributeVec_2_0 = attributeVec_lowerOverflow_2;
  wire        attributeVec_upperOverflow_2 = request_sign ? ~attributeVec_operation0Sign_2 & ~attributeVec_operation1Sign_2 & attributeVec_resultSign_2 : adderCarryOut[2] & ~isSub;
  wire        attributeVec_2_1 = attributeVec_upperOverflow_2;
  wire        attributeVec_less_2 = request_sign ? attributeVec_resultSign_2 & ~attributeVec_upperOverflow_2 | attributeVec_lowerOverflow_2 : attributeVec_uIntLess_2;
  wire        attributeVec_2_3 = attributeVec_less_2;
  wire        attributeVec_overflow_2 = (attributeVec_upperOverflow_2 | attributeVec_lowerOverflow_2) & request_saturate;
  wire        attributeVec_2_2 = attributeVec_overflow_2;
  wire        _attributeVec_maskSelect_T_66 = attributeVec_less_2 | equalVec_2;
  wire        attributeVec_maskSelect_2 =
    uopOH[2] & attributeVec_less_2 | uopOH[3] & _attributeVec_maskSelect_T_66 | uopOH[4] & ~_attributeVec_maskSelect_T_66 | uopOH[5] & ~attributeVec_less_2 | uopOH[8] & equalVec_2 | uopOH[9] & ~equalVec_2 | uopOH[10]
    & attributeVec_upperOverflow_2 | uopOH[11] & attributeVec_lowerOverflow_2;
  wire        attributeVec_2_4 = attributeVec_maskSelect_2;
  wire        attributeVec_anyNegative_2 = attributeVec_operation0Sign_2 | attributeVec_operation1Sign_2;
  wire        attributeVec_2_5 = attributeVec_anyNegative_2;
  wire        attributeVec_2_6 = attributeVec_allNegative_2;
  wire        attributeVec_sourceSign0_3 = request_src_0[31];
  wire        attributeVec_sourceSign1_3 = request_src_1[31];
  wire        attributeVec_operation0Sign_3 = attributeVec_sourceSign0_3 & request_sign ^ isSub & ~request_reverse;
  wire        attributeVec_operation1Sign_3 = attributeVec_sourceSign1_3 & request_sign ^ _attributeVec_operation1Sign_T_7;
  wire        attributeVec_resultSign_3 = adderResultVec_3[7];
  wire        attributeVec_uIntLess_3 = attributeVec_sourceSign0_3 ^ attributeVec_sourceSign1_3 ? attributeVec_sourceSign0_3 : attributeVec_resultSign_3;
  wire        attributeVec_allNegative_3 = attributeVec_operation0Sign_3 & attributeVec_operation1Sign_3;
  wire        attributeVec_lowerOverflow_3 = attributeVec_allNegative_3 & ~attributeVec_resultSign_3 | isSub & ~request_sign & attributeVec_uIntLess_3;
  wire        attributeVec_3_0 = attributeVec_lowerOverflow_3;
  wire        attributeVec_upperOverflow_3 = request_sign ? ~attributeVec_operation0Sign_3 & ~attributeVec_operation1Sign_3 & attributeVec_resultSign_3 : adderCarryOut[3] & ~isSub;
  wire        attributeVec_3_1 = attributeVec_upperOverflow_3;
  wire        attributeVec_less_3 = request_sign ? attributeVec_resultSign_3 & ~attributeVec_upperOverflow_3 | attributeVec_lowerOverflow_3 : attributeVec_uIntLess_3;
  wire        attributeVec_3_3 = attributeVec_less_3;
  wire        attributeVec_overflow_3 = (attributeVec_upperOverflow_3 | attributeVec_lowerOverflow_3) & request_saturate;
  wire        attributeVec_3_2 = attributeVec_overflow_3;
  wire        _attributeVec_maskSelect_T_97 = attributeVec_less_3 | equalVec_3;
  wire        attributeVec_maskSelect_3 =
    uopOH[2] & attributeVec_less_3 | uopOH[3] & _attributeVec_maskSelect_T_97 | uopOH[4] & ~_attributeVec_maskSelect_T_97 | uopOH[5] & ~attributeVec_less_3 | uopOH[8] & equalVec_3 | uopOH[9] & ~equalVec_3 | uopOH[10]
    & attributeVec_upperOverflow_3 | uopOH[11] & attributeVec_lowerOverflow_3;
  wire        attributeVec_3_4 = attributeVec_maskSelect_3;
  wire        attributeVec_anyNegative_3 = attributeVec_operation0Sign_3 | attributeVec_operation1Sign_3;
  wire        attributeVec_3_5 = attributeVec_anyNegative_3;
  wire        attributeVec_3_6 = attributeVec_allNegative_3;
  wire        overflowVec_0 = attributeSelect_0_2;
  wire        lessVec_0 = attributeSelect_0_3;
  wire        anyNegativeVec_0 = attributeSelect_0_5;
  wire        allNegativeVec_0 = attributeSelect_0_6;
  wire        overflowVec_1 = attributeSelect_1_2;
  wire        lessVec_1 = attributeSelect_1_3;
  wire        anyNegativeVec_1 = attributeSelect_1_5;
  wire        allNegativeVec_1 = attributeSelect_1_6;
  wire        overflowVec_2 = attributeSelect_2_2;
  wire        lessVec_2 = attributeSelect_2_3;
  wire        anyNegativeVec_2 = attributeSelect_2_5;
  wire        allNegativeVec_2 = attributeSelect_2_6;
  wire        overflowVec_3 = attributeSelect_3_2;
  wire        lessVec_3 = attributeSelect_3_3;
  wire        anyNegativeVec_3 = attributeSelect_3_5;
  wire        allNegativeVec_3 = attributeSelect_3_6;
  wire        attributeSelect_0_0 = vSew1H[0] & attributeVec_0_0 | vSew1H[1] & attributeVec_1_0 | vSew1H[2] & attributeVec_3_0;
  wire        attributeSelect_0_1 = vSew1H[0] & attributeVec_0_1 | vSew1H[1] & attributeVec_1_1 | vSew1H[2] & attributeVec_3_1;
  assign attributeSelect_0_2 = vSew1H[0] & attributeVec_0_2 | vSew1H[1] & attributeVec_1_2 | vSew1H[2] & attributeVec_3_2;
  assign attributeSelect_0_3 = vSew1H[0] & attributeVec_0_3 | vSew1H[1] & attributeVec_1_3 | vSew1H[2] & attributeVec_3_3;
  wire        attributeSelect_0_4 = vSew1H[0] & attributeVec_0_4 | vSew1H[1] & attributeVec_1_4 | vSew1H[2] & attributeVec_3_4;
  assign attributeSelect_0_5 = vSew1H[0] & attributeVec_0_5 | vSew1H[1] & attributeVec_1_5 | vSew1H[2] & attributeVec_3_5;
  assign attributeSelect_0_6 = vSew1H[0] & attributeVec_0_6 | vSew1H[1] & attributeVec_1_6 | vSew1H[2] & attributeVec_3_6;
  wire        attributeSelect_1_0 = vSew1H[0] & attributeVec_1_0 | vSew1H[1] & attributeVec_1_0 | vSew1H[2] & attributeVec_3_0;
  wire        attributeSelect_1_1 = vSew1H[0] & attributeVec_1_1 | vSew1H[1] & attributeVec_1_1 | vSew1H[2] & attributeVec_3_1;
  assign attributeSelect_1_2 = vSew1H[0] & attributeVec_1_2 | vSew1H[1] & attributeVec_1_2 | vSew1H[2] & attributeVec_3_2;
  assign attributeSelect_1_3 = vSew1H[0] & attributeVec_1_3 | vSew1H[1] & attributeVec_1_3 | vSew1H[2] & attributeVec_3_3;
  wire        attributeSelect_1_4 = vSew1H[0] & attributeVec_1_4 | vSew1H[1] & attributeVec_1_4 | vSew1H[2] & attributeVec_3_4;
  assign attributeSelect_1_5 = vSew1H[0] & attributeVec_1_5 | vSew1H[1] & attributeVec_1_5 | vSew1H[2] & attributeVec_3_5;
  assign attributeSelect_1_6 = vSew1H[0] & attributeVec_1_6 | vSew1H[1] & attributeVec_1_6 | vSew1H[2] & attributeVec_3_6;
  wire        attributeSelect_2_0 = vSew1H[0] & attributeVec_2_0 | vSew1H[1] & attributeVec_3_0 | vSew1H[2] & attributeVec_3_0;
  wire        attributeSelect_2_1 = vSew1H[0] & attributeVec_2_1 | vSew1H[1] & attributeVec_3_1 | vSew1H[2] & attributeVec_3_1;
  assign attributeSelect_2_2 = vSew1H[0] & attributeVec_2_2 | vSew1H[1] & attributeVec_3_2 | vSew1H[2] & attributeVec_3_2;
  assign attributeSelect_2_3 = vSew1H[0] & attributeVec_2_3 | vSew1H[1] & attributeVec_3_3 | vSew1H[2] & attributeVec_3_3;
  wire        attributeSelect_2_4 = vSew1H[0] & attributeVec_2_4 | vSew1H[1] & attributeVec_3_4 | vSew1H[2] & attributeVec_3_4;
  assign attributeSelect_2_5 = vSew1H[0] & attributeVec_2_5 | vSew1H[1] & attributeVec_3_5 | vSew1H[2] & attributeVec_3_5;
  assign attributeSelect_2_6 = vSew1H[0] & attributeVec_2_6 | vSew1H[1] & attributeVec_3_6 | vSew1H[2] & attributeVec_3_6;
  wire        attributeSelect_3_0 = vSew1H[0] & attributeVec_3_0 | vSew1H[1] & attributeVec_3_0 | vSew1H[2] & attributeVec_3_0;
  wire        attributeSelect_3_1 = vSew1H[0] & attributeVec_3_1 | vSew1H[1] & attributeVec_3_1 | vSew1H[2] & attributeVec_3_1;
  assign attributeSelect_3_2 = vSew1H[0] & attributeVec_3_2 | vSew1H[1] & attributeVec_3_2 | vSew1H[2] & attributeVec_3_2;
  assign attributeSelect_3_3 = vSew1H[0] & attributeVec_3_3 | vSew1H[1] & attributeVec_3_3 | vSew1H[2] & attributeVec_3_3;
  wire        attributeSelect_3_4 = vSew1H[0] & attributeVec_3_4 | vSew1H[1] & attributeVec_3_4 | vSew1H[2] & attributeVec_3_4;
  assign attributeSelect_3_5 = vSew1H[0] & attributeVec_3_5 | vSew1H[1] & attributeVec_3_5 | vSew1H[2] & attributeVec_3_5;
  assign attributeSelect_3_6 = vSew1H[0] & attributeVec_3_6 | vSew1H[1] & attributeVec_3_6 | vSew1H[2] & attributeVec_3_6;
  wire [7:0]  responseSelect_0 =
    ((|{uopOH[1:0], uopOH[11:10]}) & ~overflowVec_0
       ? {request_sign
            ? (request_average & isFirstBlock[0] ? allNegativeVec_0 & attributeVec_resultSign | adderResultVec_0[6] & anyNegativeVec_0 & (~isSub | ~attributeVec_resultSign) : attributeVec_resultSign)
            : request_average & isFirstBlock[0] & isSub ^ attributeVec_resultSign,
          adderResultVec_0[6:0]}
       : 8'h0) | (uopOH[6] & ~lessVec_0 | uopOH[7] & lessVec_0 ? request_src_1[7:0] : 8'h0) | (uopOH[6] & lessVec_0 | uopOH[7] & ~lessVec_0 ? request_src_0[7:0] : 8'h0)
    | (attributeSelect_0_1 & request_saturate ? {~(isFirstBlock[0] & request_sign), 7'h7F} : 8'h0) | (attributeSelect_0_0 & request_saturate ? {isFirstBlock[0] & request_sign, 7'h0} : 8'h0);
  wire [7:0]  responseSelect_1 =
    ((|{uopOH[1:0], uopOH[11:10]}) & ~overflowVec_1
       ? {request_sign
            ? (request_average & isFirstBlock[1] ? allNegativeVec_1 & attributeVec_resultSign_1 | adderResultVec_1[6] & anyNegativeVec_1 & (~isSub | ~attributeVec_resultSign_1) : attributeVec_resultSign_1)
            : request_average & isFirstBlock[1] & isSub ^ attributeVec_resultSign_1,
          adderResultVec_1[6:0]}
       : 8'h0) | (uopOH[6] & ~lessVec_1 | uopOH[7] & lessVec_1 ? request_src_1[15:8] : 8'h0) | (uopOH[6] & lessVec_1 | uopOH[7] & ~lessVec_1 ? request_src_0[15:8] : 8'h0)
    | (attributeSelect_1_1 & request_saturate ? {~(isFirstBlock[1] & request_sign), 7'h7F} : 8'h0) | (attributeSelect_1_0 & request_saturate ? {isFirstBlock[1] & request_sign, 7'h0} : 8'h0);
  wire [7:0]  responseSelect_2 =
    ((|{uopOH[1:0], uopOH[11:10]}) & ~overflowVec_2
       ? {request_sign
            ? (request_average & isFirstBlock[2] ? allNegativeVec_2 & attributeVec_resultSign_2 | adderResultVec_2[6] & anyNegativeVec_2 & (~isSub | ~attributeVec_resultSign_2) : attributeVec_resultSign_2)
            : request_average & isFirstBlock[2] & isSub ^ attributeVec_resultSign_2,
          adderResultVec_2[6:0]}
       : 8'h0) | (uopOH[6] & ~lessVec_2 | uopOH[7] & lessVec_2 ? request_src_1[23:16] : 8'h0) | (uopOH[6] & lessVec_2 | uopOH[7] & ~lessVec_2 ? request_src_0[23:16] : 8'h0)
    | (attributeSelect_2_1 & request_saturate ? {~(isFirstBlock[2] & request_sign), 7'h7F} : 8'h0) | (attributeSelect_2_0 & request_saturate ? {isFirstBlock[2] & request_sign, 7'h0} : 8'h0);
  wire [7:0]  responseSelect_3 =
    ((|{uopOH[1:0], uopOH[11:10]}) & ~overflowVec_3
       ? {request_sign
            ? (request_average & isFirstBlock[3] ? allNegativeVec_3 & attributeVec_resultSign_3 | adderResultVec_3[6] & anyNegativeVec_3 & (~isSub | ~attributeVec_resultSign_3) : attributeVec_resultSign_3)
            : request_average & isFirstBlock[3] & isSub ^ attributeVec_resultSign_3,
          adderResultVec_3[6:0]}
       : 8'h0) | (uopOH[6] & ~lessVec_3 | uopOH[7] & lessVec_3 ? request_src_1[31:24] : 8'h0) | (uopOH[6] & lessVec_3 | uopOH[7] & ~lessVec_3 ? request_src_0[31:24] : 8'h0)
    | (attributeSelect_3_1 & request_saturate ? {~(isFirstBlock[3] & request_sign), 7'h7F} : 8'h0) | (attributeSelect_3_0 & request_saturate ? {isFirstBlock[3] & request_sign, 7'h0} : 8'h0);
  wire        maskResponseSelect_0 = vSew1H[0] & attributeVec_0_4 | vSew1H[1] & attributeVec_1_4 | vSew1H[2] & attributeVec_3_4;
  wire        maskResponseSelect_1 = vSew1H[0] & attributeVec_1_4 | vSew1H[1] & attributeVec_3_4 | vSew1H[2] & attributeVec_1_4;
  wire        maskResponseSelect_2 = vSew1H[0] & attributeVec_2_4 | vSew1H[1] & attributeVec_3_4 | vSew1H[2] & attributeVec_2_4;
  wire        maskResponseSelect_3 = vSew1H[0] & attributeVec_3_4 | vSew1H[1] & attributeVec_3_4 | vSew1H[2] & attributeVec_3_4;
  wire [15:0] response_data_lo = {responseSelect_1, responseSelect_0};
  wire [15:0] response_data_hi = {responseSelect_3, responseSelect_2};
  assign response_data = {response_data_hi, response_data_lo};
  wire [1:0]  response_adderMaskResp_lo = {maskResponseSelect_1, maskResponseSelect_0};
  wire [1:0]  response_adderMaskResp_hi = {maskResponseSelect_3, maskResponseSelect_2};
  assign response_adderMaskResp = {response_adderMaskResp_hi, response_adderMaskResp_lo};
  wire [1:0]  response_vxsat_lo = {overflowVec_1, overflowVec_0};
  wire [1:0]  response_vxsat_hi = {overflowVec_3, overflowVec_2};
  assign response_vxsat = {response_vxsat_hi, response_vxsat_lo};
  always @(posedge clock) begin
    if (reset) begin
      requestReg_tag <= 2'h0;
      requestReg_src_0 <= 32'h0;
      requestReg_src_1 <= 32'h0;
      requestReg_mask <= 4'h0;
      requestReg_opcode <= 4'h0;
      requestReg_sign <= 1'h0;
      requestReg_reverse <= 1'h0;
      requestReg_average <= 1'h0;
      requestReg_saturate <= 1'h0;
      requestReg_vxrm <= 2'h0;
      requestReg_vSew <= 2'h0;
      requestReg_executeIndex <= 2'h0;
      requestRegValid <= 1'h0;
      request_pipeResponse_pipe_v <= 1'h0;
    end
    else begin
      if (requestIO_valid_0) begin
        requestReg_tag <= requestIO_bits_tag_0;
        requestReg_src_0 <= requestIO_bits_src_0_0;
        requestReg_src_1 <= requestIO_bits_src_1_0;
        requestReg_mask <= requestIO_bits_mask_0;
        requestReg_opcode <= requestIO_bits_opcode_0;
        requestReg_sign <= requestIO_bits_sign_0;
        requestReg_reverse <= requestIO_bits_reverse_0;
        requestReg_average <= requestIO_bits_average_0;
        requestReg_saturate <= requestIO_bits_saturate_0;
        requestReg_vxrm <= requestIO_bits_vxrm_0;
        requestReg_vSew <= requestIO_bits_vSew_0;
        requestReg_executeIndex <= requestIO_bits_executeIndex_0;
      end
      requestRegValid <= requestIO_valid_0;
      request_pipeResponse_pipe_v <= request_responseValidWire;
    end
    if (request_responseValidWire) begin
      request_pipeResponse_pipe_b_tag <= request_responseWire_tag;
      request_pipeResponse_pipe_b_data <= request_responseWire_data;
      request_pipeResponse_pipe_b_adderMaskResp <= request_responseWire_adderMaskResp;
      request_pipeResponse_pipe_b_vxsat <= request_responseWire_vxsat;
      request_pipeResponse_pipe_b_executeIndex <= request_responseWire_executeIndex;
    end
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:4];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [2:0] i = 3'h0; i < 3'h5; i += 3'h1) begin
          _RANDOM[i] = `RANDOM;
        end
        requestReg_tag = _RANDOM[3'h0][1:0];
        requestReg_src_0 = {_RANDOM[3'h0][31:2], _RANDOM[3'h1][1:0]};
        requestReg_src_1 = {_RANDOM[3'h1][31:2], _RANDOM[3'h2][1:0]};
        requestReg_mask = _RANDOM[3'h2][5:2];
        requestReg_opcode = _RANDOM[3'h2][9:6];
        requestReg_sign = _RANDOM[3'h2][10];
        requestReg_reverse = _RANDOM[3'h2][11];
        requestReg_average = _RANDOM[3'h2][12];
        requestReg_saturate = _RANDOM[3'h2][13];
        requestReg_vxrm = _RANDOM[3'h2][15:14];
        requestReg_vSew = _RANDOM[3'h2][17:16];
        requestReg_executeIndex = _RANDOM[3'h2][19:18];
        requestRegValid = _RANDOM[3'h2][20];
        request_pipeResponse_pipe_v = _RANDOM[3'h2][21];
        request_pipeResponse_pipe_b_tag = _RANDOM[3'h2][23:22];
        request_pipeResponse_pipe_b_data = {_RANDOM[3'h2][31:24], _RANDOM[3'h3][23:0]};
        request_pipeResponse_pipe_b_adderMaskResp = _RANDOM[3'h3][27:24];
        request_pipeResponse_pipe_b_vxsat = _RANDOM[3'h3][31:28];
        request_pipeResponse_pipe_b_executeIndex = _RANDOM[3'h4][1:0];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  VectorAdder adder (
    .a    (request_average ? c : subOperation0),
    .b    (request_average ? {1'h0, operation1ForAverage} : subOperation1),
    .z    (_adder_z),
    .sew  (vSew1H),
    .cin  (request_average ? 4'h0 : (vSew1H[0] ? operation2 : 4'h0) | (vSew1H[1] ? {{2{operation2[1]}}, {2{operation2[0]}}} : 4'h0) | {4{vSew1H[2] & operation2[0]}}),
    .cout (_adder_cout)
  );
  assign responseIO_valid = responseIO_valid_0;
  assign responseIO_bits_tag = responseIO_bits_tag_0;
  assign responseIO_bits_data = responseIO_bits_data_0;
  assign responseIO_bits_adderMaskResp = responseIO_bits_adderMaskResp_0;
  assign responseIO_bits_vxsat = responseIO_bits_vxsat_0;
  assign responseIO_bits_executeIndex = responseIO_bits_executeIndex_0;
endmodule

