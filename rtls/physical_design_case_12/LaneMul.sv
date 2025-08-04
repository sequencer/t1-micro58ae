
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
module LaneMul(
  input         clock,
                reset,
                requestIO_valid,
  input  [1:0]  requestIO_bits_tag,
  input  [31:0] requestIO_bits_src_0,
                requestIO_bits_src_1,
                requestIO_bits_src_2,
  input  [3:0]  requestIO_bits_opcode,
  input         requestIO_bits_saturate,
  input  [1:0]  requestIO_bits_vSew,
  input         requestIO_bits_sign0,
                requestIO_bits_sign,
  input  [1:0]  requestIO_bits_vxrm,
  output        responseIO_valid,
  output [1:0]  responseIO_bits_tag,
  output [31:0] responseIO_bits_data,
  output        responseIO_bits_vxsat
);

  wire [63:0]  _adder64_z;
  wire [63:0]  _fusionMultiplier_multiplierSum;
  wire [63:0]  _fusionMultiplier_multiplierCarry;
  wire [31:0]  _mulAbs1_Abs_z;
  wire [31:0]  _mulAbs0_Abs_z;
  wire         response_vxsat;
  wire         requestIO_valid_0 = requestIO_valid;
  wire [1:0]   requestIO_bits_tag_0 = requestIO_bits_tag;
  wire [31:0]  requestIO_bits_src_0_0 = requestIO_bits_src_0;
  wire [31:0]  requestIO_bits_src_1_0 = requestIO_bits_src_1;
  wire [31:0]  requestIO_bits_src_2_0 = requestIO_bits_src_2;
  wire [3:0]   requestIO_bits_opcode_0 = requestIO_bits_opcode;
  wire         requestIO_bits_saturate_0 = requestIO_bits_saturate;
  wire [1:0]   requestIO_bits_vSew_0 = requestIO_bits_vSew;
  wire         requestIO_bits_sign0_0 = requestIO_bits_sign0;
  wire         requestIO_bits_sign_0 = requestIO_bits_sign;
  wire [1:0]   requestIO_bits_vxrm_0 = requestIO_bits_vxrm;
  wire [6:0]   zeroExtend = 7'h0;
  wire [1:0]   response_tag = 2'h0;
  wire         adderInput_pickPreviousAdd2Carry = 1'h0;
  wire         adderInput_pickPreviousCarry = 1'h0;
  wire [7:0]   zeroByte = 8'h0;
  wire         requestIO_ready = 1'h1;
  wire         responseIO_ready = 1'h1;
  wire         request_pipeResponse_valid;
  wire [1:0]   request_pipeResponse_bits_tag;
  wire [31:0]  request_pipeResponse_bits_data;
  wire         request_pipeResponse_bits_vxsat;
  reg  [1:0]   requestReg_tag;
  wire [1:0]   request_responseWire_tag = requestReg_tag;
  reg  [31:0]  requestReg_src_0;
  reg  [31:0]  requestReg_src_1;
  reg  [31:0]  requestReg_src_2;
  reg  [3:0]   requestReg_opcode;
  reg          requestReg_saturate;
  reg  [1:0]   requestReg_vSew;
  reg          requestReg_sign0;
  reg          requestReg_sign;
  reg  [1:0]   requestReg_vxrm;
  reg          requestRegValid;
  wire         vfuRequestFire = requestRegValid;
  wire         request_responseValidWire = requestRegValid;
  wire [31:0]  response_data;
  wire         request_responseWire_vxsat = response_vxsat;
  wire [33:0]  request_responseWire_hi = {2'h0, response_data};
  wire [31:0]  request_responseWire_data = request_responseWire_hi[31:0];
  reg          request_pipeResponse_pipe_v;
  reg  [1:0]   request_pipeResponse_pipe_b_tag;
  reg  [31:0]  request_pipeResponse_pipe_b_data;
  reg          request_pipeResponse_pipe_b_vxsat;
  reg          request_pipeResponse_pipe_pipe_v;
  assign request_pipeResponse_valid = request_pipeResponse_pipe_pipe_v;
  reg  [1:0]   request_pipeResponse_pipe_pipe_b_tag;
  assign request_pipeResponse_bits_tag = request_pipeResponse_pipe_pipe_b_tag;
  reg  [31:0]  request_pipeResponse_pipe_pipe_b_data;
  assign request_pipeResponse_bits_data = request_pipeResponse_pipe_pipe_b_data;
  reg          request_pipeResponse_pipe_pipe_b_vxsat;
  assign request_pipeResponse_bits_vxsat = request_pipeResponse_pipe_pipe_b_vxsat;
  wire         responseIO_valid_0 = request_pipeResponse_valid;
  wire [1:0]   responseIO_bits_tag_0 = request_pipeResponse_bits_tag;
  wire [31:0]  responseIO_bits_data_0 = request_pipeResponse_bits_data;
  wire         responseIO_bits_vxsat_0 = request_pipeResponse_bits_vxsat;
  wire [63:0]  request_hi = {requestReg_src_2, requestReg_src_1};
  wire [2:0]   request_lo_lo = {requestReg_sign, requestReg_vxrm};
  wire [2:0]   request_lo_hi = {requestReg_vSew, requestReg_sign0};
  wire [5:0]   request_lo = {request_lo_hi, request_lo_lo};
  wire [4:0]   request_hi_lo = {requestReg_opcode, requestReg_saturate};
  wire [97:0]  request_hi_hi = {requestReg_tag, request_hi, requestReg_src_0};
  wire [102:0] request_hi_1 = {request_hi_hi, request_hi_lo};
  wire [1:0]   request_vxrm = request_lo[1:0];
  wire         request_sign = request_lo[2];
  wire         request_sign0 = request_lo[3];
  wire [1:0]   request_vSew = request_lo[5:4];
  wire         request_saturate = request_hi_1[0];
  wire [3:0]   request_opcode = request_hi_1[4:1];
  wire [31:0]  request_src_0 = request_hi_1[36:5];
  wire [31:0]  request_src_1 = request_hi_1[68:37];
  wire [31:0]  request_src_2 = request_hi_1[100:69];
  wire [1:0]   request_tag = request_hi_1[102:101];
  wire [63:0]  sew1H_hi = {requestIO_bits_src_2_0, requestIO_bits_src_1_0};
  wire [2:0]   sew1H_lo_lo = {requestIO_bits_sign_0, requestIO_bits_vxrm_0};
  wire [2:0]   sew1H_lo_hi = {requestIO_bits_vSew_0, requestIO_bits_sign0_0};
  wire [5:0]   sew1H_lo = {sew1H_lo_hi, sew1H_lo_lo};
  wire [4:0]   sew1H_hi_lo = {requestIO_bits_opcode_0, requestIO_bits_saturate_0};
  wire [97:0]  sew1H_hi_hi = {requestIO_bits_tag_0, sew1H_hi, requestIO_bits_src_0_0};
  wire [102:0] sew1H_hi_1 = {sew1H_hi_hi, sew1H_hi_lo};
  reg  [2:0]   sew1H;
  wire [3:0]   vxrm1H = 4'h1 << request_vxrm;
  wire [3:0]   opcode1H = 4'h1 << request_opcode[1:0];
  wire         ma = opcode1H[1] | opcode1H[2];
  wire         asAddend = request_opcode[2];
  wire         negative = request_opcode[3];
  wire [31:0]  mul0InputSelect = request_sign0 ? _mulAbs0_Abs_z : request_src_0;
  wire         mul0Sign_0 = request_src_0[7] & request_sign0;
  wire         mul0Sign_1 = request_src_0[15] & request_sign0;
  wire         mul0Sign_2 = request_src_0[23] & request_sign0;
  wire         mul0Sign_3 = request_src_0[31] & request_sign0;
  wire [31:0]  mul1 = asAddend | ~ma ? request_src_1 : request_src_2;
  wire [31:0]  mul1InputSelect = request_sign | ma & ~asAddend ? _mulAbs1_Abs_z : mul1;
  wire         mul1Sign_0 = mul1[7] & request_sign;
  wire         mul1Sign_1 = mul1[15] & request_sign;
  wire         mul1Sign_2 = mul1[23] & request_sign;
  wire         mul1Sign_3 = mul1[31] & request_sign;
  wire [31:0]  addend = (ma & asAddend ? request_src_2 : 32'h0) | (ma & ~asAddend ? request_src_1 : 32'h0);
  wire [15:0]  sumVec_0 = _fusionMultiplier_multiplierSum[15:0];
  wire [15:0]  sumVec_1 = _fusionMultiplier_multiplierSum[31:16];
  wire [15:0]  sumVec_2 = _fusionMultiplier_multiplierSum[47:32];
  wire [15:0]  sumVec_3 = _fusionMultiplier_multiplierSum[63:48];
  wire [15:0]  carryVec_0 = _fusionMultiplier_multiplierCarry[15:0];
  wire [15:0]  carryVec_1 = _fusionMultiplier_multiplierCarry[31:16];
  wire [15:0]  carryVec_2 = _fusionMultiplier_multiplierCarry[47:32];
  wire [15:0]  carryVec_3 = _fusionMultiplier_multiplierCarry[63:48];
  wire [3:0]   MSBBlockVec = {1'h1, sew1H[0], ~(sew1H[2]), sew1H[0]};
  wire [3:0]   LSBBlockVec = {sew1H[0], ~(sew1H[2]), sew1H[0], 1'h1};
  wire         negativeTag_0 = mul0Sign_0 ^ mul1Sign_0 ^ negative;
  wire         negativeTag_1 = mul0Sign_1 ^ mul1Sign_1 ^ negative;
  wire         negativeTag_2 = mul0Sign_2 ^ mul1Sign_2 ^ negative;
  wire         negativeTag_3 = mul0Sign_3 ^ mul1Sign_3 ^ negative;
  wire [3:0]   negativeBlock = {negativeTag_3, sew1H[0] ? negativeTag_2 : negativeTag_3, sew1H[2] ? negativeTag_3 : negativeTag_1, sew1H[0] & negativeTag_0 | sew1H[1] & negativeTag_1 | sew1H[2] & negativeTag_3};
  wire [7:0]   addendDataVec_0 = addend[7:0];
  wire [7:0]   addendDataVec_1 = addend[15:8];
  wire [7:0]   addendDataVec_2 = addend[23:16];
  wire [7:0]   addendDataVec_3 = addend[31:24];
  wire [63:0]  addendExtend =
    {8'h0,
     sew1H[0] ? addendDataVec_3 : 8'h0,
     sew1H[1] ? addendDataVec_3 : 8'h0,
     sew1H[2] ? 8'h0 : addendDataVec_2,
     sew1H[2] ? addendDataVec_3 : 8'h0,
     (sew1H[0] ? addendDataVec_1 : 8'h0) | (sew1H[2] ? addendDataVec_2 : 8'h0),
     sew1H[0] ? 8'h0 : addendDataVec_1,
     addendDataVec_0};
  wire [15:0]  addendExtendVec_0 = addendExtend[15:0];
  wire [15:0]  addendExtendVec_1 = addendExtend[31:16];
  wire [15:0]  addendExtendVec_2 = addendExtend[47:32];
  wire [15:0]  addendExtendVec_3 = addendExtend[63:48];
  wire         adderInput_isMSB = MSBBlockVec[0];
  wire         attributeVec_isMSB = MSBBlockVec[0];
  wire         adderInput_isLSB = LSBBlockVec[0];
  wire         adderInput_negativeMul = negativeBlock[0];
  wire         adderInput_needAdd2 = adderInput_negativeMul & adderInput_isLSB;
  wire [16:0]  adderInput_addCorrection = {1'h0, addendExtendVec_0} + {15'h0, adderInput_needAdd2, 1'h0};
  wire [15:0]  adderInput_csaAddInput = adderInput_addCorrection[15:0];
  wire         add2Carry_0 = adderInput_addCorrection[16];
  wire [15:0]  _GEN = {16{adderInput_negativeMul}};
  wire [15:0]  adderInput_sumSelect = _GEN ^ sumVec_0;
  wire [15:0]  adderInput_carrySelect = _GEN ^ carryVec_0;
  wire [15:0]  adderInput_xor = adderInput_sumSelect ^ adderInput_carrySelect;
  wire [15:0]  adderInput_0_1 = adderInput_xor ^ adderInput_csaAddInput;
  wire [15:0]  adderInput_csaC = adderInput_xor & adderInput_csaAddInput | adderInput_sumSelect & adderInput_carrySelect;
  wire         blockCsaCarry_0 = adderInput_csaC[15];
  wire [15:0]  adderInput_0_2 = {adderInput_csaC[14:0], 1'h0};
  wire         adderInput_isMSB_1 = MSBBlockVec[1];
  wire         attributeVec_isMSB_1 = MSBBlockVec[1];
  wire         adderInput_isLSB_1 = LSBBlockVec[1];
  wire         adderInput_negativeMul_1 = negativeBlock[1];
  wire         adderInput_needAdd2_1 = adderInput_negativeMul_1 & adderInput_isLSB_1;
  wire         adderInput_pickPreviousAdd2Carry_1 = ~adderInput_isLSB_1 & add2Carry_0;
  wire [16:0]  adderInput_addCorrection_1 = {1'h0, addendExtendVec_1} + {15'h0, adderInput_needAdd2_1, adderInput_pickPreviousAdd2Carry_1};
  wire [15:0]  adderInput_csaAddInput_1 = adderInput_addCorrection_1[15:0];
  wire         add2Carry_1 = adderInput_addCorrection_1[16];
  wire [15:0]  _GEN_0 = {16{adderInput_negativeMul_1}};
  wire [15:0]  adderInput_sumSelect_1 = _GEN_0 ^ sumVec_1;
  wire [15:0]  adderInput_carrySelect_1 = _GEN_0 ^ carryVec_1;
  wire [15:0]  adderInput_xor_1 = adderInput_sumSelect_1 ^ adderInput_carrySelect_1;
  wire [15:0]  adderInput_1_1 = adderInput_xor_1 ^ adderInput_csaAddInput_1;
  wire [15:0]  adderInput_csaC_1 = adderInput_xor_1 & adderInput_csaAddInput_1 | adderInput_sumSelect_1 & adderInput_carrySelect_1;
  wire         blockCsaCarry_1 = adderInput_csaC_1[15];
  wire         adderInput_pickPreviousCarry_1 = ~adderInput_isLSB_1 & blockCsaCarry_0;
  wire [15:0]  adderInput_1_2 = {adderInput_csaC_1[14:0], adderInput_pickPreviousCarry_1};
  wire         adderInput_isMSB_2 = MSBBlockVec[2];
  wire         attributeVec_isMSB_2 = MSBBlockVec[2];
  wire         adderInput_isLSB_2 = LSBBlockVec[2];
  wire         adderInput_negativeMul_2 = negativeBlock[2];
  wire         adderInput_needAdd2_2 = adderInput_negativeMul_2 & adderInput_isLSB_2;
  wire         adderInput_pickPreviousAdd2Carry_2 = ~adderInput_isLSB_2 & add2Carry_1;
  wire [16:0]  adderInput_addCorrection_2 = {1'h0, addendExtendVec_2} + {15'h0, adderInput_needAdd2_2, adderInput_pickPreviousAdd2Carry_2};
  wire [15:0]  adderInput_csaAddInput_2 = adderInput_addCorrection_2[15:0];
  wire         add2Carry_2 = adderInput_addCorrection_2[16];
  wire [15:0]  _GEN_1 = {16{adderInput_negativeMul_2}};
  wire [15:0]  adderInput_sumSelect_2 = _GEN_1 ^ sumVec_2;
  wire [15:0]  adderInput_carrySelect_2 = _GEN_1 ^ carryVec_2;
  wire [15:0]  adderInput_xor_2 = adderInput_sumSelect_2 ^ adderInput_carrySelect_2;
  wire [15:0]  adderInput_2_1 = adderInput_xor_2 ^ adderInput_csaAddInput_2;
  wire [15:0]  adderInput_csaC_2 = adderInput_xor_2 & adderInput_csaAddInput_2 | adderInput_sumSelect_2 & adderInput_carrySelect_2;
  wire         blockCsaCarry_2 = adderInput_csaC_2[15];
  wire         adderInput_pickPreviousCarry_2 = ~adderInput_isLSB_2 & blockCsaCarry_1;
  wire [15:0]  adderInput_2_2 = {adderInput_csaC_2[14:0], adderInput_pickPreviousCarry_2};
  wire         adderInput_isMSB_3 = MSBBlockVec[3];
  wire         attributeVec_isMSB_3 = MSBBlockVec[3];
  wire         adderInput_isLSB_3 = LSBBlockVec[3];
  wire         adderInput_negativeMul_3 = negativeBlock[3];
  wire         adderInput_needAdd2_3 = adderInput_negativeMul_3 & adderInput_isLSB_3;
  wire         adderInput_pickPreviousAdd2Carry_3 = ~adderInput_isLSB_3 & add2Carry_2;
  wire [16:0]  adderInput_addCorrection_3 = {1'h0, addendExtendVec_3} + {15'h0, adderInput_needAdd2_3, adderInput_pickPreviousAdd2Carry_3};
  wire [15:0]  adderInput_csaAddInput_3 = adderInput_addCorrection_3[15:0];
  wire         add2Carry_3 = adderInput_addCorrection_3[16];
  wire [15:0]  _GEN_2 = {16{adderInput_negativeMul_3}};
  wire [15:0]  adderInput_sumSelect_3 = _GEN_2 ^ sumVec_3;
  wire [15:0]  adderInput_carrySelect_3 = _GEN_2 ^ carryVec_3;
  wire [15:0]  adderInput_xor_3 = adderInput_sumSelect_3 ^ adderInput_carrySelect_3;
  wire [15:0]  adderInput_3_1 = adderInput_xor_3 ^ adderInput_csaAddInput_3;
  wire [15:0]  adderInput_csaC_3 = adderInput_xor_3 & adderInput_csaAddInput_3 | adderInput_sumSelect_3 & adderInput_carrySelect_3;
  wire         blockCsaCarry_3 = adderInput_csaC_3[15];
  wire         adderInput_pickPreviousCarry_3 = ~adderInput_isLSB_3 & blockCsaCarry_2;
  wire [15:0]  adderInput_3_2 = {adderInput_csaC_3[14:0], adderInput_pickPreviousCarry_3};
  wire [31:0]  lo = {adderInput_1_1, adderInput_0_1};
  wire [31:0]  hi = {adderInput_3_1, adderInput_2_1};
  wire [31:0]  lo_1 = {adderInput_1_2, adderInput_0_2};
  wire [31:0]  hi_1 = {adderInput_3_2, adderInput_2_2};
  wire [15:0]  adderResultVec_0 = _adder64_z[15:0];
  wire [15:0]  adderResultVec_1 = _adder64_z[31:16];
  wire [15:0]  adderResultVec_2 = _adder64_z[47:32];
  wire [15:0]  adderResultVec_3 = _adder64_z[63:48];
  wire         attributeVec_expectedSigForMul;
  wire         attributeVec_expectedSigForMul_1;
  wire         attributeVec_expectedSigForMul_2;
  wire         attributeVec_expectedSigForMul_3;
  wire         expectedSignVec_2;
  wire         expectedSignVec_3;
  wire         expectedSignVec_1;
  wire         expectedSignVec_0;
  wire [3:0]   expectedSignForBlockVec =
    {expectedSignVec_3, sew1H[0] ? expectedSignVec_2 : expectedSignVec_3, sew1H[2] ? expectedSignVec_3 : expectedSignVec_1, sew1H[0] & expectedSignVec_0 | sew1H[1] & expectedSignVec_1 | sew1H[2] & expectedSignVec_3};
  wire         attributeVec_sourceSign0 = request_src_0[7];
  wire         attributeVec_sourceSign1 = mul1[7];
  wire [3:0]   notZeroVec;
  wire         attributeVec_notZero = notZeroVec[0];
  wire         attributeVec_operation0Sign = attributeVec_sourceSign0 & request_sign ^ negative;
  wire         attributeVec_operation1Sign = attributeVec_sourceSign1 & request_sign ^ negative;
  assign attributeVec_expectedSigForMul = attributeVec_operation0Sign ^ attributeVec_operation1Sign;
  assign expectedSignVec_0 = attributeVec_expectedSigForMul;
  wire         resultSignVec_0;
  wire         attributeVec_0_1 = (attributeVec_expectedSigForMul ^ resultSignVec_0) & attributeVec_notZero;
  wire         attributeVec_expectedSignForBlock = expectedSignForBlockVec[0];
  wire [7:0]   attributeVec_0_2 = {attributeVec_expectedSignForBlock ^ ~attributeVec_isMSB, {7{~attributeVec_expectedSignForBlock}}};
  wire         attributeVec_sourceSign0_1 = request_src_0[15];
  wire         attributeVec_sourceSign1_1 = mul1[15];
  wire         attributeVec_notZero_1 = notZeroVec[1];
  wire         attributeVec_operation0Sign_1 = attributeVec_sourceSign0_1 & request_sign ^ negative;
  wire         attributeVec_operation1Sign_1 = attributeVec_sourceSign1_1 & request_sign ^ negative;
  assign attributeVec_expectedSigForMul_1 = attributeVec_operation0Sign_1 ^ attributeVec_operation1Sign_1;
  assign expectedSignVec_1 = attributeVec_expectedSigForMul_1;
  wire         resultSignVec_1;
  wire         attributeVec_1_1 = (attributeVec_expectedSigForMul_1 ^ resultSignVec_1) & attributeVec_notZero_1;
  wire         attributeVec_expectedSignForBlock_1 = expectedSignForBlockVec[1];
  wire [7:0]   attributeVec_1_2 = {attributeVec_expectedSignForBlock_1 ^ ~attributeVec_isMSB_1, {7{~attributeVec_expectedSignForBlock_1}}};
  wire         attributeVec_sourceSign0_2 = request_src_0[23];
  wire         attributeVec_sourceSign1_2 = mul1[23];
  wire         attributeVec_notZero_2 = notZeroVec[2];
  wire         attributeVec_operation0Sign_2 = attributeVec_sourceSign0_2 & request_sign ^ negative;
  wire         attributeVec_operation1Sign_2 = attributeVec_sourceSign1_2 & request_sign ^ negative;
  assign attributeVec_expectedSigForMul_2 = attributeVec_operation0Sign_2 ^ attributeVec_operation1Sign_2;
  assign expectedSignVec_2 = attributeVec_expectedSigForMul_2;
  wire         resultSignVec_2;
  wire         attributeVec_2_1 = (attributeVec_expectedSigForMul_2 ^ resultSignVec_2) & attributeVec_notZero_2;
  wire         attributeVec_expectedSignForBlock_2 = expectedSignForBlockVec[2];
  wire [7:0]   attributeVec_2_2 = {attributeVec_expectedSignForBlock_2 ^ ~attributeVec_isMSB_2, {7{~attributeVec_expectedSignForBlock_2}}};
  wire         attributeVec_sourceSign0_3 = request_src_0[31];
  wire         attributeVec_sourceSign1_3 = mul1[31];
  wire         attributeVec_notZero_3 = notZeroVec[3];
  wire         attributeVec_operation0Sign_3 = attributeVec_sourceSign0_3 & request_sign ^ negative;
  wire         attributeVec_operation1Sign_3 = attributeVec_sourceSign1_3 & request_sign ^ negative;
  assign attributeVec_expectedSigForMul_3 = attributeVec_operation0Sign_3 ^ attributeVec_operation1Sign_3;
  assign expectedSignVec_3 = attributeVec_expectedSigForMul_3;
  wire         resultSignVec_3;
  wire         attributeVec_3_1 = (attributeVec_expectedSigForMul_3 ^ resultSignVec_3) & attributeVec_notZero_3;
  wire         attributeVec_expectedSignForBlock_3 = expectedSignForBlockVec[3];
  wire [7:0]   attributeVec_3_2 = {attributeVec_expectedSignForBlock_3 ^ ~attributeVec_isMSB_3, {7{~attributeVec_expectedSignForBlock_3}}};
  wire         roundResultForSew8_vd1 = adderResultVec_0[6];
  wire         roundResultForSew8_vd = adderResultVec_0[7];
  wire         roundResultForSew8_vd2OR = |(adderResultVec_0[5:0]);
  wire         roundResultForSew8_roundBits1 = roundResultForSew8_vd1 & (roundResultForSew8_vd2OR | roundResultForSew8_vd);
  wire         roundResultForSew8_roundBits2 = ~roundResultForSew8_vd & (roundResultForSew8_vd2OR | roundResultForSew8_vd1);
  wire         roundResultForSew8_roundBits = vxrm1H[0] & roundResultForSew8_vd1 | vxrm1H[1] & roundResultForSew8_roundBits1 | vxrm1H[3] & roundResultForSew8_roundBits2;
  wire [8:0]   roundResultForSew8_shiftResult = adderResultVec_0[15:7];
  wire         roundResultForSew8_vd1_1 = adderResultVec_1[6];
  wire         roundResultForSew8_vd_1 = adderResultVec_1[7];
  wire         roundResultForSew8_vd2OR_1 = |(adderResultVec_1[5:0]);
  wire         roundResultForSew8_roundBits1_1 = roundResultForSew8_vd1_1 & (roundResultForSew8_vd2OR_1 | roundResultForSew8_vd_1);
  wire         roundResultForSew8_roundBits2_1 = ~roundResultForSew8_vd_1 & (roundResultForSew8_vd2OR_1 | roundResultForSew8_vd1_1);
  wire         roundResultForSew8_roundBits_1 = vxrm1H[0] & roundResultForSew8_vd1_1 | vxrm1H[1] & roundResultForSew8_roundBits1_1 | vxrm1H[3] & roundResultForSew8_roundBits2_1;
  wire [8:0]   roundResultForSew8_shiftResult_1 = adderResultVec_1[15:7];
  wire         roundResultForSew8_vd1_2 = adderResultVec_2[6];
  wire         roundResultForSew8_vd_2 = adderResultVec_2[7];
  wire         roundResultForSew8_vd2OR_2 = |(adderResultVec_2[5:0]);
  wire         roundResultForSew8_roundBits1_2 = roundResultForSew8_vd1_2 & (roundResultForSew8_vd2OR_2 | roundResultForSew8_vd_2);
  wire         roundResultForSew8_roundBits2_2 = ~roundResultForSew8_vd_2 & (roundResultForSew8_vd2OR_2 | roundResultForSew8_vd1_2);
  wire         roundResultForSew8_roundBits_2 = vxrm1H[0] & roundResultForSew8_vd1_2 | vxrm1H[1] & roundResultForSew8_roundBits1_2 | vxrm1H[3] & roundResultForSew8_roundBits2_2;
  wire [8:0]   roundResultForSew8_shiftResult_2 = adderResultVec_2[15:7];
  wire         roundResultForSew8_vd1_3 = adderResultVec_3[6];
  wire         roundResultForSew8_vd_3 = adderResultVec_3[7];
  wire         roundResultForSew8_vd2OR_3 = |(adderResultVec_3[5:0]);
  wire         roundResultForSew8_roundBits1_3 = roundResultForSew8_vd1_3 & (roundResultForSew8_vd2OR_3 | roundResultForSew8_vd_3);
  wire         roundResultForSew8_roundBits2_3 = ~roundResultForSew8_vd_3 & (roundResultForSew8_vd2OR_3 | roundResultForSew8_vd1_3);
  wire         roundResultForSew8_roundBits_3 = vxrm1H[0] & roundResultForSew8_vd1_3 | vxrm1H[1] & roundResultForSew8_roundBits1_3 | vxrm1H[3] & roundResultForSew8_roundBits2_3;
  wire [8:0]   roundResultForSew8_shiftResult_3 = adderResultVec_3[15:7];
  wire [15:0]  roundResultForSew8_lo = {roundResultForSew8_shiftResult_1[7:0] + {7'h0, roundResultForSew8_roundBits_1}, roundResultForSew8_shiftResult[7:0] + {7'h0, roundResultForSew8_roundBits}};
  wire [15:0]  roundResultForSew8_hi = {roundResultForSew8_shiftResult_3[7:0] + {7'h0, roundResultForSew8_roundBits_3}, roundResultForSew8_shiftResult_2[7:0] + {7'h0, roundResultForSew8_roundBits_2}};
  wire [31:0]  roundResultForSew8 = {roundResultForSew8_hi, roundResultForSew8_lo};
  wire         roundResultForSew16_vd1 = _adder64_z[14];
  wire         roundResultForSew16_vd = _adder64_z[15];
  wire         roundResultForSew16_vd2OR = |(_adder64_z[13:0]);
  wire         roundResultForSew16_roundBits1 = roundResultForSew16_vd1 & (roundResultForSew16_vd2OR | roundResultForSew16_vd);
  wire         roundResultForSew16_roundBits2 = ~roundResultForSew16_vd & (roundResultForSew16_vd2OR | roundResultForSew16_vd1);
  wire         roundResultForSew16_roundBits = vxrm1H[0] & roundResultForSew16_vd1 | vxrm1H[1] & roundResultForSew16_roundBits1 | vxrm1H[3] & roundResultForSew16_roundBits2;
  wire [16:0]  roundResultForSew16_shiftResult = _adder64_z[31:15];
  wire         roundResultForSew16_vd1_1 = _adder64_z[46];
  wire         roundResultForSew16_vd_1 = _adder64_z[47];
  wire         roundResultForSew16_vd2OR_1 = |(_adder64_z[45:32]);
  wire         roundResultForSew16_roundBits1_1 = roundResultForSew16_vd1_1 & (roundResultForSew16_vd2OR_1 | roundResultForSew16_vd_1);
  wire         roundResultForSew16_roundBits2_1 = ~roundResultForSew16_vd_1 & (roundResultForSew16_vd2OR_1 | roundResultForSew16_vd1_1);
  wire         roundResultForSew16_roundBits_1 = vxrm1H[0] & roundResultForSew16_vd1_1 | vxrm1H[1] & roundResultForSew16_roundBits1_1 | vxrm1H[3] & roundResultForSew16_roundBits2_1;
  wire [16:0]  roundResultForSew16_shiftResult_1 = _adder64_z[63:47];
  wire [31:0]  roundResultForSew16 = {roundResultForSew16_shiftResult_1[15:0] + {15'h0, roundResultForSew16_roundBits_1}, roundResultForSew16_shiftResult[15:0] + {15'h0, roundResultForSew16_roundBits}};
  wire         roundResultForSew32_vd1 = _adder64_z[30];
  wire         roundResultForSew32_vd = _adder64_z[31];
  wire         roundResultForSew32_vd2OR = |(_adder64_z[29:0]);
  wire         roundResultForSew32_roundBits1 = roundResultForSew32_vd1 & (roundResultForSew32_vd2OR | roundResultForSew32_vd);
  wire         roundResultForSew32_roundBits2 = ~roundResultForSew32_vd & (roundResultForSew32_vd2OR | roundResultForSew32_vd1);
  wire         roundResultForSew32_roundBits = vxrm1H[0] & roundResultForSew32_vd1 | vxrm1H[1] & roundResultForSew32_roundBits1 | vxrm1H[3] & roundResultForSew32_roundBits2;
  wire [32:0]  roundResultForSew32_shiftResult = _adder64_z[63:31];
  wire [31:0]  roundResultForSew32 = roundResultForSew32_shiftResult[31:0] + {31'h0, roundResultForSew32_roundBits};
  wire [31:0]  roundingResult = (sew1H[0] ? roundResultForSew8 : 32'h0) | (sew1H[1] ? roundResultForSew16 : 32'h0) | (sew1H[2] ? roundResultForSew32 : 32'h0);
  wire [7:0]   roundingResultVec_0 = roundingResult[7:0];
  wire [7:0]   roundingResultVec_1 = roundingResult[15:8];
  wire [7:0]   roundingResultVec_2 = roundingResult[23:16];
  wire [7:0]   roundingResultVec_3 = roundingResult[31:24];
  assign resultSignVec_0 = roundingResultVec_0[7];
  assign resultSignVec_1 = roundingResultVec_1[7];
  assign resultSignVec_2 = roundingResultVec_2[7];
  assign resultSignVec_3 = roundingResultVec_3[7];
  wire [1:0]   roundingResultOrR_lo = {|roundingResultVec_1, |roundingResultVec_0};
  wire [1:0]   roundingResultOrR_hi = {|roundingResultVec_3, |roundingResultVec_2};
  wire [3:0]   roundingResultOrR = {roundingResultOrR_hi, roundingResultOrR_lo};
  wire [1:0]   orSew16 = {roundingResultOrR[2] | roundingResultOrR[3], roundingResultOrR[0] | roundingResultOrR[1]};
  wire         orSew32 = |orSew16;
  assign notZeroVec = (sew1H[0] ? roundingResultOrR : 4'h0) | (sew1H[1] ? {{2{orSew16[1]}}, {2{orSew16[0]}}} : 4'h0) | {4{sew1H[2] & orSew32}};
  wire         overflowSelect_0 = sew1H[0] & attributeVec_0_1 | sew1H[1] & attributeVec_1_1 | sew1H[2] & attributeVec_3_1;
  wire         overflowSelect_1 = sew1H[0] & attributeVec_1_1 | sew1H[1] & attributeVec_1_1 | sew1H[2] & attributeVec_3_1;
  wire         overflowSelect_2 = sew1H[0] & attributeVec_2_1 | sew1H[1] & attributeVec_3_1 | sew1H[2] & attributeVec_3_1;
  wire         overflowSelect_3 = sew1H[0] & attributeVec_3_1 | sew1H[1] & attributeVec_3_1 | sew1H[2] & attributeVec_3_1;
  wire [7:0]   addResultCutByByte_0 = _adder64_z[7:0];
  wire [7:0]   addResultCutByByte_1 = _adder64_z[15:8];
  wire [7:0]   addResultCutByByte_2 = _adder64_z[23:16];
  wire [7:0]   addResultCutByByte_3 = _adder64_z[31:24];
  wire [7:0]   addResultCutByByte_4 = _adder64_z[39:32];
  wire [7:0]   addResultCutByByte_5 = _adder64_z[47:40];
  wire [7:0]   addResultCutByByte_6 = _adder64_z[55:48];
  wire [7:0]   addResultCutByByte_7 = _adder64_z[63:56];
  wire [31:0]  mulMSB =
    {addResultCutByByte_7,
     sew1H[0] ? addResultCutByByte_5 : addResultCutByByte_6,
     sew1H[2] ? addResultCutByByte_5 : addResultCutByByte_3,
     (sew1H[0] ? addResultCutByByte_1 : 8'h0) | (sew1H[1] ? addResultCutByByte_2 : 8'h0) | (sew1H[2] ? addResultCutByByte_4 : 8'h0)};
  wire [7:0]   msbVec_0 = mulMSB[7:0];
  wire [7:0]   msbVec_1 = mulMSB[15:8];
  wire [7:0]   msbVec_2 = mulMSB[23:16];
  wire [7:0]   msbVec_3 = mulMSB[31:24];
  wire [31:0]  mulLSB =
    {(sew1H[0] ? addResultCutByByte_6 : 8'h0) | (sew1H[1] ? addResultCutByByte_5 : 8'h0) | (sew1H[2] ? addResultCutByByte_3 : 8'h0),
     sew1H[2] ? addResultCutByByte_2 : addResultCutByByte_4,
     sew1H[0] ? addResultCutByByte_2 : addResultCutByByte_1,
     addResultCutByByte_0};
  wire [7:0]   lsbVec_0 = mulLSB[7:0];
  wire [7:0]   lsbVec_1 = mulLSB[15:8];
  wire [7:0]   lsbVec_2 = mulLSB[23:16];
  wire [7:0]   lsbVec_3 = mulLSB[31:24];
  wire [15:0]  response_data_lo =
    {(opcode1H[0] & ~request_saturate | ma ? lsbVec_1 : 8'h0) | (opcode1H[3] ? msbVec_1 : 8'h0) | (request_saturate & ~overflowSelect_1 ? roundingResultVec_1 : 8'h0) | (request_saturate & overflowSelect_1 ? attributeVec_1_2 : 8'h0),
     (opcode1H[0] & ~request_saturate | ma ? lsbVec_0 : 8'h0) | (opcode1H[3] ? msbVec_0 : 8'h0) | (request_saturate & ~overflowSelect_0 ? roundingResultVec_0 : 8'h0) | (request_saturate & overflowSelect_0 ? attributeVec_0_2 : 8'h0)};
  wire [15:0]  response_data_hi =
    {(opcode1H[0] & ~request_saturate | ma ? lsbVec_3 : 8'h0) | (opcode1H[3] ? msbVec_3 : 8'h0) | (request_saturate & ~overflowSelect_3 ? roundingResultVec_3 : 8'h0) | (request_saturate & overflowSelect_3 ? attributeVec_3_2 : 8'h0),
     (opcode1H[0] & ~request_saturate | ma ? lsbVec_2 : 8'h0) | (opcode1H[3] ? msbVec_2 : 8'h0) | (request_saturate & ~overflowSelect_2 ? roundingResultVec_2 : 8'h0) | (request_saturate & overflowSelect_2 ? attributeVec_2_2 : 8'h0)};
  assign response_data = {response_data_hi, response_data_lo};
  wire [1:0]   response_vxsat_lo = {overflowSelect_1, overflowSelect_0};
  wire [1:0]   response_vxsat_hi = {overflowSelect_3, overflowSelect_2};
  assign response_vxsat = (|{response_vxsat_hi, response_vxsat_lo}) & request_saturate;
  always @(posedge clock) begin
    if (reset) begin
      requestReg_tag <= 2'h0;
      requestReg_src_0 <= 32'h0;
      requestReg_src_1 <= 32'h0;
      requestReg_src_2 <= 32'h0;
      requestReg_opcode <= 4'h0;
      requestReg_saturate <= 1'h0;
      requestReg_vSew <= 2'h0;
      requestReg_sign0 <= 1'h0;
      requestReg_sign <= 1'h0;
      requestReg_vxrm <= 2'h0;
      requestRegValid <= 1'h0;
      request_pipeResponse_pipe_v <= 1'h0;
      request_pipeResponse_pipe_pipe_v <= 1'h0;
      sew1H <= 3'h0;
    end
    else begin
      if (requestIO_valid_0) begin
        automatic logic [3:0] _sew1H_T_15 = 4'h1 << sew1H_lo[5:4];
        requestReg_tag <= requestIO_bits_tag_0;
        requestReg_src_0 <= requestIO_bits_src_0_0;
        requestReg_src_1 <= requestIO_bits_src_1_0;
        requestReg_src_2 <= requestIO_bits_src_2_0;
        requestReg_opcode <= requestIO_bits_opcode_0;
        requestReg_saturate <= requestIO_bits_saturate_0;
        requestReg_vSew <= requestIO_bits_vSew_0;
        requestReg_sign0 <= requestIO_bits_sign0_0;
        requestReg_sign <= requestIO_bits_sign_0;
        requestReg_vxrm <= requestIO_bits_vxrm_0;
        sew1H <= _sew1H_T_15[2:0];
      end
      requestRegValid <= requestIO_valid_0;
      request_pipeResponse_pipe_v <= request_responseValidWire;
      request_pipeResponse_pipe_pipe_v <= request_pipeResponse_pipe_v;
    end
    if (request_responseValidWire) begin
      request_pipeResponse_pipe_b_tag <= request_responseWire_tag;
      request_pipeResponse_pipe_b_data <= request_responseWire_data;
      request_pipeResponse_pipe_b_vxsat <= request_responseWire_vxsat;
    end
    if (request_pipeResponse_pipe_v) begin
      request_pipeResponse_pipe_pipe_b_tag <= request_pipeResponse_pipe_b_tag;
      request_pipeResponse_pipe_pipe_b_data <= request_pipeResponse_pipe_b_data;
      request_pipeResponse_pipe_pipe_b_vxsat <= request_pipeResponse_pipe_b_vxsat;
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
        requestReg_tag = _RANDOM[3'h0][1:0];
        requestReg_src_0 = {_RANDOM[3'h0][31:2], _RANDOM[3'h1][1:0]};
        requestReg_src_1 = {_RANDOM[3'h1][31:2], _RANDOM[3'h2][1:0]};
        requestReg_src_2 = {_RANDOM[3'h2][31:2], _RANDOM[3'h3][1:0]};
        requestReg_opcode = _RANDOM[3'h3][5:2];
        requestReg_saturate = _RANDOM[3'h3][6];
        requestReg_vSew = _RANDOM[3'h3][8:7];
        requestReg_sign0 = _RANDOM[3'h3][9];
        requestReg_sign = _RANDOM[3'h3][10];
        requestReg_vxrm = _RANDOM[3'h3][12:11];
        requestRegValid = _RANDOM[3'h3][13];
        request_pipeResponse_pipe_v = _RANDOM[3'h3][14];
        request_pipeResponse_pipe_b_tag = _RANDOM[3'h3][16:15];
        request_pipeResponse_pipe_b_data = {_RANDOM[3'h3][31:17], _RANDOM[3'h4][16:0]};
        request_pipeResponse_pipe_b_vxsat = _RANDOM[3'h4][17];
        request_pipeResponse_pipe_pipe_v = _RANDOM[3'h4][18];
        request_pipeResponse_pipe_pipe_b_tag = _RANDOM[3'h4][20:19];
        request_pipeResponse_pipe_pipe_b_data = {_RANDOM[3'h4][31:21], _RANDOM[3'h5][20:0]};
        request_pipeResponse_pipe_pipe_b_vxsat = _RANDOM[3'h5][21];
        sew1H = _RANDOM[3'h5][24:22];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  Abs32 mulAbs0_Abs (
    .a   (request_src_0),
    .z   (_mulAbs0_Abs_z),
    .sew (sew1H)
  );
  Abs32 mulAbs1_Abs (
    .a   (mul1),
    .z   (_mulAbs1_Abs_z),
    .sew (sew1H)
  );
  VectorMultiplier32Unsigned fusionMultiplier (
    .a               (mul0InputSelect),
    .b               (mul1InputSelect),
    .sew             (sew1H),
    .multiplierSum   (_fusionMultiplier_multiplierSum),
    .multiplierCarry (_fusionMultiplier_multiplierCarry)
  );
  VectorAdder64 adder64 (
    .a   ({hi, lo}),
    .b   ({hi_1, lo_1}),
    .z   (_adder64_z),
    .sew ({sew1H, 1'h0})
  );
  assign responseIO_valid = responseIO_valid_0;
  assign responseIO_bits_tag = responseIO_bits_tag_0;
  assign responseIO_bits_data = responseIO_bits_data_0;
  assign responseIO_bits_vxsat = responseIO_bits_vxsat_0;
endmodule

