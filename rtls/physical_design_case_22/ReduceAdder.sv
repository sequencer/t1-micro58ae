module ReduceAdder(
  input  [31:0] request_src_0,
                request_src_1,
  input  [3:0]  request_opcode,
  input  [1:0]  request_vSew,
  input         request_sign,
  output [31:0] response_data
);

  wire [31:0] _adder_z;
  wire [3:0]  _adder_cout;
  wire [15:0] _uopOH_T = 16'h1 << request_opcode;
  wire [11:0] uopOH = _uopOH_T[11:0];
  wire        isSub = ~(uopOH[0] | uopOH[10]);
  wire [31:0] subOperation0 = {32{isSub}} ^ request_src_0;
  wire [3:0]  operation2 = {4{isSub}};
  wire [3:0]  _view__sew_T = 4'h1 << request_vSew;
  wire [2:0]  vSew1H = _view__sew_T[2:0];
  wire [3:0]  adderCarryOut = {vSew1H[0] & _adder_cout[3] | vSew1H[1] & _adder_cout[1] | vSew1H[2] & _adder_cout[0], _adder_cout[2], vSew1H[0] ? _adder_cout[1] : _adder_cout[0], _adder_cout[0]};
  wire [7:0]  adderResultVec_0 = _adder_z[7:0];
  wire [7:0]  adderResultVec_1 = _adder_z[15:8];
  wire [7:0]  adderResultVec_2 = _adder_z[23:16];
  wire [7:0]  adderResultVec_3 = _adder_z[31:24];
  wire        attributeVec_sourceSign0 = request_src_0[7];
  wire        attributeVec_sourceSign1 = request_src_1[7];
  wire        attributeVec_operation0Sign = attributeVec_sourceSign0 & request_sign ^ isSub;
  wire        attributeVec_operation1Sign = attributeVec_sourceSign1 & request_sign;
  wire        attributeVec_resultSign = adderResultVec_0[7];
  wire        attributeVec_uIntLess = attributeVec_sourceSign0 ^ attributeVec_sourceSign1 ? attributeVec_sourceSign0 : attributeVec_resultSign;
  wire        attributeVec_lowerOverflow = attributeVec_operation0Sign & attributeVec_operation1Sign & ~attributeVec_resultSign | isSub & ~request_sign & attributeVec_uIntLess;
  wire        attributeVec_upperOverflow = request_sign ? ~attributeVec_operation0Sign & ~attributeVec_operation1Sign & attributeVec_resultSign : adderCarryOut[0] & ~isSub;
  wire        attributeVec_0 = request_sign ? attributeVec_resultSign & ~attributeVec_upperOverflow | attributeVec_lowerOverflow : attributeVec_uIntLess;
  wire        attributeVec_sourceSign0_1 = request_src_0[15];
  wire        attributeVec_sourceSign1_1 = request_src_1[15];
  wire        attributeVec_operation0Sign_1 = attributeVec_sourceSign0_1 & request_sign ^ isSub;
  wire        attributeVec_operation1Sign_1 = attributeVec_sourceSign1_1 & request_sign;
  wire        attributeVec_resultSign_1 = adderResultVec_1[7];
  wire        attributeVec_uIntLess_1 = attributeVec_sourceSign0_1 ^ attributeVec_sourceSign1_1 ? attributeVec_sourceSign0_1 : attributeVec_resultSign_1;
  wire        attributeVec_lowerOverflow_1 = attributeVec_operation0Sign_1 & attributeVec_operation1Sign_1 & ~attributeVec_resultSign_1 | isSub & ~request_sign & attributeVec_uIntLess_1;
  wire        attributeVec_upperOverflow_1 = request_sign ? ~attributeVec_operation0Sign_1 & ~attributeVec_operation1Sign_1 & attributeVec_resultSign_1 : adderCarryOut[1] & ~isSub;
  wire        attributeVec_1 = request_sign ? attributeVec_resultSign_1 & ~attributeVec_upperOverflow_1 | attributeVec_lowerOverflow_1 : attributeVec_uIntLess_1;
  wire        attributeVec_sourceSign0_2 = request_src_0[23];
  wire        attributeVec_sourceSign1_2 = request_src_1[23];
  wire        attributeVec_operation0Sign_2 = attributeVec_sourceSign0_2 & request_sign ^ isSub;
  wire        attributeVec_operation1Sign_2 = attributeVec_sourceSign1_2 & request_sign;
  wire        attributeVec_resultSign_2 = adderResultVec_2[7];
  wire        attributeVec_uIntLess_2 = attributeVec_sourceSign0_2 ^ attributeVec_sourceSign1_2 ? attributeVec_sourceSign0_2 : attributeVec_resultSign_2;
  wire        attributeVec_lowerOverflow_2 = attributeVec_operation0Sign_2 & attributeVec_operation1Sign_2 & ~attributeVec_resultSign_2 | isSub & ~request_sign & attributeVec_uIntLess_2;
  wire        attributeVec_upperOverflow_2 = request_sign ? ~attributeVec_operation0Sign_2 & ~attributeVec_operation1Sign_2 & attributeVec_resultSign_2 : adderCarryOut[2] & ~isSub;
  wire        attributeVec_2 = request_sign ? attributeVec_resultSign_2 & ~attributeVec_upperOverflow_2 | attributeVec_lowerOverflow_2 : attributeVec_uIntLess_2;
  wire        attributeVec_sourceSign0_3 = request_src_0[31];
  wire        attributeVec_sourceSign1_3 = request_src_1[31];
  wire        attributeVec_operation0Sign_3 = attributeVec_sourceSign0_3 & request_sign ^ isSub;
  wire        attributeVec_operation1Sign_3 = attributeVec_sourceSign1_3 & request_sign;
  wire        attributeVec_resultSign_3 = adderResultVec_3[7];
  wire        attributeVec_uIntLess_3 = attributeVec_sourceSign0_3 ^ attributeVec_sourceSign1_3 ? attributeVec_sourceSign0_3 : attributeVec_resultSign_3;
  wire        attributeVec_lowerOverflow_3 = attributeVec_operation0Sign_3 & attributeVec_operation1Sign_3 & ~attributeVec_resultSign_3 | isSub & ~request_sign & attributeVec_uIntLess_3;
  wire        attributeVec_upperOverflow_3 = request_sign ? ~attributeVec_operation0Sign_3 & ~attributeVec_operation1Sign_3 & attributeVec_resultSign_3 : adderCarryOut[3] & ~isSub;
  wire        attributeVec_3 = request_sign ? attributeVec_resultSign_3 & ~attributeVec_upperOverflow_3 | attributeVec_lowerOverflow_3 : attributeVec_uIntLess_3;
  wire        lessVec_0 = vSew1H[0] & attributeVec_0 | vSew1H[1] & attributeVec_1 | vSew1H[2] & attributeVec_3;
  wire        lessVec_1 = vSew1H[0] & attributeVec_1 | vSew1H[1] & attributeVec_1 | vSew1H[2] & attributeVec_3;
  wire        lessVec_2 = vSew1H[0] & attributeVec_2 | vSew1H[1] & attributeVec_3 | vSew1H[2] & attributeVec_3;
  wire        lessVec_3 = vSew1H[0] & attributeVec_3 | vSew1H[1] & attributeVec_3 | vSew1H[2] & attributeVec_3;
  wire [7:0]  responseSelect_0 =
    ((|{uopOH[1:0], uopOH[11:10]}) ? adderResultVec_0 : 8'h0) | (uopOH[6] & ~lessVec_0 | uopOH[7] & lessVec_0 ? request_src_1[7:0] : 8'h0) | (uopOH[6] & lessVec_0 | uopOH[7] & ~lessVec_0 ? request_src_0[7:0] : 8'h0);
  wire [7:0]  responseSelect_1 =
    ((|{uopOH[1:0], uopOH[11:10]}) ? adderResultVec_1 : 8'h0) | (uopOH[6] & ~lessVec_1 | uopOH[7] & lessVec_1 ? request_src_1[15:8] : 8'h0) | (uopOH[6] & lessVec_1 | uopOH[7] & ~lessVec_1 ? request_src_0[15:8] : 8'h0);
  wire [7:0]  responseSelect_2 =
    ((|{uopOH[1:0], uopOH[11:10]}) ? adderResultVec_2 : 8'h0) | (uopOH[6] & ~lessVec_2 | uopOH[7] & lessVec_2 ? request_src_1[23:16] : 8'h0) | (uopOH[6] & lessVec_2 | uopOH[7] & ~lessVec_2 ? request_src_0[23:16] : 8'h0);
  wire [7:0]  responseSelect_3 =
    ((|{uopOH[1:0], uopOH[11:10]}) ? adderResultVec_3 : 8'h0) | (uopOH[6] & ~lessVec_3 | uopOH[7] & lessVec_3 ? request_src_1[31:24] : 8'h0) | (uopOH[6] & lessVec_3 | uopOH[7] & ~lessVec_3 ? request_src_0[31:24] : 8'h0);
  wire [15:0] view__response_data_lo = {responseSelect_1, responseSelect_0};
  wire [15:0] view__response_data_hi = {responseSelect_3, responseSelect_2};
  VectorAdder adder (
    .a    (subOperation0),
    .b    (request_src_1),
    .z    (_adder_z),
    .sew  (vSew1H),
    .cin  ((vSew1H[0] ? operation2 : 4'h0) | (vSew1H[1] ? {{2{operation2[1]}}, {2{operation2[0]}}} : 4'h0) | {4{vSew1H[2] & operation2[0]}}),
    .cout (_adder_cout)
  );
  assign response_data = {view__response_data_hi, view__response_data_lo};
endmodule

