module FloatCompare(
  input  [31:0] a,
                b,
  input         isMax,
  output [31:0] out
);

  wire        _compareModule_io_lt;
  wire        _compareModule_io_eq;
  wire        _compareModule_io_gt;
  wire        rec0_rawIn_sign = a[31];
  wire        rec0_rawIn_sign_0 = rec0_rawIn_sign;
  wire [7:0]  rec0_rawIn_expIn = a[30:23];
  wire [22:0] rec0_rawIn_fractIn = a[22:0];
  wire        rec0_rawIn_isZeroExpIn = rec0_rawIn_expIn == 8'h0;
  wire        rec0_rawIn_isZeroFractIn = rec0_rawIn_fractIn == 23'h0;
  wire [4:0]  rec0_rawIn_normDist =
    rec0_rawIn_fractIn[22]
      ? 5'h0
      : rec0_rawIn_fractIn[21]
          ? 5'h1
          : rec0_rawIn_fractIn[20]
              ? 5'h2
              : rec0_rawIn_fractIn[19]
                  ? 5'h3
                  : rec0_rawIn_fractIn[18]
                      ? 5'h4
                      : rec0_rawIn_fractIn[17]
                          ? 5'h5
                          : rec0_rawIn_fractIn[16]
                              ? 5'h6
                              : rec0_rawIn_fractIn[15]
                                  ? 5'h7
                                  : rec0_rawIn_fractIn[14]
                                      ? 5'h8
                                      : rec0_rawIn_fractIn[13]
                                          ? 5'h9
                                          : rec0_rawIn_fractIn[12]
                                              ? 5'hA
                                              : rec0_rawIn_fractIn[11]
                                                  ? 5'hB
                                                  : rec0_rawIn_fractIn[10]
                                                      ? 5'hC
                                                      : rec0_rawIn_fractIn[9]
                                                          ? 5'hD
                                                          : rec0_rawIn_fractIn[8]
                                                              ? 5'hE
                                                              : rec0_rawIn_fractIn[7]
                                                                  ? 5'hF
                                                                  : rec0_rawIn_fractIn[6]
                                                                      ? 5'h10
                                                                      : rec0_rawIn_fractIn[5] ? 5'h11 : rec0_rawIn_fractIn[4] ? 5'h12 : rec0_rawIn_fractIn[3] ? 5'h13 : rec0_rawIn_fractIn[2] ? 5'h14 : rec0_rawIn_fractIn[1] ? 5'h15 : 5'h16;
  wire [53:0] _rec0_rawIn_subnormFract_T = {31'h0, rec0_rawIn_fractIn} << rec0_rawIn_normDist;
  wire [22:0] rec0_rawIn_subnormFract = {_rec0_rawIn_subnormFract_T[21:0], 1'h0};
  wire [8:0]  rec0_rawIn_adjustedExp = (rec0_rawIn_isZeroExpIn ? {4'hF, ~rec0_rawIn_normDist} : {1'h0, rec0_rawIn_expIn}) + {7'h20, rec0_rawIn_isZeroExpIn ? 2'h2 : 2'h1};
  wire        rec0_rawIn_isZero = rec0_rawIn_isZeroExpIn & rec0_rawIn_isZeroFractIn;
  wire        rec0_rawIn_isZero_0 = rec0_rawIn_isZero;
  wire        rec0_rawIn_isSpecial = &(rec0_rawIn_adjustedExp[8:7]);
  wire        rec0_rawIn_isNaN = rec0_rawIn_isSpecial & ~rec0_rawIn_isZeroFractIn;
  wire        rec0_rawIn_isInf = rec0_rawIn_isSpecial & rec0_rawIn_isZeroFractIn;
  wire [9:0]  rec0_rawIn_sExp = {1'h0, rec0_rawIn_adjustedExp};
  wire [24:0] rec0_rawIn_sig = {1'h0, ~rec0_rawIn_isZero, rec0_rawIn_isZeroExpIn ? rec0_rawIn_subnormFract : rec0_rawIn_fractIn};
  wire [2:0]  _rec0_T_1 = rec0_rawIn_isZero_0 ? 3'h0 : rec0_rawIn_sExp[8:6];
  wire [32:0] rec0 = {rec0_rawIn_sign_0, _rec0_T_1[2:1], _rec0_T_1[0] | rec0_rawIn_isNaN, rec0_rawIn_sExp[5:0], rec0_rawIn_sig[22:0]};
  wire        rec1_rawIn_sign = b[31];
  wire        rec1_rawIn_sign_0 = rec1_rawIn_sign;
  wire [7:0]  rec1_rawIn_expIn = b[30:23];
  wire [22:0] rec1_rawIn_fractIn = b[22:0];
  wire        rec1_rawIn_isZeroExpIn = rec1_rawIn_expIn == 8'h0;
  wire        rec1_rawIn_isZeroFractIn = rec1_rawIn_fractIn == 23'h0;
  wire [4:0]  rec1_rawIn_normDist =
    rec1_rawIn_fractIn[22]
      ? 5'h0
      : rec1_rawIn_fractIn[21]
          ? 5'h1
          : rec1_rawIn_fractIn[20]
              ? 5'h2
              : rec1_rawIn_fractIn[19]
                  ? 5'h3
                  : rec1_rawIn_fractIn[18]
                      ? 5'h4
                      : rec1_rawIn_fractIn[17]
                          ? 5'h5
                          : rec1_rawIn_fractIn[16]
                              ? 5'h6
                              : rec1_rawIn_fractIn[15]
                                  ? 5'h7
                                  : rec1_rawIn_fractIn[14]
                                      ? 5'h8
                                      : rec1_rawIn_fractIn[13]
                                          ? 5'h9
                                          : rec1_rawIn_fractIn[12]
                                              ? 5'hA
                                              : rec1_rawIn_fractIn[11]
                                                  ? 5'hB
                                                  : rec1_rawIn_fractIn[10]
                                                      ? 5'hC
                                                      : rec1_rawIn_fractIn[9]
                                                          ? 5'hD
                                                          : rec1_rawIn_fractIn[8]
                                                              ? 5'hE
                                                              : rec1_rawIn_fractIn[7]
                                                                  ? 5'hF
                                                                  : rec1_rawIn_fractIn[6]
                                                                      ? 5'h10
                                                                      : rec1_rawIn_fractIn[5] ? 5'h11 : rec1_rawIn_fractIn[4] ? 5'h12 : rec1_rawIn_fractIn[3] ? 5'h13 : rec1_rawIn_fractIn[2] ? 5'h14 : rec1_rawIn_fractIn[1] ? 5'h15 : 5'h16;
  wire [53:0] _rec1_rawIn_subnormFract_T = {31'h0, rec1_rawIn_fractIn} << rec1_rawIn_normDist;
  wire [22:0] rec1_rawIn_subnormFract = {_rec1_rawIn_subnormFract_T[21:0], 1'h0};
  wire [8:0]  rec1_rawIn_adjustedExp = (rec1_rawIn_isZeroExpIn ? {4'hF, ~rec1_rawIn_normDist} : {1'h0, rec1_rawIn_expIn}) + {7'h20, rec1_rawIn_isZeroExpIn ? 2'h2 : 2'h1};
  wire        rec1_rawIn_isZero = rec1_rawIn_isZeroExpIn & rec1_rawIn_isZeroFractIn;
  wire        rec1_rawIn_isZero_0 = rec1_rawIn_isZero;
  wire        rec1_rawIn_isSpecial = &(rec1_rawIn_adjustedExp[8:7]);
  wire        rec1_rawIn_isNaN = rec1_rawIn_isSpecial & ~rec1_rawIn_isZeroFractIn;
  wire        rec1_rawIn_isInf = rec1_rawIn_isSpecial & rec1_rawIn_isZeroFractIn;
  wire [9:0]  rec1_rawIn_sExp = {1'h0, rec1_rawIn_adjustedExp};
  wire [24:0] rec1_rawIn_sig = {1'h0, ~rec1_rawIn_isZero, rec1_rawIn_isZeroExpIn ? rec1_rawIn_subnormFract : rec1_rawIn_fractIn};
  wire [2:0]  _rec1_T_1 = rec1_rawIn_isZero_0 ? 3'h0 : rec1_rawIn_sExp[8:6];
  wire [32:0] rec1 = {rec1_rawIn_sign_0, _rec1_T_1[2:1], _rec1_T_1[0] | rec1_rawIn_isNaN, rec1_rawIn_sExp[5:0], rec1_rawIn_sig[22:0]};
  wire [8:0]  raw0_exp = rec0[31:23];
  wire        raw0_isZero = raw0_exp[8:6] == 3'h0;
  wire        raw0_isZero_0 = raw0_isZero;
  wire        raw0_isSpecial = &(raw0_exp[8:7]);
  wire        raw0_isNaN = raw0_isSpecial & raw0_exp[6];
  wire        raw0_isInf = raw0_isSpecial & ~(raw0_exp[6]);
  wire        raw0_sign = rec0[32];
  wire [9:0]  raw0_sExp = {1'h0, raw0_exp};
  wire [24:0] raw0_sig = {1'h0, ~raw0_isZero, rec0[22:0]};
  wire [8:0]  raw1_exp = rec1[31:23];
  wire        raw1_isZero = raw1_exp[8:6] == 3'h0;
  wire        raw1_isZero_0 = raw1_isZero;
  wire        raw1_isSpecial = &(raw1_exp[8:7]);
  wire        raw1_isNaN = raw1_isSpecial & raw1_exp[6];
  wire        raw1_isInf = raw1_isSpecial & ~(raw1_exp[6]);
  wire        raw1_sign = rec1[32];
  wire [9:0]  raw1_sExp = {1'h0, raw1_exp};
  wire [24:0] raw1_sig = {1'h0, ~raw1_isZero, rec1[22:0]};
  wire        oneNaN = raw0_isNaN ^ raw1_isNaN;
  wire [31:0] hasNaNResult = oneNaN ? (raw0_isNaN ? b : a) : 32'h7FC00000;
  wire        hasNaN = raw0_isNaN | raw1_isNaN;
  wire        differentZeros = _compareModule_io_eq & (rec0_rawIn_sign ^ rec1_rawIn_sign);
  wire [31:0] noNaNResult = isMax & _compareModule_io_gt | ~isMax & _compareModule_io_lt | differentZeros & (isMax ^ ~rec1_rawIn_sign) ? a : b;
  CompareRecFN compareModule (
    .io_a              (rec0),
    .io_b              (rec1),
    .io_signaling      (1'h0),
    .io_lt             (_compareModule_io_lt),
    .io_eq             (_compareModule_io_eq),
    .io_gt             (_compareModule_io_gt),
    .io_exceptionFlags (/* unused */)
  );
  assign out = hasNaN ? hasNaNResult : noNaNResult;
endmodule

