module RoundAnyRawFNToRecFN_ie6_is32_oe8_os24(
  input         io_in_isZero,
                io_in_sign,
  input  [7:0]  io_in_sExp,
  input  [32:0] io_in_sig,
  input  [2:0]  io_roundingMode,
  output [32:0] io_out,
  output [4:0]  io_exceptionFlags
);

  wire        io_in_isZero_0 = io_in_isZero;
  wire        io_in_sign_0 = io_in_sign;
  wire [7:0]  io_in_sExp_0 = io_in_sExp;
  wire [32:0] io_in_sig_0 = io_in_sig;
  wire [2:0]  io_roundingMode_0 = io_roundingMode;
  wire [26:0] roundMask = 27'h3;
  wire [26:0] shiftedRoundMask = 27'h1;
  wire [26:0] roundPosMask = 27'h2;
  wire        io_invalidExc = 1'h0;
  wire        io_infiniteExc = 1'h0;
  wire        io_in_isNaN = 1'h0;
  wire        io_in_isInf = 1'h0;
  wire        io_detectTininess = 1'h0;
  wire        common_overflow = 1'h0;
  wire        common_totalUnderflow = 1'h0;
  wire        common_underflow = 1'h0;
  wire        isNaNOut = 1'h0;
  wire        notNaN_isSpecialInfOut = 1'h0;
  wire        overflow = 1'h0;
  wire        underflow = 1'h0;
  wire        pegMinNonzeroMagOut = 1'h0;
  wire        pegMaxFiniteMagOut = 1'h0;
  wire        notNaN_isInfOut = 1'h0;
  wire        signOut = io_in_sign_0;
  wire        roundingMode_near_even = io_roundingMode_0 == 3'h0;
  wire        roundingMode_minMag = io_roundingMode_0 == 3'h1;
  wire        roundingMode_min = io_roundingMode_0 == 3'h2;
  wire        roundingMode_max = io_roundingMode_0 == 3'h3;
  wire        roundingMode_near_maxMag = io_roundingMode_0 == 3'h4;
  wire        roundingMode_odd = io_roundingMode_0 == 3'h6;
  wire        roundMagUp = roundingMode_min & io_in_sign_0 | roundingMode_max & ~io_in_sign_0;
  wire [9:0]  sAdjustedExp = {1'h0, {io_in_sExp_0[7], io_in_sExp_0} + 9'hC0};
  wire [26:0] adjustedSig = {io_in_sig_0[32:7], |(io_in_sig_0[6:0])};
  wire        anyRound;
  wire        roundPosBit = adjustedSig[1];
  wire        anyRoundExtra = adjustedSig[0];
  assign anyRound = roundPosBit | anyRoundExtra;
  wire        common_inexact = anyRound;
  wire        _overflow_roundMagUp_T = roundingMode_near_even | roundingMode_near_maxMag;
  wire        roundIncr = _overflow_roundMagUp_T & roundPosBit | roundMagUp & anyRound;
  wire [25:0] roundedSig = roundIncr ? {1'h0, adjustedSig[26:2]} + 26'h1 & {25'h1FFFFFF, ~(roundingMode_near_even & roundPosBit & ~anyRoundExtra)} : {1'h0, adjustedSig[26:2]} | {25'h0, roundingMode_odd & anyRound};
  wire [10:0] sRoundedExp = {sAdjustedExp[9], sAdjustedExp} + {9'h0, roundedSig[25:24]};
  wire [8:0]  common_expOut = sRoundedExp[8:0];
  wire [22:0] common_fractOut = roundedSig[22:0];
  wire        unboundedRange_roundPosBit = adjustedSig[1];
  wire        unboundedRange_anyRound = |(adjustedSig[1:0]);
  wire        unboundedRange_roundIncr = _overflow_roundMagUp_T & unboundedRange_roundPosBit | roundMagUp & unboundedRange_anyRound;
  wire        roundCarry = roundedSig[24];
  wire        commonCase = ~io_in_isZero_0;
  wire        inexact = commonCase & common_inexact;
  wire        overflow_roundMagUp = _overflow_roundMagUp_T | roundMagUp;
  wire [8:0]  expOut = common_expOut & ~(io_in_isZero_0 ? 9'h1C0 : 9'h0);
  wire [22:0] fractOut = io_in_isZero_0 ? 23'h0 : common_fractOut;
  wire [32:0] io_out_0 = {signOut, expOut, fractOut};
  wire [4:0]  io_exceptionFlags_0 = {4'h0, inexact};
  assign io_out = io_out_0;
  assign io_exceptionFlags = io_exceptionFlags_0;
endmodule

