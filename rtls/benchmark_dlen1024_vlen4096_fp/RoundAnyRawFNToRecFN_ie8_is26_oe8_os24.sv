module RoundAnyRawFNToRecFN_ie8_is26_oe8_os24(
  input         io_invalidExc,
                io_in_isNaN,
                io_in_isInf,
                io_in_isZero,
                io_in_sign,
  input  [9:0]  io_in_sExp,
  input  [26:0] io_in_sig,
  input  [2:0]  io_roundingMode,
  output [32:0] io_out,
  output [4:0]  io_exceptionFlags
);

  wire        io_invalidExc_0 = io_invalidExc;
  wire        io_in_isNaN_0 = io_in_isNaN;
  wire        io_in_isInf_0 = io_in_isInf;
  wire        io_in_isZero_0 = io_in_isZero;
  wire        io_in_sign_0 = io_in_sign;
  wire [9:0]  io_in_sExp_0 = io_in_sExp;
  wire [26:0] io_in_sig_0 = io_in_sig;
  wire [2:0]  io_roundingMode_0 = io_roundingMode;
  wire        io_infiniteExc = 1'h0;
  wire        io_detectTininess = 1'h0;
  wire        notNaN_isSpecialInfOut = io_in_isInf_0;
  wire [26:0] adjustedSig = io_in_sig_0;
  wire        roundingMode_near_even = io_roundingMode_0 == 3'h0;
  wire        roundingMode_minMag = io_roundingMode_0 == 3'h1;
  wire        roundingMode_min = io_roundingMode_0 == 3'h2;
  wire        roundingMode_max = io_roundingMode_0 == 3'h3;
  wire        roundingMode_near_maxMag = io_roundingMode_0 == 3'h4;
  wire        roundingMode_odd = io_roundingMode_0 == 3'h6;
  wire        roundMagUp = roundingMode_min & io_in_sign_0 | roundingMode_max & ~io_in_sign_0;
  wire        doShiftSigDown1 = adjustedSig[26];
  wire [8:0]  _roundMask_T_1 = ~(io_in_sExp_0[8:0]);
  wire        roundMask_msb = _roundMask_T_1[8];
  wire [7:0]  roundMask_lsbs = _roundMask_T_1[7:0];
  wire        roundMask_msb_1 = roundMask_lsbs[7];
  wire [6:0]  roundMask_lsbs_1 = roundMask_lsbs[6:0];
  wire        roundMask_msb_2 = roundMask_lsbs_1[6];
  wire        roundMask_msb_3 = roundMask_lsbs_1[6];
  wire [5:0]  roundMask_lsbs_2 = roundMask_lsbs_1[5:0];
  wire [5:0]  roundMask_lsbs_3 = roundMask_lsbs_1[5:0];
  wire [64:0] roundMask_shift = $signed(65'sh10000000000000000 >>> roundMask_lsbs_2);
  wire [15:0] _roundMask_T_12 = {8'h0, roundMask_shift[57:50]} | {roundMask_shift[49:42], 8'h0};
  wire [15:0] _roundMask_T_22 = {4'h0, _roundMask_T_12[15:4] & 12'hF0F} | {_roundMask_T_12[11:0] & 12'hF0F, 4'h0};
  wire [15:0] _roundMask_T_32 = {2'h0, _roundMask_T_22[15:2] & 14'h3333} | {_roundMask_T_22[13:0] & 14'h3333, 2'h0};
  wire [64:0] roundMask_shift_1 = $signed(65'sh10000000000000000 >>> roundMask_lsbs_3);
  wire [24:0] _roundMask_T_73 =
    roundMask_msb
      ? (roundMask_msb_1
           ? {~(roundMask_msb_2
                  ? 22'h0
                  : ~{{1'h0, _roundMask_T_32[15:1] & 15'h5555} | {_roundMask_T_32[14:0] & 15'h5555, 1'h0}, roundMask_shift[58], roundMask_shift[59], roundMask_shift[60], roundMask_shift[61], roundMask_shift[62], roundMask_shift[63]}),
              3'h7}
           : {22'h0, roundMask_msb_3 ? {roundMask_shift_1[0], roundMask_shift_1[1], roundMask_shift_1[2]} : 3'h0})
      : 25'h0;
  wire [26:0] roundMask = {_roundMask_T_73[24:1], _roundMask_T_73[0] | doShiftSigDown1, 2'h3};
  wire [26:0] shiftedRoundMask = {1'h0, roundMask[26:1]};
  wire [26:0] roundPosMask = ~shiftedRoundMask & roundMask;
  wire        roundPosBit = |(adjustedSig & roundPosMask);
  wire        anyRoundExtra = |(adjustedSig & shiftedRoundMask);
  wire        anyRound = roundPosBit | anyRoundExtra;
  wire        _overflow_roundMagUp_T = roundingMode_near_even | roundingMode_near_maxMag;
  wire        roundIncr = _overflow_roundMagUp_T & roundPosBit | roundMagUp & anyRound;
  wire [25:0] roundedSig =
    roundIncr
      ? {1'h0, adjustedSig[26:2] | roundMask[26:2]} + 26'h1 & ~(roundingMode_near_even & roundPosBit & ~anyRoundExtra ? roundMask[26:1] : 26'h0)
      : {1'h0, adjustedSig[26:2] & ~(roundMask[26:2])} | (roundingMode_odd & anyRound ? roundPosMask[26:1] : 26'h0);
  wire [10:0] sRoundedExp = {io_in_sExp_0[9], io_in_sExp_0} + {9'h0, roundedSig[25:24]};
  wire [8:0]  common_expOut = sRoundedExp[8:0];
  wire [22:0] common_fractOut = doShiftSigDown1 ? roundedSig[23:1] : roundedSig[22:0];
  wire        common_overflow = $signed(sRoundedExp[10:7]) > 4'sh2;
  wire        common_totalUnderflow = $signed(sRoundedExp) < 11'sh6B;
  wire        unboundedRange_roundPosBit = doShiftSigDown1 ? adjustedSig[2] : adjustedSig[1];
  wire        unboundedRange_anyRound = |{doShiftSigDown1 & adjustedSig[2], adjustedSig[1:0]};
  wire        unboundedRange_roundIncr = _overflow_roundMagUp_T & unboundedRange_roundPosBit | roundMagUp & unboundedRange_anyRound;
  wire        roundCarry = doShiftSigDown1 ? roundedSig[25] : roundedSig[24];
  wire        common_underflow = common_totalUnderflow | anyRound & io_in_sExp_0[9:8] != 2'h1 & (doShiftSigDown1 ? roundMask[3] : roundMask[2]);
  wire        common_inexact = common_totalUnderflow | anyRound;
  wire        isNaNOut = io_invalidExc_0 | io_in_isNaN_0;
  wire        commonCase = ~isNaNOut & ~notNaN_isSpecialInfOut & ~io_in_isZero_0;
  wire        overflow = commonCase & common_overflow;
  wire        underflow = commonCase & common_underflow;
  wire        inexact = overflow | commonCase & common_inexact;
  wire        overflow_roundMagUp = _overflow_roundMagUp_T | roundMagUp;
  wire        pegMinNonzeroMagOut = commonCase & common_totalUnderflow & (roundMagUp | roundingMode_odd);
  wire        pegMaxFiniteMagOut = overflow & ~overflow_roundMagUp;
  wire        notNaN_isInfOut = notNaN_isSpecialInfOut | overflow & overflow_roundMagUp;
  wire        signOut = ~isNaNOut & io_in_sign_0;
  wire [8:0]  expOut =
    common_expOut & ~(io_in_isZero_0 | common_totalUnderflow ? 9'h1C0 : 9'h0) & ~(pegMinNonzeroMagOut ? 9'h194 : 9'h0) & {1'h1, ~pegMaxFiniteMagOut, 7'h7F} & {2'h3, ~notNaN_isInfOut, 6'h3F} | (pegMinNonzeroMagOut ? 9'h6B : 9'h0)
    | (pegMaxFiniteMagOut ? 9'h17F : 9'h0) | (notNaN_isInfOut ? 9'h180 : 9'h0) | (isNaNOut ? 9'h1C0 : 9'h0);
  wire [22:0] fractOut = (isNaNOut | io_in_isZero_0 | common_totalUnderflow ? {isNaNOut, 22'h0} : common_fractOut) | {23{pegMaxFiniteMagOut}};
  wire [32:0] io_out_0 = {signOut, expOut, fractOut};
  wire [4:0]  io_exceptionFlags_0 = {io_invalidExc_0, 1'h0, overflow, underflow, inexact};
  assign io_out = io_out_0;
  assign io_exceptionFlags = io_exceptionFlags_0;
endmodule

