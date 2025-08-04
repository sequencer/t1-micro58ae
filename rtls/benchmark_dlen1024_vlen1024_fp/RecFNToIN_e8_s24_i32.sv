module RecFNToIN_e8_s24_i32(
  input  [32:0] io_in,
  input  [2:0]  io_roundingMode,
  input         io_signedOut,
  output [31:0] io_out,
  output [2:0]  io_intExceptionFlags
);

  wire [32:0] io_in_0 = io_in;
  wire [2:0]  io_roundingMode_0 = io_roundingMode;
  wire        io_signedOut_0 = io_signedOut;
  wire [8:0]  rawIn_exp = io_in_0[31:23];
  wire        rawIn_isZero = rawIn_exp[8:6] == 3'h0;
  wire        rawIn_isZero_0 = rawIn_isZero;
  wire        rawIn_isSpecial = &(rawIn_exp[8:7]);
  wire        rawIn_isNaN = rawIn_isSpecial & rawIn_exp[6];
  wire        rawIn_isInf = rawIn_isSpecial & ~(rawIn_exp[6]);
  wire        rawIn_sign = io_in_0[32];
  wire [9:0]  rawIn_sExp = {1'h0, rawIn_exp};
  wire [24:0] rawIn_sig = {1'h0, ~rawIn_isZero, io_in_0[22:0]};
  wire        magGeOne = rawIn_sExp[8];
  wire [7:0]  posExp = rawIn_sExp[7:0];
  wire        magJustBelowOne = ~magGeOne & (&posExp);
  wire        roundingMode_near_even = io_roundingMode_0 == 3'h0;
  wire        roundingMode_minMag = io_roundingMode_0 == 3'h1;
  wire        roundingMode_min = io_roundingMode_0 == 3'h2;
  wire        roundingMode_max = io_roundingMode_0 == 3'h3;
  wire        roundingMode_near_maxMag = io_roundingMode_0 == 3'h4;
  wire        roundingMode_odd = io_roundingMode_0 == 3'h6;
  wire [54:0] shiftedSig = {31'h0, magGeOne, rawIn_sig[22:0]} << (magGeOne ? rawIn_sExp[4:0] : 5'h0);
  wire [33:0] alignedSig = {shiftedSig[54:22], |(shiftedSig[21:0])};
  wire [31:0] unroundedInt = alignedSig[33:2];
  wire        common_inexact = magGeOne ? (|(alignedSig[1:0])) : ~rawIn_isZero_0;
  wire        roundIncr_near_even = magGeOne & ((&(alignedSig[2:1])) | (&(alignedSig[1:0]))) | magJustBelowOne & (|(alignedSig[1:0]));
  wire        roundIncr_near_maxMag = magGeOne & alignedSig[1] | magJustBelowOne;
  wire        roundIncr =
    roundingMode_near_even & roundIncr_near_even | roundingMode_near_maxMag & roundIncr_near_maxMag | (roundingMode_min | roundingMode_odd) & rawIn_sign & common_inexact | roundingMode_max & ~rawIn_sign & common_inexact;
  wire [31:0] complUnroundedInt = {32{rawIn_sign}} ^ unroundedInt;
  wire [31:0] _roundedInt_T_3 = roundIncr ^ rawIn_sign ? complUnroundedInt + 32'h1 : complUnroundedInt;
  wire [31:0] roundedInt = {_roundedInt_T_3[31:1], _roundedInt_T_3[0] | roundingMode_odd & common_inexact};
  wire        magGeOne_atOverflowEdge = posExp == 8'h1F;
  wire        roundCarryBut2 = (&(unroundedInt[29:0])) & roundIncr;
  wire        common_overflow =
    magGeOne
      ? (|(posExp[7:5]))
        | (io_signedOut_0
             ? (rawIn_sign ? magGeOne_atOverflowEdge & ((|(unroundedInt[30:0])) | roundIncr) : magGeOne_atOverflowEdge | posExp == 8'h1E & roundCarryBut2)
             : rawIn_sign | magGeOne_atOverflowEdge & unroundedInt[30] & roundCarryBut2)
      : ~io_signedOut_0 & rawIn_sign & roundIncr;
  wire        invalidExc = rawIn_isNaN | rawIn_isInf;
  wire        overflow = ~invalidExc & common_overflow;
  wire        inexact = ~invalidExc & ~common_overflow & common_inexact;
  wire        excSign = ~rawIn_isNaN & rawIn_sign;
  wire [31:0] excOut = {io_signedOut_0 == excSign, {31{~excSign}}};
  wire [31:0] io_out_0 = invalidExc | common_overflow ? excOut : roundedInt;
  wire [2:0]  io_intExceptionFlags_0 = {invalidExc, overflow, inexact};
  assign io_out = io_out_0;
  assign io_intExceptionFlags = io_intExceptionFlags_0;
endmodule

