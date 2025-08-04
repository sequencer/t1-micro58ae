
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
module SRTFPWrapper(
  input         clock,
                reset,
  output        input_ready,
  input         input_valid,
  input  [31:0] input_bits_a,
                input_bits_b,
  input         input_bits_signIn,
                input_bits_opFloat,
                input_bits_opSqrt,
                input_bits_opRem,
  input  [2:0]  input_bits_roundingMode,
  output        output_valid,
  output [31:0] output_bits_result,
  output [4:0]  output_bits_exceptionFlags
);

  wire [4:0]  _roundResult_rounder_output_exceptionFlags;
  wire [31:0] _abs_io_aOut;
  wire [31:0] _abs_io_bOut;
  wire        _abs_io_aSign;
  wire        _abs_io_bSign;
  wire        _divIter_input_ready;
  wire        _divIter_resultOutput_valid;
  wire [31:0] _divIter_resultOutput_bits_reminder;
  wire [31:0] _divIter_resultOutput_bits_quotient;
  wire        _divIter_output_isLastCycle;
  wire        _sqrtIter_input_ready;
  wire        _sqrtIter_resultOutput_valid;
  wire [25:0] _sqrtIter_resultOutput_bits_result;
  wire        _sqrtIter_resultOutput_bits_zeroRemainder;
  wire [27:0] _sqrtIter_output_partialSum;
  wire [27:0] _sqrtIter_output_partialCarry;
  wire        _sqrtIter_output_isLastCycle;
  wire [25:0] _sqrtIter_reqOTF_quotient;
  wire [25:0] _sqrtIter_reqOTF_quotientMinusOne;
  wire [25:0] divSqrtMuxOut_sigToRound;
  wire [9:0]  divSqrtMuxOut_expToRound;
  wire        input_valid_0 = input_valid;
  wire [31:0] input_bits_a_0 = input_bits_a;
  wire [31:0] input_bits_b_0 = input_bits_b;
  wire        input_bits_signIn_0 = input_bits_signIn;
  wire        input_bits_opFloat_0 = input_bits_opFloat;
  wire        input_bits_opSqrt_0 = input_bits_opSqrt;
  wire        input_bits_opRem_0 = input_bits_opRem;
  wire [2:0]  input_bits_roundingMode_0 = input_bits_roundingMode;
  wire        input_ready_0;
  wire        _expSelectedReg_T = input_ready_0 & input_valid_0;
  reg         opFloatReg;
  reg         opSqrtReg;
  reg         opRemReg;
  reg  [4:0]  roundingModeReg;
  wire        rawA_sign = input_bits_a_0[31];
  wire        rawA_sign_0 = rawA_sign;
  wire [7:0]  rawA_expIn = input_bits_a_0[30:23];
  wire [22:0] rawA_fractIn = input_bits_a_0[22:0];
  wire        rawA_isZeroExpIn = rawA_expIn == 8'h0;
  wire        rawA_isZeroFractIn = rawA_fractIn == 23'h0;
  wire [4:0]  rawA_normDist =
    rawA_fractIn[22]
      ? 5'h0
      : rawA_fractIn[21]
          ? 5'h1
          : rawA_fractIn[20]
              ? 5'h2
              : rawA_fractIn[19]
                  ? 5'h3
                  : rawA_fractIn[18]
                      ? 5'h4
                      : rawA_fractIn[17]
                          ? 5'h5
                          : rawA_fractIn[16]
                              ? 5'h6
                              : rawA_fractIn[15]
                                  ? 5'h7
                                  : rawA_fractIn[14]
                                      ? 5'h8
                                      : rawA_fractIn[13]
                                          ? 5'h9
                                          : rawA_fractIn[12]
                                              ? 5'hA
                                              : rawA_fractIn[11]
                                                  ? 5'hB
                                                  : rawA_fractIn[10]
                                                      ? 5'hC
                                                      : rawA_fractIn[9]
                                                          ? 5'hD
                                                          : rawA_fractIn[8]
                                                              ? 5'hE
                                                              : rawA_fractIn[7]
                                                                  ? 5'hF
                                                                  : rawA_fractIn[6] ? 5'h10 : rawA_fractIn[5] ? 5'h11 : rawA_fractIn[4] ? 5'h12 : rawA_fractIn[3] ? 5'h13 : rawA_fractIn[2] ? 5'h14 : rawA_fractIn[1] ? 5'h15 : 5'h16;
  wire [53:0] _rawA_subnormFract_T = {31'h0, rawA_fractIn} << rawA_normDist;
  wire [22:0] rawA_subnormFract = {_rawA_subnormFract_T[21:0], 1'h0};
  wire [8:0]  rawA_adjustedExp = (rawA_isZeroExpIn ? {4'hF, ~rawA_normDist} : {1'h0, rawA_expIn}) + {7'h20, rawA_isZeroExpIn ? 2'h2 : 2'h1};
  wire        rawA_isZero = rawA_isZeroExpIn & rawA_isZeroFractIn;
  wire        rawA_isZero_0 = rawA_isZero;
  wire        rawA_isSpecial = &(rawA_adjustedExp[8:7]);
  wire        rawA_isSubnormal = rawA_isZeroExpIn & ~rawA_isZeroFractIn;
  wire        rawA_sExpIsEven = rawA_isSubnormal & (input_bits_a_0[23] ^ rawA_normDist[0]) | ~rawA_isSubnormal & input_bits_a_0[23];
  wire        rawA_sExpIsEven_0 = rawA_sExpIsEven;
  wire        rawA_isNaN = rawA_isSpecial & ~rawA_isZeroFractIn;
  wire        rawA_isInf = rawA_isSpecial & rawA_isZeroFractIn;
  wire [9:0]  rawA_sExp = {1'h0, rawA_adjustedExp};
  wire [24:0] rawA_sig = {1'h0, ~rawA_isZero, rawA_isZeroExpIn ? rawA_subnormFract : rawA_fractIn};
  wire        rawA_isSNaN = rawA_isNaN & ~(rawA_sig[22]);
  wire        rawB_sign = input_bits_b_0[31];
  wire        rawB_sign_0 = rawB_sign;
  wire [7:0]  rawB_expIn = input_bits_b_0[30:23];
  wire [22:0] rawB_fractIn = input_bits_b_0[22:0];
  wire        rawB_isZeroExpIn = rawB_expIn == 8'h0;
  wire        rawB_isZeroFractIn = rawB_fractIn == 23'h0;
  wire [4:0]  rawB_normDist =
    rawB_fractIn[22]
      ? 5'h0
      : rawB_fractIn[21]
          ? 5'h1
          : rawB_fractIn[20]
              ? 5'h2
              : rawB_fractIn[19]
                  ? 5'h3
                  : rawB_fractIn[18]
                      ? 5'h4
                      : rawB_fractIn[17]
                          ? 5'h5
                          : rawB_fractIn[16]
                              ? 5'h6
                              : rawB_fractIn[15]
                                  ? 5'h7
                                  : rawB_fractIn[14]
                                      ? 5'h8
                                      : rawB_fractIn[13]
                                          ? 5'h9
                                          : rawB_fractIn[12]
                                              ? 5'hA
                                              : rawB_fractIn[11]
                                                  ? 5'hB
                                                  : rawB_fractIn[10]
                                                      ? 5'hC
                                                      : rawB_fractIn[9]
                                                          ? 5'hD
                                                          : rawB_fractIn[8]
                                                              ? 5'hE
                                                              : rawB_fractIn[7]
                                                                  ? 5'hF
                                                                  : rawB_fractIn[6] ? 5'h10 : rawB_fractIn[5] ? 5'h11 : rawB_fractIn[4] ? 5'h12 : rawB_fractIn[3] ? 5'h13 : rawB_fractIn[2] ? 5'h14 : rawB_fractIn[1] ? 5'h15 : 5'h16;
  wire [53:0] _rawB_subnormFract_T = {31'h0, rawB_fractIn} << rawB_normDist;
  wire [22:0] rawB_subnormFract = {_rawB_subnormFract_T[21:0], 1'h0};
  wire [8:0]  rawB_adjustedExp = (rawB_isZeroExpIn ? {4'hF, ~rawB_normDist} : {1'h0, rawB_expIn}) + {7'h20, rawB_isZeroExpIn ? 2'h2 : 2'h1};
  wire        rawB_isZero = rawB_isZeroExpIn & rawB_isZeroFractIn;
  wire        rawB_isZero_0 = rawB_isZero;
  wire        rawB_isSpecial = &(rawB_adjustedExp[8:7]);
  wire        rawB_isSubnormal = rawB_isZeroExpIn & ~rawB_isZeroFractIn;
  wire        rawB_sExpIsEven = rawB_isSubnormal & (input_bits_b_0[23] ^ rawB_normDist[0]) | ~rawB_isSubnormal & input_bits_b_0[23];
  wire        rawB_sExpIsEven_0 = rawB_sExpIsEven;
  wire        rawB_isNaN = rawB_isSpecial & ~rawB_isZeroFractIn;
  wire        rawB_isInf = rawB_isSpecial & rawB_isZeroFractIn;
  wire [9:0]  rawB_sExp = {1'h0, rawB_adjustedExp};
  wire [24:0] rawB_sig = {1'h0, ~rawB_isZero, rawB_isZeroExpIn ? rawB_subnormFract : rawB_fractIn};
  wire        rawB_isSNaN = rawB_isNaN & ~(rawB_sig[22]);
  wire        divInvalidCases = rawA_isZero_0 & rawB_isZero_0 | rawA_isInf & rawB_isInf;
  wire        divDivideZero = ~rawA_isNaN & ~rawA_isInf & rawB_isZero_0;
  wire        sqrtInvalidCases = ~rawA_isNaN & ~rawA_isZero_0 & rawA_sign_0;
  wire        isNVorDZ = input_bits_opSqrt_0 ? rawA_isSNaN | sqrtInvalidCases : rawA_isSNaN | rawB_isSNaN | divInvalidCases | divDivideZero;
  wire        isNaN = input_bits_opSqrt_0 ? rawA_isNaN | sqrtInvalidCases : rawA_isNaN | rawB_isNaN | divInvalidCases;
  wire        isInf = ~input_bits_opSqrt_0 & rawB_isZero_0 | rawA_isInf;
  wire        isZero = ~input_bits_opSqrt_0 & rawB_isInf | rawA_isZero_0;
  reg         isNVorDZReg;
  reg         isNaNReg;
  reg         isInfReg;
  reg         isZeroReg;
  wire        invalidExec = isNVorDZReg & isNaNReg;
  wire        infinitExec = isNVorDZReg & ~isNaNReg;
  wire        specialCaseA = rawA_isNaN | rawA_isInf | rawA_isZero_0;
  wire        specialCaseB = rawB_isNaN | rawB_isInf | rawB_isZero_0;
  wire        normalCaseDiv = ~specialCaseA & ~specialCaseB;
  wire        normalCaseSqrt = ~specialCaseA & ~rawA_sign_0;
  wire        normalCase = input_bits_opSqrt_0 ? normalCaseSqrt : normalCaseDiv;
  wire        specialCase = ~normalCase;
  wire        bypassFloat = specialCase & input_bits_opFloat_0;
  reg         floatSpecialValid;
  wire        signNext = input_bits_opSqrt_0 ? rawA_isZero_0 & rawA_sign_0 : rawA_sign_0 ^ rawB_sign_0;
  reg         signReg;
  wire [3:0]  expfirst2 = 4'h1 << rawA_sExp[8:7];
  wire [1:0]  expstart = {expfirst2[0], 1'h0} | {2{expfirst2[1]}};
  wire [7:0]  expForSqrt = {expstart, rawA_sExp[6:1]};
  wire        sqrtExpIsOdd = ~(rawA_sExp[0]);
  wire [24:0] sqrtFractIn_hi = {1'h0, rawA_sig[23:0]};
  wire [25:0] sqrtFractIn = sqrtExpIsOdd ? {sqrtFractIn_hi, 1'h0} : {rawA_sig[23:0], 2'h0};
  wire [23:0] fractDividendIn_hi = {1'h1, rawA_sig[22:0]};
  wire [31:0] fractDividendIn = {fractDividendIn_hi, 8'h0};
  wire [23:0] fractDivisorIn_hi = {1'h1, rawB_sig[22:0]};
  wire [31:0] fractDivisorIn = {fractDivisorIn_hi, 8'h0};
  wire        negative = _abs_io_aSign ^ _abs_io_bSign;
  wire        divideZero = input_bits_b_0 == 32'h0;
  wire [32:0] dividend = {1'h0, _abs_io_aOut};
  wire [32:0] divisor = {1'h0, _abs_io_bOut};
  wire [33:0] gap = {1'h0, divisor} + {1'h0, 33'h0 - dividend};
  wire        biggerdivisor = gap[33] & (|(gap[32:0]));
  wire        bypassInteger = (divideZero | biggerdivisor) & _expSelectedReg_T & ~input_bits_opFloat_0;
  wire [5:0]  zeroHeadDividend =
    {1'h0,
     _abs_io_aOut[31]
       ? 5'h0
       : _abs_io_aOut[30]
           ? 5'h1
           : _abs_io_aOut[29]
               ? 5'h2
               : _abs_io_aOut[28]
                   ? 5'h3
                   : _abs_io_aOut[27]
                       ? 5'h4
                       : _abs_io_aOut[26]
                           ? 5'h5
                           : _abs_io_aOut[25]
                               ? 5'h6
                               : _abs_io_aOut[24]
                                   ? 5'h7
                                   : _abs_io_aOut[23]
                                       ? 5'h8
                                       : _abs_io_aOut[22]
                                           ? 5'h9
                                           : _abs_io_aOut[21]
                                               ? 5'hA
                                               : _abs_io_aOut[20]
                                                   ? 5'hB
                                                   : _abs_io_aOut[19]
                                                       ? 5'hC
                                                       : _abs_io_aOut[18]
                                                           ? 5'hD
                                                           : _abs_io_aOut[17]
                                                               ? 5'hE
                                                               : _abs_io_aOut[16]
                                                                   ? 5'hF
                                                                   : _abs_io_aOut[15]
                                                                       ? 5'h10
                                                                       : _abs_io_aOut[14]
                                                                           ? 5'h11
                                                                           : _abs_io_aOut[13]
                                                                               ? 5'h12
                                                                               : _abs_io_aOut[12]
                                                                                   ? 5'h13
                                                                                   : _abs_io_aOut[11]
                                                                                       ? 5'h14
                                                                                       : _abs_io_aOut[10]
                                                                                           ? 5'h15
                                                                                           : _abs_io_aOut[9]
                                                                                               ? 5'h16
                                                                                               : _abs_io_aOut[8]
                                                                                                   ? 5'h17
                                                                                                   : _abs_io_aOut[7]
                                                                                                       ? 5'h18
                                                                                                       : _abs_io_aOut[6]
                                                                                                           ? 5'h19
                                                                                                           : _abs_io_aOut[5]
                                                                                                               ? 5'h1A
                                                                                                               : _abs_io_aOut[4] ? 5'h1B : _abs_io_aOut[3] ? 5'h1C : _abs_io_aOut[2] ? 5'h1D : {4'hF, ~(_abs_io_aOut[1])}};
  wire [5:0]  zeroHeadDivisor =
    {1'h0,
     _abs_io_bOut[31]
       ? 5'h0
       : _abs_io_bOut[30]
           ? 5'h1
           : _abs_io_bOut[29]
               ? 5'h2
               : _abs_io_bOut[28]
                   ? 5'h3
                   : _abs_io_bOut[27]
                       ? 5'h4
                       : _abs_io_bOut[26]
                           ? 5'h5
                           : _abs_io_bOut[25]
                               ? 5'h6
                               : _abs_io_bOut[24]
                                   ? 5'h7
                                   : _abs_io_bOut[23]
                                       ? 5'h8
                                       : _abs_io_bOut[22]
                                           ? 5'h9
                                           : _abs_io_bOut[21]
                                               ? 5'hA
                                               : _abs_io_bOut[20]
                                                   ? 5'hB
                                                   : _abs_io_bOut[19]
                                                       ? 5'hC
                                                       : _abs_io_bOut[18]
                                                           ? 5'hD
                                                           : _abs_io_bOut[17]
                                                               ? 5'hE
                                                               : _abs_io_bOut[16]
                                                                   ? 5'hF
                                                                   : _abs_io_bOut[15]
                                                                       ? 5'h10
                                                                       : _abs_io_bOut[14]
                                                                           ? 5'h11
                                                                           : _abs_io_bOut[13]
                                                                               ? 5'h12
                                                                               : _abs_io_bOut[12]
                                                                                   ? 5'h13
                                                                                   : _abs_io_bOut[11]
                                                                                       ? 5'h14
                                                                                       : _abs_io_bOut[10]
                                                                                           ? 5'h15
                                                                                           : _abs_io_bOut[9]
                                                                                               ? 5'h16
                                                                                               : _abs_io_bOut[8]
                                                                                                   ? 5'h17
                                                                                                   : _abs_io_bOut[7]
                                                                                                       ? 5'h18
                                                                                                       : _abs_io_bOut[6]
                                                                                                           ? 5'h19
                                                                                                           : _abs_io_bOut[5]
                                                                                                               ? 5'h1A
                                                                                                               : _abs_io_bOut[4] ? 5'h1B : _abs_io_bOut[3] ? 5'h1C : _abs_io_bOut[2] ? 5'h1D : {4'hF, ~(_abs_io_bOut[1])}};
  wire [5:0]  sub = 6'h0 - zeroHeadDividend + zeroHeadDivisor;
  wire [6:0]  needComputerWidth = {1'h0, sub} + 7'h2;
  wire [3:0]  guardSele = 4'h1 << needComputerWidth[1:0];
  wire [1:0]  guardWidth = {2{guardSele[1]}} | {guardSele[2], 1'h0} | {1'h0, guardSele[3]};
  wire [7:0]  _counter_T = {1'h0, needComputerWidth} + {6'h0, guardWidth};
  wire [5:0]  counter = _counter_T[7:2];
  wire [5:0]  leftShiftWidthDividend = zeroHeadDividend + 6'h0 - {4'h0, guardWidth} + 6'h3;
  wire [5:0]  leftShiftWidthDivisor = {1'h0, zeroHeadDivisor[4:0]};
  reg         negativeSRT;
  reg  [5:0]  zeroHeadDivisorSRT;
  reg         dividendSignSRT;
  reg         divideZeroReg;
  reg         biggerdivisorReg;
  reg         bypassIntegerReg;
  reg  [31:0] dividendInputReg;
  wire [94:0] _divDividend_T_3 = {63'h0, _abs_io_aOut} << leftShiftWidthDividend;
  wire [34:0] divDividend = input_bits_opFloat_0 | _expSelectedReg_T & input_bits_opFloat_0 ? {3'h0, fractDividendIn} : _divDividend_T_3[34:0];
  wire [94:0] _divDivisor_T_3 = {63'h0, _abs_io_bOut} << leftShiftWidthDivisor;
  wire [31:0] divDivisor = input_bits_opFloat_0 | _expSelectedReg_T & input_bits_opFloat_0 ? fractDivisorIn : _divDivisor_T_3[31:0];
  wire [25:0] sigPlusDiv;
  wire [9:0]  expToRound = divSqrtMuxOut_expToRound;
  wire [25:0] sigToRound = divSqrtMuxOut_sigToRound;
  wire        _divSqrtMuxOut_T_2 = opSqrtReg | input_bits_opSqrt_0 & _expSelectedReg_T;
  wire        sqrtMuxIn_enable;
  wire        divMuxIn_enable;
  wire [37:0] sqrtMuxIn_partialSumInit;
  wire [37:0] divMuxIn_partialSumInit;
  wire        divSqrtMuxOut_enable = _divSqrtMuxOut_T_2 ? sqrtMuxIn_enable : divMuxIn_enable;
  wire [37:0] sqrtMuxIn_partialSumNext;
  wire [37:0] divMuxIn_partialSumNext;
  wire [37:0] divSqrtMuxOut_partialSumInit = _divSqrtMuxOut_T_2 ? sqrtMuxIn_partialSumInit : divMuxIn_partialSumInit;
  wire [37:0] sqrtMuxIn_partialCarryNext;
  wire [37:0] divMuxIn_partialCarryNext;
  wire [37:0] divSqrtMuxOut_partialSumNext = _divSqrtMuxOut_T_2 ? sqrtMuxIn_partialSumNext : divMuxIn_partialSumNext;
  wire [31:0] sqrtMuxIn_quotient;
  wire [31:0] divMuxIn_quotient;
  wire [37:0] divSqrtMuxOut_partialCarryNext = _divSqrtMuxOut_T_2 ? sqrtMuxIn_partialCarryNext : divMuxIn_partialCarryNext;
  wire [31:0] sqrtMuxIn_quotientMinusOne;
  wire [31:0] divMuxIn_quotientMinusOne;
  wire [31:0] divSqrtMuxOut_quotient = _divSqrtMuxOut_T_2 ? sqrtMuxIn_quotient : divMuxIn_quotient;
  wire [4:0]  sqrtMuxIn_selectedQuotientOH;
  wire [4:0]  divMuxIn_selectedQuotientOH;
  wire [31:0] divSqrtMuxOut_quotientMinusOne = _divSqrtMuxOut_T_2 ? sqrtMuxIn_quotientMinusOne : divMuxIn_quotientMinusOne;
  wire [9:0]  sqrtMuxIn_expToRound;
  wire [9:0]  divMuxIn_expToRound;
  wire [4:0]  divSqrtMuxOut_selectedQuotientOH = _divSqrtMuxOut_T_2 ? sqrtMuxIn_selectedQuotientOH : divMuxIn_selectedQuotientOH;
  wire [25:0] sqrtMuxIn_sigToRound;
  wire [25:0] divMuxIn_sigToRound;
  assign divSqrtMuxOut_expToRound = _divSqrtMuxOut_T_2 ? sqrtMuxIn_expToRound : divMuxIn_expToRound;
  assign divSqrtMuxOut_sigToRound = _divSqrtMuxOut_T_2 ? sqrtMuxIn_sigToRound : divMuxIn_sigToRound;
  wire        divValid = input_valid_0 & ~bypassInteger & ~bypassFloat & ~input_bits_opSqrt_0;
  wire        sqrtValid = input_valid_0 & input_bits_opSqrt_0 & normalCaseSqrt;
  reg  [37:0] partialCarry;
  reg  [37:0] partialSum;
  wire [37:0] partialSumNext = _expSelectedReg_T ? divSqrtMuxOut_partialSumInit : divSqrtMuxOut_partialSumNext;
  wire [37:0] partialCarryNext = _expSelectedReg_T ? 38'h0 : divSqrtMuxOut_partialCarryNext;
  wire [31:0] otf_0;
  wire [31:0] otf_1;
  wire [31:0] remainderAbs = _divIter_resultOutput_bits_reminder >> zeroHeadDivisorSRT[4:0];
  wire [31:0] quotientAbs;
  wire [31:0] intQuotient = divideZeroReg ? 32'hFFFFFFFF : biggerdivisorReg ? 32'h0 : negativeSRT ? 32'h0 - quotientAbs : quotientAbs;
  wire [31:0] intRemainder = divideZeroReg | biggerdivisorReg ? dividendInputReg : dividendSignSRT ? 32'h0 - remainderAbs : remainderAbs;
  wire [31:0] intResult = opRemReg ? intRemainder : intQuotient;
  wire        needRightShift = ~(_divIter_resultOutput_bits_quotient[27]);
  wire [24:0] sigPlusSqrt = {_sqrtIter_resultOutput_bits_result[24:1], ~_sqrtIter_resultOutput_bits_zeroRemainder | _sqrtIter_resultOutput_bits_result[0]};
  assign divMuxIn_sigToRound = sigPlusDiv;
  assign sigPlusDiv = {1'h0, needRightShift ? _divIter_resultOutput_bits_quotient[25:2] : _divIter_resultOutput_bits_quotient[26:3], |_divIter_resultOutput_bits_reminder};
  wire [1:0]  expSelected_hi = {2{expForSqrt[7]}};
  wire [9:0]  expSelected = input_bits_opSqrt_0 ? {expSelected_hi, expForSqrt} : rawA_sExp - rawB_sExp;
  reg  [9:0]  expSelectedReg;
  assign sqrtMuxIn_expToRound = expSelectedReg;
  assign sqrtMuxIn_enable = sqrtValid & _sqrtIter_input_ready | ~_sqrtIter_output_isLastCycle;
  assign sqrtMuxIn_partialSumInit = {12'h3, sqrtFractIn};
  assign sqrtMuxIn_partialSumNext = {10'h0, _sqrtIter_output_partialSum};
  assign sqrtMuxIn_partialCarryNext = {10'h0, _sqrtIter_output_partialCarry};
  assign sqrtMuxIn_quotient = {6'h0, _sqrtIter_reqOTF_quotient};
  assign sqrtMuxIn_quotientMinusOne = {6'h0, _sqrtIter_reqOTF_quotientMinusOne};
  assign sqrtMuxIn_sigToRound = {1'h0, sigPlusSqrt};
  assign divMuxIn_enable = divValid & _divIter_input_ready | ~_divIter_output_isLastCycle;
  assign divMuxIn_partialSumInit = {3'h0, divDividend};
  assign divMuxIn_expToRound = expSelectedReg - {9'h0, needRightShift};
  wire [31:0] roundResult_1 = {27'h0, _roundResult_rounder_output_exceptionFlags};
  assign input_ready_0 = _divIter_input_ready & _sqrtIter_input_ready;
  wire [31:0] roundResult_0;
  always @(posedge clock) begin
    if (reset) begin
      opFloatReg <= 1'h0;
      opSqrtReg <= 1'h0;
      opRemReg <= 1'h0;
      roundingModeReg <= 5'h0;
      isNVorDZReg <= 1'h0;
      isNaNReg <= 1'h0;
      isInfReg <= 1'h0;
      isZeroReg <= 1'h0;
      floatSpecialValid <= 1'h0;
      signReg <= 1'h0;
      negativeSRT <= 1'h0;
      zeroHeadDivisorSRT <= 6'h0;
      dividendSignSRT <= 1'h0;
      divideZeroReg <= 1'h0;
      biggerdivisorReg <= 1'h0;
      bypassIntegerReg <= 1'h0;
      dividendInputReg <= 32'h0;
      partialCarry <= 38'h0;
      partialSum <= 38'h0;
      expSelectedReg <= 10'h0;
    end
    else begin
      if (_expSelectedReg_T) begin
        opFloatReg <= input_bits_opFloat_0;
        opSqrtReg <= input_bits_opSqrt_0;
        opRemReg <= input_bits_opRem_0;
        roundingModeReg <= {2'h0, input_bits_roundingMode_0};
        isNVorDZReg <= isNVorDZ;
        isNaNReg <= isNaN;
        isInfReg <= isInf;
        isZeroReg <= isZero;
        signReg <= signNext;
        divideZeroReg <= divideZero;
        biggerdivisorReg <= biggerdivisor;
        dividendInputReg <= input_bits_a_0;
        expSelectedReg <= expSelected;
      end
      floatSpecialValid <= bypassFloat & _expSelectedReg_T;
      if (_divIter_input_ready & divValid) begin
        negativeSRT <= negative;
        zeroHeadDivisorSRT <= zeroHeadDivisor;
        dividendSignSRT <= _abs_io_aSign;
      end
      bypassIntegerReg <= bypassInteger;
      if (divSqrtMuxOut_enable) begin
        partialCarry <= partialCarryNext;
        partialSum <= partialSumNext;
      end
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
        opFloatReg = _RANDOM[3'h0][0];
        opSqrtReg = _RANDOM[3'h0][1];
        opRemReg = _RANDOM[3'h0][2];
        roundingModeReg = _RANDOM[3'h0][7:3];
        isNVorDZReg = _RANDOM[3'h0][8];
        isNaNReg = _RANDOM[3'h0][9];
        isInfReg = _RANDOM[3'h0][10];
        isZeroReg = _RANDOM[3'h0][11];
        floatSpecialValid = _RANDOM[3'h0][12];
        signReg = _RANDOM[3'h0][13];
        negativeSRT = _RANDOM[3'h0][14];
        zeroHeadDivisorSRT = _RANDOM[3'h0][20:15];
        dividendSignSRT = _RANDOM[3'h0][21];
        divideZeroReg = _RANDOM[3'h0][22];
        biggerdivisorReg = _RANDOM[3'h0][23];
        bypassIntegerReg = _RANDOM[3'h0][24];
        dividendInputReg = {_RANDOM[3'h0][31:25], _RANDOM[3'h1][24:0]};
        partialCarry = {_RANDOM[3'h1][31:25], _RANDOM[3'h2][30:0]};
        partialSum = {_RANDOM[3'h2][31], _RANDOM[3'h3], _RANDOM[3'h4][4:0]};
        expSelectedReg = _RANDOM[3'h4][14:5];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  assign quotientAbs = _divIter_resultOutput_bits_quotient;
  SqrtIter sqrtIter (
    .clock                           (clock),
    .reset                           (reset),
    .input_ready                     (_sqrtIter_input_ready),
    .input_valid                     (sqrtValid),
    .input_bits_partialSum           (partialSum[27:0]),
    .input_bits_partialCarry         (partialCarry[27:0]),
    .resultOutput_valid              (_sqrtIter_resultOutput_valid),
    .resultOutput_bits_result        (_sqrtIter_resultOutput_bits_result),
    .resultOutput_bits_zeroRemainder (_sqrtIter_resultOutput_bits_zeroRemainder),
    .output_partialSum               (_sqrtIter_output_partialSum),
    .output_partialCarry             (_sqrtIter_output_partialCarry),
    .output_isLastCycle              (_sqrtIter_output_isLastCycle),
    .reqOTF_quotient                 (_sqrtIter_reqOTF_quotient),
    .reqOTF_quotientMinusOne         (_sqrtIter_reqOTF_quotientMinusOne),
    .reqOTF_selectedQuotientOH       (sqrtMuxIn_selectedQuotientOH),
    .respOTF_quotient                (otf_0[25:0]),
    .respOTF_quotientMinusOne        (otf_1[25:0])
  );
  SRT16Iter divIter (
    .clock                      (clock),
    .reset                      (reset),
    .input_ready                (_divIter_input_ready),
    .input_valid                (divValid),
    .input_bits_partialSum      (partialSum),
    .input_bits_partialCarry    (partialCarry),
    .input_bits_divider         (divDivisor),
    .input_bits_counter         (opFloatReg | input_bits_opFloat_0 & _expSelectedReg_T ? 5'h8 : counter[4:0]),
    .resultOutput_valid         (_divIter_resultOutput_valid),
    .resultOutput_bits_reminder (_divIter_resultOutput_bits_reminder),
    .resultOutput_bits_quotient (_divIter_resultOutput_bits_quotient),
    .output_partialSum          (divMuxIn_partialSumNext),
    .output_partialCarry        (divMuxIn_partialCarryNext),
    .output_isLastCycle         (_divIter_output_isLastCycle),
    .reqOTF_quotient            (divMuxIn_quotient),
    .reqOTF_quotientMinusOne    (divMuxIn_quotientMinusOne),
    .reqOTF_selectedQuotientOH  (divMuxIn_selectedQuotientOH),
    .respOTF_quotient           (otf_0),
    .respOTF_quotientMinusOne   (otf_1)
  );
  Abs abs (
    .io_aIn    (input_bits_a_0),
    .io_bIn    (input_bits_b_0),
    .io_signIn (input_bits_signIn_0),
    .io_aOut   (_abs_io_aOut),
    .io_bOut   (_abs_io_bOut),
    .io_aSign  (_abs_io_aSign),
    .io_bSign  (_abs_io_bSign)
  );
  OTF otf_m (
    .input_quotient           (divSqrtMuxOut_quotient),
    .input_quotientMinusOne   (divSqrtMuxOut_quotientMinusOne),
    .input_selectedQuotientOH (divSqrtMuxOut_selectedQuotientOH),
    .output_quotient          (otf_0),
    .output_quotientMinusOne  (otf_1)
  );
  RoundingUnit roundResult_rounder (
    .input_invalidExc      (invalidExec),
    .input_infiniteExc     (infinitExec),
    .input_isInf           (isInfReg),
    .input_isZero          (isZeroReg),
    .input_isNaN           (isNaNReg),
    .input_sigPlus         (sigToRound[24:0]),
    .input_exp             (expToRound),
    .input_sign            (signReg),
    .input_roundingMode    (roundingModeReg),
    .output_data           (roundResult_0),
    .output_exceptionFlags (_roundResult_rounder_output_exceptionFlags)
  );
  assign input_ready = input_ready_0;
  assign output_valid = _divIter_resultOutput_valid | bypassIntegerReg | _sqrtIter_resultOutput_valid | floatSpecialValid;
  assign output_bits_result = opFloatReg ? roundResult_0 : intResult;
  assign output_bits_exceptionFlags = roundResult_1[4:0];
endmodule

