module RoundingUnit(
  input         input_invalidExc,
                input_infiniteExc,
                input_isInf,
                input_isZero,
                input_isNaN,
  input  [24:0] input_sigPlus,
  input  [9:0]  input_exp,
  input         input_sign,
  input  [4:0]  input_roundingMode,
  output [31:0] output_data,
  output [4:0]  output_exceptionFlags
);

  wire [24:0] allSig = input_sigPlus;
  wire [31:0] roundingMdoe = 32'h1 << input_roundingMode;
  wire        rmRNE = roundingMdoe[0];
  wire        rmRTZ = roundingMdoe[1];
  wire        rmRDN = roundingMdoe[2];
  wire        rmRUP = roundingMdoe[3];
  wire        rmRMM = roundingMdoe[4];
  wire        anyRound;
  wire [9:0]  expSubnorm = input_exp + 10'h7E;
  wire        commonUnderflow;
  wire [9:0]  subnormDist = commonUnderflow ? 10'h0 - expSubnorm : 10'h0;
  wire [25:0] adjustedSig = {1'h1, allSig};
  wire [61:0] _allMask_T_2 = $signed(-62'sh80000000 >>> subnormDist[5:0]);
  wire [31:0] allMask = _allMask_T_2[31:0];
  wire        distGT24 = (|{subnormDist[9:5], allMask[6:0]}) & commonUnderflow;
  wire [15:0] _roundMask_T_11 = {8'h0, allMask[22:15]} | {allMask[14:7], 8'h0};
  wire [15:0] _roundMask_T_21 = {4'h0, _roundMask_T_11[15:4] & 12'hF0F} | {_roundMask_T_11[11:0] & 12'hF0F, 4'h0};
  wire [15:0] _roundMask_T_31 = {2'h0, _roundMask_T_21[15:2] & 14'h3333} | {_roundMask_T_21[13:0] & 14'h3333, 2'h0};
  wire [7:0]  _roundMask_T_51 = {4'h0, allMask[30:27]} | {allMask[26:23], 4'h0};
  wire [7:0]  _roundMask_T_61 = {2'h0, _roundMask_T_51[7:2] & 6'h33} | {_roundMask_T_51[5:0] & 6'h33, 2'h0};
  wire [25:0] roundMask = distGT24 ? 26'h0 : {{1'h0, _roundMask_T_31[15:1] & 15'h5555} | {_roundMask_T_31[14:0] & 15'h5555, 1'h0}, {1'h0, _roundMask_T_61[7:1] & 7'h55} | {_roundMask_T_61[6:0] & 7'h55, 1'h0}, 2'h3};
  wire [25:0] shiftedRoundMask = distGT24 ? 26'h3FFFFFF : {1'h0, roundMask[25:1]};
  wire [25:0] guardBitMask = ~shiftedRoundMask & roundMask;
  wire        guardBit = |(adjustedSig & guardBitMask);
  wire        stickyBit = |(adjustedSig & shiftedRoundMask);
  assign anyRound = guardBit | stickyBit;
  wire        commonInexact = anyRound;
  wire [25:0] lastBitMask = {guardBitMask[24:0], 1'h0};
  wire        lastBit = |(adjustedSig & lastBitMask);
  wire        distEQ24 = subnormDist == 10'h18;
  wire [1:0]  rbits = {guardBit, stickyBit};
  wire        sigIncr = rmRNE & ((&rbits) | lastBit & rbits == 2'h2) | rmRDN & input_sign & (|rbits) | rmRUP & ~input_sign & (|rbits) | rmRMM & rbits[1];
  wire [26:0] sigAfterInc;
  wire [26:0] _subSigOut_T_3 = sigAfterInc >> subnormDist[4:0];
  wire [22:0] subSigOut = distGT24 | distEQ24 ? {22'h0, sigIncr} : _subSigOut_T_3[24:2];
  wire [7:0]  subExpOut = {7'h0, sigAfterInc[26] & (~commonUnderflow | subnormDist == 10'h1)};
  wire [25:0] sigIncrement = sigIncr ? lastBitMask : 26'h0;
  assign sigAfterInc = {1'h0, adjustedSig} + {1'h0, sigIncrement};
  wire        isNaNOut = input_invalidExc | input_isNaN;
  wire        notNaNIsSpecialInfOut = (input_infiniteExc | input_isInf) & ~input_invalidExc & ~input_isNaN;
  wire        notNaNIsZero = input_isZero & ~isNaNOut;
  wire        commonCase = ~isNaNOut & ~notNaNIsSpecialInfOut & ~input_isZero;
  wire        commonOverflow;
  wire        overflow = commonCase & commonOverflow;
  wire        underflow = commonCase & commonUnderflow & commonInexact;
  wire        inexact = overflow | commonCase & commonInexact;
  wire [3:0]  overflowSele = {rmRDN, rmRUP, rmRTZ, rmRNE | rmRMM};
  wire [31:0] infiniteOut = {input_sign, 31'h7F800000};
  wire [31:0] common_infiniteOut =
    (overflowSele[0] ? infiniteOut : 32'h0) | (overflowSele[1] ? {input_sign, 31'h7F7FFFFF} : 32'h0) | (overflowSele[2] ? (input_sign ? 32'hFF7FFFFF : 32'h7F800000) : 32'h0)
    | (overflowSele[3] ? (input_sign ? 32'hFF800000 : 32'h7F7FFFFF) : 32'h0);
  wire [31:0] zeroOut = {input_sign, 31'h0};
  wire [3:0]  outSele1H = {commonCase, notNaNIsSpecialInfOut, isNaNOut, notNaNIsZero};
  assign commonOverflow = (|(input_exp[8:7])) & ~(input_exp[9]);
  assign commonUnderflow = expSubnorm[9];
  wire [22:0] commonSigOut = sigAfterInc[24:2];
  wire [7:0]  commonExpOut = input_exp[7:0] + subExpOut + 8'h7F;
  wire [31:0] commonOut = commonOverflow ? common_infiniteOut : {input_sign, commonUnderflow ? {subExpOut, subSigOut} : {commonExpOut, commonSigOut}};
  wire [31:0] _output_data_T_4 = outSele1H[0] ? zeroOut : 32'h0;
  assign output_data = {_output_data_T_4[31], _output_data_T_4[30:0] | (outSele1H[1] ? 31'h7FC00000 : 31'h0)} | (outSele1H[2] ? infiniteOut : 32'h0) | (outSele1H[3] ? commonOut : 32'h0);
  assign output_exceptionFlags = {input_invalidExc, input_infiniteExc, overflow, underflow, inexact};
endmodule

