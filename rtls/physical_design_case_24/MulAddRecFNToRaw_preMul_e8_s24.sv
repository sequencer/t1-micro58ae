module MulAddRecFNToRaw_preMul_e8_s24(
  input  [1:0]  io_op,
  input  [32:0] io_a,
                io_b,
                io_c,
  output [23:0] io_mulAddA,
                io_mulAddB,
  output [47:0] io_mulAddC,
  output        io_toPostMul_isSigNaNAny,
                io_toPostMul_isNaNAOrB,
                io_toPostMul_isInfA,
                io_toPostMul_isZeroA,
                io_toPostMul_isInfB,
                io_toPostMul_isZeroB,
                io_toPostMul_signProd,
                io_toPostMul_isNaNC,
                io_toPostMul_isInfC,
                io_toPostMul_isZeroC,
  output [9:0]  io_toPostMul_sExpSum,
  output        io_toPostMul_doSubMags,
                io_toPostMul_CIsDominant,
  output [4:0]  io_toPostMul_CDom_CAlignDist,
  output [25:0] io_toPostMul_highAlignedSigC,
  output        io_toPostMul_bit0AlignedSigC
);

  wire [1:0]  io_op_0 = io_op;
  wire [32:0] io_a_0 = io_a;
  wire [32:0] io_b_0 = io_b;
  wire [32:0] io_c_0 = io_c;
  wire        rawA_isInf;
  wire        rawA_isZero;
  wire        rawB_isInf;
  wire        rawB_isZero;
  wire        signProd;
  wire        rawC_isNaN;
  wire        rawC_isInf;
  wire        rawC_isZero;
  wire        doSubMags;
  wire        CIsDominant;
  wire [8:0]  rawA_exp = io_a_0[31:23];
  wire        rawA_isZero_0 = rawA_exp[8:6] == 3'h0;
  assign rawA_isZero = rawA_isZero_0;
  wire        rawA_isSpecial = &(rawA_exp[8:7]);
  wire        io_toPostMul_isInfA_0 = rawA_isInf;
  wire        io_toPostMul_isZeroA_0 = rawA_isZero;
  wire        rawA_isNaN = rawA_isSpecial & rawA_exp[6];
  assign rawA_isInf = rawA_isSpecial & ~(rawA_exp[6]);
  wire        rawA_sign = io_a_0[32];
  wire [9:0]  rawA_sExp = {1'h0, rawA_exp};
  wire [24:0] rawA_sig = {1'h0, ~rawA_isZero_0, io_a_0[22:0]};
  wire [8:0]  rawB_exp = io_b_0[31:23];
  wire        rawB_isZero_0 = rawB_exp[8:6] == 3'h0;
  assign rawB_isZero = rawB_isZero_0;
  wire        rawB_isSpecial = &(rawB_exp[8:7]);
  wire        io_toPostMul_isInfB_0 = rawB_isInf;
  wire        io_toPostMul_isZeroB_0 = rawB_isZero;
  wire        rawB_isNaN = rawB_isSpecial & rawB_exp[6];
  assign rawB_isInf = rawB_isSpecial & ~(rawB_exp[6]);
  wire        rawB_sign = io_b_0[32];
  wire [9:0]  rawB_sExp = {1'h0, rawB_exp};
  wire [24:0] rawB_sig = {1'h0, ~rawB_isZero_0, io_b_0[22:0]};
  wire [8:0]  rawC_exp = io_c_0[31:23];
  wire        rawC_isZero_0 = rawC_exp[8:6] == 3'h0;
  assign rawC_isZero = rawC_isZero_0;
  wire        rawC_isSpecial = &(rawC_exp[8:7]);
  wire        io_toPostMul_isNaNC_0 = rawC_isNaN;
  wire        io_toPostMul_isInfC_0 = rawC_isInf;
  wire        io_toPostMul_isZeroC_0 = rawC_isZero;
  assign rawC_isNaN = rawC_isSpecial & rawC_exp[6];
  assign rawC_isInf = rawC_isSpecial & ~(rawC_exp[6]);
  wire        rawC_sign = io_c_0[32];
  wire [9:0]  rawC_sExp = {1'h0, rawC_exp};
  wire [24:0] rawC_sig = {1'h0, ~rawC_isZero_0, io_c_0[22:0]};
  assign signProd = rawA_sign ^ rawB_sign ^ io_op_0[1];
  wire        io_toPostMul_signProd_0 = signProd;
  wire [10:0] sExpAlignedProd = {rawA_sExp[9], rawA_sExp} + {rawB_sExp[9], rawB_sExp} - 11'hE5;
  assign doSubMags = signProd ^ rawC_sign ^ io_op_0[0];
  wire        io_toPostMul_doSubMags_0 = doSubMags;
  wire [10:0] sNatCAlignDist = sExpAlignedProd - {rawC_sExp[9], rawC_sExp};
  wire [9:0]  posNatCAlignDist = sNatCAlignDist[9:0];
  wire        isMinCAlign = rawA_isZero | rawB_isZero | $signed(sNatCAlignDist) < 11'sh0;
  assign CIsDominant = ~rawC_isZero & (isMinCAlign | posNatCAlignDist < 10'h19);
  wire        io_toPostMul_CIsDominant_0 = CIsDominant;
  wire [6:0]  CAlignDist = isMinCAlign ? 7'h0 : posNatCAlignDist < 10'h4A ? posNatCAlignDist[6:0] : 7'h4A;
  wire [77:0] mainAlignedSigC = $signed($signed({{25{doSubMags}} ^ rawC_sig, {53{doSubMags}}}) >>> CAlignDist);
  wire        reduced4CExtra_reducedVec_0 = |(rawC_sig[1:0]);
  wire        reduced4CExtra_reducedVec_1 = |(rawC_sig[5:2]);
  wire        reduced4CExtra_reducedVec_2 = |(rawC_sig[9:6]);
  wire        reduced4CExtra_reducedVec_3 = |(rawC_sig[13:10]);
  wire        reduced4CExtra_reducedVec_4 = |(rawC_sig[17:14]);
  wire        reduced4CExtra_reducedVec_5 = |(rawC_sig[21:18]);
  wire        reduced4CExtra_reducedVec_6 = |(rawC_sig[24:22]);
  wire [1:0]  reduced4CExtra_lo_hi = {reduced4CExtra_reducedVec_2, reduced4CExtra_reducedVec_1};
  wire [2:0]  reduced4CExtra_lo = {reduced4CExtra_lo_hi, reduced4CExtra_reducedVec_0};
  wire [1:0]  reduced4CExtra_hi_lo = {reduced4CExtra_reducedVec_4, reduced4CExtra_reducedVec_3};
  wire [1:0]  reduced4CExtra_hi_hi = {reduced4CExtra_reducedVec_6, reduced4CExtra_reducedVec_5};
  wire [3:0]  reduced4CExtra_hi = {reduced4CExtra_hi_hi, reduced4CExtra_hi_lo};
  wire [32:0] reduced4CExtra_shift = $signed(33'sh100000000 >>> CAlignDist[6:2]);
  wire        reduced4CExtra = |({reduced4CExtra_hi[2:0], reduced4CExtra_lo} & {reduced4CExtra_shift[14], reduced4CExtra_shift[15], reduced4CExtra_shift[16], reduced4CExtra_shift[17], reduced4CExtra_shift[18], reduced4CExtra_shift[19]});
  wire [74:0] alignedSigC_hi = mainAlignedSigC[77:3];
  wire [75:0] alignedSigC = {alignedSigC_hi, doSubMags ? (&(mainAlignedSigC[2:0])) & ~reduced4CExtra : (|(mainAlignedSigC[2:0])) | reduced4CExtra};
  wire [23:0] io_mulAddA_0 = rawA_sig[23:0];
  wire [23:0] io_mulAddB_0 = rawB_sig[23:0];
  wire [47:0] io_mulAddC_0 = alignedSigC[48:1];
  wire        io_toPostMul_isSigNaNAny_0 = rawA_isNaN & ~(rawA_sig[22]) | rawB_isNaN & ~(rawB_sig[22]) | rawC_isNaN & ~(rawC_sig[22]);
  wire        io_toPostMul_isNaNAOrB_0 = rawA_isNaN | rawB_isNaN;
  wire [9:0]  io_toPostMul_sExpSum_0 = CIsDominant ? rawC_sExp : sExpAlignedProd[9:0] - 10'h18;
  wire [4:0]  io_toPostMul_CDom_CAlignDist_0 = CAlignDist[4:0];
  wire [25:0] io_toPostMul_highAlignedSigC_0 = alignedSigC[74:49];
  wire        io_toPostMul_bit0AlignedSigC_0 = alignedSigC[0];
  assign io_mulAddA = io_mulAddA_0;
  assign io_mulAddB = io_mulAddB_0;
  assign io_mulAddC = io_mulAddC_0;
  assign io_toPostMul_isSigNaNAny = io_toPostMul_isSigNaNAny_0;
  assign io_toPostMul_isNaNAOrB = io_toPostMul_isNaNAOrB_0;
  assign io_toPostMul_isInfA = io_toPostMul_isInfA_0;
  assign io_toPostMul_isZeroA = io_toPostMul_isZeroA_0;
  assign io_toPostMul_isInfB = io_toPostMul_isInfB_0;
  assign io_toPostMul_isZeroB = io_toPostMul_isZeroB_0;
  assign io_toPostMul_signProd = io_toPostMul_signProd_0;
  assign io_toPostMul_isNaNC = io_toPostMul_isNaNC_0;
  assign io_toPostMul_isInfC = io_toPostMul_isInfC_0;
  assign io_toPostMul_isZeroC = io_toPostMul_isZeroC_0;
  assign io_toPostMul_sExpSum = io_toPostMul_sExpSum_0;
  assign io_toPostMul_doSubMags = io_toPostMul_doSubMags_0;
  assign io_toPostMul_CIsDominant = io_toPostMul_CIsDominant_0;
  assign io_toPostMul_CDom_CAlignDist = io_toPostMul_CDom_CAlignDist_0;
  assign io_toPostMul_highAlignedSigC = io_toPostMul_highAlignedSigC_0;
  assign io_toPostMul_bit0AlignedSigC = io_toPostMul_bit0AlignedSigC_0;
endmodule

