module AddRawFN(
  input         io_a_isNaN,
                io_a_isInf,
                io_a_isZero,
                io_a_sign,
  input  [9:0]  io_a_sExp,
  input  [24:0] io_a_sig,
  input         io_b_isNaN,
                io_b_isInf,
                io_b_isZero,
                io_b_sign,
  input  [9:0]  io_b_sExp,
  input  [24:0] io_b_sig,
  input  [2:0]  io_roundingMode,
  output        io_invalidExc,
                io_rawOut_isNaN,
                io_rawOut_isInf,
                io_rawOut_isZero,
                io_rawOut_sign,
  output [9:0]  io_rawOut_sExp,
  output [26:0] io_rawOut_sig
);

  wire        io_a_isNaN_0 = io_a_isNaN;
  wire        io_a_isInf_0 = io_a_isInf;
  wire        io_a_isZero_0 = io_a_isZero;
  wire        io_a_sign_0 = io_a_sign;
  wire [9:0]  io_a_sExp_0 = io_a_sExp;
  wire [24:0] io_a_sig_0 = io_a_sig;
  wire        io_b_isNaN_0 = io_b_isNaN;
  wire        io_b_isInf_0 = io_b_isInf;
  wire        io_b_isZero_0 = io_b_isZero;
  wire        io_b_sign_0 = io_b_sign;
  wire [9:0]  io_b_sExp_0 = io_b_sExp;
  wire [24:0] io_b_sig_0 = io_b_sig;
  wire [2:0]  io_roundingMode_0 = io_roundingMode;
  wire        io_subOp = 1'h0;
  wire        effSignB = io_b_sign_0;
  wire        notNaN_isInfOut;
  wire        notNaN_isZeroOut;
  wire        notNaN_signOut;
  wire [9:0]  common_sExpOut;
  wire [26:0] common_sigOut;
  wire        eqSigns = io_a_sign_0 == effSignB;
  wire        notEqSigns_signZero = io_roundingMode_0 == 3'h2;
  wire [9:0]  sDiffExps = io_a_sExp_0 - io_b_sExp_0;
  wire        _common_sExpOut_T = $signed(sDiffExps) < 10'sh0;
  wire [4:0]  modNatAlignDist = _common_sExpOut_T ? io_b_sExp_0[4:0] - io_a_sExp_0[4:0] : sDiffExps[4:0];
  wire        isMaxAlign = (|(sDiffExps[9:5])) & (sDiffExps[9:5] != 5'h1F | sDiffExps[4:0] == 5'h0);
  wire [4:0]  alignDist = isMaxAlign ? 5'h1F : modNatAlignDist;
  wire        closeSubMags = ~eqSigns & ~isMaxAlign & modNatAlignDist < 5'h2;
  wire        _close_alignedSigA_T_5 = $signed(sDiffExps) > -10'sh1;
  wire [26:0] _close_alignedSigA_T_4 = _close_alignedSigA_T_5 & sDiffExps[0] ? {io_a_sig_0, 2'h0} : 27'h0;
  wire [25:0] _GEN = _close_alignedSigA_T_4[25:0] | (_close_alignedSigA_T_5 & ~(sDiffExps[0]) ? {io_a_sig_0, 1'h0} : 26'h0);
  wire [26:0] close_alignedSigA = {_close_alignedSigA_T_4[26], _GEN[25], _GEN[24:0] | (_common_sExpOut_T ? io_a_sig_0 : 25'h0)};
  wire [26:0] close_sSigSum = close_alignedSigA - {io_b_sig_0[24], io_b_sig_0, 1'h0};
  wire        _close_notTotalCancellation_signOut_T = $signed(close_sSigSum) < 27'sh0;
  wire [25:0] close_sigSum = _close_notTotalCancellation_signOut_T ? 26'h0 - close_sSigSum[25:0] : close_sSigSum[25:0];
  wire [25:0] close_adjustedSigSum = close_sigSum;
  wire        close_reduced2SigSum_reducedVec_0 = |(close_adjustedSigSum[1:0]);
  wire        close_reduced2SigSum_reducedVec_1 = |(close_adjustedSigSum[3:2]);
  wire        close_reduced2SigSum_reducedVec_2 = |(close_adjustedSigSum[5:4]);
  wire        close_reduced2SigSum_reducedVec_3 = |(close_adjustedSigSum[7:6]);
  wire        close_reduced2SigSum_reducedVec_4 = |(close_adjustedSigSum[9:8]);
  wire        close_reduced2SigSum_reducedVec_5 = |(close_adjustedSigSum[11:10]);
  wire        close_reduced2SigSum_reducedVec_6 = |(close_adjustedSigSum[13:12]);
  wire        close_reduced2SigSum_reducedVec_7 = |(close_adjustedSigSum[15:14]);
  wire        close_reduced2SigSum_reducedVec_8 = |(close_adjustedSigSum[17:16]);
  wire        close_reduced2SigSum_reducedVec_9 = |(close_adjustedSigSum[19:18]);
  wire        close_reduced2SigSum_reducedVec_10 = |(close_adjustedSigSum[21:20]);
  wire        close_reduced2SigSum_reducedVec_11 = |(close_adjustedSigSum[23:22]);
  wire        close_reduced2SigSum_reducedVec_12 = |(close_adjustedSigSum[25:24]);
  wire [1:0]  close_reduced2SigSum_lo_lo_hi = {close_reduced2SigSum_reducedVec_2, close_reduced2SigSum_reducedVec_1};
  wire [2:0]  close_reduced2SigSum_lo_lo = {close_reduced2SigSum_lo_lo_hi, close_reduced2SigSum_reducedVec_0};
  wire [1:0]  close_reduced2SigSum_lo_hi_hi = {close_reduced2SigSum_reducedVec_5, close_reduced2SigSum_reducedVec_4};
  wire [2:0]  close_reduced2SigSum_lo_hi = {close_reduced2SigSum_lo_hi_hi, close_reduced2SigSum_reducedVec_3};
  wire [5:0]  close_reduced2SigSum_lo = {close_reduced2SigSum_lo_hi, close_reduced2SigSum_lo_lo};
  wire [1:0]  close_reduced2SigSum_hi_lo_hi = {close_reduced2SigSum_reducedVec_8, close_reduced2SigSum_reducedVec_7};
  wire [2:0]  close_reduced2SigSum_hi_lo = {close_reduced2SigSum_hi_lo_hi, close_reduced2SigSum_reducedVec_6};
  wire [1:0]  close_reduced2SigSum_hi_hi_lo = {close_reduced2SigSum_reducedVec_10, close_reduced2SigSum_reducedVec_9};
  wire [1:0]  close_reduced2SigSum_hi_hi_hi = {close_reduced2SigSum_reducedVec_12, close_reduced2SigSum_reducedVec_11};
  wire [3:0]  close_reduced2SigSum_hi_hi = {close_reduced2SigSum_hi_hi_hi, close_reduced2SigSum_hi_hi_lo};
  wire [6:0]  close_reduced2SigSum_hi = {close_reduced2SigSum_hi_hi, close_reduced2SigSum_hi_lo};
  wire [12:0] close_reduced2SigSum = {close_reduced2SigSum_hi, close_reduced2SigSum_lo};
  wire [3:0]  close_normDistReduced2 =
    close_reduced2SigSum[12]
      ? 4'h0
      : close_reduced2SigSum[11]
          ? 4'h1
          : close_reduced2SigSum[10]
              ? 4'h2
              : close_reduced2SigSum[9]
                  ? 4'h3
                  : close_reduced2SigSum[8]
                      ? 4'h4
                      : close_reduced2SigSum[7]
                          ? 4'h5
                          : close_reduced2SigSum[6] ? 4'h6 : close_reduced2SigSum[5] ? 4'h7 : close_reduced2SigSum[4] ? 4'h8 : close_reduced2SigSum[3] ? 4'h9 : close_reduced2SigSum[2] ? 4'hA : close_reduced2SigSum[1] ? 4'hB : 4'hC;
  wire [4:0]  close_nearNormDist = {close_normDistReduced2, 1'h0};
  wire [56:0] _close_sigOut_T = {31'h0, close_sigSum} << close_nearNormDist;
  wire [26:0] close_sigOut = {_close_sigOut_T[25:0], 1'h0};
  wire        close_totalCancellation = close_sigOut[26:25] == 2'h0;
  wire        close_notTotalCancellation_signOut = io_a_sign_0 ^ _close_notTotalCancellation_signOut_T;
  wire        far_signOut = _common_sExpOut_T ? effSignB : io_a_sign_0;
  wire [23:0] far_sigLarger = _common_sExpOut_T ? io_b_sig_0[23:0] : io_a_sig_0[23:0];
  wire [23:0] far_sigSmaller = _common_sExpOut_T ? io_a_sig_0[23:0] : io_b_sig_0[23:0];
  wire [28:0] far_mainAlignedSigSmaller = {far_sigSmaller, 5'h0} >> alignDist;
  wire        far_reduced4SigSmaller_reducedVec_0 = |(far_sigSmaller[1:0]);
  wire        far_reduced4SigSmaller_reducedVec_1 = |(far_sigSmaller[5:2]);
  wire        far_reduced4SigSmaller_reducedVec_2 = |(far_sigSmaller[9:6]);
  wire        far_reduced4SigSmaller_reducedVec_3 = |(far_sigSmaller[13:10]);
  wire        far_reduced4SigSmaller_reducedVec_4 = |(far_sigSmaller[17:14]);
  wire        far_reduced4SigSmaller_reducedVec_5 = |(far_sigSmaller[21:18]);
  wire        far_reduced4SigSmaller_reducedVec_6 = |(far_sigSmaller[23:22]);
  wire [1:0]  far_reduced4SigSmaller_lo_hi = {far_reduced4SigSmaller_reducedVec_2, far_reduced4SigSmaller_reducedVec_1};
  wire [2:0]  far_reduced4SigSmaller_lo = {far_reduced4SigSmaller_lo_hi, far_reduced4SigSmaller_reducedVec_0};
  wire [1:0]  far_reduced4SigSmaller_hi_lo = {far_reduced4SigSmaller_reducedVec_4, far_reduced4SigSmaller_reducedVec_3};
  wire [1:0]  far_reduced4SigSmaller_hi_hi = {far_reduced4SigSmaller_reducedVec_6, far_reduced4SigSmaller_reducedVec_5};
  wire [3:0]  far_reduced4SigSmaller_hi = {far_reduced4SigSmaller_hi_hi, far_reduced4SigSmaller_hi_lo};
  wire [6:0]  far_reduced4SigSmaller = {far_reduced4SigSmaller_hi, far_reduced4SigSmaller_lo};
  wire [8:0]  far_roundExtraMask_shift = $signed(9'sh100 >>> alignDist[4:2]);
  wire [6:0]  far_roundExtraMask = {far_roundExtraMask_shift[1], far_roundExtraMask_shift[2], far_roundExtraMask_shift[3], far_roundExtraMask_shift[4], far_roundExtraMask_shift[5], far_roundExtraMask_shift[6], far_roundExtraMask_shift[7]};
  wire [26:0] far_alignedSigSmaller = {far_mainAlignedSigSmaller[28:3], |{far_mainAlignedSigSmaller[2:0], far_reduced4SigSmaller & far_roundExtraMask}};
  wire        far_subMags;
  assign far_subMags = ~eqSigns;
  wire [27:0] far_negAlignedSigSmaller = far_subMags ? {1'h1, ~far_alignedSigSmaller} : {1'h0, far_alignedSigSmaller};
  wire [27:0] far_sigSum = {1'h0, far_sigLarger, 3'h0} + far_negAlignedSigSmaller + {27'h0, far_subMags};
  wire [26:0] far_sigOut = far_subMags ? far_sigSum[26:0] : {far_sigSum[27:2], far_sigSum[1] | far_sigSum[0]};
  wire        notSigNaN_invalidExc = io_a_isInf_0 & io_b_isInf_0 & ~eqSigns;
  assign notNaN_isInfOut = io_a_isInf_0 | io_b_isInf_0;
  wire        io_rawOut_isInf_0 = notNaN_isInfOut;
  wire        addZeros = io_a_isZero_0 & io_b_isZero_0;
  wire        notNaN_specialCase = notNaN_isInfOut | addZeros;
  assign notNaN_isZeroOut = addZeros | ~notNaN_isInfOut & closeSubMags & close_totalCancellation;
  wire        io_rawOut_isZero_0 = notNaN_isZeroOut;
  assign notNaN_signOut =
    eqSigns & io_a_sign_0 | io_a_isInf_0 & io_a_sign_0 | io_b_isInf_0 & effSignB | notNaN_isZeroOut & ~eqSigns & notEqSigns_signZero | ~notNaN_specialCase & closeSubMags & ~close_totalCancellation & close_notTotalCancellation_signOut
    | ~notNaN_specialCase & ~closeSubMags & far_signOut;
  wire        io_rawOut_sign_0 = notNaN_signOut;
  assign common_sExpOut = (closeSubMags | _common_sExpOut_T ? io_b_sExp_0 : io_a_sExp_0) - {5'h0, closeSubMags ? close_nearNormDist : {4'h0, far_subMags}};
  wire [9:0]  io_rawOut_sExp_0 = common_sExpOut;
  assign common_sigOut = closeSubMags ? close_sigOut : far_sigOut;
  wire [26:0] io_rawOut_sig_0 = common_sigOut;
  wire        io_invalidExc_0 = io_a_isNaN_0 & ~(io_a_sig_0[22]) | io_b_isNaN_0 & ~(io_b_sig_0[22]) | notSigNaN_invalidExc;
  wire        io_rawOut_isNaN_0 = io_a_isNaN_0 | io_b_isNaN_0;
  assign io_invalidExc = io_invalidExc_0;
  assign io_rawOut_isNaN = io_rawOut_isNaN_0;
  assign io_rawOut_isInf = io_rawOut_isInf_0;
  assign io_rawOut_isZero = io_rawOut_isZero_0;
  assign io_rawOut_sign = io_rawOut_sign_0;
  assign io_rawOut_sExp = io_rawOut_sExp_0;
  assign io_rawOut_sig = io_rawOut_sig_0;
endmodule

