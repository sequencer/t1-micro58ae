module MulAddRecFNToRaw_postMul_e8_s24(
  input         io_fromPreMul_isSigNaNAny,
                io_fromPreMul_isNaNAOrB,
                io_fromPreMul_isInfA,
                io_fromPreMul_isZeroA,
                io_fromPreMul_isInfB,
                io_fromPreMul_isZeroB,
                io_fromPreMul_signProd,
                io_fromPreMul_isNaNC,
                io_fromPreMul_isInfC,
                io_fromPreMul_isZeroC,
  input  [9:0]  io_fromPreMul_sExpSum,
  input         io_fromPreMul_doSubMags,
                io_fromPreMul_CIsDominant,
  input  [4:0]  io_fromPreMul_CDom_CAlignDist,
  input  [25:0] io_fromPreMul_highAlignedSigC,
  input         io_fromPreMul_bit0AlignedSigC,
  input  [48:0] io_mulAddResult,
  input  [2:0]  io_roundingMode,
  output        io_invalidExc,
                io_rawOut_isNaN,
                io_rawOut_isInf,
                io_rawOut_isZero,
                io_rawOut_sign,
  output [9:0]  io_rawOut_sExp,
  output [26:0] io_rawOut_sig
);

  wire         io_fromPreMul_isSigNaNAny_0 = io_fromPreMul_isSigNaNAny;
  wire         io_fromPreMul_isNaNAOrB_0 = io_fromPreMul_isNaNAOrB;
  wire         io_fromPreMul_isInfA_0 = io_fromPreMul_isInfA;
  wire         io_fromPreMul_isZeroA_0 = io_fromPreMul_isZeroA;
  wire         io_fromPreMul_isInfB_0 = io_fromPreMul_isInfB;
  wire         io_fromPreMul_isZeroB_0 = io_fromPreMul_isZeroB;
  wire         io_fromPreMul_signProd_0 = io_fromPreMul_signProd;
  wire         io_fromPreMul_isNaNC_0 = io_fromPreMul_isNaNC;
  wire         io_fromPreMul_isInfC_0 = io_fromPreMul_isInfC;
  wire         io_fromPreMul_isZeroC_0 = io_fromPreMul_isZeroC;
  wire [9:0]   io_fromPreMul_sExpSum_0 = io_fromPreMul_sExpSum;
  wire         io_fromPreMul_doSubMags_0 = io_fromPreMul_doSubMags;
  wire         io_fromPreMul_CIsDominant_0 = io_fromPreMul_CIsDominant;
  wire [4:0]   io_fromPreMul_CDom_CAlignDist_0 = io_fromPreMul_CDom_CAlignDist;
  wire [25:0]  io_fromPreMul_highAlignedSigC_0 = io_fromPreMul_highAlignedSigC;
  wire         io_fromPreMul_bit0AlignedSigC_0 = io_fromPreMul_bit0AlignedSigC;
  wire [48:0]  io_mulAddResult_0 = io_mulAddResult;
  wire [2:0]   io_roundingMode_0 = io_roundingMode;
  wire         notNaN_isInfOut;
  wire         roundingMode_min = io_roundingMode_0 == 3'h2;
  wire         opSignC = io_fromPreMul_signProd_0 ^ io_fromPreMul_doSubMags_0;
  wire [73:0]  sigSum_hi = {io_mulAddResult_0[48] ? io_fromPreMul_highAlignedSigC_0 + 26'h1 : io_fromPreMul_highAlignedSigC_0, io_mulAddResult_0[47:0]};
  wire [74:0]  sigSum = {sigSum_hi, io_fromPreMul_bit0AlignedSigC_0};
  wire [9:0]   CDom_sExp = io_fromPreMul_sExpSum_0 - {9'h0, io_fromPreMul_doSubMags_0};
  wire [49:0]  CDom_absSigSum = io_fromPreMul_doSubMags_0 ? ~(sigSum[74:25]) : {1'h0, io_fromPreMul_highAlignedSigC_0[25:24], sigSum[72:26]};
  wire         CDom_absSigSumExtra = io_fromPreMul_doSubMags_0 ? sigSum[24:1] != 24'hFFFFFF : (|(sigSum[25:1]));
  wire [80:0]  _CDom_mainSig_T = {31'h0, CDom_absSigSum} << io_fromPreMul_CDom_CAlignDist_0;
  wire [28:0]  CDom_mainSig = _CDom_mainSig_T[49:21];
  wire         CDom_reduced4SigExtra_reducedVec_0 = CDom_absSigSum[0];
  wire         CDom_reduced4SigExtra_reducedVec_1 = |(CDom_absSigSum[4:1]);
  wire         CDom_reduced4SigExtra_reducedVec_2 = |(CDom_absSigSum[8:5]);
  wire         CDom_reduced4SigExtra_reducedVec_3 = |(CDom_absSigSum[12:9]);
  wire         CDom_reduced4SigExtra_reducedVec_4 = |(CDom_absSigSum[16:13]);
  wire         CDom_reduced4SigExtra_reducedVec_5 = |(CDom_absSigSum[20:17]);
  wire         CDom_reduced4SigExtra_reducedVec_6 = |(CDom_absSigSum[23:21]);
  wire [1:0]   CDom_reduced4SigExtra_lo_hi = {CDom_reduced4SigExtra_reducedVec_2, CDom_reduced4SigExtra_reducedVec_1};
  wire [2:0]   CDom_reduced4SigExtra_lo = {CDom_reduced4SigExtra_lo_hi, CDom_reduced4SigExtra_reducedVec_0};
  wire [1:0]   CDom_reduced4SigExtra_hi_lo = {CDom_reduced4SigExtra_reducedVec_4, CDom_reduced4SigExtra_reducedVec_3};
  wire [1:0]   CDom_reduced4SigExtra_hi_hi = {CDom_reduced4SigExtra_reducedVec_6, CDom_reduced4SigExtra_reducedVec_5};
  wire [3:0]   CDom_reduced4SigExtra_hi = {CDom_reduced4SigExtra_hi_hi, CDom_reduced4SigExtra_hi_lo};
  wire [8:0]   CDom_reduced4SigExtra_shift = $signed(9'sh100 >>> ~(io_fromPreMul_CDom_CAlignDist_0[4:2]));
  wire         CDom_reduced4SigExtra =
    |({CDom_reduced4SigExtra_hi[2:0], CDom_reduced4SigExtra_lo}
      & {CDom_reduced4SigExtra_shift[1], CDom_reduced4SigExtra_shift[2], CDom_reduced4SigExtra_shift[3], CDom_reduced4SigExtra_shift[4], CDom_reduced4SigExtra_shift[5], CDom_reduced4SigExtra_shift[6]});
  wire [26:0]  CDom_sig = {CDom_mainSig[28:3], (|(CDom_mainSig[2:0])) | CDom_reduced4SigExtra | CDom_absSigSumExtra};
  wire         notCDom_signSigSum = sigSum[51];
  wire [50:0]  notCDom_absSigSum = notCDom_signSigSum ? ~(sigSum[50:0]) : sigSum[50:0] + {50'h0, io_fromPreMul_doSubMags_0};
  wire         notCDom_reduced2AbsSigSum_reducedVec_0 = |(notCDom_absSigSum[1:0]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_1 = |(notCDom_absSigSum[3:2]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_2 = |(notCDom_absSigSum[5:4]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_3 = |(notCDom_absSigSum[7:6]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_4 = |(notCDom_absSigSum[9:8]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_5 = |(notCDom_absSigSum[11:10]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_6 = |(notCDom_absSigSum[13:12]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_7 = |(notCDom_absSigSum[15:14]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_8 = |(notCDom_absSigSum[17:16]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_9 = |(notCDom_absSigSum[19:18]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_10 = |(notCDom_absSigSum[21:20]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_11 = |(notCDom_absSigSum[23:22]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_12 = |(notCDom_absSigSum[25:24]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_13 = |(notCDom_absSigSum[27:26]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_14 = |(notCDom_absSigSum[29:28]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_15 = |(notCDom_absSigSum[31:30]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_16 = |(notCDom_absSigSum[33:32]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_17 = |(notCDom_absSigSum[35:34]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_18 = |(notCDom_absSigSum[37:36]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_19 = |(notCDom_absSigSum[39:38]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_20 = |(notCDom_absSigSum[41:40]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_21 = |(notCDom_absSigSum[43:42]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_22 = |(notCDom_absSigSum[45:44]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_23 = |(notCDom_absSigSum[47:46]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_24 = |(notCDom_absSigSum[49:48]);
  wire         notCDom_reduced2AbsSigSum_reducedVec_25 = notCDom_absSigSum[50];
  wire [1:0]   notCDom_reduced2AbsSigSum_lo_lo_lo_hi = {notCDom_reduced2AbsSigSum_reducedVec_2, notCDom_reduced2AbsSigSum_reducedVec_1};
  wire [2:0]   notCDom_reduced2AbsSigSum_lo_lo_lo = {notCDom_reduced2AbsSigSum_lo_lo_lo_hi, notCDom_reduced2AbsSigSum_reducedVec_0};
  wire [1:0]   notCDom_reduced2AbsSigSum_lo_lo_hi_hi = {notCDom_reduced2AbsSigSum_reducedVec_5, notCDom_reduced2AbsSigSum_reducedVec_4};
  wire [2:0]   notCDom_reduced2AbsSigSum_lo_lo_hi = {notCDom_reduced2AbsSigSum_lo_lo_hi_hi, notCDom_reduced2AbsSigSum_reducedVec_3};
  wire [5:0]   notCDom_reduced2AbsSigSum_lo_lo = {notCDom_reduced2AbsSigSum_lo_lo_hi, notCDom_reduced2AbsSigSum_lo_lo_lo};
  wire [1:0]   notCDom_reduced2AbsSigSum_lo_hi_lo_hi = {notCDom_reduced2AbsSigSum_reducedVec_8, notCDom_reduced2AbsSigSum_reducedVec_7};
  wire [2:0]   notCDom_reduced2AbsSigSum_lo_hi_lo = {notCDom_reduced2AbsSigSum_lo_hi_lo_hi, notCDom_reduced2AbsSigSum_reducedVec_6};
  wire [1:0]   notCDom_reduced2AbsSigSum_lo_hi_hi_lo = {notCDom_reduced2AbsSigSum_reducedVec_10, notCDom_reduced2AbsSigSum_reducedVec_9};
  wire [1:0]   notCDom_reduced2AbsSigSum_lo_hi_hi_hi = {notCDom_reduced2AbsSigSum_reducedVec_12, notCDom_reduced2AbsSigSum_reducedVec_11};
  wire [3:0]   notCDom_reduced2AbsSigSum_lo_hi_hi = {notCDom_reduced2AbsSigSum_lo_hi_hi_hi, notCDom_reduced2AbsSigSum_lo_hi_hi_lo};
  wire [6:0]   notCDom_reduced2AbsSigSum_lo_hi = {notCDom_reduced2AbsSigSum_lo_hi_hi, notCDom_reduced2AbsSigSum_lo_hi_lo};
  wire [12:0]  notCDom_reduced2AbsSigSum_lo = {notCDom_reduced2AbsSigSum_lo_hi, notCDom_reduced2AbsSigSum_lo_lo};
  wire [1:0]   notCDom_reduced2AbsSigSum_hi_lo_lo_hi = {notCDom_reduced2AbsSigSum_reducedVec_15, notCDom_reduced2AbsSigSum_reducedVec_14};
  wire [2:0]   notCDom_reduced2AbsSigSum_hi_lo_lo = {notCDom_reduced2AbsSigSum_hi_lo_lo_hi, notCDom_reduced2AbsSigSum_reducedVec_13};
  wire [1:0]   notCDom_reduced2AbsSigSum_hi_lo_hi_hi = {notCDom_reduced2AbsSigSum_reducedVec_18, notCDom_reduced2AbsSigSum_reducedVec_17};
  wire [2:0]   notCDom_reduced2AbsSigSum_hi_lo_hi = {notCDom_reduced2AbsSigSum_hi_lo_hi_hi, notCDom_reduced2AbsSigSum_reducedVec_16};
  wire [5:0]   notCDom_reduced2AbsSigSum_hi_lo = {notCDom_reduced2AbsSigSum_hi_lo_hi, notCDom_reduced2AbsSigSum_hi_lo_lo};
  wire [1:0]   notCDom_reduced2AbsSigSum_hi_hi_lo_hi = {notCDom_reduced2AbsSigSum_reducedVec_21, notCDom_reduced2AbsSigSum_reducedVec_20};
  wire [2:0]   notCDom_reduced2AbsSigSum_hi_hi_lo = {notCDom_reduced2AbsSigSum_hi_hi_lo_hi, notCDom_reduced2AbsSigSum_reducedVec_19};
  wire [1:0]   notCDom_reduced2AbsSigSum_hi_hi_hi_lo = {notCDom_reduced2AbsSigSum_reducedVec_23, notCDom_reduced2AbsSigSum_reducedVec_22};
  wire [1:0]   notCDom_reduced2AbsSigSum_hi_hi_hi_hi = {notCDom_reduced2AbsSigSum_reducedVec_25, notCDom_reduced2AbsSigSum_reducedVec_24};
  wire [3:0]   notCDom_reduced2AbsSigSum_hi_hi_hi = {notCDom_reduced2AbsSigSum_hi_hi_hi_hi, notCDom_reduced2AbsSigSum_hi_hi_hi_lo};
  wire [6:0]   notCDom_reduced2AbsSigSum_hi_hi = {notCDom_reduced2AbsSigSum_hi_hi_hi, notCDom_reduced2AbsSigSum_hi_hi_lo};
  wire [12:0]  notCDom_reduced2AbsSigSum_hi = {notCDom_reduced2AbsSigSum_hi_hi, notCDom_reduced2AbsSigSum_hi_lo};
  wire [25:0]  notCDom_reduced2AbsSigSum = {notCDom_reduced2AbsSigSum_hi, notCDom_reduced2AbsSigSum_lo};
  wire [4:0]   notCDom_normDistReduced2 =
    notCDom_reduced2AbsSigSum[25]
      ? 5'h0
      : notCDom_reduced2AbsSigSum[24]
          ? 5'h1
          : notCDom_reduced2AbsSigSum[23]
              ? 5'h2
              : notCDom_reduced2AbsSigSum[22]
                  ? 5'h3
                  : notCDom_reduced2AbsSigSum[21]
                      ? 5'h4
                      : notCDom_reduced2AbsSigSum[20]
                          ? 5'h5
                          : notCDom_reduced2AbsSigSum[19]
                              ? 5'h6
                              : notCDom_reduced2AbsSigSum[18]
                                  ? 5'h7
                                  : notCDom_reduced2AbsSigSum[17]
                                      ? 5'h8
                                      : notCDom_reduced2AbsSigSum[16]
                                          ? 5'h9
                                          : notCDom_reduced2AbsSigSum[15]
                                              ? 5'hA
                                              : notCDom_reduced2AbsSigSum[14]
                                                  ? 5'hB
                                                  : notCDom_reduced2AbsSigSum[13]
                                                      ? 5'hC
                                                      : notCDom_reduced2AbsSigSum[12]
                                                          ? 5'hD
                                                          : notCDom_reduced2AbsSigSum[11]
                                                              ? 5'hE
                                                              : notCDom_reduced2AbsSigSum[10]
                                                                  ? 5'hF
                                                                  : notCDom_reduced2AbsSigSum[9]
                                                                      ? 5'h10
                                                                      : notCDom_reduced2AbsSigSum[8]
                                                                          ? 5'h11
                                                                          : notCDom_reduced2AbsSigSum[7]
                                                                              ? 5'h12
                                                                              : notCDom_reduced2AbsSigSum[6]
                                                                                  ? 5'h13
                                                                                  : notCDom_reduced2AbsSigSum[5]
                                                                                      ? 5'h14
                                                                                      : notCDom_reduced2AbsSigSum[4]
                                                                                          ? 5'h15
                                                                                          : notCDom_reduced2AbsSigSum[3] ? 5'h16 : notCDom_reduced2AbsSigSum[2] ? 5'h17 : {4'hC, ~(notCDom_reduced2AbsSigSum[1])};
  wire [5:0]   notCDom_nearNormDist = {notCDom_normDistReduced2, 1'h0};
  wire [9:0]   notCDom_sExp = io_fromPreMul_sExpSum_0 - {4'h0, notCDom_nearNormDist};
  wire [113:0] _notCDom_mainSig_T = {63'h0, notCDom_absSigSum} << notCDom_nearNormDist;
  wire [28:0]  notCDom_mainSig = _notCDom_mainSig_T[51:23];
  wire         notCDom_reduced4SigExtra_reducedVec_0 = |(notCDom_reduced2AbsSigSum[1:0]);
  wire         notCDom_reduced4SigExtra_reducedVec_1 = |(notCDom_reduced2AbsSigSum[3:2]);
  wire         notCDom_reduced4SigExtra_reducedVec_2 = |(notCDom_reduced2AbsSigSum[5:4]);
  wire         notCDom_reduced4SigExtra_reducedVec_3 = |(notCDom_reduced2AbsSigSum[7:6]);
  wire         notCDom_reduced4SigExtra_reducedVec_4 = |(notCDom_reduced2AbsSigSum[9:8]);
  wire         notCDom_reduced4SigExtra_reducedVec_5 = |(notCDom_reduced2AbsSigSum[11:10]);
  wire         notCDom_reduced4SigExtra_reducedVec_6 = notCDom_reduced2AbsSigSum[12];
  wire [1:0]   notCDom_reduced4SigExtra_lo_hi = {notCDom_reduced4SigExtra_reducedVec_2, notCDom_reduced4SigExtra_reducedVec_1};
  wire [2:0]   notCDom_reduced4SigExtra_lo = {notCDom_reduced4SigExtra_lo_hi, notCDom_reduced4SigExtra_reducedVec_0};
  wire [1:0]   notCDom_reduced4SigExtra_hi_lo = {notCDom_reduced4SigExtra_reducedVec_4, notCDom_reduced4SigExtra_reducedVec_3};
  wire [1:0]   notCDom_reduced4SigExtra_hi_hi = {notCDom_reduced4SigExtra_reducedVec_6, notCDom_reduced4SigExtra_reducedVec_5};
  wire [3:0]   notCDom_reduced4SigExtra_hi = {notCDom_reduced4SigExtra_hi_hi, notCDom_reduced4SigExtra_hi_lo};
  wire [16:0]  notCDom_reduced4SigExtra_shift = $signed(17'sh10000 >>> ~(notCDom_normDistReduced2[4:1]));
  wire         notCDom_reduced4SigExtra =
    |({notCDom_reduced4SigExtra_hi[2:0], notCDom_reduced4SigExtra_lo}
      & {notCDom_reduced4SigExtra_shift[1], notCDom_reduced4SigExtra_shift[2], notCDom_reduced4SigExtra_shift[3], notCDom_reduced4SigExtra_shift[4], notCDom_reduced4SigExtra_shift[5], notCDom_reduced4SigExtra_shift[6]});
  wire [26:0]  notCDom_sig = {notCDom_mainSig[28:3], (|(notCDom_mainSig[2:0])) | notCDom_reduced4SigExtra};
  wire         notCDom_completeCancellation = notCDom_sig[26:25] == 2'h0;
  wire         notCDom_sign = notCDom_completeCancellation ? roundingMode_min : io_fromPreMul_signProd_0 ^ notCDom_signSigSum;
  wire         notNaN_isInfProd = io_fromPreMul_isInfA_0 | io_fromPreMul_isInfB_0;
  assign notNaN_isInfOut = notNaN_isInfProd | io_fromPreMul_isInfC_0;
  wire         io_rawOut_isInf_0 = notNaN_isInfOut;
  wire         notNaN_addZeros = (io_fromPreMul_isZeroA_0 | io_fromPreMul_isZeroB_0) & io_fromPreMul_isZeroC_0;
  wire         io_invalidExc_0 =
    io_fromPreMul_isSigNaNAny_0 | io_fromPreMul_isInfA_0 & io_fromPreMul_isZeroB_0 | io_fromPreMul_isZeroA_0 & io_fromPreMul_isInfB_0 | ~io_fromPreMul_isNaNAOrB_0 & notNaN_isInfProd & io_fromPreMul_isInfC_0 & io_fromPreMul_doSubMags_0;
  wire         io_rawOut_isNaN_0 = io_fromPreMul_isNaNAOrB_0 | io_fromPreMul_isNaNC_0;
  wire         io_rawOut_isZero_0 = notNaN_addZeros | ~io_fromPreMul_CIsDominant_0 & notCDom_completeCancellation;
  wire         io_rawOut_sign_0 =
    notNaN_isInfProd & io_fromPreMul_signProd_0 | io_fromPreMul_isInfC_0 & opSignC | notNaN_addZeros & ~roundingMode_min & io_fromPreMul_signProd_0 & opSignC | notNaN_addZeros & roundingMode_min & (io_fromPreMul_signProd_0 | opSignC)
    | ~notNaN_isInfOut & ~notNaN_addZeros & (io_fromPreMul_CIsDominant_0 ? opSignC : notCDom_sign);
  wire [9:0]   io_rawOut_sExp_0 = io_fromPreMul_CIsDominant_0 ? CDom_sExp : notCDom_sExp;
  wire [26:0]  io_rawOut_sig_0 = io_fromPreMul_CIsDominant_0 ? CDom_sig : notCDom_sig;
  assign io_invalidExc = io_invalidExc_0;
  assign io_rawOut_isNaN = io_rawOut_isNaN_0;
  assign io_rawOut_isInf = io_rawOut_isInf_0;
  assign io_rawOut_isZero = io_rawOut_isZero_0;
  assign io_rawOut_sign = io_rawOut_sign_0;
  assign io_rawOut_sExp = io_rawOut_sExp_0;
  assign io_rawOut_sig = io_rawOut_sig_0;
endmodule

