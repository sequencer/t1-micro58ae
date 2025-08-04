module MulAddRecFN_e8_s24(
  input  [1:0]  io_op,
  input  [32:0] io_a,
                io_b,
                io_c,
  input  [2:0]  io_roundingMode,
  output [32:0] io_out,
  output [4:0]  io_exceptionFlags
);

  wire        _mulAddRecFNToRaw_postMul_io_invalidExc;
  wire        _mulAddRecFNToRaw_postMul_io_rawOut_isNaN;
  wire        _mulAddRecFNToRaw_postMul_io_rawOut_isInf;
  wire        _mulAddRecFNToRaw_postMul_io_rawOut_isZero;
  wire        _mulAddRecFNToRaw_postMul_io_rawOut_sign;
  wire [9:0]  _mulAddRecFNToRaw_postMul_io_rawOut_sExp;
  wire [26:0] _mulAddRecFNToRaw_postMul_io_rawOut_sig;
  wire [23:0] _mulAddRecFNToRaw_preMul_io_mulAddA;
  wire [23:0] _mulAddRecFNToRaw_preMul_io_mulAddB;
  wire [47:0] _mulAddRecFNToRaw_preMul_io_mulAddC;
  wire        _mulAddRecFNToRaw_preMul_io_toPostMul_isSigNaNAny;
  wire        _mulAddRecFNToRaw_preMul_io_toPostMul_isNaNAOrB;
  wire        _mulAddRecFNToRaw_preMul_io_toPostMul_isInfA;
  wire        _mulAddRecFNToRaw_preMul_io_toPostMul_isZeroA;
  wire        _mulAddRecFNToRaw_preMul_io_toPostMul_isInfB;
  wire        _mulAddRecFNToRaw_preMul_io_toPostMul_isZeroB;
  wire        _mulAddRecFNToRaw_preMul_io_toPostMul_signProd;
  wire        _mulAddRecFNToRaw_preMul_io_toPostMul_isNaNC;
  wire        _mulAddRecFNToRaw_preMul_io_toPostMul_isInfC;
  wire        _mulAddRecFNToRaw_preMul_io_toPostMul_isZeroC;
  wire [9:0]  _mulAddRecFNToRaw_preMul_io_toPostMul_sExpSum;
  wire        _mulAddRecFNToRaw_preMul_io_toPostMul_doSubMags;
  wire        _mulAddRecFNToRaw_preMul_io_toPostMul_CIsDominant;
  wire [4:0]  _mulAddRecFNToRaw_preMul_io_toPostMul_CDom_CAlignDist;
  wire [25:0] _mulAddRecFNToRaw_preMul_io_toPostMul_highAlignedSigC;
  wire        _mulAddRecFNToRaw_preMul_io_toPostMul_bit0AlignedSigC;
  wire [4:0]  io_exceptionFlags_0;
  wire [32:0] io_out_0;
  wire [1:0]  io_op_0 = io_op;
  wire [32:0] io_a_0 = io_a;
  wire [32:0] io_b_0 = io_b;
  wire [32:0] io_c_0 = io_c;
  wire [2:0]  io_roundingMode_0 = io_roundingMode;
  wire        io_detectTininess = 1'h0;
  wire [48:0] mulAddResult = {1'h0, {24'h0, _mulAddRecFNToRaw_preMul_io_mulAddA} * {24'h0, _mulAddRecFNToRaw_preMul_io_mulAddB}} + {1'h0, _mulAddRecFNToRaw_preMul_io_mulAddC};
  MulAddRecFNToRaw_preMul_e8_s24 mulAddRecFNToRaw_preMul (
    .io_op                        (io_op_0),
    .io_a                         (io_a_0),
    .io_b                         (io_b_0),
    .io_c                         (io_c_0),
    .io_mulAddA                   (_mulAddRecFNToRaw_preMul_io_mulAddA),
    .io_mulAddB                   (_mulAddRecFNToRaw_preMul_io_mulAddB),
    .io_mulAddC                   (_mulAddRecFNToRaw_preMul_io_mulAddC),
    .io_toPostMul_isSigNaNAny     (_mulAddRecFNToRaw_preMul_io_toPostMul_isSigNaNAny),
    .io_toPostMul_isNaNAOrB       (_mulAddRecFNToRaw_preMul_io_toPostMul_isNaNAOrB),
    .io_toPostMul_isInfA          (_mulAddRecFNToRaw_preMul_io_toPostMul_isInfA),
    .io_toPostMul_isZeroA         (_mulAddRecFNToRaw_preMul_io_toPostMul_isZeroA),
    .io_toPostMul_isInfB          (_mulAddRecFNToRaw_preMul_io_toPostMul_isInfB),
    .io_toPostMul_isZeroB         (_mulAddRecFNToRaw_preMul_io_toPostMul_isZeroB),
    .io_toPostMul_signProd        (_mulAddRecFNToRaw_preMul_io_toPostMul_signProd),
    .io_toPostMul_isNaNC          (_mulAddRecFNToRaw_preMul_io_toPostMul_isNaNC),
    .io_toPostMul_isInfC          (_mulAddRecFNToRaw_preMul_io_toPostMul_isInfC),
    .io_toPostMul_isZeroC         (_mulAddRecFNToRaw_preMul_io_toPostMul_isZeroC),
    .io_toPostMul_sExpSum         (_mulAddRecFNToRaw_preMul_io_toPostMul_sExpSum),
    .io_toPostMul_doSubMags       (_mulAddRecFNToRaw_preMul_io_toPostMul_doSubMags),
    .io_toPostMul_CIsDominant     (_mulAddRecFNToRaw_preMul_io_toPostMul_CIsDominant),
    .io_toPostMul_CDom_CAlignDist (_mulAddRecFNToRaw_preMul_io_toPostMul_CDom_CAlignDist),
    .io_toPostMul_highAlignedSigC (_mulAddRecFNToRaw_preMul_io_toPostMul_highAlignedSigC),
    .io_toPostMul_bit0AlignedSigC (_mulAddRecFNToRaw_preMul_io_toPostMul_bit0AlignedSigC)
  );
  MulAddRecFNToRaw_postMul_e8_s24 mulAddRecFNToRaw_postMul (
    .io_fromPreMul_isSigNaNAny     (_mulAddRecFNToRaw_preMul_io_toPostMul_isSigNaNAny),
    .io_fromPreMul_isNaNAOrB       (_mulAddRecFNToRaw_preMul_io_toPostMul_isNaNAOrB),
    .io_fromPreMul_isInfA          (_mulAddRecFNToRaw_preMul_io_toPostMul_isInfA),
    .io_fromPreMul_isZeroA         (_mulAddRecFNToRaw_preMul_io_toPostMul_isZeroA),
    .io_fromPreMul_isInfB          (_mulAddRecFNToRaw_preMul_io_toPostMul_isInfB),
    .io_fromPreMul_isZeroB         (_mulAddRecFNToRaw_preMul_io_toPostMul_isZeroB),
    .io_fromPreMul_signProd        (_mulAddRecFNToRaw_preMul_io_toPostMul_signProd),
    .io_fromPreMul_isNaNC          (_mulAddRecFNToRaw_preMul_io_toPostMul_isNaNC),
    .io_fromPreMul_isInfC          (_mulAddRecFNToRaw_preMul_io_toPostMul_isInfC),
    .io_fromPreMul_isZeroC         (_mulAddRecFNToRaw_preMul_io_toPostMul_isZeroC),
    .io_fromPreMul_sExpSum         (_mulAddRecFNToRaw_preMul_io_toPostMul_sExpSum),
    .io_fromPreMul_doSubMags       (_mulAddRecFNToRaw_preMul_io_toPostMul_doSubMags),
    .io_fromPreMul_CIsDominant     (_mulAddRecFNToRaw_preMul_io_toPostMul_CIsDominant),
    .io_fromPreMul_CDom_CAlignDist (_mulAddRecFNToRaw_preMul_io_toPostMul_CDom_CAlignDist),
    .io_fromPreMul_highAlignedSigC (_mulAddRecFNToRaw_preMul_io_toPostMul_highAlignedSigC),
    .io_fromPreMul_bit0AlignedSigC (_mulAddRecFNToRaw_preMul_io_toPostMul_bit0AlignedSigC),
    .io_mulAddResult               (mulAddResult),
    .io_roundingMode               (io_roundingMode_0),
    .io_invalidExc                 (_mulAddRecFNToRaw_postMul_io_invalidExc),
    .io_rawOut_isNaN               (_mulAddRecFNToRaw_postMul_io_rawOut_isNaN),
    .io_rawOut_isInf               (_mulAddRecFNToRaw_postMul_io_rawOut_isInf),
    .io_rawOut_isZero              (_mulAddRecFNToRaw_postMul_io_rawOut_isZero),
    .io_rawOut_sign                (_mulAddRecFNToRaw_postMul_io_rawOut_sign),
    .io_rawOut_sExp                (_mulAddRecFNToRaw_postMul_io_rawOut_sExp),
    .io_rawOut_sig                 (_mulAddRecFNToRaw_postMul_io_rawOut_sig)
  );
  RoundRawFNToRecFN_e8_s24 roundRawFNToRecFN (
    .io_invalidExc     (_mulAddRecFNToRaw_postMul_io_invalidExc),
    .io_in_isNaN       (_mulAddRecFNToRaw_postMul_io_rawOut_isNaN),
    .io_in_isInf       (_mulAddRecFNToRaw_postMul_io_rawOut_isInf),
    .io_in_isZero      (_mulAddRecFNToRaw_postMul_io_rawOut_isZero),
    .io_in_sign        (_mulAddRecFNToRaw_postMul_io_rawOut_sign),
    .io_in_sExp        (_mulAddRecFNToRaw_postMul_io_rawOut_sExp),
    .io_in_sig         (_mulAddRecFNToRaw_postMul_io_rawOut_sig),
    .io_roundingMode   (io_roundingMode_0),
    .io_out            (io_out_0),
    .io_exceptionFlags (io_exceptionFlags_0)
  );
  assign io_out = io_out_0;
  assign io_exceptionFlags = io_exceptionFlags_0;
endmodule

