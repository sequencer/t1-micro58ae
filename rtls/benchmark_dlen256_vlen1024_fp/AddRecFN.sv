module AddRecFN(
  input  [32:0] io_a,
                io_b,
  input  [2:0]  io_roundingMode,
  output [32:0] io_out,
  output [4:0]  io_exceptionFlags
);

  wire        _addRawFN_io_invalidExc;
  wire        _addRawFN_io_rawOut_isNaN;
  wire        _addRawFN_io_rawOut_isInf;
  wire        _addRawFN_io_rawOut_isZero;
  wire        _addRawFN_io_rawOut_sign;
  wire [9:0]  _addRawFN_io_rawOut_sExp;
  wire [26:0] _addRawFN_io_rawOut_sig;
  wire [4:0]  io_exceptionFlags_0;
  wire [32:0] io_out_0;
  wire [32:0] io_a_0 = io_a;
  wire [32:0] io_b_0 = io_b;
  wire [2:0]  io_roundingMode_0 = io_roundingMode;
  wire        io_subOp = 1'h0;
  wire        io_detectTininess = 1'h0;
  wire [8:0]  addRawFN_io_a_exp = io_a_0[31:23];
  wire        addRawFN_io_a_isZero = addRawFN_io_a_exp[8:6] == 3'h0;
  wire        addRawFN_io_a_out_isZero = addRawFN_io_a_isZero;
  wire        addRawFN_io_a_isSpecial = &(addRawFN_io_a_exp[8:7]);
  wire        addRawFN_io_a_out_isNaN = addRawFN_io_a_isSpecial & addRawFN_io_a_exp[6];
  wire        addRawFN_io_a_out_isInf = addRawFN_io_a_isSpecial & ~(addRawFN_io_a_exp[6]);
  wire        addRawFN_io_a_out_sign = io_a_0[32];
  wire [9:0]  addRawFN_io_a_out_sExp = {1'h0, addRawFN_io_a_exp};
  wire [24:0] addRawFN_io_a_out_sig = {1'h0, ~addRawFN_io_a_isZero, io_a_0[22:0]};
  wire [8:0]  addRawFN_io_b_exp = io_b_0[31:23];
  wire        addRawFN_io_b_isZero = addRawFN_io_b_exp[8:6] == 3'h0;
  wire        addRawFN_io_b_out_isZero = addRawFN_io_b_isZero;
  wire        addRawFN_io_b_isSpecial = &(addRawFN_io_b_exp[8:7]);
  wire        addRawFN_io_b_out_isNaN = addRawFN_io_b_isSpecial & addRawFN_io_b_exp[6];
  wire        addRawFN_io_b_out_isInf = addRawFN_io_b_isSpecial & ~(addRawFN_io_b_exp[6]);
  wire        addRawFN_io_b_out_sign = io_b_0[32];
  wire [9:0]  addRawFN_io_b_out_sExp = {1'h0, addRawFN_io_b_exp};
  wire [24:0] addRawFN_io_b_out_sig = {1'h0, ~addRawFN_io_b_isZero, io_b_0[22:0]};
  AddRawFN addRawFN (
    .io_a_isNaN       (addRawFN_io_a_out_isNaN),
    .io_a_isInf       (addRawFN_io_a_out_isInf),
    .io_a_isZero      (addRawFN_io_a_out_isZero),
    .io_a_sign        (addRawFN_io_a_out_sign),
    .io_a_sExp        (addRawFN_io_a_out_sExp),
    .io_a_sig         (addRawFN_io_a_out_sig),
    .io_b_isNaN       (addRawFN_io_b_out_isNaN),
    .io_b_isInf       (addRawFN_io_b_out_isInf),
    .io_b_isZero      (addRawFN_io_b_out_isZero),
    .io_b_sign        (addRawFN_io_b_out_sign),
    .io_b_sExp        (addRawFN_io_b_out_sExp),
    .io_b_sig         (addRawFN_io_b_out_sig),
    .io_roundingMode  (io_roundingMode_0),
    .io_invalidExc    (_addRawFN_io_invalidExc),
    .io_rawOut_isNaN  (_addRawFN_io_rawOut_isNaN),
    .io_rawOut_isInf  (_addRawFN_io_rawOut_isInf),
    .io_rawOut_isZero (_addRawFN_io_rawOut_isZero),
    .io_rawOut_sign   (_addRawFN_io_rawOut_sign),
    .io_rawOut_sExp   (_addRawFN_io_rawOut_sExp),
    .io_rawOut_sig    (_addRawFN_io_rawOut_sig)
  );
  RoundRawFNToRecFN_e8_s24 roundRawFNToRecFN (
    .io_invalidExc     (_addRawFN_io_invalidExc),
    .io_in_isNaN       (_addRawFN_io_rawOut_isNaN),
    .io_in_isInf       (_addRawFN_io_rawOut_isInf),
    .io_in_isZero      (_addRawFN_io_rawOut_isZero),
    .io_in_sign        (_addRawFN_io_rawOut_sign),
    .io_in_sExp        (_addRawFN_io_rawOut_sExp),
    .io_in_sig         (_addRawFN_io_rawOut_sig),
    .io_roundingMode   (io_roundingMode_0),
    .io_out            (io_out_0),
    .io_exceptionFlags (io_exceptionFlags_0)
  );
  assign io_out = io_out_0;
  assign io_exceptionFlags = io_exceptionFlags_0;
endmodule

