module RoundRawFNToRecFN_e8_s24(
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

  wire [4:0]  io_exceptionFlags_0;
  wire [32:0] io_out_0;
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
  RoundAnyRawFNToRecFN_ie8_is26_oe8_os24 roundAnyRawFNToRecFN (
    .io_invalidExc     (io_invalidExc_0),
    .io_in_isNaN       (io_in_isNaN_0),
    .io_in_isInf       (io_in_isInf_0),
    .io_in_isZero      (io_in_isZero_0),
    .io_in_sign        (io_in_sign_0),
    .io_in_sExp        (io_in_sExp_0),
    .io_in_sig         (io_in_sig_0),
    .io_roundingMode   (io_roundingMode_0),
    .io_out            (io_out_0),
    .io_exceptionFlags (io_exceptionFlags_0)
  );
  assign io_out = io_out_0;
  assign io_exceptionFlags = io_exceptionFlags_0;
endmodule

