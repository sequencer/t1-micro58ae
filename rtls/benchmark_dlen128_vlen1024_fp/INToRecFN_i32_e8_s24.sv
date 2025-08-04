module INToRecFN_i32_e8_s24(
  input         io_signedIn,
  input  [31:0] io_in,
  input  [2:0]  io_roundingMode,
  output [32:0] io_out,
  output [4:0]  io_exceptionFlags
);

  wire [4:0]  io_exceptionFlags_0;
  wire [32:0] io_out_0;
  wire        io_signedIn_0 = io_signedIn;
  wire [31:0] io_in_0 = io_in;
  wire [2:0]  io_roundingMode_0 = io_roundingMode;
  wire        io_detectTininess = 1'h0;
  wire        intAsRawFloat_isNaN = 1'h0;
  wire        intAsRawFloat_isInf = 1'h0;
  wire        intAsRawFloat_sign = io_signedIn_0 & io_in_0[31];
  wire        intAsRawFloat_sign_0 = intAsRawFloat_sign;
  wire [31:0] intAsRawFloat_absIn = intAsRawFloat_sign ? 32'h0 - io_in_0 : io_in_0;
  wire [31:0] intAsRawFloat_extAbsIn = intAsRawFloat_absIn;
  wire [4:0]  intAsRawFloat_adjustedNormDist =
    intAsRawFloat_extAbsIn[31]
      ? 5'h0
      : intAsRawFloat_extAbsIn[30]
          ? 5'h1
          : intAsRawFloat_extAbsIn[29]
              ? 5'h2
              : intAsRawFloat_extAbsIn[28]
                  ? 5'h3
                  : intAsRawFloat_extAbsIn[27]
                      ? 5'h4
                      : intAsRawFloat_extAbsIn[26]
                          ? 5'h5
                          : intAsRawFloat_extAbsIn[25]
                              ? 5'h6
                              : intAsRawFloat_extAbsIn[24]
                                  ? 5'h7
                                  : intAsRawFloat_extAbsIn[23]
                                      ? 5'h8
                                      : intAsRawFloat_extAbsIn[22]
                                          ? 5'h9
                                          : intAsRawFloat_extAbsIn[21]
                                              ? 5'hA
                                              : intAsRawFloat_extAbsIn[20]
                                                  ? 5'hB
                                                  : intAsRawFloat_extAbsIn[19]
                                                      ? 5'hC
                                                      : intAsRawFloat_extAbsIn[18]
                                                          ? 5'hD
                                                          : intAsRawFloat_extAbsIn[17]
                                                              ? 5'hE
                                                              : intAsRawFloat_extAbsIn[16]
                                                                  ? 5'hF
                                                                  : intAsRawFloat_extAbsIn[15]
                                                                      ? 5'h10
                                                                      : intAsRawFloat_extAbsIn[14]
                                                                          ? 5'h11
                                                                          : intAsRawFloat_extAbsIn[13]
                                                                              ? 5'h12
                                                                              : intAsRawFloat_extAbsIn[12]
                                                                                  ? 5'h13
                                                                                  : intAsRawFloat_extAbsIn[11]
                                                                                      ? 5'h14
                                                                                      : intAsRawFloat_extAbsIn[10]
                                                                                          ? 5'h15
                                                                                          : intAsRawFloat_extAbsIn[9]
                                                                                              ? 5'h16
                                                                                              : intAsRawFloat_extAbsIn[8]
                                                                                                  ? 5'h17
                                                                                                  : intAsRawFloat_extAbsIn[7]
                                                                                                      ? 5'h18
                                                                                                      : intAsRawFloat_extAbsIn[6]
                                                                                                          ? 5'h19
                                                                                                          : intAsRawFloat_extAbsIn[5]
                                                                                                              ? 5'h1A
                                                                                                              : intAsRawFloat_extAbsIn[4]
                                                                                                                  ? 5'h1B
                                                                                                                  : intAsRawFloat_extAbsIn[3] ? 5'h1C : intAsRawFloat_extAbsIn[2] ? 5'h1D : {4'hF, ~(intAsRawFloat_extAbsIn[1])};
  wire [62:0] _intAsRawFloat_sig_T = {31'h0, intAsRawFloat_extAbsIn} << intAsRawFloat_adjustedNormDist;
  wire [31:0] intAsRawFloat_sig = _intAsRawFloat_sig_T[31:0];
  wire        intAsRawFloat_isZero = ~(intAsRawFloat_sig[31]);
  wire [7:0]  intAsRawFloat_sExp = {3'h2, ~intAsRawFloat_adjustedNormDist};
  wire [32:0] intAsRawFloat_sig_0 = {1'h0, intAsRawFloat_sig};
  RoundAnyRawFNToRecFN_ie6_is32_oe8_os24 roundAnyRawFNToRecFN (
    .io_in_isZero      (intAsRawFloat_isZero),
    .io_in_sign        (intAsRawFloat_sign_0),
    .io_in_sExp        (intAsRawFloat_sExp),
    .io_in_sig         (intAsRawFloat_sig_0),
    .io_roundingMode   (io_roundingMode_0),
    .io_out            (io_out_0),
    .io_exceptionFlags (io_exceptionFlags_0)
  );
  assign io_out = io_out_0;
  assign io_exceptionFlags = io_exceptionFlags_0;
endmodule

