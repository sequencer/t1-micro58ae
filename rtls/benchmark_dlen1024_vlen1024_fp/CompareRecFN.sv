module CompareRecFN(
  input  [32:0] io_a,
                io_b,
  input         io_signaling,
  output        io_lt,
                io_eq,
                io_gt,
  output [4:0]  io_exceptionFlags
);

  wire [32:0] io_a_0 = io_a;
  wire [32:0] io_b_0 = io_b;
  wire        io_signaling_0 = io_signaling;
  wire [8:0]  rawA_exp = io_a_0[31:23];
  wire        rawA_isZero = rawA_exp[8:6] == 3'h0;
  wire        rawA_isZero_0 = rawA_isZero;
  wire        rawA_isSpecial = &(rawA_exp[8:7]);
  wire        rawA_isNaN = rawA_isSpecial & rawA_exp[6];
  wire        rawA_isInf = rawA_isSpecial & ~(rawA_exp[6]);
  wire        rawA_sign = io_a_0[32];
  wire [9:0]  rawA_sExp = {1'h0, rawA_exp};
  wire [24:0] rawA_sig = {1'h0, ~rawA_isZero, io_a_0[22:0]};
  wire [8:0]  rawB_exp = io_b_0[31:23];
  wire        rawB_isZero = rawB_exp[8:6] == 3'h0;
  wire        rawB_isZero_0 = rawB_isZero;
  wire        rawB_isSpecial = &(rawB_exp[8:7]);
  wire        rawB_isNaN = rawB_isSpecial & rawB_exp[6];
  wire        rawB_isInf = rawB_isSpecial & ~(rawB_exp[6]);
  wire        rawB_sign = io_b_0[32];
  wire [9:0]  rawB_sExp = {1'h0, rawB_exp};
  wire [24:0] rawB_sig = {1'h0, ~rawB_isZero, io_b_0[22:0]};
  wire        ordered = ~rawA_isNaN & ~rawB_isNaN;
  wire        bothInfs = rawA_isInf & rawB_isInf;
  wire        bothZeros = rawA_isZero_0 & rawB_isZero_0;
  wire        eqExps = rawA_sExp == rawB_sExp;
  wire        common_ltMags = $signed(rawA_sExp) < $signed(rawB_sExp) | eqExps & rawA_sig < rawB_sig;
  wire        common_eqMags = eqExps & rawA_sig == rawB_sig;
  wire        ordered_lt = ~bothZeros & (rawA_sign & ~rawB_sign | ~bothInfs & (rawA_sign & ~common_ltMags & ~common_eqMags | ~rawB_sign & common_ltMags));
  wire        ordered_eq = bothZeros | rawA_sign == rawB_sign & (bothInfs | common_eqMags);
  wire        invalid = rawA_isNaN & ~(rawA_sig[22]) | rawB_isNaN & ~(rawB_sig[22]) | io_signaling_0 & ~ordered;
  wire        io_lt_0 = ordered & ordered_lt;
  wire        io_eq_0 = ordered & ordered_eq;
  wire        io_gt_0 = ordered & ~ordered_lt & ~ordered_eq;
  wire [4:0]  io_exceptionFlags_0 = {invalid, 4'h0};
  assign io_lt = io_lt_0;
  assign io_eq = io_eq_0;
  assign io_gt = io_gt_0;
  assign io_exceptionFlags = io_exceptionFlags_0;
endmodule

