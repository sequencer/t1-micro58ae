
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
module FloatAdder(
  input         clock,
                reset,
  input  [31:0] a,
                b,
  input  [2:0]  roundingMode,
  output [31:0] out
);

  wire [32:0] _addRecFN_io_out;
  wire [4:0]  _addRecFN_io_exceptionFlags;
  wire        addRecFN_io_a_rawIn_sign = a[31];
  wire        addRecFN_io_a_rawIn_sign_0 = addRecFN_io_a_rawIn_sign;
  wire [7:0]  addRecFN_io_a_rawIn_expIn = a[30:23];
  wire [22:0] addRecFN_io_a_rawIn_fractIn = a[22:0];
  wire        addRecFN_io_a_rawIn_isZeroExpIn = addRecFN_io_a_rawIn_expIn == 8'h0;
  wire        addRecFN_io_a_rawIn_isZeroFractIn = addRecFN_io_a_rawIn_fractIn == 23'h0;
  wire [4:0]  addRecFN_io_a_rawIn_normDist =
    addRecFN_io_a_rawIn_fractIn[22]
      ? 5'h0
      : addRecFN_io_a_rawIn_fractIn[21]
          ? 5'h1
          : addRecFN_io_a_rawIn_fractIn[20]
              ? 5'h2
              : addRecFN_io_a_rawIn_fractIn[19]
                  ? 5'h3
                  : addRecFN_io_a_rawIn_fractIn[18]
                      ? 5'h4
                      : addRecFN_io_a_rawIn_fractIn[17]
                          ? 5'h5
                          : addRecFN_io_a_rawIn_fractIn[16]
                              ? 5'h6
                              : addRecFN_io_a_rawIn_fractIn[15]
                                  ? 5'h7
                                  : addRecFN_io_a_rawIn_fractIn[14]
                                      ? 5'h8
                                      : addRecFN_io_a_rawIn_fractIn[13]
                                          ? 5'h9
                                          : addRecFN_io_a_rawIn_fractIn[12]
                                              ? 5'hA
                                              : addRecFN_io_a_rawIn_fractIn[11]
                                                  ? 5'hB
                                                  : addRecFN_io_a_rawIn_fractIn[10]
                                                      ? 5'hC
                                                      : addRecFN_io_a_rawIn_fractIn[9]
                                                          ? 5'hD
                                                          : addRecFN_io_a_rawIn_fractIn[8]
                                                              ? 5'hE
                                                              : addRecFN_io_a_rawIn_fractIn[7]
                                                                  ? 5'hF
                                                                  : addRecFN_io_a_rawIn_fractIn[6]
                                                                      ? 5'h10
                                                                      : addRecFN_io_a_rawIn_fractIn[5]
                                                                          ? 5'h11
                                                                          : addRecFN_io_a_rawIn_fractIn[4]
                                                                              ? 5'h12
                                                                              : addRecFN_io_a_rawIn_fractIn[3] ? 5'h13 : addRecFN_io_a_rawIn_fractIn[2] ? 5'h14 : addRecFN_io_a_rawIn_fractIn[1] ? 5'h15 : 5'h16;
  wire [53:0] _addRecFN_io_a_rawIn_subnormFract_T = {31'h0, addRecFN_io_a_rawIn_fractIn} << addRecFN_io_a_rawIn_normDist;
  wire [22:0] addRecFN_io_a_rawIn_subnormFract = {_addRecFN_io_a_rawIn_subnormFract_T[21:0], 1'h0};
  wire [8:0]  addRecFN_io_a_rawIn_adjustedExp = (addRecFN_io_a_rawIn_isZeroExpIn ? {4'hF, ~addRecFN_io_a_rawIn_normDist} : {1'h0, addRecFN_io_a_rawIn_expIn}) + {7'h20, addRecFN_io_a_rawIn_isZeroExpIn ? 2'h2 : 2'h1};
  wire        addRecFN_io_a_rawIn_isZero = addRecFN_io_a_rawIn_isZeroExpIn & addRecFN_io_a_rawIn_isZeroFractIn;
  wire        addRecFN_io_a_rawIn_isZero_0 = addRecFN_io_a_rawIn_isZero;
  wire        addRecFN_io_a_rawIn_isSpecial = &(addRecFN_io_a_rawIn_adjustedExp[8:7]);
  wire        addRecFN_io_a_rawIn_isNaN = addRecFN_io_a_rawIn_isSpecial & ~addRecFN_io_a_rawIn_isZeroFractIn;
  wire        addRecFN_io_a_rawIn_isInf = addRecFN_io_a_rawIn_isSpecial & addRecFN_io_a_rawIn_isZeroFractIn;
  wire [9:0]  addRecFN_io_a_rawIn_sExp = {1'h0, addRecFN_io_a_rawIn_adjustedExp};
  wire [24:0] addRecFN_io_a_rawIn_sig = {1'h0, ~addRecFN_io_a_rawIn_isZero, addRecFN_io_a_rawIn_isZeroExpIn ? addRecFN_io_a_rawIn_subnormFract : addRecFN_io_a_rawIn_fractIn};
  wire [2:0]  _addRecFN_io_a_T_1 = addRecFN_io_a_rawIn_isZero_0 ? 3'h0 : addRecFN_io_a_rawIn_sExp[8:6];
  wire        addRecFN_io_b_rawIn_sign = b[31];
  wire        addRecFN_io_b_rawIn_sign_0 = addRecFN_io_b_rawIn_sign;
  wire [7:0]  addRecFN_io_b_rawIn_expIn = b[30:23];
  wire [22:0] addRecFN_io_b_rawIn_fractIn = b[22:0];
  wire        addRecFN_io_b_rawIn_isZeroExpIn = addRecFN_io_b_rawIn_expIn == 8'h0;
  wire        addRecFN_io_b_rawIn_isZeroFractIn = addRecFN_io_b_rawIn_fractIn == 23'h0;
  wire [4:0]  addRecFN_io_b_rawIn_normDist =
    addRecFN_io_b_rawIn_fractIn[22]
      ? 5'h0
      : addRecFN_io_b_rawIn_fractIn[21]
          ? 5'h1
          : addRecFN_io_b_rawIn_fractIn[20]
              ? 5'h2
              : addRecFN_io_b_rawIn_fractIn[19]
                  ? 5'h3
                  : addRecFN_io_b_rawIn_fractIn[18]
                      ? 5'h4
                      : addRecFN_io_b_rawIn_fractIn[17]
                          ? 5'h5
                          : addRecFN_io_b_rawIn_fractIn[16]
                              ? 5'h6
                              : addRecFN_io_b_rawIn_fractIn[15]
                                  ? 5'h7
                                  : addRecFN_io_b_rawIn_fractIn[14]
                                      ? 5'h8
                                      : addRecFN_io_b_rawIn_fractIn[13]
                                          ? 5'h9
                                          : addRecFN_io_b_rawIn_fractIn[12]
                                              ? 5'hA
                                              : addRecFN_io_b_rawIn_fractIn[11]
                                                  ? 5'hB
                                                  : addRecFN_io_b_rawIn_fractIn[10]
                                                      ? 5'hC
                                                      : addRecFN_io_b_rawIn_fractIn[9]
                                                          ? 5'hD
                                                          : addRecFN_io_b_rawIn_fractIn[8]
                                                              ? 5'hE
                                                              : addRecFN_io_b_rawIn_fractIn[7]
                                                                  ? 5'hF
                                                                  : addRecFN_io_b_rawIn_fractIn[6]
                                                                      ? 5'h10
                                                                      : addRecFN_io_b_rawIn_fractIn[5]
                                                                          ? 5'h11
                                                                          : addRecFN_io_b_rawIn_fractIn[4]
                                                                              ? 5'h12
                                                                              : addRecFN_io_b_rawIn_fractIn[3] ? 5'h13 : addRecFN_io_b_rawIn_fractIn[2] ? 5'h14 : addRecFN_io_b_rawIn_fractIn[1] ? 5'h15 : 5'h16;
  wire [53:0] _addRecFN_io_b_rawIn_subnormFract_T = {31'h0, addRecFN_io_b_rawIn_fractIn} << addRecFN_io_b_rawIn_normDist;
  wire [22:0] addRecFN_io_b_rawIn_subnormFract = {_addRecFN_io_b_rawIn_subnormFract_T[21:0], 1'h0};
  wire [8:0]  addRecFN_io_b_rawIn_adjustedExp = (addRecFN_io_b_rawIn_isZeroExpIn ? {4'hF, ~addRecFN_io_b_rawIn_normDist} : {1'h0, addRecFN_io_b_rawIn_expIn}) + {7'h20, addRecFN_io_b_rawIn_isZeroExpIn ? 2'h2 : 2'h1};
  wire        addRecFN_io_b_rawIn_isZero = addRecFN_io_b_rawIn_isZeroExpIn & addRecFN_io_b_rawIn_isZeroFractIn;
  wire        addRecFN_io_b_rawIn_isZero_0 = addRecFN_io_b_rawIn_isZero;
  wire        addRecFN_io_b_rawIn_isSpecial = &(addRecFN_io_b_rawIn_adjustedExp[8:7]);
  wire        addRecFN_io_b_rawIn_isNaN = addRecFN_io_b_rawIn_isSpecial & ~addRecFN_io_b_rawIn_isZeroFractIn;
  wire        addRecFN_io_b_rawIn_isInf = addRecFN_io_b_rawIn_isSpecial & addRecFN_io_b_rawIn_isZeroFractIn;
  wire [9:0]  addRecFN_io_b_rawIn_sExp = {1'h0, addRecFN_io_b_rawIn_adjustedExp};
  wire [24:0] addRecFN_io_b_rawIn_sig = {1'h0, ~addRecFN_io_b_rawIn_isZero, addRecFN_io_b_rawIn_isZeroExpIn ? addRecFN_io_b_rawIn_subnormFract : addRecFN_io_b_rawIn_fractIn};
  wire [2:0]  _addRecFN_io_b_T_1 = addRecFN_io_b_rawIn_isZero_0 ? 3'h0 : addRecFN_io_b_rawIn_sExp[8:6];
  wire [8:0]  view__out_rawIn_exp = _addRecFN_io_out[31:23];
  wire        view__out_rawIn_isZero = view__out_rawIn_exp[8:6] == 3'h0;
  wire        view__out_rawIn_isZero_0 = view__out_rawIn_isZero;
  wire        view__out_rawIn_isSpecial = &(view__out_rawIn_exp[8:7]);
  wire        view__out_rawIn_isNaN = view__out_rawIn_isSpecial & view__out_rawIn_exp[6];
  wire        view__out_rawIn_isInf = view__out_rawIn_isSpecial & ~(view__out_rawIn_exp[6]);
  wire        view__out_rawIn_sign = _addRecFN_io_out[32];
  wire [9:0]  view__out_rawIn_sExp = {1'h0, view__out_rawIn_exp};
  wire [24:0] view__out_rawIn_sig = {1'h0, ~view__out_rawIn_isZero, _addRecFN_io_out[22:0]};
  wire        view__out_isSubnormal = $signed(view__out_rawIn_sExp) < 10'sh82;
  wire [4:0]  view__out_denormShiftDist = 5'h1 - view__out_rawIn_sExp[4:0];
  wire [23:0] _view__out_denormFract_T_1 = view__out_rawIn_sig[24:1] >> view__out_denormShiftDist;
  wire [22:0] view__out_denormFract = _view__out_denormFract_T_1[22:0];
  wire [7:0]  view__out_expOut = (view__out_isSubnormal ? 8'h0 : view__out_rawIn_sExp[7:0] + 8'h7F) | {8{view__out_rawIn_isNaN | view__out_rawIn_isInf}};
  wire [22:0] view__out_fractOut = view__out_isSubnormal ? view__out_denormFract : view__out_rawIn_isInf ? 23'h0 : view__out_rawIn_sig[22:0];
  wire [8:0]  view__out_hi = {view__out_rawIn_sign, view__out_expOut};
  reg         view__out_pipe_v;
  wire        view__out_pipe_out_valid = view__out_pipe_v;
  reg  [31:0] view__out_pipe_b;
  wire [31:0] view__out_pipe_out_bits = view__out_pipe_b;
  reg         view__exceptionFlags_pipe_v;
  wire        view__exceptionFlags_pipe_out_valid = view__exceptionFlags_pipe_v;
  reg  [4:0]  view__exceptionFlags_pipe_b;
  wire [4:0]  view__exceptionFlags_pipe_out_bits = view__exceptionFlags_pipe_b;
  always @(posedge clock) begin
    if (reset) begin
      view__out_pipe_v <= 1'h0;
      view__exceptionFlags_pipe_v <= 1'h0;
    end
    else begin
      view__out_pipe_v <= 1'h1;
      view__exceptionFlags_pipe_v <= 1'h1;
    end
    view__out_pipe_b <= {view__out_hi, view__out_fractOut};
    view__exceptionFlags_pipe_b <= _addRecFN_io_exceptionFlags;
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:1];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [1:0] i = 2'h0; i < 2'h2; i += 2'h1) begin
          _RANDOM[i[0]] = `RANDOM;
        end
        view__out_pipe_v = _RANDOM[1'h0][0];
        view__out_pipe_b = {_RANDOM[1'h0][31:1], _RANDOM[1'h1][0]};
        view__exceptionFlags_pipe_v = _RANDOM[1'h1][1];
        view__exceptionFlags_pipe_b = _RANDOM[1'h1][6:2];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  AddRecFN addRecFN (
    .io_a              ({addRecFN_io_a_rawIn_sign_0, _addRecFN_io_a_T_1[2:1], _addRecFN_io_a_T_1[0] | addRecFN_io_a_rawIn_isNaN, addRecFN_io_a_rawIn_sExp[5:0], addRecFN_io_a_rawIn_sig[22:0]}),
    .io_b              ({addRecFN_io_b_rawIn_sign_0, _addRecFN_io_b_T_1[2:1], _addRecFN_io_b_T_1[0] | addRecFN_io_b_rawIn_isNaN, addRecFN_io_b_rawIn_sExp[5:0], addRecFN_io_b_rawIn_sig[22:0]}),
    .io_roundingMode   (roundingMode),
    .io_out            (_addRecFN_io_out),
    .io_exceptionFlags (_addRecFN_io_exceptionFlags)
  );
  assign out = view__out_pipe_out_bits;
endmodule

