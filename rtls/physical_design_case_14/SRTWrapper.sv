
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
module SRTWrapper(
  input         clock,
                reset,
  output        input_ready,
  input         input_valid,
  input  [31:0] input_bits_dividend,
                input_bits_divisor,
  input         input_bits_signIn,
  output        output_valid,
  output [31:0] output_bits_reminder,
                output_bits_quotient
);

  wire        _srt_input_ready;
  wire        _srt_output_valid;
  wire [31:0] _srt_output_bits_reminder;
  wire [31:0] _abs_io_aOut;
  wire [31:0] _abs_io_bOut;
  wire        _abs_io_aSign;
  wire        _abs_io_bSign;
  wire        input_valid_0 = input_valid;
  wire [31:0] input_bits_dividend_0 = input_bits_dividend;
  wire [31:0] input_bits_divisor_0 = input_bits_divisor;
  wire        input_bits_signIn_0 = input_bits_signIn;
  wire        negative = _abs_io_aSign ^ _abs_io_bSign;
  wire        divideZero = input_bits_divisor_0 == 32'h0;
  wire [32:0] dividend = {1'h0, _abs_io_aOut};
  wire [32:0] divisor = {1'h0, _abs_io_bOut};
  wire [33:0] gap = {1'h0, divisor} + {1'h0, 33'h0 - dividend};
  wire        biggerdivisor = gap[33] & (|(gap[32:0]));
  wire        input_ready_0;
  wire        _dividendInputReg_T_1 = input_ready_0 & input_valid_0;
  wire        bypassSRT = (divideZero | biggerdivisor) & _dividendInputReg_T_1;
  wire [5:0]  zeroHeadDividend =
    {1'h0,
     _abs_io_aOut[31]
       ? 5'h0
       : _abs_io_aOut[30]
           ? 5'h1
           : _abs_io_aOut[29]
               ? 5'h2
               : _abs_io_aOut[28]
                   ? 5'h3
                   : _abs_io_aOut[27]
                       ? 5'h4
                       : _abs_io_aOut[26]
                           ? 5'h5
                           : _abs_io_aOut[25]
                               ? 5'h6
                               : _abs_io_aOut[24]
                                   ? 5'h7
                                   : _abs_io_aOut[23]
                                       ? 5'h8
                                       : _abs_io_aOut[22]
                                           ? 5'h9
                                           : _abs_io_aOut[21]
                                               ? 5'hA
                                               : _abs_io_aOut[20]
                                                   ? 5'hB
                                                   : _abs_io_aOut[19]
                                                       ? 5'hC
                                                       : _abs_io_aOut[18]
                                                           ? 5'hD
                                                           : _abs_io_aOut[17]
                                                               ? 5'hE
                                                               : _abs_io_aOut[16]
                                                                   ? 5'hF
                                                                   : _abs_io_aOut[15]
                                                                       ? 5'h10
                                                                       : _abs_io_aOut[14]
                                                                           ? 5'h11
                                                                           : _abs_io_aOut[13]
                                                                               ? 5'h12
                                                                               : _abs_io_aOut[12]
                                                                                   ? 5'h13
                                                                                   : _abs_io_aOut[11]
                                                                                       ? 5'h14
                                                                                       : _abs_io_aOut[10]
                                                                                           ? 5'h15
                                                                                           : _abs_io_aOut[9]
                                                                                               ? 5'h16
                                                                                               : _abs_io_aOut[8]
                                                                                                   ? 5'h17
                                                                                                   : _abs_io_aOut[7]
                                                                                                       ? 5'h18
                                                                                                       : _abs_io_aOut[6]
                                                                                                           ? 5'h19
                                                                                                           : _abs_io_aOut[5]
                                                                                                               ? 5'h1A
                                                                                                               : _abs_io_aOut[4] ? 5'h1B : _abs_io_aOut[3] ? 5'h1C : _abs_io_aOut[2] ? 5'h1D : {4'hF, ~(_abs_io_aOut[1])}};
  wire [5:0]  zeroHeadDivisor =
    {1'h0,
     _abs_io_bOut[31]
       ? 5'h0
       : _abs_io_bOut[30]
           ? 5'h1
           : _abs_io_bOut[29]
               ? 5'h2
               : _abs_io_bOut[28]
                   ? 5'h3
                   : _abs_io_bOut[27]
                       ? 5'h4
                       : _abs_io_bOut[26]
                           ? 5'h5
                           : _abs_io_bOut[25]
                               ? 5'h6
                               : _abs_io_bOut[24]
                                   ? 5'h7
                                   : _abs_io_bOut[23]
                                       ? 5'h8
                                       : _abs_io_bOut[22]
                                           ? 5'h9
                                           : _abs_io_bOut[21]
                                               ? 5'hA
                                               : _abs_io_bOut[20]
                                                   ? 5'hB
                                                   : _abs_io_bOut[19]
                                                       ? 5'hC
                                                       : _abs_io_bOut[18]
                                                           ? 5'hD
                                                           : _abs_io_bOut[17]
                                                               ? 5'hE
                                                               : _abs_io_bOut[16]
                                                                   ? 5'hF
                                                                   : _abs_io_bOut[15]
                                                                       ? 5'h10
                                                                       : _abs_io_bOut[14]
                                                                           ? 5'h11
                                                                           : _abs_io_bOut[13]
                                                                               ? 5'h12
                                                                               : _abs_io_bOut[12]
                                                                                   ? 5'h13
                                                                                   : _abs_io_bOut[11]
                                                                                       ? 5'h14
                                                                                       : _abs_io_bOut[10]
                                                                                           ? 5'h15
                                                                                           : _abs_io_bOut[9]
                                                                                               ? 5'h16
                                                                                               : _abs_io_bOut[8]
                                                                                                   ? 5'h17
                                                                                                   : _abs_io_bOut[7]
                                                                                                       ? 5'h18
                                                                                                       : _abs_io_bOut[6]
                                                                                                           ? 5'h19
                                                                                                           : _abs_io_bOut[5]
                                                                                                               ? 5'h1A
                                                                                                               : _abs_io_bOut[4] ? 5'h1B : _abs_io_bOut[3] ? 5'h1C : _abs_io_bOut[2] ? 5'h1D : {4'hF, ~(_abs_io_bOut[1])}};
  wire [5:0]  sub = 6'h0 - zeroHeadDividend + zeroHeadDivisor;
  wire [6:0]  needComputerWidth = {1'h0, sub} + 7'h2;
  wire [3:0]  guardSele = 4'h1 << needComputerWidth[1:0];
  wire [1:0]  guardWidth = {2{guardSele[1]}} | {guardSele[2], 1'h0} | {1'h0, guardSele[3]};
  wire [7:0]  _counter_T = {1'h0, needComputerWidth} + {6'h0, guardWidth};
  wire [5:0]  counter = _counter_T[7:2];
  wire [5:0]  leftShiftWidthDividend = zeroHeadDividend + 6'h0 - {4'h0, guardWidth} + 6'h3;
  wire [5:0]  leftShiftWidthDivisor = {1'h0, zeroHeadDivisor[4:0]};
  reg         negativeSRT;
  reg  [5:0]  zeroHeadDivisorSRT;
  reg         dividendSignSRT;
  reg         divideZeroReg;
  reg         biggerdivisorReg;
  reg         bypassSRTReg;
  reg  [31:0] dividendInputReg;
  wire [94:0] _srt_input_bits_dividend_T = {63'h0, _abs_io_aOut} << leftShiftWidthDividend;
  wire [94:0] _srt_input_bits_divider_T = {63'h0, _abs_io_bOut} << leftShiftWidthDivisor;
  wire        srt_input_valid = input_valid_0 & ~bypassSRT;
  wire [31:0] remainderAbs = _srt_output_bits_reminder >> zeroHeadDivisorSRT[4:0];
  wire [31:0] quotientAbs;
  always @(posedge clock) begin
    if (_srt_input_ready & srt_input_valid) begin
      negativeSRT <= negative;
      zeroHeadDivisorSRT <= zeroHeadDivisor;
      dividendSignSRT <= _abs_io_aSign;
    end
    if (reset) begin
      divideZeroReg <= 1'h0;
      biggerdivisorReg <= 1'h0;
      bypassSRTReg <= 1'h0;
      dividendInputReg <= 32'h0;
    end
    else begin
      if (_dividendInputReg_T_1) begin
        divideZeroReg <= divideZero;
        biggerdivisorReg <= biggerdivisor;
        dividendInputReg <= input_bits_dividend_0;
      end
      bypassSRTReg <= bypassSRT;
    end
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
        negativeSRT = _RANDOM[1'h0][0];
        zeroHeadDivisorSRT = _RANDOM[1'h0][6:1];
        dividendSignSRT = _RANDOM[1'h0][7];
        divideZeroReg = _RANDOM[1'h0][8];
        biggerdivisorReg = _RANDOM[1'h0][9];
        bypassSRTReg = _RANDOM[1'h0][10];
        dividendInputReg = {_RANDOM[1'h0][31:11], _RANDOM[1'h1][10:0]};
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  assign input_ready_0 = _srt_input_ready;
  Abs abs (
    .io_aIn    (input_bits_dividend_0),
    .io_bIn    (input_bits_divisor_0),
    .io_signIn (input_bits_signIn_0),
    .io_aOut   (_abs_io_aOut),
    .io_bOut   (_abs_io_bOut),
    .io_aSign  (_abs_io_aSign),
    .io_bSign  (_abs_io_bSign)
  );
  SRT srt (
    .clock                (clock),
    .reset                (reset),
    .input_ready          (_srt_input_ready),
    .input_valid          (srt_input_valid),
    .input_bits_dividend  (_srt_input_bits_dividend_T[34:0]),
    .input_bits_divider   (_srt_input_bits_divider_T[31:0]),
    .input_bits_counter   (counter[4:0]),
    .output_valid         (_srt_output_valid),
    .output_bits_reminder (_srt_output_bits_reminder),
    .output_bits_quotient (quotientAbs)
  );
  assign input_ready = input_ready_0;
  assign output_valid = _srt_output_valid | bypassSRTReg;
  assign output_bits_reminder = divideZeroReg | biggerdivisorReg ? dividendInputReg : dividendSignSRT ? 32'h0 - remainderAbs : remainderAbs;
  assign output_bits_quotient = divideZeroReg ? 32'hFFFFFFFF : biggerdivisorReg ? 32'h0 : negativeSRT ? 32'h0 - quotientAbs : quotientAbs;
endmodule

