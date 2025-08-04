
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
module SqrtIter(
  input         clock,
                reset,
  output        input_ready,
  input         input_valid,
  input  [27:0] input_bits_partialSum,
                input_bits_partialCarry,
  output        resultOutput_valid,
  output [25:0] resultOutput_bits_result,
  output        resultOutput_bits_zeroRemainder,
  output [27:0] output_partialSum,
                output_partialCarry,
  output        output_isLastCycle,
  output [25:0] reqOTF_quotient,
                reqOTF_quotientMinusOne,
  output [4:0]  reqOTF_selectedQuotientOH,
  input  [25:0] respOTF_quotient,
                respOTF_quotientMinusOne
);

  wire [27:0] _csa_m_out_0;
  wire [4:0]  _selectedQuotientOH_m_output_selectedQuotientOH;
  wire        input_valid_0 = input_valid;
  wire [27:0] input_bits_partialSum_0 = input_bits_partialSum;
  wire [27:0] input_bits_partialCarry_0 = input_bits_partialCarry;
  reg         occupied;
  reg  [4:0]  counter;
  wire        input_ready_0;
  wire        _counterNext_T = input_ready_0 & input_valid_0;
  wire        isLastCycle;
  wire        occupiedNext = _counterNext_T | ~isLastCycle & occupied;
  assign isLastCycle = counter == 5'hD;
  assign input_ready_0 = ~occupied;
  wire        enable = _counterNext_T | ~isLastCycle;
  reg  [25:0] resultOrigin;
  reg  [25:0] resultMinusOne;
  wire [29:0] shiftSum = {input_bits_partialSum_0, 2'h0};
  wire [29:0] shiftCarry = {input_bits_partialCarry_0, 2'h0};
  wire [56:0] _resultOriginRestore_T_2 = {5'h0, resultOrigin, 26'h0} >> {51'h0, counter, 1'h0};
  wire [26:0] resultOriginRestore = _resultOriginRestore_T_2[26:0];
  wire [2:0]  resultForQDS = counter == 5'h0 ? 3'h5 : resultOriginRestore[26] ? 3'h7 : resultOriginRestore[24:22];
  wire [29:0] _formationForIter_T_15 = _selectedQuotientOH_m_output_selectedQuotientOH[0] ? {resultMinusOne, 4'hC} : 30'h0;
  wire [29:0] formationForIter =
    {_formationForIter_T_15[29], _formationForIter_T_15[28:0] | (_selectedQuotientOH_m_output_selectedQuotientOH[1] ? {resultMinusOne, 3'h7} : 29'h0) | (_selectedQuotientOH_m_output_selectedQuotientOH[3] ? {~resultOrigin, 3'h7} : 29'h0)}
    | (_selectedQuotientOH_m_output_selectedQuotientOH[4] ? {~resultOrigin, 4'hC} : 30'h0);
  wire [53:0] _formationFinal_T_2 = {formationForIter, 24'h0} >> {48'h0, counter, 1'h0};
  wire [28:0] formationFinal = _formationFinal_T_2[28:0];
  wire [27:0] remainderFinal = input_bits_partialSum_0 + input_bits_partialCarry_0;
  wire        needCorrect = remainderFinal[27];
  wire [25:0] resultOriginNext = _counterNext_T ? 26'h1 : respOTF_quotient;
  wire [25:0] resultMinusOneNext = _counterNext_T ? 26'h0 : respOTF_quotientMinusOne;
  wire [4:0]  counterNext = _counterNext_T ? 5'h0 : counter + 5'h1;
  always @(posedge clock) begin
    if (reset) begin
      occupied <= 1'h0;
      counter <= 5'h0;
      resultOrigin <= 26'h0;
      resultMinusOne <= 26'h0;
    end
    else begin
      occupied <= occupiedNext;
      if (enable) begin
        counter <= counterNext;
        resultOrigin <= resultOriginNext;
        resultMinusOne <= resultMinusOneNext;
      end
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
        occupied = _RANDOM[1'h0][0];
        counter = _RANDOM[1'h0][5:1];
        resultOrigin = _RANDOM[1'h0][31:6];
        resultMinusOne = _RANDOM[1'h1][25:0];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  QDS selectedQuotientOH_m (
    .input_partialReminderCarry (shiftCarry[29:23]),
    .input_partialReminderSum   (shiftSum[29:23]),
    .input_partialDivider       (resultForQDS),
    .output_selectedQuotientOH  (_selectedQuotientOH_m_output_selectedQuotientOH)
  );
  CarrySaveAdder_28 csa_m (
    .in_0  (shiftSum[27:0]),
    .in_1  (shiftCarry[27:0]),
    .in_2  (formationFinal[27:0]),
    .out_0 (_csa_m_out_0),
    .out_1 (output_partialSum)
  );
  assign input_ready = input_ready_0;
  assign resultOutput_valid = occupied & isLastCycle;
  assign resultOutput_bits_result = needCorrect ? resultMinusOne : resultOrigin;
  assign resultOutput_bits_zeroRemainder = remainderFinal == 28'h0;
  assign output_partialCarry = {_csa_m_out_0[26:0], 1'h0};
  assign output_isLastCycle = isLastCycle;
  assign reqOTF_quotient = resultOrigin;
  assign reqOTF_quotientMinusOne = resultMinusOne;
  assign reqOTF_selectedQuotientOH = _selectedQuotientOH_m_output_selectedQuotientOH;
endmodule

