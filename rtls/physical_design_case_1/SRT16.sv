
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
module SRT16(
  input         clock,
                reset,
  output        input_ready,
  input         input_valid,
  input  [34:0] input_bits_dividend,
  input  [31:0] input_bits_divider,
  input  [4:0]  input_bits_counter,
  output        output_valid,
  output [31:0] output_bits_reminder,
                output_bits_quotient
);

  wire [37:0] _csa1Out_m_out_0;
  wire [37:0] _csa1Out_m_out_1;
  wire [37:0] _csa0Out_m_out_0;
  wire [37:0] _csa0Out_m_out_1;
  wire [4:0]  _qdsOH0_m_output_selectedQuotientOH;
  wire [9:0]  _csa5_m_out_0;
  wire [9:0]  _csa5_m_out_1;
  wire [9:0]  _csa4_m_out_0;
  wire [9:0]  _csa4_m_out_1;
  wire [9:0]  _csa3_m_out_0;
  wire [9:0]  _csa3_m_out_1;
  wire [9:0]  _csa2_m_out_0;
  wire [9:0]  _csa2_m_out_1;
  wire [9:0]  _csa1_m_out_0;
  wire [9:0]  _csa1_m_out_1;
  wire [31:0] otf0_1;
  wire [31:0] otf0_0;
  wire        input_valid_0 = input_valid;
  wire [34:0] input_bits_dividend_0 = input_bits_dividend;
  wire [31:0] input_bits_divider_0 = input_bits_divider;
  wire [4:0]  input_bits_counter_0 = input_bits_counter;
  wire [37:0] dividerMap_2 = 38'h0;
  reg  [37:0] partialReminderCarry;
  reg  [37:0] partialReminderSum;
  reg  [31:0] divider;
  reg  [31:0] quotient;
  reg  [31:0] quotientMinusOne;
  reg  [4:0]  counter;
  reg         occupied;
  wire        input_ready_0;
  wire        _partialReminderCarryNext_T = input_ready_0 & input_valid_0;
  wire        isLastCycle;
  wire        occupiedNext = _partialReminderCarryNext_T | ~isLastCycle & occupied;
  assign isLastCycle = counter == 5'h0;
  assign input_ready_0 = ~occupied;
  wire        enable = _partialReminderCarryNext_T | ~isLastCycle;
  wire [34:0] divisorExtended = {divider, 3'h0};
  wire [37:0] remainderNoCorrect = partialReminderSum + partialReminderCarry;
  wire [37:0] remainderCorrect = remainderNoCorrect + {1'h0, divisorExtended, 2'h0};
  wire        needCorrect = remainderNoCorrect[37];
  wire [37:0] dividerMap_3 = {3'h7, ~divisorExtended};
  wire [37:0] dividerMap_4 = {2'h3, ~divisorExtended, 1'h1};
  wire [37:0] dividerMap_0 = {2'h0, divisorExtended, 1'h0};
  wire [37:0] dividerMap_1 = {3'h0, divisorExtended};
  wire [9:0]  csaIn1 = partialReminderSum[37:28];
  wire [9:0]  csaIn2 = partialReminderCarry[37:28];
  wire [31:0] dividerNext;
  wire [2:0]  partialDivider = dividerNext[30:28];
  wire [4:0]  qds1SelectedQuotientOHMap_0;
  wire [4:0]  qds1SelectedQuotientOHMap_1;
  wire [4:0]  qds1SelectedQuotientOHMap_2;
  wire [4:0]  qds1SelectedQuotientOHMap_3;
  wire [4:0]  qds1SelectedQuotientOHMap_4;
  wire [4:0]  qdsOH1 =
    (_qdsOH0_m_output_selectedQuotientOH[0] ? qds1SelectedQuotientOHMap_0 : 5'h0) | (_qdsOH0_m_output_selectedQuotientOH[1] ? qds1SelectedQuotientOHMap_1 : 5'h0)
    | (_qdsOH0_m_output_selectedQuotientOH[2] ? qds1SelectedQuotientOHMap_2 : 5'h0) | (_qdsOH0_m_output_selectedQuotientOH[3] ? qds1SelectedQuotientOHMap_3 : 5'h0)
    | (_qdsOH0_m_output_selectedQuotientOH[4] ? qds1SelectedQuotientOHMap_4 : 5'h0);
  wire        qds0sign = |(_qdsOH0_m_output_selectedQuotientOH[4:3]);
  wire        qds1sign = |(qdsOH1[4:3]);
  assign dividerNext = _partialReminderCarryNext_T ? input_bits_divider_0 : divider;
  wire [4:0]  counterNext = _partialReminderCarryNext_T ? input_bits_counter_0 : counter - 5'h1;
  wire [31:0] otf1_0;
  wire [31:0] quotientNext = _partialReminderCarryNext_T ? 32'h0 : otf1_0;
  wire [31:0] otf1_1;
  wire [31:0] quotientMinusOneNext = _partialReminderCarryNext_T ? 32'h0 : otf1_1;
  wire [37:0] partialReminderSumNext = _partialReminderCarryNext_T ? {3'h0, input_bits_dividend_0} : {_csa1Out_m_out_1[35:0], 2'h0};
  wire [37:0] partialReminderCarryNext = _partialReminderCarryNext_T ? 38'h0 : {_csa1Out_m_out_0[34:0], 3'h0};
  always @(posedge clock) begin
    if (reset) begin
      partialReminderCarry <= 38'h0;
      partialReminderSum <= 38'h0;
      divider <= 32'h0;
      quotient <= 32'h0;
      quotientMinusOne <= 32'h0;
      counter <= 5'h0;
      occupied <= 1'h0;
    end
    else begin
      if (enable) begin
        partialReminderCarry <= partialReminderCarryNext;
        partialReminderSum <= partialReminderSumNext;
        divider <= dividerNext;
        quotient <= quotientNext;
        quotientMinusOne <= quotientMinusOneNext;
        counter <= counterNext;
      end
      occupied <= occupiedNext;
    end
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:5];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [2:0] i = 3'h0; i < 3'h6; i += 3'h1) begin
          _RANDOM[i] = `RANDOM;
        end
        partialReminderCarry = {_RANDOM[3'h0], _RANDOM[3'h1][5:0]};
        partialReminderSum = {_RANDOM[3'h1][31:6], _RANDOM[3'h2][11:0]};
        divider = {_RANDOM[3'h2][31:12], _RANDOM[3'h3][11:0]};
        quotient = {_RANDOM[3'h3][31:12], _RANDOM[3'h4][11:0]};
        quotientMinusOne = {_RANDOM[3'h4][31:12], _RANDOM[3'h5][11:0]};
        counter = _RANDOM[3'h5][16:12];
        occupied = _RANDOM[3'h5][17];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  CarrySaveAdder_10 csa1_m (
    .in_0  (csaIn1),
    .in_1  (csaIn2),
    .in_2  (dividerMap_0[37:28]),
    .out_0 (_csa1_m_out_0),
    .out_1 (_csa1_m_out_1)
  );
  CarrySaveAdder_10 csa2_m (
    .in_0  (csaIn1),
    .in_1  (csaIn2),
    .in_2  (dividerMap_1[37:28]),
    .out_0 (_csa2_m_out_0),
    .out_1 (_csa2_m_out_1)
  );
  CarrySaveAdder_10 csa3_m (
    .in_0  (csaIn1),
    .in_1  (csaIn2),
    .in_2  (10'h0),
    .out_0 (_csa3_m_out_0),
    .out_1 (_csa3_m_out_1)
  );
  CarrySaveAdder_10 csa4_m (
    .in_0  (csaIn1),
    .in_1  (csaIn2),
    .in_2  (dividerMap_3[37:28]),
    .out_0 (_csa4_m_out_0),
    .out_1 (_csa4_m_out_1)
  );
  CarrySaveAdder_10 csa5_m (
    .in_0  (csaIn1),
    .in_1  (csaIn2),
    .in_2  (dividerMap_4[37:28]),
    .out_0 (_csa5_m_out_0),
    .out_1 (_csa5_m_out_1)
  );
  QDS qdsOH0_m (
    .input_partialReminderCarry (partialReminderCarry[37:31]),
    .input_partialReminderSum   (partialReminderSum[37:31]),
    .input_partialDivider       (partialDivider),
    .output_selectedQuotientOH  (_qdsOH0_m_output_selectedQuotientOH)
  );
  QDS qds1SelectedQuotientOH_m (
    .input_partialReminderCarry (_csa1_m_out_0[6:0]),
    .input_partialReminderSum   (_csa1_m_out_1[7:1]),
    .input_partialDivider       (partialDivider),
    .output_selectedQuotientOH  (qds1SelectedQuotientOHMap_0)
  );
  QDS qds2SelectedQuotientOH_m (
    .input_partialReminderCarry (_csa2_m_out_0[6:0]),
    .input_partialReminderSum   (_csa2_m_out_1[7:1]),
    .input_partialDivider       (partialDivider),
    .output_selectedQuotientOH  (qds1SelectedQuotientOHMap_1)
  );
  QDS qds3SelectedQuotientOH_m (
    .input_partialReminderCarry (_csa3_m_out_0[6:0]),
    .input_partialReminderSum   (_csa3_m_out_1[7:1]),
    .input_partialDivider       (partialDivider),
    .output_selectedQuotientOH  (qds1SelectedQuotientOHMap_2)
  );
  QDS qds4SelectedQuotientOH_m (
    .input_partialReminderCarry (_csa4_m_out_0[6:0]),
    .input_partialReminderSum   (_csa4_m_out_1[7:1]),
    .input_partialDivider       (partialDivider),
    .output_selectedQuotientOH  (qds1SelectedQuotientOHMap_3)
  );
  QDS qds5SelectedQuotientOH_m (
    .input_partialReminderCarry (_csa5_m_out_0[6:0]),
    .input_partialReminderSum   (_csa5_m_out_1[7:1]),
    .input_partialDivider       (partialDivider),
    .output_selectedQuotientOH  (qds1SelectedQuotientOHMap_4)
  );
  CarrySaveAdder_38 csa0Out_m (
    .in_0  (partialReminderSum),
    .in_1  ({partialReminderCarry[37:1], qds0sign}),
    .in_2
      ((_qdsOH0_m_output_selectedQuotientOH[0] ? dividerMap_0 : 38'h0) | (_qdsOH0_m_output_selectedQuotientOH[1] ? dividerMap_1 : 38'h0) | (_qdsOH0_m_output_selectedQuotientOH[3] ? dividerMap_3 : 38'h0)
       | (_qdsOH0_m_output_selectedQuotientOH[4] ? dividerMap_4 : 38'h0)),
    .out_0 (_csa0Out_m_out_0),
    .out_1 (_csa0Out_m_out_1)
  );
  CarrySaveAdder_38 csa1Out_m (
    .in_0  ({_csa0Out_m_out_1[35:0], 2'h0}),
    .in_1  ({_csa0Out_m_out_0[34:0], 2'h0, qds1sign}),
    .in_2  ((qdsOH1[0] ? dividerMap_0 : 38'h0) | (qdsOH1[1] ? dividerMap_1 : 38'h0) | (qdsOH1[3] ? dividerMap_3 : 38'h0) | (qdsOH1[4] ? dividerMap_4 : 38'h0)),
    .out_0 (_csa1Out_m_out_0),
    .out_1 (_csa1Out_m_out_1)
  );
  OTF otf0_m (
    .input_quotient           (quotient),
    .input_quotientMinusOne   (quotientMinusOne),
    .input_selectedQuotientOH (_qdsOH0_m_output_selectedQuotientOH),
    .output_quotient          (otf0_0),
    .output_quotientMinusOne  (otf0_1)
  );
  OTF otf1_m (
    .input_quotient           (otf0_0),
    .input_quotientMinusOne   (otf0_1),
    .input_selectedQuotientOH (qdsOH1),
    .output_quotient          (otf1_0),
    .output_quotientMinusOne  (otf1_1)
  );
  assign input_ready = input_ready_0;
  assign output_valid = occupied & isLastCycle;
  assign output_bits_reminder = needCorrect ? remainderCorrect[36:5] : remainderNoCorrect[36:5];
  assign output_bits_quotient = needCorrect ? quotientMinusOne : quotient;
endmodule

