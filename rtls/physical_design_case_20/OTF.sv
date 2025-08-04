module OTF(
  input  [31:0] input_quotient,
                input_quotientMinusOne,
  input  [4:0]  input_selectedQuotientOH,
  output [31:0] output_quotient,
                output_quotientMinusOne
);

  wire [2:0] _qNext_T_11 = (input_selectedQuotientOH[0] ? 3'h6 : 3'h0) | {3{input_selectedQuotientOH[1]}};
  wire [2:0] qNext = {_qNext_T_11[2], {_qNext_T_11[1], _qNext_T_11[0] | input_selectedQuotientOH[3]} | {input_selectedQuotientOH[4], 1'h0}};
  wire       cShiftQ = |(input_selectedQuotientOH[4:2]);
  wire       cShiftQM = |(input_selectedQuotientOH[2:0]);
  wire [1:0] qIn = qNext[1:0];
  wire [1:0] qmIn = qIn - 2'h1;
  assign output_quotient = {cShiftQ ? input_quotient[29:0] : input_quotientMinusOne[29:0], qIn};
  assign output_quotientMinusOne = {cShiftQM ? input_quotientMinusOne[29:0] : input_quotient[29:0], qmIn};
endmodule

