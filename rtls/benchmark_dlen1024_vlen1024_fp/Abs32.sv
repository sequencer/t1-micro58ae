module Abs32(
  input  [31:0] a,
  output [31:0] z,
  input  [2:0]  sew
);

  wire [3:0]  doComplement = (sew[0] ? {a[31], a[23], a[15], a[7]} : 4'h0) | (sew[1] ? {{2{a[31]}}, {2{a[15]}}} : 4'h0) | (sew[2] ? {4{a[31]}} : 4'h0);
  wire [7:0]  aSeq_0 = a[7:0];
  wire [7:0]  aSeq_1 = a[15:8];
  wire [7:0]  aSeq_2 = a[23:16];
  wire [7:0]  aSeq_3 = a[31:24];
  wire        doSeq_0 = doComplement[0];
  wire        doSeq_1 = doComplement[1];
  wire        doSeq_2 = doComplement[2];
  wire        doSeq_3 = doComplement[3];
  wire [7:0]  aRev_0 = {8{doSeq_0}} ^ aSeq_0;
  wire [7:0]  aRev_1 = {8{doSeq_1}} ^ aSeq_1;
  wire [7:0]  aRev_2 = {8{doSeq_2}} ^ aSeq_2;
  wire [7:0]  aRev_3 = {8{doSeq_3}} ^ aSeq_3;
  wire [8:0]  _GEN = {1'h0, aRev_0};
  wire [8:0]  aRevAddOne_sum = doSeq_0 ? _GEN + 9'h1 : _GEN;
  wire        aRevAddOne_0_1 = aRevAddOne_sum[8];
  wire [7:0]  aRevAddOne_0_2 = aRevAddOne_sum[7:0];
  wire [8:0]  _GEN_0 = {1'h0, aRev_1};
  wire [8:0]  aRevAddOne_sum_1 = doSeq_1 ? _GEN_0 + 9'h1 : _GEN_0;
  wire        aRevAddOne_1_1 = aRevAddOne_sum_1[8];
  wire [7:0]  aRevAddOne_1_2 = aRevAddOne_sum_1[7:0];
  wire [8:0]  _GEN_1 = {1'h0, aRev_2};
  wire [8:0]  aRevAddOne_sum_2 = doSeq_2 ? _GEN_1 + 9'h1 : _GEN_1;
  wire        aRevAddOne_2_1 = aRevAddOne_sum_2[8];
  wire [7:0]  aRevAddOne_2_2 = aRevAddOne_sum_2[7:0];
  wire [8:0]  _GEN_2 = {1'h0, aRev_3};
  wire [8:0]  aRevAddOne_sum_3 = doSeq_3 ? _GEN_2 + 9'h1 : _GEN_2;
  wire        aRevAddOne_3_1 = aRevAddOne_sum_3[8];
  wire [7:0]  aRevAddOne_3_2 = aRevAddOne_sum_3[7:0];
  wire [15:0] result8_lo = {aRevAddOne_1_2, aRevAddOne_0_2};
  wire [15:0] result8_hi = {aRevAddOne_3_2, aRevAddOne_2_2};
  wire [31:0] result8 = {result8_hi, result8_lo};
  wire [15:0] _GEN_3 = {aRevAddOne_0_1 ? aRevAddOne_1_2 : aRev_1, aRevAddOne_0_2};
  wire [15:0] result16_lo;
  assign result16_lo = _GEN_3;
  wire [15:0] result32_lo;
  assign result32_lo = _GEN_3;
  wire [15:0] result16_hi = {aRevAddOne_2_1 ? aRevAddOne_3_2 : aRev_3, aRevAddOne_2_2};
  wire [31:0] result16 = {result16_hi, result16_lo};
  wire        _result32_T_3 = aRevAddOne_0_1 & aRevAddOne_1_1;
  wire [15:0] result32_hi = {_result32_T_3 & aRevAddOne_2_1 ? aRevAddOne_3_2 : aRev_3, _result32_T_3 ? aRevAddOne_2_2 : aRev_2};
  wire [31:0] result32 = {result32_hi, result32_lo};
  assign z = (sew[0] ? result8 : 32'h0) | (sew[1] ? result16 : 32'h0) | (sew[2] ? result32 : 32'h0);
endmodule

