module CSA42_1(
  input  [23:0] in_0,
                in_1,
                in_2,
                in_3,
  output [24:0] out_0,
                out_1
);

  wire        _compressor_23_out_0;
  wire        _compressor_23_out_1;
  wire        _compressor_23_cout;
  wire        _compressor_22_out_0;
  wire        _compressor_22_out_1;
  wire        _compressor_22_cout;
  wire        _compressor_21_out_0;
  wire        _compressor_21_out_1;
  wire        _compressor_21_cout;
  wire        _compressor_20_out_0;
  wire        _compressor_20_out_1;
  wire        _compressor_20_cout;
  wire        _compressor_19_out_0;
  wire        _compressor_19_out_1;
  wire        _compressor_19_cout;
  wire        _compressor_18_out_0;
  wire        _compressor_18_out_1;
  wire        _compressor_18_cout;
  wire        _compressor_17_out_0;
  wire        _compressor_17_out_1;
  wire        _compressor_17_cout;
  wire        _compressor_16_out_0;
  wire        _compressor_16_out_1;
  wire        _compressor_16_cout;
  wire        _compressor_15_out_0;
  wire        _compressor_15_out_1;
  wire        _compressor_15_cout;
  wire        _compressor_14_out_0;
  wire        _compressor_14_out_1;
  wire        _compressor_14_cout;
  wire        _compressor_13_out_0;
  wire        _compressor_13_out_1;
  wire        _compressor_13_cout;
  wire        _compressor_12_out_0;
  wire        _compressor_12_out_1;
  wire        _compressor_12_cout;
  wire        _compressor_11_out_0;
  wire        _compressor_11_out_1;
  wire        _compressor_11_cout;
  wire        _compressor_10_out_0;
  wire        _compressor_10_out_1;
  wire        _compressor_10_cout;
  wire        _compressor_9_out_0;
  wire        _compressor_9_out_1;
  wire        _compressor_9_cout;
  wire        _compressor_8_out_0;
  wire        _compressor_8_out_1;
  wire        _compressor_8_cout;
  wire        _compressor_7_out_0;
  wire        _compressor_7_out_1;
  wire        _compressor_7_cout;
  wire        _compressor_6_out_0;
  wire        _compressor_6_out_1;
  wire        _compressor_6_cout;
  wire        _compressor_5_out_0;
  wire        _compressor_5_out_1;
  wire        _compressor_5_cout;
  wire        _compressor_4_out_0;
  wire        _compressor_4_out_1;
  wire        _compressor_4_cout;
  wire        _compressor_3_out_0;
  wire        _compressor_3_out_1;
  wire        _compressor_3_cout;
  wire        _compressor_2_out_0;
  wire        _compressor_2_out_1;
  wire        _compressor_2_cout;
  wire        _compressor_1_out_0;
  wire        _compressor_1_out_1;
  wire        _compressor_1_cout;
  wire        _compressor_0_out_0;
  wire        _compressor_0_out_1;
  wire        _compressor_0_cout;
  wire        cinVec_0 = 1'h0;
  wire [1:0]  coutUInt_lo_lo_lo_hi = {_compressor_2_cout, _compressor_1_cout};
  wire [2:0]  coutUInt_lo_lo_lo = {coutUInt_lo_lo_lo_hi, _compressor_0_cout};
  wire [1:0]  coutUInt_lo_lo_hi_hi = {_compressor_5_cout, _compressor_4_cout};
  wire [2:0]  coutUInt_lo_lo_hi = {coutUInt_lo_lo_hi_hi, _compressor_3_cout};
  wire [5:0]  coutUInt_lo_lo = {coutUInt_lo_lo_hi, coutUInt_lo_lo_lo};
  wire [1:0]  coutUInt_lo_hi_lo_hi = {_compressor_8_cout, _compressor_7_cout};
  wire [2:0]  coutUInt_lo_hi_lo = {coutUInt_lo_hi_lo_hi, _compressor_6_cout};
  wire [1:0]  coutUInt_lo_hi_hi_hi = {_compressor_11_cout, _compressor_10_cout};
  wire [2:0]  coutUInt_lo_hi_hi = {coutUInt_lo_hi_hi_hi, _compressor_9_cout};
  wire [5:0]  coutUInt_lo_hi = {coutUInt_lo_hi_hi, coutUInt_lo_hi_lo};
  wire [11:0] coutUInt_lo = {coutUInt_lo_hi, coutUInt_lo_lo};
  wire [1:0]  coutUInt_hi_lo_lo_hi = {_compressor_14_cout, _compressor_13_cout};
  wire [2:0]  coutUInt_hi_lo_lo = {coutUInt_hi_lo_lo_hi, _compressor_12_cout};
  wire [1:0]  coutUInt_hi_lo_hi_hi = {_compressor_17_cout, _compressor_16_cout};
  wire [2:0]  coutUInt_hi_lo_hi = {coutUInt_hi_lo_hi_hi, _compressor_15_cout};
  wire [5:0]  coutUInt_hi_lo = {coutUInt_hi_lo_hi, coutUInt_hi_lo_lo};
  wire [1:0]  coutUInt_hi_hi_lo_hi = {_compressor_20_cout, _compressor_19_cout};
  wire [2:0]  coutUInt_hi_hi_lo = {coutUInt_hi_hi_lo_hi, _compressor_18_cout};
  wire [1:0]  coutUInt_hi_hi_hi_hi = {_compressor_23_cout, _compressor_22_cout};
  wire [2:0]  coutUInt_hi_hi_hi = {coutUInt_hi_hi_hi_hi, _compressor_21_cout};
  wire [5:0]  coutUInt_hi_hi = {coutUInt_hi_hi_hi, coutUInt_hi_hi_lo};
  wire [11:0] coutUInt_hi = {coutUInt_hi_hi, coutUInt_hi_lo};
  wire [23:0] coutUInt = {coutUInt_hi, coutUInt_lo};
  wire        cinVec_1 = coutUInt[0];
  wire        cinVec_2 = coutUInt[1];
  wire        cinVec_3 = coutUInt[2];
  wire        cinVec_4 = coutUInt[3];
  wire        cinVec_5 = coutUInt[4];
  wire        cinVec_6 = coutUInt[5];
  wire        cinVec_7 = coutUInt[6];
  wire        cinVec_8 = coutUInt[7];
  wire        cinVec_9 = coutUInt[8];
  wire        cinVec_10 = coutUInt[9];
  wire        cinVec_11 = coutUInt[10];
  wire        cinVec_12 = coutUInt[11];
  wire        cinVec_13 = coutUInt[12];
  wire        cinVec_14 = coutUInt[13];
  wire        cinVec_15 = coutUInt[14];
  wire        cinVec_16 = coutUInt[15];
  wire        cinVec_17 = coutUInt[16];
  wire        cinVec_18 = coutUInt[17];
  wire        cinVec_19 = coutUInt[18];
  wire        cinVec_20 = coutUInt[19];
  wire        cinVec_21 = coutUInt[20];
  wire        cinVec_22 = coutUInt[21];
  wire        cinVec_23 = coutUInt[22];
  wire [1:0]  out_1_lo_lo_lo_hi = {_compressor_2_out_1, _compressor_1_out_1};
  wire [2:0]  out_1_lo_lo_lo = {out_1_lo_lo_lo_hi, _compressor_0_out_1};
  wire [1:0]  out_1_lo_lo_hi_hi = {_compressor_5_out_1, _compressor_4_out_1};
  wire [2:0]  out_1_lo_lo_hi = {out_1_lo_lo_hi_hi, _compressor_3_out_1};
  wire [5:0]  out_1_lo_lo = {out_1_lo_lo_hi, out_1_lo_lo_lo};
  wire [1:0]  out_1_lo_hi_lo_hi = {_compressor_8_out_1, _compressor_7_out_1};
  wire [2:0]  out_1_lo_hi_lo = {out_1_lo_hi_lo_hi, _compressor_6_out_1};
  wire [1:0]  out_1_lo_hi_hi_hi = {_compressor_11_out_1, _compressor_10_out_1};
  wire [2:0]  out_1_lo_hi_hi = {out_1_lo_hi_hi_hi, _compressor_9_out_1};
  wire [5:0]  out_1_lo_hi = {out_1_lo_hi_hi, out_1_lo_hi_lo};
  wire [11:0] out_1_lo = {out_1_lo_hi, out_1_lo_lo};
  wire [1:0]  out_1_hi_lo_lo_hi = {_compressor_14_out_1, _compressor_13_out_1};
  wire [2:0]  out_1_hi_lo_lo = {out_1_hi_lo_lo_hi, _compressor_12_out_1};
  wire [1:0]  out_1_hi_lo_hi_hi = {_compressor_17_out_1, _compressor_16_out_1};
  wire [2:0]  out_1_hi_lo_hi = {out_1_hi_lo_hi_hi, _compressor_15_out_1};
  wire [5:0]  out_1_hi_lo = {out_1_hi_lo_hi, out_1_hi_lo_lo};
  wire [1:0]  out_1_hi_hi_lo_hi = {_compressor_20_out_1, _compressor_19_out_1};
  wire [2:0]  out_1_hi_hi_lo = {out_1_hi_hi_lo_hi, _compressor_18_out_1};
  wire [1:0]  out_1_hi_hi_hi_lo = {_compressor_22_out_1, _compressor_21_out_1};
  wire [1:0]  out_1_hi_hi_hi_hi = {coutUInt[23], _compressor_23_out_1};
  wire [3:0]  out_1_hi_hi_hi = {out_1_hi_hi_hi_hi, out_1_hi_hi_hi_lo};
  wire [6:0]  out_1_hi_hi = {out_1_hi_hi_hi, out_1_hi_hi_lo};
  wire [12:0] out_1_hi = {out_1_hi_hi, out_1_hi_lo};
  wire [1:0]  out_0_lo_lo_lo_hi = {_compressor_2_out_0, _compressor_1_out_0};
  wire [2:0]  out_0_lo_lo_lo = {out_0_lo_lo_lo_hi, _compressor_0_out_0};
  wire [1:0]  out_0_lo_lo_hi_hi = {_compressor_5_out_0, _compressor_4_out_0};
  wire [2:0]  out_0_lo_lo_hi = {out_0_lo_lo_hi_hi, _compressor_3_out_0};
  wire [5:0]  out_0_lo_lo = {out_0_lo_lo_hi, out_0_lo_lo_lo};
  wire [1:0]  out_0_lo_hi_lo_hi = {_compressor_8_out_0, _compressor_7_out_0};
  wire [2:0]  out_0_lo_hi_lo = {out_0_lo_hi_lo_hi, _compressor_6_out_0};
  wire [1:0]  out_0_lo_hi_hi_hi = {_compressor_11_out_0, _compressor_10_out_0};
  wire [2:0]  out_0_lo_hi_hi = {out_0_lo_hi_hi_hi, _compressor_9_out_0};
  wire [5:0]  out_0_lo_hi = {out_0_lo_hi_hi, out_0_lo_hi_lo};
  wire [11:0] out_0_lo = {out_0_lo_hi, out_0_lo_lo};
  wire [1:0]  out_0_hi_lo_lo_hi = {_compressor_14_out_0, _compressor_13_out_0};
  wire [2:0]  out_0_hi_lo_lo = {out_0_hi_lo_lo_hi, _compressor_12_out_0};
  wire [1:0]  out_0_hi_lo_hi_hi = {_compressor_17_out_0, _compressor_16_out_0};
  wire [2:0]  out_0_hi_lo_hi = {out_0_hi_lo_hi_hi, _compressor_15_out_0};
  wire [5:0]  out_0_hi_lo = {out_0_hi_lo_hi, out_0_hi_lo_lo};
  wire [1:0]  out_0_hi_hi_lo_hi = {_compressor_20_out_0, _compressor_19_out_0};
  wire [2:0]  out_0_hi_hi_lo = {out_0_hi_hi_lo_hi, _compressor_18_out_0};
  wire [1:0]  out_0_hi_hi_hi_hi = {_compressor_23_out_0, _compressor_22_out_0};
  wire [2:0]  out_0_hi_hi_hi = {out_0_hi_hi_hi_hi, _compressor_21_out_0};
  wire [5:0]  out_0_hi_hi = {out_0_hi_hi_hi, out_0_hi_hi_lo};
  wire [11:0] out_0_hi = {out_0_hi_hi, out_0_hi_lo};
  CSACompressor4_2 compressor_0 (
    .in_0  (in_0[0]),
    .in_1  (in_1[0]),
    .in_2  (in_2[0]),
    .in_3  (in_3[0]),
    .cin   (cinVec_0),
    .out_0 (_compressor_0_out_0),
    .out_1 (_compressor_0_out_1),
    .cout  (_compressor_0_cout)
  );
  CSACompressor4_2 compressor_1 (
    .in_0  (in_0[1]),
    .in_1  (in_1[1]),
    .in_2  (in_2[1]),
    .in_3  (in_3[1]),
    .cin   (cinVec_1),
    .out_0 (_compressor_1_out_0),
    .out_1 (_compressor_1_out_1),
    .cout  (_compressor_1_cout)
  );
  CSACompressor4_2 compressor_2 (
    .in_0  (in_0[2]),
    .in_1  (in_1[2]),
    .in_2  (in_2[2]),
    .in_3  (in_3[2]),
    .cin   (cinVec_2),
    .out_0 (_compressor_2_out_0),
    .out_1 (_compressor_2_out_1),
    .cout  (_compressor_2_cout)
  );
  CSACompressor4_2 compressor_3 (
    .in_0  (in_0[3]),
    .in_1  (in_1[3]),
    .in_2  (in_2[3]),
    .in_3  (in_3[3]),
    .cin   (cinVec_3),
    .out_0 (_compressor_3_out_0),
    .out_1 (_compressor_3_out_1),
    .cout  (_compressor_3_cout)
  );
  CSACompressor4_2 compressor_4 (
    .in_0  (in_0[4]),
    .in_1  (in_1[4]),
    .in_2  (in_2[4]),
    .in_3  (in_3[4]),
    .cin   (cinVec_4),
    .out_0 (_compressor_4_out_0),
    .out_1 (_compressor_4_out_1),
    .cout  (_compressor_4_cout)
  );
  CSACompressor4_2 compressor_5 (
    .in_0  (in_0[5]),
    .in_1  (in_1[5]),
    .in_2  (in_2[5]),
    .in_3  (in_3[5]),
    .cin   (cinVec_5),
    .out_0 (_compressor_5_out_0),
    .out_1 (_compressor_5_out_1),
    .cout  (_compressor_5_cout)
  );
  CSACompressor4_2 compressor_6 (
    .in_0  (in_0[6]),
    .in_1  (in_1[6]),
    .in_2  (in_2[6]),
    .in_3  (in_3[6]),
    .cin   (cinVec_6),
    .out_0 (_compressor_6_out_0),
    .out_1 (_compressor_6_out_1),
    .cout  (_compressor_6_cout)
  );
  CSACompressor4_2 compressor_7 (
    .in_0  (in_0[7]),
    .in_1  (in_1[7]),
    .in_2  (in_2[7]),
    .in_3  (in_3[7]),
    .cin   (cinVec_7),
    .out_0 (_compressor_7_out_0),
    .out_1 (_compressor_7_out_1),
    .cout  (_compressor_7_cout)
  );
  CSACompressor4_2 compressor_8 (
    .in_0  (in_0[8]),
    .in_1  (in_1[8]),
    .in_2  (in_2[8]),
    .in_3  (in_3[8]),
    .cin   (cinVec_8),
    .out_0 (_compressor_8_out_0),
    .out_1 (_compressor_8_out_1),
    .cout  (_compressor_8_cout)
  );
  CSACompressor4_2 compressor_9 (
    .in_0  (in_0[9]),
    .in_1  (in_1[9]),
    .in_2  (in_2[9]),
    .in_3  (in_3[9]),
    .cin   (cinVec_9),
    .out_0 (_compressor_9_out_0),
    .out_1 (_compressor_9_out_1),
    .cout  (_compressor_9_cout)
  );
  CSACompressor4_2 compressor_10 (
    .in_0  (in_0[10]),
    .in_1  (in_1[10]),
    .in_2  (in_2[10]),
    .in_3  (in_3[10]),
    .cin   (cinVec_10),
    .out_0 (_compressor_10_out_0),
    .out_1 (_compressor_10_out_1),
    .cout  (_compressor_10_cout)
  );
  CSACompressor4_2 compressor_11 (
    .in_0  (in_0[11]),
    .in_1  (in_1[11]),
    .in_2  (in_2[11]),
    .in_3  (in_3[11]),
    .cin   (cinVec_11),
    .out_0 (_compressor_11_out_0),
    .out_1 (_compressor_11_out_1),
    .cout  (_compressor_11_cout)
  );
  CSACompressor4_2 compressor_12 (
    .in_0  (in_0[12]),
    .in_1  (in_1[12]),
    .in_2  (in_2[12]),
    .in_3  (in_3[12]),
    .cin   (cinVec_12),
    .out_0 (_compressor_12_out_0),
    .out_1 (_compressor_12_out_1),
    .cout  (_compressor_12_cout)
  );
  CSACompressor4_2 compressor_13 (
    .in_0  (in_0[13]),
    .in_1  (in_1[13]),
    .in_2  (in_2[13]),
    .in_3  (in_3[13]),
    .cin   (cinVec_13),
    .out_0 (_compressor_13_out_0),
    .out_1 (_compressor_13_out_1),
    .cout  (_compressor_13_cout)
  );
  CSACompressor4_2 compressor_14 (
    .in_0  (in_0[14]),
    .in_1  (in_1[14]),
    .in_2  (in_2[14]),
    .in_3  (in_3[14]),
    .cin   (cinVec_14),
    .out_0 (_compressor_14_out_0),
    .out_1 (_compressor_14_out_1),
    .cout  (_compressor_14_cout)
  );
  CSACompressor4_2 compressor_15 (
    .in_0  (in_0[15]),
    .in_1  (in_1[15]),
    .in_2  (in_2[15]),
    .in_3  (in_3[15]),
    .cin   (cinVec_15),
    .out_0 (_compressor_15_out_0),
    .out_1 (_compressor_15_out_1),
    .cout  (_compressor_15_cout)
  );
  CSACompressor4_2 compressor_16 (
    .in_0  (in_0[16]),
    .in_1  (in_1[16]),
    .in_2  (in_2[16]),
    .in_3  (in_3[16]),
    .cin   (cinVec_16),
    .out_0 (_compressor_16_out_0),
    .out_1 (_compressor_16_out_1),
    .cout  (_compressor_16_cout)
  );
  CSACompressor4_2 compressor_17 (
    .in_0  (in_0[17]),
    .in_1  (in_1[17]),
    .in_2  (in_2[17]),
    .in_3  (in_3[17]),
    .cin   (cinVec_17),
    .out_0 (_compressor_17_out_0),
    .out_1 (_compressor_17_out_1),
    .cout  (_compressor_17_cout)
  );
  CSACompressor4_2 compressor_18 (
    .in_0  (in_0[18]),
    .in_1  (in_1[18]),
    .in_2  (in_2[18]),
    .in_3  (in_3[18]),
    .cin   (cinVec_18),
    .out_0 (_compressor_18_out_0),
    .out_1 (_compressor_18_out_1),
    .cout  (_compressor_18_cout)
  );
  CSACompressor4_2 compressor_19 (
    .in_0  (in_0[19]),
    .in_1  (in_1[19]),
    .in_2  (in_2[19]),
    .in_3  (in_3[19]),
    .cin   (cinVec_19),
    .out_0 (_compressor_19_out_0),
    .out_1 (_compressor_19_out_1),
    .cout  (_compressor_19_cout)
  );
  CSACompressor4_2 compressor_20 (
    .in_0  (in_0[20]),
    .in_1  (in_1[20]),
    .in_2  (in_2[20]),
    .in_3  (in_3[20]),
    .cin   (cinVec_20),
    .out_0 (_compressor_20_out_0),
    .out_1 (_compressor_20_out_1),
    .cout  (_compressor_20_cout)
  );
  CSACompressor4_2 compressor_21 (
    .in_0  (in_0[21]),
    .in_1  (in_1[21]),
    .in_2  (in_2[21]),
    .in_3  (in_3[21]),
    .cin   (cinVec_21),
    .out_0 (_compressor_21_out_0),
    .out_1 (_compressor_21_out_1),
    .cout  (_compressor_21_cout)
  );
  CSACompressor4_2 compressor_22 (
    .in_0  (in_0[22]),
    .in_1  (in_1[22]),
    .in_2  (in_2[22]),
    .in_3  (in_3[22]),
    .cin   (cinVec_22),
    .out_0 (_compressor_22_out_0),
    .out_1 (_compressor_22_out_1),
    .cout  (_compressor_22_cout)
  );
  CSACompressor4_2 compressor_23 (
    .in_0  (in_0[23]),
    .in_1  (in_1[23]),
    .in_2  (in_2[23]),
    .in_3  (in_3[23]),
    .cin   (cinVec_23),
    .out_0 (_compressor_23_out_0),
    .out_1 (_compressor_23_out_1),
    .cout  (_compressor_23_cout)
  );
  assign out_0 = {1'h0, out_0_hi, out_0_lo};
  assign out_1 = {out_1_hi, out_1_lo};
endmodule

