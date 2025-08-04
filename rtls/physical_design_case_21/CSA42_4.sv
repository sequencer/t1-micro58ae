module CSA42_4(
  input  [15:0] in_0,
                in_1,
                in_2,
                in_3,
  output [16:0] out_0,
                out_1
);

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
  wire [1:0]  coutUInt_lo_lo_lo = {_compressor_1_cout, _compressor_0_cout};
  wire [1:0]  coutUInt_lo_lo_hi = {_compressor_3_cout, _compressor_2_cout};
  wire [3:0]  coutUInt_lo_lo = {coutUInt_lo_lo_hi, coutUInt_lo_lo_lo};
  wire [1:0]  coutUInt_lo_hi_lo = {_compressor_5_cout, _compressor_4_cout};
  wire [1:0]  coutUInt_lo_hi_hi = {_compressor_7_cout, _compressor_6_cout};
  wire [3:0]  coutUInt_lo_hi = {coutUInt_lo_hi_hi, coutUInt_lo_hi_lo};
  wire [7:0]  coutUInt_lo = {coutUInt_lo_hi, coutUInt_lo_lo};
  wire [1:0]  coutUInt_hi_lo_lo = {_compressor_9_cout, _compressor_8_cout};
  wire [1:0]  coutUInt_hi_lo_hi = {_compressor_11_cout, _compressor_10_cout};
  wire [3:0]  coutUInt_hi_lo = {coutUInt_hi_lo_hi, coutUInt_hi_lo_lo};
  wire [1:0]  coutUInt_hi_hi_lo = {_compressor_13_cout, _compressor_12_cout};
  wire [1:0]  coutUInt_hi_hi_hi = {_compressor_15_cout, _compressor_14_cout};
  wire [3:0]  coutUInt_hi_hi = {coutUInt_hi_hi_hi, coutUInt_hi_hi_lo};
  wire [7:0]  coutUInt_hi = {coutUInt_hi_hi, coutUInt_hi_lo};
  wire [15:0] coutUInt = {coutUInt_hi, coutUInt_lo};
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
  wire [1:0]  out_1_lo_lo_lo = {_compressor_1_out_1, _compressor_0_out_1};
  wire [1:0]  out_1_lo_lo_hi = {_compressor_3_out_1, _compressor_2_out_1};
  wire [3:0]  out_1_lo_lo = {out_1_lo_lo_hi, out_1_lo_lo_lo};
  wire [1:0]  out_1_lo_hi_lo = {_compressor_5_out_1, _compressor_4_out_1};
  wire [1:0]  out_1_lo_hi_hi = {_compressor_7_out_1, _compressor_6_out_1};
  wire [3:0]  out_1_lo_hi = {out_1_lo_hi_hi, out_1_lo_hi_lo};
  wire [7:0]  out_1_lo = {out_1_lo_hi, out_1_lo_lo};
  wire [1:0]  out_1_hi_lo_lo = {_compressor_9_out_1, _compressor_8_out_1};
  wire [1:0]  out_1_hi_lo_hi = {_compressor_11_out_1, _compressor_10_out_1};
  wire [3:0]  out_1_hi_lo = {out_1_hi_lo_hi, out_1_hi_lo_lo};
  wire [1:0]  out_1_hi_hi_lo = {_compressor_13_out_1, _compressor_12_out_1};
  wire [1:0]  out_1_hi_hi_hi_hi = {coutUInt[15], _compressor_15_out_1};
  wire [2:0]  out_1_hi_hi_hi = {out_1_hi_hi_hi_hi, _compressor_14_out_1};
  wire [4:0]  out_1_hi_hi = {out_1_hi_hi_hi, out_1_hi_hi_lo};
  wire [8:0]  out_1_hi = {out_1_hi_hi, out_1_hi_lo};
  wire [1:0]  out_0_lo_lo_lo = {_compressor_1_out_0, _compressor_0_out_0};
  wire [1:0]  out_0_lo_lo_hi = {_compressor_3_out_0, _compressor_2_out_0};
  wire [3:0]  out_0_lo_lo = {out_0_lo_lo_hi, out_0_lo_lo_lo};
  wire [1:0]  out_0_lo_hi_lo = {_compressor_5_out_0, _compressor_4_out_0};
  wire [1:0]  out_0_lo_hi_hi = {_compressor_7_out_0, _compressor_6_out_0};
  wire [3:0]  out_0_lo_hi = {out_0_lo_hi_hi, out_0_lo_hi_lo};
  wire [7:0]  out_0_lo = {out_0_lo_hi, out_0_lo_lo};
  wire [1:0]  out_0_hi_lo_lo = {_compressor_9_out_0, _compressor_8_out_0};
  wire [1:0]  out_0_hi_lo_hi = {_compressor_11_out_0, _compressor_10_out_0};
  wire [3:0]  out_0_hi_lo = {out_0_hi_lo_hi, out_0_hi_lo_lo};
  wire [1:0]  out_0_hi_hi_lo = {_compressor_13_out_0, _compressor_12_out_0};
  wire [1:0]  out_0_hi_hi_hi = {_compressor_15_out_0, _compressor_14_out_0};
  wire [3:0]  out_0_hi_hi = {out_0_hi_hi_hi, out_0_hi_hi_lo};
  wire [7:0]  out_0_hi = {out_0_hi_hi, out_0_hi_lo};
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
  assign out_0 = {1'h0, out_0_hi, out_0_lo};
  assign out_1 = {out_1_hi, out_1_lo};
endmodule

