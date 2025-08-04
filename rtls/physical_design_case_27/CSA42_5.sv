module CSA42_5(
  input  [47:0] in_0,
                in_1,
                in_2,
                in_3,
  output [48:0] out_0,
                out_1
);

  wire        _compressor_47_out_0;
  wire        _compressor_47_out_1;
  wire        _compressor_47_cout;
  wire        _compressor_46_out_0;
  wire        _compressor_46_out_1;
  wire        _compressor_46_cout;
  wire        _compressor_45_out_0;
  wire        _compressor_45_out_1;
  wire        _compressor_45_cout;
  wire        _compressor_44_out_0;
  wire        _compressor_44_out_1;
  wire        _compressor_44_cout;
  wire        _compressor_43_out_0;
  wire        _compressor_43_out_1;
  wire        _compressor_43_cout;
  wire        _compressor_42_out_0;
  wire        _compressor_42_out_1;
  wire        _compressor_42_cout;
  wire        _compressor_41_out_0;
  wire        _compressor_41_out_1;
  wire        _compressor_41_cout;
  wire        _compressor_40_out_0;
  wire        _compressor_40_out_1;
  wire        _compressor_40_cout;
  wire        _compressor_39_out_0;
  wire        _compressor_39_out_1;
  wire        _compressor_39_cout;
  wire        _compressor_38_out_0;
  wire        _compressor_38_out_1;
  wire        _compressor_38_cout;
  wire        _compressor_37_out_0;
  wire        _compressor_37_out_1;
  wire        _compressor_37_cout;
  wire        _compressor_36_out_0;
  wire        _compressor_36_out_1;
  wire        _compressor_36_cout;
  wire        _compressor_35_out_0;
  wire        _compressor_35_out_1;
  wire        _compressor_35_cout;
  wire        _compressor_34_out_0;
  wire        _compressor_34_out_1;
  wire        _compressor_34_cout;
  wire        _compressor_33_out_0;
  wire        _compressor_33_out_1;
  wire        _compressor_33_cout;
  wire        _compressor_32_out_0;
  wire        _compressor_32_out_1;
  wire        _compressor_32_cout;
  wire        _compressor_31_out_0;
  wire        _compressor_31_out_1;
  wire        _compressor_31_cout;
  wire        _compressor_30_out_0;
  wire        _compressor_30_out_1;
  wire        _compressor_30_cout;
  wire        _compressor_29_out_0;
  wire        _compressor_29_out_1;
  wire        _compressor_29_cout;
  wire        _compressor_28_out_0;
  wire        _compressor_28_out_1;
  wire        _compressor_28_cout;
  wire        _compressor_27_out_0;
  wire        _compressor_27_out_1;
  wire        _compressor_27_cout;
  wire        _compressor_26_out_0;
  wire        _compressor_26_out_1;
  wire        _compressor_26_cout;
  wire        _compressor_25_out_0;
  wire        _compressor_25_out_1;
  wire        _compressor_25_cout;
  wire        _compressor_24_out_0;
  wire        _compressor_24_out_1;
  wire        _compressor_24_cout;
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
  wire [1:0]  coutUInt_lo_lo_lo_lo_hi = {_compressor_2_cout, _compressor_1_cout};
  wire [2:0]  coutUInt_lo_lo_lo_lo = {coutUInt_lo_lo_lo_lo_hi, _compressor_0_cout};
  wire [1:0]  coutUInt_lo_lo_lo_hi_hi = {_compressor_5_cout, _compressor_4_cout};
  wire [2:0]  coutUInt_lo_lo_lo_hi = {coutUInt_lo_lo_lo_hi_hi, _compressor_3_cout};
  wire [5:0]  coutUInt_lo_lo_lo = {coutUInt_lo_lo_lo_hi, coutUInt_lo_lo_lo_lo};
  wire [1:0]  coutUInt_lo_lo_hi_lo_hi = {_compressor_8_cout, _compressor_7_cout};
  wire [2:0]  coutUInt_lo_lo_hi_lo = {coutUInt_lo_lo_hi_lo_hi, _compressor_6_cout};
  wire [1:0]  coutUInt_lo_lo_hi_hi_hi = {_compressor_11_cout, _compressor_10_cout};
  wire [2:0]  coutUInt_lo_lo_hi_hi = {coutUInt_lo_lo_hi_hi_hi, _compressor_9_cout};
  wire [5:0]  coutUInt_lo_lo_hi = {coutUInt_lo_lo_hi_hi, coutUInt_lo_lo_hi_lo};
  wire [11:0] coutUInt_lo_lo = {coutUInt_lo_lo_hi, coutUInt_lo_lo_lo};
  wire [1:0]  coutUInt_lo_hi_lo_lo_hi = {_compressor_14_cout, _compressor_13_cout};
  wire [2:0]  coutUInt_lo_hi_lo_lo = {coutUInt_lo_hi_lo_lo_hi, _compressor_12_cout};
  wire [1:0]  coutUInt_lo_hi_lo_hi_hi = {_compressor_17_cout, _compressor_16_cout};
  wire [2:0]  coutUInt_lo_hi_lo_hi = {coutUInt_lo_hi_lo_hi_hi, _compressor_15_cout};
  wire [5:0]  coutUInt_lo_hi_lo = {coutUInt_lo_hi_lo_hi, coutUInt_lo_hi_lo_lo};
  wire [1:0]  coutUInt_lo_hi_hi_lo_hi = {_compressor_20_cout, _compressor_19_cout};
  wire [2:0]  coutUInt_lo_hi_hi_lo = {coutUInt_lo_hi_hi_lo_hi, _compressor_18_cout};
  wire [1:0]  coutUInt_lo_hi_hi_hi_hi = {_compressor_23_cout, _compressor_22_cout};
  wire [2:0]  coutUInt_lo_hi_hi_hi = {coutUInt_lo_hi_hi_hi_hi, _compressor_21_cout};
  wire [5:0]  coutUInt_lo_hi_hi = {coutUInt_lo_hi_hi_hi, coutUInt_lo_hi_hi_lo};
  wire [11:0] coutUInt_lo_hi = {coutUInt_lo_hi_hi, coutUInt_lo_hi_lo};
  wire [23:0] coutUInt_lo = {coutUInt_lo_hi, coutUInt_lo_lo};
  wire [1:0]  coutUInt_hi_lo_lo_lo_hi = {_compressor_26_cout, _compressor_25_cout};
  wire [2:0]  coutUInt_hi_lo_lo_lo = {coutUInt_hi_lo_lo_lo_hi, _compressor_24_cout};
  wire [1:0]  coutUInt_hi_lo_lo_hi_hi = {_compressor_29_cout, _compressor_28_cout};
  wire [2:0]  coutUInt_hi_lo_lo_hi = {coutUInt_hi_lo_lo_hi_hi, _compressor_27_cout};
  wire [5:0]  coutUInt_hi_lo_lo = {coutUInt_hi_lo_lo_hi, coutUInt_hi_lo_lo_lo};
  wire [1:0]  coutUInt_hi_lo_hi_lo_hi = {_compressor_32_cout, _compressor_31_cout};
  wire [2:0]  coutUInt_hi_lo_hi_lo = {coutUInt_hi_lo_hi_lo_hi, _compressor_30_cout};
  wire [1:0]  coutUInt_hi_lo_hi_hi_hi = {_compressor_35_cout, _compressor_34_cout};
  wire [2:0]  coutUInt_hi_lo_hi_hi = {coutUInt_hi_lo_hi_hi_hi, _compressor_33_cout};
  wire [5:0]  coutUInt_hi_lo_hi = {coutUInt_hi_lo_hi_hi, coutUInt_hi_lo_hi_lo};
  wire [11:0] coutUInt_hi_lo = {coutUInt_hi_lo_hi, coutUInt_hi_lo_lo};
  wire [1:0]  coutUInt_hi_hi_lo_lo_hi = {_compressor_38_cout, _compressor_37_cout};
  wire [2:0]  coutUInt_hi_hi_lo_lo = {coutUInt_hi_hi_lo_lo_hi, _compressor_36_cout};
  wire [1:0]  coutUInt_hi_hi_lo_hi_hi = {_compressor_41_cout, _compressor_40_cout};
  wire [2:0]  coutUInt_hi_hi_lo_hi = {coutUInt_hi_hi_lo_hi_hi, _compressor_39_cout};
  wire [5:0]  coutUInt_hi_hi_lo = {coutUInt_hi_hi_lo_hi, coutUInt_hi_hi_lo_lo};
  wire [1:0]  coutUInt_hi_hi_hi_lo_hi = {_compressor_44_cout, _compressor_43_cout};
  wire [2:0]  coutUInt_hi_hi_hi_lo = {coutUInt_hi_hi_hi_lo_hi, _compressor_42_cout};
  wire [1:0]  coutUInt_hi_hi_hi_hi_hi = {_compressor_47_cout, _compressor_46_cout};
  wire [2:0]  coutUInt_hi_hi_hi_hi = {coutUInt_hi_hi_hi_hi_hi, _compressor_45_cout};
  wire [5:0]  coutUInt_hi_hi_hi = {coutUInt_hi_hi_hi_hi, coutUInt_hi_hi_hi_lo};
  wire [11:0] coutUInt_hi_hi = {coutUInt_hi_hi_hi, coutUInt_hi_hi_lo};
  wire [23:0] coutUInt_hi = {coutUInt_hi_hi, coutUInt_hi_lo};
  wire [47:0] coutUInt = {coutUInt_hi, coutUInt_lo};
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
  wire        cinVec_24 = coutUInt[23];
  wire        cinVec_25 = coutUInt[24];
  wire        cinVec_26 = coutUInt[25];
  wire        cinVec_27 = coutUInt[26];
  wire        cinVec_28 = coutUInt[27];
  wire        cinVec_29 = coutUInt[28];
  wire        cinVec_30 = coutUInt[29];
  wire        cinVec_31 = coutUInt[30];
  wire        cinVec_32 = coutUInt[31];
  wire        cinVec_33 = coutUInt[32];
  wire        cinVec_34 = coutUInt[33];
  wire        cinVec_35 = coutUInt[34];
  wire        cinVec_36 = coutUInt[35];
  wire        cinVec_37 = coutUInt[36];
  wire        cinVec_38 = coutUInt[37];
  wire        cinVec_39 = coutUInt[38];
  wire        cinVec_40 = coutUInt[39];
  wire        cinVec_41 = coutUInt[40];
  wire        cinVec_42 = coutUInt[41];
  wire        cinVec_43 = coutUInt[42];
  wire        cinVec_44 = coutUInt[43];
  wire        cinVec_45 = coutUInt[44];
  wire        cinVec_46 = coutUInt[45];
  wire        cinVec_47 = coutUInt[46];
  wire [1:0]  out_1_lo_lo_lo_lo_hi = {_compressor_2_out_1, _compressor_1_out_1};
  wire [2:0]  out_1_lo_lo_lo_lo = {out_1_lo_lo_lo_lo_hi, _compressor_0_out_1};
  wire [1:0]  out_1_lo_lo_lo_hi_hi = {_compressor_5_out_1, _compressor_4_out_1};
  wire [2:0]  out_1_lo_lo_lo_hi = {out_1_lo_lo_lo_hi_hi, _compressor_3_out_1};
  wire [5:0]  out_1_lo_lo_lo = {out_1_lo_lo_lo_hi, out_1_lo_lo_lo_lo};
  wire [1:0]  out_1_lo_lo_hi_lo_hi = {_compressor_8_out_1, _compressor_7_out_1};
  wire [2:0]  out_1_lo_lo_hi_lo = {out_1_lo_lo_hi_lo_hi, _compressor_6_out_1};
  wire [1:0]  out_1_lo_lo_hi_hi_hi = {_compressor_11_out_1, _compressor_10_out_1};
  wire [2:0]  out_1_lo_lo_hi_hi = {out_1_lo_lo_hi_hi_hi, _compressor_9_out_1};
  wire [5:0]  out_1_lo_lo_hi = {out_1_lo_lo_hi_hi, out_1_lo_lo_hi_lo};
  wire [11:0] out_1_lo_lo = {out_1_lo_lo_hi, out_1_lo_lo_lo};
  wire [1:0]  out_1_lo_hi_lo_lo_hi = {_compressor_14_out_1, _compressor_13_out_1};
  wire [2:0]  out_1_lo_hi_lo_lo = {out_1_lo_hi_lo_lo_hi, _compressor_12_out_1};
  wire [1:0]  out_1_lo_hi_lo_hi_hi = {_compressor_17_out_1, _compressor_16_out_1};
  wire [2:0]  out_1_lo_hi_lo_hi = {out_1_lo_hi_lo_hi_hi, _compressor_15_out_1};
  wire [5:0]  out_1_lo_hi_lo = {out_1_lo_hi_lo_hi, out_1_lo_hi_lo_lo};
  wire [1:0]  out_1_lo_hi_hi_lo_hi = {_compressor_20_out_1, _compressor_19_out_1};
  wire [2:0]  out_1_lo_hi_hi_lo = {out_1_lo_hi_hi_lo_hi, _compressor_18_out_1};
  wire [1:0]  out_1_lo_hi_hi_hi_hi = {_compressor_23_out_1, _compressor_22_out_1};
  wire [2:0]  out_1_lo_hi_hi_hi = {out_1_lo_hi_hi_hi_hi, _compressor_21_out_1};
  wire [5:0]  out_1_lo_hi_hi = {out_1_lo_hi_hi_hi, out_1_lo_hi_hi_lo};
  wire [11:0] out_1_lo_hi = {out_1_lo_hi_hi, out_1_lo_hi_lo};
  wire [23:0] out_1_lo = {out_1_lo_hi, out_1_lo_lo};
  wire [1:0]  out_1_hi_lo_lo_lo_hi = {_compressor_26_out_1, _compressor_25_out_1};
  wire [2:0]  out_1_hi_lo_lo_lo = {out_1_hi_lo_lo_lo_hi, _compressor_24_out_1};
  wire [1:0]  out_1_hi_lo_lo_hi_hi = {_compressor_29_out_1, _compressor_28_out_1};
  wire [2:0]  out_1_hi_lo_lo_hi = {out_1_hi_lo_lo_hi_hi, _compressor_27_out_1};
  wire [5:0]  out_1_hi_lo_lo = {out_1_hi_lo_lo_hi, out_1_hi_lo_lo_lo};
  wire [1:0]  out_1_hi_lo_hi_lo_hi = {_compressor_32_out_1, _compressor_31_out_1};
  wire [2:0]  out_1_hi_lo_hi_lo = {out_1_hi_lo_hi_lo_hi, _compressor_30_out_1};
  wire [1:0]  out_1_hi_lo_hi_hi_hi = {_compressor_35_out_1, _compressor_34_out_1};
  wire [2:0]  out_1_hi_lo_hi_hi = {out_1_hi_lo_hi_hi_hi, _compressor_33_out_1};
  wire [5:0]  out_1_hi_lo_hi = {out_1_hi_lo_hi_hi, out_1_hi_lo_hi_lo};
  wire [11:0] out_1_hi_lo = {out_1_hi_lo_hi, out_1_hi_lo_lo};
  wire [1:0]  out_1_hi_hi_lo_lo_hi = {_compressor_38_out_1, _compressor_37_out_1};
  wire [2:0]  out_1_hi_hi_lo_lo = {out_1_hi_hi_lo_lo_hi, _compressor_36_out_1};
  wire [1:0]  out_1_hi_hi_lo_hi_hi = {_compressor_41_out_1, _compressor_40_out_1};
  wire [2:0]  out_1_hi_hi_lo_hi = {out_1_hi_hi_lo_hi_hi, _compressor_39_out_1};
  wire [5:0]  out_1_hi_hi_lo = {out_1_hi_hi_lo_hi, out_1_hi_hi_lo_lo};
  wire [1:0]  out_1_hi_hi_hi_lo_hi = {_compressor_44_out_1, _compressor_43_out_1};
  wire [2:0]  out_1_hi_hi_hi_lo = {out_1_hi_hi_hi_lo_hi, _compressor_42_out_1};
  wire [1:0]  out_1_hi_hi_hi_hi_lo = {_compressor_46_out_1, _compressor_45_out_1};
  wire [1:0]  out_1_hi_hi_hi_hi_hi = {coutUInt[47], _compressor_47_out_1};
  wire [3:0]  out_1_hi_hi_hi_hi = {out_1_hi_hi_hi_hi_hi, out_1_hi_hi_hi_hi_lo};
  wire [6:0]  out_1_hi_hi_hi = {out_1_hi_hi_hi_hi, out_1_hi_hi_hi_lo};
  wire [12:0] out_1_hi_hi = {out_1_hi_hi_hi, out_1_hi_hi_lo};
  wire [24:0] out_1_hi = {out_1_hi_hi, out_1_hi_lo};
  wire [1:0]  out_0_lo_lo_lo_lo_hi = {_compressor_2_out_0, _compressor_1_out_0};
  wire [2:0]  out_0_lo_lo_lo_lo = {out_0_lo_lo_lo_lo_hi, _compressor_0_out_0};
  wire [1:0]  out_0_lo_lo_lo_hi_hi = {_compressor_5_out_0, _compressor_4_out_0};
  wire [2:0]  out_0_lo_lo_lo_hi = {out_0_lo_lo_lo_hi_hi, _compressor_3_out_0};
  wire [5:0]  out_0_lo_lo_lo = {out_0_lo_lo_lo_hi, out_0_lo_lo_lo_lo};
  wire [1:0]  out_0_lo_lo_hi_lo_hi = {_compressor_8_out_0, _compressor_7_out_0};
  wire [2:0]  out_0_lo_lo_hi_lo = {out_0_lo_lo_hi_lo_hi, _compressor_6_out_0};
  wire [1:0]  out_0_lo_lo_hi_hi_hi = {_compressor_11_out_0, _compressor_10_out_0};
  wire [2:0]  out_0_lo_lo_hi_hi = {out_0_lo_lo_hi_hi_hi, _compressor_9_out_0};
  wire [5:0]  out_0_lo_lo_hi = {out_0_lo_lo_hi_hi, out_0_lo_lo_hi_lo};
  wire [11:0] out_0_lo_lo = {out_0_lo_lo_hi, out_0_lo_lo_lo};
  wire [1:0]  out_0_lo_hi_lo_lo_hi = {_compressor_14_out_0, _compressor_13_out_0};
  wire [2:0]  out_0_lo_hi_lo_lo = {out_0_lo_hi_lo_lo_hi, _compressor_12_out_0};
  wire [1:0]  out_0_lo_hi_lo_hi_hi = {_compressor_17_out_0, _compressor_16_out_0};
  wire [2:0]  out_0_lo_hi_lo_hi = {out_0_lo_hi_lo_hi_hi, _compressor_15_out_0};
  wire [5:0]  out_0_lo_hi_lo = {out_0_lo_hi_lo_hi, out_0_lo_hi_lo_lo};
  wire [1:0]  out_0_lo_hi_hi_lo_hi = {_compressor_20_out_0, _compressor_19_out_0};
  wire [2:0]  out_0_lo_hi_hi_lo = {out_0_lo_hi_hi_lo_hi, _compressor_18_out_0};
  wire [1:0]  out_0_lo_hi_hi_hi_hi = {_compressor_23_out_0, _compressor_22_out_0};
  wire [2:0]  out_0_lo_hi_hi_hi = {out_0_lo_hi_hi_hi_hi, _compressor_21_out_0};
  wire [5:0]  out_0_lo_hi_hi = {out_0_lo_hi_hi_hi, out_0_lo_hi_hi_lo};
  wire [11:0] out_0_lo_hi = {out_0_lo_hi_hi, out_0_lo_hi_lo};
  wire [23:0] out_0_lo = {out_0_lo_hi, out_0_lo_lo};
  wire [1:0]  out_0_hi_lo_lo_lo_hi = {_compressor_26_out_0, _compressor_25_out_0};
  wire [2:0]  out_0_hi_lo_lo_lo = {out_0_hi_lo_lo_lo_hi, _compressor_24_out_0};
  wire [1:0]  out_0_hi_lo_lo_hi_hi = {_compressor_29_out_0, _compressor_28_out_0};
  wire [2:0]  out_0_hi_lo_lo_hi = {out_0_hi_lo_lo_hi_hi, _compressor_27_out_0};
  wire [5:0]  out_0_hi_lo_lo = {out_0_hi_lo_lo_hi, out_0_hi_lo_lo_lo};
  wire [1:0]  out_0_hi_lo_hi_lo_hi = {_compressor_32_out_0, _compressor_31_out_0};
  wire [2:0]  out_0_hi_lo_hi_lo = {out_0_hi_lo_hi_lo_hi, _compressor_30_out_0};
  wire [1:0]  out_0_hi_lo_hi_hi_hi = {_compressor_35_out_0, _compressor_34_out_0};
  wire [2:0]  out_0_hi_lo_hi_hi = {out_0_hi_lo_hi_hi_hi, _compressor_33_out_0};
  wire [5:0]  out_0_hi_lo_hi = {out_0_hi_lo_hi_hi, out_0_hi_lo_hi_lo};
  wire [11:0] out_0_hi_lo = {out_0_hi_lo_hi, out_0_hi_lo_lo};
  wire [1:0]  out_0_hi_hi_lo_lo_hi = {_compressor_38_out_0, _compressor_37_out_0};
  wire [2:0]  out_0_hi_hi_lo_lo = {out_0_hi_hi_lo_lo_hi, _compressor_36_out_0};
  wire [1:0]  out_0_hi_hi_lo_hi_hi = {_compressor_41_out_0, _compressor_40_out_0};
  wire [2:0]  out_0_hi_hi_lo_hi = {out_0_hi_hi_lo_hi_hi, _compressor_39_out_0};
  wire [5:0]  out_0_hi_hi_lo = {out_0_hi_hi_lo_hi, out_0_hi_hi_lo_lo};
  wire [1:0]  out_0_hi_hi_hi_lo_hi = {_compressor_44_out_0, _compressor_43_out_0};
  wire [2:0]  out_0_hi_hi_hi_lo = {out_0_hi_hi_hi_lo_hi, _compressor_42_out_0};
  wire [1:0]  out_0_hi_hi_hi_hi_hi = {_compressor_47_out_0, _compressor_46_out_0};
  wire [2:0]  out_0_hi_hi_hi_hi = {out_0_hi_hi_hi_hi_hi, _compressor_45_out_0};
  wire [5:0]  out_0_hi_hi_hi = {out_0_hi_hi_hi_hi, out_0_hi_hi_hi_lo};
  wire [11:0] out_0_hi_hi = {out_0_hi_hi_hi, out_0_hi_hi_lo};
  wire [23:0] out_0_hi = {out_0_hi_hi, out_0_hi_lo};
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
  CSACompressor4_2 compressor_24 (
    .in_0  (in_0[24]),
    .in_1  (in_1[24]),
    .in_2  (in_2[24]),
    .in_3  (in_3[24]),
    .cin   (cinVec_24),
    .out_0 (_compressor_24_out_0),
    .out_1 (_compressor_24_out_1),
    .cout  (_compressor_24_cout)
  );
  CSACompressor4_2 compressor_25 (
    .in_0  (in_0[25]),
    .in_1  (in_1[25]),
    .in_2  (in_2[25]),
    .in_3  (in_3[25]),
    .cin   (cinVec_25),
    .out_0 (_compressor_25_out_0),
    .out_1 (_compressor_25_out_1),
    .cout  (_compressor_25_cout)
  );
  CSACompressor4_2 compressor_26 (
    .in_0  (in_0[26]),
    .in_1  (in_1[26]),
    .in_2  (in_2[26]),
    .in_3  (in_3[26]),
    .cin   (cinVec_26),
    .out_0 (_compressor_26_out_0),
    .out_1 (_compressor_26_out_1),
    .cout  (_compressor_26_cout)
  );
  CSACompressor4_2 compressor_27 (
    .in_0  (in_0[27]),
    .in_1  (in_1[27]),
    .in_2  (in_2[27]),
    .in_3  (in_3[27]),
    .cin   (cinVec_27),
    .out_0 (_compressor_27_out_0),
    .out_1 (_compressor_27_out_1),
    .cout  (_compressor_27_cout)
  );
  CSACompressor4_2 compressor_28 (
    .in_0  (in_0[28]),
    .in_1  (in_1[28]),
    .in_2  (in_2[28]),
    .in_3  (in_3[28]),
    .cin   (cinVec_28),
    .out_0 (_compressor_28_out_0),
    .out_1 (_compressor_28_out_1),
    .cout  (_compressor_28_cout)
  );
  CSACompressor4_2 compressor_29 (
    .in_0  (in_0[29]),
    .in_1  (in_1[29]),
    .in_2  (in_2[29]),
    .in_3  (in_3[29]),
    .cin   (cinVec_29),
    .out_0 (_compressor_29_out_0),
    .out_1 (_compressor_29_out_1),
    .cout  (_compressor_29_cout)
  );
  CSACompressor4_2 compressor_30 (
    .in_0  (in_0[30]),
    .in_1  (in_1[30]),
    .in_2  (in_2[30]),
    .in_3  (in_3[30]),
    .cin   (cinVec_30),
    .out_0 (_compressor_30_out_0),
    .out_1 (_compressor_30_out_1),
    .cout  (_compressor_30_cout)
  );
  CSACompressor4_2 compressor_31 (
    .in_0  (in_0[31]),
    .in_1  (in_1[31]),
    .in_2  (in_2[31]),
    .in_3  (in_3[31]),
    .cin   (cinVec_31),
    .out_0 (_compressor_31_out_0),
    .out_1 (_compressor_31_out_1),
    .cout  (_compressor_31_cout)
  );
  CSACompressor4_2 compressor_32 (
    .in_0  (in_0[32]),
    .in_1  (in_1[32]),
    .in_2  (in_2[32]),
    .in_3  (in_3[32]),
    .cin   (cinVec_32),
    .out_0 (_compressor_32_out_0),
    .out_1 (_compressor_32_out_1),
    .cout  (_compressor_32_cout)
  );
  CSACompressor4_2 compressor_33 (
    .in_0  (in_0[33]),
    .in_1  (in_1[33]),
    .in_2  (in_2[33]),
    .in_3  (in_3[33]),
    .cin   (cinVec_33),
    .out_0 (_compressor_33_out_0),
    .out_1 (_compressor_33_out_1),
    .cout  (_compressor_33_cout)
  );
  CSACompressor4_2 compressor_34 (
    .in_0  (in_0[34]),
    .in_1  (in_1[34]),
    .in_2  (in_2[34]),
    .in_3  (in_3[34]),
    .cin   (cinVec_34),
    .out_0 (_compressor_34_out_0),
    .out_1 (_compressor_34_out_1),
    .cout  (_compressor_34_cout)
  );
  CSACompressor4_2 compressor_35 (
    .in_0  (in_0[35]),
    .in_1  (in_1[35]),
    .in_2  (in_2[35]),
    .in_3  (in_3[35]),
    .cin   (cinVec_35),
    .out_0 (_compressor_35_out_0),
    .out_1 (_compressor_35_out_1),
    .cout  (_compressor_35_cout)
  );
  CSACompressor4_2 compressor_36 (
    .in_0  (in_0[36]),
    .in_1  (in_1[36]),
    .in_2  (in_2[36]),
    .in_3  (in_3[36]),
    .cin   (cinVec_36),
    .out_0 (_compressor_36_out_0),
    .out_1 (_compressor_36_out_1),
    .cout  (_compressor_36_cout)
  );
  CSACompressor4_2 compressor_37 (
    .in_0  (in_0[37]),
    .in_1  (in_1[37]),
    .in_2  (in_2[37]),
    .in_3  (in_3[37]),
    .cin   (cinVec_37),
    .out_0 (_compressor_37_out_0),
    .out_1 (_compressor_37_out_1),
    .cout  (_compressor_37_cout)
  );
  CSACompressor4_2 compressor_38 (
    .in_0  (in_0[38]),
    .in_1  (in_1[38]),
    .in_2  (in_2[38]),
    .in_3  (in_3[38]),
    .cin   (cinVec_38),
    .out_0 (_compressor_38_out_0),
    .out_1 (_compressor_38_out_1),
    .cout  (_compressor_38_cout)
  );
  CSACompressor4_2 compressor_39 (
    .in_0  (in_0[39]),
    .in_1  (in_1[39]),
    .in_2  (in_2[39]),
    .in_3  (in_3[39]),
    .cin   (cinVec_39),
    .out_0 (_compressor_39_out_0),
    .out_1 (_compressor_39_out_1),
    .cout  (_compressor_39_cout)
  );
  CSACompressor4_2 compressor_40 (
    .in_0  (in_0[40]),
    .in_1  (in_1[40]),
    .in_2  (in_2[40]),
    .in_3  (in_3[40]),
    .cin   (cinVec_40),
    .out_0 (_compressor_40_out_0),
    .out_1 (_compressor_40_out_1),
    .cout  (_compressor_40_cout)
  );
  CSACompressor4_2 compressor_41 (
    .in_0  (in_0[41]),
    .in_1  (in_1[41]),
    .in_2  (in_2[41]),
    .in_3  (in_3[41]),
    .cin   (cinVec_41),
    .out_0 (_compressor_41_out_0),
    .out_1 (_compressor_41_out_1),
    .cout  (_compressor_41_cout)
  );
  CSACompressor4_2 compressor_42 (
    .in_0  (in_0[42]),
    .in_1  (in_1[42]),
    .in_2  (in_2[42]),
    .in_3  (in_3[42]),
    .cin   (cinVec_42),
    .out_0 (_compressor_42_out_0),
    .out_1 (_compressor_42_out_1),
    .cout  (_compressor_42_cout)
  );
  CSACompressor4_2 compressor_43 (
    .in_0  (in_0[43]),
    .in_1  (in_1[43]),
    .in_2  (in_2[43]),
    .in_3  (in_3[43]),
    .cin   (cinVec_43),
    .out_0 (_compressor_43_out_0),
    .out_1 (_compressor_43_out_1),
    .cout  (_compressor_43_cout)
  );
  CSACompressor4_2 compressor_44 (
    .in_0  (in_0[44]),
    .in_1  (in_1[44]),
    .in_2  (in_2[44]),
    .in_3  (in_3[44]),
    .cin   (cinVec_44),
    .out_0 (_compressor_44_out_0),
    .out_1 (_compressor_44_out_1),
    .cout  (_compressor_44_cout)
  );
  CSACompressor4_2 compressor_45 (
    .in_0  (in_0[45]),
    .in_1  (in_1[45]),
    .in_2  (in_2[45]),
    .in_3  (in_3[45]),
    .cin   (cinVec_45),
    .out_0 (_compressor_45_out_0),
    .out_1 (_compressor_45_out_1),
    .cout  (_compressor_45_cout)
  );
  CSACompressor4_2 compressor_46 (
    .in_0  (in_0[46]),
    .in_1  (in_1[46]),
    .in_2  (in_2[46]),
    .in_3  (in_3[46]),
    .cin   (cinVec_46),
    .out_0 (_compressor_46_out_0),
    .out_1 (_compressor_46_out_1),
    .cout  (_compressor_46_cout)
  );
  CSACompressor4_2 compressor_47 (
    .in_0  (in_0[47]),
    .in_1  (in_1[47]),
    .in_2  (in_2[47]),
    .in_3  (in_3[47]),
    .cin   (cinVec_47),
    .out_0 (_compressor_47_out_0),
    .out_1 (_compressor_47_out_1),
    .cout  (_compressor_47_cout)
  );
  assign out_0 = {1'h0, out_0_hi, out_0_lo};
  assign out_1 = {out_1_hi, out_1_lo};
endmodule

