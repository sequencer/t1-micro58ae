module VectorMultiplier32Unsigned(
  input  [31:0] a,
                b,
  input  [2:0]  sew,
  output [63:0] multiplierSum,
                multiplierCarry
);

  wire [64:0] _mergeSplit32_layer1_csa42_out_0;
  wire [64:0] _mergeSplit32_layer1_csa42_out_1;
  wire [48:0] _mergeSplit32_layer01_csa42_out_0;
  wire [48:0] _mergeSplit32_layer01_csa42_out_1;
  wire [48:0] _mergeSplit32_layer00_csa42_out_0;
  wire [48:0] _mergeSplit32_layer00_csa42_out_1;
  wire [31:0] _ax11_mul16_outCarry;
  wire [31:0] _ax11_mul16_outSum;
  wire [31:0] _ax00_mul16_outCarry;
  wire [31:0] _ax00_mul16_outSum;
  wire [32:0] _ax10_layer2_csa42_out_0;
  wire [32:0] _ax10_layer2_csa42_out_1;
  wire [24:0] _ax10_layer11_csa42_out_0;
  wire [24:0] _ax10_layer11_csa42_out_1;
  wire [24:0] _ax10_layer10_csa42_out_0;
  wire [24:0] _ax10_layer10_csa42_out_1;
  wire [20:0] _ax10_layer03_csa42_out_0;
  wire [20:0] _ax10_layer03_csa42_out_1;
  wire [20:0] _ax10_layer02_csa42_out_0;
  wire [20:0] _ax10_layer02_csa42_out_1;
  wire [20:0] _ax10_layer01_csa42_out_0;
  wire [20:0] _ax10_layer01_csa42_out_1;
  wire [20:0] _ax10_layer00_csa42_out_0;
  wire [20:0] _ax10_layer00_csa42_out_1;
  wire [32:0] _ax01_layer2_csa42_out_0;
  wire [32:0] _ax01_layer2_csa42_out_1;
  wire [24:0] _ax01_layer11_csa42_out_0;
  wire [24:0] _ax01_layer11_csa42_out_1;
  wire [24:0] _ax01_layer10_csa42_out_0;
  wire [24:0] _ax01_layer10_csa42_out_1;
  wire [20:0] _ax01_layer03_csa42_out_0;
  wire [20:0] _ax01_layer03_csa42_out_1;
  wire [20:0] _ax01_layer02_csa42_out_0;
  wire [20:0] _ax01_layer02_csa42_out_1;
  wire [20:0] _ax01_layer01_csa42_out_0;
  wire [20:0] _ax01_layer01_csa42_out_1;
  wire [20:0] _ax01_layer00_csa42_out_0;
  wire [20:0] _ax01_layer00_csa42_out_1;
  wire        a0Vec_0 = a[0];
  wire        a0Vec_1 = a[1];
  wire        a0Vec_2 = a[2];
  wire        a0Vec_3 = a[3];
  wire        a0Vec_4 = a[4];
  wire        a0Vec_5 = a[5];
  wire        a0Vec_6 = a[6];
  wire        a0Vec_7 = a[7];
  wire        a0Vec_8 = a[8];
  wire        a0Vec_9 = a[9];
  wire        a0Vec_10 = a[10];
  wire        a0Vec_11 = a[11];
  wire        a0Vec_12 = a[12];
  wire        a0Vec_13 = a[13];
  wire        a0Vec_14 = a[14];
  wire        a0Vec_15 = a[15];
  wire        a1Vec_0 = a[16];
  wire        a1Vec_1 = a[17];
  wire        a1Vec_2 = a[18];
  wire        a1Vec_3 = a[19];
  wire        a1Vec_4 = a[20];
  wire        a1Vec_5 = a[21];
  wire        a1Vec_6 = a[22];
  wire        a1Vec_7 = a[23];
  wire        a1Vec_8 = a[24];
  wire        a1Vec_9 = a[25];
  wire        a1Vec_10 = a[26];
  wire        a1Vec_11 = a[27];
  wire        a1Vec_12 = a[28];
  wire        a1Vec_13 = a[29];
  wire        a1Vec_14 = a[30];
  wire        a1Vec_15 = a[31];
  wire [1:0]  sewFor16 = sew[0] ? 2'h1 : 2'h2;
  wire [15:0] a1x0_0 = a1Vec_0 ? b[15:0] : 16'h0;
  wire [15:0] a1x0_exist = a1Vec_1 ? b[15:0] : 16'h0;
  wire [16:0] a1x0_1 = {a1x0_exist, 1'h0};
  wire [15:0] a1x0_exist_1 = a1Vec_2 ? b[15:0] : 16'h0;
  wire [17:0] a1x0_2 = {a1x0_exist_1, 2'h0};
  wire [15:0] a1x0_exist_2 = a1Vec_3 ? b[15:0] : 16'h0;
  wire [18:0] a1x0_3 = {a1x0_exist_2, 3'h0};
  wire [15:0] a1x0_exist_3 = a1Vec_4 ? b[15:0] : 16'h0;
  wire [19:0] a1x0_4 = {a1x0_exist_3, 4'h0};
  wire [15:0] a1x0_exist_4 = a1Vec_5 ? b[15:0] : 16'h0;
  wire [20:0] a1x0_5 = {a1x0_exist_4, 5'h0};
  wire [15:0] a1x0_exist_5 = a1Vec_6 ? b[15:0] : 16'h0;
  wire [21:0] a1x0_6 = {a1x0_exist_5, 6'h0};
  wire [15:0] a1x0_exist_6 = a1Vec_7 ? b[15:0] : 16'h0;
  wire [22:0] a1x0_7 = {a1x0_exist_6, 7'h0};
  wire [15:0] a1x0_exist_7 = a1Vec_8 ? b[15:0] : 16'h0;
  wire [23:0] a1x0_8 = {a1x0_exist_7, 8'h0};
  wire [15:0] a1x0_exist_8 = a1Vec_9 ? b[15:0] : 16'h0;
  wire [24:0] a1x0_9 = {a1x0_exist_8, 9'h0};
  wire [15:0] a1x0_exist_9 = a1Vec_10 ? b[15:0] : 16'h0;
  wire [25:0] a1x0_10 = {a1x0_exist_9, 10'h0};
  wire [15:0] a1x0_exist_10 = a1Vec_11 ? b[15:0] : 16'h0;
  wire [26:0] a1x0_11 = {a1x0_exist_10, 11'h0};
  wire [15:0] a1x0_exist_11 = a1Vec_12 ? b[15:0] : 16'h0;
  wire [27:0] a1x0_12 = {a1x0_exist_11, 12'h0};
  wire [15:0] a1x0_exist_12 = a1Vec_13 ? b[15:0] : 16'h0;
  wire [28:0] a1x0_13 = {a1x0_exist_12, 13'h0};
  wire [15:0] a1x0_exist_13 = a1Vec_14 ? b[15:0] : 16'h0;
  wire [29:0] a1x0_14 = {a1x0_exist_13, 14'h0};
  wire [15:0] a1x0_exist_14 = a1Vec_15 ? b[15:0] : 16'h0;
  wire [30:0] a1x0_15 = {a1x0_exist_14, 15'h0};
  wire [15:0] a0x1_0 = a0Vec_0 ? b[31:16] : 16'h0;
  wire [15:0] a0x1_exist = a0Vec_1 ? b[31:16] : 16'h0;
  wire [16:0] a0x1_1 = {a0x1_exist, 1'h0};
  wire [15:0] a0x1_exist_1 = a0Vec_2 ? b[31:16] : 16'h0;
  wire [17:0] a0x1_2 = {a0x1_exist_1, 2'h0};
  wire [15:0] a0x1_exist_2 = a0Vec_3 ? b[31:16] : 16'h0;
  wire [18:0] a0x1_3 = {a0x1_exist_2, 3'h0};
  wire [15:0] a0x1_exist_3 = a0Vec_4 ? b[31:16] : 16'h0;
  wire [19:0] a0x1_4 = {a0x1_exist_3, 4'h0};
  wire [15:0] a0x1_exist_4 = a0Vec_5 ? b[31:16] : 16'h0;
  wire [20:0] a0x1_5 = {a0x1_exist_4, 5'h0};
  wire [15:0] a0x1_exist_5 = a0Vec_6 ? b[31:16] : 16'h0;
  wire [21:0] a0x1_6 = {a0x1_exist_5, 6'h0};
  wire [15:0] a0x1_exist_6 = a0Vec_7 ? b[31:16] : 16'h0;
  wire [22:0] a0x1_7 = {a0x1_exist_6, 7'h0};
  wire [15:0] a0x1_exist_7 = a0Vec_8 ? b[31:16] : 16'h0;
  wire [23:0] a0x1_8 = {a0x1_exist_7, 8'h0};
  wire [15:0] a0x1_exist_8 = a0Vec_9 ? b[31:16] : 16'h0;
  wire [24:0] a0x1_9 = {a0x1_exist_8, 9'h0};
  wire [15:0] a0x1_exist_9 = a0Vec_10 ? b[31:16] : 16'h0;
  wire [25:0] a0x1_10 = {a0x1_exist_9, 10'h0};
  wire [15:0] a0x1_exist_10 = a0Vec_11 ? b[31:16] : 16'h0;
  wire [26:0] a0x1_11 = {a0x1_exist_10, 11'h0};
  wire [15:0] a0x1_exist_11 = a0Vec_12 ? b[31:16] : 16'h0;
  wire [27:0] a0x1_12 = {a0x1_exist_11, 12'h0};
  wire [15:0] a0x1_exist_12 = a0Vec_13 ? b[31:16] : 16'h0;
  wire [28:0] a0x1_13 = {a0x1_exist_12, 13'h0};
  wire [15:0] a0x1_exist_13 = a0Vec_14 ? b[31:16] : 16'h0;
  wire [29:0] a0x1_14 = {a0x1_exist_13, 14'h0};
  wire [15:0] a0x1_exist_14 = a0Vec_15 ? b[31:16] : 16'h0;
  wire [30:0] a0x1_15 = {a0x1_exist_14, 15'h0};
  wire [31:0] ax01_0 = {_ax01_layer2_csa42_out_0[30:0], 1'h0};
  wire [31:0] ax01_1 = _ax01_layer2_csa42_out_1[31:0];
  wire [31:0] ax10_0 = {_ax10_layer2_csa42_out_0[30:0], 1'h0};
  wire [31:0] ax10_1 = _ax10_layer2_csa42_out_1[31:0];
  wire [63:0] mergeSplit32_1 = {_mergeSplit32_layer1_csa42_out_0[62:0], 1'h0};
  wire [63:0] mergeSplit32_2 = _mergeSplit32_layer1_csa42_out_1[63:0];
  wire [63:0] sumForAdder = sew[2] ? mergeSplit32_1 : {_ax11_mul16_outCarry, _ax00_mul16_outCarry};
  wire [63:0] carryForAdder = sew[2] ? mergeSplit32_2 : {_ax11_mul16_outSum, _ax00_mul16_outSum};
  CSA42 ax01_layer00_csa42 (
    .in_0  ({4'h0, a0x1_0}),
    .in_1  ({3'h0, a0x1_1}),
    .in_2  ({2'h0, a0x1_2}),
    .in_3  ({1'h0, a0x1_3}),
    .out_0 (_ax01_layer00_csa42_out_0),
    .out_1 (_ax01_layer00_csa42_out_1)
  );
  CSA42 ax01_layer01_csa42 (
    .in_0  ({4'h0, a0x1_4[19:4]}),
    .in_1  ({3'h0, a0x1_5[20:4]}),
    .in_2  ({2'h0, a0x1_6[21:4]}),
    .in_3  ({1'h0, a0x1_7[22:4]}),
    .out_0 (_ax01_layer01_csa42_out_0),
    .out_1 (_ax01_layer01_csa42_out_1)
  );
  CSA42 ax01_layer02_csa42 (
    .in_0  ({4'h0, a0x1_8[23:8]}),
    .in_1  ({3'h0, a0x1_9[24:8]}),
    .in_2  ({2'h0, a0x1_10[25:8]}),
    .in_3  ({1'h0, a0x1_11[26:8]}),
    .out_0 (_ax01_layer02_csa42_out_0),
    .out_1 (_ax01_layer02_csa42_out_1)
  );
  CSA42 ax01_layer03_csa42 (
    .in_0  ({4'h0, a0x1_12[27:12]}),
    .in_1  ({3'h0, a0x1_13[28:12]}),
    .in_2  ({2'h0, a0x1_14[29:12]}),
    .in_3  ({1'h0, a0x1_15[30:12]}),
    .out_0 (_ax01_layer03_csa42_out_0),
    .out_1 (_ax01_layer03_csa42_out_1)
  );
  CSA42_1 ax01_layer10_csa42 (
    .in_0  ({2'h0, _ax01_layer00_csa42_out_0, 1'h0}),
    .in_1  ({3'h0, _ax01_layer00_csa42_out_1}),
    .in_2  ({_ax01_layer01_csa42_out_0[18:0], 5'h0}),
    .in_3  ({_ax01_layer01_csa42_out_1[19:0], 4'h0}),
    .out_0 (_ax01_layer10_csa42_out_0),
    .out_1 (_ax01_layer10_csa42_out_1)
  );
  CSA42_1 ax01_layer11_csa42 (
    .in_0  ({2'h0, _ax01_layer02_csa42_out_0, 1'h0}),
    .in_1  ({3'h0, _ax01_layer02_csa42_out_1}),
    .in_2  ({_ax01_layer03_csa42_out_0[18:0], 5'h0}),
    .in_3  ({_ax01_layer03_csa42_out_1[19:0], 4'h0}),
    .out_0 (_ax01_layer11_csa42_out_0),
    .out_1 (_ax01_layer11_csa42_out_1)
  );
  CSA42_2 ax01_layer2_csa42 (
    .in_0  ({6'h0, _ax01_layer10_csa42_out_0, 1'h0}),
    .in_1  ({7'h0, _ax01_layer10_csa42_out_1}),
    .in_2  ({_ax01_layer11_csa42_out_0[22:0], 9'h0}),
    .in_3  ({_ax01_layer11_csa42_out_1[23:0], 8'h0}),
    .out_0 (_ax01_layer2_csa42_out_0),
    .out_1 (_ax01_layer2_csa42_out_1)
  );
  CSA42 ax10_layer00_csa42 (
    .in_0  ({4'h0, a1x0_0}),
    .in_1  ({3'h0, a1x0_1}),
    .in_2  ({2'h0, a1x0_2}),
    .in_3  ({1'h0, a1x0_3}),
    .out_0 (_ax10_layer00_csa42_out_0),
    .out_1 (_ax10_layer00_csa42_out_1)
  );
  CSA42 ax10_layer01_csa42 (
    .in_0  ({4'h0, a1x0_4[19:4]}),
    .in_1  ({3'h0, a1x0_5[20:4]}),
    .in_2  ({2'h0, a1x0_6[21:4]}),
    .in_3  ({1'h0, a1x0_7[22:4]}),
    .out_0 (_ax10_layer01_csa42_out_0),
    .out_1 (_ax10_layer01_csa42_out_1)
  );
  CSA42 ax10_layer02_csa42 (
    .in_0  ({4'h0, a1x0_8[23:8]}),
    .in_1  ({3'h0, a1x0_9[24:8]}),
    .in_2  ({2'h0, a1x0_10[25:8]}),
    .in_3  ({1'h0, a1x0_11[26:8]}),
    .out_0 (_ax10_layer02_csa42_out_0),
    .out_1 (_ax10_layer02_csa42_out_1)
  );
  CSA42 ax10_layer03_csa42 (
    .in_0  ({4'h0, a1x0_12[27:12]}),
    .in_1  ({3'h0, a1x0_13[28:12]}),
    .in_2  ({2'h0, a1x0_14[29:12]}),
    .in_3  ({1'h0, a1x0_15[30:12]}),
    .out_0 (_ax10_layer03_csa42_out_0),
    .out_1 (_ax10_layer03_csa42_out_1)
  );
  CSA42_1 ax10_layer10_csa42 (
    .in_0  ({2'h0, _ax10_layer00_csa42_out_0, 1'h0}),
    .in_1  ({3'h0, _ax10_layer00_csa42_out_1}),
    .in_2  ({_ax10_layer01_csa42_out_0[18:0], 5'h0}),
    .in_3  ({_ax10_layer01_csa42_out_1[19:0], 4'h0}),
    .out_0 (_ax10_layer10_csa42_out_0),
    .out_1 (_ax10_layer10_csa42_out_1)
  );
  CSA42_1 ax10_layer11_csa42 (
    .in_0  ({2'h0, _ax10_layer02_csa42_out_0, 1'h0}),
    .in_1  ({3'h0, _ax10_layer02_csa42_out_1}),
    .in_2  ({_ax10_layer03_csa42_out_0[18:0], 5'h0}),
    .in_3  ({_ax10_layer03_csa42_out_1[19:0], 4'h0}),
    .out_0 (_ax10_layer11_csa42_out_0),
    .out_1 (_ax10_layer11_csa42_out_1)
  );
  CSA42_2 ax10_layer2_csa42 (
    .in_0  ({6'h0, _ax10_layer10_csa42_out_0, 1'h0}),
    .in_1  ({7'h0, _ax10_layer10_csa42_out_1}),
    .in_2  ({_ax10_layer11_csa42_out_0[22:0], 9'h0}),
    .in_3  ({_ax10_layer11_csa42_out_1[23:0], 8'h0}),
    .out_0 (_ax10_layer2_csa42_out_0),
    .out_1 (_ax10_layer2_csa42_out_1)
  );
  Multiplier16 ax00_mul16 (
    .a        (a[15:0]),
    .b        (b[15:0]),
    .sew      (sewFor16),
    .outCarry (_ax00_mul16_outCarry),
    .outSum   (_ax00_mul16_outSum)
  );
  Multiplier16 ax11_mul16 (
    .a        (a[31:16]),
    .b        (b[31:16]),
    .sew      (sewFor16),
    .outCarry (_ax11_mul16_outCarry),
    .outSum   (_ax11_mul16_outSum)
  );
  CSA42_5 mergeSplit32_layer00_csa42 (
    .in_0  ({16'h0, _ax00_mul16_outCarry}),
    .in_1  ({16'h0, _ax00_mul16_outSum}),
    .in_2  ({ax01_0, 16'h0}),
    .in_3  ({ax01_1, 16'h0}),
    .out_0 (_mergeSplit32_layer00_csa42_out_0),
    .out_1 (_mergeSplit32_layer00_csa42_out_1)
  );
  CSA42_5 mergeSplit32_layer01_csa42 (
    .in_0  ({16'h0, ax10_0}),
    .in_1  ({16'h0, ax10_1}),
    .in_2  ({_ax11_mul16_outCarry, 16'h0}),
    .in_3  ({_ax11_mul16_outSum, 16'h0}),
    .out_0 (_mergeSplit32_layer01_csa42_out_0),
    .out_1 (_mergeSplit32_layer01_csa42_out_1)
  );
  CSA42_6 mergeSplit32_layer1_csa42 (
    .in_0  ({14'h0, _mergeSplit32_layer00_csa42_out_0, 1'h0}),
    .in_1  ({15'h0, _mergeSplit32_layer00_csa42_out_1}),
    .in_2  ({_mergeSplit32_layer01_csa42_out_0[46:0], 17'h0}),
    .in_3  ({_mergeSplit32_layer01_csa42_out_1[47:0], 16'h0}),
    .out_0 (_mergeSplit32_layer1_csa42_out_0),
    .out_1 (_mergeSplit32_layer1_csa42_out_1)
  );
  VectorAdder64 result_adder64 (
    .a   (sumForAdder),
    .b   (carryForAdder),
    .z   (/* unused */),
    .sew ({sew, 1'h0})
  );
  assign multiplierSum = sumForAdder;
  assign multiplierCarry = carryForAdder;
endmodule

