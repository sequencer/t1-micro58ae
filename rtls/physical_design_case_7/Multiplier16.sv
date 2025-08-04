module Multiplier16(
  input  [15:0] a,
                b,
  input  [1:0]  sew,
  output [31:0] outCarry,
                outSum
);

  wire [32:0] _result16_layer1_csa42_out_0;
  wire [32:0] _result16_layer1_csa42_out_1;
  wire [24:0] _result16_layer01_csa42_out_0;
  wire [24:0] _result16_layer01_csa42_out_1;
  wire [24:0] _result16_layer00_csa42_out_0;
  wire [24:0] _result16_layer00_csa42_out_1;
  wire [16:0] _ax11_layerOut_csa42_out_0;
  wire [16:0] _ax11_layerOut_csa42_out_1;
  wire [12:0] _ax11_layer1_csa42_out_0;
  wire [12:0] _ax11_layer1_csa42_out_1;
  wire [12:0] _ax11_layer0_csa42_out_0;
  wire [12:0] _ax11_layer0_csa42_out_1;
  wire [16:0] _ax01_layerOut_csa42_out_0;
  wire [16:0] _ax01_layerOut_csa42_out_1;
  wire [12:0] _ax01_layer1_csa42_out_0;
  wire [12:0] _ax01_layer1_csa42_out_1;
  wire [12:0] _ax01_layer0_csa42_out_0;
  wire [12:0] _ax01_layer0_csa42_out_1;
  wire [16:0] _ax10_layerOut_csa42_out_0;
  wire [16:0] _ax10_layerOut_csa42_out_1;
  wire [12:0] _ax10_layer1_csa42_out_0;
  wire [12:0] _ax10_layer1_csa42_out_1;
  wire [12:0] _ax10_layer0_csa42_out_0;
  wire [12:0] _ax10_layer0_csa42_out_1;
  wire [16:0] _ax00_layerOut_csa42_out_0;
  wire [16:0] _ax00_layerOut_csa42_out_1;
  wire [12:0] _ax00_layer1_csa42_out_0;
  wire [12:0] _ax00_layer1_csa42_out_1;
  wire [12:0] _ax00_layer0_csa42_out_0;
  wire [12:0] _ax00_layer0_csa42_out_1;
  wire        a0Vec_0 = a[0];
  wire        a0Vec_1 = a[1];
  wire        a0Vec_2 = a[2];
  wire        a0Vec_3 = a[3];
  wire        a0Vec_4 = a[4];
  wire        a0Vec_5 = a[5];
  wire        a0Vec_6 = a[6];
  wire        a0Vec_7 = a[7];
  wire        a1Vec_0 = a[8];
  wire        a1Vec_1 = a[9];
  wire        a1Vec_2 = a[10];
  wire        a1Vec_3 = a[11];
  wire        a1Vec_4 = a[12];
  wire        a1Vec_5 = a[13];
  wire        a1Vec_6 = a[14];
  wire        a1Vec_7 = a[15];
  wire [7:0]  a0x0_0 = a0Vec_0 ? b[7:0] : 8'h0;
  wire [7:0]  a0x0_exist = a0Vec_1 ? b[7:0] : 8'h0;
  wire [8:0]  a0x0_1 = {a0x0_exist, 1'h0};
  wire [7:0]  a0x0_exist_1 = a0Vec_2 ? b[7:0] : 8'h0;
  wire [9:0]  a0x0_2 = {a0x0_exist_1, 2'h0};
  wire [7:0]  a0x0_exist_2 = a0Vec_3 ? b[7:0] : 8'h0;
  wire [10:0] a0x0_3 = {a0x0_exist_2, 3'h0};
  wire [7:0]  a0x0_exist_3 = a0Vec_4 ? b[7:0] : 8'h0;
  wire [11:0] a0x0_4 = {a0x0_exist_3, 4'h0};
  wire [7:0]  a0x0_exist_4 = a0Vec_5 ? b[7:0] : 8'h0;
  wire [12:0] a0x0_5 = {a0x0_exist_4, 5'h0};
  wire [7:0]  a0x0_exist_5 = a0Vec_6 ? b[7:0] : 8'h0;
  wire [13:0] a0x0_6 = {a0x0_exist_5, 6'h0};
  wire [7:0]  a0x0_exist_6 = a0Vec_7 ? b[7:0] : 8'h0;
  wire [14:0] a0x0_7 = {a0x0_exist_6, 7'h0};
  wire [7:0]  a1x0_0 = a1Vec_0 ? b[7:0] : 8'h0;
  wire [7:0]  a1x0_exist = a1Vec_1 ? b[7:0] : 8'h0;
  wire [8:0]  a1x0_1 = {a1x0_exist, 1'h0};
  wire [7:0]  a1x0_exist_1 = a1Vec_2 ? b[7:0] : 8'h0;
  wire [9:0]  a1x0_2 = {a1x0_exist_1, 2'h0};
  wire [7:0]  a1x0_exist_2 = a1Vec_3 ? b[7:0] : 8'h0;
  wire [10:0] a1x0_3 = {a1x0_exist_2, 3'h0};
  wire [7:0]  a1x0_exist_3 = a1Vec_4 ? b[7:0] : 8'h0;
  wire [11:0] a1x0_4 = {a1x0_exist_3, 4'h0};
  wire [7:0]  a1x0_exist_4 = a1Vec_5 ? b[7:0] : 8'h0;
  wire [12:0] a1x0_5 = {a1x0_exist_4, 5'h0};
  wire [7:0]  a1x0_exist_5 = a1Vec_6 ? b[7:0] : 8'h0;
  wire [13:0] a1x0_6 = {a1x0_exist_5, 6'h0};
  wire [7:0]  a1x0_exist_6 = a1Vec_7 ? b[7:0] : 8'h0;
  wire [14:0] a1x0_7 = {a1x0_exist_6, 7'h0};
  wire [7:0]  a0x1_0 = a0Vec_0 ? b[15:8] : 8'h0;
  wire [7:0]  a0x1_exist = a0Vec_1 ? b[15:8] : 8'h0;
  wire [8:0]  a0x1_1 = {a0x1_exist, 1'h0};
  wire [7:0]  a0x1_exist_1 = a0Vec_2 ? b[15:8] : 8'h0;
  wire [9:0]  a0x1_2 = {a0x1_exist_1, 2'h0};
  wire [7:0]  a0x1_exist_2 = a0Vec_3 ? b[15:8] : 8'h0;
  wire [10:0] a0x1_3 = {a0x1_exist_2, 3'h0};
  wire [7:0]  a0x1_exist_3 = a0Vec_4 ? b[15:8] : 8'h0;
  wire [11:0] a0x1_4 = {a0x1_exist_3, 4'h0};
  wire [7:0]  a0x1_exist_4 = a0Vec_5 ? b[15:8] : 8'h0;
  wire [12:0] a0x1_5 = {a0x1_exist_4, 5'h0};
  wire [7:0]  a0x1_exist_5 = a0Vec_6 ? b[15:8] : 8'h0;
  wire [13:0] a0x1_6 = {a0x1_exist_5, 6'h0};
  wire [7:0]  a0x1_exist_6 = a0Vec_7 ? b[15:8] : 8'h0;
  wire [14:0] a0x1_7 = {a0x1_exist_6, 7'h0};
  wire [7:0]  a1x1_0 = a1Vec_0 ? b[15:8] : 8'h0;
  wire [7:0]  a1x1_exist = a1Vec_1 ? b[15:8] : 8'h0;
  wire [8:0]  a1x1_1 = {a1x1_exist, 1'h0};
  wire [7:0]  a1x1_exist_1 = a1Vec_2 ? b[15:8] : 8'h0;
  wire [9:0]  a1x1_2 = {a1x1_exist_1, 2'h0};
  wire [7:0]  a1x1_exist_2 = a1Vec_3 ? b[15:8] : 8'h0;
  wire [10:0] a1x1_3 = {a1x1_exist_2, 3'h0};
  wire [7:0]  a1x1_exist_3 = a1Vec_4 ? b[15:8] : 8'h0;
  wire [11:0] a1x1_4 = {a1x1_exist_3, 4'h0};
  wire [7:0]  a1x1_exist_4 = a1Vec_5 ? b[15:8] : 8'h0;
  wire [12:0] a1x1_5 = {a1x1_exist_4, 5'h0};
  wire [7:0]  a1x1_exist_5 = a1Vec_6 ? b[15:8] : 8'h0;
  wire [13:0] a1x1_6 = {a1x1_exist_5, 6'h0};
  wire [7:0]  a1x1_exist_6 = a1Vec_7 ? b[15:8] : 8'h0;
  wire [14:0] a1x1_7 = {a1x1_exist_6, 7'h0};
  wire [15:0] ax00_1 = {_ax00_layerOut_csa42_out_0[14:0], 1'h0};
  wire [15:0] ax00_2 = _ax00_layerOut_csa42_out_1[15:0];
  wire [15:0] ax10_1 = {_ax10_layerOut_csa42_out_0[14:0], 1'h0};
  wire [15:0] ax10_2 = _ax10_layerOut_csa42_out_1[15:0];
  wire [15:0] ax01_1 = {_ax01_layerOut_csa42_out_0[14:0], 1'h0};
  wire [15:0] ax01_2 = _ax01_layerOut_csa42_out_1[15:0];
  wire [15:0] ax11_1 = {_ax11_layerOut_csa42_out_0[14:0], 1'h0};
  wire [15:0] ax11_2 = _ax11_layerOut_csa42_out_1[15:0];
  wire [31:0] result16_1 = {_result16_layer1_csa42_out_0[30:0], 1'h0};
  wire [31:0] result16_2 = _result16_layer1_csa42_out_1[31:0];
  CSA42_3 ax00_layer0_csa42 (
    .in_0  ({4'h0, a0x0_0}),
    .in_1  ({3'h0, a0x0_1}),
    .in_2  ({2'h0, a0x0_2}),
    .in_3  ({1'h0, a0x0_3}),
    .out_0 (_ax00_layer0_csa42_out_0),
    .out_1 (_ax00_layer0_csa42_out_1)
  );
  CSA42_3 ax00_layer1_csa42 (
    .in_0  ({4'h0, a0x0_4[11:4]}),
    .in_1  ({3'h0, a0x0_5[12:4]}),
    .in_2  ({2'h0, a0x0_6[13:4]}),
    .in_3  ({1'h0, a0x0_7[14:4]}),
    .out_0 (_ax00_layer1_csa42_out_0),
    .out_1 (_ax00_layer1_csa42_out_1)
  );
  CSA42_4 ax00_layerOut_csa42 (
    .in_0  ({2'h0, _ax00_layer0_csa42_out_0, 1'h0}),
    .in_1  ({3'h0, _ax00_layer0_csa42_out_1}),
    .in_2  ({_ax00_layer1_csa42_out_0[10:0], 5'h0}),
    .in_3  ({_ax00_layer1_csa42_out_1[11:0], 4'h0}),
    .out_0 (_ax00_layerOut_csa42_out_0),
    .out_1 (_ax00_layerOut_csa42_out_1)
  );
  CSA42_3 ax10_layer0_csa42 (
    .in_0  ({4'h0, a1x0_0}),
    .in_1  ({3'h0, a1x0_1}),
    .in_2  ({2'h0, a1x0_2}),
    .in_3  ({1'h0, a1x0_3}),
    .out_0 (_ax10_layer0_csa42_out_0),
    .out_1 (_ax10_layer0_csa42_out_1)
  );
  CSA42_3 ax10_layer1_csa42 (
    .in_0  ({4'h0, a1x0_4[11:4]}),
    .in_1  ({3'h0, a1x0_5[12:4]}),
    .in_2  ({2'h0, a1x0_6[13:4]}),
    .in_3  ({1'h0, a1x0_7[14:4]}),
    .out_0 (_ax10_layer1_csa42_out_0),
    .out_1 (_ax10_layer1_csa42_out_1)
  );
  CSA42_4 ax10_layerOut_csa42 (
    .in_0  ({2'h0, _ax10_layer0_csa42_out_0, 1'h0}),
    .in_1  ({3'h0, _ax10_layer0_csa42_out_1}),
    .in_2  ({_ax10_layer1_csa42_out_0[10:0], 5'h0}),
    .in_3  ({_ax10_layer1_csa42_out_1[11:0], 4'h0}),
    .out_0 (_ax10_layerOut_csa42_out_0),
    .out_1 (_ax10_layerOut_csa42_out_1)
  );
  CSA42_3 ax01_layer0_csa42 (
    .in_0  ({4'h0, a0x1_0}),
    .in_1  ({3'h0, a0x1_1}),
    .in_2  ({2'h0, a0x1_2}),
    .in_3  ({1'h0, a0x1_3}),
    .out_0 (_ax01_layer0_csa42_out_0),
    .out_1 (_ax01_layer0_csa42_out_1)
  );
  CSA42_3 ax01_layer1_csa42 (
    .in_0  ({4'h0, a0x1_4[11:4]}),
    .in_1  ({3'h0, a0x1_5[12:4]}),
    .in_2  ({2'h0, a0x1_6[13:4]}),
    .in_3  ({1'h0, a0x1_7[14:4]}),
    .out_0 (_ax01_layer1_csa42_out_0),
    .out_1 (_ax01_layer1_csa42_out_1)
  );
  CSA42_4 ax01_layerOut_csa42 (
    .in_0  ({2'h0, _ax01_layer0_csa42_out_0, 1'h0}),
    .in_1  ({3'h0, _ax01_layer0_csa42_out_1}),
    .in_2  ({_ax01_layer1_csa42_out_0[10:0], 5'h0}),
    .in_3  ({_ax01_layer1_csa42_out_1[11:0], 4'h0}),
    .out_0 (_ax01_layerOut_csa42_out_0),
    .out_1 (_ax01_layerOut_csa42_out_1)
  );
  CSA42_3 ax11_layer0_csa42 (
    .in_0  ({4'h0, a1x1_0}),
    .in_1  ({3'h0, a1x1_1}),
    .in_2  ({2'h0, a1x1_2}),
    .in_3  ({1'h0, a1x1_3}),
    .out_0 (_ax11_layer0_csa42_out_0),
    .out_1 (_ax11_layer0_csa42_out_1)
  );
  CSA42_3 ax11_layer1_csa42 (
    .in_0  ({4'h0, a1x1_4[11:4]}),
    .in_1  ({3'h0, a1x1_5[12:4]}),
    .in_2  ({2'h0, a1x1_6[13:4]}),
    .in_3  ({1'h0, a1x1_7[14:4]}),
    .out_0 (_ax11_layer1_csa42_out_0),
    .out_1 (_ax11_layer1_csa42_out_1)
  );
  CSA42_4 ax11_layerOut_csa42 (
    .in_0  ({2'h0, _ax11_layer0_csa42_out_0, 1'h0}),
    .in_1  ({3'h0, _ax11_layer0_csa42_out_1}),
    .in_2  ({_ax11_layer1_csa42_out_0[10:0], 5'h0}),
    .in_3  ({_ax11_layer1_csa42_out_1[11:0], 4'h0}),
    .out_0 (_ax11_layerOut_csa42_out_0),
    .out_1 (_ax11_layerOut_csa42_out_1)
  );
  CSA42_1 result16_layer00_csa42 (
    .in_0  ({8'h0, ax00_1}),
    .in_1  ({8'h0, ax00_2}),
    .in_2  ({ax01_1, 8'h0}),
    .in_3  ({ax01_2, 8'h0}),
    .out_0 (_result16_layer00_csa42_out_0),
    .out_1 (_result16_layer00_csa42_out_1)
  );
  CSA42_1 result16_layer01_csa42 (
    .in_0  ({8'h0, ax10_1}),
    .in_1  ({8'h0, ax10_2}),
    .in_2  ({ax11_1, 8'h0}),
    .in_3  ({ax11_2, 8'h0}),
    .out_0 (_result16_layer01_csa42_out_0),
    .out_1 (_result16_layer01_csa42_out_1)
  );
  CSA42_2 result16_layer1_csa42 (
    .in_0  ({8'h0, _result16_layer00_csa42_out_0[22:0], 1'h0}),
    .in_1  ({7'h0, _result16_layer00_csa42_out_1}),
    .in_2  ({_result16_layer01_csa42_out_0[22:0], 9'h0}),
    .in_3  ({_result16_layer01_csa42_out_1[23:0], 8'h0}),
    .out_0 (_result16_layer1_csa42_out_0),
    .out_1 (_result16_layer1_csa42_out_1)
  );
  assign outCarry = sew[0] ? {ax11_1, ax00_1} : result16_1;
  assign outSum = sew[0] ? {ax11_2, ax00_2} : result16_2;
endmodule

