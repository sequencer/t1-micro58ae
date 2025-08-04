module LaneFFO(
  input  [31:0] src_0,
                src_1,
                src_2,
                src_3,
  input  [1:0]  resultSelect,
  output        resp_valid,
  output [31:0] resp_bits,
  input         complete,
                maskType
);

  wire [31:0] truthMask = (maskType ? src_0 : 32'hFFFFFFFF) & src_3;
  wire [31:0] srcData = truthMask & src_1;
  wire        notZero = |srcData;
  wire [31:0] _lo_T_2 = srcData | {srcData[30:0], 1'h0};
  wire [31:0] _lo_T_5 = _lo_T_2 | {_lo_T_2[29:0], 2'h0};
  wire [31:0] _lo_T_8 = _lo_T_5 | {_lo_T_5[27:0], 4'h0};
  wire [31:0] _lo_T_11 = _lo_T_8 | {_lo_T_8[23:0], 8'h0};
  wire [31:0] lo = _lo_T_11 | {_lo_T_11[15:0], 16'h0};
  wire [31:0] ro = ~lo;
  wire [32:0] inc = {ro, notZero};
  wire [32:0] OH = {1'h0, inc[31:0] & lo};
  wire        index_hi = OH[32];
  wire [31:0] index_lo = OH[31:0];
  wire [31:0] _index_T_1 = {31'h0, index_hi} | index_lo;
  wire [15:0] index_hi_1 = _index_T_1[31:16];
  wire [15:0] index_lo_1 = _index_T_1[15:0];
  wire [15:0] _index_T_3 = index_hi_1 | index_lo_1;
  wire [7:0]  index_hi_2 = _index_T_3[15:8];
  wire [7:0]  index_lo_2 = _index_T_3[7:0];
  wire [7:0]  _index_T_5 = index_hi_2 | index_lo_2;
  wire [3:0]  index_hi_3 = _index_T_5[7:4];
  wire [3:0]  index_lo_3 = _index_T_5[3:0];
  wire [3:0]  _index_T_7 = index_hi_3 | index_lo_3;
  wire [1:0]  index_hi_4 = _index_T_7[3:2];
  wire [1:0]  index_lo_4 = _index_T_7[1:0];
  wire [5:0]  index = {index_hi, |index_hi_1, |index_hi_2, |index_hi_3, |index_hi_4, index_hi_4[1] | index_lo_4[1]};
  wire [3:0]  selectOH = 4'h1 << resultSelect;
  wire        first = selectOH[0];
  wire        sbf = selectOH[1];
  wire        sof = selectOH[2];
  wire        sif = selectOH[3];
  wire [32:0] _ffoResult_T_24 = {1'h0, {26'h0, ~complete & first ? index : 6'h0} | (~complete & notZero & sbf ? ro : 32'h0)} | (~complete & sof ? OH : 33'h0) | (~complete & notZero & sif ? inc : 33'h0);
  wire [32:0] ffoResult = {_ffoResult_T_24[32:1], _ffoResult_T_24[0] | ~complete & ~notZero & resultSelect[0]};
  wire [31:0] resultMask = (maskType & ~first ? src_0 : 32'hFFFFFFFF) & (first ? 32'hFFFFFFFF : src_3);
  assign resp_valid = notZero;
  assign resp_bits = ffoResult[31:0] & resultMask | src_2 & ~resultMask;
endmodule

