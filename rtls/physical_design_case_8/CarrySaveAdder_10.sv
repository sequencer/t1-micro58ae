module CarrySaveAdder_10(
  input  [9:0] in_0,
               in_1,
               in_2,
  output [9:0] out_0,
               out_1
);

  wire       out_xor01 = in_0[0] ^ in_1[0];
  wire       out_0_0 = in_0[0] & in_1[0] | out_xor01 & in_2[0];
  wire       out_1_0 = out_xor01 ^ in_2[0];
  wire       out_xor01_1 = in_0[1] ^ in_1[1];
  wire       out_0_1 = in_0[1] & in_1[1] | out_xor01_1 & in_2[1];
  wire       out_1_1 = out_xor01_1 ^ in_2[1];
  wire       out_xor01_2 = in_0[2] ^ in_1[2];
  wire       out_0_2 = in_0[2] & in_1[2] | out_xor01_2 & in_2[2];
  wire       out_1_2 = out_xor01_2 ^ in_2[2];
  wire       out_xor01_3 = in_0[3] ^ in_1[3];
  wire       out_0_3 = in_0[3] & in_1[3] | out_xor01_3 & in_2[3];
  wire       out_1_3 = out_xor01_3 ^ in_2[3];
  wire       out_xor01_4 = in_0[4] ^ in_1[4];
  wire       out_0_4 = in_0[4] & in_1[4] | out_xor01_4 & in_2[4];
  wire       out_1_4 = out_xor01_4 ^ in_2[4];
  wire       out_xor01_5 = in_0[5] ^ in_1[5];
  wire       out_0_5 = in_0[5] & in_1[5] | out_xor01_5 & in_2[5];
  wire       out_1_5 = out_xor01_5 ^ in_2[5];
  wire       out_xor01_6 = in_0[6] ^ in_1[6];
  wire       out_0_6 = in_0[6] & in_1[6] | out_xor01_6 & in_2[6];
  wire       out_1_6 = out_xor01_6 ^ in_2[6];
  wire       out_xor01_7 = in_0[7] ^ in_1[7];
  wire       out_0_7 = in_0[7] & in_1[7] | out_xor01_7 & in_2[7];
  wire       out_1_7 = out_xor01_7 ^ in_2[7];
  wire       out_xor01_8 = in_0[8] ^ in_1[8];
  wire       out_0_8 = in_0[8] & in_1[8] | out_xor01_8 & in_2[8];
  wire       out_1_8 = out_xor01_8 ^ in_2[8];
  wire       out_xor01_9 = in_0[9] ^ in_1[9];
  wire       out_0_9 = in_0[9] & in_1[9] | out_xor01_9 & in_2[9];
  wire       out_1_9 = out_xor01_9 ^ in_2[9];
  wire [1:0] out_0_lo_lo = {out_0_1, out_0_0};
  wire [1:0] out_0_lo_hi_hi = {out_0_4, out_0_3};
  wire [2:0] out_0_lo_hi = {out_0_lo_hi_hi, out_0_2};
  wire [4:0] out_0_lo = {out_0_lo_hi, out_0_lo_lo};
  wire [1:0] out_0_hi_lo = {out_0_6, out_0_5};
  wire [1:0] out_0_hi_hi_hi = {out_0_9, out_0_8};
  wire [2:0] out_0_hi_hi = {out_0_hi_hi_hi, out_0_7};
  wire [4:0] out_0_hi = {out_0_hi_hi, out_0_hi_lo};
  wire [1:0] out_1_lo_lo = {out_1_1, out_1_0};
  wire [1:0] out_1_lo_hi_hi = {out_1_4, out_1_3};
  wire [2:0] out_1_lo_hi = {out_1_lo_hi_hi, out_1_2};
  wire [4:0] out_1_lo = {out_1_lo_hi, out_1_lo_lo};
  wire [1:0] out_1_hi_lo = {out_1_6, out_1_5};
  wire [1:0] out_1_hi_hi_hi = {out_1_9, out_1_8};
  wire [2:0] out_1_hi_hi = {out_1_hi_hi_hi, out_1_7};
  wire [4:0] out_1_hi = {out_1_hi_hi, out_1_hi_lo};
  assign out_0 = {out_0_hi, out_0_lo};
  assign out_1 = {out_1_hi, out_1_lo};
endmodule

