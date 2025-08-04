module CSACompressor4_2(
  input  in_0,
         in_1,
         in_2,
         in_3,
         cin,
  output out_0,
         out_1,
         cout
);

  wire ab = in_0 ^ in_1;
  wire cd = in_2 ^ in_3;
  wire abcd = ab ^ cd;
  assign out_0 = abcd ? cin : in_3;
  assign out_1 = abcd ^ cin;
  assign cout = ab ? in_2 : in_0;
endmodule

