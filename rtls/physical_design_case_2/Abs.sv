module Abs(
  input  [31:0] io_aIn,
                io_bIn,
  input         io_signIn,
  output [31:0] io_aOut,
                io_bOut,
  output        io_aSign,
                io_bSign
);

  wire [31:0] io_aIn_0 = io_aIn;
  wire [31:0] io_bIn_0 = io_bIn;
  wire        io_signIn_0 = io_signIn;
  wire [31:0] a = io_aIn_0;
  wire [31:0] b = io_bIn_0;
  wire        aSign = io_aIn_0[31];
  wire        bSign = io_bIn_0[31];
  wire        io_aSign_0 = io_signIn_0 & aSign;
  wire [31:0] io_aOut_0 = io_aSign_0 ? 32'h0 - a : a;
  wire        io_bSign_0 = io_signIn_0 & bSign;
  wire [31:0] io_bOut_0 = io_bSign_0 ? 32'h0 - b : b;
  assign io_aOut = io_aOut_0;
  assign io_bOut = io_bOut_0;
  assign io_aSign = io_aSign_0;
  assign io_bSign = io_bSign_0;
endmodule

