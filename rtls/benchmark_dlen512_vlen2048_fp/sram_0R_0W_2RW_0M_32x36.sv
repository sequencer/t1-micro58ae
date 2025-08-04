module sram_0R_0W_2RW_0M_32x36(
// ReadWrite Port 0,
input [4:0] RW0_addr,
input RW0_en,
input RW0_clk,
input RW0_wmode,
input [35:0] RW0_wdata,
output [35:0] RW0_rdata,
// ReadWrite Port 1,
input [4:0] RW1_addr,
input RW1_en,
input RW1_clk,
input RW1_wmode,
input [35:0] RW1_wdata,
output [35:0] RW1_rdata
);
reg [35:0] Memory[0:31];
reg [4:0] _RW0_raddr;
reg _RW0_ren;
reg _RW0_rmode;
always @(posedge RW0_clk) begin // RW0
_RW0_raddr <= RW0_addr;
_RW0_ren <= RW0_en;
_RW0_rmode <= RW0_wmode;
if (RW0_en & RW0_wmode) Memory[RW0_addr] <= RW0_wdata;
end // RW0
assign RW0_rdata = _RW0_ren & ~_RW0_rmode ? Memory[_RW0_raddr] : 36'bx;
reg [4:0] _RW1_raddr;
reg _RW1_ren;
reg _RW1_rmode;
always @(posedge RW1_clk) begin // RW1
_RW1_raddr <= RW1_addr;
_RW1_ren <= RW1_en;
_RW1_rmode <= RW1_wmode;
if (RW1_en & RW1_wmode) Memory[RW1_addr] <= RW1_wdata;
end // RW1
assign RW1_rdata = _RW1_ren & ~_RW1_rmode ? Memory[_RW1_raddr] : 36'bx;
endmodule