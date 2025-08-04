module sram_0R_0W_1RW_0M_256x36(
// ReadWrite Port 0,
input [7:0] RW0_addr,
input RW0_en,
input RW0_clk,
input RW0_wmode,
input [35:0] RW0_wdata,
output [35:0] RW0_rdata
);
reg [35:0] Memory[0:255];
reg [7:0] _RW0_raddr;
reg _RW0_ren;
reg _RW0_rmode;
always @(posedge RW0_clk) begin // RW0
_RW0_raddr <= RW0_addr;
_RW0_ren <= RW0_en;
_RW0_rmode <= RW0_wmode;
if (RW0_en & RW0_wmode) Memory[RW0_addr] <= RW0_wdata;
end // RW0
assign RW0_rdata = _RW0_ren & ~_RW0_rmode ? Memory[_RW0_raddr] : 36'bx;
endmodule