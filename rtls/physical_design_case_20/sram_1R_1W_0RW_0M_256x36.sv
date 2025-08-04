module sram_1R_1W_0RW_0M_256x36(
// Write Port 0,
input [7:0] W0_addr,
input W0_en,
input W0_clk,
input [35:0] W0_data,
// Read Port 0,
input [7:0] R0_addr,
input R0_en,
input R0_clk,
output [35:0] R0_data
);
reg [35:0] Memory[0:255];
always @(posedge W0_clk) begin // W0
if (W0_en) Memory[W0_addr] <= W0_data;
end // W0
reg _R0_en;
reg [7:0] _R0_addr;
always @(posedge R0_clk) begin // R0
_R0_en <= R0_en;
_R0_addr <= R0_addr;
end // R0
assign R0_data = _R0_en ? Memory[_R0_addr] : 36'bx;
endmodule