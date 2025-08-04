module ChainingCheck(
  input  [4:0]  read_vs,
  input  [1:0]  read_offset,
  input  [2:0]  read_instructionIndex,
  input         record_bits_vd_valid,
  input  [4:0]  record_bits_vd_bits,
  input  [2:0]  record_bits_instIndex,
  input  [31:0] record_bits_elementMask,
  input         recordValid,
  output        checkResult
);

  wire         sameInst = read_instructionIndex == record_bits_instIndex;
  wire         older = sameInst | read_instructionIndex[1:0] < record_bits_instIndex[1:0] ^ read_instructionIndex[2] ^ record_bits_instIndex[2];
  wire [31:0]  readOH = 32'h1 << {27'h0, read_vs[2:0], read_offset};
  wire [126:0] _maskShifter_T_6 = {63'hFFFFFFFF, record_bits_elementMask, 32'hFFFFFFFF} << {122'h0, record_bits_vd_bits[2:0], 2'h0};
  wire [63:0]  maskShifter = _maskShifter_T_6[95:32];
  wire         hitVd = (readOH & maskShifter[31:0]) == 32'h0 & read_vs[4:3] == record_bits_vd_bits[4:3];
  wire         hitVd1 = (readOH & maskShifter[63:32]) == 32'h0 & read_vs[4:3] == record_bits_vd_bits[4:3] + 2'h1;
  wire         raw = record_bits_vd_valid & (hitVd | hitVd1);
  assign checkResult = ~(~older & raw & ~sameInst & recordValid);
endmodule

