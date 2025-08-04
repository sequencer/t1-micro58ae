module ChainingCheck(
  input  [4:0]  read_vs,
  input         read_offset,
  input  [2:0]  read_instructionIndex,
  input         record_bits_vd_valid,
  input  [4:0]  record_bits_vd_bits,
  input  [2:0]  record_bits_instIndex,
  input  [15:0] record_bits_elementMask,
  input         recordValid,
  output        checkResult
);

  wire        sameInst = read_instructionIndex == record_bits_instIndex;
  wire        older = sameInst | read_instructionIndex[1:0] < record_bits_instIndex[1:0] ^ read_instructionIndex[2] ^ record_bits_instIndex[2];
  wire [15:0] readOH = 16'h1 << {12'h0, read_vs[2:0], read_offset};
  wire [62:0] _maskShifter_T_6 = {31'hFFFF, record_bits_elementMask, 16'hFFFF} << {59'h0, record_bits_vd_bits[2:0], 1'h0};
  wire [31:0] maskShifter = _maskShifter_T_6[47:16];
  wire        hitVd = (readOH & maskShifter[15:0]) == 16'h0 & read_vs[4:3] == record_bits_vd_bits[4:3];
  wire        hitVd1 = (readOH & maskShifter[31:16]) == 16'h0 & read_vs[4:3] == record_bits_vd_bits[4:3] + 2'h1;
  wire        raw = record_bits_vd_valid & (hitVd | hitVd1);
  assign checkResult = ~(~older & raw & ~sameInst & recordValid);
endmodule

