module ChainingCheck(
  input  [4:0] read_vs,
  input  [2:0] read_instructionIndex,
  input        record_bits_vd_valid,
  input  [4:0] record_bits_vd_bits,
  input  [2:0] record_bits_instIndex,
  input  [7:0] record_bits_elementMask,
  input        recordValid,
  output       checkResult
);

  wire        sameInst = read_instructionIndex == record_bits_instIndex;
  wire        older = sameInst | read_instructionIndex[1:0] < record_bits_instIndex[1:0] ^ read_instructionIndex[2] ^ record_bits_instIndex[2];
  wire [7:0]  readOH = 8'h1 << read_vs[2:0];
  wire [30:0] _maskShifter_T_6 = {15'hFF, record_bits_elementMask, 8'hFF} << record_bits_vd_bits[2:0];
  wire [15:0] maskShifter = _maskShifter_T_6[23:8];
  wire        hitVd = (readOH & maskShifter[7:0]) == 8'h0 & read_vs[4:3] == record_bits_vd_bits[4:3];
  wire        hitVd1 = (readOH & maskShifter[15:8]) == 8'h0 & read_vs[4:3] == record_bits_vd_bits[4:3] + 2'h1;
  wire        raw = record_bits_vd_valid & (hitVd | hitVd1);
  assign checkResult = ~(~older & raw & ~sameInst & recordValid);
endmodule

