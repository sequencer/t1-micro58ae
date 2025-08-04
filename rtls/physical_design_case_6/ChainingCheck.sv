module ChainingCheck(
  input  [4:0]   read_vs,
  input  [5:0]   read_offset,
  input  [2:0]   read_instructionIndex,
  input          record_bits_vd_valid,
  input  [4:0]   record_bits_vd_bits,
  input  [2:0]   record_bits_instIndex,
  input  [511:0] record_bits_elementMask,
  input          recordValid,
  output         checkResult
);

  wire          sameInst = read_instructionIndex == record_bits_instIndex;
  wire          older = sameInst | read_instructionIndex[1:0] < record_bits_instIndex[1:0] ^ read_instructionIndex[2] ^ record_bits_instIndex[2];
  wire [511:0]  readOH = 512'h1 << {503'h0, read_vs[2:0], read_offset};
  wire [2046:0] _maskShifter_T_6 =
    {1023'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
     record_bits_elementMask,
     512'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF} << {2038'h0, record_bits_vd_bits[2:0], 6'h0};
  wire [1023:0] maskShifter = _maskShifter_T_6[1535:512];
  wire          hitVd = (readOH & maskShifter[511:0]) == 512'h0 & read_vs[4:3] == record_bits_vd_bits[4:3];
  wire          hitVd1 = (readOH & maskShifter[1023:512]) == 512'h0 & read_vs[4:3] == record_bits_vd_bits[4:3] + 2'h1;
  wire          raw = record_bits_vd_valid & (hitVd | hitVd1);
  assign checkResult = ~(~older & raw & ~sameInst & recordValid);
endmodule

