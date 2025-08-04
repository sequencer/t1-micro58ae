module ChainingCheck(
  input  [4:0]   read_vs,
                 read_offset,
  input  [2:0]   read_instructionIndex,
  input          record_bits_vd_valid,
  input  [4:0]   record_bits_vd_bits,
  input  [2:0]   record_bits_instIndex,
  input  [255:0] record_bits_elementMask,
  input          recordValid,
  output         checkResult
);

  wire          sameInst = read_instructionIndex == record_bits_instIndex;
  wire          older = sameInst | read_instructionIndex[1:0] < record_bits_instIndex[1:0] ^ read_instructionIndex[2] ^ record_bits_instIndex[2];
  wire [255:0]  readOH = 256'h1 << {248'h0, read_vs[2:0], read_offset};
  wire [1022:0] _maskShifter_T_6 =
    {511'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF, record_bits_elementMask, 256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF} << {1015'h0, record_bits_vd_bits[2:0], 5'h0};
  wire [511:0]  maskShifter = _maskShifter_T_6[767:256];
  wire          hitVd = (readOH & maskShifter[255:0]) == 256'h0 & read_vs[4:3] == record_bits_vd_bits[4:3];
  wire          hitVd1 = (readOH & maskShifter[511:256]) == 256'h0 & read_vs[4:3] == record_bits_vd_bits[4:3] + 2'h1;
  wire          raw = record_bits_vd_valid & (hitVd | hitVd1);
  assign checkResult = ~(~older & raw & ~sameInst & recordValid);
endmodule

