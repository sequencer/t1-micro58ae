module ChainingCheck(
  input  [4:0]    read_vs,
  input  [6:0]    read_offset,
  input  [2:0]    read_instructionIndex,
  input           record_bits_vd_valid,
  input  [4:0]    record_bits_vd_bits,
  input  [2:0]    record_bits_instIndex,
  input  [1023:0] record_bits_elementMask,
  input           recordValid,
  output          checkResult
);

  wire          sameInst = read_instructionIndex == record_bits_instIndex;
  wire          older = sameInst | read_instructionIndex[1:0] < record_bits_instIndex[1:0] ^ read_instructionIndex[2] ^ record_bits_instIndex[2];
  wire [1023:0] readOH = 1024'h1 << {1014'h0, read_vs[2:0], read_offset};
  wire [4094:0] _maskShifter_T_6 =
    {2047'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
     record_bits_elementMask,
     1024'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF}
    << {4085'h0, record_bits_vd_bits[2:0], 7'h0};
  wire [2047:0] maskShifter = _maskShifter_T_6[3071:1024];
  wire          hitVd = (readOH & maskShifter[1023:0]) == 1024'h0 & read_vs[4:3] == record_bits_vd_bits[4:3];
  wire          hitVd1 = (readOH & maskShifter[2047:1024]) == 1024'h0 & read_vs[4:3] == record_bits_vd_bits[4:3] + 2'h1;
  wire          raw = record_bits_vd_valid & (hitVd | hitVd1);
  assign checkResult = ~(~older & raw & ~sameInst & recordValid);
endmodule

