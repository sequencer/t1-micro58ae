module WriteCheck(
  input  [4:0]   check_vd,
                 check_offset,
  input  [2:0]   check_instructionIndex,
  input          record_valid,
                 record_bits_vd_valid,
  input  [4:0]   record_bits_vd_bits,
  input          record_bits_vs1_valid,
  input  [4:0]   record_bits_vs1_bits,
                 record_bits_vs2,
  input  [2:0]   record_bits_instIndex,
  input          record_bits_gather,
                 record_bits_gather16,
                 record_bits_onlyRead,
  input  [255:0] record_bits_elementMask,
  output         checkResult
);

  wire          sameInst = check_instructionIndex == record_bits_instIndex;
  wire          older = sameInst | check_instructionIndex[1:0] < record_bits_instIndex[1:0] ^ check_instructionIndex[2] ^ record_bits_instIndex[2];
  wire [255:0]  checkOH = 256'h1 << {248'h0, check_vd[2:0], check_offset};
  wire [1022:0] _GEN = {511'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF, record_bits_elementMask, 256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF};
  wire [1022:0] _maskShifter_T_6 = _GEN << {1015'h0, record_bits_vd_bits[2:0], 5'h0};
  wire [511:0]  maskShifter = _maskShifter_T_6[767:256];
  wire          hitVd = (checkOH & maskShifter[255:0]) == 256'h0 & check_vd[4:3] == record_bits_vd_bits[4:3];
  wire          hitVd1 = (checkOH & maskShifter[511:256]) == 256'h0 & check_vd[4:3] == record_bits_vd_bits[4:3] + 2'h1;
  wire          waw = record_bits_vd_valid & (hitVd | hitVd1);
  wire [766:0]  _vs1Mask_T_4 = {255'h0, record_bits_elementMask, 256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF} << {759'h0, record_bits_vs1_bits[2:0], 5'h0};
  wire [510:0]  vs1Mask = _vs1Mask_T_4[766:256];
  wire          notHitVs1 = (vs1Mask[255:0] & checkOH) == 256'h0 | record_bits_gather16;
  wire          war1 = record_bits_vs1_valid & check_vd[4:3] == record_bits_vs1_bits[4:3] & notHitVs1;
  wire [1022:0] _maskShifterForVs2_T_6 = _GEN << {1015'h0, record_bits_vs2[2:0], 5'h0};
  wire [511:0]  maskShifterForVs2 = _maskShifterForVs2_T_6[767:256];
  wire [255:0]  maskForVs2 = maskShifterForVs2[255:0] & {256{~record_bits_onlyRead}};
  wire          hitVs2 = ((checkOH & maskForVs2) == 256'h0 | record_bits_gather) & check_vd[4:3] == record_bits_vs2[4:3];
  wire          hitVs21 = ((checkOH & maskShifterForVs2[511:256]) == 256'h0 | record_bits_gather) & check_vd[4:3] == record_bits_vs2[4:3] + 2'h1;
  wire          war2 = hitVs2 | hitVs21;
  assign checkResult = ~(~older & (waw | war1 | war2) & ~sameInst & record_valid);
endmodule

