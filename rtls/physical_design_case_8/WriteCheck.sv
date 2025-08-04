module WriteCheck(
  input  [4:0]    check_vd,
  input  [7:0]    check_offset,
  input  [2:0]    check_instructionIndex,
  input           record_valid,
                  record_bits_vd_valid,
  input  [4:0]    record_bits_vd_bits,
  input           record_bits_vs1_valid,
  input  [4:0]    record_bits_vs1_bits,
                  record_bits_vs2,
  input  [2:0]    record_bits_instIndex,
  input           record_bits_gather,
                  record_bits_gather16,
                  record_bits_onlyRead,
  input  [2047:0] record_bits_elementMask,
  output          checkResult
);

  wire          sameInst = check_instructionIndex == record_bits_instIndex;
  wire          older = sameInst | check_instructionIndex[1:0] < record_bits_instIndex[1:0] ^ check_instructionIndex[2] ^ record_bits_instIndex[2];
  wire [2047:0] checkOH = 2048'h1 << {2037'h0, check_vd[2:0], check_offset};
  wire [8190:0] _GEN =
    {4095'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
     record_bits_elementMask,
     2048'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF};
  wire [8190:0] _maskShifter_T_6 = _GEN << {8180'h0, record_bits_vd_bits[2:0], 8'h0};
  wire [4095:0] maskShifter = _maskShifter_T_6[6143:2048];
  wire          hitVd = (checkOH & maskShifter[2047:0]) == 2048'h0 & check_vd[4:3] == record_bits_vd_bits[4:3];
  wire          hitVd1 = (checkOH & maskShifter[4095:2048]) == 2048'h0 & check_vd[4:3] == record_bits_vd_bits[4:3] + 2'h1;
  wire          waw = record_bits_vd_valid & (hitVd | hitVd1);
  wire [6142:0] _vs1Mask_T_4 =
    {2047'h0,
     record_bits_elementMask,
     2048'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF}
    << {6132'h0, record_bits_vs1_bits[2:0], 8'h0};
  wire [4094:0] vs1Mask = _vs1Mask_T_4[6142:2048];
  wire          notHitVs1 = (vs1Mask[2047:0] & checkOH) == 2048'h0 | record_bits_gather16;
  wire          war1 = record_bits_vs1_valid & check_vd[4:3] == record_bits_vs1_bits[4:3] & notHitVs1;
  wire [8190:0] _maskShifterForVs2_T_6 = _GEN << {8180'h0, record_bits_vs2[2:0], 8'h0};
  wire [4095:0] maskShifterForVs2 = _maskShifterForVs2_T_6[6143:2048];
  wire [2047:0] maskForVs2 = maskShifterForVs2[2047:0] & {2048{~record_bits_onlyRead}};
  wire          hitVs2 = ((checkOH & maskForVs2) == 2048'h0 | record_bits_gather) & check_vd[4:3] == record_bits_vs2[4:3];
  wire          hitVs21 = ((checkOH & maskShifterForVs2[4095:2048]) == 2048'h0 | record_bits_gather) & check_vd[4:3] == record_bits_vs2[4:3] + 2'h1;
  wire          war2 = hitVs2 | hitVs21;
  assign checkResult = ~(~older & (waw | war1 | war2) & ~sameInst & record_valid);
endmodule

