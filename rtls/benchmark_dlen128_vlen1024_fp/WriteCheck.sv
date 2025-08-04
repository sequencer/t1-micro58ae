module WriteCheck(
  input  [4:0]  check_vd,
  input  [2:0]  check_offset,
                check_instructionIndex,
  input         record_valid,
                record_bits_vd_valid,
  input  [4:0]  record_bits_vd_bits,
  input         record_bits_vs1_valid,
  input  [4:0]  record_bits_vs1_bits,
                record_bits_vs2,
  input  [2:0]  record_bits_instIndex,
  input         record_bits_gather,
                record_bits_gather16,
                record_bits_onlyRead,
  input  [63:0] record_bits_elementMask,
  output        checkResult
);

  wire         sameInst = check_instructionIndex == record_bits_instIndex;
  wire         older = sameInst | check_instructionIndex[1:0] < record_bits_instIndex[1:0] ^ check_instructionIndex[2] ^ record_bits_instIndex[2];
  wire [63:0]  checkOH = 64'h1 << {58'h0, check_vd[2:0], check_offset};
  wire [254:0] _GEN = {127'hFFFFFFFFFFFFFFFF, record_bits_elementMask, 64'hFFFFFFFFFFFFFFFF};
  wire [254:0] _maskShifter_T_6 = _GEN << {249'h0, record_bits_vd_bits[2:0], 3'h0};
  wire [127:0] maskShifter = _maskShifter_T_6[191:64];
  wire         hitVd = (checkOH & maskShifter[63:0]) == 64'h0 & check_vd[4:3] == record_bits_vd_bits[4:3];
  wire         hitVd1 = (checkOH & maskShifter[127:64]) == 64'h0 & check_vd[4:3] == record_bits_vd_bits[4:3] + 2'h1;
  wire         waw = record_bits_vd_valid & (hitVd | hitVd1);
  wire [190:0] _vs1Mask_T_4 = {63'h0, record_bits_elementMask, 64'hFFFFFFFFFFFFFFFF} << {185'h0, record_bits_vs1_bits[2:0], 3'h0};
  wire [126:0] vs1Mask = _vs1Mask_T_4[190:64];
  wire         notHitVs1 = (vs1Mask[63:0] & checkOH) == 64'h0 | record_bits_gather16;
  wire         war1 = record_bits_vs1_valid & check_vd[4:3] == record_bits_vs1_bits[4:3] & notHitVs1;
  wire [254:0] _maskShifterForVs2_T_6 = _GEN << {249'h0, record_bits_vs2[2:0], 3'h0};
  wire [127:0] maskShifterForVs2 = _maskShifterForVs2_T_6[191:64];
  wire [63:0]  maskForVs2 = maskShifterForVs2[63:0] & {64{~record_bits_onlyRead}};
  wire         hitVs2 = ((checkOH & maskForVs2) == 64'h0 | record_bits_gather) & check_vd[4:3] == record_bits_vs2[4:3];
  wire         hitVs21 = ((checkOH & maskShifterForVs2[127:64]) == 64'h0 | record_bits_gather) & check_vd[4:3] == record_bits_vs2[4:3] + 2'h1;
  wire         war2 = hitVs2 | hitVs21;
  assign checkResult = ~(~older & (waw | war1 | war2) & ~sameInst & record_valid);
endmodule

