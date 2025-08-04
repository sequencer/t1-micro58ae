module WriteCheck(
  input  [4:0] check_vd,
  input  [2:0] check_instructionIndex,
  input        record_valid,
               record_bits_vd_valid,
  input  [4:0] record_bits_vd_bits,
  input        record_bits_vs1_valid,
  input  [4:0] record_bits_vs1_bits,
               record_bits_vs2,
  input  [2:0] record_bits_instIndex,
  input        record_bits_gather,
               record_bits_gather16,
               record_bits_onlyRead,
  input  [7:0] record_bits_elementMask,
  output       checkResult
);

  wire        sameInst = check_instructionIndex == record_bits_instIndex;
  wire        older = sameInst | check_instructionIndex[1:0] < record_bits_instIndex[1:0] ^ check_instructionIndex[2] ^ record_bits_instIndex[2];
  wire [7:0]  checkOH = 8'h1 << check_vd[2:0];
  wire [30:0] _GEN = {15'hFF, record_bits_elementMask, 8'hFF};
  wire [30:0] _maskShifter_T_6 = _GEN << record_bits_vd_bits[2:0];
  wire [15:0] maskShifter = _maskShifter_T_6[23:8];
  wire        hitVd = (checkOH & maskShifter[7:0]) == 8'h0 & check_vd[4:3] == record_bits_vd_bits[4:3];
  wire        hitVd1 = (checkOH & maskShifter[15:8]) == 8'h0 & check_vd[4:3] == record_bits_vd_bits[4:3] + 2'h1;
  wire        waw = record_bits_vd_valid & (hitVd | hitVd1);
  wire [22:0] _vs1Mask_T_4 = {7'h0, record_bits_elementMask, 8'hFF} << record_bits_vs1_bits[2:0];
  wire [14:0] vs1Mask = _vs1Mask_T_4[22:8];
  wire        notHitVs1 = (vs1Mask[7:0] & checkOH) == 8'h0 | record_bits_gather16;
  wire        war1 = record_bits_vs1_valid & check_vd[4:3] == record_bits_vs1_bits[4:3] & notHitVs1;
  wire [30:0] _maskShifterForVs2_T_6 = _GEN << record_bits_vs2[2:0];
  wire [15:0] maskShifterForVs2 = _maskShifterForVs2_T_6[23:8];
  wire [7:0]  maskForVs2 = maskShifterForVs2[7:0] & {8{~record_bits_onlyRead}};
  wire        hitVs2 = ((checkOH & maskForVs2) == 8'h0 | record_bits_gather) & check_vd[4:3] == record_bits_vs2[4:3];
  wire        hitVs21 = ((checkOH & maskShifterForVs2[15:8]) == 8'h0 | record_bits_gather) & check_vd[4:3] == record_bits_vs2[4:3] + 2'h1;
  wire        war2 = hitVs2 | hitVs21;
  assign checkResult = ~(~older & (waw | war1 | war2) & ~sameInst & record_valid);
endmodule

