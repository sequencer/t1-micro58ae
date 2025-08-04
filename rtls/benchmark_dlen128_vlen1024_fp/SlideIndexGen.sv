
// Include register initializers in init blocks unless synthesis is set
`ifndef RANDOMIZE
  `ifdef RANDOMIZE_REG_INIT
    `define RANDOMIZE
  `endif // RANDOMIZE_REG_INIT
`endif // not def RANDOMIZE
`ifndef SYNTHESIS
  `ifndef ENABLE_INITIAL_REG_
    `define ENABLE_INITIAL_REG_
  `endif // not def ENABLE_INITIAL_REG_
`endif // not def SYNTHESIS

// Standard header to adapt well known macros for register randomization.

// RANDOM may be set to an expression that produces a 32-bit random unsigned value.
`ifndef RANDOM
  `define RANDOM $random
`endif // not def RANDOM

// Users can define INIT_RANDOM as general code that gets injected into the
// initializer block for modules with registers.
`ifndef INIT_RANDOM
  `define INIT_RANDOM
`endif // not def INIT_RANDOM

// If using random initialization, you can also define RANDOMIZE_DELAY to
// customize the delay used, otherwise 0.002 is used.
`ifndef RANDOMIZE_DELAY
  `define RANDOMIZE_DELAY 0.002
`endif // not def RANDOMIZE_DELAY

// Define INIT_RANDOM_PROLOG_ for use in our modules below.
`ifndef INIT_RANDOM_PROLOG_
  `ifdef RANDOMIZE
    `ifdef VERILATOR
      `define INIT_RANDOM_PROLOG_ `INIT_RANDOM
    `else  // VERILATOR
      `define INIT_RANDOM_PROLOG_ `INIT_RANDOM #`RANDOMIZE_DELAY begin end
    `endif // VERILATOR
  `else  // RANDOMIZE
    `define INIT_RANDOM_PROLOG_
  `endif // RANDOMIZE
`endif // not def INIT_RANDOM_PROLOG_
module SlideIndexGen(
  input         clock,
                reset,
                newInstruction,
  input  [4:0]  instructionReq_decodeResult_topUop,
  input  [31:0] instructionReq_readFromScala,
  input  [1:0]  instructionReq_sew,
  input  [2:0]  instructionReq_vlmul,
  input         instructionReq_maskType,
  input  [10:0] instructionReq_vl,
  input         indexDeq_ready,
  output        indexDeq_valid,
  output [3:0]  indexDeq_bits_needRead,
                indexDeq_bits_elementValid,
                indexDeq_bits_replaceVs1,
  output [11:0] indexDeq_bits_readOffset,
  output [1:0]  indexDeq_bits_accessLane_0,
                indexDeq_bits_accessLane_1,
                indexDeq_bits_accessLane_2,
                indexDeq_bits_accessLane_3,
  output [2:0]  indexDeq_bits_vsGrowth_0,
                indexDeq_bits_vsGrowth_1,
                indexDeq_bits_vsGrowth_2,
                indexDeq_bits_vsGrowth_3,
  output [8:0]  indexDeq_bits_executeGroup,
  output [7:0]  indexDeq_bits_readDataOffset,
  output        indexDeq_bits_last,
  output [8:0]  slideGroupOut,
  input  [3:0]  slideMaskInput
);

  wire        indexDeq_ready_0 = indexDeq_ready;
  wire [1:0]  indexVec_checkResult_2_0 = 2'h0;
  wire [1:0]  indexVec_checkResult_2_0_1 = 2'h0;
  wire [1:0]  indexVec_checkResult_2_0_2 = 2'h0;
  wire [1:0]  indexVec_checkResult_2_0_3 = 2'h0;
  wire [3:0]  indexDeq_bits_groupReadState = 4'h0;
  wire [3:0]  replaceWithVs1;
  wire [1:0]  indexVec_0_1;
  wire [1:0]  indexVec_1_1;
  wire [1:0]  indexVec_2_1;
  wire [1:0]  indexVec_3_1;
  wire [2:0]  indexVec_0_3;
  wire [2:0]  indexVec_1_3;
  wire [2:0]  indexVec_2_3;
  wire [2:0]  indexVec_3_3;
  reg         InstructionValid;
  wire        isSlide = instructionReq_decodeResult_topUop[4:2] == 3'h0;
  wire        slideUp = instructionReq_decodeResult_topUop[0];
  wire        slide1 = instructionReq_decodeResult_topUop[1];
  reg  [8:0]  slideGroup;
  wire [8:0]  indexDeq_bits_executeGroup_0 = slideGroup;
  wire [1:0]  vlTail = instructionReq_vl[1:0];
  wire [8:0]  lastSlideGroup = instructionReq_vl[10:2] - {8'h0, vlTail == 2'h0};
  wire [3:0]  _lastValidVec_T = 4'h1 << vlTail;
  wire [3:0]  _lastValidVec_T_3 = _lastValidVec_T | {_lastValidVec_T[2:0], 1'h0};
  wire [3:0]  lastValidVec = ~(_lastValidVec_T_3 | {_lastValidVec_T_3[1:0], 2'h0});
  wire        indexDeq_bits_last_0 = slideGroup == lastSlideGroup;
  wire [3:0]  groupVlValid = indexDeq_bits_last_0 & (|vlTail) ? lastValidVec : 4'hF;
  wire [3:0]  groupMaskValid = instructionReq_maskType ? slideMaskInput : 4'hF;
  wire [3:0]  validVec = groupVlValid & groupMaskValid;
  wire [3:0]  lastElementValid = ({1'h0, groupVlValid[3:1]} ^ groupVlValid) & groupMaskValid;
  assign replaceWithVs1 = (slideGroup == 9'h0 & slide1 & slideUp ? {3'h0, groupMaskValid[0]} : 4'h0) | (indexDeq_bits_last_0 & slide1 & ~slideUp ? lastElementValid : 4'h0);
  wire        indexDeq_valid_0;
  wire [3:0]  indexDeq_bits_replaceVs1_0 = replaceWithVs1;
  wire        _lastFire_T_1 = indexDeq_ready_0 & indexDeq_valid_0;
  wire        lastFire = indexDeq_bits_last_0 & _lastFire_T_1;
  wire [31:0] slideValue = slide1 ? 32'h1 : instructionReq_readFromScala;
  wire [31:0] PNSelect = {32{slideUp}} ^ slideValue;
  wire [31:0] baseIndex = {21'h0, slideGroup, 2'h0} + PNSelect + {31'h0, slideUp};
  wire [31:0] indexVec_readIndex = baseIndex;
  wire        lagerThanVL = |(slideValue[31:11]);
  wire [3:0]  _sew1H_T = 4'h1 << instructionReq_sew;
  wire [2:0]  sew1H = _sew1H_T[2:0];
  wire [31:0] indexVec_checkResult_allDataPosition = indexVec_readIndex;
  wire [3:0]  _GEN = 4'h1 << instructionReq_vlmul[1:0];
  wire [3:0]  indexVec_checkResult_intLMULInput;
  assign indexVec_checkResult_intLMULInput = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_1;
  assign indexVec_checkResult_intLMULInput_1 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_2;
  assign indexVec_checkResult_intLMULInput_2 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_3;
  assign indexVec_checkResult_intLMULInput_3 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_4;
  assign indexVec_checkResult_intLMULInput_4 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_5;
  assign indexVec_checkResult_intLMULInput_5 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_6;
  assign indexVec_checkResult_intLMULInput_6 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_7;
  assign indexVec_checkResult_intLMULInput_7 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_8;
  assign indexVec_checkResult_intLMULInput_8 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_9;
  assign indexVec_checkResult_intLMULInput_9 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_10;
  assign indexVec_checkResult_intLMULInput_10 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_11;
  assign indexVec_checkResult_intLMULInput_11 = _GEN;
  wire [9:0]  indexVec_checkResult_dataPosition = indexVec_checkResult_allDataPosition[9:0];
  wire [1:0]  indexVec_checkResult_0_0 = indexVec_checkResult_dataPosition[1:0];
  wire [1:0]  indexVec_checkResult_0_1 = indexVec_checkResult_dataPosition[3:2];
  wire [5:0]  indexVec_checkResult_dataGroup = indexVec_checkResult_dataPosition[9:4];
  wire [2:0]  indexVec_checkResult_0_2 = indexVec_checkResult_dataGroup[2:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth = indexVec_checkResult_dataGroup[5:3];
  wire [2:0]  indexVec_checkResult_0_3 = indexVec_checkResult_accessRegGrowth;
  wire [4:0]  indexVec_checkResult_decimalProportion = {indexVec_checkResult_0_2, indexVec_checkResult_0_1};
  wire [2:0]  indexVec_checkResult_decimal = indexVec_checkResult_decimalProportion[4:2];
  wire        indexVec_checkResult_overlap =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal >= indexVec_checkResult_intLMULInput[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth} >= indexVec_checkResult_intLMULInput,
      indexVec_checkResult_allDataPosition[31:10]};
  wire        indexVec_checkResult_unChange = slideUp & (indexVec_readIndex[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5 = validVec[0] & ~indexVec_checkResult_unChange;
  wire        indexVec_checkResult_0_4 = indexVec_checkResult_overlap | ~indexVec_checkResult_0_5 | lagerThanVL | indexVec_checkResult_unChange;
  wire [32:0] indexVec_checkResult_allDataPosition_1 = {indexVec_readIndex, 1'h0};
  wire [9:0]  indexVec_checkResult_dataPosition_1 = indexVec_checkResult_allDataPosition_1[9:0];
  wire [1:0]  indexVec_checkResult_1_0 = {indexVec_checkResult_dataPosition_1[1], 1'h0};
  wire [1:0]  indexVec_checkResult_1_1 = indexVec_checkResult_dataPosition_1[3:2];
  wire [5:0]  indexVec_checkResult_dataGroup_1 = indexVec_checkResult_dataPosition_1[9:4];
  wire [2:0]  indexVec_checkResult_1_2 = indexVec_checkResult_dataGroup_1[2:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_1 = indexVec_checkResult_dataGroup_1[5:3];
  wire [2:0]  indexVec_checkResult_1_3 = indexVec_checkResult_accessRegGrowth_1;
  wire [4:0]  indexVec_checkResult_decimalProportion_1 = {indexVec_checkResult_1_2, indexVec_checkResult_1_1};
  wire [2:0]  indexVec_checkResult_decimal_1 = indexVec_checkResult_decimalProportion_1[4:2];
  wire        indexVec_checkResult_overlap_1 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_1 >= indexVec_checkResult_intLMULInput_1[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_1} >= indexVec_checkResult_intLMULInput_1,
      indexVec_checkResult_allDataPosition_1[32:10]};
  wire        indexVec_checkResult_unChange_1 = slideUp & (indexVec_readIndex[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5 = validVec[0] & ~indexVec_checkResult_unChange_1;
  wire        indexVec_checkResult_1_4 = indexVec_checkResult_overlap_1 | ~indexVec_checkResult_1_5 | lagerThanVL | indexVec_checkResult_unChange_1;
  wire [33:0] indexVec_checkResult_allDataPosition_2 = {indexVec_readIndex, 2'h0};
  wire [9:0]  indexVec_checkResult_dataPosition_2 = indexVec_checkResult_allDataPosition_2[9:0];
  wire [1:0]  indexVec_checkResult_2_1 = indexVec_checkResult_dataPosition_2[3:2];
  wire [5:0]  indexVec_checkResult_dataGroup_2 = indexVec_checkResult_dataPosition_2[9:4];
  wire [2:0]  indexVec_checkResult_2_2 = indexVec_checkResult_dataGroup_2[2:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_2 = indexVec_checkResult_dataGroup_2[5:3];
  wire [2:0]  indexVec_checkResult_2_3 = indexVec_checkResult_accessRegGrowth_2;
  wire [4:0]  indexVec_checkResult_decimalProportion_2 = {indexVec_checkResult_2_2, indexVec_checkResult_2_1};
  wire [2:0]  indexVec_checkResult_decimal_2 = indexVec_checkResult_decimalProportion_2[4:2];
  wire        indexVec_checkResult_overlap_2 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_2 >= indexVec_checkResult_intLMULInput_2[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_2} >= indexVec_checkResult_intLMULInput_2,
      indexVec_checkResult_allDataPosition_2[33:10]};
  wire        indexVec_checkResult_unChange_2 = slideUp & (indexVec_readIndex[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5 = validVec[0] & ~indexVec_checkResult_unChange_2;
  wire        indexVec_checkResult_2_4 = indexVec_checkResult_overlap_2 | ~indexVec_checkResult_2_5 | lagerThanVL | indexVec_checkResult_unChange_2;
  wire [1:0]  indexVec_0_0 = (sew1H[0] ? indexVec_checkResult_0_0 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0 : 2'h0);
  assign indexVec_0_1 = (sew1H[0] ? indexVec_checkResult_0_1 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_1 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_1 : 2'h0);
  wire [1:0]  indexDeq_bits_accessLane_0_0 = indexVec_0_1;
  wire [2:0]  indexVec_0_2 = (sew1H[0] ? indexVec_checkResult_0_2 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_2 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_2 : 3'h0);
  assign indexVec_0_3 = (sew1H[0] ? indexVec_checkResult_0_3 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_0_0 = indexVec_0_3;
  wire        indexVec_0_4 = sew1H[0] & indexVec_checkResult_0_4 | sew1H[1] & indexVec_checkResult_1_4 | sew1H[2] & indexVec_checkResult_2_4;
  wire        indexVec_0_5 = sew1H[0] & indexVec_checkResult_0_5 | sew1H[1] & indexVec_checkResult_1_5 | sew1H[2] & indexVec_checkResult_2_5;
  wire [31:0] indexVec_readIndex_1 = baseIndex + 32'h1;
  wire [31:0] indexVec_checkResult_allDataPosition_3 = indexVec_readIndex_1;
  wire [9:0]  indexVec_checkResult_dataPosition_3 = indexVec_checkResult_allDataPosition_3[9:0];
  wire [1:0]  indexVec_checkResult_0_0_1 = indexVec_checkResult_dataPosition_3[1:0];
  wire [1:0]  indexVec_checkResult_0_1_1 = indexVec_checkResult_dataPosition_3[3:2];
  wire [5:0]  indexVec_checkResult_dataGroup_3 = indexVec_checkResult_dataPosition_3[9:4];
  wire [2:0]  indexVec_checkResult_0_2_1 = indexVec_checkResult_dataGroup_3[2:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_3 = indexVec_checkResult_dataGroup_3[5:3];
  wire [2:0]  indexVec_checkResult_0_3_1 = indexVec_checkResult_accessRegGrowth_3;
  wire [4:0]  indexVec_checkResult_decimalProportion_3 = {indexVec_checkResult_0_2_1, indexVec_checkResult_0_1_1};
  wire [2:0]  indexVec_checkResult_decimal_3 = indexVec_checkResult_decimalProportion_3[4:2];
  wire        indexVec_checkResult_overlap_3 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_3 >= indexVec_checkResult_intLMULInput_3[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_3} >= indexVec_checkResult_intLMULInput_3,
      indexVec_checkResult_allDataPosition_3[31:10]};
  wire        indexVec_checkResult_unChange_3 = slideUp & (indexVec_readIndex_1[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_1 = validVec[1] & ~indexVec_checkResult_unChange_3;
  wire        indexVec_checkResult_0_4_1 = indexVec_checkResult_overlap_3 | ~indexVec_checkResult_0_5_1 | lagerThanVL | indexVec_checkResult_unChange_3;
  wire [32:0] indexVec_checkResult_allDataPosition_4 = {indexVec_readIndex_1, 1'h0};
  wire [9:0]  indexVec_checkResult_dataPosition_4 = indexVec_checkResult_allDataPosition_4[9:0];
  wire [1:0]  indexVec_checkResult_1_0_1 = {indexVec_checkResult_dataPosition_4[1], 1'h0};
  wire [1:0]  indexVec_checkResult_1_1_1 = indexVec_checkResult_dataPosition_4[3:2];
  wire [5:0]  indexVec_checkResult_dataGroup_4 = indexVec_checkResult_dataPosition_4[9:4];
  wire [2:0]  indexVec_checkResult_1_2_1 = indexVec_checkResult_dataGroup_4[2:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_4 = indexVec_checkResult_dataGroup_4[5:3];
  wire [2:0]  indexVec_checkResult_1_3_1 = indexVec_checkResult_accessRegGrowth_4;
  wire [4:0]  indexVec_checkResult_decimalProportion_4 = {indexVec_checkResult_1_2_1, indexVec_checkResult_1_1_1};
  wire [2:0]  indexVec_checkResult_decimal_4 = indexVec_checkResult_decimalProportion_4[4:2];
  wire        indexVec_checkResult_overlap_4 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_4 >= indexVec_checkResult_intLMULInput_4[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_4} >= indexVec_checkResult_intLMULInput_4,
      indexVec_checkResult_allDataPosition_4[32:10]};
  wire        indexVec_checkResult_unChange_4 = slideUp & (indexVec_readIndex_1[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_1 = validVec[1] & ~indexVec_checkResult_unChange_4;
  wire        indexVec_checkResult_1_4_1 = indexVec_checkResult_overlap_4 | ~indexVec_checkResult_1_5_1 | lagerThanVL | indexVec_checkResult_unChange_4;
  wire [33:0] indexVec_checkResult_allDataPosition_5 = {indexVec_readIndex_1, 2'h0};
  wire [9:0]  indexVec_checkResult_dataPosition_5 = indexVec_checkResult_allDataPosition_5[9:0];
  wire [1:0]  indexVec_checkResult_2_1_1 = indexVec_checkResult_dataPosition_5[3:2];
  wire [5:0]  indexVec_checkResult_dataGroup_5 = indexVec_checkResult_dataPosition_5[9:4];
  wire [2:0]  indexVec_checkResult_2_2_1 = indexVec_checkResult_dataGroup_5[2:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_5 = indexVec_checkResult_dataGroup_5[5:3];
  wire [2:0]  indexVec_checkResult_2_3_1 = indexVec_checkResult_accessRegGrowth_5;
  wire [4:0]  indexVec_checkResult_decimalProportion_5 = {indexVec_checkResult_2_2_1, indexVec_checkResult_2_1_1};
  wire [2:0]  indexVec_checkResult_decimal_5 = indexVec_checkResult_decimalProportion_5[4:2];
  wire        indexVec_checkResult_overlap_5 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_5 >= indexVec_checkResult_intLMULInput_5[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_5} >= indexVec_checkResult_intLMULInput_5,
      indexVec_checkResult_allDataPosition_5[33:10]};
  wire        indexVec_checkResult_unChange_5 = slideUp & (indexVec_readIndex_1[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_1 = validVec[1] & ~indexVec_checkResult_unChange_5;
  wire        indexVec_checkResult_2_4_1 = indexVec_checkResult_overlap_5 | ~indexVec_checkResult_2_5_1 | lagerThanVL | indexVec_checkResult_unChange_5;
  wire [1:0]  indexVec_1_0 = (sew1H[0] ? indexVec_checkResult_0_0_1 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_1 : 2'h0);
  assign indexVec_1_1 = (sew1H[0] ? indexVec_checkResult_0_1_1 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_1_1 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_1_1 : 2'h0);
  wire [1:0]  indexDeq_bits_accessLane_1_0 = indexVec_1_1;
  wire [2:0]  indexVec_1_2 = (sew1H[0] ? indexVec_checkResult_0_2_1 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_2_1 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_2_1 : 3'h0);
  assign indexVec_1_3 = (sew1H[0] ? indexVec_checkResult_0_3_1 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_1 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_1 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_1_0 = indexVec_1_3;
  wire        indexVec_1_4 = sew1H[0] & indexVec_checkResult_0_4_1 | sew1H[1] & indexVec_checkResult_1_4_1 | sew1H[2] & indexVec_checkResult_2_4_1;
  wire        indexVec_1_5 = sew1H[0] & indexVec_checkResult_0_5_1 | sew1H[1] & indexVec_checkResult_1_5_1 | sew1H[2] & indexVec_checkResult_2_5_1;
  wire [31:0] indexVec_readIndex_2 = baseIndex + 32'h2;
  wire [31:0] indexVec_checkResult_allDataPosition_6 = indexVec_readIndex_2;
  wire [9:0]  indexVec_checkResult_dataPosition_6 = indexVec_checkResult_allDataPosition_6[9:0];
  wire [1:0]  indexVec_checkResult_0_0_2 = indexVec_checkResult_dataPosition_6[1:0];
  wire [1:0]  indexVec_checkResult_0_1_2 = indexVec_checkResult_dataPosition_6[3:2];
  wire [5:0]  indexVec_checkResult_dataGroup_6 = indexVec_checkResult_dataPosition_6[9:4];
  wire [2:0]  indexVec_checkResult_0_2_2 = indexVec_checkResult_dataGroup_6[2:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_6 = indexVec_checkResult_dataGroup_6[5:3];
  wire [2:0]  indexVec_checkResult_0_3_2 = indexVec_checkResult_accessRegGrowth_6;
  wire [4:0]  indexVec_checkResult_decimalProportion_6 = {indexVec_checkResult_0_2_2, indexVec_checkResult_0_1_2};
  wire [2:0]  indexVec_checkResult_decimal_6 = indexVec_checkResult_decimalProportion_6[4:2];
  wire        indexVec_checkResult_overlap_6 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_6 >= indexVec_checkResult_intLMULInput_6[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_6} >= indexVec_checkResult_intLMULInput_6,
      indexVec_checkResult_allDataPosition_6[31:10]};
  wire        indexVec_checkResult_unChange_6 = slideUp & (indexVec_readIndex_2[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_2 = validVec[2] & ~indexVec_checkResult_unChange_6;
  wire        indexVec_checkResult_0_4_2 = indexVec_checkResult_overlap_6 | ~indexVec_checkResult_0_5_2 | lagerThanVL | indexVec_checkResult_unChange_6;
  wire [32:0] indexVec_checkResult_allDataPosition_7 = {indexVec_readIndex_2, 1'h0};
  wire [9:0]  indexVec_checkResult_dataPosition_7 = indexVec_checkResult_allDataPosition_7[9:0];
  wire [1:0]  indexVec_checkResult_1_0_2 = {indexVec_checkResult_dataPosition_7[1], 1'h0};
  wire [1:0]  indexVec_checkResult_1_1_2 = indexVec_checkResult_dataPosition_7[3:2];
  wire [5:0]  indexVec_checkResult_dataGroup_7 = indexVec_checkResult_dataPosition_7[9:4];
  wire [2:0]  indexVec_checkResult_1_2_2 = indexVec_checkResult_dataGroup_7[2:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_7 = indexVec_checkResult_dataGroup_7[5:3];
  wire [2:0]  indexVec_checkResult_1_3_2 = indexVec_checkResult_accessRegGrowth_7;
  wire [4:0]  indexVec_checkResult_decimalProportion_7 = {indexVec_checkResult_1_2_2, indexVec_checkResult_1_1_2};
  wire [2:0]  indexVec_checkResult_decimal_7 = indexVec_checkResult_decimalProportion_7[4:2];
  wire        indexVec_checkResult_overlap_7 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_7 >= indexVec_checkResult_intLMULInput_7[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_7} >= indexVec_checkResult_intLMULInput_7,
      indexVec_checkResult_allDataPosition_7[32:10]};
  wire        indexVec_checkResult_unChange_7 = slideUp & (indexVec_readIndex_2[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_2 = validVec[2] & ~indexVec_checkResult_unChange_7;
  wire        indexVec_checkResult_1_4_2 = indexVec_checkResult_overlap_7 | ~indexVec_checkResult_1_5_2 | lagerThanVL | indexVec_checkResult_unChange_7;
  wire [33:0] indexVec_checkResult_allDataPosition_8 = {indexVec_readIndex_2, 2'h0};
  wire [9:0]  indexVec_checkResult_dataPosition_8 = indexVec_checkResult_allDataPosition_8[9:0];
  wire [1:0]  indexVec_checkResult_2_1_2 = indexVec_checkResult_dataPosition_8[3:2];
  wire [5:0]  indexVec_checkResult_dataGroup_8 = indexVec_checkResult_dataPosition_8[9:4];
  wire [2:0]  indexVec_checkResult_2_2_2 = indexVec_checkResult_dataGroup_8[2:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_8 = indexVec_checkResult_dataGroup_8[5:3];
  wire [2:0]  indexVec_checkResult_2_3_2 = indexVec_checkResult_accessRegGrowth_8;
  wire [4:0]  indexVec_checkResult_decimalProportion_8 = {indexVec_checkResult_2_2_2, indexVec_checkResult_2_1_2};
  wire [2:0]  indexVec_checkResult_decimal_8 = indexVec_checkResult_decimalProportion_8[4:2];
  wire        indexVec_checkResult_overlap_8 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_8 >= indexVec_checkResult_intLMULInput_8[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_8} >= indexVec_checkResult_intLMULInput_8,
      indexVec_checkResult_allDataPosition_8[33:10]};
  wire        indexVec_checkResult_unChange_8 = slideUp & (indexVec_readIndex_2[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_2 = validVec[2] & ~indexVec_checkResult_unChange_8;
  wire        indexVec_checkResult_2_4_2 = indexVec_checkResult_overlap_8 | ~indexVec_checkResult_2_5_2 | lagerThanVL | indexVec_checkResult_unChange_8;
  wire [1:0]  indexVec_2_0 = (sew1H[0] ? indexVec_checkResult_0_0_2 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_2 : 2'h0);
  assign indexVec_2_1 = (sew1H[0] ? indexVec_checkResult_0_1_2 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_1_2 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_1_2 : 2'h0);
  wire [1:0]  indexDeq_bits_accessLane_2_0 = indexVec_2_1;
  wire [2:0]  indexVec_2_2 = (sew1H[0] ? indexVec_checkResult_0_2_2 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_2_2 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_2_2 : 3'h0);
  assign indexVec_2_3 = (sew1H[0] ? indexVec_checkResult_0_3_2 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_2 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_2 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_2_0 = indexVec_2_3;
  wire        indexVec_2_4 = sew1H[0] & indexVec_checkResult_0_4_2 | sew1H[1] & indexVec_checkResult_1_4_2 | sew1H[2] & indexVec_checkResult_2_4_2;
  wire        indexVec_2_5 = sew1H[0] & indexVec_checkResult_0_5_2 | sew1H[1] & indexVec_checkResult_1_5_2 | sew1H[2] & indexVec_checkResult_2_5_2;
  wire [31:0] indexVec_readIndex_3 = baseIndex + 32'h3;
  wire [31:0] indexVec_checkResult_allDataPosition_9 = indexVec_readIndex_3;
  wire [9:0]  indexVec_checkResult_dataPosition_9 = indexVec_checkResult_allDataPosition_9[9:0];
  wire [1:0]  indexVec_checkResult_0_0_3 = indexVec_checkResult_dataPosition_9[1:0];
  wire [1:0]  indexVec_checkResult_0_1_3 = indexVec_checkResult_dataPosition_9[3:2];
  wire [5:0]  indexVec_checkResult_dataGroup_9 = indexVec_checkResult_dataPosition_9[9:4];
  wire [2:0]  indexVec_checkResult_0_2_3 = indexVec_checkResult_dataGroup_9[2:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_9 = indexVec_checkResult_dataGroup_9[5:3];
  wire [2:0]  indexVec_checkResult_0_3_3 = indexVec_checkResult_accessRegGrowth_9;
  wire [4:0]  indexVec_checkResult_decimalProportion_9 = {indexVec_checkResult_0_2_3, indexVec_checkResult_0_1_3};
  wire [2:0]  indexVec_checkResult_decimal_9 = indexVec_checkResult_decimalProportion_9[4:2];
  wire        indexVec_checkResult_overlap_9 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_9 >= indexVec_checkResult_intLMULInput_9[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_9} >= indexVec_checkResult_intLMULInput_9,
      indexVec_checkResult_allDataPosition_9[31:10]};
  wire        indexVec_checkResult_unChange_9 = slideUp & (indexVec_readIndex_3[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_3 = validVec[3] & ~indexVec_checkResult_unChange_9;
  wire        indexVec_checkResult_0_4_3 = indexVec_checkResult_overlap_9 | ~indexVec_checkResult_0_5_3 | lagerThanVL | indexVec_checkResult_unChange_9;
  wire [32:0] indexVec_checkResult_allDataPosition_10 = {indexVec_readIndex_3, 1'h0};
  wire [9:0]  indexVec_checkResult_dataPosition_10 = indexVec_checkResult_allDataPosition_10[9:0];
  wire [1:0]  indexVec_checkResult_1_0_3 = {indexVec_checkResult_dataPosition_10[1], 1'h0};
  wire [1:0]  indexVec_checkResult_1_1_3 = indexVec_checkResult_dataPosition_10[3:2];
  wire [5:0]  indexVec_checkResult_dataGroup_10 = indexVec_checkResult_dataPosition_10[9:4];
  wire [2:0]  indexVec_checkResult_1_2_3 = indexVec_checkResult_dataGroup_10[2:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_10 = indexVec_checkResult_dataGroup_10[5:3];
  wire [2:0]  indexVec_checkResult_1_3_3 = indexVec_checkResult_accessRegGrowth_10;
  wire [4:0]  indexVec_checkResult_decimalProportion_10 = {indexVec_checkResult_1_2_3, indexVec_checkResult_1_1_3};
  wire [2:0]  indexVec_checkResult_decimal_10 = indexVec_checkResult_decimalProportion_10[4:2];
  wire        indexVec_checkResult_overlap_10 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_10 >= indexVec_checkResult_intLMULInput_10[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_10} >= indexVec_checkResult_intLMULInput_10,
      indexVec_checkResult_allDataPosition_10[32:10]};
  wire        indexVec_checkResult_unChange_10 = slideUp & (indexVec_readIndex_3[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_3 = validVec[3] & ~indexVec_checkResult_unChange_10;
  wire        indexVec_checkResult_1_4_3 = indexVec_checkResult_overlap_10 | ~indexVec_checkResult_1_5_3 | lagerThanVL | indexVec_checkResult_unChange_10;
  wire [33:0] indexVec_checkResult_allDataPosition_11 = {indexVec_readIndex_3, 2'h0};
  wire [9:0]  indexVec_checkResult_dataPosition_11 = indexVec_checkResult_allDataPosition_11[9:0];
  wire [1:0]  indexVec_checkResult_2_1_3 = indexVec_checkResult_dataPosition_11[3:2];
  wire [5:0]  indexVec_checkResult_dataGroup_11 = indexVec_checkResult_dataPosition_11[9:4];
  wire [2:0]  indexVec_checkResult_2_2_3 = indexVec_checkResult_dataGroup_11[2:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_11 = indexVec_checkResult_dataGroup_11[5:3];
  wire [2:0]  indexVec_checkResult_2_3_3 = indexVec_checkResult_accessRegGrowth_11;
  wire [4:0]  indexVec_checkResult_decimalProportion_11 = {indexVec_checkResult_2_2_3, indexVec_checkResult_2_1_3};
  wire [2:0]  indexVec_checkResult_decimal_11 = indexVec_checkResult_decimalProportion_11[4:2];
  wire        indexVec_checkResult_overlap_11 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_11 >= indexVec_checkResult_intLMULInput_11[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_11} >= indexVec_checkResult_intLMULInput_11,
      indexVec_checkResult_allDataPosition_11[33:10]};
  wire        indexVec_checkResult_unChange_11 = slideUp & (indexVec_readIndex_3[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_3 = validVec[3] & ~indexVec_checkResult_unChange_11;
  wire        indexVec_checkResult_2_4_3 = indexVec_checkResult_overlap_11 | ~indexVec_checkResult_2_5_3 | lagerThanVL | indexVec_checkResult_unChange_11;
  wire [1:0]  indexVec_3_0 = (sew1H[0] ? indexVec_checkResult_0_0_3 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_3 : 2'h0);
  assign indexVec_3_1 = (sew1H[0] ? indexVec_checkResult_0_1_3 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_1_3 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_1_3 : 2'h0);
  wire [1:0]  indexDeq_bits_accessLane_3_0 = indexVec_3_1;
  wire [2:0]  indexVec_3_2 = (sew1H[0] ? indexVec_checkResult_0_2_3 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_2_3 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_2_3 : 3'h0);
  assign indexVec_3_3 = (sew1H[0] ? indexVec_checkResult_0_3_3 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_3 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_3 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_3_0 = indexVec_3_3;
  wire        indexVec_3_4 = sew1H[0] & indexVec_checkResult_0_4_3 | sew1H[1] & indexVec_checkResult_1_4_3 | sew1H[2] & indexVec_checkResult_2_4_3;
  wire        indexVec_3_5 = sew1H[0] & indexVec_checkResult_0_5_3 | sew1H[1] & indexVec_checkResult_1_5_3 | sew1H[2] & indexVec_checkResult_2_5_3;
  assign indexDeq_valid_0 = InstructionValid & isSlide;
  wire [1:0]  indexDeq_bits_needRead_lo = {~indexVec_1_4, ~indexVec_0_4};
  wire [1:0]  indexDeq_bits_needRead_hi = {~indexVec_3_4, ~indexVec_2_4};
  wire [3:0]  indexDeq_bits_needRead_0 = {indexDeq_bits_needRead_hi, indexDeq_bits_needRead_lo} & ~replaceWithVs1;
  wire [1:0]  indexDeq_bits_elementValid_lo = {indexVec_1_5, indexVec_0_5};
  wire [1:0]  indexDeq_bits_elementValid_hi = {indexVec_3_5, indexVec_2_5};
  wire [3:0]  indexDeq_bits_elementValid_0 = {indexDeq_bits_elementValid_hi, indexDeq_bits_elementValid_lo} | replaceWithVs1;
  wire [5:0]  indexDeq_bits_readOffset_lo = {indexVec_1_2, indexVec_0_2};
  wire [5:0]  indexDeq_bits_readOffset_hi = {indexVec_3_2, indexVec_2_2};
  wire [11:0] indexDeq_bits_readOffset_0 = {indexDeq_bits_readOffset_hi, indexDeq_bits_readOffset_lo};
  wire [3:0]  indexDeq_bits_readDataOffset_lo = {indexVec_1_0, indexVec_0_0};
  wire [3:0]  indexDeq_bits_readDataOffset_hi = {indexVec_3_0, indexVec_2_0};
  wire [7:0]  indexDeq_bits_readDataOffset_0 = {indexDeq_bits_readDataOffset_hi, indexDeq_bits_readDataOffset_lo};
  always @(posedge clock) begin
    if (reset) begin
      InstructionValid <= 1'h0;
      slideGroup <= 9'h0;
    end
    else begin
      if (newInstruction | lastFire)
        InstructionValid <= newInstruction;
      if (newInstruction | _lastFire_T_1)
        slideGroup <= newInstruction ? 9'h0 : slideGroup + 9'h1;
    end
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:0];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        _RANDOM[/*Zero width*/ 1'b0] = `RANDOM;
        InstructionValid = _RANDOM[/*Zero width*/ 1'b0][0];
        slideGroup = _RANDOM[/*Zero width*/ 1'b0][9:1];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  assign indexDeq_valid = indexDeq_valid_0;
  assign indexDeq_bits_needRead = indexDeq_bits_needRead_0;
  assign indexDeq_bits_elementValid = indexDeq_bits_elementValid_0;
  assign indexDeq_bits_replaceVs1 = indexDeq_bits_replaceVs1_0;
  assign indexDeq_bits_readOffset = indexDeq_bits_readOffset_0;
  assign indexDeq_bits_accessLane_0 = indexDeq_bits_accessLane_0_0;
  assign indexDeq_bits_accessLane_1 = indexDeq_bits_accessLane_1_0;
  assign indexDeq_bits_accessLane_2 = indexDeq_bits_accessLane_2_0;
  assign indexDeq_bits_accessLane_3 = indexDeq_bits_accessLane_3_0;
  assign indexDeq_bits_vsGrowth_0 = indexDeq_bits_vsGrowth_0_0;
  assign indexDeq_bits_vsGrowth_1 = indexDeq_bits_vsGrowth_1_0;
  assign indexDeq_bits_vsGrowth_2 = indexDeq_bits_vsGrowth_2_0;
  assign indexDeq_bits_vsGrowth_3 = indexDeq_bits_vsGrowth_3_0;
  assign indexDeq_bits_executeGroup = indexDeq_bits_executeGroup_0;
  assign indexDeq_bits_readDataOffset = indexDeq_bits_readDataOffset_0;
  assign indexDeq_bits_last = indexDeq_bits_last_0;
  assign slideGroupOut = slideGroup;
endmodule

