
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
  output [7:0]  indexDeq_bits_needRead,
                indexDeq_bits_elementValid,
                indexDeq_bits_replaceVs1,
  output [15:0] indexDeq_bits_readOffset,
  output [2:0]  indexDeq_bits_accessLane_0,
                indexDeq_bits_accessLane_1,
                indexDeq_bits_accessLane_2,
                indexDeq_bits_accessLane_3,
                indexDeq_bits_accessLane_4,
                indexDeq_bits_accessLane_5,
                indexDeq_bits_accessLane_6,
                indexDeq_bits_accessLane_7,
                indexDeq_bits_vsGrowth_0,
                indexDeq_bits_vsGrowth_1,
                indexDeq_bits_vsGrowth_2,
                indexDeq_bits_vsGrowth_3,
                indexDeq_bits_vsGrowth_4,
                indexDeq_bits_vsGrowth_5,
                indexDeq_bits_vsGrowth_6,
                indexDeq_bits_vsGrowth_7,
  output [7:0]  indexDeq_bits_executeGroup,
  output [15:0] indexDeq_bits_readDataOffset,
  output        indexDeq_bits_last,
  output [7:0]  slideGroupOut,
  input  [7:0]  slideMaskInput
);

  wire        indexDeq_ready_0 = indexDeq_ready;
  wire [1:0]  indexVec_checkResult_2_0 = 2'h0;
  wire [1:0]  indexVec_checkResult_2_0_1 = 2'h0;
  wire [1:0]  indexVec_checkResult_2_0_2 = 2'h0;
  wire [1:0]  indexVec_checkResult_2_0_3 = 2'h0;
  wire [1:0]  indexVec_checkResult_2_0_4 = 2'h0;
  wire [1:0]  indexVec_checkResult_2_0_5 = 2'h0;
  wire [1:0]  indexVec_checkResult_2_0_6 = 2'h0;
  wire [1:0]  indexVec_checkResult_2_0_7 = 2'h0;
  wire [7:0]  indexDeq_bits_groupReadState = 8'h0;
  wire [7:0]  replaceWithVs1;
  wire [2:0]  indexVec_0_1;
  wire [2:0]  indexVec_1_1;
  wire [2:0]  indexVec_2_1;
  wire [2:0]  indexVec_3_1;
  wire [2:0]  indexVec_4_1;
  wire [2:0]  indexVec_5_1;
  wire [2:0]  indexVec_6_1;
  wire [2:0]  indexVec_7_1;
  wire [2:0]  indexVec_0_3;
  wire [2:0]  indexVec_1_3;
  wire [2:0]  indexVec_2_3;
  wire [2:0]  indexVec_3_3;
  wire [2:0]  indexVec_4_3;
  wire [2:0]  indexVec_5_3;
  wire [2:0]  indexVec_6_3;
  wire [2:0]  indexVec_7_3;
  reg         InstructionValid;
  wire        isSlide = instructionReq_decodeResult_topUop[4:2] == 3'h0;
  wire        slideUp = instructionReq_decodeResult_topUop[0];
  wire        slide1 = instructionReq_decodeResult_topUop[1];
  reg  [7:0]  slideGroup;
  wire [7:0]  indexDeq_bits_executeGroup_0 = slideGroup;
  wire [2:0]  vlTail = instructionReq_vl[2:0];
  wire [7:0]  lastSlideGroup = instructionReq_vl[10:3] - {7'h0, vlTail == 3'h0};
  wire [7:0]  _lastValidVec_T = 8'h1 << vlTail;
  wire [7:0]  _lastValidVec_T_3 = _lastValidVec_T | {_lastValidVec_T[6:0], 1'h0};
  wire [7:0]  _lastValidVec_T_6 = _lastValidVec_T_3 | {_lastValidVec_T_3[5:0], 2'h0};
  wire [7:0]  lastValidVec = ~(_lastValidVec_T_6 | {_lastValidVec_T_6[3:0], 4'h0});
  wire        indexDeq_bits_last_0 = slideGroup == lastSlideGroup;
  wire [7:0]  groupVlValid = indexDeq_bits_last_0 & (|vlTail) ? lastValidVec : 8'hFF;
  wire [7:0]  groupMaskValid = instructionReq_maskType ? slideMaskInput : 8'hFF;
  wire [7:0]  validVec = groupVlValid & groupMaskValid;
  wire [7:0]  lastElementValid = ({1'h0, groupVlValid[7:1]} ^ groupVlValid) & groupMaskValid;
  assign replaceWithVs1 = (slideGroup == 8'h0 & slide1 & slideUp ? {7'h0, groupMaskValid[0]} : 8'h0) | (indexDeq_bits_last_0 & slide1 & ~slideUp ? lastElementValid : 8'h0);
  wire        indexDeq_valid_0;
  wire [7:0]  indexDeq_bits_replaceVs1_0 = replaceWithVs1;
  wire        _lastFire_T_1 = indexDeq_ready_0 & indexDeq_valid_0;
  wire        lastFire = indexDeq_bits_last_0 & _lastFire_T_1;
  wire [31:0] slideValue = slide1 ? 32'h1 : instructionReq_readFromScala;
  wire [31:0] PNSelect = {32{slideUp}} ^ slideValue;
  wire [31:0] baseIndex = {21'h0, slideGroup, 3'h0} + PNSelect + {31'h0, slideUp};
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
  wire [3:0]  indexVec_checkResult_intLMULInput_12;
  assign indexVec_checkResult_intLMULInput_12 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_13;
  assign indexVec_checkResult_intLMULInput_13 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_14;
  assign indexVec_checkResult_intLMULInput_14 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_15;
  assign indexVec_checkResult_intLMULInput_15 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_16;
  assign indexVec_checkResult_intLMULInput_16 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_17;
  assign indexVec_checkResult_intLMULInput_17 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_18;
  assign indexVec_checkResult_intLMULInput_18 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_19;
  assign indexVec_checkResult_intLMULInput_19 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_20;
  assign indexVec_checkResult_intLMULInput_20 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_21;
  assign indexVec_checkResult_intLMULInput_21 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_22;
  assign indexVec_checkResult_intLMULInput_22 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_23;
  assign indexVec_checkResult_intLMULInput_23 = _GEN;
  wire [9:0]  indexVec_checkResult_dataPosition = indexVec_checkResult_allDataPosition[9:0];
  wire [1:0]  indexVec_checkResult_0_0 = indexVec_checkResult_dataPosition[1:0];
  wire [2:0]  indexVec_checkResult_0_1 = indexVec_checkResult_dataPosition[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup = indexVec_checkResult_dataPosition[9:5];
  wire [1:0]  indexVec_checkResult_0_2 = indexVec_checkResult_dataGroup[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth = indexVec_checkResult_dataGroup[4:2];
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
  wire [2:0]  indexVec_checkResult_1_1 = indexVec_checkResult_dataPosition_1[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_1 = indexVec_checkResult_dataPosition_1[9:5];
  wire [1:0]  indexVec_checkResult_1_2 = indexVec_checkResult_dataGroup_1[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_1 = indexVec_checkResult_dataGroup_1[4:2];
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
  wire [2:0]  indexVec_checkResult_2_1 = indexVec_checkResult_dataPosition_2[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_2 = indexVec_checkResult_dataPosition_2[9:5];
  wire [1:0]  indexVec_checkResult_2_2 = indexVec_checkResult_dataGroup_2[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_2 = indexVec_checkResult_dataGroup_2[4:2];
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
  assign indexVec_0_1 = (sew1H[0] ? indexVec_checkResult_0_1 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_1 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_1 : 3'h0);
  wire [2:0]  indexDeq_bits_accessLane_0_0 = indexVec_0_1;
  wire [1:0]  indexVec_0_2 = (sew1H[0] ? indexVec_checkResult_0_2 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2 : 2'h0);
  assign indexVec_0_3 = (sew1H[0] ? indexVec_checkResult_0_3 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_0_0 = indexVec_0_3;
  wire        indexVec_0_4 = sew1H[0] & indexVec_checkResult_0_4 | sew1H[1] & indexVec_checkResult_1_4 | sew1H[2] & indexVec_checkResult_2_4;
  wire        indexVec_0_5 = sew1H[0] & indexVec_checkResult_0_5 | sew1H[1] & indexVec_checkResult_1_5 | sew1H[2] & indexVec_checkResult_2_5;
  wire [31:0] indexVec_readIndex_1 = baseIndex + 32'h1;
  wire [31:0] indexVec_checkResult_allDataPosition_3 = indexVec_readIndex_1;
  wire [9:0]  indexVec_checkResult_dataPosition_3 = indexVec_checkResult_allDataPosition_3[9:0];
  wire [1:0]  indexVec_checkResult_0_0_1 = indexVec_checkResult_dataPosition_3[1:0];
  wire [2:0]  indexVec_checkResult_0_1_1 = indexVec_checkResult_dataPosition_3[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_3 = indexVec_checkResult_dataPosition_3[9:5];
  wire [1:0]  indexVec_checkResult_0_2_1 = indexVec_checkResult_dataGroup_3[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_3 = indexVec_checkResult_dataGroup_3[4:2];
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
  wire [2:0]  indexVec_checkResult_1_1_1 = indexVec_checkResult_dataPosition_4[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_4 = indexVec_checkResult_dataPosition_4[9:5];
  wire [1:0]  indexVec_checkResult_1_2_1 = indexVec_checkResult_dataGroup_4[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_4 = indexVec_checkResult_dataGroup_4[4:2];
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
  wire [2:0]  indexVec_checkResult_2_1_1 = indexVec_checkResult_dataPosition_5[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_5 = indexVec_checkResult_dataPosition_5[9:5];
  wire [1:0]  indexVec_checkResult_2_2_1 = indexVec_checkResult_dataGroup_5[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_5 = indexVec_checkResult_dataGroup_5[4:2];
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
  assign indexVec_1_1 = (sew1H[0] ? indexVec_checkResult_0_1_1 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_1_1 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_1_1 : 3'h0);
  wire [2:0]  indexDeq_bits_accessLane_1_0 = indexVec_1_1;
  wire [1:0]  indexVec_1_2 = (sew1H[0] ? indexVec_checkResult_0_2_1 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_1 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_1 : 2'h0);
  assign indexVec_1_3 = (sew1H[0] ? indexVec_checkResult_0_3_1 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_1 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_1 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_1_0 = indexVec_1_3;
  wire        indexVec_1_4 = sew1H[0] & indexVec_checkResult_0_4_1 | sew1H[1] & indexVec_checkResult_1_4_1 | sew1H[2] & indexVec_checkResult_2_4_1;
  wire        indexVec_1_5 = sew1H[0] & indexVec_checkResult_0_5_1 | sew1H[1] & indexVec_checkResult_1_5_1 | sew1H[2] & indexVec_checkResult_2_5_1;
  wire [31:0] indexVec_readIndex_2 = baseIndex + 32'h2;
  wire [31:0] indexVec_checkResult_allDataPosition_6 = indexVec_readIndex_2;
  wire [9:0]  indexVec_checkResult_dataPosition_6 = indexVec_checkResult_allDataPosition_6[9:0];
  wire [1:0]  indexVec_checkResult_0_0_2 = indexVec_checkResult_dataPosition_6[1:0];
  wire [2:0]  indexVec_checkResult_0_1_2 = indexVec_checkResult_dataPosition_6[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_6 = indexVec_checkResult_dataPosition_6[9:5];
  wire [1:0]  indexVec_checkResult_0_2_2 = indexVec_checkResult_dataGroup_6[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_6 = indexVec_checkResult_dataGroup_6[4:2];
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
  wire [2:0]  indexVec_checkResult_1_1_2 = indexVec_checkResult_dataPosition_7[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_7 = indexVec_checkResult_dataPosition_7[9:5];
  wire [1:0]  indexVec_checkResult_1_2_2 = indexVec_checkResult_dataGroup_7[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_7 = indexVec_checkResult_dataGroup_7[4:2];
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
  wire [2:0]  indexVec_checkResult_2_1_2 = indexVec_checkResult_dataPosition_8[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_8 = indexVec_checkResult_dataPosition_8[9:5];
  wire [1:0]  indexVec_checkResult_2_2_2 = indexVec_checkResult_dataGroup_8[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_8 = indexVec_checkResult_dataGroup_8[4:2];
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
  assign indexVec_2_1 = (sew1H[0] ? indexVec_checkResult_0_1_2 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_1_2 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_1_2 : 3'h0);
  wire [2:0]  indexDeq_bits_accessLane_2_0 = indexVec_2_1;
  wire [1:0]  indexVec_2_2 = (sew1H[0] ? indexVec_checkResult_0_2_2 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_2 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_2 : 2'h0);
  assign indexVec_2_3 = (sew1H[0] ? indexVec_checkResult_0_3_2 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_2 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_2 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_2_0 = indexVec_2_3;
  wire        indexVec_2_4 = sew1H[0] & indexVec_checkResult_0_4_2 | sew1H[1] & indexVec_checkResult_1_4_2 | sew1H[2] & indexVec_checkResult_2_4_2;
  wire        indexVec_2_5 = sew1H[0] & indexVec_checkResult_0_5_2 | sew1H[1] & indexVec_checkResult_1_5_2 | sew1H[2] & indexVec_checkResult_2_5_2;
  wire [31:0] indexVec_readIndex_3 = baseIndex + 32'h3;
  wire [31:0] indexVec_checkResult_allDataPosition_9 = indexVec_readIndex_3;
  wire [9:0]  indexVec_checkResult_dataPosition_9 = indexVec_checkResult_allDataPosition_9[9:0];
  wire [1:0]  indexVec_checkResult_0_0_3 = indexVec_checkResult_dataPosition_9[1:0];
  wire [2:0]  indexVec_checkResult_0_1_3 = indexVec_checkResult_dataPosition_9[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_9 = indexVec_checkResult_dataPosition_9[9:5];
  wire [1:0]  indexVec_checkResult_0_2_3 = indexVec_checkResult_dataGroup_9[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_9 = indexVec_checkResult_dataGroup_9[4:2];
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
  wire [2:0]  indexVec_checkResult_1_1_3 = indexVec_checkResult_dataPosition_10[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_10 = indexVec_checkResult_dataPosition_10[9:5];
  wire [1:0]  indexVec_checkResult_1_2_3 = indexVec_checkResult_dataGroup_10[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_10 = indexVec_checkResult_dataGroup_10[4:2];
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
  wire [2:0]  indexVec_checkResult_2_1_3 = indexVec_checkResult_dataPosition_11[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_11 = indexVec_checkResult_dataPosition_11[9:5];
  wire [1:0]  indexVec_checkResult_2_2_3 = indexVec_checkResult_dataGroup_11[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_11 = indexVec_checkResult_dataGroup_11[4:2];
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
  assign indexVec_3_1 = (sew1H[0] ? indexVec_checkResult_0_1_3 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_1_3 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_1_3 : 3'h0);
  wire [2:0]  indexDeq_bits_accessLane_3_0 = indexVec_3_1;
  wire [1:0]  indexVec_3_2 = (sew1H[0] ? indexVec_checkResult_0_2_3 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_3 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_3 : 2'h0);
  assign indexVec_3_3 = (sew1H[0] ? indexVec_checkResult_0_3_3 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_3 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_3 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_3_0 = indexVec_3_3;
  wire        indexVec_3_4 = sew1H[0] & indexVec_checkResult_0_4_3 | sew1H[1] & indexVec_checkResult_1_4_3 | sew1H[2] & indexVec_checkResult_2_4_3;
  wire        indexVec_3_5 = sew1H[0] & indexVec_checkResult_0_5_3 | sew1H[1] & indexVec_checkResult_1_5_3 | sew1H[2] & indexVec_checkResult_2_5_3;
  wire [31:0] indexVec_readIndex_4 = baseIndex + 32'h4;
  wire [31:0] indexVec_checkResult_allDataPosition_12 = indexVec_readIndex_4;
  wire [9:0]  indexVec_checkResult_dataPosition_12 = indexVec_checkResult_allDataPosition_12[9:0];
  wire [1:0]  indexVec_checkResult_0_0_4 = indexVec_checkResult_dataPosition_12[1:0];
  wire [2:0]  indexVec_checkResult_0_1_4 = indexVec_checkResult_dataPosition_12[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_12 = indexVec_checkResult_dataPosition_12[9:5];
  wire [1:0]  indexVec_checkResult_0_2_4 = indexVec_checkResult_dataGroup_12[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_12 = indexVec_checkResult_dataGroup_12[4:2];
  wire [2:0]  indexVec_checkResult_0_3_4 = indexVec_checkResult_accessRegGrowth_12;
  wire [4:0]  indexVec_checkResult_decimalProportion_12 = {indexVec_checkResult_0_2_4, indexVec_checkResult_0_1_4};
  wire [2:0]  indexVec_checkResult_decimal_12 = indexVec_checkResult_decimalProportion_12[4:2];
  wire        indexVec_checkResult_overlap_12 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_12 >= indexVec_checkResult_intLMULInput_12[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_12} >= indexVec_checkResult_intLMULInput_12,
      indexVec_checkResult_allDataPosition_12[31:10]};
  wire        indexVec_checkResult_unChange_12 = slideUp & (indexVec_readIndex_4[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_4 = validVec[4] & ~indexVec_checkResult_unChange_12;
  wire        indexVec_checkResult_0_4_4 = indexVec_checkResult_overlap_12 | ~indexVec_checkResult_0_5_4 | lagerThanVL | indexVec_checkResult_unChange_12;
  wire [32:0] indexVec_checkResult_allDataPosition_13 = {indexVec_readIndex_4, 1'h0};
  wire [9:0]  indexVec_checkResult_dataPosition_13 = indexVec_checkResult_allDataPosition_13[9:0];
  wire [1:0]  indexVec_checkResult_1_0_4 = {indexVec_checkResult_dataPosition_13[1], 1'h0};
  wire [2:0]  indexVec_checkResult_1_1_4 = indexVec_checkResult_dataPosition_13[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_13 = indexVec_checkResult_dataPosition_13[9:5];
  wire [1:0]  indexVec_checkResult_1_2_4 = indexVec_checkResult_dataGroup_13[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_13 = indexVec_checkResult_dataGroup_13[4:2];
  wire [2:0]  indexVec_checkResult_1_3_4 = indexVec_checkResult_accessRegGrowth_13;
  wire [4:0]  indexVec_checkResult_decimalProportion_13 = {indexVec_checkResult_1_2_4, indexVec_checkResult_1_1_4};
  wire [2:0]  indexVec_checkResult_decimal_13 = indexVec_checkResult_decimalProportion_13[4:2];
  wire        indexVec_checkResult_overlap_13 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_13 >= indexVec_checkResult_intLMULInput_13[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_13} >= indexVec_checkResult_intLMULInput_13,
      indexVec_checkResult_allDataPosition_13[32:10]};
  wire        indexVec_checkResult_unChange_13 = slideUp & (indexVec_readIndex_4[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_4 = validVec[4] & ~indexVec_checkResult_unChange_13;
  wire        indexVec_checkResult_1_4_4 = indexVec_checkResult_overlap_13 | ~indexVec_checkResult_1_5_4 | lagerThanVL | indexVec_checkResult_unChange_13;
  wire [33:0] indexVec_checkResult_allDataPosition_14 = {indexVec_readIndex_4, 2'h0};
  wire [9:0]  indexVec_checkResult_dataPosition_14 = indexVec_checkResult_allDataPosition_14[9:0];
  wire [2:0]  indexVec_checkResult_2_1_4 = indexVec_checkResult_dataPosition_14[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_14 = indexVec_checkResult_dataPosition_14[9:5];
  wire [1:0]  indexVec_checkResult_2_2_4 = indexVec_checkResult_dataGroup_14[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_14 = indexVec_checkResult_dataGroup_14[4:2];
  wire [2:0]  indexVec_checkResult_2_3_4 = indexVec_checkResult_accessRegGrowth_14;
  wire [4:0]  indexVec_checkResult_decimalProportion_14 = {indexVec_checkResult_2_2_4, indexVec_checkResult_2_1_4};
  wire [2:0]  indexVec_checkResult_decimal_14 = indexVec_checkResult_decimalProportion_14[4:2];
  wire        indexVec_checkResult_overlap_14 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_14 >= indexVec_checkResult_intLMULInput_14[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_14} >= indexVec_checkResult_intLMULInput_14,
      indexVec_checkResult_allDataPosition_14[33:10]};
  wire        indexVec_checkResult_unChange_14 = slideUp & (indexVec_readIndex_4[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_4 = validVec[4] & ~indexVec_checkResult_unChange_14;
  wire        indexVec_checkResult_2_4_4 = indexVec_checkResult_overlap_14 | ~indexVec_checkResult_2_5_4 | lagerThanVL | indexVec_checkResult_unChange_14;
  wire [1:0]  indexVec_4_0 = (sew1H[0] ? indexVec_checkResult_0_0_4 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_4 : 2'h0);
  assign indexVec_4_1 = (sew1H[0] ? indexVec_checkResult_0_1_4 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_1_4 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_1_4 : 3'h0);
  wire [2:0]  indexDeq_bits_accessLane_4_0 = indexVec_4_1;
  wire [1:0]  indexVec_4_2 = (sew1H[0] ? indexVec_checkResult_0_2_4 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_4 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_4 : 2'h0);
  assign indexVec_4_3 = (sew1H[0] ? indexVec_checkResult_0_3_4 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_4 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_4 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_4_0 = indexVec_4_3;
  wire        indexVec_4_4 = sew1H[0] & indexVec_checkResult_0_4_4 | sew1H[1] & indexVec_checkResult_1_4_4 | sew1H[2] & indexVec_checkResult_2_4_4;
  wire        indexVec_4_5 = sew1H[0] & indexVec_checkResult_0_5_4 | sew1H[1] & indexVec_checkResult_1_5_4 | sew1H[2] & indexVec_checkResult_2_5_4;
  wire [31:0] indexVec_readIndex_5 = baseIndex + 32'h5;
  wire [31:0] indexVec_checkResult_allDataPosition_15 = indexVec_readIndex_5;
  wire [9:0]  indexVec_checkResult_dataPosition_15 = indexVec_checkResult_allDataPosition_15[9:0];
  wire [1:0]  indexVec_checkResult_0_0_5 = indexVec_checkResult_dataPosition_15[1:0];
  wire [2:0]  indexVec_checkResult_0_1_5 = indexVec_checkResult_dataPosition_15[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_15 = indexVec_checkResult_dataPosition_15[9:5];
  wire [1:0]  indexVec_checkResult_0_2_5 = indexVec_checkResult_dataGroup_15[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_15 = indexVec_checkResult_dataGroup_15[4:2];
  wire [2:0]  indexVec_checkResult_0_3_5 = indexVec_checkResult_accessRegGrowth_15;
  wire [4:0]  indexVec_checkResult_decimalProportion_15 = {indexVec_checkResult_0_2_5, indexVec_checkResult_0_1_5};
  wire [2:0]  indexVec_checkResult_decimal_15 = indexVec_checkResult_decimalProportion_15[4:2];
  wire        indexVec_checkResult_overlap_15 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_15 >= indexVec_checkResult_intLMULInput_15[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_15} >= indexVec_checkResult_intLMULInput_15,
      indexVec_checkResult_allDataPosition_15[31:10]};
  wire        indexVec_checkResult_unChange_15 = slideUp & (indexVec_readIndex_5[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_5 = validVec[5] & ~indexVec_checkResult_unChange_15;
  wire        indexVec_checkResult_0_4_5 = indexVec_checkResult_overlap_15 | ~indexVec_checkResult_0_5_5 | lagerThanVL | indexVec_checkResult_unChange_15;
  wire [32:0] indexVec_checkResult_allDataPosition_16 = {indexVec_readIndex_5, 1'h0};
  wire [9:0]  indexVec_checkResult_dataPosition_16 = indexVec_checkResult_allDataPosition_16[9:0];
  wire [1:0]  indexVec_checkResult_1_0_5 = {indexVec_checkResult_dataPosition_16[1], 1'h0};
  wire [2:0]  indexVec_checkResult_1_1_5 = indexVec_checkResult_dataPosition_16[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_16 = indexVec_checkResult_dataPosition_16[9:5];
  wire [1:0]  indexVec_checkResult_1_2_5 = indexVec_checkResult_dataGroup_16[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_16 = indexVec_checkResult_dataGroup_16[4:2];
  wire [2:0]  indexVec_checkResult_1_3_5 = indexVec_checkResult_accessRegGrowth_16;
  wire [4:0]  indexVec_checkResult_decimalProportion_16 = {indexVec_checkResult_1_2_5, indexVec_checkResult_1_1_5};
  wire [2:0]  indexVec_checkResult_decimal_16 = indexVec_checkResult_decimalProportion_16[4:2];
  wire        indexVec_checkResult_overlap_16 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_16 >= indexVec_checkResult_intLMULInput_16[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_16} >= indexVec_checkResult_intLMULInput_16,
      indexVec_checkResult_allDataPosition_16[32:10]};
  wire        indexVec_checkResult_unChange_16 = slideUp & (indexVec_readIndex_5[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_5 = validVec[5] & ~indexVec_checkResult_unChange_16;
  wire        indexVec_checkResult_1_4_5 = indexVec_checkResult_overlap_16 | ~indexVec_checkResult_1_5_5 | lagerThanVL | indexVec_checkResult_unChange_16;
  wire [33:0] indexVec_checkResult_allDataPosition_17 = {indexVec_readIndex_5, 2'h0};
  wire [9:0]  indexVec_checkResult_dataPosition_17 = indexVec_checkResult_allDataPosition_17[9:0];
  wire [2:0]  indexVec_checkResult_2_1_5 = indexVec_checkResult_dataPosition_17[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_17 = indexVec_checkResult_dataPosition_17[9:5];
  wire [1:0]  indexVec_checkResult_2_2_5 = indexVec_checkResult_dataGroup_17[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_17 = indexVec_checkResult_dataGroup_17[4:2];
  wire [2:0]  indexVec_checkResult_2_3_5 = indexVec_checkResult_accessRegGrowth_17;
  wire [4:0]  indexVec_checkResult_decimalProportion_17 = {indexVec_checkResult_2_2_5, indexVec_checkResult_2_1_5};
  wire [2:0]  indexVec_checkResult_decimal_17 = indexVec_checkResult_decimalProportion_17[4:2];
  wire        indexVec_checkResult_overlap_17 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_17 >= indexVec_checkResult_intLMULInput_17[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_17} >= indexVec_checkResult_intLMULInput_17,
      indexVec_checkResult_allDataPosition_17[33:10]};
  wire        indexVec_checkResult_unChange_17 = slideUp & (indexVec_readIndex_5[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_5 = validVec[5] & ~indexVec_checkResult_unChange_17;
  wire        indexVec_checkResult_2_4_5 = indexVec_checkResult_overlap_17 | ~indexVec_checkResult_2_5_5 | lagerThanVL | indexVec_checkResult_unChange_17;
  wire [1:0]  indexVec_5_0 = (sew1H[0] ? indexVec_checkResult_0_0_5 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_5 : 2'h0);
  assign indexVec_5_1 = (sew1H[0] ? indexVec_checkResult_0_1_5 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_1_5 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_1_5 : 3'h0);
  wire [2:0]  indexDeq_bits_accessLane_5_0 = indexVec_5_1;
  wire [1:0]  indexVec_5_2 = (sew1H[0] ? indexVec_checkResult_0_2_5 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_5 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_5 : 2'h0);
  assign indexVec_5_3 = (sew1H[0] ? indexVec_checkResult_0_3_5 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_5 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_5 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_5_0 = indexVec_5_3;
  wire        indexVec_5_4 = sew1H[0] & indexVec_checkResult_0_4_5 | sew1H[1] & indexVec_checkResult_1_4_5 | sew1H[2] & indexVec_checkResult_2_4_5;
  wire        indexVec_5_5 = sew1H[0] & indexVec_checkResult_0_5_5 | sew1H[1] & indexVec_checkResult_1_5_5 | sew1H[2] & indexVec_checkResult_2_5_5;
  wire [31:0] indexVec_readIndex_6 = baseIndex + 32'h6;
  wire [31:0] indexVec_checkResult_allDataPosition_18 = indexVec_readIndex_6;
  wire [9:0]  indexVec_checkResult_dataPosition_18 = indexVec_checkResult_allDataPosition_18[9:0];
  wire [1:0]  indexVec_checkResult_0_0_6 = indexVec_checkResult_dataPosition_18[1:0];
  wire [2:0]  indexVec_checkResult_0_1_6 = indexVec_checkResult_dataPosition_18[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_18 = indexVec_checkResult_dataPosition_18[9:5];
  wire [1:0]  indexVec_checkResult_0_2_6 = indexVec_checkResult_dataGroup_18[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_18 = indexVec_checkResult_dataGroup_18[4:2];
  wire [2:0]  indexVec_checkResult_0_3_6 = indexVec_checkResult_accessRegGrowth_18;
  wire [4:0]  indexVec_checkResult_decimalProportion_18 = {indexVec_checkResult_0_2_6, indexVec_checkResult_0_1_6};
  wire [2:0]  indexVec_checkResult_decimal_18 = indexVec_checkResult_decimalProportion_18[4:2];
  wire        indexVec_checkResult_overlap_18 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_18 >= indexVec_checkResult_intLMULInput_18[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_18} >= indexVec_checkResult_intLMULInput_18,
      indexVec_checkResult_allDataPosition_18[31:10]};
  wire        indexVec_checkResult_unChange_18 = slideUp & (indexVec_readIndex_6[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_6 = validVec[6] & ~indexVec_checkResult_unChange_18;
  wire        indexVec_checkResult_0_4_6 = indexVec_checkResult_overlap_18 | ~indexVec_checkResult_0_5_6 | lagerThanVL | indexVec_checkResult_unChange_18;
  wire [32:0] indexVec_checkResult_allDataPosition_19 = {indexVec_readIndex_6, 1'h0};
  wire [9:0]  indexVec_checkResult_dataPosition_19 = indexVec_checkResult_allDataPosition_19[9:0];
  wire [1:0]  indexVec_checkResult_1_0_6 = {indexVec_checkResult_dataPosition_19[1], 1'h0};
  wire [2:0]  indexVec_checkResult_1_1_6 = indexVec_checkResult_dataPosition_19[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_19 = indexVec_checkResult_dataPosition_19[9:5];
  wire [1:0]  indexVec_checkResult_1_2_6 = indexVec_checkResult_dataGroup_19[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_19 = indexVec_checkResult_dataGroup_19[4:2];
  wire [2:0]  indexVec_checkResult_1_3_6 = indexVec_checkResult_accessRegGrowth_19;
  wire [4:0]  indexVec_checkResult_decimalProportion_19 = {indexVec_checkResult_1_2_6, indexVec_checkResult_1_1_6};
  wire [2:0]  indexVec_checkResult_decimal_19 = indexVec_checkResult_decimalProportion_19[4:2];
  wire        indexVec_checkResult_overlap_19 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_19 >= indexVec_checkResult_intLMULInput_19[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_19} >= indexVec_checkResult_intLMULInput_19,
      indexVec_checkResult_allDataPosition_19[32:10]};
  wire        indexVec_checkResult_unChange_19 = slideUp & (indexVec_readIndex_6[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_6 = validVec[6] & ~indexVec_checkResult_unChange_19;
  wire        indexVec_checkResult_1_4_6 = indexVec_checkResult_overlap_19 | ~indexVec_checkResult_1_5_6 | lagerThanVL | indexVec_checkResult_unChange_19;
  wire [33:0] indexVec_checkResult_allDataPosition_20 = {indexVec_readIndex_6, 2'h0};
  wire [9:0]  indexVec_checkResult_dataPosition_20 = indexVec_checkResult_allDataPosition_20[9:0];
  wire [2:0]  indexVec_checkResult_2_1_6 = indexVec_checkResult_dataPosition_20[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_20 = indexVec_checkResult_dataPosition_20[9:5];
  wire [1:0]  indexVec_checkResult_2_2_6 = indexVec_checkResult_dataGroup_20[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_20 = indexVec_checkResult_dataGroup_20[4:2];
  wire [2:0]  indexVec_checkResult_2_3_6 = indexVec_checkResult_accessRegGrowth_20;
  wire [4:0]  indexVec_checkResult_decimalProportion_20 = {indexVec_checkResult_2_2_6, indexVec_checkResult_2_1_6};
  wire [2:0]  indexVec_checkResult_decimal_20 = indexVec_checkResult_decimalProportion_20[4:2];
  wire        indexVec_checkResult_overlap_20 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_20 >= indexVec_checkResult_intLMULInput_20[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_20} >= indexVec_checkResult_intLMULInput_20,
      indexVec_checkResult_allDataPosition_20[33:10]};
  wire        indexVec_checkResult_unChange_20 = slideUp & (indexVec_readIndex_6[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_6 = validVec[6] & ~indexVec_checkResult_unChange_20;
  wire        indexVec_checkResult_2_4_6 = indexVec_checkResult_overlap_20 | ~indexVec_checkResult_2_5_6 | lagerThanVL | indexVec_checkResult_unChange_20;
  wire [1:0]  indexVec_6_0 = (sew1H[0] ? indexVec_checkResult_0_0_6 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_6 : 2'h0);
  assign indexVec_6_1 = (sew1H[0] ? indexVec_checkResult_0_1_6 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_1_6 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_1_6 : 3'h0);
  wire [2:0]  indexDeq_bits_accessLane_6_0 = indexVec_6_1;
  wire [1:0]  indexVec_6_2 = (sew1H[0] ? indexVec_checkResult_0_2_6 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_6 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_6 : 2'h0);
  assign indexVec_6_3 = (sew1H[0] ? indexVec_checkResult_0_3_6 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_6 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_6 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_6_0 = indexVec_6_3;
  wire        indexVec_6_4 = sew1H[0] & indexVec_checkResult_0_4_6 | sew1H[1] & indexVec_checkResult_1_4_6 | sew1H[2] & indexVec_checkResult_2_4_6;
  wire        indexVec_6_5 = sew1H[0] & indexVec_checkResult_0_5_6 | sew1H[1] & indexVec_checkResult_1_5_6 | sew1H[2] & indexVec_checkResult_2_5_6;
  wire [31:0] indexVec_readIndex_7 = baseIndex + 32'h7;
  wire [31:0] indexVec_checkResult_allDataPosition_21 = indexVec_readIndex_7;
  wire [9:0]  indexVec_checkResult_dataPosition_21 = indexVec_checkResult_allDataPosition_21[9:0];
  wire [1:0]  indexVec_checkResult_0_0_7 = indexVec_checkResult_dataPosition_21[1:0];
  wire [2:0]  indexVec_checkResult_0_1_7 = indexVec_checkResult_dataPosition_21[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_21 = indexVec_checkResult_dataPosition_21[9:5];
  wire [1:0]  indexVec_checkResult_0_2_7 = indexVec_checkResult_dataGroup_21[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_21 = indexVec_checkResult_dataGroup_21[4:2];
  wire [2:0]  indexVec_checkResult_0_3_7 = indexVec_checkResult_accessRegGrowth_21;
  wire [4:0]  indexVec_checkResult_decimalProportion_21 = {indexVec_checkResult_0_2_7, indexVec_checkResult_0_1_7};
  wire [2:0]  indexVec_checkResult_decimal_21 = indexVec_checkResult_decimalProportion_21[4:2];
  wire        indexVec_checkResult_overlap_21 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_21 >= indexVec_checkResult_intLMULInput_21[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_21} >= indexVec_checkResult_intLMULInput_21,
      indexVec_checkResult_allDataPosition_21[31:10]};
  wire        indexVec_checkResult_unChange_21 = slideUp & (indexVec_readIndex_7[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_7 = validVec[7] & ~indexVec_checkResult_unChange_21;
  wire        indexVec_checkResult_0_4_7 = indexVec_checkResult_overlap_21 | ~indexVec_checkResult_0_5_7 | lagerThanVL | indexVec_checkResult_unChange_21;
  wire [32:0] indexVec_checkResult_allDataPosition_22 = {indexVec_readIndex_7, 1'h0};
  wire [9:0]  indexVec_checkResult_dataPosition_22 = indexVec_checkResult_allDataPosition_22[9:0];
  wire [1:0]  indexVec_checkResult_1_0_7 = {indexVec_checkResult_dataPosition_22[1], 1'h0};
  wire [2:0]  indexVec_checkResult_1_1_7 = indexVec_checkResult_dataPosition_22[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_22 = indexVec_checkResult_dataPosition_22[9:5];
  wire [1:0]  indexVec_checkResult_1_2_7 = indexVec_checkResult_dataGroup_22[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_22 = indexVec_checkResult_dataGroup_22[4:2];
  wire [2:0]  indexVec_checkResult_1_3_7 = indexVec_checkResult_accessRegGrowth_22;
  wire [4:0]  indexVec_checkResult_decimalProportion_22 = {indexVec_checkResult_1_2_7, indexVec_checkResult_1_1_7};
  wire [2:0]  indexVec_checkResult_decimal_22 = indexVec_checkResult_decimalProportion_22[4:2];
  wire        indexVec_checkResult_overlap_22 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_22 >= indexVec_checkResult_intLMULInput_22[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_22} >= indexVec_checkResult_intLMULInput_22,
      indexVec_checkResult_allDataPosition_22[32:10]};
  wire        indexVec_checkResult_unChange_22 = slideUp & (indexVec_readIndex_7[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_7 = validVec[7] & ~indexVec_checkResult_unChange_22;
  wire        indexVec_checkResult_1_4_7 = indexVec_checkResult_overlap_22 | ~indexVec_checkResult_1_5_7 | lagerThanVL | indexVec_checkResult_unChange_22;
  wire [33:0] indexVec_checkResult_allDataPosition_23 = {indexVec_readIndex_7, 2'h0};
  wire [9:0]  indexVec_checkResult_dataPosition_23 = indexVec_checkResult_allDataPosition_23[9:0];
  wire [2:0]  indexVec_checkResult_2_1_7 = indexVec_checkResult_dataPosition_23[4:2];
  wire [4:0]  indexVec_checkResult_dataGroup_23 = indexVec_checkResult_dataPosition_23[9:5];
  wire [1:0]  indexVec_checkResult_2_2_7 = indexVec_checkResult_dataGroup_23[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_23 = indexVec_checkResult_dataGroup_23[4:2];
  wire [2:0]  indexVec_checkResult_2_3_7 = indexVec_checkResult_accessRegGrowth_23;
  wire [4:0]  indexVec_checkResult_decimalProportion_23 = {indexVec_checkResult_2_2_7, indexVec_checkResult_2_1_7};
  wire [2:0]  indexVec_checkResult_decimal_23 = indexVec_checkResult_decimalProportion_23[4:2];
  wire        indexVec_checkResult_overlap_23 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_23 >= indexVec_checkResult_intLMULInput_23[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_23} >= indexVec_checkResult_intLMULInput_23,
      indexVec_checkResult_allDataPosition_23[33:10]};
  wire        indexVec_checkResult_unChange_23 = slideUp & (indexVec_readIndex_7[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_7 = validVec[7] & ~indexVec_checkResult_unChange_23;
  wire        indexVec_checkResult_2_4_7 = indexVec_checkResult_overlap_23 | ~indexVec_checkResult_2_5_7 | lagerThanVL | indexVec_checkResult_unChange_23;
  wire [1:0]  indexVec_7_0 = (sew1H[0] ? indexVec_checkResult_0_0_7 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_7 : 2'h0);
  assign indexVec_7_1 = (sew1H[0] ? indexVec_checkResult_0_1_7 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_1_7 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_1_7 : 3'h0);
  wire [2:0]  indexDeq_bits_accessLane_7_0 = indexVec_7_1;
  wire [1:0]  indexVec_7_2 = (sew1H[0] ? indexVec_checkResult_0_2_7 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_7 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_7 : 2'h0);
  assign indexVec_7_3 = (sew1H[0] ? indexVec_checkResult_0_3_7 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_7 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_7 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_7_0 = indexVec_7_3;
  wire        indexVec_7_4 = sew1H[0] & indexVec_checkResult_0_4_7 | sew1H[1] & indexVec_checkResult_1_4_7 | sew1H[2] & indexVec_checkResult_2_4_7;
  wire        indexVec_7_5 = sew1H[0] & indexVec_checkResult_0_5_7 | sew1H[1] & indexVec_checkResult_1_5_7 | sew1H[2] & indexVec_checkResult_2_5_7;
  assign indexDeq_valid_0 = InstructionValid & isSlide;
  wire [1:0]  indexDeq_bits_needRead_lo_lo = {~indexVec_1_4, ~indexVec_0_4};
  wire [1:0]  indexDeq_bits_needRead_lo_hi = {~indexVec_3_4, ~indexVec_2_4};
  wire [3:0]  indexDeq_bits_needRead_lo = {indexDeq_bits_needRead_lo_hi, indexDeq_bits_needRead_lo_lo};
  wire [1:0]  indexDeq_bits_needRead_hi_lo = {~indexVec_5_4, ~indexVec_4_4};
  wire [1:0]  indexDeq_bits_needRead_hi_hi = {~indexVec_7_4, ~indexVec_6_4};
  wire [3:0]  indexDeq_bits_needRead_hi = {indexDeq_bits_needRead_hi_hi, indexDeq_bits_needRead_hi_lo};
  wire [7:0]  indexDeq_bits_needRead_0 = {indexDeq_bits_needRead_hi, indexDeq_bits_needRead_lo} & ~replaceWithVs1;
  wire [1:0]  indexDeq_bits_elementValid_lo_lo = {indexVec_1_5, indexVec_0_5};
  wire [1:0]  indexDeq_bits_elementValid_lo_hi = {indexVec_3_5, indexVec_2_5};
  wire [3:0]  indexDeq_bits_elementValid_lo = {indexDeq_bits_elementValid_lo_hi, indexDeq_bits_elementValid_lo_lo};
  wire [1:0]  indexDeq_bits_elementValid_hi_lo = {indexVec_5_5, indexVec_4_5};
  wire [1:0]  indexDeq_bits_elementValid_hi_hi = {indexVec_7_5, indexVec_6_5};
  wire [3:0]  indexDeq_bits_elementValid_hi = {indexDeq_bits_elementValid_hi_hi, indexDeq_bits_elementValid_hi_lo};
  wire [7:0]  indexDeq_bits_elementValid_0 = {indexDeq_bits_elementValid_hi, indexDeq_bits_elementValid_lo} | replaceWithVs1;
  wire [3:0]  indexDeq_bits_readOffset_lo_lo = {indexVec_1_2, indexVec_0_2};
  wire [3:0]  indexDeq_bits_readOffset_lo_hi = {indexVec_3_2, indexVec_2_2};
  wire [7:0]  indexDeq_bits_readOffset_lo = {indexDeq_bits_readOffset_lo_hi, indexDeq_bits_readOffset_lo_lo};
  wire [3:0]  indexDeq_bits_readOffset_hi_lo = {indexVec_5_2, indexVec_4_2};
  wire [3:0]  indexDeq_bits_readOffset_hi_hi = {indexVec_7_2, indexVec_6_2};
  wire [7:0]  indexDeq_bits_readOffset_hi = {indexDeq_bits_readOffset_hi_hi, indexDeq_bits_readOffset_hi_lo};
  wire [15:0] indexDeq_bits_readOffset_0 = {indexDeq_bits_readOffset_hi, indexDeq_bits_readOffset_lo};
  wire [3:0]  indexDeq_bits_readDataOffset_lo_lo = {indexVec_1_0, indexVec_0_0};
  wire [3:0]  indexDeq_bits_readDataOffset_lo_hi = {indexVec_3_0, indexVec_2_0};
  wire [7:0]  indexDeq_bits_readDataOffset_lo = {indexDeq_bits_readDataOffset_lo_hi, indexDeq_bits_readDataOffset_lo_lo};
  wire [3:0]  indexDeq_bits_readDataOffset_hi_lo = {indexVec_5_0, indexVec_4_0};
  wire [3:0]  indexDeq_bits_readDataOffset_hi_hi = {indexVec_7_0, indexVec_6_0};
  wire [7:0]  indexDeq_bits_readDataOffset_hi = {indexDeq_bits_readDataOffset_hi_hi, indexDeq_bits_readDataOffset_hi_lo};
  wire [15:0] indexDeq_bits_readDataOffset_0 = {indexDeq_bits_readDataOffset_hi, indexDeq_bits_readDataOffset_lo};
  always @(posedge clock) begin
    if (reset) begin
      InstructionValid <= 1'h0;
      slideGroup <= 8'h0;
    end
    else begin
      if (newInstruction | lastFire)
        InstructionValid <= newInstruction;
      if (newInstruction | _lastFire_T_1)
        slideGroup <= newInstruction ? 8'h0 : slideGroup + 8'h1;
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
        slideGroup = _RANDOM[/*Zero width*/ 1'b0][8:1];
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
  assign indexDeq_bits_accessLane_4 = indexDeq_bits_accessLane_4_0;
  assign indexDeq_bits_accessLane_5 = indexDeq_bits_accessLane_5_0;
  assign indexDeq_bits_accessLane_6 = indexDeq_bits_accessLane_6_0;
  assign indexDeq_bits_accessLane_7 = indexDeq_bits_accessLane_7_0;
  assign indexDeq_bits_vsGrowth_0 = indexDeq_bits_vsGrowth_0_0;
  assign indexDeq_bits_vsGrowth_1 = indexDeq_bits_vsGrowth_1_0;
  assign indexDeq_bits_vsGrowth_2 = indexDeq_bits_vsGrowth_2_0;
  assign indexDeq_bits_vsGrowth_3 = indexDeq_bits_vsGrowth_3_0;
  assign indexDeq_bits_vsGrowth_4 = indexDeq_bits_vsGrowth_4_0;
  assign indexDeq_bits_vsGrowth_5 = indexDeq_bits_vsGrowth_5_0;
  assign indexDeq_bits_vsGrowth_6 = indexDeq_bits_vsGrowth_6_0;
  assign indexDeq_bits_vsGrowth_7 = indexDeq_bits_vsGrowth_7_0;
  assign indexDeq_bits_executeGroup = indexDeq_bits_executeGroup_0;
  assign indexDeq_bits_readDataOffset = indexDeq_bits_readDataOffset_0;
  assign indexDeq_bits_last = indexDeq_bits_last_0;
  assign slideGroupOut = slideGroup;
endmodule

