
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
  input  [11:0] instructionReq_vl,
  input         indexDeq_ready,
  output        indexDeq_valid,
  output [15:0] indexDeq_bits_needRead,
                indexDeq_bits_elementValid,
                indexDeq_bits_replaceVs1,
  output [31:0] indexDeq_bits_readOffset,
  output [3:0]  indexDeq_bits_accessLane_0,
                indexDeq_bits_accessLane_1,
                indexDeq_bits_accessLane_2,
                indexDeq_bits_accessLane_3,
                indexDeq_bits_accessLane_4,
                indexDeq_bits_accessLane_5,
                indexDeq_bits_accessLane_6,
                indexDeq_bits_accessLane_7,
                indexDeq_bits_accessLane_8,
                indexDeq_bits_accessLane_9,
                indexDeq_bits_accessLane_10,
                indexDeq_bits_accessLane_11,
                indexDeq_bits_accessLane_12,
                indexDeq_bits_accessLane_13,
                indexDeq_bits_accessLane_14,
                indexDeq_bits_accessLane_15,
  output [2:0]  indexDeq_bits_vsGrowth_0,
                indexDeq_bits_vsGrowth_1,
                indexDeq_bits_vsGrowth_2,
                indexDeq_bits_vsGrowth_3,
                indexDeq_bits_vsGrowth_4,
                indexDeq_bits_vsGrowth_5,
                indexDeq_bits_vsGrowth_6,
                indexDeq_bits_vsGrowth_7,
                indexDeq_bits_vsGrowth_8,
                indexDeq_bits_vsGrowth_9,
                indexDeq_bits_vsGrowth_10,
                indexDeq_bits_vsGrowth_11,
                indexDeq_bits_vsGrowth_12,
                indexDeq_bits_vsGrowth_13,
                indexDeq_bits_vsGrowth_14,
                indexDeq_bits_vsGrowth_15,
  output [7:0]  indexDeq_bits_executeGroup,
  output [31:0] indexDeq_bits_readDataOffset,
  output        indexDeq_bits_last,
  output [7:0]  slideGroupOut,
  input  [15:0] slideMaskInput
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
  wire [1:0]  indexVec_checkResult_2_0_8 = 2'h0;
  wire [1:0]  indexVec_checkResult_2_0_9 = 2'h0;
  wire [1:0]  indexVec_checkResult_2_0_10 = 2'h0;
  wire [1:0]  indexVec_checkResult_2_0_11 = 2'h0;
  wire [1:0]  indexVec_checkResult_2_0_12 = 2'h0;
  wire [1:0]  indexVec_checkResult_2_0_13 = 2'h0;
  wire [1:0]  indexVec_checkResult_2_0_14 = 2'h0;
  wire [1:0]  indexVec_checkResult_2_0_15 = 2'h0;
  wire [15:0] indexDeq_bits_groupReadState = 16'h0;
  wire [15:0] replaceWithVs1;
  wire [3:0]  indexVec_0_1;
  wire [3:0]  indexVec_1_1;
  wire [3:0]  indexVec_2_1;
  wire [3:0]  indexVec_3_1;
  wire [3:0]  indexVec_4_1;
  wire [3:0]  indexVec_5_1;
  wire [3:0]  indexVec_6_1;
  wire [3:0]  indexVec_7_1;
  wire [3:0]  indexVec_8_1;
  wire [3:0]  indexVec_9_1;
  wire [3:0]  indexVec_10_1;
  wire [3:0]  indexVec_11_1;
  wire [3:0]  indexVec_12_1;
  wire [3:0]  indexVec_13_1;
  wire [3:0]  indexVec_14_1;
  wire [3:0]  indexVec_15_1;
  wire [2:0]  indexVec_0_3;
  wire [2:0]  indexVec_1_3;
  wire [2:0]  indexVec_2_3;
  wire [2:0]  indexVec_3_3;
  wire [2:0]  indexVec_4_3;
  wire [2:0]  indexVec_5_3;
  wire [2:0]  indexVec_6_3;
  wire [2:0]  indexVec_7_3;
  wire [2:0]  indexVec_8_3;
  wire [2:0]  indexVec_9_3;
  wire [2:0]  indexVec_10_3;
  wire [2:0]  indexVec_11_3;
  wire [2:0]  indexVec_12_3;
  wire [2:0]  indexVec_13_3;
  wire [2:0]  indexVec_14_3;
  wire [2:0]  indexVec_15_3;
  reg         InstructionValid;
  wire        isSlide = instructionReq_decodeResult_topUop[4:2] == 3'h0;
  wire        slideUp = instructionReq_decodeResult_topUop[0];
  wire        slide1 = instructionReq_decodeResult_topUop[1];
  reg  [7:0]  slideGroup;
  wire [7:0]  indexDeq_bits_executeGroup_0 = slideGroup;
  wire [3:0]  vlTail = instructionReq_vl[3:0];
  wire [7:0]  lastSlideGroup = instructionReq_vl[11:4] - {7'h0, vlTail == 4'h0};
  wire [15:0] _lastValidVec_T = 16'h1 << vlTail;
  wire [15:0] _lastValidVec_T_3 = _lastValidVec_T | {_lastValidVec_T[14:0], 1'h0};
  wire [15:0] _lastValidVec_T_6 = _lastValidVec_T_3 | {_lastValidVec_T_3[13:0], 2'h0};
  wire [15:0] _lastValidVec_T_9 = _lastValidVec_T_6 | {_lastValidVec_T_6[11:0], 4'h0};
  wire [15:0] lastValidVec = ~(_lastValidVec_T_9 | {_lastValidVec_T_9[7:0], 8'h0});
  wire        indexDeq_bits_last_0 = slideGroup == lastSlideGroup;
  wire [15:0] groupVlValid = indexDeq_bits_last_0 & (|vlTail) ? lastValidVec : 16'hFFFF;
  wire [15:0] groupMaskValid = instructionReq_maskType ? slideMaskInput : 16'hFFFF;
  wire [15:0] validVec = groupVlValid & groupMaskValid;
  wire [15:0] lastElementValid = ({1'h0, groupVlValid[15:1]} ^ groupVlValid) & groupMaskValid;
  assign replaceWithVs1 = (slideGroup == 8'h0 & slide1 & slideUp ? {15'h0, groupMaskValid[0]} : 16'h0) | (indexDeq_bits_last_0 & slide1 & ~slideUp ? lastElementValid : 16'h0);
  wire        indexDeq_valid_0;
  wire [15:0] indexDeq_bits_replaceVs1_0 = replaceWithVs1;
  wire        _lastFire_T_1 = indexDeq_ready_0 & indexDeq_valid_0;
  wire        lastFire = indexDeq_bits_last_0 & _lastFire_T_1;
  wire [31:0] slideValue = slide1 ? 32'h1 : instructionReq_readFromScala;
  wire [31:0] PNSelect = {32{slideUp}} ^ slideValue;
  wire [31:0] baseIndex = {20'h0, slideGroup, 4'h0} + PNSelect + {31'h0, slideUp};
  wire [31:0] indexVec_readIndex = baseIndex;
  wire        lagerThanVL = |(slideValue[31:12]);
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
  wire [3:0]  indexVec_checkResult_intLMULInput_24;
  assign indexVec_checkResult_intLMULInput_24 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_25;
  assign indexVec_checkResult_intLMULInput_25 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_26;
  assign indexVec_checkResult_intLMULInput_26 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_27;
  assign indexVec_checkResult_intLMULInput_27 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_28;
  assign indexVec_checkResult_intLMULInput_28 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_29;
  assign indexVec_checkResult_intLMULInput_29 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_30;
  assign indexVec_checkResult_intLMULInput_30 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_31;
  assign indexVec_checkResult_intLMULInput_31 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_32;
  assign indexVec_checkResult_intLMULInput_32 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_33;
  assign indexVec_checkResult_intLMULInput_33 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_34;
  assign indexVec_checkResult_intLMULInput_34 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_35;
  assign indexVec_checkResult_intLMULInput_35 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_36;
  assign indexVec_checkResult_intLMULInput_36 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_37;
  assign indexVec_checkResult_intLMULInput_37 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_38;
  assign indexVec_checkResult_intLMULInput_38 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_39;
  assign indexVec_checkResult_intLMULInput_39 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_40;
  assign indexVec_checkResult_intLMULInput_40 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_41;
  assign indexVec_checkResult_intLMULInput_41 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_42;
  assign indexVec_checkResult_intLMULInput_42 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_43;
  assign indexVec_checkResult_intLMULInput_43 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_44;
  assign indexVec_checkResult_intLMULInput_44 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_45;
  assign indexVec_checkResult_intLMULInput_45 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_46;
  assign indexVec_checkResult_intLMULInput_46 = _GEN;
  wire [3:0]  indexVec_checkResult_intLMULInput_47;
  assign indexVec_checkResult_intLMULInput_47 = _GEN;
  wire [10:0] indexVec_checkResult_dataPosition = indexVec_checkResult_allDataPosition[10:0];
  wire [1:0]  indexVec_checkResult_0_0 = indexVec_checkResult_dataPosition[1:0];
  wire [3:0]  indexVec_checkResult_0_1 = indexVec_checkResult_dataPosition[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup = indexVec_checkResult_dataPosition[10:6];
  wire [1:0]  indexVec_checkResult_0_2 = indexVec_checkResult_dataGroup[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth = indexVec_checkResult_dataGroup[4:2];
  wire [2:0]  indexVec_checkResult_0_3 = indexVec_checkResult_accessRegGrowth;
  wire [5:0]  indexVec_checkResult_decimalProportion = {indexVec_checkResult_0_2, indexVec_checkResult_0_1};
  wire [2:0]  indexVec_checkResult_decimal = indexVec_checkResult_decimalProportion[5:3];
  wire        indexVec_checkResult_overlap =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal >= indexVec_checkResult_intLMULInput[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth} >= indexVec_checkResult_intLMULInput,
      indexVec_checkResult_allDataPosition[31:11]};
  wire        indexVec_checkResult_unChange = slideUp & (indexVec_readIndex[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5 = validVec[0] & ~indexVec_checkResult_unChange;
  wire        indexVec_checkResult_0_4 = indexVec_checkResult_overlap | ~indexVec_checkResult_0_5 | lagerThanVL | indexVec_checkResult_unChange;
  wire [32:0] indexVec_checkResult_allDataPosition_1 = {indexVec_readIndex, 1'h0};
  wire [10:0] indexVec_checkResult_dataPosition_1 = indexVec_checkResult_allDataPosition_1[10:0];
  wire [1:0]  indexVec_checkResult_1_0 = {indexVec_checkResult_dataPosition_1[1], 1'h0};
  wire [3:0]  indexVec_checkResult_1_1 = indexVec_checkResult_dataPosition_1[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_1 = indexVec_checkResult_dataPosition_1[10:6];
  wire [1:0]  indexVec_checkResult_1_2 = indexVec_checkResult_dataGroup_1[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_1 = indexVec_checkResult_dataGroup_1[4:2];
  wire [2:0]  indexVec_checkResult_1_3 = indexVec_checkResult_accessRegGrowth_1;
  wire [5:0]  indexVec_checkResult_decimalProportion_1 = {indexVec_checkResult_1_2, indexVec_checkResult_1_1};
  wire [2:0]  indexVec_checkResult_decimal_1 = indexVec_checkResult_decimalProportion_1[5:3];
  wire        indexVec_checkResult_overlap_1 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_1 >= indexVec_checkResult_intLMULInput_1[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_1} >= indexVec_checkResult_intLMULInput_1,
      indexVec_checkResult_allDataPosition_1[32:11]};
  wire        indexVec_checkResult_unChange_1 = slideUp & (indexVec_readIndex[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5 = validVec[0] & ~indexVec_checkResult_unChange_1;
  wire        indexVec_checkResult_1_4 = indexVec_checkResult_overlap_1 | ~indexVec_checkResult_1_5 | lagerThanVL | indexVec_checkResult_unChange_1;
  wire [33:0] indexVec_checkResult_allDataPosition_2 = {indexVec_readIndex, 2'h0};
  wire [10:0] indexVec_checkResult_dataPosition_2 = indexVec_checkResult_allDataPosition_2[10:0];
  wire [3:0]  indexVec_checkResult_2_1 = indexVec_checkResult_dataPosition_2[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_2 = indexVec_checkResult_dataPosition_2[10:6];
  wire [1:0]  indexVec_checkResult_2_2 = indexVec_checkResult_dataGroup_2[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_2 = indexVec_checkResult_dataGroup_2[4:2];
  wire [2:0]  indexVec_checkResult_2_3 = indexVec_checkResult_accessRegGrowth_2;
  wire [5:0]  indexVec_checkResult_decimalProportion_2 = {indexVec_checkResult_2_2, indexVec_checkResult_2_1};
  wire [2:0]  indexVec_checkResult_decimal_2 = indexVec_checkResult_decimalProportion_2[5:3];
  wire        indexVec_checkResult_overlap_2 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_2 >= indexVec_checkResult_intLMULInput_2[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_2} >= indexVec_checkResult_intLMULInput_2,
      indexVec_checkResult_allDataPosition_2[33:11]};
  wire        indexVec_checkResult_unChange_2 = slideUp & (indexVec_readIndex[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5 = validVec[0] & ~indexVec_checkResult_unChange_2;
  wire        indexVec_checkResult_2_4 = indexVec_checkResult_overlap_2 | ~indexVec_checkResult_2_5 | lagerThanVL | indexVec_checkResult_unChange_2;
  wire [1:0]  indexVec_0_0 = (sew1H[0] ? indexVec_checkResult_0_0 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0 : 2'h0);
  assign indexVec_0_1 = (sew1H[0] ? indexVec_checkResult_0_1 : 4'h0) | (sew1H[1] ? indexVec_checkResult_1_1 : 4'h0) | (sew1H[2] ? indexVec_checkResult_2_1 : 4'h0);
  wire [3:0]  indexDeq_bits_accessLane_0_0 = indexVec_0_1;
  wire [1:0]  indexVec_0_2 = (sew1H[0] ? indexVec_checkResult_0_2 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2 : 2'h0);
  assign indexVec_0_3 = (sew1H[0] ? indexVec_checkResult_0_3 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_0_0 = indexVec_0_3;
  wire        indexVec_0_4 = sew1H[0] & indexVec_checkResult_0_4 | sew1H[1] & indexVec_checkResult_1_4 | sew1H[2] & indexVec_checkResult_2_4;
  wire        indexVec_0_5 = sew1H[0] & indexVec_checkResult_0_5 | sew1H[1] & indexVec_checkResult_1_5 | sew1H[2] & indexVec_checkResult_2_5;
  wire [31:0] indexVec_readIndex_1 = baseIndex + 32'h1;
  wire [31:0] indexVec_checkResult_allDataPosition_3 = indexVec_readIndex_1;
  wire [10:0] indexVec_checkResult_dataPosition_3 = indexVec_checkResult_allDataPosition_3[10:0];
  wire [1:0]  indexVec_checkResult_0_0_1 = indexVec_checkResult_dataPosition_3[1:0];
  wire [3:0]  indexVec_checkResult_0_1_1 = indexVec_checkResult_dataPosition_3[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_3 = indexVec_checkResult_dataPosition_3[10:6];
  wire [1:0]  indexVec_checkResult_0_2_1 = indexVec_checkResult_dataGroup_3[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_3 = indexVec_checkResult_dataGroup_3[4:2];
  wire [2:0]  indexVec_checkResult_0_3_1 = indexVec_checkResult_accessRegGrowth_3;
  wire [5:0]  indexVec_checkResult_decimalProportion_3 = {indexVec_checkResult_0_2_1, indexVec_checkResult_0_1_1};
  wire [2:0]  indexVec_checkResult_decimal_3 = indexVec_checkResult_decimalProportion_3[5:3];
  wire        indexVec_checkResult_overlap_3 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_3 >= indexVec_checkResult_intLMULInput_3[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_3} >= indexVec_checkResult_intLMULInput_3,
      indexVec_checkResult_allDataPosition_3[31:11]};
  wire        indexVec_checkResult_unChange_3 = slideUp & (indexVec_readIndex_1[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_1 = validVec[1] & ~indexVec_checkResult_unChange_3;
  wire        indexVec_checkResult_0_4_1 = indexVec_checkResult_overlap_3 | ~indexVec_checkResult_0_5_1 | lagerThanVL | indexVec_checkResult_unChange_3;
  wire [32:0] indexVec_checkResult_allDataPosition_4 = {indexVec_readIndex_1, 1'h0};
  wire [10:0] indexVec_checkResult_dataPosition_4 = indexVec_checkResult_allDataPosition_4[10:0];
  wire [1:0]  indexVec_checkResult_1_0_1 = {indexVec_checkResult_dataPosition_4[1], 1'h0};
  wire [3:0]  indexVec_checkResult_1_1_1 = indexVec_checkResult_dataPosition_4[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_4 = indexVec_checkResult_dataPosition_4[10:6];
  wire [1:0]  indexVec_checkResult_1_2_1 = indexVec_checkResult_dataGroup_4[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_4 = indexVec_checkResult_dataGroup_4[4:2];
  wire [2:0]  indexVec_checkResult_1_3_1 = indexVec_checkResult_accessRegGrowth_4;
  wire [5:0]  indexVec_checkResult_decimalProportion_4 = {indexVec_checkResult_1_2_1, indexVec_checkResult_1_1_1};
  wire [2:0]  indexVec_checkResult_decimal_4 = indexVec_checkResult_decimalProportion_4[5:3];
  wire        indexVec_checkResult_overlap_4 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_4 >= indexVec_checkResult_intLMULInput_4[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_4} >= indexVec_checkResult_intLMULInput_4,
      indexVec_checkResult_allDataPosition_4[32:11]};
  wire        indexVec_checkResult_unChange_4 = slideUp & (indexVec_readIndex_1[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_1 = validVec[1] & ~indexVec_checkResult_unChange_4;
  wire        indexVec_checkResult_1_4_1 = indexVec_checkResult_overlap_4 | ~indexVec_checkResult_1_5_1 | lagerThanVL | indexVec_checkResult_unChange_4;
  wire [33:0] indexVec_checkResult_allDataPosition_5 = {indexVec_readIndex_1, 2'h0};
  wire [10:0] indexVec_checkResult_dataPosition_5 = indexVec_checkResult_allDataPosition_5[10:0];
  wire [3:0]  indexVec_checkResult_2_1_1 = indexVec_checkResult_dataPosition_5[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_5 = indexVec_checkResult_dataPosition_5[10:6];
  wire [1:0]  indexVec_checkResult_2_2_1 = indexVec_checkResult_dataGroup_5[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_5 = indexVec_checkResult_dataGroup_5[4:2];
  wire [2:0]  indexVec_checkResult_2_3_1 = indexVec_checkResult_accessRegGrowth_5;
  wire [5:0]  indexVec_checkResult_decimalProportion_5 = {indexVec_checkResult_2_2_1, indexVec_checkResult_2_1_1};
  wire [2:0]  indexVec_checkResult_decimal_5 = indexVec_checkResult_decimalProportion_5[5:3];
  wire        indexVec_checkResult_overlap_5 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_5 >= indexVec_checkResult_intLMULInput_5[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_5} >= indexVec_checkResult_intLMULInput_5,
      indexVec_checkResult_allDataPosition_5[33:11]};
  wire        indexVec_checkResult_unChange_5 = slideUp & (indexVec_readIndex_1[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_1 = validVec[1] & ~indexVec_checkResult_unChange_5;
  wire        indexVec_checkResult_2_4_1 = indexVec_checkResult_overlap_5 | ~indexVec_checkResult_2_5_1 | lagerThanVL | indexVec_checkResult_unChange_5;
  wire [1:0]  indexVec_1_0 = (sew1H[0] ? indexVec_checkResult_0_0_1 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_1 : 2'h0);
  assign indexVec_1_1 = (sew1H[0] ? indexVec_checkResult_0_1_1 : 4'h0) | (sew1H[1] ? indexVec_checkResult_1_1_1 : 4'h0) | (sew1H[2] ? indexVec_checkResult_2_1_1 : 4'h0);
  wire [3:0]  indexDeq_bits_accessLane_1_0 = indexVec_1_1;
  wire [1:0]  indexVec_1_2 = (sew1H[0] ? indexVec_checkResult_0_2_1 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_1 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_1 : 2'h0);
  assign indexVec_1_3 = (sew1H[0] ? indexVec_checkResult_0_3_1 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_1 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_1 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_1_0 = indexVec_1_3;
  wire        indexVec_1_4 = sew1H[0] & indexVec_checkResult_0_4_1 | sew1H[1] & indexVec_checkResult_1_4_1 | sew1H[2] & indexVec_checkResult_2_4_1;
  wire        indexVec_1_5 = sew1H[0] & indexVec_checkResult_0_5_1 | sew1H[1] & indexVec_checkResult_1_5_1 | sew1H[2] & indexVec_checkResult_2_5_1;
  wire [31:0] indexVec_readIndex_2 = baseIndex + 32'h2;
  wire [31:0] indexVec_checkResult_allDataPosition_6 = indexVec_readIndex_2;
  wire [10:0] indexVec_checkResult_dataPosition_6 = indexVec_checkResult_allDataPosition_6[10:0];
  wire [1:0]  indexVec_checkResult_0_0_2 = indexVec_checkResult_dataPosition_6[1:0];
  wire [3:0]  indexVec_checkResult_0_1_2 = indexVec_checkResult_dataPosition_6[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_6 = indexVec_checkResult_dataPosition_6[10:6];
  wire [1:0]  indexVec_checkResult_0_2_2 = indexVec_checkResult_dataGroup_6[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_6 = indexVec_checkResult_dataGroup_6[4:2];
  wire [2:0]  indexVec_checkResult_0_3_2 = indexVec_checkResult_accessRegGrowth_6;
  wire [5:0]  indexVec_checkResult_decimalProportion_6 = {indexVec_checkResult_0_2_2, indexVec_checkResult_0_1_2};
  wire [2:0]  indexVec_checkResult_decimal_6 = indexVec_checkResult_decimalProportion_6[5:3];
  wire        indexVec_checkResult_overlap_6 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_6 >= indexVec_checkResult_intLMULInput_6[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_6} >= indexVec_checkResult_intLMULInput_6,
      indexVec_checkResult_allDataPosition_6[31:11]};
  wire        indexVec_checkResult_unChange_6 = slideUp & (indexVec_readIndex_2[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_2 = validVec[2] & ~indexVec_checkResult_unChange_6;
  wire        indexVec_checkResult_0_4_2 = indexVec_checkResult_overlap_6 | ~indexVec_checkResult_0_5_2 | lagerThanVL | indexVec_checkResult_unChange_6;
  wire [32:0] indexVec_checkResult_allDataPosition_7 = {indexVec_readIndex_2, 1'h0};
  wire [10:0] indexVec_checkResult_dataPosition_7 = indexVec_checkResult_allDataPosition_7[10:0];
  wire [1:0]  indexVec_checkResult_1_0_2 = {indexVec_checkResult_dataPosition_7[1], 1'h0};
  wire [3:0]  indexVec_checkResult_1_1_2 = indexVec_checkResult_dataPosition_7[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_7 = indexVec_checkResult_dataPosition_7[10:6];
  wire [1:0]  indexVec_checkResult_1_2_2 = indexVec_checkResult_dataGroup_7[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_7 = indexVec_checkResult_dataGroup_7[4:2];
  wire [2:0]  indexVec_checkResult_1_3_2 = indexVec_checkResult_accessRegGrowth_7;
  wire [5:0]  indexVec_checkResult_decimalProportion_7 = {indexVec_checkResult_1_2_2, indexVec_checkResult_1_1_2};
  wire [2:0]  indexVec_checkResult_decimal_7 = indexVec_checkResult_decimalProportion_7[5:3];
  wire        indexVec_checkResult_overlap_7 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_7 >= indexVec_checkResult_intLMULInput_7[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_7} >= indexVec_checkResult_intLMULInput_7,
      indexVec_checkResult_allDataPosition_7[32:11]};
  wire        indexVec_checkResult_unChange_7 = slideUp & (indexVec_readIndex_2[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_2 = validVec[2] & ~indexVec_checkResult_unChange_7;
  wire        indexVec_checkResult_1_4_2 = indexVec_checkResult_overlap_7 | ~indexVec_checkResult_1_5_2 | lagerThanVL | indexVec_checkResult_unChange_7;
  wire [33:0] indexVec_checkResult_allDataPosition_8 = {indexVec_readIndex_2, 2'h0};
  wire [10:0] indexVec_checkResult_dataPosition_8 = indexVec_checkResult_allDataPosition_8[10:0];
  wire [3:0]  indexVec_checkResult_2_1_2 = indexVec_checkResult_dataPosition_8[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_8 = indexVec_checkResult_dataPosition_8[10:6];
  wire [1:0]  indexVec_checkResult_2_2_2 = indexVec_checkResult_dataGroup_8[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_8 = indexVec_checkResult_dataGroup_8[4:2];
  wire [2:0]  indexVec_checkResult_2_3_2 = indexVec_checkResult_accessRegGrowth_8;
  wire [5:0]  indexVec_checkResult_decimalProportion_8 = {indexVec_checkResult_2_2_2, indexVec_checkResult_2_1_2};
  wire [2:0]  indexVec_checkResult_decimal_8 = indexVec_checkResult_decimalProportion_8[5:3];
  wire        indexVec_checkResult_overlap_8 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_8 >= indexVec_checkResult_intLMULInput_8[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_8} >= indexVec_checkResult_intLMULInput_8,
      indexVec_checkResult_allDataPosition_8[33:11]};
  wire        indexVec_checkResult_unChange_8 = slideUp & (indexVec_readIndex_2[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_2 = validVec[2] & ~indexVec_checkResult_unChange_8;
  wire        indexVec_checkResult_2_4_2 = indexVec_checkResult_overlap_8 | ~indexVec_checkResult_2_5_2 | lagerThanVL | indexVec_checkResult_unChange_8;
  wire [1:0]  indexVec_2_0 = (sew1H[0] ? indexVec_checkResult_0_0_2 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_2 : 2'h0);
  assign indexVec_2_1 = (sew1H[0] ? indexVec_checkResult_0_1_2 : 4'h0) | (sew1H[1] ? indexVec_checkResult_1_1_2 : 4'h0) | (sew1H[2] ? indexVec_checkResult_2_1_2 : 4'h0);
  wire [3:0]  indexDeq_bits_accessLane_2_0 = indexVec_2_1;
  wire [1:0]  indexVec_2_2 = (sew1H[0] ? indexVec_checkResult_0_2_2 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_2 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_2 : 2'h0);
  assign indexVec_2_3 = (sew1H[0] ? indexVec_checkResult_0_3_2 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_2 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_2 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_2_0 = indexVec_2_3;
  wire        indexVec_2_4 = sew1H[0] & indexVec_checkResult_0_4_2 | sew1H[1] & indexVec_checkResult_1_4_2 | sew1H[2] & indexVec_checkResult_2_4_2;
  wire        indexVec_2_5 = sew1H[0] & indexVec_checkResult_0_5_2 | sew1H[1] & indexVec_checkResult_1_5_2 | sew1H[2] & indexVec_checkResult_2_5_2;
  wire [31:0] indexVec_readIndex_3 = baseIndex + 32'h3;
  wire [31:0] indexVec_checkResult_allDataPosition_9 = indexVec_readIndex_3;
  wire [10:0] indexVec_checkResult_dataPosition_9 = indexVec_checkResult_allDataPosition_9[10:0];
  wire [1:0]  indexVec_checkResult_0_0_3 = indexVec_checkResult_dataPosition_9[1:0];
  wire [3:0]  indexVec_checkResult_0_1_3 = indexVec_checkResult_dataPosition_9[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_9 = indexVec_checkResult_dataPosition_9[10:6];
  wire [1:0]  indexVec_checkResult_0_2_3 = indexVec_checkResult_dataGroup_9[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_9 = indexVec_checkResult_dataGroup_9[4:2];
  wire [2:0]  indexVec_checkResult_0_3_3 = indexVec_checkResult_accessRegGrowth_9;
  wire [5:0]  indexVec_checkResult_decimalProportion_9 = {indexVec_checkResult_0_2_3, indexVec_checkResult_0_1_3};
  wire [2:0]  indexVec_checkResult_decimal_9 = indexVec_checkResult_decimalProportion_9[5:3];
  wire        indexVec_checkResult_overlap_9 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_9 >= indexVec_checkResult_intLMULInput_9[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_9} >= indexVec_checkResult_intLMULInput_9,
      indexVec_checkResult_allDataPosition_9[31:11]};
  wire        indexVec_checkResult_unChange_9 = slideUp & (indexVec_readIndex_3[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_3 = validVec[3] & ~indexVec_checkResult_unChange_9;
  wire        indexVec_checkResult_0_4_3 = indexVec_checkResult_overlap_9 | ~indexVec_checkResult_0_5_3 | lagerThanVL | indexVec_checkResult_unChange_9;
  wire [32:0] indexVec_checkResult_allDataPosition_10 = {indexVec_readIndex_3, 1'h0};
  wire [10:0] indexVec_checkResult_dataPosition_10 = indexVec_checkResult_allDataPosition_10[10:0];
  wire [1:0]  indexVec_checkResult_1_0_3 = {indexVec_checkResult_dataPosition_10[1], 1'h0};
  wire [3:0]  indexVec_checkResult_1_1_3 = indexVec_checkResult_dataPosition_10[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_10 = indexVec_checkResult_dataPosition_10[10:6];
  wire [1:0]  indexVec_checkResult_1_2_3 = indexVec_checkResult_dataGroup_10[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_10 = indexVec_checkResult_dataGroup_10[4:2];
  wire [2:0]  indexVec_checkResult_1_3_3 = indexVec_checkResult_accessRegGrowth_10;
  wire [5:0]  indexVec_checkResult_decimalProportion_10 = {indexVec_checkResult_1_2_3, indexVec_checkResult_1_1_3};
  wire [2:0]  indexVec_checkResult_decimal_10 = indexVec_checkResult_decimalProportion_10[5:3];
  wire        indexVec_checkResult_overlap_10 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_10 >= indexVec_checkResult_intLMULInput_10[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_10} >= indexVec_checkResult_intLMULInput_10,
      indexVec_checkResult_allDataPosition_10[32:11]};
  wire        indexVec_checkResult_unChange_10 = slideUp & (indexVec_readIndex_3[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_3 = validVec[3] & ~indexVec_checkResult_unChange_10;
  wire        indexVec_checkResult_1_4_3 = indexVec_checkResult_overlap_10 | ~indexVec_checkResult_1_5_3 | lagerThanVL | indexVec_checkResult_unChange_10;
  wire [33:0] indexVec_checkResult_allDataPosition_11 = {indexVec_readIndex_3, 2'h0};
  wire [10:0] indexVec_checkResult_dataPosition_11 = indexVec_checkResult_allDataPosition_11[10:0];
  wire [3:0]  indexVec_checkResult_2_1_3 = indexVec_checkResult_dataPosition_11[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_11 = indexVec_checkResult_dataPosition_11[10:6];
  wire [1:0]  indexVec_checkResult_2_2_3 = indexVec_checkResult_dataGroup_11[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_11 = indexVec_checkResult_dataGroup_11[4:2];
  wire [2:0]  indexVec_checkResult_2_3_3 = indexVec_checkResult_accessRegGrowth_11;
  wire [5:0]  indexVec_checkResult_decimalProportion_11 = {indexVec_checkResult_2_2_3, indexVec_checkResult_2_1_3};
  wire [2:0]  indexVec_checkResult_decimal_11 = indexVec_checkResult_decimalProportion_11[5:3];
  wire        indexVec_checkResult_overlap_11 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_11 >= indexVec_checkResult_intLMULInput_11[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_11} >= indexVec_checkResult_intLMULInput_11,
      indexVec_checkResult_allDataPosition_11[33:11]};
  wire        indexVec_checkResult_unChange_11 = slideUp & (indexVec_readIndex_3[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_3 = validVec[3] & ~indexVec_checkResult_unChange_11;
  wire        indexVec_checkResult_2_4_3 = indexVec_checkResult_overlap_11 | ~indexVec_checkResult_2_5_3 | lagerThanVL | indexVec_checkResult_unChange_11;
  wire [1:0]  indexVec_3_0 = (sew1H[0] ? indexVec_checkResult_0_0_3 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_3 : 2'h0);
  assign indexVec_3_1 = (sew1H[0] ? indexVec_checkResult_0_1_3 : 4'h0) | (sew1H[1] ? indexVec_checkResult_1_1_3 : 4'h0) | (sew1H[2] ? indexVec_checkResult_2_1_3 : 4'h0);
  wire [3:0]  indexDeq_bits_accessLane_3_0 = indexVec_3_1;
  wire [1:0]  indexVec_3_2 = (sew1H[0] ? indexVec_checkResult_0_2_3 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_3 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_3 : 2'h0);
  assign indexVec_3_3 = (sew1H[0] ? indexVec_checkResult_0_3_3 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_3 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_3 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_3_0 = indexVec_3_3;
  wire        indexVec_3_4 = sew1H[0] & indexVec_checkResult_0_4_3 | sew1H[1] & indexVec_checkResult_1_4_3 | sew1H[2] & indexVec_checkResult_2_4_3;
  wire        indexVec_3_5 = sew1H[0] & indexVec_checkResult_0_5_3 | sew1H[1] & indexVec_checkResult_1_5_3 | sew1H[2] & indexVec_checkResult_2_5_3;
  wire [31:0] indexVec_readIndex_4 = baseIndex + 32'h4;
  wire [31:0] indexVec_checkResult_allDataPosition_12 = indexVec_readIndex_4;
  wire [10:0] indexVec_checkResult_dataPosition_12 = indexVec_checkResult_allDataPosition_12[10:0];
  wire [1:0]  indexVec_checkResult_0_0_4 = indexVec_checkResult_dataPosition_12[1:0];
  wire [3:0]  indexVec_checkResult_0_1_4 = indexVec_checkResult_dataPosition_12[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_12 = indexVec_checkResult_dataPosition_12[10:6];
  wire [1:0]  indexVec_checkResult_0_2_4 = indexVec_checkResult_dataGroup_12[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_12 = indexVec_checkResult_dataGroup_12[4:2];
  wire [2:0]  indexVec_checkResult_0_3_4 = indexVec_checkResult_accessRegGrowth_12;
  wire [5:0]  indexVec_checkResult_decimalProportion_12 = {indexVec_checkResult_0_2_4, indexVec_checkResult_0_1_4};
  wire [2:0]  indexVec_checkResult_decimal_12 = indexVec_checkResult_decimalProportion_12[5:3];
  wire        indexVec_checkResult_overlap_12 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_12 >= indexVec_checkResult_intLMULInput_12[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_12} >= indexVec_checkResult_intLMULInput_12,
      indexVec_checkResult_allDataPosition_12[31:11]};
  wire        indexVec_checkResult_unChange_12 = slideUp & (indexVec_readIndex_4[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_4 = validVec[4] & ~indexVec_checkResult_unChange_12;
  wire        indexVec_checkResult_0_4_4 = indexVec_checkResult_overlap_12 | ~indexVec_checkResult_0_5_4 | lagerThanVL | indexVec_checkResult_unChange_12;
  wire [32:0] indexVec_checkResult_allDataPosition_13 = {indexVec_readIndex_4, 1'h0};
  wire [10:0] indexVec_checkResult_dataPosition_13 = indexVec_checkResult_allDataPosition_13[10:0];
  wire [1:0]  indexVec_checkResult_1_0_4 = {indexVec_checkResult_dataPosition_13[1], 1'h0};
  wire [3:0]  indexVec_checkResult_1_1_4 = indexVec_checkResult_dataPosition_13[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_13 = indexVec_checkResult_dataPosition_13[10:6];
  wire [1:0]  indexVec_checkResult_1_2_4 = indexVec_checkResult_dataGroup_13[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_13 = indexVec_checkResult_dataGroup_13[4:2];
  wire [2:0]  indexVec_checkResult_1_3_4 = indexVec_checkResult_accessRegGrowth_13;
  wire [5:0]  indexVec_checkResult_decimalProportion_13 = {indexVec_checkResult_1_2_4, indexVec_checkResult_1_1_4};
  wire [2:0]  indexVec_checkResult_decimal_13 = indexVec_checkResult_decimalProportion_13[5:3];
  wire        indexVec_checkResult_overlap_13 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_13 >= indexVec_checkResult_intLMULInput_13[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_13} >= indexVec_checkResult_intLMULInput_13,
      indexVec_checkResult_allDataPosition_13[32:11]};
  wire        indexVec_checkResult_unChange_13 = slideUp & (indexVec_readIndex_4[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_4 = validVec[4] & ~indexVec_checkResult_unChange_13;
  wire        indexVec_checkResult_1_4_4 = indexVec_checkResult_overlap_13 | ~indexVec_checkResult_1_5_4 | lagerThanVL | indexVec_checkResult_unChange_13;
  wire [33:0] indexVec_checkResult_allDataPosition_14 = {indexVec_readIndex_4, 2'h0};
  wire [10:0] indexVec_checkResult_dataPosition_14 = indexVec_checkResult_allDataPosition_14[10:0];
  wire [3:0]  indexVec_checkResult_2_1_4 = indexVec_checkResult_dataPosition_14[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_14 = indexVec_checkResult_dataPosition_14[10:6];
  wire [1:0]  indexVec_checkResult_2_2_4 = indexVec_checkResult_dataGroup_14[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_14 = indexVec_checkResult_dataGroup_14[4:2];
  wire [2:0]  indexVec_checkResult_2_3_4 = indexVec_checkResult_accessRegGrowth_14;
  wire [5:0]  indexVec_checkResult_decimalProportion_14 = {indexVec_checkResult_2_2_4, indexVec_checkResult_2_1_4};
  wire [2:0]  indexVec_checkResult_decimal_14 = indexVec_checkResult_decimalProportion_14[5:3];
  wire        indexVec_checkResult_overlap_14 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_14 >= indexVec_checkResult_intLMULInput_14[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_14} >= indexVec_checkResult_intLMULInput_14,
      indexVec_checkResult_allDataPosition_14[33:11]};
  wire        indexVec_checkResult_unChange_14 = slideUp & (indexVec_readIndex_4[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_4 = validVec[4] & ~indexVec_checkResult_unChange_14;
  wire        indexVec_checkResult_2_4_4 = indexVec_checkResult_overlap_14 | ~indexVec_checkResult_2_5_4 | lagerThanVL | indexVec_checkResult_unChange_14;
  wire [1:0]  indexVec_4_0 = (sew1H[0] ? indexVec_checkResult_0_0_4 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_4 : 2'h0);
  assign indexVec_4_1 = (sew1H[0] ? indexVec_checkResult_0_1_4 : 4'h0) | (sew1H[1] ? indexVec_checkResult_1_1_4 : 4'h0) | (sew1H[2] ? indexVec_checkResult_2_1_4 : 4'h0);
  wire [3:0]  indexDeq_bits_accessLane_4_0 = indexVec_4_1;
  wire [1:0]  indexVec_4_2 = (sew1H[0] ? indexVec_checkResult_0_2_4 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_4 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_4 : 2'h0);
  assign indexVec_4_3 = (sew1H[0] ? indexVec_checkResult_0_3_4 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_4 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_4 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_4_0 = indexVec_4_3;
  wire        indexVec_4_4 = sew1H[0] & indexVec_checkResult_0_4_4 | sew1H[1] & indexVec_checkResult_1_4_4 | sew1H[2] & indexVec_checkResult_2_4_4;
  wire        indexVec_4_5 = sew1H[0] & indexVec_checkResult_0_5_4 | sew1H[1] & indexVec_checkResult_1_5_4 | sew1H[2] & indexVec_checkResult_2_5_4;
  wire [31:0] indexVec_readIndex_5 = baseIndex + 32'h5;
  wire [31:0] indexVec_checkResult_allDataPosition_15 = indexVec_readIndex_5;
  wire [10:0] indexVec_checkResult_dataPosition_15 = indexVec_checkResult_allDataPosition_15[10:0];
  wire [1:0]  indexVec_checkResult_0_0_5 = indexVec_checkResult_dataPosition_15[1:0];
  wire [3:0]  indexVec_checkResult_0_1_5 = indexVec_checkResult_dataPosition_15[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_15 = indexVec_checkResult_dataPosition_15[10:6];
  wire [1:0]  indexVec_checkResult_0_2_5 = indexVec_checkResult_dataGroup_15[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_15 = indexVec_checkResult_dataGroup_15[4:2];
  wire [2:0]  indexVec_checkResult_0_3_5 = indexVec_checkResult_accessRegGrowth_15;
  wire [5:0]  indexVec_checkResult_decimalProportion_15 = {indexVec_checkResult_0_2_5, indexVec_checkResult_0_1_5};
  wire [2:0]  indexVec_checkResult_decimal_15 = indexVec_checkResult_decimalProportion_15[5:3];
  wire        indexVec_checkResult_overlap_15 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_15 >= indexVec_checkResult_intLMULInput_15[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_15} >= indexVec_checkResult_intLMULInput_15,
      indexVec_checkResult_allDataPosition_15[31:11]};
  wire        indexVec_checkResult_unChange_15 = slideUp & (indexVec_readIndex_5[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_5 = validVec[5] & ~indexVec_checkResult_unChange_15;
  wire        indexVec_checkResult_0_4_5 = indexVec_checkResult_overlap_15 | ~indexVec_checkResult_0_5_5 | lagerThanVL | indexVec_checkResult_unChange_15;
  wire [32:0] indexVec_checkResult_allDataPosition_16 = {indexVec_readIndex_5, 1'h0};
  wire [10:0] indexVec_checkResult_dataPosition_16 = indexVec_checkResult_allDataPosition_16[10:0];
  wire [1:0]  indexVec_checkResult_1_0_5 = {indexVec_checkResult_dataPosition_16[1], 1'h0};
  wire [3:0]  indexVec_checkResult_1_1_5 = indexVec_checkResult_dataPosition_16[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_16 = indexVec_checkResult_dataPosition_16[10:6];
  wire [1:0]  indexVec_checkResult_1_2_5 = indexVec_checkResult_dataGroup_16[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_16 = indexVec_checkResult_dataGroup_16[4:2];
  wire [2:0]  indexVec_checkResult_1_3_5 = indexVec_checkResult_accessRegGrowth_16;
  wire [5:0]  indexVec_checkResult_decimalProportion_16 = {indexVec_checkResult_1_2_5, indexVec_checkResult_1_1_5};
  wire [2:0]  indexVec_checkResult_decimal_16 = indexVec_checkResult_decimalProportion_16[5:3];
  wire        indexVec_checkResult_overlap_16 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_16 >= indexVec_checkResult_intLMULInput_16[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_16} >= indexVec_checkResult_intLMULInput_16,
      indexVec_checkResult_allDataPosition_16[32:11]};
  wire        indexVec_checkResult_unChange_16 = slideUp & (indexVec_readIndex_5[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_5 = validVec[5] & ~indexVec_checkResult_unChange_16;
  wire        indexVec_checkResult_1_4_5 = indexVec_checkResult_overlap_16 | ~indexVec_checkResult_1_5_5 | lagerThanVL | indexVec_checkResult_unChange_16;
  wire [33:0] indexVec_checkResult_allDataPosition_17 = {indexVec_readIndex_5, 2'h0};
  wire [10:0] indexVec_checkResult_dataPosition_17 = indexVec_checkResult_allDataPosition_17[10:0];
  wire [3:0]  indexVec_checkResult_2_1_5 = indexVec_checkResult_dataPosition_17[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_17 = indexVec_checkResult_dataPosition_17[10:6];
  wire [1:0]  indexVec_checkResult_2_2_5 = indexVec_checkResult_dataGroup_17[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_17 = indexVec_checkResult_dataGroup_17[4:2];
  wire [2:0]  indexVec_checkResult_2_3_5 = indexVec_checkResult_accessRegGrowth_17;
  wire [5:0]  indexVec_checkResult_decimalProportion_17 = {indexVec_checkResult_2_2_5, indexVec_checkResult_2_1_5};
  wire [2:0]  indexVec_checkResult_decimal_17 = indexVec_checkResult_decimalProportion_17[5:3];
  wire        indexVec_checkResult_overlap_17 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_17 >= indexVec_checkResult_intLMULInput_17[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_17} >= indexVec_checkResult_intLMULInput_17,
      indexVec_checkResult_allDataPosition_17[33:11]};
  wire        indexVec_checkResult_unChange_17 = slideUp & (indexVec_readIndex_5[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_5 = validVec[5] & ~indexVec_checkResult_unChange_17;
  wire        indexVec_checkResult_2_4_5 = indexVec_checkResult_overlap_17 | ~indexVec_checkResult_2_5_5 | lagerThanVL | indexVec_checkResult_unChange_17;
  wire [1:0]  indexVec_5_0 = (sew1H[0] ? indexVec_checkResult_0_0_5 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_5 : 2'h0);
  assign indexVec_5_1 = (sew1H[0] ? indexVec_checkResult_0_1_5 : 4'h0) | (sew1H[1] ? indexVec_checkResult_1_1_5 : 4'h0) | (sew1H[2] ? indexVec_checkResult_2_1_5 : 4'h0);
  wire [3:0]  indexDeq_bits_accessLane_5_0 = indexVec_5_1;
  wire [1:0]  indexVec_5_2 = (sew1H[0] ? indexVec_checkResult_0_2_5 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_5 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_5 : 2'h0);
  assign indexVec_5_3 = (sew1H[0] ? indexVec_checkResult_0_3_5 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_5 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_5 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_5_0 = indexVec_5_3;
  wire        indexVec_5_4 = sew1H[0] & indexVec_checkResult_0_4_5 | sew1H[1] & indexVec_checkResult_1_4_5 | sew1H[2] & indexVec_checkResult_2_4_5;
  wire        indexVec_5_5 = sew1H[0] & indexVec_checkResult_0_5_5 | sew1H[1] & indexVec_checkResult_1_5_5 | sew1H[2] & indexVec_checkResult_2_5_5;
  wire [31:0] indexVec_readIndex_6 = baseIndex + 32'h6;
  wire [31:0] indexVec_checkResult_allDataPosition_18 = indexVec_readIndex_6;
  wire [10:0] indexVec_checkResult_dataPosition_18 = indexVec_checkResult_allDataPosition_18[10:0];
  wire [1:0]  indexVec_checkResult_0_0_6 = indexVec_checkResult_dataPosition_18[1:0];
  wire [3:0]  indexVec_checkResult_0_1_6 = indexVec_checkResult_dataPosition_18[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_18 = indexVec_checkResult_dataPosition_18[10:6];
  wire [1:0]  indexVec_checkResult_0_2_6 = indexVec_checkResult_dataGroup_18[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_18 = indexVec_checkResult_dataGroup_18[4:2];
  wire [2:0]  indexVec_checkResult_0_3_6 = indexVec_checkResult_accessRegGrowth_18;
  wire [5:0]  indexVec_checkResult_decimalProportion_18 = {indexVec_checkResult_0_2_6, indexVec_checkResult_0_1_6};
  wire [2:0]  indexVec_checkResult_decimal_18 = indexVec_checkResult_decimalProportion_18[5:3];
  wire        indexVec_checkResult_overlap_18 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_18 >= indexVec_checkResult_intLMULInput_18[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_18} >= indexVec_checkResult_intLMULInput_18,
      indexVec_checkResult_allDataPosition_18[31:11]};
  wire        indexVec_checkResult_unChange_18 = slideUp & (indexVec_readIndex_6[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_6 = validVec[6] & ~indexVec_checkResult_unChange_18;
  wire        indexVec_checkResult_0_4_6 = indexVec_checkResult_overlap_18 | ~indexVec_checkResult_0_5_6 | lagerThanVL | indexVec_checkResult_unChange_18;
  wire [32:0] indexVec_checkResult_allDataPosition_19 = {indexVec_readIndex_6, 1'h0};
  wire [10:0] indexVec_checkResult_dataPosition_19 = indexVec_checkResult_allDataPosition_19[10:0];
  wire [1:0]  indexVec_checkResult_1_0_6 = {indexVec_checkResult_dataPosition_19[1], 1'h0};
  wire [3:0]  indexVec_checkResult_1_1_6 = indexVec_checkResult_dataPosition_19[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_19 = indexVec_checkResult_dataPosition_19[10:6];
  wire [1:0]  indexVec_checkResult_1_2_6 = indexVec_checkResult_dataGroup_19[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_19 = indexVec_checkResult_dataGroup_19[4:2];
  wire [2:0]  indexVec_checkResult_1_3_6 = indexVec_checkResult_accessRegGrowth_19;
  wire [5:0]  indexVec_checkResult_decimalProportion_19 = {indexVec_checkResult_1_2_6, indexVec_checkResult_1_1_6};
  wire [2:0]  indexVec_checkResult_decimal_19 = indexVec_checkResult_decimalProportion_19[5:3];
  wire        indexVec_checkResult_overlap_19 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_19 >= indexVec_checkResult_intLMULInput_19[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_19} >= indexVec_checkResult_intLMULInput_19,
      indexVec_checkResult_allDataPosition_19[32:11]};
  wire        indexVec_checkResult_unChange_19 = slideUp & (indexVec_readIndex_6[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_6 = validVec[6] & ~indexVec_checkResult_unChange_19;
  wire        indexVec_checkResult_1_4_6 = indexVec_checkResult_overlap_19 | ~indexVec_checkResult_1_5_6 | lagerThanVL | indexVec_checkResult_unChange_19;
  wire [33:0] indexVec_checkResult_allDataPosition_20 = {indexVec_readIndex_6, 2'h0};
  wire [10:0] indexVec_checkResult_dataPosition_20 = indexVec_checkResult_allDataPosition_20[10:0];
  wire [3:0]  indexVec_checkResult_2_1_6 = indexVec_checkResult_dataPosition_20[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_20 = indexVec_checkResult_dataPosition_20[10:6];
  wire [1:0]  indexVec_checkResult_2_2_6 = indexVec_checkResult_dataGroup_20[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_20 = indexVec_checkResult_dataGroup_20[4:2];
  wire [2:0]  indexVec_checkResult_2_3_6 = indexVec_checkResult_accessRegGrowth_20;
  wire [5:0]  indexVec_checkResult_decimalProportion_20 = {indexVec_checkResult_2_2_6, indexVec_checkResult_2_1_6};
  wire [2:0]  indexVec_checkResult_decimal_20 = indexVec_checkResult_decimalProportion_20[5:3];
  wire        indexVec_checkResult_overlap_20 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_20 >= indexVec_checkResult_intLMULInput_20[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_20} >= indexVec_checkResult_intLMULInput_20,
      indexVec_checkResult_allDataPosition_20[33:11]};
  wire        indexVec_checkResult_unChange_20 = slideUp & (indexVec_readIndex_6[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_6 = validVec[6] & ~indexVec_checkResult_unChange_20;
  wire        indexVec_checkResult_2_4_6 = indexVec_checkResult_overlap_20 | ~indexVec_checkResult_2_5_6 | lagerThanVL | indexVec_checkResult_unChange_20;
  wire [1:0]  indexVec_6_0 = (sew1H[0] ? indexVec_checkResult_0_0_6 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_6 : 2'h0);
  assign indexVec_6_1 = (sew1H[0] ? indexVec_checkResult_0_1_6 : 4'h0) | (sew1H[1] ? indexVec_checkResult_1_1_6 : 4'h0) | (sew1H[2] ? indexVec_checkResult_2_1_6 : 4'h0);
  wire [3:0]  indexDeq_bits_accessLane_6_0 = indexVec_6_1;
  wire [1:0]  indexVec_6_2 = (sew1H[0] ? indexVec_checkResult_0_2_6 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_6 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_6 : 2'h0);
  assign indexVec_6_3 = (sew1H[0] ? indexVec_checkResult_0_3_6 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_6 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_6 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_6_0 = indexVec_6_3;
  wire        indexVec_6_4 = sew1H[0] & indexVec_checkResult_0_4_6 | sew1H[1] & indexVec_checkResult_1_4_6 | sew1H[2] & indexVec_checkResult_2_4_6;
  wire        indexVec_6_5 = sew1H[0] & indexVec_checkResult_0_5_6 | sew1H[1] & indexVec_checkResult_1_5_6 | sew1H[2] & indexVec_checkResult_2_5_6;
  wire [31:0] indexVec_readIndex_7 = baseIndex + 32'h7;
  wire [31:0] indexVec_checkResult_allDataPosition_21 = indexVec_readIndex_7;
  wire [10:0] indexVec_checkResult_dataPosition_21 = indexVec_checkResult_allDataPosition_21[10:0];
  wire [1:0]  indexVec_checkResult_0_0_7 = indexVec_checkResult_dataPosition_21[1:0];
  wire [3:0]  indexVec_checkResult_0_1_7 = indexVec_checkResult_dataPosition_21[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_21 = indexVec_checkResult_dataPosition_21[10:6];
  wire [1:0]  indexVec_checkResult_0_2_7 = indexVec_checkResult_dataGroup_21[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_21 = indexVec_checkResult_dataGroup_21[4:2];
  wire [2:0]  indexVec_checkResult_0_3_7 = indexVec_checkResult_accessRegGrowth_21;
  wire [5:0]  indexVec_checkResult_decimalProportion_21 = {indexVec_checkResult_0_2_7, indexVec_checkResult_0_1_7};
  wire [2:0]  indexVec_checkResult_decimal_21 = indexVec_checkResult_decimalProportion_21[5:3];
  wire        indexVec_checkResult_overlap_21 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_21 >= indexVec_checkResult_intLMULInput_21[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_21} >= indexVec_checkResult_intLMULInput_21,
      indexVec_checkResult_allDataPosition_21[31:11]};
  wire        indexVec_checkResult_unChange_21 = slideUp & (indexVec_readIndex_7[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_7 = validVec[7] & ~indexVec_checkResult_unChange_21;
  wire        indexVec_checkResult_0_4_7 = indexVec_checkResult_overlap_21 | ~indexVec_checkResult_0_5_7 | lagerThanVL | indexVec_checkResult_unChange_21;
  wire [32:0] indexVec_checkResult_allDataPosition_22 = {indexVec_readIndex_7, 1'h0};
  wire [10:0] indexVec_checkResult_dataPosition_22 = indexVec_checkResult_allDataPosition_22[10:0];
  wire [1:0]  indexVec_checkResult_1_0_7 = {indexVec_checkResult_dataPosition_22[1], 1'h0};
  wire [3:0]  indexVec_checkResult_1_1_7 = indexVec_checkResult_dataPosition_22[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_22 = indexVec_checkResult_dataPosition_22[10:6];
  wire [1:0]  indexVec_checkResult_1_2_7 = indexVec_checkResult_dataGroup_22[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_22 = indexVec_checkResult_dataGroup_22[4:2];
  wire [2:0]  indexVec_checkResult_1_3_7 = indexVec_checkResult_accessRegGrowth_22;
  wire [5:0]  indexVec_checkResult_decimalProportion_22 = {indexVec_checkResult_1_2_7, indexVec_checkResult_1_1_7};
  wire [2:0]  indexVec_checkResult_decimal_22 = indexVec_checkResult_decimalProportion_22[5:3];
  wire        indexVec_checkResult_overlap_22 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_22 >= indexVec_checkResult_intLMULInput_22[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_22} >= indexVec_checkResult_intLMULInput_22,
      indexVec_checkResult_allDataPosition_22[32:11]};
  wire        indexVec_checkResult_unChange_22 = slideUp & (indexVec_readIndex_7[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_7 = validVec[7] & ~indexVec_checkResult_unChange_22;
  wire        indexVec_checkResult_1_4_7 = indexVec_checkResult_overlap_22 | ~indexVec_checkResult_1_5_7 | lagerThanVL | indexVec_checkResult_unChange_22;
  wire [33:0] indexVec_checkResult_allDataPosition_23 = {indexVec_readIndex_7, 2'h0};
  wire [10:0] indexVec_checkResult_dataPosition_23 = indexVec_checkResult_allDataPosition_23[10:0];
  wire [3:0]  indexVec_checkResult_2_1_7 = indexVec_checkResult_dataPosition_23[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_23 = indexVec_checkResult_dataPosition_23[10:6];
  wire [1:0]  indexVec_checkResult_2_2_7 = indexVec_checkResult_dataGroup_23[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_23 = indexVec_checkResult_dataGroup_23[4:2];
  wire [2:0]  indexVec_checkResult_2_3_7 = indexVec_checkResult_accessRegGrowth_23;
  wire [5:0]  indexVec_checkResult_decimalProportion_23 = {indexVec_checkResult_2_2_7, indexVec_checkResult_2_1_7};
  wire [2:0]  indexVec_checkResult_decimal_23 = indexVec_checkResult_decimalProportion_23[5:3];
  wire        indexVec_checkResult_overlap_23 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_23 >= indexVec_checkResult_intLMULInput_23[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_23} >= indexVec_checkResult_intLMULInput_23,
      indexVec_checkResult_allDataPosition_23[33:11]};
  wire        indexVec_checkResult_unChange_23 = slideUp & (indexVec_readIndex_7[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_7 = validVec[7] & ~indexVec_checkResult_unChange_23;
  wire        indexVec_checkResult_2_4_7 = indexVec_checkResult_overlap_23 | ~indexVec_checkResult_2_5_7 | lagerThanVL | indexVec_checkResult_unChange_23;
  wire [1:0]  indexVec_7_0 = (sew1H[0] ? indexVec_checkResult_0_0_7 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_7 : 2'h0);
  assign indexVec_7_1 = (sew1H[0] ? indexVec_checkResult_0_1_7 : 4'h0) | (sew1H[1] ? indexVec_checkResult_1_1_7 : 4'h0) | (sew1H[2] ? indexVec_checkResult_2_1_7 : 4'h0);
  wire [3:0]  indexDeq_bits_accessLane_7_0 = indexVec_7_1;
  wire [1:0]  indexVec_7_2 = (sew1H[0] ? indexVec_checkResult_0_2_7 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_7 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_7 : 2'h0);
  assign indexVec_7_3 = (sew1H[0] ? indexVec_checkResult_0_3_7 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_7 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_7 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_7_0 = indexVec_7_3;
  wire        indexVec_7_4 = sew1H[0] & indexVec_checkResult_0_4_7 | sew1H[1] & indexVec_checkResult_1_4_7 | sew1H[2] & indexVec_checkResult_2_4_7;
  wire        indexVec_7_5 = sew1H[0] & indexVec_checkResult_0_5_7 | sew1H[1] & indexVec_checkResult_1_5_7 | sew1H[2] & indexVec_checkResult_2_5_7;
  wire [31:0] indexVec_readIndex_8 = baseIndex + 32'h8;
  wire [31:0] indexVec_checkResult_allDataPosition_24 = indexVec_readIndex_8;
  wire [10:0] indexVec_checkResult_dataPosition_24 = indexVec_checkResult_allDataPosition_24[10:0];
  wire [1:0]  indexVec_checkResult_0_0_8 = indexVec_checkResult_dataPosition_24[1:0];
  wire [3:0]  indexVec_checkResult_0_1_8 = indexVec_checkResult_dataPosition_24[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_24 = indexVec_checkResult_dataPosition_24[10:6];
  wire [1:0]  indexVec_checkResult_0_2_8 = indexVec_checkResult_dataGroup_24[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_24 = indexVec_checkResult_dataGroup_24[4:2];
  wire [2:0]  indexVec_checkResult_0_3_8 = indexVec_checkResult_accessRegGrowth_24;
  wire [5:0]  indexVec_checkResult_decimalProportion_24 = {indexVec_checkResult_0_2_8, indexVec_checkResult_0_1_8};
  wire [2:0]  indexVec_checkResult_decimal_24 = indexVec_checkResult_decimalProportion_24[5:3];
  wire        indexVec_checkResult_overlap_24 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_24 >= indexVec_checkResult_intLMULInput_24[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_24} >= indexVec_checkResult_intLMULInput_24,
      indexVec_checkResult_allDataPosition_24[31:11]};
  wire        indexVec_checkResult_unChange_24 = slideUp & (indexVec_readIndex_8[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_8 = validVec[8] & ~indexVec_checkResult_unChange_24;
  wire        indexVec_checkResult_0_4_8 = indexVec_checkResult_overlap_24 | ~indexVec_checkResult_0_5_8 | lagerThanVL | indexVec_checkResult_unChange_24;
  wire [32:0] indexVec_checkResult_allDataPosition_25 = {indexVec_readIndex_8, 1'h0};
  wire [10:0] indexVec_checkResult_dataPosition_25 = indexVec_checkResult_allDataPosition_25[10:0];
  wire [1:0]  indexVec_checkResult_1_0_8 = {indexVec_checkResult_dataPosition_25[1], 1'h0};
  wire [3:0]  indexVec_checkResult_1_1_8 = indexVec_checkResult_dataPosition_25[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_25 = indexVec_checkResult_dataPosition_25[10:6];
  wire [1:0]  indexVec_checkResult_1_2_8 = indexVec_checkResult_dataGroup_25[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_25 = indexVec_checkResult_dataGroup_25[4:2];
  wire [2:0]  indexVec_checkResult_1_3_8 = indexVec_checkResult_accessRegGrowth_25;
  wire [5:0]  indexVec_checkResult_decimalProportion_25 = {indexVec_checkResult_1_2_8, indexVec_checkResult_1_1_8};
  wire [2:0]  indexVec_checkResult_decimal_25 = indexVec_checkResult_decimalProportion_25[5:3];
  wire        indexVec_checkResult_overlap_25 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_25 >= indexVec_checkResult_intLMULInput_25[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_25} >= indexVec_checkResult_intLMULInput_25,
      indexVec_checkResult_allDataPosition_25[32:11]};
  wire        indexVec_checkResult_unChange_25 = slideUp & (indexVec_readIndex_8[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_8 = validVec[8] & ~indexVec_checkResult_unChange_25;
  wire        indexVec_checkResult_1_4_8 = indexVec_checkResult_overlap_25 | ~indexVec_checkResult_1_5_8 | lagerThanVL | indexVec_checkResult_unChange_25;
  wire [33:0] indexVec_checkResult_allDataPosition_26 = {indexVec_readIndex_8, 2'h0};
  wire [10:0] indexVec_checkResult_dataPosition_26 = indexVec_checkResult_allDataPosition_26[10:0];
  wire [3:0]  indexVec_checkResult_2_1_8 = indexVec_checkResult_dataPosition_26[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_26 = indexVec_checkResult_dataPosition_26[10:6];
  wire [1:0]  indexVec_checkResult_2_2_8 = indexVec_checkResult_dataGroup_26[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_26 = indexVec_checkResult_dataGroup_26[4:2];
  wire [2:0]  indexVec_checkResult_2_3_8 = indexVec_checkResult_accessRegGrowth_26;
  wire [5:0]  indexVec_checkResult_decimalProportion_26 = {indexVec_checkResult_2_2_8, indexVec_checkResult_2_1_8};
  wire [2:0]  indexVec_checkResult_decimal_26 = indexVec_checkResult_decimalProportion_26[5:3];
  wire        indexVec_checkResult_overlap_26 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_26 >= indexVec_checkResult_intLMULInput_26[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_26} >= indexVec_checkResult_intLMULInput_26,
      indexVec_checkResult_allDataPosition_26[33:11]};
  wire        indexVec_checkResult_unChange_26 = slideUp & (indexVec_readIndex_8[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_8 = validVec[8] & ~indexVec_checkResult_unChange_26;
  wire        indexVec_checkResult_2_4_8 = indexVec_checkResult_overlap_26 | ~indexVec_checkResult_2_5_8 | lagerThanVL | indexVec_checkResult_unChange_26;
  wire [1:0]  indexVec_8_0 = (sew1H[0] ? indexVec_checkResult_0_0_8 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_8 : 2'h0);
  assign indexVec_8_1 = (sew1H[0] ? indexVec_checkResult_0_1_8 : 4'h0) | (sew1H[1] ? indexVec_checkResult_1_1_8 : 4'h0) | (sew1H[2] ? indexVec_checkResult_2_1_8 : 4'h0);
  wire [3:0]  indexDeq_bits_accessLane_8_0 = indexVec_8_1;
  wire [1:0]  indexVec_8_2 = (sew1H[0] ? indexVec_checkResult_0_2_8 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_8 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_8 : 2'h0);
  assign indexVec_8_3 = (sew1H[0] ? indexVec_checkResult_0_3_8 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_8 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_8 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_8_0 = indexVec_8_3;
  wire        indexVec_8_4 = sew1H[0] & indexVec_checkResult_0_4_8 | sew1H[1] & indexVec_checkResult_1_4_8 | sew1H[2] & indexVec_checkResult_2_4_8;
  wire        indexVec_8_5 = sew1H[0] & indexVec_checkResult_0_5_8 | sew1H[1] & indexVec_checkResult_1_5_8 | sew1H[2] & indexVec_checkResult_2_5_8;
  wire [31:0] indexVec_readIndex_9 = baseIndex + 32'h9;
  wire [31:0] indexVec_checkResult_allDataPosition_27 = indexVec_readIndex_9;
  wire [10:0] indexVec_checkResult_dataPosition_27 = indexVec_checkResult_allDataPosition_27[10:0];
  wire [1:0]  indexVec_checkResult_0_0_9 = indexVec_checkResult_dataPosition_27[1:0];
  wire [3:0]  indexVec_checkResult_0_1_9 = indexVec_checkResult_dataPosition_27[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_27 = indexVec_checkResult_dataPosition_27[10:6];
  wire [1:0]  indexVec_checkResult_0_2_9 = indexVec_checkResult_dataGroup_27[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_27 = indexVec_checkResult_dataGroup_27[4:2];
  wire [2:0]  indexVec_checkResult_0_3_9 = indexVec_checkResult_accessRegGrowth_27;
  wire [5:0]  indexVec_checkResult_decimalProportion_27 = {indexVec_checkResult_0_2_9, indexVec_checkResult_0_1_9};
  wire [2:0]  indexVec_checkResult_decimal_27 = indexVec_checkResult_decimalProportion_27[5:3];
  wire        indexVec_checkResult_overlap_27 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_27 >= indexVec_checkResult_intLMULInput_27[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_27} >= indexVec_checkResult_intLMULInput_27,
      indexVec_checkResult_allDataPosition_27[31:11]};
  wire        indexVec_checkResult_unChange_27 = slideUp & (indexVec_readIndex_9[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_9 = validVec[9] & ~indexVec_checkResult_unChange_27;
  wire        indexVec_checkResult_0_4_9 = indexVec_checkResult_overlap_27 | ~indexVec_checkResult_0_5_9 | lagerThanVL | indexVec_checkResult_unChange_27;
  wire [32:0] indexVec_checkResult_allDataPosition_28 = {indexVec_readIndex_9, 1'h0};
  wire [10:0] indexVec_checkResult_dataPosition_28 = indexVec_checkResult_allDataPosition_28[10:0];
  wire [1:0]  indexVec_checkResult_1_0_9 = {indexVec_checkResult_dataPosition_28[1], 1'h0};
  wire [3:0]  indexVec_checkResult_1_1_9 = indexVec_checkResult_dataPosition_28[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_28 = indexVec_checkResult_dataPosition_28[10:6];
  wire [1:0]  indexVec_checkResult_1_2_9 = indexVec_checkResult_dataGroup_28[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_28 = indexVec_checkResult_dataGroup_28[4:2];
  wire [2:0]  indexVec_checkResult_1_3_9 = indexVec_checkResult_accessRegGrowth_28;
  wire [5:0]  indexVec_checkResult_decimalProportion_28 = {indexVec_checkResult_1_2_9, indexVec_checkResult_1_1_9};
  wire [2:0]  indexVec_checkResult_decimal_28 = indexVec_checkResult_decimalProportion_28[5:3];
  wire        indexVec_checkResult_overlap_28 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_28 >= indexVec_checkResult_intLMULInput_28[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_28} >= indexVec_checkResult_intLMULInput_28,
      indexVec_checkResult_allDataPosition_28[32:11]};
  wire        indexVec_checkResult_unChange_28 = slideUp & (indexVec_readIndex_9[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_9 = validVec[9] & ~indexVec_checkResult_unChange_28;
  wire        indexVec_checkResult_1_4_9 = indexVec_checkResult_overlap_28 | ~indexVec_checkResult_1_5_9 | lagerThanVL | indexVec_checkResult_unChange_28;
  wire [33:0] indexVec_checkResult_allDataPosition_29 = {indexVec_readIndex_9, 2'h0};
  wire [10:0] indexVec_checkResult_dataPosition_29 = indexVec_checkResult_allDataPosition_29[10:0];
  wire [3:0]  indexVec_checkResult_2_1_9 = indexVec_checkResult_dataPosition_29[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_29 = indexVec_checkResult_dataPosition_29[10:6];
  wire [1:0]  indexVec_checkResult_2_2_9 = indexVec_checkResult_dataGroup_29[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_29 = indexVec_checkResult_dataGroup_29[4:2];
  wire [2:0]  indexVec_checkResult_2_3_9 = indexVec_checkResult_accessRegGrowth_29;
  wire [5:0]  indexVec_checkResult_decimalProportion_29 = {indexVec_checkResult_2_2_9, indexVec_checkResult_2_1_9};
  wire [2:0]  indexVec_checkResult_decimal_29 = indexVec_checkResult_decimalProportion_29[5:3];
  wire        indexVec_checkResult_overlap_29 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_29 >= indexVec_checkResult_intLMULInput_29[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_29} >= indexVec_checkResult_intLMULInput_29,
      indexVec_checkResult_allDataPosition_29[33:11]};
  wire        indexVec_checkResult_unChange_29 = slideUp & (indexVec_readIndex_9[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_9 = validVec[9] & ~indexVec_checkResult_unChange_29;
  wire        indexVec_checkResult_2_4_9 = indexVec_checkResult_overlap_29 | ~indexVec_checkResult_2_5_9 | lagerThanVL | indexVec_checkResult_unChange_29;
  wire [1:0]  indexVec_9_0 = (sew1H[0] ? indexVec_checkResult_0_0_9 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_9 : 2'h0);
  assign indexVec_9_1 = (sew1H[0] ? indexVec_checkResult_0_1_9 : 4'h0) | (sew1H[1] ? indexVec_checkResult_1_1_9 : 4'h0) | (sew1H[2] ? indexVec_checkResult_2_1_9 : 4'h0);
  wire [3:0]  indexDeq_bits_accessLane_9_0 = indexVec_9_1;
  wire [1:0]  indexVec_9_2 = (sew1H[0] ? indexVec_checkResult_0_2_9 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_9 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_9 : 2'h0);
  assign indexVec_9_3 = (sew1H[0] ? indexVec_checkResult_0_3_9 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_9 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_9 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_9_0 = indexVec_9_3;
  wire        indexVec_9_4 = sew1H[0] & indexVec_checkResult_0_4_9 | sew1H[1] & indexVec_checkResult_1_4_9 | sew1H[2] & indexVec_checkResult_2_4_9;
  wire        indexVec_9_5 = sew1H[0] & indexVec_checkResult_0_5_9 | sew1H[1] & indexVec_checkResult_1_5_9 | sew1H[2] & indexVec_checkResult_2_5_9;
  wire [31:0] indexVec_readIndex_10 = baseIndex + 32'hA;
  wire [31:0] indexVec_checkResult_allDataPosition_30 = indexVec_readIndex_10;
  wire [10:0] indexVec_checkResult_dataPosition_30 = indexVec_checkResult_allDataPosition_30[10:0];
  wire [1:0]  indexVec_checkResult_0_0_10 = indexVec_checkResult_dataPosition_30[1:0];
  wire [3:0]  indexVec_checkResult_0_1_10 = indexVec_checkResult_dataPosition_30[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_30 = indexVec_checkResult_dataPosition_30[10:6];
  wire [1:0]  indexVec_checkResult_0_2_10 = indexVec_checkResult_dataGroup_30[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_30 = indexVec_checkResult_dataGroup_30[4:2];
  wire [2:0]  indexVec_checkResult_0_3_10 = indexVec_checkResult_accessRegGrowth_30;
  wire [5:0]  indexVec_checkResult_decimalProportion_30 = {indexVec_checkResult_0_2_10, indexVec_checkResult_0_1_10};
  wire [2:0]  indexVec_checkResult_decimal_30 = indexVec_checkResult_decimalProportion_30[5:3];
  wire        indexVec_checkResult_overlap_30 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_30 >= indexVec_checkResult_intLMULInput_30[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_30} >= indexVec_checkResult_intLMULInput_30,
      indexVec_checkResult_allDataPosition_30[31:11]};
  wire        indexVec_checkResult_unChange_30 = slideUp & (indexVec_readIndex_10[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_10 = validVec[10] & ~indexVec_checkResult_unChange_30;
  wire        indexVec_checkResult_0_4_10 = indexVec_checkResult_overlap_30 | ~indexVec_checkResult_0_5_10 | lagerThanVL | indexVec_checkResult_unChange_30;
  wire [32:0] indexVec_checkResult_allDataPosition_31 = {indexVec_readIndex_10, 1'h0};
  wire [10:0] indexVec_checkResult_dataPosition_31 = indexVec_checkResult_allDataPosition_31[10:0];
  wire [1:0]  indexVec_checkResult_1_0_10 = {indexVec_checkResult_dataPosition_31[1], 1'h0};
  wire [3:0]  indexVec_checkResult_1_1_10 = indexVec_checkResult_dataPosition_31[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_31 = indexVec_checkResult_dataPosition_31[10:6];
  wire [1:0]  indexVec_checkResult_1_2_10 = indexVec_checkResult_dataGroup_31[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_31 = indexVec_checkResult_dataGroup_31[4:2];
  wire [2:0]  indexVec_checkResult_1_3_10 = indexVec_checkResult_accessRegGrowth_31;
  wire [5:0]  indexVec_checkResult_decimalProportion_31 = {indexVec_checkResult_1_2_10, indexVec_checkResult_1_1_10};
  wire [2:0]  indexVec_checkResult_decimal_31 = indexVec_checkResult_decimalProportion_31[5:3];
  wire        indexVec_checkResult_overlap_31 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_31 >= indexVec_checkResult_intLMULInput_31[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_31} >= indexVec_checkResult_intLMULInput_31,
      indexVec_checkResult_allDataPosition_31[32:11]};
  wire        indexVec_checkResult_unChange_31 = slideUp & (indexVec_readIndex_10[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_10 = validVec[10] & ~indexVec_checkResult_unChange_31;
  wire        indexVec_checkResult_1_4_10 = indexVec_checkResult_overlap_31 | ~indexVec_checkResult_1_5_10 | lagerThanVL | indexVec_checkResult_unChange_31;
  wire [33:0] indexVec_checkResult_allDataPosition_32 = {indexVec_readIndex_10, 2'h0};
  wire [10:0] indexVec_checkResult_dataPosition_32 = indexVec_checkResult_allDataPosition_32[10:0];
  wire [3:0]  indexVec_checkResult_2_1_10 = indexVec_checkResult_dataPosition_32[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_32 = indexVec_checkResult_dataPosition_32[10:6];
  wire [1:0]  indexVec_checkResult_2_2_10 = indexVec_checkResult_dataGroup_32[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_32 = indexVec_checkResult_dataGroup_32[4:2];
  wire [2:0]  indexVec_checkResult_2_3_10 = indexVec_checkResult_accessRegGrowth_32;
  wire [5:0]  indexVec_checkResult_decimalProportion_32 = {indexVec_checkResult_2_2_10, indexVec_checkResult_2_1_10};
  wire [2:0]  indexVec_checkResult_decimal_32 = indexVec_checkResult_decimalProportion_32[5:3];
  wire        indexVec_checkResult_overlap_32 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_32 >= indexVec_checkResult_intLMULInput_32[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_32} >= indexVec_checkResult_intLMULInput_32,
      indexVec_checkResult_allDataPosition_32[33:11]};
  wire        indexVec_checkResult_unChange_32 = slideUp & (indexVec_readIndex_10[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_10 = validVec[10] & ~indexVec_checkResult_unChange_32;
  wire        indexVec_checkResult_2_4_10 = indexVec_checkResult_overlap_32 | ~indexVec_checkResult_2_5_10 | lagerThanVL | indexVec_checkResult_unChange_32;
  wire [1:0]  indexVec_10_0 = (sew1H[0] ? indexVec_checkResult_0_0_10 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_10 : 2'h0);
  assign indexVec_10_1 = (sew1H[0] ? indexVec_checkResult_0_1_10 : 4'h0) | (sew1H[1] ? indexVec_checkResult_1_1_10 : 4'h0) | (sew1H[2] ? indexVec_checkResult_2_1_10 : 4'h0);
  wire [3:0]  indexDeq_bits_accessLane_10_0 = indexVec_10_1;
  wire [1:0]  indexVec_10_2 = (sew1H[0] ? indexVec_checkResult_0_2_10 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_10 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_10 : 2'h0);
  assign indexVec_10_3 = (sew1H[0] ? indexVec_checkResult_0_3_10 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_10 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_10 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_10_0 = indexVec_10_3;
  wire        indexVec_10_4 = sew1H[0] & indexVec_checkResult_0_4_10 | sew1H[1] & indexVec_checkResult_1_4_10 | sew1H[2] & indexVec_checkResult_2_4_10;
  wire        indexVec_10_5 = sew1H[0] & indexVec_checkResult_0_5_10 | sew1H[1] & indexVec_checkResult_1_5_10 | sew1H[2] & indexVec_checkResult_2_5_10;
  wire [31:0] indexVec_readIndex_11 = baseIndex + 32'hB;
  wire [31:0] indexVec_checkResult_allDataPosition_33 = indexVec_readIndex_11;
  wire [10:0] indexVec_checkResult_dataPosition_33 = indexVec_checkResult_allDataPosition_33[10:0];
  wire [1:0]  indexVec_checkResult_0_0_11 = indexVec_checkResult_dataPosition_33[1:0];
  wire [3:0]  indexVec_checkResult_0_1_11 = indexVec_checkResult_dataPosition_33[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_33 = indexVec_checkResult_dataPosition_33[10:6];
  wire [1:0]  indexVec_checkResult_0_2_11 = indexVec_checkResult_dataGroup_33[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_33 = indexVec_checkResult_dataGroup_33[4:2];
  wire [2:0]  indexVec_checkResult_0_3_11 = indexVec_checkResult_accessRegGrowth_33;
  wire [5:0]  indexVec_checkResult_decimalProportion_33 = {indexVec_checkResult_0_2_11, indexVec_checkResult_0_1_11};
  wire [2:0]  indexVec_checkResult_decimal_33 = indexVec_checkResult_decimalProportion_33[5:3];
  wire        indexVec_checkResult_overlap_33 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_33 >= indexVec_checkResult_intLMULInput_33[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_33} >= indexVec_checkResult_intLMULInput_33,
      indexVec_checkResult_allDataPosition_33[31:11]};
  wire        indexVec_checkResult_unChange_33 = slideUp & (indexVec_readIndex_11[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_11 = validVec[11] & ~indexVec_checkResult_unChange_33;
  wire        indexVec_checkResult_0_4_11 = indexVec_checkResult_overlap_33 | ~indexVec_checkResult_0_5_11 | lagerThanVL | indexVec_checkResult_unChange_33;
  wire [32:0] indexVec_checkResult_allDataPosition_34 = {indexVec_readIndex_11, 1'h0};
  wire [10:0] indexVec_checkResult_dataPosition_34 = indexVec_checkResult_allDataPosition_34[10:0];
  wire [1:0]  indexVec_checkResult_1_0_11 = {indexVec_checkResult_dataPosition_34[1], 1'h0};
  wire [3:0]  indexVec_checkResult_1_1_11 = indexVec_checkResult_dataPosition_34[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_34 = indexVec_checkResult_dataPosition_34[10:6];
  wire [1:0]  indexVec_checkResult_1_2_11 = indexVec_checkResult_dataGroup_34[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_34 = indexVec_checkResult_dataGroup_34[4:2];
  wire [2:0]  indexVec_checkResult_1_3_11 = indexVec_checkResult_accessRegGrowth_34;
  wire [5:0]  indexVec_checkResult_decimalProportion_34 = {indexVec_checkResult_1_2_11, indexVec_checkResult_1_1_11};
  wire [2:0]  indexVec_checkResult_decimal_34 = indexVec_checkResult_decimalProportion_34[5:3];
  wire        indexVec_checkResult_overlap_34 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_34 >= indexVec_checkResult_intLMULInput_34[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_34} >= indexVec_checkResult_intLMULInput_34,
      indexVec_checkResult_allDataPosition_34[32:11]};
  wire        indexVec_checkResult_unChange_34 = slideUp & (indexVec_readIndex_11[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_11 = validVec[11] & ~indexVec_checkResult_unChange_34;
  wire        indexVec_checkResult_1_4_11 = indexVec_checkResult_overlap_34 | ~indexVec_checkResult_1_5_11 | lagerThanVL | indexVec_checkResult_unChange_34;
  wire [33:0] indexVec_checkResult_allDataPosition_35 = {indexVec_readIndex_11, 2'h0};
  wire [10:0] indexVec_checkResult_dataPosition_35 = indexVec_checkResult_allDataPosition_35[10:0];
  wire [3:0]  indexVec_checkResult_2_1_11 = indexVec_checkResult_dataPosition_35[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_35 = indexVec_checkResult_dataPosition_35[10:6];
  wire [1:0]  indexVec_checkResult_2_2_11 = indexVec_checkResult_dataGroup_35[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_35 = indexVec_checkResult_dataGroup_35[4:2];
  wire [2:0]  indexVec_checkResult_2_3_11 = indexVec_checkResult_accessRegGrowth_35;
  wire [5:0]  indexVec_checkResult_decimalProportion_35 = {indexVec_checkResult_2_2_11, indexVec_checkResult_2_1_11};
  wire [2:0]  indexVec_checkResult_decimal_35 = indexVec_checkResult_decimalProportion_35[5:3];
  wire        indexVec_checkResult_overlap_35 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_35 >= indexVec_checkResult_intLMULInput_35[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_35} >= indexVec_checkResult_intLMULInput_35,
      indexVec_checkResult_allDataPosition_35[33:11]};
  wire        indexVec_checkResult_unChange_35 = slideUp & (indexVec_readIndex_11[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_11 = validVec[11] & ~indexVec_checkResult_unChange_35;
  wire        indexVec_checkResult_2_4_11 = indexVec_checkResult_overlap_35 | ~indexVec_checkResult_2_5_11 | lagerThanVL | indexVec_checkResult_unChange_35;
  wire [1:0]  indexVec_11_0 = (sew1H[0] ? indexVec_checkResult_0_0_11 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_11 : 2'h0);
  assign indexVec_11_1 = (sew1H[0] ? indexVec_checkResult_0_1_11 : 4'h0) | (sew1H[1] ? indexVec_checkResult_1_1_11 : 4'h0) | (sew1H[2] ? indexVec_checkResult_2_1_11 : 4'h0);
  wire [3:0]  indexDeq_bits_accessLane_11_0 = indexVec_11_1;
  wire [1:0]  indexVec_11_2 = (sew1H[0] ? indexVec_checkResult_0_2_11 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_11 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_11 : 2'h0);
  assign indexVec_11_3 = (sew1H[0] ? indexVec_checkResult_0_3_11 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_11 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_11 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_11_0 = indexVec_11_3;
  wire        indexVec_11_4 = sew1H[0] & indexVec_checkResult_0_4_11 | sew1H[1] & indexVec_checkResult_1_4_11 | sew1H[2] & indexVec_checkResult_2_4_11;
  wire        indexVec_11_5 = sew1H[0] & indexVec_checkResult_0_5_11 | sew1H[1] & indexVec_checkResult_1_5_11 | sew1H[2] & indexVec_checkResult_2_5_11;
  wire [31:0] indexVec_readIndex_12 = baseIndex + 32'hC;
  wire [31:0] indexVec_checkResult_allDataPosition_36 = indexVec_readIndex_12;
  wire [10:0] indexVec_checkResult_dataPosition_36 = indexVec_checkResult_allDataPosition_36[10:0];
  wire [1:0]  indexVec_checkResult_0_0_12 = indexVec_checkResult_dataPosition_36[1:0];
  wire [3:0]  indexVec_checkResult_0_1_12 = indexVec_checkResult_dataPosition_36[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_36 = indexVec_checkResult_dataPosition_36[10:6];
  wire [1:0]  indexVec_checkResult_0_2_12 = indexVec_checkResult_dataGroup_36[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_36 = indexVec_checkResult_dataGroup_36[4:2];
  wire [2:0]  indexVec_checkResult_0_3_12 = indexVec_checkResult_accessRegGrowth_36;
  wire [5:0]  indexVec_checkResult_decimalProportion_36 = {indexVec_checkResult_0_2_12, indexVec_checkResult_0_1_12};
  wire [2:0]  indexVec_checkResult_decimal_36 = indexVec_checkResult_decimalProportion_36[5:3];
  wire        indexVec_checkResult_overlap_36 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_36 >= indexVec_checkResult_intLMULInput_36[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_36} >= indexVec_checkResult_intLMULInput_36,
      indexVec_checkResult_allDataPosition_36[31:11]};
  wire        indexVec_checkResult_unChange_36 = slideUp & (indexVec_readIndex_12[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_12 = validVec[12] & ~indexVec_checkResult_unChange_36;
  wire        indexVec_checkResult_0_4_12 = indexVec_checkResult_overlap_36 | ~indexVec_checkResult_0_5_12 | lagerThanVL | indexVec_checkResult_unChange_36;
  wire [32:0] indexVec_checkResult_allDataPosition_37 = {indexVec_readIndex_12, 1'h0};
  wire [10:0] indexVec_checkResult_dataPosition_37 = indexVec_checkResult_allDataPosition_37[10:0];
  wire [1:0]  indexVec_checkResult_1_0_12 = {indexVec_checkResult_dataPosition_37[1], 1'h0};
  wire [3:0]  indexVec_checkResult_1_1_12 = indexVec_checkResult_dataPosition_37[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_37 = indexVec_checkResult_dataPosition_37[10:6];
  wire [1:0]  indexVec_checkResult_1_2_12 = indexVec_checkResult_dataGroup_37[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_37 = indexVec_checkResult_dataGroup_37[4:2];
  wire [2:0]  indexVec_checkResult_1_3_12 = indexVec_checkResult_accessRegGrowth_37;
  wire [5:0]  indexVec_checkResult_decimalProportion_37 = {indexVec_checkResult_1_2_12, indexVec_checkResult_1_1_12};
  wire [2:0]  indexVec_checkResult_decimal_37 = indexVec_checkResult_decimalProportion_37[5:3];
  wire        indexVec_checkResult_overlap_37 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_37 >= indexVec_checkResult_intLMULInput_37[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_37} >= indexVec_checkResult_intLMULInput_37,
      indexVec_checkResult_allDataPosition_37[32:11]};
  wire        indexVec_checkResult_unChange_37 = slideUp & (indexVec_readIndex_12[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_12 = validVec[12] & ~indexVec_checkResult_unChange_37;
  wire        indexVec_checkResult_1_4_12 = indexVec_checkResult_overlap_37 | ~indexVec_checkResult_1_5_12 | lagerThanVL | indexVec_checkResult_unChange_37;
  wire [33:0] indexVec_checkResult_allDataPosition_38 = {indexVec_readIndex_12, 2'h0};
  wire [10:0] indexVec_checkResult_dataPosition_38 = indexVec_checkResult_allDataPosition_38[10:0];
  wire [3:0]  indexVec_checkResult_2_1_12 = indexVec_checkResult_dataPosition_38[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_38 = indexVec_checkResult_dataPosition_38[10:6];
  wire [1:0]  indexVec_checkResult_2_2_12 = indexVec_checkResult_dataGroup_38[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_38 = indexVec_checkResult_dataGroup_38[4:2];
  wire [2:0]  indexVec_checkResult_2_3_12 = indexVec_checkResult_accessRegGrowth_38;
  wire [5:0]  indexVec_checkResult_decimalProportion_38 = {indexVec_checkResult_2_2_12, indexVec_checkResult_2_1_12};
  wire [2:0]  indexVec_checkResult_decimal_38 = indexVec_checkResult_decimalProportion_38[5:3];
  wire        indexVec_checkResult_overlap_38 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_38 >= indexVec_checkResult_intLMULInput_38[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_38} >= indexVec_checkResult_intLMULInput_38,
      indexVec_checkResult_allDataPosition_38[33:11]};
  wire        indexVec_checkResult_unChange_38 = slideUp & (indexVec_readIndex_12[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_12 = validVec[12] & ~indexVec_checkResult_unChange_38;
  wire        indexVec_checkResult_2_4_12 = indexVec_checkResult_overlap_38 | ~indexVec_checkResult_2_5_12 | lagerThanVL | indexVec_checkResult_unChange_38;
  wire [1:0]  indexVec_12_0 = (sew1H[0] ? indexVec_checkResult_0_0_12 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_12 : 2'h0);
  assign indexVec_12_1 = (sew1H[0] ? indexVec_checkResult_0_1_12 : 4'h0) | (sew1H[1] ? indexVec_checkResult_1_1_12 : 4'h0) | (sew1H[2] ? indexVec_checkResult_2_1_12 : 4'h0);
  wire [3:0]  indexDeq_bits_accessLane_12_0 = indexVec_12_1;
  wire [1:0]  indexVec_12_2 = (sew1H[0] ? indexVec_checkResult_0_2_12 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_12 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_12 : 2'h0);
  assign indexVec_12_3 = (sew1H[0] ? indexVec_checkResult_0_3_12 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_12 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_12 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_12_0 = indexVec_12_3;
  wire        indexVec_12_4 = sew1H[0] & indexVec_checkResult_0_4_12 | sew1H[1] & indexVec_checkResult_1_4_12 | sew1H[2] & indexVec_checkResult_2_4_12;
  wire        indexVec_12_5 = sew1H[0] & indexVec_checkResult_0_5_12 | sew1H[1] & indexVec_checkResult_1_5_12 | sew1H[2] & indexVec_checkResult_2_5_12;
  wire [31:0] indexVec_readIndex_13 = baseIndex + 32'hD;
  wire [31:0] indexVec_checkResult_allDataPosition_39 = indexVec_readIndex_13;
  wire [10:0] indexVec_checkResult_dataPosition_39 = indexVec_checkResult_allDataPosition_39[10:0];
  wire [1:0]  indexVec_checkResult_0_0_13 = indexVec_checkResult_dataPosition_39[1:0];
  wire [3:0]  indexVec_checkResult_0_1_13 = indexVec_checkResult_dataPosition_39[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_39 = indexVec_checkResult_dataPosition_39[10:6];
  wire [1:0]  indexVec_checkResult_0_2_13 = indexVec_checkResult_dataGroup_39[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_39 = indexVec_checkResult_dataGroup_39[4:2];
  wire [2:0]  indexVec_checkResult_0_3_13 = indexVec_checkResult_accessRegGrowth_39;
  wire [5:0]  indexVec_checkResult_decimalProportion_39 = {indexVec_checkResult_0_2_13, indexVec_checkResult_0_1_13};
  wire [2:0]  indexVec_checkResult_decimal_39 = indexVec_checkResult_decimalProportion_39[5:3];
  wire        indexVec_checkResult_overlap_39 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_39 >= indexVec_checkResult_intLMULInput_39[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_39} >= indexVec_checkResult_intLMULInput_39,
      indexVec_checkResult_allDataPosition_39[31:11]};
  wire        indexVec_checkResult_unChange_39 = slideUp & (indexVec_readIndex_13[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_13 = validVec[13] & ~indexVec_checkResult_unChange_39;
  wire        indexVec_checkResult_0_4_13 = indexVec_checkResult_overlap_39 | ~indexVec_checkResult_0_5_13 | lagerThanVL | indexVec_checkResult_unChange_39;
  wire [32:0] indexVec_checkResult_allDataPosition_40 = {indexVec_readIndex_13, 1'h0};
  wire [10:0] indexVec_checkResult_dataPosition_40 = indexVec_checkResult_allDataPosition_40[10:0];
  wire [1:0]  indexVec_checkResult_1_0_13 = {indexVec_checkResult_dataPosition_40[1], 1'h0};
  wire [3:0]  indexVec_checkResult_1_1_13 = indexVec_checkResult_dataPosition_40[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_40 = indexVec_checkResult_dataPosition_40[10:6];
  wire [1:0]  indexVec_checkResult_1_2_13 = indexVec_checkResult_dataGroup_40[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_40 = indexVec_checkResult_dataGroup_40[4:2];
  wire [2:0]  indexVec_checkResult_1_3_13 = indexVec_checkResult_accessRegGrowth_40;
  wire [5:0]  indexVec_checkResult_decimalProportion_40 = {indexVec_checkResult_1_2_13, indexVec_checkResult_1_1_13};
  wire [2:0]  indexVec_checkResult_decimal_40 = indexVec_checkResult_decimalProportion_40[5:3];
  wire        indexVec_checkResult_overlap_40 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_40 >= indexVec_checkResult_intLMULInput_40[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_40} >= indexVec_checkResult_intLMULInput_40,
      indexVec_checkResult_allDataPosition_40[32:11]};
  wire        indexVec_checkResult_unChange_40 = slideUp & (indexVec_readIndex_13[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_13 = validVec[13] & ~indexVec_checkResult_unChange_40;
  wire        indexVec_checkResult_1_4_13 = indexVec_checkResult_overlap_40 | ~indexVec_checkResult_1_5_13 | lagerThanVL | indexVec_checkResult_unChange_40;
  wire [33:0] indexVec_checkResult_allDataPosition_41 = {indexVec_readIndex_13, 2'h0};
  wire [10:0] indexVec_checkResult_dataPosition_41 = indexVec_checkResult_allDataPosition_41[10:0];
  wire [3:0]  indexVec_checkResult_2_1_13 = indexVec_checkResult_dataPosition_41[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_41 = indexVec_checkResult_dataPosition_41[10:6];
  wire [1:0]  indexVec_checkResult_2_2_13 = indexVec_checkResult_dataGroup_41[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_41 = indexVec_checkResult_dataGroup_41[4:2];
  wire [2:0]  indexVec_checkResult_2_3_13 = indexVec_checkResult_accessRegGrowth_41;
  wire [5:0]  indexVec_checkResult_decimalProportion_41 = {indexVec_checkResult_2_2_13, indexVec_checkResult_2_1_13};
  wire [2:0]  indexVec_checkResult_decimal_41 = indexVec_checkResult_decimalProportion_41[5:3];
  wire        indexVec_checkResult_overlap_41 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_41 >= indexVec_checkResult_intLMULInput_41[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_41} >= indexVec_checkResult_intLMULInput_41,
      indexVec_checkResult_allDataPosition_41[33:11]};
  wire        indexVec_checkResult_unChange_41 = slideUp & (indexVec_readIndex_13[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_13 = validVec[13] & ~indexVec_checkResult_unChange_41;
  wire        indexVec_checkResult_2_4_13 = indexVec_checkResult_overlap_41 | ~indexVec_checkResult_2_5_13 | lagerThanVL | indexVec_checkResult_unChange_41;
  wire [1:0]  indexVec_13_0 = (sew1H[0] ? indexVec_checkResult_0_0_13 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_13 : 2'h0);
  assign indexVec_13_1 = (sew1H[0] ? indexVec_checkResult_0_1_13 : 4'h0) | (sew1H[1] ? indexVec_checkResult_1_1_13 : 4'h0) | (sew1H[2] ? indexVec_checkResult_2_1_13 : 4'h0);
  wire [3:0]  indexDeq_bits_accessLane_13_0 = indexVec_13_1;
  wire [1:0]  indexVec_13_2 = (sew1H[0] ? indexVec_checkResult_0_2_13 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_13 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_13 : 2'h0);
  assign indexVec_13_3 = (sew1H[0] ? indexVec_checkResult_0_3_13 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_13 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_13 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_13_0 = indexVec_13_3;
  wire        indexVec_13_4 = sew1H[0] & indexVec_checkResult_0_4_13 | sew1H[1] & indexVec_checkResult_1_4_13 | sew1H[2] & indexVec_checkResult_2_4_13;
  wire        indexVec_13_5 = sew1H[0] & indexVec_checkResult_0_5_13 | sew1H[1] & indexVec_checkResult_1_5_13 | sew1H[2] & indexVec_checkResult_2_5_13;
  wire [31:0] indexVec_readIndex_14 = baseIndex + 32'hE;
  wire [31:0] indexVec_checkResult_allDataPosition_42 = indexVec_readIndex_14;
  wire [10:0] indexVec_checkResult_dataPosition_42 = indexVec_checkResult_allDataPosition_42[10:0];
  wire [1:0]  indexVec_checkResult_0_0_14 = indexVec_checkResult_dataPosition_42[1:0];
  wire [3:0]  indexVec_checkResult_0_1_14 = indexVec_checkResult_dataPosition_42[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_42 = indexVec_checkResult_dataPosition_42[10:6];
  wire [1:0]  indexVec_checkResult_0_2_14 = indexVec_checkResult_dataGroup_42[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_42 = indexVec_checkResult_dataGroup_42[4:2];
  wire [2:0]  indexVec_checkResult_0_3_14 = indexVec_checkResult_accessRegGrowth_42;
  wire [5:0]  indexVec_checkResult_decimalProportion_42 = {indexVec_checkResult_0_2_14, indexVec_checkResult_0_1_14};
  wire [2:0]  indexVec_checkResult_decimal_42 = indexVec_checkResult_decimalProportion_42[5:3];
  wire        indexVec_checkResult_overlap_42 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_42 >= indexVec_checkResult_intLMULInput_42[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_42} >= indexVec_checkResult_intLMULInput_42,
      indexVec_checkResult_allDataPosition_42[31:11]};
  wire        indexVec_checkResult_unChange_42 = slideUp & (indexVec_readIndex_14[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_14 = validVec[14] & ~indexVec_checkResult_unChange_42;
  wire        indexVec_checkResult_0_4_14 = indexVec_checkResult_overlap_42 | ~indexVec_checkResult_0_5_14 | lagerThanVL | indexVec_checkResult_unChange_42;
  wire [32:0] indexVec_checkResult_allDataPosition_43 = {indexVec_readIndex_14, 1'h0};
  wire [10:0] indexVec_checkResult_dataPosition_43 = indexVec_checkResult_allDataPosition_43[10:0];
  wire [1:0]  indexVec_checkResult_1_0_14 = {indexVec_checkResult_dataPosition_43[1], 1'h0};
  wire [3:0]  indexVec_checkResult_1_1_14 = indexVec_checkResult_dataPosition_43[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_43 = indexVec_checkResult_dataPosition_43[10:6];
  wire [1:0]  indexVec_checkResult_1_2_14 = indexVec_checkResult_dataGroup_43[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_43 = indexVec_checkResult_dataGroup_43[4:2];
  wire [2:0]  indexVec_checkResult_1_3_14 = indexVec_checkResult_accessRegGrowth_43;
  wire [5:0]  indexVec_checkResult_decimalProportion_43 = {indexVec_checkResult_1_2_14, indexVec_checkResult_1_1_14};
  wire [2:0]  indexVec_checkResult_decimal_43 = indexVec_checkResult_decimalProportion_43[5:3];
  wire        indexVec_checkResult_overlap_43 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_43 >= indexVec_checkResult_intLMULInput_43[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_43} >= indexVec_checkResult_intLMULInput_43,
      indexVec_checkResult_allDataPosition_43[32:11]};
  wire        indexVec_checkResult_unChange_43 = slideUp & (indexVec_readIndex_14[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_14 = validVec[14] & ~indexVec_checkResult_unChange_43;
  wire        indexVec_checkResult_1_4_14 = indexVec_checkResult_overlap_43 | ~indexVec_checkResult_1_5_14 | lagerThanVL | indexVec_checkResult_unChange_43;
  wire [33:0] indexVec_checkResult_allDataPosition_44 = {indexVec_readIndex_14, 2'h0};
  wire [10:0] indexVec_checkResult_dataPosition_44 = indexVec_checkResult_allDataPosition_44[10:0];
  wire [3:0]  indexVec_checkResult_2_1_14 = indexVec_checkResult_dataPosition_44[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_44 = indexVec_checkResult_dataPosition_44[10:6];
  wire [1:0]  indexVec_checkResult_2_2_14 = indexVec_checkResult_dataGroup_44[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_44 = indexVec_checkResult_dataGroup_44[4:2];
  wire [2:0]  indexVec_checkResult_2_3_14 = indexVec_checkResult_accessRegGrowth_44;
  wire [5:0]  indexVec_checkResult_decimalProportion_44 = {indexVec_checkResult_2_2_14, indexVec_checkResult_2_1_14};
  wire [2:0]  indexVec_checkResult_decimal_44 = indexVec_checkResult_decimalProportion_44[5:3];
  wire        indexVec_checkResult_overlap_44 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_44 >= indexVec_checkResult_intLMULInput_44[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_44} >= indexVec_checkResult_intLMULInput_44,
      indexVec_checkResult_allDataPosition_44[33:11]};
  wire        indexVec_checkResult_unChange_44 = slideUp & (indexVec_readIndex_14[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_14 = validVec[14] & ~indexVec_checkResult_unChange_44;
  wire        indexVec_checkResult_2_4_14 = indexVec_checkResult_overlap_44 | ~indexVec_checkResult_2_5_14 | lagerThanVL | indexVec_checkResult_unChange_44;
  wire [1:0]  indexVec_14_0 = (sew1H[0] ? indexVec_checkResult_0_0_14 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_14 : 2'h0);
  assign indexVec_14_1 = (sew1H[0] ? indexVec_checkResult_0_1_14 : 4'h0) | (sew1H[1] ? indexVec_checkResult_1_1_14 : 4'h0) | (sew1H[2] ? indexVec_checkResult_2_1_14 : 4'h0);
  wire [3:0]  indexDeq_bits_accessLane_14_0 = indexVec_14_1;
  wire [1:0]  indexVec_14_2 = (sew1H[0] ? indexVec_checkResult_0_2_14 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_14 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_14 : 2'h0);
  assign indexVec_14_3 = (sew1H[0] ? indexVec_checkResult_0_3_14 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_14 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_14 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_14_0 = indexVec_14_3;
  wire        indexVec_14_4 = sew1H[0] & indexVec_checkResult_0_4_14 | sew1H[1] & indexVec_checkResult_1_4_14 | sew1H[2] & indexVec_checkResult_2_4_14;
  wire        indexVec_14_5 = sew1H[0] & indexVec_checkResult_0_5_14 | sew1H[1] & indexVec_checkResult_1_5_14 | sew1H[2] & indexVec_checkResult_2_5_14;
  wire [31:0] indexVec_readIndex_15 = baseIndex + 32'hF;
  wire [31:0] indexVec_checkResult_allDataPosition_45 = indexVec_readIndex_15;
  wire [10:0] indexVec_checkResult_dataPosition_45 = indexVec_checkResult_allDataPosition_45[10:0];
  wire [1:0]  indexVec_checkResult_0_0_15 = indexVec_checkResult_dataPosition_45[1:0];
  wire [3:0]  indexVec_checkResult_0_1_15 = indexVec_checkResult_dataPosition_45[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_45 = indexVec_checkResult_dataPosition_45[10:6];
  wire [1:0]  indexVec_checkResult_0_2_15 = indexVec_checkResult_dataGroup_45[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_45 = indexVec_checkResult_dataGroup_45[4:2];
  wire [2:0]  indexVec_checkResult_0_3_15 = indexVec_checkResult_accessRegGrowth_45;
  wire [5:0]  indexVec_checkResult_decimalProportion_45 = {indexVec_checkResult_0_2_15, indexVec_checkResult_0_1_15};
  wire [2:0]  indexVec_checkResult_decimal_45 = indexVec_checkResult_decimalProportion_45[5:3];
  wire        indexVec_checkResult_overlap_45 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_45 >= indexVec_checkResult_intLMULInput_45[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_45} >= indexVec_checkResult_intLMULInput_45,
      indexVec_checkResult_allDataPosition_45[31:11]};
  wire        indexVec_checkResult_unChange_45 = slideUp & (indexVec_readIndex_15[31] | lagerThanVL);
  wire        indexVec_checkResult_0_5_15 = validVec[15] & ~indexVec_checkResult_unChange_45;
  wire        indexVec_checkResult_0_4_15 = indexVec_checkResult_overlap_45 | ~indexVec_checkResult_0_5_15 | lagerThanVL | indexVec_checkResult_unChange_45;
  wire [32:0] indexVec_checkResult_allDataPosition_46 = {indexVec_readIndex_15, 1'h0};
  wire [10:0] indexVec_checkResult_dataPosition_46 = indexVec_checkResult_allDataPosition_46[10:0];
  wire [1:0]  indexVec_checkResult_1_0_15 = {indexVec_checkResult_dataPosition_46[1], 1'h0};
  wire [3:0]  indexVec_checkResult_1_1_15 = indexVec_checkResult_dataPosition_46[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_46 = indexVec_checkResult_dataPosition_46[10:6];
  wire [1:0]  indexVec_checkResult_1_2_15 = indexVec_checkResult_dataGroup_46[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_46 = indexVec_checkResult_dataGroup_46[4:2];
  wire [2:0]  indexVec_checkResult_1_3_15 = indexVec_checkResult_accessRegGrowth_46;
  wire [5:0]  indexVec_checkResult_decimalProportion_46 = {indexVec_checkResult_1_2_15, indexVec_checkResult_1_1_15};
  wire [2:0]  indexVec_checkResult_decimal_46 = indexVec_checkResult_decimalProportion_46[5:3];
  wire        indexVec_checkResult_overlap_46 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_46 >= indexVec_checkResult_intLMULInput_46[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_46} >= indexVec_checkResult_intLMULInput_46,
      indexVec_checkResult_allDataPosition_46[32:11]};
  wire        indexVec_checkResult_unChange_46 = slideUp & (indexVec_readIndex_15[31] | lagerThanVL);
  wire        indexVec_checkResult_1_5_15 = validVec[15] & ~indexVec_checkResult_unChange_46;
  wire        indexVec_checkResult_1_4_15 = indexVec_checkResult_overlap_46 | ~indexVec_checkResult_1_5_15 | lagerThanVL | indexVec_checkResult_unChange_46;
  wire [33:0] indexVec_checkResult_allDataPosition_47 = {indexVec_readIndex_15, 2'h0};
  wire [10:0] indexVec_checkResult_dataPosition_47 = indexVec_checkResult_allDataPosition_47[10:0];
  wire [3:0]  indexVec_checkResult_2_1_15 = indexVec_checkResult_dataPosition_47[5:2];
  wire [4:0]  indexVec_checkResult_dataGroup_47 = indexVec_checkResult_dataPosition_47[10:6];
  wire [1:0]  indexVec_checkResult_2_2_15 = indexVec_checkResult_dataGroup_47[1:0];
  wire [2:0]  indexVec_checkResult_accessRegGrowth_47 = indexVec_checkResult_dataGroup_47[4:2];
  wire [2:0]  indexVec_checkResult_2_3_15 = indexVec_checkResult_accessRegGrowth_47;
  wire [5:0]  indexVec_checkResult_decimalProportion_47 = {indexVec_checkResult_2_2_15, indexVec_checkResult_2_1_15};
  wire [2:0]  indexVec_checkResult_decimal_47 = indexVec_checkResult_decimalProportion_47[5:3];
  wire        indexVec_checkResult_overlap_47 =
    |{instructionReq_vlmul[2] & indexVec_checkResult_decimal_47 >= indexVec_checkResult_intLMULInput_47[3:1] | ~(instructionReq_vlmul[2]) & {1'h0, indexVec_checkResult_accessRegGrowth_47} >= indexVec_checkResult_intLMULInput_47,
      indexVec_checkResult_allDataPosition_47[33:11]};
  wire        indexVec_checkResult_unChange_47 = slideUp & (indexVec_readIndex_15[31] | lagerThanVL);
  wire        indexVec_checkResult_2_5_15 = validVec[15] & ~indexVec_checkResult_unChange_47;
  wire        indexVec_checkResult_2_4_15 = indexVec_checkResult_overlap_47 | ~indexVec_checkResult_2_5_15 | lagerThanVL | indexVec_checkResult_unChange_47;
  wire [1:0]  indexVec_15_0 = (sew1H[0] ? indexVec_checkResult_0_0_15 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_0_15 : 2'h0);
  assign indexVec_15_1 = (sew1H[0] ? indexVec_checkResult_0_1_15 : 4'h0) | (sew1H[1] ? indexVec_checkResult_1_1_15 : 4'h0) | (sew1H[2] ? indexVec_checkResult_2_1_15 : 4'h0);
  wire [3:0]  indexDeq_bits_accessLane_15_0 = indexVec_15_1;
  wire [1:0]  indexVec_15_2 = (sew1H[0] ? indexVec_checkResult_0_2_15 : 2'h0) | (sew1H[1] ? indexVec_checkResult_1_2_15 : 2'h0) | (sew1H[2] ? indexVec_checkResult_2_2_15 : 2'h0);
  assign indexVec_15_3 = (sew1H[0] ? indexVec_checkResult_0_3_15 : 3'h0) | (sew1H[1] ? indexVec_checkResult_1_3_15 : 3'h0) | (sew1H[2] ? indexVec_checkResult_2_3_15 : 3'h0);
  wire [2:0]  indexDeq_bits_vsGrowth_15_0 = indexVec_15_3;
  wire        indexVec_15_4 = sew1H[0] & indexVec_checkResult_0_4_15 | sew1H[1] & indexVec_checkResult_1_4_15 | sew1H[2] & indexVec_checkResult_2_4_15;
  wire        indexVec_15_5 = sew1H[0] & indexVec_checkResult_0_5_15 | sew1H[1] & indexVec_checkResult_1_5_15 | sew1H[2] & indexVec_checkResult_2_5_15;
  assign indexDeq_valid_0 = InstructionValid & isSlide;
  wire [1:0]  indexDeq_bits_needRead_lo_lo_lo = {~indexVec_1_4, ~indexVec_0_4};
  wire [1:0]  indexDeq_bits_needRead_lo_lo_hi = {~indexVec_3_4, ~indexVec_2_4};
  wire [3:0]  indexDeq_bits_needRead_lo_lo = {indexDeq_bits_needRead_lo_lo_hi, indexDeq_bits_needRead_lo_lo_lo};
  wire [1:0]  indexDeq_bits_needRead_lo_hi_lo = {~indexVec_5_4, ~indexVec_4_4};
  wire [1:0]  indexDeq_bits_needRead_lo_hi_hi = {~indexVec_7_4, ~indexVec_6_4};
  wire [3:0]  indexDeq_bits_needRead_lo_hi = {indexDeq_bits_needRead_lo_hi_hi, indexDeq_bits_needRead_lo_hi_lo};
  wire [7:0]  indexDeq_bits_needRead_lo = {indexDeq_bits_needRead_lo_hi, indexDeq_bits_needRead_lo_lo};
  wire [1:0]  indexDeq_bits_needRead_hi_lo_lo = {~indexVec_9_4, ~indexVec_8_4};
  wire [1:0]  indexDeq_bits_needRead_hi_lo_hi = {~indexVec_11_4, ~indexVec_10_4};
  wire [3:0]  indexDeq_bits_needRead_hi_lo = {indexDeq_bits_needRead_hi_lo_hi, indexDeq_bits_needRead_hi_lo_lo};
  wire [1:0]  indexDeq_bits_needRead_hi_hi_lo = {~indexVec_13_4, ~indexVec_12_4};
  wire [1:0]  indexDeq_bits_needRead_hi_hi_hi = {~indexVec_15_4, ~indexVec_14_4};
  wire [3:0]  indexDeq_bits_needRead_hi_hi = {indexDeq_bits_needRead_hi_hi_hi, indexDeq_bits_needRead_hi_hi_lo};
  wire [7:0]  indexDeq_bits_needRead_hi = {indexDeq_bits_needRead_hi_hi, indexDeq_bits_needRead_hi_lo};
  wire [15:0] indexDeq_bits_needRead_0 = {indexDeq_bits_needRead_hi, indexDeq_bits_needRead_lo} & ~replaceWithVs1;
  wire [1:0]  indexDeq_bits_elementValid_lo_lo_lo = {indexVec_1_5, indexVec_0_5};
  wire [1:0]  indexDeq_bits_elementValid_lo_lo_hi = {indexVec_3_5, indexVec_2_5};
  wire [3:0]  indexDeq_bits_elementValid_lo_lo = {indexDeq_bits_elementValid_lo_lo_hi, indexDeq_bits_elementValid_lo_lo_lo};
  wire [1:0]  indexDeq_bits_elementValid_lo_hi_lo = {indexVec_5_5, indexVec_4_5};
  wire [1:0]  indexDeq_bits_elementValid_lo_hi_hi = {indexVec_7_5, indexVec_6_5};
  wire [3:0]  indexDeq_bits_elementValid_lo_hi = {indexDeq_bits_elementValid_lo_hi_hi, indexDeq_bits_elementValid_lo_hi_lo};
  wire [7:0]  indexDeq_bits_elementValid_lo = {indexDeq_bits_elementValid_lo_hi, indexDeq_bits_elementValid_lo_lo};
  wire [1:0]  indexDeq_bits_elementValid_hi_lo_lo = {indexVec_9_5, indexVec_8_5};
  wire [1:0]  indexDeq_bits_elementValid_hi_lo_hi = {indexVec_11_5, indexVec_10_5};
  wire [3:0]  indexDeq_bits_elementValid_hi_lo = {indexDeq_bits_elementValid_hi_lo_hi, indexDeq_bits_elementValid_hi_lo_lo};
  wire [1:0]  indexDeq_bits_elementValid_hi_hi_lo = {indexVec_13_5, indexVec_12_5};
  wire [1:0]  indexDeq_bits_elementValid_hi_hi_hi = {indexVec_15_5, indexVec_14_5};
  wire [3:0]  indexDeq_bits_elementValid_hi_hi = {indexDeq_bits_elementValid_hi_hi_hi, indexDeq_bits_elementValid_hi_hi_lo};
  wire [7:0]  indexDeq_bits_elementValid_hi = {indexDeq_bits_elementValid_hi_hi, indexDeq_bits_elementValid_hi_lo};
  wire [15:0] indexDeq_bits_elementValid_0 = {indexDeq_bits_elementValid_hi, indexDeq_bits_elementValid_lo} | replaceWithVs1;
  wire [3:0]  indexDeq_bits_readOffset_lo_lo_lo = {indexVec_1_2, indexVec_0_2};
  wire [3:0]  indexDeq_bits_readOffset_lo_lo_hi = {indexVec_3_2, indexVec_2_2};
  wire [7:0]  indexDeq_bits_readOffset_lo_lo = {indexDeq_bits_readOffset_lo_lo_hi, indexDeq_bits_readOffset_lo_lo_lo};
  wire [3:0]  indexDeq_bits_readOffset_lo_hi_lo = {indexVec_5_2, indexVec_4_2};
  wire [3:0]  indexDeq_bits_readOffset_lo_hi_hi = {indexVec_7_2, indexVec_6_2};
  wire [7:0]  indexDeq_bits_readOffset_lo_hi = {indexDeq_bits_readOffset_lo_hi_hi, indexDeq_bits_readOffset_lo_hi_lo};
  wire [15:0] indexDeq_bits_readOffset_lo = {indexDeq_bits_readOffset_lo_hi, indexDeq_bits_readOffset_lo_lo};
  wire [3:0]  indexDeq_bits_readOffset_hi_lo_lo = {indexVec_9_2, indexVec_8_2};
  wire [3:0]  indexDeq_bits_readOffset_hi_lo_hi = {indexVec_11_2, indexVec_10_2};
  wire [7:0]  indexDeq_bits_readOffset_hi_lo = {indexDeq_bits_readOffset_hi_lo_hi, indexDeq_bits_readOffset_hi_lo_lo};
  wire [3:0]  indexDeq_bits_readOffset_hi_hi_lo = {indexVec_13_2, indexVec_12_2};
  wire [3:0]  indexDeq_bits_readOffset_hi_hi_hi = {indexVec_15_2, indexVec_14_2};
  wire [7:0]  indexDeq_bits_readOffset_hi_hi = {indexDeq_bits_readOffset_hi_hi_hi, indexDeq_bits_readOffset_hi_hi_lo};
  wire [15:0] indexDeq_bits_readOffset_hi = {indexDeq_bits_readOffset_hi_hi, indexDeq_bits_readOffset_hi_lo};
  wire [31:0] indexDeq_bits_readOffset_0 = {indexDeq_bits_readOffset_hi, indexDeq_bits_readOffset_lo};
  wire [3:0]  indexDeq_bits_readDataOffset_lo_lo_lo = {indexVec_1_0, indexVec_0_0};
  wire [3:0]  indexDeq_bits_readDataOffset_lo_lo_hi = {indexVec_3_0, indexVec_2_0};
  wire [7:0]  indexDeq_bits_readDataOffset_lo_lo = {indexDeq_bits_readDataOffset_lo_lo_hi, indexDeq_bits_readDataOffset_lo_lo_lo};
  wire [3:0]  indexDeq_bits_readDataOffset_lo_hi_lo = {indexVec_5_0, indexVec_4_0};
  wire [3:0]  indexDeq_bits_readDataOffset_lo_hi_hi = {indexVec_7_0, indexVec_6_0};
  wire [7:0]  indexDeq_bits_readDataOffset_lo_hi = {indexDeq_bits_readDataOffset_lo_hi_hi, indexDeq_bits_readDataOffset_lo_hi_lo};
  wire [15:0] indexDeq_bits_readDataOffset_lo = {indexDeq_bits_readDataOffset_lo_hi, indexDeq_bits_readDataOffset_lo_lo};
  wire [3:0]  indexDeq_bits_readDataOffset_hi_lo_lo = {indexVec_9_0, indexVec_8_0};
  wire [3:0]  indexDeq_bits_readDataOffset_hi_lo_hi = {indexVec_11_0, indexVec_10_0};
  wire [7:0]  indexDeq_bits_readDataOffset_hi_lo = {indexDeq_bits_readDataOffset_hi_lo_hi, indexDeq_bits_readDataOffset_hi_lo_lo};
  wire [3:0]  indexDeq_bits_readDataOffset_hi_hi_lo = {indexVec_13_0, indexVec_12_0};
  wire [3:0]  indexDeq_bits_readDataOffset_hi_hi_hi = {indexVec_15_0, indexVec_14_0};
  wire [7:0]  indexDeq_bits_readDataOffset_hi_hi = {indexDeq_bits_readDataOffset_hi_hi_hi, indexDeq_bits_readDataOffset_hi_hi_lo};
  wire [15:0] indexDeq_bits_readDataOffset_hi = {indexDeq_bits_readDataOffset_hi_hi, indexDeq_bits_readDataOffset_hi_lo};
  wire [31:0] indexDeq_bits_readDataOffset_0 = {indexDeq_bits_readDataOffset_hi, indexDeq_bits_readDataOffset_lo};
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
  assign indexDeq_bits_accessLane_8 = indexDeq_bits_accessLane_8_0;
  assign indexDeq_bits_accessLane_9 = indexDeq_bits_accessLane_9_0;
  assign indexDeq_bits_accessLane_10 = indexDeq_bits_accessLane_10_0;
  assign indexDeq_bits_accessLane_11 = indexDeq_bits_accessLane_11_0;
  assign indexDeq_bits_accessLane_12 = indexDeq_bits_accessLane_12_0;
  assign indexDeq_bits_accessLane_13 = indexDeq_bits_accessLane_13_0;
  assign indexDeq_bits_accessLane_14 = indexDeq_bits_accessLane_14_0;
  assign indexDeq_bits_accessLane_15 = indexDeq_bits_accessLane_15_0;
  assign indexDeq_bits_vsGrowth_0 = indexDeq_bits_vsGrowth_0_0;
  assign indexDeq_bits_vsGrowth_1 = indexDeq_bits_vsGrowth_1_0;
  assign indexDeq_bits_vsGrowth_2 = indexDeq_bits_vsGrowth_2_0;
  assign indexDeq_bits_vsGrowth_3 = indexDeq_bits_vsGrowth_3_0;
  assign indexDeq_bits_vsGrowth_4 = indexDeq_bits_vsGrowth_4_0;
  assign indexDeq_bits_vsGrowth_5 = indexDeq_bits_vsGrowth_5_0;
  assign indexDeq_bits_vsGrowth_6 = indexDeq_bits_vsGrowth_6_0;
  assign indexDeq_bits_vsGrowth_7 = indexDeq_bits_vsGrowth_7_0;
  assign indexDeq_bits_vsGrowth_8 = indexDeq_bits_vsGrowth_8_0;
  assign indexDeq_bits_vsGrowth_9 = indexDeq_bits_vsGrowth_9_0;
  assign indexDeq_bits_vsGrowth_10 = indexDeq_bits_vsGrowth_10_0;
  assign indexDeq_bits_vsGrowth_11 = indexDeq_bits_vsGrowth_11_0;
  assign indexDeq_bits_vsGrowth_12 = indexDeq_bits_vsGrowth_12_0;
  assign indexDeq_bits_vsGrowth_13 = indexDeq_bits_vsGrowth_13_0;
  assign indexDeq_bits_vsGrowth_14 = indexDeq_bits_vsGrowth_14_0;
  assign indexDeq_bits_vsGrowth_15 = indexDeq_bits_vsGrowth_15_0;
  assign indexDeq_bits_executeGroup = indexDeq_bits_executeGroup_0;
  assign indexDeq_bits_readDataOffset = indexDeq_bits_readDataOffset_0;
  assign indexDeq_bits_last = indexDeq_bits_last_0;
  assign slideGroupOut = slideGroup;
endmodule

