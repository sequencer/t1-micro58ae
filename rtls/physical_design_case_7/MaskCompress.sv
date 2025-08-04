
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
module MaskCompress(
  input          clock,
                 reset,
                 in_valid,
                 in_bits_maskType,
  input  [1:0]   in_bits_eew,
  input  [2:0]   in_bits_uop,
  input  [31:0]  in_bits_readFromScalar,
                 in_bits_source1,
                 in_bits_mask,
  input  [127:0] in_bits_source2,
                 in_bits_pipeData,
  input  [10:0]  in_bits_groupCounter,
  input  [3:0]   in_bits_ffoInput,
                 in_bits_validInput,
  input          in_bits_lastCompress,
  output [127:0] out_data,
  output [15:0]  out_mask,
  output [10:0]  out_groupCounter,
  output [3:0]   out_ffoOutput,
  output         out_compressValid,
  input          newInstruction,
                 ffoInstruction,
  output [31:0]  writeData,
  output         stageValid
);

  wire         compressDataVec_useTail_15 = 1'h0;
  wire         compressTailMask_elementValid_15 = 1'h0;
  reg          in_1_valid;
  reg          in_1_bits_maskType;
  reg  [1:0]   in_1_bits_eew;
  reg  [2:0]   in_1_bits_uop;
  reg  [31:0]  in_1_bits_readFromScalar;
  reg  [31:0]  in_1_bits_source1;
  reg  [31:0]  in_1_bits_mask;
  reg  [127:0] in_1_bits_source2;
  reg  [127:0] in_1_bits_pipeData;
  reg  [10:0]  in_1_bits_groupCounter;
  reg  [3:0]   in_1_bits_ffoInput;
  reg  [3:0]   in_1_bits_validInput;
  reg          in_1_bits_lastCompress;
  wire         compress = in_1_bits_uop == 3'h1;
  wire         viota = in_1_bits_uop == 3'h0;
  wire         mv = in_1_bits_uop == 3'h2;
  wire         mvRd = in_1_bits_uop == 3'h3;
  wire         writeRD = &(in_1_bits_uop[1:0]);
  wire         ffoType = &(in_1_bits_uop[2:1]);
  wire [3:0]   _eew1H_T = 4'h1 << in_1_bits_eew;
  wire [2:0]   eew1H = _eew1H_T[2:0];
  reg  [13:0]  compressInit;
  wire [13:0]  compressVec_0 = compressInit;
  wire [15:0]  maskInput = in_1_bits_source1[15:0] & in_1_bits_mask[15:0];
  wire         compressMaskVec_0 = maskInput[0];
  wire         compressMaskVec_1 = maskInput[1];
  wire         compressMaskVec_2 = maskInput[2];
  wire         compressMaskVec_3 = maskInput[3];
  wire         compressMaskVec_4 = maskInput[4];
  wire         compressMaskVec_5 = maskInput[5];
  wire         compressMaskVec_6 = maskInput[6];
  wire         compressMaskVec_7 = maskInput[7];
  wire         compressMaskVec_8 = maskInput[8];
  wire         compressMaskVec_9 = maskInput[9];
  wire         compressMaskVec_10 = maskInput[10];
  wire         compressMaskVec_11 = maskInput[11];
  wire         compressMaskVec_12 = maskInput[12];
  wire         compressMaskVec_13 = maskInput[13];
  wire         compressMaskVec_14 = maskInput[14];
  wire         compressMaskVec_15 = maskInput[15];
  wire [13:0]  compressCount =
    compressInit
    + {9'h0,
       {1'h0,
        {1'h0, {1'h0, {1'h0, compressMaskVec_0} + {1'h0, compressMaskVec_1}} + {1'h0, {1'h0, compressMaskVec_2} + {1'h0, compressMaskVec_3}}}
          + {1'h0, {1'h0, {1'h0, compressMaskVec_4} + {1'h0, compressMaskVec_5}} + {1'h0, {1'h0, compressMaskVec_6} + {1'h0, compressMaskVec_7}}}}
         + {1'h0,
            {1'h0, {1'h0, {1'h0, compressMaskVec_8} + {1'h0, compressMaskVec_9}} + {1'h0, {1'h0, compressMaskVec_10} + {1'h0, compressMaskVec_11}}}
              + {1'h0, {1'h0, {1'h0, compressMaskVec_12} + {1'h0, compressMaskVec_13}} + {1'h0, {1'h0, compressMaskVec_14} + {1'h0, compressMaskVec_15}}}}};
  wire [13:0]  compressVec_1 = compressInit + {13'h0, compressMaskVec_0};
  wire [13:0]  compressVec_2 = compressVec_1 + {13'h0, compressMaskVec_1};
  wire [13:0]  compressVec_3 = compressVec_2 + {13'h0, compressMaskVec_2};
  wire [13:0]  compressVec_4 = compressVec_3 + {13'h0, compressMaskVec_3};
  wire [13:0]  compressVec_5 = compressVec_4 + {13'h0, compressMaskVec_4};
  wire [13:0]  compressVec_6 = compressVec_5 + {13'h0, compressMaskVec_5};
  wire [13:0]  compressVec_7 = compressVec_6 + {13'h0, compressMaskVec_6};
  wire [13:0]  compressVec_8 = compressVec_7 + {13'h0, compressMaskVec_7};
  wire [13:0]  compressVec_9 = compressVec_8 + {13'h0, compressMaskVec_8};
  wire [13:0]  compressVec_10 = compressVec_9 + {13'h0, compressMaskVec_9};
  wire [13:0]  compressVec_11 = compressVec_10 + {13'h0, compressMaskVec_10};
  wire [13:0]  compressVec_12 = compressVec_11 + {13'h0, compressMaskVec_11};
  wire [13:0]  compressVec_13 = compressVec_12 + {13'h0, compressMaskVec_12};
  wire [13:0]  compressVec_14 = compressVec_13 + {13'h0, compressMaskVec_13};
  wire [13:0]  compressVec_15 = compressVec_14 + {13'h0, compressMaskVec_14};
  reg  [31:0]  ffoIndex;
  reg          ffoValid;
  wire         countSplit_0_1 = compressCount[4];
  wire [3:0]   countSplit_0_2 = compressCount[3:0];
  wire         countSplit_1_1 = compressCount[3];
  wire [2:0]   countSplit_1_2 = compressCount[2:0];
  wire         countSplit_2_1 = compressCount[2];
  wire [1:0]   countSplit_2_2 = compressCount[1:0];
  wire         compressDeqValid = eew1H[0] & countSplit_0_1 | eew1H[1] & countSplit_1_1 | eew1H[2] & countSplit_2_1 | ~compress;
  wire [3:0]   _compressCountSelect_T_3 = eew1H[0] ? countSplit_0_2 : 4'h0;
  wire [2:0]   _GEN = _compressCountSelect_T_3[2:0] | (eew1H[1] ? countSplit_1_2 : 3'h0);
  wire [3:0]   compressCountSelect = {_compressCountSelect_T_3[3], _GEN[2], _GEN[1:0] | (eew1H[2] ? countSplit_2_2 : 2'h0)};
  reg  [13:0]  compressVecPipe_0;
  reg  [13:0]  compressVecPipe_1;
  reg  [13:0]  compressVecPipe_2;
  reg  [13:0]  compressVecPipe_3;
  reg  [13:0]  compressVecPipe_4;
  reg  [13:0]  compressVecPipe_5;
  reg  [13:0]  compressVecPipe_6;
  reg  [13:0]  compressVecPipe_7;
  reg  [13:0]  compressVecPipe_8;
  reg  [13:0]  compressVecPipe_9;
  reg  [13:0]  compressVecPipe_10;
  reg  [13:0]  compressVecPipe_11;
  reg  [13:0]  compressVecPipe_12;
  reg  [13:0]  compressVecPipe_13;
  reg  [13:0]  compressVecPipe_14;
  reg  [13:0]  compressVecPipe_15;
  reg          compressMaskVecPipe_0;
  reg          compressMaskVecPipe_1;
  reg          compressMaskVecPipe_2;
  reg          compressMaskVecPipe_3;
  reg          compressMaskVecPipe_4;
  reg          compressMaskVecPipe_5;
  reg          compressMaskVecPipe_6;
  reg          compressMaskVecPipe_7;
  reg          compressMaskVecPipe_8;
  reg          compressMaskVecPipe_9;
  reg          compressMaskVecPipe_10;
  reg          compressMaskVecPipe_11;
  reg          compressMaskVecPipe_12;
  reg          compressMaskVecPipe_13;
  reg          compressMaskVecPipe_14;
  reg          compressMaskVecPipe_15;
  reg  [31:0]  maskPipe;
  reg  [127:0] source2Pipe;
  reg          lastCompressPipe;
  reg          stage2Valid;
  reg          newInstructionPipe;
  reg  [13:0]  compressInitPipe;
  reg          compressDeqValidPipe;
  reg  [10:0]  groupCounterPipe;
  wire [7:0]   viotaResult_res_0 = compressVecPipe_0[7:0];
  wire [7:0]   viotaResult_res_1 = compressVecPipe_1[7:0];
  wire [7:0]   viotaResult_res_2 = compressVecPipe_2[7:0];
  wire [7:0]   viotaResult_res_3 = compressVecPipe_3[7:0];
  wire [15:0]  viotaResult_lo = {viotaResult_res_1, viotaResult_res_0};
  wire [15:0]  viotaResult_hi = {viotaResult_res_3, viotaResult_res_2};
  wire [7:0]   viotaResult_res_0_1 = compressVecPipe_4[7:0];
  wire [7:0]   viotaResult_res_1_1 = compressVecPipe_5[7:0];
  wire [7:0]   viotaResult_res_2_1 = compressVecPipe_6[7:0];
  wire [7:0]   viotaResult_res_3_1 = compressVecPipe_7[7:0];
  wire [15:0]  viotaResult_lo_1 = {viotaResult_res_1_1, viotaResult_res_0_1};
  wire [15:0]  viotaResult_hi_1 = {viotaResult_res_3_1, viotaResult_res_2_1};
  wire [7:0]   viotaResult_res_0_2 = compressVecPipe_8[7:0];
  wire [7:0]   viotaResult_res_1_2 = compressVecPipe_9[7:0];
  wire [7:0]   viotaResult_res_2_2 = compressVecPipe_10[7:0];
  wire [7:0]   viotaResult_res_3_2 = compressVecPipe_11[7:0];
  wire [15:0]  viotaResult_lo_2 = {viotaResult_res_1_2, viotaResult_res_0_2};
  wire [15:0]  viotaResult_hi_2 = {viotaResult_res_3_2, viotaResult_res_2_2};
  wire [7:0]   viotaResult_res_0_3 = compressVecPipe_12[7:0];
  wire [7:0]   viotaResult_res_1_3 = compressVecPipe_13[7:0];
  wire [7:0]   viotaResult_res_2_3 = compressVecPipe_14[7:0];
  wire [7:0]   viotaResult_res_3_3 = compressVecPipe_15[7:0];
  wire [15:0]  viotaResult_lo_3 = {viotaResult_res_1_3, viotaResult_res_0_3};
  wire [15:0]  viotaResult_hi_3 = {viotaResult_res_3_3, viotaResult_res_2_3};
  wire [63:0]  viotaResult_lo_4 = {viotaResult_hi_1, viotaResult_lo_1, viotaResult_hi, viotaResult_lo};
  wire [63:0]  viotaResult_hi_4 = {viotaResult_hi_3, viotaResult_lo_3, viotaResult_hi_2, viotaResult_lo_2};
  wire [15:0]  viotaResult_res_0_4 = {2'h0, compressVecPipe_0};
  wire [15:0]  viotaResult_res_1_4 = {2'h0, compressVecPipe_1};
  wire [15:0]  viotaResult_res_0_5 = {2'h0, compressVecPipe_2};
  wire [15:0]  viotaResult_res_1_5 = {2'h0, compressVecPipe_3};
  wire [15:0]  viotaResult_res_0_6 = {2'h0, compressVecPipe_4};
  wire [15:0]  viotaResult_res_1_6 = {2'h0, compressVecPipe_5};
  wire [15:0]  viotaResult_res_0_7 = {2'h0, compressVecPipe_6};
  wire [15:0]  viotaResult_res_1_7 = {2'h0, compressVecPipe_7};
  wire [63:0]  viotaResult_lo_5 = {viotaResult_res_1_5, viotaResult_res_0_5, viotaResult_res_1_4, viotaResult_res_0_4};
  wire [63:0]  viotaResult_hi_5 = {viotaResult_res_1_7, viotaResult_res_0_7, viotaResult_res_1_6, viotaResult_res_0_6};
  wire [31:0]  viotaResult_res_0_8 = {18'h0, compressVecPipe_0};
  wire [31:0]  viotaResult_res_0_9 = {18'h0, compressVecPipe_1};
  wire [31:0]  viotaResult_res_0_10 = {18'h0, compressVecPipe_2};
  wire [31:0]  viotaResult_res_0_11 = {18'h0, compressVecPipe_3};
  wire [63:0]  viotaResult_lo_6 = {viotaResult_res_0_9, viotaResult_res_0_8};
  wire [63:0]  viotaResult_hi_6 = {viotaResult_res_0_11, viotaResult_res_0_10};
  wire [127:0] viotaResult = (eew1H[0] ? {viotaResult_hi_4, viotaResult_lo_4} : 128'h0) | (eew1H[1] ? {viotaResult_hi_5, viotaResult_lo_5} : 128'h0) | (eew1H[2] ? {viotaResult_hi_6, viotaResult_lo_6} : 128'h0);
  wire         viotaMask_res_0 = maskPipe[0];
  wire         viotaMask_res_1 = maskPipe[1];
  wire         viotaMask_res_2 = maskPipe[2];
  wire         viotaMask_res_3 = maskPipe[3];
  wire [1:0]   viotaMask_lo = {viotaMask_res_1, viotaMask_res_0};
  wire [1:0]   viotaMask_hi = {viotaMask_res_3, viotaMask_res_2};
  wire         viotaMask_res_0_1 = maskPipe[4];
  wire         viotaMask_res_1_1 = maskPipe[5];
  wire         viotaMask_res_2_1 = maskPipe[6];
  wire         viotaMask_res_3_1 = maskPipe[7];
  wire [1:0]   viotaMask_lo_1 = {viotaMask_res_1_1, viotaMask_res_0_1};
  wire [1:0]   viotaMask_hi_1 = {viotaMask_res_3_1, viotaMask_res_2_1};
  wire         viotaMask_res_0_2 = maskPipe[8];
  wire         viotaMask_res_1_2 = maskPipe[9];
  wire         viotaMask_res_2_2 = maskPipe[10];
  wire         viotaMask_res_3_2 = maskPipe[11];
  wire [1:0]   viotaMask_lo_2 = {viotaMask_res_1_2, viotaMask_res_0_2};
  wire [1:0]   viotaMask_hi_2 = {viotaMask_res_3_2, viotaMask_res_2_2};
  wire         viotaMask_res_0_3 = maskPipe[12];
  wire         viotaMask_res_1_3 = maskPipe[13];
  wire         viotaMask_res_2_3 = maskPipe[14];
  wire         viotaMask_res_3_3 = maskPipe[15];
  wire [1:0]   viotaMask_lo_3 = {viotaMask_res_1_3, viotaMask_res_0_3};
  wire [1:0]   viotaMask_hi_3 = {viotaMask_res_3_3, viotaMask_res_2_3};
  wire [7:0]   viotaMask_lo_4 = {viotaMask_hi_1, viotaMask_lo_1, viotaMask_hi, viotaMask_lo};
  wire [7:0]   viotaMask_hi_4 = {viotaMask_hi_3, viotaMask_lo_3, viotaMask_hi_2, viotaMask_lo_2};
  wire [1:0]   viotaMask_res_0_4 = {2{viotaMask_res_0}};
  wire [1:0]   viotaMask_res_1_4 = {2{viotaMask_res_1}};
  wire [1:0]   viotaMask_res_0_5 = {2{viotaMask_res_2}};
  wire [1:0]   viotaMask_res_1_5 = {2{viotaMask_res_3}};
  wire [1:0]   viotaMask_res_0_6 = {2{viotaMask_res_0_1}};
  wire [1:0]   viotaMask_res_1_6 = {2{viotaMask_res_1_1}};
  wire [1:0]   viotaMask_res_0_7 = {2{viotaMask_res_2_1}};
  wire [1:0]   viotaMask_res_1_7 = {2{viotaMask_res_3_1}};
  wire [7:0]   viotaMask_lo_5 = {viotaMask_res_1_5, viotaMask_res_0_5, viotaMask_res_1_4, viotaMask_res_0_4};
  wire [7:0]   viotaMask_hi_5 = {viotaMask_res_1_7, viotaMask_res_0_7, viotaMask_res_1_6, viotaMask_res_0_6};
  wire [3:0]   viotaMask_res_0_8 = {4{viotaMask_res_0}};
  wire [3:0]   viotaMask_res_0_9 = {4{viotaMask_res_1}};
  wire [3:0]   viotaMask_res_0_10 = {4{viotaMask_res_2}};
  wire [3:0]   viotaMask_res_0_11 = {4{viotaMask_res_3}};
  wire [7:0]   viotaMask_lo_6 = {viotaMask_res_0_9, viotaMask_res_0_8};
  wire [7:0]   viotaMask_hi_6 = {viotaMask_res_0_11, viotaMask_res_0_10};
  wire [15:0]  viotaMask = (eew1H[0] ? {viotaMask_hi_4, viotaMask_lo_4} : 16'h0) | (eew1H[1] ? {viotaMask_hi_5, viotaMask_lo_5} : 16'h0) | (eew1H[2] ? {viotaMask_hi_6, viotaMask_lo_6} : 16'h0);
  wire [3:0]   tailCount = compressInitPipe[3:0];
  wire [3:0]   tailCountForMask = compressInit[3:0];
  reg  [127:0] compressDataReg;
  reg          compressTailValid;
  reg  [10:0]  compressWriteGroupCount;
  wire         _GEN_0 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h0;
  wire         compressDataVec_hitReq_0;
  assign compressDataVec_hitReq_0 = _GEN_0;
  wire         compressDataVec_hitReq_0_32;
  assign compressDataVec_hitReq_0_32 = _GEN_0;
  wire         compressDataVec_hitReq_0_48;
  assign compressDataVec_hitReq_0_48 = _GEN_0;
  wire         _GEN_1 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h0;
  wire         compressDataVec_hitReq_1;
  assign compressDataVec_hitReq_1 = _GEN_1;
  wire         compressDataVec_hitReq_1_32;
  assign compressDataVec_hitReq_1_32 = _GEN_1;
  wire         compressDataVec_hitReq_1_48;
  assign compressDataVec_hitReq_1_48 = _GEN_1;
  wire         _GEN_2 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h0;
  wire         compressDataVec_hitReq_2;
  assign compressDataVec_hitReq_2 = _GEN_2;
  wire         compressDataVec_hitReq_2_32;
  assign compressDataVec_hitReq_2_32 = _GEN_2;
  wire         compressDataVec_hitReq_2_48;
  assign compressDataVec_hitReq_2_48 = _GEN_2;
  wire         _GEN_3 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h0;
  wire         compressDataVec_hitReq_3;
  assign compressDataVec_hitReq_3 = _GEN_3;
  wire         compressDataVec_hitReq_3_32;
  assign compressDataVec_hitReq_3_32 = _GEN_3;
  wire         compressDataVec_hitReq_3_48;
  assign compressDataVec_hitReq_3_48 = _GEN_3;
  wire         _GEN_4 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h0;
  wire         compressDataVec_hitReq_4;
  assign compressDataVec_hitReq_4 = _GEN_4;
  wire         compressDataVec_hitReq_4_32;
  assign compressDataVec_hitReq_4_32 = _GEN_4;
  wire         _GEN_5 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h0;
  wire         compressDataVec_hitReq_5;
  assign compressDataVec_hitReq_5 = _GEN_5;
  wire         compressDataVec_hitReq_5_32;
  assign compressDataVec_hitReq_5_32 = _GEN_5;
  wire         _GEN_6 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h0;
  wire         compressDataVec_hitReq_6;
  assign compressDataVec_hitReq_6 = _GEN_6;
  wire         compressDataVec_hitReq_6_32;
  assign compressDataVec_hitReq_6_32 = _GEN_6;
  wire         _GEN_7 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h0;
  wire         compressDataVec_hitReq_7;
  assign compressDataVec_hitReq_7 = _GEN_7;
  wire         compressDataVec_hitReq_7_32;
  assign compressDataVec_hitReq_7_32 = _GEN_7;
  wire         compressDataVec_hitReq_8 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h0;
  wire         compressDataVec_hitReq_9 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h0;
  wire         compressDataVec_hitReq_10 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h0;
  wire         compressDataVec_hitReq_11 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h0;
  wire         compressDataVec_hitReq_12 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h0;
  wire         compressDataVec_hitReq_13 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h0;
  wire         compressDataVec_hitReq_14 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h0;
  wire         compressDataVec_hitReq_15 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h0;
  wire [7:0]   compressDataVec_selectReqData =
    (compressDataVec_hitReq_0 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6 ? source2Pipe[55:48] : 8'h0) | (compressDataVec_hitReq_7 ? source2Pipe[63:56] : 8'h0)
    | (compressDataVec_hitReq_8 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9 ? source2Pipe[79:72] : 8'h0) | (compressDataVec_hitReq_10 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11 ? source2Pipe[95:88] : 8'h0)
    | (compressDataVec_hitReq_12 ? source2Pipe[103:96] : 8'h0) | (compressDataVec_hitReq_13 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14 ? source2Pipe[119:112] : 8'h0)
    | (compressDataVec_hitReq_15 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_useTail;
  assign compressDataVec_useTail = |tailCount;
  wire         _GEN_8 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h1;
  wire         compressDataVec_hitReq_0_1;
  assign compressDataVec_hitReq_0_1 = _GEN_8;
  wire         compressDataVec_hitReq_0_33;
  assign compressDataVec_hitReq_0_33 = _GEN_8;
  wire         compressDataVec_hitReq_0_49;
  assign compressDataVec_hitReq_0_49 = _GEN_8;
  wire         _GEN_9 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h1;
  wire         compressDataVec_hitReq_1_1;
  assign compressDataVec_hitReq_1_1 = _GEN_9;
  wire         compressDataVec_hitReq_1_33;
  assign compressDataVec_hitReq_1_33 = _GEN_9;
  wire         compressDataVec_hitReq_1_49;
  assign compressDataVec_hitReq_1_49 = _GEN_9;
  wire         _GEN_10 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h1;
  wire         compressDataVec_hitReq_2_1;
  assign compressDataVec_hitReq_2_1 = _GEN_10;
  wire         compressDataVec_hitReq_2_33;
  assign compressDataVec_hitReq_2_33 = _GEN_10;
  wire         compressDataVec_hitReq_2_49;
  assign compressDataVec_hitReq_2_49 = _GEN_10;
  wire         _GEN_11 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h1;
  wire         compressDataVec_hitReq_3_1;
  assign compressDataVec_hitReq_3_1 = _GEN_11;
  wire         compressDataVec_hitReq_3_33;
  assign compressDataVec_hitReq_3_33 = _GEN_11;
  wire         compressDataVec_hitReq_3_49;
  assign compressDataVec_hitReq_3_49 = _GEN_11;
  wire         _GEN_12 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h1;
  wire         compressDataVec_hitReq_4_1;
  assign compressDataVec_hitReq_4_1 = _GEN_12;
  wire         compressDataVec_hitReq_4_33;
  assign compressDataVec_hitReq_4_33 = _GEN_12;
  wire         _GEN_13 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h1;
  wire         compressDataVec_hitReq_5_1;
  assign compressDataVec_hitReq_5_1 = _GEN_13;
  wire         compressDataVec_hitReq_5_33;
  assign compressDataVec_hitReq_5_33 = _GEN_13;
  wire         _GEN_14 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h1;
  wire         compressDataVec_hitReq_6_1;
  assign compressDataVec_hitReq_6_1 = _GEN_14;
  wire         compressDataVec_hitReq_6_33;
  assign compressDataVec_hitReq_6_33 = _GEN_14;
  wire         _GEN_15 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h1;
  wire         compressDataVec_hitReq_7_1;
  assign compressDataVec_hitReq_7_1 = _GEN_15;
  wire         compressDataVec_hitReq_7_33;
  assign compressDataVec_hitReq_7_33 = _GEN_15;
  wire         compressDataVec_hitReq_8_1 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h1;
  wire         compressDataVec_hitReq_9_1 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h1;
  wire         compressDataVec_hitReq_10_1 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h1;
  wire         compressDataVec_hitReq_11_1 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h1;
  wire         compressDataVec_hitReq_12_1 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h1;
  wire         compressDataVec_hitReq_13_1 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h1;
  wire         compressDataVec_hitReq_14_1 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h1;
  wire         compressDataVec_hitReq_15_1 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h1;
  wire [7:0]   compressDataVec_selectReqData_1 =
    (compressDataVec_hitReq_0_1 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_1 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_1 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_1 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_1 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_1 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_1 ? source2Pipe[55:48] : 8'h0) | (compressDataVec_hitReq_7_1 ? source2Pipe[63:56] : 8'h0)
    | (compressDataVec_hitReq_8_1 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_1 ? source2Pipe[79:72] : 8'h0) | (compressDataVec_hitReq_10_1 ? source2Pipe[87:80] : 8'h0)
    | (compressDataVec_hitReq_11_1 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_1 ? source2Pipe[103:96] : 8'h0) | (compressDataVec_hitReq_13_1 ? source2Pipe[111:104] : 8'h0)
    | (compressDataVec_hitReq_14_1 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_1 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_useTail_1;
  assign compressDataVec_useTail_1 = |(tailCount[3:1]);
  wire         _GEN_16 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h2;
  wire         compressDataVec_hitReq_0_2;
  assign compressDataVec_hitReq_0_2 = _GEN_16;
  wire         compressDataVec_hitReq_0_34;
  assign compressDataVec_hitReq_0_34 = _GEN_16;
  wire         compressDataVec_hitReq_0_50;
  assign compressDataVec_hitReq_0_50 = _GEN_16;
  wire         _GEN_17 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h2;
  wire         compressDataVec_hitReq_1_2;
  assign compressDataVec_hitReq_1_2 = _GEN_17;
  wire         compressDataVec_hitReq_1_34;
  assign compressDataVec_hitReq_1_34 = _GEN_17;
  wire         compressDataVec_hitReq_1_50;
  assign compressDataVec_hitReq_1_50 = _GEN_17;
  wire         _GEN_18 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h2;
  wire         compressDataVec_hitReq_2_2;
  assign compressDataVec_hitReq_2_2 = _GEN_18;
  wire         compressDataVec_hitReq_2_34;
  assign compressDataVec_hitReq_2_34 = _GEN_18;
  wire         compressDataVec_hitReq_2_50;
  assign compressDataVec_hitReq_2_50 = _GEN_18;
  wire         _GEN_19 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h2;
  wire         compressDataVec_hitReq_3_2;
  assign compressDataVec_hitReq_3_2 = _GEN_19;
  wire         compressDataVec_hitReq_3_34;
  assign compressDataVec_hitReq_3_34 = _GEN_19;
  wire         compressDataVec_hitReq_3_50;
  assign compressDataVec_hitReq_3_50 = _GEN_19;
  wire         _GEN_20 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h2;
  wire         compressDataVec_hitReq_4_2;
  assign compressDataVec_hitReq_4_2 = _GEN_20;
  wire         compressDataVec_hitReq_4_34;
  assign compressDataVec_hitReq_4_34 = _GEN_20;
  wire         _GEN_21 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h2;
  wire         compressDataVec_hitReq_5_2;
  assign compressDataVec_hitReq_5_2 = _GEN_21;
  wire         compressDataVec_hitReq_5_34;
  assign compressDataVec_hitReq_5_34 = _GEN_21;
  wire         _GEN_22 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h2;
  wire         compressDataVec_hitReq_6_2;
  assign compressDataVec_hitReq_6_2 = _GEN_22;
  wire         compressDataVec_hitReq_6_34;
  assign compressDataVec_hitReq_6_34 = _GEN_22;
  wire         _GEN_23 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h2;
  wire         compressDataVec_hitReq_7_2;
  assign compressDataVec_hitReq_7_2 = _GEN_23;
  wire         compressDataVec_hitReq_7_34;
  assign compressDataVec_hitReq_7_34 = _GEN_23;
  wire         compressDataVec_hitReq_8_2 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h2;
  wire         compressDataVec_hitReq_9_2 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h2;
  wire         compressDataVec_hitReq_10_2 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h2;
  wire         compressDataVec_hitReq_11_2 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h2;
  wire         compressDataVec_hitReq_12_2 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h2;
  wire         compressDataVec_hitReq_13_2 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h2;
  wire         compressDataVec_hitReq_14_2 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h2;
  wire         compressDataVec_hitReq_15_2 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h2;
  wire [7:0]   compressDataVec_selectReqData_2 =
    (compressDataVec_hitReq_0_2 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_2 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_2 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_2 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_2 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_2 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_2 ? source2Pipe[55:48] : 8'h0) | (compressDataVec_hitReq_7_2 ? source2Pipe[63:56] : 8'h0)
    | (compressDataVec_hitReq_8_2 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_2 ? source2Pipe[79:72] : 8'h0) | (compressDataVec_hitReq_10_2 ? source2Pipe[87:80] : 8'h0)
    | (compressDataVec_hitReq_11_2 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_2 ? source2Pipe[103:96] : 8'h0) | (compressDataVec_hitReq_13_2 ? source2Pipe[111:104] : 8'h0)
    | (compressDataVec_hitReq_14_2 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_2 ? source2Pipe[127:120] : 8'h0);
  wire         _GEN_24 = tailCount > 4'h2;
  wire         compressDataVec_useTail_2;
  assign compressDataVec_useTail_2 = _GEN_24;
  wire         compressDataVec_useTail_18;
  assign compressDataVec_useTail_18 = _GEN_24;
  wire         compressDataVec_useTail_26;
  assign compressDataVec_useTail_26 = _GEN_24;
  wire         _GEN_25 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h3;
  wire         compressDataVec_hitReq_0_3;
  assign compressDataVec_hitReq_0_3 = _GEN_25;
  wire         compressDataVec_hitReq_0_35;
  assign compressDataVec_hitReq_0_35 = _GEN_25;
  wire         compressDataVec_hitReq_0_51;
  assign compressDataVec_hitReq_0_51 = _GEN_25;
  wire         _GEN_26 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h3;
  wire         compressDataVec_hitReq_1_3;
  assign compressDataVec_hitReq_1_3 = _GEN_26;
  wire         compressDataVec_hitReq_1_35;
  assign compressDataVec_hitReq_1_35 = _GEN_26;
  wire         compressDataVec_hitReq_1_51;
  assign compressDataVec_hitReq_1_51 = _GEN_26;
  wire         _GEN_27 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h3;
  wire         compressDataVec_hitReq_2_3;
  assign compressDataVec_hitReq_2_3 = _GEN_27;
  wire         compressDataVec_hitReq_2_35;
  assign compressDataVec_hitReq_2_35 = _GEN_27;
  wire         compressDataVec_hitReq_2_51;
  assign compressDataVec_hitReq_2_51 = _GEN_27;
  wire         _GEN_28 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h3;
  wire         compressDataVec_hitReq_3_3;
  assign compressDataVec_hitReq_3_3 = _GEN_28;
  wire         compressDataVec_hitReq_3_35;
  assign compressDataVec_hitReq_3_35 = _GEN_28;
  wire         compressDataVec_hitReq_3_51;
  assign compressDataVec_hitReq_3_51 = _GEN_28;
  wire         _GEN_29 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h3;
  wire         compressDataVec_hitReq_4_3;
  assign compressDataVec_hitReq_4_3 = _GEN_29;
  wire         compressDataVec_hitReq_4_35;
  assign compressDataVec_hitReq_4_35 = _GEN_29;
  wire         _GEN_30 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h3;
  wire         compressDataVec_hitReq_5_3;
  assign compressDataVec_hitReq_5_3 = _GEN_30;
  wire         compressDataVec_hitReq_5_35;
  assign compressDataVec_hitReq_5_35 = _GEN_30;
  wire         _GEN_31 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h3;
  wire         compressDataVec_hitReq_6_3;
  assign compressDataVec_hitReq_6_3 = _GEN_31;
  wire         compressDataVec_hitReq_6_35;
  assign compressDataVec_hitReq_6_35 = _GEN_31;
  wire         _GEN_32 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h3;
  wire         compressDataVec_hitReq_7_3;
  assign compressDataVec_hitReq_7_3 = _GEN_32;
  wire         compressDataVec_hitReq_7_35;
  assign compressDataVec_hitReq_7_35 = _GEN_32;
  wire         compressDataVec_hitReq_8_3 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h3;
  wire         compressDataVec_hitReq_9_3 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h3;
  wire         compressDataVec_hitReq_10_3 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h3;
  wire         compressDataVec_hitReq_11_3 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h3;
  wire         compressDataVec_hitReq_12_3 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h3;
  wire         compressDataVec_hitReq_13_3 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h3;
  wire         compressDataVec_hitReq_14_3 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h3;
  wire         compressDataVec_hitReq_15_3 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h3;
  wire [7:0]   compressDataVec_selectReqData_3 =
    (compressDataVec_hitReq_0_3 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_3 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_3 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_3 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_3 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_3 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_3 ? source2Pipe[55:48] : 8'h0) | (compressDataVec_hitReq_7_3 ? source2Pipe[63:56] : 8'h0)
    | (compressDataVec_hitReq_8_3 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_3 ? source2Pipe[79:72] : 8'h0) | (compressDataVec_hitReq_10_3 ? source2Pipe[87:80] : 8'h0)
    | (compressDataVec_hitReq_11_3 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_3 ? source2Pipe[103:96] : 8'h0) | (compressDataVec_hitReq_13_3 ? source2Pipe[111:104] : 8'h0)
    | (compressDataVec_hitReq_14_3 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_3 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_useTail_3;
  assign compressDataVec_useTail_3 = |(tailCount[3:2]);
  wire         _GEN_33 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h4;
  wire         compressDataVec_hitReq_0_4;
  assign compressDataVec_hitReq_0_4 = _GEN_33;
  wire         compressDataVec_hitReq_0_36;
  assign compressDataVec_hitReq_0_36 = _GEN_33;
  wire         compressDataVec_hitReq_0_52;
  assign compressDataVec_hitReq_0_52 = _GEN_33;
  wire         _GEN_34 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h4;
  wire         compressDataVec_hitReq_1_4;
  assign compressDataVec_hitReq_1_4 = _GEN_34;
  wire         compressDataVec_hitReq_1_36;
  assign compressDataVec_hitReq_1_36 = _GEN_34;
  wire         compressDataVec_hitReq_1_52;
  assign compressDataVec_hitReq_1_52 = _GEN_34;
  wire         _GEN_35 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h4;
  wire         compressDataVec_hitReq_2_4;
  assign compressDataVec_hitReq_2_4 = _GEN_35;
  wire         compressDataVec_hitReq_2_36;
  assign compressDataVec_hitReq_2_36 = _GEN_35;
  wire         compressDataVec_hitReq_2_52;
  assign compressDataVec_hitReq_2_52 = _GEN_35;
  wire         _GEN_36 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h4;
  wire         compressDataVec_hitReq_3_4;
  assign compressDataVec_hitReq_3_4 = _GEN_36;
  wire         compressDataVec_hitReq_3_36;
  assign compressDataVec_hitReq_3_36 = _GEN_36;
  wire         compressDataVec_hitReq_3_52;
  assign compressDataVec_hitReq_3_52 = _GEN_36;
  wire         _GEN_37 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h4;
  wire         compressDataVec_hitReq_4_4;
  assign compressDataVec_hitReq_4_4 = _GEN_37;
  wire         compressDataVec_hitReq_4_36;
  assign compressDataVec_hitReq_4_36 = _GEN_37;
  wire         _GEN_38 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h4;
  wire         compressDataVec_hitReq_5_4;
  assign compressDataVec_hitReq_5_4 = _GEN_38;
  wire         compressDataVec_hitReq_5_36;
  assign compressDataVec_hitReq_5_36 = _GEN_38;
  wire         _GEN_39 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h4;
  wire         compressDataVec_hitReq_6_4;
  assign compressDataVec_hitReq_6_4 = _GEN_39;
  wire         compressDataVec_hitReq_6_36;
  assign compressDataVec_hitReq_6_36 = _GEN_39;
  wire         _GEN_40 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h4;
  wire         compressDataVec_hitReq_7_4;
  assign compressDataVec_hitReq_7_4 = _GEN_40;
  wire         compressDataVec_hitReq_7_36;
  assign compressDataVec_hitReq_7_36 = _GEN_40;
  wire         compressDataVec_hitReq_8_4 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h4;
  wire         compressDataVec_hitReq_9_4 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h4;
  wire         compressDataVec_hitReq_10_4 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h4;
  wire         compressDataVec_hitReq_11_4 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h4;
  wire         compressDataVec_hitReq_12_4 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h4;
  wire         compressDataVec_hitReq_13_4 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h4;
  wire         compressDataVec_hitReq_14_4 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h4;
  wire         compressDataVec_hitReq_15_4 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h4;
  wire [7:0]   compressDataVec_selectReqData_4 =
    (compressDataVec_hitReq_0_4 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_4 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_4 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_4 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_4 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_4 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_4 ? source2Pipe[55:48] : 8'h0) | (compressDataVec_hitReq_7_4 ? source2Pipe[63:56] : 8'h0)
    | (compressDataVec_hitReq_8_4 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_4 ? source2Pipe[79:72] : 8'h0) | (compressDataVec_hitReq_10_4 ? source2Pipe[87:80] : 8'h0)
    | (compressDataVec_hitReq_11_4 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_4 ? source2Pipe[103:96] : 8'h0) | (compressDataVec_hitReq_13_4 ? source2Pipe[111:104] : 8'h0)
    | (compressDataVec_hitReq_14_4 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_4 ? source2Pipe[127:120] : 8'h0);
  wire         _GEN_41 = tailCount > 4'h4;
  wire         compressDataVec_useTail_4;
  assign compressDataVec_useTail_4 = _GEN_41;
  wire         compressDataVec_useTail_20;
  assign compressDataVec_useTail_20 = _GEN_41;
  wire         _GEN_42 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h5;
  wire         compressDataVec_hitReq_0_5;
  assign compressDataVec_hitReq_0_5 = _GEN_42;
  wire         compressDataVec_hitReq_0_37;
  assign compressDataVec_hitReq_0_37 = _GEN_42;
  wire         compressDataVec_hitReq_0_53;
  assign compressDataVec_hitReq_0_53 = _GEN_42;
  wire         _GEN_43 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h5;
  wire         compressDataVec_hitReq_1_5;
  assign compressDataVec_hitReq_1_5 = _GEN_43;
  wire         compressDataVec_hitReq_1_37;
  assign compressDataVec_hitReq_1_37 = _GEN_43;
  wire         compressDataVec_hitReq_1_53;
  assign compressDataVec_hitReq_1_53 = _GEN_43;
  wire         _GEN_44 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h5;
  wire         compressDataVec_hitReq_2_5;
  assign compressDataVec_hitReq_2_5 = _GEN_44;
  wire         compressDataVec_hitReq_2_37;
  assign compressDataVec_hitReq_2_37 = _GEN_44;
  wire         compressDataVec_hitReq_2_53;
  assign compressDataVec_hitReq_2_53 = _GEN_44;
  wire         _GEN_45 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h5;
  wire         compressDataVec_hitReq_3_5;
  assign compressDataVec_hitReq_3_5 = _GEN_45;
  wire         compressDataVec_hitReq_3_37;
  assign compressDataVec_hitReq_3_37 = _GEN_45;
  wire         compressDataVec_hitReq_3_53;
  assign compressDataVec_hitReq_3_53 = _GEN_45;
  wire         _GEN_46 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h5;
  wire         compressDataVec_hitReq_4_5;
  assign compressDataVec_hitReq_4_5 = _GEN_46;
  wire         compressDataVec_hitReq_4_37;
  assign compressDataVec_hitReq_4_37 = _GEN_46;
  wire         _GEN_47 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h5;
  wire         compressDataVec_hitReq_5_5;
  assign compressDataVec_hitReq_5_5 = _GEN_47;
  wire         compressDataVec_hitReq_5_37;
  assign compressDataVec_hitReq_5_37 = _GEN_47;
  wire         _GEN_48 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h5;
  wire         compressDataVec_hitReq_6_5;
  assign compressDataVec_hitReq_6_5 = _GEN_48;
  wire         compressDataVec_hitReq_6_37;
  assign compressDataVec_hitReq_6_37 = _GEN_48;
  wire         _GEN_49 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h5;
  wire         compressDataVec_hitReq_7_5;
  assign compressDataVec_hitReq_7_5 = _GEN_49;
  wire         compressDataVec_hitReq_7_37;
  assign compressDataVec_hitReq_7_37 = _GEN_49;
  wire         compressDataVec_hitReq_8_5 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h5;
  wire         compressDataVec_hitReq_9_5 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h5;
  wire         compressDataVec_hitReq_10_5 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h5;
  wire         compressDataVec_hitReq_11_5 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h5;
  wire         compressDataVec_hitReq_12_5 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h5;
  wire         compressDataVec_hitReq_13_5 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h5;
  wire         compressDataVec_hitReq_14_5 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h5;
  wire         compressDataVec_hitReq_15_5 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h5;
  wire [7:0]   compressDataVec_selectReqData_5 =
    (compressDataVec_hitReq_0_5 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_5 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_5 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_5 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_5 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_5 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_5 ? source2Pipe[55:48] : 8'h0) | (compressDataVec_hitReq_7_5 ? source2Pipe[63:56] : 8'h0)
    | (compressDataVec_hitReq_8_5 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_5 ? source2Pipe[79:72] : 8'h0) | (compressDataVec_hitReq_10_5 ? source2Pipe[87:80] : 8'h0)
    | (compressDataVec_hitReq_11_5 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_5 ? source2Pipe[103:96] : 8'h0) | (compressDataVec_hitReq_13_5 ? source2Pipe[111:104] : 8'h0)
    | (compressDataVec_hitReq_14_5 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_5 ? source2Pipe[127:120] : 8'h0);
  wire         _GEN_50 = tailCount > 4'h5;
  wire         compressDataVec_useTail_5;
  assign compressDataVec_useTail_5 = _GEN_50;
  wire         compressDataVec_useTail_21;
  assign compressDataVec_useTail_21 = _GEN_50;
  wire         _GEN_51 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h6;
  wire         compressDataVec_hitReq_0_6;
  assign compressDataVec_hitReq_0_6 = _GEN_51;
  wire         compressDataVec_hitReq_0_38;
  assign compressDataVec_hitReq_0_38 = _GEN_51;
  wire         compressDataVec_hitReq_0_54;
  assign compressDataVec_hitReq_0_54 = _GEN_51;
  wire         _GEN_52 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h6;
  wire         compressDataVec_hitReq_1_6;
  assign compressDataVec_hitReq_1_6 = _GEN_52;
  wire         compressDataVec_hitReq_1_38;
  assign compressDataVec_hitReq_1_38 = _GEN_52;
  wire         compressDataVec_hitReq_1_54;
  assign compressDataVec_hitReq_1_54 = _GEN_52;
  wire         _GEN_53 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h6;
  wire         compressDataVec_hitReq_2_6;
  assign compressDataVec_hitReq_2_6 = _GEN_53;
  wire         compressDataVec_hitReq_2_38;
  assign compressDataVec_hitReq_2_38 = _GEN_53;
  wire         compressDataVec_hitReq_2_54;
  assign compressDataVec_hitReq_2_54 = _GEN_53;
  wire         _GEN_54 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h6;
  wire         compressDataVec_hitReq_3_6;
  assign compressDataVec_hitReq_3_6 = _GEN_54;
  wire         compressDataVec_hitReq_3_38;
  assign compressDataVec_hitReq_3_38 = _GEN_54;
  wire         compressDataVec_hitReq_3_54;
  assign compressDataVec_hitReq_3_54 = _GEN_54;
  wire         _GEN_55 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h6;
  wire         compressDataVec_hitReq_4_6;
  assign compressDataVec_hitReq_4_6 = _GEN_55;
  wire         compressDataVec_hitReq_4_38;
  assign compressDataVec_hitReq_4_38 = _GEN_55;
  wire         _GEN_56 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h6;
  wire         compressDataVec_hitReq_5_6;
  assign compressDataVec_hitReq_5_6 = _GEN_56;
  wire         compressDataVec_hitReq_5_38;
  assign compressDataVec_hitReq_5_38 = _GEN_56;
  wire         _GEN_57 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h6;
  wire         compressDataVec_hitReq_6_6;
  assign compressDataVec_hitReq_6_6 = _GEN_57;
  wire         compressDataVec_hitReq_6_38;
  assign compressDataVec_hitReq_6_38 = _GEN_57;
  wire         _GEN_58 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h6;
  wire         compressDataVec_hitReq_7_6;
  assign compressDataVec_hitReq_7_6 = _GEN_58;
  wire         compressDataVec_hitReq_7_38;
  assign compressDataVec_hitReq_7_38 = _GEN_58;
  wire         compressDataVec_hitReq_8_6 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h6;
  wire         compressDataVec_hitReq_9_6 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h6;
  wire         compressDataVec_hitReq_10_6 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h6;
  wire         compressDataVec_hitReq_11_6 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h6;
  wire         compressDataVec_hitReq_12_6 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h6;
  wire         compressDataVec_hitReq_13_6 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h6;
  wire         compressDataVec_hitReq_14_6 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h6;
  wire         compressDataVec_hitReq_15_6 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h6;
  wire [7:0]   compressDataVec_selectReqData_6 =
    (compressDataVec_hitReq_0_6 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_6 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_6 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_6 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_6 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_6 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_6 ? source2Pipe[55:48] : 8'h0) | (compressDataVec_hitReq_7_6 ? source2Pipe[63:56] : 8'h0)
    | (compressDataVec_hitReq_8_6 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_6 ? source2Pipe[79:72] : 8'h0) | (compressDataVec_hitReq_10_6 ? source2Pipe[87:80] : 8'h0)
    | (compressDataVec_hitReq_11_6 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_6 ? source2Pipe[103:96] : 8'h0) | (compressDataVec_hitReq_13_6 ? source2Pipe[111:104] : 8'h0)
    | (compressDataVec_hitReq_14_6 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_6 ? source2Pipe[127:120] : 8'h0);
  wire         _GEN_59 = tailCount > 4'h6;
  wire         compressDataVec_useTail_6;
  assign compressDataVec_useTail_6 = _GEN_59;
  wire         compressDataVec_useTail_22;
  assign compressDataVec_useTail_22 = _GEN_59;
  wire         _GEN_60 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h7;
  wire         compressDataVec_hitReq_0_7;
  assign compressDataVec_hitReq_0_7 = _GEN_60;
  wire         compressDataVec_hitReq_0_39;
  assign compressDataVec_hitReq_0_39 = _GEN_60;
  wire         compressDataVec_hitReq_0_55;
  assign compressDataVec_hitReq_0_55 = _GEN_60;
  wire         _GEN_61 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h7;
  wire         compressDataVec_hitReq_1_7;
  assign compressDataVec_hitReq_1_7 = _GEN_61;
  wire         compressDataVec_hitReq_1_39;
  assign compressDataVec_hitReq_1_39 = _GEN_61;
  wire         compressDataVec_hitReq_1_55;
  assign compressDataVec_hitReq_1_55 = _GEN_61;
  wire         _GEN_62 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h7;
  wire         compressDataVec_hitReq_2_7;
  assign compressDataVec_hitReq_2_7 = _GEN_62;
  wire         compressDataVec_hitReq_2_39;
  assign compressDataVec_hitReq_2_39 = _GEN_62;
  wire         compressDataVec_hitReq_2_55;
  assign compressDataVec_hitReq_2_55 = _GEN_62;
  wire         _GEN_63 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h7;
  wire         compressDataVec_hitReq_3_7;
  assign compressDataVec_hitReq_3_7 = _GEN_63;
  wire         compressDataVec_hitReq_3_39;
  assign compressDataVec_hitReq_3_39 = _GEN_63;
  wire         compressDataVec_hitReq_3_55;
  assign compressDataVec_hitReq_3_55 = _GEN_63;
  wire         _GEN_64 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h7;
  wire         compressDataVec_hitReq_4_7;
  assign compressDataVec_hitReq_4_7 = _GEN_64;
  wire         compressDataVec_hitReq_4_39;
  assign compressDataVec_hitReq_4_39 = _GEN_64;
  wire         _GEN_65 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h7;
  wire         compressDataVec_hitReq_5_7;
  assign compressDataVec_hitReq_5_7 = _GEN_65;
  wire         compressDataVec_hitReq_5_39;
  assign compressDataVec_hitReq_5_39 = _GEN_65;
  wire         _GEN_66 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h7;
  wire         compressDataVec_hitReq_6_7;
  assign compressDataVec_hitReq_6_7 = _GEN_66;
  wire         compressDataVec_hitReq_6_39;
  assign compressDataVec_hitReq_6_39 = _GEN_66;
  wire         _GEN_67 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h7;
  wire         compressDataVec_hitReq_7_7;
  assign compressDataVec_hitReq_7_7 = _GEN_67;
  wire         compressDataVec_hitReq_7_39;
  assign compressDataVec_hitReq_7_39 = _GEN_67;
  wire         compressDataVec_hitReq_8_7 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h7;
  wire         compressDataVec_hitReq_9_7 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h7;
  wire         compressDataVec_hitReq_10_7 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h7;
  wire         compressDataVec_hitReq_11_7 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h7;
  wire         compressDataVec_hitReq_12_7 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h7;
  wire         compressDataVec_hitReq_13_7 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h7;
  wire         compressDataVec_hitReq_14_7 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h7;
  wire         compressDataVec_hitReq_15_7 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h7;
  wire [7:0]   compressDataVec_selectReqData_7 =
    (compressDataVec_hitReq_0_7 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_7 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_7 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_7 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_7 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_7 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_7 ? source2Pipe[55:48] : 8'h0) | (compressDataVec_hitReq_7_7 ? source2Pipe[63:56] : 8'h0)
    | (compressDataVec_hitReq_8_7 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_7 ? source2Pipe[79:72] : 8'h0) | (compressDataVec_hitReq_10_7 ? source2Pipe[87:80] : 8'h0)
    | (compressDataVec_hitReq_11_7 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_7 ? source2Pipe[103:96] : 8'h0) | (compressDataVec_hitReq_13_7 ? source2Pipe[111:104] : 8'h0)
    | (compressDataVec_hitReq_14_7 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_7 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_useTail_7 = tailCount[3];
  wire         compressDataVec_useTail_23 = tailCount[3];
  wire         _GEN_68 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h8;
  wire         compressDataVec_hitReq_0_8;
  assign compressDataVec_hitReq_0_8 = _GEN_68;
  wire         compressDataVec_hitReq_0_40;
  assign compressDataVec_hitReq_0_40 = _GEN_68;
  wire         _GEN_69 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h8;
  wire         compressDataVec_hitReq_1_8;
  assign compressDataVec_hitReq_1_8 = _GEN_69;
  wire         compressDataVec_hitReq_1_40;
  assign compressDataVec_hitReq_1_40 = _GEN_69;
  wire         _GEN_70 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h8;
  wire         compressDataVec_hitReq_2_8;
  assign compressDataVec_hitReq_2_8 = _GEN_70;
  wire         compressDataVec_hitReq_2_40;
  assign compressDataVec_hitReq_2_40 = _GEN_70;
  wire         _GEN_71 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h8;
  wire         compressDataVec_hitReq_3_8;
  assign compressDataVec_hitReq_3_8 = _GEN_71;
  wire         compressDataVec_hitReq_3_40;
  assign compressDataVec_hitReq_3_40 = _GEN_71;
  wire         _GEN_72 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h8;
  wire         compressDataVec_hitReq_4_8;
  assign compressDataVec_hitReq_4_8 = _GEN_72;
  wire         compressDataVec_hitReq_4_40;
  assign compressDataVec_hitReq_4_40 = _GEN_72;
  wire         _GEN_73 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h8;
  wire         compressDataVec_hitReq_5_8;
  assign compressDataVec_hitReq_5_8 = _GEN_73;
  wire         compressDataVec_hitReq_5_40;
  assign compressDataVec_hitReq_5_40 = _GEN_73;
  wire         _GEN_74 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h8;
  wire         compressDataVec_hitReq_6_8;
  assign compressDataVec_hitReq_6_8 = _GEN_74;
  wire         compressDataVec_hitReq_6_40;
  assign compressDataVec_hitReq_6_40 = _GEN_74;
  wire         _GEN_75 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h8;
  wire         compressDataVec_hitReq_7_8;
  assign compressDataVec_hitReq_7_8 = _GEN_75;
  wire         compressDataVec_hitReq_7_40;
  assign compressDataVec_hitReq_7_40 = _GEN_75;
  wire         compressDataVec_hitReq_8_8 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h8;
  wire         compressDataVec_hitReq_9_8 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h8;
  wire         compressDataVec_hitReq_10_8 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h8;
  wire         compressDataVec_hitReq_11_8 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h8;
  wire         compressDataVec_hitReq_12_8 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h8;
  wire         compressDataVec_hitReq_13_8 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h8;
  wire         compressDataVec_hitReq_14_8 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h8;
  wire         compressDataVec_hitReq_15_8 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h8;
  wire [7:0]   compressDataVec_selectReqData_8 =
    (compressDataVec_hitReq_0_8 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_8 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_8 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_8 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_8 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_8 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_8 ? source2Pipe[55:48] : 8'h0) | (compressDataVec_hitReq_7_8 ? source2Pipe[63:56] : 8'h0)
    | (compressDataVec_hitReq_8_8 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_8 ? source2Pipe[79:72] : 8'h0) | (compressDataVec_hitReq_10_8 ? source2Pipe[87:80] : 8'h0)
    | (compressDataVec_hitReq_11_8 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_8 ? source2Pipe[103:96] : 8'h0) | (compressDataVec_hitReq_13_8 ? source2Pipe[111:104] : 8'h0)
    | (compressDataVec_hitReq_14_8 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_8 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_useTail_8 = tailCount > 4'h8;
  wire         _GEN_76 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h9;
  wire         compressDataVec_hitReq_0_9;
  assign compressDataVec_hitReq_0_9 = _GEN_76;
  wire         compressDataVec_hitReq_0_41;
  assign compressDataVec_hitReq_0_41 = _GEN_76;
  wire         _GEN_77 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h9;
  wire         compressDataVec_hitReq_1_9;
  assign compressDataVec_hitReq_1_9 = _GEN_77;
  wire         compressDataVec_hitReq_1_41;
  assign compressDataVec_hitReq_1_41 = _GEN_77;
  wire         _GEN_78 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h9;
  wire         compressDataVec_hitReq_2_9;
  assign compressDataVec_hitReq_2_9 = _GEN_78;
  wire         compressDataVec_hitReq_2_41;
  assign compressDataVec_hitReq_2_41 = _GEN_78;
  wire         _GEN_79 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h9;
  wire         compressDataVec_hitReq_3_9;
  assign compressDataVec_hitReq_3_9 = _GEN_79;
  wire         compressDataVec_hitReq_3_41;
  assign compressDataVec_hitReq_3_41 = _GEN_79;
  wire         _GEN_80 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h9;
  wire         compressDataVec_hitReq_4_9;
  assign compressDataVec_hitReq_4_9 = _GEN_80;
  wire         compressDataVec_hitReq_4_41;
  assign compressDataVec_hitReq_4_41 = _GEN_80;
  wire         _GEN_81 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h9;
  wire         compressDataVec_hitReq_5_9;
  assign compressDataVec_hitReq_5_9 = _GEN_81;
  wire         compressDataVec_hitReq_5_41;
  assign compressDataVec_hitReq_5_41 = _GEN_81;
  wire         _GEN_82 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h9;
  wire         compressDataVec_hitReq_6_9;
  assign compressDataVec_hitReq_6_9 = _GEN_82;
  wire         compressDataVec_hitReq_6_41;
  assign compressDataVec_hitReq_6_41 = _GEN_82;
  wire         _GEN_83 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h9;
  wire         compressDataVec_hitReq_7_9;
  assign compressDataVec_hitReq_7_9 = _GEN_83;
  wire         compressDataVec_hitReq_7_41;
  assign compressDataVec_hitReq_7_41 = _GEN_83;
  wire         compressDataVec_hitReq_8_9 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h9;
  wire         compressDataVec_hitReq_9_9 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h9;
  wire         compressDataVec_hitReq_10_9 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h9;
  wire         compressDataVec_hitReq_11_9 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h9;
  wire         compressDataVec_hitReq_12_9 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h9;
  wire         compressDataVec_hitReq_13_9 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h9;
  wire         compressDataVec_hitReq_14_9 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h9;
  wire         compressDataVec_hitReq_15_9 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h9;
  wire [7:0]   compressDataVec_selectReqData_9 =
    (compressDataVec_hitReq_0_9 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_9 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_9 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_9 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_9 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_9 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_9 ? source2Pipe[55:48] : 8'h0) | (compressDataVec_hitReq_7_9 ? source2Pipe[63:56] : 8'h0)
    | (compressDataVec_hitReq_8_9 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_9 ? source2Pipe[79:72] : 8'h0) | (compressDataVec_hitReq_10_9 ? source2Pipe[87:80] : 8'h0)
    | (compressDataVec_hitReq_11_9 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_9 ? source2Pipe[103:96] : 8'h0) | (compressDataVec_hitReq_13_9 ? source2Pipe[111:104] : 8'h0)
    | (compressDataVec_hitReq_14_9 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_9 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_useTail_9 = tailCount > 4'h9;
  wire         _GEN_84 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'hA;
  wire         compressDataVec_hitReq_0_10;
  assign compressDataVec_hitReq_0_10 = _GEN_84;
  wire         compressDataVec_hitReq_0_42;
  assign compressDataVec_hitReq_0_42 = _GEN_84;
  wire         _GEN_85 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'hA;
  wire         compressDataVec_hitReq_1_10;
  assign compressDataVec_hitReq_1_10 = _GEN_85;
  wire         compressDataVec_hitReq_1_42;
  assign compressDataVec_hitReq_1_42 = _GEN_85;
  wire         _GEN_86 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'hA;
  wire         compressDataVec_hitReq_2_10;
  assign compressDataVec_hitReq_2_10 = _GEN_86;
  wire         compressDataVec_hitReq_2_42;
  assign compressDataVec_hitReq_2_42 = _GEN_86;
  wire         _GEN_87 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'hA;
  wire         compressDataVec_hitReq_3_10;
  assign compressDataVec_hitReq_3_10 = _GEN_87;
  wire         compressDataVec_hitReq_3_42;
  assign compressDataVec_hitReq_3_42 = _GEN_87;
  wire         _GEN_88 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'hA;
  wire         compressDataVec_hitReq_4_10;
  assign compressDataVec_hitReq_4_10 = _GEN_88;
  wire         compressDataVec_hitReq_4_42;
  assign compressDataVec_hitReq_4_42 = _GEN_88;
  wire         _GEN_89 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'hA;
  wire         compressDataVec_hitReq_5_10;
  assign compressDataVec_hitReq_5_10 = _GEN_89;
  wire         compressDataVec_hitReq_5_42;
  assign compressDataVec_hitReq_5_42 = _GEN_89;
  wire         _GEN_90 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'hA;
  wire         compressDataVec_hitReq_6_10;
  assign compressDataVec_hitReq_6_10 = _GEN_90;
  wire         compressDataVec_hitReq_6_42;
  assign compressDataVec_hitReq_6_42 = _GEN_90;
  wire         _GEN_91 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'hA;
  wire         compressDataVec_hitReq_7_10;
  assign compressDataVec_hitReq_7_10 = _GEN_91;
  wire         compressDataVec_hitReq_7_42;
  assign compressDataVec_hitReq_7_42 = _GEN_91;
  wire         compressDataVec_hitReq_8_10 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'hA;
  wire         compressDataVec_hitReq_9_10 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'hA;
  wire         compressDataVec_hitReq_10_10 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'hA;
  wire         compressDataVec_hitReq_11_10 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'hA;
  wire         compressDataVec_hitReq_12_10 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'hA;
  wire         compressDataVec_hitReq_13_10 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'hA;
  wire         compressDataVec_hitReq_14_10 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'hA;
  wire         compressDataVec_hitReq_15_10 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'hA;
  wire [7:0]   compressDataVec_selectReqData_10 =
    (compressDataVec_hitReq_0_10 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_10 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_10 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_10 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_10 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_10 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_10 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_10 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_10 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_10 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_10 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_10 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_10 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_10 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_10 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_10 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_useTail_10 = tailCount > 4'hA;
  wire         _GEN_92 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'hB;
  wire         compressDataVec_hitReq_0_11;
  assign compressDataVec_hitReq_0_11 = _GEN_92;
  wire         compressDataVec_hitReq_0_43;
  assign compressDataVec_hitReq_0_43 = _GEN_92;
  wire         _GEN_93 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'hB;
  wire         compressDataVec_hitReq_1_11;
  assign compressDataVec_hitReq_1_11 = _GEN_93;
  wire         compressDataVec_hitReq_1_43;
  assign compressDataVec_hitReq_1_43 = _GEN_93;
  wire         _GEN_94 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'hB;
  wire         compressDataVec_hitReq_2_11;
  assign compressDataVec_hitReq_2_11 = _GEN_94;
  wire         compressDataVec_hitReq_2_43;
  assign compressDataVec_hitReq_2_43 = _GEN_94;
  wire         _GEN_95 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'hB;
  wire         compressDataVec_hitReq_3_11;
  assign compressDataVec_hitReq_3_11 = _GEN_95;
  wire         compressDataVec_hitReq_3_43;
  assign compressDataVec_hitReq_3_43 = _GEN_95;
  wire         _GEN_96 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'hB;
  wire         compressDataVec_hitReq_4_11;
  assign compressDataVec_hitReq_4_11 = _GEN_96;
  wire         compressDataVec_hitReq_4_43;
  assign compressDataVec_hitReq_4_43 = _GEN_96;
  wire         _GEN_97 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'hB;
  wire         compressDataVec_hitReq_5_11;
  assign compressDataVec_hitReq_5_11 = _GEN_97;
  wire         compressDataVec_hitReq_5_43;
  assign compressDataVec_hitReq_5_43 = _GEN_97;
  wire         _GEN_98 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'hB;
  wire         compressDataVec_hitReq_6_11;
  assign compressDataVec_hitReq_6_11 = _GEN_98;
  wire         compressDataVec_hitReq_6_43;
  assign compressDataVec_hitReq_6_43 = _GEN_98;
  wire         _GEN_99 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'hB;
  wire         compressDataVec_hitReq_7_11;
  assign compressDataVec_hitReq_7_11 = _GEN_99;
  wire         compressDataVec_hitReq_7_43;
  assign compressDataVec_hitReq_7_43 = _GEN_99;
  wire         compressDataVec_hitReq_8_11 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'hB;
  wire         compressDataVec_hitReq_9_11 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'hB;
  wire         compressDataVec_hitReq_10_11 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'hB;
  wire         compressDataVec_hitReq_11_11 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'hB;
  wire         compressDataVec_hitReq_12_11 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'hB;
  wire         compressDataVec_hitReq_13_11 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'hB;
  wire         compressDataVec_hitReq_14_11 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'hB;
  wire         compressDataVec_hitReq_15_11 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'hB;
  wire [7:0]   compressDataVec_selectReqData_11 =
    (compressDataVec_hitReq_0_11 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_11 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_11 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_11 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_11 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_11 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_11 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_11 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_11 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_11 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_11 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_11 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_11 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_11 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_11 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_11 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_useTail_11 = tailCount > 4'hB;
  wire         _GEN_100 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'hC;
  wire         compressDataVec_hitReq_0_12;
  assign compressDataVec_hitReq_0_12 = _GEN_100;
  wire         compressDataVec_hitReq_0_44;
  assign compressDataVec_hitReq_0_44 = _GEN_100;
  wire         _GEN_101 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'hC;
  wire         compressDataVec_hitReq_1_12;
  assign compressDataVec_hitReq_1_12 = _GEN_101;
  wire         compressDataVec_hitReq_1_44;
  assign compressDataVec_hitReq_1_44 = _GEN_101;
  wire         _GEN_102 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'hC;
  wire         compressDataVec_hitReq_2_12;
  assign compressDataVec_hitReq_2_12 = _GEN_102;
  wire         compressDataVec_hitReq_2_44;
  assign compressDataVec_hitReq_2_44 = _GEN_102;
  wire         _GEN_103 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'hC;
  wire         compressDataVec_hitReq_3_12;
  assign compressDataVec_hitReq_3_12 = _GEN_103;
  wire         compressDataVec_hitReq_3_44;
  assign compressDataVec_hitReq_3_44 = _GEN_103;
  wire         _GEN_104 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'hC;
  wire         compressDataVec_hitReq_4_12;
  assign compressDataVec_hitReq_4_12 = _GEN_104;
  wire         compressDataVec_hitReq_4_44;
  assign compressDataVec_hitReq_4_44 = _GEN_104;
  wire         _GEN_105 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'hC;
  wire         compressDataVec_hitReq_5_12;
  assign compressDataVec_hitReq_5_12 = _GEN_105;
  wire         compressDataVec_hitReq_5_44;
  assign compressDataVec_hitReq_5_44 = _GEN_105;
  wire         _GEN_106 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'hC;
  wire         compressDataVec_hitReq_6_12;
  assign compressDataVec_hitReq_6_12 = _GEN_106;
  wire         compressDataVec_hitReq_6_44;
  assign compressDataVec_hitReq_6_44 = _GEN_106;
  wire         _GEN_107 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'hC;
  wire         compressDataVec_hitReq_7_12;
  assign compressDataVec_hitReq_7_12 = _GEN_107;
  wire         compressDataVec_hitReq_7_44;
  assign compressDataVec_hitReq_7_44 = _GEN_107;
  wire         compressDataVec_hitReq_8_12 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'hC;
  wire         compressDataVec_hitReq_9_12 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'hC;
  wire         compressDataVec_hitReq_10_12 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'hC;
  wire         compressDataVec_hitReq_11_12 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'hC;
  wire         compressDataVec_hitReq_12_12 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'hC;
  wire         compressDataVec_hitReq_13_12 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'hC;
  wire         compressDataVec_hitReq_14_12 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'hC;
  wire         compressDataVec_hitReq_15_12 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'hC;
  wire [7:0]   compressDataVec_selectReqData_12 =
    (compressDataVec_hitReq_0_12 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_12 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_12 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_12 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_12 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_12 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_12 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_12 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_12 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_12 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_12 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_12 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_12 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_12 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_12 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_12 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_useTail_12 = tailCount > 4'hC;
  wire         _GEN_108 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'hD;
  wire         compressDataVec_hitReq_0_13;
  assign compressDataVec_hitReq_0_13 = _GEN_108;
  wire         compressDataVec_hitReq_0_45;
  assign compressDataVec_hitReq_0_45 = _GEN_108;
  wire         _GEN_109 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'hD;
  wire         compressDataVec_hitReq_1_13;
  assign compressDataVec_hitReq_1_13 = _GEN_109;
  wire         compressDataVec_hitReq_1_45;
  assign compressDataVec_hitReq_1_45 = _GEN_109;
  wire         _GEN_110 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'hD;
  wire         compressDataVec_hitReq_2_13;
  assign compressDataVec_hitReq_2_13 = _GEN_110;
  wire         compressDataVec_hitReq_2_45;
  assign compressDataVec_hitReq_2_45 = _GEN_110;
  wire         _GEN_111 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'hD;
  wire         compressDataVec_hitReq_3_13;
  assign compressDataVec_hitReq_3_13 = _GEN_111;
  wire         compressDataVec_hitReq_3_45;
  assign compressDataVec_hitReq_3_45 = _GEN_111;
  wire         _GEN_112 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'hD;
  wire         compressDataVec_hitReq_4_13;
  assign compressDataVec_hitReq_4_13 = _GEN_112;
  wire         compressDataVec_hitReq_4_45;
  assign compressDataVec_hitReq_4_45 = _GEN_112;
  wire         _GEN_113 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'hD;
  wire         compressDataVec_hitReq_5_13;
  assign compressDataVec_hitReq_5_13 = _GEN_113;
  wire         compressDataVec_hitReq_5_45;
  assign compressDataVec_hitReq_5_45 = _GEN_113;
  wire         _GEN_114 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'hD;
  wire         compressDataVec_hitReq_6_13;
  assign compressDataVec_hitReq_6_13 = _GEN_114;
  wire         compressDataVec_hitReq_6_45;
  assign compressDataVec_hitReq_6_45 = _GEN_114;
  wire         _GEN_115 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'hD;
  wire         compressDataVec_hitReq_7_13;
  assign compressDataVec_hitReq_7_13 = _GEN_115;
  wire         compressDataVec_hitReq_7_45;
  assign compressDataVec_hitReq_7_45 = _GEN_115;
  wire         compressDataVec_hitReq_8_13 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'hD;
  wire         compressDataVec_hitReq_9_13 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'hD;
  wire         compressDataVec_hitReq_10_13 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'hD;
  wire         compressDataVec_hitReq_11_13 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'hD;
  wire         compressDataVec_hitReq_12_13 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'hD;
  wire         compressDataVec_hitReq_13_13 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'hD;
  wire         compressDataVec_hitReq_14_13 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'hD;
  wire         compressDataVec_hitReq_15_13 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'hD;
  wire [7:0]   compressDataVec_selectReqData_13 =
    (compressDataVec_hitReq_0_13 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_13 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_13 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_13 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_13 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_13 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_13 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_13 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_13 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_13 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_13 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_13 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_13 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_13 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_13 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_13 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_useTail_13 = tailCount > 4'hD;
  wire         _GEN_116 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'hE;
  wire         compressDataVec_hitReq_0_14;
  assign compressDataVec_hitReq_0_14 = _GEN_116;
  wire         compressDataVec_hitReq_0_46;
  assign compressDataVec_hitReq_0_46 = _GEN_116;
  wire         _GEN_117 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'hE;
  wire         compressDataVec_hitReq_1_14;
  assign compressDataVec_hitReq_1_14 = _GEN_117;
  wire         compressDataVec_hitReq_1_46;
  assign compressDataVec_hitReq_1_46 = _GEN_117;
  wire         _GEN_118 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'hE;
  wire         compressDataVec_hitReq_2_14;
  assign compressDataVec_hitReq_2_14 = _GEN_118;
  wire         compressDataVec_hitReq_2_46;
  assign compressDataVec_hitReq_2_46 = _GEN_118;
  wire         _GEN_119 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'hE;
  wire         compressDataVec_hitReq_3_14;
  assign compressDataVec_hitReq_3_14 = _GEN_119;
  wire         compressDataVec_hitReq_3_46;
  assign compressDataVec_hitReq_3_46 = _GEN_119;
  wire         _GEN_120 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'hE;
  wire         compressDataVec_hitReq_4_14;
  assign compressDataVec_hitReq_4_14 = _GEN_120;
  wire         compressDataVec_hitReq_4_46;
  assign compressDataVec_hitReq_4_46 = _GEN_120;
  wire         _GEN_121 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'hE;
  wire         compressDataVec_hitReq_5_14;
  assign compressDataVec_hitReq_5_14 = _GEN_121;
  wire         compressDataVec_hitReq_5_46;
  assign compressDataVec_hitReq_5_46 = _GEN_121;
  wire         _GEN_122 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'hE;
  wire         compressDataVec_hitReq_6_14;
  assign compressDataVec_hitReq_6_14 = _GEN_122;
  wire         compressDataVec_hitReq_6_46;
  assign compressDataVec_hitReq_6_46 = _GEN_122;
  wire         _GEN_123 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'hE;
  wire         compressDataVec_hitReq_7_14;
  assign compressDataVec_hitReq_7_14 = _GEN_123;
  wire         compressDataVec_hitReq_7_46;
  assign compressDataVec_hitReq_7_46 = _GEN_123;
  wire         compressDataVec_hitReq_8_14 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'hE;
  wire         compressDataVec_hitReq_9_14 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'hE;
  wire         compressDataVec_hitReq_10_14 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'hE;
  wire         compressDataVec_hitReq_11_14 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'hE;
  wire         compressDataVec_hitReq_12_14 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'hE;
  wire         compressDataVec_hitReq_13_14 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'hE;
  wire         compressDataVec_hitReq_14_14 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'hE;
  wire         compressDataVec_hitReq_15_14 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'hE;
  wire [7:0]   compressDataVec_selectReqData_14 =
    (compressDataVec_hitReq_0_14 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_14 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_14 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_14 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_14 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_14 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_14 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_14 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_14 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_14 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_14 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_14 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_14 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_14 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_14 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_14 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_useTail_14 = &tailCount;
  wire         _GEN_124 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'hF;
  wire         compressDataVec_hitReq_0_15;
  assign compressDataVec_hitReq_0_15 = _GEN_124;
  wire         compressDataVec_hitReq_0_47;
  assign compressDataVec_hitReq_0_47 = _GEN_124;
  wire         _GEN_125 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'hF;
  wire         compressDataVec_hitReq_1_15;
  assign compressDataVec_hitReq_1_15 = _GEN_125;
  wire         compressDataVec_hitReq_1_47;
  assign compressDataVec_hitReq_1_47 = _GEN_125;
  wire         _GEN_126 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'hF;
  wire         compressDataVec_hitReq_2_15;
  assign compressDataVec_hitReq_2_15 = _GEN_126;
  wire         compressDataVec_hitReq_2_47;
  assign compressDataVec_hitReq_2_47 = _GEN_126;
  wire         _GEN_127 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'hF;
  wire         compressDataVec_hitReq_3_15;
  assign compressDataVec_hitReq_3_15 = _GEN_127;
  wire         compressDataVec_hitReq_3_47;
  assign compressDataVec_hitReq_3_47 = _GEN_127;
  wire         _GEN_128 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'hF;
  wire         compressDataVec_hitReq_4_15;
  assign compressDataVec_hitReq_4_15 = _GEN_128;
  wire         compressDataVec_hitReq_4_47;
  assign compressDataVec_hitReq_4_47 = _GEN_128;
  wire         _GEN_129 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'hF;
  wire         compressDataVec_hitReq_5_15;
  assign compressDataVec_hitReq_5_15 = _GEN_129;
  wire         compressDataVec_hitReq_5_47;
  assign compressDataVec_hitReq_5_47 = _GEN_129;
  wire         _GEN_130 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'hF;
  wire         compressDataVec_hitReq_6_15;
  assign compressDataVec_hitReq_6_15 = _GEN_130;
  wire         compressDataVec_hitReq_6_47;
  assign compressDataVec_hitReq_6_47 = _GEN_130;
  wire         _GEN_131 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'hF;
  wire         compressDataVec_hitReq_7_15;
  assign compressDataVec_hitReq_7_15 = _GEN_131;
  wire         compressDataVec_hitReq_7_47;
  assign compressDataVec_hitReq_7_47 = _GEN_131;
  wire         compressDataVec_hitReq_8_15 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'hF;
  wire         compressDataVec_hitReq_9_15 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'hF;
  wire         compressDataVec_hitReq_10_15 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'hF;
  wire         compressDataVec_hitReq_11_15 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'hF;
  wire         compressDataVec_hitReq_12_15 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'hF;
  wire         compressDataVec_hitReq_13_15 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'hF;
  wire         compressDataVec_hitReq_14_15 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'hF;
  wire         compressDataVec_hitReq_15_15 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'hF;
  wire [7:0]   compressDataVec_selectReqData_15 =
    (compressDataVec_hitReq_0_15 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_15 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_15 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_15 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_15 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_15 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_15 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_15 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_15 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_15 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_15 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_15 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_15 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_15 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_15 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_15 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_hitReq_0_16 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h10;
  wire         compressDataVec_hitReq_1_16 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h10;
  wire         compressDataVec_hitReq_2_16 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h10;
  wire         compressDataVec_hitReq_3_16 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h10;
  wire         compressDataVec_hitReq_4_16 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h10;
  wire         compressDataVec_hitReq_5_16 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h10;
  wire         compressDataVec_hitReq_6_16 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h10;
  wire         compressDataVec_hitReq_7_16 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h10;
  wire         compressDataVec_hitReq_8_16 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h10;
  wire         compressDataVec_hitReq_9_16 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h10;
  wire         compressDataVec_hitReq_10_16 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h10;
  wire         compressDataVec_hitReq_11_16 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h10;
  wire         compressDataVec_hitReq_12_16 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h10;
  wire         compressDataVec_hitReq_13_16 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h10;
  wire         compressDataVec_hitReq_14_16 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h10;
  wire         compressDataVec_hitReq_15_16 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h10;
  wire [7:0]   compressDataVec_selectReqData_16 =
    (compressDataVec_hitReq_0_16 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_16 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_16 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_16 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_16 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_16 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_16 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_16 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_16 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_16 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_16 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_16 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_16 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_16 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_16 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_16 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_hitReq_0_17 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h11;
  wire         compressDataVec_hitReq_1_17 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h11;
  wire         compressDataVec_hitReq_2_17 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h11;
  wire         compressDataVec_hitReq_3_17 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h11;
  wire         compressDataVec_hitReq_4_17 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h11;
  wire         compressDataVec_hitReq_5_17 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h11;
  wire         compressDataVec_hitReq_6_17 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h11;
  wire         compressDataVec_hitReq_7_17 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h11;
  wire         compressDataVec_hitReq_8_17 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h11;
  wire         compressDataVec_hitReq_9_17 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h11;
  wire         compressDataVec_hitReq_10_17 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h11;
  wire         compressDataVec_hitReq_11_17 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h11;
  wire         compressDataVec_hitReq_12_17 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h11;
  wire         compressDataVec_hitReq_13_17 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h11;
  wire         compressDataVec_hitReq_14_17 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h11;
  wire         compressDataVec_hitReq_15_17 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h11;
  wire [7:0]   compressDataVec_selectReqData_17 =
    (compressDataVec_hitReq_0_17 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_17 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_17 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_17 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_17 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_17 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_17 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_17 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_17 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_17 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_17 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_17 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_17 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_17 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_17 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_17 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_hitReq_0_18 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h12;
  wire         compressDataVec_hitReq_1_18 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h12;
  wire         compressDataVec_hitReq_2_18 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h12;
  wire         compressDataVec_hitReq_3_18 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h12;
  wire         compressDataVec_hitReq_4_18 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h12;
  wire         compressDataVec_hitReq_5_18 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h12;
  wire         compressDataVec_hitReq_6_18 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h12;
  wire         compressDataVec_hitReq_7_18 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h12;
  wire         compressDataVec_hitReq_8_18 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h12;
  wire         compressDataVec_hitReq_9_18 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h12;
  wire         compressDataVec_hitReq_10_18 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h12;
  wire         compressDataVec_hitReq_11_18 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h12;
  wire         compressDataVec_hitReq_12_18 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h12;
  wire         compressDataVec_hitReq_13_18 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h12;
  wire         compressDataVec_hitReq_14_18 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h12;
  wire         compressDataVec_hitReq_15_18 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h12;
  wire [7:0]   compressDataVec_selectReqData_18 =
    (compressDataVec_hitReq_0_18 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_18 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_18 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_18 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_18 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_18 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_18 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_18 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_18 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_18 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_18 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_18 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_18 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_18 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_18 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_18 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_hitReq_0_19 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h13;
  wire         compressDataVec_hitReq_1_19 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h13;
  wire         compressDataVec_hitReq_2_19 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h13;
  wire         compressDataVec_hitReq_3_19 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h13;
  wire         compressDataVec_hitReq_4_19 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h13;
  wire         compressDataVec_hitReq_5_19 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h13;
  wire         compressDataVec_hitReq_6_19 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h13;
  wire         compressDataVec_hitReq_7_19 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h13;
  wire         compressDataVec_hitReq_8_19 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h13;
  wire         compressDataVec_hitReq_9_19 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h13;
  wire         compressDataVec_hitReq_10_19 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h13;
  wire         compressDataVec_hitReq_11_19 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h13;
  wire         compressDataVec_hitReq_12_19 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h13;
  wire         compressDataVec_hitReq_13_19 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h13;
  wire         compressDataVec_hitReq_14_19 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h13;
  wire         compressDataVec_hitReq_15_19 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h13;
  wire [7:0]   compressDataVec_selectReqData_19 =
    (compressDataVec_hitReq_0_19 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_19 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_19 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_19 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_19 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_19 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_19 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_19 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_19 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_19 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_19 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_19 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_19 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_19 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_19 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_19 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_hitReq_0_20 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h14;
  wire         compressDataVec_hitReq_1_20 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h14;
  wire         compressDataVec_hitReq_2_20 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h14;
  wire         compressDataVec_hitReq_3_20 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h14;
  wire         compressDataVec_hitReq_4_20 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h14;
  wire         compressDataVec_hitReq_5_20 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h14;
  wire         compressDataVec_hitReq_6_20 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h14;
  wire         compressDataVec_hitReq_7_20 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h14;
  wire         compressDataVec_hitReq_8_20 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h14;
  wire         compressDataVec_hitReq_9_20 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h14;
  wire         compressDataVec_hitReq_10_20 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h14;
  wire         compressDataVec_hitReq_11_20 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h14;
  wire         compressDataVec_hitReq_12_20 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h14;
  wire         compressDataVec_hitReq_13_20 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h14;
  wire         compressDataVec_hitReq_14_20 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h14;
  wire         compressDataVec_hitReq_15_20 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h14;
  wire [7:0]   compressDataVec_selectReqData_20 =
    (compressDataVec_hitReq_0_20 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_20 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_20 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_20 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_20 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_20 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_20 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_20 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_20 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_20 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_20 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_20 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_20 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_20 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_20 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_20 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_hitReq_0_21 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h15;
  wire         compressDataVec_hitReq_1_21 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h15;
  wire         compressDataVec_hitReq_2_21 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h15;
  wire         compressDataVec_hitReq_3_21 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h15;
  wire         compressDataVec_hitReq_4_21 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h15;
  wire         compressDataVec_hitReq_5_21 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h15;
  wire         compressDataVec_hitReq_6_21 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h15;
  wire         compressDataVec_hitReq_7_21 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h15;
  wire         compressDataVec_hitReq_8_21 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h15;
  wire         compressDataVec_hitReq_9_21 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h15;
  wire         compressDataVec_hitReq_10_21 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h15;
  wire         compressDataVec_hitReq_11_21 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h15;
  wire         compressDataVec_hitReq_12_21 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h15;
  wire         compressDataVec_hitReq_13_21 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h15;
  wire         compressDataVec_hitReq_14_21 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h15;
  wire         compressDataVec_hitReq_15_21 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h15;
  wire [7:0]   compressDataVec_selectReqData_21 =
    (compressDataVec_hitReq_0_21 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_21 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_21 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_21 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_21 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_21 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_21 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_21 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_21 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_21 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_21 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_21 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_21 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_21 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_21 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_21 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_hitReq_0_22 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h16;
  wire         compressDataVec_hitReq_1_22 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h16;
  wire         compressDataVec_hitReq_2_22 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h16;
  wire         compressDataVec_hitReq_3_22 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h16;
  wire         compressDataVec_hitReq_4_22 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h16;
  wire         compressDataVec_hitReq_5_22 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h16;
  wire         compressDataVec_hitReq_6_22 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h16;
  wire         compressDataVec_hitReq_7_22 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h16;
  wire         compressDataVec_hitReq_8_22 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h16;
  wire         compressDataVec_hitReq_9_22 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h16;
  wire         compressDataVec_hitReq_10_22 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h16;
  wire         compressDataVec_hitReq_11_22 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h16;
  wire         compressDataVec_hitReq_12_22 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h16;
  wire         compressDataVec_hitReq_13_22 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h16;
  wire         compressDataVec_hitReq_14_22 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h16;
  wire         compressDataVec_hitReq_15_22 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h16;
  wire [7:0]   compressDataVec_selectReqData_22 =
    (compressDataVec_hitReq_0_22 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_22 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_22 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_22 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_22 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_22 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_22 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_22 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_22 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_22 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_22 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_22 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_22 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_22 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_22 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_22 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_hitReq_0_23 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h17;
  wire         compressDataVec_hitReq_1_23 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h17;
  wire         compressDataVec_hitReq_2_23 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h17;
  wire         compressDataVec_hitReq_3_23 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h17;
  wire         compressDataVec_hitReq_4_23 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h17;
  wire         compressDataVec_hitReq_5_23 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h17;
  wire         compressDataVec_hitReq_6_23 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h17;
  wire         compressDataVec_hitReq_7_23 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h17;
  wire         compressDataVec_hitReq_8_23 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h17;
  wire         compressDataVec_hitReq_9_23 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h17;
  wire         compressDataVec_hitReq_10_23 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h17;
  wire         compressDataVec_hitReq_11_23 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h17;
  wire         compressDataVec_hitReq_12_23 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h17;
  wire         compressDataVec_hitReq_13_23 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h17;
  wire         compressDataVec_hitReq_14_23 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h17;
  wire         compressDataVec_hitReq_15_23 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h17;
  wire [7:0]   compressDataVec_selectReqData_23 =
    (compressDataVec_hitReq_0_23 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_23 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_23 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_23 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_23 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_23 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_23 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_23 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_23 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_23 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_23 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_23 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_23 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_23 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_23 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_23 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_hitReq_0_24 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h18;
  wire         compressDataVec_hitReq_1_24 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h18;
  wire         compressDataVec_hitReq_2_24 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h18;
  wire         compressDataVec_hitReq_3_24 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h18;
  wire         compressDataVec_hitReq_4_24 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h18;
  wire         compressDataVec_hitReq_5_24 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h18;
  wire         compressDataVec_hitReq_6_24 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h18;
  wire         compressDataVec_hitReq_7_24 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h18;
  wire         compressDataVec_hitReq_8_24 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h18;
  wire         compressDataVec_hitReq_9_24 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h18;
  wire         compressDataVec_hitReq_10_24 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h18;
  wire         compressDataVec_hitReq_11_24 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h18;
  wire         compressDataVec_hitReq_12_24 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h18;
  wire         compressDataVec_hitReq_13_24 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h18;
  wire         compressDataVec_hitReq_14_24 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h18;
  wire         compressDataVec_hitReq_15_24 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h18;
  wire [7:0]   compressDataVec_selectReqData_24 =
    (compressDataVec_hitReq_0_24 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_24 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_24 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_24 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_24 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_24 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_24 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_24 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_24 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_24 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_24 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_24 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_24 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_24 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_24 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_24 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_hitReq_0_25 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h19;
  wire         compressDataVec_hitReq_1_25 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h19;
  wire         compressDataVec_hitReq_2_25 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h19;
  wire         compressDataVec_hitReq_3_25 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h19;
  wire         compressDataVec_hitReq_4_25 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h19;
  wire         compressDataVec_hitReq_5_25 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h19;
  wire         compressDataVec_hitReq_6_25 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h19;
  wire         compressDataVec_hitReq_7_25 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h19;
  wire         compressDataVec_hitReq_8_25 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h19;
  wire         compressDataVec_hitReq_9_25 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h19;
  wire         compressDataVec_hitReq_10_25 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h19;
  wire         compressDataVec_hitReq_11_25 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h19;
  wire         compressDataVec_hitReq_12_25 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h19;
  wire         compressDataVec_hitReq_13_25 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h19;
  wire         compressDataVec_hitReq_14_25 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h19;
  wire         compressDataVec_hitReq_15_25 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h19;
  wire [7:0]   compressDataVec_selectReqData_25 =
    (compressDataVec_hitReq_0_25 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_25 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_25 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_25 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_25 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_25 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_25 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_25 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_25 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_25 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_25 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_25 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_25 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_25 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_25 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_25 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_hitReq_0_26 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h1A;
  wire         compressDataVec_hitReq_1_26 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h1A;
  wire         compressDataVec_hitReq_2_26 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h1A;
  wire         compressDataVec_hitReq_3_26 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h1A;
  wire         compressDataVec_hitReq_4_26 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h1A;
  wire         compressDataVec_hitReq_5_26 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h1A;
  wire         compressDataVec_hitReq_6_26 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h1A;
  wire         compressDataVec_hitReq_7_26 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h1A;
  wire         compressDataVec_hitReq_8_26 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h1A;
  wire         compressDataVec_hitReq_9_26 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h1A;
  wire         compressDataVec_hitReq_10_26 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h1A;
  wire         compressDataVec_hitReq_11_26 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h1A;
  wire         compressDataVec_hitReq_12_26 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h1A;
  wire         compressDataVec_hitReq_13_26 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h1A;
  wire         compressDataVec_hitReq_14_26 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h1A;
  wire         compressDataVec_hitReq_15_26 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h1A;
  wire [7:0]   compressDataVec_selectReqData_26 =
    (compressDataVec_hitReq_0_26 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_26 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_26 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_26 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_26 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_26 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_26 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_26 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_26 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_26 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_26 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_26 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_26 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_26 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_26 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_26 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_hitReq_0_27 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h1B;
  wire         compressDataVec_hitReq_1_27 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h1B;
  wire         compressDataVec_hitReq_2_27 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h1B;
  wire         compressDataVec_hitReq_3_27 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h1B;
  wire         compressDataVec_hitReq_4_27 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h1B;
  wire         compressDataVec_hitReq_5_27 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h1B;
  wire         compressDataVec_hitReq_6_27 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h1B;
  wire         compressDataVec_hitReq_7_27 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h1B;
  wire         compressDataVec_hitReq_8_27 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h1B;
  wire         compressDataVec_hitReq_9_27 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h1B;
  wire         compressDataVec_hitReq_10_27 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h1B;
  wire         compressDataVec_hitReq_11_27 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h1B;
  wire         compressDataVec_hitReq_12_27 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h1B;
  wire         compressDataVec_hitReq_13_27 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h1B;
  wire         compressDataVec_hitReq_14_27 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h1B;
  wire         compressDataVec_hitReq_15_27 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h1B;
  wire [7:0]   compressDataVec_selectReqData_27 =
    (compressDataVec_hitReq_0_27 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_27 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_27 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_27 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_27 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_27 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_27 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_27 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_27 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_27 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_27 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_27 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_27 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_27 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_27 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_27 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_hitReq_0_28 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h1C;
  wire         compressDataVec_hitReq_1_28 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h1C;
  wire         compressDataVec_hitReq_2_28 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h1C;
  wire         compressDataVec_hitReq_3_28 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h1C;
  wire         compressDataVec_hitReq_4_28 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h1C;
  wire         compressDataVec_hitReq_5_28 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h1C;
  wire         compressDataVec_hitReq_6_28 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h1C;
  wire         compressDataVec_hitReq_7_28 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h1C;
  wire         compressDataVec_hitReq_8_28 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h1C;
  wire         compressDataVec_hitReq_9_28 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h1C;
  wire         compressDataVec_hitReq_10_28 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h1C;
  wire         compressDataVec_hitReq_11_28 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h1C;
  wire         compressDataVec_hitReq_12_28 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h1C;
  wire         compressDataVec_hitReq_13_28 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h1C;
  wire         compressDataVec_hitReq_14_28 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h1C;
  wire         compressDataVec_hitReq_15_28 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h1C;
  wire [7:0]   compressDataVec_selectReqData_28 =
    (compressDataVec_hitReq_0_28 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_28 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_28 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_28 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_28 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_28 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_28 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_28 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_28 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_28 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_28 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_28 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_28 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_28 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_28 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_28 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_hitReq_0_29 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h1D;
  wire         compressDataVec_hitReq_1_29 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h1D;
  wire         compressDataVec_hitReq_2_29 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h1D;
  wire         compressDataVec_hitReq_3_29 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h1D;
  wire         compressDataVec_hitReq_4_29 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h1D;
  wire         compressDataVec_hitReq_5_29 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h1D;
  wire         compressDataVec_hitReq_6_29 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h1D;
  wire         compressDataVec_hitReq_7_29 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h1D;
  wire         compressDataVec_hitReq_8_29 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h1D;
  wire         compressDataVec_hitReq_9_29 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h1D;
  wire         compressDataVec_hitReq_10_29 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h1D;
  wire         compressDataVec_hitReq_11_29 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h1D;
  wire         compressDataVec_hitReq_12_29 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h1D;
  wire         compressDataVec_hitReq_13_29 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h1D;
  wire         compressDataVec_hitReq_14_29 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h1D;
  wire         compressDataVec_hitReq_15_29 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h1D;
  wire [7:0]   compressDataVec_selectReqData_29 =
    (compressDataVec_hitReq_0_29 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_29 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_29 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_29 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_29 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_29 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_29 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_29 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_29 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_29 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_29 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_29 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_29 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_29 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_29 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_29 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_hitReq_0_30 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h1E;
  wire         compressDataVec_hitReq_1_30 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h1E;
  wire         compressDataVec_hitReq_2_30 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h1E;
  wire         compressDataVec_hitReq_3_30 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h1E;
  wire         compressDataVec_hitReq_4_30 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h1E;
  wire         compressDataVec_hitReq_5_30 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h1E;
  wire         compressDataVec_hitReq_6_30 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h1E;
  wire         compressDataVec_hitReq_7_30 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h1E;
  wire         compressDataVec_hitReq_8_30 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h1E;
  wire         compressDataVec_hitReq_9_30 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h1E;
  wire         compressDataVec_hitReq_10_30 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h1E;
  wire         compressDataVec_hitReq_11_30 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h1E;
  wire         compressDataVec_hitReq_12_30 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h1E;
  wire         compressDataVec_hitReq_13_30 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h1E;
  wire         compressDataVec_hitReq_14_30 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h1E;
  wire         compressDataVec_hitReq_15_30 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h1E;
  wire [7:0]   compressDataVec_selectReqData_30 =
    (compressDataVec_hitReq_0_30 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_30 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_30 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_30 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_30 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_30 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_30 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_30 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_30 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_30 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_30 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_30 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_30 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_30 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_30 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_30 ? source2Pipe[127:120] : 8'h0);
  wire         compressDataVec_hitReq_0_31 = compressMaskVecPipe_0 & compressVecPipe_0 == 14'h1F;
  wire         compressDataVec_hitReq_1_31 = compressMaskVecPipe_1 & compressVecPipe_1 == 14'h1F;
  wire         compressDataVec_hitReq_2_31 = compressMaskVecPipe_2 & compressVecPipe_2 == 14'h1F;
  wire         compressDataVec_hitReq_3_31 = compressMaskVecPipe_3 & compressVecPipe_3 == 14'h1F;
  wire         compressDataVec_hitReq_4_31 = compressMaskVecPipe_4 & compressVecPipe_4 == 14'h1F;
  wire         compressDataVec_hitReq_5_31 = compressMaskVecPipe_5 & compressVecPipe_5 == 14'h1F;
  wire         compressDataVec_hitReq_6_31 = compressMaskVecPipe_6 & compressVecPipe_6 == 14'h1F;
  wire         compressDataVec_hitReq_7_31 = compressMaskVecPipe_7 & compressVecPipe_7 == 14'h1F;
  wire         compressDataVec_hitReq_8_31 = compressMaskVecPipe_8 & compressVecPipe_8 == 14'h1F;
  wire         compressDataVec_hitReq_9_31 = compressMaskVecPipe_9 & compressVecPipe_9 == 14'h1F;
  wire         compressDataVec_hitReq_10_31 = compressMaskVecPipe_10 & compressVecPipe_10 == 14'h1F;
  wire         compressDataVec_hitReq_11_31 = compressMaskVecPipe_11 & compressVecPipe_11 == 14'h1F;
  wire         compressDataVec_hitReq_12_31 = compressMaskVecPipe_12 & compressVecPipe_12 == 14'h1F;
  wire         compressDataVec_hitReq_13_31 = compressMaskVecPipe_13 & compressVecPipe_13 == 14'h1F;
  wire         compressDataVec_hitReq_14_31 = compressMaskVecPipe_14 & compressVecPipe_14 == 14'h1F;
  wire         compressDataVec_hitReq_15_31 = compressMaskVecPipe_15 & compressVecPipe_15 == 14'h1F;
  wire [7:0]   compressDataVec_selectReqData_31 =
    (compressDataVec_hitReq_0_31 ? source2Pipe[7:0] : 8'h0) | (compressDataVec_hitReq_1_31 ? source2Pipe[15:8] : 8'h0) | (compressDataVec_hitReq_2_31 ? source2Pipe[23:16] : 8'h0) | (compressDataVec_hitReq_3_31 ? source2Pipe[31:24] : 8'h0)
    | (compressDataVec_hitReq_4_31 ? source2Pipe[39:32] : 8'h0) | (compressDataVec_hitReq_5_31 ? source2Pipe[47:40] : 8'h0) | (compressDataVec_hitReq_6_31 ? source2Pipe[55:48] : 8'h0)
    | (compressDataVec_hitReq_7_31 ? source2Pipe[63:56] : 8'h0) | (compressDataVec_hitReq_8_31 ? source2Pipe[71:64] : 8'h0) | (compressDataVec_hitReq_9_31 ? source2Pipe[79:72] : 8'h0)
    | (compressDataVec_hitReq_10_31 ? source2Pipe[87:80] : 8'h0) | (compressDataVec_hitReq_11_31 ? source2Pipe[95:88] : 8'h0) | (compressDataVec_hitReq_12_31 ? source2Pipe[103:96] : 8'h0)
    | (compressDataVec_hitReq_13_31 ? source2Pipe[111:104] : 8'h0) | (compressDataVec_hitReq_14_31 ? source2Pipe[119:112] : 8'h0) | (compressDataVec_hitReq_15_31 ? source2Pipe[127:120] : 8'h0);
  wire [15:0]  compressDataVec_lo_lo_lo_lo = {compressDataVec_useTail_1 ? compressDataReg[15:8] : compressDataVec_selectReqData_1, compressDataVec_useTail ? compressDataReg[7:0] : compressDataVec_selectReqData};
  wire [15:0]  compressDataVec_lo_lo_lo_hi = {compressDataVec_useTail_3 ? compressDataReg[31:24] : compressDataVec_selectReqData_3, compressDataVec_useTail_2 ? compressDataReg[23:16] : compressDataVec_selectReqData_2};
  wire [31:0]  compressDataVec_lo_lo_lo = {compressDataVec_lo_lo_lo_hi, compressDataVec_lo_lo_lo_lo};
  wire [15:0]  compressDataVec_lo_lo_hi_lo = {compressDataVec_useTail_5 ? compressDataReg[47:40] : compressDataVec_selectReqData_5, compressDataVec_useTail_4 ? compressDataReg[39:32] : compressDataVec_selectReqData_4};
  wire [15:0]  compressDataVec_lo_lo_hi_hi = {compressDataVec_useTail_7 ? compressDataReg[63:56] : compressDataVec_selectReqData_7, compressDataVec_useTail_6 ? compressDataReg[55:48] : compressDataVec_selectReqData_6};
  wire [31:0]  compressDataVec_lo_lo_hi = {compressDataVec_lo_lo_hi_hi, compressDataVec_lo_lo_hi_lo};
  wire [63:0]  compressDataVec_lo_lo = {compressDataVec_lo_lo_hi, compressDataVec_lo_lo_lo};
  wire [15:0]  compressDataVec_lo_hi_lo_lo = {compressDataVec_useTail_9 ? compressDataReg[79:72] : compressDataVec_selectReqData_9, compressDataVec_useTail_8 ? compressDataReg[71:64] : compressDataVec_selectReqData_8};
  wire [15:0]  compressDataVec_lo_hi_lo_hi = {compressDataVec_useTail_11 ? compressDataReg[95:88] : compressDataVec_selectReqData_11, compressDataVec_useTail_10 ? compressDataReg[87:80] : compressDataVec_selectReqData_10};
  wire [31:0]  compressDataVec_lo_hi_lo = {compressDataVec_lo_hi_lo_hi, compressDataVec_lo_hi_lo_lo};
  wire [15:0]  compressDataVec_lo_hi_hi_lo = {compressDataVec_useTail_13 ? compressDataReg[111:104] : compressDataVec_selectReqData_13, compressDataVec_useTail_12 ? compressDataReg[103:96] : compressDataVec_selectReqData_12};
  wire [15:0]  compressDataVec_lo_hi_hi_hi = {compressDataVec_selectReqData_15, compressDataVec_useTail_14 ? compressDataReg[119:112] : compressDataVec_selectReqData_14};
  wire [31:0]  compressDataVec_lo_hi_hi = {compressDataVec_lo_hi_hi_hi, compressDataVec_lo_hi_hi_lo};
  wire [63:0]  compressDataVec_lo_hi = {compressDataVec_lo_hi_hi, compressDataVec_lo_hi_lo};
  wire [127:0] compressDataVec_lo = {compressDataVec_lo_hi, compressDataVec_lo_lo};
  wire [15:0]  compressDataVec_hi_lo_lo_lo = {compressDataVec_selectReqData_17, compressDataVec_selectReqData_16};
  wire [15:0]  compressDataVec_hi_lo_lo_hi = {compressDataVec_selectReqData_19, compressDataVec_selectReqData_18};
  wire [31:0]  compressDataVec_hi_lo_lo = {compressDataVec_hi_lo_lo_hi, compressDataVec_hi_lo_lo_lo};
  wire [15:0]  compressDataVec_hi_lo_hi_lo = {compressDataVec_selectReqData_21, compressDataVec_selectReqData_20};
  wire [15:0]  compressDataVec_hi_lo_hi_hi = {compressDataVec_selectReqData_23, compressDataVec_selectReqData_22};
  wire [31:0]  compressDataVec_hi_lo_hi = {compressDataVec_hi_lo_hi_hi, compressDataVec_hi_lo_hi_lo};
  wire [63:0]  compressDataVec_hi_lo = {compressDataVec_hi_lo_hi, compressDataVec_hi_lo_lo};
  wire [15:0]  compressDataVec_hi_hi_lo_lo = {compressDataVec_selectReqData_25, compressDataVec_selectReqData_24};
  wire [15:0]  compressDataVec_hi_hi_lo_hi = {compressDataVec_selectReqData_27, compressDataVec_selectReqData_26};
  wire [31:0]  compressDataVec_hi_hi_lo = {compressDataVec_hi_hi_lo_hi, compressDataVec_hi_hi_lo_lo};
  wire [15:0]  compressDataVec_hi_hi_hi_lo = {compressDataVec_selectReqData_29, compressDataVec_selectReqData_28};
  wire [15:0]  compressDataVec_hi_hi_hi_hi = {compressDataVec_selectReqData_31, compressDataVec_selectReqData_30};
  wire [31:0]  compressDataVec_hi_hi_hi = {compressDataVec_hi_hi_hi_hi, compressDataVec_hi_hi_hi_lo};
  wire [63:0]  compressDataVec_hi_hi = {compressDataVec_hi_hi_hi, compressDataVec_hi_hi_lo};
  wire [127:0] compressDataVec_hi = {compressDataVec_hi_hi, compressDataVec_hi_lo};
  wire [255:0] compressDataVec_0 = {compressDataVec_hi, compressDataVec_lo};
  wire [15:0]  compressDataVec_selectReqData_32 =
    (compressDataVec_hitReq_0_32 ? source2Pipe[15:0] : 16'h0) | (compressDataVec_hitReq_1_32 ? source2Pipe[31:16] : 16'h0) | (compressDataVec_hitReq_2_32 ? source2Pipe[47:32] : 16'h0)
    | (compressDataVec_hitReq_3_32 ? source2Pipe[63:48] : 16'h0) | (compressDataVec_hitReq_4_32 ? source2Pipe[79:64] : 16'h0) | (compressDataVec_hitReq_5_32 ? source2Pipe[95:80] : 16'h0)
    | (compressDataVec_hitReq_6_32 ? source2Pipe[111:96] : 16'h0) | (compressDataVec_hitReq_7_32 ? source2Pipe[127:112] : 16'h0);
  wire         compressDataVec_useTail_16;
  assign compressDataVec_useTail_16 = |tailCount;
  wire [15:0]  compressDataVec_selectReqData_33 =
    (compressDataVec_hitReq_0_33 ? source2Pipe[15:0] : 16'h0) | (compressDataVec_hitReq_1_33 ? source2Pipe[31:16] : 16'h0) | (compressDataVec_hitReq_2_33 ? source2Pipe[47:32] : 16'h0)
    | (compressDataVec_hitReq_3_33 ? source2Pipe[63:48] : 16'h0) | (compressDataVec_hitReq_4_33 ? source2Pipe[79:64] : 16'h0) | (compressDataVec_hitReq_5_33 ? source2Pipe[95:80] : 16'h0)
    | (compressDataVec_hitReq_6_33 ? source2Pipe[111:96] : 16'h0) | (compressDataVec_hitReq_7_33 ? source2Pipe[127:112] : 16'h0);
  wire         compressDataVec_useTail_17;
  assign compressDataVec_useTail_17 = |(tailCount[3:1]);
  wire [15:0]  compressDataVec_selectReqData_34 =
    (compressDataVec_hitReq_0_34 ? source2Pipe[15:0] : 16'h0) | (compressDataVec_hitReq_1_34 ? source2Pipe[31:16] : 16'h0) | (compressDataVec_hitReq_2_34 ? source2Pipe[47:32] : 16'h0)
    | (compressDataVec_hitReq_3_34 ? source2Pipe[63:48] : 16'h0) | (compressDataVec_hitReq_4_34 ? source2Pipe[79:64] : 16'h0) | (compressDataVec_hitReq_5_34 ? source2Pipe[95:80] : 16'h0)
    | (compressDataVec_hitReq_6_34 ? source2Pipe[111:96] : 16'h0) | (compressDataVec_hitReq_7_34 ? source2Pipe[127:112] : 16'h0);
  wire [15:0]  compressDataVec_selectReqData_35 =
    (compressDataVec_hitReq_0_35 ? source2Pipe[15:0] : 16'h0) | (compressDataVec_hitReq_1_35 ? source2Pipe[31:16] : 16'h0) | (compressDataVec_hitReq_2_35 ? source2Pipe[47:32] : 16'h0)
    | (compressDataVec_hitReq_3_35 ? source2Pipe[63:48] : 16'h0) | (compressDataVec_hitReq_4_35 ? source2Pipe[79:64] : 16'h0) | (compressDataVec_hitReq_5_35 ? source2Pipe[95:80] : 16'h0)
    | (compressDataVec_hitReq_6_35 ? source2Pipe[111:96] : 16'h0) | (compressDataVec_hitReq_7_35 ? source2Pipe[127:112] : 16'h0);
  wire         compressDataVec_useTail_19;
  assign compressDataVec_useTail_19 = |(tailCount[3:2]);
  wire [15:0]  compressDataVec_selectReqData_36 =
    (compressDataVec_hitReq_0_36 ? source2Pipe[15:0] : 16'h0) | (compressDataVec_hitReq_1_36 ? source2Pipe[31:16] : 16'h0) | (compressDataVec_hitReq_2_36 ? source2Pipe[47:32] : 16'h0)
    | (compressDataVec_hitReq_3_36 ? source2Pipe[63:48] : 16'h0) | (compressDataVec_hitReq_4_36 ? source2Pipe[79:64] : 16'h0) | (compressDataVec_hitReq_5_36 ? source2Pipe[95:80] : 16'h0)
    | (compressDataVec_hitReq_6_36 ? source2Pipe[111:96] : 16'h0) | (compressDataVec_hitReq_7_36 ? source2Pipe[127:112] : 16'h0);
  wire [15:0]  compressDataVec_selectReqData_37 =
    (compressDataVec_hitReq_0_37 ? source2Pipe[15:0] : 16'h0) | (compressDataVec_hitReq_1_37 ? source2Pipe[31:16] : 16'h0) | (compressDataVec_hitReq_2_37 ? source2Pipe[47:32] : 16'h0)
    | (compressDataVec_hitReq_3_37 ? source2Pipe[63:48] : 16'h0) | (compressDataVec_hitReq_4_37 ? source2Pipe[79:64] : 16'h0) | (compressDataVec_hitReq_5_37 ? source2Pipe[95:80] : 16'h0)
    | (compressDataVec_hitReq_6_37 ? source2Pipe[111:96] : 16'h0) | (compressDataVec_hitReq_7_37 ? source2Pipe[127:112] : 16'h0);
  wire [15:0]  compressDataVec_selectReqData_38 =
    (compressDataVec_hitReq_0_38 ? source2Pipe[15:0] : 16'h0) | (compressDataVec_hitReq_1_38 ? source2Pipe[31:16] : 16'h0) | (compressDataVec_hitReq_2_38 ? source2Pipe[47:32] : 16'h0)
    | (compressDataVec_hitReq_3_38 ? source2Pipe[63:48] : 16'h0) | (compressDataVec_hitReq_4_38 ? source2Pipe[79:64] : 16'h0) | (compressDataVec_hitReq_5_38 ? source2Pipe[95:80] : 16'h0)
    | (compressDataVec_hitReq_6_38 ? source2Pipe[111:96] : 16'h0) | (compressDataVec_hitReq_7_38 ? source2Pipe[127:112] : 16'h0);
  wire [15:0]  compressDataVec_selectReqData_39 =
    (compressDataVec_hitReq_0_39 ? source2Pipe[15:0] : 16'h0) | (compressDataVec_hitReq_1_39 ? source2Pipe[31:16] : 16'h0) | (compressDataVec_hitReq_2_39 ? source2Pipe[47:32] : 16'h0)
    | (compressDataVec_hitReq_3_39 ? source2Pipe[63:48] : 16'h0) | (compressDataVec_hitReq_4_39 ? source2Pipe[79:64] : 16'h0) | (compressDataVec_hitReq_5_39 ? source2Pipe[95:80] : 16'h0)
    | (compressDataVec_hitReq_6_39 ? source2Pipe[111:96] : 16'h0) | (compressDataVec_hitReq_7_39 ? source2Pipe[127:112] : 16'h0);
  wire [15:0]  compressDataVec_selectReqData_40 =
    (compressDataVec_hitReq_0_40 ? source2Pipe[15:0] : 16'h0) | (compressDataVec_hitReq_1_40 ? source2Pipe[31:16] : 16'h0) | (compressDataVec_hitReq_2_40 ? source2Pipe[47:32] : 16'h0)
    | (compressDataVec_hitReq_3_40 ? source2Pipe[63:48] : 16'h0) | (compressDataVec_hitReq_4_40 ? source2Pipe[79:64] : 16'h0) | (compressDataVec_hitReq_5_40 ? source2Pipe[95:80] : 16'h0)
    | (compressDataVec_hitReq_6_40 ? source2Pipe[111:96] : 16'h0) | (compressDataVec_hitReq_7_40 ? source2Pipe[127:112] : 16'h0);
  wire [15:0]  compressDataVec_selectReqData_41 =
    (compressDataVec_hitReq_0_41 ? source2Pipe[15:0] : 16'h0) | (compressDataVec_hitReq_1_41 ? source2Pipe[31:16] : 16'h0) | (compressDataVec_hitReq_2_41 ? source2Pipe[47:32] : 16'h0)
    | (compressDataVec_hitReq_3_41 ? source2Pipe[63:48] : 16'h0) | (compressDataVec_hitReq_4_41 ? source2Pipe[79:64] : 16'h0) | (compressDataVec_hitReq_5_41 ? source2Pipe[95:80] : 16'h0)
    | (compressDataVec_hitReq_6_41 ? source2Pipe[111:96] : 16'h0) | (compressDataVec_hitReq_7_41 ? source2Pipe[127:112] : 16'h0);
  wire [15:0]  compressDataVec_selectReqData_42 =
    (compressDataVec_hitReq_0_42 ? source2Pipe[15:0] : 16'h0) | (compressDataVec_hitReq_1_42 ? source2Pipe[31:16] : 16'h0) | (compressDataVec_hitReq_2_42 ? source2Pipe[47:32] : 16'h0)
    | (compressDataVec_hitReq_3_42 ? source2Pipe[63:48] : 16'h0) | (compressDataVec_hitReq_4_42 ? source2Pipe[79:64] : 16'h0) | (compressDataVec_hitReq_5_42 ? source2Pipe[95:80] : 16'h0)
    | (compressDataVec_hitReq_6_42 ? source2Pipe[111:96] : 16'h0) | (compressDataVec_hitReq_7_42 ? source2Pipe[127:112] : 16'h0);
  wire [15:0]  compressDataVec_selectReqData_43 =
    (compressDataVec_hitReq_0_43 ? source2Pipe[15:0] : 16'h0) | (compressDataVec_hitReq_1_43 ? source2Pipe[31:16] : 16'h0) | (compressDataVec_hitReq_2_43 ? source2Pipe[47:32] : 16'h0)
    | (compressDataVec_hitReq_3_43 ? source2Pipe[63:48] : 16'h0) | (compressDataVec_hitReq_4_43 ? source2Pipe[79:64] : 16'h0) | (compressDataVec_hitReq_5_43 ? source2Pipe[95:80] : 16'h0)
    | (compressDataVec_hitReq_6_43 ? source2Pipe[111:96] : 16'h0) | (compressDataVec_hitReq_7_43 ? source2Pipe[127:112] : 16'h0);
  wire [15:0]  compressDataVec_selectReqData_44 =
    (compressDataVec_hitReq_0_44 ? source2Pipe[15:0] : 16'h0) | (compressDataVec_hitReq_1_44 ? source2Pipe[31:16] : 16'h0) | (compressDataVec_hitReq_2_44 ? source2Pipe[47:32] : 16'h0)
    | (compressDataVec_hitReq_3_44 ? source2Pipe[63:48] : 16'h0) | (compressDataVec_hitReq_4_44 ? source2Pipe[79:64] : 16'h0) | (compressDataVec_hitReq_5_44 ? source2Pipe[95:80] : 16'h0)
    | (compressDataVec_hitReq_6_44 ? source2Pipe[111:96] : 16'h0) | (compressDataVec_hitReq_7_44 ? source2Pipe[127:112] : 16'h0);
  wire [15:0]  compressDataVec_selectReqData_45 =
    (compressDataVec_hitReq_0_45 ? source2Pipe[15:0] : 16'h0) | (compressDataVec_hitReq_1_45 ? source2Pipe[31:16] : 16'h0) | (compressDataVec_hitReq_2_45 ? source2Pipe[47:32] : 16'h0)
    | (compressDataVec_hitReq_3_45 ? source2Pipe[63:48] : 16'h0) | (compressDataVec_hitReq_4_45 ? source2Pipe[79:64] : 16'h0) | (compressDataVec_hitReq_5_45 ? source2Pipe[95:80] : 16'h0)
    | (compressDataVec_hitReq_6_45 ? source2Pipe[111:96] : 16'h0) | (compressDataVec_hitReq_7_45 ? source2Pipe[127:112] : 16'h0);
  wire [15:0]  compressDataVec_selectReqData_46 =
    (compressDataVec_hitReq_0_46 ? source2Pipe[15:0] : 16'h0) | (compressDataVec_hitReq_1_46 ? source2Pipe[31:16] : 16'h0) | (compressDataVec_hitReq_2_46 ? source2Pipe[47:32] : 16'h0)
    | (compressDataVec_hitReq_3_46 ? source2Pipe[63:48] : 16'h0) | (compressDataVec_hitReq_4_46 ? source2Pipe[79:64] : 16'h0) | (compressDataVec_hitReq_5_46 ? source2Pipe[95:80] : 16'h0)
    | (compressDataVec_hitReq_6_46 ? source2Pipe[111:96] : 16'h0) | (compressDataVec_hitReq_7_46 ? source2Pipe[127:112] : 16'h0);
  wire [15:0]  compressDataVec_selectReqData_47 =
    (compressDataVec_hitReq_0_47 ? source2Pipe[15:0] : 16'h0) | (compressDataVec_hitReq_1_47 ? source2Pipe[31:16] : 16'h0) | (compressDataVec_hitReq_2_47 ? source2Pipe[47:32] : 16'h0)
    | (compressDataVec_hitReq_3_47 ? source2Pipe[63:48] : 16'h0) | (compressDataVec_hitReq_4_47 ? source2Pipe[79:64] : 16'h0) | (compressDataVec_hitReq_5_47 ? source2Pipe[95:80] : 16'h0)
    | (compressDataVec_hitReq_6_47 ? source2Pipe[111:96] : 16'h0) | (compressDataVec_hitReq_7_47 ? source2Pipe[127:112] : 16'h0);
  wire [31:0]  compressDataVec_lo_lo_lo_1 = {compressDataVec_useTail_17 ? compressDataReg[31:16] : compressDataVec_selectReqData_33, compressDataVec_useTail_16 ? compressDataReg[15:0] : compressDataVec_selectReqData_32};
  wire [31:0]  compressDataVec_lo_lo_hi_1 = {compressDataVec_useTail_19 ? compressDataReg[63:48] : compressDataVec_selectReqData_35, compressDataVec_useTail_18 ? compressDataReg[47:32] : compressDataVec_selectReqData_34};
  wire [63:0]  compressDataVec_lo_lo_1 = {compressDataVec_lo_lo_hi_1, compressDataVec_lo_lo_lo_1};
  wire [31:0]  compressDataVec_lo_hi_lo_1 = {compressDataVec_useTail_21 ? compressDataReg[95:80] : compressDataVec_selectReqData_37, compressDataVec_useTail_20 ? compressDataReg[79:64] : compressDataVec_selectReqData_36};
  wire [31:0]  compressDataVec_lo_hi_hi_1 = {compressDataVec_useTail_23 ? compressDataReg[127:112] : compressDataVec_selectReqData_39, compressDataVec_useTail_22 ? compressDataReg[111:96] : compressDataVec_selectReqData_38};
  wire [63:0]  compressDataVec_lo_hi_1 = {compressDataVec_lo_hi_hi_1, compressDataVec_lo_hi_lo_1};
  wire [127:0] compressDataVec_lo_1 = {compressDataVec_lo_hi_1, compressDataVec_lo_lo_1};
  wire [31:0]  compressDataVec_hi_lo_lo_1 = {compressDataVec_selectReqData_41, compressDataVec_selectReqData_40};
  wire [31:0]  compressDataVec_hi_lo_hi_1 = {compressDataVec_selectReqData_43, compressDataVec_selectReqData_42};
  wire [63:0]  compressDataVec_hi_lo_1 = {compressDataVec_hi_lo_hi_1, compressDataVec_hi_lo_lo_1};
  wire [31:0]  compressDataVec_hi_hi_lo_1 = {compressDataVec_selectReqData_45, compressDataVec_selectReqData_44};
  wire [31:0]  compressDataVec_hi_hi_hi_1 = {compressDataVec_selectReqData_47, compressDataVec_selectReqData_46};
  wire [63:0]  compressDataVec_hi_hi_1 = {compressDataVec_hi_hi_hi_1, compressDataVec_hi_hi_lo_1};
  wire [127:0] compressDataVec_hi_1 = {compressDataVec_hi_hi_1, compressDataVec_hi_lo_1};
  wire [255:0] compressDataVec_1 = {compressDataVec_hi_1, compressDataVec_lo_1};
  wire [31:0]  compressDataVec_selectReqData_48 =
    (compressDataVec_hitReq_0_48 ? source2Pipe[31:0] : 32'h0) | (compressDataVec_hitReq_1_48 ? source2Pipe[63:32] : 32'h0) | (compressDataVec_hitReq_2_48 ? source2Pipe[95:64] : 32'h0)
    | (compressDataVec_hitReq_3_48 ? source2Pipe[127:96] : 32'h0);
  wire         compressDataVec_useTail_24;
  assign compressDataVec_useTail_24 = |tailCount;
  wire [31:0]  compressDataVec_selectReqData_49 =
    (compressDataVec_hitReq_0_49 ? source2Pipe[31:0] : 32'h0) | (compressDataVec_hitReq_1_49 ? source2Pipe[63:32] : 32'h0) | (compressDataVec_hitReq_2_49 ? source2Pipe[95:64] : 32'h0)
    | (compressDataVec_hitReq_3_49 ? source2Pipe[127:96] : 32'h0);
  wire         compressDataVec_useTail_25;
  assign compressDataVec_useTail_25 = |(tailCount[3:1]);
  wire [31:0]  compressDataVec_selectReqData_50 =
    (compressDataVec_hitReq_0_50 ? source2Pipe[31:0] : 32'h0) | (compressDataVec_hitReq_1_50 ? source2Pipe[63:32] : 32'h0) | (compressDataVec_hitReq_2_50 ? source2Pipe[95:64] : 32'h0)
    | (compressDataVec_hitReq_3_50 ? source2Pipe[127:96] : 32'h0);
  wire [31:0]  compressDataVec_selectReqData_51 =
    (compressDataVec_hitReq_0_51 ? source2Pipe[31:0] : 32'h0) | (compressDataVec_hitReq_1_51 ? source2Pipe[63:32] : 32'h0) | (compressDataVec_hitReq_2_51 ? source2Pipe[95:64] : 32'h0)
    | (compressDataVec_hitReq_3_51 ? source2Pipe[127:96] : 32'h0);
  wire         compressDataVec_useTail_27;
  assign compressDataVec_useTail_27 = |(tailCount[3:2]);
  wire [31:0]  compressDataVec_selectReqData_52 =
    (compressDataVec_hitReq_0_52 ? source2Pipe[31:0] : 32'h0) | (compressDataVec_hitReq_1_52 ? source2Pipe[63:32] : 32'h0) | (compressDataVec_hitReq_2_52 ? source2Pipe[95:64] : 32'h0)
    | (compressDataVec_hitReq_3_52 ? source2Pipe[127:96] : 32'h0);
  wire [31:0]  compressDataVec_selectReqData_53 =
    (compressDataVec_hitReq_0_53 ? source2Pipe[31:0] : 32'h0) | (compressDataVec_hitReq_1_53 ? source2Pipe[63:32] : 32'h0) | (compressDataVec_hitReq_2_53 ? source2Pipe[95:64] : 32'h0)
    | (compressDataVec_hitReq_3_53 ? source2Pipe[127:96] : 32'h0);
  wire [31:0]  compressDataVec_selectReqData_54 =
    (compressDataVec_hitReq_0_54 ? source2Pipe[31:0] : 32'h0) | (compressDataVec_hitReq_1_54 ? source2Pipe[63:32] : 32'h0) | (compressDataVec_hitReq_2_54 ? source2Pipe[95:64] : 32'h0)
    | (compressDataVec_hitReq_3_54 ? source2Pipe[127:96] : 32'h0);
  wire [31:0]  compressDataVec_selectReqData_55 =
    (compressDataVec_hitReq_0_55 ? source2Pipe[31:0] : 32'h0) | (compressDataVec_hitReq_1_55 ? source2Pipe[63:32] : 32'h0) | (compressDataVec_hitReq_2_55 ? source2Pipe[95:64] : 32'h0)
    | (compressDataVec_hitReq_3_55 ? source2Pipe[127:96] : 32'h0);
  wire [63:0]  compressDataVec_lo_lo_2 = {compressDataVec_useTail_25 ? compressDataReg[63:32] : compressDataVec_selectReqData_49, compressDataVec_useTail_24 ? compressDataReg[31:0] : compressDataVec_selectReqData_48};
  wire [63:0]  compressDataVec_lo_hi_2 = {compressDataVec_useTail_27 ? compressDataReg[127:96] : compressDataVec_selectReqData_51, compressDataVec_useTail_26 ? compressDataReg[95:64] : compressDataVec_selectReqData_50};
  wire [127:0] compressDataVec_lo_2 = {compressDataVec_lo_hi_2, compressDataVec_lo_lo_2};
  wire [63:0]  compressDataVec_hi_lo_2 = {compressDataVec_selectReqData_53, compressDataVec_selectReqData_52};
  wire [63:0]  compressDataVec_hi_hi_2 = {compressDataVec_selectReqData_55, compressDataVec_selectReqData_54};
  wire [127:0] compressDataVec_hi_2 = {compressDataVec_hi_hi_2, compressDataVec_hi_lo_2};
  wire [255:0] compressDataVec_2 = {compressDataVec_hi_2, compressDataVec_lo_2};
  wire [255:0] compressResult = (eew1H[0] ? compressDataVec_0 : 256'h0) | (eew1H[1] ? compressDataVec_1 : 256'h0) | (eew1H[2] ? compressDataVec_2 : 256'h0);
  wire         lastCompressEnq = stage2Valid & lastCompressPipe;
  wire [127:0] splitCompressResult_0 = compressResult[127:0];
  wire [127:0] splitCompressResult_1 = compressResult[255:128];
  wire         compressTailMask_elementValid;
  assign compressTailMask_elementValid = |tailCountForMask;
  wire         compressTailMask_elementValid_1;
  assign compressTailMask_elementValid_1 = |(tailCountForMask[3:1]);
  wire         _GEN_132 = tailCountForMask > 4'h2;
  wire         compressTailMask_elementValid_2;
  assign compressTailMask_elementValid_2 = _GEN_132;
  wire         compressTailMask_elementValid_18;
  assign compressTailMask_elementValid_18 = _GEN_132;
  wire         compressTailMask_elementValid_26;
  assign compressTailMask_elementValid_26 = _GEN_132;
  wire         compressTailMask_elementValid_3;
  assign compressTailMask_elementValid_3 = |(tailCountForMask[3:2]);
  wire         _GEN_133 = tailCountForMask > 4'h4;
  wire         compressTailMask_elementValid_4;
  assign compressTailMask_elementValid_4 = _GEN_133;
  wire         compressTailMask_elementValid_20;
  assign compressTailMask_elementValid_20 = _GEN_133;
  wire         _GEN_134 = tailCountForMask > 4'h5;
  wire         compressTailMask_elementValid_5;
  assign compressTailMask_elementValid_5 = _GEN_134;
  wire         compressTailMask_elementValid_21;
  assign compressTailMask_elementValid_21 = _GEN_134;
  wire         _GEN_135 = tailCountForMask > 4'h6;
  wire         compressTailMask_elementValid_6;
  assign compressTailMask_elementValid_6 = _GEN_135;
  wire         compressTailMask_elementValid_22;
  assign compressTailMask_elementValid_22 = _GEN_135;
  wire         compressTailMask_elementValid_7 = tailCountForMask[3];
  wire         compressTailMask_elementValid_23 = tailCountForMask[3];
  wire         compressTailMask_elementValid_8 = tailCountForMask > 4'h8;
  wire         compressTailMask_elementValid_9 = tailCountForMask > 4'h9;
  wire         compressTailMask_elementValid_10 = tailCountForMask > 4'hA;
  wire         compressTailMask_elementValid_11 = tailCountForMask > 4'hB;
  wire         compressTailMask_elementValid_12 = tailCountForMask > 4'hC;
  wire         compressTailMask_elementValid_13 = tailCountForMask > 4'hD;
  wire         compressTailMask_elementValid_14 = &tailCountForMask;
  wire [1:0]   compressTailMask_lo_lo_lo = {compressTailMask_elementValid_1, compressTailMask_elementValid};
  wire [1:0]   compressTailMask_lo_lo_hi = {compressTailMask_elementValid_3, compressTailMask_elementValid_2};
  wire [3:0]   compressTailMask_lo_lo = {compressTailMask_lo_lo_hi, compressTailMask_lo_lo_lo};
  wire [1:0]   compressTailMask_lo_hi_lo = {compressTailMask_elementValid_5, compressTailMask_elementValid_4};
  wire [1:0]   compressTailMask_lo_hi_hi = {compressTailMask_elementValid_7, compressTailMask_elementValid_6};
  wire [3:0]   compressTailMask_lo_hi = {compressTailMask_lo_hi_hi, compressTailMask_lo_hi_lo};
  wire [7:0]   compressTailMask_lo = {compressTailMask_lo_hi, compressTailMask_lo_lo};
  wire [1:0]   compressTailMask_hi_lo_lo = {compressTailMask_elementValid_9, compressTailMask_elementValid_8};
  wire [1:0]   compressTailMask_hi_lo_hi = {compressTailMask_elementValid_11, compressTailMask_elementValid_10};
  wire [3:0]   compressTailMask_hi_lo = {compressTailMask_hi_lo_hi, compressTailMask_hi_lo_lo};
  wire [1:0]   compressTailMask_hi_hi_lo = {compressTailMask_elementValid_13, compressTailMask_elementValid_12};
  wire [1:0]   compressTailMask_hi_hi_hi = {1'h0, compressTailMask_elementValid_14};
  wire [3:0]   compressTailMask_hi_hi = {compressTailMask_hi_hi_hi, compressTailMask_hi_hi_lo};
  wire [7:0]   compressTailMask_hi = {compressTailMask_hi_hi, compressTailMask_hi_lo};
  wire         compressTailMask_elementValid_16;
  assign compressTailMask_elementValid_16 = |tailCountForMask;
  wire [1:0]   compressTailMask_elementMask = {2{compressTailMask_elementValid_16}};
  wire         compressTailMask_elementValid_17;
  assign compressTailMask_elementValid_17 = |(tailCountForMask[3:1]);
  wire [1:0]   compressTailMask_elementMask_1 = {2{compressTailMask_elementValid_17}};
  wire [1:0]   compressTailMask_elementMask_2 = {2{compressTailMask_elementValid_18}};
  wire         compressTailMask_elementValid_19;
  assign compressTailMask_elementValid_19 = |(tailCountForMask[3:2]);
  wire [1:0]   compressTailMask_elementMask_3 = {2{compressTailMask_elementValid_19}};
  wire [1:0]   compressTailMask_elementMask_4 = {2{compressTailMask_elementValid_20}};
  wire [1:0]   compressTailMask_elementMask_5 = {2{compressTailMask_elementValid_21}};
  wire [1:0]   compressTailMask_elementMask_6 = {2{compressTailMask_elementValid_22}};
  wire [1:0]   compressTailMask_elementMask_7 = {2{compressTailMask_elementValid_23}};
  wire [3:0]   compressTailMask_lo_lo_1 = {compressTailMask_elementMask_1, compressTailMask_elementMask};
  wire [3:0]   compressTailMask_lo_hi_1 = {compressTailMask_elementMask_3, compressTailMask_elementMask_2};
  wire [7:0]   compressTailMask_lo_1 = {compressTailMask_lo_hi_1, compressTailMask_lo_lo_1};
  wire [3:0]   compressTailMask_hi_lo_1 = {compressTailMask_elementMask_5, compressTailMask_elementMask_4};
  wire [3:0]   compressTailMask_hi_hi_1 = {compressTailMask_elementMask_7, compressTailMask_elementMask_6};
  wire [7:0]   compressTailMask_hi_1 = {compressTailMask_hi_hi_1, compressTailMask_hi_lo_1};
  wire         compressTailMask_elementValid_24;
  assign compressTailMask_elementValid_24 = |tailCountForMask;
  wire [3:0]   compressTailMask_elementMask_8 = {4{compressTailMask_elementValid_24}};
  wire         compressTailMask_elementValid_25;
  assign compressTailMask_elementValid_25 = |(tailCountForMask[3:1]);
  wire [3:0]   compressTailMask_elementMask_9 = {4{compressTailMask_elementValid_25}};
  wire [3:0]   compressTailMask_elementMask_10 = {4{compressTailMask_elementValid_26}};
  wire         compressTailMask_elementValid_27;
  assign compressTailMask_elementValid_27 = |(tailCountForMask[3:2]);
  wire [3:0]   compressTailMask_elementMask_11 = {4{compressTailMask_elementValid_27}};
  wire [7:0]   compressTailMask_lo_2 = {compressTailMask_elementMask_9, compressTailMask_elementMask_8};
  wire [7:0]   compressTailMask_hi_2 = {compressTailMask_elementMask_11, compressTailMask_elementMask_10};
  wire [15:0]  compressTailMask = (eew1H[0] ? {compressTailMask_hi, compressTailMask_lo} : 16'h0) | (eew1H[1] ? {compressTailMask_hi_1, compressTailMask_lo_1} : 16'h0) | (eew1H[2] ? {compressTailMask_hi_2, compressTailMask_lo_2} : 16'h0);
  wire [15:0]  compressMask = compressTailValid ? compressTailMask : 16'hFFFF;
  reg  [3:0]   validInputPipe;
  reg  [31:0]  readFromScalarPipe;
  wire [3:0]   mvMask = {2'h0, {1'h0, eew1H[0]} | {2{eew1H[1]}}} | {4{eew1H[2]}};
  wire [7:0]   ffoMask_lo = {{4{validInputPipe[1]}}, {4{validInputPipe[0]}}};
  wire [7:0]   ffoMask_hi = {{4{validInputPipe[3]}}, {4{validInputPipe[2]}}};
  wire [15:0]  ffoMask = {ffoMask_hi, ffoMask_lo};
  wire [3:0]   outWire_ffoOutput;
  wire [63:0]  ffoData_lo = {outWire_ffoOutput[1] ? in_1_bits_pipeData[63:32] : in_1_bits_source2[63:32], outWire_ffoOutput[0] ? in_1_bits_pipeData[31:0] : in_1_bits_source2[31:0]};
  wire [63:0]  ffoData_hi = {outWire_ffoOutput[3] ? in_1_bits_pipeData[127:96] : in_1_bits_source2[127:96], outWire_ffoOutput[2] ? in_1_bits_pipeData[95:64] : in_1_bits_source2[95:64]};
  wire [127:0] ffoData = {ffoData_hi, ffoData_lo};
  wire [127:0] _GEN_136 = (compress ? compressResult[127:0] : 128'h0) | (viota ? viotaResult : 128'h0);
  wire [127:0] outWire_data = {_GEN_136[127:32], _GEN_136[31:0] | (mv ? readFromScalarPipe : 32'h0)} | (ffoType ? ffoData : 128'h0);
  wire [15:0]  _outWire_mask_T_4 = (compress ? compressMask : 16'h0) | (viota ? viotaMask : 16'h0);
  wire [15:0]  outWire_mask = {_outWire_mask_T_4[15:4], _outWire_mask_T_4[3:0] | (mv ? mvMask : 4'h0)} | (ffoType ? ffoMask : 16'h0);
  wire         outWire_compressValid = (compressTailValid | compressDeqValidPipe & stage2Valid) & ~writeRD;
  wire [10:0]  outWire_groupCounter = compress ? compressWriteGroupCount : groupCounterPipe;
  wire [2:0]   _completedLeftOr_T_2 = in_1_bits_ffoInput[2:0] | {in_1_bits_ffoInput[1:0], 1'h0};
  wire [2:0]   _GEN_137 = {_completedLeftOr_T_2[0], 2'h0};
  wire [3:0]   firstLane = {~(_completedLeftOr_T_2 | _GEN_137), 1'h1} & in_1_bits_ffoInput;
  wire [1:0]   firstLaneIndex_hi = firstLane[3:2];
  wire [1:0]   firstLaneIndex_lo = firstLane[1:0];
  wire [1:0]   firstLaneIndex = {|firstLaneIndex_hi, firstLaneIndex_hi[1] | firstLaneIndex_lo[1]};
  wire [31:0]  source1SigExtend = (eew1H[0] ? {{24{in_1_bits_source1[7]}}, in_1_bits_source1[7:0]} : 32'h0) | (eew1H[1] ? {{16{in_1_bits_source1[15]}}, in_1_bits_source1[15:0]} : 32'h0) | (eew1H[2] ? in_1_bits_source1 : 32'h0);
  wire [3:0]   completedLeftOr = {_completedLeftOr_T_2 | _GEN_137, 1'h0};
  reg  [3:0]   ffoOutPipe;
  assign outWire_ffoOutput = ffoOutPipe;
  reg  [127:0] view__out_REG_data;
  reg  [15:0]  view__out_REG_mask;
  reg  [10:0]  view__out_REG_groupCounter;
  reg  [3:0]   view__out_REG_ffoOutput;
  reg          view__out_REG_compressValid;
  always @(posedge clock) begin
    if (reset) begin
      in_1_valid <= 1'h0;
      in_1_bits_maskType <= 1'h0;
      in_1_bits_eew <= 2'h0;
      in_1_bits_uop <= 3'h0;
      in_1_bits_readFromScalar <= 32'h0;
      in_1_bits_source1 <= 32'h0;
      in_1_bits_mask <= 32'h0;
      in_1_bits_source2 <= 128'h0;
      in_1_bits_pipeData <= 128'h0;
      in_1_bits_groupCounter <= 11'h0;
      in_1_bits_ffoInput <= 4'h0;
      in_1_bits_validInput <= 4'h0;
      in_1_bits_lastCompress <= 1'h0;
      compressInit <= 14'h0;
      ffoIndex <= 32'h0;
      ffoValid <= 1'h0;
      compressVecPipe_0 <= 14'h0;
      compressVecPipe_1 <= 14'h0;
      compressVecPipe_2 <= 14'h0;
      compressVecPipe_3 <= 14'h0;
      compressVecPipe_4 <= 14'h0;
      compressVecPipe_5 <= 14'h0;
      compressVecPipe_6 <= 14'h0;
      compressVecPipe_7 <= 14'h0;
      compressVecPipe_8 <= 14'h0;
      compressVecPipe_9 <= 14'h0;
      compressVecPipe_10 <= 14'h0;
      compressVecPipe_11 <= 14'h0;
      compressVecPipe_12 <= 14'h0;
      compressVecPipe_13 <= 14'h0;
      compressVecPipe_14 <= 14'h0;
      compressVecPipe_15 <= 14'h0;
      compressMaskVecPipe_0 <= 1'h0;
      compressMaskVecPipe_1 <= 1'h0;
      compressMaskVecPipe_2 <= 1'h0;
      compressMaskVecPipe_3 <= 1'h0;
      compressMaskVecPipe_4 <= 1'h0;
      compressMaskVecPipe_5 <= 1'h0;
      compressMaskVecPipe_6 <= 1'h0;
      compressMaskVecPipe_7 <= 1'h0;
      compressMaskVecPipe_8 <= 1'h0;
      compressMaskVecPipe_9 <= 1'h0;
      compressMaskVecPipe_10 <= 1'h0;
      compressMaskVecPipe_11 <= 1'h0;
      compressMaskVecPipe_12 <= 1'h0;
      compressMaskVecPipe_13 <= 1'h0;
      compressMaskVecPipe_14 <= 1'h0;
      compressMaskVecPipe_15 <= 1'h0;
      maskPipe <= 32'h0;
      source2Pipe <= 128'h0;
      lastCompressPipe <= 1'h0;
      stage2Valid <= 1'h0;
      newInstructionPipe <= 1'h0;
      compressInitPipe <= 14'h0;
      compressDeqValidPipe <= 1'h0;
      groupCounterPipe <= 11'h0;
      compressDataReg <= 128'h0;
      compressTailValid <= 1'h0;
      compressWriteGroupCount <= 11'h0;
      validInputPipe <= 4'h0;
      readFromScalarPipe <= 32'h0;
      ffoOutPipe <= 4'h0;
      view__out_REG_data <= 128'h0;
      view__out_REG_mask <= 16'h0;
      view__out_REG_groupCounter <= 11'h0;
      view__out_REG_ffoOutput <= 4'h0;
      view__out_REG_compressValid <= 1'h0;
    end
    else begin
      automatic logic _GEN_138;
      automatic logic _GEN_139;
      _GEN_138 = newInstruction & ffoInstruction;
      _GEN_139 = in_1_valid & (|in_1_bits_ffoInput) & ffoType;
      in_1_valid <= in_valid;
      in_1_bits_maskType <= in_bits_maskType;
      in_1_bits_eew <= in_bits_eew;
      in_1_bits_uop <= in_bits_uop;
      in_1_bits_readFromScalar <= in_bits_readFromScalar;
      in_1_bits_source1 <= in_bits_source1;
      in_1_bits_mask <= in_bits_mask;
      in_1_bits_source2 <= in_bits_source2;
      in_1_bits_pipeData <= in_bits_pipeData;
      in_1_bits_groupCounter <= in_bits_groupCounter;
      in_1_bits_ffoInput <= in_bits_ffoInput;
      in_1_bits_validInput <= in_bits_validInput;
      in_1_bits_lastCompress <= in_bits_lastCompress;
      if (in_1_valid) begin
        compressInit <= viota ? compressCount : {10'h0, compressCountSelect};
        compressVecPipe_0 <= compressVec_0;
        compressVecPipe_1 <= compressVec_1;
        compressVecPipe_2 <= compressVec_2;
        compressVecPipe_3 <= compressVec_3;
        compressVecPipe_4 <= compressVec_4;
        compressVecPipe_5 <= compressVec_5;
        compressVecPipe_6 <= compressVec_6;
        compressVecPipe_7 <= compressVec_7;
        compressVecPipe_8 <= compressVec_8;
        compressVecPipe_9 <= compressVec_9;
        compressVecPipe_10 <= compressVec_10;
        compressVecPipe_11 <= compressVec_11;
        compressVecPipe_12 <= compressVec_12;
        compressVecPipe_13 <= compressVec_13;
        compressVecPipe_14 <= compressVec_14;
        compressVecPipe_15 <= compressVec_15;
        compressMaskVecPipe_0 <= compressMaskVec_0;
        compressMaskVecPipe_1 <= compressMaskVec_1;
        compressMaskVecPipe_2 <= compressMaskVec_2;
        compressMaskVecPipe_3 <= compressMaskVec_3;
        compressMaskVecPipe_4 <= compressMaskVec_4;
        compressMaskVecPipe_5 <= compressMaskVec_5;
        compressMaskVecPipe_6 <= compressMaskVec_6;
        compressMaskVecPipe_7 <= compressMaskVec_7;
        compressMaskVecPipe_8 <= compressMaskVec_8;
        compressMaskVecPipe_9 <= compressMaskVec_9;
        compressMaskVecPipe_10 <= compressMaskVec_10;
        compressMaskVecPipe_11 <= compressMaskVec_11;
        compressMaskVecPipe_12 <= compressMaskVec_12;
        compressMaskVecPipe_13 <= compressMaskVec_13;
        compressMaskVecPipe_14 <= compressMaskVec_14;
        compressMaskVecPipe_15 <= compressMaskVec_15;
        maskPipe <= in_1_bits_mask;
        source2Pipe <= in_1_bits_source2;
        lastCompressPipe <= in_1_bits_lastCompress;
        compressInitPipe <= compressInit;
        compressDeqValidPipe <= compressDeqValid;
        groupCounterPipe <= in_1_bits_groupCounter;
        validInputPipe <= in_1_bits_validInput;
        readFromScalarPipe <= in_1_bits_readFromScalar;
        ffoOutPipe <= completedLeftOr | {4{ffoValid}};
      end
      else if (newInstruction)
        compressInit <= 14'h0;
      if (_GEN_139) begin
        if (ffoValid) begin
          if (_GEN_138)
            ffoIndex <= 32'hFFFFFFFF;
        end
        else
          ffoIndex <=
            {1'h0,
             (firstLane[0] ? {in_1_bits_source2[28:5], firstLaneIndex, in_1_bits_source2[4:0]} : 31'h0) | (firstLane[1] ? {in_1_bits_source2[60:37], firstLaneIndex, in_1_bits_source2[36:32]} : 31'h0)
               | (firstLane[2] ? {in_1_bits_source2[92:69], firstLaneIndex, in_1_bits_source2[68:64]} : 31'h0) | (firstLane[3] ? {in_1_bits_source2[124:101], firstLaneIndex, in_1_bits_source2[100:96]} : 31'h0)};
      end
      else if (mvRd)
        ffoIndex <= source1SigExtend;
      else if (_GEN_138)
        ffoIndex <= 32'hFFFFFFFF;
      ffoValid <= _GEN_139 | ~_GEN_138 & ffoValid;
      stage2Valid <= in_1_valid;
      newInstructionPipe <= newInstruction;
      if (stage2Valid)
        compressDataReg <= compressDeqValidPipe ? splitCompressResult_1 : splitCompressResult_0;
      if (newInstructionPipe | lastCompressEnq | outWire_compressValid)
        compressTailValid <= lastCompressEnq & compress;
      if (newInstructionPipe | outWire_compressValid)
        compressWriteGroupCount <= newInstructionPipe ? 11'h0 : compressWriteGroupCount + 11'h1;
      view__out_REG_data <= outWire_data;
      view__out_REG_mask <= outWire_mask;
      view__out_REG_groupCounter <= outWire_groupCounter;
      view__out_REG_ffoOutput <= outWire_ffoOutput;
      view__out_REG_compressValid <= outWire_compressValid;
    end
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:37];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [5:0] i = 6'h0; i < 6'h26; i += 6'h1) begin
          _RANDOM[i] = `RANDOM;
        end
        in_1_valid = _RANDOM[6'h0][0];
        in_1_bits_maskType = _RANDOM[6'h0][1];
        in_1_bits_eew = _RANDOM[6'h0][3:2];
        in_1_bits_uop = _RANDOM[6'h0][6:4];
        in_1_bits_readFromScalar = {_RANDOM[6'h0][31:7], _RANDOM[6'h1][6:0]};
        in_1_bits_source1 = {_RANDOM[6'h1][31:7], _RANDOM[6'h2][6:0]};
        in_1_bits_mask = {_RANDOM[6'h2][31:7], _RANDOM[6'h3][6:0]};
        in_1_bits_source2 = {_RANDOM[6'h3][31:7], _RANDOM[6'h4], _RANDOM[6'h5], _RANDOM[6'h6], _RANDOM[6'h7][6:0]};
        in_1_bits_pipeData = {_RANDOM[6'h7][31:7], _RANDOM[6'h8], _RANDOM[6'h9], _RANDOM[6'hA], _RANDOM[6'hB][6:0]};
        in_1_bits_groupCounter = _RANDOM[6'hB][17:7];
        in_1_bits_ffoInput = _RANDOM[6'hB][21:18];
        in_1_bits_validInput = _RANDOM[6'hB][25:22];
        in_1_bits_lastCompress = _RANDOM[6'hB][26];
        compressInit = {_RANDOM[6'hB][31:27], _RANDOM[6'hC][8:0]};
        ffoIndex = {_RANDOM[6'hC][31:9], _RANDOM[6'hD][8:0]};
        ffoValid = _RANDOM[6'hD][9];
        compressVecPipe_0 = _RANDOM[6'hD][23:10];
        compressVecPipe_1 = {_RANDOM[6'hD][31:24], _RANDOM[6'hE][5:0]};
        compressVecPipe_2 = _RANDOM[6'hE][19:6];
        compressVecPipe_3 = {_RANDOM[6'hE][31:20], _RANDOM[6'hF][1:0]};
        compressVecPipe_4 = _RANDOM[6'hF][15:2];
        compressVecPipe_5 = _RANDOM[6'hF][29:16];
        compressVecPipe_6 = {_RANDOM[6'hF][31:30], _RANDOM[6'h10][11:0]};
        compressVecPipe_7 = _RANDOM[6'h10][25:12];
        compressVecPipe_8 = {_RANDOM[6'h10][31:26], _RANDOM[6'h11][7:0]};
        compressVecPipe_9 = _RANDOM[6'h11][21:8];
        compressVecPipe_10 = {_RANDOM[6'h11][31:22], _RANDOM[6'h12][3:0]};
        compressVecPipe_11 = _RANDOM[6'h12][17:4];
        compressVecPipe_12 = _RANDOM[6'h12][31:18];
        compressVecPipe_13 = _RANDOM[6'h13][13:0];
        compressVecPipe_14 = _RANDOM[6'h13][27:14];
        compressVecPipe_15 = {_RANDOM[6'h13][31:28], _RANDOM[6'h14][9:0]};
        compressMaskVecPipe_0 = _RANDOM[6'h14][10];
        compressMaskVecPipe_1 = _RANDOM[6'h14][11];
        compressMaskVecPipe_2 = _RANDOM[6'h14][12];
        compressMaskVecPipe_3 = _RANDOM[6'h14][13];
        compressMaskVecPipe_4 = _RANDOM[6'h14][14];
        compressMaskVecPipe_5 = _RANDOM[6'h14][15];
        compressMaskVecPipe_6 = _RANDOM[6'h14][16];
        compressMaskVecPipe_7 = _RANDOM[6'h14][17];
        compressMaskVecPipe_8 = _RANDOM[6'h14][18];
        compressMaskVecPipe_9 = _RANDOM[6'h14][19];
        compressMaskVecPipe_10 = _RANDOM[6'h14][20];
        compressMaskVecPipe_11 = _RANDOM[6'h14][21];
        compressMaskVecPipe_12 = _RANDOM[6'h14][22];
        compressMaskVecPipe_13 = _RANDOM[6'h14][23];
        compressMaskVecPipe_14 = _RANDOM[6'h14][24];
        compressMaskVecPipe_15 = _RANDOM[6'h14][25];
        maskPipe = {_RANDOM[6'h14][31:26], _RANDOM[6'h15][25:0]};
        source2Pipe = {_RANDOM[6'h15][31:26], _RANDOM[6'h16], _RANDOM[6'h17], _RANDOM[6'h18], _RANDOM[6'h19][25:0]};
        lastCompressPipe = _RANDOM[6'h19][26];
        stage2Valid = _RANDOM[6'h19][27];
        newInstructionPipe = _RANDOM[6'h19][28];
        compressInitPipe = {_RANDOM[6'h19][31:29], _RANDOM[6'h1A][10:0]};
        compressDeqValidPipe = _RANDOM[6'h1A][11];
        groupCounterPipe = _RANDOM[6'h1A][22:12];
        compressDataReg = {_RANDOM[6'h1A][31:23], _RANDOM[6'h1B], _RANDOM[6'h1C], _RANDOM[6'h1D], _RANDOM[6'h1E][22:0]};
        compressTailValid = _RANDOM[6'h1E][23];
        compressWriteGroupCount = {_RANDOM[6'h1E][31:24], _RANDOM[6'h1F][2:0]};
        validInputPipe = _RANDOM[6'h1F][6:3];
        readFromScalarPipe = {_RANDOM[6'h1F][31:7], _RANDOM[6'h20][6:0]};
        ffoOutPipe = _RANDOM[6'h20][10:7];
        view__out_REG_data = {_RANDOM[6'h20][31:11], _RANDOM[6'h21], _RANDOM[6'h22], _RANDOM[6'h23], _RANDOM[6'h24][10:0]};
        view__out_REG_mask = _RANDOM[6'h24][26:11];
        view__out_REG_groupCounter = {_RANDOM[6'h24][31:27], _RANDOM[6'h25][5:0]};
        view__out_REG_ffoOutput = _RANDOM[6'h25][9:6];
        view__out_REG_compressValid = _RANDOM[6'h25][10];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  assign out_data = view__out_REG_data;
  assign out_mask = view__out_REG_mask;
  assign out_groupCounter = view__out_REG_groupCounter;
  assign out_ffoOutput = view__out_REG_ffoOutput;
  assign out_compressValid = view__out_REG_compressValid;
  assign writeData = ffoIndex;
  assign stageValid = stage2Valid | in_1_valid | compressTailValid;
endmodule

