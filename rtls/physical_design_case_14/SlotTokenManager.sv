
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
module SlotTokenManager(
  input        clock,
               reset,
               enqReports_0_valid,
               enqReports_0_bits_decodeResult_sWrite,
               enqReports_0_bits_decodeResult_crossWrite,
               enqReports_0_bits_decodeResult_maskUnit,
  input  [2:0] enqReports_0_bits_instructionIndex,
  input        enqReports_0_bits_sSendResponse,
               enqReports_1_valid,
               enqReports_1_bits_decodeResult_sWrite,
               enqReports_1_bits_decodeResult_maskUnit,
  input  [2:0] enqReports_1_bits_instructionIndex,
  input        enqReports_2_valid,
               enqReports_2_bits_decodeResult_sWrite,
               enqReports_2_bits_decodeResult_maskUnit,
  input  [2:0] enqReports_2_bits_instructionIndex,
  input        enqReports_3_valid,
               enqReports_3_bits_decodeResult_sWrite,
               enqReports_3_bits_decodeResult_maskUnit,
  input  [2:0] enqReports_3_bits_instructionIndex,
  input        crossWriteReports_0_valid,
  input  [2:0] crossWriteReports_0_bits,
  input        crossWriteReports_1_valid,
  input  [2:0] crossWriteReports_1_bits,
  input        responseReport_valid,
  input  [2:0] responseReport_bits,
  input        responseFeedbackReport_valid,
  input  [2:0] responseFeedbackReport_bits,
  input        slotWriteReport_0_valid,
  input  [2:0] slotWriteReport_0_bits,
  input        slotWriteReport_1_valid,
  input  [2:0] slotWriteReport_1_bits,
  input        slotWriteReport_2_valid,
  input  [2:0] slotWriteReport_2_bits,
  input        slotWriteReport_3_valid,
  input  [2:0] slotWriteReport_3_bits,
  input        writePipeEnqReport_valid,
  input  [2:0] writePipeEnqReport_bits,
  input        writePipeDeqReport_valid,
  input  [2:0] writePipeDeqReport_bits,
  input        topWriteEnq_valid,
  input  [2:0] topWriteEnq_bits,
  input        topWriteDeq_valid,
  input  [2:0] topWriteDeq_bits,
  output [7:0] instructionValid,
               dataInWritePipe,
  input  [7:0] maskUnitLastReport
);

  reg  [4:0] instructionInSlot_writeToken_0;
  reg  [4:0] instructionInSlot_writeToken_1;
  reg  [4:0] instructionInSlot_writeToken_2;
  reg  [4:0] instructionInSlot_writeToken_3;
  reg  [4:0] instructionInSlot_writeToken_4;
  reg  [4:0] instructionInSlot_writeToken_5;
  reg  [4:0] instructionInSlot_writeToken_6;
  reg  [4:0] instructionInSlot_writeToken_7;
  wire [7:0] instructionInSlot_enqOH = 8'h1 << enqReports_0_bits_instructionIndex;
  wire [7:0] instructionInSlot_writeDoEnq = enqReports_0_valid & ~enqReports_0_bits_decodeResult_sWrite & ~enqReports_0_bits_decodeResult_maskUnit ? instructionInSlot_enqOH : 8'h0;
  wire [7:0] instructionInSlot_writeEnqSelect = instructionInSlot_writeDoEnq;
  wire [7:0] instructionInSlot_writeDoDeq = slotWriteReport_0_valid ? 8'h1 << slotWriteReport_0_bits : 8'h0;
  wire       instructionInSlot_pendingSlotWrite_e = instructionInSlot_writeEnqSelect[0];
  wire       instructionInSlot_pendingSlotWrite_d = instructionInSlot_writeDoDeq[0];
  wire [4:0] instructionInSlot_pendingSlotWrite_change = instructionInSlot_pendingSlotWrite_e ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_1 = instructionInSlot_writeEnqSelect[1];
  wire       instructionInSlot_pendingSlotWrite_d_1 = instructionInSlot_writeDoDeq[1];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_1 = instructionInSlot_pendingSlotWrite_e_1 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_2 = instructionInSlot_writeEnqSelect[2];
  wire       instructionInSlot_pendingSlotWrite_d_2 = instructionInSlot_writeDoDeq[2];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_2 = instructionInSlot_pendingSlotWrite_e_2 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_3 = instructionInSlot_writeEnqSelect[3];
  wire       instructionInSlot_pendingSlotWrite_d_3 = instructionInSlot_writeDoDeq[3];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_3 = instructionInSlot_pendingSlotWrite_e_3 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_4 = instructionInSlot_writeEnqSelect[4];
  wire       instructionInSlot_pendingSlotWrite_d_4 = instructionInSlot_writeDoDeq[4];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_4 = instructionInSlot_pendingSlotWrite_e_4 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_5 = instructionInSlot_writeEnqSelect[5];
  wire       instructionInSlot_pendingSlotWrite_d_5 = instructionInSlot_writeDoDeq[5];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_5 = instructionInSlot_pendingSlotWrite_e_5 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_6 = instructionInSlot_writeEnqSelect[6];
  wire       instructionInSlot_pendingSlotWrite_d_6 = instructionInSlot_writeDoDeq[6];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_6 = instructionInSlot_pendingSlotWrite_e_6 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_7 = instructionInSlot_writeEnqSelect[7];
  wire       instructionInSlot_pendingSlotWrite_d_7 = instructionInSlot_writeDoDeq[7];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_7 = instructionInSlot_pendingSlotWrite_e_7 ? 5'h1 : 5'h1F;
  wire [1:0] instructionInSlot_pendingSlotWrite_lo_lo = {|instructionInSlot_writeToken_1, |instructionInSlot_writeToken_0};
  wire [1:0] instructionInSlot_pendingSlotWrite_lo_hi = {|instructionInSlot_writeToken_3, |instructionInSlot_writeToken_2};
  wire [3:0] instructionInSlot_pendingSlotWrite_lo = {instructionInSlot_pendingSlotWrite_lo_hi, instructionInSlot_pendingSlotWrite_lo_lo};
  wire [1:0] instructionInSlot_pendingSlotWrite_hi_lo = {|instructionInSlot_writeToken_5, |instructionInSlot_writeToken_4};
  wire [1:0] instructionInSlot_pendingSlotWrite_hi_hi = {|instructionInSlot_writeToken_7, |instructionInSlot_writeToken_6};
  wire [3:0] instructionInSlot_pendingSlotWrite_hi = {instructionInSlot_pendingSlotWrite_hi_hi, instructionInSlot_pendingSlotWrite_hi_lo};
  wire [7:0] instructionInSlot_pendingSlotWrite = {instructionInSlot_pendingSlotWrite_hi, instructionInSlot_pendingSlotWrite_lo};
  reg  [4:0] instructionInSlot_responseToken_0;
  reg  [4:0] instructionInSlot_responseToken_1;
  reg  [4:0] instructionInSlot_responseToken_2;
  reg  [4:0] instructionInSlot_responseToken_3;
  reg  [4:0] instructionInSlot_responseToken_4;
  reg  [4:0] instructionInSlot_responseToken_5;
  reg  [4:0] instructionInSlot_responseToken_6;
  reg  [4:0] instructionInSlot_responseToken_7;
  reg  [4:0] instructionInSlot_feedbackToken_0;
  reg  [4:0] instructionInSlot_feedbackToken_1;
  reg  [4:0] instructionInSlot_feedbackToken_2;
  reg  [4:0] instructionInSlot_feedbackToken_3;
  reg  [4:0] instructionInSlot_feedbackToken_4;
  reg  [4:0] instructionInSlot_feedbackToken_5;
  reg  [4:0] instructionInSlot_feedbackToken_6;
  reg  [4:0] instructionInSlot_feedbackToken_7;
  reg  [4:0] instructionInSlot_crossWriteTokenLSB_0;
  reg  [4:0] instructionInSlot_crossWriteTokenLSB_1;
  reg  [4:0] instructionInSlot_crossWriteTokenLSB_2;
  reg  [4:0] instructionInSlot_crossWriteTokenLSB_3;
  reg  [4:0] instructionInSlot_crossWriteTokenLSB_4;
  reg  [4:0] instructionInSlot_crossWriteTokenLSB_5;
  reg  [4:0] instructionInSlot_crossWriteTokenLSB_6;
  reg  [4:0] instructionInSlot_crossWriteTokenLSB_7;
  reg  [4:0] instructionInSlot_crossWriteTokenMSB_0;
  reg  [4:0] instructionInSlot_crossWriteTokenMSB_1;
  reg  [4:0] instructionInSlot_crossWriteTokenMSB_2;
  reg  [4:0] instructionInSlot_crossWriteTokenMSB_3;
  reg  [4:0] instructionInSlot_crossWriteTokenMSB_4;
  reg  [4:0] instructionInSlot_crossWriteTokenMSB_5;
  reg  [4:0] instructionInSlot_crossWriteTokenMSB_6;
  reg  [4:0] instructionInSlot_crossWriteTokenMSB_7;
  wire [7:0] instructionInSlot_crossWriteDoEnq = enqReports_0_valid & enqReports_0_bits_decodeResult_crossWrite ? instructionInSlot_enqOH : 8'h0;
  wire [7:0] instructionInSlot_crossWriteDeqLSB = crossWriteReports_0_valid ? 8'h1 << crossWriteReports_0_bits : 8'h0;
  wire [7:0] instructionInSlot_crossWriteDeqMSB = crossWriteReports_1_valid ? 8'h1 << crossWriteReports_1_bits : 8'h0;
  wire       instructionInSlot_pendingCrossWriteLSB_e = instructionInSlot_crossWriteDoEnq[0];
  wire       instructionInSlot_pendingCrossWriteMSB_e = instructionInSlot_crossWriteDoEnq[0];
  wire       instructionInSlot_pendingCrossWriteLSB_d = instructionInSlot_crossWriteDeqLSB[0];
  wire [4:0] instructionInSlot_pendingCrossWriteLSB_change = instructionInSlot_pendingCrossWriteLSB_e ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingCrossWriteLSB_e_1 = instructionInSlot_crossWriteDoEnq[1];
  wire       instructionInSlot_pendingCrossWriteMSB_e_1 = instructionInSlot_crossWriteDoEnq[1];
  wire       instructionInSlot_pendingCrossWriteLSB_d_1 = instructionInSlot_crossWriteDeqLSB[1];
  wire [4:0] instructionInSlot_pendingCrossWriteLSB_change_1 = instructionInSlot_pendingCrossWriteLSB_e_1 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingCrossWriteLSB_e_2 = instructionInSlot_crossWriteDoEnq[2];
  wire       instructionInSlot_pendingCrossWriteMSB_e_2 = instructionInSlot_crossWriteDoEnq[2];
  wire       instructionInSlot_pendingCrossWriteLSB_d_2 = instructionInSlot_crossWriteDeqLSB[2];
  wire [4:0] instructionInSlot_pendingCrossWriteLSB_change_2 = instructionInSlot_pendingCrossWriteLSB_e_2 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingCrossWriteLSB_e_3 = instructionInSlot_crossWriteDoEnq[3];
  wire       instructionInSlot_pendingCrossWriteMSB_e_3 = instructionInSlot_crossWriteDoEnq[3];
  wire       instructionInSlot_pendingCrossWriteLSB_d_3 = instructionInSlot_crossWriteDeqLSB[3];
  wire [4:0] instructionInSlot_pendingCrossWriteLSB_change_3 = instructionInSlot_pendingCrossWriteLSB_e_3 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingCrossWriteLSB_e_4 = instructionInSlot_crossWriteDoEnq[4];
  wire       instructionInSlot_pendingCrossWriteMSB_e_4 = instructionInSlot_crossWriteDoEnq[4];
  wire       instructionInSlot_pendingCrossWriteLSB_d_4 = instructionInSlot_crossWriteDeqLSB[4];
  wire [4:0] instructionInSlot_pendingCrossWriteLSB_change_4 = instructionInSlot_pendingCrossWriteLSB_e_4 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingCrossWriteLSB_e_5 = instructionInSlot_crossWriteDoEnq[5];
  wire       instructionInSlot_pendingCrossWriteMSB_e_5 = instructionInSlot_crossWriteDoEnq[5];
  wire       instructionInSlot_pendingCrossWriteLSB_d_5 = instructionInSlot_crossWriteDeqLSB[5];
  wire [4:0] instructionInSlot_pendingCrossWriteLSB_change_5 = instructionInSlot_pendingCrossWriteLSB_e_5 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingCrossWriteLSB_e_6 = instructionInSlot_crossWriteDoEnq[6];
  wire       instructionInSlot_pendingCrossWriteMSB_e_6 = instructionInSlot_crossWriteDoEnq[6];
  wire       instructionInSlot_pendingCrossWriteLSB_d_6 = instructionInSlot_crossWriteDeqLSB[6];
  wire [4:0] instructionInSlot_pendingCrossWriteLSB_change_6 = instructionInSlot_pendingCrossWriteLSB_e_6 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingCrossWriteLSB_e_7 = instructionInSlot_crossWriteDoEnq[7];
  wire       instructionInSlot_pendingCrossWriteMSB_e_7 = instructionInSlot_crossWriteDoEnq[7];
  wire       instructionInSlot_pendingCrossWriteLSB_d_7 = instructionInSlot_crossWriteDeqLSB[7];
  wire [4:0] instructionInSlot_pendingCrossWriteLSB_change_7 = instructionInSlot_pendingCrossWriteLSB_e_7 ? 5'h1 : 5'h1F;
  wire [1:0] instructionInSlot_pendingCrossWriteLSB_lo_lo = {|instructionInSlot_crossWriteTokenLSB_1, |instructionInSlot_crossWriteTokenLSB_0};
  wire [1:0] instructionInSlot_pendingCrossWriteLSB_lo_hi = {|instructionInSlot_crossWriteTokenLSB_3, |instructionInSlot_crossWriteTokenLSB_2};
  wire [3:0] instructionInSlot_pendingCrossWriteLSB_lo = {instructionInSlot_pendingCrossWriteLSB_lo_hi, instructionInSlot_pendingCrossWriteLSB_lo_lo};
  wire [1:0] instructionInSlot_pendingCrossWriteLSB_hi_lo = {|instructionInSlot_crossWriteTokenLSB_5, |instructionInSlot_crossWriteTokenLSB_4};
  wire [1:0] instructionInSlot_pendingCrossWriteLSB_hi_hi = {|instructionInSlot_crossWriteTokenLSB_7, |instructionInSlot_crossWriteTokenLSB_6};
  wire [3:0] instructionInSlot_pendingCrossWriteLSB_hi = {instructionInSlot_pendingCrossWriteLSB_hi_hi, instructionInSlot_pendingCrossWriteLSB_hi_lo};
  wire [7:0] instructionInSlot_pendingCrossWriteLSB = {instructionInSlot_pendingCrossWriteLSB_hi, instructionInSlot_pendingCrossWriteLSB_lo};
  wire       instructionInSlot_pendingCrossWriteMSB_d = instructionInSlot_crossWriteDeqMSB[0];
  wire [4:0] instructionInSlot_pendingCrossWriteMSB_change = instructionInSlot_pendingCrossWriteMSB_e ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingCrossWriteMSB_d_1 = instructionInSlot_crossWriteDeqMSB[1];
  wire [4:0] instructionInSlot_pendingCrossWriteMSB_change_1 = instructionInSlot_pendingCrossWriteMSB_e_1 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingCrossWriteMSB_d_2 = instructionInSlot_crossWriteDeqMSB[2];
  wire [4:0] instructionInSlot_pendingCrossWriteMSB_change_2 = instructionInSlot_pendingCrossWriteMSB_e_2 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingCrossWriteMSB_d_3 = instructionInSlot_crossWriteDeqMSB[3];
  wire [4:0] instructionInSlot_pendingCrossWriteMSB_change_3 = instructionInSlot_pendingCrossWriteMSB_e_3 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingCrossWriteMSB_d_4 = instructionInSlot_crossWriteDeqMSB[4];
  wire [4:0] instructionInSlot_pendingCrossWriteMSB_change_4 = instructionInSlot_pendingCrossWriteMSB_e_4 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingCrossWriteMSB_d_5 = instructionInSlot_crossWriteDeqMSB[5];
  wire [4:0] instructionInSlot_pendingCrossWriteMSB_change_5 = instructionInSlot_pendingCrossWriteMSB_e_5 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingCrossWriteMSB_d_6 = instructionInSlot_crossWriteDeqMSB[6];
  wire [4:0] instructionInSlot_pendingCrossWriteMSB_change_6 = instructionInSlot_pendingCrossWriteMSB_e_6 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingCrossWriteMSB_d_7 = instructionInSlot_crossWriteDeqMSB[7];
  wire [4:0] instructionInSlot_pendingCrossWriteMSB_change_7 = instructionInSlot_pendingCrossWriteMSB_e_7 ? 5'h1 : 5'h1F;
  wire [1:0] instructionInSlot_pendingCrossWriteMSB_lo_lo = {|instructionInSlot_crossWriteTokenMSB_1, |instructionInSlot_crossWriteTokenMSB_0};
  wire [1:0] instructionInSlot_pendingCrossWriteMSB_lo_hi = {|instructionInSlot_crossWriteTokenMSB_3, |instructionInSlot_crossWriteTokenMSB_2};
  wire [3:0] instructionInSlot_pendingCrossWriteMSB_lo = {instructionInSlot_pendingCrossWriteMSB_lo_hi, instructionInSlot_pendingCrossWriteMSB_lo_lo};
  wire [1:0] instructionInSlot_pendingCrossWriteMSB_hi_lo = {|instructionInSlot_crossWriteTokenMSB_5, |instructionInSlot_crossWriteTokenMSB_4};
  wire [1:0] instructionInSlot_pendingCrossWriteMSB_hi_hi = {|instructionInSlot_crossWriteTokenMSB_7, |instructionInSlot_crossWriteTokenMSB_6};
  wire [3:0] instructionInSlot_pendingCrossWriteMSB_hi = {instructionInSlot_pendingCrossWriteMSB_hi_hi, instructionInSlot_pendingCrossWriteMSB_hi_lo};
  wire [7:0] instructionInSlot_pendingCrossWriteMSB = {instructionInSlot_pendingCrossWriteMSB_hi, instructionInSlot_pendingCrossWriteMSB_lo};
  wire [7:0] instructionInSlot_responseDoEnq = enqReports_0_valid & ~enqReports_0_bits_sSendResponse ? instructionInSlot_enqOH : 8'h0;
  wire [7:0] instructionInSlot_responseDoDeq = responseReport_valid ? 8'h1 << responseReport_bits : 8'h0;
  wire [7:0] instructionInSlot_feedbackDoDeq = responseFeedbackReport_valid ? 8'h1 << responseFeedbackReport_bits : 8'h0;
  wire       instructionInSlot_pendingResponse_e = instructionInSlot_responseDoEnq[0];
  wire       instructionInSlot_pendingFeedback_e = instructionInSlot_responseDoEnq[0];
  wire       instructionInSlot_pendingResponse_d = instructionInSlot_responseDoDeq[0];
  wire [4:0] instructionInSlot_pendingResponse_change = instructionInSlot_pendingResponse_e ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingResponse_e_1 = instructionInSlot_responseDoEnq[1];
  wire       instructionInSlot_pendingFeedback_e_1 = instructionInSlot_responseDoEnq[1];
  wire       instructionInSlot_pendingResponse_d_1 = instructionInSlot_responseDoDeq[1];
  wire [4:0] instructionInSlot_pendingResponse_change_1 = instructionInSlot_pendingResponse_e_1 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingResponse_e_2 = instructionInSlot_responseDoEnq[2];
  wire       instructionInSlot_pendingFeedback_e_2 = instructionInSlot_responseDoEnq[2];
  wire       instructionInSlot_pendingResponse_d_2 = instructionInSlot_responseDoDeq[2];
  wire [4:0] instructionInSlot_pendingResponse_change_2 = instructionInSlot_pendingResponse_e_2 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingResponse_e_3 = instructionInSlot_responseDoEnq[3];
  wire       instructionInSlot_pendingFeedback_e_3 = instructionInSlot_responseDoEnq[3];
  wire       instructionInSlot_pendingResponse_d_3 = instructionInSlot_responseDoDeq[3];
  wire [4:0] instructionInSlot_pendingResponse_change_3 = instructionInSlot_pendingResponse_e_3 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingResponse_e_4 = instructionInSlot_responseDoEnq[4];
  wire       instructionInSlot_pendingFeedback_e_4 = instructionInSlot_responseDoEnq[4];
  wire       instructionInSlot_pendingResponse_d_4 = instructionInSlot_responseDoDeq[4];
  wire [4:0] instructionInSlot_pendingResponse_change_4 = instructionInSlot_pendingResponse_e_4 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingResponse_e_5 = instructionInSlot_responseDoEnq[5];
  wire       instructionInSlot_pendingFeedback_e_5 = instructionInSlot_responseDoEnq[5];
  wire       instructionInSlot_pendingResponse_d_5 = instructionInSlot_responseDoDeq[5];
  wire [4:0] instructionInSlot_pendingResponse_change_5 = instructionInSlot_pendingResponse_e_5 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingResponse_e_6 = instructionInSlot_responseDoEnq[6];
  wire       instructionInSlot_pendingFeedback_e_6 = instructionInSlot_responseDoEnq[6];
  wire       instructionInSlot_pendingResponse_d_6 = instructionInSlot_responseDoDeq[6];
  wire [4:0] instructionInSlot_pendingResponse_change_6 = instructionInSlot_pendingResponse_e_6 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingResponse_e_7 = instructionInSlot_responseDoEnq[7];
  wire       instructionInSlot_pendingFeedback_e_7 = instructionInSlot_responseDoEnq[7];
  wire       instructionInSlot_pendingResponse_d_7 = instructionInSlot_responseDoDeq[7];
  wire [4:0] instructionInSlot_pendingResponse_change_7 = instructionInSlot_pendingResponse_e_7 ? 5'h1 : 5'h1F;
  wire [1:0] instructionInSlot_pendingResponse_lo_lo = {|instructionInSlot_responseToken_1, |instructionInSlot_responseToken_0};
  wire [1:0] instructionInSlot_pendingResponse_lo_hi = {|instructionInSlot_responseToken_3, |instructionInSlot_responseToken_2};
  wire [3:0] instructionInSlot_pendingResponse_lo = {instructionInSlot_pendingResponse_lo_hi, instructionInSlot_pendingResponse_lo_lo};
  wire [1:0] instructionInSlot_pendingResponse_hi_lo = {|instructionInSlot_responseToken_5, |instructionInSlot_responseToken_4};
  wire [1:0] instructionInSlot_pendingResponse_hi_hi = {|instructionInSlot_responseToken_7, |instructionInSlot_responseToken_6};
  wire [3:0] instructionInSlot_pendingResponse_hi = {instructionInSlot_pendingResponse_hi_hi, instructionInSlot_pendingResponse_hi_lo};
  wire [7:0] instructionInSlot_pendingResponse = {instructionInSlot_pendingResponse_hi, instructionInSlot_pendingResponse_lo};
  wire       instructionInSlot_pendingFeedback_d = instructionInSlot_feedbackDoDeq[0];
  wire       instructionInSlot_pendingFeedback_c = maskUnitLastReport[0];
  wire [4:0] instructionInSlot_pendingFeedback_change = instructionInSlot_pendingFeedback_e ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingFeedback_d_1 = instructionInSlot_feedbackDoDeq[1];
  wire       instructionInSlot_pendingFeedback_c_1 = maskUnitLastReport[1];
  wire [4:0] instructionInSlot_pendingFeedback_change_1 = instructionInSlot_pendingFeedback_e_1 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingFeedback_d_2 = instructionInSlot_feedbackDoDeq[2];
  wire       instructionInSlot_pendingFeedback_c_2 = maskUnitLastReport[2];
  wire [4:0] instructionInSlot_pendingFeedback_change_2 = instructionInSlot_pendingFeedback_e_2 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingFeedback_d_3 = instructionInSlot_feedbackDoDeq[3];
  wire       instructionInSlot_pendingFeedback_c_3 = maskUnitLastReport[3];
  wire [4:0] instructionInSlot_pendingFeedback_change_3 = instructionInSlot_pendingFeedback_e_3 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingFeedback_d_4 = instructionInSlot_feedbackDoDeq[4];
  wire       instructionInSlot_pendingFeedback_c_4 = maskUnitLastReport[4];
  wire [4:0] instructionInSlot_pendingFeedback_change_4 = instructionInSlot_pendingFeedback_e_4 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingFeedback_d_5 = instructionInSlot_feedbackDoDeq[5];
  wire       instructionInSlot_pendingFeedback_c_5 = maskUnitLastReport[5];
  wire [4:0] instructionInSlot_pendingFeedback_change_5 = instructionInSlot_pendingFeedback_e_5 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingFeedback_d_6 = instructionInSlot_feedbackDoDeq[6];
  wire       instructionInSlot_pendingFeedback_c_6 = maskUnitLastReport[6];
  wire [4:0] instructionInSlot_pendingFeedback_change_6 = instructionInSlot_pendingFeedback_e_6 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingFeedback_d_7 = instructionInSlot_feedbackDoDeq[7];
  wire       instructionInSlot_pendingFeedback_c_7 = maskUnitLastReport[7];
  wire [4:0] instructionInSlot_pendingFeedback_change_7 = instructionInSlot_pendingFeedback_e_7 ? 5'h1 : 5'h1F;
  wire [1:0] instructionInSlot_pendingFeedback_lo_lo = {|instructionInSlot_feedbackToken_1, |instructionInSlot_feedbackToken_0};
  wire [1:0] instructionInSlot_pendingFeedback_lo_hi = {|instructionInSlot_feedbackToken_3, |instructionInSlot_feedbackToken_2};
  wire [3:0] instructionInSlot_pendingFeedback_lo = {instructionInSlot_pendingFeedback_lo_hi, instructionInSlot_pendingFeedback_lo_lo};
  wire [1:0] instructionInSlot_pendingFeedback_hi_lo = {|instructionInSlot_feedbackToken_5, |instructionInSlot_feedbackToken_4};
  wire [1:0] instructionInSlot_pendingFeedback_hi_hi = {|instructionInSlot_feedbackToken_7, |instructionInSlot_feedbackToken_6};
  wire [3:0] instructionInSlot_pendingFeedback_hi = {instructionInSlot_pendingFeedback_hi_hi, instructionInSlot_pendingFeedback_hi_lo};
  wire [7:0] instructionInSlot_pendingFeedback = {instructionInSlot_pendingFeedback_hi, instructionInSlot_pendingFeedback_lo};
  reg  [4:0] instructionInSlot_writeToken_0_1;
  reg  [4:0] instructionInSlot_writeToken_1_1;
  reg  [4:0] instructionInSlot_writeToken_2_1;
  reg  [4:0] instructionInSlot_writeToken_3_1;
  reg  [4:0] instructionInSlot_writeToken_4_1;
  reg  [4:0] instructionInSlot_writeToken_5_1;
  reg  [4:0] instructionInSlot_writeToken_6_1;
  reg  [4:0] instructionInSlot_writeToken_7_1;
  wire [7:0] instructionInSlot_enqOH_1 = 8'h1 << enqReports_1_bits_instructionIndex;
  wire [7:0] instructionInSlot_writeDoEnq_1 = enqReports_1_valid & ~enqReports_1_bits_decodeResult_sWrite & ~enqReports_1_bits_decodeResult_maskUnit ? instructionInSlot_enqOH_1 : 8'h0;
  wire [7:0] instructionInSlot_writeEnqSelect_1 = instructionInSlot_writeDoEnq_1;
  wire [7:0] instructionInSlot_writeDoDeq_1 = slotWriteReport_1_valid ? 8'h1 << slotWriteReport_1_bits : 8'h0;
  wire       instructionInSlot_pendingSlotWrite_e_8 = instructionInSlot_writeEnqSelect_1[0];
  wire       instructionInSlot_pendingSlotWrite_d_8 = instructionInSlot_writeDoDeq_1[0];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_8 = instructionInSlot_pendingSlotWrite_e_8 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_9 = instructionInSlot_writeEnqSelect_1[1];
  wire       instructionInSlot_pendingSlotWrite_d_9 = instructionInSlot_writeDoDeq_1[1];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_9 = instructionInSlot_pendingSlotWrite_e_9 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_10 = instructionInSlot_writeEnqSelect_1[2];
  wire       instructionInSlot_pendingSlotWrite_d_10 = instructionInSlot_writeDoDeq_1[2];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_10 = instructionInSlot_pendingSlotWrite_e_10 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_11 = instructionInSlot_writeEnqSelect_1[3];
  wire       instructionInSlot_pendingSlotWrite_d_11 = instructionInSlot_writeDoDeq_1[3];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_11 = instructionInSlot_pendingSlotWrite_e_11 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_12 = instructionInSlot_writeEnqSelect_1[4];
  wire       instructionInSlot_pendingSlotWrite_d_12 = instructionInSlot_writeDoDeq_1[4];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_12 = instructionInSlot_pendingSlotWrite_e_12 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_13 = instructionInSlot_writeEnqSelect_1[5];
  wire       instructionInSlot_pendingSlotWrite_d_13 = instructionInSlot_writeDoDeq_1[5];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_13 = instructionInSlot_pendingSlotWrite_e_13 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_14 = instructionInSlot_writeEnqSelect_1[6];
  wire       instructionInSlot_pendingSlotWrite_d_14 = instructionInSlot_writeDoDeq_1[6];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_14 = instructionInSlot_pendingSlotWrite_e_14 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_15 = instructionInSlot_writeEnqSelect_1[7];
  wire       instructionInSlot_pendingSlotWrite_d_15 = instructionInSlot_writeDoDeq_1[7];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_15 = instructionInSlot_pendingSlotWrite_e_15 ? 5'h1 : 5'h1F;
  wire [1:0] instructionInSlot_pendingSlotWrite_lo_lo_1 = {|instructionInSlot_writeToken_1_1, |instructionInSlot_writeToken_0_1};
  wire [1:0] instructionInSlot_pendingSlotWrite_lo_hi_1 = {|instructionInSlot_writeToken_3_1, |instructionInSlot_writeToken_2_1};
  wire [3:0] instructionInSlot_pendingSlotWrite_lo_1 = {instructionInSlot_pendingSlotWrite_lo_hi_1, instructionInSlot_pendingSlotWrite_lo_lo_1};
  wire [1:0] instructionInSlot_pendingSlotWrite_hi_lo_1 = {|instructionInSlot_writeToken_5_1, |instructionInSlot_writeToken_4_1};
  wire [1:0] instructionInSlot_pendingSlotWrite_hi_hi_1 = {|instructionInSlot_writeToken_7_1, |instructionInSlot_writeToken_6_1};
  wire [3:0] instructionInSlot_pendingSlotWrite_hi_1 = {instructionInSlot_pendingSlotWrite_hi_hi_1, instructionInSlot_pendingSlotWrite_hi_lo_1};
  wire [7:0] instructionInSlot_pendingSlotWrite_1 = {instructionInSlot_pendingSlotWrite_hi_1, instructionInSlot_pendingSlotWrite_lo_1};
  reg  [4:0] instructionInSlot_writeToken_0_2;
  reg  [4:0] instructionInSlot_writeToken_1_2;
  reg  [4:0] instructionInSlot_writeToken_2_2;
  reg  [4:0] instructionInSlot_writeToken_3_2;
  reg  [4:0] instructionInSlot_writeToken_4_2;
  reg  [4:0] instructionInSlot_writeToken_5_2;
  reg  [4:0] instructionInSlot_writeToken_6_2;
  reg  [4:0] instructionInSlot_writeToken_7_2;
  wire [7:0] instructionInSlot_enqOH_2 = 8'h1 << enqReports_2_bits_instructionIndex;
  wire [7:0] instructionInSlot_writeDoEnq_2 = enqReports_2_valid & ~enqReports_2_bits_decodeResult_sWrite & ~enqReports_2_bits_decodeResult_maskUnit ? instructionInSlot_enqOH_2 : 8'h0;
  wire [7:0] instructionInSlot_writeEnqSelect_2 = instructionInSlot_writeDoEnq_2;
  wire [7:0] instructionInSlot_writeDoDeq_2 = slotWriteReport_2_valid ? 8'h1 << slotWriteReport_2_bits : 8'h0;
  wire       instructionInSlot_pendingSlotWrite_e_16 = instructionInSlot_writeEnqSelect_2[0];
  wire       instructionInSlot_pendingSlotWrite_d_16 = instructionInSlot_writeDoDeq_2[0];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_16 = instructionInSlot_pendingSlotWrite_e_16 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_17 = instructionInSlot_writeEnqSelect_2[1];
  wire       instructionInSlot_pendingSlotWrite_d_17 = instructionInSlot_writeDoDeq_2[1];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_17 = instructionInSlot_pendingSlotWrite_e_17 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_18 = instructionInSlot_writeEnqSelect_2[2];
  wire       instructionInSlot_pendingSlotWrite_d_18 = instructionInSlot_writeDoDeq_2[2];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_18 = instructionInSlot_pendingSlotWrite_e_18 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_19 = instructionInSlot_writeEnqSelect_2[3];
  wire       instructionInSlot_pendingSlotWrite_d_19 = instructionInSlot_writeDoDeq_2[3];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_19 = instructionInSlot_pendingSlotWrite_e_19 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_20 = instructionInSlot_writeEnqSelect_2[4];
  wire       instructionInSlot_pendingSlotWrite_d_20 = instructionInSlot_writeDoDeq_2[4];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_20 = instructionInSlot_pendingSlotWrite_e_20 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_21 = instructionInSlot_writeEnqSelect_2[5];
  wire       instructionInSlot_pendingSlotWrite_d_21 = instructionInSlot_writeDoDeq_2[5];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_21 = instructionInSlot_pendingSlotWrite_e_21 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_22 = instructionInSlot_writeEnqSelect_2[6];
  wire       instructionInSlot_pendingSlotWrite_d_22 = instructionInSlot_writeDoDeq_2[6];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_22 = instructionInSlot_pendingSlotWrite_e_22 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_23 = instructionInSlot_writeEnqSelect_2[7];
  wire       instructionInSlot_pendingSlotWrite_d_23 = instructionInSlot_writeDoDeq_2[7];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_23 = instructionInSlot_pendingSlotWrite_e_23 ? 5'h1 : 5'h1F;
  wire [1:0] instructionInSlot_pendingSlotWrite_lo_lo_2 = {|instructionInSlot_writeToken_1_2, |instructionInSlot_writeToken_0_2};
  wire [1:0] instructionInSlot_pendingSlotWrite_lo_hi_2 = {|instructionInSlot_writeToken_3_2, |instructionInSlot_writeToken_2_2};
  wire [3:0] instructionInSlot_pendingSlotWrite_lo_2 = {instructionInSlot_pendingSlotWrite_lo_hi_2, instructionInSlot_pendingSlotWrite_lo_lo_2};
  wire [1:0] instructionInSlot_pendingSlotWrite_hi_lo_2 = {|instructionInSlot_writeToken_5_2, |instructionInSlot_writeToken_4_2};
  wire [1:0] instructionInSlot_pendingSlotWrite_hi_hi_2 = {|instructionInSlot_writeToken_7_2, |instructionInSlot_writeToken_6_2};
  wire [3:0] instructionInSlot_pendingSlotWrite_hi_2 = {instructionInSlot_pendingSlotWrite_hi_hi_2, instructionInSlot_pendingSlotWrite_hi_lo_2};
  wire [7:0] instructionInSlot_pendingSlotWrite_2 = {instructionInSlot_pendingSlotWrite_hi_2, instructionInSlot_pendingSlotWrite_lo_2};
  reg  [4:0] instructionInSlot_writeToken_0_3;
  reg  [4:0] instructionInSlot_writeToken_1_3;
  reg  [4:0] instructionInSlot_writeToken_2_3;
  reg  [4:0] instructionInSlot_writeToken_3_3;
  reg  [4:0] instructionInSlot_writeToken_4_3;
  reg  [4:0] instructionInSlot_writeToken_5_3;
  reg  [4:0] instructionInSlot_writeToken_6_3;
  reg  [4:0] instructionInSlot_writeToken_7_3;
  wire [7:0] instructionInSlot_enqOH_3 = 8'h1 << enqReports_3_bits_instructionIndex;
  wire [7:0] instructionInSlot_writeDoEnq_3 = enqReports_3_valid & ~enqReports_3_bits_decodeResult_sWrite & ~enqReports_3_bits_decodeResult_maskUnit ? instructionInSlot_enqOH_3 : 8'h0;
  wire [7:0] instructionInSlot_writeEnqSelect_3 = instructionInSlot_writeDoEnq_3;
  wire [7:0] instructionInSlot_writeDoDeq_3 = slotWriteReport_3_valid ? 8'h1 << slotWriteReport_3_bits : 8'h0;
  wire       instructionInSlot_pendingSlotWrite_e_24 = instructionInSlot_writeEnqSelect_3[0];
  wire       instructionInSlot_pendingSlotWrite_d_24 = instructionInSlot_writeDoDeq_3[0];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_24 = instructionInSlot_pendingSlotWrite_e_24 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_25 = instructionInSlot_writeEnqSelect_3[1];
  wire       instructionInSlot_pendingSlotWrite_d_25 = instructionInSlot_writeDoDeq_3[1];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_25 = instructionInSlot_pendingSlotWrite_e_25 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_26 = instructionInSlot_writeEnqSelect_3[2];
  wire       instructionInSlot_pendingSlotWrite_d_26 = instructionInSlot_writeDoDeq_3[2];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_26 = instructionInSlot_pendingSlotWrite_e_26 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_27 = instructionInSlot_writeEnqSelect_3[3];
  wire       instructionInSlot_pendingSlotWrite_d_27 = instructionInSlot_writeDoDeq_3[3];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_27 = instructionInSlot_pendingSlotWrite_e_27 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_28 = instructionInSlot_writeEnqSelect_3[4];
  wire       instructionInSlot_pendingSlotWrite_d_28 = instructionInSlot_writeDoDeq_3[4];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_28 = instructionInSlot_pendingSlotWrite_e_28 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_29 = instructionInSlot_writeEnqSelect_3[5];
  wire       instructionInSlot_pendingSlotWrite_d_29 = instructionInSlot_writeDoDeq_3[5];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_29 = instructionInSlot_pendingSlotWrite_e_29 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_30 = instructionInSlot_writeEnqSelect_3[6];
  wire       instructionInSlot_pendingSlotWrite_d_30 = instructionInSlot_writeDoDeq_3[6];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_30 = instructionInSlot_pendingSlotWrite_e_30 ? 5'h1 : 5'h1F;
  wire       instructionInSlot_pendingSlotWrite_e_31 = instructionInSlot_writeEnqSelect_3[7];
  wire       instructionInSlot_pendingSlotWrite_d_31 = instructionInSlot_writeDoDeq_3[7];
  wire [4:0] instructionInSlot_pendingSlotWrite_change_31 = instructionInSlot_pendingSlotWrite_e_31 ? 5'h1 : 5'h1F;
  wire [1:0] instructionInSlot_pendingSlotWrite_lo_lo_3 = {|instructionInSlot_writeToken_1_3, |instructionInSlot_writeToken_0_3};
  wire [1:0] instructionInSlot_pendingSlotWrite_lo_hi_3 = {|instructionInSlot_writeToken_3_3, |instructionInSlot_writeToken_2_3};
  wire [3:0] instructionInSlot_pendingSlotWrite_lo_3 = {instructionInSlot_pendingSlotWrite_lo_hi_3, instructionInSlot_pendingSlotWrite_lo_lo_3};
  wire [1:0] instructionInSlot_pendingSlotWrite_hi_lo_3 = {|instructionInSlot_writeToken_5_3, |instructionInSlot_writeToken_4_3};
  wire [1:0] instructionInSlot_pendingSlotWrite_hi_hi_3 = {|instructionInSlot_writeToken_7_3, |instructionInSlot_writeToken_6_3};
  wire [3:0] instructionInSlot_pendingSlotWrite_hi_3 = {instructionInSlot_pendingSlotWrite_hi_hi_3, instructionInSlot_pendingSlotWrite_hi_lo_3};
  wire [7:0] instructionInSlot_pendingSlotWrite_3 = {instructionInSlot_pendingSlotWrite_hi_3, instructionInSlot_pendingSlotWrite_lo_3};
  wire [7:0] instructionInSlot =
    instructionInSlot_pendingSlotWrite | instructionInSlot_pendingCrossWriteLSB | instructionInSlot_pendingCrossWriteMSB | instructionInSlot_pendingResponse | instructionInSlot_pendingFeedback | instructionInSlot_pendingSlotWrite_1
    | instructionInSlot_pendingSlotWrite_2 | instructionInSlot_pendingSlotWrite_3;
  reg  [4:0] writePipeToken_0;
  reg  [4:0] writePipeToken_1;
  reg  [4:0] writePipeToken_2;
  reg  [4:0] writePipeToken_3;
  reg  [4:0] writePipeToken_4;
  reg  [4:0] writePipeToken_5;
  reg  [4:0] writePipeToken_6;
  reg  [4:0] writePipeToken_7;
  wire [7:0] writePipeEnq = writePipeEnqReport_valid ? 8'h1 << writePipeEnqReport_bits : 8'h0;
  wire [7:0] writePipeDeq = writePipeDeqReport_valid ? 8'h1 << writePipeDeqReport_bits : 8'h0;
  wire       instructionInWritePipe_e = writePipeEnq[0];
  wire       instructionInWritePipe_d = writePipeDeq[0];
  wire [4:0] instructionInWritePipe_change = instructionInWritePipe_e ? 5'h1 : 5'h1F;
  wire       instructionInWritePipe_e_1 = writePipeEnq[1];
  wire       instructionInWritePipe_d_1 = writePipeDeq[1];
  wire [4:0] instructionInWritePipe_change_1 = instructionInWritePipe_e_1 ? 5'h1 : 5'h1F;
  wire       instructionInWritePipe_e_2 = writePipeEnq[2];
  wire       instructionInWritePipe_d_2 = writePipeDeq[2];
  wire [4:0] instructionInWritePipe_change_2 = instructionInWritePipe_e_2 ? 5'h1 : 5'h1F;
  wire       instructionInWritePipe_e_3 = writePipeEnq[3];
  wire       instructionInWritePipe_d_3 = writePipeDeq[3];
  wire [4:0] instructionInWritePipe_change_3 = instructionInWritePipe_e_3 ? 5'h1 : 5'h1F;
  wire       instructionInWritePipe_e_4 = writePipeEnq[4];
  wire       instructionInWritePipe_d_4 = writePipeDeq[4];
  wire [4:0] instructionInWritePipe_change_4 = instructionInWritePipe_e_4 ? 5'h1 : 5'h1F;
  wire       instructionInWritePipe_e_5 = writePipeEnq[5];
  wire       instructionInWritePipe_d_5 = writePipeDeq[5];
  wire [4:0] instructionInWritePipe_change_5 = instructionInWritePipe_e_5 ? 5'h1 : 5'h1F;
  wire       instructionInWritePipe_e_6 = writePipeEnq[6];
  wire       instructionInWritePipe_d_6 = writePipeDeq[6];
  wire [4:0] instructionInWritePipe_change_6 = instructionInWritePipe_e_6 ? 5'h1 : 5'h1F;
  wire       instructionInWritePipe_e_7 = writePipeEnq[7];
  wire       instructionInWritePipe_d_7 = writePipeDeq[7];
  wire [4:0] instructionInWritePipe_change_7 = instructionInWritePipe_e_7 ? 5'h1 : 5'h1F;
  wire [1:0] instructionInWritePipe_lo_lo = {|writePipeToken_1, |writePipeToken_0};
  wire [1:0] instructionInWritePipe_lo_hi = {|writePipeToken_3, |writePipeToken_2};
  wire [3:0] instructionInWritePipe_lo = {instructionInWritePipe_lo_hi, instructionInWritePipe_lo_lo};
  wire [1:0] instructionInWritePipe_hi_lo = {|writePipeToken_5, |writePipeToken_4};
  wire [1:0] instructionInWritePipe_hi_hi = {|writePipeToken_7, |writePipeToken_6};
  wire [3:0] instructionInWritePipe_hi = {instructionInWritePipe_hi_hi, instructionInWritePipe_hi_lo};
  wire [7:0] instructionInWritePipe = {instructionInWritePipe_hi, instructionInWritePipe_lo};
  reg  [4:0] topWriteToken_0;
  reg  [4:0] topWriteToken_1;
  reg  [4:0] topWriteToken_2;
  reg  [4:0] topWriteToken_3;
  reg  [4:0] topWriteToken_4;
  reg  [4:0] topWriteToken_5;
  reg  [4:0] topWriteToken_6;
  reg  [4:0] topWriteToken_7;
  wire [7:0] topWriteDoEnq = topWriteEnq_valid ? 8'h1 << topWriteEnq_bits : 8'h0;
  wire [7:0] topWriteDoDeq = topWriteDeq_valid ? 8'h1 << topWriteDeq_bits : 8'h0;
  wire       topWrite_e = topWriteDoEnq[0];
  wire       topWrite_d = topWriteDoDeq[0];
  wire [4:0] topWrite_change = topWrite_e ? 5'h1 : 5'h1F;
  wire       topWrite_e_1 = topWriteDoEnq[1];
  wire       topWrite_d_1 = topWriteDoDeq[1];
  wire [4:0] topWrite_change_1 = topWrite_e_1 ? 5'h1 : 5'h1F;
  wire       topWrite_e_2 = topWriteDoEnq[2];
  wire       topWrite_d_2 = topWriteDoDeq[2];
  wire [4:0] topWrite_change_2 = topWrite_e_2 ? 5'h1 : 5'h1F;
  wire       topWrite_e_3 = topWriteDoEnq[3];
  wire       topWrite_d_3 = topWriteDoDeq[3];
  wire [4:0] topWrite_change_3 = topWrite_e_3 ? 5'h1 : 5'h1F;
  wire       topWrite_e_4 = topWriteDoEnq[4];
  wire       topWrite_d_4 = topWriteDoDeq[4];
  wire [4:0] topWrite_change_4 = topWrite_e_4 ? 5'h1 : 5'h1F;
  wire       topWrite_e_5 = topWriteDoEnq[5];
  wire       topWrite_d_5 = topWriteDoDeq[5];
  wire [4:0] topWrite_change_5 = topWrite_e_5 ? 5'h1 : 5'h1F;
  wire       topWrite_e_6 = topWriteDoEnq[6];
  wire       topWrite_d_6 = topWriteDoDeq[6];
  wire [4:0] topWrite_change_6 = topWrite_e_6 ? 5'h1 : 5'h1F;
  wire       topWrite_e_7 = topWriteDoEnq[7];
  wire       topWrite_d_7 = topWriteDoDeq[7];
  wire [4:0] topWrite_change_7 = topWrite_e_7 ? 5'h1 : 5'h1F;
  wire [1:0] topWrite_lo_lo = {|topWriteToken_1, |topWriteToken_0};
  wire [1:0] topWrite_lo_hi = {|topWriteToken_3, |topWriteToken_2};
  wire [3:0] topWrite_lo = {topWrite_lo_hi, topWrite_lo_lo};
  wire [1:0] topWrite_hi_lo = {|topWriteToken_5, |topWriteToken_4};
  wire [1:0] topWrite_hi_hi = {|topWriteToken_7, |topWriteToken_6};
  wire [3:0] topWrite_hi = {topWrite_hi_hi, topWrite_hi_lo};
  wire [7:0] topWrite = {topWrite_hi, topWrite_lo};
  wire [7:0] _dataInWritePipe_output = instructionInWritePipe | topWrite;
  always @(posedge clock) begin
    if (reset) begin
      instructionInSlot_writeToken_0 <= 5'h0;
      instructionInSlot_writeToken_1 <= 5'h0;
      instructionInSlot_writeToken_2 <= 5'h0;
      instructionInSlot_writeToken_3 <= 5'h0;
      instructionInSlot_writeToken_4 <= 5'h0;
      instructionInSlot_writeToken_5 <= 5'h0;
      instructionInSlot_writeToken_6 <= 5'h0;
      instructionInSlot_writeToken_7 <= 5'h0;
      instructionInSlot_responseToken_0 <= 5'h0;
      instructionInSlot_responseToken_1 <= 5'h0;
      instructionInSlot_responseToken_2 <= 5'h0;
      instructionInSlot_responseToken_3 <= 5'h0;
      instructionInSlot_responseToken_4 <= 5'h0;
      instructionInSlot_responseToken_5 <= 5'h0;
      instructionInSlot_responseToken_6 <= 5'h0;
      instructionInSlot_responseToken_7 <= 5'h0;
      instructionInSlot_feedbackToken_0 <= 5'h0;
      instructionInSlot_feedbackToken_1 <= 5'h0;
      instructionInSlot_feedbackToken_2 <= 5'h0;
      instructionInSlot_feedbackToken_3 <= 5'h0;
      instructionInSlot_feedbackToken_4 <= 5'h0;
      instructionInSlot_feedbackToken_5 <= 5'h0;
      instructionInSlot_feedbackToken_6 <= 5'h0;
      instructionInSlot_feedbackToken_7 <= 5'h0;
      instructionInSlot_crossWriteTokenLSB_0 <= 5'h0;
      instructionInSlot_crossWriteTokenLSB_1 <= 5'h0;
      instructionInSlot_crossWriteTokenLSB_2 <= 5'h0;
      instructionInSlot_crossWriteTokenLSB_3 <= 5'h0;
      instructionInSlot_crossWriteTokenLSB_4 <= 5'h0;
      instructionInSlot_crossWriteTokenLSB_5 <= 5'h0;
      instructionInSlot_crossWriteTokenLSB_6 <= 5'h0;
      instructionInSlot_crossWriteTokenLSB_7 <= 5'h0;
      instructionInSlot_crossWriteTokenMSB_0 <= 5'h0;
      instructionInSlot_crossWriteTokenMSB_1 <= 5'h0;
      instructionInSlot_crossWriteTokenMSB_2 <= 5'h0;
      instructionInSlot_crossWriteTokenMSB_3 <= 5'h0;
      instructionInSlot_crossWriteTokenMSB_4 <= 5'h0;
      instructionInSlot_crossWriteTokenMSB_5 <= 5'h0;
      instructionInSlot_crossWriteTokenMSB_6 <= 5'h0;
      instructionInSlot_crossWriteTokenMSB_7 <= 5'h0;
      instructionInSlot_writeToken_0_1 <= 5'h0;
      instructionInSlot_writeToken_1_1 <= 5'h0;
      instructionInSlot_writeToken_2_1 <= 5'h0;
      instructionInSlot_writeToken_3_1 <= 5'h0;
      instructionInSlot_writeToken_4_1 <= 5'h0;
      instructionInSlot_writeToken_5_1 <= 5'h0;
      instructionInSlot_writeToken_6_1 <= 5'h0;
      instructionInSlot_writeToken_7_1 <= 5'h0;
      instructionInSlot_writeToken_0_2 <= 5'h0;
      instructionInSlot_writeToken_1_2 <= 5'h0;
      instructionInSlot_writeToken_2_2 <= 5'h0;
      instructionInSlot_writeToken_3_2 <= 5'h0;
      instructionInSlot_writeToken_4_2 <= 5'h0;
      instructionInSlot_writeToken_5_2 <= 5'h0;
      instructionInSlot_writeToken_6_2 <= 5'h0;
      instructionInSlot_writeToken_7_2 <= 5'h0;
      instructionInSlot_writeToken_0_3 <= 5'h0;
      instructionInSlot_writeToken_1_3 <= 5'h0;
      instructionInSlot_writeToken_2_3 <= 5'h0;
      instructionInSlot_writeToken_3_3 <= 5'h0;
      instructionInSlot_writeToken_4_3 <= 5'h0;
      instructionInSlot_writeToken_5_3 <= 5'h0;
      instructionInSlot_writeToken_6_3 <= 5'h0;
      instructionInSlot_writeToken_7_3 <= 5'h0;
      writePipeToken_0 <= 5'h0;
      writePipeToken_1 <= 5'h0;
      writePipeToken_2 <= 5'h0;
      writePipeToken_3 <= 5'h0;
      writePipeToken_4 <= 5'h0;
      writePipeToken_5 <= 5'h0;
      writePipeToken_6 <= 5'h0;
      writePipeToken_7 <= 5'h0;
      topWriteToken_0 <= 5'h0;
      topWriteToken_1 <= 5'h0;
      topWriteToken_2 <= 5'h0;
      topWriteToken_3 <= 5'h0;
      topWriteToken_4 <= 5'h0;
      topWriteToken_5 <= 5'h0;
      topWriteToken_6 <= 5'h0;
      topWriteToken_7 <= 5'h0;
    end
    else begin
      if (instructionInSlot_pendingSlotWrite_e ^ instructionInSlot_pendingSlotWrite_d)
        instructionInSlot_writeToken_0 <= instructionInSlot_writeToken_0 + instructionInSlot_pendingSlotWrite_change;
      if (instructionInSlot_pendingSlotWrite_e_1 ^ instructionInSlot_pendingSlotWrite_d_1)
        instructionInSlot_writeToken_1 <= instructionInSlot_writeToken_1 + instructionInSlot_pendingSlotWrite_change_1;
      if (instructionInSlot_pendingSlotWrite_e_2 ^ instructionInSlot_pendingSlotWrite_d_2)
        instructionInSlot_writeToken_2 <= instructionInSlot_writeToken_2 + instructionInSlot_pendingSlotWrite_change_2;
      if (instructionInSlot_pendingSlotWrite_e_3 ^ instructionInSlot_pendingSlotWrite_d_3)
        instructionInSlot_writeToken_3 <= instructionInSlot_writeToken_3 + instructionInSlot_pendingSlotWrite_change_3;
      if (instructionInSlot_pendingSlotWrite_e_4 ^ instructionInSlot_pendingSlotWrite_d_4)
        instructionInSlot_writeToken_4 <= instructionInSlot_writeToken_4 + instructionInSlot_pendingSlotWrite_change_4;
      if (instructionInSlot_pendingSlotWrite_e_5 ^ instructionInSlot_pendingSlotWrite_d_5)
        instructionInSlot_writeToken_5 <= instructionInSlot_writeToken_5 + instructionInSlot_pendingSlotWrite_change_5;
      if (instructionInSlot_pendingSlotWrite_e_6 ^ instructionInSlot_pendingSlotWrite_d_6)
        instructionInSlot_writeToken_6 <= instructionInSlot_writeToken_6 + instructionInSlot_pendingSlotWrite_change_6;
      if (instructionInSlot_pendingSlotWrite_e_7 ^ instructionInSlot_pendingSlotWrite_d_7)
        instructionInSlot_writeToken_7 <= instructionInSlot_writeToken_7 + instructionInSlot_pendingSlotWrite_change_7;
      if (instructionInSlot_pendingResponse_e ^ instructionInSlot_pendingResponse_d)
        instructionInSlot_responseToken_0 <= instructionInSlot_responseToken_0 + instructionInSlot_pendingResponse_change;
      if (instructionInSlot_pendingResponse_e_1 ^ instructionInSlot_pendingResponse_d_1)
        instructionInSlot_responseToken_1 <= instructionInSlot_responseToken_1 + instructionInSlot_pendingResponse_change_1;
      if (instructionInSlot_pendingResponse_e_2 ^ instructionInSlot_pendingResponse_d_2)
        instructionInSlot_responseToken_2 <= instructionInSlot_responseToken_2 + instructionInSlot_pendingResponse_change_2;
      if (instructionInSlot_pendingResponse_e_3 ^ instructionInSlot_pendingResponse_d_3)
        instructionInSlot_responseToken_3 <= instructionInSlot_responseToken_3 + instructionInSlot_pendingResponse_change_3;
      if (instructionInSlot_pendingResponse_e_4 ^ instructionInSlot_pendingResponse_d_4)
        instructionInSlot_responseToken_4 <= instructionInSlot_responseToken_4 + instructionInSlot_pendingResponse_change_4;
      if (instructionInSlot_pendingResponse_e_5 ^ instructionInSlot_pendingResponse_d_5)
        instructionInSlot_responseToken_5 <= instructionInSlot_responseToken_5 + instructionInSlot_pendingResponse_change_5;
      if (instructionInSlot_pendingResponse_e_6 ^ instructionInSlot_pendingResponse_d_6)
        instructionInSlot_responseToken_6 <= instructionInSlot_responseToken_6 + instructionInSlot_pendingResponse_change_6;
      if (instructionInSlot_pendingResponse_e_7 ^ instructionInSlot_pendingResponse_d_7)
        instructionInSlot_responseToken_7 <= instructionInSlot_responseToken_7 + instructionInSlot_pendingResponse_change_7;
      if (instructionInSlot_pendingFeedback_c)
        instructionInSlot_feedbackToken_0 <= 5'h0;
      else if ((instructionInSlot_pendingFeedback_e ^ instructionInSlot_pendingFeedback_d) & (|{instructionInSlot_pendingFeedback_e, instructionInSlot_feedbackToken_0}))
        instructionInSlot_feedbackToken_0 <= instructionInSlot_feedbackToken_0 + instructionInSlot_pendingFeedback_change;
      if (instructionInSlot_pendingFeedback_c_1)
        instructionInSlot_feedbackToken_1 <= 5'h0;
      else if ((instructionInSlot_pendingFeedback_e_1 ^ instructionInSlot_pendingFeedback_d_1) & (|{instructionInSlot_pendingFeedback_e_1, instructionInSlot_feedbackToken_1}))
        instructionInSlot_feedbackToken_1 <= instructionInSlot_feedbackToken_1 + instructionInSlot_pendingFeedback_change_1;
      if (instructionInSlot_pendingFeedback_c_2)
        instructionInSlot_feedbackToken_2 <= 5'h0;
      else if ((instructionInSlot_pendingFeedback_e_2 ^ instructionInSlot_pendingFeedback_d_2) & (|{instructionInSlot_pendingFeedback_e_2, instructionInSlot_feedbackToken_2}))
        instructionInSlot_feedbackToken_2 <= instructionInSlot_feedbackToken_2 + instructionInSlot_pendingFeedback_change_2;
      if (instructionInSlot_pendingFeedback_c_3)
        instructionInSlot_feedbackToken_3 <= 5'h0;
      else if ((instructionInSlot_pendingFeedback_e_3 ^ instructionInSlot_pendingFeedback_d_3) & (|{instructionInSlot_pendingFeedback_e_3, instructionInSlot_feedbackToken_3}))
        instructionInSlot_feedbackToken_3 <= instructionInSlot_feedbackToken_3 + instructionInSlot_pendingFeedback_change_3;
      if (instructionInSlot_pendingFeedback_c_4)
        instructionInSlot_feedbackToken_4 <= 5'h0;
      else if ((instructionInSlot_pendingFeedback_e_4 ^ instructionInSlot_pendingFeedback_d_4) & (|{instructionInSlot_pendingFeedback_e_4, instructionInSlot_feedbackToken_4}))
        instructionInSlot_feedbackToken_4 <= instructionInSlot_feedbackToken_4 + instructionInSlot_pendingFeedback_change_4;
      if (instructionInSlot_pendingFeedback_c_5)
        instructionInSlot_feedbackToken_5 <= 5'h0;
      else if ((instructionInSlot_pendingFeedback_e_5 ^ instructionInSlot_pendingFeedback_d_5) & (|{instructionInSlot_pendingFeedback_e_5, instructionInSlot_feedbackToken_5}))
        instructionInSlot_feedbackToken_5 <= instructionInSlot_feedbackToken_5 + instructionInSlot_pendingFeedback_change_5;
      if (instructionInSlot_pendingFeedback_c_6)
        instructionInSlot_feedbackToken_6 <= 5'h0;
      else if ((instructionInSlot_pendingFeedback_e_6 ^ instructionInSlot_pendingFeedback_d_6) & (|{instructionInSlot_pendingFeedback_e_6, instructionInSlot_feedbackToken_6}))
        instructionInSlot_feedbackToken_6 <= instructionInSlot_feedbackToken_6 + instructionInSlot_pendingFeedback_change_6;
      if (instructionInSlot_pendingFeedback_c_7)
        instructionInSlot_feedbackToken_7 <= 5'h0;
      else if ((instructionInSlot_pendingFeedback_e_7 ^ instructionInSlot_pendingFeedback_d_7) & (|{instructionInSlot_pendingFeedback_e_7, instructionInSlot_feedbackToken_7}))
        instructionInSlot_feedbackToken_7 <= instructionInSlot_feedbackToken_7 + instructionInSlot_pendingFeedback_change_7;
      if (instructionInSlot_pendingCrossWriteLSB_e ^ instructionInSlot_pendingCrossWriteLSB_d)
        instructionInSlot_crossWriteTokenLSB_0 <= instructionInSlot_crossWriteTokenLSB_0 + instructionInSlot_pendingCrossWriteLSB_change;
      if (instructionInSlot_pendingCrossWriteLSB_e_1 ^ instructionInSlot_pendingCrossWriteLSB_d_1)
        instructionInSlot_crossWriteTokenLSB_1 <= instructionInSlot_crossWriteTokenLSB_1 + instructionInSlot_pendingCrossWriteLSB_change_1;
      if (instructionInSlot_pendingCrossWriteLSB_e_2 ^ instructionInSlot_pendingCrossWriteLSB_d_2)
        instructionInSlot_crossWriteTokenLSB_2 <= instructionInSlot_crossWriteTokenLSB_2 + instructionInSlot_pendingCrossWriteLSB_change_2;
      if (instructionInSlot_pendingCrossWriteLSB_e_3 ^ instructionInSlot_pendingCrossWriteLSB_d_3)
        instructionInSlot_crossWriteTokenLSB_3 <= instructionInSlot_crossWriteTokenLSB_3 + instructionInSlot_pendingCrossWriteLSB_change_3;
      if (instructionInSlot_pendingCrossWriteLSB_e_4 ^ instructionInSlot_pendingCrossWriteLSB_d_4)
        instructionInSlot_crossWriteTokenLSB_4 <= instructionInSlot_crossWriteTokenLSB_4 + instructionInSlot_pendingCrossWriteLSB_change_4;
      if (instructionInSlot_pendingCrossWriteLSB_e_5 ^ instructionInSlot_pendingCrossWriteLSB_d_5)
        instructionInSlot_crossWriteTokenLSB_5 <= instructionInSlot_crossWriteTokenLSB_5 + instructionInSlot_pendingCrossWriteLSB_change_5;
      if (instructionInSlot_pendingCrossWriteLSB_e_6 ^ instructionInSlot_pendingCrossWriteLSB_d_6)
        instructionInSlot_crossWriteTokenLSB_6 <= instructionInSlot_crossWriteTokenLSB_6 + instructionInSlot_pendingCrossWriteLSB_change_6;
      if (instructionInSlot_pendingCrossWriteLSB_e_7 ^ instructionInSlot_pendingCrossWriteLSB_d_7)
        instructionInSlot_crossWriteTokenLSB_7 <= instructionInSlot_crossWriteTokenLSB_7 + instructionInSlot_pendingCrossWriteLSB_change_7;
      if (instructionInSlot_pendingCrossWriteMSB_e ^ instructionInSlot_pendingCrossWriteMSB_d)
        instructionInSlot_crossWriteTokenMSB_0 <= instructionInSlot_crossWriteTokenMSB_0 + instructionInSlot_pendingCrossWriteMSB_change;
      if (instructionInSlot_pendingCrossWriteMSB_e_1 ^ instructionInSlot_pendingCrossWriteMSB_d_1)
        instructionInSlot_crossWriteTokenMSB_1 <= instructionInSlot_crossWriteTokenMSB_1 + instructionInSlot_pendingCrossWriteMSB_change_1;
      if (instructionInSlot_pendingCrossWriteMSB_e_2 ^ instructionInSlot_pendingCrossWriteMSB_d_2)
        instructionInSlot_crossWriteTokenMSB_2 <= instructionInSlot_crossWriteTokenMSB_2 + instructionInSlot_pendingCrossWriteMSB_change_2;
      if (instructionInSlot_pendingCrossWriteMSB_e_3 ^ instructionInSlot_pendingCrossWriteMSB_d_3)
        instructionInSlot_crossWriteTokenMSB_3 <= instructionInSlot_crossWriteTokenMSB_3 + instructionInSlot_pendingCrossWriteMSB_change_3;
      if (instructionInSlot_pendingCrossWriteMSB_e_4 ^ instructionInSlot_pendingCrossWriteMSB_d_4)
        instructionInSlot_crossWriteTokenMSB_4 <= instructionInSlot_crossWriteTokenMSB_4 + instructionInSlot_pendingCrossWriteMSB_change_4;
      if (instructionInSlot_pendingCrossWriteMSB_e_5 ^ instructionInSlot_pendingCrossWriteMSB_d_5)
        instructionInSlot_crossWriteTokenMSB_5 <= instructionInSlot_crossWriteTokenMSB_5 + instructionInSlot_pendingCrossWriteMSB_change_5;
      if (instructionInSlot_pendingCrossWriteMSB_e_6 ^ instructionInSlot_pendingCrossWriteMSB_d_6)
        instructionInSlot_crossWriteTokenMSB_6 <= instructionInSlot_crossWriteTokenMSB_6 + instructionInSlot_pendingCrossWriteMSB_change_6;
      if (instructionInSlot_pendingCrossWriteMSB_e_7 ^ instructionInSlot_pendingCrossWriteMSB_d_7)
        instructionInSlot_crossWriteTokenMSB_7 <= instructionInSlot_crossWriteTokenMSB_7 + instructionInSlot_pendingCrossWriteMSB_change_7;
      if (instructionInSlot_pendingSlotWrite_e_8 ^ instructionInSlot_pendingSlotWrite_d_8)
        instructionInSlot_writeToken_0_1 <= instructionInSlot_writeToken_0_1 + instructionInSlot_pendingSlotWrite_change_8;
      if (instructionInSlot_pendingSlotWrite_e_9 ^ instructionInSlot_pendingSlotWrite_d_9)
        instructionInSlot_writeToken_1_1 <= instructionInSlot_writeToken_1_1 + instructionInSlot_pendingSlotWrite_change_9;
      if (instructionInSlot_pendingSlotWrite_e_10 ^ instructionInSlot_pendingSlotWrite_d_10)
        instructionInSlot_writeToken_2_1 <= instructionInSlot_writeToken_2_1 + instructionInSlot_pendingSlotWrite_change_10;
      if (instructionInSlot_pendingSlotWrite_e_11 ^ instructionInSlot_pendingSlotWrite_d_11)
        instructionInSlot_writeToken_3_1 <= instructionInSlot_writeToken_3_1 + instructionInSlot_pendingSlotWrite_change_11;
      if (instructionInSlot_pendingSlotWrite_e_12 ^ instructionInSlot_pendingSlotWrite_d_12)
        instructionInSlot_writeToken_4_1 <= instructionInSlot_writeToken_4_1 + instructionInSlot_pendingSlotWrite_change_12;
      if (instructionInSlot_pendingSlotWrite_e_13 ^ instructionInSlot_pendingSlotWrite_d_13)
        instructionInSlot_writeToken_5_1 <= instructionInSlot_writeToken_5_1 + instructionInSlot_pendingSlotWrite_change_13;
      if (instructionInSlot_pendingSlotWrite_e_14 ^ instructionInSlot_pendingSlotWrite_d_14)
        instructionInSlot_writeToken_6_1 <= instructionInSlot_writeToken_6_1 + instructionInSlot_pendingSlotWrite_change_14;
      if (instructionInSlot_pendingSlotWrite_e_15 ^ instructionInSlot_pendingSlotWrite_d_15)
        instructionInSlot_writeToken_7_1 <= instructionInSlot_writeToken_7_1 + instructionInSlot_pendingSlotWrite_change_15;
      if (instructionInSlot_pendingSlotWrite_e_16 ^ instructionInSlot_pendingSlotWrite_d_16)
        instructionInSlot_writeToken_0_2 <= instructionInSlot_writeToken_0_2 + instructionInSlot_pendingSlotWrite_change_16;
      if (instructionInSlot_pendingSlotWrite_e_17 ^ instructionInSlot_pendingSlotWrite_d_17)
        instructionInSlot_writeToken_1_2 <= instructionInSlot_writeToken_1_2 + instructionInSlot_pendingSlotWrite_change_17;
      if (instructionInSlot_pendingSlotWrite_e_18 ^ instructionInSlot_pendingSlotWrite_d_18)
        instructionInSlot_writeToken_2_2 <= instructionInSlot_writeToken_2_2 + instructionInSlot_pendingSlotWrite_change_18;
      if (instructionInSlot_pendingSlotWrite_e_19 ^ instructionInSlot_pendingSlotWrite_d_19)
        instructionInSlot_writeToken_3_2 <= instructionInSlot_writeToken_3_2 + instructionInSlot_pendingSlotWrite_change_19;
      if (instructionInSlot_pendingSlotWrite_e_20 ^ instructionInSlot_pendingSlotWrite_d_20)
        instructionInSlot_writeToken_4_2 <= instructionInSlot_writeToken_4_2 + instructionInSlot_pendingSlotWrite_change_20;
      if (instructionInSlot_pendingSlotWrite_e_21 ^ instructionInSlot_pendingSlotWrite_d_21)
        instructionInSlot_writeToken_5_2 <= instructionInSlot_writeToken_5_2 + instructionInSlot_pendingSlotWrite_change_21;
      if (instructionInSlot_pendingSlotWrite_e_22 ^ instructionInSlot_pendingSlotWrite_d_22)
        instructionInSlot_writeToken_6_2 <= instructionInSlot_writeToken_6_2 + instructionInSlot_pendingSlotWrite_change_22;
      if (instructionInSlot_pendingSlotWrite_e_23 ^ instructionInSlot_pendingSlotWrite_d_23)
        instructionInSlot_writeToken_7_2 <= instructionInSlot_writeToken_7_2 + instructionInSlot_pendingSlotWrite_change_23;
      if (instructionInSlot_pendingSlotWrite_e_24 ^ instructionInSlot_pendingSlotWrite_d_24)
        instructionInSlot_writeToken_0_3 <= instructionInSlot_writeToken_0_3 + instructionInSlot_pendingSlotWrite_change_24;
      if (instructionInSlot_pendingSlotWrite_e_25 ^ instructionInSlot_pendingSlotWrite_d_25)
        instructionInSlot_writeToken_1_3 <= instructionInSlot_writeToken_1_3 + instructionInSlot_pendingSlotWrite_change_25;
      if (instructionInSlot_pendingSlotWrite_e_26 ^ instructionInSlot_pendingSlotWrite_d_26)
        instructionInSlot_writeToken_2_3 <= instructionInSlot_writeToken_2_3 + instructionInSlot_pendingSlotWrite_change_26;
      if (instructionInSlot_pendingSlotWrite_e_27 ^ instructionInSlot_pendingSlotWrite_d_27)
        instructionInSlot_writeToken_3_3 <= instructionInSlot_writeToken_3_3 + instructionInSlot_pendingSlotWrite_change_27;
      if (instructionInSlot_pendingSlotWrite_e_28 ^ instructionInSlot_pendingSlotWrite_d_28)
        instructionInSlot_writeToken_4_3 <= instructionInSlot_writeToken_4_3 + instructionInSlot_pendingSlotWrite_change_28;
      if (instructionInSlot_pendingSlotWrite_e_29 ^ instructionInSlot_pendingSlotWrite_d_29)
        instructionInSlot_writeToken_5_3 <= instructionInSlot_writeToken_5_3 + instructionInSlot_pendingSlotWrite_change_29;
      if (instructionInSlot_pendingSlotWrite_e_30 ^ instructionInSlot_pendingSlotWrite_d_30)
        instructionInSlot_writeToken_6_3 <= instructionInSlot_writeToken_6_3 + instructionInSlot_pendingSlotWrite_change_30;
      if (instructionInSlot_pendingSlotWrite_e_31 ^ instructionInSlot_pendingSlotWrite_d_31)
        instructionInSlot_writeToken_7_3 <= instructionInSlot_writeToken_7_3 + instructionInSlot_pendingSlotWrite_change_31;
      if (instructionInWritePipe_e ^ instructionInWritePipe_d)
        writePipeToken_0 <= writePipeToken_0 + instructionInWritePipe_change;
      if (instructionInWritePipe_e_1 ^ instructionInWritePipe_d_1)
        writePipeToken_1 <= writePipeToken_1 + instructionInWritePipe_change_1;
      if (instructionInWritePipe_e_2 ^ instructionInWritePipe_d_2)
        writePipeToken_2 <= writePipeToken_2 + instructionInWritePipe_change_2;
      if (instructionInWritePipe_e_3 ^ instructionInWritePipe_d_3)
        writePipeToken_3 <= writePipeToken_3 + instructionInWritePipe_change_3;
      if (instructionInWritePipe_e_4 ^ instructionInWritePipe_d_4)
        writePipeToken_4 <= writePipeToken_4 + instructionInWritePipe_change_4;
      if (instructionInWritePipe_e_5 ^ instructionInWritePipe_d_5)
        writePipeToken_5 <= writePipeToken_5 + instructionInWritePipe_change_5;
      if (instructionInWritePipe_e_6 ^ instructionInWritePipe_d_6)
        writePipeToken_6 <= writePipeToken_6 + instructionInWritePipe_change_6;
      if (instructionInWritePipe_e_7 ^ instructionInWritePipe_d_7)
        writePipeToken_7 <= writePipeToken_7 + instructionInWritePipe_change_7;
      if (topWrite_e ^ topWrite_d)
        topWriteToken_0 <= topWriteToken_0 + topWrite_change;
      if (topWrite_e_1 ^ topWrite_d_1)
        topWriteToken_1 <= topWriteToken_1 + topWrite_change_1;
      if (topWrite_e_2 ^ topWrite_d_2)
        topWriteToken_2 <= topWriteToken_2 + topWrite_change_2;
      if (topWrite_e_3 ^ topWrite_d_3)
        topWriteToken_3 <= topWriteToken_3 + topWrite_change_3;
      if (topWrite_e_4 ^ topWrite_d_4)
        topWriteToken_4 <= topWriteToken_4 + topWrite_change_4;
      if (topWrite_e_5 ^ topWrite_d_5)
        topWriteToken_5 <= topWriteToken_5 + topWrite_change_5;
      if (topWrite_e_6 ^ topWrite_d_6)
        topWriteToken_6 <= topWriteToken_6 + topWrite_change_6;
      if (topWrite_e_7 ^ topWrite_d_7)
        topWriteToken_7 <= topWriteToken_7 + topWrite_change_7;
    end
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:12];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [3:0] i = 4'h0; i < 4'hD; i += 4'h1) begin
          _RANDOM[i] = `RANDOM;
        end
        instructionInSlot_writeToken_0 = _RANDOM[4'h0][4:0];
        instructionInSlot_writeToken_1 = _RANDOM[4'h0][9:5];
        instructionInSlot_writeToken_2 = _RANDOM[4'h0][14:10];
        instructionInSlot_writeToken_3 = _RANDOM[4'h0][19:15];
        instructionInSlot_writeToken_4 = _RANDOM[4'h0][24:20];
        instructionInSlot_writeToken_5 = _RANDOM[4'h0][29:25];
        instructionInSlot_writeToken_6 = {_RANDOM[4'h0][31:30], _RANDOM[4'h1][2:0]};
        instructionInSlot_writeToken_7 = _RANDOM[4'h1][7:3];
        instructionInSlot_responseToken_0 = _RANDOM[4'h1][12:8];
        instructionInSlot_responseToken_1 = _RANDOM[4'h1][17:13];
        instructionInSlot_responseToken_2 = _RANDOM[4'h1][22:18];
        instructionInSlot_responseToken_3 = _RANDOM[4'h1][27:23];
        instructionInSlot_responseToken_4 = {_RANDOM[4'h1][31:28], _RANDOM[4'h2][0]};
        instructionInSlot_responseToken_5 = _RANDOM[4'h2][5:1];
        instructionInSlot_responseToken_6 = _RANDOM[4'h2][10:6];
        instructionInSlot_responseToken_7 = _RANDOM[4'h2][15:11];
        instructionInSlot_feedbackToken_0 = _RANDOM[4'h2][20:16];
        instructionInSlot_feedbackToken_1 = _RANDOM[4'h2][25:21];
        instructionInSlot_feedbackToken_2 = _RANDOM[4'h2][30:26];
        instructionInSlot_feedbackToken_3 = {_RANDOM[4'h2][31], _RANDOM[4'h3][3:0]};
        instructionInSlot_feedbackToken_4 = _RANDOM[4'h3][8:4];
        instructionInSlot_feedbackToken_5 = _RANDOM[4'h3][13:9];
        instructionInSlot_feedbackToken_6 = _RANDOM[4'h3][18:14];
        instructionInSlot_feedbackToken_7 = _RANDOM[4'h3][23:19];
        instructionInSlot_crossWriteTokenLSB_0 = _RANDOM[4'h3][28:24];
        instructionInSlot_crossWriteTokenLSB_1 = {_RANDOM[4'h3][31:29], _RANDOM[4'h4][1:0]};
        instructionInSlot_crossWriteTokenLSB_2 = _RANDOM[4'h4][6:2];
        instructionInSlot_crossWriteTokenLSB_3 = _RANDOM[4'h4][11:7];
        instructionInSlot_crossWriteTokenLSB_4 = _RANDOM[4'h4][16:12];
        instructionInSlot_crossWriteTokenLSB_5 = _RANDOM[4'h4][21:17];
        instructionInSlot_crossWriteTokenLSB_6 = _RANDOM[4'h4][26:22];
        instructionInSlot_crossWriteTokenLSB_7 = _RANDOM[4'h4][31:27];
        instructionInSlot_crossWriteTokenMSB_0 = _RANDOM[4'h5][4:0];
        instructionInSlot_crossWriteTokenMSB_1 = _RANDOM[4'h5][9:5];
        instructionInSlot_crossWriteTokenMSB_2 = _RANDOM[4'h5][14:10];
        instructionInSlot_crossWriteTokenMSB_3 = _RANDOM[4'h5][19:15];
        instructionInSlot_crossWriteTokenMSB_4 = _RANDOM[4'h5][24:20];
        instructionInSlot_crossWriteTokenMSB_5 = _RANDOM[4'h5][29:25];
        instructionInSlot_crossWriteTokenMSB_6 = {_RANDOM[4'h5][31:30], _RANDOM[4'h6][2:0]};
        instructionInSlot_crossWriteTokenMSB_7 = _RANDOM[4'h6][7:3];
        instructionInSlot_writeToken_0_1 = _RANDOM[4'h6][12:8];
        instructionInSlot_writeToken_1_1 = _RANDOM[4'h6][17:13];
        instructionInSlot_writeToken_2_1 = _RANDOM[4'h6][22:18];
        instructionInSlot_writeToken_3_1 = _RANDOM[4'h6][27:23];
        instructionInSlot_writeToken_4_1 = {_RANDOM[4'h6][31:28], _RANDOM[4'h7][0]};
        instructionInSlot_writeToken_5_1 = _RANDOM[4'h7][5:1];
        instructionInSlot_writeToken_6_1 = _RANDOM[4'h7][10:6];
        instructionInSlot_writeToken_7_1 = _RANDOM[4'h7][15:11];
        instructionInSlot_writeToken_0_2 = _RANDOM[4'h7][20:16];
        instructionInSlot_writeToken_1_2 = _RANDOM[4'h7][25:21];
        instructionInSlot_writeToken_2_2 = _RANDOM[4'h7][30:26];
        instructionInSlot_writeToken_3_2 = {_RANDOM[4'h7][31], _RANDOM[4'h8][3:0]};
        instructionInSlot_writeToken_4_2 = _RANDOM[4'h8][8:4];
        instructionInSlot_writeToken_5_2 = _RANDOM[4'h8][13:9];
        instructionInSlot_writeToken_6_2 = _RANDOM[4'h8][18:14];
        instructionInSlot_writeToken_7_2 = _RANDOM[4'h8][23:19];
        instructionInSlot_writeToken_0_3 = _RANDOM[4'h8][28:24];
        instructionInSlot_writeToken_1_3 = {_RANDOM[4'h8][31:29], _RANDOM[4'h9][1:0]};
        instructionInSlot_writeToken_2_3 = _RANDOM[4'h9][6:2];
        instructionInSlot_writeToken_3_3 = _RANDOM[4'h9][11:7];
        instructionInSlot_writeToken_4_3 = _RANDOM[4'h9][16:12];
        instructionInSlot_writeToken_5_3 = _RANDOM[4'h9][21:17];
        instructionInSlot_writeToken_6_3 = _RANDOM[4'h9][26:22];
        instructionInSlot_writeToken_7_3 = _RANDOM[4'h9][31:27];
        writePipeToken_0 = _RANDOM[4'hA][4:0];
        writePipeToken_1 = _RANDOM[4'hA][9:5];
        writePipeToken_2 = _RANDOM[4'hA][14:10];
        writePipeToken_3 = _RANDOM[4'hA][19:15];
        writePipeToken_4 = _RANDOM[4'hA][24:20];
        writePipeToken_5 = _RANDOM[4'hA][29:25];
        writePipeToken_6 = {_RANDOM[4'hA][31:30], _RANDOM[4'hB][2:0]};
        writePipeToken_7 = _RANDOM[4'hB][7:3];
        topWriteToken_0 = _RANDOM[4'hB][12:8];
        topWriteToken_1 = _RANDOM[4'hB][17:13];
        topWriteToken_2 = _RANDOM[4'hB][22:18];
        topWriteToken_3 = _RANDOM[4'hB][27:23];
        topWriteToken_4 = {_RANDOM[4'hB][31:28], _RANDOM[4'hC][0]};
        topWriteToken_5 = _RANDOM[4'hC][5:1];
        topWriteToken_6 = _RANDOM[4'hC][10:6];
        topWriteToken_7 = _RANDOM[4'hC][15:11];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  assign instructionValid = _dataInWritePipe_output | instructionInSlot;
  assign dataInWritePipe = _dataInWritePipe_output;
endmodule

