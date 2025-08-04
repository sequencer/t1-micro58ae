
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
module CrossReadUnit(
  input         clock,
                reset,
  output        dataInputLSB_ready,
  input         dataInputLSB_valid,
  input  [31:0] dataInputLSB_bits,
  output        dataInputMSB_ready,
  input         dataInputMSB_valid,
  input  [31:0] dataInputMSB_bits,
  input  [10:0] dataGroup,
  output [10:0] currentGroup,
  output        readBusDequeue_0_ready,
  input         readBusDequeue_0_valid,
  input  [31:0] readBusDequeue_0_bits_data,
  output        readBusDequeue_1_ready,
  input         readBusDequeue_1_valid,
  input  [31:0] readBusDequeue_1_bits_data,
  input         readBusRequest_0_ready,
  output        readBusRequest_0_valid,
  output [31:0] readBusRequest_0_bits_data,
  input         readBusRequest_1_ready,
  output        readBusRequest_1_valid,
  output [31:0] readBusRequest_1_bits_data,
  input         crossReadDequeue_ready,
  output        crossReadDequeue_valid,
  output [63:0] crossReadDequeue_bits,
  output        crossReadStageFree
);

  wire        dataInputLSB_valid_0 = dataInputLSB_valid;
  wire [31:0] dataInputLSB_bits_0 = dataInputLSB_bits;
  wire        dataInputMSB_valid_0 = dataInputMSB_valid;
  wire [31:0] dataInputMSB_bits_0 = dataInputMSB_bits;
  wire        readBusDequeue_0_valid_0 = readBusDequeue_0_valid;
  wire [31:0] readBusDequeue_0_bits_data_0 = readBusDequeue_0_bits_data;
  wire        readBusDequeue_1_valid_0 = readBusDequeue_1_valid;
  wire [31:0] readBusDequeue_1_bits_data_0 = readBusDequeue_1_bits_data;
  wire        readBusRequest_0_ready_0 = readBusRequest_0_ready;
  wire        readBusRequest_1_ready_0 = readBusRequest_1_ready;
  wire        crossReadDequeue_ready_0 = crossReadDequeue_ready;
  reg         stageValid;
  reg         sSendCrossReadResultLSB;
  reg         sSendCrossReadResultMSB;
  reg         wCrossReadLSB;
  reg         wCrossReadMSB;
  reg  [31:0] sendDataVec_0;
  wire [31:0] readBusRequest_0_bits_data_0 = sendDataVec_0;
  reg  [31:0] sendDataVec_1;
  wire [31:0] readBusRequest_1_bits_data_0 = sendDataVec_1;
  reg  [10:0] groupCounter;
  reg  [31:0] receiveDataVec_0;
  reg  [31:0] receiveDataVec_1;
  wire        readBusRequest_0_valid_0 = stageValid & ~sSendCrossReadResultLSB;
  wire        readBusRequest_1_valid_0 = stageValid & ~sSendCrossReadResultMSB;
  wire        readBusDequeue_0_ready_0 = ~wCrossReadLSB;
  wire        readBusDequeue_1_ready_0 = ~wCrossReadMSB;
  wire        _crossReadStageFree_T_1 = sSendCrossReadResultLSB & sSendCrossReadResultMSB;
  wire        _GEN_1 = wCrossReadLSB & wCrossReadMSB;
  wire        allStateReady = _crossReadStageFree_T_1 & _GEN_1;
  wire        stageReady = ~stageValid | allStateReady & crossReadDequeue_ready_0;
  wire        allSourceValid = dataInputLSB_valid_0 & dataInputMSB_valid_0;
  wire        _dataInputMSB_ready_T = stageReady & allSourceValid;
  wire        dataInputLSB_ready_0;
  assign dataInputLSB_ready_0 = _dataInputMSB_ready_T;
  wire        dataInputMSB_ready_0;
  assign dataInputMSB_ready_0 = _dataInputMSB_ready_T;
  wire        enqueueFire;
  assign enqueueFire = _dataInputMSB_ready_T;
  wire [63:0] crossReadDequeue_bits_0 = {receiveDataVec_1, receiveDataVec_0};
  wire        crossReadDequeue_valid_0 = allStateReady & stageValid;
  always @(posedge clock) begin
    if (reset) begin
      stageValid <= 1'h0;
      sSendCrossReadResultLSB <= 1'h1;
      sSendCrossReadResultMSB <= 1'h1;
      wCrossReadLSB <= 1'h1;
      wCrossReadMSB <= 1'h1;
      sendDataVec_0 <= 32'h0;
      sendDataVec_1 <= 32'h0;
      groupCounter <= 11'h0;
      receiveDataVec_0 <= 32'h0;
      receiveDataVec_1 <= 32'h0;
    end
    else begin
      automatic logic _GEN = readBusDequeue_0_ready_0 & readBusDequeue_0_valid_0;
      automatic logic _GEN_0 = readBusDequeue_1_ready_0 & readBusDequeue_1_valid_0;
      if (enqueueFire ^ crossReadDequeue_ready_0 & crossReadDequeue_valid_0)
        stageValid <= enqueueFire;
      sSendCrossReadResultLSB <= ~enqueueFire & (readBusRequest_0_ready_0 & readBusRequest_0_valid_0 | sSendCrossReadResultLSB);
      sSendCrossReadResultMSB <= ~enqueueFire & (readBusRequest_1_ready_0 & readBusRequest_1_valid_0 | sSendCrossReadResultMSB);
      wCrossReadLSB <= ~enqueueFire & (_GEN | wCrossReadLSB);
      wCrossReadMSB <= ~enqueueFire & (_GEN_0 | wCrossReadMSB);
      if (enqueueFire) begin
        sendDataVec_0 <= dataInputLSB_bits_0;
        sendDataVec_1 <= dataInputMSB_bits_0;
        groupCounter <= dataGroup;
      end
      if (_GEN)
        receiveDataVec_0 <= readBusDequeue_0_bits_data_0;
      if (_GEN_0)
        receiveDataVec_1 <= readBusDequeue_1_bits_data_0;
    end
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:4];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [2:0] i = 3'h0; i < 3'h5; i += 3'h1) begin
          _RANDOM[i] = `RANDOM;
        end
        stageValid = _RANDOM[3'h0][0];
        sSendCrossReadResultLSB = _RANDOM[3'h0][1];
        sSendCrossReadResultMSB = _RANDOM[3'h0][2];
        wCrossReadLSB = _RANDOM[3'h0][3];
        wCrossReadMSB = _RANDOM[3'h0][4];
        sendDataVec_0 = {_RANDOM[3'h0][31:5], _RANDOM[3'h1][4:0]};
        sendDataVec_1 = {_RANDOM[3'h1][31:5], _RANDOM[3'h2][4:0]};
        groupCounter = _RANDOM[3'h2][15:5];
        receiveDataVec_0 = {_RANDOM[3'h2][31:16], _RANDOM[3'h3][15:0]};
        receiveDataVec_1 = {_RANDOM[3'h3][31:16], _RANDOM[3'h4][15:0]};
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  assign dataInputLSB_ready = dataInputLSB_ready_0;
  assign dataInputMSB_ready = dataInputMSB_ready_0;
  assign currentGroup = groupCounter;
  assign readBusDequeue_0_ready = readBusDequeue_0_ready_0;
  assign readBusDequeue_1_ready = readBusDequeue_1_ready_0;
  assign readBusRequest_0_valid = readBusRequest_0_valid_0;
  assign readBusRequest_0_bits_data = readBusRequest_0_bits_data_0;
  assign readBusRequest_1_valid = readBusRequest_1_valid_0;
  assign readBusRequest_1_bits_data = readBusRequest_1_bits_data_0;
  assign crossReadDequeue_valid = crossReadDequeue_valid_0;
  assign crossReadDequeue_bits = crossReadDequeue_bits_0;
  assign crossReadStageFree = ~stageValid & _crossReadStageFree_T_1 & _GEN_1;
endmodule

