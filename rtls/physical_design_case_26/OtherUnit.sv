
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
module OtherUnit(
  input         clock,
                reset,
                requestIO_valid,
  input  [1:0]  requestIO_bits_tag,
  input  [31:0] requestIO_bits_src_0,
                requestIO_bits_src_1,
                requestIO_bits_src_2,
                requestIO_bits_src_3,
  input  [9:0]  requestIO_bits_popInit,
  input  [3:0]  requestIO_bits_opcode,
  input  [4:0]  requestIO_bits_groupIndex,
  input  [1:0]  requestIO_bits_laneIndex,
                requestIO_bits_executeIndex,
  input         requestIO_bits_sign,
                requestIO_bits_mask,
                requestIO_bits_maskType,
  input  [1:0]  requestIO_bits_vSew,
                requestIO_bits_vxrm,
  input         requestIO_bits_narrow,
  output        responseIO_valid,
  output [1:0]  responseIO_bits_tag,
  output [31:0] responseIO_bits_data,
  output        responseIO_bits_clipFail,
                responseIO_bits_ffoSuccess
);

  wire [31:0]  _popCount_resp;
  wire         _ffo_resp_valid;
  wire [31:0]  _ffo_resp_bits;
  wire         requestIO_valid_0 = requestIO_valid;
  wire [1:0]   requestIO_bits_tag_0 = requestIO_bits_tag;
  wire [31:0]  requestIO_bits_src_0_0 = requestIO_bits_src_0;
  wire [31:0]  requestIO_bits_src_1_0 = requestIO_bits_src_1;
  wire [31:0]  requestIO_bits_src_2_0 = requestIO_bits_src_2;
  wire [31:0]  requestIO_bits_src_3_0 = requestIO_bits_src_3;
  wire [9:0]   requestIO_bits_popInit_0 = requestIO_bits_popInit;
  wire [3:0]   requestIO_bits_opcode_0 = requestIO_bits_opcode;
  wire [4:0]   requestIO_bits_groupIndex_0 = requestIO_bits_groupIndex;
  wire [1:0]   requestIO_bits_laneIndex_0 = requestIO_bits_laneIndex;
  wire [1:0]   requestIO_bits_executeIndex_0 = requestIO_bits_executeIndex;
  wire         requestIO_bits_sign_0 = requestIO_bits_sign;
  wire         requestIO_bits_mask_0 = requestIO_bits_mask;
  wire         requestIO_bits_maskType_0 = requestIO_bits_maskType;
  wire [1:0]   requestIO_bits_vSew_0 = requestIO_bits_vSew;
  wire [1:0]   requestIO_bits_vxrm_0 = requestIO_bits_vxrm;
  wire         requestIO_bits_narrow_0 = requestIO_bits_narrow;
  wire [1:0]   response_tag = 2'h0;
  wire         requestIO_ready = 1'h1;
  wire         responseIO_ready = 1'h1;
  wire         requestIO_bits_complete = 1'h0;
  wire         request_pipeResponse_valid;
  wire [1:0]   request_pipeResponse_bits_tag;
  wire [31:0]  request_pipeResponse_bits_data;
  wire         request_pipeResponse_bits_clipFail;
  wire         request_pipeResponse_bits_ffoSuccess;
  reg  [1:0]   requestReg_tag;
  wire [1:0]   request_responseWire_tag = requestReg_tag;
  reg  [31:0]  requestReg_src_0;
  reg  [31:0]  requestReg_src_1;
  reg  [31:0]  requestReg_src_2;
  reg  [31:0]  requestReg_src_3;
  reg  [9:0]   requestReg_popInit;
  reg  [3:0]   requestReg_opcode;
  reg  [4:0]   requestReg_groupIndex;
  reg  [1:0]   requestReg_laneIndex;
  reg  [1:0]   requestReg_executeIndex;
  reg          requestReg_sign;
  reg          requestReg_mask;
  reg          requestReg_maskType;
  reg  [1:0]   requestReg_vSew;
  reg  [1:0]   requestReg_vxrm;
  reg          requestReg_narrow;
  reg          requestRegValid;
  wire         vfuRequestFire = requestRegValid;
  wire         request_responseValidWire = requestRegValid;
  wire [31:0]  result;
  wire         response_clipFail;
  wire         response_ffoSuccess;
  wire [1:0]   request_responseWire_lo = {response_clipFail, response_ffoSuccess};
  wire [31:0]  response_data;
  wire [33:0]  request_responseWire_hi = {2'h0, response_data};
  wire         request_responseWire_ffoSuccess = request_responseWire_lo[0];
  wire         request_responseWire_clipFail = request_responseWire_lo[1];
  wire [31:0]  request_responseWire_data = request_responseWire_hi[31:0];
  reg          request_pipeResponse_pipe_v;
  assign request_pipeResponse_valid = request_pipeResponse_pipe_v;
  reg  [1:0]   request_pipeResponse_pipe_b_tag;
  assign request_pipeResponse_bits_tag = request_pipeResponse_pipe_b_tag;
  reg  [31:0]  request_pipeResponse_pipe_b_data;
  assign request_pipeResponse_bits_data = request_pipeResponse_pipe_b_data;
  reg          request_pipeResponse_pipe_b_clipFail;
  assign request_pipeResponse_bits_clipFail = request_pipeResponse_pipe_b_clipFail;
  reg          request_pipeResponse_pipe_b_ffoSuccess;
  assign request_pipeResponse_bits_ffoSuccess = request_pipeResponse_pipe_b_ffoSuccess;
  wire         responseIO_valid_0 = request_pipeResponse_valid;
  wire [1:0]   responseIO_bits_tag_0 = request_pipeResponse_bits_tag;
  wire [31:0]  responseIO_bits_data_0 = request_pipeResponse_bits_data;
  wire         responseIO_bits_clipFail_0 = request_pipeResponse_bits_clipFail;
  wire         responseIO_bits_ffoSuccess_0 = request_pipeResponse_bits_ffoSuccess;
  wire [63:0]  request_lo = {requestReg_src_1, requestReg_src_0};
  wire [63:0]  request_hi = {requestReg_src_3, requestReg_src_2};
  wire [3:0]   request_lo_lo_hi = {requestReg_vSew, requestReg_vxrm};
  wire [4:0]   request_lo_lo = {request_lo_lo_hi, requestReg_narrow};
  wire [1:0]   request_lo_hi_lo = {1'h0, requestReg_maskType};
  wire [1:0]   request_lo_hi_hi = {requestReg_sign, requestReg_mask};
  wire [3:0]   request_lo_hi = {request_lo_hi_hi, request_lo_hi_lo};
  wire [8:0]   request_lo_1 = {request_lo_hi, request_lo_lo};
  wire [6:0]   request_hi_lo_hi = {requestReg_groupIndex, requestReg_laneIndex};
  wire [8:0]   request_hi_lo = {request_hi_lo_hi, requestReg_executeIndex};
  wire [13:0]  request_hi_hi_lo = {requestReg_popInit, requestReg_opcode};
  wire [129:0] request_hi_hi_hi = {requestReg_tag, request_hi, request_lo};
  wire [143:0] request_hi_hi = {request_hi_hi_hi, request_hi_hi_lo};
  wire [152:0] request_hi_1 = {request_hi_hi, request_hi_lo};
  wire         request_narrow = request_lo_1[0];
  wire [1:0]   request_vxrm = request_lo_1[2:1];
  wire [1:0]   request_vSew = request_lo_1[4:3];
  wire         request_maskType = request_lo_1[5];
  wire         request_complete = request_lo_1[6];
  wire         request_mask = request_lo_1[7];
  wire         request_sign = request_lo_1[8];
  wire [1:0]   request_executeIndex = request_hi_1[1:0];
  wire [1:0]   request_laneIndex = request_hi_1[3:2];
  wire [4:0]   request_groupIndex = request_hi_1[8:4];
  wire [3:0]   request_opcode = request_hi_1[12:9];
  wire [9:0]   request_popInit = request_hi_1[22:13];
  wire [31:0]  request_src_0 = request_hi_1[54:23];
  wire [31:0]  request_src_1 = request_hi_1[86:55];
  wire [31:0]  request_src_2 = request_hi_1[118:87];
  wire [31:0]  request_src_3 = request_hi_1[150:119];
  wire [1:0]   request_tag = request_hi_1[152:151];
  wire [3:0]   _vSewOH_T_1 = 4'h1 << request_vSew >> request_narrow;
  wire [2:0]   vSewOH = _vSewOH_T_1[2:0];
  wire [15:0]  _opcodeOH_T = 16'h1 << request_opcode;
  wire [9:0]   opcodeOH = _opcodeOH_T[9:0];
  wire         isffo = |(opcodeOH[3:0]);
  wire [5:0]   originalOpcodeOH = opcodeOH[9:4];
  wire         signValue = request_src_1[31] & request_sign;
  wire [31:0]  signExtend = {32{signValue}};
  wire [5:0]   clipSize = {(vSewOH[1] ? {1'h0, request_src_0[4]} : 2'h0) | (vSewOH[2] ? request_src_0[5:4] : 2'h0), request_src_0[3:0]};
  wire [15:0]  clipMask_lo = {{8{|(vSewOH[2:1])}}, 8'hFF};
  wire [15:0]  clipMask_hi = {16{vSewOH[2]}};
  wire [31:0]  clipMask = {clipMask_hi, clipMask_lo};
  wire [31:0]  largestClipResult = clipMask >> request_sign;
  wire [15:0]  clipMaskRemainder_lo = {{8{vSewOH[0]}}, 8'h0};
  wire [15:0]  clipMaskRemainder_hi = {{8{~(vSewOH[2])}}, {8{~(vSewOH[2])}}};
  wire [31:0]  clipMaskRemainder = {clipMaskRemainder_hi, clipMaskRemainder_lo};
  wire [63:0]  _GEN = {58'h0, clipSize};
  wire [63:0]  roundTail = 64'h1 << _GEN;
  wire [62:0]  lostMSB = roundTail[63:1];
  wire [63:0]  roundMask = roundTail - 64'h1;
  wire         vds1 = |(lostMSB[31:0] & request_src_1);
  wire         vLostLSB = |(roundMask[32:1] & request_src_1);
  wire         vd = |(roundTail[31:0] & request_src_1);
  wire [3:0]   _roundR_T = 4'h1 << request_vxrm;
  wire         roundR = _roundR_T[0] & vds1 | _roundR_T[1] & vds1 & (vLostLSB | vd) | _roundR_T[3] & ~vd & (vds1 | vLostLSB);
  wire [63:0]  _roundResult_T_1 = {signExtend, request_src_1} >> _GEN;
  wire [31:0]  roundResult = _roundResult_T_1[31:0] + {31'h0, roundR};
  wire [31:0]  roundRemainder = roundResult & clipMaskRemainder;
  wire         roundSignBits = vSewOH[0] & roundResult[7] | vSewOH[1] & roundResult[15] | vSewOH[2] & roundResult[31];
  wire         roundResultOverlap = (|roundRemainder) & ~(request_sign & (&(roundRemainder | clipMask)) & roundSignBits);
  wire         differentSign = request_sign & roundSignBits & ~(request_src_1[31]);
  assign response_clipFail = roundResultOverlap | differentSign;
  wire [31:0]  clipResult = response_clipFail ? largestClipResult : roundResult;
  wire [8:0]   indexRes = {request_groupIndex, request_laneIndex, request_executeIndex} >> request_vSew;
  wire         extendSign = request_sign & (vSewOH[0] & request_src_0[7] | vSewOH[1] & request_src_0[15] | vSewOH[2] & request_src_0[31]);
  wire         selectSource1 = (originalOpcodeOH[0] | originalOpcodeOH[1]) & request_mask | originalOpcodeOH[3];
  wire         selectSource2 = originalOpcodeOH[1] & ~request_mask;
  wire [1:0]   resultSelect_lo_hi = originalOpcodeOH[5:4];
  wire [2:0]   resultSelect_lo = {resultSelect_lo_hi, isffo};
  wire [1:0]   resultSelect_hi_hi = {selectSource2, selectSource1};
  wire [2:0]   resultSelect_hi = {resultSelect_hi_hi, originalOpcodeOH[2]};
  wire [5:0]   resultSelect = {resultSelect_hi, resultSelect_lo};
  wire [31:0]  popCountResult = _popCount_resp + {22'h0, request_popInit};
  wire [31:0]  _result_T_12 = (resultSelect[0] ? _ffo_resp_bits : 32'h0) | (resultSelect[1] ? popCountResult : 32'h0);
  assign result = {_result_T_12[31:9], _result_T_12[8:0] | (resultSelect[2] ? indexRes : 9'h0)} | (resultSelect[3] ? clipResult : 32'h0) | (resultSelect[4] ? request_src_0 : 32'h0) | (resultSelect[5] ? request_src_1 : 32'h0);
  assign response_data = result;
  assign response_ffoSuccess = _ffo_resp_valid & isffo;
  always @(posedge clock) begin
    if (reset) begin
      requestReg_tag <= 2'h0;
      requestReg_src_0 <= 32'h0;
      requestReg_src_1 <= 32'h0;
      requestReg_src_2 <= 32'h0;
      requestReg_src_3 <= 32'h0;
      requestReg_popInit <= 10'h0;
      requestReg_opcode <= 4'h0;
      requestReg_groupIndex <= 5'h0;
      requestReg_laneIndex <= 2'h0;
      requestReg_executeIndex <= 2'h0;
      requestReg_sign <= 1'h0;
      requestReg_mask <= 1'h0;
      requestReg_maskType <= 1'h0;
      requestReg_vSew <= 2'h0;
      requestReg_vxrm <= 2'h0;
      requestReg_narrow <= 1'h0;
      requestRegValid <= 1'h0;
      request_pipeResponse_pipe_v <= 1'h0;
    end
    else begin
      if (requestIO_valid_0) begin
        requestReg_tag <= requestIO_bits_tag_0;
        requestReg_src_0 <= requestIO_bits_src_0_0;
        requestReg_src_1 <= requestIO_bits_src_1_0;
        requestReg_src_2 <= requestIO_bits_src_2_0;
        requestReg_src_3 <= requestIO_bits_src_3_0;
        requestReg_popInit <= requestIO_bits_popInit_0;
        requestReg_opcode <= requestIO_bits_opcode_0;
        requestReg_groupIndex <= requestIO_bits_groupIndex_0;
        requestReg_laneIndex <= requestIO_bits_laneIndex_0;
        requestReg_executeIndex <= requestIO_bits_executeIndex_0;
        requestReg_sign <= requestIO_bits_sign_0;
        requestReg_mask <= requestIO_bits_mask_0;
        requestReg_maskType <= requestIO_bits_maskType_0;
        requestReg_vSew <= requestIO_bits_vSew_0;
        requestReg_vxrm <= requestIO_bits_vxrm_0;
        requestReg_narrow <= requestIO_bits_narrow_0;
      end
      requestRegValid <= requestIO_valid_0;
      request_pipeResponse_pipe_v <= request_responseValidWire;
    end
    if (request_responseValidWire) begin
      request_pipeResponse_pipe_b_tag <= request_responseWire_tag;
      request_pipeResponse_pipe_b_data <= request_responseWire_data;
      request_pipeResponse_pipe_b_clipFail <= request_responseWire_clipFail;
      request_pipeResponse_pipe_b_ffoSuccess <= request_responseWire_ffoSuccess;
    end
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:6];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [2:0] i = 3'h0; i < 3'h7; i += 3'h1) begin
          _RANDOM[i] = `RANDOM;
        end
        requestReg_tag = _RANDOM[3'h0][1:0];
        requestReg_src_0 = {_RANDOM[3'h0][31:2], _RANDOM[3'h1][1:0]};
        requestReg_src_1 = {_RANDOM[3'h1][31:2], _RANDOM[3'h2][1:0]};
        requestReg_src_2 = {_RANDOM[3'h2][31:2], _RANDOM[3'h3][1:0]};
        requestReg_src_3 = {_RANDOM[3'h3][31:2], _RANDOM[3'h4][1:0]};
        requestReg_popInit = _RANDOM[3'h4][11:2];
        requestReg_opcode = _RANDOM[3'h4][15:12];
        requestReg_groupIndex = _RANDOM[3'h4][20:16];
        requestReg_laneIndex = _RANDOM[3'h4][22:21];
        requestReg_executeIndex = _RANDOM[3'h4][24:23];
        requestReg_sign = _RANDOM[3'h4][25];
        requestReg_mask = _RANDOM[3'h4][26];
        requestReg_maskType = _RANDOM[3'h4][28];
        requestReg_vSew = _RANDOM[3'h4][30:29];
        requestReg_vxrm = {_RANDOM[3'h4][31], _RANDOM[3'h5][0]};
        requestReg_narrow = _RANDOM[3'h5][1];
        requestRegValid = _RANDOM[3'h5][2];
        request_pipeResponse_pipe_v = _RANDOM[3'h5][3];
        request_pipeResponse_pipe_b_tag = _RANDOM[3'h5][5:4];
        request_pipeResponse_pipe_b_data = {_RANDOM[3'h5][31:6], _RANDOM[3'h6][5:0]};
        request_pipeResponse_pipe_b_clipFail = _RANDOM[3'h6][6];
        request_pipeResponse_pipe_b_ffoSuccess = _RANDOM[3'h6][7];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  LaneFFO ffo (
    .src_0        (request_src_0),
    .src_1        (request_src_1),
    .src_2        (request_src_2),
    .src_3        (request_src_3),
    .resultSelect (request_opcode[1:0]),
    .resp_valid   (_ffo_resp_valid),
    .resp_bits    (_ffo_resp_bits),
    .complete     (request_complete),
    .maskType     (request_maskType)
  );
  LanePopCount popCount (
    .src  (request_src_1 & (request_maskType ? request_src_0 : 32'hFFFFFFFF) & request_src_3),
    .resp (_popCount_resp)
  );
  assign responseIO_valid = responseIO_valid_0;
  assign responseIO_bits_tag = responseIO_bits_tag_0;
  assign responseIO_bits_data = responseIO_bits_data_0;
  assign responseIO_bits_clipFail = responseIO_bits_clipFail_0;
  assign responseIO_bits_ffoSuccess = responseIO_bits_ffoSuccess_0;
endmodule

