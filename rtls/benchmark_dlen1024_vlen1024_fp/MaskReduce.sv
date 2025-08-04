
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
module MaskReduce(
  input           clock,
                  reset,
  output          in_ready,
  input           in_valid,
                  in_bits_maskType,
  input  [1:0]    in_bits_eew,
  input  [2:0]    in_bits_uop,
  input  [31:0]   in_bits_readVS1,
  input  [1023:0] in_bits_source2,
  input  [31:0]   in_bits_sourceValid,
  input           in_bits_lastGroup,
  input  [2:0]    in_bits_vxrm,
  input  [3:0]    in_bits_aluUop,
  input           in_bits_sign,
  input  [31:0]   in_bits_fpSourceValid,
  output          out_valid,
  output [31:0]   out_bits_data,
  output [3:0]    out_bits_mask,
  input           firstGroup,
                  newInstruction,
                  validInst,
                  pop
);

  wire [31:0]   _flotCompare_out;
  wire [31:0]   _floatAdder_out;
  wire [31:0]   _logicUnit_resp;
  wire [31:0]   _adder_response_data;
  wire          in_valid_0 = in_valid;
  wire          in_bits_maskType_0 = in_bits_maskType;
  wire [1:0]    in_bits_eew_0 = in_bits_eew;
  wire [2:0]    in_bits_uop_0 = in_bits_uop;
  wire [31:0]   in_bits_readVS1_0 = in_bits_readVS1;
  wire [1023:0] in_bits_source2_0 = in_bits_source2;
  wire [31:0]   in_bits_sourceValid_0 = in_bits_sourceValid;
  wire          in_bits_lastGroup_0 = in_bits_lastGroup;
  wire [2:0]    in_bits_vxrm_0 = in_bits_vxrm;
  wire [3:0]    in_bits_aluUop_0 = in_bits_aluUop;
  wire          in_bits_sign_0 = in_bits_sign;
  wire [31:0]   in_bits_fpSourceValid_0 = in_bits_fpSourceValid;
  wire          skipFlotReduce = 1'h0;
  wire          stateIdle;
  wire          order = in_bits_uop_0 == 3'h5;
  wire          reqWiden = in_bits_uop_0 == 3'h1 | (&(in_bits_uop_0[2:1]));
  wire [3:0]    _eew1H_T = 4'h1 << in_bits_eew_0;
  wire [2:0]    eew1H = _eew1H_T[2:0];
  wire          nextFoldCount = eew1H[0] & ~reqWiden;
  reg  [31:0]   reduceInit;
  reg  [4:0]    crossFoldCount;
  reg           lastFoldCount;
  reg           reqReg_maskType;
  reg  [1:0]    reqReg_eew;
  reg  [2:0]    reqReg_uop;
  reg  [31:0]   reqReg_readVS1;
  reg  [1023:0] reqReg_source2;
  reg  [31:0]   reqReg_sourceValid;
  reg           reqReg_lastGroup;
  reg  [2:0]    reqReg_vxrm;
  reg  [3:0]    reqReg_aluUop;
  reg           reqReg_sign;
  reg  [31:0]   reqReg_fpSourceValid;
  wire          groupLastReduce = &crossFoldCount;
  wire          lastFoldEnd = ~lastFoldCount;
  wire [3:0]    _eew1HReg_T = 4'h1 << reqReg_eew;
  wire [2:0]    eew1HReg = _eew1HReg_T[2:0];
  wire          floatType = reqReg_uop[2] | (&(reqReg_uop[1:0]));
  wire          NotAdd = reqReg_uop[1];
  wire          widen = reqReg_uop == 3'h1 | (&(reqReg_uop[2:1]));
  wire          floatAdd = floatType & ~NotAdd;
  wire [1:0]    writeEEW = pop ? 2'h2 : reqReg_eew + {1'h0, widen};
  wire [3:0]    _writeEEW1H_T = 4'h1 << writeEEW;
  wire [2:0]    writeEEW1H = _writeEEW1H_T[2:0];
  wire [3:0]    writeMask = {{2{writeEEW1H[2]}}, ~(writeEEW1H[0]), 1'h1};
  reg  [2:0]    state;
  assign stateIdle = state == 3'h0;
  wire          in_ready_0 = stateIdle;
  wire          stateCross = state == 3'h1;
  wire          stateLast = state == 3'h2;
  wire          stateOrder = state == 3'h3;
  wire          stateWait = state == 3'h4;
  reg           waitCount;
  wire          resFire = stateWait & ~waitCount;
  wire          sourceValid;
  wire          updateResult = stateLast | stateCross & sourceValid & ~floatAdd | resFire & sourceValid;
  wire          waiteDeq = stateWait & resFire;
  wire          _GEN = stateLast & lastFoldEnd;
  wire          outValid = _GEN | (waiteDeq | stateCross & ~floatAdd) & groupLastReduce & reqReg_lastGroup;
  wire [3:0]    widenEnqMask = {{2{|in_bits_eew_0}}, 2'h3};
  wire [3:0]    normalMask = {{2{in_bits_eew_0[1]}}, |in_bits_eew_0, 1'h1};
  wire [3:0]    enqWriteMask = reqWiden ? widenEnqMask : normalMask;
  wire [15:0]   updateInitMask_lo = {{8{enqWriteMask[1]}}, {8{enqWriteMask[0]}}};
  wire [15:0]   updateInitMask_hi = {{8{enqWriteMask[3]}}, {8{enqWriteMask[2]}}};
  wire [31:0]   updateInitMask = {updateInitMask_hi, updateInitMask_lo};
  wire [15:0]   updateMask_lo = {{8{writeMask[1]}}, {8{writeMask[0]}}};
  wire [15:0]   updateMask_hi = {{8{writeMask[3]}}, {8{writeMask[2]}}};
  wire [31:0]   updateMask = {updateMask_hi, updateMask_lo};
  wire [31:0]   _sourceValid_T = 32'h1 << crossFoldCount;
  wire [31:0]   selectLaneResult =
    (_sourceValid_T[0] ? reqReg_source2[31:0] : 32'h0) | (_sourceValid_T[1] ? reqReg_source2[63:32] : 32'h0) | (_sourceValid_T[2] ? reqReg_source2[95:64] : 32'h0) | (_sourceValid_T[3] ? reqReg_source2[127:96] : 32'h0)
    | (_sourceValid_T[4] ? reqReg_source2[159:128] : 32'h0) | (_sourceValid_T[5] ? reqReg_source2[191:160] : 32'h0) | (_sourceValid_T[6] ? reqReg_source2[223:192] : 32'h0) | (_sourceValid_T[7] ? reqReg_source2[255:224] : 32'h0)
    | (_sourceValid_T[8] ? reqReg_source2[287:256] : 32'h0) | (_sourceValid_T[9] ? reqReg_source2[319:288] : 32'h0) | (_sourceValid_T[10] ? reqReg_source2[351:320] : 32'h0) | (_sourceValid_T[11] ? reqReg_source2[383:352] : 32'h0)
    | (_sourceValid_T[12] ? reqReg_source2[415:384] : 32'h0) | (_sourceValid_T[13] ? reqReg_source2[447:416] : 32'h0) | (_sourceValid_T[14] ? reqReg_source2[479:448] : 32'h0) | (_sourceValid_T[15] ? reqReg_source2[511:480] : 32'h0)
    | (_sourceValid_T[16] ? reqReg_source2[543:512] : 32'h0) | (_sourceValid_T[17] ? reqReg_source2[575:544] : 32'h0) | (_sourceValid_T[18] ? reqReg_source2[607:576] : 32'h0) | (_sourceValid_T[19] ? reqReg_source2[639:608] : 32'h0)
    | (_sourceValid_T[20] ? reqReg_source2[671:640] : 32'h0) | (_sourceValid_T[21] ? reqReg_source2[703:672] : 32'h0) | (_sourceValid_T[22] ? reqReg_source2[735:704] : 32'h0) | (_sourceValid_T[23] ? reqReg_source2[767:736] : 32'h0)
    | (_sourceValid_T[24] ? reqReg_source2[799:768] : 32'h0) | (_sourceValid_T[25] ? reqReg_source2[831:800] : 32'h0) | (_sourceValid_T[26] ? reqReg_source2[863:832] : 32'h0) | (_sourceValid_T[27] ? reqReg_source2[895:864] : 32'h0)
    | (_sourceValid_T[28] ? reqReg_source2[927:896] : 32'h0) | (_sourceValid_T[29] ? reqReg_source2[959:928] : 32'h0) | (_sourceValid_T[30] ? reqReg_source2[991:960] : 32'h0) | (_sourceValid_T[31] ? reqReg_source2[1023:992] : 32'h0);
  wire [31:0]   sourceValidCalculate = ({32{~floatType}} | reqReg_fpSourceValid) & reqReg_sourceValid;
  assign sourceValid =
    _sourceValid_T[0] & sourceValidCalculate[0] | _sourceValid_T[1] & sourceValidCalculate[1] | _sourceValid_T[2] & sourceValidCalculate[2] | _sourceValid_T[3] & sourceValidCalculate[3] | _sourceValid_T[4] & sourceValidCalculate[4]
    | _sourceValid_T[5] & sourceValidCalculate[5] | _sourceValid_T[6] & sourceValidCalculate[6] | _sourceValid_T[7] & sourceValidCalculate[7] | _sourceValid_T[8] & sourceValidCalculate[8] | _sourceValid_T[9] & sourceValidCalculate[9]
    | _sourceValid_T[10] & sourceValidCalculate[10] | _sourceValid_T[11] & sourceValidCalculate[11] | _sourceValid_T[12] & sourceValidCalculate[12] | _sourceValid_T[13] & sourceValidCalculate[13] | _sourceValid_T[14]
    & sourceValidCalculate[14] | _sourceValid_T[15] & sourceValidCalculate[15] | _sourceValid_T[16] & sourceValidCalculate[16] | _sourceValid_T[17] & sourceValidCalculate[17] | _sourceValid_T[18] & sourceValidCalculate[18]
    | _sourceValid_T[19] & sourceValidCalculate[19] | _sourceValid_T[20] & sourceValidCalculate[20] | _sourceValid_T[21] & sourceValidCalculate[21] | _sourceValid_T[22] & sourceValidCalculate[22] | _sourceValid_T[23]
    & sourceValidCalculate[23] | _sourceValid_T[24] & sourceValidCalculate[24] | _sourceValid_T[25] & sourceValidCalculate[25] | _sourceValid_T[26] & sourceValidCalculate[26] | _sourceValid_T[27] & sourceValidCalculate[27]
    | _sourceValid_T[28] & sourceValidCalculate[28] | _sourceValid_T[29] & sourceValidCalculate[29] | _sourceValid_T[30] & sourceValidCalculate[30] | _sourceValid_T[31] & sourceValidCalculate[31];
  wire [7:0]    reduceDataVec_0 = reduceInit[7:0];
  wire [7:0]    reduceDataVec_1 = reduceInit[15:8];
  wire [7:0]    reduceDataVec_2 = reduceInit[23:16];
  wire [7:0]    reduceDataVec_3 = reduceInit[31:24];
  wire [23:0]   lastFoldSource1 = {{2{reduceDataVec_3}}, lastFoldCount ? reduceDataVec_1 : reduceDataVec_2};
  wire [31:0]   source2Select = stateCross | stateOrder ? selectLaneResult : {8'h0, lastFoldSource1};
  wire [31:0]   flotReduceResult = NotAdd ? _flotCompare_out : _floatAdder_out;
  wire [31:0]   reduceResult = floatType ? flotReduceResult : NotAdd ? _logicUnit_resp : _adder_response_data;
  always @(posedge clock) begin
    if (reset) begin
      reduceInit <= 32'h0;
      crossFoldCount <= 5'h0;
      lastFoldCount <= 1'h0;
      reqReg_maskType <= 1'h0;
      reqReg_eew <= 2'h0;
      reqReg_uop <= 3'h0;
      reqReg_readVS1 <= 32'h0;
      reqReg_source2 <= 1024'h0;
      reqReg_sourceValid <= 32'h0;
      reqReg_lastGroup <= 1'h0;
      reqReg_vxrm <= 3'h0;
      reqReg_aluUop <= 4'h0;
      reqReg_sign <= 1'h0;
      reqReg_fpSourceValid <= 32'h0;
      state <= 3'h0;
      waitCount <= 1'h0;
    end
    else begin
      automatic logic _crossFoldCount_T;
      automatic logic _GEN_0;
      _crossFoldCount_T = in_ready_0 & in_valid_0;
      _GEN_0 = firstGroup | newInstruction;
      if (updateResult)
        reduceInit <= reduceResult & updateMask;
      else if (_GEN_0)
        reduceInit <= pop | newInstruction ? 32'h0 : in_bits_readVS1_0 & updateInitMask;
      if (stateCross & ~floatAdd | waiteDeq | _crossFoldCount_T)
        crossFoldCount <= _crossFoldCount_T ? 5'h0 : crossFoldCount + 5'h1;
      else if (_GEN_0)
        crossFoldCount <= 5'h0;
      lastFoldCount <= ~stateLast & (_GEN_0 ? nextFoldCount : lastFoldCount);
      if (_crossFoldCount_T) begin
        reqReg_maskType <= in_bits_maskType_0;
        reqReg_eew <= in_bits_eew_0;
        reqReg_uop <= in_bits_uop_0;
        reqReg_readVS1 <= in_bits_readVS1_0;
        reqReg_source2 <= in_bits_source2_0;
        reqReg_sourceValid <= in_bits_sourceValid_0;
        reqReg_lastGroup <= in_bits_lastGroup_0;
        reqReg_vxrm <= in_bits_vxrm_0;
        reqReg_aluUop <= in_bits_aluUop_0;
        reqReg_sign <= in_bits_sign_0;
        reqReg_fpSourceValid <= in_bits_fpSourceValid_0;
      end
      if (_GEN)
        state <= 3'h0;
      else if (stateOrder)
        state <= 3'h4;
      else if (waiteDeq)
        state <= groupLastReduce ? 3'h0 : {1'h0, order, 1'h1};
      else begin
        automatic logic [2:0] _state_T;
        automatic logic       _GEN_1;
        _state_T = {1'h0, order, 1'h1};
        _GEN_1 = stateIdle & in_valid_0;
        if (stateCross) begin
          if (floatAdd)
            state <= 3'h4;
          else if (groupLastReduce)
            state <= 3'h0;
          else if (_GEN_1)
            state <= _state_T;
        end
        else if (_GEN_1)
          state <= _state_T;
      end
      waitCount <= ~(stateOrder | stateCross & floatAdd) & (stateWait ? waitCount - 1'h1 : waitCount);
    end
  end // always @(posedge)
  `ifdef ENABLE_INITIAL_REG_
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM[0:36];
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        for (logic [5:0] i = 6'h0; i < 6'h25; i += 6'h1) begin
          _RANDOM[i] = `RANDOM;
        end
        reduceInit = _RANDOM[6'h0];
        crossFoldCount = _RANDOM[6'h1][4:0];
        lastFoldCount = _RANDOM[6'h1][5];
        reqReg_maskType = _RANDOM[6'h1][6];
        reqReg_eew = _RANDOM[6'h1][8:7];
        reqReg_uop = _RANDOM[6'h1][11:9];
        reqReg_readVS1 = {_RANDOM[6'h1][31:12], _RANDOM[6'h2][11:0]};
        reqReg_source2 =
          {_RANDOM[6'h2][31:12],
           _RANDOM[6'h3],
           _RANDOM[6'h4],
           _RANDOM[6'h5],
           _RANDOM[6'h6],
           _RANDOM[6'h7],
           _RANDOM[6'h8],
           _RANDOM[6'h9],
           _RANDOM[6'hA],
           _RANDOM[6'hB],
           _RANDOM[6'hC],
           _RANDOM[6'hD],
           _RANDOM[6'hE],
           _RANDOM[6'hF],
           _RANDOM[6'h10],
           _RANDOM[6'h11],
           _RANDOM[6'h12],
           _RANDOM[6'h13],
           _RANDOM[6'h14],
           _RANDOM[6'h15],
           _RANDOM[6'h16],
           _RANDOM[6'h17],
           _RANDOM[6'h18],
           _RANDOM[6'h19],
           _RANDOM[6'h1A],
           _RANDOM[6'h1B],
           _RANDOM[6'h1C],
           _RANDOM[6'h1D],
           _RANDOM[6'h1E],
           _RANDOM[6'h1F],
           _RANDOM[6'h20],
           _RANDOM[6'h21],
           _RANDOM[6'h22][11:0]};
        reqReg_sourceValid = {_RANDOM[6'h22][31:12], _RANDOM[6'h23][11:0]};
        reqReg_lastGroup = _RANDOM[6'h23][12];
        reqReg_vxrm = _RANDOM[6'h23][15:13];
        reqReg_aluUop = _RANDOM[6'h23][19:16];
        reqReg_sign = _RANDOM[6'h23][20];
        reqReg_fpSourceValid = {_RANDOM[6'h23][31:21], _RANDOM[6'h24][20:0]};
        state = _RANDOM[6'h24][23:21];
        waitCount = _RANDOM[6'h24][24];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // ENABLE_INITIAL_REG_
  ReduceAdder adder (
    .request_src_0  (reduceInit),
    .request_src_1  (source2Select),
    .request_opcode (pop ? 4'h0 : reqReg_aluUop),
    .request_vSew   (writeEEW),
    .request_sign   (reqReg_sign),
    .response_data  (_adder_response_data)
  );
  LaneLogic logicUnit (
    .req_src_0  (reduceInit),
    .req_src_1  (source2Select),
    .req_opcode (reqReg_aluUop),
    .resp       (_logicUnit_resp)
  );
  FloatAdder floatAdder (
    .clock        (clock),
    .reset        (reset),
    .a            (reduceInit),
    .b            (source2Select),
    .roundingMode (reqReg_vxrm),
    .out          (_floatAdder_out)
  );
  FloatCompare flotCompare (
    .a     (reduceInit),
    .b     (source2Select),
    .isMax (reqReg_aluUop[2]),
    .out   (_flotCompare_out)
  );
  assign in_ready = in_ready_0;
  assign out_valid = outValid & ~pop;
  assign out_bits_data = updateResult ? reduceResult : reduceInit;
  assign out_bits_mask = writeMask & {4{validInst}};
endmodule

