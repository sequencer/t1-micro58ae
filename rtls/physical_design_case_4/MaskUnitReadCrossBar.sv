module MaskUnitReadCrossBar(
  output       input_0_ready,
  input        input_0_valid,
  input  [4:0] input_0_bits_vs,
  input  [3:0] input_0_bits_offset,
  input  [1:0] input_0_bits_readLane,
               input_0_bits_dataOffset,
  output       input_1_ready,
  input        input_1_valid,
  input  [4:0] input_1_bits_vs,
  input  [3:0] input_1_bits_offset,
  input  [1:0] input_1_bits_readLane,
               input_1_bits_dataOffset,
  output       input_2_ready,
  input        input_2_valid,
  input  [4:0] input_2_bits_vs,
  input  [3:0] input_2_bits_offset,
  input  [1:0] input_2_bits_readLane,
               input_2_bits_dataOffset,
  output       input_3_ready,
  input        input_3_valid,
  input  [4:0] input_3_bits_vs,
  input  [3:0] input_3_bits_offset,
  input  [1:0] input_3_bits_readLane,
               input_3_bits_dataOffset,
  input        output_0_ready,
  output       output_0_valid,
  output [4:0] output_0_bits_vs,
  output [3:0] output_0_bits_offset,
  output [1:0] output_0_bits_writeIndex,
               output_0_bits_dataOffset,
  input        output_1_ready,
  output       output_1_valid,
  output [4:0] output_1_bits_vs,
  output [3:0] output_1_bits_offset,
  output [1:0] output_1_bits_writeIndex,
               output_1_bits_dataOffset,
  input        output_2_ready,
  output       output_2_valid,
  output [4:0] output_2_bits_vs,
  output [3:0] output_2_bits_offset,
  output [1:0] output_2_bits_writeIndex,
               output_2_bits_dataOffset,
  input        output_3_ready,
  output       output_3_valid,
  output [4:0] output_3_bits_vs,
  output [3:0] output_3_bits_offset,
  output [1:0] output_3_bits_writeIndex,
               output_3_bits_dataOffset
);

  wire       input_0_valid_0 = input_0_valid;
  wire [4:0] input_0_bits_vs_0 = input_0_bits_vs;
  wire [3:0] input_0_bits_offset_0 = input_0_bits_offset;
  wire [1:0] input_0_bits_readLane_0 = input_0_bits_readLane;
  wire [1:0] input_0_bits_dataOffset_0 = input_0_bits_dataOffset;
  wire       input_1_valid_0 = input_1_valid;
  wire [4:0] input_1_bits_vs_0 = input_1_bits_vs;
  wire [3:0] input_1_bits_offset_0 = input_1_bits_offset;
  wire [1:0] input_1_bits_readLane_0 = input_1_bits_readLane;
  wire [1:0] input_1_bits_dataOffset_0 = input_1_bits_dataOffset;
  wire       input_2_valid_0 = input_2_valid;
  wire [4:0] input_2_bits_vs_0 = input_2_bits_vs;
  wire [3:0] input_2_bits_offset_0 = input_2_bits_offset;
  wire [1:0] input_2_bits_readLane_0 = input_2_bits_readLane;
  wire [1:0] input_2_bits_dataOffset_0 = input_2_bits_dataOffset;
  wire       input_3_valid_0 = input_3_valid;
  wire [4:0] input_3_bits_vs_0 = input_3_bits_vs;
  wire [3:0] input_3_bits_offset_0 = input_3_bits_offset;
  wire [1:0] input_3_bits_readLane_0 = input_3_bits_readLane;
  wire [1:0] input_3_bits_dataOffset_0 = input_3_bits_dataOffset;
  wire       output_0_ready_0 = output_0_ready;
  wire       output_1_ready_0 = output_1_ready;
  wire       output_2_ready_0 = output_2_ready;
  wire       output_3_ready_0 = output_3_ready;
  wire [1:0] input_3_bits_requestIndex = 2'h3;
  wire [1:0] input_2_bits_requestIndex = 2'h2;
  wire [1:0] input_1_bits_requestIndex = 2'h1;
  wire [1:0] input_0_bits_requestIndex = 2'h0;
  wire [4:0] selectReq_bits_vs;
  wire [3:0] selectReq_bits_offset;
  wire [1:0] selectReq_bits_requestIndex;
  wire [1:0] selectReq_bits_dataOffset;
  wire [4:0] selectReq_1_bits_vs;
  wire [3:0] selectReq_1_bits_offset;
  wire [1:0] selectReq_1_bits_requestIndex;
  wire [1:0] selectReq_1_bits_dataOffset;
  wire [4:0] selectReq_2_bits_vs;
  wire [3:0] selectReq_2_bits_offset;
  wire [1:0] selectReq_2_bits_requestIndex;
  wire [1:0] selectReq_2_bits_dataOffset;
  wire [4:0] selectReq_3_bits_vs;
  wire [3:0] selectReq_3_bits_offset;
  wire [1:0] selectReq_3_bits_requestIndex;
  wire [1:0] selectReq_3_bits_dataOffset;
  wire [3:0] requestReadLane = 4'h1 << input_0_bits_readLane_0;
  wire       free = |requestReadLane;
  wire       outReady = requestReadLane[0] & output_0_ready_0 | requestReadLane[1] & output_1_ready_0 | requestReadLane[2] & output_2_ready_0 | requestReadLane[3] & output_3_ready_0;
  wire       input_0_ready_0 = free & outReady;
  wire [3:0] inputSelect1H_0 = input_0_valid_0 & free ? requestReadLane : 4'h0;
  wire [3:0] requestReadLane_1 = 4'h1 << input_1_bits_readLane_0;
  wire       free_1 = |(requestReadLane_1 & ~inputSelect1H_0);
  wire       outReady_1 = requestReadLane_1[0] & output_0_ready_0 | requestReadLane_1[1] & output_1_ready_0 | requestReadLane_1[2] & output_2_ready_0 | requestReadLane_1[3] & output_3_ready_0;
  wire       input_1_ready_0 = free_1 & outReady_1;
  wire [3:0] inputSelect1H_1 = input_1_valid_0 & free_1 ? requestReadLane_1 : 4'h0;
  wire [3:0] _GEN = inputSelect1H_0 | inputSelect1H_1;
  wire [3:0] requestReadLane_2 = 4'h1 << input_2_bits_readLane_0;
  wire       free_2 = |(requestReadLane_2 & ~_GEN);
  wire       outReady_2 = requestReadLane_2[0] & output_0_ready_0 | requestReadLane_2[1] & output_1_ready_0 | requestReadLane_2[2] & output_2_ready_0 | requestReadLane_2[3] & output_3_ready_0;
  wire       input_2_ready_0 = free_2 & outReady_2;
  wire [3:0] inputSelect1H_2 = input_2_valid_0 & free_2 ? requestReadLane_2 : 4'h0;
  wire [3:0] requestReadLane_3 = 4'h1 << input_3_bits_readLane_0;
  wire       free_3 = |(requestReadLane_3 & ~(_GEN | inputSelect1H_2));
  wire       outReady_3 = requestReadLane_3[0] & output_0_ready_0 | requestReadLane_3[1] & output_1_ready_0 | requestReadLane_3[2] & output_2_ready_0 | requestReadLane_3[3] & output_3_ready_0;
  wire       input_3_ready_0 = free_3 & outReady_3;
  wire [3:0] inputSelect1H_3 = input_3_valid_0 & free_3 ? requestReadLane_3 : 4'h0;
  wire [1:0] tryToRead_lo = {inputSelect1H_1[0], inputSelect1H_0[0]};
  wire [1:0] tryToRead_hi = {inputSelect1H_3[0], inputSelect1H_2[0]};
  wire [3:0] tryToRead = {tryToRead_hi, tryToRead_lo};
  wire       output_0_valid_0 = |tryToRead;
  wire [4:0] output_0_bits_vs_0 = selectReq_bits_vs;
  wire [3:0] output_0_bits_offset_0 = selectReq_bits_offset;
  wire [1:0] output_0_bits_writeIndex_0 = selectReq_bits_requestIndex;
  wire [1:0] output_0_bits_dataOffset_0 = selectReq_bits_dataOffset;
  assign selectReq_bits_dataOffset =
    (tryToRead[0] ? input_0_bits_dataOffset_0 : 2'h0) | (tryToRead[1] ? input_1_bits_dataOffset_0 : 2'h0) | (tryToRead[2] ? input_2_bits_dataOffset_0 : 2'h0) | (tryToRead[3] ? input_3_bits_dataOffset_0 : 2'h0);
  assign selectReq_bits_requestIndex = {1'h0, tryToRead[1]} | {tryToRead[2], 1'h0} | {2{tryToRead[3]}};
  wire [1:0] selectReq_bits_readLane = (tryToRead[0] ? input_0_bits_readLane_0 : 2'h0) | (tryToRead[1] ? input_1_bits_readLane_0 : 2'h0) | (tryToRead[2] ? input_2_bits_readLane_0 : 2'h0) | (tryToRead[3] ? input_3_bits_readLane_0 : 2'h0);
  assign selectReq_bits_offset = (tryToRead[0] ? input_0_bits_offset_0 : 4'h0) | (tryToRead[1] ? input_1_bits_offset_0 : 4'h0) | (tryToRead[2] ? input_2_bits_offset_0 : 4'h0) | (tryToRead[3] ? input_3_bits_offset_0 : 4'h0);
  assign selectReq_bits_vs = (tryToRead[0] ? input_0_bits_vs_0 : 5'h0) | (tryToRead[1] ? input_1_bits_vs_0 : 5'h0) | (tryToRead[2] ? input_2_bits_vs_0 : 5'h0) | (tryToRead[3] ? input_3_bits_vs_0 : 5'h0);
  wire       selectReq_valid = tryToRead[0] & input_0_valid_0 | tryToRead[1] & input_1_valid_0 | tryToRead[2] & input_2_valid_0 | tryToRead[3] & input_3_valid_0;
  wire       selectReq_ready = tryToRead[0] & input_0_ready_0 | tryToRead[1] & input_1_ready_0 | tryToRead[2] & input_2_ready_0 | tryToRead[3] & input_3_ready_0;
  wire [1:0] tryToRead_lo_1 = {inputSelect1H_1[1], inputSelect1H_0[1]};
  wire [1:0] tryToRead_hi_1 = {inputSelect1H_3[1], inputSelect1H_2[1]};
  wire [3:0] tryToRead_1 = {tryToRead_hi_1, tryToRead_lo_1};
  wire       output_1_valid_0 = |tryToRead_1;
  wire [4:0] output_1_bits_vs_0 = selectReq_1_bits_vs;
  wire [3:0] output_1_bits_offset_0 = selectReq_1_bits_offset;
  wire [1:0] output_1_bits_writeIndex_0 = selectReq_1_bits_requestIndex;
  wire [1:0] output_1_bits_dataOffset_0 = selectReq_1_bits_dataOffset;
  assign selectReq_1_bits_dataOffset =
    (tryToRead_1[0] ? input_0_bits_dataOffset_0 : 2'h0) | (tryToRead_1[1] ? input_1_bits_dataOffset_0 : 2'h0) | (tryToRead_1[2] ? input_2_bits_dataOffset_0 : 2'h0) | (tryToRead_1[3] ? input_3_bits_dataOffset_0 : 2'h0);
  assign selectReq_1_bits_requestIndex = {1'h0, tryToRead_1[1]} | {tryToRead_1[2], 1'h0} | {2{tryToRead_1[3]}};
  wire [1:0] selectReq_1_bits_readLane =
    (tryToRead_1[0] ? input_0_bits_readLane_0 : 2'h0) | (tryToRead_1[1] ? input_1_bits_readLane_0 : 2'h0) | (tryToRead_1[2] ? input_2_bits_readLane_0 : 2'h0) | (tryToRead_1[3] ? input_3_bits_readLane_0 : 2'h0);
  assign selectReq_1_bits_offset = (tryToRead_1[0] ? input_0_bits_offset_0 : 4'h0) | (tryToRead_1[1] ? input_1_bits_offset_0 : 4'h0) | (tryToRead_1[2] ? input_2_bits_offset_0 : 4'h0) | (tryToRead_1[3] ? input_3_bits_offset_0 : 4'h0);
  assign selectReq_1_bits_vs = (tryToRead_1[0] ? input_0_bits_vs_0 : 5'h0) | (tryToRead_1[1] ? input_1_bits_vs_0 : 5'h0) | (tryToRead_1[2] ? input_2_bits_vs_0 : 5'h0) | (tryToRead_1[3] ? input_3_bits_vs_0 : 5'h0);
  wire       selectReq_1_valid = tryToRead_1[0] & input_0_valid_0 | tryToRead_1[1] & input_1_valid_0 | tryToRead_1[2] & input_2_valid_0 | tryToRead_1[3] & input_3_valid_0;
  wire       selectReq_1_ready = tryToRead_1[0] & input_0_ready_0 | tryToRead_1[1] & input_1_ready_0 | tryToRead_1[2] & input_2_ready_0 | tryToRead_1[3] & input_3_ready_0;
  wire [1:0] tryToRead_lo_2 = {inputSelect1H_1[2], inputSelect1H_0[2]};
  wire [1:0] tryToRead_hi_2 = {inputSelect1H_3[2], inputSelect1H_2[2]};
  wire [3:0] tryToRead_2 = {tryToRead_hi_2, tryToRead_lo_2};
  wire       output_2_valid_0 = |tryToRead_2;
  wire [4:0] output_2_bits_vs_0 = selectReq_2_bits_vs;
  wire [3:0] output_2_bits_offset_0 = selectReq_2_bits_offset;
  wire [1:0] output_2_bits_writeIndex_0 = selectReq_2_bits_requestIndex;
  wire [1:0] output_2_bits_dataOffset_0 = selectReq_2_bits_dataOffset;
  assign selectReq_2_bits_dataOffset =
    (tryToRead_2[0] ? input_0_bits_dataOffset_0 : 2'h0) | (tryToRead_2[1] ? input_1_bits_dataOffset_0 : 2'h0) | (tryToRead_2[2] ? input_2_bits_dataOffset_0 : 2'h0) | (tryToRead_2[3] ? input_3_bits_dataOffset_0 : 2'h0);
  assign selectReq_2_bits_requestIndex = {1'h0, tryToRead_2[1]} | {tryToRead_2[2], 1'h0} | {2{tryToRead_2[3]}};
  wire [1:0] selectReq_2_bits_readLane =
    (tryToRead_2[0] ? input_0_bits_readLane_0 : 2'h0) | (tryToRead_2[1] ? input_1_bits_readLane_0 : 2'h0) | (tryToRead_2[2] ? input_2_bits_readLane_0 : 2'h0) | (tryToRead_2[3] ? input_3_bits_readLane_0 : 2'h0);
  assign selectReq_2_bits_offset = (tryToRead_2[0] ? input_0_bits_offset_0 : 4'h0) | (tryToRead_2[1] ? input_1_bits_offset_0 : 4'h0) | (tryToRead_2[2] ? input_2_bits_offset_0 : 4'h0) | (tryToRead_2[3] ? input_3_bits_offset_0 : 4'h0);
  assign selectReq_2_bits_vs = (tryToRead_2[0] ? input_0_bits_vs_0 : 5'h0) | (tryToRead_2[1] ? input_1_bits_vs_0 : 5'h0) | (tryToRead_2[2] ? input_2_bits_vs_0 : 5'h0) | (tryToRead_2[3] ? input_3_bits_vs_0 : 5'h0);
  wire       selectReq_2_valid = tryToRead_2[0] & input_0_valid_0 | tryToRead_2[1] & input_1_valid_0 | tryToRead_2[2] & input_2_valid_0 | tryToRead_2[3] & input_3_valid_0;
  wire       selectReq_2_ready = tryToRead_2[0] & input_0_ready_0 | tryToRead_2[1] & input_1_ready_0 | tryToRead_2[2] & input_2_ready_0 | tryToRead_2[3] & input_3_ready_0;
  wire [1:0] tryToRead_lo_3 = {inputSelect1H_1[3], inputSelect1H_0[3]};
  wire [1:0] tryToRead_hi_3 = {inputSelect1H_3[3], inputSelect1H_2[3]};
  wire [3:0] tryToRead_3 = {tryToRead_hi_3, tryToRead_lo_3};
  wire       output_3_valid_0 = |tryToRead_3;
  wire [4:0] output_3_bits_vs_0 = selectReq_3_bits_vs;
  wire [3:0] output_3_bits_offset_0 = selectReq_3_bits_offset;
  wire [1:0] output_3_bits_writeIndex_0 = selectReq_3_bits_requestIndex;
  wire [1:0] output_3_bits_dataOffset_0 = selectReq_3_bits_dataOffset;
  assign selectReq_3_bits_dataOffset =
    (tryToRead_3[0] ? input_0_bits_dataOffset_0 : 2'h0) | (tryToRead_3[1] ? input_1_bits_dataOffset_0 : 2'h0) | (tryToRead_3[2] ? input_2_bits_dataOffset_0 : 2'h0) | (tryToRead_3[3] ? input_3_bits_dataOffset_0 : 2'h0);
  assign selectReq_3_bits_requestIndex = {1'h0, tryToRead_3[1]} | {tryToRead_3[2], 1'h0} | {2{tryToRead_3[3]}};
  wire [1:0] selectReq_3_bits_readLane =
    (tryToRead_3[0] ? input_0_bits_readLane_0 : 2'h0) | (tryToRead_3[1] ? input_1_bits_readLane_0 : 2'h0) | (tryToRead_3[2] ? input_2_bits_readLane_0 : 2'h0) | (tryToRead_3[3] ? input_3_bits_readLane_0 : 2'h0);
  assign selectReq_3_bits_offset = (tryToRead_3[0] ? input_0_bits_offset_0 : 4'h0) | (tryToRead_3[1] ? input_1_bits_offset_0 : 4'h0) | (tryToRead_3[2] ? input_2_bits_offset_0 : 4'h0) | (tryToRead_3[3] ? input_3_bits_offset_0 : 4'h0);
  assign selectReq_3_bits_vs = (tryToRead_3[0] ? input_0_bits_vs_0 : 5'h0) | (tryToRead_3[1] ? input_1_bits_vs_0 : 5'h0) | (tryToRead_3[2] ? input_2_bits_vs_0 : 5'h0) | (tryToRead_3[3] ? input_3_bits_vs_0 : 5'h0);
  wire       selectReq_3_valid = tryToRead_3[0] & input_0_valid_0 | tryToRead_3[1] & input_1_valid_0 | tryToRead_3[2] & input_2_valid_0 | tryToRead_3[3] & input_3_valid_0;
  wire       selectReq_3_ready = tryToRead_3[0] & input_0_ready_0 | tryToRead_3[1] & input_1_ready_0 | tryToRead_3[2] & input_2_ready_0 | tryToRead_3[3] & input_3_ready_0;
  assign input_0_ready = input_0_ready_0;
  assign input_1_ready = input_1_ready_0;
  assign input_2_ready = input_2_ready_0;
  assign input_3_ready = input_3_ready_0;
  assign output_0_valid = output_0_valid_0;
  assign output_0_bits_vs = output_0_bits_vs_0;
  assign output_0_bits_offset = output_0_bits_offset_0;
  assign output_0_bits_writeIndex = output_0_bits_writeIndex_0;
  assign output_0_bits_dataOffset = output_0_bits_dataOffset_0;
  assign output_1_valid = output_1_valid_0;
  assign output_1_bits_vs = output_1_bits_vs_0;
  assign output_1_bits_offset = output_1_bits_offset_0;
  assign output_1_bits_writeIndex = output_1_bits_writeIndex_0;
  assign output_1_bits_dataOffset = output_1_bits_dataOffset_0;
  assign output_2_valid = output_2_valid_0;
  assign output_2_bits_vs = output_2_bits_vs_0;
  assign output_2_bits_offset = output_2_bits_offset_0;
  assign output_2_bits_writeIndex = output_2_bits_writeIndex_0;
  assign output_2_bits_dataOffset = output_2_bits_dataOffset_0;
  assign output_3_valid = output_3_valid_0;
  assign output_3_bits_vs = output_3_bits_vs_0;
  assign output_3_bits_offset = output_3_bits_offset_0;
  assign output_3_bits_writeIndex = output_3_bits_writeIndex_0;
  assign output_3_bits_dataOffset = output_3_bits_dataOffset_0;
endmodule

