func.func @pack_vl_STEP_PLACEHOLDER(%src: memref<?xui8>, %dst: memref<?xui32>,
                                    %src_size: index, %dst_size: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index

  // VectorIR
  // vector.set_vl %dst_size {
  //   %vec_u8 = vector.load %src
  //   %vec_u32 = vector.bitcast %vec_u8
  //   vector.store %vec_u32
  // }

  // STEP_PLACEHOLDER 代表 DST(ui32) 的 VL
  // STEP_4_PLACEHOLDER 代表 SRC(ui8) 的 VL

  %vl_step = arith.constant STEP_PLACEHOLDER : index
  %vl_ub_ = arith.subi %dst_size, %vl_step : index
  %vl_ub = arith.addi %vl_ub_, %c1 : index

  %iter_idx = scf.for %i = %c0 to %vl_ub step %vl_step
      iter_args(%iter_init = %c0) -> (index) {
    %i_u8 = arith.muli %i, %c4 : index
    %vec_u8 = vector.load %src[%i_u8] : memref<?xui8>, vector<STEP_4_PLACEHOLDERxui8>
    %vec_u32 = vector.bitcast %vec_u8 : vector<STEP_4_PLACEHOLDERxui8> to vector<STEP_PLACEHOLDERxui32>
    vector.store %vec_u32, %dst[%i] : memref<?xui32>, vector<STEP_PLACEHOLDERxui32>
    %i_next = arith.addi %i, %vl_step : index
    scf.yield %i_next : index
  }

  return
}
