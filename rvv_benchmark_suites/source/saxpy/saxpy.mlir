func.func @saxpy_vl_STEP_PLACEHOLDER(%size : index, %x: memref<?xf32>, %y: memref<?xf32>, %a: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // %size = memref.dim %x, %c0 : memref<?xf32>

  %vs = vector.vscale
  %factor = arith.constant STEP_PLACEHOLDER : index
  %step = arith.muli %vs, %factor : index
  
  %i_bound_ = arith.subi %size, %step : index
  %i_bound = arith.addi %i_bound_, %c1 : index

  %a_vec = vector.broadcast %a : f32 to vector<[STEP_PLACEHOLDER]xf32>

  // body
  %iter_idx = scf.for %i = %c0 to %i_bound step %step 
      iter_args(%iter = %c0) -> (index) {
    %xi = vector.load %x[%i] : memref<?xf32>, vector<[STEP_PLACEHOLDER]xf32>
    %yi = vector.load %y[%i] : memref<?xf32>, vector<[STEP_PLACEHOLDER]xf32>
    %updated_yi = vector.fma %a_vec, %xi, %yi : vector<[STEP_PLACEHOLDER]xf32>
    vector.store %updated_yi, %y[%i] : memref<?xf32>, vector<[STEP_PLACEHOLDER]xf32>
    %i_next = arith.addi %i, %step : index
    scf.yield %i_next : index
  }
  // tail
  affine.for %i = %iter_idx to %size {
    %xi = affine.load %x[%i] : memref<?xf32>
    %yi = affine.load %y[%i] : memref<?xf32>
    %axi = arith.mulf %a, %xi : f32
    %updated_yi = arith.addf %axi, %yi : f32
    affine.store %updated_yi, %y[%i] : memref<?xf32>
  }
  return
}
