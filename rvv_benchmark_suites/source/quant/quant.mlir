func.func @quant_vl_STEP_PLACEHOLDER(%data: memref<?xf32>, %res: memref<?xi8>, 
                                    %size: index, %scale: f32, %zero_point: i8) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %int8_min = arith.constant -128 : i32
  %int8_max = arith.constant 127 : i32
  %int8_min_vec = vector.broadcast %int8_min : i32 to vector<[STEP_PLACEHOLDER]xi32>
  %int8_max_vec = vector.broadcast %int8_max : i32 to vector<[STEP_PLACEHOLDER]xi32>
  %scale_vec = vector.broadcast %scale : f32 to vector<[STEP_PLACEHOLDER]xf32>
  %zero_point_i32 = arith.extsi %zero_point : i8 to i32
  %zero_point_vec = vector.broadcast %zero_point_i32 : i32 to vector<[STEP_PLACEHOLDER]xi32>

  %vs = vector.vscale
  %factor = arith.constant STEP_PLACEHOLDER : index
  %vl_step = arith.muli %vs, %factor : index
  %vl_ub_ = arith.subi %size, %vl_step : index
  %vl_ub = arith.addi %vl_ub_, %c1 : index
  // Vectorized loop for quantization
  %iter_idx = scf.for %i = %c0 to %vl_ub step %vl_step 
    iter_args(%iter_init = %c0) -> (index) {
    // Load a batch of float data
    %data_vec = vector.load %data[%i] : memref<?xf32>, vector<[STEP_PLACEHOLDER]xf32>
    // Quantize: scaled_data = int(round(data / scale)) + zero_point
    %div_scale = arith.divf %data_vec, %scale_vec : vector<[STEP_PLACEHOLDER]xf32>
    %round_f32 = math.round %div_scale : vector<[STEP_PLACEHOLDER]xf32>
    %round_i32 = arith.fptosi %round_f32: vector<[STEP_PLACEHOLDER]xf32> to vector<[STEP_PLACEHOLDER]xi32>
    %add_zero_point = arith.addi %round_i32, %zero_point_vec : vector<[STEP_PLACEHOLDER]xi32>
    //  Quantize: Clamp to int8 range
    %clamp_min = arith.minsi %add_zero_point, %int8_max_vec : vector<[STEP_PLACEHOLDER]xi32>
    %clamp_max = arith.maxsi %clamp_min, %int8_min_vec : vector<[STEP_PLACEHOLDER]xi32>
    %clamp_i8 = arith.trunci %clamp_max : vector<[STEP_PLACEHOLDER]xi32> to vector<[STEP_PLACEHOLDER]xi8>
    // Store quantized values
    vector.store %clamp_i8, %res[%i] : memref<?xi8>, vector<[STEP_PLACEHOLDER]xi8>
    %i_next = arith.addi %i, %vl_step : index
    scf.yield %i_next : index
  }
  
  // Scalar loop for remaining elements
  scf.for %tail_i = %iter_idx to %size step %c1 {
    // Load a batch of float data
    %data_ele = memref.load %data[%tail_i] : memref<?xf32>
    // Quantize: scaled_data = int(round(data / scale)) + zero_point
    %div_scale = arith.divf %data_ele, %scale : f32
    %round_f32 = math.round %div_scale : f32
    %round_i32 = arith.fptosi %round_f32: f32 to i32
    %add_zero_point = arith.addi %round_i32, %zero_point_i32 : i32
    //  Quantize: Clamp to int8 range
    %clamp_min = arith.minsi %add_zero_point, %int8_max : i32
    %clamp_max = arith.maxsi %clamp_min, %int8_min : i32
    %clamp_i8 = arith.trunci %clamp_max : i32 to i8
    // Store quantized values
    memref.store %clamp_i8, %res[%tail_i] : memref<?xi8>
  }
  return
}

func.func @dequant_vs_STEP_PLACEHOLDER(%quantized_data: memref<?xi8>, %res: memref<?xf32>,
                                      %size: index, %scale: f32, %zero_point: i8) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero_point_vec = vector.broadcast %zero_point : i8 to vector<[STEP_PLACEHOLDER]xi8>
  %scale_vec = vector.broadcast %scale : f32 to vector<[STEP_PLACEHOLDER]xf32>

  %vl_step = arith.constant STEP_PLACEHOLDER : index
  %vl_ub_ = arith.subi %size, %vl_step : index
  %vl_ub = arith.addi %vl_ub_, %c1 : index

  // Vectorized loop for dequantization
  %iter_idx = scf.for %i = %c0 to %vl_ub step %vl_step 
    iter_args(%iter_init = %c0) -> (index) {
    %quantized_vec = vector.load %quantized_data[%i] : memref<?xi8>, vector<[STEP_PLACEHOLDER]xi8>
    %sub_zero_point = arith.subi %quantized_vec, %zero_point_vec : vector<[STEP_PLACEHOLDER]xi8>
    %quantized_i32 = arith.extsi %sub_zero_point : vector<[STEP_PLACEHOLDER]xi8> to vector<[STEP_PLACEHOLDER]xi32>
    %quantized_f32 = arith.sitofp %quantized_i32 : vector<[STEP_PLACEHOLDER]xi32> to vector<[STEP_PLACEHOLDER]xf32>
    %result = arith.mulf %quantized_f32, %scale_vec : vector<[STEP_PLACEHOLDER]xf32>
    vector.store %result, %res[%i] : memref<?xf32>, vector<[STEP_PLACEHOLDER]xf32>
    %i_next = arith.addi %i, %vl_step : index
    scf.yield %i_next : index
  }

  // Scalar loop for remaining elements
  scf.for %tail_i = %iter_idx to %size step %c1 {
    %quantized_ele = memref.load %quantized_data[%tail_i] : memref<?xi8>
    %sub_zero_point = arith.subi %quantized_ele, %zero_point : i8
    %quantized_i32 = arith.extsi %sub_zero_point : i8 to i32
    %quantized_f32 = arith.sitofp %quantized_i32 : i32 to f32
    %result = arith.mulf %quantized_f32, %scale : f32
    memref.store %result, %res[%tail_i] : memref<?xf32>
  }
  return
}
