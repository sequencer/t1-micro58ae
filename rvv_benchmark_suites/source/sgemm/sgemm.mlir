// Modify from https://github.com/buddy-compiler/buddy-mlir

func.func @sgemm_vs_STEP_PLACEHOLDER(%mm : i32, %nn : i32, %kk : i32, %a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index

  %m = arith.index_cast %mm : i32 to index
  %n = arith.index_cast %nn : i32 to index
  %k = arith.index_cast %kk : i32 to index

  // 向量化处理 beta
  %vs = vector.vscale
  %factor = arith.constant STEP_PLACEHOLDER : index
  %step = arith.muli %vs, %factor : index

  affine.for %m_idx = 0 to %m step 8 {
    %m_idx_1 = arith.addi %m_idx, %c1 : index
    %m_idx_2 = arith.addi %m_idx, %c2 : index
    %m_idx_3 = arith.addi %m_idx, %c3 : index
    %m_idx_4 = arith.addi %m_idx, %c4 : index
    %m_idx_5 = arith.addi %m_idx, %c5 : index
    %m_idx_6 = arith.addi %m_idx, %c6 : index
    %m_idx_7 = arith.addi %m_idx, %c7 : index
    
    // VectorIR
    // vector.setvl %N {
    //   %sum_iter_vec = affine.for %n_idx = 0 to %n
    //       iter_args(%sum_vec = %sum_init) -> (vector<[STEP_PLACEHOLDER]xf32>) {
    //     %a_ele = memref.load %a[%m_idx, %n_idx] : memref<?x?xf32>
    //     %a_vec = vector.broadcast %a_ele : f32 to vector<[STEP_PLACEHOLDER]xf32>
    //     %b_vec = vector.load %b[%n_idx, %k_idx] : memref<?x?xf32>, vector<[STEP_PLACEHOLDER]xf32>
    //     %res_sum_vec = vector.fma %a_vec, %b_vec, %sum_vec : vector<[STEP_PLACEHOLDER]xf32>
    //     affine.yield %res_sum_vec : vector<[STEP_PLACEHOLDER]xf32>
    //   }
    //   vector.store %sum_iter_vec, %c[%m_idx, %k_idx] : memref<?x?xf32>, vector<[STEP_PLACEHOLDER]xf32>
    // }
    %n_body_bound_ = arith.subi %n, %step : index
    %n_body_bound = arith.addi %n_body_bound_, %c1 : index

    %n_iter_idx = scf.for %n_idx = %c0 to %n_body_bound step %step
        iter_args(%n_iter_idx_init = %c0) -> (index) {
      %sum_init_0 = arith.constant dense<0.> : vector<[STEP_PLACEHOLDER]xf32>
      %sum_init_1 = arith.constant dense<0.> : vector<[STEP_PLACEHOLDER]xf32>
      %sum_init_2 = arith.constant dense<0.> : vector<[STEP_PLACEHOLDER]xf32>
      %sum_init_3 = arith.constant dense<0.> : vector<[STEP_PLACEHOLDER]xf32>
      %sum_init_4 = arith.constant dense<0.> : vector<[STEP_PLACEHOLDER]xf32>
      %sum_init_5 = arith.constant dense<0.> : vector<[STEP_PLACEHOLDER]xf32>
      %sum_init_6 = arith.constant dense<0.> : vector<[STEP_PLACEHOLDER]xf32>
      %sum_init_7 = arith.constant dense<0.> : vector<[STEP_PLACEHOLDER]xf32>
      %sum_iter_vec_0, %sum_iter_vec_1, %sum_iter_vec_2, %sum_iter_vec_3,
      %sum_iter_vec_4, %sum_iter_vec_5, %sum_iter_vec_6, %sum_iter_vec_7
          = affine.for %k_idx = 0 to %k
          iter_args(%sum_vec_0 = %sum_init_0,
                    %sum_vec_1 = %sum_init_1,
                    %sum_vec_2 = %sum_init_2,
                    %sum_vec_3 = %sum_init_3,
                    %sum_vec_4 = %sum_init_4,
                    %sum_vec_5 = %sum_init_5,
                    %sum_vec_6 = %sum_init_6,
                    %sum_vec_7 = %sum_init_7
                    ) 
          -> (vector<[STEP_PLACEHOLDER]xf32>, vector<[STEP_PLACEHOLDER]xf32>, vector<[STEP_PLACEHOLDER]xf32>, vector<[STEP_PLACEHOLDER]xf32>,
              vector<[STEP_PLACEHOLDER]xf32>, vector<[STEP_PLACEHOLDER]xf32>, vector<[STEP_PLACEHOLDER]xf32>, vector<[STEP_PLACEHOLDER]xf32>) {
        %a_ele_0 = memref.load %a[%m_idx, %k_idx] : memref<?x?xf32>
        %a_ele_1 = memref.load %a[%m_idx_1, %k_idx] : memref<?x?xf32>
        %a_ele_2 = memref.load %a[%m_idx_2, %k_idx] : memref<?x?xf32>
        %a_ele_3 = memref.load %a[%m_idx_3, %k_idx] : memref<?x?xf32>
        %a_ele_4 = memref.load %a[%m_idx_4, %k_idx] : memref<?x?xf32>
        %a_ele_5 = memref.load %a[%m_idx_5, %k_idx] : memref<?x?xf32>
        %a_ele_6 = memref.load %a[%m_idx_6, %k_idx] : memref<?x?xf32>
        %a_ele_7 = memref.load %a[%m_idx_7, %k_idx] : memref<?x?xf32>
        %a_vec_0 = vector.broadcast %a_ele_0 : f32 to vector<[STEP_PLACEHOLDER]xf32>
        %a_vec_1 = vector.broadcast %a_ele_1 : f32 to vector<[STEP_PLACEHOLDER]xf32>
        %a_vec_2 = vector.broadcast %a_ele_2 : f32 to vector<[STEP_PLACEHOLDER]xf32>
        %a_vec_3 = vector.broadcast %a_ele_3 : f32 to vector<[STEP_PLACEHOLDER]xf32>
        %a_vec_4 = vector.broadcast %a_ele_4 : f32 to vector<[STEP_PLACEHOLDER]xf32>
        %a_vec_5 = vector.broadcast %a_ele_5 : f32 to vector<[STEP_PLACEHOLDER]xf32>
        %a_vec_6 = vector.broadcast %a_ele_6 : f32 to vector<[STEP_PLACEHOLDER]xf32>
        %a_vec_7 = vector.broadcast %a_ele_7 : f32 to vector<[STEP_PLACEHOLDER]xf32>
        %b_vec = vector.load %b[%k_idx, %n_idx] : memref<?x?xf32>, vector<[STEP_PLACEHOLDER]xf32>
        %res_sum_vec_0 = vector.fma %a_vec_0, %b_vec, %sum_vec_0 : vector<[STEP_PLACEHOLDER]xf32>
        %res_sum_vec_1 = vector.fma %a_vec_1, %b_vec, %sum_vec_1 : vector<[STEP_PLACEHOLDER]xf32>
        %res_sum_vec_2 = vector.fma %a_vec_2, %b_vec, %sum_vec_2 : vector<[STEP_PLACEHOLDER]xf32>
        %res_sum_vec_3 = vector.fma %a_vec_3, %b_vec, %sum_vec_3 : vector<[STEP_PLACEHOLDER]xf32>
        %res_sum_vec_4 = vector.fma %a_vec_4, %b_vec, %sum_vec_4 : vector<[STEP_PLACEHOLDER]xf32>
        %res_sum_vec_5 = vector.fma %a_vec_5, %b_vec, %sum_vec_5 : vector<[STEP_PLACEHOLDER]xf32>
        %res_sum_vec_6 = vector.fma %a_vec_6, %b_vec, %sum_vec_6 : vector<[STEP_PLACEHOLDER]xf32>
        %res_sum_vec_7 = vector.fma %a_vec_7, %b_vec, %sum_vec_7 : vector<[STEP_PLACEHOLDER]xf32>
        affine.yield %res_sum_vec_0, %res_sum_vec_1, %res_sum_vec_2, %res_sum_vec_3,
                     %res_sum_vec_4, %res_sum_vec_5, %res_sum_vec_6, %res_sum_vec_7
            : vector<[STEP_PLACEHOLDER]xf32>, vector<[STEP_PLACEHOLDER]xf32>, vector<[STEP_PLACEHOLDER]xf32>, vector<[STEP_PLACEHOLDER]xf32>,
              vector<[STEP_PLACEHOLDER]xf32>, vector<[STEP_PLACEHOLDER]xf32>, vector<[STEP_PLACEHOLDER]xf32>, vector<[STEP_PLACEHOLDER]xf32>
      }
      vector.store %sum_iter_vec_0, %c[%m_idx, %n_idx] : memref<?x?xf32>, vector<[STEP_PLACEHOLDER]xf32>
      vector.store %sum_iter_vec_1, %c[%m_idx_1, %n_idx] : memref<?x?xf32>, vector<[STEP_PLACEHOLDER]xf32>
      vector.store %sum_iter_vec_2, %c[%m_idx_2, %n_idx] : memref<?x?xf32>, vector<[STEP_PLACEHOLDER]xf32>
      vector.store %sum_iter_vec_3, %c[%m_idx_3, %n_idx] : memref<?x?xf32>, vector<[STEP_PLACEHOLDER]xf32>
      vector.store %sum_iter_vec_4, %c[%m_idx_4, %n_idx] : memref<?x?xf32>, vector<[STEP_PLACEHOLDER]xf32>
      vector.store %sum_iter_vec_5, %c[%m_idx_5, %n_idx] : memref<?x?xf32>, vector<[STEP_PLACEHOLDER]xf32>
      vector.store %sum_iter_vec_6, %c[%m_idx_6, %n_idx] : memref<?x?xf32>, vector<[STEP_PLACEHOLDER]xf32>
      vector.store %sum_iter_vec_7, %c[%m_idx_7, %n_idx] : memref<?x?xf32>, vector<[STEP_PLACEHOLDER]xf32>
      %k_next = arith.addi %n_idx, %step : index
      scf.yield %k_next : index
    }
    // TODO: Add unroll to the tail processing.
    scf.for %n_idx = %n_iter_idx to %n step %c1 {
      %sum_init = arith.constant 0. : f32
      %sum_iter = affine.for %k_idx = 0 to %k
          iter_args(%sum = %sum_init) -> (f32) {
        %a_ele = memref.load %a[%m_idx, %k_idx] : memref<?x?xf32>
        %b_ele = memref.load %b[%k_idx, %n_idx] : memref<?x?xf32>
        %tmp_ele = arith.mulf %a_ele, %b_ele : f32
        %res_sum = arith.addf %tmp_ele, %sum : f32
        affine.yield %res_sum : f32
      }
      memref.store %sum_iter, %c[%m_idx, %n_idx] : memref<?x?xf32>
    }
  }
  return
}
