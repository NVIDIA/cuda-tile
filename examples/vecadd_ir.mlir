func @main(%A_ptr: tile<ptr<f32>>, %B_ptr: tile<ptr<f32>>, %C_ptr: tile<ptr<f32>>) {
  %bid, %_, %__ = tileir.get_tile_block_id : i32, i32, i32
  %A_tv = tileir.make_tensor_view %A_ptr : tensor_view<f32, [1024], [1]>
  %A_pv = tileir.make_partition_view %A_tv : partition_view<[128], tensor_view<f32, [1024], [1]>, [0], zero>
  %B_tv = tileir.make_tensor_view %B_ptr : tensor_view<f32, [1024], [1]>
  %B_pv = tileir.make_partition_view %B_tv : partition_view<[128], tensor_view<f32, [1024], [1]>, [0], zero>
  %C_tv = tileir.make_tensor_view %C_ptr : tensor_view<f32, [1024], [1]>
  %C_pv = tileir.make_partition_view %C_tv : partition_view<[128], tensor_view<f32, [1024], [1]>, [0], zero>
  %tok0 = tileir.make_token : token
  %A_tile, %tok1 = tileir.load_view %A_pv[%bid], %tok0 : tile<f32, [128]>, token
  %B_tile, %tok2 = tileir.load_view %B_pv[%bid], %tok1 : tile<f32, [128]>, token
  %v3 = tileir.addf %A_tile, %B_tile : tile<f32, [128]>
  %tok3 = tileir.join_tokens %tok1, %tok2 : token
  tileir.store_view %v3, %C_pv[%bid], %tok3 : tile<f32, [128]>, token
  tileir.return
}