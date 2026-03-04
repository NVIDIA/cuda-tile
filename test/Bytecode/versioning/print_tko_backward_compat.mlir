// Regression test for bytecode backward compatibility when an operation
// gains a new result in a newer version.
//
// In 13.1, `print` had 0 results.
// In 13.2, it was renamed to `print_tko` and gained 1 result (token).
//
// This test verifies that 13.1 bytecode containing `print` (0 results)
// can be correctly read by the 13.2 reader as `print_tko` (1 result),
// without corrupting SSA value numbering.
//

// COM: The 13.1 bytecode was generated from:
// COM: cuda_tile.module @kernels {
// COM:   entry @mutated_kernel(%arg0: tile<ptr<f64>>, %arg1: tile<ptr<f64>>, %arg2: tile<ptr<f64>>) {
// COM:     %assume = assume div_by<256>, %arg2 : tile<ptr<f64>>
// COM:     %assume_1 = assume div_by<256>, %arg0 : tile<ptr<f64>>
// COM:     %tview = make_tensor_view %assume_1, shape = [1024, 1024], strides = [1024, 1] : tensor_view<1024x1024xf64, strides=[1024,1]>
// COM:     %tview_3 = make_tensor_view %assume, shape = [1024, 512], strides = [512, 1] : tensor_view<1024x512xf64, strides=[512,1]>
// COM:     %pview = make_partition_view %tview_3 : partition_view<tile=(256x256), tensor_view<1024x512xf64, strides=[512,1]>>
// COM:     %pview_5 = make_partition_view %tview : partition_view<tile=(256x256), tensor_view<1024x1024xf64, strides=[1024,1]>>
// COM:     %blockId_x, %blockId_y, %blockId_z = get_tile_block_id : tile<i32>
// COM:     %tile, %result_token = load_view_tko weak %pview_5[%blockId_x, %blockId_y] : partition_view<tile=(256x256), tensor_view<1024x1024xf64, strides=[1024,1]>>, tile<i32> -> tile<256x256xf64>, token
// COM:     %0 = loop iter_values(%arg3 = %tile) : tile<256x256xf64> -> tile<256x256xf64> {
// COM:       print "Iteration result"  // <-- This was 0 results in 13.1
// COM:       %tile_6, %result_token_7 = load_view_tko weak %pview[%blockId_x, %blockId_y] : partition_view<tile=(256x256), tensor_view<1024x512xf64, strides=[512,1]>>, tile<i32> -> tile<256x256xf64>, token
// COM:       %2 = mmaf %tile_6, %tile_6, %arg3 : tile<256x256xf64>, tile<256x256xf64>, tile<256x256xf64>
// COM:       continue %2 : tile<256x256xf64>
// COM:     }
// COM:     return
// COM:   }
// COM: }

// RUN: cuda-tile-translate -cudatilebc-to-mlir %S/Inputs/13.1/print-op-13.1.tileirbc | FileCheck %s

// Verify the module structure is preserved
// CHECK: cuda_tile.module @kernels

// Verify print is now print_tko with a token result.
// CHECK: print_tko "Iteration result" -> token

// Verify mmaf gets tile operands, not token operands.
// CHECK: mmaf %{{.*}}, %{{.*}}, %{{.*}} : tile<256x256xf64>, tile<256x256xf64>, tile<256x256xf64>

