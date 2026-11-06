// Round-trips every public-preview access and synchronization enum value
// through the public operations that own those attributes.
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module \
// RUN:   --preview=simt %s -o /dev/null

cuda_tile.module @kernels {
  cuda_tile.preview$func extern @access_modes(
      %input: !cuda_tile.tile<1xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %output: !cuda_tile.tile<1xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<out>},
      %inout: !cuda_tile.tile<1xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
    -> (!cuda_tile.tile<1xf32>, !cuda_tile.tile<1xf32>)

  cuda_tile.preview$func extern @synchronization_target(
      %input: !cuda_tile.tile<1xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  cuda_tile.preview$extern_impl @sync_none
      implements @synchronization_target(
        %input: !cuda_tile.preview$fragment<1xf32, local,
          mem_layout = #cuda_tile.preview$layout<(1):(1)>,
          fragmap = #cuda_tile.preview$layout<(1, 1, 1):(0, 0, 0)>> {
            cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
      synchronization = #cuda_tile.preview$synchronization<none>

  cuda_tile.preview$extern_impl @sync_warp
      implements @synchronization_target(
        %input: !cuda_tile.preview$fragment<1xf32, local,
          mem_layout = #cuda_tile.preview$layout<(1):(1)>,
          fragmap = #cuda_tile.preview$layout<(1, 32, 1):(0, 0, 0)>> {
            cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
      synchronization = #cuda_tile.preview$synchronization<warp>

  cuda_tile.preview$extern_impl @sync_warpgroup
      implements @synchronization_target(
        %input: !cuda_tile.preview$fragment<1xf32, local,
          mem_layout = #cuda_tile.preview$layout<(1):(1)>,
          fragmap = #cuda_tile.preview$layout<(1, 128, 1):(0, 0, 0)>> {
            cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
      synchronization = #cuda_tile.preview$synchronization<warpgroup>

  cuda_tile.preview$extern_impl @sync_cta
      implements @synchronization_target(
        %input: !cuda_tile.preview$fragment<1xf32, local,
          mem_layout = #cuda_tile.preview$layout<(1):(1)>,
          fragmap = #cuda_tile.preview$layout<(1, 128, 1):(0, 0, 0)>> {
            cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
      synchronization = #cuda_tile.preview$synchronization<cta>
}
