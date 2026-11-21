// Generated quantize-tiles instantiation for RDNA (v1.5.0). Mirrors
// quant/comp_units/quantize_tiles_inst_k4.cu with the kernel headers swapped
// for the RDNA siblings (which swap codebook.cuh for codebook_rdna.hip.h).
//
// The optimized getter now returns the ported dense specialisation
// (quantize_tiles_optimized_rdna.hip.h); quantize_rdna.hip enables it for
// K = 4 only. Bit-identity against the plain kernel is gated by
// rocm_tools/qt_bitwise.py.
//
// These exist because ROCM_EXCLUDE drops all of quant/comp_units/, which is
// where upstream defines these getters; without them the ROCm build fails at
// link with undefined quantize_tiles_kernel_k*_cb*.

#include "../../../quant/comp_units/quantize_tiles_instances.cuh"
#include "../quantize_tiles_kernel_rdna.hip.h"
#include "../quantize_tiles_optimized_rdna.hip.h"

fp_quantize_tiles_kernel quantize_tiles_kernel_k4_cb0(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<4, 0> : quantize_tiles_kernel<4, 0>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k4_cb1(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<4, 1> : quantize_tiles_kernel<4, 1>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k4_cb2(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<4, 2> : quantize_tiles_kernel<4, 2>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k4_cb2_l160(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<4, 2, 160> : quantize_tiles_kernel<4, 2, 160>; }
