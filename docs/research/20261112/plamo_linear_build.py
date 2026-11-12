import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path


def generated_kernel(root):
    source = (root / 'src/shelf/kernels/optimized/linear/exl3_family/mul24/decode.hip').read_text()
    start = source.index('template <int BITS, int MT, bool kHighOcc>')
    end = source.index('template <int BITS>\nhipError_t launch_gemm_bits', start)
    kernel = source[start:end]
    kernel = kernel.replace('template <int BITS, int MT, bool kHighOcc>',
                            'template <int BITS, int MT, bool kHighOcc, bool CACHED, int WAVES = 4, bool PIPE = true>')
    kernel = kernel.replace('__launch_bounds__(128)', '__launch_bounds__(WAVES * 32)')
    kernel = kernel.replace('nb * 4u + wave', 'nb * WAVES + wave')
    kernel = kernel.replace('const uint32_t row_blocks)', 'const uint32_t row_blocks, const int8_t* cached)')
    kernel = kernel.replace('for (uint32_t s = 0u; s < 8u; ++s) {\n        const uint32_t* tile',
                            'for (uint32_t s = 0u; s < (CACHED ? 0u : 8u); ++s) {\n        const uint32_t* tile', 1)
    kernel = kernel.replace('if (g + 1u < g1)', 'if constexpr (!CACHED && PIPE) if (g + 1u < g1)')
    begin = kernel.index('            const uint32_t w0 = cur[s][0];')
    finish = kernel.index('            #pragma unroll\n            for (int mt = 0; mt < MT; ++mt) {\n                acc[mt]', begin)
    decode = kernel[begin:finish]
    decode = decode.replace('            int32x2_t b;\n', '')
    decode = decode.replace('const uint32_t w0 = cur[s][0];', 'const uint32_t* tile = tile_base + static_cast<size_t>(g * 8u + s) * kt_stride;\n            const uint32_t w0 = PIPE ? cur[s][0] : tile[i0];')
    decode = decode.replace('const uint32_t w1 = cur[s][1];', 'const uint32_t w1 = PIPE ? cur[s][1] : tile[i1];')
    decode = decode.replace('const uint32_t w2 = cur[s][2];', 'const uint32_t w2 = PIPE ? cur[s][2] : (kNeedW2 ? tile[i2] : 0u);')
    kernel = kernel[:begin] + '''            int32x2_t b;
            if constexpr (CACHED) {
                b = *reinterpret_cast<const int32x2_t*>(cached +
                    (static_cast<size_t>(nt) * (k >> 4u) + g * 8u + s) * 256u + lane * 8u);
            } else {
''' + decode + '            }\n' + kernel[finish:]
    kernel = kernel.replace('for (uint32_t s = 0u; s < 8u; ++s) {\n            cur[s][0] = nxt[s][0];',
                            'for (uint32_t s = 0u; s < ((!CACHED && PIPE) ? 8u : 0u); ++s) {\n            cur[s][0] = nxt[s][0];')
    kernel = '\n'.join(line for line in kernel.splitlines() if not line.lstrip().startswith('//'))
    inst = '''
template<int B, int MT, bool CACHE, int WAVES = 4, bool PIPE = true>
hipError_t launch_variant(const int8_t* a, const float* s, uint32_t cs, uint32_t ss,
    const uint16_t* t, const int8_t* w, float* y, uint32_t m, uint32_t n, uint32_t k,
    uint32_t sp, hipStream_t stream) {
    const uint32_t rb = (m + 16u * MT - 1u) / (16u * MT);
    exl3_mul24_gemm_kernel<B, MT, B == 4, CACHE, WAVES, PIPE>
        <<<dim3(n / (WAVES * 16u) * rb, 1, sp), WAVES * 32, 0, stream>>>(
            a, s, cs, ss / 4u, reinterpret_cast<const uint32_t*>(t), y, m, nullptr,
            n, k, (k / 128u + sp - 1u) / sp, rb, w);
    return hipGetLastError();
}
template<int B>
hipError_t variant_bits(int variant, int mt, const int8_t* a, const float* s, uint32_t cs,
    uint32_t ss, const uint16_t* t, const int8_t* w, float* y, uint32_t m, uint32_t n,
    uint32_t k, uint32_t sp, hipStream_t stream) {
    if (variant == 1) {
        if (mt == 1) return launch_variant<B, 1, false>(a,s,cs,ss,t,w,y,m,n,k,sp,stream);
        if (mt == 2) return launch_variant<B, 2, false>(a,s,cs,ss,t,w,y,m,n,k,sp,stream);
        if (mt == 3) return launch_variant<B, 3, false>(a,s,cs,ss,t,w,y,m,n,k,sp,stream);
        return launch_variant<B, 4, false>(a,s,cs,ss,t,w,y,m,n,k,sp,stream);
    }
    if (variant == 2) {
        if (mt == 1) return launch_variant<B, 1, true>(a,s,cs,ss,t,w,y,m,n,k,sp,stream);
        if (mt == 2) return launch_variant<B, 2, true>(a,s,cs,ss,t,w,y,m,n,k,sp,stream);
        return launch_variant<B, 4, true>(a,s,cs,ss,t,w,y,m,n,k,sp,stream);
    }
    if (variant == 3) {
        if (m <= 16) return launch_variant<B, 1, false, 8>(a,s,cs,ss,t,w,y,m,n,k,sp,stream);
        return launch_variant<B, 4, false, 8>(a,s,cs,ss,t,w,y,m,n,k,sp,stream);
    }
    if (m <= 16) return launch_variant<B, 1, false, 4, false>(a,s,cs,ss,t,w,y,m,n,k,sp,stream);
    return launch_variant<B, 4, false, 4, false>(a,s,cs,ss,t,w,y,m,n,k,sp,stream);
}
hipError_t variant_gemm(int variant, int mt, const int8_t* a, const float* s, uint32_t cs,
    uint32_t ss, const uint16_t* t, const int8_t* w, float* y, uint32_t m, uint32_t n,
    uint32_t k, uint32_t bits, uint32_t sp, hipStream_t stream) {
    if (bits == 4) return variant_bits<4>(variant,mt,a,s,cs,ss,t,w,y,m,n,k,sp,stream);
    if (bits == 5) return variant_bits<5>(variant,mt,a,s,cs,ss,t,w,y,m,n,k,sp,stream);
    return variant_bits<8>(variant,mt,a,s,cs,ss,t,w,y,m,n,k,sp,stream);
}
'''
    extra = generated_extra(source)
    grouped = generated_grouped(root, kernel)
    return '''#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>
namespace plamo_research {
using namespace shelf::kernel;
using namespace shelf::kernel::exl3_detail;
using namespace shelf::kernel::exl3_detail::mul24;
''' + kernel + inst + extra + grouped + '\n}\n'


def kernel_body(source, name):
    start = source.index('{', source.index(name + '('))
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start+1:end-1]


def generated_grouped(root, kernel):
    name = 'exl3_mul24_gemm_kernel('
    start = kernel.index(name) + len(name)
    core = '''template<int BITS,int MT,bool kHighOcc,bool CACHED,int WAVES=4,bool PIPE=true>
__device__ __forceinline__ void group_body(''' + kernel[start:]
    core = core.replace('const int8_t* cached)', 'const int8_t* cached, uint32_t tile_x, uint32_t tile_z)')
    core = core.replace('blockIdx.x', 'tile_x').replace('blockIdx.z', 'tile_z')
    rotate = (root/'src/shelf/kernels/optimized/linear/exl3_family/mul24/rotate_quantize.hip').read_text()
    rbody = kernel_body(rotate, 'exl3_mul24_rotate_quantize_kernel')
    fbody = kernel_body(rotate, 'exl3_mul24_rotate_quantize_f4_kernel')
    obody = kernel_body(generated_extra((root/'src/shelf/kernels/optimized/linear/exl3_family/mul24/decode.hip').read_text()), 'tiled_out_kernel')
    prefix = '''
struct Projection {
    const uint16_t *trellis, *suh, *svh;
    int8_t* codes;
    float *scales, *yprime;
    shelf::bf16_t* out;
    uint32_t n, splits;
};
struct ProjectionGroup {
    const shelf::bf16_t* input;
    Projection p[3];
    uint32_t rows,k,ss,count;
};
__global__ void __launch_bounds__(128) group_rotate(ProjectionGroup args) {
    const auto p=args.p[blockIdx.z];
    const auto input=args.input;
    const auto suh=p.suh;
    const auto codes=p.codes;
    const auto scales=p.scales;
    const uint32_t input_row_stride=args.k, code_row_stride=args.k, scale_row_stride_floats=args.ss/4;
''' + rbody + '''
}
__global__ void __launch_bounds__(256) group_rotate_f4(ProjectionGroup args) {
    const auto p=args.p[blockIdx.z];
    const auto input=args.input;
    const auto suh=p.suh;
    const auto codes=p.codes;
    const auto scales=p.scales;
    const uint32_t input_row_stride=args.k, code_row_stride=args.k, scale_row_stride_floats=args.ss/4, groups=args.k/128u;
''' + fbody + '''
}
__global__ void __launch_bounds__(128) group_out(ProjectionGroup args) {
    const auto p=args.p[blockIdx.z];
    const auto yprime=p.yprime;
    const auto svh=p.svh;
    const auto out=p.out;
    const uint32_t n=p.n, rows=args.rows, splits=p.splits, out_row_stride=p.n;
    constexpr bool BF16=true;
    constexpr int WAVES=4;
''' + obody + '\n}\n'
    suffix = '''
template<int B,int MT>
__global__ void __launch_bounds__(128) group_gemm(ProjectionGroup args) {
    uint32_t tile=blockIdx.x;
    const uint32_t rb=(args.rows+16u*MT-1u)/(16u*MT);
    for(uint32_t i=0;i<args.count;++i) {
        const auto p=args.p[i];
        const uint32_t blocks=p.n/64u*rb, total=blocks*p.splits;
        if(tile<total) {
            group_body<B,MT,B==4,false>(p.codes,p.scales,args.k,args.ss/4,
                reinterpret_cast<const uint32_t*>(p.trellis),p.yprime,args.rows,nullptr,
                p.n,args.k,(args.k/128u+p.splits-1u)/p.splits,rb,nullptr,tile%blocks,tile/blocks);
            return;
        }
        tile-=total;
    }
}
template<int B,int MT>
hipError_t launch_group_bits(ProjectionGroup args) {
    uint32_t blocks=0;
    const uint32_t rb=(args.rows+16u*MT-1)/(16u*MT);
    for(uint32_t i=0;i<args.count;++i)blocks+=args.p[i].n/64u*rb*args.p[i].splits;
    group_gemm<B,MT><<<blocks,128>>>(args);
    return hipGetLastError();
}
hipError_t launch_group(ProjectionGroup args,uint32_t bits) {
    if(args.rows>=32)group_rotate_f4<<<dim3(args.rows,1,args.count),256>>>(args);
    else group_rotate<<<dim3(args.rows,args.k/128u,args.count),128>>>(args);
    hipError_t err=hipGetLastError();if(err!=hipSuccess)return err;
    if(bits==4)err=args.rows==1?launch_group_bits<4,1>(args):launch_group_bits<4,4>(args);
    else err=args.rows==1?launch_group_bits<5,1>(args):launch_group_bits<5,4>(args);
    if(err!=hipSuccess)return err;
    uint32_t n=0;for(uint32_t i=0;i<args.count;++i)n=max(n,args.p[i].n);
    group_out<<<dim3(args.rows,(n/128u+3u)/4u,args.count),128>>>(args);
    return hipGetLastError();
}
'''
    return prefix + core + suffix


def generated_extra(source):
    start = source.index('template <bool BF16>\n__global__ void __launch_bounds__(256)')
    end = source.index('// Prefill path', start)
    epi = source[start:end]
    epi = epi.replace('template <bool BF16>', 'template <bool BF16, int WAVES>')
    epi = epi.replace('__launch_bounds__(256)', '__launch_bounds__(WAVES * 32)')
    epi = epi.replace('exl3_mul24_rotate_out_f4_kernel', 'tiled_out_kernel')
    epi = epi.replace('blockIdx.x * 8u + (threadIdx.x >> 5u)', 'blockIdx.x')
    epi = epi.replace('for (uint32_t cb = 0u; cb < n / kBlock; ++cb)',
                      'for (uint32_t cb = blockIdx.y * WAVES + (threadIdx.x >> 5u); cb < min(n / kBlock, (blockIdx.y + 1u) * WAVES); ++cb)')
    epi = '\n'.join(line for line in epi.splitlines() if not line.lstrip().startswith('//'))
    dot = '''
template<int BITS, int MR>
__global__ void __launch_bounds__(128) dot_kernel(const int8_t* codes, const float* scales,
    uint32_t cs, uint32_t ss, const uint32_t* trellis, float* out,
    uint32_t rows, uint32_t n, uint32_t k, uint32_t gps) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t nt = blockIdx.x * 4u + (threadIdx.x >> 5u);
    const uint32_t half = lane >> 4u;
    const uint32_t col = lane & 15u;
    const uint32_t m0 = blockIdx.y * MR;
    constexpr uint32_t kWords = 8u * BITS, kBits = 256u * BITS;
    const uint32_t base = (lane * 8u * BITS + BITS + kBits - 16u) % kBits;
    const uint32_t i0 = base >> 5u, off = base & 31u;
    const uint32_t g0 = blockIdx.z * gps, g1 = min(k / 128u, g0 + gps);
    float acc[MR] = {};
    for(uint32_t g=g0;g<g1;++g) {
        int sums[MR] = {};
        #pragma unroll
        for(uint32_t s=0;s<8;++s) {
            const uint32_t* tile = trellis + (static_cast<size_t>(nt) * (k / 16u) + g*8u+s)*kWords;
            const uint32_t w0=tile[i0], w1=tile[(i0+1)%kWords], w2=tile[(i0+2)%kWords];
            uint32_t packed[2]={};
            #pragma unroll
            for(uint32_t i=0;i<8;++i) {
                const uint32_t e=off+i*BITS+16u, wl=(e-1u)>>5u, r=32u-(e-32u*wl);
                const uint32_t lo=wl==0?w0:(wl==1?w1:w2), hi=wl==0?0:(wl==1?w0:w1);
                const uint32_t state=__builtin_amdgcn_alignbit(hi,lo,r)&0xffffu;
                const uint32_t q=__builtin_amdgcn_udot4(__umul24(state,0xDCD12Du),0x01010101u,3u,false)>>2u;
                packed[i>>2u] |= q<<((i&3u)*8u);
            }
            #pragma unroll
            for(int r=0;r<MR;++r) if(m0+r<rows) {
                const auto a=*reinterpret_cast<const int32x2_t*>(codes+static_cast<size_t>(m0+r)*cs+g*128u+s*16u+half*8u);
                sums[r]=__builtin_amdgcn_sudot4(true,a[0],true,packed[0]^0x80808080u,sums[r],false);
                sums[r]=__builtin_amdgcn_sudot4(true,a[1],true,packed[1]^0x80808080u,sums[r],false);
            }
        }
        #pragma unroll
        for(int r=0;r<MR;++r) {
            const int sum=sums[r]+__shfl_xor(sums[r],16);
            const float scale=m0+r<rows?scales[static_cast<size_t>(m0+r)*ss+g]:0.0f;
            acc[r]+=static_cast<float>(sum)*scale;
        }
    }
    #pragma unroll
    for(int r=0;r<MR;++r) if(half==0 && m0+r<rows)
        out[(static_cast<size_t>(blockIdx.z)*rows+m0+r)*n+nt*16u+col]=acc[r]*shelf::quantization::exl3::kExl3Int8GridStep;
}
template<int B>
hipError_t launch_dot_bits(const int8_t* a,const float* s,uint32_t cs,uint32_t ss,
    const uint16_t* t,float* y,uint32_t m,uint32_t n,uint32_t k,uint32_t sp) {
    if(m==1) dot_kernel<B,1><<<dim3(n/64u,1,sp),128>>>(a,s,cs,ss/4,reinterpret_cast<const uint32_t*>(t),y,m,n,k,(k/128u+sp-1)/sp);
    else dot_kernel<B,4><<<dim3(n/64u,(m+3)/4,sp),128>>>(a,s,cs,ss/4,reinterpret_cast<const uint32_t*>(t),y,m,n,k,(k/128u+sp-1)/sp);
    return hipGetLastError();
}
hipError_t launch_dot(const int8_t* a,const float* s,uint32_t cs,uint32_t ss,
    const uint16_t* t,float* y,uint32_t m,uint32_t n,uint32_t k,uint32_t b,uint32_t sp) {
    if(b==4)return launch_dot_bits<4>(a,s,cs,ss,t,y,m,n,k,sp);
    if(b==5)return launch_dot_bits<5>(a,s,cs,ss,t,y,m,n,k,sp);
    return launch_dot_bits<8>(a,s,cs,ss,t,y,m,n,k,sp);
}
hipError_t tiled_out(const float* y,const uint16_t* svh,void* out,uint32_t m,uint32_t n,uint32_t sp) {
    tiled_out_kernel<true,4><<<dim3(m,(n/128u+3u)/4u),128>>>(y,svh,out,n,n,sp,m);
    return hipGetLastError();
}
'''
    return epi + dot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', type=Path, default=Path('build-gfx1201'))
    parser.add_argument('--wide-mt2', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    build = args.build_dir.resolve()
    output = build / 'plamo-linear'
    output.mkdir(exist_ok=True)
    source = output / 'micro.hip'
    selected = (root / 'src/shelf/kernels/optimized/linear/exl3_family/mul24/projection.hip').read_text()
    if args.wide_mt2:
        selected, count = re.subn(r'err = args\.rows <= 16u \? launch_group_bits<4, 1, true>\(args, stream\).*?;',
            'err = args.rows <= 16u ? launch_group_bits<4, 1, true>(args, stream) : launch_group_bits<4, 2, true>(args, stream);',
            selected,count=1,flags=re.S)
        if count != 1:
            raise ValueError('wide MT2 selector anchor missing')
    selected = selected.replace('namespace shelf::kernel {', 'namespace plamo_selected {\nusing namespace shelf::kernel;', 1)
    selected += '''
namespace plamo_selected {
hipError_t selected_stage(int stage,PlamoExl3ProjectionGroup args,uint32_t bits) {
    if(stage==0) {
        if(args.rows>=32)group_rotate_f4<<<dim3(args.rows,1,args.count),256>>>(args);
        else group_rotate<<<dim3(args.rows,args.k/128u,args.count),128>>>(args);
    } else if(stage==1) {
        if(bits==4)return args.rows<=16?launch_group_bits<4,1,true>(args,nullptr):args.rows<=128?launch_group_bits<4,2,true>(args,nullptr):launch_group_bits<4,4,true>(args,nullptr);
        if(bits==5)return args.rows<=16?launch_group_bits<5,1,true>(args,nullptr):launch_group_bits<5,4,false>(args,nullptr);
        return args.rows<=16?launch_group_bits<8,1,true>(args,nullptr):launch_group_bits<8,2,true>(args,nullptr);
    } else group_out<<<dim3(args.rows,(args.p[0].n/128u+3u)/4u,args.count),128>>>(args);
    return hipGetLastError();
}
}
'''
    if args.wide_mt2:
        selected = selected.replace('args.rows<=128?launch_group_bits<4,2,true>(args,nullptr):launch_group_bits<4,4,true>(args,nullptr)',
                                    'launch_group_bits<4,2,true>(args,nullptr)')
    source.write_text(generated_kernel(root) + selected + (Path(__file__).with_name('plamo_linear_micro.hip')).read_text())
    commands = json.loads((build / 'compile_commands.json').read_text())
    command = shlex.split(next(c['command'] for c in commands if c['file'].endswith('test_exl3_mul24_linear.hip')))
    pos = command.index('-o')
    command = command[:pos] + ['-x', 'hip', str(source), '-o', str(output / 'micro'),
                              '-x', 'none', str(build / 'libshelf_kernels_optimized.a'),
                              '-L/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib',
                              '-L/opt/rocm/core-7.14/lib', '-lamdhip64', '-lc++', '-lc++abi']
    subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()
