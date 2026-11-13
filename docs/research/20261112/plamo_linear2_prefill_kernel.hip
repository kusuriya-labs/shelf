#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <hip/hip_runtime.h>
#include <cstdint>

#ifndef PLAMO_MT
#define PLAMO_MT 2
#endif
#ifndef PLAMO_RECON_GROUP_FAST
#define PLAMO_RECON_GROUP_FAST 1
#endif
#ifndef PLAMO_PIPE_B
#define PLAMO_PIPE_B 1
#endif
#ifndef PLAMO_SWIZZLE
#define PLAMO_SWIZZLE 1
#endif
#ifndef PLAMO_NT_B
#define PLAMO_NT_B 0
#endif
#ifndef PLAMO_NT_RECON
#define PLAMO_NT_RECON 0
#endif
#ifndef PLAMO_SCHED_BARRIER
#define PLAMO_SCHED_BARRIER 0
#endif
#ifndef PLAMO_WAVES_MIN
#define PLAMO_WAVES_MIN 1
#endif
#ifndef PLAMO_LAUNCH_BOUND
#define PLAMO_LAUNCH_BOUND 256
#endif

namespace plamo_prefill_study {
using namespace shelf::kernel::exl3_detail;
using namespace shelf::kernel::exl3_detail::mul24;
using I2=int32_t __attribute__((ext_vector_type(2)));
using I4=int32_t __attribute__((ext_vector_type(4)));
using I8=int32_t __attribute__((ext_vector_type(8)));
struct Args {
    const int8_t* codes;
    const float* scales;
    const uint16_t* trellis;
    const int8_t* cache;
    const uint16_t* svh;
    shelf::bf16_t* out;
    uint32_t rows,n,k,code_stride,scale_stride,out_stride;
};

template<int BITS>
__device__ __forceinline__ I2 decode(const uint32_t* tile,uint32_t lane) {
    constexpr uint32_t words=8u*BITS,bits=256u*BITS;
    const uint32_t bit=(lane*8u*BITS+BITS+bits-16u)%bits;
    const uint32_t i0=bit>>5u,off=bit&31u;
#if PLAMO_NT_RECON
    const uint32_t w0=__builtin_nontemporal_load(tile+i0),w1=__builtin_nontemporal_load(tile+(i0+1u)%words),w2=__builtin_nontemporal_load(tile+(i0+2u)%words);
#else
    const uint32_t w0=tile[i0],w1=tile[(i0+1u)%words],w2=tile[(i0+2u)%words];
#endif
    uint32_t p0=0u,p1=0u;
    #pragma unroll
    for(uint32_t i=0u;i<8u;++i) {
        const uint32_t e=off+i*BITS+16u,wl=(e-1u)>>5u,r=32u-(e-wl*32u);
        const uint32_t lo=wl==0u?w0:wl==1u?w1:w2,hi=wl==0u?0u:wl==1u?w0:w1;
        const uint32_t state=__builtin_amdgcn_alignbit(hi,lo,r)&0xffffu;
        const uint32_t q=__builtin_amdgcn_udot4(__umul24(state,0xDCD12Du),0x01010101u,3u,false)>>2u;
        if(i<4u)p0|=q<<(i*8u);else p1|=q<<((i-4u)*8u);
    }
    return I2{int32_t(p0^0x80808080u),int32_t(p1^0x80808080u)};
}

template<int BITS>
__global__ void __launch_bounds__(128) recon(const uint16_t* trellis,int8_t* cache,uint32_t k) {
    const uint32_t nt=PLAMO_RECON_GROUP_FAST?blockIdx.y:blockIdx.x,g=PLAMO_RECON_GROUP_FAST?blockIdx.x:blockIdx.y,lane=threadIdx.x>>2u,pair=threadIdx.x&3u;
    const uint32_t* source=reinterpret_cast<const uint32_t*>(trellis)+(size_t(nt)*(k/16u)+g*8u+pair*2u)*(8u*BITS);
    const I2 b0=decode<BITS>(source,lane),b1=decode<BITS>(source+8u*BITS,lane);
    *reinterpret_cast<I4*>(cache+((size_t(nt)*(k/128u)+g)*32u+lane)*64u+pair*16u)=I4{b0[0],b0[1],b1[0],b1[1]};
}

hipError_t pack(const uint16_t* trellis,int8_t* cache,uint32_t n,uint32_t k,uint32_t bits,hipStream_t stream=nullptr) {
    if(bits==4u)recon<4><<<dim3(PLAMO_RECON_GROUP_FAST?k/128u:n/16u,PLAMO_RECON_GROUP_FAST?n/16u:k/128u),128,0,stream>>>(trellis,cache,k);
    else if(bits==5u)recon<5><<<dim3(PLAMO_RECON_GROUP_FAST?k/128u:n/16u,PLAMO_RECON_GROUP_FAST?n/16u:k/128u),128,0,stream>>>(trellis,cache,k);
    else return hipErrorInvalidValue;
    return hipGetLastError();
}

template<int BITS,bool CACHE>
__device__ __forceinline__ I4 load_b(const Args& a,uint32_t nt,uint32_t group,uint32_t pair,uint32_t lane) {
    if constexpr(CACHE) {
        const I4* p=reinterpret_cast<const I4*>(a.cache+((size_t(nt)*(a.k/128u)+group)*32u+lane)*64u+pair*16u);
#if PLAMO_NT_B
        return __builtin_nontemporal_load(p);
#else
        return *p;
#endif
    } else {
        const uint32_t* p=reinterpret_cast<const uint32_t*>(a.trellis)+(size_t(nt)*(a.k/16u)+group*8u+pair*2u)*(8u*BITS);
        const I2 x=decode<BITS>(p,lane),y=decode<BITS>(p+8u*BITS,lane);
        return I4{x[0],x[1],y[0],y[1]};
    }
}

template<int MT>struct AStage {uint4 data[(MT*16u*8u+255u)/256u];float scale;};
template<int MT>
__device__ __forceinline__ AStage<MT> prefetch_a(const Args& a,uint32_t m0,uint32_t g) {
    AStage<MT> s{};
    #pragma unroll
    for(uint32_t j=0u;j<(MT*16u*8u+255u)/256u;++j) {
        const uint32_t t=threadIdx.x+j*256u,r=t/8u,chunk=t&7u;
        if(r<MT*16u && m0+r<a.rows)s.data[j]=*reinterpret_cast<const uint4*>(a.codes+size_t(m0+r)*a.code_stride+g*128u+chunk*16u);
    }
    if(threadIdx.x<MT*16u && m0+threadIdx.x<a.rows)s.scale=a.scales[size_t(m0+threadIdx.x)*a.scale_stride+g];
    return s;
}
template<int MT>
__device__ __forceinline__ void stage_a(uint8_t* shared,const AStage<MT>& s) {
    #pragma unroll
    for(uint32_t j=0u;j<(MT*16u*8u+255u)/256u;++j) {
        const uint32_t t=threadIdx.x+j*256u,r=t/8u,chunk=t&7u;
        if(r<MT*16u) {
            const uint32_t sw=PLAMO_SWIZZLE?(r&7u)*16u:0u;
            const uint32_t pos=(chunk/2u)*32u+(chunk&1u)*8u;
            const uint4 raw=s.data[j];
            *reinterpret_cast<uint2*>(shared+r*128u+(pos^sw))=uint2{raw.x,raw.y};
            *reinterpret_cast<uint2*>(shared+r*128u+((pos+16u)^sw))=uint2{raw.z,raw.w};
        }
    }
    if(threadIdx.x<MT*16u)reinterpret_cast<float*>(shared+MT*16u*128u)[threadIdx.x]=s.scale;
}

template<int BITS,int MT,bool CACHE>
__global__ void __launch_bounds__(PLAMO_LAUNCH_BOUND) __attribute__((amdgpu_waves_per_eu(PLAMO_WAVES_MIN))) gemm(Args a) {
    constexpr uint32_t rows_tile=MT*16u,plane=rows_tile*132u;
    alignas(16) __shared__ uint8_t shared[2u*plane];
    const uint32_t m0=blockIdx.x*rows_tile,nt=blockIdx.y*8u+(threadIdx.x>>5u);
    if(m0>=a.rows)return;
    const uint32_t lane=threadIdx.x&31u,col=lane&15u,half=lane>>4u,groups=a.k/128u;
    float accf[MT][8]{};
    I4 cur[4];
    #pragma unroll
    for(uint32_t p=0u;p<4u;++p)cur[p]=load_b<BITS,CACHE>(a,nt,0u,p,lane);
    stage_a<MT>(shared,prefetch_a<MT>(a,m0,0u));
    __syncthreads();
    for(uint32_t g=0u;g<groups;++g) {
        AStage<MT> next{};I4 bnext[4];
        if(g+1u<groups) {
            next=prefetch_a<MT>(a,m0,g+1u);
#if PLAMO_PIPE_B
            #pragma unroll
            for(uint32_t p=0u;p<4u;++p)bnext[p]=load_b<BITS,CACHE>(a,nt,g+1u,p,lane);
#endif
        }
#if PLAMO_SCHED_BARRIER
        __builtin_amdgcn_sched_barrier(0);
#endif
        uint8_t* current=shared+(g&1u)*plane;
        I8 acc[MT]{};
        #pragma unroll
        for(uint32_t p=0u;p<4u;++p) {
            const I4 b=cur[p];
            #pragma unroll
            for(uint32_t mt=0u;mt<MT;++mt) {
                const uint32_t r=mt*16u+col,sw=PLAMO_SWIZZLE?(r&7u)*16u:0u;
                const I4 x=*reinterpret_cast<const I4*>(current+r*128u+((p*32u+half*16u)^sw));
                acc[mt]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,I2{x[0],x[1]},true,I2{b[0],b[1]},acc[mt],false);
                acc[mt]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,I2{x[2],x[3]},true,I2{b[2],b[3]},acc[mt],false);
            }
        }
        #pragma unroll
        for(uint32_t mt=0u;mt<MT;++mt)
            #pragma unroll
            for(uint32_t i=0u;i<8u;++i) {
                const uint32_t r=mt*16u+half*8u+i;
                const float scale=reinterpret_cast<const float*>(current+rows_tile*128u)[r];
                accf[mt][i]+=static_cast<float>(acc[mt][i])*scale;
            }
        if(g+1u<groups) {
            stage_a<MT>(shared+((g+1u)&1u)*plane,next);
            #pragma unroll
            for(uint32_t p=0u;p<4u;++p) {
#if PLAMO_PIPE_B
                cur[p]=bnext[p];
#else
                cur[p]=load_b<BITS,CACHE>(a,nt,g+1u,p,lane);
#endif
            }
        }
        __syncthreads();
    }
    float* epi=reinterpret_cast<float*>(shared);
    #pragma unroll
    for(uint32_t phase=0u;phase<rows_tile/8u;++phase) {
        #pragma unroll
        for(uint32_t mt=0u;mt<MT;++mt)
            if(mt*2u+half==phase)
                #pragma unroll
                for(uint32_t i=0u;i<8u;++i)
                    epi[i*128u+(threadIdx.x>>5u)*16u+col]=accf[mt][i]*shelf::quantization::exl3::kExl3Int8GridStep;
        __syncthreads();
        const uint32_t row=m0+phase*8u+(threadIdx.x>>5u);
        if(row<a.rows) {
            const uint32_t c=blockIdx.y*128u+lane*4u;
            const float4 value=*reinterpret_cast<const float4*>(epi+(threadIdx.x>>5u)*128u+lane*4u);
            float v0=0.0f+value.x,v1=0.0f+value.y,v2=0.0f+value.z,v3=0.0f+value.w;
            shelf::kernel::exl3_detail::fwht128(v0,v1,v2,v3,lane);
            v0=(v0*kInvSqrt128)*f16_bits_to_f32(a.svh[c]);v1=(v1*kInvSqrt128)*f16_bits_to_f32(a.svh[c+1u]);
            v2=(v2*kInvSqrt128)*f16_bits_to_f32(a.svh[c+2u]);v3=(v3*kInvSqrt128)*f16_bits_to_f32(a.svh[c+3u]);
            const uint2 value_out={uint32_t(f32_to_bf16_rne_device(v0))|(uint32_t(f32_to_bf16_rne_device(v1))<<16u),uint32_t(f32_to_bf16_rne_device(v2))|(uint32_t(f32_to_bf16_rne_device(v3))<<16u)};
            *reinterpret_cast<uint2*>(a.out+size_t(row)*a.out_stride+c)=value_out;
        }
        __syncthreads();
    }
}

hipError_t run(Args args,uint32_t bits,bool cached,hipStream_t stream=nullptr) {
    const dim3 grid((args.rows+PLAMO_MT*16u-1u)/(PLAMO_MT*16u),args.n/128u);
    if(cached)gemm<4,PLAMO_MT,true><<<grid,256,0,stream>>>(args);
    else if(bits==4u)gemm<4,PLAMO_MT,false><<<grid,256,0,stream>>>(args);
    else if(bits==5u)gemm<5,PLAMO_MT,false><<<grid,256,0,stream>>>(args);
    else return hipErrorInvalidValue;
    return hipGetLastError();
}
}
