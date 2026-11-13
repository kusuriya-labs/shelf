#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

void ck(hipError_t e) { if (e != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(e)); std::exit(1); } }
template<class T> struct Buffer { T* p; explicit Buffer(size_t n) { ck(hipMalloc(&p,n*sizeof(T))); } ~Buffer() { ck(hipFree(p)); } };
template<bool BF16>
__global__ void scalar(const float* y, const uint16_t* svh, void* out, uint32_t rows,
                       uint32_t n, uint32_t splits) {
    using namespace shelf::kernel::exl3_detail;
    using namespace shelf::kernel::exl3_detail::mul24;
    __shared__ float smem[128];
    const uint32_t m=blockIdx.x, col=blockIdx.y*128u+threadIdx.x;
    float v=0.0f;
    for(uint32_t z=0;z<splits;++z) v+=y[(static_cast<size_t>(z)*rows+m)*n+col];
    v=fwht128_smem(v,smem,threadIdx.x)*kInvSqrt128*f16_bits_to_f32(svh[col]);
    if constexpr(BF16) reinterpret_cast<uint16_t*>(out)[size_t(m)*n+col]=f32_to_bf16_rne_device(v);
    else reinterpret_cast<float*>(out)[size_t(m)*n+col]=v;
}
int main(int argc,char** argv) {
    if(argc!=5)return 2;
    const uint32_t rows=std::atoi(argv[1]),n=std::atoi(argv[2]),splits=std::atoi(argv[3]);
    const bool bf16=std::atoi(argv[4])!=0;
    Buffer<float> y(size_t(rows)*n*splits);Buffer<uint16_t> svh(n);Buffer<float> out(size_t(rows)*n);
    ck(hipMemset(y.p,0x3f,size_t(rows)*n*splits*4));ck(hipMemset(svh.p,0x3c,size_t(n)*2));
    auto run=[&](bool candidate) {
        if(candidate) ck(shelf::kernel::launch_exl3_mul24_rotate_out(y.p,svh.p,out.p,
            bf16?shelf::kernel::Exl3OutputDType::BF16:shelf::kernel::Exl3OutputDType::F32,
            rows,n,n,splits,nullptr));
        else { if(bf16) scalar<true><<<dim3(rows,n/128u),128>>>(y.p,svh.p,out.p,rows,n,splits);
               else scalar<false><<<dim3(rows,n/128u),128>>>(y.p,svh.p,out.p,rows,n,splits);ck(hipGetLastError()); }
    };
    hipEvent_t a,b;ck(hipEventCreate(&a));ck(hipEventCreate(&b));
    for(int i=0;i<20;++i){run(false);run(true);}ck(hipDeviceSynchronize());
    std::printf("{\"rows\":%u,\"n\":%u,\"splits\":%u,\"bf16\":%s,\"samples\":[",rows,n,splits,bf16?"true":"false");
    bool first=true;
    for(int rep=0;rep<5;++rep)for(int j=0;j<2;++j){const bool candidate=(j^(rep&1))!=0;
        ck(hipEventRecord(a));for(int i=0;i<20;++i)run(candidate);ck(hipEventRecord(b));ck(hipEventSynchronize(b));
        float ms;ck(hipEventElapsedTime(&ms,a,b));std::printf("%s{\"candidate\":%s,\"us\":%.4f}",first?"":",",candidate?"true":"false",ms*50.0f);first=false;}
    std::puts("]}");ck(hipEventDestroy(a));ck(hipEventDestroy(b));return 0;
}
