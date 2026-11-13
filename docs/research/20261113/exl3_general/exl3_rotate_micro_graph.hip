#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

void ck_at(hipError_t e, int line) { if (e != hipSuccess) { std::fprintf(stderr, "%d: %s\n", line, hipGetErrorString(e)); std::exit(1); } }
#define ck(e) ck_at((e), __LINE__)
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
    hipStream_t stream;ck(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    auto run=[&](bool candidate) {
        if(candidate) ck(shelf::kernel::launch_exl3_mul24_rotate_out(y.p,svh.p,out.p,
            bf16?shelf::kernel::Exl3OutputDType::BF16:shelf::kernel::Exl3OutputDType::F32,
            rows,n,n,splits,stream));
        else { if(bf16) scalar<true><<<dim3(rows,n/128u),128,0,stream>>>(y.p,svh.p,out.p,rows,n,splits);
               else scalar<false><<<dim3(rows,n/128u),128,0,stream>>>(y.p,svh.p,out.p,rows,n,splits);ck(hipGetLastError()); }
    };
    hipEvent_t a[2],b[2];hipGraph_t graph[2];hipGraphExec_t exec[2];
    for(int c=0;c<2;++c){ck(hipEventCreate(&a[c]));ck(hipEventCreate(&b[c]));
        ck(hipStreamBeginCapture(stream,hipStreamCaptureModeGlobal));ck(hipEventRecordWithFlags(a[c],stream,hipEventRecordExternal));
        run(c!=0);ck(hipEventRecordWithFlags(b[c],stream,hipEventRecordExternal));ck(hipStreamEndCapture(stream,&graph[c]));
        ck(hipGraphInstantiate(&exec[c],graph[c],nullptr,nullptr,0));}
    for(int i=0;i<20;++i){ck(hipGraphLaunch(exec[0],stream));ck(hipGraphLaunch(exec[1],stream));}ck(hipStreamSynchronize(stream));
    std::printf("{\"rows\":%u,\"n\":%u,\"splits\":%u,\"bf16\":%s,\"samples\":[",rows,n,splits,bf16?"true":"false");
    bool first=true;
    for(int rep=0;rep<5;++rep)for(int j=0;j<2;++j){const bool candidate=(j^(rep&1))!=0;
        const int warm=candidate?1:0;
        for(int i=0;i<20;++i)ck(hipGraphLaunch(exec[warm],stream));
        ck(hipStreamSynchronize(stream));
        std::vector<float> values;
        for(int i=0;i<20;++i){const int c=candidate?1:0;ck(hipGraphLaunch(exec[c],stream));ck(hipStreamSynchronize(stream));
            float ms;ck(hipEventElapsedTime(&ms,a[c],b[c]));values.push_back(ms*1000.0f);}
        std::sort(values.begin(),values.end());
        std::printf("%s{\"candidate\":%s,\"us\":%.4f}",first?"":",",candidate?"true":"false",(values[9]+values[10])*0.5f);first=false;}
    std::puts("]}");for(int c=0;c<2;++c){ck(hipGraphExecDestroy(exec[c]));ck(hipGraphDestroy(graph[c]));ck(hipEventDestroy(a[c]));ck(hipEventDestroy(b[c]));}ck(hipStreamDestroy(stream));return 0;
}
