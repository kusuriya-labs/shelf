#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace plamo_candidate {
hipError_t launch_plamo_exl3_projection(const shelf::kernel::PlamoExl3ProjectionGroup&,uint32_t,hipStream_t,uint32_t* = nullptr);
}
void check(hipError_t e) {if(e!=hipSuccess){std::fprintf(stderr,"%s\n",hipGetErrorString(e));std::exit(2);}}
template<class T>struct Buffer {
    T* p=nullptr;
    explicit Buffer(size_t size){check(hipMalloc(&p,size*sizeof(T)));}
    ~Buffer(){if(p)check(hipFree(p));}
};
int main(int argc,char** argv) {
    if(argc<5)return 1;
    const uint32_t m=std::atoi(argv[1]),n=std::atoi(argv[2]),k=std::atoi(argv[3]),bits=std::atoi(argv[4]);
    const uint32_t count=argc>5?std::atoi(argv[5]):1u,ss=(k/128u*4u+15u)&~15u,os=n+4u;
    const uint32_t sp=shelf::kernel::exl3_mul24_gemm_splits(m,n,k,4096u,bits);
    const size_t tw=size_t(n)*k*bits/16u,plane=size_t(m)*os;
    Buffer<uint16_t>x(size_t(m)*(k+4u)),t(tw*count),suh(size_t(k)*count),svh(size_t(n)*count),out(plane*count);
    Buffer<int8_t>a(size_t(m)*k*count);Buffer<float>s(size_t(m)*ss/4u*count),y(size_t(m)*n*16u*count);
    Buffer<uint8_t>flush(128u*1024u*1024u);
    std::mt19937 rng(17u);std::vector<uint16_t>hx(size_t(m)*(k+4u)),ht(tw*count),hu(size_t(k)*count),hv(size_t(n)*count);
    for(auto&v:hx)v=uint16_t(0x3e00u+rng()%512u);for(auto&v:ht)v=uint16_t(rng());
    for(auto&v:hu)v=rng()&1u?0x3c00u:0xbc00u;for(auto&v:hv)v=uint16_t(0x3800u+rng()%2048u);
    check(hipMemcpy(x.p,hx.data(),hx.size()*2u,hipMemcpyHostToDevice));check(hipMemcpy(t.p,ht.data(),ht.size()*2u,hipMemcpyHostToDevice));
    check(hipMemcpy(suh.p,hu.data(),hu.size()*2u,hipMemcpyHostToDevice));check(hipMemcpy(svh.p,hv.data(),hv.size()*2u,hipMemcpyHostToDevice));
    shelf::kernel::PlamoExl3ProjectionGroup args{};args.input=reinterpret_cast<const shelf::bf16_t*>(x.p);args.rows=m;args.k=k;args.input_stride=k+4u;args.scale_stride_bytes=ss;args.count=count;
    for(uint32_t i=0;i<count;++i){auto&p=args.p[i];p.trellis=t.p+tw*i;p.suh=suh.p+size_t(k)*i;p.svh=svh.p+size_t(n)*i;p.codes=a.p+size_t(m)*k*i;p.scales=s.p+size_t(m)*ss/4u*i;p.yprime=y.p+size_t(m)*n*16u*i;p.out=reinterpret_cast<shelf::bf16_t*>(out.p+plane*i);p.n=count==3u&&i>0u?512u:n;p.splits=argc>6?std::atoi(argv[6]):shelf::kernel::exl3_mul24_gemm_splits(m,p.n,k,4096u,bits);p.out_stride=os;}
    auto before=[&]{check(shelf::kernel::launch_plamo_exl3_projection(args,bits,nullptr));};auto after=[&]{check(plamo_candidate::launch_plamo_exl3_projection(args,bits,nullptr));};
    std::vector<uint16_t>ref(plane*count),candidate(ref.size());check(hipMemset(out.p,0x55,ref.size()*2u));before();check(hipMemcpy(ref.data(),out.p,ref.size()*2u,hipMemcpyDeviceToHost));check(hipMemset(out.p,0x55,ref.size()*2u));after();check(hipMemcpy(candidate.data(),out.p,candidate.size()*2u,hipMemcpyDeviceToHost));
    size_t diff=0;for(size_t i=0;i<ref.size();++i)diff+=ref[i]!=candidate[i];if(diff){std::fprintf(stderr,"bit mismatch %zu\n",diff);return 3;}
    hipEvent_t e0,e1;check(hipEventCreate(&e0));check(hipEventCreate(&e1));
    auto time=[&](auto fn){for(int i=0;i<30;++i)fn();check(hipDeviceSynchronize());std::vector<float>v;for(int i=0;i<20;++i){check(hipMemsetAsync(flush.p,i,128u*1024u*1024u));check(hipEventRecord(e0));fn();check(hipEventRecord(e1));check(hipEventSynchronize(e1));float ms;check(hipEventElapsedTime(&ms,e0,e1));v.push_back(ms*1000.0f);}std::sort(v.begin(),v.end());return(v[9]+v[10])*0.5f;};
    std::vector<float>b,c;for(int r=0;r<7;++r){if(r%2){c.push_back(time(after));b.push_back(time(before));}else{b.push_back(time(before));c.push_back(time(after));}}
    std::printf("{\"m\":%u,\"n\":%u,\"k\":%u,\"bits\":%u,\"count\":%u,\"bit_equal\":true,\"before_us\":[",m,n,k,bits,count);for(size_t i=0;i<b.size();++i)std::printf("%s%.4f",i?",":"",b[i]);std::printf("],\"candidate_us\":[");for(size_t i=0;i<c.size();++i)std::printf("%s%.4f",i?",":"",c[i]);std::puts("]}");check(hipEventDestroy(e0));check(hipEventDestroy(e1));
}
