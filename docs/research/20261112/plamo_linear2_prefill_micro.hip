#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>
#include <shelf/inference/execution/exl3_dense_gemm.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

void check(hipError_t e){if(e!=hipSuccess){std::fprintf(stderr,"%s\n",hipGetErrorString(e));std::exit(2);}}
template<class T>struct Buffer{T*p=nullptr;explicit Buffer(size_t n){check(hipMalloc(&p,n*sizeof(T)));}~Buffer(){if(p)check(hipFree(p));}};
int main(int argc,char** argv){
    if(argc<5)return 1;
    const uint32_t m=std::atoi(argv[1]),n=std::atoi(argv[2]),k=std::atoi(argv[3]),bits=std::atoi(argv[4]);
    const uint32_t flag_mode=argc>5?std::atoi(argv[5]):1u,ss=(k/128u*4u+15u)&~15u,cap=shelf::kernel::exl3_mul24_flag_cap(m),os=n+4u;
    const size_t tw=size_t(n)*k*bits/16u,plane=size_t(m)*os;
    Buffer<uint16_t>x(size_t(m)*k),t(tw),suh(k),svh(n),out(plane);
    Buffer<int8_t>codes(size_t(m)*k),codesf(size_t(cap)*k),dense(size_t(n)*k),packed(size_t(n)*k);
    Buffer<float>scales(size_t(m)*ss/4u),scalesf(size_t(cap)*ss/4u),rowscale(m),yp(size_t(m)*n),yf(size_t(cap)*n);
    Buffer<int32_t>slots(m);Buffer<uint32_t>count(1u);Buffer<uint8_t>flush(128u*1024u*1024u);
    std::mt19937 rng(17u);std::vector<uint16_t>hx(size_t(m)*k),ht(tw),hu(k),hv(n);
    for(uint32_t r=0u;r<m;++r)for(uint32_t c=0u;c<k;++c){uint16_t v=uint16_t(0x3e00u+rng()%384u);if(flag_mode && (flag_mode==2u || r%2u==0u) && c<128u)v+=0x0200u;hx[size_t(r)*k+c]=v;}
    for(auto&v:ht)v=uint16_t(rng());for(auto&v:hu)v=rng()&1u?0x3c00u:0xbc00u;for(auto&v:hv)v=uint16_t(0x3800u+rng()%2048u);
    check(hipMemcpy(x.p,hx.data(),hx.size()*2u,hipMemcpyHostToDevice));check(hipMemcpy(t.p,ht.data(),ht.size()*2u,hipMemcpyHostToDevice));check(hipMemcpy(suh.p,hu.data(),hu.size()*2u,hipMemcpyHostToDevice));check(hipMemcpy(svh.p,hv.data(),hv.size()*2u,hipMemcpyHostToDevice));
    auto rotate=[&]{check(shelf::kernel::launch_exl3_mul24_rotate_quantize(reinterpret_cast<const shelf::bf16_t*>(x.p),k,suh.p,codes.p,scales.p,m,k,k,ss,nullptr));};
    auto exact=[&]{rotate();check(shelf::kernel::launch_exl3_mul24_gemm(codes.p,scales.p,k,ss,t.p,bits,yp.p,m,n,k,1u,nullptr));check(shelf::kernel::launch_exl3_mul24_rotate_out(yp.p,svh.p,out.p,shelf::kernel::Exl3OutputDType::BF16,m,n,os,1u,nullptr));};
    auto baseline=[&]{rotate();check(shelf::kernel::launch_exl3_mul24_rowscale(codes.p,scales.p,rowscale.p,slots.p,count.p,codesf.p,scalesf.p,m,cap,k,k,ss,0.5f,nullptr));check(shelf::kernel::launch_exl3_mul24_recon(t.p,bits,dense.p,n,k,nullptr));check(shelf::execution::exl3_dense_gemm_int8(m,n,k,dense.p,codes.p,yp.p,nullptr));check(shelf::kernel::launch_plamo_exl3_flagged_gemm(codesf.p,scalesf.p,k,ss,t.p,bits,yf.p,cap,count.p,n,k,nullptr));check(shelf::kernel::launch_plamo_exl3_rotate_out_i32(reinterpret_cast<const int32_t*>(yp.p),rowscale.p,slots.p,yf.p,svh.p,out.p,true,m,n,os,nullptr));};
    plamo_prefill_study::Args args{codes.p,scales.p,t.p,packed.p,svh.p,reinterpret_cast<shelf::bf16_t*>(out.p),m,n,k,k,ss/4u,os};
    auto direct=[&]{rotate();check(plamo_prefill_study::run(args,bits,false));};
    auto cached=[&]{rotate();check(plamo_prefill_study::pack(t.p,packed.p,n,k,bits));check(plamo_prefill_study::run(args,bits,true));};
    std::vector<uint16_t>gold(plane),after(plane);check(hipMemset(out.p,0x55,plane*2u));exact();check(hipMemcpy(gold.data(),out.p,plane*2u,hipMemcpyDeviceToHost));
    for(int mode:{2,3}){check(hipMemset(out.p,0x55,plane*2u));if(mode==2)direct();else cached();check(hipMemcpy(after.data(),out.p,plane*2u,hipMemcpyDeviceToHost));size_t diff=0u;for(size_t i=0;i<plane;++i)if(gold[i]!=after[i])++diff;std::fprintf(stderr,"mode=%d m=%u n=%u k=%u bits=%u diff=%zu\n",mode,m,n,k,bits,diff);if(diff)return 3;}
    check(hipMemset(out.p,0x55,plane*2u));baseline();check(hipMemcpy(after.data(),out.p,plane*2u,hipMemcpyDeviceToHost));size_t dense_diff=0;for(size_t i=0;i<plane;++i)dense_diff+=gold[i]!=after[i];uint32_t active=0;check(hipMemcpy(&active,count.p,4u,hipMemcpyDeviceToHost));
    hipEvent_t e0,e1;check(hipEventCreate(&e0));check(hipEventCreate(&e1));
    auto time=[&](auto fn,bool cold){for(int i=0;i<20;++i)fn();check(hipDeviceSynchronize());std::vector<float>v;for(int i=0;i<20;++i){if(cold)check(hipMemsetAsync(flush.p,i,128u*1024u*1024u));check(hipEventRecord(e0));fn();check(hipEventRecord(e1));check(hipEventSynchronize(e1));float ms;check(hipEventElapsedTime(&ms,e0,e1));v.push_back(ms*1000.0f);}std::sort(v.begin(),v.end());return(v[9]+v[10])*0.5f;};
    auto emit=[&](const char*stage,int variant,const std::vector<float>&v){std::printf("{\"m\":%u,\"n\":%u,\"k\":%u,\"bits\":%u,\"mt\":%d,\"stage\":\"%s\",\"variant\":%d,\"flag_mode\":%u,\"flag_count\":%u,\"dense_vs_exact_diff\":%zu,\"us\":[",m,n,k,bits,PLAMO_MT,stage,variant,flag_mode,active,dense_diff);for(size_t i=0;i<v.size();++i)std::printf("%s%.4f",i?",":"",v[i]);std::puts("]}");std::fflush(stdout);};
    for(int mode:{1,2,3}){std::vector<float>b,c;for(int r=0;r<5;++r){auto cand=[&]{if(mode==1)exact();else if(mode==2)direct();else cached();};if(r%2){c.push_back(time(cand,true));b.push_back(time(baseline,true));}else{b.push_back(time(baseline,true));c.push_back(time(cand,true));}}emit("linear",0,b);emit("linear",mode,c);}
    rotate();check(plamo_prefill_study::pack(t.p,packed.p,n,k,bits));std::vector<float>g,p;for(int r=0;r<5;++r){g.push_back(time([&]{check(plamo_prefill_study::run(args,bits,true));},false));p.push_back(time([&]{check(plamo_prefill_study::pack(t.p,packed.p,n,k,bits));},true));}emit("gemm_hot",3,g);emit("pack",3,p);
    std::vector<float>resident;for(int r=0;r<5;++r)resident.push_back(time([&]{rotate();check(plamo_prefill_study::run(args,bits,true));},true));emit("resident_linear",3,resident);
    check(hipEventDestroy(e0));check(hipEventDestroy(e1));
}
