#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
void ck(hipError_t e) { if(e != hipSuccess) { std::fprintf(stderr,"%s\n",hipGetErrorString(e)); std::exit(1); } }
template<class T> struct Buf { T* p; Buf(size_t n) { ck(hipMalloc(&p,n*sizeof(T))); } ~Buf() { ck(hipFree(p)); } };
int main(int argc,char** argv) {
 if(argc<5) return 2;
 uint32_t rows=std::atoi(argv[1]), k=std::atoi(argv[2]), n0=std::atoi(argv[3]), n1=std::atoi(argv[4]);
 uint32_t n2=argc>5?std::atoi(argv[5]):0, count=n2?3:2, ns[]={n0,n1,n2}, ss=(k/128*4+15)&~15u;
 Buf<uint16_t> input(size_t(rows)*k), suh(size_t(k)*count), svh(size_t(n0+n1+n2)), out(size_t(rows)*(n0+n1+n2));
 Buf<uint16_t> weight(size_t(n0+n1+n2)*k*4/16); Buf<int8_t> codes(size_t(rows)*k*count); Buf<float> scales(size_t(rows)*ss/4*count);
 size_t ys=0;uint32_t sp[3]={};for(uint32_t i=0;i<count;++i){sp[i]=shelf::kernel::exl3_mul24_gemm_splits(rows,ns[i],k,1024,4);ys+=size_t(rows)*ns[i]*sp[i];}
 Buf<float> y(ys);Buf<unsigned> flush(32u*1024*1024);
 ck(hipMemset(input.p,0x3f,size_t(rows)*k*2));ck(hipMemset(suh.p,0x3c,size_t(k)*count*2));ck(hipMemset(svh.p,0x3c,size_t(n0+n1+n2)*2));ck(hipMemset(weight.p,0x35,size_t(n0+n1+n2)*k/2));
 shelf::kernel::PlamoExl3ProjectionGroup g{};g.input=reinterpret_cast<shelf::bf16_t*>(input.p);g.rows=rows;g.k=k;g.input_stride=k;g.scale_stride_bytes=ss;g.count=count;
 size_t no=0,yo=0;for(uint32_t i=0;i<count;++i){auto& p=g.p[i];p.n=ns[i];p.splits=sp[i];p.out_stride=ns[i];p.trellis=weight.p+no*k/4;p.suh=suh.p+size_t(i)*k;p.svh=svh.p+no;p.codes=codes.p+size_t(i)*rows*k;p.scales=scales.p+size_t(i)*rows*ss/4;p.yprime=y.p+yo;p.out=reinterpret_cast<shelf::bf16_t*>(out.p+size_t(rows)*no);no+=ns[i];yo+=size_t(rows)*p.n*p.splits;}
 auto base=[&]{for(uint32_t i=0;i<count;++i){auto& p=g.p[i];ck(shelf::kernel::launch_exl3_mul24_rotate_quantize(g.input,k,p.suh,p.codes,p.scales,rows,k,k,ss,nullptr));ck(shelf::kernel::launch_exl3_mul24_gemm(p.codes,p.scales,k,ss,p.trellis,4,p.yprime,rows,p.n,k,p.splits,nullptr));ck(shelf::kernel::launch_exl3_mul24_rotate_out(p.yprime,p.svh,p.out,shelf::kernel::Exl3OutputDType::BF16,rows,p.n,p.n,p.splits,nullptr));}};
 auto cand=[&]{uint32_t l=0;ck(shelf::kernel::launch_plamo_exl3_projection(g,4,nullptr,&l));};
 base();std::vector<uint16_t>a(size_t(rows)*no),b(a.size());ck(hipMemcpy(a.data(),out.p,a.size()*2,hipMemcpyDeviceToHost));cand();ck(hipMemcpy(b.data(),out.p,b.size()*2,hipMemcpyDeviceToHost));size_t diff=0;for(size_t i=0;i<a.size();++i)diff+=a[i]!=b[i];
 hipEvent_t start,end;ck(hipEventCreate(&start));ck(hipEventCreate(&end));std::printf("{\"rows\":%u,\"k\":%u,\"n\":[%u,%u,%u],\"splits\":[%u,%u,%u],\"diff\":%zu,\"samples\":[",rows,k,n0,n1,n2,sp[0],sp[1],sp[2],diff);
 bool first=true;for(int rep=0;rep<5;++rep){for(int j=0;j<2;++j){bool c=(j^(rep&1));std::vector<float> times;for(int z=0;z<22;++z){ck(hipMemsetAsync(flush.p,z,128u*1024*1024));ck(hipEventRecord(start));if(c)cand();else base();ck(hipEventRecord(end));ck(hipEventSynchronize(end));float ms;ck(hipEventElapsedTime(&ms,start,end));if(z>=2)times.push_back(ms*1000); }std::sort(times.begin(),times.end());std::printf("%s{\"candidate\":%s,\"us\":%.4f}",first?"":",",c?"true":"false",(times[9]+times[10])/2);first=false;}}std::puts("]}");ck(hipEventDestroy(start));ck(hipEventDestroy(end));return diff?1:0;
}
