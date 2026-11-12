#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

void check(hipError_t e) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(e)); std::exit(1); }
}
template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { check(hipMalloc(&p, n * sizeof(T))); }
    ~Buffer() { if (p) check(hipFree(p)); }
};

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    const uint32_t m = std::atoi(argv[1]), n = std::atoi(argv[2]), k = std::atoi(argv[3]);
    const uint32_t bits = std::atoi(argv[4]);
    const int sweep = argc > 5 ? std::atoi(argv[5]) : 0;
    const bool mean20 = argc > 6 && std::atoi(argv[6]) != 0;
    const uint32_t ss = (k / 128u * 4u + 15u) & ~15u;
    const uint32_t base_sp = m > 16 && m <= 64 && n == 2560 && k == 10240 ? 4u :
        shelf::kernel::exl3_mul24_gemm_splits(m, n, k, 1024, bits);
    std::mt19937 rng(17);
    Buffer<uint16_t> x(size_t(m) * k), suh(k), svh(n), t(size_t(n) * k * bits / 16u);
    Buffer<int8_t> a(size_t(m) * k), w(size_t(n) * k);
    Buffer<float> s(size_t(m) * ss / 4), y(size_t(m) * n * 16), ref(size_t(m) * n * 16);
    Buffer<uint16_t> out(size_t(m) * n);
    Buffer<uint32_t> flush(32u * 1024 * 1024);
    std::vector<uint16_t> hx(size_t(m) * k), ht(size_t(n) * k * bits / 16u);
    for (auto& v : hx) v = static_cast<uint16_t>(0x3f00u + rng() % 128u);
    for (auto& v : ht) v = rng();
    std::vector<uint16_t> hs(std::max(n, k), 0x3c00u);
    check(hipMemcpy(x.p,hx.data(),hx.size()*2,hipMemcpyHostToDevice));
    check(hipMemcpy(t.p,ht.data(),ht.size()*2,hipMemcpyHostToDevice));
    check(hipMemcpy(suh.p,hs.data(),k*2,hipMemcpyHostToDevice));
    check(hipMemcpy(svh.p,hs.data(),n*2,hipMemcpyHostToDevice));
    auto rotate = [&] { check(shelf::kernel::launch_exl3_mul24_rotate_quantize(
        reinterpret_cast<shelf::bf16_t*>(x.p),k,suh.p,a.p,s.p,m,k,k,ss,nullptr)); };
    auto gemm = [&](int v, int mt, uint32_t sp) {
        if (v == 0) check(shelf::kernel::launch_exl3_mul24_gemm(a.p,s.p,k,ss,t.p,bits,y.p,m,n,k,sp,nullptr));
        else if (v == 5) check(shelf::kernel::launch_exl3_mul24_prefill_wmma(a.p,s.p,k,ss,w.p,y.p,m,n,k,nullptr));
        else if (v == 6) check(plamo_research::launch_dot(a.p,s.p,k,ss,t.p,y.p,m,n,k,bits,sp));
        else check(plamo_research::variant_gemm(v,mt,a.p,s.p,k,ss,t.p,w.p,y.p,m,n,k,bits,sp,nullptr));
    };
    auto epilogue = [&](uint32_t sp) { check(shelf::kernel::launch_exl3_mul24_rotate_out(
        y.p,svh.p,out.p,shelf::kernel::Exl3OutputDType::BF16,m,n,n,sp,nullptr)); };
    check(shelf::kernel::launch_exl3_mul24_recon_wmma(t.p,bits,w.p,n,k,nullptr));
    rotate();
    hipEvent_t ev[2]; for(auto& e : ev) check(hipEventCreate(&e));
    auto time = [&](auto fn, bool cold, int runs = 5) {
        for(int i=0;i<20;++i) fn();
        check(hipDeviceSynchronize());
        std::vector<float> samples;
        for(int run=0;run<runs;++run) {
            std::vector<float> values;
            for(int i=0;i<20;++i) {
                if (cold) check(hipMemsetAsync(flush.p,i,128u*1024*1024));
                check(hipEventRecord(ev[0])); fn(); check(hipEventRecord(ev[1]));
                check(hipEventSynchronize(ev[1]));
                float ms; check(hipEventElapsedTime(&ms,ev[0],ev[1])); values.push_back(ms * 1000);
            }
            std::sort(values.begin(),values.end());
            float sum=0;for(float value:values)sum+=value;
            samples.push_back(mean20?sum/20:(values[9]+values[10])/2);
        }
        return samples;
    };
    auto emit = [&](const char* stage,int v,int mt,uint32_t sp,bool cold,const std::vector<float>& samples) {
        std::printf("{\"m\":%u,\"n\":%u,\"k\":%u,\"bits\":%u,\"stage\":\"%s\",\"variant\":%d,\"mt\":%d,\"splits\":%u,\"cold\":%s,\"us\":[",m,n,k,bits,stage,v,mt,sp,cold?"true":"false");
        for(size_t i=0;i<samples.size();++i) std::printf("%s%.4f",i?",":"",samples[i]);
        std::puts("]}"); std::fflush(stdout);
    };
    auto print = [&](const char* stage,int v,int mt,uint32_t sp,bool cold,auto fn) {
        emit(stage,v,mt,sp,cold,time(fn,cold));
    };
    if(sweep == 4) {
        shelf::kernel::PlamoExl3ProjectionGroup args{};
        args.input=reinterpret_cast<const shelf::bf16_t*>(x.p);args.rows=m;args.k=k;args.input_stride=k;args.scale_stride_bytes=ss;args.count=1;
        auto& p=args.p[0];p.trellis=t.p;p.suh=suh.p;p.svh=svh.p;p.codes=a.p;p.scales=s.p;p.yprime=y.p;
        p.out=reinterpret_cast<shelf::bf16_t*>(out.p);p.n=n;p.splits=base_sp;p.out_stride=n;
        rotate();gemm(0,0,base_sp);epilogue(base_sp);
        std::vector<uint16_t> before(size_t(m)*n),after(before.size());
        check(hipMemcpy(before.data(),out.p,before.size()*2,hipMemcpyDeviceToHost));
        check(plamo_selected::launch_plamo_exl3_projection(args,bits,nullptr));
        check(hipMemcpy(after.data(),out.p,after.size()*2,hipMemcpyDeviceToHost));
        size_t diff=0;for(size_t i=0;i<before.size();++i)if(before[i]!=after[i])++diff;
        std::fprintf(stderr,"selected m=%u n=%u bits=%u diff=%zu\n",m,n,bits,diff);
        const char* names[]={"rotate","gemm","out","linear"};
        for(int stage=0;stage<4;++stage) {
            const bool cold=stage==1 || stage==3;
            std::vector<float> before_us,after_us;
            auto before=[&]{if(stage==0)rotate();else if(stage==1)gemm(0,0,base_sp);else if(stage==2)epilogue(base_sp);else {rotate();gemm(0,0,base_sp);epilogue(base_sp);}};
            auto after=[&]{if(stage==3)check(plamo_selected::launch_plamo_exl3_projection(args,bits,nullptr));else check(plamo_selected::selected_stage(stage,args,bits));};
            for(int run=0;run<5;++run) {
                if(run%2==0){before_us.push_back(time(before,cold,1)[0]);after_us.push_back(time(after,cold,1)[0]);}
                else {after_us.push_back(time(after,cold,1)[0]);before_us.push_back(time(before,cold,1)[0]);}
            }
            emit(names[stage],0,0,base_sp,cold,before_us);emit(names[stage],9,0,base_sp,cold,after_us);
        }
    } else if(sweep == 3) {
        const uint32_t count=n==10240?2:3;
        Buffer<uint16_t> gt(ht.size()*count), gu(size_t(k)*count), gv(size_t(n)*count);
        Buffer<int8_t> ga(size_t(m)*k*count);
        Buffer<float> gs(size_t(m)*ss/4*count), gy(size_t(m)*n*16*count);
        Buffer<uint16_t> go(size_t(m)*n*count);
        plamo_research::ProjectionGroup args{};
        args.input=reinterpret_cast<const shelf::bf16_t*>(x.p);args.rows=m;args.k=k;args.ss=ss;args.count=count;
        for(uint32_t i=0;i<count;++i) {
            for(auto& v:ht)v=rng();
            check(hipMemcpy(gt.p+i*ht.size(),ht.data(),ht.size()*2,hipMemcpyHostToDevice));
            for(uint32_t j=0;j<k;++j)hs[j]=(rng()&1)?0x3c00:0xbc00;
            check(hipMemcpy(gu.p+size_t(i)*k,hs.data(),k*2,hipMemcpyHostToDevice));
            std::fill(hs.begin(),hs.end(),0x3c00);
            check(hipMemcpy(gv.p+size_t(i)*n,hs.data(),n*2,hipMemcpyHostToDevice));
            auto& p=args.p[i];
            p.n=i==0||count==2?n:512;
            p.splits=shelf::kernel::exl3_mul24_gemm_splits(m,p.n,k,1024,bits);
            p.trellis=gt.p+i*ht.size();p.suh=gu.p+size_t(i)*k;p.svh=gv.p+size_t(i)*n;
            p.codes=ga.p+size_t(i)*m*k;p.scales=gs.p+size_t(i)*m*ss/4;
            p.yprime=gy.p+size_t(i)*m*n*16;p.out=reinterpret_cast<shelf::bf16_t*>(go.p+size_t(i)*m*n);
        }
        auto sequence=[&](bool tiled) {
            for(uint32_t i=0;i<count;++i) {
                auto p=args.p[i];
                check(shelf::kernel::launch_exl3_mul24_rotate_quantize(args.input,k,p.suh,p.codes,p.scales,m,k,k,ss,nullptr));
                check(shelf::kernel::launch_exl3_mul24_gemm(p.codes,p.scales,k,ss,p.trellis,bits,p.yprime,m,p.n,k,p.splits,nullptr));
                if(tiled)check(plamo_research::tiled_out(p.yprime,p.svh,p.out,m,p.n,p.splits));
                else check(shelf::kernel::launch_exl3_mul24_rotate_out(p.yprime,p.svh,p.out,shelf::kernel::Exl3OutputDType::BF16,m,p.n,p.n,p.splits,nullptr));
            }
        };
        check(hipMemset(go.p,0,size_t(m)*n*count*2));sequence(false);
        std::vector<uint16_t> before(size_t(m)*n*count),after(before.size());
        check(hipMemcpy(before.data(),go.p,before.size()*2,hipMemcpyDeviceToHost));
        check(plamo_research::launch_group(args,bits));
        check(hipMemcpy(after.data(),go.p,after.size()*2,hipMemcpyDeviceToHost));
        size_t diff=0;for(size_t i=0;i<before.size();++i)if(before[i]!=after[i])++diff;
        std::fprintf(stderr,"group m=%u n=%u bits=%u diff=%zu\n",m,n,bits,diff);
        shelf::kernel::PlamoExl3ProjectionGroup selected{};
        selected.input=args.input;selected.rows=m;selected.k=k;selected.input_stride=k;selected.scale_stride_bytes=ss;selected.count=count;
        for(uint32_t i=0;i<count;++i) {
            const auto p=args.p[i];auto& q=selected.p[i];
            q.trellis=p.trellis;q.suh=p.suh;q.svh=p.svh;q.codes=p.codes;q.scales=p.scales;
            q.yprime=p.yprime;q.out=p.out;q.n=p.n;q.splits=p.splits;q.out_stride=p.n;
        }
        check(plamo_selected::launch_plamo_exl3_projection(selected,bits,nullptr));
        check(hipMemcpy(after.data(),go.p,after.size()*2,hipMemcpyDeviceToHost));
        diff=0;for(size_t i=0;i<before.size();++i)if(before[i]!=after[i])++diff;
        std::fprintf(stderr,"selected_group m=%u n=%u bits=%u diff=%zu\n",m,n,bits,diff);
        std::vector<float> samples[4];
        for(int run=0;run<5;++run)for(int index=0;index<4;++index) {
            const int variant=run%2==0?index:3-index;
            samples[variant].push_back(time([&]{if(variant<2)sequence(variant==1);else if(variant==2)check(plamo_research::launch_group(args,bits));else check(plamo_selected::launch_plamo_exl3_projection(selected,bits,nullptr));},true,1)[0]);
        }
        emit("group",0,0,base_sp,true,samples[0]);emit("group",7,0,base_sp,true,samples[1]);emit("group",8,0,base_sp,true,samples[2]);emit("group",9,0,base_sp,true,samples[3]);
    } else if (!sweep) {
        print("rotate",0,0,base_sp,false,rotate);
        print("gemm",0,0,base_sp,true,[&]{gemm(0,0,base_sp);});
        print("out",0,0,base_sp,false,[&]{epilogue(base_sp);});
        print("linear",0,0,base_sp,true,[&]{rotate();gemm(0,0,base_sp);epilogue(base_sp);});
    } else if(sweep == 2) {
        for(uint32_t sp : {base_sp,1u,2u,4u,8u}) {
            gemm(0,0,sp);check(hipMemcpy(ref.p,y.p,size_t(m)*n*sp*4,hipMemcpyDeviceToDevice));
            gemm(6,0,sp);check(hipDeviceSynchronize());
            std::vector<float> before(size_t(m)*n*sp),after(before.size());
            check(hipMemcpy(before.data(),ref.p,before.size()*4,hipMemcpyDeviceToHost));
            check(hipMemcpy(after.data(),y.p,after.size()*4,hipMemcpyDeviceToHost));
            size_t diff=0;float maxdiff=0;
            for(size_t i=0;i<before.size();++i){if(std::memcmp(&before[i],&after[i],4))++diff;maxdiff=std::max(maxdiff,std::fabs(before[i]-after[i]));}
            std::fprintf(stderr,"dot m=%u n=%u sp=%u diff=%zu max=%.8g\n",m,n,sp,diff,maxdiff);
            print("gemm",6,0,sp,true,[&]{gemm(6,0,sp);});
        }
        gemm(0,0,base_sp);epilogue(base_sp);
        std::vector<uint16_t> before(size_t(m)*n),after(before.size());
        check(hipMemcpy(before.data(),out.p,before.size()*2,hipMemcpyDeviceToHost));
        check(plamo_research::tiled_out(y.p,svh.p,out.p,m,n,base_sp));
        check(hipMemcpy(after.data(),out.p,after.size()*2,hipMemcpyDeviceToHost));
        size_t diff=0;for(size_t i=0;i<before.size();++i)if(before[i]!=after[i])++diff;
        std::fprintf(stderr,"tiled_out m=%u n=%u diff=%zu\n",m,n,diff);
        print("out",7,0,base_sp,false,[&]{check(plamo_research::tiled_out(y.p,svh.p,out.p,m,n,base_sp));});
    } else {
        const int base_mt = (m + 15u)/16u;
        for (int v : {0,1,2,3,4,5}) {
            for (int mt : (v==1 || v==2 ? std::vector<int>{1,2,4} : std::vector<int>{base_mt})) {
                if (v==2 && mt==2 && m==1) continue;
                for (uint32_t sp : (v==5 ? std::vector<uint32_t>{1} : std::vector<uint32_t>{1,2,4,5,8,10})) {
                    if (k/128u<sp) continue;
                    gemm(0,0,sp); check(hipMemcpy(ref.p,y.p,size_t(m)*n*sp*4,hipMemcpyDeviceToDevice));
                    gemm(v,mt,sp); check(hipDeviceSynchronize());
                    std::vector<float> before(size_t(m)*n*sp),after(before.size());
                    check(hipMemcpy(before.data(),ref.p,before.size()*4,hipMemcpyDeviceToHost));
                    check(hipMemcpy(after.data(),y.p,after.size()*4,hipMemcpyDeviceToHost));
                    size_t diff=0; double maxdiff=0;
                    if(v!=5) for(size_t i=0;i<before.size();++i) {
                        if(std::memcmp(&before[i],&after[i],4)) ++diff;
                        maxdiff=std::max(maxdiff,double(std::fabs(before[i]-after[i])));
                    }
                    std::fprintf(stderr,"check m=%u n=%u k=%u b=%u v=%d mt=%d sp=%u diff=%zu max=%.8g\n",m,n,k,bits,v,mt,sp,diff,maxdiff);
                    print("gemm",v,mt,sp,true,[&]{gemm(v,mt,sp);});
                }
            }
        }
    }
    for(auto e:ev)check(hipEventDestroy(e));
}
