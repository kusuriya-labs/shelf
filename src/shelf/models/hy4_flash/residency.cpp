#include <shelf/models/hy4_flash/residency.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string_view>

namespace shelf::hy4 {
namespace {
size_t align(size_t n) {return (n+255)&~size_t(255);}
void check(hipError_t e) {
    if(e!=hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
}
void* Region::next(size_t length) {
    size_t start=align(cursor);
    if(start>size || length>size-start) throw std::runtime_error("作業域の容量が不足しています");
    cursor=start+length;
    return base+start;
}
Residency::Residency(const Checkpoint& model,size_t tokens):model(model),tokens(tokens) {
    if(!tokens || tokens>2097152) throw std::runtime_error("KV の予約長が範囲外です");
}
void Residency::reserve() {
    int count=0;
    check(hipGetDeviceCount(&count));
    if(count!=3) throw std::runtime_error("R9700、V620、V620 の順に 3 台を指定してください");
    for(int r=0;r<3;++r) {
        hipDeviceProp_t properties{};
        check(hipGetDeviceProperties(&properties,r));
        if(!std::string_view(properties.gcnArchName).starts_with(r?"gfx1030":"gfx1201"))
            throw std::runtime_error("rank と GPU の ISA が一致しません");
        check(hipSetDevice(r));
        size_t free=0,total=0;
        check(hipMemGetInfo(&free,&total));
        size_t bytes=align(model.bytes[r]);
        size_t work=r?size_t(512)<<20:size_t(2)<<30;
        if(!r) {
            for(int i=0;i<42;++i) {
                bool sparse=i==0 || i%4==1;
                bytes+=align(294*(sparse?tokens:std::min(tokens,size_t(518))));
                if(sparse) bytes+=align((tokens/4+12)*256);
            }
        }
        bytes+=work;
        fprintf(stderr,"rank=%d weight=%zu workspace=%zu request=%zu free=%zu total=%zu\n",
                r,model.bytes[r],work,bytes,free,total);
        auto result=gpu::GpuArena::create(r,bytes);
        if(!result.ok()) throw std::runtime_error(result.status().message());
        arenas.push_back(result.release());
        auto allocation=arenas.back().allocate_aligned(bytes,256);
        if(!allocation.ok()) throw std::runtime_error(allocation.status().message());
        Region buffer{static_cast<uint8_t*>(allocation.value().data()),bytes};
        weight_base[r]=static_cast<uint8_t*>(buffer.next(model.bytes[r]));
        if(!r) for(int i=0;i<42;++i) {
            bool sparse=i==0 || i%4==1;
            cache[i]=static_cast<uint8_t*>(buffer.next(294*(sparse?tokens:std::min(tokens,size_t(518)))));
            if(sparse) index_keys[i]=static_cast<uint16_t*>(buffer.next((tokens/4+12)*256));
        }
        workspace[r]={static_cast<uint8_t*>(buffer.next(work)),work};
    }
    check(hipSetDevice(0));
}
void Residency::upload() {
    for(int r=0;r<3;++r) {
        check(hipSetDevice(r));
        for(const auto& [name,w]:model.weights)
            if(w.rank==r) check(hipMemcpy(weight_base[r]+w.offset,w.host,w.bytes,hipMemcpyHostToDevice));
    }
    check(hipSetDevice(0));
}
void Residency::shutdown() {
    std::string errors;
    for(auto it=arenas.rbegin();it!=arenas.rend();++it) {
        auto status=it->shutdown();
        if(!status.ok()) errors+=status.message()+"\n";
    }
    arenas.clear();
    if(!errors.empty()) throw std::runtime_error(errors);
}
const void* Residency::device(const Weight& w) const {
    if(w.rank<0) throw std::runtime_error("ホスト常駐 tensor です");
    return weight_base.at(w.rank)+w.offset;
}
}
