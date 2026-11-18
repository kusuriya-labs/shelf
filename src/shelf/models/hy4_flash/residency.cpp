#include <shelf/models/hy4_flash/residency.h>
#include <shelf/models/hy4_flash/absorbed.h>
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
Residency::Residency(Checkpoint& model,size_t tokens):model(model),tokens(tokens) {
    auto mode=std::getenv("SHELF_HY4_ABSORB");
    if(mode && std::string_view(mode)=="1")
        for(const auto& [name,dims]:model.shape) if(name.ends_with(".self_attn.kv_b_proj")) folded[name]=nullptr;
    if(!tokens || tokens>2097152) throw std::runtime_error("KV の予約長が範囲外です");
}
void Residency::reserve() {
    int count=0;
    check(hipGetDeviceCount(&count));
    if(count!=3) throw std::runtime_error("R9700、V620、V620 の順に 3 台を指定してください");
    std::array<size_t,3> available{},overhead{};
    std::array<int,3> used{},limit{};
    for(const auto& row:model.owners) for(int r:row) ++used[r];
    constexpr size_t expert_bytes=3*(1536ull*1536/2+1536*4);
    for(int r=0;r<3;++r) {
        hipDeviceProp_t props{};
        check(hipGetDeviceProperties(&props,r));
        if(!std::string_view(props.gcnArchName).starts_with(r?"gfx1030":"gfx1201"))
            throw std::runtime_error("rank と GPU の ISA が一致しません");
        check(hipSetDevice(r));
        size_t total=0;
        check(hipMemGetInfo(&available[r],&total));
        overhead[r]=r?size_t(512)<<20:size_t(2)<<30;
        if(!r) for(int i=0;i<42;++i) {
            bool sparse=i==0 || i%4==1;
            overhead[r]+=align(294*(sparse?tokens:std::min(tokens,size_t(518))));
            if(sparse) overhead[r]+=align((tokens/4+12)*256);
        }
        if(!r) for(const auto& [name,ignored]:folded) overhead[0]+=size_t(model.shape.at(name)[1])*512*2;
        size_t fixed=align(model.bytes[r])-used[r]*expert_bytes+overhead[r];
        if(available[r]<fixed) throw std::bad_alloc();
        limit[r]=(available[r]-fixed)/expert_bytes;
    }
    if(limit[0]+limit[1]+limit[2]<41*512) throw std::bad_alloc();
    auto plan=model.owners;
    int moved=0;
    for(int expert=511;expert>=0;--expert) for(auto& row:plan) {
        int from=row[expert];
        if(used[from]<=limit[from]) continue;
        int dest=0;
        for(int r=1;r<3;++r)
            if(limit[r]-used[r]>limit[dest]-used[dest]) dest=r;
        if(dest==from || used[dest]>=limit[dest]) throw std::bad_alloc();
        row[expert]=dest;
        --used[from]; ++used[dest]; ++moved;
    }
    if(moved) model.relocate(plan);
    fprintf(stderr,"relocated_experts=%d rank_counts=%d,%d,%d\n",moved,used[0],used[1],used[2]);
    for(int r=0;r<3;++r) {
        check(hipSetDevice(r));
        size_t free,total;
        check(hipMemGetInfo(&free,&total));
        size_t work=r?size_t(512)<<20:size_t(2)<<30;
        size_t bytes=align(model.bytes[r])+overhead[r];
        fprintf(stderr,"rank=%d weight=%zu workspace=%zu request=%zu free=%zu total=%zu\n",
                r,model.bytes[r],work,bytes,free,total);
        auto result=gpu::GpuArena::create(r,bytes);
        if(!result.ok()) {
            if(result.status().code()==Status::Code::insufficient_memory) throw std::bad_alloc();
            throw std::runtime_error(result.status().message());
        }
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
        if(!r) for(auto& [name,ptr]:folded) ptr=buffer.next(size_t(model.shape.at(name)[1])*512*2);
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
    if(!folded.empty()) fold_kv_weights(*this);
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
