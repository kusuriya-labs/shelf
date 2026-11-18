#include <shelf/models/hy4_flash/decoder.h>
#include <shelf/models/hy4_flash/generate.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace shelf::hy4 {
namespace {
using namespace shelf::kernel::exl3_detail::mul24;
constexpr std::array<int,5> sources{25,29,33,37,41};
__global__ void best_ids(Mat logits,int* tokens) {
    __shared__ float vals[256];
    __shared__ int ids[256];
    int tid=threadIdx.x,row=blockIdx.x,id=0;
    float best=-INFINITY;
    for(int c=tid;c<logits.cols;c+=256) {
        float v=bf16_bits_to_f32(logits.p[size_t(row)*logits.cols+c]);
        if(v>best) {best=v;id=c;}
    }
    vals[tid]=best;
    ids[tid]=id;
    __syncthreads();
    for(int d=128;d;d/=2) {
        if(tid<d && (vals[tid+d]>vals[tid] || (vals[tid+d]==vals[tid] && ids[tid+d]<ids[tid]))) {
            vals[tid]=vals[tid+d];
            ids[tid]=ids[tid+d];
        }
        __syncthreads();
    }
    if(!tid) tokens[row]=ids[0];
}
__global__ void fuse_anchor(Mat anchor,Mat fusion,const uint16_t* gate) {
    int i=blockIdx.x*256+threadIdx.x;
    anchor.p[i]=f32_to_bf16_rne_device(bf16_bits_to_f32(anchor.p[i])+
        bf16_bits_to_f32(fusion.p[i])*bf16_bits_to_f32(gate[i]));
}
struct Target {
    Ops op;
    std::vector<int> history;
    std::array<Selection,5> tapped;
    Mat state{};
    size_t base=0;
    explicit Target(Residency& gpu):op(gpu) {
        state=op.mat(6,3072);
        base=op.arena.cursor;
    }
    std::vector<int> predict(Mat hidden) {
        Mat logits=op.linear("lm_head",hidden);
        int* device_ids=static_cast<int*>(op.arena.next(hidden.rows*4));
        best_ids<<<hidden.rows,256>>>(logits,device_ids);
        checked(hipGetLastError());
        std::vector<int> result(hidden.rows);
        op.copy(result.data(),device_ids,result.size()*4,hipMemcpyDeviceToHost);
        return result;
    }
    std::vector<int> forward(const std::vector<int>& ids,bool verify) {
        op.arena.cursor=base;
        int start=history.size();
        if(ids.empty() || start+ids.size()>op.gpu.tokens || (verify && ids.size()>6))
            throw std::runtime_error("入力列が KV の範囲外です");
        Mat x=op.embed(ids,true);
        size_t save=op.arena.cursor;
        for(int layer=0;layer<42;++layer) {
            auto p="model.layers."+std::to_string(layer)+".";
            if(layer%10==1 && layer<=31) engram(op,layer,x,history,ids);
            float* post;
            Mat a=op.collapse(x,p+"hc_attn_layer.hc_pre.hc_",post);
            a=op.norm(a,p+"input_layernorm.weight");
            Selection selection;
            a=attention(op,p,layer,start,a,selection,false);
            Mat z=op.expand(x,a,post);
            checked(hipMemcpyAsync(x.p,z.p,z.bytes(),hipMemcpyDeviceToDevice));
            auto it=std::find(sources.begin(),sources.end(),layer);
            if(it!=sources.end()) {
                if(!verify) selection=Selection{std::move(selection.back())};
                tapped[it-sources.begin()]=std::move(selection);
            }
            op.arena.cursor=save;
            Mat f=op.collapse(x,p+"hc_mlp_layer.hc_pre.hc_",post);
            f=op.norm(f,p+"post_attention_layernorm.weight");
            if(layer==0) f=op.norm(feedforward(op,p+"mlp",f),p+"mlp.post_fc2_norm.weight");
            else f=moe(op,layer,f);
            z=op.expand(x,f,post);
            checked(hipMemcpyAsync(x.p,z.p,z.bytes(),hipMemcpyDeviceToDevice));
            op.arena.cursor=save;
        }
        if(!verify) x=x.row(x.rows-1);
        float* unused;
        Mat h=op.norm(op.collapse(x,"model.hc_head.hc_head_",unused),"model.norm.weight");
        checked(hipMemcpyAsync(state.p,h.p,h.bytes(),hipMemcpyDeviceToDevice));
        history.insert(history.end(),ids.begin(),ids.end());
        return predict(h);
    }
    std::vector<int> propose(int anchor) {
        op.arena.cursor=base;
        std::vector<int> block(6,120009);
        block[0]=anchor;
        Mat x=op.embed(block,false);
        std::string p="model.dfly_layers.";
        auto e=op.norm(x.row(0),p+"mtp_fusion.enorm.weight");
        auto h=op.norm(state.row(0),p+"mtp_fusion.hnorm.weight");
        auto fused=op.linear(p+"mtp_fusion.eh_proj",op.join(e,h));
        fuse_anchor<<<12,256>>>(x.row(0),fused,static_cast<const uint16_t*>(op.param(p+"mtp_fusion.gate").p));
        checked(hipGetLastError());
        int start=history.size();
        size_t save=op.arena.cursor;
        for(int i=0;i<5;++i) {
            std::string prefix=p+std::to_string(i)+".";
            Selection select(6,tapped[i][0]);
            for(int row=0;row<6;++row) for(int j=0;j<=row;++j) select[row].push_back(start+j);
            auto a=op.norm(x,prefix+"input_layernorm.weight");
            auto y=op.add(x,attention(op,prefix,sources[i],start,a,select,true));
            auto f=feedforward(op,prefix+"mlp",op.norm(y,prefix+"post_attention_layernorm.weight"));
            auto next=op.add(y,f);
            checked(hipMemcpyAsync(x.p,next.p,next.bytes(),hipMemcpyDeviceToDevice));
            op.arena.cursor=save;
        }
        x=op.norm(x,p+"final_norm.weight");
        std::string correction=p+"hidden_correction.";
        int previous=anchor;
        std::vector<int> draft;
        for(int i=1;i<=5;++i) {
            auto q=op.norm(x.row(i),correction+"hidden_norm.weight");
            auto k=op.norm(op.embed({previous},false),correction+"embed_norm.weight");
            auto corrected=op.add(x.row(i),feedforward(op,p+"hidden_correction",op.join(q,k)));
            previous=predict(corrected)[0];
            draft.push_back(previous);
        }
        return draft;
    }
    void accept(int keep,int submitted) {
        history.resize(history.size()-submitted+keep);
        if(keep>1) checked(hipMemcpyAsync(state.p,state.row(keep-1).p,3072*2,hipMemcpyDeviceToDevice));
        for(auto& s:tapped) s=Selection{std::move(s[keep-1])};
    }
};
}
void generate(Residency& gpu,const std::vector<int>& ids,int count,int chunk) {
    if(ids.empty() || !count || ids.size()+count>gpu.tokens || gpu.tokens>1048576 || chunk<1 || chunk>128)
        throw std::runtime_error("入力・出力長または prefill chunk が範囲外です");
    auto now=std::chrono::steady_clock::now;
    Target target(gpu);
    auto begin=now();
    int next=0;
    for(size_t first=0;first<ids.size();first+=chunk) {
        auto end=std::min(first+chunk,ids.size());
        next=target.forward({ids.begin()+first,ids.begin()+end},false).back();
    }
    auto prefill=now();
    std::vector<int> generated{next};
    int rounds=0,accepted=0;
    while(int(generated.size())<count && generated.back()!=120025) {
        int anchor=generated.back();
        if(!gpu.model.has_draft || target.history.size()+6>gpu.tokens) {
            generated.push_back(target.forward({anchor},false)[0]);
            continue;
        }
        auto draft=target.propose(anchor);
        std::vector<int> block{anchor};
        block.insert(block.end(),draft.begin(),draft.end());
        auto actual=target.forward(block,true);
        int keep=1;
        while(keep<6 && draft[keep-1]==actual[keep-1]) ++keep;
        ++rounds;
        accepted+=keep-1;
        target.accept(keep,6);
        for(int i=0;i<keep && int(generated.size())<count;++i) {
            generated.push_back(actual[i]);
            if(actual[i]==120025) break;
        }
    }
    double pp=std::chrono::duration<double>(prefill-begin).count();
    double tg=std::chrono::duration<double>(now()-prefill).count();
    fprintf(stderr,"prefill_tokens=%zu prefill_s=%.6f decode_tokens=%zu decode_s=%.6f draft_rounds=%d accepted=%d\n",
        ids.size(),pp,generated.size()-1,tg,rounds,accepted);
    printf("GENERATED_IDS=");
    for(size_t i=0;i<generated.size();++i) printf("%s%d",i?",":"",generated[i]);
    puts("");
}
}
