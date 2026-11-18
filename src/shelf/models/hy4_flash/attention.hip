#include <hip/hip_fp16.h>
#include <shelf/models/hy4_flash/attention.h>
#include <shelf/models/hy4_flash/absorbed.h>
#include <shelf/kernels/fused/detail/octave_common.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace shelf::hy4 {
namespace {
using namespace shelf::kernel::exl3_detail::mul24;
using shelf::kernel::detail::octave::sign_of;
__device__ constexpr float levels[]={
    -2.73258948f,-2.06901717f,-1.61804640f,-1.25623119f,-0.94234043f,-0.65675914f,-0.38804829f,-0.12839504f,
    0.12839504f,0.38804829f,0.65675914f,0.94234043f,1.25623119f,1.61804640f,2.06901717f,2.73258948f};
__device__ float read(Param p,size_t i) {
    return p.type==io::SType::F32?static_cast<const float*>(p.p)[i]:
        bf16_bits_to_f32(static_cast<const uint16_t*>(p.p)[i]);
}
__device__ float bv(Mat x,int row,int col) {return bf16_bits_to_f32(x.p[size_t(row)*x.cols+col]);}
__device__ float wht(float v,float* scratch,int width) {
    int c=threadIdx.x;
    for(int step=1;step<width;step*=2) {
        scratch[c]=v;
        __syncthreads();
        float other=scratch[c^step];
        __syncthreads();
        v=(c&step)?other-v:v+other;
    }
    return v*rsqrtf(float(width));
}
template<bool Max>
__device__ float reduce(float v,float* scratch,int width) {
    int c=threadIdx.x;
    scratch[c]=v;
    __syncthreads();
    for(int step=width/2;step;step/=2) {
        if(c<step) scratch[c]=Max?fmaxf(scratch[c],scratch[c+step]):scratch[c]+scratch[c+step];
        __syncthreads();
    }
    float result=scratch[0];
    __syncthreads();
    return result;
}
template<int Width,int Salt>
__global__ void pack(Mat x,uint8_t* records,int offset) {
    __shared__ float scratch[Width];
    __shared__ uint8_t codes[Width];
    int c=threadIdx.x,sub=blockIdx.x,row=blockIdx.y;
    uint8_t* dst=records+size_t(row)*294+offset+sub*(Width/2+2);
    float v=wht(bv(x,row,sub*Width+c)*sign_of(c,Salt),scratch,Width);
    float ss=reduce<false>(v*v,scratch,Width);
    float z=ss>0?v*rsqrtf(ss/Width):0;
    int lo=0,hi=15;
    while(lo<hi) {
        int mid=(lo+hi)/2;
        if(z>(levels[mid]+levels[mid+1])*0.5f) lo=mid+1;
        else hi=mid;
    }
    codes[c]=ss>0?lo:0;
    float dot=reduce<false>(v*levels[lo],scratch,Width);
    if(c<Width/2) dst[c]=codes[c*2]|(codes[c*2+1]<<4);
    if(!c) {
        uint16_t h=__half_as_ushort(__float2half(dot>0?ss/dot:0));
        dst[Width/2]=h&255;
        dst[Width/2+1]=h>>8;
    }
}
template<int Width,int Salt>
__global__ void unpack(const uint8_t* cache,const uint8_t* fresh,const int* ids,int start,int slots,
                       Mat out,int offset) {
    __shared__ float scratch[Width];
    int c=threadIdx.x,sub=blockIdx.x,row=blockIdx.y,id=ids[row];
    const uint8_t* rec=id>=start?fresh+size_t(id-start)*294:cache+size_t(id%slots)*294;
    const uint8_t* src=rec+offset+sub*(Width/2+2);
    int code=(src[c/2]>>(4*(c%2)))&15;
    float scale=__half2float(__ushort_as_half(src[Width/2]|(src[Width/2+1]<<8)));
    float value=wht(levels[code]*scale,scratch,Width)*sign_of(c,Salt);
    out.p[size_t(row)*out.cols+sub*Width+c]=f32_to_bf16_rne_device(value);
}
__global__ void save_records(uint8_t* cache,const uint8_t* fresh,int rows,int start,int slots) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<294*rows) cache[size_t((start+i/294)%slots)*294+i%294]=fresh[i];
}
__global__ void pool(Mat keys,uint16_t* history,int first,int start,Param values,Param gates,Param ape,Param norm,
                     const uint16_t* pending) {
    __shared__ float pooled[128],scratch[128];
    int c=threadIdx.x,group=first+blockIdx.x;
    float v[4],g[4],peak=-INFINITY;
    for(int i=0;i<4;++i) {
        int pos=4*group+i;
        auto x=pos>=start?keys.p+size_t(pos-start)*128:pending+size_t(pos%12)*128;
        v[i]=0;
        g[i]=read(ape,i*128+c);
        for(int j=0;j<128;++j) {
            float a=bf16_bits_to_f32(x[j]);
            v[i]+=a*read(values,c*128+j);
            g[i]+=a*read(gates,c*128+j);
        }
        peak=fmaxf(peak,g[i]);
    }
    float sum=0,total=0;
    for(int i=0;i<4;++i) {float p=expf(g[i]-peak);sum+=p*v[i];total+=p;}
    float out=sum/total;
    float inv=rsqrtf(reduce<false>(out*out,scratch,128)/128+1e-5f);
    pooled[c]=bf16_bits_to_f32(f32_to_bf16_rne_device(out*inv*read(norm,c)));
    __syncthreads();
    out=pooled[c];
    if(c>=64) {
        float phase=group*4*powf(1e7f,-float((c-64)/2)/32);
        float co=bf16_bits_to_f32(f32_to_bf16_rne_device(cosf(phase)));
        float si=bf16_bits_to_f32(f32_to_bf16_rne_device(sinf(phase)));
        out=out*co+pooled[c^1]*((c&1)?si:-si);
    }
    history[size_t(group)*128+c]=f32_to_bf16_rne_device(out);
}
__global__ void save_pending(Mat k,int start,uint16_t* pending) {
    int row=k.rows-1-blockIdx.x;
    pending[((start+row)%12)*128+threadIdx.x]=k.p[row*128+threadIdx.x];
}
__global__ void index_scores(Mat q,Mat weights,const uint16_t* groups,int row,int count,float* scores) {
    int group=blockIdx.x*4+threadIdx.x/32;
    int lane=threadIdx.x%32;
    if(group>=count) return;
    float sum=0;
    for(int h=0;h<16;++h) {
        float dot=0;
        for(int j=lane;j<128;j+=32)
            dot+=bv(q,row,h*128+j)*bf16_bits_to_f32(groups[size_t(group)*128+j]);
        for(int dist=16;dist;dist/=2) dot+=__shfl_down(dot,dist,32);
        sum+=fmaxf(dot,0)*bv(weights,row,h);
    }
    if(!lane) scores[group]=sum;
}
Selection select(Ops& op,const std::string& p,int layer,int start,Mat x,Mat qres) {
    int end=start+x.rows;
    Selection selected(x.rows);
    bool sliding=layer!=0 && layer%4!=1;
    if(!sliding) {
        Mat k=op.norm(op.linear(p+"token_key_projection",x),p+"token_key_norm.weight");
        auto groups=op.gpu.index_keys[layer];
        auto pending=groups+(op.gpu.tokens/4)*128;
        int first=start/4,ng=end/4-first;
        if(ng) {
            pool<<<ng,128>>>(k,groups,first,start,op.param(p+"key_value_projection.weight"),
                op.param(p+"key_gate_projection.weight"),op.param(p+"intra_group_ape"),
                op.param(p+"key_norm.weight"),pending);
            checked(hipGetLastError());
        }
        save_pending<<<std::min(k.rows,12),128>>>(k,start,pending);
        checked(hipGetLastError());
        if(end>2048) {
            Mat q=op.rope(op.linear(p+"query_projection",qres),16,start,1e7f);
            Mat w=op.linear(p+"head_weight_projection",x);
            auto buffer=static_cast<float*>(op.arena.next(size_t(end/4)*4));
            for(int i=std::max(0,2048-start);i<x.rows;++i) {
                int pos=start+i,count=pos/4;
                index_scores<<<(count+3)/4,128>>>(q,w,groups,i,count,buffer);
                checked(hipGetLastError());
                std::vector<float> scores(count);
                op.copy(scores.data(),buffer,count*4,hipMemcpyDeviceToHost);
                std::vector<int> ids(count);
                std::iota(ids.begin(),ids.end(),0);
                std::partial_sort(ids.begin(),ids.begin()+511,ids.end(),[&](int a,int b){
                    return scores[a]==scores[b]?a<b:scores[a]>scores[b];
                });
                for(int g=0;g<511;++g) for(int j=0;j<4;++j) selected[i].push_back(ids[g]*4+j);
                for(int j=count*4;j<=pos;++j) selected[i].push_back(j);
            }
        }
    }
    for(int i=0;i<x.rows;++i) if(selected[i].empty()) {
        int pos=start+i,first=sliding?std::max(0,pos-511):0;
        selected[i].resize(pos-first+1);
        std::iota(selected[i].begin(),selected[i].end(),first);
    }
    return selected;
}
__global__ void attend(Mat q,Mat kv,Mat rotary,const int* indices,int stride,Param sink,Mat gate,Mat output) {
    extern __shared__ float scores[];
    __shared__ float scratch[256];
    int head=blockIdx.x,row=blockIdx.y,c=threadIdx.x;
    float maxval=read(sink,head);
    for(int t=c;t<stride;t+=256) {
        int idx=indices[row*stride+t];
        float s=-INFINITY;
        if(idx>=0) {
            float dot=0;
            for(int j=0;j<192;++j) dot+=bv(q,row,head*256+j)*bv(kv,idx,head*448+j);
            for(int j=0;j<64;++j) dot+=bv(q,row,head*256+192+j)*bv(rotary,idx,j);
            s=dot*0.0625f;
        }
        scores[t]=s;
        maxval=fmaxf(maxval,s);
    }
    float peak=reduce<true>(maxval,scratch,256);
    float denom=c==0?expf(read(sink,head)-peak):0;
    for(int t=c;t<stride;t+=256) {scores[t]=expf(scores[t]-peak);denom+=scores[t];}
    float sum=reduce<false>(denom,scratch,256),v=0;
    for(int t=0;t<stride;++t) {
        int idx=indices[row*stride+t];
        if(idx>=0) v+=scores[t]*bv(kv,idx,head*448+192+c);
    }
    output.p[size_t(row)*output.cols+head*256+c]=f32_to_bf16_rne_device(
        v/sum/(1.f+expf(-bv(gate,row,head*256+c))));
}
}
Mat attention(Ops& op,const std::string& prefix,int layer,int start,Mat hidden,Selection& selected,bool draft) {
    std::string p=prefix+"self_attn.";
    bool sliding=layer!=0 && layer%4!=1;
    int heads=sliding?64:32;
    float theta=sliding?1e4f:1e7f;
    Mat qres=op.norm(op.linear(p+"q_a_proj",hidden),p+"q_a_layernorm.weight");
    Mat q=op.rope(op.linear(p+"q_b_proj",qres),heads,start,theta);
    Mat ka=op.linear(p+"kv_a_proj_with_mqa",hidden);
    Mat lat=op.norm(op.slice(ka,0,512),p+"kv_a_layernorm.weight");
    Mat rot=op.rope(op.slice(ka,512,64),1,start,theta);
    Mat gate=op.linear(p+"linear_gate",hidden);
    auto fresh=static_cast<uint8_t*>(op.arena.next(size_t(hidden.rows)*294));
    pack<256,29><<<dim3(2,hidden.rows),256>>>(lat,fresh,0);
    checked(hipGetLastError());
    pack<64,11><<<dim3(1,hidden.rows),64>>>(rot,fresh,260);
    checked(hipGetLastError());
    if(!draft) selected=select(op,p+"compressed_indexer.",layer,start,hidden,qres);
    Mat out=op.mat(hidden.rows,heads*256);
    int slots=sliding?std::min(op.gpu.tokens,size_t(518)):op.gpu.tokens;
    auto cache=op.gpu.cache[layer];
    for(int i=0;i<hidden.rows;) {
        size_t save=op.arena.cursor;
        int n=(start+hidden.rows<=2048 || sliding || draft)?hidden.rows-i:1;
        std::vector<int> ids;
        int stride=0;
        for(int r=i;r<i+n;++r) {
            ids.insert(ids.end(),selected[r].begin(),selected[r].end());
            stride=std::max(stride,int(selected[r].size()));
        }
        std::sort(ids.begin(),ids.end());
        ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
        std::vector<int> offsets(stride*n,-1);
        for(int r=0;r<n;++r) for(size_t t=0;t<selected[i+r].size();++t)
            offsets[r*stride+t]=std::lower_bound(ids.begin(),ids.end(),selected[i+r][t])-ids.begin();
        auto positions=op.indices(ids);
        Mat a=op.mat(ids.size(),512),b=op.mat(ids.size(),64);
        unpack<256,29><<<dim3(2,ids.size()),256>>>(cache,fresh,positions,start,slots,a,0);
        checked(hipGetLastError());
        unpack<64,11><<<dim3(1,ids.size()),64>>>(cache,fresh,positions,start,slots,b,260);
        checked(hipGetLastError());
        auto offsets_gpu=op.indices(offsets);
        if(!op.gpu.folded.empty()) {
            attend_absorbed(op,p+"kv_b_proj",a,b,q.row(i,n),gate.row(i,n),offsets_gpu,stride,
                            op.param(p+"learnable_sink_param"),out.row(i,n));
        } else {
            Mat expanded=op.linear(p+"kv_b_proj",a);
            attend<<<dim3(heads,n),256,stride*sizeof(float)>>>(q.row(i,n),expanded,b,offsets_gpu,stride,
                op.param(p+"learnable_sink_param"),gate.row(i,n),out.row(i,n));
            checked(hipGetLastError());
        }
        op.arena.cursor=save;
        i+=n;
    }
    if(!draft) {
        save_records<<<(hidden.rows*294+255)/256,256>>>(cache,fresh,hidden.rows,start,slots);
        checked(hipGetLastError());
    }
    return op.norm(op.linear(p+"o_proj",out),p+"post_proj_norm.weight");
}
}
