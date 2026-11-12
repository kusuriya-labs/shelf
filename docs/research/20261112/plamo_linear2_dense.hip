#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

void hc(hipError_t e) { if(e!=hipSuccess)std::exit(2); }
void bc(hipblasStatus_t e) { if(e!=HIPBLAS_STATUS_SUCCESS)std::exit(3); }
int main(int argc,char** argv) {
    if(argc!=4)return 1;
    const uint32_t m=std::atoi(argv[1]),n=std::atoi(argv[2]),k=std::atoi(argv[3]);
    hipblasLtHandle_t handle;bc(hipblasLtCreate(&handle));
    hipblasLtMatmulDesc_t desc;bc(hipblasLtMatmulDescCreate(&desc,HIPBLAS_COMPUTE_32I,HIP_R_32I));
    hipblasOperation_t ta=HIPBLAS_OP_T,tb=HIPBLAS_OP_N;
    bc(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
    bc(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
    hipblasLtMatrixLayout_t la,lb,lc;
    bc(hipblasLtMatrixLayoutCreate(&la,HIP_R_8I,k,n,k));
    bc(hipblasLtMatrixLayoutCreate(&lb,HIP_R_8I,k,m,k));
    bc(hipblasLtMatrixLayoutCreate(&lc,HIP_R_32I,n,m,n));
    hipblasLtMatmulPreference_t pref;bc(hipblasLtMatmulPreferenceCreate(&pref));
    const size_t limit=0u;
    bc(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&limit,sizeof(limit)));
    hipblasLtMatmulHeuristicResult_t candidates[16]{};int found=0;
    bc(hipblasLtMatmulAlgoGetHeuristic(handle,desc,la,lb,lc,lc,pref,16,candidates,&found));
    if(!found)return 4;
    int8_t *a,*b;int32_t* c;uint8_t* flush;
    hc(hipMalloc(&a,size_t(n)*k));hc(hipMalloc(&b,size_t(m)*k));
    hc(hipMalloc(&c,size_t(m)*n*4u));hc(hipMalloc(&flush,128u*1024u*1024u));
    std::mt19937 rng(17u);std::vector<int8_t> ha(size_t(n)*k),hb(size_t(m)*k);
    for(auto& v:ha)v=static_cast<int8_t>(rng());for(auto& v:hb)v=static_cast<int8_t>(rng());
    hc(hipMemcpy(a,ha.data(),ha.size(),hipMemcpyHostToDevice));hc(hipMemcpy(b,hb.data(),hb.size(),hipMemcpyHostToDevice));
    const int32_t alpha=1,beta=0;
    std::vector<int32_t> reference(size_t(m)*n),current(reference.size());
    hipEvent_t e0,e1;hc(hipEventCreate(&e0));hc(hipEventCreate(&e1));
    for(int i=0;i<found;++i) {
        if(candidates[i].state!=HIPBLAS_STATUS_SUCCESS || candidates[i].workspaceSize!=0u)continue;
        auto run=[&]{bc(hipblasLtMatmul(handle,desc,&alpha,a,la,b,lb,&beta,c,lc,c,lc,&candidates[i].algo,nullptr,0u,nullptr));};
        run();hc(hipMemcpy(current.data(),c,current.size()*4u,hipMemcpyDeviceToHost));
        if(i==0)reference=current;
        if(current!=reference)return 5;
        std::vector<float> medians;
        for(int repeat=0;repeat<5;++repeat) {
            for(int j=0;j<20;++j)run();hc(hipDeviceSynchronize());
            std::vector<float> us;
            for(int j=0;j<20;++j) {
                hc(hipMemsetAsync(flush,j,128u*1024u*1024u));hc(hipEventRecord(e0));run();hc(hipEventRecord(e1));hc(hipEventSynchronize(e1));
                float ms;hc(hipEventElapsedTime(&ms,e0,e1));us.push_back(ms*1000.0f);
            }
            std::sort(us.begin(),us.end());medians.push_back((us[9]+us[10])*0.5f);
        }
        std::printf("{\"m\":%u,\"n\":%u,\"k\":%u,\"heuristic_rank\":%d,\"index\":%d,\"bit_equal\":true,\"us\":[",m,n,k,i,hipblaslt_ext::getIndexFromAlgo(candidates[i].algo));
        for(size_t j=0;j<medians.size();++j)std::printf("%s%.4f",j?",":"",medians[j]);std::puts("]}");std::fflush(stdout);
    }
    hc(hipEventDestroy(e0));hc(hipEventDestroy(e1));hc(hipFree(a));hc(hipFree(b));hc(hipFree(c));hc(hipFree(flush));
    bc(hipblasLtMatmulPreferenceDestroy(pref));bc(hipblasLtMatrixLayoutDestroy(la));bc(hipblasLtMatrixLayoutDestroy(lb));bc(hipblasLtMatrixLayoutDestroy(lc));bc(hipblasLtMatmulDescDestroy(desc));bc(hipblasLtDestroy(handle));
}
