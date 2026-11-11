// Standalone microbenchmark: compile alongside recurrence.hip with the project's HIP flags.
#include <shelf/kernels/optimized/gdn/recurrence.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace shelf;
#define HIP(x) do { auto e=(x); if(e!=hipSuccess){std::fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); std::exit(1);} } while(0)
template<class T> T* alloc(size_t n) { T* p=nullptr; HIP(hipMalloc(&p,n*sizeof(T))); HIP(hipMemset(p,0,n*sizeof(T))); return p; }
int main() {
  constexpr unsigned kh=16, vh=48, d=128, layers=48;
  for (auto [nr,nrow] : std::vector<std::pair<unsigned,unsigned>>{{1,8},{1,10},{1,16},{4,8},{8,8},{16,4},{64,1}}) {
    unsigned rows=nr*nrow; uint64_t layer=(uint64_t)vh*d*d, slot=layers*layer;
    kernel::GdnRecurrenceArgs a{};
    a.q=alloc<float>(rows*kh*d); a.k=alloc<float>(rows*kh*d); a.v=alloc<bf16_t>(rows*vh*d);
    a.a=alloc<bf16_t>(rows*vh); a.b=alloc<bf16_t>(rows*vh); a.dt_bias=alloc<bf16_t>(vh); a.a_log=alloc<bf16_t>(vh);
    a.output=alloc<float>(rows*vh*d); a.recurrent_state=alloc<fp16_t>(nr*slot);
    auto* history=alloc<fp16_t>(rows*layer);
    std::vector<runtime::DeviceRequestDescriptor> req(nr);
    for(unsigned i=0;i<nr;++i){req[i].request_handle.slot=i;req[i].row_begin=i*nrow;req[i].row_count=nrow;}
    auto* dr=alloc<runtime::DeviceRequestDescriptor>(nr); HIP(hipMemcpy(dr,req.data(),nr*sizeof(req[0]),hipMemcpyHostToDevice));
    a.requests=dr;a.num_requests=nr;a.actual_rows=rows;a.max_sequences=nr;a.num_gdn_states=layers;
    a.key_heads=kh;a.num_v_heads=vh;a.head_k=d;a.head_v=d;a.recurrent_slot_stride=slot;a.recurrent_layer_stride=layer;
    a.q_row_stride=a.k_row_stride=kh*d;a.v_row_stride=a.output_row_stride=vh*d;a.a_row_stride=a.b_row_stride=vh;
    hipEvent_t start,stop; HIP(hipEventCreate(&start));HIP(hipEventCreate(&stop));
    for(int mode=0;mode<5;++mode){
      if (mode==4 && nrow!=1) continue;
      a.recurrent_history=mode==2?history:nullptr; a.capture_rows=mode==2?rows:0;
      a.recurrent_history_row_stride=layer; a.skip_state_store=mode==3;
      auto run=[&](){return mode==0?kernel::launch_gdn_recurrence_f32_wmma(a,nullptr):mode==4?kernel::launch_gdn_recurrence_f32_wmma_single_row_experimental(a,nullptr):kernel::launch_gdn_recurrence_f32_wmma_verify(a,nullptr);};
      for(int i=0;i<3;++i) HIP(run()); HIP(hipDeviceSynchronize());
      HIP(hipEventRecord(start));for(int i=0;i<20;++i) HIP(run());HIP(hipEventRecord(stop));HIP(hipEventSynchronize(stop));
      float ms=0;HIP(hipEventElapsedTime(&ms,start,stop));
      std::printf("%u x %u %s %.6f ms\n",nr,nrow,(const char*[]){"decode","verify","history","no_store","single_row_experimental"}[mode],ms/20);std::fflush(stdout);
    }
    HIP(hipEventDestroy(start));HIP(hipEventDestroy(stop));
    for (const void* p : {(const void*)a.q,(const void*)a.k,(const void*)a.v,(const void*)a.a,(const void*)a.b,(const void*)a.dt_bias,(const void*)a.a_log,(const void*)a.output,(const void*)a.recurrent_state,(const void*)history,(const void*)dr}) HIP(hipFree(const_cast<void*>(p)));
  }
}
