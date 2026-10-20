// Crane-owned launch adapters; dependency headers are included read-only.
// Match the original framework's narrowly scoped softmax rounding boundary;
// other kernel arithmetic retains normal CUDA fused multiply-add behavior.
#define UNFUSE_FMA
#include "flash_fwd_launch_template.h"
#include <cuda_runtime_api.h>
#include <cmath>

static Flash_fwd_params make_params(
    const void* q,const void* k,const void* v,void* out,void* lse,void* output_accum,void* lse_accum,
    int batch,int sequence,int heads,int kv_heads,int width,int splits,bool causal) {
    Flash_fwd_params p{};
    p.q_ptr=const_cast<void*>(q);p.k_ptr=const_cast<void*>(k);p.v_ptr=const_cast<void*>(v);
    p.o_ptr=out;p.softmax_lse_ptr=lse;p.oaccum_ptr=output_accum;p.softmax_lseaccum_ptr=lse_accum;
    p.b=batch;p.h=heads;p.h_k=kv_heads;p.h_h_k_ratio=heads/kv_heads;
    p.d=width;p.d_rounded=width;p.seqlen_q=sequence;p.seqlen_k=sequence;
    p.seqlen_q_rounded=((sequence+127)/128)*128;p.seqlen_k_rounded=p.seqlen_q_rounded;p.total_q=batch*sequence;
    p.q_batch_stride=static_cast<int64_t>(sequence)*heads*width;
    p.k_batch_stride=static_cast<int64_t>(sequence)*kv_heads*width;
    p.v_batch_stride=p.k_batch_stride;p.o_batch_stride=p.q_batch_stride;
    p.q_row_stride=heads*width;p.o_row_stride=p.q_row_stride;
    p.k_row_stride=kv_heads*width;p.v_row_stride=p.k_row_stride;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=p.o_head_stride=width;
    p.scale_softmax=1.f/std::sqrt(static_cast<float>(width));p.scale_softmax_log2=p.scale_softmax*M_LOG2E;
    p.p_dropout=1.f;p.p_dropout_in_uint8_t=255;p.rp_dropout=1.f;p.scale_softmax_rp_dropout=p.scale_softmax;
    p.is_bf16=true;p.is_causal=causal;p.window_size_left=causal?sequence:-1;p.window_size_right=causal?0:-1;
    p.is_seqlens_k_cumulative=true;p.num_splits=splits;
    return p;
}
extern "C" int crane_split_kv_bf16_256(
    const void* q,const void* k,const void* v,void* out,void* lse,void* output_accum,void* lse_accum,
    int batch,int sequence,int heads,int kv_heads,int splits,void* stream_ptr) {
    auto p=make_params(q,k,v,out,lse,output_accum,lse_accum,batch,sequence,heads,kv_heads,256,splits,true);
    try {
        if (splits>1) run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t,256,true>(p,reinterpret_cast<cudaStream_t>(stream_ptr));
        else run_mha_fwd_hdim256<cutlass::bfloat16_t,true>(p,reinterpret_cast<cudaStream_t>(stream_ptr));
        return static_cast<int>(cudaGetLastError());
    } catch (...) {return static_cast<int>(cudaErrorUnknown);}
}
extern "C" int crane_noncausal_bf16_64(
    const void* q,const void* k,const void* v,void* out,void* lse,void* output_accum,void* lse_accum,
    int batch,int sequence,int heads,int kv_heads,int splits,void* stream_ptr) {
    auto p=make_params(q,k,v,out,lse,nullptr,nullptr,batch,sequence,heads,kv_heads,64,1,false);
    try {
        run_flash_fwd<Flash_fwd_kernel_traits<64,128,128,4,false,false,cutlass::bfloat16_t>,false,false>(
            p,reinterpret_cast<cudaStream_t>(stream_ptr));
        return static_cast<int>(cudaGetLastError());
    } catch (...) {return static_cast<int>(cudaErrorUnknown);}
}
