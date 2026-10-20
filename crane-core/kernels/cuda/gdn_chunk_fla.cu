/*
MIT License

Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// SPDX-License-Identifier: MIT
// Source implementation of FLA's scalar-gate, 64-token WY formulation.
// Algorithm: flash-linear-attention, Copyright (c) 2023-2026 FLA contributors.
// No generated kernel or launch-plan assets are consumed by these kernels.
#include <cuda_bf16.h>
#include <mma.h>
#include <math.h>
using BF = __nv_bfloat16;
namespace wm = nvcuda::wmma;

__device__ __forceinline__ float f32(BF x) {return __bfloat162float(x);}
__device__ __forceinline__ BF bf(float x) {return __float2bfloat16_rn(x);}
__device__ __forceinline__ float e2(float x) {float y;asm("ex2.approx.ftz.f32 %0, %1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ unsigned pair(BF a,BF b) {return (unsigned)__bfloat16_as_ushort(a)|((unsigned)__bfloat16_as_ushort(b)<<16);}
__device__ __forceinline__ unsigned pair_f32(float a,float b) {
    unsigned p;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p) : "f"(a),"f"(b));
    return p;
}
__device__ __forceinline__ unsigned load_pair(const BF* __restrict__ x) {return *reinterpret_cast<const unsigned*>(x);}
__device__ __forceinline__ void mma_bf16(float (&d)[4],const unsigned (&a)[4],const unsigned (&b)[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
// Convert an accumulator-layout BF16 tile to a dot-B fragment with warp
// shuffles. No shared staging, block barrier, or local-memory spill is needed.
__device__ __forceinline__ void c_to_b(unsigned c0,unsigned c8,unsigned (&b)[2]) {
    int lane=threadIdx.x&31,g=lane/4,t=lane%4,s0=t*8+g/2,s1=s0+4,shift=(g&1)*16;
    unsigned a=__shfl_sync(0xffffffff,c0,s0),c=__shfl_sync(0xffffffff,c0,s1);
    unsigned d=__shfl_sync(0xffffffff,c8,s0),e=__shfl_sync(0xffffffff,c8,s1);
    b[0]=((a>>shift)&65535)|(((c>>shift)&65535)<<16);
    b[1]=((d>>shift)&65535)|(((e>>shift)&65535)<<16);
}
__device__ __forceinline__ void float_c_to_b(const float (&c)[4],unsigned (&b)[2]) {
    c_to_b(pair_f32(c[0],c[1]),pair_f32(c[2],c[3]),b);
}

// FLA vectorizes eight adjacent elements per lane and fuses its local sum.
__device__ __forceinline__ void l2_impl(const BF* __restrict__ x, BF* __restrict__ y, int rows, int D) {
    int lanes=D/8, lane=threadIdx.x%lanes, row=(blockIdx.x*blockDim.x+threadIdx.x)/lanes;
    float a[8];
    const BF* input=x+row*D+lane*8;
    uint4 raw=make_uint4(0,0,0,0);
    bool aligned=(reinterpret_cast<unsigned long long>(x)&15)==0;
    if(row<rows&&aligned)raw=*reinterpret_cast<const uint4*>(input);
    unsigned words[4]={raw.x,raw.y,raw.z,raw.w};
#pragma unroll
    for(int j=0;j<8;++j) {
        a[j]=f32(__ushort_as_bfloat16((words[j/2]>>((j%2)*16))&65535));
        if(row<rows&&!aligned)a[j]=f32(input[j]);
    }
    float sum=__fmul_rn(a[1],a[1]);
    sum=__fmaf_rn(a[0],a[0],sum);
#pragma unroll
    for(int j=2;j<8;++j)sum=__fmaf_rn(a[j],a[j],sum);
    for(int off=lanes/2;off>0;off/=2)sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,off));
    sum=__fadd_rn(sum,1.e-6f);
    float root, inv;
    asm("sqrt.approx.ftz.f32 %0, %1;":"=f"(root):"f"(sum));
    asm("div.full.f32 %0, 1.0, %1;":"=f"(inv):"f"(root));
    if(row<rows) {
        uint4 packed=make_uint4(
            pair_f32(__fmul_rn(a[0],inv),__fmul_rn(a[1],inv)),
            pair_f32(__fmul_rn(a[2],inv),__fmul_rn(a[3],inv)),
            pair_f32(__fmul_rn(a[4],inv),__fmul_rn(a[5],inv)),
            pair_f32(__fmul_rn(a[6],inv),__fmul_rn(a[7],inv)));
        *reinterpret_cast<uint4*>(y+row*D+lane*8)=packed;
    }
}

extern "C" __global__ void gdn_fla_l2(const BF* x,BF* y,int rows,int D) {l2_impl(x,y,rows,D);}
extern "C" __global__ void gdn_fla_l2_pair(const BF* q,const BF* k,BF* qn,BF* kn,int rows,int D) {
    l2_impl(blockIdx.y?k:q,blockIdx.y?kn:qn,rows,D);
}

__device__ __forceinline__ void cumsum_impl(const float* __restrict__ g, float* __restrict__ out, int T, int HV) {
    int lane=threadIdx.x, head=blockIdx.y%HV, batch=blockIdx.y/HV, base=blockIdx.x*64;
    int t0=base+lane,t1=t0+32;
    float a=t0<T?g[(batch*T+t0)*HV+head]:0.f;
    float b=t1<T?g[(batch*T+t1)*HV+head]:0.f;
    for(int off=1;off<32;off*=2) {
        float av=__shfl_up_sync(0xffffffff,a,off),bv=__shfl_up_sync(0xffffffff,b,off);
        if(lane>=off) {a=__fadd_rn(a,av);b=__fadd_rn(b,bv);}
    }
    b=__fadd_rn(b,__shfl_sync(0xffffffff,a,31));
    if(t0<T)out[(batch*T+t0)*HV+head]=__fmul_rn(a,1.4426950216293334961f);
    if(t1<T)out[(batch*T+t1)*HV+head]=__fmul_rn(b,1.4426950216293334961f);
}

extern "C" __global__ void gdn_fla_cumsum(const float* g,float* out,int T,int HV) {cumsum_impl(g,out,T,HV);}

// Float input TF32 dots reproduce FLA's block merge precision. The hardware
// consumes TF32 significands; no extra BF16 rounding is introduced here.
// Diagonal/elementwise operands use even/odd K slices; a nested-dot result
// uses contiguous K slices. Keep dot-add in the MMA accumulator, not a
// separate rounded matrix addition.
template<bool InterleavedK,bool Accumulate=false>
__device__ __forceinline__ void dot_tf32_16(const float* __restrict__ A,int as,const float* __restrict__ B,int bs,float* __restrict__ C,int cs,float sign) {
    int lane=threadIdx.x&31;
    int row=lane/4,col=lane%4;
#pragma unroll
    for(int n=0;n<16;n+=8) {
        float d0=0.f,d1=0.f,d2=0.f,d3=0.f;
        if(Accumulate) {
            d0=C[row*cs+n+col*2];d1=C[row*cs+n+col*2+1];d2=C[(row+8)*cs+n+col*2];d3=C[(row+8)*cs+n+col*2+1];
        }
#pragma unroll
        for(int k=0;k<16;k+=8) {
            int p0=InterleavedK?col*2+k/8:k+col,p1=InterleavedK?(col+4)*2+k/8:k+col+4;
            unsigned a0=__float_as_uint(A[row*as+p0]);
            unsigned a1=__float_as_uint(A[(row+8)*as+p0]);
            unsigned a2=__float_as_uint(A[row*as+p1]);
            unsigned a3=__float_as_uint(A[(row+8)*as+p1]);
            unsigned b0=__float_as_uint(B[p0*bs+n+row]);
            unsigned b1=__float_as_uint(B[p1*bs+n+row]);
            asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
        C[row*cs+n+col*2]=sign*d0;C[row*cs+n+col*2+1]=sign*d1;
        C[(row+8)*cs+n+col*2]=sign*d2;C[(row+8)*cs+n+col*2+1]=sign*d3;
    }
    __syncwarp();
}

__device__ __forceinline__ void intra_impl(const BF* __restrict__ k,const float* __restrict__ g,const BF* __restrict__ beta,BF* __restrict__ output,
                                        int T,int H,int HV,int K,float* __restrict__ debug_inverse,float* __restrict__ debug_scaled_kkt,float* __restrict__ debug_merge) {
    // Reuse each scaled KKT block once all its raw consumers have run.
    __shared__ __align__(32) float A[64*68],tmp[1][256];
    __shared__ __align__(32) BF ba_all[4][256],bb_all[4][256];
    int lane=threadIdx.x&31,warp=threadIdx.x/32;
    int chunk=blockIdx.x,bh=blockIdx.y,batch=bh/HV,head=bh%HV,kh=head/(HV/H),base=chunk*64;
    BF* __restrict__ ba=ba_all[warp];BF* __restrict__ bb=bb_all[warp];
    __shared__ __align__(32) BF keys[64*136];
    if(K<=128) {
        for(int z=threadIdx.x*8;z<64*128;z+=blockDim.x*8) {
            int row=z/128,d=z%128,token=base+row;
            uint4 raw=make_uint4(0,0,0,0);
            if(token<T&&d<K)raw=*reinterpret_cast<const uint4*>(k+((batch*T+token)*H+kh)*K+d);
            *reinterpret_cast<uint4*>(keys+row*136+d)=raw;
        }
        __syncthreads();
        for(int tile=warp;tile<10;tile+=4) {
            int ri=0,ci=0,index=0;
            for(int r=0;r<4;++r)for(int c=0;c<=r;++c){if(index==tile){ri=r;ci=c;}++index;}
            float acc[2][4]={};
            for(int kk=0;kk<K;kk+=16) {
                unsigned a[4],b[2],addr=(unsigned)__cvta_generic_to_shared(keys+(ri*16+(lane&15))*136+kk+(lane/16)*8);
                asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]) : "r"(addr));
#pragma unroll
                for(int n=0;n<2;++n) {
                    addr=(unsigned)__cvta_generic_to_shared(keys+(ci*16+n*8+(lane&7))*136+kk+((lane/8)&1)*8);
                    asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                        : "=r"(b[0]),"=r"(b[1]) : "r"(addr));
                    mma_bf16(acc[n],a,b);
                }
            }
#pragma unroll
            for(int n=0;n<2;++n) {
#pragma unroll
                for(int j=0;j<4;++j)A[(ri*16+lane/4+(j/2)*8)*68+ci*16+n*8+(lane%4)*2+j%2]=acc[n][j];
            }
        }
    } else {
    for(int tile=warp;tile<10;tile+=4) {
        int ri=0,ci=0,n=0;
        for(int r=0;r<4;++r)for(int c=0;c<=r;++c) {if(n==tile){ri=r;ci=c;}++n;}
        wm::fragment<wm::matrix_a,16,16,16,BF,wm::row_major> fa;
        wm::fragment<wm::matrix_b,16,16,16,BF,wm::col_major> fb;
        wm::fragment<wm::accumulator,16,16,16,float> acc;
        wm::fill_fragment(acc,0.f);
        for(int kk=0;kk<K;kk+=16) {
            for(int z=lane;z<256;z+=32) {
                int token=base+ri*16+z/16,d=kk+z%16;
                ba[z]=token<T&&d<K?k[((batch*T+token)*H+kh)*K+d]:bf(0.f);
                token=base+ci*16+z/16;
                bb[z]=token<T&&d<K?k[((batch*T+token)*H+kh)*K+d]:bf(0.f);
            }
            __syncwarp();wm::load_matrix_sync(fa,ba,16);wm::load_matrix_sync(fb,bb,16);
            wm::mma_sync(acc,fa,fb,acc);__syncwarp();
        }
        wm::store_matrix_sync(A+ri*16*68+ci*16,acc,68,wm::mem_row_major);
    }
    }
    __syncthreads();
    for(int z=threadIdx.x;z<4096;z+=blockDim.x) {
        int r=z/64,c=z%64,tr=base+r,tc=base+c;
        float a=0.f;
        if(r>c&&tr<T&&tc<T) {
            a=__fmul_rn(A[r*68+c],e2(g[(batch*T+tr)*HV+head]-g[(batch*T+tc)*HV+head]));
            a=__fmul_rn(a,f32(beta[(batch*T+tr)*HV+head]));
        }
        if(debug_scaled_kkt&&tr<T)debug_scaled_kkt[((batch*T+tr)*HV+head)*64+c]=a;
        A[r*68+c]=(r/16==c/16)?-a:a;
    }
    __syncthreads();
    // Four independent 16x16 diagonal solves. Pairwise reduction across the
    // row dimension follows the associative reduction in the FLA source.
    if(lane<16) {
        int start=warp*16;
        int count=min(16,T-base-start);
        for(int row=2;row<count;++row) {
            float values[8];
#pragma unroll
            for(int j=0;j<8;++j) {
                float a0=j<row?A[(start+row)*68+start+j]:0.f;
                float a8=j+8<row?A[(start+row)*68+start+j+8]:0.f;
                values[j]=__fmaf_rn(a0,A[(start+j)*68+start+lane],__fmul_rn(a8,A[(start+j+8)*68+start+lane]));
            }
#pragma unroll
            for(int width=4;width>0;width/=2) {
#pragma unroll
                for(int j=0;j<width;++j)values[j]=__fadd_rn(values[j],values[j+width]);
            }
            float initial=lane<row?A[(start+row)*68+start+lane]:0.f;
            __syncwarp(0xffff);
            A[(start+row)*68+start+lane]=__fadd_rn(initial,values[0]);
            __syncwarp(0xffff);
        }
        A[(start+lane)*68+start+lane]=1.f;
    }
    __syncthreads();
    if(warp==0) {
#define AA(r,c) (A+(r)*16*68+(c)*16)
#define II(r,c) AA(r,c)
#define DOT(a,as,b,bs,c,cs,sign) dot_tf32_16<((sign)>0.f)>(a,as,b,bs,c,cs,sign)
#define DOTADD(a,as,b,bs,c,cs) dot_tf32_16<true,true>(a,as,b,bs,c,cs,1.f)
        DOT(II(1,1),68,AA(1,0),68,tmp[0],16,1.f);
        if(debug_merge)for(int z=lane;z<256;z+=32) {
            int t=base+16+z/16;
            if(t<T)debug_merge[((batch*T+t)*HV+head)*64+z%16]=tmp[0][z];
        }
        DOT(tmp[0],16,II(0,0),68,II(1,0),68,-1.f);
        // Consume raw (2,1) before replacing it by its solved block.
        DOT(AA(2,0),68,II(0,0),68,tmp[0],16,1.f);DOTADD(AA(2,1),68,II(1,0),68,tmp[0],16);
        DOT(II(2,2),68,tmp[0],16,II(2,0),68,-1.f);
        DOT(II(2,2),68,AA(2,1),68,tmp[0],16,1.f);DOT(tmp[0],16,II(1,1),68,II(2,1),68,-1.f);
        // Raw (3,1)/(3,2) still feed (3,0); retain them until it is solved.
        DOT(AA(3,0),68,II(0,0),68,tmp[0],16,1.f);DOTADD(AA(3,1),68,II(1,0),68,tmp[0],16);DOTADD(AA(3,2),68,II(2,0),68,tmp[0],16);
        DOT(II(3,3),68,tmp[0],16,II(3,0),68,-1.f);
        DOT(AA(3,1),68,II(1,1),68,tmp[0],16,1.f);DOTADD(AA(3,2),68,II(2,1),68,tmp[0],16);
        DOT(II(3,3),68,tmp[0],16,II(3,1),68,-1.f);
        DOT(II(3,3),68,AA(3,2),68,tmp[0],16,1.f);DOT(tmp[0],16,II(2,2),68,II(3,2),68,-1.f);
#undef DOT
#undef II
#undef AA
    }
    __syncthreads();
    for(int z=threadIdx.x;z<4096;z+=blockDim.x) {
        int r=z/64,c=z%64,t=base+r;
        if(t<T) {
            int dst=((batch*T+t)*HV+head)*64+c;
#undef DOTADD
            output[dst]=bf(A[r*68+c]);
            if(debug_inverse)debug_inverse[dst]=A[r*68+c];
        }
    }
}

extern "C" __global__ void gdn_fla_intra(
    const BF* k,const float* g,const BF* beta,BF* output,int T,int H,int HV,int K,
    float* debug_inverse,float* debug_scaled_kkt,float* debug_merge) {
    intra_impl(k,g,beta,output,T,H,HV,K,debug_inverse,debug_scaled_kkt,debug_merge);
}
extern "C" __global__ void gdn_fla_intra_scan(
    const BF* k,const float* g,const BF* beta,BF* output,float* gc,int T,int H,int HV,int K,
    float* debug_inverse,float* debug_scaled_kkt,float* debug_merge) {
    if(threadIdx.x<32)cumsum_impl(g,gc,T,HV);
    __syncthreads();
    intra_impl(k,gc,beta,output,T,H,HV,K,debug_inverse,debug_scaled_kkt,debug_merge);
}

extern "C" __global__ void gdn_fla_wu_reg(const BF* __restrict__ k,const BF* __restrict__ v,const BF* __restrict__ beta,const BF* __restrict__ inverse,
                                         const float* __restrict__ g,BF* __restrict__ w,BF* __restrict__ u,int T,int H,int HV,int K,int V) {
    int lane=threadIdx.x&31,r=(threadIdx.x/32)*16+lane/4,t=lane%4,base=blockIdx.y*64;
    int batch=blockIdx.z/HV,head=blockIdx.z%HV,kh=head/(HV/H),kw=(K+31)/32;
    bool is_w=blockIdx.x<kw;int D=is_w?K:V,col=(is_w?blockIdx.x:blockIdx.x-kw)*32;
    __shared__ __align__(32) BF tile[64*40];
    const BF* x=is_w?k:v;int hd=is_w?kh:head,heads=is_w?H:HV;
    for(int z=threadIdx.x*8;z<64*32;z+=blockDim.x*8) {
        int row=z/32,d=z%32,token=base+row,vc=col+d;
        uint4 raw=make_uint4(0,0,0,0);
        // Widths not divisible by eight retain the scalar tail path.
        if(token<T&&vc+7<D&&(D%8)==0)
            raw=*reinterpret_cast<const uint4*>(x+((batch*T+token)*heads+hd)*D+vc);
        unsigned words[4]={raw.x,raw.y,raw.z,raw.w};
        float weight=token<T?f32(beta[(batch*T+token)*HV+head]):0.f;
        float gate=is_w&&token<T?e2(g[(batch*T+token)*HV+head]):1.f;
#pragma unroll
        for(int j=0;j<8;++j) {
            BF value=__ushort_as_bfloat16((words[j/2]>>((j%2)*16))&65535);
            if((D%8)!=0)value=token<T&&vc+j<D?x[((batch*T+token)*heads+hd)*D+vc+j]:bf(0.f);
            value=bf(__fmul_rn(f32(value),weight));
            tile[row*40+d+j]=is_w?bf(__fmul_rn(f32(value),gate)):value;
        }
    }
    __syncthreads();
    float acc[4][4]={};
#pragma unroll
    for(int kk=0;kk<64;kk+=16) {
        unsigned a[4],b[2];
        a[0]=base+r<T?load_pair(inverse+((batch*T+base+r)*HV+head)*64+kk+t*2):0;
        a[1]=base+r+8<T?load_pair(inverse+((batch*T+base+r+8)*HV+head)*64+kk+t*2):0;
        a[2]=base+r<T?load_pair(inverse+((batch*T+base+r)*HV+head)*64+kk+t*2+8):0;
        a[3]=base+r+8<T?load_pair(inverse+((batch*T+base+r+8)*HV+head)*64+kk+t*2+8):0;
#pragma unroll
        for(int n=0;n<4;++n) {
            unsigned address=(unsigned)__cvta_generic_to_shared(tile+(kk+(lane&15))*40+n*8);
            asm("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                : "=r"(b[0]),"=r"(b[1]) : "r"(address));
            mma_bf16(acc[n],a,b);
        }
    }
#pragma unroll
    for(int n=0;n<4;++n) {
#pragma unroll
        for(int j=0;j<4;++j) {
            int row=base+r+(j/2)*8,d=col+n*8+t*2+j%2;
            if(row<T&&d<D)(is_w?w:u)[((batch*T+row)*HV+head)*D+d]=bf(acc[n][j]);
        }
    }
}

#define GDN_STATE_PARAMS const BF* __restrict__ k,const BF* __restrict__ w,const BF* __restrict__ u,const float* __restrict__ g,const float* __restrict__ initial,BF* __restrict__ snapshots,BF* __restrict__ vnew,float* __restrict__ final_state,int T,int H,int HV,int K,int V
#define GDN_STATE_PASS k,w,u,g,initial,snapshots,vnew,final_state,T,H,HV,K,V
__device__ __forceinline__ void load_matrix_a(unsigned (&a)[4],const BF* __restrict__ p) {
    unsigned address=(unsigned)__cvta_generic_to_shared(p);
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]) : "r"(address));
}
// Transpose a 16x16 tile in the shared-memory load instruction.
// Swap the middle 8x8 quadrants to retain the MMA A register layout.
__device__ __forceinline__ void load_matrix_at(unsigned (&a)[4],const BF* p) {
    unsigned address=(unsigned)__cvta_generic_to_shared(p);
    asm("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a[0]),"=r"(a[2]),"=r"(a[1]),"=r"(a[3]) : "r"(address));
}
template<int MAX_K>
__device__ __forceinline__ void stage_state_matrix(const BF* __restrict__ source,BF* __restrict__ matrix,int batch,int head,int heads,int T,int K,int base) {
    for(int z=threadIdx.x*8;z<64*MAX_K;z+=blockDim.x*8) {
        int row=z/MAX_K,d=z%MAX_K,token=base+row;
        uint4 value=make_uint4(0,0,0,0);
        if(token<T&&d<K)value=*reinterpret_cast<const uint4*>(source+((batch*T+token)*heads+head)*K+d);
        *reinterpret_cast<uint4*>(matrix+row*(MAX_K+8)+d)=value;
    }
}
template<int MAX_K,int VC=2>
__device__ __forceinline__ void state_register(GDN_STATE_PARAMS) {
    constexpr int KT=MAX_K/16;
    __shared__ __align__(32) BF matrix[64*(MAX_K+8)];
    int lane=threadIdx.x&31,warp=threadIdx.x/32,r=lane/4,t=lane%4;
    int batch=blockIdx.y/HV,head=blockIdx.y%HV,kh=head/(HV/H),vbase=blockIdx.x*(16*VC)+warp*(8*VC),NT=(T+63)/64;
    float state[KT][VC][4];
#pragma unroll
    for(int kt=0;kt<KT;++kt) {
#pragma unroll
        for(int n=0;n<VC;++n) {
#pragma unroll
            for(int j=0;j<4;j+=2) {
                int kd=kt*16+r+(j/2)*8,vd=vbase+n*8+t*2;
                if(initial&&kd<K&&vd+1<V&&(V%2)==0) {
                    float2 raw=*reinterpret_cast<const float2*>(initial+((batch*HV+head)*K+kd)*V+vd);
                    state[kt][n][j]=raw.x;state[kt][n][j+1]=raw.y;
                } else {
                    state[kt][n][j]=initial&&kd<K&&vd<V?initial[((batch*HV+head)*K+kd)*V+vd]:0.f;
                    state[kt][n][j+1]=initial&&kd<K&&vd+1<V?initial[((batch*HV+head)*K+kd)*V+vd+1]:0.f;
                }
            }
        }
    }
    for(int chunk=0;chunk<NT;++chunk) {
        int base=chunk*64;
#pragma unroll
        for(int kt=0;kt<KT;++kt) {
#pragma unroll
            for(int n=0;n<VC;++n) {
#pragma unroll
                for(int j=0;j<4;j+=2) {
                    int kd=kt*16+r+(j/2)*8,vd=vbase+n*8+t*2;
                    if(kd<K&&vd+1<V&&(V%2)==0)
                        *reinterpret_cast<unsigned*>(snapshots+(((batch*NT+chunk)*HV+head)*K+kd)*V+vd)=pair_f32(state[kt][n][j],state[kt][n][j+1]);
                    else {
                        if(kd<K&&vd<V)snapshots[(((batch*NT+chunk)*HV+head)*K+kd)*V+vd]=bf(state[kt][n][j]);
                        if(kd<K&&vd+1<V)snapshots[(((batch*NT+chunk)*HV+head)*K+kd)*V+vd+1]=bf(state[kt][n][j+1]);
                    }
                }
            }
        }
        // Cache conversion for K<=128 only: wider states would spill heavily.
        unsigned state_b[KT][VC][2];
#pragma unroll
        for(int kt=0;kt<KT;++kt) {
#pragma unroll
            for(int n=0;n<VC;++n)if constexpr(MAX_K<=128)float_c_to_b(state[kt][n],state_b[kt][n]);
        }
        stage_state_matrix<MAX_K>(w,matrix,batch,head,HV,T,K,base);
        __syncthreads();
        unsigned values[4][VC][2];
        float glast=g[(batch*T+min(base+64,T)-1)*HV+head];
#pragma unroll
        for(int rt=0;rt<4;++rt) {
            float prediction[VC][4]={};
#pragma unroll
            for(int kt=0;kt<KT;++kt)if(kt*16<K) {
                unsigned a[4];
                load_matrix_a(a,matrix+(rt*16+(lane&15))*(MAX_K+8)+kt*16+(lane/16)*8);
#pragma unroll
                for(int n=0;n<VC;++n) {
                    if constexpr(MAX_K<=128)mma_bf16(prediction[n],a,state_b[kt][n]);
                    else {unsigned b[2];float_c_to_b(state[kt][n],b);mma_bf16(prediction[n],a,b);}
                }
            }
#pragma unroll
            for(int n=0;n<VC;++n) {
                float value[4];
#pragma unroll
                for(int j=0;j<4;++j) {
                    int token=base+rt*16+r+(j/2)*8,vd=vbase+n*8+t*2+j%2;
                    value[j]=token<T&&vd<V?__fsub_rn(f32(u[((batch*T+token)*HV+head)*V+vd]),prediction[n][j]):0.f;
                    if(token<T&&vd<V)vnew[((batch*T+token)*HV+head)*V+vd]=bf(value[j]);
                    float gate=token<T?e2(glast-g[(batch*T+token)*HV+head]):0.f;
                    value[j]=__fmul_rn(value[j],gate);
                }
                // Convert once per value tile, rather than once per K tile.
                c_to_b(pair_f32(value[0],value[1]),pair_f32(value[2],value[3]),values[rt][n]);
            }
        }
        __syncthreads();
        stage_state_matrix<MAX_K>(k,matrix,batch,kh,H,T,K,base);
        __syncthreads();
        float decay=e2(glast);
#pragma unroll
        for(int kt=0;kt<KT;++kt)if(kt*16<K) {
#pragma unroll
            for(int n=0;n<VC;++n) {
#pragma unroll
                for(int j=0;j<4;++j)state[kt][n][j]=__fmul_rn(state[kt][n][j],decay);
            }
#pragma unroll
            for(int rt=0;rt<4;++rt) {
                unsigned a[4];
                load_matrix_at(a,matrix+(rt*16+(lane&15))*(MAX_K+8)+kt*16+(lane/16)*8);
#pragma unroll
                for(int n=0;n<VC;++n)mma_bf16(state[kt][n],a,values[rt][n]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(int kt=0;kt<KT;++kt) {
#pragma unroll
        for(int n=0;n<VC;++n) {
#pragma unroll
            for(int j=0;j<4;j+=2) {
                int kd=kt*16+r+(j/2)*8,vd=vbase+n*8+t*2;
                if(final_state&&kd<K&&vd+1<V&&(V%2)==0)
                    *reinterpret_cast<float2*>(final_state+((batch*HV+head)*K+kd)*V+vd)=make_float2(state[kt][n][j],state[kt][n][j+1]);
                else {
                    if(final_state&&kd<K&&vd<V)final_state[((batch*HV+head)*K+kd)*V+vd]=state[kt][n][j];
                    if(final_state&&kd<K&&vd+1<V)final_state[((batch*HV+head)*K+kd)*V+vd+1]=state[kt][n][j+1];
                }
            }
        }
    }
}
extern "C" __global__ void gdn_fla_state_128(GDN_STATE_PARAMS) {state_register<128>(GDN_STATE_PASS);}
extern "C" __global__ void gdn_fla_state_256(GDN_STATE_PARAMS) {state_register<256>(GDN_STATE_PASS);}
extern "C" __global__ void gdn_fla_state_128_v16(GDN_STATE_PARAMS) {state_register<128,1>(GDN_STATE_PASS);}
extern "C" __global__ void gdn_fla_state_256_v16(GDN_STATE_PARAMS) {state_register<256,1>(GDN_STATE_PASS);}
#undef GDN_STATE_PASS
#undef GDN_STATE_PARAMS

extern "C" __global__ void gdn_fla_output_reg(const BF* __restrict__ q,const BF* __restrict__ k,const BF* __restrict__ v,const BF* __restrict__ snapshots,
                                             const float* __restrict__ g,BF* __restrict__ output,int T,int H,int HV,int K,int V,float scale) {
    int lane=threadIdx.x&31,warp=threadIdx.x/32,r=warp*16+lane/4,t=lane%4;
    int base=blockIdx.y*64,batch=blockIdx.z/HV,head=blockIdx.z%HV,kh=head/(HV/H),NT=(T+63)/64,vbase=blockIdx.x*32;
    // One cooperative load feeds all four row warps. Keep total shared
    // storage at 32 KiB; wider K retains the register/global-load path.
    __shared__ __align__(32) BF keys[64*136],values[64*40],history[128*40];
    const BF* hp=snapshots+(((batch*NT+blockIdx.y)*HV+head)*K)*V;
    if(K<=128) {
        for(int z=threadIdx.x*8;z<64*128;z+=blockDim.x*8) {
            int row=z/128,d=z%128,token=base+row;
            uint4 raw=make_uint4(0,0,0,0);
            if(token<T&&d<K)raw=*reinterpret_cast<const uint4*>(k+((batch*T+token)*H+kh)*K+d);
            *reinterpret_cast<uint4*>(keys+row*136+d)=raw;
        }
        for(int z=threadIdx.x*8;z<128*32;z+=blockDim.x*8) {
            int row=z/32,d=z%32,vd=vbase+d;
            uint4 raw=make_uint4(0,0,0,0);
            if(row<K&&vd+7<V&&(V%8)==0)raw=*reinterpret_cast<const uint4*>(hp+row*V+vd);
            if(V%8) {
#pragma unroll
                for(int j=0;j<8;++j)history[row*40+d+j]=row<K&&vd+j<V?hp[row*V+vd+j]:bf(0.f);
            } else *reinterpret_cast<uint4*>(history+row*40+d)=raw;
        }
    }
    for(int z=threadIdx.x*8;z<64*32;z+=blockDim.x*8) {
        int row=z/32,d=z%32,token=base+row,vd=vbase+d;
        uint4 raw=make_uint4(0,0,0,0);
        if(token<T&&vd+7<V&&(V%8)==0)raw=*reinterpret_cast<const uint4*>(v+((batch*T+token)*HV+head)*V+vd);
        if(V%8) {
#pragma unroll
            for(int j=0;j<8;++j)values[row*40+d+j]=token<T&&vd+j<V?v[((batch*T+token)*HV+head)*V+vd+j]:bf(0.f);
        } else *reinterpret_cast<uint4*>(values+row*40+d)=raw;
    }
    __syncthreads();
    float inter[4][4]={},score[8][4]={};
    for(int kk=0;kk<K;kk+=16) {
        unsigned a[4],b[2];int kt=kk+t*2;
        a[0]=base+r<T?load_pair(q+((batch*T+base+r)*H+kh)*K+kt):0;
        a[1]=base+r+8<T?load_pair(q+((batch*T+base+r+8)*H+kh)*K+kt):0;
        a[2]=base+r<T?load_pair(q+((batch*T+base+r)*H+kh)*K+kt+8):0;
        a[3]=base+r+8<T?load_pair(q+((batch*T+base+r+8)*H+kh)*K+kt+8):0;
#pragma unroll
        for(int n=0;n<4;++n) {
            int vd=vbase+n*8+lane/4;
            if(K<=128) {
                unsigned address=(unsigned)__cvta_generic_to_shared(history+(kk+(lane&15))*40+n*8);
                asm("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                    : "=r"(b[0]),"=r"(b[1]) : "r"(address));
            } else {
                b[0]=vd<V?pair(hp[kt*V+vd],hp[(kt+1)*V+vd]):0;
                b[1]=vd<V?pair(hp[(kt+8)*V+vd],hp[(kt+9)*V+vd]):0;
            }
            mma_bf16(inter[n],a,b);
        }
#pragma unroll
        for(int n=0;n<8;++n) {
            int token=base+n*8+lane/4;
            if(K<=128) {
                unsigned address=(unsigned)__cvta_generic_to_shared(keys+(n*8+(lane&7))*136+kk+((lane/8)&1)*8);
                asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                    : "=r"(b[0]),"=r"(b[1]) : "r"(address));
            } else {
                b[0]=token<T?load_pair(k+((batch*T+token)*H+kh)*K+kt):0;
                b[1]=token<T?load_pair(k+((batch*T+token)*H+kh)*K+kt+8):0;
            }
            mma_bf16(score[n],a,b);
        }
    }
    unsigned coefficient[4][4];
#pragma unroll
    for(int n=0;n<8;++n) {
        float gated[4];
#pragma unroll
        for(int j=0;j<4;++j) {
            int tr=base+r+(j/2)*8,tc=base+n*8+t*2+j%2;
            gated[j]=tr<T&&tc<=tr?__fmul_rn(score[n][j],e2(g[(batch*T+tr)*HV+head]-g[(batch*T+tc)*HV+head])):0.f;
        }
        coefficient[n/2][(n%2)*2]=pair_f32(gated[0],gated[1]);
        coefficient[n/2][(n%2)*2+1]=pair_f32(gated[2],gated[3]);
    }
    float local[4][4]={};
#pragma unroll
    for(int rt=0;rt<4;++rt) {
        unsigned b[2];
#pragma unroll
        for(int n=0;n<4;++n) {
            unsigned address=(unsigned)__cvta_generic_to_shared(values+(rt*16+(lane&15))*40+n*8);
            asm("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                : "=r"(b[0]),"=r"(b[1]) : "r"(address));
            mma_bf16(local[n],coefficient[rt],b);
        }
    }
#pragma unroll
    for(int n=0;n<4;++n) {
#pragma unroll
        for(int j=0;j<4;++j) {
            int token=base+r+(j/2)*8,vd=vbase+n*8+t*2+j%2;
            if(token<T&&vd<V) {
                // Preserve FLA's fused carry*scale + rounded(local*scale) ordering.
                float carry=__fmul_rn(inter[n][j],e2(g[(batch*T+token)*HV+head]));
                output[((batch*T+token)*HV+head)*V+vd]=bf(__fmaf_rn(carry,scale,__fmul_rn(local[n][j],scale)));
            }
        }
    }
}
