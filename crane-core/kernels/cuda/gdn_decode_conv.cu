// Single-token GDN convolution + SiLU, preserving the reference's BF16 rounding.

#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ void gdn_decode_conv_silu_bf16(
    const __nv_bfloat16 *input, // [channels]: new token's values
    const __nv_bfloat16 *state, // [channels, 4]: cached history
    const float *weights,       // [channels, 4]: preconverted weights
    __nv_bfloat16 *output,      // [channels]: convolution + SiLU
    __nv_bfloat16 *next_state,  // [channels, 4]: updated history
    int channels)
{
    // Give each GPU thread one channel.
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels)
    {
        return;
    }

    // Drop the oldest cached value and append the new token.
    const int base = channel * 4;
    const float x0 = __bfloat162float(state[base + 1]);
    const float x1 = __bfloat162float(state[base + 2]);
    const float x2 = __bfloat162float(state[base + 3]);
    const float x3 = __bfloat162float(input[channel]);

    const float p0 = __fmul_rn(x0, weights[base + 0]);
    const float p1 = __fmul_rn(x1, weights[base + 1]);
    const float p2 = __fmul_rn(x2, weights[base + 2]);
    const float p3 = __fmul_rn(x3, weights[base + 3]);

    const float sum02 = __fadd_rn(p0, p2);
    const float sum13 = __fadd_rn(p1, p3);

    const float sum = __fadd_rn(sum02, sum13);

    // Match the BF16 convolution output consumed by the FP32 activation.
    const float value = __bfloat162float(__float2bfloat16_rn(sum));
    const float denom = 1 + expf(-value);

    const float silu = __fdiv_rn(value, denom);

    output[channel] = __float2bfloat16_rn(silu);

    // Carry the window forward without changing its BF16 representation.
    next_state[base + 0] = state[base + 1];
    next_state[base + 1] = state[base + 2];
    next_state[base + 2] = state[base + 3];
    next_state[base + 3] = input[channel];
}
