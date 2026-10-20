//! Explicit split-KV inference using unmodified dependency CUDA templates.
use candle_core::{DType, Result, Tensor};
use std::ffi::c_void;
unsafe extern "C" {
    fn crane_split_kv_bf16_256(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        lse: *mut c_void,
        output_accum: *mut c_void,
        lse_accum: *mut c_void,
        batch: i32,
        sequence: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        stream: *mut c_void,
    ) -> i32;
    fn crane_noncausal_bf16_64(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        lse: *mut c_void,
        output_accum: *mut c_void,
        lse_accum: *mut c_void,
        batch: i32,
        sequence: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        stream: *mut c_void,
    ) -> i32;
}
/// CUDA causal attention, BS H D layout, BF16, D=256, equal Q/K lengths.
/// Intermediate outputs and log-sum-exp stay F32 until partition combination.
pub fn causal_bf16_256(q: &Tensor, k: &Tensor, v: &Tensor, splits: usize) -> Result<Tensor> {
    run(q, k, v, splits, 256, true)
}
/// BF16 vision attention using the same noncausal 64-wide tile and arithmetic policy.
pub fn noncausal_bf16_64(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    run(q, k, v, 1, 64, false)
}
fn run(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    splits: usize,
    expected_width: usize,
    causal: bool,
) -> Result<Tensor> {
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::{CudaStorage, Storage, op::BackpropOp};
    let (batch, sequence, heads, width) = q.dims4()?;
    let (kb, ks, kh, kw) = k.dims4()?;
    if !q.device().is_cuda()
        || [q, k, v].iter().any(|x| x.dtype() != DType::BF16)
        || !q.device().same_device(k.device())
        || !q.device().same_device(v.device())
        || width != expected_width
        || (kb, ks, kw) != (batch, sequence, width)
        || k.dims() != v.dims()
        || kh == 0
        || heads % kh != 0
        || batch == 0
        || sequence == 0
        || heads == 0
        || !(1..=128).contains(&splits)
    {
        candle_core::bail!(
            "split-KV requires CUDA BF16, matching BS, D=256, divisible GQA heads, 2..=128 partitions"
        );
    }
    let count = q
        .elem_count()
        .checked_mul(splits)
        .ok_or_else(|| candle_core::Error::Msg("split-KV allocation overflow".into()))?;
    let _ = u32::try_from(count).map_err(candle_core::Error::wrap)?;
    let dims = [batch, sequence, heads, kh, splits].map(i32::try_from);
    let dims = dims
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(candle_core::Error::wrap)?;
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let dev = q.device().as_cuda_device()?.clone();
    let stream = dev.cuda_stream();
    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = k.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let slice = |s: &Storage| match s {
        Storage::Cuda(_) => Ok(()),
        _ => Err(candle_core::Error::Msg("CUDA storage required".into())),
    };
    slice(&qs)?;
    slice(&ks)?;
    slice(&vs)?;
    let Storage::Cuda(qs) = &*qs else {
        unreachable!()
    };
    let Storage::Cuda(ks) = &*ks else {
        unreachable!()
    };
    let Storage::Cuda(vs) = &*vs else {
        unreachable!()
    };
    let qv = qs.as_cuda_slice::<half::bf16>()?.slice(ql.start_offset()..);
    let kv = ks.as_cuda_slice::<half::bf16>()?.slice(kl.start_offset()..);
    let vv = vs.as_cuda_slice::<half::bf16>()?.slice(vl.start_offset()..);
    let mut out = unsafe { dev.alloc::<half::bf16>(q.elem_count()) }?;
    let lse_count = batch * heads * sequence;
    let mut lse = unsafe { dev.alloc::<f32>(lse_count) }?;
    let mut accum = unsafe { dev.alloc::<f32>(count) }?;
    let mut lse_accum = unsafe { dev.alloc::<f32>(lse_count * splits) }?;
    {
        let (qp, _q_guard) = qv.device_ptr(&stream);
        let (kp, _k_guard) = kv.device_ptr(&stream);
        let (vp, _v_guard) = vv.device_ptr(&stream);
        let (op, _o_guard) = out.device_ptr_mut(&stream);
        let (lp, _l_guard) = lse.device_ptr_mut(&stream);
        let (ap, _a_guard) = accum.device_ptr_mut(&stream);
        let (sp, _s_guard) = lse_accum.device_ptr_mut(&stream);
        // SAFETY: validated dimensions, contiguous device buffers and all stream guards stay live through launch.
        let status = unsafe {
            if causal {
                crane_split_kv_bf16_256(
                    qp as _,
                    kp as _,
                    vp as _,
                    op as _,
                    lp as _,
                    ap as _,
                    sp as _,
                    dims[0],
                    dims[1],
                    dims[2],
                    dims[3],
                    dims[4],
                    stream.cu_stream() as _,
                )
            } else {
                crane_noncausal_bf16_64(
                    qp as _,
                    kp as _,
                    vp as _,
                    op as _,
                    lp as _,
                    ap as _,
                    sp as _,
                    dims[0],
                    dims[1],
                    dims[2],
                    dims[3],
                    dims[4],
                    stream.cu_stream() as _,
                )
            }
        };
        if status != 0 {
            candle_core::bail!("split-KV CUDA launch failed: error {status}");
        }
    }
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)),
        q.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

/// Occupancy-based partition selection (the FA2 wave-efficiency policy).
/// Reference: Dao-AILab/flash-attention v2.6.3, flash_api.cpp.
fn recommended_parts(work: usize, sm_slots: usize, kv_blocks: usize) -> usize {
    if sm_slots == 0 || kv_blocks == 0 || work as f32 >= 0.8f32 * sm_slots as f32 {
        return 1;
    }
    let limit = 128usize.min(sm_slots).min(kv_blocks);
    let mut efficiency = vec![0f32; limit];
    let mut best = 0f32;
    for parts in 1..=limit {
        if parts > 1 && kv_blocks.div_ceil(parts) == kv_blocks.div_ceil(parts - 1) {
            continue;
        }
        let waves = (work * parts) as f32 / sm_slots as f32;
        let value = waves / waves.ceil();
        efficiency[parts - 1] = value;
        best = best.max(value);
    }
    efficiency
        .iter()
        .position(|&value| value as f64 >= 0.85f64 * best as f64)
        .map_or(1, |i| i + 1)
}
/// Choose the original wave-efficiency partition count from the current CUDA device.
pub fn causal_bf16_256_auto(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute;
    let (batch, sequence, heads, _) = q.dims4()?;
    let stream = q.device().as_cuda_device()?.cuda_stream();
    let sms = stream
        .context()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(candle_core::Error::wrap)? as usize;
    let parts = recommended_parts(
        batch * heads * sequence.div_ceil(64),
        sms * 2,
        k.dim(1)?.div_ceil(64),
    );
    causal_bf16_256(q, k, v, parts)
}

/// Inference kernels required by the native XFuse backbone. Dependency headers
/// are read-only; the full feature additionally supports other head shapes.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> Result<Tensor> {
    if q.dtype() == DType::BF16 && q.dim(3)? == 64 && !causal {
        return noncausal_bf16_64(q, k, v);
    }
    if q.dtype() == DType::BF16 && q.dim(3)? == 256 && causal && q.dim(1)? == k.dim(1)? {
        return causal_bf16_256_auto(q, k, v);
    }
    #[cfg(feature = "flash-attn")]
    {
        candle_flash_attn::flash_attn(q, k, v, scale, causal)
    }
    #[cfg(not(feature = "flash-attn"))]
    {
        let _ = scale;
        candle_core::bail!(
            "flash-inference supports BF16 head 64 noncausal or head 256 causal; enable flash-attn for other shapes"
        )
    }
}
