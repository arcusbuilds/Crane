//! Scalar-gate WY chunk recurrence, following FLA's mixed-precision graph.
//! CUDA source is built and embedded by Crane's normal kernel build pipeline.
//! Batch, sequence, heads, GQA and value width are runtime parameters.
use candle_core::cuda_backend::{
    WrapErr,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::op::BackpropOp;
use candle_core::{CudaStorage, DType, Result, Storage, Tensor};
use half::bf16;
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
}

macro_rules! view {
    ($v:ident,$guard:ident,$x:expr,$ty:ty) => {
        let ($guard, layout) = $x.storage_and_layout();
        let $v = match &*$guard {
            Storage::Cuda(s) => s.as_cuda_slice::<$ty>()?.slice(layout.start_offset()..),
            _ => candle_core::bail!("FLA chunk recurrence requires CUDA"),
        };
    };
}

/// Q/K: BF16 `[B,T,H,K]`, V: BF16 `[B,T,HV,V]`, scalar gates `[B,T,HV]`.
/// G is F32, beta BF16, initial/final recurrent state F32 `[B,HV,K,V]`.
/// Q/K enter unnormalized. Returns BF16 output and a fresh F32 final state.
/// No model identity, exported plan, Python runtime or fixed token count.
pub fn forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    initial: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (batch, time, heads, kdim) = k.dims4()?;
    let (vb, vt, vheads, vdim) = v.dims4()?;
    if q.dims() != k.dims()
        || vb != batch
        || vt != time
        || heads == 0
        || vheads == 0
        || vheads % heads != 0
        || time == 0
        || batch == 0
        || !matches!(kdim, 64 | 128 | 256)
        || vdim == 0
        || g.dims() != [batch, time, vheads]
        || beta.dims() != g.dims()
        || initial.dims() != [batch, vheads, kdim, vdim]
    {
        candle_core::bail!(
            "unsupported/inconsistent FLA chunk dimensions: Q={:?}, K={:?}, V={:?}, G={:?}, beta={:?}, state={:?}",
            q.dims(),
            k.dims(),
            v.dims(),
            g.dims(),
            beta.dims(),
            initial.dims()
        );
    }
    for (x, dtype) in [
        (q, DType::BF16),
        (k, DType::BF16),
        (v, DType::BF16),
        (beta, DType::BF16),
        (g, DType::F32),
        (initial, DType::F32),
    ] {
        if x.dtype() != dtype || !x.device().same_device(q.device()) {
            candle_core::bail!("FLA chunk tensor dtype/device mismatch");
        }
    }
    if [batch, time, heads, vheads, kdim, vdim]
        .iter()
        .any(|&d| d > i32::MAX as usize)
        || batch
            .checked_mul(time)
            .and_then(|n| n.checked_mul(vheads))
            .and_then(|n| n.checked_mul(kdim.max(vdim)))
            .is_none_or(|n| n > i32::MAX as usize)
    {
        candle_core::bail!("FLA chunk dimensions exceed CUDA index range");
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let initial = initial.contiguous()?;
    let dev = q.device().as_cuda_device()?.clone();
    use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute_enum as Attr;
    if dev
        .cuda_stream()
        .context()
        .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .w()?
        < 8
    {
        candle_core::bail!("FLA BF16 Tensor Cores require CUDA compute capability >= 8.0");
    }
    view!(qv, qg, q, bf16);
    view!(kv, kg, k, bf16);
    view!(vv, vg, v, bf16);
    view!(gv, gg, g, f32);
    view!(bv, bg, beta, bf16);
    view!(sv, sg, initial, f32);
    let chunks = time.div_ceil(64);
    let mut qn = unsafe { dev.alloc::<bf16>(q.elem_count()) }?;
    let mut kn = unsafe { dev.alloc::<bf16>(k.elem_count()) }?;
    let mut gc = unsafe { dev.alloc::<f32>(g.elem_count()) }?;
    let mut inverse = unsafe { dev.alloc::<bf16>(batch * time * vheads * 64) }?;
    let mut w = unsafe { dev.alloc::<bf16>(batch * time * vheads * kdim) }?;
    let mut u = unsafe { dev.alloc::<bf16>(v.elem_count()) }?;
    let mut snapshots = unsafe { dev.alloc::<bf16>(batch * chunks * vheads * kdim * vdim) }?;
    let mut new_values = unsafe { dev.alloc::<bf16>(v.elem_count()) }?;
    let mut final_state = unsafe { dev.alloc::<f32>(initial.elem_count()) }?;
    let mut output = unsafe { dev.alloc::<bf16>(v.elem_count()) }?;
    let rows = (batch * time * heads) as i32;
    let (t, h, hv, kd, vd) = (
        time as i32,
        heads as i32,
        vheads as i32,
        kdim as i32,
        vdim as i32,
    );
    let null = 0u64;
    let scale = (1.0 / (kdim as f64).sqrt()) as f32;
    macro_rules! launch {
        ($name:expr,$grid:expr,$threads:expr,$($arg:expr),*) => {{
            let fun=dev.get_or_load_custom_func($name,"crane_gdn_chunk_fla",ptx::GDN_CHUNK_FLA)?;
            let mut builder=fun.builder();$(builder.arg($arg);)*
            // SAFETY: validated shapes/dtypes and buffer extents match this kernel;
            // all storage guards and intermediate allocations live through launch.
            unsafe{builder.launch(LaunchConfig{grid_dim:$grid,block_dim:($threads,1,1),shared_mem_bytes:0})}.w()?;
        }};
    }
    let norm_grid = ((rows as u32 * (kdim as u32 / 8)).div_ceil(128), 2, 1);
    launch!(
        "gdn_fla_l2_pair",
        norm_grid,
        128,
        &qv,
        &kv,
        &mut qn,
        &mut kn,
        &rows,
        &kd
    );
    let chunk_grid = (chunks as u32, (batch * vheads) as u32, 1);
    launch!(
        "gdn_fla_intra_scan",
        chunk_grid,
        128,
        &kn,
        &gv,
        &bv,
        &mut inverse,
        &mut gc,
        &t,
        &h,
        &hv,
        &kd,
        &null,
        &null,
        &null
    );
    launch!(
        "gdn_fla_wu_reg",
        (
            (kdim.div_ceil(32) + vdim.div_ceil(32)) as u32,
            chunks as u32,
            (batch * vheads) as u32
        ),
        128,
        &kn,
        &vv,
        &bv,
        &inverse,
        &gc,
        &mut w,
        &mut u,
        &t,
        &h,
        &hv,
        &kd,
        &vd
    );
    let state_kernel = if kdim <= 128 {
        "gdn_fla_state_128_v16"
    } else {
        "gdn_fla_state_256_v16"
    };
    launch!(
        state_kernel,
        (vdim.div_ceil(16) as u32, (batch * vheads) as u32, 1),
        64,
        &kn,
        &w,
        &u,
        &gc,
        &sv,
        &mut snapshots,
        &mut new_values,
        &mut final_state,
        &t,
        &h,
        &hv,
        &kd,
        &vd
    );
    launch!(
        "gdn_fla_output_reg",
        (
            vdim.div_ceil(32) as u32,
            chunks as u32,
            (batch * vheads) as u32
        ),
        128,
        &qn,
        &kn,
        &new_values,
        &snapshots,
        &gc,
        &mut output,
        &t,
        &h,
        &hv,
        &kd,
        &vd,
        &scale
    );
    let result = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
        (batch, time, vheads, vdim),
        BackpropOp::none(),
        false,
    );
    let state = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(final_state, dev)),
        (batch, vheads, kdim, vdim),
        BackpropOp::none(),
        false,
    );
    Ok((result, state))
}
