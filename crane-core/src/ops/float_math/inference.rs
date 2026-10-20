//! Contiguous inference primitives preserving separate F32 rounding boundaries.
use candle_core::{D, DType, Result, Tensor};
fn validate(xs: &[&Tensor]) -> Result<()> {
    if xs
        .iter()
        .any(|x| x.dtype() != DType::F32 || !x.device().same_device(xs[0].device()))
    {
        candle_core::bail!("F32 inference inputs must share a device and dtype");
    }
    Ok(())
}
pub fn add(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    validate(&[x, y])?;
    if x.dims() != y.dims() {
        candle_core::bail!("contiguous add shapes differ");
    }
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() {
        return Ok(launch("contiguous_add_f32", &[x, y], x.shape(), &[count(x)?], 1)?.remove(0));
    }
    x + y
}
/// Modulation is [B,2D] or [B,T,2D]. Output is (x * (1+scale)) + shift,
/// with separately rounded operations matching the unfused framework graph.
pub fn modulate(x: &Tensor, m: &Tensor, scale_first: bool) -> Result<Tensor> {
    validate(&[x, m])?;
    let (b, t, d) = x.dims3()?;
    if !((m.rank() == 2 && m.dims() == [b, 2 * d]) || (m.rank() == 3 && m.dims() == [b, t, 2 * d]))
    {
        candle_core::bail!("adaptive modulation requires [B,2D] or [B,T,2D]");
    }
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() {
        let args = [
            count(x)?,
            u32::try_from(d).map_err(candle_core::Error::wrap)?,
            u32::try_from(t).map_err(candle_core::Error::wrap)?,
            u32::from(m.rank() == 3),
            u32::from(scale_first),
        ];
        return Ok(launch("adaptive_modulate_f32", &[x, m], x.shape(), &args, 1)?.remove(0));
    }
    let scale = (m.narrow(D::Minus1, if scale_first { 0 } else { d }, d)? + 1.)?;
    let shift = m.narrow(D::Minus1, if scale_first { d } else { 0 }, d)?;
    let (scale, shift) = if m.rank() == 2 {
        (scale.unsqueeze(1)?, shift.unsqueeze(1)?)
    } else {
        (scale, shift)
    };
    x.broadcast_mul(&scale)?.broadcast_add(&shift)
}
/// Multiply K/V [B,S,H*D] by paired per-channel gates [B,2H*D], and
/// directly write the contiguous [B,H,S,D] attention layout.
pub fn gated_kv_heads(
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    heads: usize,
) -> Result<(Tensor, Tensor)> {
    validate(&[k, v, g])?;
    let (b, s, w) = k.dims3()?;
    if heads == 0 || w % heads != 0 || k.dims() != v.dims() || g.dims() != [b, 2 * w] {
        candle_core::bail!("gated KV requires matching [B,S,H*D] and [B,2H*D]");
    }
    let d = w / heads;
    #[cfg(feature = "cuda")]
    if k.device().is_cuda() {
        let shape = candle_core::Shape::from((b, heads, s, d));
        let args = [
            count(k)?,
            u32::try_from(s).map_err(candle_core::Error::wrap)?,
            u32::try_from(heads).map_err(candle_core::Error::wrap)?,
            u32::try_from(d).map_err(candle_core::Error::wrap)?,
        ];
        let mut outputs = launch("gated_kv_heads_f32", &[k, v, g], &shape, &args, 2)?;
        let v = outputs.pop().unwrap();
        let k = outputs.pop().unwrap();
        return Ok((k, v));
    }
    let cvt = |x: &Tensor, offset| {
        x.broadcast_mul(&g.narrow(1, offset, w)?.unsqueeze(1)?)?
            .reshape((b, s, heads, d))?
            .transpose(1, 2)?
            .contiguous()
    };
    Ok((cvt(k, 0)?, cvt(v, w)?))
}
#[cfg(feature = "cuda")]
fn count(x: &Tensor) -> Result<u32> {
    u32::try_from(x.elem_count()).map_err(candle_core::Error::wrap)
}
#[cfg(feature = "cuda")]
fn launch(
    name: &str,
    inputs: &[&Tensor],
    shape: &candle_core::Shape,
    args: &[u32],
    outputs: usize,
) -> Result<Vec<Tensor>> {
    use candle_core::cuda_backend::{
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    };
    use candle_core::{CudaStorage, Storage, op::BackpropOp};
    let inputs = inputs
        .iter()
        .map(|x| x.contiguous())
        .collect::<Result<Vec<_>>>()?;
    let dev = inputs[0].device().as_cuda_device()?.clone();
    let guards = inputs
        .iter()
        .map(Tensor::storage_and_layout)
        .collect::<Vec<_>>();
    let mut views = Vec::new();
    for (storage, layout) in &guards {
        let Storage::Cuda(storage) = &**storage else {
            candle_core::bail!("CUDA inputs required");
        };
        views.push(
            storage
                .as_cuda_slice::<f32>()?
                .slice(layout.start_offset()..),
        );
    }
    let mut buffers = (0..outputs)
        .map(|_| unsafe { dev.alloc::<f32>(shape.elem_count()) })
        .collect::<Result<Vec<_>>>()?;
    if args[0] > 0 {
        let function =
            dev.get_or_load_custom_func(name, "crane_float_math", super::ptx::FLOAT_MATH)?;
        let mut builder = function.builder();
        for view in &views {
            builder.arg(view);
        }
        for buffer in &mut buffers {
            builder.arg(buffer);
        }
        for arg in args {
            builder.arg(arg);
        }
        // Shapes, dtype and extents are validated; storage guards live through launch.
        unsafe { builder.launch(LaunchConfig::for_num_elems(args[0])) }.w()?;
    }
    Ok(buffers
        .into_iter()
        .map(|out| {
            Tensor::from_storage(
                Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
                shape.clone(),
                BackpropOp::none(),
                false,
            )
        })
        .collect())
}

pub fn split_heads(x: &Tensor, heads: usize) -> Result<Tensor> {
    validate(&[x])?;
    let (b, t, w) = x.dims3()?;
    if heads == 0 || w % heads != 0 {
        candle_core::bail!("head count must divide width");
    }
    let d = w / heads;
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() {
        let shape = candle_core::Shape::from((b, heads, t, d));
        let args = [
            count(x)?,
            u32::try_from(t).map_err(candle_core::Error::wrap)?,
            u32::try_from(heads).map_err(candle_core::Error::wrap)?,
            u32::try_from(d).map_err(candle_core::Error::wrap)?,
            0,
        ];
        return Ok(launch("transpose_heads_f32", &[x], &shape, &args, 1)?.remove(0));
    }
    x.reshape((b, t, heads, d))?.transpose(1, 2)?.contiguous()
}
pub fn merge_heads(x: &Tensor) -> Result<Tensor> {
    validate(&[x])?;
    let (b, h, t, d) = x.dims4()?;
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() {
        let shape = candle_core::Shape::from((b, t, h * d));
        let args = [
            count(x)?,
            u32::try_from(t).map_err(candle_core::Error::wrap)?,
            u32::try_from(h).map_err(candle_core::Error::wrap)?,
            u32::try_from(d).map_err(candle_core::Error::wrap)?,
            1,
        ];
        return Ok(launch("transpose_heads_f32", &[x], &shape, &args, 1)?.remove(0));
    }
    x.transpose(1, 2)?.contiguous()?.reshape((b, t, h * d))
}

#[cfg(feature = "cuda")]
pub fn add_selected_expert(
    base: &Tensor,
    value: &Tensor,
    gates: &Tensor,
    selected: &Tensor,
    expert: usize,
) -> Result<Tensor> {
    use candle_core::cuda_backend::{
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    };
    use candle_core::{CudaStorage, Storage, op::BackpropOp};
    validate(&[base, value, gates])?;
    let (rows, width) = base.dims2()?;
    let (gr, experts) = gates.dims2()?;
    if base.dims() != value.dims()
        || gr != rows
        || selected.dims() != [rows]
        || selected.dtype() != DType::U32
        || !base.device().same_device(selected.device())
        || expert >= experts
    {
        candle_core::bail!("invalid selected-expert combination shapes/dtypes");
    }
    let inputs = [base.contiguous()?, value.contiguous()?, gates.contiguous()?];
    let selected = selected.contiguous()?;
    let dev = base.device().as_cuda_device()?.clone();
    let guards = inputs
        .iter()
        .map(Tensor::storage_and_layout)
        .collect::<Vec<_>>();
    let mut views = Vec::new();
    for (storage, layout) in &guards {
        let Storage::Cuda(storage) = &**storage else {
            candle_core::bail!("CUDA required");
        };
        views.push(
            storage
                .as_cuda_slice::<f32>()?
                .slice(layout.start_offset()..),
        );
    }
    let (ss, sl) = selected.storage_and_layout();
    let Storage::Cuda(ss) = &*ss else {
        candle_core::bail!("CUDA routes required");
    };
    let sv = ss.as_cuda_slice::<u32>()?.slice(sl.start_offset()..);
    let args = [
        count(base)?,
        u32::try_from(width).map_err(candle_core::Error::wrap)?,
        u32::try_from(experts).map_err(candle_core::Error::wrap)?,
        expert as u32,
    ];
    let mut out = unsafe { dev.alloc::<f32>(base.elem_count()) }?;
    if args[0] > 0 {
        let f = dev.get_or_load_custom_func(
            "moe_selected_add_f32",
            "crane_float_math",
            super::ptx::FLOAT_MATH,
        )?;
        let mut b = f.builder();
        for v in &views {
            b.arg(v);
        }
        b.arg(&sv).arg(&mut out);
        for a in &args {
            b.arg(a);
        }
        unsafe { b.launch(LaunchConfig::for_num_elems(args[0])) }.w()?;
    }
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)),
        base.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}
