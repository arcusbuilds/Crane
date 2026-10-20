//! Float32 transcendental helpers with explicit rounding boundaries.
use candle_core::{DType, Result, Tensor};
pub mod inference;
mod scan;
pub use scan::cumsum_last_dim;
#[cfg(feature = "cuda")]
#[allow(dead_code)]
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
}

/// `base ** exponent`, using powf directly rather than exp(log(base)*x).
/// The distinction matters when the result is subsequently rounded to BF16.
pub fn scalar_power(base: f32, exponent: &Tensor) -> Result<Tensor> {
    if !base.is_finite() || base <= 0. {
        candle_core::bail!("scalar_power requires a positive finite base");
    }
    if exponent.dtype() != DType::F32 {
        candle_core::bail!("scalar_power requires F32 exponents");
    }
    #[cfg(feature = "cuda")]
    if exponent.device().is_cuda() {
        return cuda_power(base, exponent);
    }
    if exponent.device().is_cpu() {
        let values = exponent
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|x| base.powf(x))
            .collect::<Vec<_>>();
        return Tensor::from_vec(values, exponent.shape().clone(), exponent.device());
    }
    exponent.affine((base as f64).ln(), 0.)?.exp()
}
#[cfg(feature = "cuda")]
fn cuda_power(base: f32, exponent: &Tensor) -> Result<Tensor> {
    use candle_core::cuda_backend::{
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    };
    use candle_core::{CudaStorage, Storage, op::BackpropOp};
    let exponent = exponent.contiguous()?;
    let dev = exponent.device().as_cuda_device()?.clone();
    let (storage, layout) = exponent.storage_and_layout();
    let slice = match &*storage {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => unreachable!(),
    };
    let input = slice.slice(layout.start_offset()..);
    let n = u32::try_from(exponent.elem_count()).map_err(candle_core::Error::wrap)?;
    if n == 0 {
        return Ok(exponent.clone());
    }
    let out = unsafe { dev.alloc::<f32>(n as usize) }?;
    let function =
        dev.get_or_load_custom_func("scalar_power_f32", "crane_float_math", ptx::FLOAT_MATH)?;
    let mut builder = function.builder();
    builder.arg(&input);
    builder.arg(&out);
    builder.arg(&n);
    builder.arg(&base);
    // SAFETY: input/output hold n contiguous F32 elements on this stream.
    unsafe { builder.launch(LaunchConfig::for_num_elems(n)) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)),
        exponent.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    fn check(device: &Device) -> Result<()> {
        let x = Tensor::new(&[[0f32, 1.], [0.5, -1.]], device)?;
        assert_eq!(
            scalar_power(4., &x)?.to_vec2::<f32>()?,
            vec![vec![1., 4.], vec![2., 0.25]]
        );
        assert!(scalar_power(-4., &x).is_err());
        Ok(())
    }
    #[test]
    fn cpu_known_powers() -> Result<()> {
        check(&Device::Cpu)
    }
    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_known_powers() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        check(&device)
    }
}

/// F32 RMS normalization with explicit square/reduction/multiply rounding.
/// A missing alpha requests unit-weight normalization.
pub fn rms_norm(x: &Tensor, alpha: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    let width = x.dim(candle_core::D::Minus1)?;
    if x.dtype() != DType::F32
        || alpha.is_some_and(|a| {
            a.dtype() != DType::F32
                || a.dims() != [width]
                || a.device().location() != x.device().location()
        })
    {
        candle_core::bail!("ordered RMS norm requires F32 inputs and a width-sized F32 alpha");
    }
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() && width >= 128 && width % 128 == 0 {
        return cuda_rms_norm(x, alpha, eps);
    }
    let alpha = match alpha {
        Some(a) => a.clone(),
        None => Tensor::ones(width, DType::F32, x.device())?,
    };
    candle_nn::ops::rms_norm(&x.contiguous()?, &alpha, eps)
}
#[cfg(feature = "cuda")]
fn cuda_rms_norm(x: &Tensor, alpha: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    use candle_core::cuda_backend::{
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    };
    use candle_core::{CudaStorage, Storage, op::BackpropOp};
    let x = x.contiguous()?;
    let width = u32::try_from(x.dim(candle_core::D::Minus1)?).map_err(candle_core::Error::wrap)?;
    let _count = u32::try_from(x.elem_count()).map_err(candle_core::Error::wrap)?;
    let rows = u32::try_from(x.elem_count() / width as usize).map_err(candle_core::Error::wrap)?;
    if rows == 0 {
        return Ok(x.clone());
    }
    let dev = x.device().as_cuda_device()?.clone();
    let (storage, layout) = x.storage_and_layout();
    let slice = match &*storage {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => unreachable!(),
    };
    let input = slice.slice(layout.start_offset()..);
    let alpha = alpha.map(Tensor::contiguous).transpose()?;
    let a_storage = alpha.as_ref().map(Tensor::storage_and_layout);
    let a_view = match &a_storage {
        Some((storage, layout)) => match &**storage {
            Storage::Cuda(c) => Some(c.as_cuda_slice::<f32>()?.slice(layout.start_offset()..)),
            _ => candle_core::bail!("ordered RMS norm alpha must be on CUDA"),
        },
        None => None,
    };
    let out = unsafe { dev.alloc::<f32>(x.elem_count()) }?;
    let function =
        dev.get_or_load_custom_func("rms_norm_f32_ordered", "crane_float_math", ptx::FLOAT_MATH)?;
    let mut builder = function.builder();
    builder.arg(&input);
    let null = 0u64;
    match &a_view {
        Some(view) => {
            builder.arg(view);
        },
        None => {
            builder.arg(&null);
        },
    }
    builder.arg(&out);
    builder.arg(&rows);
    builder.arg(&width);
    builder.arg(&eps);
    // SAFETY: one warp owns each contiguous row; all tensor guards remain
    // alive through launch and alpha is either null or width F32 values.
    // A warp owns one row. Small text prefills need enough blocks to fill
    // the GPU rather than grouping 32 rows into each 1024-thread block.
    let cfg = if width == 2048 && rows <= 256 {
        LaunchConfig {
            grid_dim: (rows, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        LaunchConfig::for_num_elems(rows * 32)
    };
    unsafe { builder.launch(cfg) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)),
        x.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

#[path = "float_math/layer_norm.rs"]
pub mod layer_norm;

/// Ordered F32 RMS arithmetic with low-precision input/output and an F32
/// scale. This is exactly the cast -> F32 normalize/scale -> cast graph.
pub fn rms_norm_low_precision(x: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor> {
    let width = x.dim(candle_core::D::Minus1)?;
    if !matches!(x.dtype(), DType::BF16 | DType::F16)
        || alpha.dtype() != DType::F32
        || alpha.dims() != [width]
        || !alpha.device().same_device(x.device())
    {
        candle_core::bail!(
            "low-precision ordered RMS requires BF16/F16 input and an F32 width-sized scale"
        );
    }
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() && width > 0 && width % 128 == 0 {
        return match x.dtype() {
            DType::BF16 => cuda_rms_low::<half::bf16>(x, alpha, eps, "rms_norm_bf16_ordered"),
            DType::F16 => cuda_rms_low::<half::f16>(x, alpha, eps, "rms_norm_f16_ordered"),
            _ => unreachable!(),
        };
    }
    rms_norm(&x.to_dtype(DType::F32)?, Some(alpha), eps)?.to_dtype(x.dtype())
}
#[cfg(feature = "cuda")]
fn cuda_rms_low<
    T: candle_core::cuda_backend::CudaDType
        + candle_core::WithDType
        + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
>(
    x: &Tensor,
    alpha: &Tensor,
    eps: f32,
    name: &str,
) -> Result<Tensor> {
    use candle_core::cuda_backend::{
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    };
    use candle_core::{CudaStorage, Storage, op::BackpropOp};
    let x = x.contiguous()?;
    let alpha = alpha.contiguous()?;
    let width = u32::try_from(x.dim(candle_core::D::Minus1)?).map_err(candle_core::Error::wrap)?;
    let count = u32::try_from(x.elem_count()).map_err(candle_core::Error::wrap)?;
    let rows = count / width;
    if rows == 0 {
        return Ok(x);
    }
    let dev = x.device().as_cuda_device()?.clone();
    let (xs, xl) = x.storage_and_layout();
    let (as_, al) = alpha.storage_and_layout();
    let Storage::Cuda(xs) = &*xs else {
        unreachable!()
    };
    let Storage::Cuda(as_) = &*as_ else {
        unreachable!()
    };
    let xv = xs.as_cuda_slice::<T>()?.slice(xl.start_offset()..);
    let av = as_.as_cuda_slice::<f32>()?.slice(al.start_offset()..);
    let mut out = unsafe { dev.alloc::<T>(count as usize) }?;
    let fun = dev.get_or_load_custom_func(name, "crane_float_math", ptx::FLOAT_MATH)?;
    let mut b = fun.builder();
    b.arg(&xv)
        .arg(&av)
        .arg(&mut out)
        .arg(&rows)
        .arg(&width)
        .arg(&eps);
    let cfg = if width == 2048 && rows <= 256 {
        LaunchConfig {
            grid_dim: (rows, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        LaunchConfig::for_num_elems(rows * 32)
    };
    unsafe { b.launch(cfg) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)),
        x.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

pub fn swiglu_low_precision(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if !matches!(gate.dtype(), DType::BF16 | DType::F16)
        || up.dtype() != gate.dtype()
        || up.dims() != gate.dims()
        || !up.device().same_device(gate.device())
    {
        candle_core::bail!("low-precision SwiGLU requires matching BF16/F16 tensors");
    }
    #[cfg(feature = "cuda")]
    if gate.device().is_cuda() {
        return match gate.dtype() {
            DType::BF16 => cuda_swiglu_low::<half::bf16>(gate, up, "swiglu_bf16_ordered"),
            DType::F16 => cuda_swiglu_low::<half::f16>(gate, up, "swiglu_f16_ordered"),
            _ => unreachable!(),
        };
    }
    gate.to_dtype(DType::F32)?
        .silu()?
        .to_dtype(gate.dtype())?
        .mul(up)
}
#[cfg(feature = "cuda")]
fn cuda_swiglu_low<
    T: candle_core::cuda_backend::CudaDType
        + candle_core::WithDType
        + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
>(
    gate: &Tensor,
    up: &Tensor,
    name: &str,
) -> Result<Tensor> {
    use candle_core::cuda_backend::{
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    };
    use candle_core::{CudaStorage, Storage, op::BackpropOp};
    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let n = u32::try_from(gate.elem_count()).map_err(candle_core::Error::wrap)?;
    if n == 0 {
        return Ok(gate);
    }
    let dev = gate.device().as_cuda_device()?.clone();
    let (gs, gl) = gate.storage_and_layout();
    let (us, ul) = up.storage_and_layout();
    let Storage::Cuda(gs) = &*gs else {
        unreachable!()
    };
    let Storage::Cuda(us) = &*us else {
        unreachable!()
    };
    let gv = gs.as_cuda_slice::<T>()?.slice(gl.start_offset()..);
    let uv = us.as_cuda_slice::<T>()?.slice(ul.start_offset()..);
    let mut out = unsafe { dev.alloc::<T>(n as usize) }?;
    let f = dev.get_or_load_custom_func(name, "crane_float_math", ptx::FLOAT_MATH)?;
    let mut b = f.builder();
    b.arg(&gv).arg(&uv).arg(&mut out).arg(&n);
    unsafe { b.launch(LaunchConfig::for_num_elems(n)) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)),
        gate.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}
