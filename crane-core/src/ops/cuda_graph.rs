//! Replay fixed-shape CUDA inference with explicitly owned mutable inputs.
//! No exported graph/kernel assets or Python runtime are involved.
use candle_core::cuda_backend::{
    WrapErr,
    cudarc::driver::{
        CudaGraph, CudaStream,
        sys::{CUgraphInstantiate_flags, CUstreamCaptureMode},
    },
};
use candle_core::{Result, Tensor, Var};
use std::sync::Arc;
pub struct GraphPlan {
    inputs: Vec<Var>,
    outputs: Vec<Tensor>,
    graph: Option<CudaGraph>,
    stream: Arc<CudaStream>,
}
// CudaGraph itself is not Sync. This owner only exposes replay through &mut
// self, serializes destruction, and the stream binds its context on each call.
unsafe impl Send for GraphPlan {}
impl GraphPlan {
    /// Warm kernels/metadata, capture, instantiate, and initialize outputs.
    ///
    /// # Safety
    /// External allocations referenced by `forward` (for example model
    /// weights) must outlive this plan. Captured operations must use the
    /// supplied device stream and must not depend on host input values.
    /// Mutable inputs are owned by the plan; replay returns independent copies.
    pub unsafe fn capture<F>(inputs: &[&Tensor], mut forward: F) -> Result<Self>
    where
        F: FnMut(&[Tensor]) -> Result<Vec<Tensor>>,
    {
        if inputs.is_empty()
            || !inputs[0].device().is_cuda()
            || inputs
                .iter()
                .any(|x| !x.device().same_device(inputs[0].device()))
        {
            candle_core::bail!("CUDA graph inputs must share one device/stream");
        }
        let dev = inputs[0].device().as_cuda_device()?;
        let stream = dev.cuda_stream();
        let _params = dev.enable_cuda_graph_htod_cache();
        let variables = inputs
            .iter()
            .map(|x| {
                let variable = Var::zeros(x.shape().clone(), x.dtype(), x.device())?;
                variable.set(x)?;
                Ok(variable)
            })
            .collect::<Result<Vec<_>>>()?;
        let values = variables.iter().map(|x| x.detach()).collect::<Vec<_>>();
        for _ in 0..2 {
            let _ = forward(&values)?;
        }
        stream.synchronize().w()?;
        stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .w()?;
        let outputs = forward(&values);
        // End capture even when the closure rejects an operation.
        let graph = stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .w();
        let outputs = outputs?;
        let graph = graph?.ok_or_else(|| candle_core::Error::Msg("empty CUDA graph".into()))?;
        graph.upload().w()?;
        graph.launch().w()?;
        stream.synchronize().w()?;
        Ok(Self {
            inputs: variables,
            outputs,
            graph: Some(graph),
            stream,
        })
    }
    pub fn matches(&self, inputs: &[&Tensor]) -> bool {
        inputs.len() == self.inputs.len()
            && self.inputs.iter().zip(inputs).all(|(a, b)| {
                a.dims() == b.dims() && a.dtype() == b.dtype() && a.device().same_device(b.device())
            })
    }
    pub fn replay(&mut self, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        if !self.matches(inputs) {
            candle_core::bail!("CUDA graph input signature changed");
        }
        for (dst, src) in self.inputs.iter().zip(inputs) {
            dst.set(src)?;
        }
        self.graph.as_ref().expect("live CUDA graph").launch().w()?;
        // Queue copies before releasing exclusive access: later replays can
        // safely overwrite graph buffers without changing returned tensors.
        self.outputs.iter().map(Tensor::copy).collect()
    }
}
impl Drop for GraphPlan {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        self.outputs.clear();
        let _ = self.stream.synchronize();
        self.graph.take();
    }
}
