use anyhow::*;
use ndarray::{Array, IxDyn};
use std::{fs, sync::Arc};

use async_tensorrt::{
    logger::Logger,
    runtime::Runtime,
    engine::Engine,
    context::ExecutionContext,
    cuda::{CudaStream, DeviceBuffer},
};

pub struct TrtSession {
    rt: tokio::runtime::Runtime,
    logger: Logger,
    runtime: Runtime,
    engine: Arc<Engine>,
    context: ExecutionContext,
    stream: CudaStream,
    input_name: String,
    output_name: String,
    output_elems: usize,
    dynamic_input: bool,
}

impl TrtSession {
    pub fn from_engine_file<P: AsRef<std::path::Path>>(engine_path:P) -> Result<Self> {
        let rt = tokio::runtime::Runtime::new()?;

        let (logger, runtime, engine, context, stream, input_name, output_name, output_elems, dynamic_input) =
        rt.block_on(async {
            let logger = Logger::warning();
            let runtime = Runtime::new(&logger).await?;
            let bytes = fs::read(&engine_path)
                .with_context(|| format!("gagal baca engine: {}", engine_path.as_ref().display()))?;
            let engine = runtime.deserialize_cuda_engine(&bytes).await?;
            let engine = Arc::new(engine);
            let mut context = engine.create_execution_context().await?;
            let stream = CudaStream::new().await?;

            // cari IO tensor
            let num = engine.num_io_tensors();
            let mut input_name = None;
            let mut output_name = None;
            for i in 0..num {
                let name = engine.get_tensor_name(i).await?;
                let is_input = engine.is_input(&name).await?;
                if is_input && input_name.is_none() { input_name = Some(name.clone()); }
                if !is_input && output_name.is_none() { output_name = Some(name.clone()); }
            }
            let input_name = input_name.ok_or_else(|| anyhow!("input tensor tidak ditemukan"))?;
            let output_name = output_name.ok_or_else(|| anyhow!("output tensor tidak ditemukan"))?;

            // cek shape output → hitung elemen
            let odim = engine.get_tensor_shape(&output_name).await?;
            let output_elems: usize = odim.iter().map(|&d| d as usize).product();

            // cek apakah input dynamic (ada -1)
            let in_dim = engine.get_tensor_shape(&input_name).await?;
            let dynamic_input = in_dim.iter().any(|&d| d < 0);

            Ok::<_, anyhow::Error>((logger, runtime, engine, context, stream, input_name, output_name, output_elems, dynamic_input))
        })?;

        Ok(Self {
            rt, logger, runtime, engine, context, stream,
            input_name, output_name, output_elems, dynamic_input
        })
    }

    /// input: [1,3,H,W] f32 NCHW (biasanya 640x640)
    pub fn infer(&mut self, input:&Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>> {
        let shape: Vec<i32> = input.shape().iter().map(|&x| x as i32).collect();
        let elems = input.len();
        let mut host_out = vec![0f32; self.output_elems];

        self.rt.block_on(async {
            // kalau engine dynamic, set shape input
            if self.dynamic_input {
                self.context.set_input_shape(&self.input_name, &shape).await?;
            }

            // alok device
            let mut d_in  = DeviceBuffer::<f32>::new(elems).await?;
            let mut d_out = DeviceBuffer::<f32>::new(self.output_elems).await?;

            // copy H->D
            d_in.copy_from(input.as_slice().unwrap(), &self.stream).await?;

            // set address
            self.context.set_tensor_address(&self.input_name,  d_in.as_device_ptr()).await?;
            self.context.set_tensor_address(&self.output_name, d_out.as_device_ptr()).await?;

            // eksekusi
            self.context.execute_async_v3(&self.stream).await?;
            self.stream.synchronize().await?;

            // copy D->H
            d_out.copy_to(&mut host_out, &self.stream).await?;
            Ok::<_, anyhow::Error>(())
        })?;

        // bentuk ndarray pakai dim output engine
        let odim = self.rt.block_on(self.engine.get_tensor_shape(&self.output_name))?;
        let oshp: Vec<usize> = odim.iter().map(|&d| d as usize).collect();
        let arr = Array::from_shape_vec(IxDyn(&oshp), host_out)?;
        Ok(arr)
    }
}
