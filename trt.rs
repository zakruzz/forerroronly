use anyhow::*;
use ndarray::{Array, IxDyn};
use std::{fs, sync::Arc};

use async_tensorrt::{
    runtime::Runtime,
    engine::Engine,
    execution_context::ExecutionContext,
    memory::DeviceBuffer,
    stream::CudaStream,
};

pub struct TrtSession {
    rt: tokio::runtime::Runtime,
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
    pub fn from_engine_file<P: AsRef<std::path::Path>>(engine_path: P) -> Result<Self> {
        let rt = tokio::runtime::Runtime::new()?;
        // block_on karena API async
        let (runtime, engine, context, stream, input_name, output_name, output_elems, dynamic_input) =
        rt.block_on(async {
            let runtime = Runtime::new(); // tidak mengembalikan Result
            let bytes = fs::read(&engine_path)
                .with_context(|| format!("gagal baca engine: {}", engine_path.as_ref().display()))?;
            let engine = runtime.deserialize_engine(&bytes).await?;
            let engine = Arc::new(engine);
            let context = engine.create_execution_context().await?;
            let stream = CudaStream::new().await?;

            // pilih IO pertama
            let io = engine.io_tensors().await?; // Vec<String>
            let mut input_name = None;
            let mut output_name = None;
            for name in &io {
                if engine.is_input(name).await? {
                    if input_name.is_none() { input_name = Some(name.clone()); }
                } else {
                    if output_name.is_none() { output_name = Some(name.clone()); }
                }
            }
            let input_name = input_name.ok_or_else(|| anyhow!("input tensor tidak ditemukan"))?;
            let output_name = output_name.ok_or_else(|| anyhow!("output tensor tidak ditemukan"))?;

            // dim output & input
            let odim = engine.tensor_shape(&output_name).await?; // Vec<i32>
            let output_elems: usize = odim.iter().map(|&d| d as usize).product();
            let in_dim = engine.tensor_shape(&input_name).await?;
            let dynamic_input = in_dim.iter().any(|&d| d < 0);

            Ok::<_, anyhow::Error>((runtime, engine, context, stream, input_name, output_name, output_elems, dynamic_input))
        })?;

        Ok(Self {
            rt, runtime, engine, context, stream,
            input_name, output_name, output_elems, dynamic_input
        })
    }

    /// input: [1,3,H,W] f32 NCHW
    pub fn infer(&mut self, input: &Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>> {
        let shape: Vec<i32> = input.shape().iter().map(|&x| x as i32).collect();
        let elems = input.len();
        let mut host_out = vec![0f32; self.output_elems];

        self.rt.block_on(async {
            if self.dynamic_input {
                self.context.set_input_shape(&self.input_name, &shape).await?;
            }

            let mut d_in  = DeviceBuffer::<f32>::new(elems).await?;
            let mut d_out = DeviceBuffer::<f32>::new(self.output_elems).await?;

            d_in.copy_from(input.as_slice().unwrap(), &self.stream).await?;

            self.context.set_tensor_address(&self.input_name,  d_in.as_device_ptr()).await?;
            self.context.set_tensor_address(&self.output_name, d_out.as_device_ptr()).await?;

            self.context.execute_async_v3(&self.stream).await?;
            self.stream.synchronize().await?;

            d_out.copy_to(&mut host_out, &self.stream).await?;
            Ok::<_, anyhow::Error>(())
        })?;

        let odim = self.rt.block_on(self.engine.tensor_shape(&self.output_name))?;
        let oshp: Vec<usize> = odim.iter().map(|&d| d as usize).collect();
        let arr = Array::from_shape_vec(IxDyn(&oshp), host_out)?;
        Ok(arr)
    }
}
