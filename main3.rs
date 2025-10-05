use anyhow::*;
use std::env;
use tract_onnx::prelude::*;

fn main() -> Result<()> {
    let path = env::args().nth(1).context("usage: onnx_io_inspect <model.onnx>")?;
    let model = tract_onnx::onnx().model_for_path(&path)?;
    println!("== Inputs ==");
    for (ix, outlet) in model.input_outlets()?.iter().enumerate() {
        let fact = model.outlet_fact(*outlet)?;
        let name = &model.node(outlet.node).name;
        println!("{ix}: name={name} dtype={:?} shape={:?}", fact.datum_type, fact.shape);
    }
    println!("== Outputs ==");
    for (ix, outlet) in model.output_outlets()?.iter().enumerate() {
        let fact = model.outlet_fact(*outlet)?;
        let name = &model.node(outlet.node).name;
        println!("{ix}: name={name} dtype={:?} shape={:?}", fact.datum_type, fact.shape);
    }
    Ok(())
}
