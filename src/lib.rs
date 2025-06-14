use pyo3::prelude::*;

pub fn run_lib() {
    println!("Hello, world!");
    let f64_val: f64 = 0.5;

    println!("{:?}", f64_val);
}

#[pyfunction]
fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[pymodule]
fn rs_trijectory(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}
