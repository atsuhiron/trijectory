pub mod rs_trijectory;

pub use rs_trijectory::engine_param::{Method, TrijectoryParam};

use numpy::PyReadonlyArrayDyn;
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use rs_trijectory::rust_engine::life;

pub fn run_lib() {
    println!("Hello, world!");
    let f64_val: f64 = 0.5;

    println!("{:?}", f64_val);
}

#[pyfunction]
fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[pyfunction]
#[pyo3(name = "calc_life")]
fn py_calc_life<'py>(
    r_py: PyReadonlyArrayDyn<'py, f64>,
    v_py: PyReadonlyArrayDyn<'py, f64>,
    max_time: f64,
    time_step: f64,
    escape_debounce_time: f64,
    min_distance: f64,
    method_str: &str,
    m_py: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<f64> {
    let r_ndarray = r_py
        .as_array()
        .to_owned()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| PyValueError::new_err(format!("Input array must be 2D: {}", e)))?;
    let v_ndarray = v_py
        .as_array()
        .to_owned()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| PyValueError::new_err(format!("Input array must be 2D: {}", e)))?;
    let m_ndarray = m_py
        .as_array()
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| PyValueError::new_err(format!("Input array must be 2D: {}", e)))?;

    Ok(life(
        &r_ndarray,
        &v_ndarray,
        max_time,
        time_step,
        escape_debounce_time,
        min_distance,
        method_str,
        m_ndarray,
    ))
}

#[pymodule]
#[pyo3(name = "rs_trijectory")]
fn trijectory(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(py_calc_life, m)?)?;
    Ok(())
}
