mod rs_trijectory;

use numpy::{IntoPyArray, PyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyResult;
use pyo3::Python;

use rs_trijectory::geometric_procedure::calc_relative_vector;

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
#[pyo3(name = "calc_relative_vector")]
fn py_calc_relative_vector<'py>(py: Python<'py>, r_py: PyReadonlyArrayDyn<'py, f64>) -> PyResult<Py<PyArray<f64, ndarray::Dim<[usize; 3]>>>> {
    // PythonのNumPy配列をRustのndarray::Array2<f64>に変換
    let r_ndarray = r_py.as_array().to_owned().into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| PyValueError::new_err(format!("Input array must be 2D: {}", e)))?;

    // Rustのロジックを呼び出し
    let result_ndarray = calc_relative_vector(&r_ndarray);

    // Rustのndarray::Array3<f64>をPythonのNumPy配列に変換して返す
    Ok(result_ndarray.into_pyarray(py).into())
}


#[pymodule]
#[pyo3(name="rs_trijectory")]
fn trijectory(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(py_calc_relative_vector, m)?)?;
    Ok(())
}
