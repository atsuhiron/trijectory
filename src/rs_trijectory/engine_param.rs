use ndarray::Array1;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Euler,
    Rk22,
    Rk44,
}

impl FromStr for Method {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "euler" => Ok(Method::Euler),
            "rk22" => Ok(Method::Rk22),
            "rk44" => Ok(Method::Rk44),
            _ => Err(()),
        }
    }
}

pub struct TrijectoryParam {
    pub max_time: f64,
    pub time_step: f64,
    pub escape_debounce_time: f64,
    pub min_distance: f64,
    pub method: Method,
    pub mass: Array1<f64>,
}
