use crate::rs_trijectory::debounce_metric::{CollisionMetric, EscapeMetric, Metric};
use crate::rs_trijectory::engine_param::{Method, TrijectoryParam};
use crate::rs_trijectory::geometric_procedure;
use ndarray::{Array1, Array2};
use std::str::FromStr;

pub struct RV {
    pub r: Array2<f64>,
    pub v: Array2<f64>,
}

pub struct Euler {
    pub func: fn(r: &Array2<f64>, v: &Array2<f64>, mass: &Array1<f64>) -> RV,
}

pub struct RK22 {
    pub func: fn(r: &Array2<f64>, v: &Array2<f64>, mass: &Array1<f64>) -> RV,
}

pub struct RK44 {
    pub func: fn(r: &Array2<f64>, v: &Array2<f64>, mass: &Array1<f64>) -> RV,
}

pub trait NumericalMethodsForODE {
    fn step(&self, r: &Array2<f64>, v: &Array2<f64>, mass: &Array1<f64>, delta_t: f64) -> RV;
}

impl NumericalMethodsForODE for Euler {
    fn step(&self, r: &Array2<f64>, v: &Array2<f64>, mass: &Array1<f64>, delta_t: f64) -> RV {
        let delta_rv = (self.func)(&r, &v, &mass);
        RV {
            r: r + delta_t * delta_rv.r,
            v: v + delta_t * delta_rv.v,
        }
    }
}

impl NumericalMethodsForODE for RK22 {
    fn step(&self, r: &Array2<f64>, v: &Array2<f64>, mass: &Array1<f64>, delta_t: f64) -> RV {
        let k1 = (self.func)(&r, &v, &mass);
        let k2 = (self.func)(&(r + &k1.r * delta_t), &(v + &k1.v * delta_t), &mass);

        RV {
            r: r + (k1.r + k2.r) * delta_t * 0.5,
            v: v + (k1.v + k2.v) * delta_t * 0.5,
        }
    }
}

impl NumericalMethodsForODE for RK44 {
    fn step(&self, r: &Array2<f64>, v: &Array2<f64>, mass: &Array1<f64>, delta_t: f64) -> RV {
        let k1 = (self.func)(&r, &v, &mass);
        let k2 = (self.func)(
            &(r + &k1.r * delta_t * 0.5),
            &(v + &k1.v * delta_t * 0.5),
            &mass,
        );
        let k3 = (self.func)(
            &(r + &k2.r * delta_t * 0.5),
            &(v + &k2.v * delta_t * 0.5),
            &mass,
        );
        let k4 = (self.func)(&(r + &k3.r * delta_t), &(v + &k3.v * delta_t), &mass);

        RV {
            r: r + (k1.r / 6.0 + k2.r / 3.0 + k3.r / 3.0 + k4.r / 6.0) * delta_t,
            v: v + (k1.v / 6.0 + k2.v / 3.0 + k3.v / 3.0 + k4.v / 6.0) * delta_t,
        }
    }
}

pub fn f(r: &Array2<f64>, v: &Array2<f64>, mass: &Array1<f64>) -> RV {
    let inv_square = geometric_procedure::calc_vectored_inv_square(r);
    let num_bodies = r.shape()[0];
    let mut ret_r = Array2::<f64>::zeros((num_bodies, r.shape()[1]));
    let mut ret_v = Array2::<f64>::zeros((num_bodies, v.shape()[1]));

    for i_body in 0..num_bodies {
        for j in 0..r.shape()[1] {
            ret_r[[i_body, j]] = v[[i_body, j]];
        }

        let mut fv = Array1::<f64>::zeros(r.shape()[1]);
        for j_body in 0..num_bodies {
            for k in 0..r.shape()[1] {
                fv[k] += inv_square[[i_body, j_body, k]] * mass[j_body];
            }
        }

        for j in 0..v.shape()[1] {
            ret_v[[i_body, j]] = fv[j];
        }
    }

    RV { r: ret_r, v: ret_v }
}

fn create_solver(param: &TrijectoryParam) -> Box<dyn NumericalMethodsForODE> {
    match param.method {
        Method::Euler => Box::new(Euler { func: f }),
        Method::Rk22 => Box::new(RK22 { func: f }),
        Method::Rk44 => Box::new(RK44 { func: f }),
    }
}

fn run_without_traj(
    solver: &dyn NumericalMethodsForODE,
    param: &TrijectoryParam,
    r: &Array2<f64>,
    v: &Array2<f64>,
) -> usize {
    let iterations = (param.max_time / param.time_step) as usize;
    let mass = param.mass.clone();
    let mut met_escape = EscapeMetric::new((param.escape_debounce_time / param.time_step) as usize);
    let mut met_collision = CollisionMetric::new(param.min_distance);

    let mut metrics: Vec<&mut dyn Metric> = vec![&mut met_escape, &mut met_collision];

    let mut _r = r.clone();
    let mut _v = v.clone();

    for step_i in 0..iterations {
        let rv = solver.step(&_r, &_v, &mass, param.time_step);
        _r = rv.r;
        _v = rv.v;

        let detected = metrics.iter_mut().any(|met| met.detect(&_r, &_v, &mass));
        if detected {
            return step_i + 1;
        }
    }

    iterations + 1
}

pub fn life(
    r: &Array2<f64>,
    v: &Array2<f64>,
    max_time: f64,
    time_step: f64,
    escape_debounce_time: f64,
    min_distance: f64,
    method_str: &str,
    mass: Array1<f64>,
) -> f64 {
    let param = TrijectoryParam {
        max_time,
        time_step,
        escape_debounce_time,
        min_distance,
        method: Method::from_str(method_str).unwrap(),
        mass,
    };
    let solver = create_solver(&param);
    let iter_num = run_without_traj(solver.as_ref(), &param, r, v);

    iter_num as f64 * param.time_step
}
