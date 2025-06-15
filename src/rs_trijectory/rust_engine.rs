use ndarray::{Array1, Array2};


struct RV {
    r: Array2<f64>,
    v: Array2<f64>,
}


struct Euler {
    func: fn(r: &Array2<f64>, v: &Array2<f64>, mass: &Array1<f64>) -> RV,
}


struct RK22 {
    func: fn(r: &Array2<f64>, v: &Array2<f64>, mass: &Array1<f64>) -> RV,
}


struct RK44 {
    func: fn(r: &Array2<f64>, v: &Array2<f64>, mass: &Array1<f64>) -> RV,
}


trait NumericalMethodsForODE {
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
        let k2 = (self.func)(&(r + &k1.r * delta_t * 0.5), &(v + &k1.v * delta_t * 0.5), &mass);
        let k3 = (self.func)(&(r + &k2.r * delta_t * 0.5), &(v + &k2.v * delta_t * 0.5), &mass);
        let k4 = (self.func)(&(r + &k3.r * delta_t), &(v + &k3.v * delta_t), &mass);
        
        RV {
            r: r + (k1.r/6.0 + k2.r/3.0 + k3.r/3.0 + k4.r/6.0) * delta_t,
            v: v + (k1.v/6.0 + k2.v/3.0 + k3.v/3.0 + k4.v/6.0) * delta_t,
        }
    }
}