use crate::rs_trijectory::geometric_procedure;
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Norm;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    LessEqual,    // "le"
    LessThan,     // "lt"
    GreaterEqual, // "ge"
    GreaterThan,  // "gt"
}

pub trait Metric {
    fn new(max_steps: usize, threshold: f64, ope: Operation) -> Self
    where
        Self: Sized;

    fn measure(&self, r: &Array2<f64>, v: &Array2<f64>, mass: &Array1<f64>) -> f64;

    fn detect(&mut self, r: &Array2<f64>, v: &Array2<f64>, mass: &Array1<f64>) -> bool;

    fn operate(&self, measure_value: f64) -> bool;
}

pub struct MetricBase {
    max_steps: usize,
    threshold: f64,
    ope: Operation,
    status: bool,
    step_count: usize,
}

impl MetricBase {
    pub fn new(max_steps: usize, threshold: f64, ope: Operation) -> Self {
        Self {
            max_steps,
            threshold,
            ope,
            status: false,
            step_count: 0,
        }
    }

    pub fn detect(&mut self, measure_value: f64) -> bool {
        let status = self.operate(measure_value);
        if status {
            self.status = true;
            self.step_count += 1;
        } else {
            self.status = false;
            self.step_count = 0;
        }
        self.step_count >= self.max_steps
    }

    pub fn operate(&self, measure_value: f64) -> bool {
        match self.ope {
            Operation::LessEqual => measure_value <= self.threshold,
            Operation::LessThan => measure_value < self.threshold,
            Operation::GreaterEqual => measure_value >= self.threshold,
            Operation::GreaterThan => measure_value > self.threshold,
        }
    }
}

pub struct EscapeMetric {
    base: MetricBase,
}

impl EscapeMetric {
    pub fn new(max_steps: usize) -> Self {
        Self {
            base: MetricBase::new(max_steps, 0.0, Operation::GreaterThan),
        }
    }
}

impl Metric for EscapeMetric {
    fn new(max_steps: usize, _threshold: f64, _ope: Operation) -> Self {
        Self::new(max_steps)
    }

    fn measure(&self, r: &Array2<f64>, v: &Array2<f64>, mass: &Array1<f64>) -> f64 {
        let rel = geometric_procedure::calc_relative_vector(r);
        let dist = rel.map_axis(Axis(2), |row| row.norm_l2());

        let dist_array = vec![dist[[0, 1]], dist[[0, 2]], dist[[1, 2]]];

        let min_idx = dist_array
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let (e_3, b_12) = match min_idx {
            0 => (2, (0, 1)),
            1 => (1, (0, 2)),
            _ => (0, (1, 2)),
        };

        let binary_mass = mass[b_12.0] + mass[b_12.1];

        let dim = r.shape()[1]; // Dimension of space
        let mut bound_r = Array1::<f64>::zeros(dim);
        let mut bound_v = Array1::<f64>::zeros(dim);

        for i in 0..dim {
            bound_r[i] =
                (r[[b_12.0, i]] * mass[b_12.0] + r[[b_12.1, i]] * mass[b_12.1]) / binary_mass;
            bound_v[i] =
                (v[[b_12.0, i]] * mass[b_12.0] + v[[b_12.1, i]] * mass[b_12.1]) / binary_mass;
        }

        let mut rel_r = Array1::<f64>::zeros(dim);
        let mut rel_v = Array1::<f64>::zeros(dim);

        for i in 0..dim {
            rel_r[i] = r[[e_3, i]] - bound_r[i];
            rel_v[i] = v[[e_3, i]] - bound_v[i];
        }

        let energy_k = 0.5 * rel_v.norm_l2().powi(2);
        let energy_u = binary_mass / rel_r.norm_l2();

        energy_k - energy_u
    }

    fn detect(&mut self, r: &Array2<f64>, v: &Array2<f64>, mass: &Array1<f64>) -> bool {
        let measure_value = self.measure(r, v, mass);
        self.base.detect(measure_value)
    }

    fn operate(&self, measure_value: f64) -> bool {
        self.base.operate(measure_value)
    }
}

pub struct CollisionMetric {
    base: MetricBase,
}

impl CollisionMetric {
    pub fn new(min_distance: f64) -> Self {
        Self {
            base: MetricBase::new(1, min_distance, Operation::LessThan),
        }
    }
}

impl Metric for CollisionMetric {
    fn new(_max_steps: usize, threshold: f64, _ope: Operation) -> Self {
        Self::new(threshold)
    }

    fn measure(&self, r: &Array2<f64>, _v: &Array2<f64>, _mass: &Array1<f64>) -> f64 {
        let rel = geometric_procedure::calc_relative_vector(r);
        let dist = rel.map_axis(Axis(2), |row| row.norm_l2());

        let num_bodies = r.shape()[0];
        let mut min_dist = f64::INFINITY;

        for i in 0..num_bodies {
            for j in (i + 1)..num_bodies {
                if dist[[i, j]] < min_dist {
                    min_dist = dist[[i, j]];
                }
            }
        }

        min_dist
    }

    fn detect(&mut self, r: &Array2<f64>, v: &Array2<f64>, mass: &Array1<f64>) -> bool {
        let measure_value = self.measure(r, v, mass);
        self.base.detect(measure_value)
    }

    fn operate(&self, measure_value: f64) -> bool {
        self.base.operate(measure_value)
    }
}
