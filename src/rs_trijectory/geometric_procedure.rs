use ndarray::{Array2, Array3, Axis};
use ndarray_linalg::Norm;


pub fn calc_relative_vector(r: &Array2<f64>) -> Array3<f64> {
    // Equivalent of r[np.newaxis, :, :]
    let r_expanded_0 = r.view().insert_axis(Axis(0));
    // Equivalent of r[:, np.newaxis, :]
    let r_expanded_1 = r.view().insert_axis(Axis(1));

    // Broadcasting subtraction
    &r_expanded_0 - &r_expanded_1
}


pub fn calc_vectored_inv_square(r: &Array2<f64>) -> Array3<f64> {
    let rel = calc_relative_vector(r);
    let num_bodies = r.shape()[0];

    // Calculate the norm along the last axis (axis=2 in numpy)
    let non_zero_dist = rel.map_axis(Axis(2), |row| row.norm_l2());

    // Create an identity matrix and add it to non_zero_dist
    // This handles the np.eye(len(r)) part to avoid division by zero for self-interactions
    let identity_matrix = Array2::<f64>::eye(num_bodies);
    let non_zero_dist_with_identity = &non_zero_dist + &identity_matrix;

    // Equivalent of np.power(non_zero_dist[:, :, np.newaxis], 3)
    let powered_dist = non_zero_dist_with_identity
        .insert_axis(Axis(2))
        .mapv(|x| x.powi(3));

    // Equivalent of rel / powered_dist
    rel / powered_dist
}
