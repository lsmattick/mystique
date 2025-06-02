use ndarray::prelude::*;
use numpy::{PyArray1, PyArray2, PyArrayDyn};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

// The name of the module must be the same as the rust package name
#[pymodule]
fn rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn calc_norm_l2<'py>(x_vec: &PyArrayDyn<f64>) -> PyResult<f64> {
        let x_shape = x_vec.shape();
        let xs = unsafe {
            x_vec
                .as_array()
                .into_shape((x_shape[0], x_shape[1]))
                .unwrap()
                .to_owned()
        };
        let norm = rust_fn::calculate_norm(xs);

        Ok(norm)
    }

    #[pyfn(m)]
    fn cost_w<'py>(
        w: &PyArray1<f64>,
        treatment_vector: &PyArray2<f64>,
        cntrl_feat_mtx: &PyArray2<f64>,
        lam: f64,
    ) -> PyResult<f64> {
        let w_shape = w.shape();
        let treatment_vector_shape = treatment_vector.shape();
        let cntrl_feat_mtx_shape = cntrl_feat_mtx.shape();

        let w_reshaped = unsafe { w.as_array().into_shape((w_shape[0], 1)).unwrap() };
        let treatment_vector = unsafe {
            treatment_vector
                .as_array()
                .into_shape((treatment_vector_shape[0], 1))
                .unwrap()
        };
        let cntrl_feat_mtx = unsafe {
            cntrl_feat_mtx
                .as_array()
                .into_shape((cntrl_feat_mtx_shape[0], cntrl_feat_mtx_shape[1]))
                .unwrap()
        };

        let dot_product = cntrl_feat_mtx.dot(&w_reshaped);
        let dist_vector = &treatment_vector - &dot_product;

        // Penalty Section
        let penalty: f64 = w_reshaped
            .iter()
            .enumerate()
            .map(|(i, &w_i)| {
                let column_i = cntrl_feat_mtx
                    .slice(s![.., i])
                    .to_owned()
                    .into_shape((cntrl_feat_mtx_shape[0], 1))
                    .unwrap();
                w_i * rust_fn::calculate_norm(&treatment_vector - &column_i).powi(2)
            })
            .sum();

        // Result
        let result = rust_fn::calculate_norm(dist_vector).powi(2) + lam * penalty;

        Ok(result)
    }

    #[pyfn(m)]
    fn norm_x_over_v<'py>(x: &PyArray2<f64>, v: &PyArray1<f64>) -> PyResult<f64> {
        let x = unsafe { x.as_array() };
        let v = unsafe { v.as_array() };

        let mut V = Array2::<f64>::eye(v.len());
        V.diag_mut().zip_mut_with(&v, |a, &b| *a *= b);

        let final_result = x.t().dot(&V).dot(&x);
        let first_element = final_result.mapv(f64::sqrt).as_slice().unwrap()[0];
        Ok(first_element)
    }

    #[pyfn(m)]
    fn dtw_distance<'py>(s1: &PyArray1<f64>, s2: &PyArray1<f64>) -> PyResult<f64> {
        let s1 = unsafe { s1.as_array() };
        let s2 = unsafe { s2.as_array() };

        Ok(rust_fn::dtw_distance(s1, s2))
    }

    Ok(())
}

// The rust side functions
// Put it in mod to separate it from the python bindings
// These are just some random operations
// you probably want to do something more meaningful.
mod rust_fn {
    use ndarray::{Array2, ArrayView1, Dimension};
    use std::collections::HashMap;

    pub fn calculate_norm(dist_vector: Array2<f64>) -> f64 {
        let shape = dist_vector.raw_dim();
        let total_elements = shape.size();
        let flattened_vector = dist_vector.into_shape((total_elements,)).unwrap();
        let norm = flattened_vector.fold(0., |acc, &x| acc + x.powi(2)).sqrt();

        norm
    }

    pub fn dtw_distance<'py>(s1: ArrayView1<f64>, s2: ArrayView1<f64>) -> f64 {
        let mut dtw = HashMap::new();

        for i in 0..s1.len() {
            dtw.insert((i, usize::MAX), f64::INFINITY);
        }
        for i in 0..s2.len() {
            dtw.insert((usize::MAX, i), f64::INFINITY);
        }
        dtw.insert((usize::MAX, usize::MAX), 0.0);

        for (i, &x) in s1.iter().enumerate() {
            for (j, &y) in s2.iter().enumerate() {
                let dist = (x - y).powi(2);
                let min_prev = *[
                    dtw.get(&(i - 1, j)).unwrap(),
                    dtw.get(&(i, j - 1)).unwrap(),
                    dtw.get(&(i - 1, j - 1)).unwrap(),
                ]
                .iter()
                .map(|&v| v)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
                dtw.insert((i, j), dist + min_prev);
            }
        }

        dtw[&(s1.len() - 1, s2.len() - 1)].sqrt()
    }
}
