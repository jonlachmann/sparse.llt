extern crate faer;

use extendr_api::prelude::*;
use dyn_stack::{GlobalPodBuffer, StackReq, PodStack, ReborrowMut};
use faer::sparse::linalg::cholesky::supernodal;
use faer::sparse::linalg::cholesky::simplicial;

fn load_dgc_matrix<'a>(mat: S4, x: &'a Robj) -> faer::sparse::SparseColMatRef<'a, u32, f64> {
    use extendr_api::AsTypedSlice;
    let i = mat.get_slot("i").unwrap();
    let p = mat.get_slot("p").unwrap();
    let dim = mat.get_slot("Dim").unwrap();

    let i = i.as_typed_slice().unwrap();
    let p = p.as_typed_slice().unwrap();
    let x = x.as_real_slice().unwrap();
    let dim = dim.as_integer_slice().unwrap();
    let ncols = dim[0] as _;
    let nrows = dim[1] as _;

    // Map the sparse matrix
    let symbolic_1 = faer::sparse::SymbolicSparseColMatRef::new_checked(nrows, ncols, p, None, i);
    let a_mat = faer::sparse::SparseColMatRef::<u32, f64>::new({
            symbolic_1
        },
        x
    );

    return a_mat;
}

/// Perform a symbolic decomposition of invomega0 + x_invsigma_x and return a pointer to it
/// @export
#[extendr]
fn beta_symbolic(invomega0: S4, x_invsigma_x: S4) -> Robj {
    let invomega0_x = invomega0.get_slot("x").unwrap();
    let invomega0_mat = load_dgc_matrix(invomega0, &invomega0_x);

    let x_invsigma_x_x = x_invsigma_x.get_slot("x").unwrap();
    let x_invsigma_x_mat = load_dgc_matrix(x_invsigma_x, &x_invsigma_x_x);

    let invomegabar = invomega0_mat + x_invsigma_x_mat;

    let etree = &mut *vec![0i32; invomegabar.nrows()];
    let col_counts = &mut *vec![0u32; invomegabar.ncols()];

    let mut mem = GlobalPodBuffer::new(
        StackReq::new::<u32>(invomegabar.nrows())
            .or(supernodal::factorize_supernodal_symbolic_cholesky_req::<u32>(invomegabar.nrows()).unwrap()),
    );

    let mut stack = PodStack::new(&mut mem);
    let etree = simplicial::prefactorize_symbolic_cholesky::<u32>(
        etree, col_counts, invomegabar.as_ref().symbolic(), stack.rb_mut(),
    );
    let symbolic = supernodal::factorize_supernodal_symbolic::<u32>(
        invomegabar.as_ref().symbolic(), etree, col_counts, stack.rb_mut(), Default::default(),
    ).unwrap();

    drop(mem);

    let externalptr_symb = ExternalPtr::new(symbolic);

    return externalptr_symb.into()
}

/// Draw beta from N(omegabar * x_invsigma_y, omegabar)
/// @export
#[extendr]
unsafe fn beta_draw(invomega0: S4, x_invsigma_x: S4, x_invsigma_y: &[f64], random_vec: &[f64], symb: Robj) -> Vec<f64> {
    // Get the symbolic decomposition from the pointer
    let symbolic_ptr: ExternalPtr<faer::sparse::linalg::cholesky::supernodal::SymbolicSupernodalCholesky<u32>> = symb.try_into().unwrap();
    let symbolic = symbolic_ptr.addr();

    // Map invomega0 and x_invsigma_x
    let invomega0_x = invomega0.get_slot("x").unwrap();
    let invomega0_mat = load_dgc_matrix(invomega0, &invomega0_x);

    let x_invsigma_x_x = x_invsigma_x.get_slot("x").unwrap();
    let x_invsigma_x_mat = load_dgc_matrix(x_invsigma_x, &x_invsigma_x_x);

    let invomegabar = invomega0_mat + x_invsigma_x_mat;

    // Copy x_invsigma_y into owned memory and map it to a matrix
    let mut x_invsigma_y_bind = x_invsigma_y.to_owned();
    let x_invsigma_y_mut: &mut [f64] = x_invsigma_y_bind.as_mut();
    let mut x_invsigma_y_mat = faer::mat::from_column_major_slice_mut::<f64>(x_invsigma_y_mut, invomegabar.nrows(), 1);

    // Map random_vec to a matrix
    let random_vec_mat = faer::mat::from_column_major_slice::<f64>(&random_vec, invomegabar.nrows(), 1);

    let mut mem = GlobalPodBuffer::new(
        supernodal::factorize_supernodal_numeric_llt_req::<u32, f64>(
            &symbolic, faer::Parallelism::None
        ).unwrap(),
    );
    let mut stack = PodStack::new(&mut mem);

    let mut l_values = Vec::with_capacity(symbolic.len_values());
    l_values.set_len(symbolic.len_values());

    let _ = supernodal::factorize_supernodal_numeric_llt::<u32, f64>(
        &mut l_values,
        invomegabar.as_ref(),
        Default::default(),
        &symbolic,
        faer::Parallelism::None,
        stack.rb_mut()
    );

    let llt = supernodal::SupernodalLltRef::<'_, u32, f64>::new(&symbolic, &l_values);

    llt.l_solve_with_conj(
        faer::Conj::No,
        x_invsigma_y_mat.rb_mut(),
        faer::Parallelism::None,
        stack.rb_mut()
    );

    x_invsigma_y_mat += random_vec_mat;

    llt.l_transpose_solve_with_conj(
        faer::Conj::No,
        x_invsigma_y_mat.rb_mut(),
        faer::Parallelism::None,
        stack.rb_mut()
    );

    x_invsigma_y_mut.to_vec()
}


// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod sparse_llt;
    fn beta_symbolic;
    fn beta_draw;
}
