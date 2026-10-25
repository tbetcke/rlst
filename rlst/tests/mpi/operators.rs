//! Distributed vector spaces, the distributed CSR operator, and iterative solvers on them.

use std::rc::Rc;

use mpi::{
    topology::{CommunicatorRelation, SimpleCommunicator},
    traits::Communicator,
};
use rlst::{
    AsMatrixApply, FromAijDistributed, InnerProductSpace, LinearSpace, NormedSpace,
    abstract_operator::OperatorBase,
    c64,
    distributed_tools::IndexLayout,
    operator::{
        abstract_operator::Operator,
        algorithms::{conjugate_gradients::CgIteration, gmres::GmresIteration},
        element::Element,
        interface::{
            distributed_array_vector_space::DistributedArrayVectorSpace,
            matrix_operator::DistributedCsrOperator,
        },
    },
    sparse::distributed_csr_mat::DistributedCsrMatrix,
};

use crate::common::{
    Case, Layout, TOL, all_close, close, collective_assert, complex_value_at, expected,
    local_values, uneven_count, value_at, vector,
};

/// The cases of this module.
pub const CASES: &[Case] = &[
    ("vector-space-ops", vector_space_ops),
    ("distributed-operator-apply", distributed_operator_apply),
    ("distributed-cg", distributed_cg),
    ("distributed-gmres", distributed_gmres),
];

/// An uneven layout with `scale` times the indices of [`uneven_count`].
fn layout(comm: &SimpleCommunicator, salt: usize, scale: usize) -> Layout<'_> {
    let count = uneven_count(comm.rank() as usize, comm.size() as usize, salt);
    Rc::new(IndexLayout::from_local_counts(scale * count, comm))
}

/// The tridiagonal matrix with `lower`, `diagonal` and `upper` on `layout`. Each rank
/// contributes the rows it owns.
fn tridiagonal<'a>(
    layout: &Layout<'a>,
    lower: f64,
    diagonal: f64,
    upper: f64,
) -> DistributedCsrMatrix<'a, f64, SimpleCommunicator> {
    let n = layout.number_of_global_indices();
    let (start, end) = layout.local_range();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for row in start..end {
        for (col, value) in [
            (row.checked_sub(1), lower),
            (Some(row), diagonal),
            ((row + 1 < n).then_some(row + 1), upper),
        ] {
            if let Some(col) = col {
                rows.push(row);
                cols.push(col);
                data.push(value);
            }
        }
    }
    DistributedCsrMatrix::from_aij(layout.clone(), layout.clone(), &rows, &cols, &data)
}

/// The local rows of the tridiagonal matrix times the vector with entries `x(g)`.
fn tridiagonal_times(
    layout: &Layout<'_>,
    (lower, diagonal, upper): (f64, f64, f64),
    x: impl Fn(usize) -> f64,
) -> Vec<f64> {
    let n = layout.number_of_global_indices();
    expected(layout, |g| {
        let below = if g > 0 { lower * x(g - 1) } else { 0.0 };
        let above = if g + 1 < n { upper * x(g + 1) } else { 0.0 };
        below + diagonal * x(g) + above
    })
}

/// Contract: every operation of the distributed vector space acts on the local entries, and
/// its inner product and norm combine all ranks.
fn vector_space_ops(comm: &SimpleCommunicator) {
    let layout = layout(comm, 0, 2);
    let total = layout.number_of_global_indices();
    let space = DistributedArrayVectorSpace::<_, f64>::new(layout.clone());
    let f = |g: usize| value_at(g, 0);
    let h = |g: usize| value_at(g, 1);
    let x = Element::new(&space, vector(&layout, f));
    let y = Element::new(&space, vector(&layout, h));
    let e = |op: &dyn Fn(usize) -> f64| expected(&layout, op);
    let values = |element: &Element<DistributedArrayVectorSpace<SimpleCommunicator, f64>>| {
        local_values(element.imp())
    };
    let mut ok = space.comm().compare(comm) == CommunicatorRelation::Identical
        && space.index_layout().is_same(&layout);

    ok &= values(&space.zero()) == vec![0.0; layout.number_of_local_indices()];
    ok &= values(&space.add(&x, &y)) == e(&|g| f(g) + h(g));
    ok &= values(&space.sub(&x, &y)) == e(&|g| f(g) - h(g));
    ok &= values(&space.scalar_mul(&2.5, &x)) == e(&|g| 2.5 * f(g));
    ok &= values(&space.neg(&x)) == e(&|g| -f(g));
    ok &= values(&(&x + &y)) == e(&|g| f(g) + h(g));
    ok &= values(&(3.0 * &x - &y)) == e(&|g| 3.0 * f(g) - h(g));

    let mut z = space.copy_from(&x);
    space.sum_inplace(&mut z, &y);
    ok &= values(&z) == e(&|g| f(g) + h(g));
    space.sub_inplace(&mut z, &y);
    ok &= all_close(values(&z), e(&f), 0.0);
    space.scale_inplace(&4.0, &mut z);
    ok &= all_close(values(&z), e(&|g| 4.0 * f(g)), 0.0);
    space.axpy_inplace(&-2.0, &y, &mut z);
    ok &= all_close(values(&z), e(&|g| 4.0 * f(g) - 2.0 * h(g)), 0.0);
    // The copy is independent of its source.
    ok &= values(&x) == e(&f);

    let inner: f64 = (0..total).map(|g| f(g) * h(g)).sum();
    let norm = (0..total).map(|g| f(g) * f(g)).sum::<f64>().sqrt();
    ok &= close(space.inner_product(&x, &y), inner, TOL);
    ok &= close(space.norm(&x), norm, TOL);

    // Complex inner products conjugate the second argument.
    let complex_space = DistributedArrayVectorSpace::<_, c64>::new(layout.clone());
    let xc = Element::new(&complex_space, vector(&layout, |g| complex_value_at(g, 0)));
    let yc = Element::new(&complex_space, vector(&layout, |g| complex_value_at(g, 1)));
    let inner: c64 = (0..total)
        .map(|g| complex_value_at(g, 0) * complex_value_at(g, 1).conj())
        .sum();
    ok &= close(complex_space.inner_product(&xc, &yc), inner, TOL);
    ok &= close(
        complex_space.norm(&xc),
        (0..total)
            .map(|g| complex_value_at(g, 0).norm_sqr())
            .sum::<f64>()
            .sqrt(),
        TOL,
    );

    collective_assert(
        comm,
        ok,
        "distributed vector spaces must implement their operations on all ranks",
    );
}

/// Contract: the distributed CSR operator, built either way, applies its matrix in the spaces
/// of the matrix layouts, also inside operator sums.
fn distributed_operator_apply(comm: &SimpleCommunicator) {
    let layout = layout(comm, 1, 3);
    let coefficients = (-1.0, 3.0, -0.5);
    let mat = tridiagonal(&layout, coefficients.0, coefficients.1, coefficients.2);
    let f = |g: usize| value_at(g, 0);
    let h = |g: usize| value_at(g, 2);
    let ax = tridiagonal_times(&layout, coefficients, f);

    let op = Operator::from(&mat);
    let direct = Operator::new(DistributedCsrOperator::new(&mat));
    let mut ok = op.domain().index_layout().is_same(&layout)
        && op.range().index_layout().is_same(&layout)
        && direct.domain().index_layout().is_same(&layout);

    let x = Element::new(op.domain(), vector(&layout, f));
    let mut y = Element::new(op.range(), vector(&layout, h));
    op.apply(2.0, &x, -1.0, &mut y);
    let reference: Vec<f64> = ax
        .iter()
        .zip(expected(&layout, h))
        .map(|(&ax, h)| 2.0 * ax - h)
        .collect();
    ok &= all_close(local_values(y.imp()), reference.clone(), TOL);

    let mut z = vector(&layout, h);
    mat.apply(2.0, x.imp(), -1.0, &mut z);
    ok &= local_values(&z) == local_values(y.imp());

    let x = Element::new(direct.domain(), vector(&layout, f));
    ok &= all_close(local_values(direct.dot(&x).imp()), ax.clone(), TOL);

    // Operator algebra over distributed operators.
    let combined = op.r() + 2.0 * direct.r();
    let x = Element::new(combined.domain(), vector(&layout, f));
    ok &= all_close(
        local_values(combined.dot(&x).imp()),
        ax.iter().map(|&value| 3.0 * value),
        TOL,
    );

    collective_assert(
        comm,
        ok,
        "the distributed CSR operator must apply its matrix",
    );
}

/// Contract: conjugate gradients solve a distributed 1-D Laplacian on an uneven layout with an
/// empty rank.
fn distributed_cg(comm: &SimpleCommunicator) {
    let layout = layout(comm, 2, 5);
    let mat = tridiagonal(&layout, -1.0, 2.0, -1.0);
    let op = Operator::from(&mat);
    let solution = |g: usize| value_at(g, 0);

    let exact = Element::new(op.domain(), vector(&layout, solution));
    let rhs = op.dot(&exact);
    let mut x = op.domain().zero();
    let residual = CgIteration::new(&op, &rhs, &mut x)
        .set_tol(1e-13)
        .set_max_iter(1000)
        .run();

    // An independent residual.
    let mut check = op.range().copy_from(&rhs);
    op.apply(-1.0, &x, 1.0, &mut check);
    let rhs_norm = op.range().norm(&rhs);
    let check_norm = op.range().norm(&check);
    let ok = residual < 1e-13
        && check_norm <= 1e-11 * rhs_norm
        && all_close(local_values(x.imp()), expected(&layout, solution), 1e-8);

    collective_assert(
        comm,
        ok,
        "conjugate gradients must solve the distributed Laplacian",
    );
}

/// Contract: restarted GMRES solves a distributed nonsymmetric tridiagonal system.
fn distributed_gmres(comm: &SimpleCommunicator) {
    let layout = layout(comm, 3, 5);
    let mat = tridiagonal(&layout, -1.0, 4.0, -2.0);
    let op = Operator::from(&mat);
    let solution = |g: usize| value_at(g, 1);

    let exact = Element::new(op.domain(), vector(&layout, solution));
    let rhs = op.dot(&exact);
    let mut x = op.domain().zero();
    let residual = GmresIteration::new(&op, &rhs, &mut x)
        .set_tol(1e-13)
        .set_restart(8)
        .set_max_iter(1000)
        .run();

    let mut check = op.range().copy_from(&rhs);
    op.apply(-1.0, &x, 1.0, &mut check);
    let rhs_norm = op.range().norm(&rhs);
    let check_norm = op.range().norm(&check);
    let ok = residual < 1e-13
        && check_norm <= 1e-11 * rhs_norm
        && all_close(local_values(x.imp()), expected(&layout, solution), 1e-8);

    collective_assert(
        comm,
        ok,
        "restarted GMRES must solve the distributed nonsymmetric system",
    );
}
