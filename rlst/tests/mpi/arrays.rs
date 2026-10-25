//! Distributed arrays on uneven layouts with an empty rank.
//!
//! Every rank fills its local rows from [`value_at`] of the global row, so the expected value
//! of every local entry is known without communication.

use std::rc::Rc;

use mpi::{topology::SimpleCommunicator, traits::Communicator};
use num::traits::MulAdd;
use rlst::{
    DynArray, Inner, c64, distributed_tools::IndexLayout,
    sparse::distributed_array::DistributedArray,
};

use crate::common::{
    Case, Layout, TOL, all_close, close, collective_assert, complex_value_at, expected, filled,
    global_array, local_values, multi_indices, panic_message, panics, uneven_layout, value_at,
    vector,
};

/// The cases of this module.
pub const CASES: &[Case] = &[
    ("dist-array-scatter-gather", dist_array_scatter_gather),
    ("dist-array-arithmetic", dist_array_arithmetic),
    ("dist-array-state", dist_array_state),
    ("dist-array-reductions", dist_array_reductions),
    ("dist-array-incompatible-ops", dist_array_incompatible_ops),
];

/// Scatter `f` of `trailing` columns from every root and gather it back in every way.
fn scatter_gather_rank<const N: usize>(
    comm: &SimpleCommunicator,
    layout: &Layout<'_>,
    trailing: [usize; N],
) -> bool {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let mut shape = trailing;
    shape[0] = layout.number_of_global_indices();
    let global = global_array(shape, value_at);
    let reference = filled(layout, trailing, value_at);
    let mut ok = true;

    for root in [0, size - 1] {
        let scattered = if rank == root {
            global.scatter_from_one_root(layout.clone())
        } else {
            DynArray::<f64, N>::scatter_from_one(root, layout.clone())
        };
        ok &= scattered.index_layout.is_same(layout)
            && scattered.local.shape() == reference.local.shape()
            && local_values(&scattered) == local_values(&reference);

        if rank == root {
            let gathered = reference.gather_to_one_root();
            ok &= gathered.shape() == shape && gathered.iter_value().eq(global.iter_value());
        } else {
            reference.gather_to_one(root);
        }
    }
    let everywhere = reference.gather_to_all();
    ok && everywhere.shape() == shape && everywhere.iter_value().eq(global.iter_value())
}

/// Contract: arrays of rank 1, 2 and 3 scatter from the first and last rank to the owners of
/// their rows, and gather back exactly to one rank or to all.
///
/// The checks of `examples/distributed_array.rs`, on a layout with an empty rank.
fn dist_array_scatter_gather(comm: &SimpleCommunicator) {
    let layout = uneven_layout(comm, 0);
    // Each call is collective, so none may be skipped after a failure on some rank.
    let mut ok = scatter_gather_rank(comm, &layout, [0]);
    ok &= scatter_gather_rank(comm, &layout, [0, 3]);
    ok &= scatter_gather_rank(comm, &layout, [0, 2, 3]);
    collective_assert(
        comm,
        ok,
        "scatter and gather must place every row on its owner and back",
    );
}

/// Contract: every elementwise operation of distributed arrays computes the local values of
/// the corresponding operation of their entries.
fn dist_array_arithmetic(comm: &SimpleCommunicator) {
    let layout = uneven_layout(comm, 1);
    let f = |g: usize| value_at(g, 0);
    let h = |g: usize| value_at(g, 1);
    let x = vector(&layout, f);
    let y = vector(&layout, h);
    let e = |op: &dyn Fn(usize) -> f64| expected(&layout, op);
    let mut ok = true;

    ok &= all_close(
        local_values(&(x.r() + y.r()).eval()),
        e(&|g| f(g) + h(g)),
        0.0,
    );
    ok &= all_close(
        local_values(&(x.r() - y.r()).eval()),
        e(&|g| f(g) - h(g)),
        0.0,
    );
    ok &= all_close(
        local_values(&(x.r() * y.r()).eval()),
        e(&|g| f(g) * h(g)),
        0.0,
    );
    ok &= all_close(
        local_values(&(x.r() / y.r()).eval()),
        e(&|g| f(g) / h(g)),
        0.0,
    );
    ok &= all_close(local_values(&(2.5 * x.r()).eval()), e(&|g| 2.5 * f(g)), 0.0);
    ok &= all_close(local_values(&(x.r() * 2.5).eval()), e(&|g| 2.5 * f(g)), 0.0);
    ok &= all_close(
        local_values(&x.r().scalar_mul(3.0).eval()),
        e(&|g| 3.0 * f(g)),
        0.0,
    );
    ok &= all_close(local_values(&(-x.r()).eval()), e(&|g| -f(g)), 0.0);
    ok &= all_close(
        local_values(&x.r().mul_add(2.0, y.r()).eval()),
        e(&|g| 2.0 * f(g) + h(g)),
        0.0,
    );
    ok &= all_close(
        local_values(&(x.r() + y.r()).eval_row_major()),
        e(&|g| f(g) + h(g)),
        0.0,
    );

    // Unary operations.
    ok &= all_close(local_values(&(-x.r()).abs().eval()), e(&f), 0.0);
    ok &= all_close(
        local_values(&x.r().square().eval()),
        e(&|g| f(g) * f(g)),
        TOL,
    );
    ok &= all_close(local_values(&x.r().sqrt().eval()), e(&|g| f(g).sqrt()), TOL);
    ok &= all_close(local_values(&x.r().exp().eval()), e(&|g| f(g).exp()), TOL);
    ok &= all_close(local_values(&x.r().ln().eval()), e(&|g| f(g).ln()), TOL);
    ok &= all_close(
        local_values(&x.r().recip().eval()),
        e(&|g| f(g).recip()),
        TOL,
    );
    ok &= all_close(local_values(&x.r().sin().eval()), e(&|g| f(g).sin()), TOL);
    ok &= all_close(local_values(&x.r().cos().eval()), e(&|g| f(g).cos()), TOL);
    ok &= all_close(local_values(&x.r().tanh().eval()), e(&|g| f(g).tanh()), TOL);
    ok &= all_close(
        local_values(&x.r().unary_op(|v: f64| v * v + 1.0).eval()),
        e(&|g| f(g) * f(g) + 1.0),
        0.0,
    );

    // Conversions.
    ok &= local_values(&x.r().cast::<f32>().eval()) == expected(&layout, |g| f(g) as f32);
    ok &= local_values(&x.r().into_type::<c64>().eval())
        == expected(&layout, |g| c64::new(f(g), 0.0));
    ok &= x
        .r()
        .coerce_dim::<1>()
        .is_ok_and(|same| local_values(&same.eval()) == e(&f));
    ok &= x.r().coerce_dim::<2>().is_err();

    // In-place operations.
    let mut z = x.r().eval();
    z += y.r();
    ok &= all_close(local_values(&z), e(&|g| f(g) + h(g)), 0.0);
    z -= y.r();
    ok &= all_close(local_values(&z), e(&f), TOL);
    z *= 2.0;
    ok &= all_close(local_values(&z), e(&|g| 2.0 * f(g)), TOL);
    z /= 4.0;
    ok &= all_close(local_values(&z), e(&|g| 0.5 * f(g)), TOL);

    // Complex entries, and a matrix.
    let xc = vector(&layout, |g| complex_value_at(g, 0));
    let yc = vector(&layout, |g| complex_value_at(g, 1));
    let s = c64::new(0.5, -2.0);
    ok &= all_close(
        local_values(&(s * xc.r() + xc.r() * yc.r()).eval()),
        expected(&layout, |g| {
            s * complex_value_at(g, 0) + complex_value_at(g, 0) * complex_value_at(g, 1)
        }),
        TOL,
    );
    ok &= all_close(
        local_values(&xc.r().abs().eval()),
        expected(&layout, |g| complex_value_at(g, 0).norm()),
        TOL,
    );
    let a = filled(&layout, [0, 3], value_at);
    let b = filled(&layout, [0, 3], |g, c| value_at(g, c + 1));
    let sum = (a.r() - 0.5 * b.r()).eval();
    let shape = sum.local.shape();
    ok &= multi_indices(shape).into_iter().all(|index| {
        let g = layout.local_range().0 + index[0];
        close(
            sum.local[index],
            value_at(g, index[1]) - 0.5 * value_at(g, index[1] + 1),
            0.0,
        )
    });

    collective_assert(
        comm,
        ok,
        "elementwise operations on distributed arrays must act on every local entry",
    );
}

/// Contract: construction, views, fills, compatibility, shapes and column iterators of
/// distributed arrays.
fn dist_array_state(comm: &SimpleCommunicator) {
    let layout = uneven_layout(comm, 2);
    let total = layout.number_of_global_indices();
    let local = layout.number_of_local_indices();
    let f = |g: usize| value_at(g, 0);
    let x = vector(&layout, f);
    let mut ok = true;

    // `new` checks the local row count on every rank.
    ok &= panics(|| {
        DistributedArray::new(layout.clone(), DynArray::<f64, 1>::from_shape([local + 1]))
    });

    // Views and fills.
    let mut a = x.r().eval();
    a.r_mut().fill_with_value(7.0);
    ok &= local_values(&a) == vec![7.0; local];
    a.fill_from(&x.r());
    ok &= local_values(&a) == expected(&layout, f);
    a.fill_with_value(-1.0);
    ok &= local_values(&a) == vec![-1.0; local];

    // Compatibility needs the same layout handle and the same global shape.
    let copy: Layout = Rc::new((*layout).clone());
    let same_counts = vector(&copy, f);
    ok &= x.is_compatible_with(&a.r()) && !x.is_compatible_with(&same_counts);
    let m3 = filled(&layout, [0, 3], value_at);
    let m2 = filled(&layout, [0, 2], value_at);
    ok &= m3.is_compatible_with(&m3.r()) && !m3.is_compatible_with(&m2);

    // Global shapes.
    ok &= x.shape() == [total] && x.len() == total && !x.is_empty();
    ok &= m3.shape() == [total, 3] && m3.len() == 3 * total && !m3.is_empty();
    let no_columns = filled(&layout, [0, 0], value_at);
    ok &= no_columns.shape() == [total, 0] && no_columns.is_empty();
    let nothing: Layout = Rc::new(IndexLayout::from_local_counts(0, comm));
    let empty = vector(&nothing, f);
    ok &= empty.shape() == [0] && empty.is_empty();

    // Column iterators.
    ok &= m3.col_iter().count() == 3
        && m3
            .col_iter()
            .enumerate()
            .all(|(c, column)| local_values(&column) == expected(&layout, |g| value_at(g, c)));
    let mut m = m3.r().eval();
    for (c, mut column) in m.col_iter_mut().enumerate() {
        column.fill_with_value(c as f64);
    }
    ok &= multi_indices(m.local.shape())
        .into_iter()
        .all(|index| m.local[index] == index[1] as f64);

    collective_assert(
        comm,
        ok,
        "distributed arrays must report their state and give views of their columns",
    );
}

/// Contract: global reductions of distributed arrays equal the reductions of all entries,
/// and are `None` for an empty array.
fn dist_array_reductions(comm: &SimpleCommunicator) {
    let layout = uneven_layout(comm, 3);
    let total = layout.number_of_global_indices();
    let f = |g: usize| value_at(g, 0);
    let x = vector(&layout, f);
    let all = |op: &dyn Fn(usize) -> f64| (0..total).map(op).collect::<Vec<_>>();
    let mut ok = true;

    ok &= x
        .sum()
        .is_some_and(|sum| close(sum, all(&f).iter().sum(), 0.0));
    ok &= (-x.r())
        .max_abs()
        .is_some_and(|max| max == all(&f).iter().copied().fold(0.0, f64::max));
    ok &= x
        .norm_2()
        .is_some_and(|norm| close(norm, all(&|g| f(g) * f(g)).iter().sum::<f64>().sqrt(), TOL));
    // The check of `examples/distributed_array_operations.rs`.
    ok &= (5.0 * x.r().sin()).norm_2().is_some_and(|norm| {
        close(
            norm,
            all(&|g| 25.0 * f(g).sin() * f(g).sin())
                .iter()
                .sum::<f64>()
                .sqrt(),
            TOL,
        )
    });
    let m = filled(&layout, [0, 3], value_at);
    ok &= m.sum().is_some_and(|sum| {
        close(
            sum,
            (0..3)
                .flat_map(|c| (0..total).map(move |g| value_at(g, c)))
                .sum(),
            0.0,
        )
    });

    // Complex inner products conjugate the second argument.
    let xc = vector(&layout, |g| complex_value_at(g, 0));
    let yc = vector(&layout, |g| complex_value_at(g, 1));
    let inner: c64 = (0..total)
        .map(|g| complex_value_at(g, 0) * complex_value_at(g, 1).conj())
        .sum();
    ok &= xc
        .inner(&yc.r())
        .is_some_and(|value| close(value, inner, TOL));
    ok &= close(Inner::inner(&xc, &yc), inner, TOL);
    ok &= xc
        .sum()
        .is_some_and(|sum| close(sum, (0..total).map(|g| complex_value_at(g, 0)).sum(), TOL));
    ok &= xc.norm_2().is_some_and(|norm| {
        close(
            norm,
            (0..total)
                .map(|g| complex_value_at(g, 0).norm_sqr())
                .sum::<f64>()
                .sqrt(),
            TOL,
        )
    });
    ok &= xc.max_abs().is_some_and(|max| {
        close(
            max,
            (0..total)
                .map(|g| complex_value_at(g, 0).norm())
                .fold(0.0, f64::max),
            TOL,
        )
    });

    // Every reduction of a globally empty array is `None`.
    let nothing: Layout = Rc::new(IndexLayout::from_local_counts(0, comm));
    let empty = vector(&nothing, f);
    ok &= empty.sum().is_none();
    ok &= empty.max_abs().is_none();
    ok &= empty.norm_2().is_none();
    ok &= empty.inner(&empty.r()).is_none();
    ok &= Inner::inner(&empty, &empty) == 0.0;

    collective_assert(
        comm,
        ok,
        "reductions of distributed arrays must combine the entries of every rank",
    );
}

/// Contract: operations on arrays with different layout handles panic on every rank, and
/// name the operation.
fn dist_array_incompatible_ops(comm: &SimpleCommunicator) {
    let layout = uneven_layout(comm, 0);
    let copy: Layout = Rc::new((*layout).clone());
    let f = |g: usize| value_at(g, 0);
    let x = vector(&layout, f);
    let z = vector(&copy, f);
    let names = |message: Option<String>, name: &str| {
        message.is_some_and(|message| message.contains(&format!("DistributedArray::{name}:")))
    };
    let mut ok = true;

    ok &= names(panic_message(|| x.r() + z.r()), "add");
    ok &= names(panic_message(|| x.r() - z.r()), "sub");
    ok &= names(panic_message(|| x.r() * z.r()), "mul");
    ok &= names(panic_message(|| x.r() / z.r()), "div");
    ok &= names(panic_message(|| x.r().mul_add(2.0, z.r())), "mul_add");
    ok &= names(panic_message(|| x.inner(&z.r())), "inner");
    ok &= names(
        panic_message(|| {
            let mut a = x.r().eval();
            a += z.r();
        }),
        "add_assign",
    );
    ok &= names(
        panic_message(|| {
            let mut a = x.r().eval();
            a -= z.r();
        }),
        "sub_assign",
    );
    ok &= names(
        panic_message(|| {
            let mut a = x.r().eval();
            a.fill_from(&z.r());
        }),
        "fill_from",
    );

    collective_assert(
        comm,
        ok,
        "operations on incompatible distributed arrays must panic with their own name",
    );
}
