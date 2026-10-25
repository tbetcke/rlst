//! Distributed CSR matrices and lazy sparse operations, checked against dense references.
//!
//! Every rank generates the same random triplets from a seed and contributes a share of them
//! that does not depend on which rank owns their rows. The dense sum of all triplets, known on
//! every rank, is the reference.

use std::{collections::BTreeSet, rc::Rc};

use mpi::{
    topology::{CommunicatorRelation, SimpleCommunicator},
    traits::{Communicator, Equivalence, Root},
};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rlst::{
    Abs, AijIteratorByValue, AijIteratorMut, AsMatrixApply, DynArray, FromAijDistributed, Nonzeros,
    RlstScalar, Shape, SparseMatrixType, Sqrt, Square, c64,
    distributed_tools::IndexLayout,
    io::matrix_market::{read_array_mm, read_coordinate_mm},
    sparse::{
        SparseMatType, distributed_csr_mat::DistributedCsrMatrix,
        mat_operations_distributed::DistributedSparseMatOpIterator,
    },
};

use crate::common::{
    Case, DistArray, Layout, all_close, close, collective_assert, filled, local_values, panics,
    uneven_count, vector,
};

/// The cases of this module.
pub const CASES: &[Case] = &[
    ("csr-random-matvec", csr_random_matvec),
    ("csr-new-direct", csr_new_direct),
    ("csr-matrix-market", csr_matrix_market),
    ("sparse-mat-ops", sparse_mat_ops),
];

type Triplets<T> = Vec<([usize; 2], T)>;

/// An uneven layout with three times the indices of [`uneven_count`].
fn layout(comm: &SimpleCommunicator, salt: usize) -> Layout<'_> {
    let count = uneven_count(comm.rank() as usize, comm.size() as usize, salt);
    Rc::new(IndexLayout::from_local_counts(3 * count, comm))
}

/// A real value in eighths.
fn real(a: i32, _: i32) -> f64 {
    f64::from(a) / 8.0
}

/// A complex value in eighths.
fn complex(a: i32, b: i32) -> c64 {
    c64::new(f64::from(a) / 8.0, f64::from(b) / 8.0)
}

/// The triplets of a random matrix of `shape`, the same on every rank, with repeated
/// positions and pairs of entries that cancel.
fn random_triplets<T: RlstScalar>(
    shape: [usize; 2],
    seed: u64,
    make: fn(i32, i32) -> T,
) -> Triplets<T> {
    let mut rng = StdRng::seed_from_u64(seed);
    let position =
        |rng: &mut StdRng| [rng.random_range(0..shape[0]), rng.random_range(0..shape[1])];
    let mut triplets = Vec::new();
    for k in 0..3 * shape[0] {
        let at = position(&mut rng);
        let value = make(rng.random_range(-16..16), rng.random_range(-16..16));
        triplets.push((at, value));
        if k % 4 == 0 {
            let other = make(rng.random_range(-16..16), rng.random_range(-16..16));
            triplets.push((at, other));
        }
        if k % 5 == 0 {
            let at = position(&mut rng);
            triplets.push((at, value));
            triplets.push((at, -value));
        }
    }
    triplets
}

/// The row-major dense matrix of `triplets`.
fn dense_of<T: RlstScalar>(shape: [usize; 2], triplets: &Triplets<T>) -> Vec<T> {
    let mut dense = vec![T::zero(); shape[0] * shape[1]];
    for &([row, col], value) in triplets {
        dense[row * shape[1] + col] += value;
    }
    dense
}

/// The triplets that this rank contributes: every `size`-th, starting at its rank.
fn share<T: Copy>(comm: &SimpleCommunicator, triplets: &Triplets<T>) -> Triplets<T> {
    let (rank, size) = (comm.rank() as usize, comm.size() as usize);
    triplets
        .iter()
        .enumerate()
        .filter(|(k, _)| k % size == rank)
        .map(|(_, &triplet)| triplet)
        .collect()
}

/// Assemble the matrix of `triplets` from the shares of all ranks.
fn assemble<'a, T: RlstScalar + Equivalence>(
    comm: &SimpleCommunicator,
    domain: &Layout<'a>,
    range: &Layout<'a>,
    triplets: &Triplets<T>,
) -> DistributedCsrMatrix<'a, T, SimpleCommunicator> {
    let mine = share(comm, triplets);
    let rows: Vec<usize> = mine.iter().map(|(at, _)| at[0]).collect();
    let cols: Vec<usize> = mine.iter().map(|(at, _)| at[1]).collect();
    let data: Vec<T> = mine.iter().map(|&(_, value)| value).collect();
    DistributedCsrMatrix::from_aij(domain.clone(), range.clone(), &rows, &cols, &data)
}

/// Whether the local rows of `mat` hold exactly the nonzero entries of `dense`, listed in
/// row-major order by `iter_aij_value`, and its global shape and number of nonzeros agree.
///
/// This is collective. Like every collective in these cases, it must not sit behind a `&&`
/// whose left side can differ between ranks.
fn matches_dense<T: RlstScalar + Equivalence>(
    mat: &DistributedCsrMatrix<'_, T, SimpleCommunicator>,
    shape: [usize; 2],
    dense: &[T],
    tol: f64,
) -> bool {
    let nnz = mat.nnz();
    let (start, end) = mat.range_layout().local_range();
    let actual: Triplets<T> = mat.iter_aij_value().collect();
    let expected: Triplets<T> = (start..end)
        .flat_map(|row| (0..shape[1]).map(move |col| [row, col]))
        .map(|at| (at, dense[at[0] * shape[1] + at[1]]))
        .filter(|&(_, value)| value != T::zero())
        .collect();
    let global_nonzeros = dense.iter().filter(|&&value| value != T::zero()).count();
    mat.shape() == shape
        && nnz == global_nonzeros
        && actual.len() == expected.len()
        && actual
            .iter()
            .zip(&expected)
            .all(|(a, e)| a.0 == e.0 && close(a.1, e.1, tol))
}

/// `alpha * dense * x + beta * y` for the local rows of `range`, column `c` of `x` and `y`.
#[allow(clippy::too_many_arguments)]
fn dense_apply<T: RlstScalar>(
    range: &Layout<'_>,
    shape: [usize; 2],
    dense: &[T],
    alpha: T,
    x: impl Fn(usize) -> T,
    beta: T,
    y: impl Fn(usize) -> T,
) -> Vec<T> {
    let (start, end) = range.local_range();
    (start..end)
        .map(|row| {
            let product = (0..shape[1])
                .map(|col| dense[row * shape[1] + col] * x(col))
                .fold(T::zero(), |sum, term| sum + term);
            alpha * product + beta * y(row)
        })
        .collect()
}

/// The checks of `csr-random-matvec` for one scalar type.
fn random_matvec_for<T: RlstScalar + Equivalence>(
    comm: &SimpleCommunicator,
    make: fn(i32, i32) -> T,
    alpha: T,
    beta: T,
) -> bool {
    let domain = layout(comm, 4);
    let range = layout(comm, 5);
    let shape = [
        range.number_of_global_indices(),
        domain.number_of_global_indices(),
    ];
    let triplets = random_triplets(shape, 2024, make);
    let dense = dense_of(shape, &triplets);
    let mat = assemble(comm, &domain, &range, &triplets);
    let from_iter = DistributedCsrMatrix::from_aij_iter(
        domain.clone(),
        range.clone(),
        share(comm, &triplets).into_iter(),
    );
    let mut ok = matches_dense(&mat, shape, &dense, 0.0);
    ok &= matches_dense(&from_iter, shape, &dense, 0.0);
    ok &= matches!(mat.mat_type(), SparseMatType::DistCsr);

    // Index maps: local domain columns first, then the ghost columns of the local rows.
    let (dstart, dend) = domain.local_range();
    let owned = dend - dstart;
    let ghosts: BTreeSet<usize> = mat
        .iter_aij_value()
        .map(|(at, _)| at[1])
        .filter(|col| !(dstart..dend).contains(col))
        .collect();
    let local_columns = owned + ghosts.len();
    ok &= (0..owned).all(|l| {
        mat.domain_local2global(l) == Some(dstart + l)
            && mat.domain_global2local(dstart + l) == Some(l)
    });
    ok &= (owned..local_columns)
        .map(|l| mat.domain_local2global(l))
        .collect::<Option<BTreeSet<usize>>>()
        == Some(ghosts.clone());
    ok &= (0..local_columns).all(|l| {
        mat.domain_local2global(l)
            .is_some_and(|g| mat.domain_global2local(g) == Some(l))
    });
    ok &= mat.domain_local2global(local_columns).is_none();
    ok &= (0..shape[1])
        .filter(|g| !(dstart..dend).contains(g) && !ghosts.contains(g))
        .all(|g| mat.domain_global2local(g).is_none());
    ok &= mat.local().shape() == [range.number_of_local_indices(), local_columns];
    let (rstart, rend) = range.local_range();
    ok &= (0..shape[0])
        .all(|g| mat.range_global2local(g) == (rstart..rend).contains(&g).then(|| g - rstart));
    ok &= (0..rend - rstart).all(|i| mat.range_local2global(i) == Some(rstart + i))
        && mat.range_local2global(rend - rstart).is_none();

    // Products with a vector and with three columns.
    let xf = |g: usize, c: usize| make(g as i32 % 7 - 3 + c as i32, 2 - (g as i32 % 3));
    let yf = |g: usize, c: usize| make(1 + g as i32 % 5, c as i32 - 1);
    let tol = 1e-12;
    let x = vector(&domain, |g| xf(g, 0));
    let mut y = vector(&range, |g| yf(g, 0));
    mat.apply(alpha, &x, beta, &mut y);
    ok &= all_close(
        local_values(&y),
        dense_apply(
            &range,
            shape,
            &dense,
            alpha,
            |g| xf(g, 0),
            beta,
            |g| yf(g, 0),
        ),
        tol,
    );
    let product: DistArray<T, 1> = rlst::dot!(mat, x.r());
    ok &= product.index_layout.is_same(&range)
        && all_close(
            local_values(&product),
            dense_apply(
                &range,
                shape,
                &dense,
                T::one(),
                |g| xf(g, 0),
                T::zero(),
                |_| T::zero(),
            ),
            tol,
        );

    let xs = filled(&domain, [0, 3], xf);
    let mut ys = filled(&range, [0, 3], yf);
    mat.apply(alpha, &xs, beta, &mut ys);
    let products = mat.dot(&xs);
    for c in 0..3 {
        let column = |arr: &DistArray<T, 2>| {
            (0..arr.local.shape()[0])
                .map(|i| arr.local[[i, c]])
                .collect::<Vec<_>>()
        };
        ok &= all_close(
            column(&ys),
            dense_apply(
                &range,
                shape,
                &dense,
                alpha,
                |g| xf(g, c),
                beta,
                |g| yf(g, c),
            ),
            tol,
        );
        ok &= all_close(
            column(&products),
            dense_apply(
                &range,
                shape,
                &dense,
                T::one(),
                |g| xf(g, c),
                T::zero(),
                |_| T::zero(),
            ),
            tol,
        );
    }
    ok
}

/// Contract: a random non-square matrix, contributed by ranks that do not own its rows, with
/// repeated and cancelling entries, assembles to the sum of its triplets on uneven layouts; its
/// index maps describe the local and ghost columns; and its products with vectors and
/// matrices equal the dense products.
fn csr_random_matvec(comm: &SimpleCommunicator) {
    let mut ok = random_matvec_for(comm, real, 1.5, -0.5);
    ok &= random_matvec_for(comm, complex, c64::new(0.5, 1.0), c64::new(-1.0, 0.25));
    collective_assert(
        comm,
        ok,
        "distributed CSR matrices must assemble and multiply like their dense references",
    );
}

/// Contract: `DistributedCsrMatrix::new` from global column indices gives the matrix of
/// `from_aij`, and changes of the local matrix change its products.
fn csr_new_direct(comm: &SimpleCommunicator) {
    let domain = layout(comm, 1);
    let range = layout(comm, 2);
    let shape = [
        range.number_of_global_indices(),
        domain.number_of_global_indices(),
    ];
    let triplets = random_triplets(shape, 99, real);
    let dense = dense_of(shape, &triplets);

    let (start, end) = range.local_range();
    let mut indices = Vec::new();
    let mut indptr = vec![0];
    let mut data = Vec::new();
    for row in start..end {
        for col in 0..shape[1] {
            let value = dense[row * shape[1] + col];
            if value != 0.0 {
                indices.push(col);
                data.push(value);
            }
        }
        indptr.push(indices.len());
    }
    let mut mat = DistributedCsrMatrix::new(
        DynArray::from(indices),
        DynArray::from(indptr),
        DynArray::from(data),
        domain.clone(),
        range.clone(),
    );
    let mut ok = matches_dense(&mat, shape, &dense, 0.0);
    ok &= mat.domain_layout().is_same(&domain)
        && mat.range_layout().is_same(&range)
        && mat.comm().compare(comm) == CommunicatorRelation::Identical;

    let xf = |g: usize| real(g as i32 % 5 - 2, 0);
    let x = vector(&domain, xf);
    let mut y = vector(&range, |_| 0.0);
    mat.apply(1.0, &x, 0.0, &mut y);
    ok &= all_close(
        local_values(&y),
        dense_apply(&range, shape, &dense, 1.0, xf, 0.0, |_| 0.0),
        0.0,
    );

    for (_, value) in mat.local_mut().iter_aij_mut() {
        *value *= 2.0;
    }
    mat.apply(1.0, &x, 0.0, &mut y);
    ok &= all_close(
        local_values(&y),
        dense_apply(&range, shape, &dense, 2.0, xf, 0.0, |_| 0.0),
        0.0,
    );

    collective_assert(
        comm,
        ok,
        "a matrix built from local CSR data must equal the assembled matrix",
    );
}

/// The path of a file of the examples.
fn example_file(name: &str) -> String {
    format!("{}/examples/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// Contract: the 507 x 313 matrix of the examples, read on rank 0 and distributed over
/// equidistributed layouts, multiplies its vector to the stored result.
///
/// The check of `examples/distributed_csr_matrix.rs`.
fn csr_matrix_market(comm: &SimpleCommunicator) {
    let root = comm.process_at_rank(0);
    let read = comm.rank() == 0;
    let matrix = read.then(|| read_coordinate_mm::<f64>(&example_file("mat_507_313.mm")).unwrap());
    let mut shape = matrix.as_ref().map_or([0, 0], |matrix| matrix.shape());
    root.broadcast_into(shape.as_mut_slice());

    let domain: Layout = Rc::new(IndexLayout::from_equidistributed_chunks(shape[1], 1, comm));
    let range: Layout = Rc::new(IndexLayout::from_equidistributed_chunks(shape[0], 1, comm));
    let mat = match &matrix {
        Some(matrix) => DistributedCsrMatrix::from_aij_iter(
            domain.clone(),
            range.clone(),
            matrix.iter_aij_value(),
        ),
        None => {
            DistributedCsrMatrix::<f64, _>::from_aij(domain.clone(), range.clone(), &[], &[], &[])
        }
    };
    let x = if read {
        let x = read_array_mm::<f64>(&example_file("x_313.mm")).unwrap();
        x.r().slice(1, 0).scatter_from_one_root(domain.clone())
    } else {
        DynArray::<f64, 1>::scatter_from_one(0, domain.clone())
    };

    let mut y = rlst::dist_vec!(f64, range.clone());
    mat.apply(1.0, &x, 0.0, &mut y);
    let gathered = y.gather_to_all();
    let expected = read_array_mm::<f64>(&example_file("y_507.mm")).unwrap();
    let ok = mat.nnz() == 793
        && mat.shape() == [507, 313]
        && gathered.shape() == [507]
        && gathered
            .iter_value()
            .zip(expected.r().slice::<1>(1, 0).iter_value())
            .all(|(actual, expected)| close(actual, expected, 1e-10));

    collective_assert(
        comm,
        ok,
        "the Matrix Market example must multiply to its stored result",
    );
}

/// The lazy operator of the entries of `mat`.
#[allow(clippy::type_complexity)]
fn op<'m, 'a>(
    mat: &'m DistributedCsrMatrix<'a, f64, SimpleCommunicator>,
    domain: &Layout<'a>,
) -> DistributedSparseMatOpIterator<
    'a,
    f64,
    impl Iterator<Item = ([usize; 2], f64)> + 'm,
    SimpleCommunicator,
> {
    DistributedSparseMatOpIterator::new(mat.iter_aij_value(), domain.clone(), mat.range_layout())
}

/// Contract: sums, differences, products and maps of lazy sparse operators assemble to the
/// corresponding dense results, and operators on different layouts are rejected.
fn sparse_mat_ops(comm: &SimpleCommunicator) {
    let domain = layout(comm, 3);
    let range = layout(comm, 0);
    let shape = [
        range.number_of_global_indices(),
        domain.number_of_global_indices(),
    ];
    let a_triplets = random_triplets(shape, 11, real);
    let b_triplets = random_triplets(shape, 12, real);
    let a = assemble(comm, &domain, &range, &a_triplets);
    let b = assemble(comm, &domain, &range, &b_triplets);
    let da = dense_of(shape, &a_triplets);
    let db = dense_of(shape, &b_triplets);
    let combine = |f: &dyn Fn(f64, f64) -> f64| -> Vec<f64> {
        da.iter().zip(&db).map(|(&x, &y)| f(x, y)).collect()
    };
    let stored = |f: &dyn Fn(f64) -> f64| -> Vec<f64> {
        da.iter()
            .map(|&x| if x != 0.0 { f(x) } else { 0.0 })
            .collect()
    };
    let check = |mat: DistributedCsrMatrix<'_, f64, SimpleCommunicator>, dense: Vec<f64>| {
        matches_dense(&mat, shape, &dense, 1e-14)
            && mat.domain_layout().is_same(&domain)
            && mat.range_layout().is_same(&range)
    };
    let mut ok = true;

    ok &= check(
        (op(&a, &domain) + op(&b, &domain)).into_distributed_csr(),
        combine(&|x, y| x + y),
    );
    ok &= check(
        (op(&a, &domain) - op(&b, &domain)).into_distributed_csr(),
        combine(&|x, y| x - y),
    );
    ok &= check(
        (op(&a, &domain) * op(&b, &domain)).into_distributed_csr(),
        combine(&|x, y| x * y),
    );
    ok &= check(
        (2.0 * op(&a, &domain)).into_distributed_csr(),
        stored(&|x| 2.0 * x),
    );
    ok &= check(
        (op(&a, &domain) * -0.5).into_distributed_csr(),
        stored(&|x| -0.5 * x),
    );
    ok &= check(
        op(&a, &domain)
            .map_stored(|x| x * x + 1.0)
            .into_distributed_csr(),
        stored(&|x| x * x + 1.0),
    );
    ok &= check(
        Abs::abs(op(&a, &domain)).into_distributed_csr(),
        stored(&f64::abs),
    );
    ok &= check(
        Sqrt::sqrt(Square::square(op(&a, &domain))).into_distributed_csr(),
        stored(&f64::abs),
    );
    ok &= check(
        (op(&a, &domain) + 3.0 * op(&b, &domain).map_stored(f64::exp)).into_distributed_csr(),
        da.iter()
            .zip(&db)
            .map(|(&x, &y)| x + if y != 0.0 { 3.0 * y.exp() } else { 0.0 })
            .collect(),
    );

    let lazy = op(&a, &domain);
    ok &= lazy.shape() == shape
        && lazy.domain_layout().is_same(&domain)
        && lazy.range_layout().is_same(&range);

    // A different domain handle is rejected, even with the same ranges.
    let copy: Layout = Rc::new((*domain).clone());
    ok &= panics(|| op(&a, &domain) + op(&b, &copy));
    ok &= panics(|| op(&a, &domain) * op(&b, &copy));

    collective_assert(
        comm,
        ok,
        "lazy sparse operations must assemble to their dense results",
    );
}
