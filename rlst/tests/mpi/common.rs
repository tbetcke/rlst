//! Helpers shared by the cases.

use std::{panic::AssertUnwindSafe, rc::Rc};

use mpi::{topology::SimpleCommunicator, traits::CommunicatorCollectives};
use rlst::{
    DynArray, RlstScalar, Shape, UnsafeRandom1DAccessByValue, c64,
    dense::{base_array::BaseArray, data_container::VectorContainer},
    distributed_tools::IndexLayout,
    sparse::distributed_array::DistributedArray,
};

/// A named case: a function that every rank calls with the world communicator.
pub type Case = (&'static str, fn(&SimpleCommunicator));

/// Make every rank observe an assertion result before this case exits.
pub fn collective_assert<C: CommunicatorCollectives>(comm: &C, local: bool, message: &str) {
    let mut global = false;
    comm.all_reduce_into(
        &local,
        &mut global,
        mpi::collective::SystemOperation::logical_and(),
    );
    assert!(global, "{message}");
}

/// Whether `f` panics. The panic message still reaches standard error.
pub fn panics<R>(f: impl FnOnce() -> R) -> bool {
    std::panic::catch_unwind(AssertUnwindSafe(f)).is_err()
}

/// The message of the panic of `f`, or `None` if it does not panic.
pub fn panic_message<R>(f: impl FnOnce() -> R) -> Option<String> {
    let payload = std::panic::catch_unwind(AssertUnwindSafe(f)).err()?;
    Some(
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default(),
    )
}

/// A positive real value determined by a global index and a column.
///
/// The values are multiples of 1/8, so they and their sums are exact in floating point.
pub fn value_at(index: usize, column: usize) -> f64 {
    1.0 + 0.5 * index as f64 + 0.25 * column as f64 + 0.125 * ((7 * index + 3 * column) % 5) as f64
}

/// A complex value determined by a global index and a column, with a nonzero imaginary part
/// of either sign.
pub fn complex_value_at(index: usize, column: usize) -> c64 {
    c64::new(
        value_at(index, column),
        0.375 * index as f64 - 1.0 - 0.5 * column as f64 + 0.0625,
    )
}

/// Whether `actual` is within `tol` of `expected`, relative to `1 + |expected|`.
pub fn close<T: RlstScalar>(actual: T, expected: T, tol: f64) -> bool {
    RlstScalar::abs(actual - expected) <= T::real(tol) * (T::real(1.0) + RlstScalar::abs(expected))
}

/// Whether the two sequences have the same length and are elementwise [`close`].
pub fn all_close<T: RlstScalar>(
    actual: impl IntoIterator<Item = T>,
    expected: impl IntoIterator<Item = T>,
    tol: f64,
) -> bool {
    let actual: Vec<T> = actual.into_iter().collect();
    let expected: Vec<T> = expected.into_iter().collect();
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(&expected)
            .all(|(&actual, &expected)| close(actual, expected, tol))
}

/// The number of local indices of rank `rank` in an uneven layout over `size` ranks.
///
/// With several ranks, rank `(salt + 1) % size` holds no index; the others hold between one
/// and seven. Different salts give different layouts.
pub fn uneven_count(rank: usize, size: usize, salt: usize) -> usize {
    if size > 1 && rank == (salt + 1) % size {
        0
    } else {
        1 + (3 * rank + 5 * salt) % 7
    }
}

/// The local counts of all ranks of [`uneven_count`].
pub fn uneven_counts(size: usize, salt: usize) -> Vec<usize> {
    (0..size)
        .map(|rank| uneven_count(rank, size, salt))
        .collect()
}

/// The layout of [`uneven_count`] on `comm`.
pub fn uneven_layout<C: CommunicatorCollectives>(comm: &C, salt: usize) -> Rc<IndexLayout<'_, C>> {
    Rc::new(IndexLayout::from_local_counts(
        uneven_count(comm.rank() as usize, comm.size() as usize, salt),
        comm,
    ))
}

/// A shared index layout on the world communicator.
pub type Layout<'a> = Rc<IndexLayout<'a, SimpleCommunicator>>;
/// An owned distributed array.
pub type DistArray<'a, T, const N: usize> =
    DistributedArray<'a, SimpleCommunicator, BaseArray<VectorContainer<T>, N>, N>;

/// The tolerance of results computed in a different order than their reference.
pub const TOL: f64 = 1e-13;

/// All multi-indices of `shape`, in column-major order.
pub fn multi_indices<const N: usize>(shape: [usize; N]) -> Vec<[usize; N]> {
    let count: usize = shape.iter().product();
    (0..count)
        .map(|mut linear| {
            let mut index = [0; N];
            for (entry, &extent) in index.iter_mut().zip(&shape) {
                *entry = linear % extent;
                linear /= extent;
            }
            index
        })
        .collect()
}

/// The column of a multi-index: its trailing indices, linearized column-major.
pub fn column_of<const N: usize>(index: [usize; N], shape: [usize; N]) -> usize {
    index[1..]
        .iter()
        .zip(&shape[1..])
        .rev()
        .fold(0, |column, (&i, &extent)| column * extent + i)
}

/// A distributed array whose entry with global row `g` and column `c` is `f(g, c)`.
pub fn filled<'a, T: Copy + Default, const N: usize>(
    layout: &Layout<'a>,
    trailing: [usize; N],
    f: impl Fn(usize, usize) -> T,
) -> DistArray<'a, T, N> {
    let mut shape = trailing;
    shape[0] = layout.number_of_local_indices();
    let start = layout.local_range().0;
    let mut local = DynArray::<T, N>::from_shape(shape);
    for index in multi_indices(shape) {
        local[index] = f(start + index[0], column_of(index, shape));
    }
    DistributedArray::new(layout.clone(), local)
}

/// A distributed vector with entries `f(g)`.
pub fn vector<'a, T: Copy + Default>(
    layout: &Layout<'a>,
    f: impl Fn(usize) -> T,
) -> DistArray<'a, T, 1> {
    filled(layout, [0], |g, _| f(g))
}

/// The local entries of `arr`, in column-major order.
pub fn local_values<T, A, const N: usize>(
    arr: &DistributedArray<'_, SimpleCommunicator, A, N>,
) -> Vec<T>
where
    A: UnsafeRandom1DAccessByValue<Item = T> + Shape<N>,
{
    arr.local.iter_value().collect()
}

/// The expected local entries of a vector with entries `f(g)`.
pub fn expected<T>(layout: &Layout<'_>, f: impl Fn(usize) -> T) -> Vec<T> {
    let (start, end) = layout.local_range();
    (start..end).map(f).collect()
}

/// The entries `f(g, c)` of a global array of `shape`, as a local array.
pub fn global_array<const N: usize>(
    shape: [usize; N],
    f: impl Fn(usize, usize) -> f64,
) -> DynArray<f64, N> {
    let mut array = DynArray::<f64, N>::from_shape(shape);
    for index in multi_indices(shape) {
        array[index] = f(index[0], column_of(index, shape));
    }
    array
}
