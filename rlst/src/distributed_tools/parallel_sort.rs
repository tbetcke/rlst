//! An implementation of distributed sample sort.

use std::cmp::Ordering;

use itertools::Itertools;
use mpi::traits::{CommunicatorCollectives, Equivalence};
use paste::paste;
use rand::Rng;

use crate::{
    RlstResult, TotalCmp,
    distributed_tools::{
        all_to_allv,
        array_tools::{gather_to_all, global_min, global_size},
        sort_to_bins,
    },
};

/// Maximum number of weighted stratified representatives per nonempty rank.
pub const OVERSAMPLING: usize = 4;

/// A helper trait for parallel sorts to convert items into unique items.
pub trait AsUnique: TotalCmp + Clone + Copy {
    /// The unique output type.
    type Unique: Default + Copy + Clone + TotalCmp + Equivalence;
    /// Convert `self` into the unique type.
    fn into_unique(value: Self, rank: usize, index: usize) -> Self::Unique;
    /// Convert a unique item back into the original type.
    fn from_unique(elem: Self::Unique) -> Self;
}

/// Globally sort `arr` while retaining every input record and stable tie order.
///
/// Empty local inputs and empty output buckets are valid. The output balance is
/// approximate; no rank is guaranteed to receive an element.
pub fn parsort<Item: AsUnique, C: CommunicatorCollectives, R: Rng + ?Sized>(
    arr: &[Item],
    comm: &C,
    rng: &mut R,
) -> RlstResult<Vec<Item>> {
    let rank = comm.rank() as usize;
    let mut local = arr.to_vec();
    if comm.size() == 1 {
        local.sort_by(|left, right| left.total_cmp(*right));
        return Ok(local);
    }
    if global_size(arr, comm) == 0 {
        return Ok(Vec::new());
    }
    let mut unique = local
        .iter()
        .enumerate()
        .map(|(index, item)| Item::into_unique(*item, rank, index))
        .collect_vec();
    unique.sort_by(|left, right| left.total_cmp(*right));
    let bins = get_bins::<Item, _, _>(&unique, comm, rng);
    let counts = sort_to_bins(&unique, &bins);
    let mut received = all_to_allv(comm, &counts, &unique).1;
    received.sort_by(|left, right| left.total_cmp(*right));
    Ok(received.into_iter().map(Item::from_unique).collect())
}

/// Create exactly one lower bound per rank from weighted local representatives.
fn get_bins<Item, C, R>(arr: &[Item::Unique], comm: &C, _rng: &mut R) -> Vec<Item::Unique>
where
    Item: AsUnique,
    C: CommunicatorCollectives,
    R: Rng + ?Sized,
{
    let size = comm.size() as usize;
    let samples = arr.len().min(OVERSAMPLING);
    let mut local_values = Vec::with_capacity(samples);
    let mut local_weights = Vec::with_capacity(samples);
    for sample in 0..samples {
        let start = sample * arr.len() / samples;
        let end = (sample + 1) * arr.len() / samples;
        local_values.push(arr[start]);
        local_weights.push(end - start);
    }
    let values = gather_to_all(&local_values, comm);
    let weights = gather_to_all(&local_weights, comm);
    let mut weighted = values.into_iter().zip(weights).collect_vec();
    weighted.sort_by(|(left, _), (right, _)| left.total_cmp(*right));
    let minimum = match global_min(arr, comm) {
        Some(value) => value,
        None => unreachable!("global input was checked nonempty"),
    };
    let total_weight: usize = weighted.iter().map(|(_, weight)| weight).sum();
    debug_assert!(total_weight > 0);
    let mut bounds = Vec::with_capacity(size);
    bounds.push(minimum);
    for bucket in 1..size {
        let target = bucket * total_weight / size;
        let mut accumulated = 0;
        let mut bound = weighted.last().expect("nonempty global samples").0;
        for &(value, weight) in &weighted {
            accumulated += weight;
            if accumulated >= target {
                bound = value;
                break;
            }
        }
        bounds.push(bound);
    }
    bounds
}

macro_rules! impl_as_unique_for {
    ($dtype:ty) => { paste! {
        #[derive(Copy, Clone, PartialOrd, PartialEq, Equivalence, Default)]
        #[allow(non_camel_case_types, missing_docs)]
        pub struct [<UniqueItem_ $dtype>] { pub value: $dtype, pub rank: usize, pub index: usize }
        impl TotalCmp for [<UniqueItem_$dtype>] {
            fn total_cmp(self, other: Self) -> Ordering {
                match self.value.total_cmp(other.value) {
                    Ordering::Equal => match self.rank.cmp(&other.rank) {
                        Ordering::Equal => self.index.cmp(&other.index), order => order,
                    }, order => order,
                }
            }
        }
        impl AsUnique for $dtype {
            type Unique = [<UniqueItem_$dtype>];
            fn into_unique(value: $dtype, rank: usize, index: usize) -> Self::Unique { Self::Unique { value, rank, index } }
            fn from_unique(elem: Self::Unique) -> Self { elem.value }
        }
    }};
}
impl_as_unique_for!(f64);
impl_as_unique_for!(f32);
impl_as_unique_for!(i8);
impl_as_unique_for!(i16);
impl_as_unique_for!(i32);
impl_as_unique_for!(i64);
impl_as_unique_for!(isize);
impl_as_unique_for!(u8);
impl_as_unique_for!(u16);
impl_as_unique_for!(u32);
impl_as_unique_for!(u64);
impl_as_unique_for!(usize);
