//! Array tools
//!
//! This module contains tools for working with distributed arrays.

use std::ops::Add;

use itertools::Itertools;
use mpi::{
    collective::{SystemOperation, UserOperation},
    datatype::PartitionMut,
    traits::{CommunicatorCollectives, Equivalence, Root},
};
use num::Zero;

use crate::{Max, Min, TotalCmp};

/// Distribute a sorted sequence into bins.
///
/// `bins` contains the inclusive lower bound for each half-open bucket; the
/// final bucket has no upper bound. Empty keys and empty bins produce no
/// buckets. Nonempty keys require nonempty, nondecreasing bounds and must not
/// precede the first bound. Repeated bounds deliberately produce empty buckets.
pub fn sort_to_bins<T: TotalCmp>(sorted_keys: &[T], bins: &[T]) -> Vec<usize> {
    if bins.is_empty() {
        assert!(
            sorted_keys.is_empty(),
            "nonempty keys require at least one bin"
        );
        return Vec::new();
    }
    assert!(
        bins.windows(2).all(|window| window[0].le(window[1])),
        "bin lower bounds must be nondecreasing"
    );
    if let Some(first) = sorted_keys.first() {
        assert!(bins[0].le(*first), "key precedes the first bin lower bound");
    }
    assert!(
        sorted_keys.windows(2).all(|window| window[0].le(window[1])),
        "keys passed to sort_to_bins must be sorted"
    );

    let mut counts = vec![0; bins.len()];
    for key in sorted_keys {
        // The last lower bound not greater than key selects its bucket. This
        // deliberately selects the final of repeated lower bounds.
        let bucket = bins.partition_point(|bound| bound.le(*key)) - 1;
        counts[bucket] += 1;
    }
    counts
}

/// Compute displacements on n processes from a vector of n counts of local indices.
pub fn displacements(counts: &[i32]) -> Vec<i32> {
    checked_displacements(counts).expect("counts and displacements must fit in i32")
}

fn checked_displacements(counts: &[i32]) -> Option<Vec<i32>> {
    let mut total = 0_i32;
    counts
        .iter()
        .map(|&count| {
            if count < 0 {
                return None;
            }
            let displacement = total;
            total += count;
            Some(displacement)
        })
        .collect()
}

fn collective_valid<C: CommunicatorCollectives>(comm: &C, valid: bool, message: &str) {
    let mut all_valid = false;
    comm.all_reduce_into(&valid, &mut all_valid, SystemOperation::logical_and());
    assert!(all_valid, "{message}");
}

fn collective_root_valid<C: CommunicatorCollectives>(comm: &C, root: usize, message: &str) {
    let mut roots = vec![0_usize; comm.size() as usize];
    comm.all_gather_into(&root, &mut roots);
    collective_valid(
        comm,
        root < comm.size() as usize && roots.iter().all(|&candidate| candidate == root),
        message,
    );
}

fn checked_counts_for_data(
    counts: &[usize],
    process_count: usize,
    data_len: usize,
) -> Option<Vec<i32>> {
    let converted = counts
        .iter()
        .map(|&count| i32::try_from(count))
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    (counts.len() == process_count && counts.iter().copied().sum::<usize>() == data_len)
        .then_some(converted)
}

fn checked_counts<C: CommunicatorCollectives>(
    comm: &C,
    counts: &[usize],
    data_len: usize,
) -> Vec<i32> {
    let converted = checked_counts_for_data(counts, comm.size() as usize, data_len);
    collective_valid(
        comm,
        converted.is_some(),
        "invalid all-to-all counts or data length",
    );
    let converted = converted.expect("collectively validated counts");
    collective_valid(
        comm,
        checked_displacements(&converted).is_some(),
        "all-to-all displacements overflow i32",
    );
    converted
}

/// Performs an all-to-all communication.
pub fn all_to_allv<T: Equivalence + Default + Clone, C: CommunicatorCollectives>(
    comm: &C,
    counts: &[usize],
    data: &[T],
) -> (Vec<usize>, Vec<T>) {
    let counts = checked_counts(comm, counts, data.len());
    let mut recv_counts = vec![0_i32; comm.size() as usize];
    comm.all_to_all_into(&counts, &mut recv_counts);
    collective_valid(
        comm,
        checked_displacements(&recv_counts).is_some(),
        "received counts or displacements overflow i32",
    );
    let receive_len = recv_counts
        .iter()
        .map(|&count| usize::try_from(count).expect("validated nonnegative count"))
        .sum::<usize>();
    collective_valid(comm, true, "received count total is valid");
    let mut receive_data = vec![T::default(); receive_len];
    let send_displacements = checked_displacements(&counts).expect("validated displacements");
    let receive_displacements =
        checked_displacements(&recv_counts).expect("validated displacements");
    let send_partition =
        mpi::datatype::Partition::new(data, counts.as_slice(), send_displacements.as_slice());
    let mut receive_partition = PartitionMut::new(
        &mut receive_data,
        recv_counts.as_slice(),
        receive_displacements.as_slice(),
    );
    comm.all_to_all_varcount_into(&send_partition, &mut receive_partition);
    (
        recv_counts
            .into_iter()
            .map(|count| count as usize)
            .collect(),
        receive_data,
    )
}

/// Scatter data across processes from the calling root process.
pub fn scatterv_root<T: Equivalence + Default + Clone, C: CommunicatorCollectives>(
    comm: &C,
    counts: &[usize],
    out_data: &[T],
) -> Vec<T> {
    collective_root_valid(
        comm,
        comm.rank() as usize,
        "scatter root must agree on every rank",
    );
    let counts_total = counts.iter().copied().sum::<usize>();
    let mut valid = counts.len() == comm.size() as usize
        && counts.iter().all(|&count| i32::try_from(count).is_ok())
        && counts_total == out_data.len();
    let send_counts = counts.iter().map(|&count| count as i32).collect_vec();
    valid &= checked_displacements(&send_counts).is_some();
    // The matching non-root helper receives this root-only validation before
    // either rank enters the scatter payload collective.
    comm.this_process().broadcast_into(&mut valid);
    assert!(
        valid,
        "invalid scatter counts, displacements, or data length"
    );
    let rank = comm.rank() as usize;
    let mut recv_count = 0_i32;
    comm.this_process()
        .scatter_into_root(&send_counts, &mut recv_count);
    let mut recvbuf = vec![T::default(); recv_count as usize];
    let displacements = checked_displacements(&send_counts).expect("validated displacements");
    let send_partition =
        mpi::datatype::Partition::new(out_data, send_counts.as_slice(), displacements.as_slice());
    comm.this_process()
        .scatter_varcount_into_root(&send_partition, &mut recvbuf);
    debug_assert_eq!(recvbuf.len(), send_counts[rank] as usize);
    recvbuf
}

/// Receive data scattered from `root`.
pub fn scatterv<T: Equivalence + Default + Clone>(
    comm: &impl CommunicatorCollectives,
    root: usize,
) -> Vec<T> {
    collective_root_valid(comm, root, "scatter root must agree and be in range");
    let root_process = comm.process_at_rank(root as i32);
    let mut valid = false;
    root_process.broadcast_into(&mut valid);
    assert!(
        valid,
        "invalid scatter counts, displacements, or data length"
    );
    let mut recv_count = 0_i32;
    root_process.scatter_into(&mut recv_count);
    assert!(recv_count >= 0, "MPI returned a negative scatter count");
    let mut recvbuf = vec![T::default(); recv_count as usize];
    root_process.scatter_varcount_into(&mut recvbuf);
    recvbuf
}

/// Get the minimum value across all ranks, or `None` when every rank is empty.
pub fn global_min<T: Equivalence + Default + Copy + Min<Output = T>, C: CommunicatorCollectives>(
    arr: &[T],
    comm: &C,
) -> Option<T> {
    global_extremum(arr, comm, |left, right| left.min(right))
}

/// Get the maximum value across all ranks, or `None` when every rank is empty.
pub fn global_max<T: Equivalence + Default + Copy + Max<Output = T>, C: CommunicatorCollectives>(
    arr: &[T],
    comm: &C,
) -> Option<T> {
    global_extremum(arr, comm, |left, right| left.max(right))
}

fn global_extremum<T: Equivalence + Default + Copy, C: CommunicatorCollectives>(
    arr: &[T],
    comm: &C,
    operation: impl Fn(T, T) -> T + Copy + Sync,
) -> Option<T> {
    let local = arr.iter().copied().reduce(operation);
    let occupied = !arr.is_empty();
    let mut occupancy = vec![false; comm.size() as usize];
    comm.all_gather_into(&occupied, &mut occupancy);
    let seed_rank = occupancy.iter().position(|&present| present)?;
    // Default only initializes storage on non-seed ranks before broadcast; it
    // is never used as an extremum sentinel.
    let mut seed = local.unwrap_or_default();
    comm.process_at_rank(seed_rank as i32)
        .broadcast_into(&mut seed);
    let candidate = local.unwrap_or(seed);
    let mut result = candidate;
    comm.all_reduce_into(
        &candidate,
        &mut result,
        &UserOperation::commutative(move |x, y| {
            let x: &[T] = x.downcast().expect("MPI datatype must be T");
            let y: &mut [T] = y.downcast().expect("MPI datatype must be T");
            for (&value, output) in x.iter().zip(y) {
                *output = operation(value, *output);
            }
        }),
    );
    Some(result)
}

/// Get global size of a distributed array.
pub fn global_size<T, C: CommunicatorCollectives>(arr: &[T], comm: &C) -> usize {
    let mut global_size = 0;
    comm.all_reduce_into(&arr.len(), &mut global_size, SystemOperation::sum());
    global_size
}

/// Gather a distributed array to a specific rank.
pub fn gather_to_rank<T: Equivalence + Default + Clone, C: CommunicatorCollectives>(
    arr: &[T],
    root: usize,
    comm: &C,
) -> Option<Vec<T>> {
    collective_root_valid(comm, root, "gather root must agree and be in range");
    collective_valid(
        comm,
        i32::try_from(arr.len()).is_ok(),
        "local gather count exceeds i32",
    );
    let count = arr.len() as i32;
    // All ranks learn the counts before the payload phase, so overflow is
    // rejected collectively rather than leaving a root panic to strand peers.
    let mut counts = vec![0_i32; comm.size() as usize];
    comm.all_gather_into(&count, &mut counts);
    let len = counts
        .iter()
        .map(|&count| usize::try_from(count).expect("validated nonnegative count"))
        .sum::<usize>();
    collective_valid(
        comm,
        checked_displacements(&counts).is_some(),
        "gather count displacement or receive length overflow",
    );
    let root_process = comm.process_at_rank(root as i32);
    if comm.rank() as usize == root {
        let mut data = vec![T::default(); len];
        let displacements = checked_displacements(&counts).expect("validated");
        let mut partition =
            PartitionMut::new(&mut data, counts.as_slice(), displacements.as_slice());
        root_process.gather_varcount_into_root(arr, &mut partition);
        Some(data)
    } else {
        root_process.gather_varcount_into(arr);
        None
    }
}

/// Gather an array to all processes.
pub fn gather_to_all<T: Equivalence + Default + Clone, C: CommunicatorCollectives>(
    arr: &[T],
    comm: &C,
) -> Vec<T> {
    collective_valid(
        comm,
        i32::try_from(arr.len()).is_ok(),
        "local gather count exceeds i32",
    );
    let mut counts = vec![0_i32; comm.size() as usize];
    comm.all_gather_into(&(arr.len() as i32), &mut counts);
    collective_valid(
        comm,
        checked_displacements(&counts).is_some(),
        "gather count displacement overflow",
    );
    let len = counts
        .iter()
        .map(|&count| usize::try_from(count).expect("validated nonnegative count"))
        .sum::<usize>();
    let mut data = vec![T::default(); len];
    let displacements = checked_displacements(&counts).expect("validated");
    let mut partition = PartitionMut::new(&mut data, counts.as_slice(), displacements.as_slice());
    comm.all_gather_varcount_into(arr, &mut partition);
    data
}

/// Perform a global inclusive cumulative sum operation.
pub fn global_inclusive_cumsum<
    T: Equivalence + Zero + Copy + Add<Output = T>,
    C: CommunicatorCollectives,
>(
    arr: &[T],
    comm: &C,
) -> Vec<T> {
    let mut scan = arr
        .iter()
        .scan(T::zero(), |sum, value| {
            *sum = *sum + *value;
            Some(*sum)
        })
        .collect_vec();
    let local_total = scan.last().copied().unwrap_or_else(T::zero);
    let mut offset = T::zero();
    comm.exclusive_scan_into(&local_total, &mut offset, SystemOperation::sum());
    if comm.rank() == 0 {
        offset = T::zero();
    }
    for value in &mut scan {
        *value = *value + offset;
    }
    scan
}

/// Return the first value on the next nonempty rank, if one exists.
pub fn communicate_back<T: Equivalence + Default + Clone, C: CommunicatorCollectives>(
    arr: &[T],
    comm: &C,
) -> Option<T> {
    let occupied = !arr.is_empty();
    let mut occupancy = vec![false; comm.size() as usize];
    comm.all_gather_into(&occupied, &mut occupancy);
    let first_values = gather_to_all(&arr[..std::cmp::min(arr.len(), 1)], comm);
    let mut value_index = 0;
    let mut first_by_rank = vec![None; comm.size() as usize];
    for (rank, &present) in occupancy.iter().enumerate() {
        if present {
            first_by_rank[rank] = Some(first_values[value_index].clone());
            value_index += 1;
        }
    }
    first_by_rank
        .into_iter()
        .skip(comm.rank() as usize + 1)
        .flatten()
        .next()
}

/// Check whether a distributed array is sorted.
pub fn is_sorted_array<T: Equivalence + Default + Clone + TotalCmp, C: CommunicatorCollectives>(
    arr: &[T],
    comm: &C,
) -> bool {
    let local_sorted = arr.windows(2).all(|pair| pair[0].le(pair[1]));
    let boundary_sorted = match (arr.last(), communicate_back(arr, comm)) {
        (Some(last), Some(next)) => last.le(next),
        _ => true,
    };
    let mut sorted = false;
    comm.all_reduce_into(
        &(local_sorted && boundary_sorted),
        &mut sorted,
        SystemOperation::logical_and(),
    );
    sorted
}

#[cfg(test)]
mod test {
    use super::{checked_counts_for_data, checked_displacements};
    use crate::distributed_tools::{displacements, sort_to_bins};

    #[test]
    fn test_sort_to_bins() {
        assert_eq!(sort_to_bins::<i32>(&[], &[]), []);
        assert_eq!(sort_to_bins(&[1, 1, 2], &[1, 1, 2]), [0, 2, 1]);
    }

    #[test]
    #[should_panic(expected = "nonempty keys require at least one bin")]
    fn sort_to_bins_rejects_nonempty_keys_without_bins() {
        sort_to_bins(&[1_i32], &[]);
    }

    #[test]
    #[should_panic(expected = "bin lower bounds must be nondecreasing")]
    fn sort_to_bins_rejects_descending_bounds() {
        sort_to_bins(&[], &[1_i32, 0]);
    }

    #[test]
    #[should_panic(expected = "key precedes the first bin lower bound")]
    fn sort_to_bins_rejects_keys_before_first_bound() {
        sort_to_bins(&[0_i32], &[1]);
    }

    #[test]
    fn test_displacements() {
        assert_eq!(displacements(&[3, 4, 5]), [0, 3, 7]);
        assert_eq!(checked_displacements(&[-1]), None);
        assert_eq!(checked_displacements(&[i32::MAX, 1]), None);
        let too_large_for_mpi = i32::MAX as usize + 1;
        assert_eq!(
            checked_counts_for_data(&[too_large_for_mpi], 1, too_large_for_mpi),
            None
        );
    }
}
