//! A distributed histogram sort with sampling.
//!
//! [`parsort`] sorts every rank's records into one global order and moves them so
//! that rank `r` of `P` receives the records of global positions `t_r..t_{r+1}`,
//! with `t_r = ⌊r N / P⌋` for `N` records in all: an exactly balanced output.
//!
//! The splitters are found by histogram refinement (histogram sort with sampling,
//! Harsh, Kalé and Solomonik, "Histogram Sort with Sampling", SPAA 2019). Every
//! splitter target `t_r` keeps a bracket, the closest records known below and above
//! it. In each round every rank draws random candidates from its own records inside
//! the open brackets, about [`OVERSAMPLING`] per unresolved target; the candidates are
//! gathered to every rank, and one all-reduce of their local ranks gives each
//! candidate's global position. A candidate at position `t_r` resolves target `r`;
//! the others narrow the brackets. A bracket holding at most `4 * OVERSAMPLING`
//! records is resolved exactly in the next round, because every record in it becomes
//! a candidate. Each round shrinks a bracket by about a factor of `OVERSAMPLING`, so
//! the number of rounds grows like `log(N / P) / log(OVERSAMPLING)`, and the data
//! gathered per round is about `OVERSAMPLING` records per unresolved target, whatever
//! the input distribution. In particular the result does not depend on how the
//! records are spread over the ranks: inputs whose local distributions all look
//! alike (a random share of one data set) are balanced as exactly as sorted ones.

use std::cmp::Ordering;

use itertools::Itertools;
use mpi::{
    collective::SystemOperation,
    traits::{CommunicatorCollectives, Equivalence},
};
use paste::paste;
use rand::{Rng, RngExt, seq::index};

use crate::{
    RlstResult, TotalCmp,
    distributed_tools::{
        all_to_allv,
        array_tools::{gather_to_all, global_size},
        sort_to_bins,
    },
};

/// The expected number of splitter candidates drawn per unresolved target in each
/// refinement round of [`parsort`].
pub const OVERSAMPLING: usize = 16;

/// A bracket holding at most this many records is resolved exactly: every record in
/// it becomes a candidate.
const EXACT_BRACKET: usize = 4 * OVERSAMPLING;

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
/// Records that compare equal keep the order of (input rank, input position). Rank
/// `r` of `P` receives exactly the records at global positions `⌊r N / P⌋` up to
/// `⌊(r + 1) N / P⌋`, `N` the number of records over all ranks, so every rank receives
/// `⌊N / P⌋` or `⌈N / P⌉` records whatever the input distribution. Empty local
/// inputs are valid, and with fewer records than ranks some ranks receive none.
///
/// The random number generator draws the splitter candidates; it changes the
/// communication, never the result.
///
/// # Collective operation
///
/// Every rank must call this function. With more than one rank and a nonempty
/// global input: one all-reduce of the input size; per refinement round (see the
/// [module documentation](self)), one gather of the candidates to every rank and one
/// all-reduce of their positions; then one all-to-all-v of the records.
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
    let total = global_size(arr, comm);
    if total == 0 {
        return Ok(Vec::new());
    }
    let mut unique = local
        .iter()
        .enumerate()
        .map(|(index, item)| Item::into_unique(*item, rank, index))
        .collect_vec();
    unique.sort_by(|left, right| left.total_cmp(*right));
    let bins = histogram_splitters(&unique, total, comm, rng);
    let counts = sort_to_bins(&unique, &bins);
    let mut received = all_to_allv(comm, &counts, &unique).1;
    received.sort_by(|left, right| left.total_cmp(*right));
    Ok(received.into_iter().map(Item::from_unique).collect())
}

/// Find the record at global position `⌊r N / P⌋` for every rank `r`, by histogram
/// refinement: one lower bound per rank, nondecreasing, the first the global minimum.
fn histogram_splitters<U, C, R>(arr: &[U], total: usize, comm: &C, rng: &mut R) -> Vec<U>
where
    U: Default + Copy + TotalCmp + Equivalence,
    C: CommunicatorCollectives,
    R: Rng + ?Sized,
{
    let mut brackets = initial_brackets(total, comm.size() as usize);
    while brackets.iter().any(|bracket| bracket.splitter.is_none()) {
        let local_candidates = choose_candidates(arr, &brackets, total, rng);
        let mut candidates = gather_to_all(&local_candidates, comm);
        candidates.sort_by(|left, right| left.total_cmp(*right));
        let local = local_positions(arr, &candidates);
        let mut positions = vec![0usize; candidates.len()];
        comm.all_reduce_into(&local[..], &mut positions[..], SystemOperation::sum());
        refine(&mut brackets, &candidates, &positions);
    }
    brackets
        .into_iter()
        .map(|bracket| bracket.splitter.expect("every bracket is resolved"))
        .collect()
}

/// The search state of one splitter target.
#[derive(Clone, Copy, Debug)]
struct Bracket<U> {
    /// The global position whose record is wanted.
    target: usize,
    /// The closest candidate known below the target, with its global position.
    lower: Option<(U, usize)>,
    /// The closest candidate known above the target, with its global position.
    upper: Option<(U, usize)>,
    /// The record at the target, once found.
    splitter: Option<U>,
    /// Doubles the candidates drawn in the bracket after a round without progress, so
    /// that the refinement terminates whatever the random draws.
    boost: usize,
}

impl<U: Copy + TotalCmp> Bracket<U> {
    /// The global positions strictly inside the bracket, as a half-open range.
    fn positions(&self, total: usize) -> (usize, usize) {
        let start = self.lower.map_or(0, |(_, position)| position + 1);
        let end = self.upper.map_or(total, |(_, position)| position);
        (start, end)
    }

    /// The local records strictly inside the bracket, as a range of `arr`.
    fn local_range(&self, arr: &[U]) -> (usize, usize) {
        let start = self.lower.map_or(0, |(value, _)| {
            arr.partition_point(|item| item.total_cmp(value) != Ordering::Greater)
        });
        let end = self.upper.map_or(arr.len(), |(value, _)| {
            arr.partition_point(|item| item.total_cmp(value) == Ordering::Less)
        });
        (start, end.max(start))
    }
}

/// One open bracket per rank, for the targets `⌊r N / P⌋`.
fn initial_brackets<U>(total: usize, size: usize) -> Vec<Bracket<U>> {
    (0..size)
        .map(|r| Bracket {
            target: (r as u128 * total as u128 / size as u128) as usize,
            lower: None,
            upper: None,
            splitter: None,
            boost: 1,
        })
        .collect()
}

/// This rank's candidates for one round: from each distinct open bracket, every local
/// record if the bracket holds at most [`EXACT_BRACKET`] records over all ranks, else a
/// random sample whose expected size over all ranks is [`OVERSAMPLING`] per target in
/// the bracket (times its boost).
fn choose_candidates<U, R>(arr: &[U], brackets: &[Bracket<U>], total: usize, rng: &mut R) -> Vec<U>
where
    U: Copy + TotalCmp,
    R: Rng + ?Sized,
{
    let mut candidates = Vec::new();
    // Open brackets with the same positions are the same interval; they are adjacent
    // because the targets are nondecreasing and the brackets nest in position order.
    let open = brackets
        .iter()
        .filter(|bracket| bracket.splitter.is_none())
        .collect_vec();
    for group in open.chunk_by(|left, right| left.positions(total) == right.positions(total)) {
        let (start, end) = group[0].positions(total);
        let inside = end - start;
        let (first, last) = group[0].local_range(arr);
        let local = last - first;
        if local == 0 {
            continue;
        }
        if inside <= EXACT_BRACKET {
            candidates.extend_from_slice(&arr[first..last]);
            continue;
        }
        let boost = group.iter().map(|bracket| bracket.boost).max().unwrap_or(1);
        let wanted = OVERSAMPLING
            .saturating_mul(group.len())
            .saturating_mul(boost);
        let fraction = (wanted as f64 / inside as f64).min(1.0);
        // Randomised rounding keeps the expected count exact on every rank.
        let draws = ((local as f64 * fraction + rng.random::<f64>()).floor() as usize).min(local);
        let mut picked = index::sample(rng, local, draws).into_vec();
        picked.sort_unstable();
        candidates.extend(picked.into_iter().map(|offset| arr[first + offset]));
    }
    candidates
}

/// For every candidate, the number of local records that sort before it.
fn local_positions<U: Copy + TotalCmp>(arr: &[U], candidates: &[U]) -> Vec<usize> {
    candidates
        .iter()
        .map(|&candidate| arr.partition_point(|item| item.total_cmp(candidate) == Ordering::Less))
        .collect()
}

/// Narrow every open bracket with the round's candidates (sorted, distinct) and their
/// global positions (strictly increasing).
fn refine<U: Copy + TotalCmp>(brackets: &mut [Bracket<U>], candidates: &[U], positions: &[usize]) {
    for bracket in brackets
        .iter_mut()
        .filter(|bracket| bracket.splitter.is_none())
    {
        let p = positions.partition_point(|&position| position < bracket.target);
        if p < positions.len() && positions[p] == bracket.target {
            bracket.splitter = Some(candidates[p]);
            continue;
        }
        let mut progress = false;
        if p > 0
            && bracket
                .lower
                .is_none_or(|(_, position)| positions[p - 1] > position)
        {
            bracket.lower = Some((candidates[p - 1], positions[p - 1]));
            progress = true;
        }
        if p < positions.len()
            && bracket
                .upper
                .is_none_or(|(_, position)| positions[p] < position)
        {
            bracket.upper = Some((candidates[p], positions[p]));
            progress = true;
        }
        if !progress {
            bracket.boost = bracket.boost.saturating_mul(2);
        }
    }
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

#[cfg(test)]
mod test {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    type Unique = <u64 as AsUnique>::Unique;

    /// Run the refinement on simulated ranks (one local array each) and return the
    /// splitters, the sorted local arrays and the number of rounds.
    fn simulate(ranks: &[Vec<u64>], seed: u64) -> (Vec<Unique>, Vec<Vec<Unique>>, usize) {
        let mut rng = StdRng::seed_from_u64(seed);
        let local: Vec<Vec<Unique>> = ranks
            .iter()
            .enumerate()
            .map(|(rank, values)| {
                let mut unique = values
                    .iter()
                    .enumerate()
                    .map(|(index, &value)| u64::into_unique(value, rank, index))
                    .collect_vec();
                unique.sort_by(|left, right| left.total_cmp(*right));
                unique
            })
            .collect();
        let total = local.iter().map(Vec::len).sum();
        let mut brackets = initial_brackets::<Unique>(total, ranks.len());
        let mut rounds = 0;
        while brackets.iter().any(|bracket| bracket.splitter.is_none()) {
            rounds += 1;
            assert!(rounds < 100, "the refinement must terminate");
            let mut candidates = local
                .iter()
                .flat_map(|arr| choose_candidates(arr, &brackets, total, &mut rng))
                .collect_vec();
            candidates.sort_by(|left, right| left.total_cmp(*right));
            let mut positions = vec![0usize; candidates.len()];
            for arr in &local {
                for (sum, position) in positions.iter_mut().zip(local_positions(arr, &candidates)) {
                    *sum += position;
                }
            }
            refine(&mut brackets, &candidates, &positions);
        }
        let splitters = brackets.iter().map(|b| b.splitter.unwrap()).collect();
        (splitters, local, rounds)
    }

    /// Check that every splitter is the record at its target position; return the
    /// number of rounds.
    fn check_exact(ranks: &[Vec<u64>], seed: u64) -> usize {
        let (splitters, local, rounds) = simulate(ranks, seed);
        let mut all = local.concat();
        all.sort_by(|left, right| left.total_cmp(*right));
        let size = ranks.len();
        for (r, splitter) in splitters.iter().enumerate() {
            let target = r * all.len() / size;
            assert!(
                all[target].total_cmp(*splitter) == Ordering::Equal,
                "splitter {r} is not the record at position {target}"
            );
        }
        rounds
    }

    fn uniform(rng: &mut StdRng, n: usize) -> Vec<u64> {
        (0..n).map(|_| rng.random::<u64>() >> 16).collect()
    }

    #[test]
    fn test_alike_local_distributions_are_balanced_exactly() {
        // Every rank holds a random share of one distribution: the case in which
        // splitters taken from local quantiles collapse onto a few values.
        let mut rng = StdRng::seed_from_u64(1);
        let ranks: Vec<Vec<u64>> = (0..72).map(|_| uniform(&mut rng, 2000)).collect();
        let rounds = check_exact(&ranks, 2);
        assert!(rounds <= 8, "{rounds} rounds");
    }

    #[test]
    fn test_sorted_and_skewed_inputs() {
        let mut rng = StdRng::seed_from_u64(3);
        // Each rank a contiguous range.
        let sorted: Vec<Vec<u64>> = (0..16)
            .map(|r| (0..500).map(|i| (r * 500 + i) as u64).collect())
            .collect();
        check_exact(&sorted, 4);
        // Every record on rank 0.
        let mut on_zero = vec![Vec::new(); 8];
        on_zero[0] = uniform(&mut rng, 10_000);
        check_exact(&on_zero, 5);
        // One heavy rank and many light ones.
        let mut skewed: Vec<Vec<u64>> = (0..32).map(|_| uniform(&mut rng, 3)).collect();
        skewed[7] = uniform(&mut rng, 50_000);
        check_exact(&skewed, 6);
    }

    #[test]
    fn test_duplicates_few_records_and_empty_ranks() {
        // Every value equal: the order comes from (rank, index) alone.
        let equal: Vec<Vec<u64>> = (0..12).map(|_| vec![7; 300]).collect();
        check_exact(&equal, 7);
        // Fewer records than ranks: repeated targets, empty buckets.
        let mut few = vec![Vec::new(); 8];
        few[3] = vec![5, 1, 4];
        few[6] = vec![2, 2];
        check_exact(&few, 8);
        // A single record.
        let mut one = vec![Vec::new(); 5];
        one[4] = vec![42];
        check_exact(&one, 9);
    }

    #[test]
    fn test_initial_brackets_targets() {
        let brackets = initial_brackets::<Unique>(10, 4);
        let targets = brackets.iter().map(|b| b.target).collect_vec();
        assert_eq!(targets, vec![0, 2, 5, 7]);
        let brackets = initial_brackets::<Unique>(3, 5);
        let targets = brackets.iter().map(|b| b.target).collect_vec();
        assert_eq!(targets, vec![0, 0, 1, 1, 2]);
    }
}
