//! Multi-rank regression contracts for empty local MPI partitions.
//!
//! This executable deliberately has `harness = false`: `scripts/run_mpi_tests.py`
//! launches each case in its own bounded MPI process group.  Keeping cases isolated
//! means that a pre-fix panic or deadlock cannot prevent the remaining contracts
//! from being reported.

use std::{cmp::Ordering, env, panic::AssertUnwindSafe, rc::Rc};

use mpi::{
    topology::{Color, SimpleCommunicator},
    traits::{Communicator, CommunicatorCollectives, Equivalence},
};
use rand::{SeedableRng, rngs::StdRng};
use rlst::distributed_tools::{
    ChunkSizes, DataPermutation, GhostCommunicatorBuilder, Global2LocalDataMapper, IndexLayout,
    all_to_allv,
    array_tools::{
        communicate_back, gather_to_all, gather_to_rank, global_inclusive_cumsum, global_max,
        global_min, is_sorted_array, scatterv, scatterv_root,
    },
    index_embedding::IndexEmbedding,
    parallel_sort::{AsUnique, parsort},
};
use rlst::{AsMatrixApply, FromAijDistributed, InnerProductSpace, LinearSpace, TotalCmp};

#[derive(Copy, Clone, Default, Equivalence, Debug, Eq, PartialEq)]
struct SortRecord {
    key: i32,
    payload: i32,
}

#[derive(Copy, Clone, Default, Equivalence)]
struct UniqueSortRecord {
    record: SortRecord,
    rank: usize,
    index: usize,
}

impl TotalCmp for SortRecord {
    fn total_cmp(self, other: Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

impl TotalCmp for UniqueSortRecord {
    fn total_cmp(self, other: Self) -> Ordering {
        match self.record.key.cmp(&other.record.key) {
            Ordering::Equal => match self.rank.cmp(&other.rank) {
                Ordering::Equal => self.index.cmp(&other.index),
                order => order,
            },
            order => order,
        }
    }
}

impl AsUnique for SortRecord {
    type Unique = UniqueSortRecord;

    fn into_unique(value: Self, rank: usize, index: usize) -> Self::Unique {
        UniqueSortRecord {
            record: value,
            rank,
            index,
        }
    }

    fn from_unique(elem: Self::Unique) -> Self {
        elem.record
    }
}

/// Make every rank observe an assertion result before this case exits.
fn collective_assert<C: CommunicatorCollectives>(comm: &C, local: bool, message: &str) {
    let mut global = false;
    comm.all_reduce_into(
        &local,
        &mut global,
        mpi::collective::SystemOperation::logical_and(),
    );
    assert!(global, "{message}");
}

/// Contract: all zero-count variable collectives retain all ranks and return empty buffers.
fn zero_count_collectives<C: CommunicatorCollectives>(comm: &C) {
    let empty: [i32; 0] = [];
    let gathered = gather_to_all(&empty, comm);
    let rooted = gather_to_rank(&empty, 0, comm);
    let (received_counts, received) = all_to_allv(comm, &vec![0; comm.size() as usize], &empty);
    let self_traffic = vec![comm.rank()];
    let (_, self_received) = all_to_allv(
        comm,
        &(0..comm.size() as usize)
            .map(|rank| usize::from(rank == comm.rank() as usize))
            .collect::<Vec<_>>(),
        &self_traffic,
    );
    let scattered = if comm.rank() == 0 {
        scatterv_root(comm, &vec![0; comm.size() as usize], &empty)
    } else {
        scatterv::<i32>(comm, 0)
    };

    collective_assert(
        comm,
        gathered.is_empty()
            && received.is_empty()
            && received_counts.iter().all(|&count| count == 0)
            && self_received == [comm.rank()]
            && scattered.is_empty()
            && if comm.rank() == 0 {
                rooted.is_some_and(|values| values.is_empty())
            } else {
                rooted.is_none()
            },
        "zero-count collectives must complete with empty buffers on every rank",
    );
}

/// Contract: empty local inputs contribute an additive identity to the global cumsum.
fn empty_cumsum<C: CommunicatorCollectives>(comm: &C) {
    let input: Vec<i32> = (0..comm.rank() + 1).collect();
    let result = global_inclusive_cumsum(&input, comm);
    let serial = gather_to_all(&input, comm);
    let expected = serial
        .iter()
        .scan(0, |sum, value| {
            *sum += value;
            Some(*sum)
        })
        .collect::<Vec<_>>();
    let result_all = gather_to_all(&result, comm);
    let empty: [i32; 0] = [];
    let empty_result = global_inclusive_cumsum(&empty, comm);
    collective_assert(
        comm,
        result_all == expected && empty_result.is_empty(),
        "cumsums must match a serial rank-ordered oracle and accept empty partitions",
    );
}

/// Contract: boundaries skip empty ranks; an all-empty distributed sequence is sorted.
fn empty_boundaries<C: CommunicatorCollectives>(comm: &C) {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let mut all_masks_ok = true;
    for mask in 0..(1_usize << size) {
        let local = if mask & (1 << rank) != 0 {
            vec![rank as i32 * 10]
        } else {
            Vec::new()
        };
        let expected = ((rank + 1)..size)
            .find(|next| mask & (1 << next) != 0)
            .map(|next| next as i32 * 10);
        let sorted = is_sorted_array(&local, comm);
        all_masks_ok &= communicate_back(&local, comm) == expected;
        all_masks_ok &= sorted;
    }
    let empty: [i32; 0] = [];

    let empty_sorted = is_sorted_array(&empty, comm);
    let unsorted = if rank == 0 {
        vec![1_i32, 0]
    } else {
        Vec::new()
    };
    let unsorted_sorted = is_sorted_array(&unsorted, comm);
    collective_assert(
        comm,
        all_masks_ok && empty_sorted && !unsorted_sorted,
        "boundary exchange and sortedness must accept every empty-rank occupancy mask",
    );
}

/// Contract: extrema are computed from nonempty peers even when this rank is empty.
fn empty_extrema<C: CommunicatorCollectives>(comm: &C) {
    let local = if comm.rank() % 2 == 0 {
        vec![i32::MIN + comm.rank(), i32::MAX - comm.rank()]
    } else {
        Vec::new()
    };
    let all = gather_to_all(&local, comm);
    let empty: [i32; 0] = [];
    let minimum_matches = global_min(&local, comm) == all.iter().copied().min();
    let maximum_matches = global_max(&local, comm) == all.iter().copied().max();
    let empty_minimum = global_min(&empty, comm).is_none();
    let empty_maximum = global_max(&empty, comm).is_none();
    collective_assert(
        comm,
        minimum_matches && maximum_matches && empty_minimum && empty_maximum,
        "global extrema must match serial negative/extreme-value oracles and handle global emptiness",
    );
}

/// Contract: sorting preserves all records, stable equal-key payload order, and a generous balance.
fn sparse_sort<C: CommunicatorCollectives>(comm: &C) {
    let rank = comm.rank() as usize;
    let mut rng = StdRng::seed_from_u64(0);
    let mut masks_ok = true;
    for mask in 0..(1_usize << comm.size()) {
        let local = if mask & (1 << rank) == 0 {
            Vec::new()
        } else {
            vec![
                SortRecord {
                    key: 2,
                    payload: (rank * 10) as i32,
                },
                SortRecord {
                    key: 1,
                    payload: (rank * 10 + 1) as i32,
                },
                SortRecord {
                    key: 1,
                    payload: (rank * 10 + 2) as i32,
                },
            ]
        };
        let mut expected = gather_to_all(&local, comm);
        // Stable serial sorting retains rank/local input order for equal keys.
        expected.sort_by_key(|record| record.key);
        let sorted = parsort(&local, comm, &mut rng).expect("sort occupancy mask");
        masks_ok &= gather_to_all(&sorted, comm) == expected;
    }

    let mut lone_ok = true;
    for owner in 0..comm.size() as usize {
        let lone_local = if rank == owner {
            vec![SortRecord {
                key: 3,
                payload: owner as i32,
            }]
        } else {
            Vec::new()
        };
        let lone_expected = gather_to_all(&lone_local, comm);
        let lone_sorted = parsort(&lone_local, comm, &mut rng).expect("sort lone record");
        lone_ok &= gather_to_all(&lone_sorted, comm) == lone_expected;
    }

    let all_identical = vec![SortRecord {
        key: 7,
        payload: rank as i32,
    }];
    let mut identical_expected = gather_to_all(&all_identical, comm);
    identical_expected.sort_by_key(|record| record.key);
    let identical = parsort(&all_identical, comm, &mut rng).expect("sort identical keys");
    let identical_ok = gather_to_all(&identical, comm) == identical_expected;

    // Deliberately unequal nonzero populations exercise weighted samples rather
    // than treating every occupied rank as an equally weighted representative.
    let uneven_local = (0..(1_usize << rank))
        .map(|index| SortRecord {
            key: ((rank * 13 + index * 7) % 31) as i32,
            payload: (rank * 10_000 + index) as i32,
        })
        .collect::<Vec<_>>();
    let mut uneven_expected = gather_to_all(&uneven_local, comm);
    uneven_expected.sort_by_key(|record| record.key);
    let uneven = parsort(&uneven_local, comm, &mut rng).expect("sort uneven data");
    let uneven_gathered = gather_to_all(&uneven, comm);
    let local_sizes = gather_to_all(&[uneven.len()], comm);
    let average = uneven_expected.len().div_ceil(comm.size() as usize);
    let balance_ok = local_sizes.iter().all(|&count| count <= 2 * average + 4);
    let globally_empty = parsort::<i32, _, _>(&[], comm, &mut rng).expect("sort empty input");

    collective_assert(
        comm,
        masks_ok
            && lone_ok
            && identical_ok
            && globally_empty.is_empty()
            && uneven_gathered == uneven_expected
            && balance_ok,
        "parallel sorting must match stable serial payload oracles across occupancy masks and remain approximately balanced",
    );
}

/// Contract: mapper and permutation exchanges preserve off-rank data in both directions.
fn mapper_and_permutation<C: CommunicatorCollectives>(comm: &C) {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let layout = Rc::new(IndexLayout::from_local_counts(1, comm));
    let requested = [(rank + 1) % size];
    let mapper = Global2LocalDataMapper::new(layout.clone(), &requested, 2);
    let mapped = mapper.map_data(&[rank as i32, -(rank as i32)]);
    let permutation = DataPermutation::new(layout.clone(), &requested, 2);
    let mut forward = [0_i32; 2];
    permutation.forward_permute(&[rank as i32, -(rank as i32)], &mut forward);
    let mut backward = [0_i32; 2];
    permutation.backward_permute(&forward, &mut backward);

    // Validation routes this deliberately owner-unsorted list separately from
    // the caller-visible permutation order. Rank zero's two entries have
    // different owners, exercising the contiguous all-to-all partitioning.
    let unordered_custom = if size == 1 {
        vec![0]
    } else if rank == 0 {
        vec![1, 0]
    } else if rank == 1 {
        Vec::new()
    } else {
        vec![rank]
    };
    let unordered = DataPermutation::new(layout, &unordered_custom, 1);
    let mut unordered_forward = vec![0_i32; unordered_custom.len()];
    unordered.forward_permute(&[rank as i32], &mut unordered_forward);
    let mut unordered_backward = vec![0_i32; 1];
    unordered.backward_permute(&unordered_forward, &mut unordered_backward);
    let unordered_ok = unordered_forward
        == if size == 1 {
            vec![0]
        } else if rank == 0 {
            vec![1, 0]
        } else if rank == 1 {
            Vec::new()
        } else {
            vec![rank as i32]
        }
        && unordered_backward == [rank as i32];
    collective_assert(
        comm,
        mapped == [((rank + 1) % size) as i32, -(((rank + 1) % size) as i32)]
            && mapped == forward
            && backward == [rank as i32, -(rank as i32)]
            && unordered_ok,
        "mapper and permutation must exchange chunked off-rank values in both directions",
    );
}

/// Contract: wrapper roots and AIJ metadata are rejected before rooted payloads or routing.
fn rooted_and_aij_rejection<C: CommunicatorCollectives>(comm: &C) {
    let layout = Rc::new(IndexLayout::from_local_counts(1, comm));
    let invalid_root = comm.size() as usize;
    let gather_rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let distributed = rlst::dist_vec!(i32, layout.clone());
        distributed.gather_to_one(invalid_root);
    }))
    .is_err();
    let scatter_rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = rlst::DynArray::<i32, 1>::scatter_from_one(invalid_root, layout.clone());
    }))
    .is_err();
    let malformed_rows = if comm.rank() == 0 {
        vec![0]
    } else {
        Vec::new()
    };
    let malformed_data = if comm.rank() == 0 {
        vec![1_f64]
    } else {
        Vec::new()
    };
    let aij_length_rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = rlst::sparse::distributed_csr_mat::DistributedCsrMatrix::from_aij(
            layout.clone(),
            layout.clone(),
            &malformed_rows,
            &[],
            &malformed_data,
        );
    }))
    .is_err();
    let invalid_rows = if comm.rank() == 0 {
        vec![comm.size() as usize]
    } else {
        Vec::new()
    };
    let valid_columns = vec![0; invalid_rows.len()];
    let valid_values = vec![1_f64; invalid_rows.len()];
    let valid_rows = vec![0; invalid_rows.len()];
    let invalid_columns = invalid_rows.clone();
    let aij_row_rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = rlst::sparse::distributed_csr_mat::DistributedCsrMatrix::from_aij(
            layout.clone(),
            layout.clone(),
            &invalid_rows,
            &valid_columns,
            &valid_values,
        );
    }))
    .is_err();
    let aij_column_rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = rlst::sparse::distributed_csr_mat::DistributedCsrMatrix::from_aij(
            layout.clone(),
            layout.clone(),
            &valid_rows,
            &invalid_columns,
            &valid_values,
        );
    }))
    .is_err();
    collective_assert(
        comm,
        gather_rejected
            && scatter_rejected
            && aij_length_rejected
            && aij_row_rejected
            && aij_column_rejected,
        "invalid wrapper roots and individually malformed AIJ lengths/rows/columns must be rejected uniformly before payload communication",
    );
}

/// Contract: invalid permutation and ghost metadata is rejected after every rank participates.
fn collective_metadata_rejection<C: CommunicatorCollectives>(comm: &C) {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let empty_layout = Rc::new(IndexLayout::from_local_counts(0, comm));
    let zero_chunk_rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        DataPermutation::new(empty_layout.clone(), &[], 0);
    }))
    .is_err()
        && std::panic::catch_unwind(AssertUnwindSafe(|| {
            Global2LocalDataMapper::new(empty_layout.clone(), &[], 0);
        }))
        .is_err()
        && std::panic::catch_unwind(AssertUnwindSafe(|| {
            GhostCommunicatorBuilder::<usize>::new()
                .chunk_sizes(ChunkSizes::Uniform(0))
                .build(comm);
        }))
        .is_err();
    // Chunk sizes are agreed at construction, even between ranks without ghost traffic.
    let positive_disagreement_rejected = if size == 1 {
        true
    } else {
        let chunk_size = if rank == 0 { 1 } else { 2 };
        std::panic::catch_unwind(AssertUnwindSafe(|| {
            DataPermutation::new(empty_layout.clone(), &[], chunk_size);
        }))
        .is_err()
            && std::panic::catch_unwind(AssertUnwindSafe(|| {
                Global2LocalDataMapper::new(empty_layout.clone(), &[], chunk_size);
            }))
            .is_err()
    };
    // A receiver that declares a different uniform chunk size than the owner of its ghost
    // is rejected, including on ranks that are not part of the mismatched pair.
    let ghost_disagreement_rejected = if size == 1 {
        true
    } else {
        std::panic::catch_unwind(AssertUnwindSafe(|| {
            let mut builder = GhostCommunicatorBuilder::<usize>::new();
            if rank == 1 {
                builder.add_ghost(0, 0);
            }
            builder
                .chunk_sizes(ChunkSizes::Uniform(if rank == 0 { 1 } else { 2 }))
                .build(comm);
        }))
        .is_err()
    };

    let layout = Rc::new(IndexLayout::from_local_counts(
        if comm.size() == 1 { 2 } else { 1 },
        comm,
    ));
    let invalid_count_rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        all_to_allv::<i32, _>(comm, &[], &[]);
    }))
    .is_err();
    let duplicate_rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        DataPermutation::new(layout.clone(), &[rank, rank], 1);
    }))
    .is_err();
    let missing_coverage_rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let incomplete: &[usize] = if rank == 0 { &[0] } else { &[] };
        DataPermutation::new(layout.clone(), incomplete, 1);
    }))
    .is_err();
    let mapper_index_rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        Global2LocalDataMapper::new(layout.clone(), &[layout.number_of_global_indices()], 1);
    }))
    .is_err();
    let invalid_owner_rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        GhostCommunicatorBuilder::new()
            .add_ghost(0_usize, size)
            .build(comm);
    }))
    .is_err();
    collective_assert(
        comm,
        zero_chunk_rejected
            && positive_disagreement_rejected
            && ghost_disagreement_rejected
            && invalid_count_rejected
            && duplicate_rejected
            && missing_coverage_rejected
            && mapper_index_rejected
            && invalid_owner_rejected,
        "invalid counts, owners, mapper indices, chunks, and permutation coverage must be rejected independently at construction",
    );
}

/// Contract: layouts, ghost value exchange, and CSR matvec retain ranks with independent emptiness.
/// Contract: mixed occupancy, mapper/permutation edge topologies, embedding, and
/// chunked value traffic retain data without depending on nonempty local partitions.
fn expanded_empty_contracts<C: CommunicatorCollectives>(comm: &C) {
    let rank = comm.rank() as usize;
    let input = if rank.is_multiple_of(2) {
        vec![rank as i32, 1]
    } else {
        Vec::new()
    };
    let cumsum = global_inclusive_cumsum(&input, comm);
    let serial = gather_to_all(&input, comm);
    let expected_cumsum = serial
        .iter()
        .scan(0_i32, |sum, value| {
            *sum += value;
            Some(*sum)
        })
        .collect::<Vec<_>>();
    let cumsum_ok = gather_to_all(&cumsum, comm) == expected_cumsum;

    // Rank zero owns all data but requests none; every other rank requests it.
    let concentrated = Rc::new(IndexLayout::from_local_counts(
        usize::from(rank == 0) * 2,
        comm,
    ));
    let required = if rank == 0 { Vec::new() } else { vec![0, 1] };
    let mapper = Global2LocalDataMapper::new(concentrated.clone(), &required, 1);
    let mapper_input = if rank == 0 {
        vec![10_i32, 20]
    } else {
        Vec::new()
    };
    let mapped = mapper.map_data(&mapper_input);
    let mapper_ok = if rank == 0 {
        mapped.is_empty()
    } else {
        mapped == [10, 20]
    };

    // Move all indices off rank zero (except in the single-rank execution), then
    // verify forward/backward roundtrips and a globally empty permutation.
    let custom = if comm.size() == 1 || rank == 1 {
        vec![0, 1]
    } else {
        Vec::new()
    };
    let permutation = DataPermutation::new(concentrated.clone(), &custom, 1);
    let mut forward = vec![0_i32; custom.len()];
    permutation.forward_permute(&mapper_input, &mut forward);
    let mut backward = vec![0_i32; mapper_input.len()];
    permutation.backward_permute(&forward, &mut backward);
    let empty_layout = Rc::new(IndexLayout::from_local_counts(0, comm));
    let empty_permutation = DataPermutation::new(empty_layout, &[], 1);
    let mut empty: [i32; 0] = [];
    empty_permutation.forward_permute::<i32>(&[], &mut empty);
    empty_permutation.backward_permute::<i32>(&empty, &mut []);
    let permutation_ok = (if comm.size() == 1 || rank == 1 {
        forward == [10, 20]
    } else {
        forward.is_empty()
    }) && backward == mapper_input;

    let local_count = usize::from(rank == 0) * 4;
    let embedding_layout = Rc::new(IndexLayout::from_local_counts(local_count, comm));
    let subset = if rank == 0 { vec![3, 1] } else { Vec::new() };
    let embedding = IndexEmbedding::new(embedding_layout, &subset, comm);
    let embedded = if rank == 0 {
        vec![30_i32, 10]
    } else {
        Vec::new()
    };
    let mut expanded = vec![-1_i32; local_count];
    embedding.embed_data(&embedded, &mut expanded, 1);
    let embedding_ok = if rank == 0 {
        expanded == [0, 10, 0, 30] && embedding.extract_embedded_data(&expanded, 1) == embedded
    } else {
        expanded.is_empty() && embedding.extract_embedded_data(&expanded, 1).is_empty()
    };

    let ghost_builder = |chunk_size| {
        let mut builder = GhostCommunicatorBuilder::new();
        if rank != 0 {
            builder.add_ghost(0_usize, 0);
        }
        builder
            .chunk_sizes(ChunkSizes::Uniform(chunk_size))
            .build(comm)
    };
    let ghosts = ghost_builder(1);
    let chunked = ghost_builder(2);
    // Exercise both directions and chunk sizes one/two for every supported
    // value representation.  The one-way topology also covers send-only,
    // receive-only, and isolated ranks.
    let sent_f32 = vec![1.5_f32; chunked.send_buffer_len()];
    let mut received_f32 = vec![0_f32; chunked.receive_buffer_len()];
    chunked.forward_send_values(&sent_f32, &mut received_f32);
    let mut received_f32_scalar = vec![0_f32; ghosts.total_receive_count()];
    ghosts.forward_send_values(
        &vec![1.5_f32; ghosts.total_send_count()],
        &mut received_f32_scalar,
    );
    let mut backward_f32_scalar = vec![0_f32; ghosts.total_send_count()];
    ghosts.backward_send_values(
        &vec![2.5_f32; ghosts.total_receive_count()],
        &mut backward_f32_scalar,
    );
    let mut backward_f32 = vec![0_f32; chunked.send_buffer_len()];
    chunked.backward_send_values(
        &vec![2.5_f32; chunked.receive_buffer_len()],
        &mut backward_f32,
    );

    let mut received_f64_scalar = vec![0_f64; ghosts.total_receive_count()];
    ghosts.forward_send_values(
        &vec![4.0_f64; ghosts.total_send_count()],
        &mut received_f64_scalar,
    );
    let mut received_f64 = vec![0_f64; chunked.receive_buffer_len()];
    chunked.forward_send_values(&vec![4.0_f64; chunked.send_buffer_len()], &mut received_f64);
    let mut backward_f64_scalar = vec![0_f64; ghosts.total_send_count()];
    ghosts.backward_send_values(
        &vec![7.0_f64; ghosts.total_receive_count()],
        &mut backward_f64_scalar,
    );
    let mut backward_f64 = vec![0_f64; chunked.send_buffer_len()];
    chunked.backward_send_values(
        &vec![7.0_f64; chunked.receive_buffer_len()],
        &mut backward_f64,
    );

    let complex = rlst::c64::new(3.0, -2.0);
    let mut received_complex_scalar = vec![rlst::c64::new(0.0, 0.0); ghosts.total_receive_count()];
    ghosts.forward_send_values(
        &vec![complex; ghosts.total_send_count()],
        &mut received_complex_scalar,
    );
    let mut received_complex = vec![rlst::c64::new(0.0, 0.0); chunked.receive_buffer_len()];
    chunked.forward_send_values(
        &vec![complex; chunked.send_buffer_len()],
        &mut received_complex,
    );
    let back_complex = rlst::c64::new(-1.0, 5.0);
    let mut backward_complex_scalar = vec![rlst::c64::new(0.0, 0.0); ghosts.total_send_count()];
    ghosts.backward_send_values(
        &vec![back_complex; ghosts.total_receive_count()],
        &mut backward_complex_scalar,
    );
    let mut backward_complex = vec![rlst::c64::new(0.0, 0.0); chunked.send_buffer_len()];
    chunked.backward_send_values(
        &vec![back_complex; chunked.receive_buffer_len()],
        &mut backward_complex,
    );
    let ghost_ok = chunked.send_buffer_len() == 2 * chunked.total_send_count()
        && chunked.receive_buffer_len() == 2 * chunked.total_receive_count()
        && received_f32.iter().all(|&value| value == 1.5)
        && received_f32_scalar.iter().all(|&value| value == 1.5)
        && backward_f32_scalar.iter().all(|&value| value == 2.5)
        && backward_f32.iter().all(|&value| value == 2.5)
        && received_f64_scalar.iter().all(|&value| value == 4.0)
        && received_f64.iter().all(|&value| value == 4.0)
        && backward_f64_scalar.iter().all(|&value| value == 7.0)
        && backward_f64.iter().all(|&value| value == 7.0)
        && received_complex_scalar
            .iter()
            .all(|&value| value == complex)
        && received_complex.iter().all(|&value| value == complex)
        && backward_complex_scalar
            .iter()
            .all(|&value| value == back_complex)
        && backward_complex.iter().all(|&value| value == back_complex);

    // A successful rooted scatter/gather round-trip includes an empty root
    // partition when more than one rank participates.
    let rooted_layout = Rc::new(IndexLayout::from_local_counts(usize::from(rank != 0), comm));
    let mut rooted_source =
        rlst::DynArray::<i32, 1>::from_shape([rooted_layout.number_of_global_indices()]);
    if rank == 0 {
        rooted_source
            .data_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
            .for_each(|(index, value)| *value = index as i32 + 40);
    }
    let rooted_distributed = if rank == 0 {
        rooted_source.scatter_from_one_root(rooted_layout.clone())
    } else {
        rlst::DynArray::<i32, 1>::scatter_from_one(0, rooted_layout.clone())
    };
    let rooted_ok = if rank == 0 {
        rooted_distributed.gather_to_one_root().data() == rooted_source.data()
    } else {
        rooted_distributed.gather_to_one(0);
        true
    };

    let zero_layout = Rc::new(IndexLayout::from_local_counts(0, comm));
    let zero_vector = rlst::dist_vec!(f64, zero_layout.clone());
    let zero_matrix = rlst::dist_mat!(f64, zero_layout.clone(), 2);
    let no_column_matrix = rlst::dist_mat!(f64, concentrated.clone(), 0);
    let zero_vector_gathered = zero_vector.gather_to_all();
    let zero_vector_row_major = zero_vector.eval_row_major().gather_to_all();
    let zero_vector_column_major = zero_vector.eval_row_major().eval().gather_to_all();
    let zero_vector_sum = zero_vector.sum();
    let zero_vector_max_abs = zero_vector.max_abs();
    let zero_vector_norm_2 = zero_vector.norm_2();
    let zero_matrix_gathered = zero_matrix.gather_to_all();
    let zero_matrix_row_major = zero_matrix.eval_row_major().gather_to_all();
    let no_column_matrix_gathered = no_column_matrix.gather_to_all();
    let no_column_matrix_row_major = no_column_matrix.eval_row_major().gather_to_all();
    let zero_vector_shape_ok = zero_vector_gathered.shape() == [0]
        && zero_vector_row_major.shape() == [0]
        && zero_vector_column_major.shape() == [0];
    let zero_vector_reductions_ok =
        zero_vector_sum.is_none() && zero_vector_max_abs.is_none() && zero_vector_norm_2.is_none();
    let zero_matrix_shape_ok =
        zero_matrix_gathered.shape() == [0, 2] && zero_matrix_row_major.shape() == [0, 2];
    let zero_matrix_columns_ok = zero_matrix.col_iter().count() == 2;
    let no_column_matrix_shape_ok =
        no_column_matrix_gathered.shape() == [2, 0] && no_column_matrix_row_major.shape() == [2, 0];
    let no_column_matrix_columns_ok = no_column_matrix.col_iter().count() == 0;
    let mut reduction_vector = rlst::dist_vec!(f64, concentrated.clone());
    let mut reduction_other = rlst::dist_vec!(f64, concentrated.clone());
    if rank == 0 {
        reduction_vector
            .local
            .data_mut()
            .expect("nonempty local reduction vector")
            .copy_from_slice(&[3.0, -4.0]);
        reduction_other
            .local
            .data_mut()
            .expect("nonempty local reduction vector")
            .copy_from_slice(&[4.0, -3.0]);
    }
    let reduction_values = gather_to_all(reduction_vector.local.data().unwrap_or(&[]), comm);
    let reduction_other_values = gather_to_all(reduction_other.local.data().unwrap_or(&[]), comm);
    let serial_sum: f64 = reduction_values.iter().sum();
    let serial_max_abs = reduction_values
        .iter()
        .map(|value| value.abs())
        .reduce(f64::max);
    let serial_norm = reduction_values
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let serial_inner: f64 = reduction_values
        .iter()
        .zip(&reduction_other_values)
        .map(|(left, right)| left * right)
        .sum();
    // Every distributed reduction must execute before checking any result, so a
    // rank-local failure cannot alter the collective schedule.
    let reduction_sum = reduction_vector.sum();
    let reduction_max_abs = reduction_vector.max_abs();
    let reduction_norm_2 = reduction_vector.norm_2();
    let reduction_inner = reduction_vector.inner(&reduction_other);
    let reductions_ok = reduction_sum == Some(serial_sum)
        && reduction_max_abs == serial_max_abs
        && reduction_norm_2 == Some(serial_norm)
        && reduction_inner == Some(serial_inner);
    let arrays_ok = zero_vector_shape_ok
        && zero_vector_reductions_ok
        && zero_matrix_shape_ok
        && zero_matrix_columns_ok
        && no_column_matrix_shape_ok
        && no_column_matrix_columns_ok
        && reductions_ok;

    collective_assert(
        comm,
        cumsum_ok
            && mapper_ok
            && permutation_ok
            && embedding_ok
            && ghost_ok
            && rooted_ok
            && arrays_ok,
        "mixed occupancy contracts must preserve cumsums, mappings, permutations, embeddings, and chunked values",
    );
}

/// Contract: cancellation, positive zero-nnz rows, beta scaling, multiple RHS,
/// and globally zero matrices remain valid with empty partitions.
fn csr_empty_matrix_contracts<C: CommunicatorCollectives>(comm: &C) {
    let rank = comm.rank() as usize;
    let domain = Rc::new(IndexLayout::from_local_counts(
        usize::from(rank == 0) * 2,
        comm,
    ));
    let range = Rc::new(IndexLayout::from_local_counts(
        usize::from(rank == 0) * 2,
        comm,
    ));
    let rows = if rank == 0 { vec![0, 0, 1] } else { Vec::new() };
    let cols = if rank == 0 { vec![0, 0, 1] } else { Vec::new() };
    let values = if rank == 0 {
        vec![1_f64, -1.0, 2.0]
    } else {
        Vec::new()
    };
    let matrix = rlst::sparse::distributed_csr_mat::DistributedCsrMatrix::from_aij(
        domain.clone(),
        range.clone(),
        &rows,
        &cols,
        &values,
    );
    let mut x = rlst::dist_mat!(f64, domain.clone(), 2);
    let mut y = rlst::dist_mat!(f64, range.clone(), 2);
    if rank == 0 {
        x.local
            .data_mut()
            .unwrap()
            .copy_from_slice(&[1.0, 10.0, 2.0, 20.0]);
        y.local
            .data_mut()
            .unwrap()
            .copy_from_slice(&[4.0, 5.0, 6.0, 7.0]);
    }
    matrix.apply(1.0, &x, 3.0, &mut y);
    let output = gather_to_all(y.local.data().unwrap_or(&[]), comm);
    let zero_matrix = rlst::sparse::distributed_csr_mat::DistributedCsrMatrix::<f64, C>::from_aij(
        domain.clone(),
        range.clone(),
        &[],
        &[],
        &[],
    );
    let mut zero_output = rlst::dist_mat!(f64, range.clone(), 2);
    if rank == 0 {
        zero_output
            .local
            .data_mut()
            .unwrap()
            .copy_from_slice(&[9.0, 8.0, 7.0, 6.0]);
    }
    let zero_input = rlst::dist_mat!(f64, domain.clone(), 2);
    zero_matrix.apply(1.0, &zero_input, 3.0, &mut zero_output);
    let zero_applied = gather_to_all(zero_output.local.data().unwrap_or(&[]), comm);
    let expression = rlst::sparse::mat_operations_distributed::DistributedSparseMatOpIterator::new(
        if rank == 0 {
            vec![([0, 0], 2_f64), ([0, 0], -2.0)].into_iter()
        } else {
            Vec::new().into_iter()
        },
        domain.clone(),
        range.clone(),
    )
    .into_distributed_csr();
    let expression_ok = expression.local().indptr().len() == if rank == 0 { 3 } else { 1 }
        && expression.local().indices().is_empty();
    collective_assert(
        comm,
        matrix.local().indptr().len() == if rank == 0 { 3 } else { 1 }
            && output == [12.0, 35.0, 18.0, 61.0]
            && zero_matrix.local().indptr().len() == if rank == 0 { 3 } else { 1 }
            && zero_matrix.local().indices().is_empty()
            && zero_applied == [27.0, 24.0, 21.0, 18.0]
            && expression_ok,
        "CSR cancellation and positive-row zero-nnz matrices must retain valid row pointers and beta-scaled multi-RHS results",
    );
}

/// Contract: allocating distributed products preserve layouts and communicate
/// ghosts even when domain and range ownership are on disjoint ranks.
fn distributed_dot<C: CommunicatorCollectives>(comm: &C) {
    let rank = comm.rank();
    let domain = Rc::new(IndexLayout::from_local_counts(
        if rank == comm.size() - 1 { 2 } else { 0 },
        comm,
    ));
    let range = Rc::new(IndexLayout::from_local_counts(
        if rank == 0 { 3 } else { 0 },
        comm,
    ));
    let matrix = rlst::sparse::distributed_csr_mat::DistributedCsrMatrix::from_aij(
        domain.clone(),
        range.clone(),
        if rank == 0 { &[0, 0, 2] } else { &[] },
        if rank == 0 { &[0, 1, 0] } else { &[] },
        if rank == 0 {
            &[2.0_f64, 3.0, -1.0]
        } else {
            &[]
        },
    );
    let mut x = rlst::dist_vec!(f64, domain.clone());
    for i in 0..x.local.len() {
        x.local[[i]] = 4.0 + (domain.local_range().0 + i) as f64;
    }
    let calls = std::cell::Cell::new(0);
    let y = rlst::dot!(
        {
            calls.set(calls.get() + 1);
            &matrix
        },
        &x
    );
    let values = y.gather_to_all();
    let doubled = rlst::dot!(matrix, x.r() + x.r()).gather_to_all();
    collective_assert(
        comm,
        calls.get() == 1
            && Rc::ptr_eq(&y.index_layout, &range)
            && values.data().unwrap() == [23.0, 0.0, -4.0]
            && doubled.data().unwrap() == [46.0, 0.0, -8.0],
        "distributed dot must borrow once, preserve the range layout, and exchange remote vector values",
    );

    let mut rhs = rlst::dist_mat!(f64, domain.clone(), 2);
    for j in 0..2 {
        for i in 0..domain.number_of_local_indices() {
            rhs.local[[i, j]] = 4.0 + (domain.local_range().0 + i + 10 * j) as f64;
        }
    }
    let result = rlst::dot!(matrix, rhs.r());
    let values = result.gather_to_all();
    collective_assert(
        comm,
        Rc::ptr_eq(&result.index_layout, &range)
            && values.shape() == [3, 2]
            && values.data().unwrap() == [23.0, 0.0, -4.0, 73.0, 0.0, -14.0],
        "distributed dot must preserve multiple RHS columns and zero-nnz rows",
    );
    let no_columns = rlst::dist_mat!(f64, domain.clone(), 0);
    let result = rlst::dot!(matrix, no_columns);
    collective_assert(
        comm,
        result.local.shape() == [range.number_of_local_indices(), 0],
        "zero RHS columns must produce empty local matrices",
    );

    for (rows, cols) in [(0, 0), (0, 2), (3, 0), (3, 2)] {
        let domain = Rc::new(IndexLayout::from_local_counts(
            if rank == 0 { cols } else { 0 },
            comm,
        ));
        let range = Rc::new(IndexLayout::from_local_counts(
            if rank == 0 { rows } else { 0 },
            comm,
        ));
        let matrix = rlst::sparse::distributed_csr_mat::DistributedCsrMatrix::<f64, C>::from_aij(
            domain.clone(),
            range.clone(),
            &[],
            &[],
            &[],
        );
        let vector = rlst::dist_vec!(f64, domain.clone());
        let rhs = rlst::dist_mat!(f64, domain.clone(), 2);
        // The returned storage must not borrow the matrix or either input.
        let y = rlst::dot!(matrix, vector);
        let z = rlst::dot!(matrix, rhs);
        drop((matrix, vector, rhs));
        let y = y.gather_to_all();
        let z = z.gather_to_all();
        collective_assert(
            comm,
            y.shape() == [rows]
                && z.shape() == [rows, 2]
                && y.iter_value().all(|v| v == 0.0)
                && z.iter_value().all(|v| v == 0.0),
            "globally empty and zero-nnz distributed products must have correct zero outputs",
        );
    }
}

/// Contract: invalid metadata on one rank is rejected before peers exchange ghosts.
fn distributed_dot_rejection<C: CommunicatorCollectives>(comm: &C) {
    let layout = Rc::new(IndexLayout::from_local_counts(1, comm));
    let other_layout = Rc::new(IndexLayout::from_local_counts(1, comm));
    let row = layout.local_range().0;
    let matrix = rlst::sparse::distributed_csr_mat::DistributedCsrMatrix::from_aij(
        layout.clone(),
        layout.clone(),
        &[row],
        &[(row + 1) % comm.size() as usize],
        &[1.0_f64],
    );
    let wrong_layout = rlst::dist_vec!(
        f64,
        if comm.rank() == 0 {
            other_layout.clone()
        } else {
            layout.clone()
        }
    );
    let layout_rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = rlst::dot!(matrix, wrong_layout);
    }))
    .is_err();
    let mut wrong_rows = rlst::dist_vec!(f64, layout.clone());
    if comm.rank() == 0 {
        wrong_rows.local = rlst::DynArray::from_shape([2]);
    }
    let rows_rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = rlst::dot!(matrix, wrong_rows);
    }))
    .is_err();
    let mut columns_rejected = true;
    let mut ranks_rejected = true;
    if comm.size() > 1 {
        let wrong_columns = rlst::dist_mat!(f64, layout.clone(), usize::from(comm.rank() == 0));
        columns_rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = rlst::dot!(matrix, wrong_columns);
        }))
        .is_err();
        ranks_rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
            if comm.rank() == 0 {
                let x = rlst::dist_vec!(f64, layout.clone());
                let _ = rlst::dot!(matrix, x);
            } else {
                let x = rlst::dist_mat!(f64, layout.clone(), 1);
                let _ = rlst::dot!(matrix, x);
            }
        }))
        .is_err();
    }
    collective_assert(
        comm,
        layout_rejected && rows_rejected && columns_rejected && ranks_rejected,
        "invalid distributed dot metadata must be rejected on every rank",
    );
}

/// Contract: remapping rejects per-rank disagreement about either layout before routing.
fn remap_layout_agreement(comm: &SimpleCommunicator) {
    if comm.size() == 1 {
        collective_assert(comm, true, "single-rank layouts are trivially consistent");
        return;
    }
    let size = comm.size() as usize;
    let source = IndexLayout::from_local_counts(1, comm);
    let mut target_offsets = vec![0; size + 1];
    if comm.rank() == 0 {
        target_offsets[size] = size;
    } else {
        target_offsets[1..].fill(size);
    }
    let inconsistent_target = IndexLayout::new(target_offsets, comm);
    let rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = source.remap(&inconsistent_target, &[comm.rank()]);
    }))
    .is_err();
    let mut source_offsets = (0..=size).collect::<Vec<_>>();
    if comm.rank() == 0 {
        source_offsets[1] = 0;
    }
    let inconsistent_source = IndexLayout::new(source_offsets, comm);
    let inconsistent_source_data: &[i32] = if comm.rank() == 0 {
        &[]
    } else {
        &[comm.rank()]
    };
    let source_rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = inconsistent_source.remap(&source, inconsistent_source_data);
    }))
    .is_err();
    let self_comm = comm
        .split_by_color(Color::with_value(comm.rank()))
        .expect("each rank belongs to its singleton communicator");
    let self_layout = IndexLayout::from_local_counts(size, &self_comm);
    let different_communicator_rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = source.remap(&self_layout, &[comm.rank()]);
    }))
    .is_err();
    let empty = IndexLayout::from_local_counts(0, comm);
    let empty_remap = empty.remap(&empty, &[] as &[i32]);
    collective_assert(
        comm,
        rejected && source_rejected && different_communicator_rejected && empty_remap.is_empty(),
        "remap must reject inconsistent offsets and differing-size communicators before routing and support globally empty layouts",
    );
}

fn layouts_and_consumers<C: CommunicatorCollectives>(comm: &C) {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let source = Rc::new(IndexLayout::from_local_counts(usize::from(rank == 0), comm));
    let target = Rc::new(IndexLayout::from_local_counts(
        usize::from(rank == size / 2),
        comm,
    ));
    let source_values: &[i32] = if rank == 0 { &[7_i32] } else { &[] };
    let remapped = source.remap(&target, source_values);

    // Both layouts contain an interior empty owner when more than two ranks
    // participate.  Compare exact values for the singleton nonempty path and
    // the globally empty path too, rather than merely accepting construction.
    let middle_source_count = if size > 1 && rank == 1 { 0 } else { rank + 1 };
    let middle_target_count = if size > 1 && rank == 1 { 0 } else { rank + 1 };
    let middle_source = IndexLayout::from_local_counts(middle_source_count, comm);
    let middle_target = IndexLayout::from_local_counts(middle_target_count, comm);
    let middle_source_range = middle_source.local_range();
    let middle_values = (middle_source_range.0..middle_source_range.1)
        .map(|global_index| global_index as i32)
        .collect::<Vec<_>>();
    let middle_remapped = middle_source.remap(&middle_target, &middle_values);
    let middle_target_range = middle_target.local_range();
    let expected_middle = (middle_target_range.0..middle_target_range.1)
        .map(|global_index| global_index as i32)
        .collect::<Vec<_>>();
    let empty_layout = IndexLayout::from_local_counts(0, comm);
    let empty_remapped = empty_layout.remap(&empty_layout, &[] as &[i32]);

    let mut ghost_builder = GhostCommunicatorBuilder::new();
    if rank != 0 {
        ghost_builder.add_ghost(0_usize, 0);
    }
    let ghosts = ghost_builder.build(comm);
    let mut received = vec![0_f64; ghosts.total_receive_count()];
    let sent = vec![4_f64; ghosts.total_send_count()];
    ghosts.forward_send_values(&sent, &mut received);

    let domain = Rc::new(IndexLayout::from_local_counts(usize::from(rank == 0), comm));
    let range = Rc::new(IndexLayout::from_local_counts(
        usize::from(rank + 1 == size),
        comm,
    ));
    let rows = if rank == 0 { vec![0] } else { vec![] };
    let cols = if rank == 0 { vec![0] } else { vec![] };
    let values = if rank == 0 { vec![2_f64] } else { vec![] };
    let matrix = rlst::sparse::distributed_csr_mat::DistributedCsrMatrix::from_aij(
        domain.clone(),
        range.clone(),
        &rows,
        &cols,
        &values,
    );
    let mut x = rlst::dist_vec!(f64, domain.clone());
    let mut y = rlst::dist_vec!(f64, range.clone());
    if rank == 0 {
        x.local.data_mut().unwrap()[0] = 1.0;
    }
    matrix.apply(1.0, &x, 0.0, &mut y);
    let output =
        rlst::distributed_tools::array_tools::gather_to_all(y.local.data().unwrap_or(&[]), comm);

    // All ranks execute each RHS exchange even though the domain and range
    // ownership are disjoint and most ranks have no local rows.
    let mut x_multi = rlst::dist_mat!(f64, domain.clone(), 2);
    let mut y_multi = rlst::dist_mat!(f64, range.clone(), 2);
    if rank == 0 {
        x_multi
            .local
            .data_mut()
            .unwrap()
            .copy_from_slice(&[1.0, 3.0]);
    }
    matrix.apply(1.0, &x_multi, 0.0, &mut y_multi);
    let multi_output = rlst::distributed_tools::array_tools::gather_to_all(
        y_multi.local.data().unwrap_or(&[]),
        comm,
    );
    let x_zero_columns = rlst::dist_mat!(f64, domain.clone(), 0);
    let mut y_zero_columns = rlst::dist_mat!(f64, range.clone(), 0);
    matrix.apply(1.0, &x_zero_columns, 0.0, &mut y_zero_columns);
    let space =
        rlst::operator::interface::distributed_array_vector_space::DistributedArrayVectorSpace::<
            _,
            f64,
        >::new(domain);
    let zero = space.zero();
    let zero_inner = space.inner_product(&zero, &zero);
    let empty_domain = Rc::new(IndexLayout::from_local_counts(0, comm));
    let empty_space =
        rlst::operator::interface::distributed_array_vector_space::DistributedArrayVectorSpace::<
            _,
            f64,
        >::new(empty_domain);
    let empty_zero = empty_space.zero();
    let empty_inner = empty_space.inner_product(&empty_zero, &empty_zero);

    collective_assert(
        comm,
        remapped == if rank == size / 2 { vec![7] } else { vec![] }
            && middle_remapped == expected_middle
            && empty_remapped.is_empty()
            && received.iter().all(|&value| value == 4.0)
            && matrix.local().indptr().len() == matrix.range_layout().number_of_local_indices() + 1
            && output == [2.0]
            && multi_output == [2.0, 6.0]
            && y_zero_columns
                .local
                .data()
                .is_some_and(|data| data.is_empty())
            && zero_inner == 0.0
            && empty_inner == 0.0,
        "layouts, exact middle-empty remaps, ghost communication, and zero-nnz CSR ranks must remain valid",
    );
}

/// Contract: owner-declared variable chunk sizes, including empty chunks, reach the
/// receivers at build time; forward and backward exchanges use the resulting layout, and
/// ghosts arrive grouped by owner in insertion order.
fn ghost_variable_chunks<C: CommunicatorCollectives>(comm: &C) {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let owner = |index: usize| index % size;

    // Odd ranks require every index they do not own, added in descending order, so the
    // receive order differs from the insertion order. Even ranks require no ghosts.
    let ghosts = if rank % 2 == 1 {
        (0..4 * size)
            .rev()
            .filter(|&index| owner(index) != rank)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let ghost_comm = GhostCommunicatorBuilder::new()
        .add_ghosts(ghosts.iter().map(|&index| (index, owner(index))))
        .chunk_sizes(ChunkSizes::PerIndex(Box::new(|index: usize| index % 3)))
        .build(comm);

    let mut expected_order = ghosts.clone();
    expected_order.sort_by_key(|&index| owner(index));
    let order_ok = ghost_comm.receive_indices() == expected_order.as_slice();

    let value = |index: usize, k: usize| (1000 * index + k) as i64;
    let send_offsets = ghost_comm.send_offsets();
    let mut send_data = vec![0_i64; ghost_comm.send_buffer_len()];
    for (i, &index) in ghost_comm.send_indices().iter().enumerate() {
        for (k, entry) in send_data[send_offsets[i]..send_offsets[i + 1]]
            .iter_mut()
            .enumerate()
        {
            *entry = value(index, k);
        }
    }
    let mut received = vec![0_i64; ghost_comm.receive_buffer_len()];
    ghost_comm.forward_send_values(&send_data, &mut received);

    let receive_offsets = ghost_comm.receive_offsets();
    let layout_ok = receive_offsets.len() == 1 + ghost_comm.total_receive_count()
        && receive_offsets[0] == 0
        && *receive_offsets.last().unwrap() == received.len()
        && send_offsets.len() == 1 + ghost_comm.total_send_count()
        && ghost_comm
            .send_indices()
            .iter()
            .all(|&index| owner(index) == rank)
        && ghost_comm
            .send_indices()
            .iter()
            .zip(ghost_comm.send_chunk_sizes())
            .all(|(&index, &chunk_size)| chunk_size == index % 3)
        && ghost_comm
            .receive_indices()
            .iter()
            .zip(ghost_comm.receive_chunk_sizes())
            .all(|(&index, &chunk_size)| chunk_size == index % 3);
    let forward_ok = ghost_comm
        .receive_indices()
        .iter()
        .enumerate()
        .all(|(i, &index)| {
            let chunk = &received[receive_offsets[i]..receive_offsets[i + 1]];
            chunk.len() == index % 3
                && chunk
                    .iter()
                    .enumerate()
                    .all(|(k, &entry)| entry == value(index, k))
        });

    // Receivers send negated values back; every owner sees each requester's chunk.
    let returned_values = received.iter().map(|&entry| -entry).collect::<Vec<_>>();
    let mut returned = vec![0_i64; ghost_comm.send_buffer_len()];
    ghost_comm.backward_send_values(&returned_values, &mut returned);
    let backward_ok = returned
        .iter()
        .zip(&send_data)
        .all(|(&back, &sent)| back == -sent);

    // The owned buffers give the same payloads and round-trip in both directions.
    let owned_send = ghost_comm.send_buffer_from_fn(|index, chunk: &mut [i64]| {
        for (k, entry) in chunk.iter_mut().enumerate() {
            *entry = value(index, k);
        }
    });
    let owned_received = owned_send.forward();
    let mut owned_returned_values = ghost_comm.receive_buffer::<i64>();
    for ((index, chunk), (received_index, received_chunk)) in owned_returned_values
        .chunks_mut()
        .zip(owned_received.chunks())
    {
        assert_eq!(index, received_index);
        for (entry, &received_entry) in chunk.iter_mut().zip(received_chunk) {
            *entry = -received_entry;
        }
    }
    let owned_returned = owned_returned_values.backward();
    let owned_ok = owned_send.as_slice() == send_data.as_slice()
        && owned_received.as_slice() == received.as_slice()
        && owned_returned.as_slice() == returned.as_slice()
        && (0..ghost_comm.total_receive_count()).all(|i| {
            owned_received.chunk(i) == &received[receive_offsets[i]..receive_offsets[i + 1]]
        });

    // Every odd rank requests every index it does not own.
    let requesters = |index: usize| {
        (0..size)
            .filter(|&other| other % 2 == 1 && other != owner(index))
            .count()
    };
    let owned_requests = (0..4 * size)
        .filter(|&index| owner(index) == rank)
        .map(requesters)
        .sum::<usize>();
    let count_ok = ghost_comm.total_send_count() == owned_requests
        && ghost_comm.total_receive_count() == ghosts.len();

    collective_assert(
        comm,
        order_ok && layout_ok && forward_ok && backward_ok && owned_ok && count_ok,
        "variable ghost chunks must keep owner-declared sizes, owner-grouped order, and exact payloads in both directions",
    );
}

/// Contract: an exchange with a wrong local buffer length panics only on that rank, after
/// it has taken part in the exchange, so its neighbours complete with default values
/// instead of blocking.
fn exchange_buffer_mismatch_panics<C: CommunicatorCollectives>(comm: &C) {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let mut builder = GhostCommunicatorBuilder::new();
    if rank != 0 {
        builder.add_ghost(0_usize, 0);
    }
    let ghosts = builder.build(comm);

    // Rank zero sends to every other rank, with one extra value in its send buffer.
    let send = vec![7_i32; ghosts.send_buffer_len() + usize::from(rank == 0)];
    let mut receive = vec![-1_i32; ghosts.receive_buffer_len()];
    let ghost_panicked = std::panic::catch_unwind(AssertUnwindSafe(|| {
        ghosts.forward_send_values(&send, &mut receive);
    }))
    .is_err();
    let ghost_ok = ghost_panicked == (rank == 0) && receive.iter().all(|&value| value == 0);

    // The last rank passes a receive buffer with one value too many. Rank zero's values reach
    // every other receiver, while the offending rank's own slice stays untouched.
    let last = size - 1;
    let send = vec![7_i32; ghosts.send_buffer_len()];
    let mut receive = vec![-1_i32; ghosts.receive_buffer_len() + usize::from(rank == last)];
    let receive_panicked = std::panic::catch_unwind(AssertUnwindSafe(|| {
        ghosts.forward_send_values(&send, &mut receive);
    }))
    .is_err();
    let receive_ok = receive_panicked == (rank == last)
        && receive
            .iter()
            .all(|&value| value == if rank == last { -1 } else { 7 });

    // In the backward direction, the last rank sends one value too many; its owner receives
    // default values from it and the values of every other receiver.
    let back_send = vec![3_i32; ghosts.receive_buffer_len() + usize::from(rank == last)];
    let mut back_receive = vec![-1_i32; ghosts.send_buffer_len()];
    let backward_panicked = std::panic::catch_unwind(AssertUnwindSafe(|| {
        ghosts.backward_send_values(&back_send, &mut back_receive);
    }))
    .is_err();
    // Rank zero's send slots are ordered by the requesting ranks 1, 2, ...
    let backward_ok = backward_panicked == (rank == last)
        && back_receive
            .iter()
            .enumerate()
            .all(|(slot, &value)| value == if slot + 1 == last { 0 } else { 3 });

    // Permutation over a ring: each rank takes the index of its successor. Rank zero passes
    // one input value too many forwards, so its predecessor receives a default value.
    let ring_layout = Rc::new(IndexLayout::from_local_counts(1, comm));
    let permutation = DataPermutation::new(ring_layout, &[(rank + 1) % size], 1);
    let forward_data = vec![10 + rank as i32; 1 + usize::from(rank == 0)];
    let mut forward = [-1_i32];
    let forward_panicked = std::panic::catch_unwind(AssertUnwindSafe(|| {
        permutation.forward_permute(&forward_data, &mut forward);
    }))
    .is_err();
    let permute_forward_ok = forward_panicked == (rank == 0)
        && (rank == 0
            || forward[0]
                == if (rank + 1).is_multiple_of(size) {
                    0
                } else {
                    10 + (rank as i32 + 1)
                });
    // Backwards, rank zero passes an output buffer that is too long, so its successor
    // receives a default value.
    let mut backward = vec![-1_i32; 1 + usize::from(rank == 0)];
    let backward_permute_panicked = std::panic::catch_unwind(AssertUnwindSafe(|| {
        permutation.backward_permute(&[100 + rank as i32], &mut backward);
    }))
    .is_err();
    let permute_backward_ok = backward_permute_panicked == (rank == 0)
        && (rank == 0 || backward[0] == if rank == 1 { 0 } else { 100 + rank as i32 - 1 });

    // Rank zero passes one value too many to the mapper; the others request its dof.
    let layout = Rc::new(IndexLayout::from_local_counts(1, comm));
    let required = if rank == 0 { Vec::new() } else { vec![0] };
    let mapper = Global2LocalDataMapper::new(layout, &required, 1);
    let data = vec![rank as i32 + 5; 1 + usize::from(rank == 0)];
    let mapped = std::panic::catch_unwind(AssertUnwindSafe(|| mapper.map_data(&data)));
    let mapper_ok = match mapped {
        Err(_) => rank == 0,
        Ok(mapped) => rank != 0 && mapped == [0],
    };

    collective_assert(
        comm,
        ghost_ok
            && receive_ok
            && backward_ok
            && mapper_ok
            && permute_forward_ok
            && permute_backward_ok,
        "a buffer mismatch must panic only on its rank, without blocking the neighbours",
    );
}

/// Contract: ghosts owned by the requesting rank itself and duplicated ghosts are separate
/// entries that are routed exactly in both directions, and chunk sizes whose value counts
/// overflow `usize` or MPI's `i32` counts are rejected collectively by the build.
fn ghost_edge_topologies<C: CommunicatorCollectives>(comm: &C) {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let owner = |index: usize| index % size;
    // Each rank requests its own index and, twice, the index of its successor.
    let requests =
        |requester: usize| vec![requester, (requester + 1) % size, (requester + 1) % size];
    let receive_order = |requester: usize| {
        let mut order = requests(requester);
        order.sort_by_key(|&index| owner(index));
        order
    };
    let chunk_size = |index: usize| index % 2 + 1;

    let ghosts = GhostCommunicatorBuilder::new()
        .add_ghosts(
            requests(rank)
                .into_iter()
                .map(|index| (index, owner(index))),
        )
        .chunk_sizes(ChunkSizes::PerIndex(Box::new(chunk_size)))
        .build(comm);

    let order_ok = ghosts.receive_indices() == receive_order(rank).as_slice();
    let self_ok =
        ghosts.in_ranks().contains(&(rank as i32)) && ghosts.out_ranks().contains(&(rank as i32));

    // The block sent to each requester lists its requests for our indices, duplicates
    // included, in its receive order.
    let mut start = 0;
    let mut routing_ok = true;
    for (&requester, &count) in ghosts.out_ranks().iter().zip(ghosts.send_counts()) {
        let expected = receive_order(requester as usize)
            .into_iter()
            .filter(|&index| owner(index) == rank)
            .collect::<Vec<_>>();
        routing_ok &= ghosts.send_indices()[start..start + count as usize] == expected;
        start += count as usize;
    }
    routing_ok &= start == ghosts.total_send_count();

    let value = |index: usize, k: usize| (10 * index + k) as i64;
    let send = ghosts.send_buffer_from_fn(|index, chunk: &mut [i64]| {
        for (k, entry) in chunk.iter_mut().enumerate() {
            *entry = value(index, k);
        }
    });
    let received = send.forward();
    let forward_ok = received.chunks().all(|(index, chunk)| {
        chunk.len() == chunk_size(index)
            && chunk
                .iter()
                .enumerate()
                .all(|(k, &entry)| entry == value(index, k))
    });

    // Filling a plain buffer chunk by chunk gives the same payload.
    let mut refilled = ghosts.send_buffer::<i64>();
    for i in 0..ghosts.total_send_count() {
        let index = ghosts.send_indices()[i];
        for (k, entry) in refilled.chunk_mut(i).iter_mut().enumerate() {
            *entry = value(index, k);
        }
    }
    let buffers_ok = std::ptr::eq(refilled.ghosts(), &ghosts)
        && refilled.forward().into_vec() == received.as_slice();

    // Receivers tag every value with their rank and the ghost's receive position, so the
    // owner can check that each value comes back to the right slot.
    let tag = |requester: usize, position: usize, k: usize| {
        (1000 * (requester + 1) + 100 * position + k) as i64
    };
    let mut returned_values = ghosts.receive_buffer::<i64>();
    for position in 0..ghosts.total_receive_count() {
        for (k, entry) in returned_values.chunk_mut(position).iter_mut().enumerate() {
            *entry = tag(rank, position, k);
        }
    }
    let returned_raw = returned_values.as_slice().to_vec();
    let returned = returned_values.backward();
    let mut slot = 0;
    let mut backward_ok = true;
    for (&requester, &count) in ghosts.out_ranks().iter().zip(ghosts.send_counts()) {
        let positions = receive_order(requester as usize)
            .into_iter()
            .enumerate()
            .filter(|&(_, index)| owner(index) == rank)
            .map(|(position, _)| position)
            .collect::<Vec<_>>();
        backward_ok &= positions.len() == count as usize;
        for &position in &positions {
            backward_ok &= returned
                .chunk(slot)
                .iter()
                .enumerate()
                .all(|(k, &entry)| entry == tag(requester as usize, position, k));
            slot += 1;
        }
    }
    backward_ok &= slot == ghosts.total_send_count();

    // The slice path agrees with the owned buffers.
    let mut slice_returned = vec![0_i64; ghosts.send_buffer_len()];
    ghosts.backward_send_values(&returned_raw, &mut slice_returned);
    backward_ok &= slice_returned == returned.as_slice();

    // Every rank requests index zero twice, so rank zero owns at least two chunks.
    let build_huge = |size: usize| {
        std::panic::catch_unwind(AssertUnwindSafe(|| {
            GhostCommunicatorBuilder::new()
                .add_ghosts([(0_usize, 0), (0_usize, 0)])
                .chunk_sizes(ChunkSizes::PerIndex(Box::new(move |_| size)))
                .build(comm);
        }))
        .is_err()
    };
    let overflow_rejected = build_huge(i32::MAX as usize) && build_huge(usize::MAX / 2 + 1);

    collective_assert(
        comm,
        order_ok
            && self_ok
            && routing_ok
            && forward_ok
            && buffers_ok
            && backward_ok
            && overflow_rejected,
        "self-owned and duplicated ghosts must be routed exactly, and oversized chunks rejected",
    );
}

fn main() {
    let universe = mpi::initialize().expect("MPI must initialize once per executable");
    let world = universe.world();
    let case = env::var("RLST_MPI_EMPTY_CASE").unwrap_or_else(|_| "zero-count-collectives".into());

    match case.as_str() {
        "zero-count-collectives" => zero_count_collectives(&world),
        "empty-cumsum" => empty_cumsum(&world),
        "empty-boundaries" => empty_boundaries(&world),
        "empty-extrema" => empty_extrema(&world),
        "sparse-sort" => sparse_sort(&world),
        "mapper-and-permutation" => mapper_and_permutation(&world),
        "rooted-and-aij-rejection" => rooted_and_aij_rejection(&world),
        "collective-metadata-rejection" => collective_metadata_rejection(&world),
        "expanded-empty-contracts" => expanded_empty_contracts(&world),
        "csr-empty-matrix-contracts" => csr_empty_matrix_contracts(&world),
        "distributed-dot" => distributed_dot(&world),
        "distributed-dot-rejection" => distributed_dot_rejection(&world),
        "remap-layout-agreement" => remap_layout_agreement(&world),
        "layouts-and-consumers" => layouts_and_consumers(&world),
        "ghost-variable-chunks" => ghost_variable_chunks(&world),
        "exchange-buffer-mismatch-panics" => exchange_buffer_mismatch_panics(&world),
        "ghost-edge-topologies" => ghost_edge_topologies(&world),
        _ => panic!("unknown RLST_MPI_EMPTY_CASE={case}"),
    }
}
