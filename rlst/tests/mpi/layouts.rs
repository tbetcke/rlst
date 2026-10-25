//! Index layouts, the array tools, embeddings, permutations and ghost buffers on data spread
//! over the ranks.
//!
//! Expected values are computed on every rank from formulas of the global index, not from
//! another collective.

use std::collections::BTreeSet;

use mpi::{
    topology::{CommunicatorRelation, SimpleCommunicator},
    traits::{AsRaw, Communicator},
};
use rand::{RngExt, SeedableRng, rngs::StdRng, seq::SliceRandom};
use rlst::distributed_tools::{
    ChunkSizes, DataPermutation, GhostCommunicator, GhostCommunicatorBuilder,
    Global2LocalDataMapper, IndexLayout, SparseRoutes, all_to_allv,
    array_tools::{gather_to_rank, global_size, scatterv, scatterv_root},
    index_embedding::IndexEmbedding,
    sparse_all_to_allv,
};

use crate::common::{Case, collective_assert, panics, uneven_counts, uneven_layout, value_at};

/// The cases of this module.
pub const CASES: &[Case] = &[
    ("layout-queries", layout_queries),
    ("layout-remap-data", layout_remap_data),
    ("array-tools-data", array_tools_data),
    ("index-embedding", index_embedding),
    ("permutation-and-mapper-data", permutation_and_mapper_data),
    ("ghost-buffer-accessors", ghost_buffer_accessors),
    ("ghost-example-port", ghost_example_port),
];

/// The exclusive prefix sums of `counts`, with the total appended.
fn scan_of(counts: &[usize]) -> Vec<usize> {
    std::iter::once(0)
        .chain(counts.iter().scan(0, |sum, &count| {
            *sum += count;
            Some(*sum)
        }))
        .collect()
}

/// Whether every query of `layout` agrees with the local `counts` of all ranks.
fn layout_matches(layout: &IndexLayout<'_, SimpleCommunicator>, counts: &[usize]) -> bool {
    let comm = layout.comm();
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let scan = scan_of(counts);
    let total = scan[size];
    let local = counts[rank];
    let owner = |index: usize| (0..size).find(|&r| scan[r] <= index && index < scan[r + 1]);

    layout.scan() == scan
        && layout.local_counts() == counts
        && layout.local_range() == (scan[rank], scan[rank + 1])
        && layout.number_of_global_indices() == total
        && layout.number_of_local_indices() == local
        && (0..size).all(|r| layout.index_range(r) == Some((scan[r], scan[r + 1])))
        && layout.index_range(size).is_none()
        && (0..local).all(|i| layout.local2global(i) == Some(scan[rank] + i))
        && layout.local2global(local).is_none()
        && (0..total).all(|index| {
            let owner = owner(index).expect("every index has an owner");
            layout.rank_from_index(index) == Some(owner)
                && layout.global2local(owner, index) == Some(index - scan[owner])
                && (0..size)
                    .filter(|&r| r != owner)
                    .all(|r| layout.global2local(r, index).is_none())
        })
        && layout.rank_from_index(total).is_none()
        && layout.global2local(size, 0).is_none()
        && format!("{layout:?}")
            == format!("IndexLayout with {total} global indices and {local} local indices.")
}

/// Contract: every constructor of `IndexLayout` gives the expected ranges, and every query
/// agrees with them.
fn layout_queries(comm: &SimpleCommunicator) {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let mut ok = true;

    // Equidistributed chunks: rank r receives `nchunks / size` chunks, plus one if
    // `r < nchunks % size`. This covers no chunk, fewer chunks than ranks, exactly one chunk
    // per rank and a remainder.
    for (nchunks, chunk_size) in [
        (0, 1),
        (size - 1, 2),
        (size, 2),
        (size + 1, 3),
        (3 * size + 2, 1),
        (5 * size + size / 2, 4),
    ] {
        let layout = IndexLayout::from_equidistributed_chunks(nchunks, chunk_size, comm);
        let counts: Vec<usize> = (0..size)
            .map(|r| (nchunks / size + usize::from(r < nchunks % size)) * chunk_size)
            .collect();
        ok &= layout_matches(&layout, &counts);
    }

    // Local counts, and the same layout from its scan.
    for salt in 0..3 {
        let counts = uneven_counts(size, salt);
        let layout = IndexLayout::from_local_counts(counts[rank], comm);
        ok &= layout_matches(&layout, &counts);
        ok &= layout_matches(&IndexLayout::new(scan_of(&counts), comm), &counts);
    }

    // `is_same` compares identity, not contents.
    let layout = uneven_layout(comm, 0);
    let shared = layout.clone();
    let copy = (*layout).clone();
    ok &= layout.is_same(&shared)
        && !layout.is_same(&copy)
        && layout_matches(&copy, &uneven_counts(size, 0));
    ok &= layout.comm().compare(comm) == CommunicatorRelation::Identical;

    // Invalid scans are rejected.
    ok &= panics(|| IndexLayout::new(vec![0; size], comm));
    ok &= panics(|| IndexLayout::new(vec![1; size + 1], comm));
    if size > 1 {
        let mut decreasing = scan_of(&vec![1; size]);
        decreasing.swap(1, 2);
        ok &= panics(|| IndexLayout::new(decreasing, comm));
    }

    collective_assert(
        comm,
        ok,
        "index layouts must report the ranges of their construction",
    );
}

/// Contract: `remap` moves every value to the rank that owns its index in the target layout,
/// for layouts with empty ranks and for all data on one rank, and back again.
fn layout_remap_data(comm: &SimpleCommunicator) {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let source_counts = uneven_counts(size, 1);
    let total: usize = source_counts.iter().sum();
    let source = IndexLayout::from_local_counts(source_counts[rank], comm);

    let targets = [
        // Equidistributed.
        (0..size)
            .map(|r| total / size + usize::from(r < total % size))
            .collect::<Vec<_>>(),
        // Everything on the last rank.
        (0..size)
            .map(|r| if r + 1 == size { total } else { 0 })
            .collect(),
        // Everything on the first rank.
        (0..size).map(|r| if r == 0 { total } else { 0 }).collect(),
    ];

    let (start, end) = source.local_range();
    let indices: Vec<u64> = (start..end).map(|index| index as u64).collect();
    let values: Vec<f64> = (start..end).map(|index| value_at(index, 0)).collect();
    let mut ok = true;
    for target_counts in targets {
        let target = IndexLayout::new(scan_of(&target_counts), comm);
        let (target_start, target_end) = target.local_range();

        let remapped_indices = source.remap(&target, &indices);
        let remapped_values = source.remap(&target, &values);
        ok &= remapped_indices
            == (target_start..target_end)
                .map(|index| index as u64)
                .collect::<Vec<_>>();
        ok &= remapped_values
            == (target_start..target_end)
                .map(|index| value_at(index, 0))
                .collect::<Vec<_>>();

        // The way back restores the original data.
        ok &= target.remap(&source, &remapped_values) == values;
    }

    collective_assert(
        comm,
        ok,
        "remap must deliver each value to the owner of its index in the target layout",
    );
}

/// A value that identifies the source rank, the destination rank and the position.
fn encode(source: usize, destination: usize, position: usize) -> u64 {
    (source * 1_000_000 + destination * 1_000 + position) as u64
}

/// Contract: the rooted scatter and gather, `global_size` and the all-to-all exchanges move
/// nontrivial data, with zero counts on some ranks, to the expected places.
fn array_tools_data(comm: &SimpleCommunicator) {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let count = |r: usize| (3 * r + 1) % 4;
    let mut ok = true;

    // Scatter from the last rank.
    let root = size - 1;
    let received = if rank == root {
        let counts: Vec<usize> = (0..size).map(count).collect();
        let data: Vec<u64> = (0..size)
            .flat_map(|r| (0..count(r)).map(move |k| encode(root, r, k)))
            .collect();
        scatterv_root(comm, &counts, &data)
    } else {
        scatterv::<u64>(comm, root)
    };
    ok &= received
        == (0..count(rank))
            .map(|k| encode(root, rank, k))
            .collect::<Vec<_>>();

    // Gather to every rank in turn.
    let local: Vec<u64> = (0..count(rank)).map(|k| encode(rank, 0, k)).collect();
    for root in 0..size {
        let gathered = gather_to_rank(&local, root, comm);
        ok &= if rank == root {
            gathered
                == Some(
                    (0..size)
                        .flat_map(|r| (0..count(r)).map(move |k| encode(r, 0, k)))
                        .collect(),
                )
        } else {
            gathered.is_none()
        };
    }
    ok &= global_size(&local, comm) == (0..size).map(count).sum::<usize>();

    // All-to-all with a different count for every pair, including zero.
    let pair_count = |source: usize, destination: usize| (source + 2 * destination) % 3;
    let counts: Vec<usize> = (0..size).map(|d| pair_count(rank, d)).collect();
    let data: Vec<u64> = (0..size)
        .flat_map(|d| (0..pair_count(rank, d)).map(move |k| encode(rank, d, k)))
        .collect();
    let expected_counts: Vec<usize> = (0..size).map(|s| pair_count(s, rank)).collect();
    let expected: Vec<u64> = (0..size)
        .flat_map(|s| (0..pair_count(s, rank)).map(move |k| encode(s, rank, k)))
        .collect();
    let (dense_counts, dense) = all_to_allv(comm, &counts, &data);
    let (sparse_counts, sparse) = sparse_all_to_allv(comm, &counts, &data);
    ok &= dense_counts == expected_counts && dense == expected;
    ok &= sparse_counts == expected_counts && sparse == expected;

    collective_assert(
        comm,
        ok,
        "the array tools must move every value between the expected ranks",
    );
}

/// Contract: an index embedding maps between embedded, local and global indices, and embeds
/// and extracts chunked data, with an empty rank and a subset in reverse order.
fn index_embedding(comm: &SimpleCommunicator) {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let counts = uneven_counts(size, 2);
    let scan = scan_of(&counts);
    let global = uneven_layout(comm, 2);
    let member = |index: usize| index % 3 != 1;

    // Every local index whose global index is a member, in reverse order.
    let subset: Vec<usize> = (0..counts[rank])
        .rev()
        .filter(|&i| member(scan[rank] + i))
        .collect();
    let embedding = IndexEmbedding::new(global.clone(), &subset, comm);

    let expected_counts: Vec<usize> = (0..size)
        .map(|r| {
            (scan[r]..scan[r + 1])
                .filter(|&index| member(index))
                .count()
        })
        .collect();
    let mut ok = embedding.embedded_layout().local_counts() == expected_counts
        && embedding.global_layout().is_same(&global);

    for (e, &local) in subset.iter().enumerate() {
        ok &= embedding.embedded_index_to_local_index(e) == local
            && embedding.local_index_to_embedded_index(local) == Some(e)
            && embedding.embedded_index_to_global_index(e) == scan[rank] + local;
    }
    for index in 0..scan[size] {
        let expected = (scan[rank]..scan[rank + 1])
            .contains(&index)
            .then(|| subset.iter().position(|&local| scan[rank] + local == index))
            .flatten();
        ok &= embedding.global_index_to_embedded_index(index) == expected;
    }
    ok &= (0..counts[rank])
        .filter(|&i| !member(scan[rank] + i))
        .all(|i| embedding.local_index_to_embedded_index(i).is_none());

    let offset = scan[rank];
    for chunk_size in [1, 3] {
        let data: Vec<f64> = subset
            .iter()
            .flat_map(|&local| (0..chunk_size).map(move |j| value_at(offset + local, j)))
            .collect();
        let mut embedded = vec![-1.0; counts[rank] * chunk_size];
        embedding.embed_data(&data, &mut embedded, chunk_size);
        let expected: Vec<f64> = (0..counts[rank])
            .flat_map(|i| {
                (0..chunk_size).map(move |j| {
                    if member(offset + i) {
                        value_at(offset + i, j)
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        ok &= embedded == expected;
        ok &= embedding.extract_embedded_data(&embedded, chunk_size) == data;
    }

    collective_assert(
        comm,
        ok,
        "index embeddings must map indices and data between the subset and the layout",
    );
}

/// Contract: a random global permutation and a mapper with random owned and ghost dofs move
/// chunked data to the expected positions, over an uneven layout with an empty rank.
fn permutation_and_mapper_data(comm: &SimpleCommunicator) {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let layout = uneven_layout(comm, 3);
    let (start, end) = layout.local_range();
    let total = layout.number_of_global_indices();
    let mut ok = true;

    // The same permutation on every rank, split into custom index sets whose sizes differ from
    // the layout's.
    let mut permutation: Vec<usize> = (0..total).collect();
    permutation.shuffle(&mut StdRng::seed_from_u64(42));
    let custom_counts = uneven_counts(size, 5);
    let custom_scan = scan_of(&custom_counts);
    // The last rank takes the indices beyond the custom counts; the counts are clipped to the
    // total.
    let bounds = |r: usize| {
        let begin = custom_scan[r].min(total);
        let finish = if r + 1 == size {
            total
        } else {
            custom_scan[r + 1].min(total)
        };
        (begin, finish.max(begin))
    };
    let (begin, finish) = bounds(rank);
    let custom = &permutation[begin..finish];

    for chunk_size in [1, 3] {
        let data: Vec<f64> = (start..end)
            .flat_map(|index| (0..chunk_size).map(move |j| value_at(index, j)))
            .collect();
        let permutation = DataPermutation::new(layout.clone(), custom, chunk_size);
        ok &= permutation.chunk_size() == chunk_size;

        let mut permuted = vec![0.0; custom.len() * chunk_size];
        permutation.forward_permute(&data, &mut permuted);
        ok &= permuted
            == custom
                .iter()
                .flat_map(|&index| (0..chunk_size).map(move |j| value_at(index, j)))
                .collect::<Vec<_>>();

        let mut restored = vec![0.0; data.len()];
        permutation.backward_permute(&permuted, &mut restored);
        ok &= restored == data;
    }

    // A mapper with owned and ghost dofs in random order, with repetitions.
    let mut rng = StdRng::seed_from_u64(7 + rank as u64);
    let required: Vec<usize> = (0..2 * (end - start) + 3)
        .map(|_| rng.random_range(0..total))
        .collect();
    let ghosts: BTreeSet<usize> = required
        .iter()
        .copied()
        .filter(|index| !(start..end).contains(index))
        .collect();
    for chunk_size in [1, 2] {
        let mapper = Global2LocalDataMapper::new(layout.clone(), &required, chunk_size);
        let data: Vec<f64> = (start..end)
            .flat_map(|index| (0..chunk_size).map(move |j| value_at(index, j)))
            .collect();
        ok &= mapper.map_data(&data)
            == required
                .iter()
                .flat_map(|&index| (0..chunk_size).map(move |j| value_at(index, j)))
                .collect::<Vec<_>>();
        ok &= mapper.index_layout().is_same(&layout) && mapper.chunk_size() == chunk_size;
        let received: BTreeSet<usize> = mapper
            .ghost_communicator()
            .receive_indices()
            .iter()
            .copied()
            .collect();
        ok &= received == ghosts
            && mapper.ghost_communicator().total_receive_count() == ghosts.len()
            && mapper.ghost_communicator().receive_buffer_len() == ghosts.len() * chunk_size;
    }

    collective_assert(
        comm,
        ok,
        "permutations and mappers must deliver the value of each requested index",
    );
}

/// The in- and out-degrees of a distributed graph communicator.
fn graph_degrees(comm: &SimpleCommunicator) -> (usize, usize) {
    let mut indegree = 0;
    let mut outdegree = 0;
    let mut weighted = 0;
    // SAFETY: `comm` is a valid distributed graph communicator for the duration of the call,
    // and the three outputs are valid for writes.
    unsafe {
        mpi_sys::MPI_Dist_graph_neighbors_count(
            comm.as_raw(),
            &mut indegree,
            &mut outdegree,
            &mut weighted,
        );
    }
    (indegree as usize, outdegree as usize)
}

/// Contract: the buffer accessors not covered elsewhere, the neighbourhood communicators of a
/// ghost communicator and the duplicated communicator of sparse routes behave as documented.
fn ghost_buffer_accessors(comm: &SimpleCommunicator) {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    let next = (rank + 1) % size;
    let value = |index: usize, j: usize| 10 * index + j;

    // A ring: every rank ghosts both indices of the next rank, which carry one and two values.
    let mut builder = GhostCommunicatorBuilder::new();
    if size > 1 {
        builder.add_ghosts([(2 * next + 1, next), (2 * next, next)]);
    }
    let ghosts: GhostCommunicator<usize> = builder
        .chunk_sizes(ChunkSizes::PerIndex(Box::new(|index: usize| index % 2 + 1)))
        .build(comm);

    let mut send = ghosts.send_buffer::<usize>();
    let mut ok = send.as_slice().iter().all(|&v| v == 0);
    for (index, chunk) in send.chunks_mut() {
        for (j, entry) in chunk.iter_mut().enumerate() {
            *entry = value(index, j);
        }
    }
    ok &= std::ptr::eq(send.ghosts(), &ghosts);
    let mut received = send.forward();
    ok &= std::ptr::eq(received.ghosts(), &ghosts);
    ok &= received.chunks().all(|(index, chunk)| {
        chunk.len() == index % 2 + 1 && chunk.iter().enumerate().all(|(j, &v)| v == value(index, j))
    });
    for entry in received.as_mut_slice() {
        *entry *= 2;
    }
    let returned = received.backward();
    // Every owned index has exactly one receiver, so every value comes back doubled.
    ok &= returned.chunks().all(|(index, chunk)| {
        chunk
            .iter()
            .enumerate()
            .all(|(j, &v)| v == 2 * value(index, j))
    });
    let snapshot = send.as_slice().to_vec();
    send.as_mut_slice().fill(1);
    ok &= send.into_vec() == vec![1; snapshot.len()];
    ok &= received.into_vec().len() == ghosts.receive_buffer_len();

    // The forward communicator receives from the in-ranks and sends to the out-ranks; the
    // backward communicator reverses both.
    let forward = ghosts.forward_comm();
    let backward = ghosts.backward_comm();
    ok &= forward.size() == comm.size() && forward.rank() == comm.rank();
    ok &= backward.size() == comm.size() && backward.rank() == comm.rank();
    let (ins, outs) = (ghosts.in_ranks().len(), ghosts.out_ranks().len());
    ok &= graph_degrees(forward) == (ins, outs) && graph_degrees(backward) == (outs, ins);
    ok &= ins == usize::from(size > 1) && outs == usize::from(size > 1);

    // Sparse routes exchange on a duplicate of the communicator.
    let routes = SparseRoutes::new(comm, &vec![1; size]);
    ok &= routes.comm().compare(comm) == CommunicatorRelation::Congruent;
    ok &= routes.exchange(&vec![rank as u64; size], 1) == (0..size as u64).collect::<Vec<_>>();

    collective_assert(
        comm,
        ok,
        "ghost buffers and communicators must expose the documented views",
    );
}

/// Contract: the scenario of `examples/ghost_communicator.rs`, which is written for three
/// ranks. The other rank counts build an empty communicator.
fn ghost_example_port(comm: &SimpleCommunicator) {
    let rank = comm.rank() as usize;
    let three = comm.size() == 3;
    let ghosts: &[(usize, usize)] = match (three, rank) {
        (false, _) => &[],
        (true, 0) => &[(5, 1), (6, 1)],
        (true, 1) => &[(10, 2)],
        (true, _) => &[(5, 1), (0, 0), (1, 0), (2, 0)],
    };
    let ghost_comm = GhostCommunicatorBuilder::new()
        .add_ghosts(ghosts.iter().copied())
        .build(comm);

    let data: &[i32] = match (three, rank) {
        (false, _) => &[],
        (true, 0) => &[10, 11, 12],
        (true, 1) => &[13, 14, 13],
        (true, _) => &[15],
    };
    let mut received = vec![0; ghost_comm.total_receive_count()];
    ghost_comm.forward_send_values(data, &mut received);
    let mut sent_back = vec![0; ghost_comm.total_send_count()];
    ghost_comm.backward_send_values(&received, &mut sent_back);

    let mut ok = sent_back == data;
    if three {
        ok &= match rank {
            0 => {
                ghost_comm.receive_indices() == [5, 6]
                    && received == [13, 14]
                    && ghost_comm.send_indices() == [0, 1, 2]
                    && ghost_comm.out_ranks() == [2]
                    && ghost_comm.in_ranks() == [1]
            }
            1 => {
                ghost_comm.receive_indices() == [10]
                    && received == [15]
                    && ghost_comm.send_indices() == [5, 6, 5]
                    && ghost_comm.out_ranks() == [0, 2]
                    && ghost_comm.send_counts() == [2, 1]
            }
            _ => {
                ghost_comm.receive_indices() == [0, 1, 2, 5]
                    && received == [10, 11, 12, 13]
                    && ghost_comm.send_indices() == [10]
                    && ghost_comm.in_ranks() == [0, 1]
                    && ghost_comm.receive_counts() == [3, 1]
            }
        };
    } else {
        ok &= ghost_comm.total_receive_count() == 0 && ghost_comm.total_send_count() == 0;
    }

    // Owner-chosen chunk sizes, exchanged in buffers, and back.
    let chunked = GhostCommunicatorBuilder::new()
        .add_ghosts(ghosts.iter().copied())
        .chunk_sizes(ChunkSizes::PerIndex(Box::new(|index: usize| index % 3 + 1)))
        .build(comm);
    let send = chunked.send_buffer_from_fn(|index, chunk: &mut [usize]| {
        for (k, value) in chunk.iter_mut().enumerate() {
            *value = 100 * index + k;
        }
    });
    let receive = send.forward();
    ok &= receive.chunks().count() == ghosts.len()
        && receive.chunks().all(|(index, chunk)| {
            chunk.len() == index % 3 + 1
                && chunk.iter().enumerate().all(|(k, &v)| v == 100 * index + k)
        });
    ok &= receive.backward().as_slice() == send.as_slice();

    collective_assert(
        comm,
        ok,
        "the ghost communicator example must exchange the documented values",
    );
}
