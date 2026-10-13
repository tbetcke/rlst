//! Distributed computations in RLST
//!
//! RLST supports MPI distributed arrays and sparse matrices. Moreover,
//! it provides a number of utility routines for working with distributed data layouts.
//!
//! # Index Layouts
//!
//! The basic structure to store information about data distributions is the [IndexLayout](crate::distributed_tools::IndexLayout).
//! It stores an MPI communicator `comm` and a vector `scan` of length `1 + nprocs` where `nprocs` is the number of processes in the MPI
//! communicator. The range `scan[i]..scan[i + 1]` contains the indices on rank `i` (excluding the index `scan[i + 1]`).
//!
//! The easiest way to create an index layout is to use the function [IndexLayout::from_local_counts](crate::distributed_tools::IndexLayout::from_local_counts).
//! Each process calls this function with the number of local variables. They then communicate their local count to all other processes so that each process
//! can construct the complete `scan` vector.
//!
//! It is also possible to construct an index layout from [IndexLayout::from_equidistributed_chunks](crate::distributed_tools::IndexLayout::from_equidistributed_chunks). This expects the `nchunks` describing the overall number of chunks across all processes and a parameter `chunk_size` that specifies how many indices
//! are associated with each chunk. This is useful for example when we want to distribute n vectors with 3 elements each. Then `nchunks` is `n` and `chunk_size`
//! is 3. The resulting distribution will ensure that each process has a multiple of 3 local indices.
//!
//!
//!
//! # Distributed arrays
//!
//! RLST provides a [DistributedArray](crate::sparse::distributed_array::DistributedArray) type.
//! This consists of an [IndexLayout](crate::distributed_tools::IndexLayout) and a local [Array](crate::dense::array::Array) type.
//! Data is always distributed along the first dimension, meaning that for distributed matrices
//! each process owns consecutive rows. A distributed array stores its layout in an `Rc`, so callers
//! can cheaply clone the shared layout handle and use it independently while mutating local data.
//!
//! Distributed arrays support most operations from standard arrays whereas the result always refers to the global array, e.g. the method `inner` performs an inner product across all processes, or `shape` returns the global shape of the distributed array.
//!
//! For distributed arrays the following methods are available to distribute data to multiple processes or gather data on a single process.
//!
//! - [DistributedArray::gather_to_all](crate::sparse::distributed_array::DistributedArray::gather_to_all):
//!   Gather the global array into local arrays on each process.
//! - [`DistributedArray::gather_to_one`](crate::sparse::distributed_array::DistributedArray::gather_to_one):
//!   non-root side of a gather to `root`.
//! - [`DistributedArray::gather_to_one_root`](crate::sparse::distributed_array::DistributedArray::gather_to_one_root):
//!   root side of the same gather.
//! - [`ScatterFromOne::scatter_from_one`](crate::traits::distributed::ScatterFromOne::scatter_from_one):
//!   non-root side of scattering an array along its first dimension.
//! - [`ScatterFromOne::scatter_from_one_root`](crate::traits::distributed::ScatterFromOne::scatter_from_one_root):
//!   root side of the same scatter.
//!
//! These paired methods still form one collective operation: every rank must call the appropriate
//! side with the same root and compatible layout. On the root, the source first dimension must
//! equal the layout's global size; all trailing dimensions are preserved.
//!
//! ## Empty partitions and migration
//!
//! An [`IndexLayout`](crate::distributed_tools::IndexLayout) may contain repeated
//! offsets, so a rank may own no indices and the entire layout may be empty. Every
//! rank must still call distributed operations in the same order; an empty local
//! slice is not a reason to skip a collective. Gather and redistribution routines
//! preserve empty slices, and sortedness of a globally empty sequence is `true`.
//!
//! `global_min` and `global_max` now return `Option<T>`: they return `None` when
//! the global input is empty. Callers that previously assumed a value must handle
//! this case explicitly. `communicate_back` returns the first value on the next
//! *nonempty* rank (or `None`), including when the calling rank is empty.
//! `parsort` accepts empty local input and preserves all records. Its output is
//! exactly balanced: rank `r` of `P` receives the records at global positions
//! `⌊r N / P⌋..⌊(r + 1) N / P⌋`, so ranks are empty only when there are fewer
//! records than ranks.
//!
//! Chunked ghost, mapping, and permutation operations require a positive,
//! communicator-consistent chunk size even when the local input is empty. Roots,
//! layouts, and other collective metadata must likewise agree on every rank.
//! RLST validates metadata before a payload phase where possible, but cannot
//! recover from ranks that do not enter the same collective sequence, mismatched
//! communicators, arbitrary user panics, allocation failure, or MPI runtime
//! failure. Empty subsets embedded with
//! [`IndexEmbedding`](crate::distributed_tools::index_embedding::IndexEmbedding) leave all
//! unselected output positions at `Default::default()`.
//!
//! # Distributed sparse matrices
//!
//! RLST supports distributed CSR matrices via the [DistributedCsrMatrix](crate::sparse::distributed_csr_mat::DistributedCsrMatrix)
//! that operate on distributed arrays. A distributed matrix is defined by a domain layout and a range layout.
//! The domain layout determines the distribution of vectors that are multiplied with the sparse matrix and the range layout determines the distribution of
//! the rows of the distributed sparse matrix. To create a distributed sparse matrix the method
//! [DistributedCsrMatrix::from_aij](crate::traits::sparse::FromAijDistributed::from_aij)
//! from the [FromAijDistributed](crate::traits::sparse::FromAijDistributed) trait. Each process can contribute arbitrary global elements, also for rows that
//! are not associated with the current rank. The constructor routines ensures that elements are communicated to the correct process. Multiple elements associated
//! with the same global row and column are summed up on creation. All ranks must participate,
//! including ranks contributing empty slices. The three local AIJ slices must have equal lengths;
//! global row indices are bounded by the range layout and columns by the domain layout.
//!
//! # Distributed communication patterns
//!
//! RLST has support routines for a number of distributed communication patterns.
//!
//! ## Tools for distributed arrays
//!
//! - [sort_to_bins](crate::distributed_tools::array_tools::sort_to_bins): Take a sorted sequence and distribute it into half-open bins. This is useful
//!   for redistributing data across MPI ranks when the data distribution changes.
//! - [all_to_allv](crate::distributed_tools::array_tools::all_to_allv): A simplified MPI all_to_all communication routine.
//! - [sparse_all_to_allv](crate::distributed_tools::array_tools::sparse_all_to_allv): The same arguments and result as `all_to_allv`, for
//!   sparse counts. See "Sparse exchanges" below.
//! - [scatterv](crate::distributed_tools::array_tools::scatterv): A simplified MPI scatter communication routine.
//! - [scatterv_root](crate::distributed_tools::array_tools::scatterv_root): A simplified MPI scatter communication routine. Call from root.
//!
//! ## Ghost communicators
//!
//! We are often in situations where we need data associated with indices on other processes. These indices are called ghost indices.
//! Underlying this is the pattern of a communication graph in which a process `j` receives data from all processes that are original owners of
//! ghost indices that `j` requires and in which `j` sends out data to all processes who have ghost indices from process `j`. A ghost communicator
//! is created with a [GhostCommunicatorBuilder](crate::distributed_tools::ghost_communicator::GhostCommunicatorBuilder): each process adds the
//! global indices of the ghosts it needs together with their owning ranks, and then calls the collective
//! [build](crate::distributed_tools::ghost_communicator::GhostCommunicatorBuilder::build) method.
//!
//! The builder does not assign positions to the ghosts. The order in which ghost values arrive is given by
//! [GhostCommunicator::receive_indices](crate::distributed_tools::ghost_communicator::GhostCommunicator::receive_indices): the ghosts are
//! grouped by ascending owning rank and keep their insertion order within one owner. The owners send the values of
//! [GhostCommunicator::send_indices](crate::distributed_tools::ghost_communicator::GhostCommunicator::send_indices) in that order.
//!
//! Each index may carry several values. The number of values is fixed when the communicator is built, with
//! [ChunkSizes](crate::distributed_tools::ghost_communicator::ChunkSizes): either a uniform chunk size, or a function with which each owner
//! determines the chunk size of each of its indices. The chunk of the `i`-th receive index is at
//! `receive_offsets()[i]..receive_offsets()[i + 1]` in a receive buffer of length `receive_buffer_len()`, and correspondingly for send buffers.
//!
//! Ghost data is exchanged in buffers created by the communicator. Their layout always matches the communicator, so an exchange cannot fail
//! because of a wrong buffer length.
//!
//! - [GhostCommunicator::send_buffer_from_fn](crate::distributed_tools::ghost_communicator::GhostCommunicator::send_buffer_from_fn) (or
//!   [send_buffer](crate::distributed_tools::ghost_communicator::GhostCommunicator::send_buffer)): the owners fill the chunk of each index they send.
//! - [GhostSendBuffer::forward](crate::distributed_tools::ghost_communicator::GhostSendBuffer::forward): each process forward sends data to the processes that require them and
//!   receives a [GhostReceiveBuffer](crate::distributed_tools::ghost_communicator::GhostReceiveBuffer) whose chunks follow `receive_indices()`.
//! - [GhostReceiveBuffer::backward](crate::distributed_tools::ghost_communicator::GhostReceiveBuffer::backward): send data from the ghost processes backwards to the owners.
//!   Receivers can start from [GhostCommunicator::receive_buffer](crate::distributed_tools::ghost_communicator::GhostCommunicator::receive_buffer).
//!
//! To exchange directly into existing storage, [GhostCommunicator::forward_send_values](crate::distributed_tools::ghost_communicator::GhostCommunicator::forward_send_values) and
//! [GhostCommunicator::backward_send_values](crate::distributed_tools::ghost_communicator::GhostCommunicator::backward_send_values) take slices instead. Their lengths are checked
//! locally. If a slice has the wrong length, the rank still takes part in the exchange with default values, so its neighbours do not block but
//! receive default values from it, and then panics.
//!
//! Only the neighbouring processes communicate in these exchanges. Invalid metadata, such as unknown owners or disagreeing uniform chunk
//! sizes, is rejected collectively when the communicator is built. The build itself sends each process's ghost indices to their owners by a
//! sparse exchange (see below), so it exchanges no array with one entry per process.
//!
//! A complete worked out example for the use of the ghost communicator can be found in `examples/ghost_communicator.rs`.
//!
//! ## Sparse exchanges
//!
//! `all_to_allv` exchanges the counts of every pair of processes in an `MPI_Alltoall` and then the data in an `MPI_Alltoallv`. Both take
//! time proportional to the number of processes `P`, even when each process sends to a few others only. For such sparse patterns RLST
//! offers two alternatives whose cost is that of the messages actually sent plus `O(log P)`.
//!
//! - [sparse_all_to_allv](crate::distributed_tools::array_tools::sparse_all_to_allv) is a drop-in replacement for `all_to_allv`: the same
//!   arguments, and for valid input the same counts and data, element for element. It sends one message to each process with a nonzero
//!   count and learns its sources by non-blocking consensus (a synchronous send per message, a matched probe for whatever arrives, and a
//!   non-blocking barrier once its own messages have been received), then agrees the validity of the input in one all-reduce. Use it for
//!   patterns that change from call to call. With dense counts `all_to_allv` is usually faster.
//! - [SparseRoutes](crate::distributed_tools::sparse_routes::SparseRoutes) is for a pattern that is built once and used many times. It is
//!   created from each process's item counts, as passed to `all_to_allv`, and learns its sources by the same consensus. After that,
//!   [exchange](crate::distributed_tools::sparse_routes::SparseRoutes::exchange) sends any number of values per item along the routes and
//!   [exchange_back](crate::distributed_tools::sparse_routes::SparseRoutes::exchange_back) returns values along the reversed routes, with
//!   point-to-point messages between the processes that share a route only and no collective. The values arrive laid out as with
//!   `all_to_allv`.
//!
//! Every process must take part in each call, including processes that send and receive nothing. `sparse_all_to_allv` and the ghost
//! communicator build send their messages on the caller's communicator with the tag
//! [SPARSE_EXCHANGE_TAG](crate::distributed_tools::array_tools::SPARSE_EXCHANGE_TAG) and probe for that tag only: during the call the caller
//! must not use the tag on the communicator, nor have a receive with `MPI_ANY_TAG` pending on it. `SparseRoutes` duplicates the communicator
//! when it is created, so its exchanges never meet the caller's messages.
//!
//! None of these routines passes an empty buffer to MPI. With Open MPI, an empty `Vec` of a one-byte type has the address `0x1`, which is
//! `MPI_IN_PLACE`, and a collective given it as a buffer fails with `MPI_ERR_ARG`; code that calls MPI directly should guard against it too.
//!
//! ## Mapping global to local data
//!
//! Imagine a global index set with indices from 0 to 9.  The first process may need the indices `0, 1, 2, 3, 4, 5, 6`, and the second process
//! may need the indices `5, 6, 7, 8, 9`. Notice that 5 and 6 are repeated for both processes.
//! The [Global2LocalDataMapper](crate::distributed_tools::data_mapper::Global2LocalDataMapper) manages this communication
//! pattern. It is instantiated with [Global2LocalDataMapper::new](crate::distributed_tools::data_mapper::Global2LocalDataMapper) which takes the
//! index layout, the required global dofs, and the chunk size, which is the number of values associated with each dof.
//! The method [Global2LocalDataMapper::map_data](crate::distributed_tools::data_mapper::Global2LocalDataMapper::map_data) then maps a vector of global data
//! to the required local data.
//!
//! ## Data permutations
//!
//! Consider an index layout in which the first process owns indices `0, 1, 2, 3, 4` and the second process owns the indices `5, 6, 7, 8, 9`.
//! We want to create a new index layout in which the first process owns `0, 2, 4, 6, 8` and the second process owns `1, 3, 5, 7, 9`. Notice that this is
//! a permutation of the original layout. Each index still occurs exactly once. The [DataPermutation](crate::distributed_tools::permutation::DataPermutation)
//! type provides [forward_permute](crate::distributed_tools::permutation::DataPermutation::forward_permute) and a
//! [backward_permute](crate::distributed_tools::permutation::DataPermutation::backward_permute) method to map data from a given index layout to a custom
//! permutation and back. As for the mapper, the chunk size is given when the permutation is created.
