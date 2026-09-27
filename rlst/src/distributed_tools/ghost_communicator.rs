//! Ghost Communicator
//!
//! This module provides a simple ghost communication framework on top of `rsmpi`.
//! Each process declares the ghost indices it requires together with the ranks that own
//! them. The library then sets up neighbourhood communicators that forward the values of
//! owned indices to every process that holds them as ghosts. A backward communicator allows
//! a receiver to update ghost values and send them back to the original owner.
//!
//! Communicators are created with [GhostCommunicatorBuilder]. Every index may carry a chunk of
//! several values; the chunk sizes are fixed when the communicator is built, see [ChunkSizes].
//! Values are exchanged in buffers created by the communicator, [GhostSendBuffer] and
//! [GhostReceiveBuffer], whose layout always matches it. Slice-based exchanges are available
//! for callers that need to write into existing storage.
//!
//! # Example
//!
//! A fully worked example is provided in the file `examples/ghost_communicator.rs`.

use std::os::raw::c_void;

use mpi_sys;

use mpi::topology::SimpleCommunicator;
use mpi::traits::{AsRaw, Communicator, CommunicatorCollectives, Equivalence, FromRaw};

/// Neighbourhood all-to-all exchange with variable counts.
///
/// # Safety
///
/// `send_counts` and `send_displacements` must have one entry per out-neighbour of `comm`,
/// and every range `displacement..displacement + count` must lie within `send`. The same
/// holds for the receive arguments with respect to the in-neighbours and `receive`.
unsafe fn neighbor_alltoallv<T: Equivalence>(
    comm: &SimpleCommunicator,
    send: &[T],
    send_counts: &[i32],
    send_displacements: &[i32],
    receive: &mut [T],
    receive_counts: &[i32],
    receive_displacements: &[i32],
) {
    // SAFETY: The caller guarantees that counts and displacements describe ranges inside the
    // buffers and match the neighbourhood of `comm`. The datatype matches the element type.
    unsafe {
        mpi_sys::MPI_Neighbor_alltoallv(
            send.as_ptr() as *const c_void,
            send_counts.as_ptr(),
            send_displacements.as_ptr(),
            <T as Equivalence>::equivalent_datatype().as_raw(),
            receive.as_mut_ptr() as *mut c_void,
            receive_counts.as_ptr(),
            receive_displacements.as_ptr(),
            <T as Equivalence>::equivalent_datatype().as_raw(),
            comm.as_raw(),
        );
    }
}

/// Create a distributed graph communicator with the given in- and out-neighbours.
fn dist_graph_create<C: Communicator>(
    comm: &C,
    sources: &[i32],
    destinations: &[i32],
) -> SimpleCommunicator {
    let source_count = i32::try_from(sources.len()).expect("collectively validated neighbours");
    let destination_count =
        i32::try_from(destinations.len()).expect("collectively validated neighbours");
    // SAFETY: The neighbour arrays are valid for the given lengths, and the new communicator
    // handle is written by MPI before it is wrapped.
    unsafe {
        let mut raw_comm = mpi_sys::RSMPI_COMM_NULL;
        mpi_sys::MPI_Dist_graph_create_adjacent(
            comm.as_raw(),
            source_count,
            sources.as_ptr(),
            mpi_sys::RSMPI_UNWEIGHTED(),
            destination_count,
            destinations.as_ptr(),
            mpi_sys::RSMPI_UNWEIGHTED(),
            mpi_sys::RSMPI_INFO_NULL,
            0,
            &mut raw_comm,
        );
        SimpleCommunicator::from_raw(raw_comm)
    }
}

/// Collectively assert that a local condition holds on every rank.
fn collective_assert<C: CommunicatorCollectives>(comm: &C, local: bool, message: &str) {
    let mut global = false;
    comm.all_reduce_into(
        &local,
        &mut global,
        mpi::collective::SystemOperation::logical_and(),
    );
    assert!(global, "{message}");
}

/// Prefix sums of `sizes` with a leading zero, or `None` on overflow.
fn offsets(sizes: &[usize]) -> Option<Vec<usize>> {
    let mut offsets = Vec::with_capacity(1 + sizes.len());
    let mut total = 0_usize;
    offsets.push(total);
    for &size in sizes {
        total = total.checked_add(size)?;
        offsets.push(total);
    }
    Some(offsets)
}

/// Per-neighbour value counts and displacements for the index blocks given by
/// `index_counts`, or `None` if they do not fit into MPI's `i32` counts.
fn value_counts(index_counts: &[i32], offsets: &[usize]) -> Option<(Vec<i32>, Vec<i32>)> {
    i32::try_from(*offsets.last()?).ok()?;
    let mut counts = Vec::with_capacity(index_counts.len());
    let mut displacements = Vec::with_capacity(index_counts.len());
    let mut start = 0_usize;
    for &count in index_counts {
        let end = start + count as usize;
        displacements.push(offsets[start] as i32);
        counts.push((offsets[end] - offsets[start]) as i32);
        start = end;
    }
    Some((counts, displacements))
}

/// Number of values that each index carries in a ghost exchange.
///
/// The chunk sizes are fixed when a [GhostCommunicator] is built. The owner of an index
/// determines the size of its chunk; receivers learn it during the build.
pub enum ChunkSizes<'a, I> {
    /// Every index owned by this rank carries the same positive number of values.
    ///
    /// The rank also expects this many values for each of its ghosts. The build rejects the
    /// communicator if an owner of one of its ghosts declares a different size.
    Uniform(usize),
    /// The number of values for an owned index is given by a function of its global index.
    ///
    /// The function is called on the owning rank, once for each index that another rank
    /// requests as a ghost. A size of zero is allowed.
    PerIndex(Box<dyn Fn(I) -> usize + 'a>),
}

impl<I> Default for ChunkSizes<'_, I> {
    fn default() -> Self {
        Self::Uniform(1)
    }
}

/// Builder for a [GhostCommunicator].
///
/// Ghosts are added one at a time together with their owning ranks. The builder does not
/// assign positions to the ghosts: the order in which the built communicator expects and
/// delivers ghost values is given by [GhostCommunicator::receive_indices].
///
/// ```no_run
/// use rlst::distributed_tools::{ChunkSizes, GhostCommunicatorBuilder};
///
/// let universe = mpi::initialize().unwrap();
/// let world = universe.world();
///
/// // Run with at least two ranks. Every rank requires index 5 from rank 1 and index 0
/// // from rank 0, with two values per index.
/// let ghosts = GhostCommunicatorBuilder::<usize>::new()
///     .add_ghost(5, 1)
///     .add_ghost(0, 0)
///     .chunk_sizes(ChunkSizes::Uniform(2))
///     .build(&world);
///
/// // The ghosts arrive grouped by owning rank.
/// assert_eq!(ghosts.receive_indices(), [0, 5]);
/// assert_eq!(ghosts.receive_buffer_len(), 4);
/// ```
pub struct GhostCommunicatorBuilder<'a, I> {
    ghosts: Vec<(I, usize)>,
    chunk_sizes: ChunkSizes<'a, I>,
}

impl<I> Default for GhostCommunicatorBuilder<'_, I> {
    fn default() -> Self {
        Self {
            ghosts: Vec::new(),
            chunk_sizes: ChunkSizes::default(),
        }
    }
}

impl<'a, I: Default + Copy + Equivalence> GhostCommunicatorBuilder<'a, I> {
    /// Create a builder without ghosts and with a uniform chunk size of one.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add the ghost `index`, owned by the rank `owner`.
    ///
    /// The owner may be the current rank; its values are then exchanged with itself. An index
    /// may be added more than once: every occurrence is a separate ghost with its own chunk in
    /// the receive buffer, and the owner sends and receives one chunk per occurrence.
    pub fn add_ghost(&mut self, index: I, owner: usize) -> &mut Self {
        self.ghosts.push((index, owner));
        self
    }

    /// Add several ghosts, given as pairs of global index and owning rank.
    pub fn add_ghosts(&mut self, ghosts: impl IntoIterator<Item = (I, usize)>) -> &mut Self {
        self.ghosts.extend(ghosts);
        self
    }

    /// Set the number of values that each index carries. The default is `Uniform(1)`.
    pub fn chunk_sizes(&mut self, chunk_sizes: ChunkSizes<'a, I>) -> &mut Self {
        self.chunk_sizes = chunk_sizes;
        self
    }

    /// Build the ghost communicator.
    ///
    /// This is a collective operation on `comm`. Every rank validates its metadata before
    /// any rank panics: owners must be ranks of `comm`, `Uniform` chunk sizes must be
    /// positive and agree between owners and receivers, and all value counts must fit into
    /// MPI's `i32` counts.
    pub fn build<C: CommunicatorCollectives>(&self, comm: &C) -> GhostCommunicator<I> {
        let size = comm.size() as usize;

        // Validate on every rank before the all-to-all graph setup so invalid local
        // ownership cannot strand peers in that collective.
        let mut receive_counts = vec![0_i32; size];
        let mut valid = !matches!(self.chunk_sizes, ChunkSizes::Uniform(0));
        for &(_, owner) in &self.ghosts {
            valid &= owner < size;
            if owner < size {
                receive_counts[owner] += 1;
            }
        }
        collective_assert(comm, valid, "invalid ghost owner or chunk size metadata");

        // The receive indices are the ghosts sorted by owning rank. The sort is stable, so
        // ghosts with the same owner keep the order in which they were added.
        let receive_indices = {
            let mut ghosts = self.ghosts.clone();
            ghosts.sort_by_key(|&(_, owner)| owner);
            ghosts
                .into_iter()
                .map(|(index, _)| index)
                .collect::<Vec<_>>()
        };

        // Tell the owners how many of their indices each rank requires.
        let mut send_counts = vec![0_i32; size];
        comm.all_to_all_into(&receive_counts, &mut send_counts);

        // Restrict the counts to the neighbours. In-neighbours are the ranks from which we
        // receive data and out-neighbours are the ranks to which we send data.
        let mut in_ranks = Vec::<i32>::new();
        let mut out_ranks = Vec::<i32>::new();
        let mut neighbor_receive_counts = Vec::<i32>::new();
        let mut neighbor_send_counts = Vec::<i32>::new();
        let mut receive_displacements = Vec::<i32>::new();
        let mut send_displacements = Vec::<i32>::new();
        let mut total_receive_count = 0_i32;
        let mut total_send_count = 0_i32;
        for rank in 0..size {
            if receive_counts[rank] != 0 {
                in_ranks.push(rank as i32);
                neighbor_receive_counts.push(receive_counts[rank]);
                receive_displacements.push(total_receive_count);
                total_receive_count += receive_counts[rank];
            }
            if send_counts[rank] != 0 {
                out_ranks.push(rank as i32);
                neighbor_send_counts.push(send_counts[rank]);
                send_displacements.push(total_send_count);
                total_send_count += send_counts[rank];
            }
        }
        collective_assert(
            comm,
            i32::try_from(in_ranks.len()).is_ok() && i32::try_from(out_ranks.len()).is_ok(),
            "ghost communication metadata exceeds MPI limits",
        );

        let forward_comm = dist_graph_create(comm, &in_ranks, &out_ranks);
        // The backward communicator simply reverses the in and out neighbours.
        let backward_comm = dist_graph_create(comm, &out_ranks, &in_ranks);

        // The receivers know which indices they need from each process, but the owners do
        // not know yet which indices to send. So we send the receive indices to the owners.
        let mut send_indices = vec![I::default(); total_send_count as usize];
        // SAFETY: The counts and displacements partition `receive_indices` and
        // `send_indices` over the neighbours of the backward communicator.
        unsafe {
            neighbor_alltoallv(
                &backward_comm,
                &receive_indices,
                &neighbor_receive_counts,
                &receive_displacements,
                &mut send_indices,
                &neighbor_send_counts,
                &send_displacements,
            );
        }

        // The owners now determine the chunk sizes of their indices and tell the receivers.
        let send_chunk_sizes = match &self.chunk_sizes {
            ChunkSizes::Uniform(chunk_size) => vec![*chunk_size; send_indices.len()],
            ChunkSizes::PerIndex(chunk_size) => send_indices
                .iter()
                .map(|&index| chunk_size(index))
                .collect(),
        };
        let mut receive_chunk_sizes = vec![0_usize; receive_indices.len()];
        // SAFETY: The counts and displacements partition the chunk size buffers over the
        // neighbours of the forward communicator.
        unsafe {
            neighbor_alltoallv(
                &forward_comm,
                &send_chunk_sizes,
                &neighbor_send_counts,
                &send_displacements,
                &mut receive_chunk_sizes,
                &neighbor_receive_counts,
                &receive_displacements,
            );
        }

        let send_offsets = offsets(&send_chunk_sizes);
        let receive_offsets = offsets(&receive_chunk_sizes);
        let send_values = send_offsets
            .as_deref()
            .and_then(|offsets| value_counts(&neighbor_send_counts, offsets));
        let receive_values = receive_offsets
            .as_deref()
            .and_then(|offsets| value_counts(&neighbor_receive_counts, offsets));
        let chunks_agree = match self.chunk_sizes {
            ChunkSizes::Uniform(chunk_size) => {
                receive_chunk_sizes.iter().all(|&size| size == chunk_size)
            }
            ChunkSizes::PerIndex(_) => true,
        };
        collective_assert(
            comm,
            chunks_agree && send_values.is_some() && receive_values.is_some(),
            "invalid or inconsistent ghost chunk sizes",
        );
        let (send_value_counts, send_value_displacements) = send_values.unwrap();
        let (receive_value_counts, receive_value_displacements) = receive_values.unwrap();

        GhostCommunicator {
            out_ranks,
            in_ranks,
            send_indices,
            receive_indices,
            send_counts: neighbor_send_counts,
            receive_counts: neighbor_receive_counts,
            send_chunk_sizes,
            receive_chunk_sizes,
            send_offsets: send_offsets.unwrap(),
            receive_offsets: receive_offsets.unwrap(),
            send_value_counts,
            receive_value_counts,
            send_value_displacements,
            receive_value_displacements,
            forward_comm,
            backward_comm,
        }
    }
}

/// Ghost communicator
///
/// Created with [GhostCommunicatorBuilder]. Values are exchanged in flat buffers:
/// - the send buffer of a forward exchange holds the values of [send_indices](Self::send_indices)
///   in that order, the chunk of the `i`-th index at `send_offsets()[i]..send_offsets()[i + 1]`;
/// - the receive buffer holds the values of [receive_indices](Self::receive_indices), laid out
///   by [receive_offsets](Self::receive_offsets) in the same way.
///
/// A backward exchange uses the same two layouts with their roles reversed.
pub struct GhostCommunicator<I: Default + Copy + Equivalence> {
    out_ranks: Vec<i32>,
    in_ranks: Vec<i32>,
    send_indices: Vec<I>,
    receive_indices: Vec<I>,
    send_counts: Vec<i32>,
    receive_counts: Vec<i32>,
    send_chunk_sizes: Vec<usize>,
    receive_chunk_sizes: Vec<usize>,
    send_offsets: Vec<usize>,
    receive_offsets: Vec<usize>,
    send_value_counts: Vec<i32>,
    receive_value_counts: Vec<i32>,
    send_value_displacements: Vec<i32>,
    receive_value_displacements: Vec<i32>,
    forward_comm: SimpleCommunicator,
    backward_comm: SimpleCommunicator,
}

impl<I: Default + Copy + Equivalence> GhostCommunicator<I> {
    /// Return the ranks to which the current process sends to.
    pub fn out_ranks(&self) -> &[i32] {
        &self.out_ranks
    }

    /// Return the ranks from which the current process receives from.
    pub fn in_ranks(&self) -> &[i32] {
        &self.in_ranks
    }

    /// Return the owned indices that are sent out from the current process.
    ///
    /// The indices are grouped by the ranks in [out_ranks](Self::out_ranks).
    pub fn send_indices(&self) -> &[I] {
        &self.send_indices
    }

    /// Return the ghost indices that are received on the current process.
    ///
    /// This is the order of the ghosts in every receive buffer. The ghosts are sorted by
    /// ascending owning rank; ghosts with the same owner keep the order in which they were
    /// added to the [GhostCommunicatorBuilder].
    pub fn receive_indices(&self) -> &[I] {
        &self.receive_indices
    }

    /// Return the number of indices that are sent to each rank in [out_ranks](Self::out_ranks).
    pub fn send_counts(&self) -> &[i32] {
        &self.send_counts
    }

    /// Return the number of indices that are received from each rank in
    /// [in_ranks](Self::in_ranks).
    pub fn receive_counts(&self) -> &[i32] {
        &self.receive_counts
    }

    /// Return the total number of indices that are sent.
    pub fn total_send_count(&self) -> usize {
        self.send_indices.len()
    }

    /// Return the total number of indices that are received.
    pub fn total_receive_count(&self) -> usize {
        self.receive_indices.len()
    }

    /// Return the chunk size of each index in [send_indices](Self::send_indices).
    pub fn send_chunk_sizes(&self) -> &[usize] {
        &self.send_chunk_sizes
    }

    /// Return the chunk size of each index in [receive_indices](Self::receive_indices).
    pub fn receive_chunk_sizes(&self) -> &[usize] {
        &self.receive_chunk_sizes
    }

    /// Return the offsets of the send chunks in a send buffer.
    ///
    /// The values of the `i`-th send index are at `send_offsets()[i]..send_offsets()[i + 1]`.
    /// The slice has `total_send_count() + 1` entries.
    pub fn send_offsets(&self) -> &[usize] {
        &self.send_offsets
    }

    /// Return the offsets of the receive chunks in a receive buffer.
    ///
    /// The values of the `i`-th receive index are at
    /// `receive_offsets()[i]..receive_offsets()[i + 1]`. The slice has
    /// `total_receive_count() + 1` entries.
    pub fn receive_offsets(&self) -> &[usize] {
        &self.receive_offsets
    }

    /// Return the number of values in a send buffer.
    pub fn send_buffer_len(&self) -> usize {
        *self.send_offsets.last().unwrap()
    }

    /// Return the number of values in a receive buffer.
    pub fn receive_buffer_len(&self) -> usize {
        *self.receive_offsets.last().unwrap()
    }

    /// Return the forward communicator.
    ///
    /// This is a neighbourhood MPI communicator that sends values to the `out` processes
    /// and receives from the `in` processes.
    pub fn forward_comm(&self) -> &SimpleCommunicator {
        &self.forward_comm
    }

    /// Return the backward communicator.
    ///
    /// This is a neighbourhood MPI communicator that sends values to the `in` processes and
    /// receives values from the `out` processes.
    pub fn backward_comm(&self) -> &SimpleCommunicator {
        &self.backward_comm
    }

    /// Create a send buffer with default values, laid out by [send_offsets](Self::send_offsets).
    pub fn send_buffer<T: Equivalence + Copy + Default>(&self) -> GhostSendBuffer<'_, I, T> {
        GhostSendBuffer {
            ghosts: self,
            data: vec![T::default(); self.send_buffer_len()],
        }
    }

    /// Create a send buffer and fill the chunk of each send index with `fill`.
    ///
    /// `fill` is called once per index in [send_indices](Self::send_indices), in order, with
    /// the global index and its chunk of default values.
    pub fn send_buffer_from_fn<T: Equivalence + Copy + Default>(
        &self,
        mut fill: impl FnMut(I, &mut [T]),
    ) -> GhostSendBuffer<'_, I, T> {
        let mut buffer = self.send_buffer();
        for (index, chunk) in buffer.chunks_mut() {
            fill(index, chunk);
        }
        buffer
    }

    /// Create a receive buffer with default values, laid out by
    /// [receive_offsets](Self::receive_offsets).
    ///
    /// Receivers fill such a buffer to send updated ghost values back to their owners with
    /// [GhostReceiveBuffer::backward].
    pub fn receive_buffer<T: Equivalence + Copy + Default>(&self) -> GhostReceiveBuffer<'_, I, T> {
        GhostReceiveBuffer {
            ghosts: self,
            data: vec![T::default(); self.receive_buffer_len()],
        }
    }

    /// Exchange buffers whose lengths match `direction`.
    fn exchange<T: Equivalence>(
        &self,
        direction: Direction,
        out_values: &[T],
        in_values: &mut [T],
    ) {
        let (comm, out_counts, out_displacements, in_counts, in_displacements) = match direction {
            Direction::Forward => (
                &self.forward_comm,
                &self.send_value_counts,
                &self.send_value_displacements,
                &self.receive_value_counts,
                &self.receive_value_displacements,
            ),
            Direction::Backward => (
                &self.backward_comm,
                &self.receive_value_counts,
                &self.receive_value_displacements,
                &self.send_value_counts,
                &self.send_value_displacements,
            ),
        };
        let (out_len, in_len) = self.buffer_lens(direction);
        assert!(
            out_values.len() == out_len && in_values.len() == in_len,
            "ghost exchange buffers must match the communicator layout"
        );
        // SAFETY: The value counts and displacements partition buffers of exactly these
        // lengths over the neighbours of the communicator for this direction.
        unsafe {
            neighbor_alltoallv(
                comm,
                out_values,
                out_counts,
                out_displacements,
                in_values,
                in_counts,
                in_displacements,
            );
        }
    }

    /// Expected lengths of the outgoing and incoming buffer for `direction`.
    fn buffer_lens(&self, direction: Direction) -> (usize, usize) {
        match direction {
            Direction::Forward => (self.send_buffer_len(), self.receive_buffer_len()),
            Direction::Backward => (self.receive_buffer_len(), self.send_buffer_len()),
        }
    }

    /// Exchange caller-provided slices, taking part in the exchange even if they are wrong.
    ///
    /// A slice with the wrong length is replaced by a scratch buffer of default values, so the
    /// neighbours complete the exchange; the rank then panics.
    fn exchange_or_panic<T: Equivalence + Copy + Default>(
        &self,
        direction: Direction,
        out_values: &[T],
        in_values: &mut [T],
    ) {
        let (out_len, in_len) = self.buffer_lens(direction);
        let (out_actual, in_actual) = (out_values.len(), in_values.len());
        let scratch_out;
        let out_values = if out_actual == out_len {
            out_values
        } else {
            scratch_out = vec![T::default(); out_len];
            &scratch_out
        };
        if in_actual == in_len {
            self.exchange(direction, out_values, in_values);
        } else {
            let mut scratch_in = vec![T::default(); in_len];
            self.exchange(direction, out_values, &mut scratch_in);
        }
        assert!(
            out_actual == out_len && in_actual == in_len,
            "ghost exchange buffer mismatch in {direction:?} exchange: outgoing buffer has length {out_actual}, expected {out_len}; incoming buffer has length {in_actual}, expected {in_len}"
        );
    }

    /// Forward send values from caller-provided slices.
    ///
    /// This updates ghosts on the receiver process with the values of the ghosts from
    /// their owning process. `out_values` must have [send_buffer_len](Self::send_buffer_len)
    /// values and `in_values` must have [receive_buffer_len](Self::receive_buffer_len) values.
    /// Prefer [GhostSendBuffer::forward], whose buffers always have the right lengths; use
    /// this method to receive directly into existing storage.
    ///
    /// The lengths are checked locally, without communication. If a slice has the wrong
    /// length, this rank still takes part in the exchange with default values in its place,
    /// so its neighbours complete normally but receive default values from it, and then
    /// panics.
    pub fn forward_send_values<T: Equivalence + Copy + Default>(
        &self,
        out_values: &[T],
        in_values: &mut [T],
    ) {
        self.exchange_or_panic(Direction::Forward, out_values, in_values);
    }

    /// Backward send values from caller-provided slices.
    ///
    /// This back propagates updated ghost values from the receiver to the original owning
    /// process. `out_values` must have [receive_buffer_len](Self::receive_buffer_len) values
    /// and `in_values` must have [send_buffer_len](Self::send_buffer_len) values. Prefer
    /// [GhostReceiveBuffer::backward], whose buffers always have the right lengths.
    ///
    /// Wrong lengths are handled as for [forward_send_values](Self::forward_send_values): this
    /// rank takes part in the exchange with default values and then panics.
    pub fn backward_send_values<T: Equivalence + Copy + Default>(
        &self,
        out_values: &[T],
        in_values: &mut [T],
    ) {
        self.exchange_or_panic(Direction::Backward, out_values, in_values);
    }
}

/// Direction of a ghost exchange.
#[derive(Clone, Copy, Debug)]
enum Direction {
    /// From the owners to the ghosts.
    Forward,
    /// From the ghosts back to the owners.
    Backward,
}

/// Iterate over the chunks of `data`, laid out by `offsets`, together with their indices.
fn chunks<'s, I: Copy, T>(
    indices: &'s [I],
    offsets: &'s [usize],
    data: &'s [T],
) -> impl Iterator<Item = (I, &'s [T])> {
    indices
        .iter()
        .zip(offsets.windows(2))
        .map(move |(&index, range)| (index, &data[range[0]..range[1]]))
}

/// Iterate mutably over the chunks of `data`, laid out by `sizes`, with their indices.
fn chunks_mut<'s, I: Copy, T>(
    indices: &'s [I],
    sizes: &'s [usize],
    mut data: &'s mut [T],
) -> impl Iterator<Item = (I, &'s mut [T])> {
    indices.iter().zip(sizes).map(move |(&index, &size)| {
        let (chunk, rest) = std::mem::take(&mut data).split_at_mut(size);
        data = rest;
        (index, chunk)
    })
}

/// Values of the [send indices](GhostCommunicator::send_indices) of a ghost communicator.
///
/// Created by [GhostCommunicator::send_buffer] or [GhostCommunicator::send_buffer_from_fn],
/// or returned by [GhostReceiveBuffer::backward]. The buffer always has the layout of its
/// communicator, so exchanging it cannot fail.
pub struct GhostSendBuffer<'g, I: Default + Copy + Equivalence, T> {
    ghosts: &'g GhostCommunicator<I>,
    data: Vec<T>,
}

impl<'g, I: Default + Copy + Equivalence, T: Equivalence + Copy + Default>
    GhostSendBuffer<'g, I, T>
{
    /// Send the values to the ghosts on the receiving processes.
    ///
    /// This is a neighbourhood collective: every rank of the communicator must call it.
    pub fn forward(&self) -> GhostReceiveBuffer<'g, I, T> {
        let mut received = self.ghosts.receive_buffer();
        self.ghosts
            .exchange(Direction::Forward, &self.data, &mut received.data);
        received
    }

    /// Return the ghost communicator of this buffer.
    pub fn ghosts(&self) -> &'g GhostCommunicator<I> {
        self.ghosts
    }

    /// Return all values.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Return all values mutably.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Return the values of the `i`-th send index.
    pub fn chunk(&self, i: usize) -> &[T] {
        let offsets = self.ghosts.send_offsets();
        &self.data[offsets[i]..offsets[i + 1]]
    }

    /// Return the values of the `i`-th send index mutably.
    pub fn chunk_mut(&mut self, i: usize) -> &mut [T] {
        let offsets = self.ghosts.send_offsets();
        &mut self.data[offsets[i]..offsets[i + 1]]
    }

    /// Iterate over the send indices together with their values.
    pub fn chunks(&self) -> impl Iterator<Item = (I, &[T])> {
        chunks(
            self.ghosts.send_indices(),
            self.ghosts.send_offsets(),
            &self.data,
        )
    }

    /// Iterate over the send indices together with their values, mutably.
    pub fn chunks_mut(&mut self) -> impl Iterator<Item = (I, &mut [T])> {
        chunks_mut(
            self.ghosts.send_indices(),
            self.ghosts.send_chunk_sizes(),
            &mut self.data,
        )
    }

    /// Return the values as a vector.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

/// Values of the [receive indices](GhostCommunicator::receive_indices) of a ghost communicator.
///
/// Returned by [GhostSendBuffer::forward] or created by [GhostCommunicator::receive_buffer].
/// The buffer always has the layout of its communicator, so exchanging it cannot fail.
pub struct GhostReceiveBuffer<'g, I: Default + Copy + Equivalence, T> {
    ghosts: &'g GhostCommunicator<I>,
    data: Vec<T>,
}

impl<'g, I: Default + Copy + Equivalence, T: Equivalence + Copy + Default>
    GhostReceiveBuffer<'g, I, T>
{
    /// Send the ghost values back to their owning processes.
    ///
    /// This is a neighbourhood collective: every rank of the communicator must call it.
    pub fn backward(&self) -> GhostSendBuffer<'g, I, T> {
        let mut returned = self.ghosts.send_buffer();
        self.ghosts
            .exchange(Direction::Backward, &self.data, &mut returned.data);
        returned
    }

    /// Return the ghost communicator of this buffer.
    pub fn ghosts(&self) -> &'g GhostCommunicator<I> {
        self.ghosts
    }

    /// Return all values.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Return all values mutably.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Return the values of the `i`-th receive index.
    pub fn chunk(&self, i: usize) -> &[T] {
        let offsets = self.ghosts.receive_offsets();
        &self.data[offsets[i]..offsets[i + 1]]
    }

    /// Return the values of the `i`-th receive index mutably.
    pub fn chunk_mut(&mut self, i: usize) -> &mut [T] {
        let offsets = self.ghosts.receive_offsets();
        &mut self.data[offsets[i]..offsets[i + 1]]
    }

    /// Iterate over the receive indices together with their values.
    pub fn chunks(&self) -> impl Iterator<Item = (I, &[T])> {
        chunks(
            self.ghosts.receive_indices(),
            self.ghosts.receive_offsets(),
            &self.data,
        )
    }

    /// Iterate over the receive indices together with their values, mutably.
    pub fn chunks_mut(&mut self) -> impl Iterator<Item = (I, &mut [T])> {
        chunks_mut(
            self.ghosts.receive_indices(),
            self.ghosts.receive_chunk_sizes(),
            &mut self.data,
        )
    }

    /// Return the values as a vector.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

#[cfg(test)]
mod test {
    use super::{offsets, value_counts};

    #[test]
    fn test_offsets() {
        assert_eq!(offsets(&[]), Some(vec![0]));
        assert_eq!(offsets(&[2, 0, 3]), Some(vec![0, 2, 2, 5]));
        assert_eq!(offsets(&[usize::MAX, 1]), None);
    }

    #[test]
    fn test_value_counts() {
        // Two neighbours with two and one indices; chunk sizes 2, 0 and 3.
        let offsets = [0, 2, 2, 5];
        assert_eq!(
            value_counts(&[2, 1], &offsets),
            Some((vec![2, 3], vec![0, 2]))
        );
        assert_eq!(value_counts(&[], &[0]), Some((vec![], vec![])));
        // A neighbour whose indices all have empty chunks.
        assert_eq!(
            value_counts(&[1, 2], &[0, 0, 4, 4]),
            Some((vec![0, 4], vec![0, 0]))
        );
        let limit = i32::MAX as usize;
        assert_eq!(
            value_counts(&[1], &[0, limit]),
            Some((vec![i32::MAX], vec![0]))
        );
        assert_eq!(value_counts(&[1], &[0, limit + 1]), None);
    }
}
