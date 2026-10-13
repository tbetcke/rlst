//! Sparse routes
//!
//! A fixed communication pattern for data that is exchanged repeatedly between the same
//! pairs of ranks: each rank sends a fixed number of items to each of a few ranks, and the
//! values of every item change from one exchange to the next. [SparseRoutes] learns the
//! pattern once, by a sparse dynamic exchange of the counts, and then exchanges values with
//! point-to-point messages between the ranks that share a route only. No exchange
//! communicates an array with one entry per rank or runs a collective, and the received
//! values are laid out as [all_to_allv](crate::distributed_tools::all_to_allv) lays them
//! out, so the two give the same bytes in the same order.
//!
//! ```no_run
//! use mpi::traits::Communicator;
//! use rlst::distributed_tools::SparseRoutes;
//!
//! let universe = mpi::initialize().unwrap();
//! let world = universe.world();
//! let rank = world.rank() as usize;
//! let size = world.size() as usize;
//!
//! // Every rank sends two items to its successor, with three values per item.
//! let mut counts = vec![0; size];
//! counts[(rank + 1) % size] = 2;
//! let routes = SparseRoutes::new(&world, &counts);
//!
//! let values = vec![rank as f64; 2 * 3];
//! let received = routes.exchange(&values, 3);
//! assert_eq!(received, vec![((rank + size - 1) % size) as f64; 2 * 3]);
//!
//! // The values return along the same routes.
//! let returned = routes.exchange_back(&received, 3);
//! assert_eq!(returned, values);
//! ```

use mpi::{
    Tag,
    collective::SystemOperation,
    topology::SimpleCommunicator,
    traits::{Communicator, CommunicatorCollectives, Destination, Equivalence, Source},
};

use crate::distributed_tools::array_tools::{SPARSE_EXCHANGE_TAG, nbx};

/// The tag of the values sent from the senders to the receivers.
const FORWARD_TAG: Tag = 1;
/// The tag of the values sent back from the receivers to the senders.
const BACKWARD_TAG: Tag = 2;

/// Ranks and the number of items for each, one direction of the routes.
type Routes<'s> = (&'s [usize], &'s [usize]);

/// A fixed set of routes between ranks, for repeated sparse exchanges.
///
/// Built from each rank's number of items for every rank, as passed to
/// [all_to_allv](crate::distributed_tools::all_to_allv). Then
/// [exchange](Self::exchange) sends `per_item` values of every item along the routes,
/// and [exchange_back](Self::exchange_back) returns values along the reversed routes.
///
/// Values are sent grouped by destination rank in ascending order, as the data of
/// `all_to_allv`, and received grouped by source rank in ascending order, as its
/// result. Items a rank sends to itself are copied without MPI.
pub struct SparseRoutes {
    comm: SimpleCommunicator,
    rank: usize,
    destinations: Vec<usize>,
    send_counts: Vec<usize>,
    sources: Vec<usize>,
    receive_counts: Vec<usize>,
    max_per_item: usize,
}

impl SparseRoutes {
    /// Learn the routes from the number of items this rank sends to each rank.
    ///
    /// `counts` has one entry per rank of `comm`, as in
    /// [all_to_allv](crate::distributed_tools::all_to_allv).
    ///
    /// # Collective operation
    ///
    /// Every rank of `comm` must call this. It duplicates `comm` (the exchanges run on the
    /// duplicate, so they never meet the caller's messages), sends each count to its rank
    /// by a sparse dynamic exchange on the duplicate, and agrees the validity and
    /// [max_per_item](Self::max_per_item) in one all-reduce.
    ///
    /// # Panics
    ///
    /// Panics on every rank, after the exchange of the counts, if on any rank `counts` does
    /// not have one entry per rank, or a rank sends or receives `2^31` items or more along
    /// one route.
    pub fn new<C: CommunicatorCollectives>(comm: &C, counts: &[usize]) -> Self {
        let size = comm.size() as usize;
        let rank = comm.rank() as usize;
        let comm = comm.duplicate();
        let valid =
            counts.len() == size && counts.iter().all(|&count| i32::try_from(count).is_ok());

        let mut destinations = Vec::new();
        let mut send_counts = Vec::new();
        if valid {
            for (destination, &count) in counts.iter().enumerate() {
                if count != 0 {
                    destinations.push(destination);
                    send_counts.push(count);
                }
            }
        }
        let messages = destinations
            .iter()
            .filter(|&&destination| destination != rank)
            .map(|&destination| (destination, &counts[destination..destination + 1]));
        let mut received = nbx(&comm, SPARSE_EXCHANGE_TAG, messages);
        if valid && counts[rank] != 0 {
            let position = received.partition_point(|&(source, _)| source < rank);
            received.insert(position, (rank, vec![counts[rank]]));
        }
        let (sources, receive_counts): (Vec<usize>, Vec<usize>) = received
            .into_iter()
            .map(|(source, count)| (source, count[0]))
            .unzip();

        // The largest number of values per item for which every route's values fit MPI's
        // i32 counts, agreed as a minimum together with the validity.
        let largest = send_counts
            .iter()
            .chain(&receive_counts)
            .copied()
            .max()
            .unwrap_or(0)
            .max(1);
        let local_max = i32::MAX as usize / largest;
        let mine = [i64::from(!valid), -(local_max as i64)];
        let mut all = [0_i64; 2];
        comm.all_reduce_into(&mine[..], &mut all[..], SystemOperation::max());
        assert!(all[0] == 0, "invalid sparse route counts");

        Self {
            comm,
            rank,
            destinations,
            send_counts,
            sources,
            receive_counts,
            max_per_item: (-all[1]) as usize,
        }
    }

    /// Return the duplicated communicator on which the exchanges run.
    pub fn comm(&self) -> &SimpleCommunicator {
        &self.comm
    }

    /// Return the ranks to which this rank sends items, in ascending order.
    ///
    /// These are the ranks with a nonzero count, the calling rank included if it sends
    /// items to itself.
    pub fn destinations(&self) -> &[usize] {
        &self.destinations
    }

    /// Return the number of items sent to each rank in [destinations](Self::destinations).
    pub fn send_counts(&self) -> &[usize] {
        &self.send_counts
    }

    /// Return the ranks from which this rank receives items, in ascending order.
    pub fn sources(&self) -> &[usize] {
        &self.sources
    }

    /// Return the number of items received from each rank in [sources](Self::sources).
    pub fn receive_counts(&self) -> &[usize] {
        &self.receive_counts
    }

    /// Return the total number of items this rank sends.
    pub fn total_send_count(&self) -> usize {
        self.send_counts.iter().sum()
    }

    /// Return the total number of items this rank receives.
    pub fn total_receive_count(&self) -> usize {
        self.receive_counts.iter().sum()
    }

    /// Return the largest number of values per item that an exchange accepts.
    ///
    /// The same on every rank: for it, the values along every route of every rank fit
    /// MPI's `i32` counts.
    pub fn max_per_item(&self) -> usize {
        self.max_per_item
    }

    /// Send `per_item` values of every item to its destination and return the values
    /// received, grouped by source rank in ascending order.
    ///
    /// `values` holds [total_send_count](Self::total_send_count) times `per_item` values,
    /// grouped by destination rank in ascending order. The result holds
    /// [total_receive_count](Self::total_receive_count) times `per_item` values.
    ///
    /// # Collective operation
    ///
    /// Every rank of the communicator must call this with the same `per_item`, and call
    /// the exchanges of one set of routes in the same order. Only the ranks that share a
    /// route exchange messages; there is no collective.
    ///
    /// # Panics
    ///
    /// Panics on every rank, before communicating, if `per_item` exceeds
    /// [max_per_item](Self::max_per_item); and, after taking part in the exchange with
    /// default values, on a rank whose `values` has the wrong length.
    pub fn exchange<T: Equivalence + Copy + Default>(
        &self,
        values: &[T],
        per_item: usize,
    ) -> Vec<T> {
        self.check_per_item(per_item);
        let mut received = vec![T::default(); self.total_receive_count() * per_item];
        self.exchange_into(values, per_item, &mut received);
        received
    }

    /// [exchange](Self::exchange) into `received`, which must hold
    /// [total_receive_count](Self::total_receive_count) times `per_item` values.
    ///
    /// # Collective operation
    ///
    /// As [exchange](Self::exchange).
    ///
    /// # Panics
    ///
    /// As [exchange](Self::exchange), and after taking part in the exchange on a rank
    /// whose `received` has the wrong length; its slice is then left unchanged.
    pub fn exchange_into<T: Equivalence + Copy + Default>(
        &self,
        values: &[T],
        per_item: usize,
        received: &mut [T],
    ) {
        self.route_or_panic(Direction::Forward, values, per_item, received);
    }

    /// Send `per_item` values of every received item back to the rank it came from, and
    /// return the values that come back, grouped by destination rank in ascending order.
    ///
    /// `values` holds [total_receive_count](Self::total_receive_count) times `per_item`
    /// values, laid out as the result of [exchange](Self::exchange). The result holds
    /// [total_send_count](Self::total_send_count) times `per_item` values, laid out as
    /// the values passed to [exchange](Self::exchange).
    ///
    /// # Collective operation
    ///
    /// As [exchange](Self::exchange).
    ///
    /// # Panics
    ///
    /// As [exchange](Self::exchange).
    pub fn exchange_back<T: Equivalence + Copy + Default>(
        &self,
        values: &[T],
        per_item: usize,
    ) -> Vec<T> {
        self.check_per_item(per_item);
        let mut returned = vec![T::default(); self.total_send_count() * per_item];
        self.exchange_back_into(values, per_item, &mut returned);
        returned
    }

    /// [exchange_back](Self::exchange_back) into `returned`, which must hold
    /// [total_send_count](Self::total_send_count) times `per_item` values.
    ///
    /// # Collective operation
    ///
    /// As [exchange](Self::exchange).
    ///
    /// # Panics
    ///
    /// As [exchange_into](Self::exchange_into).
    pub fn exchange_back_into<T: Equivalence + Copy + Default>(
        &self,
        values: &[T],
        per_item: usize,
        returned: &mut [T],
    ) {
        self.route_or_panic(Direction::Backward, values, per_item, returned);
    }

    /// Panic, alike on every rank, if `per_item` exceeds the agreed maximum.
    fn check_per_item(&self, per_item: usize) {
        assert!(
            per_item <= self.max_per_item,
            "{per_item} values per item exceed the {} that fit MPI's i32 counts",
            self.max_per_item
        );
    }

    /// The outgoing and incoming routes of `direction`: (ranks, items per rank) each.
    fn routes(&self, direction: Direction) -> (Routes<'_>, Routes<'_>) {
        let forward = (&self.destinations[..], &self.send_counts[..]);
        let backward = (&self.sources[..], &self.receive_counts[..]);
        match direction {
            Direction::Forward => (forward, backward),
            Direction::Backward => (backward, forward),
        }
    }

    /// Exchange caller-provided slices, taking part in the exchange even if they are wrong.
    ///
    /// A slice with the wrong length is replaced by a scratch buffer of default values, so
    /// the other ranks complete the exchange; the rank then panics.
    fn route_or_panic<T: Equivalence + Copy + Default>(
        &self,
        direction: Direction,
        values: &[T],
        per_item: usize,
        received: &mut [T],
    ) {
        self.check_per_item(per_item);
        let ((_, out_counts), (_, in_counts)) = self.routes(direction);
        let out_len = out_counts.iter().sum::<usize>() * per_item;
        let in_len = in_counts.iter().sum::<usize>() * per_item;
        let (out_actual, in_actual) = (values.len(), received.len());
        let scratch_out;
        let values = if out_actual == out_len {
            values
        } else {
            scratch_out = vec![T::default(); out_len];
            &scratch_out
        };
        if in_actual == in_len {
            self.route(direction, values, per_item, received);
        } else {
            let mut scratch_in = vec![T::default(); in_len];
            self.route(direction, values, per_item, &mut scratch_in);
        }
        assert!(
            out_actual == out_len && in_actual == in_len,
            "sparse route buffer mismatch in {direction:?} exchange: outgoing buffer has length {out_actual}, expected {out_len}; incoming buffer has length {in_actual}, expected {in_len}"
        );
    }

    /// Exchange buffers whose lengths match `direction` and `per_item`.
    fn route<T: Equivalence + Copy>(
        &self,
        direction: Direction,
        values: &[T],
        per_item: usize,
        received: &mut [T],
    ) {
        let ((out_ranks, out_counts), (in_ranks, in_counts)) = self.routes(direction);
        let tag = match direction {
            Direction::Forward => FORWARD_TAG,
            Direction::Backward => BACKWARD_TAG,
        };
        mpi::request::scope(|scope| {
            let mut requests = Vec::with_capacity(out_ranks.len() + in_ranks.len());
            // The receives first, then the sends. Empty blocks (`per_item` zero) are never
            // passed to MPI, on either side alike.
            let mut own_receive = None;
            let mut rest = &mut received[..];
            for (&source, &count) in in_ranks.iter().zip(in_counts) {
                let (block, tail) = std::mem::take(&mut rest).split_at_mut(count * per_item);
                rest = tail;
                if source == self.rank {
                    own_receive = Some(block);
                } else if !block.is_empty() {
                    requests.push(
                        self.comm
                            .process_at_rank(source as i32)
                            .immediate_receive_into_with_tag(scope, block, tag),
                    );
                }
            }
            let mut own_send = None;
            let mut start = 0;
            for (&destination, &count) in out_ranks.iter().zip(out_counts) {
                let block = &values[start..start + count * per_item];
                start += count * per_item;
                if destination == self.rank {
                    own_send = Some(block);
                } else if !block.is_empty() {
                    requests.push(
                        self.comm
                            .process_at_rank(destination as i32)
                            .immediate_send_with_tag(scope, block, tag),
                    );
                }
            }
            if let (Some(own_receive), Some(own_send)) = (own_receive, own_send) {
                own_receive.copy_from_slice(own_send);
            }
            for request in requests {
                request.wait_without_status();
            }
        });
    }
}

/// Direction of an exchange along the routes.
#[derive(Clone, Copy, Debug)]
enum Direction {
    /// From the senders to the receivers.
    Forward,
    /// From the receivers back to the senders.
    Backward,
}
