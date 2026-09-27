//! Data Permutations
//!
//! This module provides a `Permutation` struct that can be used to permute data
//! from the layout given by an `index_set` to a custom index layout.

use std::rc::Rc;

use itertools::izip;
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence};

use super::{
    ChunkSizes, GhostCommunicatorBuilder, array_tools::all_to_allv, index_layout::IndexLayout,
};

/// Permuation of data.
pub struct DataPermutation<'a, C: Communicator> {
    index_layout: Rc<IndexLayout<'a, C>>,
    nindices: usize,
    my_rank: usize,
    custom_local_indices: Vec<usize>,
    local_to_custom_map: Vec<usize>,
    receive_to_custom_map: Vec<usize>,
    ghost_communicator: super::GhostCommunicator<usize>,
    chunk_size: usize,
}

impl<'a, C: Communicator> DataPermutation<'a, C> {
    /// Create a new permutation object.
    ///
    /// The `custom_indices` are the global indices of the custom layout on this rank. Each
    /// index is associated with `chunk_size` consecutive values in the data vectors. The
    /// chunk size must be positive and the same on every rank.
    ///
    /// This is a collective operation.
    pub fn new(
        index_layout: Rc<IndexLayout<'a, C>>,
        custom_indices: &[usize],
        chunk_size: usize,
    ) -> Self
    where
        C: CommunicatorCollectives,
    {
        // We first need to identify which custom indices are local and which are global.

        let comm = index_layout.comm();
        let mut max_chunk_size = 0_usize;
        comm.all_reduce_into(
            &chunk_size,
            &mut max_chunk_size,
            mpi::collective::SystemOperation::max(),
        );
        let local_valid = chunk_size > 0
            && chunk_size == max_chunk_size
            && custom_indices
                .iter()
                .all(|&index| index_layout.rank_from_index(index).is_some());
        let mut valid = false;
        comm.all_reduce_into(
            &local_valid,
            &mut valid,
            mpi::collective::SystemOperation::logical_and(),
        );
        assert!(
            valid,
            "permutation index is outside the global layout or chunk sizes are invalid"
        );

        // Route every requested index to its owner.  This includes requests made
        // by the owner itself, so each owner can verify exact global multiplicity.
        let mut owner_counts = vec![0_usize; comm.size() as usize];
        let mut owner_grouped_indices = Vec::with_capacity(custom_indices.len());
        for (owner, owner_count) in owner_counts.iter_mut().enumerate() {
            for &index in custom_indices {
                if index_layout.rank_from_index(index) == Some(owner) {
                    *owner_count += 1;
                    owner_grouped_indices.push(index);
                }
            }
        }
        // `all_to_allv` partitions its input contiguously by destination rank.
        // Keep this validation-only copy owner grouped; the original order below
        // remains the caller's permutation order.
        let (_, owned_requests) = all_to_allv(comm, &owner_counts, &owner_grouped_indices);
        let mut multiplicities = vec![0_usize; index_layout.number_of_local_indices()];
        let my_rank = comm.rank() as usize;
        let mut multiplicity_valid = true;
        for index in owned_requests {
            if let Some(local_index) = index_layout.global2local(my_rank, index) {
                multiplicities[local_index] += 1;
            } else {
                multiplicity_valid = false;
            }
        }
        multiplicity_valid &= multiplicities.iter().all(|&count| count == 1);
        let mut exact_permutation = false;
        comm.all_reduce_into(
            &multiplicity_valid,
            &mut exact_permutation,
            mpi::collective::SystemOperation::logical_and(),
        );
        assert!(
            exact_permutation,
            "custom indices must contain every global index exactly once"
        );

        let mut custom_local_indices = Vec::new();
        let mut custom_ghost_indices = Vec::new();
        let mut ghost_builder = GhostCommunicatorBuilder::new();

        let mut local_to_custom_map = Vec::<usize>::new();
        let mut ghost_to_custom_map = Vec::<usize>::new();

        for (pos, &index) in custom_indices.iter().enumerate() {
            let index_rank = index_layout.rank_from_index(index).unwrap();
            if index_rank == my_rank {
                custom_local_indices.push(index_layout.global2local(my_rank, index).unwrap());
                local_to_custom_map.push(pos);
            } else {
                custom_ghost_indices.push(index);
                ghost_builder.add_ghost(index, index_rank);
                ghost_to_custom_map.push(pos);
            }
        }

        // We can now send up the ghost communicator for all indices that are not local.

        let ghost_communicator = ghost_builder
            .chunk_sizes(ChunkSizes::Uniform(chunk_size))
            .build(comm);

        // We now need the map from the receive indices to our custom indices. For
        // this we first compute the map from the receive indices to custom ghost indices
        // and then map on to actual positions in the custom indices.

        let ghost_permutation =
            permutation_map(ghost_communicator.receive_indices(), &custom_ghost_indices);

        // We now use this to map from receive indices to the corresponding positions in the custom Arc::new(indices)

        let mut receive_to_custom_map =
            Vec::<usize>::with_capacity(ghost_communicator.total_receive_count());

        for index in 0..ghost_to_custom_map.len() {
            receive_to_custom_map.push(ghost_to_custom_map[ghost_permutation[index]]);
        }

        Self {
            index_layout,
            nindices: custom_indices.len(),
            my_rank,
            custom_local_indices,
            local_to_custom_map,
            receive_to_custom_map,
            ghost_communicator,
            chunk_size,
        }
    }

    /// Panic unless the input and output buffers of a permutation have their expected lengths.
    ///
    /// Called after the ghost exchange, so that a rank with wrong buffers still takes part.
    fn check_buffers(data_len: [usize; 2], expected_len: [usize; 2]) {
        assert!(
            data_len == expected_len,
            "permutation input and output data have lengths {data_len:?}, expected {expected_len:?}"
        );
    }

    /// Return the number of values associated with each index.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Permute data from the layout given by the `index_set` to the custom index layout.
    ///
    /// Both buffers hold `chunk_size` values per index. Their lengths are checked locally,
    /// without communication. If they are wrong, this rank still takes part in the exchange
    /// with default values, so the other ranks complete normally but receive default values
    /// from it, and then panics.
    pub fn forward_permute<T: Equivalence + Copy + Default>(
        &self,
        data: &[T],
        permuted_data: &mut [T],
    ) {
        let chunk_size = self.chunk_size;
        let lengths = [data.len(), permuted_data.len()];
        let expected_lengths = [
            self.index_layout.number_of_local_indices() * chunk_size,
            self.nindices * chunk_size,
        ];
        let valid = lengths == expected_lengths;

        // We first need to get the send data. This is quite easy. We can just
        // use the global2local method from the index layout.

        let send_buffer = self.ghost_communicator.send_buffer_from_fn(|index, chunk| {
            if valid {
                let local_start_index =
                    chunk_size * self.index_layout.global2local(self.my_rank, index).unwrap();
                chunk.copy_from_slice(&data[local_start_index..local_start_index + chunk_size]);
            }
        });

        // Now we do the data exchange across ranks.

        let receive_buffer = send_buffer.forward();
        Self::check_buffers(lengths, expected_lengths);

        // The data exchange is done. Now we have to fit everything back together to get to our custom data layout.

        // First we iterate through the local data.

        for (&pos, &local_index) in izip!(&self.local_to_custom_map, &self.custom_local_indices) {
            permuted_data[chunk_size * pos..chunk_size * (1 + pos)]
                .copy_from_slice(&data[chunk_size * local_index..chunk_size * (1 + local_index)]);
        }

        // Now we iterate through the ghost data and assign it to the right position in the permuted data.

        for (&permuted_index, (_, chunk)) in
            izip!(&self.receive_to_custom_map, receive_buffer.chunks())
        {
            permuted_data[chunk_size * permuted_index..chunk_size * (1 + permuted_index)]
                .copy_from_slice(chunk);
        }
    }

    /// Permute data from the custom index layout to the layout given by the `index_set`.
    ///
    /// Both buffers hold `chunk_size` values per index. Their lengths are checked locally,
    /// without communication. If they are wrong, this rank still takes part in the exchange
    /// with default values, so the other ranks complete normally but receive default values
    /// from it, and then panics.
    pub fn backward_permute<T: Equivalence + Copy + Default>(
        &self,
        data: &[T],
        permuted_data: &mut [T],
    ) {
        let chunk_size = self.chunk_size;
        let lengths = [data.len(), permuted_data.len()];
        let expected_lengths = [
            self.nindices * chunk_size,
            self.index_layout.number_of_local_indices() * chunk_size,
        ];
        let valid = lengths == expected_lengths;

        // We need to fill up the receive indices as this is the data that is sent around.
        let mut receive_buffer = self.ghost_communicator.receive_buffer();
        if valid {
            for (&custom_index, (_, chunk)) in
                izip!(&self.receive_to_custom_map, receive_buffer.chunks_mut())
            {
                chunk.copy_from_slice(
                    &data[custom_index * chunk_size..(1 + custom_index) * chunk_size],
                );
            }
        }

        // We now send data backwards from receiver to sender.
        let send_buffer = receive_buffer.backward();
        Self::check_buffers(lengths, expected_lengths);

        // We now go through the send indices and fill the output data with the corresponding values.

        for (index, chunk) in send_buffer.chunks() {
            let local_start_index =
                chunk_size * self.index_layout.global2local(self.my_rank, index).unwrap();
            let local_end_index = local_start_index + chunk_size;
            permuted_data[local_start_index..local_end_index].copy_from_slice(chunk);
        }

        // We still have to handle the indices that lived only locally.
        for (&pos, &local_index) in izip!(&self.local_to_custom_map, &self.custom_local_indices) {
            permuted_data[local_index * chunk_size..(1 + local_index) * chunk_size]
                .copy_from_slice(&data[pos * chunk_size..(1 + pos) * chunk_size]);
        }
    }
}

/// Create a permutation map.
///
/// Returns a map m such that
/// `permuted_indices[m[i]] = original_indices[i]`
pub fn permutation_map(original_indices: &[usize], permuted_indices: &[usize]) -> Vec<usize> {
    concatenate_permutations(
        &invert_permutation(&argsort(original_indices)),
        &argsort(permuted_indices),
    )
}

/// Return the permutation that sorts a sequence.
pub fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..data.len()).collect();
    indices.sort_by(|&a, &b| data[a].cmp(&data[b]));
    indices
}

/// Invert a permutation.
pub fn invert_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inverse[p] = i;
    }
    inverse
}

/// Concatenate permutations.
pub fn concatenate_permutations(perm1: &[usize], perm2: &[usize]) -> Vec<usize> {
    assert_eq!(perm1.len(), perm2.len());
    let mut result = vec![0; perm1.len()];
    for (i, &p) in perm1.iter().enumerate() {
        result[i] = perm2[p];
    }
    result
}

#[cfg(test)]
mod test {

    #[test]
    fn test_permutation() {
        let original_indices = vec![6, 3, 4, 4, 2];
        let permuted_indices = vec![6, 4, 3, 2, 4];

        let permutation = super::permutation_map(&original_indices, &permuted_indices);

        for index in 0..original_indices.len() {
            assert_eq!(
                permuted_indices[permutation[index]],
                original_indices[index]
            );
        }
    }
}
