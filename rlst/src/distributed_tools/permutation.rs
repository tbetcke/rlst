//! Data Permutations
//!
//! This module provides a `Permutation` struct that can be used to permute data
//! from the layout given by an `index_set` to a custom index layout.

use std::rc::Rc;

use itertools::izip;
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence};

use super::{array_tools::all_to_allv, index_layout::IndexLayout};

/// Permuation of data.
pub struct DataPermutation<'a, C: Communicator> {
    index_layout: Rc<IndexLayout<'a, C>>,
    nindices: usize,
    my_rank: usize,
    custom_local_indices: Vec<usize>,
    local_to_custom_map: Vec<usize>,
    receive_to_custom_map: Vec<usize>,
    ghost_communicator: super::GhostCommunicator<usize>,
}

impl<'a, C: Communicator> DataPermutation<'a, C> {
    /// Create a new permutation object.
    pub fn new(index_layout: Rc<IndexLayout<'a, C>>, custom_indices: &[usize]) -> Self
    where
        C: CommunicatorCollectives,
    {
        // We first need to identify which custom indices are local and which are global.

        let comm = index_layout.comm();
        let local_valid = custom_indices
            .iter()
            .all(|&index| index_layout.rank_from_index(index).is_some());
        let mut valid = false;
        comm.all_reduce_into(
            &local_valid,
            &mut valid,
            mpi::collective::SystemOperation::logical_and(),
        );
        assert!(valid, "permutation index is outside the global layout");

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
        let mut ghost_ranks = Vec::new();

        let mut local_to_custom_map = Vec::<usize>::new();
        let mut ghost_to_custom_map = Vec::<usize>::new();

        for (pos, &index) in custom_indices.iter().enumerate() {
            let index_rank = index_layout.rank_from_index(index).unwrap();
            if index_rank == my_rank {
                custom_local_indices.push(index_layout.global2local(my_rank, index).unwrap());
                local_to_custom_map.push(pos);
            } else {
                custom_ghost_indices.push(index);
                ghost_ranks.push(index_rank);
                ghost_to_custom_map.push(pos);
            }
        }

        // We can now send up the ghost communicator for all indices that are not local.

        let ghost_communicator =
            super::GhostCommunicator::new(&custom_ghost_indices, &ghost_ranks, comm);

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
        }
    }

    fn validate_buffers(
        &self,
        actual_lengths: [usize; 2],
        chunk_size: usize,
        expected_lengths: [usize; 4],
    ) {
        let mut chunks = vec![0_usize; self.index_layout.comm().size() as usize];
        self.index_layout
            .comm()
            .all_gather_into(&chunk_size, &mut chunks);
        let valid = chunk_size > 0
            && chunks.iter().all(|&candidate| candidate == chunk_size)
            && expected_lengths[0] == actual_lengths[0]
            && expected_lengths[1] == actual_lengths[1];
        let mut all_valid = false;
        self.index_layout.comm().all_reduce_into(
            &valid,
            &mut all_valid,
            mpi::collective::SystemOperation::logical_and(),
        );
        assert!(
            all_valid,
            "invalid or inconsistent permutation exchange metadata"
        );
    }

    /// Permute data from the layout given by the `index_set` to the custom index layout.
    pub fn forward_permute<T: Equivalence + Copy + Default>(
        &self,
        data: &[T],
        permuted_data: &mut [T],
        chunk_size: usize,
    ) {
        let expected_data_len = self.index_layout.number_of_local_indices() * chunk_size;
        let expected_output_len = self.nindices * chunk_size;
        let expected_send_len = self.ghost_communicator.total_send_count() * chunk_size;
        let expected_receive_len = self.ghost_communicator.total_receive_count() * chunk_size;
        self.validate_buffers(
            [data.len(), permuted_data.len()],
            chunk_size,
            [
                expected_data_len,
                expected_output_len,
                expected_send_len,
                expected_receive_len,
            ],
        );

        // We first need to get the send data. This is quite easy. We can just
        // use the global2local method from the index layout.

        let mut send_data = Vec::<T>::with_capacity(expected_send_len);

        for &index in self.ghost_communicator.send_indices() {
            let local_start_index =
                chunk_size * self.index_layout.global2local(self.my_rank, index).unwrap();
            let local_end_index = local_start_index + chunk_size;
            send_data.extend_from_slice(&data[local_start_index..local_end_index]);
        }

        // Now we do the data exchange across ranks.

        let mut received_data = vec![T::default(); expected_receive_len];
        self.ghost_communicator.forward_send_values_by_chunks(
            &send_data,
            &mut received_data,
            chunk_size,
        );

        // The data exchange is done. Now we have to fit everything back together to get to our custom data layout.

        // First we iterate through the local data.

        for (&pos, &local_index) in izip!(&self.local_to_custom_map, &self.custom_local_indices) {
            permuted_data[chunk_size * pos..chunk_size * (1 + pos)]
                .copy_from_slice(&data[chunk_size * local_index..chunk_size * (1 + local_index)]);
        }

        // Now we iterate through the ghost data and assign it to the right position in the permuted data.

        for (&permuted_index, chunk) in izip!(
            &self.receive_to_custom_map,
            received_data.chunks(chunk_size)
        ) {
            permuted_data[chunk_size * permuted_index..chunk_size * (1 + permuted_index)]
                .copy_from_slice(chunk);
        }
    }

    /// Permute data from the custom index layout to the layout given by the `index_set`.
    pub fn backward_permute<T: Equivalence + Copy + Default>(
        &self,
        data: &[T],
        permuted_data: &mut [T],
        chunk_size: usize,
    ) {
        let expected_data_len = self.nindices * chunk_size;
        let expected_output_len = self.index_layout.number_of_local_indices() * chunk_size;
        let expected_send_len = self.ghost_communicator.total_send_count() * chunk_size;
        let expected_receive_len = self.ghost_communicator.total_receive_count() * chunk_size;
        self.validate_buffers(
            [data.len(), permuted_data.len()],
            chunk_size,
            [
                expected_data_len,
                expected_output_len,
                expected_send_len,
                expected_receive_len,
            ],
        );

        // We need to fill up the receive indices as this is the data that is sent around.
        let mut receive_data = Vec::<T>::with_capacity(expected_receive_len);
        for &custom_index in self.receive_to_custom_map.iter() {
            receive_data.extend_from_slice(
                &data[custom_index * chunk_size..(1 + custom_index) * chunk_size],
            )
        }

        // We can now send back the receive indices.

        let mut send_data = vec![T::default(); expected_send_len];

        // We now send data backwards from receiver to sender.
        self.ghost_communicator.backward_send_values_by_chunks(
            &receive_data,
            &mut send_data,
            chunk_size,
        );

        // We now go through the send indices and fill the output data with the corresponding values.

        for (&index, chunk) in izip!(
            self.ghost_communicator.send_indices(),
            send_data.chunks(chunk_size)
        ) {
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
