//! Maps global data to local data required by processes.
//!
//! Consider ten global dofs with indices 0 to 9 and two processes. The first process
//! may need the dofs 0, 1, 2, 3, 4, 5, 6 and the second process the dofs 3, 4, 5, 6, 7, 8, 9.
//! Hence, some dofs are needed on both processes. The `Global2LocalDataMapper` establishes the corresponding
//! communication and maps distributed vectors of global dofs to the required dofs on each process.

use std::{collections::HashMap, rc::Rc};

use itertools::{Itertools, izip};
use mpi::traits::{CommunicatorCollectives, Equivalence};

use super::IndexLayout;

/// Maps global data to local data.
pub struct Global2LocalDataMapper<'a, C: CommunicatorCollectives> {
    index_layout: Rc<IndexLayout<'a, C>>,
    ghost_communicator: super::GhostCommunicator<usize>,
    ghost_to_position: HashMap<usize, usize>,
    required_dofs: Vec<usize>,
}

impl<'a, C: CommunicatorCollectives> Global2LocalDataMapper<'a, C> {
    /// Create a new data mapper.
    ///
    /// The `required_dofs` are the dofs that are required on the local process.
    pub fn new(index_layout: Rc<IndexLayout<'a, C>>, required_dofs: &[usize]) -> Self {
        let comm = index_layout.comm();
        let rank = comm.rank() as usize;

        // First we go through the required dofs and get the ghosts

        let valid = required_dofs
            .iter()
            .all(|&dof| index_layout.rank_from_index(dof).is_some());
        let mut all_valid = false;
        comm.all_reduce_into(
            &valid,
            &mut all_valid,
            mpi::collective::SystemOperation::logical_and(),
        );
        assert!(all_valid, "required dof is outside the global layout");
        let mut ghost_dofs = Vec::<usize>::new();

        for &dof in required_dofs {
            if index_layout.rank_from_index(dof).expect("validated dof") != rank {
                ghost_dofs.push(dof);
            }
        }

        // Now make sure that the ghosts are unique

        let ghost_dofs = ghost_dofs.iter().unique().copied().collect_vec();

        // Get the ranks of the ghost

        let ghost_owners = ghost_dofs
            .iter()
            .map(|&dof| index_layout.rank_from_index(dof).unwrap())
            .collect_vec();

        // We now setup the ghost communicator

        let ghost_communicator = super::GhostCommunicator::new(&ghost_dofs, &ghost_owners, comm);

        // We now create a dof to position array for the ghosts

        let ghost_to_position = HashMap::<usize, usize>::from_iter(
            ghost_communicator
                .receive_indices
                .iter()
                .enumerate()
                .map(|(i, &d)| (d, i)),
        );

        Self {
            index_layout,
            ghost_communicator,
            ghost_to_position,
            required_dofs: required_dofs.to_vec(),
        }
    }

    /// Map global data to the local required data
    ///
    /// The input data is a vector of global data. A chunk size can be given in case multiple elements
    /// are associated with each dof.
    pub fn map_data<T: Equivalence + Copy + Default + std::fmt::Debug>(
        &self,
        data: &[T],
        chunk_size: usize,
    ) -> Vec<T> {
        let expected_data_len = self.index_layout.number_of_local_indices() * chunk_size;
        let expected_send_len = self.ghost_communicator.total_send_count() * chunk_size;
        let expected_receive_len = self.ghost_communicator.total_receive_count() * chunk_size;
        let output_len = self.required_dofs.len() * chunk_size;
        let mut chunks = vec![0_usize; self.index_layout.comm().size() as usize];
        self.index_layout
            .comm()
            .all_gather_into(&chunk_size, &mut chunks);
        let valid = chunk_size > 0
            && chunks.iter().all(|&candidate| candidate == chunk_size)
            && expected_data_len == data.len();
        let mut all_valid = false;
        self.index_layout.comm().all_reduce_into(
            &valid,
            &mut all_valid,
            mpi::collective::SystemOperation::logical_and(),
        );
        assert!(
            all_valid,
            "invalid or inconsistent mapper exchange metadata"
        );
        let rank = self.index_layout.comm().rank() as usize;
        let mut send_data = vec![T::default(); expected_send_len];
        for (global_send_index, send_buffer_chunk) in izip!(
            self.ghost_communicator.send_indices().iter(),
            send_data.chunks_mut(chunk_size)
        ) {
            let local_start_index = self
                .index_layout
                .global2local(rank, *global_send_index)
                .expect("ghost sender must own its requested index")
                * chunk_size;
            send_buffer_chunk
                .copy_from_slice(&data[local_start_index..local_start_index + chunk_size]);
        }

        let mut receive_data = vec![T::default(); expected_receive_len];
        self.ghost_communicator.forward_send_values_by_chunks(
            &send_data,
            &mut receive_data,
            chunk_size,
        );

        let mut output_data = vec![T::default(); output_len];
        for (&dof, output_chunk) in izip!(
            self.required_dofs.iter(),
            output_data.chunks_mut(chunk_size)
        ) {
            if self
                .index_layout
                .rank_from_index(dof)
                .expect("required dof is out of range")
                == rank
            {
                let local_dof = self.index_layout.global2local(rank, dof).unwrap();
                output_chunk
                    .copy_from_slice(&data[local_dof * chunk_size..(local_dof + 1) * chunk_size]);
            } else {
                let receive_start = self.ghost_to_position[&dof] * chunk_size;
                output_chunk
                    .copy_from_slice(&receive_data[receive_start..receive_start + chunk_size]);
            }
        }
        output_data
    }

    /// Return the index layout
    pub fn index_layout(&self) -> Rc<IndexLayout<'a, C>> {
        self.index_layout.clone()
    }

    /// Return the ghost communicator
    pub fn ghost_communicator(&self) -> &super::GhostCommunicator<usize> {
        &self.ghost_communicator
    }
}
