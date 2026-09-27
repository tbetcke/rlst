//! Maps global data to local data required by processes.
//!
//! Consider ten global dofs with indices 0 to 9 and two processes. The first process
//! may need the dofs 0, 1, 2, 3, 4, 5, 6 and the second process the dofs 3, 4, 5, 6, 7, 8, 9.
//! Hence, some dofs are needed on both processes. The `Global2LocalDataMapper` establishes the corresponding
//! communication and maps distributed vectors of global dofs to the required dofs on each process.

use std::{collections::HashMap, rc::Rc};

use itertools::{Itertools, izip};
use mpi::traits::{CommunicatorCollectives, Equivalence};

use super::{ChunkSizes, GhostCommunicatorBuilder, IndexLayout};

/// Maps global data to local data.
pub struct Global2LocalDataMapper<'a, C: CommunicatorCollectives> {
    index_layout: Rc<IndexLayout<'a, C>>,
    ghost_communicator: super::GhostCommunicator<usize>,
    ghost_to_position: HashMap<usize, usize>,
    required_dofs: Vec<usize>,
    chunk_size: usize,
}

impl<'a, C: CommunicatorCollectives> Global2LocalDataMapper<'a, C> {
    /// Create a new data mapper.
    ///
    /// The `required_dofs` are the dofs that are required on the local process. Each dof is
    /// associated with `chunk_size` consecutive values in the data vectors. The chunk size
    /// must be positive and the same on every rank.
    ///
    /// This is a collective operation.
    pub fn new(
        index_layout: Rc<IndexLayout<'a, C>>,
        required_dofs: &[usize],
        chunk_size: usize,
    ) -> Self {
        let comm = index_layout.comm();
        let rank = comm.rank() as usize;

        // First we go through the required dofs and get the ghosts

        let mut max_chunk_size = 0_usize;
        comm.all_reduce_into(
            &chunk_size,
            &mut max_chunk_size,
            mpi::collective::SystemOperation::max(),
        );
        let valid = chunk_size > 0
            && chunk_size == max_chunk_size
            && required_dofs
                .iter()
                .all(|&dof| index_layout.rank_from_index(dof).is_some());
        let mut all_valid = false;
        comm.all_reduce_into(
            &valid,
            &mut all_valid,
            mpi::collective::SystemOperation::logical_and(),
        );
        assert!(
            all_valid,
            "required dof is outside the global layout or chunk sizes are invalid"
        );
        let mut ghost_dofs = Vec::<usize>::new();

        for &dof in required_dofs {
            if index_layout.rank_from_index(dof).expect("validated dof") != rank {
                ghost_dofs.push(dof);
            }
        }

        // Now make sure that the ghosts are unique

        let ghost_dofs = ghost_dofs.iter().unique().copied().collect_vec();

        // We now setup the ghost communicator

        let ghost_communicator = GhostCommunicatorBuilder::new()
            .add_ghosts(
                ghost_dofs
                    .iter()
                    .map(|&dof| (dof, index_layout.rank_from_index(dof).unwrap())),
            )
            .chunk_sizes(ChunkSizes::Uniform(chunk_size))
            .build(comm);

        // We now create a dof to position array for the ghosts

        let ghost_to_position = HashMap::<usize, usize>::from_iter(
            ghost_communicator
                .receive_indices()
                .iter()
                .enumerate()
                .map(|(i, &d)| (d, i)),
        );

        Self {
            index_layout,
            ghost_communicator,
            ghost_to_position,
            required_dofs: required_dofs.to_vec(),
            chunk_size,
        }
    }

    /// Map global data to the local required data
    ///
    /// The input data is the vector of locally owned data, with `chunk_size` values for each
    /// dof. The output holds the chunks of the required dofs in their given order.
    ///
    /// The input length is checked locally, without communication. If it is wrong, this rank
    /// still takes part in the exchange with default values, so the other ranks complete
    /// normally but receive default values from it, and then panics.
    pub fn map_data<T: Equivalence + Copy + Default + std::fmt::Debug>(
        &self,
        data: &[T],
    ) -> Vec<T> {
        let chunk_size = self.chunk_size;
        let rank = self.index_layout.comm().rank() as usize;
        let data_valid = data.len() == self.index_layout.number_of_local_indices() * chunk_size;

        let send_buffer = self.ghost_communicator.send_buffer_from_fn(|index, chunk| {
            if data_valid {
                let local_start_index = self
                    .index_layout
                    .global2local(rank, index)
                    .expect("ghost sender must own its requested index")
                    * chunk_size;
                chunk.copy_from_slice(&data[local_start_index..local_start_index + chunk_size]);
            }
        });
        let receive_buffer = send_buffer.forward();
        assert!(
            data_valid,
            "mapper input data has length {}, expected {}",
            data.len(),
            self.index_layout.number_of_local_indices() * chunk_size
        );

        let mut output_data = vec![T::default(); self.required_dofs.len() * chunk_size];
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
                output_chunk.copy_from_slice(receive_buffer.chunk(self.ghost_to_position[&dof]));
            }
        }
        output_data
    }

    /// Return the index layout
    pub fn index_layout(&self) -> Rc<IndexLayout<'a, C>> {
        self.index_layout.clone()
    }

    /// Return the number of values associated with each dof.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Return the ghost communicator
    pub fn ghost_communicator(&self) -> &super::GhostCommunicator<usize> {
        &self.ghost_communicator
    }
}
