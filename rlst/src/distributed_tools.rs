//! Useful tools for arrays on MPI systems.

pub mod array_tools;
pub mod data_mapper;
pub mod ghost_communicator;
pub mod index_embedding;
pub mod index_layout;
pub mod parallel_sort;
pub mod permutation;
pub mod sparse_routes;

pub use array_tools::{
    SPARSE_EXCHANGE_TAG, all_to_allv, displacements, scatterv, scatterv_root, sort_to_bins,
    sparse_all_to_allv,
};
pub use data_mapper::Global2LocalDataMapper;
pub use ghost_communicator::{
    ChunkSizes, GhostCommunicator, GhostCommunicatorBuilder, GhostReceiveBuffer, GhostSendBuffer,
};
pub use index_layout::IndexLayout;
pub use permutation::DataPermutation;
pub use sparse_routes::SparseRoutes;
