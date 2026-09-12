//! Traits for distributed patterns.
//!
//! The traits in this module provide typical patterns for working with distributed array data.

use std::rc::Rc;

use mpi::traits::Communicator;

use crate::distributed_tools::IndexLayout;

/// Collective gather to every process.
///
/// Every rank in the communicator must call the operation in the same order.
pub trait GatherToAll {
    /// The output type after gathering the distributed data.
    type Output;
    /// Gather the complete global data to every rank, preserving global index order.
    fn gather_to_all(&self) -> Self::Output;
}

/// Collective gather to one process using distinct root and non-root calls.
pub trait GatherToOne {
    /// The output type after gathering the distributed array to one process.
    type Output;
    /// Participate as a non-root rank in a gather to `root`.
    ///
    /// All ranks must participate, select the same root, and use compatible layouts.
    fn gather_to_one(&self, root: usize);
    /// Participate as the root rank and return the gathered global data.
    ///
    /// Every other rank must simultaneously call [`Self::gather_to_one`] with this rank.
    fn gather_to_one_root(&self) -> Self::Output;
}

/// Collective scatter from one process using distinct root and non-root calls.
///
/// Data is partitioned along its first dimension according to the index layout.
pub trait ScatterFromOne {
    /// The output type after scattering the data.
    type Output<'a, C>
    where
        C: 'a,
        C: Communicator;
    /// Participate as root, scattering this array according to `index_layout`.
    ///
    /// The source first dimension must equal the layout's global size. All other
    /// ranks must call [`Self::scatter_from_one`] with this root and a compatible layout.
    fn scatter_from_one_root<'a, C: Communicator>(
        &self,
        index_layout: Rc<IndexLayout<'a, C>>,
    ) -> Self::Output<'a, C>;

    /// Participate as a non-root receiver in a scatter from `root`.
    ///
    /// All ranks must select the same root and use compatible layouts.
    fn scatter_from_one<'a, C: Communicator>(
        root: usize,
        index_layout: Rc<IndexLayout<'a, C>>,
    ) -> Self::Output<'a, C>;
}
