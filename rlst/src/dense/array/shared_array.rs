//! Runtime controlled mutable access for arrays.

use std::{
    cell::{Ref, RefCell, RefMut},
    ops::Deref,
    rc::Rc,
    sync::{Arc, LockResult, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use crate::{Array, BorrowAtRuntime, BorrowAtRuntimeMut};

/// A shared runtime non-thread safe reference to an array.
#[derive(Clone)]
pub struct ArraySharedRef<ArrayImpl, const NDIM: usize>(Rc<RefCell<Array<ArrayImpl, NDIM>>>);

/// A shared runtime thread-safe reference to an array.
#[derive(Clone)]
pub struct ArrayAtomicSharedRef<ArrayImpl, const NDIM: usize>(Arc<RwLock<Array<ArrayImpl, NDIM>>>);

impl<ArrayImpl, const NDIM: usize> BorrowAtRuntime for ArraySharedRef<ArrayImpl, NDIM> {
    type Target<'a>
        = Ref<'a, Array<ArrayImpl, NDIM>>
    where
        Self: 'a;

    fn borrow(&self) -> Self::Target<'_> {
        RefCell::borrow(self.0.deref())
    }
}

impl<ArrayImpl, const NDIM: usize> BorrowAtRuntimeMut for ArraySharedRef<ArrayImpl, NDIM> {
    type Target<'a>
        = RefMut<'a, Array<ArrayImpl, NDIM>>
    where
        Self: 'a;

    fn borrow_mut(&self) -> Self::Target<'_> {
        RefCell::borrow_mut(self.0.deref())
    }
}

impl<ArrayImpl, const NDIM: usize> BorrowAtRuntime for ArrayAtomicSharedRef<ArrayImpl, NDIM> {
    type Target<'a>
        = LockResult<RwLockReadGuard<'a, Array<ArrayImpl, NDIM>>>
    where
        Self: 'a;

    fn borrow(&self) -> Self::Target<'_> {
        self.0.read()
    }
}

impl<ArrayImpl, const NDIM: usize> BorrowAtRuntimeMut for ArrayAtomicSharedRef<ArrayImpl, NDIM> {
    type Target<'a>
        = LockResult<RwLockWriteGuard<'a, Array<ArrayImpl, NDIM>>>
    where
        Self: 'a;

    fn borrow_mut(&self) -> Self::Target<'_> {
        self.0.write()
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM> {
    /// Create a non thread-safe runtime shared reference from the array
    ///
    /// Use [borrow](BorrowAtRuntime::borrow) and [borrow_mut](BorrowAtRuntime::borrow_mut)
    /// to access the underlying array.
    ///
    /// The rational of this function is to obtain a cloneable shared reference structure
    /// to the original array with runtime borrow checking when compile-time checking is
    /// not sufficient.
    ///
    /// This function is not thread-safe. For a thread-safe variant use [Array::into_shared_atomic].
    pub fn into_shared(self) -> ArraySharedRef<ArrayImpl, NDIM> {
        ArraySharedRef(Rc::new(RefCell::new(self)))
    }

    /// Create an atomic thread-safe runtime shared reference from the array
    /// Use [borrow](BorrowAtRuntime::borrow) and [borrow_mut](BorrowAtRuntime::borrow_mut)
    /// to access the underlying array.
    ///
    /// The rational of this function is to obtain a cloneable shared reference structure
    /// to the original array with runtime borrow checking when compile-time checking is
    /// not sufficient.
    ///
    /// This function is thread-safe. For a slightly faster non thread-safe variant use [Array::into_shared].
    pub fn into_shared_atomic(self) -> ArrayAtomicSharedRef<ArrayImpl, NDIM> {
        ArrayAtomicSharedRef(Arc::new(RwLock::new(self)))
    }
}
