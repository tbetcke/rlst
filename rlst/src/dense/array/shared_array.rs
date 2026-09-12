//! Runtime controlled mutable access for arrays.

use std::{
    cell::{Ref, RefCell, RefMut},
    ops::Deref,
    rc::Rc,
    sync::{Arc, LockResult, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use crate::{Array, BorrowAtRuntime, BorrowAtRuntimeMut};

/// Cloneable, single-threaded owner with runtime-checked borrowing.
///
/// Clones refer to the same array through `Rc<RefCell<_>>`. [`BorrowAtRuntime::borrow`] and
/// [`BorrowAtRuntimeMut::borrow_mut`] panic if the `RefCell` borrow rules are violated.
pub struct ArraySharedRef<ArrayImpl, const NDIM: usize>(Rc<RefCell<Array<ArrayImpl, NDIM>>>);

impl<ArrayImpl, const NDIM: usize> Clone for ArraySharedRef<ArrayImpl, NDIM> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<ArrayImpl, const NDIM: usize> ArraySharedRef<ArrayImpl, NDIM> {
    /// Move an array into a new single-threaded shared owner.
    pub fn new(arr: Array<ArrayImpl, NDIM>) -> Self {
        ArraySharedRef(Rc::new(RefCell::new(arr)))
    }
}

/// Cloneable, thread-safe owner protected by a reader-writer lock.
///
/// Clones refer to the same array through `Arc<RwLock<_>>`. Borrow methods return a `LockResult`,
/// which reports poisoning if a writer panicked while holding the lock.
pub struct ArrayAtomicSharedRef<ArrayImpl, const NDIM: usize>(Arc<RwLock<Array<ArrayImpl, NDIM>>>);

impl<ArrayImpl, const NDIM: usize> Clone for ArrayAtomicSharedRef<ArrayImpl, NDIM> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<ArrayImpl, const NDIM: usize> ArrayAtomicSharedRef<ArrayImpl, NDIM> {
    /// Move an array into a new thread-safe shared owner.
    pub fn new(arr: Array<ArrayImpl, NDIM>) -> Self {
        ArrayAtomicSharedRef(Arc::new(RwLock::new(arr)))
    }
}

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
    /// Move the array into a cloneable, single-threaded shared owner.
    ///
    /// Use [borrow](BorrowAtRuntime::borrow) and [borrow_mut](BorrowAtRuntimeMut::borrow_mut)
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

    /// Move the array into a cloneable, thread-safe shared owner.
    /// Use [borrow](BorrowAtRuntime::borrow) and [borrow_mut](BorrowAtRuntimeMut::borrow_mut)
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
