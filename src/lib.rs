//! # Assumptions
//!
//! CAS operation is wait free - it will complete in a finite number of steps
//! Failures are ok (as long as it means another thread is making progress)
//!
//! The HelpQueue is only wait free for insertion, peek and remove. Creating a handle is NOT wait
//! free, although that doesn't effect the wait freedom of most algorithms.
//!
//! # Structure
//!
//! The HelpQueue is a circular linked list, and each handle takes ownership of one Node in the
//! list. New nodes can be created as needed, and inserted without effecting the validity of the
//! list. Nodes are never removed, rather they are marked as unused, and can then be re-used by
//! future handles.
//!
//! Since each handle has ownership of one Node, insertion is guaranteed to succeed in a small
//! number of steps. There is a limitation that a given handle can only insert a single element
//! into the queue at a time, so the element the handle inserted must be removed before the handle
//! can insert another element.
//!
//! peek is also a constant time operation, since an atomic load never fails.
//!
//! try_remove_head is also a constant time operation. The only way for try_remove_head to fail is
//! if another thread has already removed it.
//!
//! Note that the queue makes no guarantees about whether readers see the same state. It is
//! guaranteed that the readers see a valid help queue, and can get elements. Hazard Pointers are
//! used to reclaim the memory of the elements of the queue

use std::{ptr::NonNull, sync::{Arc, atomic::{AtomicBool, AtomicPtr}}};
use std::sync::atomic::Ordering::*;

struct Node<T> {
    /// MUST be valid OR null
    element: AtomicPtr<T>,
    /// MUST be valid and NEVER null
    next: AtomicPtr<Node<T>>,
    /// Whether this Node is still valid. If set to false, next should be considered dangling
    valid: AtomicBool,
}

impl<T> std::fmt::Debug for Node<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node {{ element: {:?}, next: {:?}, valid: {:?} }}", self.element, self.next, self.valid)
    }
}

impl<T> Node<T> {
    /// Creates an empty node
    fn empty(ptr: *mut Self) -> Self {
        Self {
            element: AtomicPtr::new(std::ptr::null_mut()),
            next: AtomicPtr::new(ptr),
            valid: AtomicBool::new(true),
        }
    }
}

/// A Help Queue
pub struct HelpQueue<T> {
    /// MUST always be a valid pointer
    ///
    /// This is a pointer into a valid Circular Linked List
    current: AtomicPtr<Node<T>>,
    /// Hazard Pointer Domain
    domain: haz_ptr::HazPtrDomain,
}

impl<T> HelpQueue<T> {
    /// Constructs a new help queue
    pub fn new() -> Arc<Self> {
        // Sentinel node, which need to be placed on the stack and needs to point to itself
        // Technically, this violates the requirement that next never be null, but it's okay since
        // we set it right away
        let ptr = Box::into_raw(Box::new(Node::empty(std::ptr::null_mut())));
        unsafe { &* ptr }.next.store(ptr, Release);
        // Mark this node as unused, so it can be claimed
        unsafe { &* ptr }.valid.store(false, Release);

        let s = Self {
            current: AtomicPtr::new(ptr),
            domain: haz_ptr::HazPtrDomain::new(),
        };
        Arc::new(s)
    }

    /// Gets a handle to access the help queue
    ///
    /// Note that this operation isn't wait free, but it is lock free
    pub fn get_handle(self: &Arc<Self>) -> HelpQueueHandle<T> {
        HelpQueueHandle {
            inner: Arc::clone(self),
            current: self.get_node().unwrap_or_else(|| self.create_node()),
            lock: self.domain.get_lock(),
        }
    }

    fn get_node(&self) -> Option<&'static Node<T>> {
        let first = self.current.load(Acquire);
        let mut cur = unsafe { &* first }.next.load(Acquire);
        while cur != first {
            let node = unsafe { &* cur };
            if node.valid.load(Acquire) == false {
                if let Ok(_p) = node.valid.compare_exchange(false, true, AcqRel, Relaxed) {
                    return Some(unsafe { &* cur });
                }
            }
            cur = node.next.load(Acquire);
        }
        None
    }

    fn create_node(&self) -> &'static Node<T> {
        // Safety: Current MUST always be valid
        let mut cur = unsafe { &*self.current.load(Acquire) };
        loop {
            cur = unsafe { &*cur.next.load(Acquire) };

            let next = cur.next.load(Acquire);
            let new = Box::into_raw(Box::new(Node::empty(next)));
            match cur.next.compare_exchange_weak(next, new, AcqRel, Relaxed) {
                Ok(_p) => {
                    // If the previous node (the one that points at us) is still valid, we return.
                    // Otherwise we start over.
                    return unsafe { &* new };
                },
                Err(_) => {
                    // Safety: Free box iff it wasn't added
                    let _ = unsafe { Box::from_raw(new) };
                },
            }
        }
    }

    /// Advance the current pointer iff it's safe to do so.
    fn advance(&self) {
        let cur_ptr = self.current.load(Acquire);
        let cur = unsafe { &* cur_ptr };
        if !cur.valid.load(Acquire) || cur.element.load(Acquire).is_null() {
            let next = cur.next.load(Acquire);
            let _e = self.current.compare_exchange(cur_ptr, next, AcqRel, Relaxed);
        }
    }
}

impl<T> Drop for HelpQueue<T> {
    fn drop(&mut self) {
        let first = self.current.load(Acquire);
        let mut cur = first;
        while cur != first {
            let current = unsafe { Box::from_raw(cur) };
            cur = current.next.load(Acquire);
        }
        // The hazptr domain will not be dropped until AFTER this, so it can still be used to free
        // memory
    }
}

/// HelpQueue handle, this contains a reference to the HelpQueue, and the associated data to
/// ineract with it.
pub struct HelpQueueHandle<T: 'static> {
    inner: Arc<HelpQueue<T>>,
    /// Basically a Box<Node<T>>, but I'm responsible for freeing it
    current: &'static Node<T>,
    lock: haz_ptr::HazLock,
}

impl<T> HelpQueueHandle<T> {
    /// Enqueues the operation into the help queue. Note that each handle is only permited to
    /// insert a single operation into the help queue at a time.
    pub fn enqueue(&mut self, op: T) -> Result<(), ()> {
        self.release();
        let cur = self.current.element.load(Acquire);
        if cur.is_null() {
            match self.current.element.compare_exchange(cur, Box::into_raw(Box::new(op)), Release, Relaxed) {
                Ok(_p) => Ok(()),
                Err(_p) => Err(()),
            }
        } else {
            Err(())
        }
    }

    /// Gets the item this handle has added to the queue
    pub fn get_current<'a>(&'a mut self) -> Option<&'a T> {
        NonNull::new(self.lock.load_locked(&self.current.element))
            .map(|p| unsafe { &* p.as_ptr() })
    }

    /// Gets the first element from the queue
    ///
    /// `'a` represents the lifetime based on hazard pointers
    pub fn peek<'a>(&'a mut self) -> Option<&'a T> {
        self.inner.advance();
        self.release();
        // Safety:
        //
        // self.inner.first MUST always be a valid pointer, so load + deref is valid
        let first = unsafe { &* self.inner.current.load(Acquire) };

        // Safety:
        //
        // if element is Null, NonNull will clean it up
        if first.valid.load(Acquire) {
            NonNull::new(self.lock.load_locked(&first.element)).map(|p| {
                unsafe { &* p.as_ptr() }
            })
        } else {
            None
        }
    }

    /// Release the last borrow. This is called anywhere the borrow ends, although there is no
    /// guarantee made.
    pub fn release(&mut self) {
        self.lock.reset();
    }

    /// Removes the head item iff it is still last
    ///
    /// How does this work? I imagine the user must convert the reference to a pointer, which is
    /// always safe... Alternatively, we could just remove the last value locked... (i.e. the one
    /// locked)
    ///
    /// ```rust,no_run
    /// # let queue: HelpQueue<()> = HelpQueue::new();
    /// # let mut handle = queue.get_handle();
    /// # handle.enqueue(()).expect("First never fails");
    /// let front = handle.peek().unwrap();
    ///
    /// // let front = front as *const _;
    /// handle.try_remove_head();
    /// ```
    ///
    /// If the current is null (i.e. this handle has dropped the reference to the last element
    /// peeked), Err() is returned
    pub fn try_remove_last_peeked(&mut self) -> Result<(), ()> {
        let ret = if !self.lock.current().is_null() {
            let tmp = unsafe { &* self.inner.current.load(Acquire) };
            match tmp.element.compare_exchange(self.lock.current().cast(), std::ptr::null_mut(), Release, Relaxed) {
                Ok(p) => {
                    self.inner.advance();
                    // Safety: p is no longer accessible, and therefore will not be dropped again
                    unsafe { self.inner.domain.retire(p) };
                    Ok(())
                },
                Err(_p) => Err(()),
            }
        } else {
            Err(())
        };
        // Release AFTER the element has been removed from the list, and can now be retired
        self.release();
        ret
    }
}

impl<T> Drop for HelpQueueHandle<T> {
    fn drop(&mut self) {
        self.release();
        let old = self.current.element.swap(std::ptr::null_mut(), AcqRel);
        // Safety: old is no longer accessible, and therefore will not be dropped again
        // If old is null, retire does nothing
        unsafe { self.inner.domain.retire(old) };
        self.current.valid.store(false, Release);
    }
}

#[test]
fn feels_good() {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Operation { Add, Remove }

    let queue: Arc<HelpQueue<Operation>> = HelpQueue::new();
    let mut handle = queue.get_handle();
    let mut handle2 = queue.get_handle();
    handle.enqueue(Operation::Add).expect("First never fails");
    handle2.enqueue(Operation::Remove).expect("First never fails");
    //queue.get_handle().enqueue(Operation::Remove).expect("First never fails");

    //queue.get_handle().enqueue(Operation::Remove).expect("First never fails");

    let front = handle.peek();
    assert_eq!(front, Some(&Operation::Add));
    assert_eq!(handle2.peek(), Some(&Operation::Add));
    let _ = handle.try_remove_last_peeked();
    // Not allowed because _front is no longer valid
    //println!("{:?}", _front);
    let front = handle.peek();
    assert_eq!(front, Some(&Operation::Remove));
    assert_eq!(handle2.peek(), Some(&Operation::Remove));
    let _ = handle.try_remove_last_peeked();

    assert_eq!(handle.peek(), None);
    assert_eq!(handle2.peek(), None);
}

// TODO: remove and just use an existing implementation
#[allow(unused)]
mod haz_ptr {
    use std::sync::atomic::{AtomicPtr, AtomicBool, Ordering::*};

    struct RetiredPointer {
        retire: Box<dyn FnOnce()>,
        ptr: AtomicPtr<()>,
        next: AtomicPtr<RetiredPointer>
    }

    impl RetiredPointer {
        pub fn new(ptr: *mut (), retire: Box<dyn FnOnce()>, next: *mut Self) -> Self {
            Self {
                retire, ptr: AtomicPtr::new(ptr), next: AtomicPtr::new(next),
            }
        }
    }

    pub struct HazPtrDomain {
        first: AtomicPtr<HazPtrLock>,
        retired: AtomicPtr<RetiredPointer>,
    }

    impl HazPtrDomain {
        pub fn new() -> Self {
            Self {
                first: AtomicPtr::new(std::ptr::null_mut()),
                retired: AtomicPtr::new(std::ptr::null_mut()),
            }
        }

        fn get_inner_lock(&self) -> &'static HazPtrLock {
            let mut cur = self.first.load(Acquire);
            if cur.is_null() {
                let new = HazPtrLock::new();
                self.first.store(new, Release);
                return unsafe { &* new };
            }
            loop {
                if let Ok(_) = unsafe { &* cur }
                    .in_use
                    .compare_exchange(false, true, AcqRel, Relaxed)
                {
                    return unsafe { &* cur };
                }
                let next = unsafe { &* cur }.next.load(Acquire);
                if next.is_null() {
                    let new = HazPtrLock::new();
                    if let Ok(_) = unsafe { &* cur }.next.compare_exchange(next, new, AcqRel, Relaxed) {
                        return unsafe { &* new };
                    } else {
                        // Free allocation if it wasn't added
                        let _ = unsafe { Box::from_raw(new) };
                        continue;
                    }
                }
                cur = next;
            }
        }

        pub fn get_lock(&self) -> HazLock {
            HazLock::new(self.get_inner_lock())
        }

        fn in_use(&self, ptr: *mut ()) -> bool {
            let mut cur = self.first.load(Acquire);
            while !cur.is_null() {
                if unsafe { &* cur }.hazptr.load(Acquire) == ptr {
                    return true;
                }
            }
            false
        }

        /// Retires this pointer to eventually be dropped. It is up to the caller to verify that
        /// the pointer is no longer accessible.
        ///
        /// # Safety
        ///
        /// The caller must guarantee that the ptr is no longer accessible, such that no new hazptr
        /// locks will be created for it. The caller must also guarantee that this function is
        /// called at most once.
        ///
        /// Calling this on a null pointer is a no-op
        pub unsafe fn retire<T: 'static>(&self, ptr: *mut T) {
            if ptr.is_null() {
                return;
            }

            // Trivial case, where the pointer is not in use by any lock
            if !self.in_use(ptr.cast()) {
                unsafe { ptr.drop_in_place() };
            }
            let mut next = self.retired.load(Acquire);
            let retire = Box::into_raw(Box::new(RetiredPointer::new(
                ptr.cast(),
                Box::new(move || unsafe { ptr.drop_in_place() }),
                next,
            )));
            loop {
                if let Ok(_) = self.retired.compare_exchange(next, retire, AcqRel, Relaxed) {
                    break;
                }
                next = self.retired.load(Acquire);
                unsafe { &* retire }.next.store(next, Release);
            }
        }

        fn retire_old(&self) {
            let mut cur = self.retired.load(Acquire);
            while !cur.is_null() {
                let retired = unsafe { &* cur };
                let ptr = retired.ptr.load(Acquire);
                if ptr.is_null() {
                    // Remove node
                } else if !self.in_use(ptr) {
                    // swap and call destructor
                    match retired.ptr.compare_exchange(ptr, std::ptr::null_mut(), AcqRel, Relaxed) {
                        Ok(_p) => (),// We have handled it, the actual memory will be freed by 
                        // .retire
                        Err(_p) => (),// We assume someone else has handled it
                    }
                }
                cur = retired.next.load(Acquire);
            }
        }
    }

    struct HazPtrLock {
        hazptr: AtomicPtr<()>,
        next: AtomicPtr<HazPtrLock>,
        in_use: AtomicBool,
    }

    impl HazPtrLock {
        pub fn new() -> *mut Self {
            Box::into_raw(Box::new(Self {
                hazptr: AtomicPtr::new(std::ptr::null_mut()),
                next: AtomicPtr::new(std::ptr::null_mut()),
                in_use: AtomicBool::new(true),
            }))
        }
    }

    pub struct HazLock {
        lock: &'static HazPtrLock,
    }

    impl HazLock {
        fn new(lock: *const HazPtrLock) -> Self {
            Self { lock: unsafe { &* lock } }
        }

        pub fn load_locked<T>(&self, ptr: &AtomicPtr<T>) -> *mut T {
            let mut current = ptr.load(Acquire);
            loop {
                self.lock.hazptr.store(current.cast(), Release);
                let new = ptr.load(Acquire);
                if current == new {
                    return new;
                }
                current = new;
            }
        }
        
        pub fn reset(&mut self) {
            self.lock.hazptr.store(std::ptr::null_mut(), Release);
        }

        pub fn current(&self) -> *mut () {
            self.lock.hazptr.load(Acquire)
        }
    }
}

