use super::*;
use std::{
    sync::{atomic::AtomicBool, mpsc},
    time::{Duration, Instant},
};

const TIMEOUT: Duration = Duration::from_secs(5);
const BLOCKED_CHECK: Duration = Duration::from_millis(50);

#[test]
fn two_outstanding_jobs_have_independent_completion() {
    let worker = make_worker_thread();
    let (release, gate) = mpsc::channel();
    let (started, entered) = mpsc::channel();
    // SAFETY: Owned captures; both handles are joined below.
    let mut first = unsafe { worker.dispatch_borrowed(|| std::thread::sleep(BLOCKED_CHECK)) };
    let mut second = unsafe {
        worker.dispatch_borrowed(move || {
            started.send(()).unwrap();
            gate.recv_timeout(TIMEOUT).unwrap();
        })
    };
    entered.recv_timeout(TIMEOUT).unwrap();
    first.join();
    let (joining, join_started) = mpsc::channel();
    let (done, completed) = mpsc::channel();
    let waiter = std::thread::spawn(move || {
        joining.send(()).unwrap();
        second.join();
        done.send(()).unwrap();
    });
    join_started.recv_timeout(TIMEOUT).unwrap();
    let early = completed.recv_timeout(BLOCKED_CHECK).is_ok();
    release.send(()).unwrap();
    waiter.join().unwrap();
    worker.join();
    assert!(!early, "first job's completion satisfied the second join");
}

#[test]
fn dropped_handle_waits_for_its_borrowed_job() {
    let worker = make_worker_thread();
    let finished = AtomicBool::new(false);
    let (release, gate) = mpsc::channel();
    let (started, entered) = mpsc::channel();
    // SAFETY: The private handle is dropped/joined by a test thread that is
    // itself joined before finished leaves this function, even on unwind.
    let finished_ref = &finished;
    let handle = unsafe {
        worker.dispatch_borrowed(move || {
            started.send(()).unwrap();
            gate.recv_timeout(TIMEOUT).unwrap();
            finished_ref.store(true, Ordering::Release);
        })
    };
    entered.recv_timeout(TIMEOUT).unwrap();
    let early = std::thread::scope(|scope| {
        let (dropping, drop_started) = mpsc::channel();
        let (done, completed) = mpsc::channel();
        let waiter = scope.spawn(move || {
            dropping.send(()).unwrap();
            drop(handle);
            done.send(()).unwrap();
        });
        drop_started.recv_timeout(TIMEOUT).unwrap();
        let early = completed.recv_timeout(BLOCKED_CHECK).is_ok();
        release.send(()).unwrap();
        waiter.join().unwrap();
        early
    });
    worker.join();
    assert!(!early, "Drop returned while the borrowed job was blocked");
    assert!(finished.load(Ordering::Acquire));
}

#[test]
fn panic_signals_join_and_same_os_worker_runs_next_job() {
    let worker = make_worker_thread();
    let id = worker.handle.thread().id();
    let (sent, received) = mpsc::channel();
    // SAFETY: Owned captures and an explicitly joined handle.
    let mut failed = unsafe {
        worker.dispatch_borrowed(move || {
            sent.send(std::thread::current().id()).unwrap();
            panic!("job panic");
        })
    };
    let panic = catch_unwind(AssertUnwindSafe(|| failed.join())).unwrap_err();
    assert_eq!(panic.downcast_ref::<&str>(), Some(&"job panic"));
    assert_eq!(received.recv_timeout(TIMEOUT).unwrap(), id);
    let mut next_id = None;
    // SAFETY: next is dropped before next_id on every path.
    let mut next = unsafe { worker.dispatch_borrowed(|| next_id = Some(std::thread::current().id())) };
    next.join();
    drop(next);
    assert_eq!(next_id, Some(id));
    worker.join();
}

#[test]
fn capture_destructor_panic_also_signals_completion() {
    struct PanicOnDrop;
    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!("capture destructor");
        }
    }
    let worker = make_worker_thread();
    let bomb = PanicOnDrop;
    // SAFETY: No borrowed captures; the handle is joined.
    let mut job = unsafe { worker.dispatch_borrowed(move || drop(bomb)) };
    let panic = catch_unwind(AssertUnwindSafe(|| job.join())).unwrap_err();
    assert_eq!(panic.downcast_ref::<&str>(), Some(&"capture destructor"));
    worker.join();
}

#[test]
fn drop_during_unwind_waits_without_dropping_a_panicking_payload() {
    struct PanicOnDrop;
    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!("panic payload destructor must not run during cleanup");
        }
    }
    let worker = make_worker_thread();
    let finished = AtomicBool::new(false);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The handle is owned locally and dropped before finished expires.
        let _job = unsafe {
            worker.dispatch_borrowed(|| {
                std::thread::sleep(BLOCKED_CHECK);
                finished.store(true, Ordering::Release);
                std::panic::panic_any(PanicOnDrop);
            })
        };
        panic!("caller panic");
    }))
    .unwrap_err();
    assert_eq!(panic.downcast_ref::<&str>(), Some(&"caller panic"));
    assert!(finished.load(Ordering::Acquire));
    // This must still be the same live worker after the discarded panic.
    let mut next = unsafe { worker.dispatch_borrowed(|| {}) };
    next.join();
    worker.join();
}

// Test-only controlled workers exercise the REAL execute_searches control flow
// without adding production hooks or changing search logic. Accepted work is
// always consumed and signalled, even when a timeout/assertion panics.
fn controlled_worker(f: impl FnOnce(mpsc::Receiver<WorkItem>) + Send + 'static) -> WorkerThread {
    let (comms, receiver) = mpsc::sync_channel(0);
    let handle = std::thread::spawn(move || f(receiver));
    WorkerThread { handle, comms }
}

fn delayed_search_worker(shared: Arc<SharedContext>, finished: Arc<AtomicBool>) -> WorkerThread {
    controlled_worker(move |receiver| {
        let WorkItem { task, completion } = receiver.recv_timeout(TIMEOUT).unwrap();
        let result = catch_unwind(AssertUnwindSafe(move || {
            // Only the parent unwind guard can release this task. In particular,
            // cleanup must request stop BEFORE waiting on pending handles.
            let deadline = Instant::now() + TIMEOUT;
            while shared.status.get() != Status::STOPPED {
                assert!(Instant::now() < deadline, "cleanup did not stop searches before joining");
                std::thread::yield_now();
            }
            assert!(!shared.infinite_search.load(Ordering::Acquire));
            std::thread::sleep(BLOCKED_CHECK);
            task(); // Real search still accesses borrowed tt and WorkerData.
        }));
        finished.store(true, Ordering::Release);
        completion.finish(result);
    })
}

fn execute_unwind_case(dispatch_panics: bool) {
    let shared = Arc::new(SharedContext::default());
    let finished = [Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false))];
    let mut workers = Vec::new();
    if !dispatch_panics {
        workers.push(controlled_worker(|receiver| {
            let WorkItem { task, completion } = receiver.recv_timeout(TIMEOUT).unwrap();
            let result = catch_unwind(AssertUnwindSafe(move || {
                drop(task); // Consume the borrowed captures before signalling.
                panic!("first joined job panic");
            }));
            completion.finish(result);
        }));
    }
    for flag in &finished {
        workers.push(delayed_search_worker(Arc::clone(&shared), Arc::clone(flag)));
    }
    if dispatch_panics {
        let dead = controlled_worker(drop);
        // Ensure the receiver is disconnected before dispatching to this worker.
        while !dead.handle.is_finished() {
            std::thread::yield_now();
        }
        workers.push(dead);
    }
    let mut pool = ThreadPool {
        vector: (0..workers.len()).map(WorkerData::new).collect(),
        workers,
    };
    let board = Board::starting_position();
    let tt = TranspositionTable::new(1);
    pool.prepare_search(&tt, true, &shared);
    let result = catch_unwind(AssertUnwindSafe(|| {
        pool.execute_searches(&board, &tt, TimeManager::new(Limits::Depth(1), 1, 0), 1, &shared);
    }));
    let completed_before_return = finished.iter().all(|flag| flag.load(Ordering::Acquire));
    // Also drain the OS workers before assertions, so a broken regression cannot
    // leave test borrows alive after their referents, even if it returned early.
    pool.workers.drain(..).for_each(WorkerThread::join);
    let panic = result.unwrap_err();
    let expected = if dispatch_panics { "Failed to send task to worker thread" } else { "first joined job panic" };
    assert_eq!(panic.downcast_ref::<&str>(), Some(&expected));
    assert!(completed_before_return, "execute_searches returned before all borrowed jobs completed");
    assert_eq!(shared.status.get(), Status::STOPPED);
    assert!(!shared.infinite_search.load(Ordering::Acquire));
    let start = usize::from(!dispatch_panics);
    for worker in &pool.vector[start..start + 2] {
        assert!(worker.best_move.is_some(), "pending borrowed search was not executed");
    }
}

#[test]
fn dispatch_panic_joins_all_previous_borrowed_searches() {
    execute_unwind_case(true);
}

#[test]
fn one_join_panicking_still_joins_all_other_borrowed_searches() {
    execute_unwind_case(false);
}
