use std::{
    collections::HashMap,
    ops::{Index, IndexMut},
    sync::{
        Arc, Condvar, Mutex,
        atomic::Ordering,
        mpsc::{Receiver, SyncSender},
    },
    thread::Scope,
};

use crate::{
    board::Board,
    history::HistoryTables,
    search::start_worker_search,
    thread::{MAX_THREADS, PvLine, RootMoves, SharedContext, Status, ThreadData},
    time::{Limits, TimeManager},
    transposition::TranspositionTable,
    types::{Move, Score, is_decisive, is_loss, is_win},
};

/// Persistent per-worker search state retained across UCI searches.
pub struct WorkerData {
    pub id: usize,
    pub history: HistoryTables,
    pub root_moves: RootMoves,
    pub pv: PvLine,
    pub completed_depth: u8,
    pub sel_depth: u16,
    pub nodes: u64,
    pub score: i32,
    pub best_move: Option<Move>,
    pub previous_best_score: i32,
    pub multi_pv: usize,
}

impl WorkerData {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            history: HistoryTables::default(),
            root_moves: RootMoves::default(),
            pv: PvLine::new(),
            completed_depth: 0,
            sel_depth: 0,
            nodes: 0,
            score: -Score::INFINITE,
            best_move: None,
            previous_best_score: -Score::INFINITE,
            multi_pv: 1,
        }
    }

    pub fn clear(&mut self) {
        self.history.clear();
        self.root_moves = RootMoves::default();
        self.pv = PvLine::new();
        self.completed_depth = 0;
        self.sel_depth = 0;
        self.nodes = 0;
        self.score = -Score::INFINITE;
        self.best_move = None;
        self.previous_best_score = -Score::INFINITE;
    }
}

pub struct ThreadPool {
    pub workers: Vec<WorkerThread>,
    pub vector: Vec<WorkerData>,
}

impl ThreadPool {
    pub fn available_threads() -> usize {
        match std::thread::available_parallelism() {
            Ok(threads) => (4 * threads.get()).clamp(1, MAX_THREADS),
            Err(_) => MAX_THREADS,
        }
    }

    pub fn new() -> Self {
        let workers = make_worker_threads(1);
        let vector = (0..1).map(WorkerData::new).collect();
        Self { workers, vector }
    }

    pub fn set_count(&mut self, threads: usize) {
        let threads = threads.clamp(1, Self::available_threads());
        self.workers.drain(..).for_each(WorkerThread::join);
        self.workers = make_worker_threads(threads);
        self.vector = (0..threads).map(WorkerData::new).collect();
    }

    pub fn main_thread(&mut self) -> &mut WorkerData {
        &mut self.vector[0]
    }

    pub const fn len(&self) -> usize {
        self.vector.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.vector.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WorkerData> {
        self.vector.iter()
    }

    pub fn clear(&mut self) {
        for worker in &mut self.vector {
            worker.clear();
        }
    }

    pub fn execute_searches(
        &mut self, board: &Board, tt: &TranspositionTable, time_manager: TimeManager, multi_pv: usize,
        shared: &Arc<SharedContext>,
    ) {
        tt.increment_age();
        shared.nodes.reset();
        shared.soft_stop_votes.store(0, Ordering::Release);
        shared.infinite_search.store(matches!(time_manager.limits(), Limits::Infinite), Ordering::Release);
        shared.status.set(Status::RUNNING);
        let init_stat = (self.vector[0].previous_best_score + 32_768).clamp(0, 65_535) as u32;
        for stat in &shared.best_stats {
            stat.store(init_stat, Ordering::Release);
        }

        let thread_count = self.vector.len();
        std::thread::scope(|scope| {
            let mut handlers = Vec::with_capacity(thread_count);
            for (index, (worker_data, worker_thread)) in self.vector.iter_mut().zip(self.workers.iter()).enumerate() {
                let tm = time_manager.clone();
                let shared_ref = Arc::clone(shared);
                let position = board.clone();
                let worker_multi_pv = if index == 0 { multi_pv } else { 1 };

                handlers.push(scope.spawn_into(
                    move || {
                        worker_data.id = index;
                        worker_data.multi_pv = worker_multi_pv;
                        let mut td =
                            ThreadData::with_time_manager(&position, tm, Some(tt), Some(&mut worker_data.history));
                        td.id = index;
                        td.multi_pv = worker_multi_pv;
                        td.thread_count = thread_count;
                        td.shared = Some(&shared_ref);

                        let mut pv = PvLine::new();
                        let (score, best_move) = start_worker_search(&mut td, &mut pv);
                        if index == 0 {
                            shared_ref.status.set(Status::STOPPED);
                        }

                        worker_data.root_moves = td.root_moves;
                        worker_data.pv = pv;
                        worker_data.completed_depth = td.completed_depth;
                        worker_data.sel_depth = td.sel_depth;
                        worker_data.nodes = td.nodes;
                        worker_data.score = score;
                        worker_data.best_move = best_move;
                        worker_data.previous_best_score = score;
                    },
                    worker_thread,
                ));
            }

            for handler in handlers {
                handler.join();
            }
        });
        shared.status.set(Status::STOPPED);
        shared.infinite_search.store(false, Ordering::Release);
    }

    /// Select the strongest worker thread result using depth-weighted move voting.
    pub fn select_best_thread(&self, limits: Limits) -> usize {
        if self.vector.len() <= 1 || self.vector[0].root_moves.is_empty() {
            return 0;
        }
        if matches!(limits, Limits::Depth(_)) || self.vector[0].multi_pv > 1 {
            return 0;
        }

        let min_score = self
            .vector
            .iter()
            .filter(|w| !w.root_moves.is_empty() && w.score > -Score::INFINITE)
            .map(|w| w.score)
            .min()
            .unwrap_or(0);

        let vote_value = |w: &WorkerData| (w.score - min_score + 10) * i32::from(w.completed_depth.max(1));

        let mut votes: HashMap<Move, i32> = HashMap::new();
        for w in &self.vector {
            if let Some(mv) = w.best_move
                && w.score > -Score::INFINITE
            {
                *votes.entry(mv).or_default() += vote_value(w);
            }
        }

        let mut best = 0;
        for current in 1..self.vector.len() {
            let b = &self.vector[best];
            let c = &self.vector[current];
            let Some(c_mv) = c.best_move else { continue };
            let Some(b_mv) = b.best_move else {
                best = current;
                continue;
            };
            if c.score <= -Score::INFINITE || c.completed_depth == 0 {
                continue;
            }

            let is_better = if is_win(b.score) {
                c.score > b.score
            } else if b.score != -Score::INFINITE && is_loss(b.score) {
                c.score < b.score
            } else if is_decisive(c.score) {
                true
            } else {
                let best_vote = votes.get(&b_mv).copied().unwrap_or(0);
                let current_vote = votes.get(&c_mv).copied().unwrap_or(0);
                !is_loss(c.score)
                    && (current_vote > best_vote || (current_vote == best_vote && vote_value(c) > vote_value(b)))
            };

            if is_better {
                best = current;
            }
        }
        best
    }
}

impl Default for ThreadPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Index<usize> for ThreadPool {
    type Output = WorkerData;

    fn index(&self, index: usize) -> &Self::Output {
        &self.vector[index]
    }
}

impl IndexMut<usize> for ThreadPool {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.vector[index]
    }
}

type WorkTask = Box<dyn FnOnce() + Send>;

pub struct WorkerThread {
    handle: std::thread::JoinHandle<()>,
    comms: WorkSender,
}

impl WorkerThread {
    pub fn join(self) {
        drop(self.comms);
        self.handle.join().expect("Worker thread panicked");
    }
}

struct WorkSender {
    sender: SyncSender<WorkTask>,
    completion_signal: Arc<(Mutex<bool>, Condvar)>,
}

struct WorkReceiver {
    receiver: Receiver<WorkTask>,
    completion_signal: Arc<(Mutex<bool>, Condvar)>,
}

fn make_work_channel() -> (WorkSender, WorkReceiver) {
    let (sender, receiver) = std::sync::mpsc::sync_channel(0);
    let completion_signal = Arc::new((Mutex::new(false), Condvar::new()));
    (
        WorkSender { sender, completion_signal: Arc::clone(&completion_signal) },
        WorkReceiver { receiver, completion_signal },
    )
}

pub struct ReceiverHandle<'scope> {
    completion_signal: &'scope Arc<(Mutex<bool>, Condvar)>,
    received: bool,
}

impl ReceiverHandle<'_> {
    pub fn join(mut self) {
        let (lock, cvar) = &**self.completion_signal;
        let mut completed = lock.lock().unwrap();
        while !*completed {
            completed = cvar.wait(completed).unwrap();
        }
        drop(completed);
        self.received = true;
    }
}

impl Drop for ReceiverHandle<'_> {
    fn drop(&mut self) {
        assert!(self.received, "ReceiverHandle was dropped without receiving a value");
    }
}

pub trait ScopeExt<'scope, 'env> {
    fn spawn_into<F>(&'scope self, f: F, comms: &'scope WorkerThread) -> ReceiverHandle<'scope>
    where
        F: FnOnce() + Send + 'scope;
}

impl<'scope, 'env> ScopeExt<'scope, 'env> for Scope<'scope, 'env> {
    fn spawn_into<F>(&'scope self, f: F, thread: &'scope WorkerThread) -> ReceiverHandle<'scope>
    where
        F: FnOnce() + Send + 'scope,
    {
        // Safety: ReceiverHandle::join waits for completion before 'scope ends.
        let f = unsafe { std::mem::transmute::<Box<dyn FnOnce() + Send + 'scope>, WorkTask>(Box::new(f)) };
        {
            let (lock, _) = &*thread.comms.completion_signal;
            let mut completed = lock.lock().unwrap();
            *completed = false;
        }
        thread.comms.sender.send(f).expect("Failed to send task to worker thread");
        ReceiverHandle {
            completion_signal: &thread.comms.completion_signal,
            received: false,
        }
    }
}

fn make_worker_thread() -> WorkerThread {
    let (sender, receiver) = make_work_channel();
    let handle = std::thread::spawn(move || {
        while let Ok(work) = receiver.receiver.recv() {
            work();
            let (lock, cvar) = &*receiver.completion_signal;
            let mut completed = lock.lock().unwrap();
            *completed = true;
            drop(completed);
            cvar.notify_one();
        }
    });
    WorkerThread { handle, comms: sender }
}

fn make_worker_threads(num_threads: usize) -> Vec<WorkerThread> {
    std::iter::repeat_with(make_worker_thread).take(num_threads).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thread::RootMove;

    #[test]
    fn select_best_thread_prefers_higher_depth_weighted_vote_and_faster_mates() {
        let mut pool = ThreadPool::new();
        pool.set_count(3);

        let board = Board::starting_position();
        let moves = board.generate_all_moves();
        let m0 = moves[0];
        let m1 = moves[1];

        for w in &mut pool.vector {
            w.root_moves = RootMoves::new(&board);
        }

        // Thread 0 picked m0 at depth 4, score 20; Threads 1 and 2 picked m1 at depth 6, score 35.
        pool.vector[0].best_move = Some(m0);
        pool.vector[0].score = 20;
        pool.vector[0].completed_depth = 4;

        pool.vector[1].best_move = Some(m1);
        pool.vector[1].score = 35;
        pool.vector[1].completed_depth = 6;

        pool.vector[2].best_move = Some(m1);
        pool.vector[2].score = 35;
        pool.vector[2].completed_depth = 6;

        let best = pool.select_best_thread(Limits::Nodes(10_000));
        assert!(best == 1 || best == 2, "expected thread 1 or 2 to win vote, got {best}");
        assert_eq!(pool[best].best_move, Some(m1));

        // If Thread 0 finds a forced win, only a faster forced win can beat it.
        pool.vector[0].score = Score::MATE - 4;
        pool.vector[2].score = Score::MATE - 2;
        pool.vector[0].root_moves.records = vec![RootMove::new(m0)];
        assert_eq!(pool.select_best_thread(Limits::Nodes(10_000)), 2);
    }
}
