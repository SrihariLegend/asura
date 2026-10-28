use std::{
    alloc::{Layout, alloc_zeroed, dealloc},
    sync::atomic::{AtomicPtr, AtomicU8, AtomicU64, AtomicUsize, Ordering},
};

use crate::{
    search::{MATE_SCORE, MATE_THRESHOLD},
    types::Move,
};

pub const DEFAULT_TT_SIZE: usize = 16;

const MEGABYTE: usize = 1024 * 1024;
const CLUSTER_SIZE: usize = std::mem::size_of::<Cluster>();
const CACHE_LINE_ALIGN: usize = 64;
const ENTRIES_PER_CLUSTER: usize = 3;

const AGE_CYCLE: u8 = 1 << 5;
const AGE_MASK: u8 = AGE_CYCLE - 1;

const _: () = assert!(std::mem::size_of::<Cluster>() == 32);
const _: () = assert!(std::mem::size_of::<InternalEntry>() == 8);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Entry {
    pub mv: Move,
    pub score: i32,
    pub raw_eval: i32,
    pub depth: i32,
    pub bound: Bound,
    pub tt_pv: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Flags {
    data: u8,
}

impl Flags {
    pub const fn new(bound: Bound, tt_pv: bool, age: u8) -> Self {
        debug_assert!(age <= AGE_MASK);
        Self { data: (bound as u8) | ((tt_pv as u8) << 2) | (age << 3) }
    }

    pub const fn bound(self) -> Bound {
        match self.data & 0b11 {
            0 => Bound::None,
            1 => Bound::Exact,
            2 => Bound::Lower,
            3 => Bound::Upper,
            _ => unreachable!(),
        }
    }

    pub const fn tt_pv(self) -> bool {
        (self.data & (1 << 2)) != 0
    }

    pub const fn age(self) -> u8 {
        self.data >> 3
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Bound {
    None = 0,
    Exact = 1,
    Lower = 2,
    Upper = 3,
}

#[derive(Copy, Clone)]
#[repr(C)]
struct InternalEntry {
    mv: Move,
    score: i16,
    raw_eval: i16,
    offset_depth: u8,
    flags: Flags,
}

impl InternalEntry {
    const fn from_u64(raw: u64) -> Self {
        unsafe { std::mem::transmute(raw) }
    }

    const fn to_u64(self) -> u64 {
        unsafe { std::mem::transmute(self) }
    }

    const fn depth(self) -> i32 {
        TtDepth::from_tt(self.offset_depth)
    }

    const fn relative_age(self, tt_age: u8) -> i32 {
        ((AGE_CYCLE + tt_age - self.flags.age()) & AGE_MASK) as i32
    }
}

pub enum TtDepth {}

impl TtDepth {
    pub const NONE: i32 = -6;
    pub const EVAL_ONLY: i32 = -5;
    pub const QS_RECAPTURES: i32 = -4;
    pub const QS_PRE_RECAPTURES: i32 = -3;
    pub const QS_NO_CHECKS: i32 = -2;
    pub const QS_SHALLOW_CHECKS: i32 = -1;
    pub const QS_CHECKS: i32 = 0;
    pub const NONE_SCORE: i32 = 32_002;

    const fn from_tt(offset_depth: u8) -> i32 {
        offset_depth as i32 - 6
    }

    fn to_tt(depth: i32) -> u8 {
        (depth + 6).clamp(u8::MIN as i32, u8::MAX as i32) as u8
    }
}

#[repr(align(32))]
struct Cluster {
    entries: [AtomicU64; ENTRIES_PER_CLUSTER],
    keys: AtomicU64,
}

#[derive(Copy, Clone, PartialEq, Eq)]
struct VerificationKey(u32);

const fn verification_key(hash: u64) -> VerificationKey {
    VerificationKey((hash & Cluster::KEY_MASK) as u32)
}

impl Cluster {
    const KEY_BITS: usize = 21;
    const KEY_MASK: u64 = (1 << Self::KEY_BITS) - 1;
    // Bit 63 of keys is unused by the three 21-bit keys (3 * 21 = 63 bits)
    // and serves as a per-cluster writer lock bit.
    const LOCK_BIT: u64 = 1 << 63;

    fn lock(&self) -> u64 {
        loop {
            let prev = self.keys.fetch_or(Self::LOCK_BIT, Ordering::Acquire);
            if (prev & Self::LOCK_BIT) == 0 {
                return prev;
            }
            std::hint::spin_loop();
        }
    }

    fn unlock_with_key(&self, keys: u64, index: usize, key: VerificationKey) {
        let shift = index * Self::KEY_BITS;
        let mask = Self::KEY_MASK << shift;
        let next = (keys & !Self::LOCK_BIT & !mask) | ((key.0 as u64) << shift);
        self.keys.store(next, Ordering::Release);
    }

    fn unlock(&self, keys: u64) {
        self.keys.store(keys & !Self::LOCK_BIT, Ordering::Release);
    }

    fn load_entry(&self, index: usize) -> InternalEntry {
        InternalEntry::from_u64(self.entries[index].load(Ordering::Acquire))
    }

    fn store_entry(&self, index: usize, entry: InternalEntry) {
        self.entries[index].store(entry.to_u64(), Ordering::Release);
    }

    const fn lookup_in_keys(keys: u64, key: VerificationKey) -> usize {
        let bits = 1 | (1 << Self::KEY_BITS) | (1 << (Self::KEY_BITS * 2));
        let needle = key.0 as u64 * bits;
        let zeros = (keys & !Self::LOCK_BIT) ^ needle;
        let matches = zeros.wrapping_sub(bits) & !zeros & (bits << (Self::KEY_BITS - 1));
        matches.trailing_zeros() as usize / Self::KEY_BITS
    }
}

pub struct TranspositionTable {
    ptr: AtomicPtr<Cluster>,
    len: AtomicUsize,
    age: AtomicU8,
}

unsafe impl Sync for TranspositionTable {}

impl TranspositionTable {
    pub fn new(megabytes: usize) -> Self {
        let (ptr, len) = unsafe { allocate(1, megabytes.max(1)) };
        Self {
            ptr: AtomicPtr::new(ptr),
            len: AtomicUsize::new(len),
            age: AtomicU8::new(0),
        }
    }

    pub fn clear(&mut self, threads: usize) {
        let clusters = unsafe { std::slice::from_raw_parts(self.ptr(), self.len()) };
        parallel_clear_clusters(threads, clusters);
        self.age.store(0, Ordering::Relaxed);
    }

    pub fn resize(&mut self, threads: usize, megabytes: usize) {
        let old_ptr = self.ptr();
        let old_len = self.len();
        let (new_ptr, new_len) = unsafe { allocate(threads, megabytes.max(1)) };
        self.ptr.store(new_ptr, Ordering::Relaxed);
        self.len.store(new_len, Ordering::Relaxed);
        self.age.store(0, Ordering::Relaxed);
        unsafe { deallocate(old_ptr, old_len) };
    }

    pub fn hashfull(&self) -> usize {
        let age = self.age();
        let clusters = unsafe { std::slice::from_raw_parts(self.ptr(), self.len()) };

        let mut count = 0;
        for cluster in clusters.iter().take(1000) {
            for slot in 0..ENTRIES_PER_CLUSTER {
                let entry = cluster.load_entry(slot);
                count += (entry.depth() != TtDepth::NONE && entry.flags.age() == age) as usize;
            }
        }

        count / ENTRIES_PER_CLUSTER
    }

    pub fn increment_age(&self) {
        self.age.store((self.age() + 1) & AGE_MASK, Ordering::Relaxed);
    }

    pub fn read(&self, hash: u64, halfmove_clock: u8, ply: isize) -> Option<Entry> {
        let cluster = {
            let index = index(hash, self.len());
            unsafe { &*self.ptr().add(index) }
        };

        let keys1 = cluster.keys.load(Ordering::Acquire);
        if (keys1 & Cluster::LOCK_BIT) != 0 {
            return None;
        }

        let key = verification_key(hash);
        let slot = Cluster::lookup_in_keys(keys1, key);

        if slot < ENTRIES_PER_CLUSTER {
            let entry = cluster.load_entry(slot);
            let keys2 = cluster.keys.load(Ordering::Acquire);
            if keys1 == keys2 && entry.depth() != TtDepth::NONE {
                return Some(Entry {
                    depth: entry.depth(),
                    score: score_from_tt(entry.score as i32, ply, halfmove_clock),
                    raw_eval: entry.raw_eval as i32,
                    bound: entry.flags.bound(),
                    tt_pv: entry.flags.tt_pv(),
                    mv: entry.mv,
                });
            }
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write(
        &self, hash: u64, depth: i32, raw_eval: i32, mut score: i32, bound: Bound, mv: Move, ply: isize, tt_pv: bool,
        force: bool,
    ) {
        debug_assert!(depth > TtDepth::NONE);

        let cluster = {
            let index = index(hash, self.len());
            unsafe { &*self.ptr().add(index) }
        };

        let key = verification_key(hash);
        let tt_age = self.age();
        let locked_keys = cluster.lock();

        let lookup_index = Cluster::lookup_in_keys(locked_keys, key);
        let (replacement_index, matched_occupied) =
            if lookup_index < ENTRIES_PER_CLUSTER && cluster.load_entry(lookup_index).depth() != TtDepth::NONE {
                (lookup_index, true)
            } else {
                let mut chosen = 0;
                let mut lowest_quality = i32::MAX;

                for idx in 0..ENTRIES_PER_CLUSTER {
                    let candidate = cluster.load_entry(idx);
                    if candidate.depth() == TtDepth::NONE {
                        chosen = idx;
                        break;
                    }
                    let quality = candidate.depth() - 4 * candidate.relative_age(tt_age);
                    if quality < lowest_quality {
                        chosen = idx;
                        lowest_quality = quality;
                    }
                }
                (chosen, false)
            };

        let mut entry = cluster.load_entry(replacement_index);

        if !(matched_occupied && mv.is_null()) {
            entry.mv = mv;
        }

        if !force
            && matched_occupied
            && bound != Bound::Exact
            && depth + 4 + 2 * tt_pv as i32 <= entry.depth()
            && entry.flags.age() == tt_age
        {
            cluster.store_entry(replacement_index, entry);
            cluster.unlock(locked_keys);
            return;
        }

        if is_decisive_score(score) && score != TtDepth::NONE_SCORE {
            score += score.signum() * ply as i32;
        }

        entry.offset_depth = TtDepth::to_tt(depth);
        entry.score = score as i16;
        entry.raw_eval = raw_eval as i16;
        entry.flags = Flags::new(bound, tt_pv, tt_age);
        cluster.store_entry(replacement_index, entry);
        cluster.unlock_with_key(locked_keys, replacement_index, key);
    }

    pub fn prefetch(&self, hash: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

            let index = index(hash, self.len());
            let ptr = self.ptr().add(index).cast();
            _mm_prefetch::<_MM_HINT_T0>(ptr);
        }

        #[cfg(not(target_arch = "x86_64"))]
        let _ = hash;
    }

    fn age(&self) -> u8 {
        self.age.load(Ordering::Relaxed)
    }

    fn ptr(&self) -> *mut Cluster {
        self.ptr.load(Ordering::Relaxed)
    }

    fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }
}

const fn index(hash: u64, len: usize) -> usize {
    (((hash as u128) * (len as u128)) >> 64) as usize
}

const fn is_decisive_score(score: i32) -> bool {
    score != TtDepth::NONE_SCORE && score.abs() >= MATE_THRESHOLD
}

const fn score_from_tt(score: i32, ply: isize, halfmove_clock: u8) -> i32 {
    if score == TtDepth::NONE_SCORE {
        return TtDepth::NONE_SCORE;
    }

    if score >= MATE_THRESHOLD {
        if MATE_SCORE - score > 100 - halfmove_clock as i32 {
            return MATE_THRESHOLD - 1;
        }
        return score - ply as i32;
    }

    if score <= -MATE_THRESHOLD {
        if MATE_SCORE + score > 100 - halfmove_clock as i32 {
            return -MATE_THRESHOLD + 1;
        }
        return score + ply as i32;
    }

    score
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new(DEFAULT_TT_SIZE)
    }
}

impl Drop for TranspositionTable {
    fn drop(&mut self) {
        unsafe { deallocate(self.ptr(), self.len()) };
    }
}

unsafe fn allocate(threads: usize, size_mb: usize) -> (*mut Cluster, usize) {
    let size = size_mb * MEGABYTE;
    let len = size / CLUSTER_SIZE;
    let layout = Layout::from_size_align(len * CLUSTER_SIZE, CACHE_LINE_ALIGN).unwrap();
    let ptr: *mut Cluster = unsafe { alloc_zeroed(layout).cast() };
    assert!(!ptr.is_null(), "failed to allocate transposition table");
    let clusters = unsafe { std::slice::from_raw_parts(ptr, len) };
    parallel_clear_clusters(threads, clusters);
    (ptr, len)
}

unsafe fn deallocate(ptr: *mut Cluster, len: usize) {
    let layout = Layout::from_size_align(len * CLUSTER_SIZE, CACHE_LINE_ALIGN).unwrap();
    unsafe { dealloc(ptr.cast(), layout) };
}

fn clear_cluster_slice(clusters: &[Cluster]) {
    for cluster in clusters {
        for entry in &cluster.entries {
            entry.store(0, Ordering::Relaxed);
        }
        cluster.keys.store(0, Ordering::Relaxed);
    }
}

fn parallel_clear_clusters(threads: usize, clusters: &[Cluster]) {
    if threads <= 1 || clusters.len() < 4096 {
        clear_cluster_slice(clusters);
        return;
    }
    std::thread::scope(|scope| {
        let chunk_size = clusters.len().div_ceil(threads);
        for chunk in clusters.chunks(chunk_size) {
            scope.spawn(|| clear_cluster_slice(chunk));
        }
    });
}
