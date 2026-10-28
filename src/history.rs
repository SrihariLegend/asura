use crate::{
    board::Board,
    types::{Bitboard, Color, Move, Piece, PieceType, Square},
};

type FromToHistory<T> = [[T; 64]; 64];
type PieceToHistory<T> = [[T; 64]; 13];
type QuietContextArray = [[[[FromToHistory<i16>; 2]; 2]; 2]; 2];

const PLY_OFFSET: usize = 6;
const MAX_HISTORY_PLY: usize = 264;
const CONT_KEYS: usize = 2 * 2 * 13 * 64 + 1;

fn apply_bonus<const MAX: i32>(entry: &mut i16, bonus: i32) {
    let bonus = bonus.clamp(-MAX, MAX);
    *entry += (bonus - bonus.abs() * (*entry) as i32 / MAX) as i16;
}

pub struct QuietHistory {
    // [side_to_move][from_threatened][to_threatened][creates_threat][from][to]
    entries: Box<QuietContextArray>,
}

impl QuietHistory {
    pub const MAX_HISTORY: i32 = 10_000;

    pub fn get(&self, threats: Bitboard, creates_threat: bool, stm: Color, mv: Move) -> i32 {
        self.entries[stm][threats.contains(mv.from()) as usize][threats.contains(mv.to()) as usize]
            [creates_threat as usize][mv.from()][mv.to()] as i32
    }

    pub fn update(&mut self, threats: Bitboard, creates_threat: bool, stm: Color, mv: Move, bonus: i32) {
        let entry = &mut self.entries[stm][threats.contains(mv.from()) as usize][threats.contains(mv.to()) as usize]
            [creates_threat as usize][mv.from()][mv.to()];
        apply_bonus::<{ Self::MAX_HISTORY }>(entry, bonus);
    }

    pub fn clear(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.entries.as_mut() as *mut _, 0, 1);
        }
    }
}

impl Default for QuietHistory {
    fn default() -> Self {
        Self { entries: zeroed_box() }
    }
}

pub struct PawnHistory {
    // [pawn_key_bucket][piece][to]
    entries: Box<[PieceToHistory<i16>; Self::SIZE]>,
}

impl PawnHistory {
    pub const MAX_HISTORY: i32 = 10_000;
    const SIZE: usize = 512;
    const MASK: usize = Self::SIZE - 1;

    pub fn get(&self, pawn_key: u64, piece: Piece, to: Square) -> i32 {
        self.entries[pawn_key as usize & Self::MASK][piece][to] as i32
    }

    pub fn update(&mut self, pawn_key: u64, piece: Piece, to: Square, bonus: i32) {
        let entry = &mut self.entries[pawn_key as usize & Self::MASK][piece][to];
        apply_bonus::<{ Self::MAX_HISTORY }>(entry, bonus);
    }

    pub fn clear(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.entries.as_mut() as *mut _, 0, 1);
        }
    }
}

impl Default for PawnHistory {
    fn default() -> Self {
        Self { entries: zeroed_box() }
    }
}

pub struct NoisyHistory {
    // [piece][to][captured_piece_type][to_threatened]
    entries: Box<PieceToHistory<[[i16; 2]; 7]>>,
}

impl NoisyHistory {
    pub const MAX_HISTORY: i32 = 10_000;

    pub fn get(&self, threats: Bitboard, piece: Piece, sq: Square, captured: PieceType) -> i32 {
        self.entries[piece][sq][captured][threats.contains(sq) as usize] as i32
    }

    pub fn update(&mut self, threats: Bitboard, piece: Piece, sq: Square, captured: PieceType, bonus: i32) {
        let entry = &mut self.entries[piece][sq][captured][threats.contains(sq) as usize];
        apply_bonus::<{ Self::MAX_HISTORY }>(entry, bonus);
    }

    pub fn clear(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.entries.as_mut() as *mut _, 0, 1);
        }
    }
}

impl Default for NoisyHistory {
    fn default() -> Self {
        Self { entries: zeroed_box() }
    }
}

pub struct ContinuationHistory {
    // Lazily allocated subtables indexed by ((in_check * 2 + noisy) * 13 + piece) * 64 + to.
    // Index 0 is reserved as the "no predecessor" sentinel.
    tables: Box<[Option<Box<PieceToHistory<i16>>>; CONT_KEYS]>,
    active_keys: Vec<u16>,
    active_flags: Box<[bool; CONT_KEYS]>,
    plies: [u16; MAX_HISTORY_PLY],
}

impl ContinuationHistory {
    pub const MAX_HISTORY: i32 = 10_000;

    const fn encode_key(in_check: bool, is_noisy: bool, piece: Piece, to: Square) -> u16 {
        if matches!(piece, Piece::None) || (to as usize) >= Square::NUM {
            return 0;
        }
        let idx = (((in_check as usize) * 2 + (is_noisy as usize)) * 13 + (piece as usize)) * 64 + (to as usize);
        (idx + 1) as u16
    }

    pub fn seed_root(&mut self, board: &Board) {
        self.plies.fill(0);
        for distance in 1..=PLY_OFFSET {
            if let Some((mv, piece, was_in_check)) = board.recent_move_from_root(distance) {
                self.plies[PLY_OFFSET - distance] = Self::encode_key(was_in_check, mv.is_noisy(), piece, mv.to());
            }
        }
    }

    pub fn remember(&mut self, ply: usize, in_check: bool, mv: Move, piece: Piece) {
        let slot = ply + PLY_OFFSET;
        if slot >= MAX_HISTORY_PLY {
            return;
        }
        if mv.is_null() || piece == Piece::None {
            self.plies[slot] = 0;
        } else {
            self.plies[slot] = Self::encode_key(in_check, mv.is_noisy(), piece, mv.to());
        }
    }

    pub fn get(&self, ply: usize, offset: usize, piece: Piece, to: Square) -> i32 {
        let slot = (ply + PLY_OFFSET).wrapping_sub(offset);
        if slot >= MAX_HISTORY_PLY {
            return 0;
        }
        let key = self.plies[slot] as usize;
        if key == 0 || key >= CONT_KEYS || !self.active_flags[key] {
            return 0;
        }
        self.tables[key].as_ref().map_or(0, |table| table[piece][to] as i32)
    }

    pub fn update(&mut self, ply: usize, offset: usize, piece: Piece, to: Square, bonus: i32) {
        let slot = (ply + PLY_OFFSET).wrapping_sub(offset);
        if slot >= MAX_HISTORY_PLY {
            return;
        }
        let key = self.plies[slot] as usize;
        if key == 0 || key >= CONT_KEYS {
            return;
        }
        if !self.active_flags[key] {
            self.active_flags[key] = true;
            self.active_keys.push(key as u16);
            if let Some(table) = &mut self.tables[key] {
                unsafe {
                    std::ptr::write_bytes(table.as_mut() as *mut _, 0, 1);
                }
            } else {
                self.tables[key] = Some(zeroed_box());
            }
        }
        let table = self.tables[key].as_mut().unwrap();
        apply_bonus::<{ Self::MAX_HISTORY }>(&mut table[piece][to], bonus);
    }

    pub fn clear(&mut self) {
        for &key in &self.active_keys {
            self.active_flags[key as usize] = false;
        }
        self.active_keys.clear();
        self.plies.fill(0);
    }
}

impl Default for ContinuationHistory {
    fn default() -> Self {
        Self {
            tables: Box::new([const { None }; CONT_KEYS]),
            active_keys: Vec::with_capacity(256),
            active_flags: zeroed_box(),
            plies: [0; MAX_HISTORY_PLY],
        }
    }
}

pub struct HistoryTables {
    pub quiet_history: QuietHistory,
    pub pawn_history: PawnHistory,
    pub noisy_history: NoisyHistory,
    pub continuation_history: ContinuationHistory,
    killers: [[Move; 2]; MAX_HISTORY_PLY],
    countermoves: Box<FromToHistory<Move>>,
}

impl HistoryTables {
    pub fn prepare_search(&mut self, board: &Board) {
        self.killers.fill([Move::NULL; 2]);
        self.continuation_history.seed_root(board);
    }

    pub fn clear_child_killers(&mut self, ply: usize) {
        if ply + 1 < MAX_HISTORY_PLY {
            self.killers[ply + 1] = [Move::NULL; 2];
        }
    }

    pub fn killers(&self, ply: usize) -> [Move; 2] {
        if ply < MAX_HISTORY_PLY { self.killers[ply] } else { [Move::NULL; 2] }
    }

    pub fn countermove(&self, previous: Move) -> Move {
        if previous.is_null() { Move::NULL } else { self.countermoves[previous.from()][previous.to()] }
    }

    pub fn update_killer_and_countermove(&mut self, ply: usize, previous: Move, best: Move) {
        if best.is_null() || best.is_noisy() {
            return;
        }
        if ply < MAX_HISTORY_PLY && self.killers[ply][0] != best {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = best;
        }
        if previous.is_present() {
            self.countermoves[previous.from()][previous.to()] = best;
        }
    }

    pub fn conthist(&self, ply: usize, offset: usize, piece: Piece, to: Square) -> i32 {
        self.continuation_history.get(ply, offset, piece, to)
    }

    pub fn update_continuation_histories(&mut self, ply: usize, piece: Piece, to: Square, bonus: i32) {
        for offset in [1, 2, 4, 6] {
            self.continuation_history.update(ply, offset, piece, to, bonus);
        }
    }

    pub fn clear(&mut self) {
        self.quiet_history.clear();
        self.pawn_history.clear();
        self.noisy_history.clear();
        self.continuation_history.clear();
        self.killers.fill([Move::NULL; 2]);
        unsafe {
            std::ptr::write_bytes(self.countermoves.as_mut() as *mut _, 0, 1);
        }
    }
}

impl Default for HistoryTables {
    fn default() -> Self {
        Self {
            quiet_history: QuietHistory::default(),
            pawn_history: PawnHistory::default(),
            noisy_history: NoisyHistory::default(),
            continuation_history: ContinuationHistory::default(),
            killers: [[Move::NULL; 2]; MAX_HISTORY_PLY],
            countermoves: zeroed_box(),
        }
    }
}

fn zeroed_box<T>() -> Box<T> {
    unsafe {
        let layout = std::alloc::Layout::new::<T>();
        let ptr = std::alloc::alloc_zeroed(layout);
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Box::<T>::from_raw(ptr.cast())
    }
}
