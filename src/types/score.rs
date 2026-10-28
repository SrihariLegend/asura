use super::MAX_PLY;

pub struct Score;

#[rustfmt::skip]
impl Score {
    pub const ZERO:        i32 = 0;
    pub const NONE:        i32 = 32_002;
    pub const INFINITE:    i32 = 32_001;
    pub const MATE:        i32 = 32_000;
    pub const MATE_IN_MAX: i32 = Self::MATE - MAX_PLY as i32;
}

pub const fn mated_in(ply: isize) -> i32 {
    -Score::MATE + ply as i32
}

pub const fn mate_in(ply: isize) -> i32 {
    Score::MATE - ply as i32
}

pub const fn is_win(score: i32) -> bool {
    score >= Score::MATE_IN_MAX
}

pub const fn is_loss(score: i32) -> bool {
    score <= -Score::MATE_IN_MAX
}

pub const fn is_decisive(score: i32) -> bool {
    is_win(score) || is_loss(score)
}

pub const fn is_valid(score: i32) -> bool {
    score != Score::NONE
}
