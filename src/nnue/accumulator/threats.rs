use std::sync::OnceLock;

use super::psq::orient_flip;
use crate::{
    board::Board,
    lookup::{attacks, bishop_attacks, knight_attacks, pawn_attacks, rook_attacks},
    nnue::{PAWN_PAIR_ROWS, PSQ_ROWS, THREAT_ROWS},
    types::{Bitboard, Color, Piece, PieceType, Square},
};

pub struct ThreatLookupTables {
    pub slot_base: [[[i32; 12]; 12]; 2],
    pub semi_excl: [[u8; 12]; 12],
    pub att_from_offset: [[i32; 64]; 10],
    pub att_ray_index: [[[u8; 64]; 64]; 5],
    pub att_kind: [[u8; 12]; 2],
    pub pp_band_mask: [u64; 64],
    pub front_span: [[u64; 64]; 2],
    pub front_file: [[u64; 64]; 2],
}

static TABLES: OnceLock<ThreatLookupTables> = OnceLock::new();

pub fn tables() -> &'static ThreatLookupTables {
    TABLES.get_or_init(|| {
        crate::lookup::initialize();
        let mut t = ThreatLookupTables {
            slot_base: [[[-1; 12]; 12]; 2],
            semi_excl: [[0; 12]; 12],
            att_from_offset: [[0; 64]; 10],
            att_ray_index: [[[0; 64]; 64]; 5],
            att_kind: [[0; 12]; 2],
            pp_band_mask: [0; 64],
            front_span: [[0; 64]; 2],
            front_file: [[0; 64]; 2],
        };

        const SLOT_MAP: [[[i32; 6]; 2]; 5] = [
            [[-1, 0, -1, 1, -1, -1], [-1, 2, -1, 3, -1, -1]],
            [[0, 1, 2, 3, 4, -1], [5, 6, 7, 8, 9, -1]],
            [[0, 1, -1, 2, -1, -1], [3, 4, 5, 6, -1, -1]],
            [[0, 1, 2, 3, -1, -1], [4, 5, 6, 7, -1, -1]],
            [[0, 1, 2, 3, -1, -1], [4, 5, 6, 7, 8, -1]],
        ];
        const VALID_SLOTS: [i32; 5] = [4, 10, 7, 8, 9];

        let mut running_base = 0i32;
        for kind in 0..10 {
            let rel_c = kind / 5;
            let pt_idx = kind % 5;
            let color = Color::new(rel_c as u8);
            let pt = PieceType::new(pt_idx);
            let piece = Piece::new(color, pt);

            let mut count = 0i32;
            for sq in 0..64 {
                t.att_from_offset[kind][sq] = count;
                let att = if pt == PieceType::Pawn {
                    if (8..56).contains(&sq) {
                        attacks(piece, Square::new(sq as u8), Bitboard(0))
                    } else {
                        Bitboard(0)
                    }
                } else {
                    attacks(piece, Square::new(sq as u8), Bitboard(0))
                };
                if rel_c == 0 {
                    for to in 0..64 {
                        if pt_idx == 0 {
                            let f_from = sq & 7;
                            let f_to = to & 7;
                            t.att_ray_index[0][sq][to] = if f_from == 0 || f_to < f_from { 0 } else { 1 };
                        } else {
                            t.att_ray_index[pt_idx][sq][to] = (att.0 & ((1u64 << to) - 1)).count_ones() as u8;
                        }
                    }
                }
                count += att.popcount() as i32;
            }

            for pov in 0..2 {
                let abs_att_c = rel_c ^ pov;
                let abs_att_pc = abs_att_c * 6 + pt_idx;
                t.att_kind[pov][abs_att_pc] = kind as u8;

                for abs_def_c in 0..2 {
                    let rel_def_c = abs_def_c ^ pov;
                    for def_pt in 0..6 {
                        let abs_def_pc = abs_def_c * 6 + def_pt;
                        let slot = SLOT_MAP[pt_idx][rel_def_c][def_pt];
                        if slot >= 0 {
                            t.slot_base[pov][abs_att_pc][abs_def_pc] = running_base + slot * count;
                        }
                    }
                }
            }
            running_base += VALID_SLOTS[pt_idx] * count;
        }
        debug_assert_eq!(running_base as usize, THREAT_ROWS);

        for a in 0..12 {
            for d in 0..12 {
                let pt_a = a % 6;
                let pt_d = d % 6;
                let enemy = (a / 6) != (d / 6);
                if pt_a == pt_d && (enemy || pt_a != 0) {
                    t.semi_excl[a][d] = 1;
                }
            }
        }

        for s in 0..64 {
            let f = s & 7;
            let r = s >> 3;
            let mut files = 0x0101_0101_0101_0101u64 << f;
            if f > 0 {
                files |= 0x0101_0101_0101_0101u64 << (f - 1);
            }
            if f < 7 {
                files |= 0x0101_0101_0101_0101u64 << (f + 1);
            }
            t.pp_band_mask[s] = files & !0xFFu64 & !(0xFFu64 << 56) & !(1u64 << s);

            let mut ahead_w = 0u64;
            let mut ahead_b = 0u64;
            for rr in 0..8 {
                if rr > r {
                    ahead_w |= 0xFFu64 << (rr * 8);
                } else if rr < r {
                    ahead_b |= 0xFFu64 << (rr * 8);
                }
            }
            let own_file = 0x0101_0101_0101_0101u64 << f;
            t.front_file[0][s] = ahead_w & own_file;
            t.front_file[1][s] = ahead_b & own_file;
            t.front_span[0][s] = ahead_w & files;
            t.front_span[1][s] = ahead_b & files;
        }

        t
    })
}

#[inline]
pub fn threat_row_index(
    att_color: Color, att_pt: PieceType, from: Square, def_color: Color, def_pt: PieceType, to: Square, ksq: Square,
    pov: Color,
) -> Option<usize> {
    if att_pt == PieceType::King || def_pt == PieceType::King {
        return None;
    }
    let tbl = tables();
    let p = pov as usize;
    let apc = (att_color as usize) * 6 + (att_pt as usize);
    let dpc = (def_color as usize) * 6 + (def_pt as usize);
    let base = tbl.slot_base[p][apc][dpc];
    if base < 0 {
        return None;
    }
    let flip = orient_flip(pov, ksq);
    let fo = (from as u8 ^ flip) as usize;
    let to_or = (to as u8 ^ flip) as usize;
    if tbl.semi_excl[apc][dpc] != 0 && fo < to_or {
        return None;
    }
    let kind = tbl.att_kind[p][apc] as usize;
    let ray_idx = if att_pt == PieceType::Pawn {
        let f_from = fo & 7;
        let f_to = to_or & 7;
        if f_from == 0 || f_to < f_from { 0 } else { 1 }
    } else {
        tbl.att_ray_index[att_pt as usize][fo][to_or] as usize
    };
    let idx = base as usize + tbl.att_from_offset[kind][fo] as usize + ray_idx;
    Some(PSQ_ROWS + idx)
}

#[inline]
pub fn pawn_pair_row_index(c1: Color, s1: Square, c2: Color, s2: Square, ksq: Square, pov: Color) -> usize {
    let flip = orient_flip(pov, ksq);
    let o1 = (s1 as u8 ^ flip) as usize;
    let o2 = (s2 as u8 ^ flip) as usize;
    let id1 = ((c1 != pov) as usize) * 48 + (o1 - 8);
    let id2 = ((c2 != pov) as usize) * 48 + (o2 - 8);
    let hi = id1.max(id2);
    let lo = id1.min(id2);
    PSQ_ROWS + THREAT_ROWS + (hi * (hi - 1)) / 2 + lo
}

#[inline]
pub fn passed_pawn_row_index(c: Color, sq: Square, ksq: Square, pov: Color) -> usize {
    let flip = orient_flip(pov, ksq);
    let o = (sq as u8 ^ flip) as usize;
    let id = ((c != pov) as usize) * 48 + (o - 8);
    PSQ_ROWS + THREAT_ROWS + PAWN_PAIR_ROWS + id
}

pub fn collect_threat_rows(board: &Board, pov: Color, out: &mut Vec<usize>) {
    let tbl = tables();
    let p = pov as usize;
    let flip = orient_flip(pov, board.king_square(pov));
    let occ = board.occupancies();
    let kings = board.pieces(PieceType::King);
    let non_kings = occ & !kings;
    for sq in non_kings {
        let att_pc = board.piece_on(sq);
        let att_pt = att_pc.piece_type();
        let apc = (att_pc.color() as usize) * 6 + (att_pt as usize);
        let fo = (sq as u8 ^ flip) as usize;
        let kind = tbl.att_kind[p][apc] as usize;
        let from_off = tbl.att_from_offset[kind][fo] as usize;
        let targets = attacks(att_pc, sq, occ) & non_kings;
        for to in targets {
            let def_pc = board.piece_on(to);
            let dpc = (def_pc.color() as usize) * 6 + (def_pc.piece_type() as usize);
            let base = tbl.slot_base[p][apc][dpc];
            if base < 0 {
                continue;
            }
            let to_or = (to as u8 ^ flip) as usize;
            if tbl.semi_excl[apc][dpc] != 0 && fo < to_or {
                continue;
            }
            let ray_idx = if att_pt == PieceType::Pawn {
                let f_from = fo & 7;
                let f_to = to_or & 7;
                if f_from == 0 || f_to < f_from { 0 } else { 1 }
            } else {
                tbl.att_ray_index[att_pt as usize][fo][to_or] as usize
            };
            out.push(PSQ_ROWS + base as usize + from_off + ray_idx);
        }
    }
}

pub fn collect_threat_rows_both(board: &Board, out_w: &mut Vec<usize>, out_b: &mut Vec<usize>) {
    let tbl = tables();
    let flip_w = orient_flip(Color::White, board.king_square(Color::White));
    let flip_b = orient_flip(Color::Black, board.king_square(Color::Black));
    let occ = board.occupancies();
    let kings = board.pieces(PieceType::King);
    let non_kings = occ & !kings;
    for sq in non_kings {
        let att_pc = board.piece_on(sq);
        let att_pt = att_pc.piece_type();
        let pt_u = att_pt as usize;
        let apc = (att_pc.color() as usize) * 6 + pt_u;
        let fo_w = (sq as u8 ^ flip_w) as usize;
        let fo_b = (sq as u8 ^ flip_b) as usize;
        let kind_w = tbl.att_kind[0][apc] as usize;
        let kind_b = tbl.att_kind[1][apc] as usize;
        let from_off_w = tbl.att_from_offset[kind_w][fo_w] as usize;
        let from_off_b = tbl.att_from_offset[kind_b][fo_b] as usize;
        let targets = attacks(att_pc, sq, occ) & non_kings;
        for to in targets {
            let def_pc = board.piece_on(to);
            let dpc = (def_pc.color() as usize) * 6 + (def_pc.piece_type() as usize);
            let semi = tbl.semi_excl[apc][dpc] != 0;

            let base_w = tbl.slot_base[0][apc][dpc];
            if base_w >= 0 {
                let to_w = (to as u8 ^ flip_w) as usize;
                if !(semi && fo_w < to_w) {
                    let ray_w = if att_pt == PieceType::Pawn {
                        let f_from = fo_w & 7;
                        let f_to = to_w & 7;
                        if f_from == 0 || f_to < f_from { 0 } else { 1 }
                    } else {
                        tbl.att_ray_index[pt_u][fo_w][to_w] as usize
                    };
                    out_w.push(PSQ_ROWS + base_w as usize + from_off_w + ray_w);
                }
            }

            let base_b = tbl.slot_base[1][apc][dpc];
            if base_b >= 0 {
                let to_b = (to as u8 ^ flip_b) as usize;
                if !(semi && fo_b < to_b) {
                    let ray_b = if att_pt == PieceType::Pawn {
                        let f_from = fo_b & 7;
                        let f_to = to_b & 7;
                        if f_from == 0 || f_to < f_from { 0 } else { 1 }
                    } else {
                        tbl.att_ray_index[pt_u][fo_b][to_b] as usize
                    };
                    out_b.push(PSQ_ROWS + base_b as usize + from_off_b + ray_b);
                }
            }
        }
    }
}

#[inline(always)]
fn push_threat_both(
    tbl: &ThreatLookupTables, flip_w: u8, flip_b: u8, att_pc: Piece, sq: Square, def_pc: Piece, to: Square,
    out_w: &mut Vec<usize>, out_b: &mut Vec<usize>,
) {
    let pt_u = att_pc.piece_type() as usize;
    let apc = (att_pc.color() as usize) * 6 + pt_u;
    let dpc = (def_pc.color() as usize) * 6 + (def_pc.piece_type() as usize);
    let semi = tbl.semi_excl[apc][dpc] != 0;

    let base_w = tbl.slot_base[0][apc][dpc];
    let fo_w = (sq as u8 ^ flip_w) as usize;
    let to_w = (to as u8 ^ flip_w) as usize;
    if base_w >= 0 && !(semi && fo_w < to_w) {
        let kind_w = tbl.att_kind[0][apc] as usize;
        let idx_w = PSQ_ROWS
            + base_w as usize
            + tbl.att_from_offset[kind_w][fo_w] as usize
            + tbl.att_ray_index[pt_u][fo_w][to_w] as usize;
        out_w.push(idx_w);
    }

    let base_b = tbl.slot_base[1][apc][dpc];
    let fo_b = (sq as u8 ^ flip_b) as usize;
    let to_b = (to as u8 ^ flip_b) as usize;
    if base_b >= 0 && !(semi && fo_b < to_b) {
        let kind_b = tbl.att_kind[1][apc] as usize;
        let idx_b = PSQ_ROWS
            + base_b as usize
            + tbl.att_from_offset[kind_b][fo_b] as usize
            + tbl.att_ray_index[pt_u][fo_b][to_b] as usize;
        out_b.push(idx_b);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn push_targets_from_attacker_both(
    tbl: &ThreatLookupTables, flip_w: u8, flip_b: u8, att_pc: Piece, sq: Square, targets: Bitboard,
    board: &BoardSnapshot, out_w: &mut Vec<usize>, out_b: &mut Vec<usize>,
) {
    let pt_u = att_pc.piece_type() as usize;
    let apc = (att_pc.color() as usize) * 6 + pt_u;
    let fo_w = (sq as u8 ^ flip_w) as usize;
    let fo_b = (sq as u8 ^ flip_b) as usize;
    let off_w = PSQ_ROWS + tbl.att_from_offset[tbl.att_kind[0][apc] as usize][fo_w] as usize;
    let off_b = PSQ_ROWS + tbl.att_from_offset[tbl.att_kind[1][apc] as usize][fo_b] as usize;
    let sb_w = &tbl.slot_base[0][apc];
    let sb_b = &tbl.slot_base[1][apc];
    let semi_row = &tbl.semi_excl[apc];
    let ray_row_w = &tbl.att_ray_index[pt_u][fo_w];
    let ray_row_b = &tbl.att_ray_index[pt_u][fo_b];

    for to in targets {
        let def_pc = board.piece_on(to);
        let dpc = (def_pc.color() as usize) * 6 + (def_pc.piece_type() as usize);
        let semi = semi_row[dpc] != 0;

        let base_w = sb_w[dpc];
        let to_w = (to as u8 ^ flip_w) as usize;
        if base_w >= 0 && !(semi && fo_w < to_w) {
            out_w.push(off_w + base_w as usize + ray_row_w[to_w] as usize);
        }

        let base_b = sb_b[dpc];
        let to_b = (to as u8 ^ flip_b) as usize;
        if base_b >= 0 && !(semi && fo_b < to_b) {
            out_b.push(off_b + base_b as usize + ray_row_b[to_b] as usize);
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn push_threat_both_replace(
    tbl: &ThreatLookupTables, flip_w: u8, flip_b: u8, att_pc: Piece, sq: Square, old_def_pc: Piece, new_def_pc: Piece,
    to: Square, subs_w: &mut Vec<usize>, adds_w: &mut Vec<usize>, subs_b: &mut Vec<usize>, adds_b: &mut Vec<usize>,
) {
    let pt_u = att_pc.piece_type() as usize;
    let apc = (att_pc.color() as usize) * 6 + pt_u;
    let dpc_old = (old_def_pc.color() as usize) * 6 + (old_def_pc.piece_type() as usize);
    let dpc_new = (new_def_pc.color() as usize) * 6 + (new_def_pc.piece_type() as usize);

    let fo_w = (sq as u8 ^ flip_w) as usize;
    let to_w = (to as u8 ^ flip_w) as usize;
    let off_ray_w = PSQ_ROWS
        + tbl.att_from_offset[tbl.att_kind[0][apc] as usize][fo_w] as usize
        + tbl.att_ray_index[pt_u][fo_w][to_w] as usize;
    let bw_old = tbl.slot_base[0][apc][dpc_old];
    if bw_old >= 0 && !(tbl.semi_excl[apc][dpc_old] != 0 && fo_w < to_w) {
        subs_w.push(off_ray_w + bw_old as usize);
    }
    let bw_new = tbl.slot_base[0][apc][dpc_new];
    if bw_new >= 0 && !(tbl.semi_excl[apc][dpc_new] != 0 && fo_w < to_w) {
        adds_w.push(off_ray_w + bw_new as usize);
    }

    let fo_b = (sq as u8 ^ flip_b) as usize;
    let to_b = (to as u8 ^ flip_b) as usize;
    let off_ray_b = PSQ_ROWS
        + tbl.att_from_offset[tbl.att_kind[1][apc] as usize][fo_b] as usize
        + tbl.att_ray_index[pt_u][fo_b][to_b] as usize;
    let bb_old = tbl.slot_base[1][apc][dpc_old];
    if bb_old >= 0 && !(tbl.semi_excl[apc][dpc_old] != 0 && fo_b < to_b) {
        subs_b.push(off_ray_b + bb_old as usize);
    }
    let bb_new = tbl.slot_base[1][apc][dpc_new];
    if bb_new >= 0 && !(tbl.semi_excl[apc][dpc_new] != 0 && fo_b < to_b) {
        adds_b.push(off_ray_b + bb_new as usize);
    }
}

/// Lightweight 136-byte `Copy` snapshot of board piece placement (avoids cloning `Board::state_stack`).
#[derive(Copy, Clone)]
pub struct BoardSnapshot {
    pub pieces: [Bitboard; 6],
    pub colors: [Bitboard; 2],
    pub mailbox: [Piece; 64],
    pub hash: u64,
}

impl Default for BoardSnapshot {
    fn default() -> Self {
        Self {
            pieces: [Bitboard(0); 6],
            colors: [Bitboard(0); 2],
            mailbox: [Piece::None; 64],
            hash: 0,
        }
    }
}

impl BoardSnapshot {
    #[inline(always)]
    pub fn from_board(b: &Board) -> Self {
        Self {
            pieces: *b.pieces_array(),
            colors: *b.colors_array(),
            mailbox: *b.mailbox_array(),
            hash: b.hash(),
        }
    }

    #[inline(always)]
    pub const fn occupancies(&self) -> Bitboard {
        Bitboard(self.colors[0].0 | self.colors[1].0)
    }

    #[inline(always)]
    pub const fn pieces(&self, pt: PieceType) -> Bitboard {
        self.pieces[pt as usize]
    }

    #[inline(always)]
    pub const fn colors(&self, c: Color) -> Bitboard {
        self.colors[c as usize]
    }

    #[inline(always)]
    pub const fn colored_pieces(&self, c: Color, pt: PieceType) -> Bitboard {
        Bitboard(self.colors[c as usize].0 & self.pieces[pt as usize].0)
    }

    #[inline(always)]
    pub const fn piece_on(&self, sq: Square) -> Piece {
        self.mailbox[sq as usize]
    }

    #[inline(always)]
    pub const fn king_square(&self, c: Color) -> Square {
        Square::new(self.colored_pieces(c, PieceType::King).0.trailing_zeros() as u8)
    }

    #[inline(always)]
    pub const fn heavy_regime(&self, pov: Color) -> usize {
        let opp_c = (pov as usize) ^ 1;
        if (self.colors[opp_c].0 & self.pieces[PieceType::Queen as usize].0) != 0 {
            2
        } else if (self.pieces[PieceType::Rook as usize].0 | self.pieces[PieceType::Queen as usize].0) != 0 {
            1
        } else {
            0
        }
    }
}

#[inline(always)]
fn push_threat_one(
    tbl: &ThreatLookupTables, p: usize, flip: u8, att_pc: Piece, sq: Square, def_pc: Piece, to: Square,
    out: &mut Vec<usize>,
) {
    let pt_u = att_pc.piece_type() as usize;
    let apc = (att_pc.color() as usize) * 6 + pt_u;
    let dpc = (def_pc.color() as usize) * 6 + (def_pc.piece_type() as usize);
    let base = tbl.slot_base[p][apc][dpc];
    let fo = (sq as u8 ^ flip) as usize;
    let to_or = (to as u8 ^ flip) as usize;
    if base >= 0 && !(tbl.semi_excl[apc][dpc] != 0 && fo < to_or) {
        let kind = tbl.att_kind[p][apc] as usize;
        let idx = PSQ_ROWS
            + base as usize
            + tbl.att_from_offset[kind][fo] as usize
            + tbl.att_ray_index[pt_u][fo][to_or] as usize;
        out.push(idx);
    }
}

pub fn diff_threat_rows_one(
    prev_board: &BoardSnapshot, board: &BoardSnapshot, pov: Color, flip: u8, subs: &mut Vec<usize>,
    adds: &mut Vec<usize>,
) -> Bitboard {
    let tbl = tables();
    let p = pov as usize;
    let occ_old = prev_board.occupancies();
    let occ_new = board.occupancies();
    let non_kings_old = occ_old & !prev_board.pieces(PieceType::King);
    let non_kings_new = occ_new & !board.pieces(PieceType::King);

    let changed_sq = (prev_board.colors(Color::White) ^ board.colors(Color::White))
        | (prev_board.colors(Color::Black) ^ board.colors(Color::Black))
        | (prev_board.pieces(PieceType::Pawn) ^ board.pieces(PieceType::Pawn))
        | (prev_board.pieces(PieceType::Knight) ^ board.pieces(PieceType::Knight))
        | (prev_board.pieces(PieceType::Bishop) ^ board.pieces(PieceType::Bishop))
        | (prev_board.pieces(PieceType::Rook) ^ board.pieces(PieceType::Rook))
        | (prev_board.pieces(PieceType::Queen) ^ board.pieces(PieceType::Queen))
        | (prev_board.pieces(PieceType::King) ^ board.pieces(PieceType::King));

    let occ_diff = occ_old ^ occ_new;
    let occ_common = occ_old & occ_new;
    let mut ortho_rays = Bitboard(0);
    let mut diag_rays = Bitboard(0);
    for d_sq in occ_diff {
        ortho_rays |= rook_attacks(d_sq, occ_common);
        diag_rays |= bishop_attacks(d_sq, occ_common);
    }

    let unchanged_non_kings = non_kings_new & !changed_sq;
    let rooks_queens = board.pieces(PieceType::Rook) | board.pieces(PieceType::Queen);
    let bishops_queens = board.pieces(PieceType::Bishop) | board.pieces(PieceType::Queen);
    let dirty_sliders = unchanged_non_kings & ((ortho_rays & rooks_queens) | (diag_rays & bishops_queens));
    let set3 = unchanged_non_kings & !dirty_sliders;

    for sq in changed_sq & non_kings_old {
        let att_pc = prev_board.piece_on(sq);
        let targets = attacks(att_pc, sq, occ_old) & non_kings_old;
        for to in targets {
            push_threat_one(tbl, p, flip, att_pc, sq, prev_board.piece_on(to), to, subs);
        }
    }
    for sq in changed_sq & non_kings_new {
        let att_pc = board.piece_on(sq);
        let targets = attacks(att_pc, sq, occ_new) & non_kings_new;
        for to in targets {
            push_threat_one(tbl, p, flip, att_pc, sq, board.piece_on(to), to, adds);
        }
    }

    for sq in dirty_sliders {
        let att_pc = board.piece_on(sq);
        let old_targets = attacks(att_pc, sq, occ_old) & non_kings_old;
        let new_targets = attacks(att_pc, sq, occ_new) & non_kings_new;
        for to in old_targets & (!new_targets | changed_sq) {
            push_threat_one(tbl, p, flip, att_pc, sq, prev_board.piece_on(to), to, subs);
        }
        for to in new_targets & (!old_targets | changed_sq) {
            push_threat_one(tbl, p, flip, att_pc, sq, board.piece_on(to), to, adds);
        }
    }

    let wp = board.colored_pieces(Color::White, PieceType::Pawn);
    let bp = board.colored_pieces(Color::Black, PieceType::Pawn);
    let knights = board.pieces(PieceType::Knight);
    for to in changed_sq {
        let old_valid = non_kings_old.contains(to);
        let new_valid = non_kings_new.contains(to);
        if !old_valid && !new_valid {
            continue;
        }
        let attackers = set3
            & ((pawn_attacks(to, Color::Black) & wp)
                | (pawn_attacks(to, Color::White) & bp)
                | (knight_attacks(to) & knights)
                | (bishop_attacks(to, occ_new) & bishops_queens)
                | (rook_attacks(to, occ_new) & rooks_queens));
        if old_valid {
            let old_def_pc = prev_board.piece_on(to);
            for sq in attackers {
                push_threat_one(tbl, p, flip, board.piece_on(sq), sq, old_def_pc, to, subs);
            }
        }
        if new_valid {
            let new_def_pc = board.piece_on(to);
            for sq in attackers {
                push_threat_one(tbl, p, flip, board.piece_on(sq), sq, new_def_pc, to, adds);
            }
        }
    }
    changed_sq
}

/// Theorem 38: Bitwise Kogge-Stone North/South Fill for Passed Pawns (`0` loops, `0` table lookups!).
#[inline(always)]
pub const fn passed_pawns_mask(wp: u64, bp: u64) -> (u64, u64) {
    let mut b_south = bp >> 8;
    b_south |= b_south >> 8;
    b_south |= b_south >> 16;
    b_south |= b_south >> 32;
    let b_front_span = b_south | ((b_south >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((b_south << 1) & 0xfefe_fefe_fefe_fefe);

    let mut w_south = wp >> 8;
    w_south |= w_south >> 8;
    w_south |= w_south >> 16;
    w_south |= w_south >> 32;

    let mut w_north = wp << 8;
    w_north |= w_north << 8;
    w_north |= w_north << 16;
    w_north |= w_north << 32;
    let w_front_span = w_north | ((w_north >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((w_north << 1) & 0xfefe_fefe_fefe_fefe);

    let mut b_north = bp << 8;
    b_north |= b_north << 8;
    b_north |= b_north << 16;
    b_north |= b_north << 32;

    (wp & !b_front_span & !w_south, bp & !w_front_span & !b_north)
}

pub fn diff_pawn_rows_one(
    prev_board: &BoardSnapshot, board: &BoardSnapshot, pov: Color, flip: u8, subs: &mut Vec<usize>,
    adds: &mut Vec<usize>,
) {
    let tbl = tables();
    let p = pov as usize;
    let wp_old = prev_board.colored_pieces(Color::White, PieceType::Pawn).0;
    let bp_old = prev_board.colored_pieces(Color::Black, PieceType::Pawn).0;
    let wp_new = board.colored_pieces(Color::White, PieceType::Pawn).0;
    let bp_new = board.colored_pieces(Color::Black, PieceType::Pawn).0;

    let all_old = wp_old | bp_old;
    let all_new = wp_new | bp_new;
    let changed_pawns = (wp_old ^ wp_new) | (bp_old ^ bp_new);
    let base_pp = PSQ_ROWS + THREAT_ROWS;
    let base_pass = base_pp + PAWN_PAIR_ROWS;

    let mut rem_it = changed_pawns & all_old;
    while rem_it != 0 {
        let s1 = rem_it.trailing_zeros() as usize;
        rem_it &= rem_it - 1;
        let c1 = usize::from(((bp_old >> s1) & 1) != 0);
        let lower_mask = (1u64 << s1) - 1;
        let mut partners = tbl.pp_band_mask[s1] & all_old & (!changed_pawns | lower_mask);
        while partners != 0 {
            let s2 = partners.trailing_zeros() as usize;
            partners &= partners - 1;
            let c2 = usize::from(((bp_old >> s2) & 1) != 0);
            let o1 = (s1 as u8 ^ flip) as usize;
            let o2 = (s2 as u8 ^ flip) as usize;
            let id1 = (c1 ^ p) * 48 + (o1 - 8);
            let id2 = (c2 ^ p) * 48 + (o2 - 8);
            let hi = id1.max(id2);
            let lo = id1.min(id2);
            subs.push(base_pp + (hi * (hi - 1)) / 2 + lo);
        }
    }

    let mut add_it = changed_pawns & all_new;
    while add_it != 0 {
        let s1 = add_it.trailing_zeros() as usize;
        add_it &= add_it - 1;
        let c1 = usize::from(((bp_new >> s1) & 1) != 0);
        let lower_mask = (1u64 << s1) - 1;
        let mut partners = tbl.pp_band_mask[s1] & all_new & (!changed_pawns | lower_mask);
        while partners != 0 {
            let s2 = partners.trailing_zeros() as usize;
            partners &= partners - 1;
            let c2 = usize::from(((bp_new >> s2) & 1) != 0);
            let o1 = (s1 as u8 ^ flip) as usize;
            let o2 = (s2 as u8 ^ flip) as usize;
            let id1 = (c1 ^ p) * 48 + (o1 - 8);
            let id2 = (c2 ^ p) * 48 + (o2 - 8);
            let hi = id1.max(id2);
            let lo = id1.min(id2);
            adds.push(base_pp + (hi * (hi - 1)) / 2 + lo);
        }
    }

    let (w_pass_old, b_pass_old) = passed_pawns_mask(wp_old, bp_old);
    let (w_pass_new, b_pass_new) = passed_pawns_mask(wp_new, bp_new);
    let mut rem_wp = w_pass_old & !w_pass_new;
    while rem_wp != 0 {
        let s = rem_wp.trailing_zeros() as usize;
        rem_wp &= rem_wp - 1;
        subs.push(base_pass + p * 48 + (((s as u8 ^ flip) as usize) - 8));
    }
    let mut add_wp = w_pass_new & !w_pass_old;
    while add_wp != 0 {
        let s = add_wp.trailing_zeros() as usize;
        add_wp &= add_wp - 1;
        adds.push(base_pass + p * 48 + (((s as u8 ^ flip) as usize) - 8));
    }
    let mut rem_bp = b_pass_old & !b_pass_new;
    while rem_bp != 0 {
        let s = rem_bp.trailing_zeros() as usize;
        rem_bp &= rem_bp - 1;
        subs.push(base_pass + (1 ^ p) * 48 + (((s as u8 ^ flip) as usize) - 8));
    }
    let mut add_bp = b_pass_new & !b_pass_old;
    while add_bp != 0 {
        let s = add_bp.trailing_zeros() as usize;
        add_bp &= add_bp - 1;
        adds.push(base_pass + (1 ^ p) * 48 + (((s as u8 ^ flip) as usize) - 8));
    }
}

/// Theorem 28: Exact 3-Set Ray-Incremental Threat Diff when `flip_w` and `flip_b` are invariant.
/// Returns `changed_sq` so the caller can also update `PSQ` in `O(|changed_sq|)` time.
#[allow(clippy::too_many_arguments)]
pub fn diff_threat_rows_both(
    prev_board: &BoardSnapshot, board: &BoardSnapshot, flip_w: u8, flip_b: u8, subs_w: &mut Vec<usize>,
    adds_w: &mut Vec<usize>, subs_b: &mut Vec<usize>, adds_b: &mut Vec<usize>,
) -> Bitboard {
    let tbl = tables();
    let occ_old = prev_board.occupancies();
    let occ_new = board.occupancies();
    let non_kings_old = occ_old & !prev_board.pieces(PieceType::King);
    let non_kings_new = occ_new & !board.pieces(PieceType::King);

    let changed_sq = (prev_board.colors(Color::White) ^ board.colors(Color::White))
        | (prev_board.colors(Color::Black) ^ board.colors(Color::Black))
        | (prev_board.pieces(PieceType::Pawn) ^ board.pieces(PieceType::Pawn))
        | (prev_board.pieces(PieceType::Knight) ^ board.pieces(PieceType::Knight))
        | (prev_board.pieces(PieceType::Bishop) ^ board.pieces(PieceType::Bishop))
        | (prev_board.pieces(PieceType::Rook) ^ board.pieces(PieceType::Rook))
        | (prev_board.pieces(PieceType::Queen) ^ board.pieces(PieceType::Queen))
        | (prev_board.pieces(PieceType::King) ^ board.pieces(PieceType::King));

    let occ_diff = occ_old ^ occ_new;
    let occ_common = occ_old & occ_new;
    let mut ortho_rays = Bitboard(0);
    let mut diag_rays = Bitboard(0);
    for d_sq in occ_diff {
        ortho_rays |= rook_attacks(d_sq, occ_common);
        diag_rays |= bishop_attacks(d_sq, occ_common);
    }

    let unchanged_non_kings = non_kings_new & !changed_sq;
    let rooks_queens = board.pieces(PieceType::Rook) | board.pieces(PieceType::Queen);
    let bishops_queens = board.pieces(PieceType::Bishop) | board.pieces(PieceType::Queen);
    let dirty_sliders = unchanged_non_kings & ((ortho_rays & rooks_queens) | (diag_rays & bishops_queens));
    let set3 = unchanged_non_kings & !dirty_sliders;

    // Set 1: Attackers on `changed_sq` (hoisted per-attacker lookup!)
    for sq in changed_sq & non_kings_old {
        let att_pc = prev_board.piece_on(sq);
        let targets = attacks(att_pc, sq, occ_old) & non_kings_old;
        push_targets_from_attacker_both(tbl, flip_w, flip_b, att_pc, sq, targets, prev_board, subs_w, subs_b);
    }
    for sq in changed_sq & non_kings_new {
        let att_pc = board.piece_on(sq);
        let targets = attacks(att_pc, sq, occ_new) & non_kings_new;
        push_targets_from_attacker_both(tbl, flip_w, flip_b, att_pc, sq, targets, board, adds_w, adds_b);
    }

    // Set 2: Unchanged sliders whose rays were touched by `occ_diff` (hoisted per-attacker lookup!)
    for sq in dirty_sliders {
        let att_pc = board.piece_on(sq);
        let old_targets = attacks(att_pc, sq, occ_old) & non_kings_old;
        let new_targets = attacks(att_pc, sq, occ_new) & non_kings_new;
        push_targets_from_attacker_both(
            tbl,
            flip_w,
            flip_b,
            att_pc,
            sq,
            old_targets & (!new_targets | changed_sq),
            prev_board,
            subs_w,
            subs_b,
        );
        push_targets_from_attacker_both(
            tbl,
            flip_w,
            flip_b,
            att_pc,
            sq,
            new_targets & (!old_targets | changed_sq),
            board,
            adds_w,
            adds_b,
        );
    }

    // Set 3: Unchanged attackers with invariant attack masks hitting `to in changed_sq`
    // Theorem 44: Sliders in `set3` cannot attack any `to in occ_diff`; only check sliders when `!occ_diff.contains(to)`!
    let wp = board.colored_pieces(Color::White, PieceType::Pawn);
    let bp = board.colored_pieces(Color::Black, PieceType::Pawn);
    let knights = board.pieces(PieceType::Knight);
    for to in changed_sq {
        let old_valid = non_kings_old.contains(to);
        let new_valid = non_kings_new.contains(to);
        if !old_valid && !new_valid {
            continue;
        }
        let mut att_mask = (pawn_attacks(to, Color::Black) & wp)
            | (pawn_attacks(to, Color::White) & bp)
            | (knight_attacks(to) & knights);
        if !occ_diff.contains(to) {
            att_mask |= (bishop_attacks(to, occ_new) & bishops_queens) | (rook_attacks(to, occ_new) & rooks_queens);
        }
        let attackers = set3 & att_mask;
        if old_valid && new_valid {
            let old_def_pc = prev_board.piece_on(to);
            let new_def_pc = board.piece_on(to);
            for sq in attackers {
                push_threat_both_replace(
                    tbl,
                    flip_w,
                    flip_b,
                    board.piece_on(sq),
                    sq,
                    old_def_pc,
                    new_def_pc,
                    to,
                    subs_w,
                    adds_w,
                    subs_b,
                    adds_b,
                );
            }
        } else if old_valid {
            let old_def_pc = prev_board.piece_on(to);
            for sq in attackers {
                push_threat_both(tbl, flip_w, flip_b, board.piece_on(sq), sq, old_def_pc, to, subs_w, subs_b);
            }
        } else {
            let new_def_pc = board.piece_on(to);
            for sq in attackers {
                push_threat_both(tbl, flip_w, flip_b, board.piece_on(sq), sq, new_def_pc, to, adds_w, adds_b);
            }
        }
    }
    changed_sq
}

pub fn collect_pawn_rows(board: &Board, pov: Color, out: &mut Vec<usize>) {
    let ksq = board.king_square(pov);
    let tbl = tables();
    let wp = board.colored_pieces(Color::White, PieceType::Pawn).0;
    let bp = board.colored_pieces(Color::Black, PieceType::Pawn).0;
    let all_pawns = wp | bp;
    let mut p_iter = all_pawns;
    while p_iter != 0 {
        let s = p_iter.trailing_zeros() as usize;
        p_iter &= p_iter - 1;
        let sq = Square::new(s as u8);
        let c1 = if ((bp >> s) & 1) != 0 { Color::Black } else { Color::White };
        let mut partners = tbl.pp_band_mask[s] & all_pawns & ((1u64 << s) - 1);
        while partners != 0 {
            let to = partners.trailing_zeros() as usize;
            partners &= partners - 1;
            let c2 = if ((bp >> to) & 1) != 0 { Color::Black } else { Color::White };
            out.push(pawn_pair_row_index(c1, sq, c2, Square::new(to as u8), ksq, pov));
        }
    }

    let mut w_iter = wp;
    while w_iter != 0 {
        let s = w_iter.trailing_zeros() as usize;
        w_iter &= w_iter - 1;
        if (bp & tbl.front_span[0][s]) == 0 && (wp & tbl.front_file[0][s]) == 0 {
            out.push(passed_pawn_row_index(Color::White, Square::new(s as u8), ksq, pov));
        }
    }
    let mut b_iter = bp;
    while b_iter != 0 {
        let s = b_iter.trailing_zeros() as usize;
        b_iter &= b_iter - 1;
        if (wp & tbl.front_span[1][s]) == 0 && (bp & tbl.front_file[1][s]) == 0 {
            out.push(passed_pawn_row_index(Color::Black, Square::new(s as u8), ksq, pov));
        }
    }
}

#[inline(always)]
fn push_pawn_pair_both(
    flip_w: u8, flip_b: u8, c1: Color, s1: usize, c2: Color, s2: usize, out_w: &mut Vec<usize>, out_b: &mut Vec<usize>,
) {
    let base = PSQ_ROWS + THREAT_ROWS;

    let ow1 = (s1 as u8 ^ flip_w) as usize;
    let ow2 = (s2 as u8 ^ flip_w) as usize;
    let idw1 = (c1 as usize) * 48 + (ow1 - 8);
    let idw2 = (c2 as usize) * 48 + (ow2 - 8);
    let hiw = idw1.max(idw2);
    let low = idw1.min(idw2);
    out_w.push(base + (hiw * (hiw - 1)) / 2 + low);

    let ob1 = (s1 as u8 ^ flip_b) as usize;
    let ob2 = (s2 as u8 ^ flip_b) as usize;
    let idb1 = ((c1 as usize) ^ 1) * 48 + (ob1 - 8);
    let idb2 = ((c2 as usize) ^ 1) * 48 + (ob2 - 8);
    let hib = idb1.max(idb2);
    let lob = idb1.min(idb2);
    out_b.push(base + (hib * (hib - 1)) / 2 + lob);
}

#[inline(always)]
fn push_passed_pawn_both(
    flip_w: u8, flip_b: u8, c: Color, s: usize, out_w: &mut Vec<usize>, out_b: &mut Vec<usize>,
) {
    let base = PSQ_ROWS + THREAT_ROWS + PAWN_PAIR_ROWS;
    let ow = (s as u8 ^ flip_w) as usize;
    out_w.push(base + (c as usize) * 48 + (ow - 8));
    let ob = (s as u8 ^ flip_b) as usize;
    out_b.push(base + ((c as usize) ^ 1) * 48 + (ob - 8));
}

/// Theorem 31: Exact O(|changed_pawns|) incremental diff of `PP_3Wide` and `PassedPawns` for both perspectives.
#[allow(clippy::too_many_arguments)]
pub fn diff_pawn_rows_both(
    prev_board: &BoardSnapshot, board: &BoardSnapshot, flip_w: u8, flip_b: u8, subs_w: &mut Vec<usize>,
    adds_w: &mut Vec<usize>, subs_b: &mut Vec<usize>, adds_b: &mut Vec<usize>,
) {
    let tbl = tables();
    let wp_old = prev_board.colored_pieces(Color::White, PieceType::Pawn).0;
    let bp_old = prev_board.colored_pieces(Color::Black, PieceType::Pawn).0;
    let wp_new = board.colored_pieces(Color::White, PieceType::Pawn).0;
    let bp_new = board.colored_pieces(Color::Black, PieceType::Pawn).0;

    let all_old = wp_old | bp_old;
    let all_new = wp_new | bp_new;
    let changed_pawns = (wp_old ^ wp_new) | (bp_old ^ bp_new);

    // 1. Removed pawn pairs (`s1 in changed_pawns & all_old`)
    let mut rem_it = changed_pawns & all_old;
    while rem_it != 0 {
        let s1 = rem_it.trailing_zeros() as usize;
        rem_it &= rem_it - 1;
        let c1 = if ((bp_old >> s1) & 1) != 0 { Color::Black } else { Color::White };
        let lower_mask = (1u64 << s1) - 1;
        let mut partners = tbl.pp_band_mask[s1] & all_old & (!changed_pawns | lower_mask);
        while partners != 0 {
            let s2 = partners.trailing_zeros() as usize;
            partners &= partners - 1;
            let c2 = if ((bp_old >> s2) & 1) != 0 { Color::Black } else { Color::White };
            push_pawn_pair_both(flip_w, flip_b, c1, s1, c2, s2, subs_w, subs_b);
        }
    }

    // 2. Added pawn pairs (`s1 in changed_pawns & all_new`)
    let mut add_it = changed_pawns & all_new;
    while add_it != 0 {
        let s1 = add_it.trailing_zeros() as usize;
        add_it &= add_it - 1;
        let c1 = if ((bp_new >> s1) & 1) != 0 { Color::Black } else { Color::White };
        let lower_mask = (1u64 << s1) - 1;
        let mut partners = tbl.pp_band_mask[s1] & all_new & (!changed_pawns | lower_mask);
        while partners != 0 {
            let s2 = partners.trailing_zeros() as usize;
            partners &= partners - 1;
            let c2 = if ((bp_new >> s2) & 1) != 0 { Color::Black } else { Color::White };
            push_pawn_pair_both(flip_w, flip_b, c1, s1, c2, s2, adds_w, adds_b);
        }
    }

    // 3. Passed pawns bitmask diff (Theorem 38: 0 loops!)
    let (w_pass_old, b_pass_old) = passed_pawns_mask(wp_old, bp_old);
    let (w_pass_new, b_pass_new) = passed_pawns_mask(wp_new, bp_new);
    let mut rem_wp = w_pass_old & !w_pass_new;
    while rem_wp != 0 {
        let s = rem_wp.trailing_zeros() as usize;
        rem_wp &= rem_wp - 1;
        push_passed_pawn_both(flip_w, flip_b, Color::White, s, subs_w, subs_b);
    }
    let mut add_wp = w_pass_new & !w_pass_old;
    while add_wp != 0 {
        let s = add_wp.trailing_zeros() as usize;
        add_wp &= add_wp - 1;
        push_passed_pawn_both(flip_w, flip_b, Color::White, s, adds_w, adds_b);
    }
    let mut rem_bp = b_pass_old & !b_pass_new;
    while rem_bp != 0 {
        let s = rem_bp.trailing_zeros() as usize;
        rem_bp &= rem_bp - 1;
        push_passed_pawn_both(flip_w, flip_b, Color::Black, s, subs_w, subs_b);
    }
    let mut add_bp = b_pass_new & !b_pass_old;
    while add_bp != 0 {
        let s = add_bp.trailing_zeros() as usize;
        add_bp &= add_bp - 1;
        push_passed_pawn_both(flip_w, flip_b, Color::Black, s, adds_w, adds_b);
    }
}

pub fn collect_active_rows_snap(snap: &BoardSnapshot, pov: Color, out: &mut Vec<usize>) {
    use super::psq::psq_row_index;
    out.clear();
    let ksq = snap.king_square(pov);
    let regime = snap.heavy_regime(pov);
    let occ = snap.occupancies();
    for sq in occ {
        let pc = snap.piece_on(sq);
        out.push(psq_row_index(pc.color(), pc.piece_type(), sq, ksq, regime, pov));
    }

    let tbl = tables();
    let p = pov as usize;
    let flip = orient_flip(pov, ksq);
    let non_kings = occ & !snap.pieces(PieceType::King);
    for sq in non_kings {
        let att_pc = snap.piece_on(sq);
        let att_pt = att_pc.piece_type();
        let apc = (att_pc.color() as usize) * 6 + (att_pt as usize);
        let fo = (sq as u8 ^ flip) as usize;
        let kind = tbl.att_kind[p][apc] as usize;
        let from_off = tbl.att_from_offset[kind][fo] as usize;
        let targets = attacks(att_pc, sq, occ) & non_kings;
        for to in targets {
            let def_pc = snap.piece_on(to);
            let dpc = (def_pc.color() as usize) * 6 + (def_pc.piece_type() as usize);
            let base = tbl.slot_base[p][apc][dpc];
            if base < 0 {
                continue;
            }
            let to_or = (to as u8 ^ flip) as usize;
            if tbl.semi_excl[apc][dpc] != 0 && fo < to_or {
                continue;
            }
            let ray_idx = if att_pt == PieceType::Pawn {
                let f_from = fo & 7;
                let f_to = to_or & 7;
                if f_from == 0 || f_to < f_from { 0 } else { 1 }
            } else {
                tbl.att_ray_index[att_pt as usize][fo][to_or] as usize
            };
            out.push(PSQ_ROWS + base as usize + from_off + ray_idx);
        }
    }

    let wp = snap.colored_pieces(Color::White, PieceType::Pawn).0;
    let bp = snap.colored_pieces(Color::Black, PieceType::Pawn).0;
    let all_pawns = wp | bp;
    let mut p_iter = all_pawns;
    while p_iter != 0 {
        let s = p_iter.trailing_zeros() as usize;
        p_iter &= p_iter - 1;
        let sq = Square::new(s as u8);
        let c1 = if ((bp >> s) & 1) != 0 { Color::Black } else { Color::White };
        let mut partners = tbl.pp_band_mask[s] & all_pawns & ((1u64 << s) - 1);
        while partners != 0 {
            let to = partners.trailing_zeros() as usize;
            partners &= partners - 1;
            let c2 = if ((bp >> to) & 1) != 0 { Color::Black } else { Color::White };
            out.push(pawn_pair_row_index(c1, sq, c2, Square::new(to as u8), ksq, pov));
        }
    }

    let (mut w_pass, mut b_pass) = passed_pawns_mask(wp, bp);
    while w_pass != 0 {
        let s = w_pass.trailing_zeros() as usize;
        w_pass &= w_pass - 1;
        out.push(passed_pawn_row_index(Color::White, Square::new(s as u8), ksq, pov));
    }
    while b_pass != 0 {
        let s = b_pass.trailing_zeros() as usize;
        b_pass &= b_pass - 1;
        out.push(passed_pawn_row_index(Color::Black, Square::new(s as u8), ksq, pov));
    }
}
