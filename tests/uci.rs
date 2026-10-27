use asura::board::Board;

use std::{
    io::Write,
    process::{Command, Stdio},
};

fn exchange(commands: &str) -> String {
    let mut child = Command::new(env!("CARGO_BIN_EXE_asura"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(commands.as_bytes()).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "engine failed: {}", String::from_utf8_lossy(&output.stderr));
    assert!(output.stderr.is_empty(), "unexpected stderr: {}", String::from_utf8_lossy(&output.stderr));
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn handshake_only_advertises_implemented_options() {
    let output = exchange("uci\nisready\nquit\n");
    assert!(output.starts_with(&format!("id name Asura {}\n", env!("CARGO_PKG_VERSION"))));
    assert!(output.contains("option name UCI_Chess960 type check default false\n"));
    assert!(output.ends_with("uciok\nreadyok\n"));
    assert!(!output.contains("name Hash"));
    assert!(!output.contains("name Threads"));
}

#[test]
fn position_applies_a_move_sequence() {
    let output = exchange("position startpos moves e2e4 e7e5 g1f3\nd\nquit\n");
    assert!(output.contains("FEN: rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"));
}

#[test]
fn handles_special_moves_from_fen() {
    for (command, expected) in [
        ("position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1 moves e1g1", "r3k2r/8/8/8/8/8/8/R4RK1 b kq - 1 1"),
        ("position fen 4k3/8/8/5pP1/8/8/8/4K3 w - f6 0 1 moves g5f6", "4k3/8/5P2/8/8/8/8/4K3 b - - 0 1"),
        ("position fen 4k3/P7/8/8/8/8/8/4K3 w - - 0 1 moves a7a8n", "N3k3/8/8/8/8/8/8/4K3 b - - 0 1"),
    ] {
        let output = exchange(&format!("{command}\nd\nquit\n"));
        assert!(output.contains(&format!("FEN: {expected}")), "{output}");
    }
}

#[test]
fn chess960_mode_applies_to_startpos_castling_and_new_games() {
    let output = exchange(
        "setoption name UCI_Chess960 value true\n         position startpos\nd\n         position fen 4k3/8/8/8/8/8/8/4K2R w H - 0 1 moves e1h1\nd\n         ucinewgame\nd\nquit\n",
    );
    assert_eq!(output.matches("FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w HAha - 0 1").count(), 2,);
    assert!(output.contains("FEN: 4k3/8/8/8/8/8/8/5RK1 b - - 1 1"));
}

#[test]
fn invalid_positions_leave_the_current_board_unchanged() {
    let output = exchange(
        "position startpos moves e2e4\n         position startpos moves d2d4 not-a-move\n         position fen 8/8/8/8/8/8/8/8 w - - 0 1\n         d\nisready\nquit\n",
    );
    assert!(output.contains("info string Illegal move: not-a-move"));
    assert!(output.contains("info string Invalid FEN:"));
    assert!(output.contains("FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"));
    assert!(output.ends_with("readyok\n"));
}

#[test]
fn perft_counts_and_restores_the_position() {
    let output = exchange("position startpos\nperft 0\nperft 3\nd\nquit\n");
    assert!(output.contains("total: 1\n"));
    assert!(output.contains("total: 8902\n"));
    assert!(output.contains("FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
}

#[test]
fn invalid_diagnostic_input_does_not_crash() {
    let output = exchange(
        "perft\nperft -1\nperft nope\n         setoption name UCI_Chess960 value nope\n         unknown-command\n\n  isready \r\nquit\n",
    );
    assert_eq!(output.matches("info string ").count(), 4);
    assert!(output.ends_with("readyok\n"));
}

#[test]
fn go_depth_returns_a_legal_bestmove_without_changing_the_position() {
    for (command, fen) in [
        ("position startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("position startpos moves e2e4 e7e5 g1f3", "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"),
    ] {
        let board = Board::from_fen(fen).unwrap();
        for depth in [1, 2, 3] {
            let output = exchange(&format!("{command}\nd\ngo depth {depth}\nd\nstop\nisready\nquit\n"));
            let moves: Vec<_> = output.lines().filter_map(|line| line.strip_prefix("bestmove ")).collect();
            assert_eq!(moves.len(), 1);
            assert!(board.generate_all_moves().iter().any(|mv| mv.to_uci(&board) == moves[0]), "{output}");
            assert!(output.contains(&format!("info depth {depth} score ")), "{output}");
            assert_eq!(output.matches(&format!("FEN: {fen}\n")).count(), 2, "{output}");
            assert!(output.ends_with("readyok\n"));
        }
    }
}

#[test]
fn eof_and_quit_exit_cleanly() {
    assert_eq!(exchange(""), "");
    assert_eq!(exchange("isready"), "readyok\n");
    assert_eq!(exchange("quit\nisready\n"), "");
}

#[test]
fn queued_cli_commands_run_without_stdin() {
    let output = Command::new(env!("CARGO_BIN_EXE_asura"))
        .args(["position startpos", "perft 3"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert!(String::from_utf8(output.stdout).unwrap().contains("total: 8902\n"));
}

#[test]
fn go_depth_returns_0000_for_checkmate_and_stalemate() {
    for (fen, score) in [("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "mate 0"), ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "cp -2")] {
        for depth in [1, 3, 255] {
            let output = exchange(&format!("position fen {fen}\ngo depth {depth}\nquit\n"));
            assert_eq!(output, format!("info depth {depth} score {score} nodes 0\nbestmove 0000\n"));
        }
    }
}

#[test]
fn go_depth_reports_mate_in_one_for_either_side() {
    for fen in ["7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", "8/8/8/8/8/6k1/5q2/7K b - - 0 1"] {
        let output = exchange(&format!("position fen {fen}\ngo depth 1\nquit\n"));
        assert!(output.starts_with("info depth 1 score mate 1 nodes "), "{output}");
        let uci = output.lines().find_map(|line| line.strip_prefix("bestmove ")).unwrap();
        let mut board = Board::from_fen(fen).unwrap();
        let mv = board.generate_all_moves().iter().copied().find(|mv| mv.to_uci(&board) == uci).unwrap();
        board.make_move(mv);
        assert!(board.in_check() && !board.has_legal_moves());
    }
}

#[test]
fn go_depth_formats_a_forced_loss_as_negative_mate() {
    let output = exchange("position fen 7k/8/5K1Q/8/8/8/8/8 b - - 0 1\ngo depth 2\nquit\n");
    assert!(output.starts_with("info depth 2 score mate -1 nodes "), "{output}");
    assert!(output.ends_with("bestmove h8g8\n"));
}

#[test]
fn go_depth_reports_material_and_encodes_promotions_and_en_passant() {
    for (fen, best_move, score) in [
        ("4k3/8/8/8/8/8/q7/R3K3 w - - 0 1", "a1a2", 500),
        ("r3k3/Q7/8/8/8/8/8/4K3 b - - 0 1", "a8a7", 500),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q", 900),
        ("4k3/8/8/5pP1/8/8/8/4K3 w - f6 0 1", "g5f6", 100),
    ] {
        let output = exchange(&format!("position fen {fen}\ngo depth 1\nquit\n"));
        assert!(output.starts_with(&format!("info depth 1 score cp {score} nodes ")), "{output}");
        assert!(output.ends_with(&format!("bestmove {best_move}\n")), "{output}");
    }
}

#[test]
fn go_depth_reports_mate_found_beyond_the_full_width_horizon() {
    for (fen, best_move) in [("7k/8/5K2/8/8/8/3Q4/8 w - - 0 1", "d2h6"), ("8/3q4/8/8/8/5k2/8/7K b - - 0 1", "d7h3")] {
        let output = exchange(&format!("position fen {fen}\ngo depth 1\nd\nquit\n"));
        assert!(output.starts_with("info depth 1 score mate 2 nodes "), "{output}");
        assert!(output.contains(&format!("bestmove {best_move}\n")), "{output}");
        assert!(output.contains(&format!("FEN: {fen}\n")), "{output}");
    }
}

#[test]
fn go_depth_rejects_invalid_or_unsupported_limits_without_fake_moves() {
    for command in [
        "go",
        "go depth",
        "go depth 0",
        "go depth -1",
        "go depth nope",
        "go depth 256",
        "go depth 999999999999999999999",
        "go infinite",
        "go movetime 10",
        "go depth 1 extra",
    ] {
        let output = exchange(&format!("{command}\nisready\nquit\n"));
        assert!(output.starts_with("info string "), "{command}: {output}");
        assert!(!output.contains("bestmove "), "{command}: {output}");
        assert!(output.ends_with("readyok\n"));
    }
}

#[test]
fn go_depth_returns_a_legal_move_for_a_material_draw() {
    for fen in [
        "7k/8/8/8/8/8/8/7K w - - 0 1",
        "7k/8/8/8/8/8/NN6/7K w - - 0 1",
        "7k/8/nn6/8/8/8/8/7K b - - 0 1",
        "7k/8/1b6/8/8/8/N7/7K w - - 0 1",
    ] {
        let board = Board::from_fen(fen).unwrap();
        for depth in [1, 3] {
            let output = exchange(&format!("position fen {fen}\ngo depth {depth}\nd\nquit\n"));
            assert!(output.starts_with(&format!("info depth {depth} score cp -2 nodes 0\n")), "{output}");
            let moves: Vec<_> = output.lines().filter_map(|line| line.strip_prefix("bestmove ")).collect();
            assert_eq!(moves.len(), 1);
            assert!(board.generate_all_moves().iter().any(|mv| mv.to_uci(&board) == moves[0]), "{output}");
            assert!(output.contains(&format!("FEN: {fen}\n")));
        }
    }
}
