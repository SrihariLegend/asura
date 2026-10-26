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
fn go_does_not_pretend_to_search() {
    let output = exchange("go depth 1\nstop\nisready\nquit\n");
    assert_eq!(output, "info string Search is not implemented\nbestmove 0000\nreadyok\n");
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
