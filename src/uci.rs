use std::{
    collections::VecDeque,
    io::{self, BufRead, Write},
};

use crate::{board::Board, tools};

pub fn message_loop(mut commands: VecDeque<String>, mut input: impl BufRead, mut output: impl Write) -> io::Result<()> {
    let mut board = Board::starting_position();
    let mut frc = false;
    let mut use_stdin = commands.is_empty();

    loop {
        let message = if let Some(command) = commands.pop_front() {
            command
        } else if use_stdin {
            let mut message = String::new();
            if input.read_line(&mut message)? == 0 {
                break;
            }
            message
        } else {
            break;
        };

        let tokens: Vec<_> = message.split_whitespace().collect();
        match tokens.as_slice() {
            ["uci"] => {
                writeln!(output, "id name Asura {}", env!("CARGO_PKG_VERSION"))?;
                writeln!(output, "id author Asura contributors")?;
                writeln!(output, "option name UCI_Chess960 type check default false")?;
                writeln!(output, "uciok")?;
                use_stdin = true;
            }
            ["isready"] => writeln!(output, "readyok")?,
            ["ucinewgame"] => {
                board = Board::starting_position();
                board.set_frc(frc);
            }
            ["position", tokens @ ..] => match position(tokens, frc) {
                Ok(position) => board = position,
                Err(error) => writeln!(output, "info string {error}")?,
            },
            ["setoption", "name", "UCI_Chess960", "value", value] => match value.parse::<bool>() {
                Ok(value) => {
                    frc = value;
                    board.set_frc(frc);
                }
                Err(_) => writeln!(output, "info string UCI_Chess960 requires true or false")?,
            },
            ["perft", depth] => match depth.parse::<usize>() {
                Ok(depth) => {
                    tools::divide(&mut board, depth, &mut output)?;
                }
                Err(_) => writeln!(output, "info string perft depth must be a non-negative integer")?,
            },
            ["perft", ..] => writeln!(output, "info string Usage: perft <depth>")?,
            ["d"] => writeln!(output, "{board}")?,
            ["go", ..] => {
                writeln!(output, "info string Search is not implemented")?;
                writeln!(output, "bestmove 0000")?;
            }
            ["quit"] => break,
            // No active search to stop. Unknown commands are ignored per UCI.
            ["stop"] | [] => (),
            _ => (),
        }
        output.flush()?;
    }
    output.flush()
}

/// Construct a new position before replacing the live board, so a malformed FEN
/// or an illegal move cannot leave a partially applied command behind.
fn position(tokens: &[&str], frc: bool) -> Result<Board, String> {
    let moves_index = tokens.iter().position(|&token| token == "moves").unwrap_or(tokens.len());
    let (position, moves) = tokens.split_at(moves_index);
    let mut board = match position {
        ["startpos"] => Board::starting_position(),
        ["fen", fen @ ..] => Board::from_fen(&fen.join(" ")).map_err(|error| format!("Invalid FEN: {error}"))?,
        _ => return Err("Usage: position startpos|fen <fen> [moves <moves...>]".to_string()),
    };
    board.set_frc(frc);

    for &token in moves.iter().skip(1) {
        let mv = board
            .generate_all_moves()
            .iter()
            .copied()
            .find(|mv| mv.to_uci(&board) == token)
            .ok_or_else(|| format!("Illegal move: {token}"))?;
        board.make_move(mv);
    }
    Ok(board)
}
