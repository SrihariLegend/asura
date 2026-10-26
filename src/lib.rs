use std::{collections::VecDeque, io};

pub mod board;
pub mod tools;
pub mod types;

mod lookup;
mod setwise;
mod uci;

/// Execute queued commands, or read UCI commands from stdin if the queue is empty.
/// Each queued string is one complete command. Sending `uci` enables stdin mode.
pub fn run(commands: VecDeque<String>) -> io::Result<()> {
    uci::message_loop(commands, io::stdin().lock(), io::stdout().lock())
}
