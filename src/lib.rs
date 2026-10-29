use std::{collections::VecDeque, io};

pub mod board;
pub mod history;
pub mod movepick;
pub mod search;
pub mod thread;
pub mod time;
pub mod tools;
pub mod transposition;
pub mod types;

mod evaluation;
mod lookup;
mod parameters;
mod setwise;
mod stack;
mod uci;

/// Execute queued commands, or read UCI commands from stdin if the queue is empty.
/// Each queued string is one complete command. Sending `uci` enables stdin mode.
pub fn run(commands: VecDeque<String>) -> io::Result<()> {
    lookup::initialize();
    uci::message_loop(commands, io::stdin().lock(), io::stdout().lock())
}
