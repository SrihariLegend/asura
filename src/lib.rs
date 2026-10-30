#![allow(unsafe_op_in_unsafe_fn)]
#![warn(clippy::large_types_passed_by_value)]
#![warn(clippy::trivially_copy_pass_by_ref)]
#![warn(clippy::redundant_clone)]

use std::{collections::VecDeque, io};

pub mod board;
pub mod history;
pub mod movepick;
pub mod search;
pub mod thread;
pub mod threadpool;
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
