use std::{
    env,
    fs::File,
    io::{self, BufWriter, Write},
    path::Path,
};

mod attacks;
mod magics;
mod maps;

fn main() -> io::Result<()> {
    println!("cargo:rerun-if-changed=build");

    let dir = env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR");
    let out = File::create(Path::new(&dir).join("lookup.rs"))?;
    write_maps(BufWriter::new(out))
}

fn write_maps(mut buf: BufWriter<File>) -> io::Result<()> {
    macro_rules! write_map {
        ($name:literal, $type:literal, $items:expr) => {{
            let items = $items;
            writeln!(buf, "static {}: [{}; {}] = {:?};", $name, $type, items.len(), items)?;
        }};
    }

    write_map!("DIAGONALS", "[u64; 64]", maps::generate_diagonal_tables());
    write_map!("KING_MAP", "u64", maps::generate_king_map());
    write_map!("KNIGHT_MAP", "u64", maps::generate_knight_map());
    write_map!("PAWN_MAP", "[u64; 64]", maps::generate_pawn_map());
    write_map!("RAYPASS", "[u64; 64]", maps::generate_rays_map());
    write_map!("BETWEEN", "[u64; 64]", maps::generate_between_map());
    write_map!("ROOK_MAP", "u64", maps::generate_rook_map());
    write_map!("BISHOP_MAP", "u64", maps::generate_bishop_map());
    write_map!("ROOK_MAGICS", "MagicEntry", magics::ROOK_MAGICS);
    write_map!("BISHOP_MAGICS", "MagicEntry", magics::BISHOP_MAGICS);

    writeln!(buf, "struct MagicEntry {{ mask: u64, magic: u64, shift: u32, offset: u32 }}")?;
    buf.flush()
}
