fn main() -> std::io::Result<()> {
    asura::run(std::env::args().skip(1).collect())
}
