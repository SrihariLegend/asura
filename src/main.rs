fn main() -> std::io::Result<()> {
    if std::env::var_os("ASURA_NNUE_PATH").is_none() && asura::nnue::global_model().is_none() {
        let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf()));
        let exe_nnue = exe_dir.as_ref().map(|d| d.join("asura_atn1.nnue"));
        let repo_nnue = exe_dir
            .as_ref()
            .and_then(|d| d.parent())
            .and_then(|d| d.parent())
            .map(|d| d.join("asura_atn1.nnue"));
        let candidates = [
            Some(std::path::PathBuf::from("asura_atn1.nnue")),
            exe_nnue,
            repo_nnue,
            Some(std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/asura_atn1.nnue"))),
        ];
        for path in candidates.into_iter().flatten() {
            if path.exists() && asura::nnue::load_global_model(&path).is_ok() {
                break;
            }
        }
    }
    asura::run(std::env::args().skip(1).collect())
}
