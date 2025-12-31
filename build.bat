rem cargo build --release > nul 2>&1
set RUSTC_LOG=info
cargo build --release --jobs 1 > build.log 2>&1
