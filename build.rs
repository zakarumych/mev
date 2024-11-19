
fn main() {
    println!("cargo::rustc-check-cfg=cfg(mev_backend, values(\"metal, vulkan\"))");

    let windows = std::env::var_os("CARGO_CFG_WINDOWS").is_some();
    let unix = std::env::var_os("CARGO_CFG_UNIX").is_some();
    let macos = std::env::var_os("CARGO_CFG_TARGET_OS").map(|os| os == "macos").unwrap_or(false);
    let ios = std::env::var_os("CARGO_CFG_TARGET_OS").map(|os| os == "ios").unwrap_or(false);

    if windows || (unix && !(macos || ios)) {
        println!("cargo::rustc-cfg=mev_backend=\"vulkan\"");
    } else {
        println!("cargo::rustc-cfg=mev_backend=\"metal\"");
    }
}
