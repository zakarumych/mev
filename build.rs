fn main() {
    println!("cargo::rustc-check-cfg=cfg(mev_backend, values(\"metal, vulkan, webgpu\"))");

    // println!("cargo::rustc-cfg=mev_backend=\"webgpu\"");
    // return;

    let windows = std::env::var_os("CARGO_CFG_WINDOWS").is_some();
    let unix = std::env::var_os("CARGO_CFG_UNIX").is_some();
    let macos = std::env::var_os("CARGO_CFG_TARGET_OS").map_or(false, |os| os == "macos");
    let ios = std::env::var_os("CARGO_CFG_TARGET_OS").map_or(false, |os| os == "ios");
    let wasm32 = std::env::var("CARGO_CFG_TARGET_ARCH").map_or(false, |os| os == "wasm32");

    if wasm32 {
        println!("cargo::rustc-cfg=mev_backend=\"webgpu\"");
        eprintln!("mev selects WebGPU");
    } else if macos || ios {
        println!("cargo::rustc-cfg=mev_backend=\"metal\"");
        eprintln!("mev selects Metal");
    } else if windows || unix {
        println!("cargo::rustc-cfg=mev_backend=\"vulkan\"");
        eprintln!("mev selects Vulkan");
    } else {
        panic!("Only Windows, macOS, iOS, and Unix are currently supported");
    }
}
