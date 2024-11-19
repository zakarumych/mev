# MEV is a GAPI

that can use Metal et Vulkan GAPIs.

With ergonomics in mind, MEV is designed to be:

- 🪓 **Simple**: MEV is a no-nonsense GAPI keeping WTF per minute low.
- 🎯 **Flexible**: MEV is designed to be used in a variety of applications, exposing the full power of the underlying APIs.
- 🚀 **Fast**: MEV is designed to be fast, adding as little overhead as possible.
- 😵 **Safe-ish**: MEV is designed to be safe, but it's still possible to shoot yourself in the foot.

## Backends

MEV supports two backends: Metal and Vulkan.
It automatically picks one based on the platform it's built for.

On MacOS and iOS, MEV uses Metal.
On other platforms, MEV uses Vulkan 🌋
