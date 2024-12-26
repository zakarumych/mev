use std::{borrow::Cow, error::Error, fmt};

use codespan_reporting::{
    diagnostic::{Diagnostic, Label},
    files::SimpleFile,
    term::{self, termcolor::Buffer},
};
use naga::FastHashMap;

use crate::{backend::Library, generic::OutOfMemory};

/// Shader stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    /// Vertex shader stage.
    Vertex,

    /// Fragment shader stage.
    Fragment,

    /// Compute shader stage.
    Compute,
}

impl fmt::Display for ShaderStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShaderStage::Vertex => write!(f, "vertex"),
            ShaderStage::Fragment => write!(f, "fragment"),
            ShaderStage::Compute => write!(f, "compute"),
        }
    }
}

bitflags::bitflags! {
    /// Flags that describe the shader stages.
    ///
    /// Each flag corresponds to one [`ShaderStage`].
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct ShaderStages : u32 {
        /// Bit for [`Vertex`](ShaderStage::Vertex) stage.
        const VERTEX = 1 << ShaderStage::Vertex as u32;
        /// Bit for [`Fragment`](ShaderStage::Fragment) stage.
        const FRAGMENT = 1 << ShaderStage::Fragment as u32;
        /// Bit for [`Compute`](ShaderStage::Compute) stage.
        const COMPUTE = 1 << ShaderStage::Compute as u32;
    }
}

/// Shader language.
///
/// Must be specified when loading shader source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderLanguage {
    /// SPIR-V.
    /// This is native format for Vulkan.
    /// Shader source code will be compiled to native format on other backends.
    SpirV,

    /// WGSL (WebGPU Shading Language).
    /// This is native format for WebGPU.
    /// Shader source code will be compiled to native format.
    Wgsl,

    /// GLSL (OpenGL Shading Language).
    /// Shader source code will be compiled to native format.
    ///
    /// Requires specifying shader stage.
    Glsl { stage: ShaderStage },

    /// MSL (Metal Shading Language).
    /// This is native format for Metal.
    /// Shader source code will be compiled to native format on other backends.
    Msl,
}

impl fmt::Display for ShaderLanguage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShaderLanguage::SpirV => write!(f, "SPIR-V"),
            ShaderLanguage::Wgsl => write!(f, "WGSL"),
            ShaderLanguage::Glsl { stage } => write!(f, "GLSL {}", stage),
            ShaderLanguage::Msl => write!(f, "MSL"),
        }
    }
}

/// Describes shader source.
#[derive(Clone, Debug, PartialEq, Hash)]
pub struct ShaderSource<'a> {
    /// Code of the shader.
    pub code: Cow<'a, [u8]>,

    /// Optional filename of the shader.
    pub filename: Option<&'a str>,

    /// Language of the shader code.
    pub language: ShaderLanguage,
}

/// Convenience macro to include shader source code from a file during compilation.
#[macro_export]
macro_rules! include_shader_source {
    ($filename:literal as $lang:expr) => {
        $crate::for_macro::ShaderSource {
            code: std::borrow::Cow::Borrowed(std::include_bytes!($filename)),
            filename: std::option::Option::Some($filename),
            language: $lang,
        }
    };
}

/// Input for the library.
/// Currently only source code is supported.
#[derive(Clone, Debug, PartialEq, Hash)]
pub enum LibraryInput<'a> {
    /// Shader source code.
    Source(ShaderSource<'a>),
}

/// Convenience macro to include shader library input from a source code file during compilation.
#[macro_export]
macro_rules! include_library {
    ($filename:literal as $lang:expr) => {
        $crate::for_macro::LibraryInput::Source($crate::include_shader_source!($filename as $lang))
    };
}

/// Describes shader library.
#[derive(Clone, Debug, PartialEq, Hash)]
pub struct LibraryDesc<'a> {
    /// Name of the library.
    pub name: &'a str,

    /// Input for the library.
    pub input: LibraryInput<'a>,
}

/// Shader from the library.
#[derive(Clone)]
pub struct Shader<'a> {
    /// Library that contains the shader.
    pub library: Library,

    /// Shader entry point.
    pub entry: Cow<'a, str>,
}

/// Error that can occur during library creation.
#[derive(Debug)]
pub enum CreateLibraryError {
    /// Out of memory.
    OutOfMemory,

    /// Shader compilation error.
    CompileError(ShaderCompileError),
}

impl From<OutOfMemory> for CreateLibraryError {
    #[inline(always)]
    fn from(_: OutOfMemory) -> Self {
        CreateLibraryError::OutOfMemory
    }
}

impl From<ShaderCompileError> for CreateLibraryError {
    #[inline(always)]
    fn from(err: ShaderCompileError) -> Self {
        CreateLibraryError::CompileError(err)
    }
}

impl fmt::Display for CreateLibraryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CreateLibraryError::OutOfMemory => fmt::Display::fmt(&OutOfMemory, f),
            CreateLibraryError::CompileError(err) => fmt::Display::fmt(err, f),
        }
    }
}

impl Error for CreateLibraryError {}

#[derive(Debug)]
pub(crate) enum ShaderCompileError {
    NonUtf8(std::str::Utf8Error),
    ParseSpirV(naga::front::spv::Error),
    ParseWgsl(naga::front::wgsl::ParseError),
    ParseGlsl(naga::front::glsl::ParseErrors),
    ValidationFailed,

    #[cfg(any(windows, all(unix, not(any(target_os = "macos", target_os = "ios")))))]
    GenSpirV(naga::back::spv::Error),

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    GenMsl(naga::back::msl::Error),
}

impl fmt::Display for ShaderCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShaderCompileError::NonUtf8(err) => write!(f, "non-utf8: {}", err),
            ShaderCompileError::ParseSpirV(err) => write!(f, "parse SPIR-V: {}", err),
            ShaderCompileError::ParseWgsl(err) => write!(f, "parse WGSL: {}", err),
            ShaderCompileError::ParseGlsl(err) => write!(f, "parse GLSL: {}", err),
            ShaderCompileError::ValidationFailed => write!(f, "validation failed"),
            #[cfg(any(windows, all(unix, not(any(target_os = "macos", target_os = "ios")))))]
            ShaderCompileError::GenSpirV(err) => write!(f, "generate SPIR-V: {}", err),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            ShaderCompileError::GenMsl(err) => write!(f, "generate MSL: {}", err),
        }
    }
}

pub(crate) fn parse_shader<'a>(
    code: &'a [u8],
    filename: Option<&str>,
    lang: ShaderLanguage,
) -> Result<(naga::Module, naga::valid::ModuleInfo, Option<&'a str>), ShaderCompileError> {
    let mut source_code = None;
    let module = match lang {
        ShaderLanguage::SpirV => {
            naga::front::spv::parse_u8_slice(code, &naga::front::spv::Options::default())
                .map_err(ShaderCompileError::ParseSpirV)?
        }
        ShaderLanguage::Msl => {
            unimplemented!("Compilation from MSL is not supported")
        }
        ShaderLanguage::Wgsl => {
            let code = std::str::from_utf8(code).map_err(ShaderCompileError::NonUtf8)?;
            source_code = Some(code);
            naga::front::wgsl::parse_str(code).map_err(ShaderCompileError::ParseWgsl)?
        }
        ShaderLanguage::Glsl { stage } => {
            let code = std::str::from_utf8(code).map_err(ShaderCompileError::NonUtf8)?;
            source_code = Some(code);
            naga::front::glsl::Frontend::default()
                .parse(
                    &naga::front::glsl::Options {
                        defines: FastHashMap::default(),
                        stage: match stage {
                            ShaderStage::Vertex => naga::ShaderStage::Vertex,
                            ShaderStage::Fragment => naga::ShaderStage::Fragment,
                            ShaderStage::Compute => naga::ShaderStage::Compute,
                        },
                    },
                    code,
                )
                .map_err(ShaderCompileError::ParseGlsl)?
        }
    };

    let flags = naga::valid::ValidationFlags::all();
    let caps = naga::valid::Capabilities::all();
    let info = naga::valid::Validator::new(flags, caps)
        .validate(&module)
        .map_err(|e| {
            emit_annotated_error(
                &e,
                filename.and_then(|filename| {
                    std::str::from_utf8(code)
                        .ok()
                        .map(|source| (filename, source))
                }),
            );
            ShaderCompileError::ValidationFailed
        })?;

    Ok((module, info, source_code))
}

fn emit_annotated_error<E: std::error::Error>(
    error: &naga::WithSpan<E>,
    file: Option<(&str, &str)>,
) {
    if let Some((filename, source)) = file {
        let files = SimpleFile::new(filename, source);
        let config = term::Config::default();
        let mut writer = Buffer::no_color();

        let diagnostic = Diagnostic::error().with_labels(
            error
                .spans()
                .map(|(span, desc)| {
                    Label::primary((), span.to_range().unwrap()).with_message(desc.to_owned())
                })
                .collect(),
        );

        term::emit(&mut writer, &config, &files, &diagnostic).expect("cannot write error");

        if let Ok(s) = std::str::from_utf8(writer.as_slice()) {
            tracing::event!(
                target: "naga",
                tracing::Level::ERROR,
                error = error.as_inner().to_string(),
                diagnostic = s,
            );
            return;
        }
    }

    tracing::event!(
        target: "naga",
        tracing::Level::ERROR,
        error = error.as_inner().to_string(),
    );
}
