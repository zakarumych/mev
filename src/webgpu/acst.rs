use std::{fmt, marker::PhantomData};

use crate::generic::{BlasBuildDesc, BlasDesc, TlasBuildDesc, TlasDesc};

pub struct Blas {
    _phantom: PhantomData<()>,
}

impl fmt::Debug for Blas {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Blas").finish()
    }
}

impl crate::traits::Resource for Blas {}

pub struct Tlas {
    _phantom: PhantomData<()>,
}

impl fmt::Debug for Tlas {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Tlas").finish()
    }
}

impl crate::traits::Resource for Tlas {}
