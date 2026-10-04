//!
//! Output archive (ozz `OArchive`), the counterpart of `Archive`.
//!

use std::io::{self, Write};

use crate::animation::{Animation, Float3Key, QuaternionKey};
use crate::base::OzzError;
use crate::math::SoaTransform;
use crate::skeleton::Skeleton;

///
/// Implements the output archive concept used to save/serialize data.
///
/// Writes native endianness and records it in the first byte, so `Archive` swaps on read when needed.
///
pub struct ArchiveWriter<W: Write> {
    write: W,
}

impl<W: Write> ArchiveWriter<W> {
    /// Creates an `ArchiveWriter`, writing the endianness tag.
    pub fn new(mut write: W) -> Result<ArchiveWriter<W>, OzzError> {
        let little_endian = cfg!(target_endian = "little") as u8;
        write.write_all(&[little_endian])?;
        Ok(ArchiveWriter { write })
    }

    /// Returns the underlying writer.
    pub fn into_inner(self) -> W {
        self.write
    }

    /// Writes `T` to the archive.
    #[inline]
    pub fn write<T: ArchiveWrite + ?Sized>(&mut self, value: &T) -> Result<(), OzzError> {
        value.write(self)
    }

    /// Writes `[T]` to the archive, without its length.
    #[inline]
    pub fn write_slice<T: ArchiveWrite>(&mut self, values: &[T]) -> Result<(), OzzError> {
        for value in values {
            value.write(self)?;
        }
        Ok(())
    }

    /// Writes an object tag (zero terminated).
    #[inline]
    pub fn write_tag(&mut self, tag: &str) -> Result<(), OzzError> {
        self.write_bytes(tag.as_bytes())?;
        self.write_bytes(&[0])
    }

    /// Writes an object version.
    #[inline]
    pub fn write_version(&mut self, version: u32) -> Result<(), OzzError> {
        self.write(&version)
    }

    /// Writes raw bytes.
    #[inline]
    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), OzzError> {
        self.write.write_all(bytes).map_err(OzzError::from)
    }
}

impl ArchiveWriter<Vec<u8>> {
    /// Creates an `ArchiveWriter` into a new `Vec<u8>`.
    pub fn new_vec() -> ArchiveWriter<Vec<u8>> {
        ArchiveWriter::new(Vec::new()).expect("writing to a Vec doesn't fail")
    }
}

/// Implements `ArchiveWrite` to write `Self` to an `ArchiveWriter`.
pub trait ArchiveWrite {
    /// Writes `self` to the archive.
    fn write<W: Write>(&self, archive: &mut ArchiveWriter<W>) -> Result<(), OzzError>;
}

macro_rules! primitive_writer {
    ($type:ty) => {
        impl ArchiveWrite for $type {
            #[inline]
            fn write<W: Write>(&self, archive: &mut ArchiveWriter<W>) -> Result<(), OzzError> {
                archive.write_bytes(&self.to_ne_bytes())
            }
        }
    };
}

primitive_writer!(u8);
primitive_writer!(i8);
primitive_writer!(u16);
primitive_writer!(i16);
primitive_writer!(u32);
primitive_writer!(i32);
primitive_writer!(u64);
primitive_writer!(i64);
primitive_writer!(f32);
primitive_writer!(f64);

impl ArchiveWrite for bool {
    #[inline]
    fn write<W: Write>(&self, archive: &mut ArchiveWriter<W>) -> Result<(), OzzError> {
        archive.write_bytes(&[*self as u8])
    }
}

/// Zero-terminated string, as `Archive` reads `String`.
impl ArchiveWrite for str {
    #[inline]
    fn write<W: Write>(&self, archive: &mut ArchiveWriter<W>) -> Result<(), OzzError> {
        if self.as_bytes().contains(&0) {
            return Err(OzzError::IO(io::ErrorKind::InvalidInput));
        }
        archive.write_tag(self)
    }
}

impl ArchiveWrite for Float3Key {
    #[inline]
    fn write<W: Write>(&self, archive: &mut ArchiveWriter<W>) -> Result<(), OzzError> {
        archive.write_slice(&self.0)
    }
}

impl ArchiveWrite for QuaternionKey {
    #[inline]
    fn write<W: Write>(&self, archive: &mut ArchiveWriter<W>) -> Result<(), OzzError> {
        archive.write_slice(&self.0)
    }
}

impl ArchiveWrite for SoaTransform {
    fn write<W: Write>(&self, archive: &mut ArchiveWriter<W>) -> Result<(), OzzError> {
        let t = &self.translation;
        let r = &self.rotation;
        let s = &self.scale;
        for v in [t.x, t.y, t.z, r.x, r.y, r.z, r.w, s.x, s.y, s.z] {
            archive.write_slice(&v.to_array())?;
        }
        Ok(())
    }
}

/// "ozz-skeleton" v2: joint count, names, parents, rest poses.
impl ArchiveWrite for Skeleton {
    fn write<W: Write>(&self, archive: &mut ArchiveWriter<W>) -> Result<(), OzzError> {
        archive.write_tag(Skeleton::tag())?;
        archive.write_version(Skeleton::version())?;

        let num_joints = self.num_joints();
        archive.write(&(num_joints as u32))?;
        if num_joints == 0 {
            return Ok(());
        }

        let mut names = Vec::with_capacity(num_joints);
        for joint in 0..num_joints {
            let name = self.name_by_joint(joint as i16).ok_or(OzzError::InvalidIndex)?;
            names.push(name);
        }
        let chars_count: usize = names.iter().map(|name| name.len() + 1).sum();
        archive.write(&(chars_count as u32))?;
        for name in names {
            archive.write(name)?;
        }
        archive.write_slice(self.joint_parents())?;
        archive.write_slice(self.joint_rest_poses())
    }
}

/// "ozz-animation" v7: counts, name, time points, then per component (translation, rotation, scale)
/// ratios (`u8` when there are at most 255 time points), previouses, iframes and values.
impl ArchiveWrite for Animation {
    fn write<W: Write>(&self, archive: &mut ArchiveWriter<W>) -> Result<(), OzzError> {
        archive.write_tag(Animation::tag())?;
        archive.write_version(Animation::version())?;

        let t = self.translations_ctrl();
        let r = self.rotations_ctrl();
        let s = self.scales_ctrl();

        archive.write(&self.duration())?;
        archive.write(&(self.num_tracks() as u32))?;
        archive.write(&(self.name().len() as u32))?;
        archive.write(&(self.timepoints().len() as u32))?;
        archive.write(&(self.translations().len() as u32))?;
        archive.write(&(self.rotations().len() as u32))?;
        archive.write(&(self.scales().len() as u32))?;
        archive.write(&(t.iframe_entries.len() as u32))?;
        archive.write(&(t.iframe_desc.len() as u32))?;
        archive.write(&(r.iframe_entries.len() as u32))?;
        archive.write(&(r.iframe_desc.len() as u32))?;
        archive.write(&(s.iframe_entries.len() as u32))?;
        archive.write(&(s.iframe_desc.len() as u32))?;

        archive.write_bytes(self.name().as_bytes())?;
        archive.write_slice(self.timepoints())?;

        let ratio_u8 = self.timepoints().len() <= u8::MAX as usize;
        for (ctrl, keys) in [
            (t, Keys::Float3(self.translations())),
            (r, Keys::Quaternion(self.rotations())),
            (s, Keys::Float3(self.scales())),
        ] {
            if ratio_u8 {
                for ratio in ctrl.ratios {
                    archive.write(&(*ratio as u8))?;
                }
            } else {
                archive.write_slice(ctrl.ratios)?;
            }
            archive.write_slice(ctrl.previouses)?;
            archive.write_bytes(ctrl.iframe_entries)?;
            archive.write_slice(ctrl.iframe_desc)?;
            archive.write(&ctrl.iframe_interval)?;
            match keys {
                Keys::Float3(keys) => archive.write_slice(keys)?,
                Keys::Quaternion(keys) => archive.write_slice(keys)?,
            }
        }
        Ok(())
    }
}

enum Keys<'t> {
    Float3(&'t [Float3Key]),
    Quaternion(&'t [QuaternionKey]),
}

impl Skeleton {
    /// Writes an "ozz-skeleton" archive (endianness tag included) to bytes.
    pub fn to_archive_bytes(&self) -> Result<Vec<u8>, OzzError> {
        let mut archive = ArchiveWriter::new_vec();
        archive.write(self)?;
        Ok(archive.into_inner())
    }
}

impl Animation {
    /// Writes an "ozz-animation" archive (endianness tag included) to bytes.
    pub fn to_archive_bytes(&self) -> Result<Vec<u8>, OzzError> {
        let mut archive = ArchiveWriter::new_vec();
        archive.write(self)?;
        Ok(archive.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archive::Archive;

    fn assert_animation_eq(a: &Animation, b: &Animation) {
        assert_eq!(a.duration(), b.duration());
        assert_eq!(a.num_tracks(), b.num_tracks());
        assert_eq!(a.name(), b.name());
        assert_eq!(a.timepoints(), b.timepoints());
        assert_eq!(a.translations(), b.translations());
        assert_eq!(a.rotations(), b.rotations());
        assert_eq!(a.scales(), b.scales());
        for (x, y) in [
            (a.translations_ctrl(), b.translations_ctrl()),
            (a.rotations_ctrl(), b.rotations_ctrl()),
            (a.scales_ctrl(), b.scales_ctrl()),
        ] {
            assert_eq!(x.ratios, y.ratios);
            assert_eq!(x.previouses, y.previouses);
            assert_eq!(x.iframe_entries, y.iframe_entries);
            assert_eq!(x.iframe_desc, y.iframe_desc);
            assert_eq!(x.iframe_interval, y.iframe_interval);
        }
    }

    #[test]
    fn test_rewrite_skeleton_bytes() {
        let bytes = std::fs::read("./resource/playback/skeleton.ozz").unwrap();
        let skeleton = Skeleton::from_archive(&mut Archive::from_slice(&bytes).unwrap()).unwrap();
        assert_eq!(skeleton.to_archive_bytes().unwrap(), bytes);
    }

    #[test]
    fn test_rewrite_animation_bytes() {
        for path in [
            "./resource/playback/animation.ozz",
            "./resource/blend/animation1.ozz",
            "./resource/additive/animation_base.ozz",
        ] {
            let bytes = std::fs::read(path).unwrap();
            let animation = Animation::from_archive(&mut Archive::from_slice(&bytes).unwrap()).unwrap();
            let written = animation.to_archive_bytes().unwrap();
            assert_eq!(written, bytes, "{}", path);
            let reread = Animation::from_archive(&mut Archive::from_slice(&written).unwrap()).unwrap();
            assert_animation_eq(&animation, &reread);
        }
    }

    #[test]
    fn test_string_with_nul_rejected() {
        let mut archive = ArchiveWriter::new_vec();
        assert!(archive.write("a\0b").is_err());
    }
}
