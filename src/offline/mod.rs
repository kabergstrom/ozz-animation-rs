//!
//! Offline data building, ported from ozz-animation 0.16's offline library (feature `offline`).
//!
//! - `RawSkeleton` + `SkeletonBuilder` build a runtime `Skeleton`.
//! - `RawAnimation` + `AnimationBuilder` build a runtime `Animation`.
//! - `AnimationOptimizer` removes keyframes within a hierarchical tolerance.
//! - `ArchiveWriter` writes "ozz-skeleton" v2 and "ozz-animation" v7 archives, the versions `Archive` reads.
//! - `gltf` (feature `gltf`) converts glTF nodes and animation channels to raw data, as gltf2ozz does.
//!
//! Builders allocate freely: they run at import time, never in the runtime path.
//!

mod animation_builder;
mod animation_optimizer;
mod archive_writer;
#[cfg(feature = "gltf")]
pub mod gltf;
mod raw_animation;
mod raw_skeleton;

pub use animation_builder::AnimationBuilder;
pub use animation_optimizer::{AnimationOptimizer, OptimizerSetting};
pub use archive_writer::{ArchiveWrite, ArchiveWriter};
pub use raw_animation::{
    lerp_rotation, lerp_scale, lerp_translation, RawAnimation, RawJointTrack, RawRotationKey, RawScaleKey,
    RawTranslationKey,
};
pub use raw_skeleton::{RawJoint, RawSkeleton, SkeletonBuilder};

use glam::{Quat, Vec3};
use thiserror::Error;

/// Offline building error.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum OfflineError {
    /// More joints than `SKELETON_MAX_JOINTS`.
    #[error("Too many joints: {0}")]
    TooManyJoints(usize),
    /// Skeleton has no joint.
    #[error("Empty skeleton")]
    EmptySkeleton,
    /// Two joints share a name. The runtime `Skeleton` maps names to joints one to one.
    #[error("Duplicate joint name: {0}")]
    DuplicateJointName(String),
    /// Animation duration is not greater than 0.
    #[error("Invalid duration: {0}")]
    InvalidDuration(f32),
    /// More tracks than `SKELETON_MAX_JOINTS`.
    #[error("Too many tracks: {0}")]
    TooManyTracks(usize),
    /// Keyframe times of a track are not strictly ascending or not within `[0, duration]`.
    #[error("Invalid keyframes on track {0}")]
    InvalidKeyframes(usize),
    /// More distinct keyframe times than a `u16` can index.
    #[error("Too many time points: {0}")]
    TooManyTimepoints(usize),
    /// Animation track count differs from skeleton joint count.
    #[error("Track count {tracks} differs from joint count {joints}")]
    TrackCountMismatch { tracks: usize, joints: usize },
}

// Scalar math matching ozz's offline math (base/maths), operation for operation, so that keyframe
// decisions (optimizer tolerance tests, quaternion fix-ups) match the C++ builders.

#[inline]
pub(crate) fn lerp3(a: Vec3, b: Vec3, alpha: f32) -> Vec3 {
    Vec3::new(
        (b.x - a.x) * alpha + a.x,
        (b.y - a.y) * alpha + a.y,
        (b.z - a.z) * alpha + a.z,
    )
}

#[inline]
pub(crate) fn length3(v: Vec3) -> f32 {
    (v.x * v.x + v.y * v.y + v.z * v.z).sqrt()
}

#[inline]
pub(crate) fn length_sqr3(v: Vec3) -> f32 {
    v.x * v.x + v.y * v.y + v.z * v.z
}

#[inline]
pub(crate) fn dot4(a: Quat, b: Quat) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
}

#[inline]
pub(crate) fn normalize_quat(q: Quat) -> Quat {
    let inv_len = 1.0 / dot4(q, q).sqrt();
    Quat::from_xyzw(q.x * inv_len, q.y * inv_len, q.z * inv_len, q.w * inv_len)
}

#[inline]
pub(crate) fn normalize_safe_quat(q: Quat, safer: Quat) -> Quat {
    if dot4(q, q) == 0.0 {
        return safer;
    }
    normalize_quat(q)
}

#[inline]
pub(crate) fn nlerp(a: Quat, b: Quat, alpha: f32) -> Quat {
    normalize_quat(Quat::from_xyzw(
        (b.x - a.x) * alpha + a.x,
        (b.y - a.y) * alpha + a.y,
        (b.z - a.z) * alpha + a.z,
        (b.w - a.w) * alpha + a.w,
    ))
}

/// ozz's `FloatToHalf` (reference implementation): round to nearest, overflow clamps to infinity.
pub(crate) fn f32_to_f16(f: f32) -> u16 {
    const F32_INFTY: u32 = 255 << 23;
    const F16_INFTY: u32 = 31 << 23;
    const MAGIC: u32 = 15 << 23;
    const SIGN_MASK: u32 = 0x8000_0000;
    const ROUND_MASK: u32 = !0x0fff;

    let bits = f.to_bits();
    let sign = bits & SIGN_MASK;
    let f_nosign = bits & !SIGN_MASK;
    if f_nosign >= F32_INFTY {
        // NaN -> qNaN, Inf -> Inf.
        let result = (if f_nosign > F32_INFTY { 0x7e00 } else { 0x7c00 }) | (sign >> 16);
        result as u16
    } else {
        let rounded = f32::from_bits(f_nosign & ROUND_MASK);
        let exp = (rounded * f32::from_bits(MAGIC)).to_bits();
        let re_rounded = exp.wrapping_sub(ROUND_MASK);
        let clamped = if re_rounded > F16_INFTY { F16_INFTY } else { re_rounded };
        ((clamped >> 13) | (sign >> 16)) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::f16_to_f32;

    #[test]
    fn test_f32_to_f16() {
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-1.0), 0xbc00);
        assert_eq!(f32_to_f16(3.5), 0x4300);
        assert_eq!(f32_to_f16(0.0), 0);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(f32_to_f16(1e6), 0x7c00);
        assert_eq!(f32_to_f16(f32::NAN) & 0x7e00, 0x7e00);
        for v in [0.1f32, -0.25, 1.37, 100.5, 6.1e-5, -3.0e-6, 65504.0] {
            let back = f16_to_f32(f32_to_f16(v));
            assert!((back - v).abs() <= v.abs() * 1e-3 + 6e-8, "{} -> {}", v, back);
        }
    }
}
