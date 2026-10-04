//!
//! Offline animation (ozz `RawAnimation`).
//!

use glam::{Quat, Vec3};

use super::{dot4, lerp3, nlerp, OfflineError};
use crate::base::SKELETON_MAX_JOINTS;

/// Raw translation keyframe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawTranslationKey {
    pub time: f32,
    pub value: Vec3,
}

/// Raw rotation keyframe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawRotationKey {
    pub time: f32,
    pub value: Quat,
}

/// Raw scale keyframe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawScaleKey {
    pub time: f32,
    pub value: Vec3,
}

/// Keyframe operations shared by the builder, the optimizer and the glTF import.
pub(crate) trait RawKey: Copy {
    type Value: Copy;
    fn new(time: f32, value: Self::Value) -> Self;
    fn time(&self) -> f32;
    fn value(&self) -> Self::Value;
    /// Value of a track without keys.
    fn identity() -> Self::Value;
    /// Interpolation matching the runtime sampling job.
    fn lerp(a: Self::Value, b: Self::Value, alpha: f32) -> Self::Value;
}

macro_rules! raw_key {
    ($key:ty, $value:ty, $identity:expr, $lerp:path) => {
        impl RawKey for $key {
            type Value = $value;
            #[inline]
            fn new(time: f32, value: $value) -> Self {
                Self { time, value }
            }
            #[inline]
            fn time(&self) -> f32 {
                self.time
            }
            #[inline]
            fn value(&self) -> $value {
                self.value
            }
            #[inline]
            fn identity() -> $value {
                $identity
            }
            #[inline]
            fn lerp(a: $value, b: $value, alpha: f32) -> $value {
                $lerp(a, b, alpha)
            }
        }
    };
}

raw_key!(RawTranslationKey, Vec3, Vec3::ZERO, lerp_translation);
raw_key!(RawRotationKey, Quat, Quat::IDENTITY, lerp_rotation);
raw_key!(RawScaleKey, Vec3, Vec3::ONE, lerp_scale);

/// Translation interpolation, as the sampling job does it.
#[inline]
pub fn lerp_translation(a: Vec3, b: Vec3, alpha: f32) -> Vec3 {
    lerp3(a, b, alpha)
}

/// Rotation interpolation along the shortest path. The runtime gets the shortest path from
/// `AnimationBuilder`'s quaternion fix-up; offline code takes it here.
#[inline]
pub fn lerp_rotation(a: Quat, b: Quat, alpha: f32) -> Quat {
    let b = if dot4(a, b) < 0.0 { -b } else { b };
    nlerp(a, b, alpha)
}

/// Scale interpolation, as the sampling job does it.
#[inline]
pub fn lerp_scale(a: Vec3, b: Vec3, alpha: f32) -> Vec3 {
    lerp3(a, b, alpha)
}

/// Keyframes of one joint. Each component is independent; an empty component is identity.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RawJointTrack {
    pub translations: Vec<RawTranslationKey>,
    pub rotations: Vec<RawRotationKey>,
    pub scales: Vec<RawScaleKey>,
}

impl RawJointTrack {
    /// Keyframe times strictly ascending and within `[0, duration]`.
    /// Use an infinite `duration` to check only the order.
    pub fn validate(&self, duration: f32) -> bool {
        validate_keys(&self.translations, duration)
            && validate_keys(&self.rotations, duration)
            && validate_keys(&self.scales, duration)
    }
}

fn validate_keys<K: RawKey>(keys: &[K], duration: f32) -> bool {
    let mut previous_time = -1.0;
    for key in keys {
        let time = key.time();
        if !(0.0..=duration).contains(&time) || time <= previous_time {
            return false;
        }
        previous_time = time;
    }
    true
}

///
/// Offline animation: per-joint tracks of translation, rotation and scale keyframes.
/// `AnimationBuilder` converts it to the runtime `Animation`.
///
/// Valid when:
/// - `duration` is greater than 0;
/// - there are at most `SKELETON_MAX_JOINTS` tracks;
/// - keyframe times of each component are strictly ascending and within `[0, duration]`.
///
#[derive(Debug, Clone, PartialEq)]
pub struct RawAnimation {
    /// Per-joint tracks, in skeleton joint order.
    pub tracks: Vec<RawJointTrack>,
    /// Duration in seconds.
    pub duration: f32,
    /// Animation name.
    pub name: String,
}

impl Default for RawAnimation {
    /// An empty animation with a 1 s duration, as ozz's default.
    fn default() -> RawAnimation {
        RawAnimation {
            tracks: Vec::new(),
            duration: 1.0,
            name: String::new(),
        }
    }
}

impl RawAnimation {
    /// Number of tracks.
    #[inline]
    pub fn num_tracks(&self) -> usize {
        self.tracks.len()
    }

    /// Tests validity, see `RawAnimation`.
    pub fn validate(&self) -> Result<(), OfflineError> {
        if !(self.duration > 0.0) {
            return Err(OfflineError::InvalidDuration(self.duration));
        }
        if self.tracks.len() > SKELETON_MAX_JOINTS as usize {
            return Err(OfflineError::TooManyTracks(self.tracks.len()));
        }
        for (index, track) in self.tracks.iter().enumerate() {
            if !track.validate(self.duration) {
                return Err(OfflineError::InvalidKeyframes(index));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(time: f32) -> RawTranslationKey {
        RawTranslationKey {
            time,
            value: Vec3::ZERO,
        }
    }

    #[test]
    fn test_validate() {
        let mut raw = RawAnimation::default();
        assert_eq!(raw.validate(), Ok(()));

        raw.duration = 0.0;
        assert_eq!(raw.validate(), Err(OfflineError::InvalidDuration(0.0)));
        raw.duration = 1.0;

        raw.tracks.push(RawJointTrack {
            translations: vec![t(0.0), t(0.5), t(1.0)],
            ..Default::default()
        });
        assert_eq!(raw.validate(), Ok(()));

        raw.tracks.push(RawJointTrack {
            translations: vec![t(0.5), t(0.5)],
            ..Default::default()
        });
        assert_eq!(raw.validate(), Err(OfflineError::InvalidKeyframes(1)));

        raw.tracks[1].translations = vec![t(0.5), t(1.5)];
        assert_eq!(raw.validate(), Err(OfflineError::InvalidKeyframes(1)));

        raw.tracks[1].translations = vec![t(-0.1)];
        assert_eq!(raw.validate(), Err(OfflineError::InvalidKeyframes(1)));

        raw.tracks = vec![RawJointTrack::default(); SKELETON_MAX_JOINTS as usize + 1];
        assert!(matches!(raw.validate(), Err(OfflineError::TooManyTracks(_))));
    }

    #[test]
    fn test_lerp_rotation_shortest_path() {
        let a = Quat::IDENTITY;
        let b = -Quat::from_rotation_z(0.5);
        let mid = lerp_rotation(a, b, 0.5);
        assert!(mid.abs_diff_eq(Quat::from_rotation_z(0.25), 1e-6));
    }
}
