//!
//! Keyframe reduction (ozz `AnimationOptimizer`).
//!

use std::collections::BTreeMap;

use glam::{Quat, Vec3};

use super::raw_animation::RawKey;
use super::{dot4, length3, length_sqr3, OfflineError, RawAnimation, RawJointTrack};
use crate::base::SKELETON_NO_PARENT;
use crate::skeleton::Skeleton;

/// Optimization setting of a joint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimizerSetting {
    /// Maximum error, in skeleton units (meters for metric data), an optimization may introduce on a
    /// whole joint hierarchy.
    pub tolerance: f32,
    /// Distance from the joint, in skeleton units, at which the error is measured (if larger than the joint
    /// hierarchy). Emulates the effect on skinning.
    pub distance: f32,
}

impl Default for OptimizerSetting {
    /// 1 mm tolerance at 10 cm, ozz's defaults.
    fn default() -> OptimizerSetting {
        OptimizerSetting {
            tolerance: 1e-3,
            distance: 1e-1,
        }
    }
}

///
/// Removes keyframes that linear interpolation of their neighbours reproduces within tolerance
/// (Ramer-Douglas-Peucker per track component).
///
/// Tolerance is hierarchical: a joint's rotation and scale error is measured at the length of its
/// whole child hierarchy (scaled by accumulated parent scales), and a joint gets the smallest
/// tolerance of its descendants.
///
#[derive(Debug, Clone, Default)]
pub struct AnimationOptimizer {
    /// Setting for all joints without an override.
    pub setting: OptimizerSetting,
    /// Per-joint overrides, by skeleton joint index.
    pub joints_setting_override: BTreeMap<usize, OptimizerSetting>,
}

#[derive(Debug, Clone, Copy, Default)]
struct Spec {
    /// Length of the joint hierarchy (max of all children).
    length: f32,
    /// Scale of the joint hierarchy (accumulated from all parents).
    scale: f32,
    /// Tolerance of the joint hierarchy (min of all children).
    tolerance: f32,
}

impl AnimationOptimizer {
    /// An optimizer with ozz's default distance and the given tolerance (`optimize_tolerance_m` in metric data).
    pub fn with_tolerance(tolerance: f32) -> AnimationOptimizer {
        AnimationOptimizer {
            setting: OptimizerSetting {
                tolerance,
                ..Default::default()
            },
            joints_setting_override: BTreeMap::new(),
        }
    }

    fn joint_setting(&self, joint: usize) -> OptimizerSetting {
        self.joints_setting_override
            .get(&joint)
            .copied()
            .unwrap_or(self.setting)
    }

    fn hierarchy_specs(&self, input: &RawAnimation, skeleton: &Skeleton) -> Vec<Spec> {
        let mut specs = vec![Spec::default(); input.num_tracks()];

        // Scales, root to leaf.
        skeleton.iter_depth_first(-1, |joint, parent| {
            let joint = joint as usize;
            let track = &input.tracks[joint];
            let max_scale = if track.scales.is_empty() {
                1.0
            } else {
                track.scales.iter().fold(0.0f32, |max, key| {
                    let v = key.value;
                    f32::max(max, f32::max(f32::max(v.x.abs(), v.y.abs()), v.z.abs()))
                })
            };
            let mut scale = max_scale;
            if parent as i32 != SKELETON_NO_PARENT {
                scale *= specs[parent as usize].scale;
            }
            let setting = self.joint_setting(joint);
            specs[joint] = Spec {
                length: setting.distance * scale,
                scale,
                tolerance: setting.tolerance,
            };
        });

        // Lengths and tolerances, leaf to root.
        skeleton.iter_depth_first_reverse(|joint, parent| {
            if parent as i32 == SKELETON_NO_PARENT {
                return;
            }
            let track = &input.tracks[joint as usize];
            let max_length_sq = track
                .translations
                .iter()
                .fold(0.0f32, |max, key| f32::max(max, length_sqr3(key.value)));
            let max_length = max_length_sq.sqrt();

            let joint_spec = specs[joint as usize];
            let parent_spec = &mut specs[parent as usize];
            parent_spec.length = f32::max(parent_spec.length, joint_spec.length + max_length * parent_spec.scale);
            parent_spec.tolerance = f32::min(parent_spec.tolerance, joint_spec.tolerance);
        });

        specs
    }

    /// Optimizes `input`, whose tracks follow `skeleton`'s joints.
    ///
    /// Fails if `input` doesn't validate or its track count differs from the joint count.
    pub fn optimize(&self, input: &RawAnimation, skeleton: &Skeleton) -> Result<RawAnimation, OfflineError> {
        input.validate()?;
        if input.num_tracks() != skeleton.num_joints() {
            return Err(OfflineError::TrackCountMismatch {
                tracks: input.num_tracks(),
                joints: skeleton.num_joints(),
            });
        }

        let specs = self.hierarchy_specs(input, skeleton);

        let tracks = input
            .tracks
            .iter()
            .enumerate()
            .map(|(i, track)| {
                let spec = specs[i];
                let parent = skeleton.joint_parent(i);
                let parent_scale = if parent as i32 != SKELETON_NO_PARENT {
                    specs[parent as usize].scale
                } else {
                    1.0
                };
                RawJointTrack {
                    // Translation error scales with parent scale.
                    translations: decimate(&track.translations, spec.tolerance, |a: Vec3, b: Vec3| {
                        length3(a - b) * parent_scale
                    }),
                    // Rotation and scale errors move children: measured at the hierarchy length.
                    rotations: decimate(&track.rotations, spec.tolerance, |a: Quat, b: Quat| {
                        rotation_distance(a, b, spec.length)
                    }),
                    scales: decimate(&track.scales, spec.tolerance, |a: Vec3, b: Vec3| {
                        length3(a - b) * spec.length
                    }),
                }
            })
            .collect();

        let output = RawAnimation {
            tracks,
            duration: input.duration,
            name: input.name.clone(),
        };
        output.validate()?;
        Ok(output)
    }
}

/// Distance between 2 points on a circle of `radius`, at the shortest angle between 2 rotations.
fn rotation_distance(a: Quat, b: Quat, radius: f32) -> f32 {
    // cos_half_angle is the w component of a^-1 * b.
    let cos_half_angle = dot4(a, b);
    let sine_half_angle = (1.0 - f32::min(1.0, cos_half_angle * cos_half_angle)).sqrt();
    2.0 * sine_half_angle * radius
}

/// Ramer-Douglas-Peucker decimation, then removal of trailing keys that the previous key (or identity
/// for a single key) reproduces within tolerance.
fn decimate<K: RawKey>(src: &[K], tolerance: f32, distance: impl Fn(K::Value, K::Value) -> f32) -> Vec<K> {
    let mut output: Vec<K>;
    if src.len() < 2 {
        output = src.to_vec();
    } else {
        let mut included = vec![false; src.len()];
        included[0] = true;
        included[src.len() - 1] = true;
        let mut segments = vec![(0, src.len() - 1)];

        while let Some((first, second)) = segments.pop() {
            // Furthest point from the segment.
            let mut max = -1.0;
            let mut candidate = first;
            let left = src[first];
            let right = src[second];
            for (i, test) in src.iter().enumerate().take(second).skip(first + 1) {
                debug_assert!(!included[i]);
                let alpha = (test.time() - left.time()) / (right.time() - left.time());
                let lerped = K::lerp(left.value(), right.value(), alpha);
                let d = distance(lerped, test.value());
                if d > tolerance && d > max {
                    max = d;
                    candidate = i;
                }
            }

            if candidate != first {
                included[candidate] = true;
                if candidate - first > 1 {
                    segments.push((first, candidate));
                }
                if second - candidate > 1 {
                    segments.push((candidate, second));
                }
            }
        }

        output = src
            .iter()
            .zip(&included)
            .filter(|(_, inc)| **inc)
            .map(|(k, _)| *k)
            .collect();
    }

    // Removes last keys while the track is constant (or identity, for the last one).
    while let Some(back) = output.last() {
        let penultimate = if output.len() == 1 {
            K::identity()
        } else {
            output[output.len() - 2].value()
        };
        if distance(penultimate, back.value()) > tolerance {
            break;
        }
        output.pop();
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offline::{RawJoint, RawRotationKey, RawSkeleton, RawTranslationKey, SkeletonBuilder};
    use glam_ext::Transform3A;

    fn t(time: f32, x: f32) -> RawTranslationKey {
        RawTranslationKey {
            time,
            value: Vec3::new(x, 0.0, 0.0),
        }
    }

    #[test]
    fn test_decimate() {
        let dist = |a: Vec3, b: Vec3| length3(a - b);

        // Collinear keys collapse to the end points.
        let keys = vec![t(0.0, 0.0), t(0.25, 0.25), t(0.5, 0.5), t(1.0, 1.0)];
        assert_eq!(decimate(&keys, 1e-3, dist), vec![keys[0], keys[3]]);

        // A peak above tolerance stays; below tolerance goes.
        let keys = vec![t(0.0, 0.0), t(0.5, 0.01), t(1.0, 0.0), t(2.0, 1.0)];
        assert_eq!(decimate(&keys, 1e-3, dist), keys);
        assert_eq!(decimate(&keys, 2e-2, dist), vec![keys[0], keys[2], keys[3]]);

        // A constant track at identity disappears, a constant non-identity one keeps one key.
        let keys = vec![t(0.0, 0.0), t(1.0, 0.0)];
        assert!(decimate(&keys, 1e-3, dist).is_empty());
        let keys = vec![t(0.0, 1.0), t(0.5, 1.0), t(1.0, 1.0)];
        assert_eq!(decimate(&keys, 1e-3, dist), vec![keys[0]]);
    }

    #[test]
    fn test_hierarchical_tolerance() {
        // A child 1 m away: a small root rotation moves it by ~angle meters.
        let mut root = RawJoint::new("root", Transform3A::IDENTITY);
        let mut child = RawJoint::new("child", Transform3A::IDENTITY);
        child.transform.translation = glam::Vec3A::new(1.0, 0.0, 0.0);
        root.children.push(child);
        let skeleton = SkeletonBuilder.build(&RawSkeleton { roots: vec![root] }).unwrap();

        let r = |time: f32, angle: f32| RawRotationKey {
            time,
            value: Quat::from_rotation_z(angle),
        };
        let mut raw = RawAnimation {
            duration: 1.0,
            name: String::new(),
            tracks: vec![RawJointTrack::default(), RawJointTrack::default()],
        };
        // 0.5 mrad bump: 0.5 mm at the child, under 1 mm tolerance; 2 mrad: 2 mm, over.
        raw.tracks[0].rotations = vec![r(0.0, 0.0), r(0.5, 0.0005), r(1.0, 0.0)];
        raw.tracks[1].translations = vec![t(0.0, 1.0), t(1.0, 1.0)];
        let optimized = AnimationOptimizer::default().optimize(&raw, &skeleton).unwrap();
        assert!(optimized.tracks[0].rotations.is_empty());
        assert_eq!(optimized.tracks[1].translations.len(), 1);

        raw.tracks[0].rotations[1] = r(0.5, 0.002);
        let optimized = AnimationOptimizer::default().optimize(&raw, &skeleton).unwrap();
        assert_eq!(optimized.tracks[0].rotations.len(), 3);

        // Wider tolerance removes it again.
        let optimized = AnimationOptimizer::with_tolerance(0.01)
            .optimize(&raw, &skeleton)
            .unwrap();
        assert!(optimized.tracks[0].rotations.is_empty());

        raw.tracks.pop();
        assert_eq!(
            AnimationOptimizer::default().optimize(&raw, &skeleton).unwrap_err(),
            OfflineError::TrackCountMismatch { tracks: 1, joints: 2 }
        );
    }
}
