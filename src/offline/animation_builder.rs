//!
//! Builds a runtime `Animation` from a `RawAnimation` (ozz `AnimationBuilder`).
//!

use std::cmp::Ordering;

use super::raw_animation::{RawKey, RawRotationKey, RawScaleKey, RawTranslationKey};
use super::{dot4, f32_to_f16, normalize_safe_quat, OfflineError, RawAnimation};
use crate::animation::{Animation, AnimationRaw, Float3Key, QuaternionKey};
use glam::{Quat, Vec3};

/// Offsets to previous keyframes are stored on `u16`.
const MAX_PREVIOUS_OFFSET: usize = (1 << 16) - 1;

///
/// Builds a runtime `Animation` from a `RawAnimation`.
///
/// - Every track gets a key at t = 0 and t = duration (copied from its first and last key,
///   identity for an empty track). Tracks are padded to a multiple of 4 with identity tracks.
/// - Keys are sorted by the time of the previous key of their track, then by track, so that
///   sampling reads keyframes forward.
/// - Translations and scales are stored as half floats, rotations as smallest-three quaternions.
///
#[derive(Debug, Clone, Copy, Default)]
pub struct AnimationBuilder {
    /// Interval in seconds between iframes, which let the sampler seek without reading keys
    /// sequentially. 0 generates none. Any positive value guarantees one at the end of the animation.
    pub iframe_interval: f32,
}

#[derive(Debug, Clone, Copy)]
struct SortingKey<K> {
    track: u16,
    prev_key_time: f32,
    key: K,
}

fn sorting_less<K>(left: &SortingKey<K>, right: &SortingKey<K>) -> bool {
    let time_diff = left.prev_key_time - right.prev_key_time;
    time_diff < 0.0 || (time_diff == 0.0 && left.track < right.track)
}

fn sorting_cmp<K>(left: &SortingKey<K>, right: &SortingKey<K>) -> Ordering {
    if sorting_less(left, right) {
        Ordering::Less
    } else if sorting_less(right, left) {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn push_back_identity_key<K: RawKey>(track: u16, time: f32, dest: &mut Vec<SortingKey<K>>) {
    let prev_key_time = match dest.last() {
        Some(last) if last.track == track => last.key.time(),
        _ => -1.0,
    };
    dest.push(SortingKey {
        track,
        prev_key_time,
        key: K::new(time, K::identity()),
    });
}

/// Copies a raw track, adding keys at t = 0 and t = duration if missing.
fn copy_raw<K: RawKey>(src: &[K], track: u16, duration: f32, dest: &mut Vec<SortingKey<K>>) {
    match src {
        [] => {
            push_back_identity_key(track, 0.0, dest);
            push_back_identity_key(track, duration, dest);
        }
        [key] => {
            dest.push(SortingKey {
                track,
                prev_key_time: -1.0,
                key: K::new(0.0, key.value()),
            });
            dest.push(SortingKey {
                track,
                prev_key_time: 0.0,
                key: K::new(duration, key.value()),
            });
        }
        [first, .., last] => {
            let mut prev_time = -1.0;
            if first.time() != 0.0 {
                dest.push(SortingKey {
                    track,
                    prev_key_time: prev_time,
                    key: K::new(0.0, first.value()),
                });
                prev_time = 0.0;
            }
            for key in src {
                dest.push(SortingKey {
                    track,
                    prev_key_time: prev_time,
                    key: *key,
                });
                prev_time = key.time();
            }
            if last.time() - duration != 0.0 {
                dest.push(SortingKey {
                    track,
                    prev_key_time: prev_time,
                    key: K::new(duration, last.value()),
                });
            }
        }
    }
}

/// Sorts keys, then injects keys wherever a key is further than `MAX_PREVIOUS_OFFSET` from the
/// previous key of its track.
fn sort_keys<K: RawKey>(src: &mut Vec<SortingKey<K>>, num_tracks: usize) {
    src.sort_by(sorting_cmp);

    // Last and penultimate key per track.
    let mut previouses: Vec<(Option<usize>, Option<usize>)> = vec![(None, None); num_tracks];
    loop {
        let mut changed = false;
        previouses.fill((None, None));

        for i in 0..src.len() {
            let track = src[i].track;
            let previous = previouses[track as usize];

            if let Some(first) = previous.0 {
                if i - first > MAX_PREVIOUS_OFFSET {
                    let second = previous.1.expect("a key that far has a penultimate key");
                    let mut last = src[first];
                    let penultimate = src[second];

                    let insert_time = (penultimate.key.time() + last.key.time()) * 0.5;
                    let insert = SortingKey {
                        track,
                        prev_key_time: penultimate.key.time(),
                        key: K::new(insert_time, K::lerp(penultimate.key.value(), last.key.value(), 0.5)),
                    };
                    last.prev_key_time = insert_time;

                    // `last` changes its sorting key: removes it and merges both keys back, stable,
                    // into the range after the penultimate key (nothing changed before it).
                    src.remove(first);
                    let mut lower = second;
                    for key in [insert, last] {
                        let pos = lower + src[lower..].partition_point(|k| !sorting_less(&key, k));
                        src.insert(pos, key);
                        lower = pos + 1;
                    }

                    changed = true;
                    break;
                }
            }

            previouses[track as usize] = (Some(i), previous.0);
        }

        if !changed {
            break;
        }
    }
}

/// Normalizes quaternions and flips successive opposite ones so that normalized lerp takes the
/// shortest path. Keys are still grouped per track at this point.
fn fixup_quaternions(src: &mut [SortingKey<RawRotationKey>]) {
    let mut track = None;
    for i in 0..src.len() {
        let mut normalized = normalize_safe_quat(src[i].key.value, Quat::IDENTITY);
        if track != Some(src[i].track) {
            // First key of the track: w is the dot with identity.
            if normalized.w < 0.0 {
                normalized = -normalized;
            }
        } else if dot4(src[i - 1].key.value, normalized) < 0.0 {
            normalized = -normalized;
        }
        src[i].key.value = normalized;
        track = Some(src[i].track);
    }
}

fn build_timepoints(
    translations: &[SortingKey<RawTranslationKey>],
    rotations: &[SortingKey<RawRotationKey>],
    scales: &[SortingKey<RawScaleKey>],
) -> Vec<f32> {
    let mut timepoints: Vec<f32> = Vec::with_capacity(translations.len() + rotations.len() + scales.len());
    timepoints.extend(translations.iter().map(|k| k.key.time));
    timepoints.extend(rotations.iter().map(|k| k.key.time));
    timepoints.extend(scales.iter().map(|k| k.key.time));
    timepoints.sort_by(|a, b| a.partial_cmp(b).unwrap());
    timepoints.dedup();
    timepoints
}

/// Group varint encoding of 4 integers (ozz `EncodeGV4`).
fn encode_gv4(input: &[u32], out: &mut Vec<u8>) {
    debug_assert_eq!(input.len(), 4);
    let tag = |v: u32| (v >= (1 << 24)) as u8 + (v >= (1 << 16)) as u8 + (v >= (1 << 8)) as u8;
    let tags = [tag(input[0]), tag(input[1]), tag(input[2]), tag(input[3])];
    out.push((tags[3] << 6) | (tags[2] << 4) | (tags[1] << 2) | tags[0]);
    for (value, tag) in input.iter().zip(tags) {
        out.extend_from_slice(&value.to_le_bytes()[..tag as usize + 1]);
    }
}

struct BuilderIFrames {
    entries: Vec<u8>,
    desc: Vec<u32>,
    interval: f32,
}

/// Splits keys into iframes: per iframe, the index of the last key of each track whose
/// previous key is before the iframe time.
fn build_iframes<K>(src: &[SortingKey<K>], num_soa_tracks: usize, interval: f32, duration: f32) -> BuilderIFrames {
    let mut iframes = BuilderIFrames {
        entries: Vec::new(),
        desc: Vec::new(),
        interval: 1.0,
    };
    if num_soa_tracks == 0 || interval <= 0.0 {
        return iframes;
    }

    let divs = f32::max(1.0, duration / interval) as usize;
    let mut cache = vec![0u32; num_soa_tracks];
    for i in 0..divs {
        let time = duration * (i + 1) as f32 / divs as f32;

        cache.fill(0);
        let mut last = 0;
        for (index, key) in src.iter().enumerate() {
            if key.prev_key_time > time {
                break;
            }
            cache[key.track as usize] = index as u32;
            last = index;
        }
        debug_assert!(last >= num_soa_tracks * 2 - 1);

        // No iframe for the first set of keyframes, nor without keys since the previous one.
        if last <= num_soa_tracks * 2 - 1 {
            continue;
        }
        if let Some(&previous) = iframes.desc.last() {
            if last <= previous as usize {
                continue;
            }
        }

        iframes.desc.push(iframes.entries.len() as u32);
        iframes.desc.push(last as u32);
        for chunk in cache.chunks_exact(4) {
            encode_gv4(chunk, &mut iframes.entries);
        }
    }

    iframes.interval = if iframes.entries.is_empty() {
        1.0
    } else {
        1.0 / (iframes.desc.len() / 2) as f32
    };
    iframes
}

/// Ratio indices, offsets to previous keys of the same track, and compressed values.
fn compress<K: RawKey, D: Copy>(
    timepoints: &[f32],
    src: &[SortingKey<K>],
    num_soa_tracks: usize,
    compressor: impl Fn(K::Value) -> D,
) -> (Vec<u16>, Vec<u16>, Vec<D>) {
    let mut ratios = Vec::with_capacity(src.len());
    let mut previouses = Vec::with_capacity(src.len());
    let mut values = Vec::with_capacity(src.len());
    let mut track_positions: Vec<Option<usize>> = vec![None; num_soa_tracks];
    for (i, key) in src.iter().enumerate() {
        let time = key.key.time();
        let ratio = timepoints.partition_point(|&t| t < time);
        debug_assert!(timepoints[ratio] == time);
        debug_assert!(timepoints.len() > u8::MAX as usize || ratio <= u8::MAX as usize);
        ratios.push(ratio as u16);

        let diff = track_positions[key.track as usize].map_or(0, |p| i - p);
        debug_assert!(diff < MAX_PREVIOUS_OFFSET);
        previouses.push(diff as u16);

        values.push(compressor(key.key.value()));
        track_positions[key.track as usize] = Some(i);
    }
    (ratios, previouses, values)
}

fn compress_float3(value: Vec3) -> Float3Key {
    Float3Key::new([f32_to_f16(value.x), f32_to_f16(value.y), f32_to_f16(value.z)])
}

/// Smallest-three quantization: the 3 smallest components (each within ±√2/2) on 15 bits,
/// the largest one's index and sign; the largest is restored from unit length.
fn compress_quaternion(value: Quat) -> QuaternionKey {
    const BITS: u32 = 15;
    const ISCALE: i32 = (1 << BITS) - 1;
    const SCALE: f32 = ISCALE as f32 / core::f32::consts::SQRT_2;
    const OFFSET: f32 = -core::f32::consts::FRAC_1_SQRT_2;
    const MAPPING: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];

    let quat = [value.x, value.y, value.z, value.w];
    let mut largest = 0;
    for i in 1..4 {
        if quat[largest].abs() < quat[i].abs() {
            largest = i;
        }
    }

    let map = &MAPPING[largest];
    let quantize = |v: f32| i32::min(((v - OFFSET) * SCALE + 0.5) as i32, ISCALE);
    let cpnt = [quantize(quat[map[0]]), quantize(quat[map[1]]), quantize(quat[map[2]])];
    let sign = (quat[largest] < 0.0) as u64;

    let packed: u64 = (largest as u64 & 0x3)
        | (sign << 2)
        | ((cpnt[0] as u64 & 0x7fff) << 3)
        | ((cpnt[1] as u64 & 0x7fff) << 18)
        | ((cpnt[2] as u64 & 0x7fff) << 33);
    QuaternionKey::new([
        (packed & 0xffff) as u16,
        ((packed >> 16) & 0xffff) as u16,
        ((packed >> 32) & 0xffff) as u16,
    ])
}

impl AnimationBuilder {
    /// Builds the runtime `Animation`.
    ///
    /// Fails if `raw` doesn't validate or has more distinct keyframe times than a `u16` indexes.
    pub fn build(&self, raw: &RawAnimation) -> Result<Animation, OfflineError> {
        raw.validate()?;

        let duration = raw.duration;
        let inv_duration = 1.0 / raw.duration;
        let num_tracks = raw.num_tracks() as u16;
        let num_soa_tracks = ((num_tracks + 3) & !3) as usize;

        let (mut translation_count, mut rotation_count, mut scale_count) = (0, 0, 0);
        for track in &raw.tracks {
            // +2: worst case adds the first and last keys.
            translation_count += track.translations.len() + 2;
            rotation_count += track.rotations.len() + 2;
            scale_count += track.scales.len() + 2;
        }
        let mut translations: Vec<SortingKey<RawTranslationKey>> = Vec::with_capacity(translation_count);
        let mut rotations: Vec<SortingKey<RawRotationKey>> = Vec::with_capacity(rotation_count);
        let mut scales: Vec<SortingKey<RawScaleKey>> = Vec::with_capacity(scale_count);

        for (i, track) in raw.tracks.iter().enumerate() {
            copy_raw(&track.translations, i as u16, duration, &mut translations);
            copy_raw(&track.rotations, i as u16, duration, &mut rotations);
            copy_raw(&track.scales, i as u16, duration, &mut scales);
        }
        for i in num_tracks..num_soa_tracks as u16 {
            push_back_identity_key(i, 0.0, &mut translations);
            push_back_identity_key(i, duration, &mut translations);
            push_back_identity_key(i, 0.0, &mut rotations);
            push_back_identity_key(i, duration, &mut rotations);
            push_back_identity_key(i, 0.0, &mut scales);
            push_back_identity_key(i, duration, &mut scales);
        }

        fixup_quaternions(&mut rotations);

        sort_keys(&mut translations, num_soa_tracks);
        sort_keys(&mut rotations, num_soa_tracks);
        sort_keys(&mut scales, num_soa_tracks);

        // After sorting: sorting may inject keys.
        let timepoints = build_timepoints(&translations, &rotations, &scales);
        if timepoints.len() > u16::MAX as usize {
            return Err(OfflineError::TooManyTimepoints(timepoints.len()));
        }

        let t_iframes = build_iframes(&translations, num_soa_tracks, self.iframe_interval, duration);
        let r_iframes = build_iframes(&rotations, num_soa_tracks, self.iframe_interval, duration);
        let s_iframes = build_iframes(&scales, num_soa_tracks, self.iframe_interval, duration);

        let (t_ratios, t_previouses, translations) =
            compress(&timepoints, &translations, num_soa_tracks, compress_float3);
        let (r_ratios, r_previouses, rotations) =
            compress(&timepoints, &rotations, num_soa_tracks, compress_quaternion);
        let (s_ratios, s_previouses, scales) = compress(&timepoints, &scales, num_soa_tracks, compress_float3);

        let raw = AnimationRaw {
            duration,
            num_tracks: num_tracks as u32,
            name: raw.name.clone(),
            timepoints: timepoints.iter().map(|t| t * inv_duration).collect(),

            translations,
            t_ratios,
            t_previouses,
            t_iframe_interval: t_iframes.interval,
            t_iframe_entries: t_iframes.entries,
            t_iframe_desc: t_iframes.desc,

            rotations,
            r_ratios,
            r_previouses,
            r_iframe_interval: r_iframes.interval,
            r_iframe_entries: r_iframes.entries,
            r_iframe_desc: r_iframes.desc,

            scales,
            s_ratios,
            s_previouses,
            s_iframe_interval: s_iframes.interval,
            s_iframe_entries: s_iframes.entries,
            s_iframe_desc: s_iframes.desc,
        };
        Ok(Animation::from_raw(&raw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offline::RawJointTrack;

    #[test]
    fn test_compress_quaternion() {
        let quats = [
            Quat::IDENTITY,
            -Quat::IDENTITY,
            Quat::from_xyzw(0.5, -0.5, 0.5, -0.5),
            Quat::from_rotation_x(2.5),
            Quat::from_rotation_y(-1.0),
            Quat::from_euler(glam::EulerRot::XYZ, 0.3, -1.2, 2.9),
        ];
        for q in quats {
            let back = compress_quaternion(q).decompress();
            assert!(dot4(q, back).abs() > 1.0 - 1e-7, "{:?} -> {:?}", q, back);
        }
    }

    #[test]
    fn test_encode_gv4() {
        let mut out = Vec::new();
        encode_gv4(&[1, 256, 65536, 16777216], &mut out);
        assert_eq!(out, vec![0b11_10_01_00, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1]);
    }

    #[test]
    fn test_build_layout() {
        // Two tracks: one keyed at 0, 0.5, 1; one empty (identity).
        let raw = RawAnimation {
            duration: 2.0,
            name: "test".into(),
            tracks: vec![
                RawJointTrack {
                    translations: vec![
                        RawTranslationKey {
                            time: 0.5,
                            value: Vec3::X,
                        },
                        RawTranslationKey {
                            time: 1.0,
                            value: Vec3::Y,
                        },
                    ],
                    rotations: vec![RawRotationKey {
                        time: 1.0,
                        value: -Quat::from_rotation_z(1.0),
                    }],
                    ..Default::default()
                },
                RawJointTrack::default(),
            ],
        };
        let animation = AnimationBuilder::default().build(&raw).unwrap();
        assert_eq!(animation.duration(), 2.0);
        assert_eq!(animation.num_tracks(), 2);
        assert_eq!(animation.name(), "test");
        assert_eq!(animation.timepoints(), &[0.0, 0.25, 0.5, 1.0]);

        // Track 0 gets keys at 0 (copied), 0.5, 1 and 2 (copied); 3 identity tracks get 2 keys each.
        assert_eq!(animation.translations().len(), 4 + 3 * 2);
        let ctrl = animation.translations_ctrl();
        // First keys of the 4 tracks, then second keys (sorted by previous key time).
        assert_eq!(&ctrl.ratios[..8], &[0, 0, 0, 0, 1, 3, 3, 3]);
        assert_eq!(&ctrl.previouses[..8], &[0, 0, 0, 0, 4, 4, 4, 4]);
        assert_eq!(animation.translations()[4].decompress(), Vec3::X);
        assert!(ctrl.iframe_desc.is_empty());

        // Rotations: the w < 0 first key is flipped to w > 0.
        assert_eq!(animation.rotations().len(), 8);
        let r0 = animation.rotations()[0].decompress();
        assert!(r0.abs_diff_eq(Quat::from_rotation_z(1.0), 1e-4));
    }

    #[test]
    fn test_build_iframes() {
        let keys = (0..=100)
            .map(|i| RawTranslationKey {
                time: i as f32 * 0.1,
                value: Vec3::splat(i as f32),
            })
            .collect::<Vec<_>>();
        let raw = RawAnimation {
            duration: 10.0,
            name: String::new(),
            tracks: vec![RawJointTrack {
                translations: keys,
                ..Default::default()
            }],
        };
        let animation = AnimationBuilder { iframe_interval: 2.0 }.build(&raw).unwrap();
        let ctrl = animation.translations_ctrl();
        assert_eq!(ctrl.iframe_desc.len(), 10);
        assert_eq!(ctrl.iframe_interval, 0.2);
        // The scale and rotation tracks have only first and last keys: no iframes.
        assert!(animation.scales_ctrl().iframe_desc.is_empty());
        assert_eq!(animation.scales_ctrl().iframe_interval, 1.0);
    }
}
