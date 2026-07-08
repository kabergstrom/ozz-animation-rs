//! Scalar stand-in for the `std::simd` (portable_simd, nightly-only) subset
//! used by this crate, so the fork builds on stable Rust.
//!
//! Types mirror `std::simd` layout (`#[repr(C, align(16))]`, 4 lanes) and the
//! subset of the API that `math.rs` / `animation.rs` / `skeleton.rs` /
//! `sampling_job.rs` use. Semantics are element-wise IEEE f32 — deterministic,
//! just not vectorized. `simd_swizzle!` is re-implemented as a macro over
//! generic lane-shuffle helpers (both one- and two-vector forms).

#![allow(non_camel_case_types)]

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
    DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Not, Shl, Shr, Sub, SubAssign,
};

#[repr(C, align(16))]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct f32x4(pub(crate) [f32; 4]);

#[repr(C, align(16))]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct i32x4(pub(crate) [i32; 4]);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct mask32x4(pub(crate) [bool; 4]);

macro_rules! lanewise {
    ($a:expr, $b:expr, $op:tt) => {
        [
            $a.0[0] $op $b.0[0],
            $a.0[1] $op $b.0[1],
            $a.0[2] $op $b.0[2],
            $a.0[3] $op $b.0[3],
        ]
    };
}

macro_rules! lanecmp {
    ($a:expr, $b:expr, $op:tt) => {
        mask32x4([
            $a.0[0] $op $b.0[0],
            $a.0[1] $op $b.0[1],
            $a.0[2] $op $b.0[2],
            $a.0[3] $op $b.0[3],
        ])
    };
}

impl f32x4 {
    pub const fn from_array(a: [f32; 4]) -> Self {
        Self(a)
    }
    pub const fn splat(v: f32) -> Self {
        Self([v; 4])
    }
    pub const fn to_array(self) -> [f32; 4] {
        self.0
    }
    pub const fn as_array(&self) -> &[f32; 4] {
        &self.0
    }
    pub fn as_mut_array(&mut self) -> &mut [f32; 4] {
        &mut self.0
    }
    pub fn from_slice(s: &[f32]) -> Self {
        Self([s[0], s[1], s[2], s[3]])
    }

    pub fn abs(self) -> Self {
        Self(self.0.map(f32::abs))
    }
    pub fn sqrt(self) -> Self {
        Self(self.0.map(f32::sqrt))
    }
    pub fn recip(self) -> Self {
        Self(self.0.map(f32::recip))
    }
    pub fn floor(self) -> Self {
        Self(self.0.map(f32::floor))
    }
    pub fn fract(self) -> Self {
        Self(self.0.map(f32::fract))
    }
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self([
            self.0[0].mul_add(a.0[0], b.0[0]),
            self.0[1].mul_add(a.0[1], b.0[1]),
            self.0[2].mul_add(a.0[2], b.0[2]),
            self.0[3].mul_add(a.0[3], b.0[3]),
        ])
    }
    pub fn simd_min(self, o: Self) -> Self {
        Self(lanewise_min_f(self.0, o.0))
    }
    pub fn simd_max(self, o: Self) -> Self {
        Self(lanewise_max_f(self.0, o.0))
    }
    pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
        self.simd_max(lo).simd_min(hi)
    }
    pub fn simd_eq(self, o: Self) -> mask32x4 {
        lanecmp!(self, o, ==)
    }
    pub fn simd_ne(self, o: Self) -> mask32x4 {
        lanecmp!(self, o, !=)
    }
    pub fn simd_lt(self, o: Self) -> mask32x4 {
        lanecmp!(self, o, <)
    }
    pub fn simd_le(self, o: Self) -> mask32x4 {
        lanecmp!(self, o, <=)
    }
    pub fn simd_gt(self, o: Self) -> mask32x4 {
        lanecmp!(self, o, >)
    }
    pub fn simd_ge(self, o: Self) -> mask32x4 {
        lanecmp!(self, o, >=)
    }
    pub fn reduce_sum(self) -> f32 {
        self.0[0] + self.0[1] + self.0[2] + self.0[3]
    }
    pub fn reduce_min(self) -> f32 {
        self.0[0].min(self.0[1]).min(self.0[2]).min(self.0[3])
    }
    pub fn reduce_max(self) -> f32 {
        self.0[0].max(self.0[1]).max(self.0[2]).max(self.0[3])
    }
    pub fn is_nan(self) -> mask32x4 {
        mask32x4(self.0.map(f32::is_nan))
    }
    pub fn to_bits(self) -> i32x4 {
        i32x4(self.0.map(|v| v.to_bits() as i32))
    }
    /// Lane-wise `f32 as i32` without the checked conversion (mirrors
    /// `Simd::to_int_unchecked`).
    ///
    /// # Safety
    /// Each lane must be finite and in `i32` range after truncation.
    pub unsafe fn to_int_unchecked(self) -> i32x4 {
        i32x4(self.0.map(|v| unsafe { v.to_int_unchecked::<i32>() }))
    }
}

fn lanewise_min_f(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    // portable_simd simd_min is IEEE minimum-number: propagates the number
    // when one operand is NaN, like f32::min.
    [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2]), a[3].min(b[3])]
}
fn lanewise_max_f(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2]), a[3].max(b[3])]
}

impl i32x4 {
    pub const fn from_array(a: [i32; 4]) -> Self {
        Self(a)
    }
    pub const fn splat(v: i32) -> Self {
        Self([v; 4])
    }
    pub const fn to_array(self) -> [i32; 4] {
        self.0
    }
    pub const fn as_array(&self) -> &[i32; 4] {
        &self.0
    }
    pub fn simd_eq(self, o: Self) -> mask32x4 {
        lanecmp!(self, o, ==)
    }
    pub fn simd_ne(self, o: Self) -> mask32x4 {
        lanecmp!(self, o, !=)
    }
    pub fn simd_lt(self, o: Self) -> mask32x4 {
        lanecmp!(self, o, <)
    }
    pub fn simd_le(self, o: Self) -> mask32x4 {
        lanecmp!(self, o, <=)
    }
    pub fn simd_gt(self, o: Self) -> mask32x4 {
        lanecmp!(self, o, >)
    }
    pub fn simd_ge(self, o: Self) -> mask32x4 {
        lanecmp!(self, o, >=)
    }
    pub fn simd_min(self, o: Self) -> Self {
        Self([
            self.0[0].min(o.0[0]),
            self.0[1].min(o.0[1]),
            self.0[2].min(o.0[2]),
            self.0[3].min(o.0[3]),
        ])
    }
    pub fn simd_max(self, o: Self) -> Self {
        Self([
            self.0[0].max(o.0[0]),
            self.0[1].max(o.0[1]),
            self.0[2].max(o.0[2]),
            self.0[3].max(o.0[3]),
        ])
    }
    pub fn cast_f32(self) -> f32x4 {
        f32x4(self.0.map(|v| v as f32))
    }
}

impl mask32x4 {
    pub fn all(self) -> bool {
        self.0[0] && self.0[1] && self.0[2] && self.0[3]
    }
    pub fn any(self) -> bool {
        self.0[0] || self.0[1] || self.0[2] || self.0[3]
    }
    pub fn to_int(self) -> i32x4 {
        i32x4(self.0.map(|b| if b { -1 } else { 0 }))
    }
    /// Bit `i` = lane `i` (LSB = lane 0), matching `std::simd::Mask`.
    pub fn to_bitmask(self) -> u64 {
        (self.0[0] as u64) | ((self.0[1] as u64) << 1) | ((self.0[2] as u64) << 2) | ((self.0[3] as u64) << 3)
    }
    pub fn select<T: SelectLanes>(self, t: T, f: T) -> T {
        T::select_lanes(self, t, f)
    }
}

pub trait SelectLanes: Copy {
    fn select_lanes(m: mask32x4, t: Self, f: Self) -> Self;
}
impl SelectLanes for f32x4 {
    fn select_lanes(m: mask32x4, t: Self, f: Self) -> Self {
        Self([
            if m.0[0] { t.0[0] } else { f.0[0] },
            if m.0[1] { t.0[1] } else { f.0[1] },
            if m.0[2] { t.0[2] } else { f.0[2] },
            if m.0[3] { t.0[3] } else { f.0[3] },
        ])
    }
}
impl SelectLanes for i32x4 {
    fn select_lanes(m: mask32x4, t: Self, f: Self) -> Self {
        Self([
            if m.0[0] { t.0[0] } else { f.0[0] },
            if m.0[1] { t.0[1] } else { f.0[1] },
            if m.0[2] { t.0[2] } else { f.0[2] },
            if m.0[3] { t.0[3] } else { f.0[3] },
        ])
    }
}

macro_rules! impl_binop {
    ($ty:ident, $trait:ident, $fn:ident, $op:tt) => {
        impl $trait for $ty {
            type Output = $ty;
            fn $fn(self, o: $ty) -> $ty {
                $ty(lanewise!(self, o, $op))
            }
        }
    };
}

impl_binop!(f32x4, Add, add, +);
impl_binop!(f32x4, Sub, sub, -);
impl_binop!(f32x4, Mul, mul, *);
impl_binop!(f32x4, Div, div, /);
impl_binop!(i32x4, Add, add, +);
impl_binop!(i32x4, Sub, sub, -);
impl_binop!(i32x4, BitAnd, bitand, &);
impl_binop!(i32x4, BitOr, bitor, |);
impl_binop!(i32x4, BitXor, bitxor, ^);

macro_rules! impl_assign {
    ($ty:ident, $trait:ident, $fn:ident, $op:tt) => {
        impl $trait for $ty {
            fn $fn(&mut self, o: $ty) {
                *self = *self $op o;
            }
        }
    };
}

impl_assign!(f32x4, AddAssign, add_assign, +);
impl_assign!(f32x4, SubAssign, sub_assign, -);
impl_assign!(f32x4, MulAssign, mul_assign, *);
impl_assign!(f32x4, DivAssign, div_assign, /);
impl_assign!(i32x4, AddAssign, add_assign, +);
impl_assign!(i32x4, SubAssign, sub_assign, -);
impl_assign!(i32x4, BitAndAssign, bitand_assign, &);
impl_assign!(i32x4, BitOrAssign, bitor_assign, |);
impl_assign!(i32x4, BitXorAssign, bitxor_assign, ^);

impl Neg for f32x4 {
    type Output = f32x4;
    fn neg(self) -> f32x4 {
        f32x4(self.0.map(|v| -v))
    }
}
impl Neg for i32x4 {
    type Output = i32x4;
    fn neg(self) -> i32x4 {
        i32x4(self.0.map(|v| -v))
    }
}
impl Not for i32x4 {
    type Output = i32x4;
    fn not(self) -> i32x4 {
        i32x4(self.0.map(|v| !v))
    }
}

macro_rules! impl_shift {
    ($rhs:ty) => {
        impl Shl<$rhs> for i32x4 {
            type Output = i32x4;
            fn shl(self, s: $rhs) -> i32x4 {
                i32x4(self.0.map(|v| v << s))
            }
        }
        impl Shr<$rhs> for i32x4 {
            type Output = i32x4;
            fn shr(self, s: $rhs) -> i32x4 {
                i32x4(self.0.map(|v| v >> s))
            }
        }
    };
}
impl_shift!(i32);
impl_shift!(u32);

impl Shl<i32x4> for i32x4 {
    type Output = i32x4;
    fn shl(self, s: i32x4) -> i32x4 {
        i32x4(core::array::from_fn(|i| self.0[i] << s.0[i]))
    }
}
impl Shr<i32x4> for i32x4 {
    type Output = i32x4;
    fn shr(self, s: i32x4) -> i32x4 {
        i32x4(core::array::from_fn(|i| self.0[i] >> s.0[i]))
    }
}

impl Index<usize> for f32x4 {
    type Output = f32;
    fn index(&self, i: usize) -> &f32 {
        &self.0[i]
    }
}
impl IndexMut<usize> for f32x4 {
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        &mut self.0[i]
    }
}
impl Index<usize> for i32x4 {
    type Output = i32;
    fn index(&self, i: usize) -> &i32 {
        &self.0[i]
    }
}
impl IndexMut<usize> for i32x4 {
    fn index_mut(&mut self, i: usize) -> &mut i32 {
        &mut self.0[i]
    }
}

pub trait SwizzleVec: Copy {
    type Elem: Copy;
    fn lane(self, i: usize) -> Self::Elem;
    fn from_lanes(l: [Self::Elem; 4]) -> Self;
}
impl SwizzleVec for f32x4 {
    type Elem = f32;
    fn lane(self, i: usize) -> f32 {
        self.0[i]
    }
    fn from_lanes(l: [f32; 4]) -> Self {
        Self(l)
    }
}
impl SwizzleVec for i32x4 {
    type Elem = i32;
    fn lane(self, i: usize) -> i32 {
        self.0[i]
    }
    fn from_lanes(l: [i32; 4]) -> Self {
        Self(l)
    }
}

pub fn swizzle1<T: SwizzleVec>(v: T, idx: [usize; 4]) -> T {
    T::from_lanes([v.lane(idx[0]), v.lane(idx[1]), v.lane(idx[2]), v.lane(idx[3])])
}

/// Two-vector form: indices 0..3 pick from `a`, 4..7 from `b`.
pub fn swizzle2<T: SwizzleVec>(a: T, b: T, idx: [usize; 4]) -> T {
    let pick = |i: usize| if i < 4 { a.lane(i) } else { b.lane(i - 4) };
    T::from_lanes([pick(idx[0]), pick(idx[1]), pick(idx[2]), pick(idx[3])])
}

macro_rules! simd_swizzle {
    ($v:expr, $idx:expr $(,)?) => {
        $crate::simd_compat::swizzle1($v, $idx)
    };
    ($a:expr, $b:expr, $idx:expr $(,)?) => {
        $crate::simd_compat::swizzle2($a, $b, $idx)
    };
}
pub(crate) use simd_swizzle;
