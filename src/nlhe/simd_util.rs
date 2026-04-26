//! f32x4 SIMD helpers for CFR hot loops.
//! All functions operate on contiguous slices; callers are responsible for
//! ensuring slices have the expected lengths. The `_k` functions work on
//! slices of arbitrary length k — they process f32x4 lanes then handle
//! the trailing remainder scalarly.

use wide::f32x4;

/// `out[i] += a[i] * b[i]` (fused-multiply-add, length k).
#[inline]
pub fn fma_slice(out: &mut [f32], a: &[f32], b: &[f32]) {
    let k = out.len();
    debug_assert_eq!(a.len(), k);
    debug_assert_eq!(b.len(), k);
    let lanes = k / 4;
    let rem = k % 4;
    for i in 0..lanes {
        let o = f32x4::from(&out[i*4..i*4+4]);
        let av = f32x4::from(&a[i*4..i*4+4]);
        let bv = f32x4::from(&b[i*4..i*4+4]);
        let r = o + av * bv;
        out[i*4..i*4+4].copy_from_slice(&r.to_array());
    }
    let base = lanes * 4;
    for i in 0..rem { out[base+i] += a[base+i] * b[base+i]; }
}

/// `out[i] += scale * a[i]` (axpy, length k).
#[inline]
pub fn axpy(out: &mut [f32], scale: f32, a: &[f32]) {
    let k = out.len();
    debug_assert_eq!(a.len(), k);
    let sv = f32x4::splat(scale);
    let lanes = k / 4;
    let rem = k % 4;
    for i in 0..lanes {
        let o = f32x4::from(&out[i*4..i*4+4]);
        let av = f32x4::from(&a[i*4..i*4+4]);
        let r = o + sv * av;
        out[i*4..i*4+4].copy_from_slice(&r.to_array());
    }
    let base = lanes * 4;
    for i in 0..rem { out[base+i] += scale * a[base+i]; }
}

/// `out[i] = a[i] * b[i]` (element-wise multiply into out, length k).
#[inline]
pub fn mul_into(out: &mut [f32], a: &[f32], b: &[f32]) {
    let k = out.len();
    debug_assert_eq!(a.len(), k);
    debug_assert_eq!(b.len(), k);
    let lanes = k / 4;
    let rem = k % 4;
    for i in 0..lanes {
        let av = f32x4::from(&a[i*4..i*4+4]);
        let bv = f32x4::from(&b[i*4..i*4+4]);
        (av * bv).to_array().iter().enumerate().for_each(|(j, &v)| out[i*4+j] = v);
    }
    let base = lanes * 4;
    for i in 0..rem { out[base+i] = a[base+i] * b[base+i]; }
}

/// Horizontal dot product: sum_i a[i]*b[i] (length k, returns scalar).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let k = a.len();
    debug_assert_eq!(b.len(), k);
    let lanes = k / 4;
    let rem = k % 4;
    let mut acc = f32x4::splat(0.0);
    for i in 0..lanes {
        let av = f32x4::from(&a[i*4..i*4+4]);
        let bv = f32x4::from(&b[i*4..i*4+4]);
        acc += av * bv;
    }
    let mut s: f32 = acc.to_array().iter().sum();
    let base = lanes * 4;
    for i in 0..rem { s += a[base+i] * b[base+i]; }
    s
}

/// CFR+ regret update for one action's per-bucket slice (length k).
/// `regrets[i] = max(0, regrets[i] + own_reach[i] * (action_util[i] - node_ev[i]))`
/// `strat_sum[i] += t * own_reach[i] * strategy[i]`
#[inline]
pub fn cfr_plus_update(
    regrets: &mut [f32],
    strat_sum: &mut [f32],
    own_reach: &[f32],
    action_util: &[f32],
    node_ev: &[f32],
    strategy: &[f32],
    t_f: f32,
) {
    let k = regrets.len();
    debug_assert_eq!(strat_sum.len(), k);
    debug_assert_eq!(own_reach.len(), k);
    debug_assert_eq!(action_util.len(), k);
    debug_assert_eq!(node_ev.len(), k);
    debug_assert_eq!(strategy.len(), k);
    let zero = f32x4::splat(0.0);
    let tv = f32x4::splat(t_f);
    let lanes = k / 4;
    let rem = k % 4;
    for i in 0..lanes {
        let base = i * 4;
        let r = f32x4::from(&regrets[base..base+4]);
        let reach = f32x4::from(&own_reach[base..base+4]);
        let au = f32x4::from(&action_util[base..base+4]);
        let ev = f32x4::from(&node_ev[base..base+4]);
        let st = f32x4::from(&strategy[base..base+4]);
        let new_r = r + reach * (au - ev);
        let clamped = new_r.max(zero);
        clamped.to_array().iter().enumerate().for_each(|(j, &v)| regrets[base+j] = v);
        let ss_delta = tv * reach * st;
        let ss = f32x4::from(&strat_sum[base..base+4]) + ss_delta;
        ss.to_array().iter().enumerate().for_each(|(j, &v)| strat_sum[base+j] = v);
    }
    let base = lanes * 4;
    for i in 0..rem {
        let own_r = own_reach[base+i];
        let regret = action_util[base+i] - node_ev[base+i];
        let new_r = regrets[base+i] + own_r * regret;
        regrets[base+i] = if new_r > 0.0 { new_r } else { 0.0 };
        strat_sum[base+i] += t_f * own_r * strategy[base+i];
    }
}
