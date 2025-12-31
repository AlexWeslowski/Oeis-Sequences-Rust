use num::{Unsigned, NumCast, PrimInt};
use std::default::Default;
use std::fmt::Display;
use tinyvec::{tiny_vec, TinyVec};

pub trait Num: Sized + Default + NumCast + Copy + PrimInt + Display {
	const ARYSIZE: usize;
	type BackingArray: tinyvec::Array<Item = Self>;
}

/*

for pair in [("u8", 20), ("u16", 78), ("u32", 480), ("u64", 3840), ("u128", 38400), ("usize", 3840), ("i32", 480), ("i64", 3840), ("i128", 38400)]:
	print(f"""impl Num for {pair[0]} {{
    const ARYSIZE: usize = {pair[1]};
	type BackingArray = [{pair[0]}; Self::ARYSIZE];
}}""")

*/
impl Num for u8 {
    const ARYSIZE: usize = 20;
    type BackingArray = [u8; Self::ARYSIZE];
}
impl Num for u16 {
    const ARYSIZE: usize = 78;
    type BackingArray = [u16; Self::ARYSIZE];
}
impl Num for u32 {
    const ARYSIZE: usize = 480;
    type BackingArray = [u32; Self::ARYSIZE];
}
impl Num for u64 {
    const ARYSIZE: usize = 3840;
    type BackingArray = [u64; Self::ARYSIZE];
}
impl Num for u128 {
    const ARYSIZE: usize = 38400;
    type BackingArray = [u128; Self::ARYSIZE];
}
impl Num for usize {
    const ARYSIZE: usize = 3840;
    type BackingArray = [usize; Self::ARYSIZE];
}
impl Num for i32 {
    const ARYSIZE: usize = 480;
    type BackingArray = [i32; Self::ARYSIZE];
}
impl Num for i64 {
    const ARYSIZE: usize = 3840;
    type BackingArray = [i64; Self::ARYSIZE];
}
impl Num for i128 {
    const ARYSIZE: usize = 38400;
    type BackingArray = [i128; Self::ARYSIZE];
}

pub fn get_divisors<T: Num + Default>(n: T) -> TinyVec<T::BackingArray> {
    
    let _0: T = T::zero();
    let _1: T = T::one();
    let _2: T = T::from(2).unwrap();
    let mut _n = n;
    let mut v: TinyVec<T::BackingArray> = TinyVec::new();
    
    let mut count_divisors_2: usize = 0;
    while _n & _1 == _0 {
        v.push(_2 << count_divisors_2);
        count_divisors_2 += 1;
        _n = _n >> 1;
    }
    
    let mut _x: T = T::from(3).unwrap();
    let mut _n_sqrt: T = approximated_sqrt(_n);
    while _x < _n_sqrt {        
        let mut _pow_x = _x;
        let v_len = v.len();
        let mut x_is_a_divisors = false;

        let mut pow_x_is_a_divisors = _n % _x == _0;
        while pow_x_is_a_divisors == true {
            _n = _n.div(_x);
            v.push(_pow_x);
            push_new_divisors(&mut v, v_len, _pow_x);
            pow_x_is_a_divisors = _n % _x == _0;
            if pow_x_is_a_divisors == true {
                _pow_x = _pow_x.mul(_x);                
            }
            x_is_a_divisors = true;
        }
        _x = _x + _2;
        if x_is_a_divisors == true {
            _n_sqrt = approximated_sqrt(_n);
        }
    }
    
    if _n > _1 && _n != n {
        let v_len = v.len();
        v.push(_n);
        push_new_divisors(&mut v, v_len, _n);
    }

    if v.len() > 1 {
        v.sort();
        v.pop();
    }
    
    v
}

pub fn approximated_sqrt<T: Num>(n: T) -> T {
    let _0: T = T::zero();
    let _1: T = T::one();
    let mut num_bits = (std::mem::size_of::<T>() << 3) - 1;
    while ((n >> num_bits) & _1) == _0 {
        num_bits -= 1;
    }
    
    _1 << ((num_bits >> 1) + 1)
}


fn push_new_divisors<T: Num + Default>(v: &mut TinyVec<T::BackingArray>, v_len: usize, _x: T) {
    for i in 0..v_len {
        v.push(_x.mul(unsafe { *v.get_unchecked(i) }));
    }
}