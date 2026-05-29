#![allow(unused_imports)]
#![allow(unused)]
#![allow(non_snake_case)] 

//extern crate crossbeam;
//extern crate divisors;
//extern crate flurry;
extern crate itertools;
extern crate num;
extern crate primes;
//extern crate seize;
//extern crate shared_memory;
extern crate std;
extern crate thousands;

mod enums;
mod divisors;
mod sequence24;
mod sequence2;
#[cfg(test)]
mod tests;

use ahash::{AHasher, AHashMap, AHashSet, HashSetExt, RandomState};
//use bit_vec::BitVec;
use chrono::{Local, Timelike};
use clap::{Parser, Subcommand, ValueEnum};
use enums::{CalcDensityType, DataType, Backtrack};
use fixedbitset::FixedBitSet;
//use flurry::HashSet as FlurryHashSet;
use hashbrown::HashMap;
use itertools::Itertools;
use lazy_static::lazy_static;
use num::rational::{Ratio, Rational32, Rational64};
//use perf;
//use perf::Perf;
//use perf_macro;
use primes::{PrimeSet, Sieve};
use raw_cpuid::CpuId;
use sequence24::{PRIMES_SIZE, PrimesType, Sequence24};
use serde_json;
//use shared_memory::*;
use smallvec::{smallvec, SmallVec};
use sysinfo::{Pid, System};
use thousands::Separable;
use time_graph;
use time_graph_macros::instrument;
use tinyvec::{array_vec, ArrayVec};
use tinyvec::{tiny_vec, TinyVec};

use std::clone::Clone;
use std::collections::{BTreeSet, HashSet};
use std::cmp::min;
use std::env;
use std::error::Error;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::hash::Hash;
use std::hint;
use std::io::{self, Write};
use std::ops::{BitOr, BitOrAssign};
use std::process::{self, Command};
use std::slice::Iter;
use std::thread;
use std::time::{Duration, Instant};
use std::str::{self, FromStr};
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicI8, AtomicI32, AtomicI64, AtomicU32, Ordering};
use std::sync::mpsc;
use std::sync::mpsc::{Sender};


/*
lazy_static! {
    static ref Seq: Sequence24 = Sequence24::new(2, 11457600, false);
}
*/


struct LockedBool {
    lock: AtomicBool,
	val: AtomicBool,
}

impl LockedBool {
	pub fn new(bln: bool) -> Self {
		Self { 
			lock: AtomicBool::new(false),
			val: AtomicBool::new(bln),
		}
	}
	
    pub fn lock_and_load(&self) -> bool {
        while self.lock.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            hint::spin_loop();
        }
		return self.val.load(Ordering::SeqCst);
    }
	
	pub fn store(&self, bln: bool) {
		self.val.store(bln, Ordering::SeqCst);
	}
	
    pub fn unlock(&self) {
        self.lock.store(false, Ordering::Release);
    }
}

struct Message {
    threadid: u32,
    datatype: DataType,
    calcdensity: CalcDensityType,
	n: u32,
	ratio: Ratio<i32>,
	msg: String
}	

#[derive(Clone, Copy, Debug, PartialEq)]
struct RatioVec {
    n: i32, 
    ratio: Ratio<i32>,
    //slice: &'a[i32],
    slice: [i32; 10]
}

macro_rules! message_format {
    //() => ("{}    {}    [{}]    {}");
    () => ("{}\t{}\t[{}]\t{}");
}

impl RatioVec {
    fn to_string(self) -> String {        
        //let s = Itertools::join(&mut self.slice.iter(), ", ");
        //let s: String = self.slice.iter().filter(|&&x| x != 0).map(|&x| x.to_string()).collect::<Vec<String>>().join(",");
        let mut v: Vec<i32> = self.slice.iter().filter(|&&x| x != 0).cloned().collect();
        return format!(message_format!(), self.ratio, self.n.separate_with_commas(), v.clone().into_iter().join(", "), v.len());
    }
}

struct Main {
	debug: bool,
    logging: bool,
    ithread: u32,
    anumthreads: AtomicU32,
    t1: Instant,
    astart: AtomicU32,
    afinish: AtomicU32,
	datatype: DataType,
	calcdensity: CalcDensityType,
    matches: Vec<Ratio<i32>>,
    predefined: HashMap<u32, Vec<RatioVec>, RandomState>,
	outmap: Arc<Mutex<AHashMap<Ratio<i32>, ArrayVec<[u32; 4096]>>>>,
	tx: Option<Sender<Option<Message>>>,
}



#[derive(Clone, PartialEq, Eq, Hash)]
enum CombinationType {
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    Usize(usize),
}

fn as_usize(val: &CombinationType) -> usize {
    match val {
        CombinationType::I32(x) => *x as usize,
        CombinationType::U32(x) => *x as usize,
        CombinationType::I64(x) => *x as usize,
        CombinationType::U64(x) => *x as usize,
        CombinationType::Usize(x) => *x as usize,
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Combination {
    singles: Vec<usize>,
    multiples: Vec<Vec<usize>>,
    multiples_flat: Vec<usize>,
}

impl Main
{

pub fn print_duration(&mut self, seq: &Sequence24, mut n: u32, mut icount: u32, inumthreads: u32)
{
    if icount <= 10 {
        return;
    }
    for i in 1..10 {
        if n % 100 == i {
            n -= i;
        }
        if icount % 100 == i {
            icount -= i;
        }
    }
    let fcount = (icount as f32)/(inumthreads as f32);
    let fmins = (self.t1.elapsed().as_secs() as f32)/60.0;
    let fhrs = fmins/60.0;
    if fmins <= 0.005 {
        return;
    }
	let msg = if fhrs < 1.0 {
        format!("thread #{}, ({}) {:.2} mins ~ {} per min", self.ithread + 1, icount.separate_with_commas(), (100.0*fmins).round()/100.0, (fcount/fmins).round().separate_with_commas())
    } else {
        format!("thread #{}, ({}) {:.2} hrs ~ {} per min", self.ithread + 1, icount.separate_with_commas(), (100.0*fhrs).round()/100.0, (fcount/fmins).round().separate_with_commas())
    };
    if self.logging && self.anumthreads.load(Ordering::Relaxed) == 1 {
        println!("{}", msg);
    }
	if let Some(tx) = &self.tx {
		tx.send(Some(Message { threadid: self.ithread + 1, datatype: self.datatype, calcdensity: self.calcdensity, n: n, ratio: Ratio::<i32>::new(0, 1), msg: msg }));
	}
    //seq.print_capacity();
    //println!("lcm_map.capacity() = {}", seq.lcm_map.lock().unwrap().capacity().separate_with_commas());
    //println!("factors.keys.len() = {}, factors.values.len() = {}", seq.factors.lock().unwrap().len().0.separate_with_commas(), seq.factors.lock().unwrap().len().1.separate_with_commas());
    //println!("divisors.keys.len() = {}, divisors.values.len() = {}", seq.divisors.lock().unwrap().len().0.separate_with_commas(), seq.divisors.lock().unwrap().len().1.separate_with_commas());
}

#[instrument]
fn output(&mut self, n: &u32, this_combination: &TinyVec<[i32; 24]>, density: &Ratio<i32>, datatype: &DataType) 
{
	if let Some(ratio) = self.matches.iter().find(|&x| x == density)
	{
		let vec: String = Itertools::join(&mut this_combination.iter(), ", ");
		let num: String = (*n).separate_with_commas();
		let msg: String = format!(message_format!(), ratio, num, vec, this_combination.len());
		self.outmap.lock().unwrap().get_mut(ratio).unwrap().push(*n);
		if self.anumthreads.load(Ordering::Relaxed) == 1 {
			println!("{}", msg);
		}
		if let Some(tx) = &self.tx {
			tx.send(Some(Message { threadid: self.ithread + 1, datatype: *datatype, calcdensity: self.calcdensity, n: *n, ratio: *ratio, msg: msg }));
		}
	}
}

/*
nstart, nfinish, nstepby, ithousands = 2, 4000000, 2, 1000000
#[int(((n - nstart) / nstepby) % 4) for n in range(nstart, nstart + 24, nstepby)]
for inumthreads in range(4, 4 + 1):
	minhsh = {ithread:[] for ithread in range(0, inumthreads)}
	nums = []
	for ithread in range(0, inumthreads):
		n = nstart - nstepby
		while n < nfinish:
			n += nstepby
			if ((n - nstart) / nstepby) % inumthreads != ithread:
				continue
			nums.append(n)
			if (n - nstart - nstepby * ithread) % ithousands < inumthreads:
				minhsh[ithread].append(n - nstart)
			if (n - nstart - nstepby * ithread) % ithousands < inumthreads:
				print(f"ithread = {ithread}, n % inumthreads = {n % inumthreads}, n - nstart - nstepby * ithread = {n - nstart - nstepby * ithread}, {n - nstart - nstepby * ithread} % ithousands = {(n - nstart - nstepby * ithread) % ithousands}")
            if (n - nstart - nstepby * ithread) % ithousands == 0:
				print(f"ithread = {ithread}, n % inumthreads = {n % inumthreads}, n - nstart - ithread = {n - nstart - ithread}, {n - nstart - ithread} % ithousands = {(n - nstart - ithread) % ithousands}")
	#print(f"inumthreads = {inumthreads}, nums = {sorted(nums)}")
	print(f"inumthreads = {inumthreads}, minhsh = {minhsh}")
    print(f"len(nums) = {len(nums)}, nfinish - nstart + 1 = {nfinish - nstart + 1}")    
    #sorted(nums) == list(range(nstart, nfinish+1, nstepby))
	if len(nums) != nfinish - nstart + 1 or sorted(nums) != list(range(nstart, nfinish+1, nstepby)):
		break
*/
#[instrument]
#[function_name::named]
pub fn do_work(&mut self, mut nstart: u32, nfinish: u32, inumthreads: u32, mut seq: Sequence24) -> (Vec<usize>, Vec<usize>)
{
    let nstepby: u32 = if self.matches.len() == 1 { *self.matches[0].denom() as u32 } else { 1 };
    if nstart % nstepby > 0
    {
        nstart += nstepby - (nstart % nstepby);
    }
    /*while nstart < 12 {
        nstart += nstepby;
    }*/
    let ihundreds: u32 = 100;
    let ithousands: u32 = 1000000;
	let mut n: u32 = if nstepby > nstart { 0 } else { nstart - nstepby };
	let mut nvec: Vec<u32> = Vec::new();
	let mut vecmaxcombinations = vec![0];
	
    //let bln_array = self.datatype.is_set(DataType::ARRAY);
	let bln_vec = self.datatype.is_set(DataType::VEC);
	let bln_tinyvec = self.datatype.is_set(DataType::TINYVEC);
	let bln_arrayvec = self.datatype.is_set(DataType::ARRAYVEC);
	let bln_smallvec = self.datatype.is_set(DataType::SMALLVEC);
    let mut vec_types: TinyVec<[DataType; 4]> = TinyVec::new();
    for dt in [DataType::VEC, DataType::TINYVEC, DataType::ARRAYVEC, DataType::SMALLVEC] {
        if self.datatype.is_set(dt) {
            vec_types.push(dt);
        }
    }
    if vec_types.len() == 1 {
        seq.datatype = vec_types[0];
    }
    
	println!("do_work() n = {}, nfinish = {}, nstepby = {}, n % nstepby = {}, ithread = {}", n, nfinish, nstepby, n % nstepby, self.ithread);
	
	while n < nfinish
    {
		n += nstepby;
        if ((n - nstart) / nstepby) % inumthreads != self.ithread
        {
            continue;
        }
		if self.debug {
			nvec.push(n);
		}
        match n {
            // 1/2    3, 4         12
            // 1/2    3, 7, 8     168
            // 1/2    3, 5, 16    240
            2 | 12 | 168 | 240 |
            3 | 24 | 30 | 36 | 378 | 480 | 504 | 540 | 600 | 660 | 720 | 840 | 936 | 1260 | 1320 | 1404 | 1980 |
            4 | 48 | 56 | 80 | 864 | 1344 | 1512 | 1680 | 1824 | 1920 | 2240 | 2496 | 3024 | 3840 | 4032 | 4480 | 4960 | 5280 | 6720 | 8640 | 9120 | 11520 | 21760 => {
                if !self.matches.iter().find(|&m| self.predefined[&n].iter().any(|p| p.ratio == *m)).is_none() {
					for ratiovec in &self.predefined[&n] {
                        if self.matches.iter().find(|&m| *m == ratiovec.ratio).is_none() {
                            continue;
                        }
						let msg: String = ratiovec.to_string();
						//println!("{} line {}", function_name!(), line!());
						self.outmap.lock().unwrap().get_mut(&ratiovec.ratio).unwrap().push(n);
						if inumthreads == 1 {
							println!("{}", msg.clone());
						}
						/*
                        if bln_array {
                            if let Some(tx) = &self.tx {
                                tx.send(Some(Message { threadid: self.ithread + 1, datatype: DataType::ARRAY, calcdensity: seq.calcdensity, n: n, ratio: ratiovec.ratio, msg: msg.clone() }));
                            }
                        }
						*/
                        for vt in 0..vec_types.len() {
                            if let Some(tx) = &self.tx {
                                tx.send(Some(Message { threadid: self.ithread + 1, datatype: vec_types[vt], calcdensity: seq.calcdensity, n: n, ratio: ratiovec.ratio, msg: msg.clone() }));
                            }
                        }
					}
                }
            },
            _ => {
				let mut icombinations = 0;
				
				/*
				if bln_array {
					let mut prev_combination: [i32; 24] = [0; 24];
					for this_combination in seq.factor_combinations_ary(n as i32)
					{ 
						icombinations += 1;
						//println!("icombinations = {}", icombinations);
						if this_combination != prev_combination 
						{
							let this_vec: TinyVec<[i32; 24]> = this_combination.iter().map(|&x| x as i32).filter(|&x| x != 0).collect();
							let density: Ratio<i32> = seq.calc_density(n as usize, &this_vec);
							//println!("n = {}, comb = {:?}", n, this_combination);
							//println!("n = {}, avec = {:?}", n, avec);
							//println!("n = {}, dens = {}/{}", n, density.numer(), density.denom());
							self.output(&n, &this_vec, &density, &DataType::ARRAY);
							prev_combination = this_combination;
						}
					}
				}
				*/
				if bln_vec {
					let mut prev_combination: TinyVec<[i32; 24]> = tiny_vec![0; 24];
					for this_combination in seq.factor_combinations_vec(n as i32)
					{ 
						icombinations += 1;
						if this_combination != prev_combination 
						{
							let density: Ratio<i32> = seq.calc_density(n as usize, &this_combination);							
							self.output(&n, &this_combination, &density, &DataType::VEC);
							prev_combination = this_combination;
						}
					}
				}
				if bln_tinyvec {
					let mut prev_combination: TinyVec<[i32; 24]> = tiny_vec![0; 24];
					for this_combination in seq.factor_combinations_tinyvec(n as i32)
					{ 
						icombinations += 1;
						if this_combination != prev_combination 
						{
							let density: Ratio<i32> = seq.calc_density(n as usize, &this_combination);
							self.output(&n, &this_combination, &density, &DataType::TINYVEC);
							prev_combination = this_combination;
						}
					}
				}
				if bln_arrayvec {
					let mut prev_combination: TinyVec<[i32; 24]> = tiny_vec![0; 24];
					for this_combination in seq.factor_combinations_arrayvec(n as i32)
					{ 
						icombinations += 1;
						if this_combination != prev_combination 
						{
							let density: Ratio<i32> = seq.calc_density(n as usize, &this_combination);
							self.output(&n, &this_combination, &density, &DataType::ARRAYVEC);
							prev_combination = this_combination;
						}
					}
				}
				if bln_smallvec {
					let mut prev_combination: TinyVec<[i32; 24]> = tiny_vec![0; 24];
					for this_combination in seq.factor_combinations_smallvec(n as i32)
					{ 
						icombinations += 1;
						if this_combination != prev_combination 
						{
							let density: Ratio<i32> = seq.calc_density(n as usize, &this_combination);
							self.output(&n, &this_combination, &density, &DataType::SMALLVEC);
							prev_combination = this_combination;
						}
					}
				}
				if icombinations > vecmaxcombinations[vecmaxcombinations.len() - 1] {
					vecmaxcombinations.push(icombinations);
				}
                if (n - nstart - nstepby * self.ithread) % ithousands < inumthreads
                {
                    self.print_duration(&seq, n, n - nstart, inumthreads);   
                }                
            },
        }
    }
	
	//println!("maxcombinations = {:?}", vecmaxcombinations);
	if self.debug {
		println!("n = {}", n);
		println!("afinish = {}", self.afinish.load(Ordering::SeqCst));
        println!("nvec.len() = {}, {} - {} + 1 = {}", nvec.len(), nvec[nvec.len()-1], nvec[0], nvec[nvec.len()-1] - nvec[0] + 1);
		println!("nvec = {:?}", nvec);
	}
	if let Some(tx) = &self.tx {
		tx.send(None);
	}
    seq.max_combinations.sort();
	return (seq.max_stack, seq.max_combinations);
}
}



fn init(maxprime: u32) -> PrimesType
{
    let mut primes: PrimesType = PrimesType::new();
    let mut pset = Sieve::new();
    for p in pset.iter()
    {
        let up = p as u32;
        if up > maxprime
        {
            break;
        }
        primes.push(up);
    }
    return primes;
}


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(index = 1)]
    numthreads: u8,
    #[arg(index = 2)]
    ratios: String,
    #[arg(index = 3)]
    start: u32,
    #[arg(index = 4)]
    finish: u32,
    #[arg(index = 5)]
	method: CalcDensityType,
	#[arg(index = 6)]
	datatype: DataType,
    #[arg(short, long, default_value_t = false)]
    verbose: bool,
    #[arg(short, long, default_value_t = false)]
    logging: bool,
    #[arg(short, long, default_value_t = false)]
    debug: bool,
    #[arg(short, long, default_value_t = false)]
    perf: bool,
    #[arg(short, long, default_value_t = false)]
    file: bool,
    #[arg(long)]
    filepath: Option<String>,
	//    8*32*1024 .... 262144
	//   32*32*1024 ... 1048576
	//   64*32*1024 ... 2097152
	//  128*32*1024 ... 4194304
	//  256*32*1024 ... 8388608
	// 1024*32*1024 .. 33554432
	#[arg(long, aliases = ["stack", "stacksize", "stack_size"], default_value_t = 96*32*1024)]
    stacksize: usize,
}

/*
import collections
import bitarray
import fractions
import math
import divisors

def calc_density1(n, tpl):
    bits = bitarray.bitarray(n + 1)
    for t in tpl:
        temp = bitarray.bitarray(n + 1)
        for b in range(t, n + 1, t):
            temp[b] = True
        bits |= temp
    return fractions.Fraction(bits.count(), n)

match = fractions.Fraction(1, 3)
for n in range(2, 374220 + 2):
    combinations = divisors.Combinations(n)
    combinations.backtrack(n, [])
    bln = False
    for combo in combinations:
        bln = True
        if calc_density1(n, combo) == match:
            print(f"{calc_density1(n, combo)}    {n:,}    {combo}    {len(combo)}")
    if not bln and fractions.Fraction(1, n) == match:
        print(f"{match}    {n:,}    {[n,]}    {1}")

dir = "D:\\Rust\\Sequence"
[int(line.split("\t")[1].replace(",", "")) for line in open("{dir}\\sequence 1_2.txt").readlines() if not line.startswith("#")]
ary = []
with open(f"{dir}\\sequence (1,3).txt") as f:
    for line in f:
		line = line.strip()
        if len(line) <= 1:
            continue
        n = int(line.split("\t")[1].replace(",", ""))
        ary.append(n)

# 1/2    6,720    [3, 7, 10, 32]
hsh = collections.defaultdict(list)
dir = "D:\\Rust\\Sequence\\src"
with open(f"{dir}\\predefined.txt") as f:
	prev_factors = []
    for line in f:
		line = line.strip()
        if len(line) <= 1:
            continue
		ary = line.split("    ")
        n = int(ary[1].replace(",", ""))
		this_factors = list(ary[2])
		if this_factors == prev_factors:
			print(f"Warning, duplicate: {line}")
        combinations = divisors.Combinations(n)
        combinations.backtrack(n, [])
        for combo in combinations:
            frac1 = calc_density1(n, combo)
            if (frac1.denominator >= 5 or frac1.numerator > 1) and combo not in hsh[frac1]:
                hsh[frac1].append(combo)
		prev_factors = this_factors

icount = 0
for frac1 in sorted(hsh.keys()):
    for combo in hsh[frac1]:
		icount += 1
        print(f"{frac1}    {math.prod(combo):,}    {combo}")
*/
const PREDEFINED: &str = include_str!("predefined.txt");

/*
 * 
 * https://play.rust-lang.org/?version=stable&mode=debug&edition=2021
 * 
 * cargo build
 * Finished `dev` profile [unoptimized + debuginfo] target(s) in 5m 05s
 * cargo build --release
 * Finished `release` profile [optimized] target(s) in 32m 23s
 * 
 * target\debug\sequence_rust 2 3 2 1024
 * target/debug/sequence_rust 2 2 65536
 * target/debug/sequence_rust 2 2 4408320
 * target\release\sequence_rust 2 2 2 65536
 * target/release/sequence_rust 1 2 4408320
 * target/release/sequence_rust 1 2 39621120 (2^25.2)
 * target\release\sequence_rust 1 2 4194304 268435456
 * 
 * target\release\sequence_rust.exe 4 "[(1,2), (1,3)]" 2 128
 * target\release\sequence_rust.exe 4 "[(1,2)]" 2 65536 OR vec --file --perf
 * target\release\sequence_rust.exe 1 "[(1,2)]" 2 1048576 RATIO tinyvec --perf
 * target\debug\sequence_rust.exe 1 "[(1,2)]" 2 1048576 RATIO tinyvec --perf --stacksize 16777216
 * target\release\sequence_rust.exe 1 "[(1,2)]" 2 4194304 RATIO tinyvec --perf --stacksize 33554432
 * target\release\sequence_rust.exe 1 "[(1,2)]" 2 8388608 OR arrayvec
 * 33554432
 * 134217728
 * target\release\sequence_rust.exe 1 "[(1,2)]" 4194304 8388608 --file --array --stacksize 1048576
 * target\release\sequence_rust.exe 4 "[(1,2), (1,3), (1,4)]" 2 8388608 ratio tinyvec --file
 * target\release\sequence_rust.exe 1 "[(1,2)]" 2 4408322 RATIO tinyvec --file --stacksize 1048576
 * target\release\sequence_rust.exe 1 "[(1,2)]" 2 4440482 RATIO tinyvec --file --stacksize 1048576
 * target\release\sequence_rust.exe 1 "[(1,4)]" 2 25522560 RATIO smallvec --file --stacksize 33554432
 * target\release\sequence_rust.exe 1 "[(1,5)]" 58069444 134217728 RATIO smallvec --file --stacksize 33554432
 * target\release\sequence_rust.exe 1 "[(1,2)]" 61501442 134217728 RATIO tinyvec --file
 * target\release\sequence_rust.exe 1 "[(1,3)]" 59875202 134217728 RATIO tinyvec --file
 * python.exe "E:\Python\Sequence\sequence_th.py" 1 [(1,3)] 15301442 134217728
 * 
 * target\debug\sequence_rust.exe 1 "[(1,2)]" 549120 591360 ratio array|vec --perf
 * target\debug\sequence_rust.exe 1 "[(1,2)]" 3655680 3886080 ratio vec|arrayvec --perf
 * target\release\sequence_rust.exe 1 "[(1,2)]" 1145750 1145770 RATIO tinyvec --stacksize 33554432
 * target\release\sequence_rust.exe 1 "[(1,2)]" 3655680 3886080 ratio array|tinyvec --perf
 * target\release\sequence_rust.exe 1 "[(1,2)]" 2 1048576 ratio tinyvec --stacksize 33554432
 * 
 * 
 * C:\Python\Python314\python.exe "E:\Python\Sequence\sequence_th.py" 1 [(1,2)] 2 1048576
 * i7-1165G7 @ 2.80GHz [(1,2)] from 2 to 1048576 with 1 thread  in    6.62 minutes
 * i7-1165G7 @ 2.80GHz [(1,2)] from 2 to 4194304 with 1 thread  in   34.17 minutes
 * i7-1165G7 @ 2.80GHz [(1,2)] from 2 to 4194304 with 1 thread  in    2.06 minutes (2,035,417 per min)
 * 503.22 MB physical memory
 * 706.79 MB virtual memory
 * 
 *                     [(1,3)] from 2 to 4194304 with 4 threads in 4299.50 minutes (71.66 hours)
 * i7-1165G7 @ 2.80GHz [(1,3)] from 2 to 4194304 with 1 thread  in   20.50 minutes ( 0.34 hours)
 * i7-1165G7 @ 2.80GHz [(1,2)] from 2 to 1048576 with 1 thread  in    1.83 minutes (RATIO array   )
 * i7-1165G7 @ 2.80GHz [(1,2)] from 2 to 1048576 with 1 thread  in    2.26 minutes (RATIO arrayvec)
 * i7-1165G7 @ 2.80GHz [(1,2)] from 2 to 1048576 with 1 thread  in   17.82 minutes (   OR arrayvec) 17.82/2.26 = 7.88
 * i7-1165G7 @ 2.80GHz [(1,2)] from 2 to 1048576 with 1 thread  in    2.45 minutes (RATIO tinyvec )
 * i7-1165G7 @ 2.80GHz [(1,2)] from 2 to 1048576 with 1 thread  in    0.78 minutes (RATIO tinyvec ) backtrack_stack
 * i7-1165G7 @ 2.80GHz [(1,2)] from 2 to 1048576 with 1 thread  in    0.41 minutes (RATIO tinyvec ) backtrack_recurse
 * i7-1165G7 @ 2.80GHz [(1,2)] from 2 to 1048576 with 1 thread  in   21.39 minutes (   OR tinyvec ) 21.39/2.45 = 8.73
 * i7-1165G7 @ 2.80GHz [(1,2)] from 2 to 1048576 with 1 thread  in    2.83 minutes (RATIO array   )
 * i7-1165G7 @ 2.80GHz [(1,2)] from 2 to 1048576 with 1 thread  in    3.36 minutes (RATIO vec     )
 *
 *  i7-1360P @ 2.20GHz [(1,2)] from 2 to 1048576 with 1 thread  in   20.43 seccons (RATIO vec     ) 3,079,861 per min
 * 
 *                     [(1,2)] (1,000,000) 28.5 mins ~ 35,108 per min
 *                     [(1,2)] from 2 to 1048576 with 1 thread  in   30.7 minutes ( 0.51 hours)
 *                     [(1,2)] from 2 to 1048576 with 4 threads in   27.6 minutes ( 0.46 hours)
 *
 * physical_memory=27 MB
 * virtual_memory=25 MB
 *  
 */

fn print_time(f: &str, s: &str)
{
    let time = Local::now().time();
    let (pm, hour) = time.hour12();
    println!("{}() {}={:02}:{:02}:{:02}{}", f, s, hour, time.minute(), time.second(), if pm { "pm" } else { "am" });	
}

fn print_memory(f: &str, pid: &Pid) -> bool {
    let mut system = System::new_all();
    system.refresh_all();
    if let Some(process) = system.process(*pid) {
		let memory_mb = process.memory() / 1024 / 1024;
		let virtual_mb = process.virtual_memory() / 1024 / 1024;
		println!("{}() physical_memory={} MB, virtual_memory={} MB", f, memory_mb.separate_with_commas(), virtual_mb.separate_with_commas());
		return true;
    }
	return false;
}

fn predefined() -> HashMap<u32, Vec<RatioVec>, RandomState> {
	let mut predefined1: HashMap<u32, Vec<RatioVec>, RandomState> = HashMap::with_hasher(RandomState::new());
    for line in PREDEFINED.lines() {
        if line.len() <= 1 {
            continue;
        }
        let mut vars: Vec<&str> = line.split("    ").collect();
        if !vars[0].contains('/') {
            continue;
        }
        vars[0] = vars[0].split('/').collect::<Vec<_>>()[1];
        let vars0: i32 = vars[0].parse::<i32>().ok().unwrap();
        let vars1s = &vars[1].replace(",", "");
        let vars1i = vars1s.parse::<i32>().ok().unwrap();
        let vars1u = vars1i as u32;
        vars[2] = vars[2].trim_matches(|p| p == '[' || p == ']');
        let mut vars2: Vec<i32> = vars[2].split(',').filter_map(|s| s.trim().parse().ok()).collect();
        //println!("1/{}    {}    {:?}", vars0, vars1i, vars2);
        while vars2.len() < 10 {
            vars2.push(0);
        }
		let ratiovec = RatioVec { n: vars1i, ratio: Ratio::<i32>::new(1, vars0), slice: vars2.try_into().unwrap() };
		let vec = predefined1.entry(vars1u).or_insert_with(Vec::new);
		if !vec.contains(&ratiovec) {
			vec.push(ratiovec);
		}
    }
	return predefined1;
}

fn print_elapsed(total_sec: f64, istart: u32, ifinish: u32, inumthreads: u8) {	
    let total_min = total_sec/60.0;
    let total_hrs = total_min/60.0;
    let count = (ifinish - istart) as f64;    
    if total_sec < 60.0 {
        println!("finished from {} to {} with {} threads in {:.2} sec ({:.2} min) ({:.2} hrs) ~ {} per min", istart, ifinish, inumthreads, total_sec, total_min, total_hrs, (count/total_min).round().separate_with_commas());
    } else {
        println!("finished from {} to {} with {} threads in {} sec ({:.2} min) ({:.2} hrs) ~ {} per min", istart, ifinish, inumthreads, total_sec.round() as i64, total_min, total_hrs, (count/total_min).round().separate_with_commas());
    }
}

#[function_name::named]
fn main() 
{
	let line_numbers: bool = false;
	let mut bln_gt_half: bool = false;
	let mut min_factors_len: usize = 4;
	
    let output = Command::new("rustc").arg("--version").output().unwrap();
    println!("{}() rust_version=\"{}\"", function_name!(), String::from_utf8_lossy(&output.stdout).trim_end());
    let cpuid = CpuId::new();
    if let Some(brand_string) = cpuid.get_processor_brand_string() && let Some(freq_info) = cpuid.get_processor_frequency_info() {
        let base_ghz = freq_info.processor_base_frequency() as f64 / 1000.0;
        let max_ghz = freq_info.processor_max_frequency() as f64 / 1000.0;
		let mut brand = brand_string.as_str().trim().to_string();
		if !brand.contains("@") && !brand.contains("GHz") {
			if base_ghz > 0.0 {
				brand.push_str(&format!(" @ {:.2} GHz", base_ghz));
			} else if max_ghz > 0.0 {
				brand.push_str(&format!(" @ {:.2} GHz", max_ghz));
			} else {
				let mut system = System::new_all();
				system.refresh_all();
				let mut cpu_ghz = 0.0;
				for cpu in system.cpus() {
					cpu_ghz = cpu.frequency() as f64 / 1000.0;
					break;
				}
				if cpu_ghz > 0.0 {
					brand.push_str(&format!(" @ {:.2} GHz", cpu_ghz));
				}
			}
		}
        println!("{}() processor=\"{}\"", function_name!(), brand);
    }
    print_time(function_name!(), "start_time");
	
	if line_numbers { println!("{}() line {}", function_name!(), line!()); }
    let vecargs: Vec<String> = env::args().collect();
    let mut args = Args::parse();
    if let Some(fp) = args.filepath.clone().filter(|fp| fp.len() > 0) {
        args.file = true;
    }
	if line_numbers { println!("{}() line {}", function_name!(), line!()); }
    if args.debug {
        //divisors::bln_debug.store(true, Ordering::Relaxed);
        println!("{:#?}", args);
    }
	if line_numbers { println!("{}() line {}", function_name!(), line!()); }
    if args.perf {
        //divisors::bln_perf.store(true, Ordering::Relaxed);
        time_graph::enable_data_collection(true);
    }
    
	let t1: Instant = Instant::now();
	let pid = Pid::from_u32(process::id());
	args.ratios = args.ratios.replace("[", "").replace("]", "").replace(" ", "");
	
	if line_numbers { println!("{}() line {}", function_name!(), line!()); }
    let mut vecprimes1: Arc<PrimesType> = Arc::new(init(args.finish as u32));

	if line_numbers { println!("{}() line {}", function_name!(), line!()); }
    //let vecratios1: Vec<Ratio<i32>> = (iratio..=iratio).map(|x| Ratio::<i32>::new(1, x as i32)).collect();
    let mut vecratios1: Vec<Ratio<i32>> = Vec::new();
	let mut outmap1: Arc<Mutex<AHashMap<Ratio<i32>, ArrayVec<[u32; 4096]>>>> = Arc::new(Mutex::new(AHashMap::new()));
	let tuples: Vec<&str> = args.ratios.split("),(").collect();
    let half: Ratio<i32> = Ratio::<i32>::new(1, 2);
	for tuple1 in tuples {
		let tuple2 = tuple1.replace("(", "").replace(")", "");
		let parts: Vec<&str> = tuple2.split(",").collect();
		let num: i32 = parts[0].parse::<i32>().ok().unwrap();
		let den: i32 = parts[1].parse::<i32>().ok().unwrap();
        let rat: Ratio<i32> = Ratio::<i32>::new(num, den);
        if rat > half {
            bln_gt_half = true;
        }
		vecratios1.push(rat);
		outmap1.lock().unwrap().insert(rat, ArrayVec::<[u32; 4096]>::new());
		if den > 4 || num > 1 {
			min_factors_len = 2;
		}
	}
    
    if args.filepath.is_none() {
        args.filepath = Some(format!("sequence {}.txt", args.ratios.replace("),(", ") (")));
    }
    
	if line_numbers { println!("{}() line {}", function_name!(), line!()); }
	let mut predefined1 = predefined();
	
	if false {
		for rat in &vecratios1 {
			println!("matches {}", rat);
		}
		for (key, val) in &predefined1 {
			println!("predefined {}: {:?}", key, val.into_iter().map(|&ratiovec| ratiovec.n));
		}
	}
	
	if line_numbers { println!("{}() line {}", function_name!(), line!()); }
    println!("{}() num_threads={}, vec_ratios=[{}], start={}, finish={}, method={}, stacksize={}", function_name!(), args.numthreads, args.ratios.replace(" ", ""), args.start.separate_with_commas(), args.finish.separate_with_commas(), args.method.to_string(), args.stacksize.separate_with_commas());
    println!("{}() vec={}, tinyvec={}, arrayvec={}, smallvec={}", function_name!(), args.datatype.is_set(DataType::VEC), args.datatype.is_set(DataType::TINYVEC), args.datatype.is_set(DataType::ARRAYVEC), args.datatype.is_set(DataType::SMALLVEC));
	println!("{}() logging={}, debug={}, perf={}, file={}{}", function_name!(), args.logging, args.debug, args.perf, args.file, if args.file && let Some(ref fp) = args.filepath { format!(", file_path=\"{}\"", fp) } else { "".to_string() });


    let mut istacksize = args.numthreads as usize * args.stacksize;
    if args.numthreads == 1 {
        istacksize += args.finish as usize;
    }
	
	
    if false {
		let builder_test = thread::Builder::new().stack_size(istacksize);
		let handle_test = builder_test.spawn(move || {
			let bprint: bool = false;
			/*
			#   75576
			primes = list(sympy.primerange(2, 70))
			len([x for x in list(itertools.combinations(primes, 6)) + list(itertools.combinations(primes, 7)) + list(itertools.combinations(primes, 8)) if math.prod(x) < 2**32])
			# 8503548 (5 + 6 + 7 + 8)
			primes = sorted(3 * list(sympy.primerange(2, 30)))
			len([x for x in list(itertools.combinations(primes, 5)) + list(itertools.combinations(primes, 6)) + list(itertools.combinations(primes, 7)) + list(itertools.combinations(primes, 8)) if math.prod(x) < 2**32])
			#   66099 (3 + 4 + 5)
			#  126165 (4 + 5 + 6)
			primes = [2, 2, 2, 2, 3, 3, 3, 5] + [list(sympy.primerange(128-i, 128)) for i in range(40, 512) if len(list(sympy.primerange(128-i, 128))) == 10][0] + [list(sympy.primerange(1620, 1620+i)) for i in range(40, 512) if len(list(sympy.primerange(1620, 1620+i))) == 10][0]
			len([x for x in list(itertools.combinations(primes, 3)) + list(itertools.combinations(primes, 4)) + list(itertools.combinations(primes, 5)) if math.prod(x) < 2**32])
			len([x for x in list(itertools.combinations(primes, 4)) + list(itertools.combinations(primes, 5)) + list(itertools.combinations(primes, 6)) if math.prod(x) < 2**32])
			*/
			// bprint = true    75576 ...  6.13- 6.91 (7.45-7.92 previous get_divisors() method)
			// perf =  true, bprint = false, 8503548 ... 10.65-15.16
			// perf =  true, bprint = false,  126165 ...  3.60- 4.81
			// perf = false, bprint = false, 8503548 ...  9.85-13.46 (10.17-13.62 previous get_divisors() method)
			// perf = false, bprint = false,  126165 ...  3.17- 4.48 ( 5.74- 8.27 previous get_divisors() method)
			// this get_divisors ran for 155.86-191.34 sec, approximated_sqrt ran for 10.32-11.98 sec
			// prev get_divisors ran for 157.20-160.15 sec, approximated_sqrt ran for 10.35-12.13 sec
			const PRIMES1: [u32; 30] = [2, 2, 2, 3, 3, 3, 5, 5, 5, 7, 7, 7, 11, 11, 11, 13, 13, 13, 17, 17, 17, 19, 19, 19, 23, 23, 23, 29, 29, 29];
			const PRIMES2: [u32; 28] = [2, 2, 2, 2, 3, 3, 3, 5, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699];
			let mut comboa = PRIMES1.iter().combinations(5);
			let mut combob = PRIMES1.iter().combinations(6);
			let mut comboc = PRIMES1.iter().combinations(7);
			let mut combod = PRIMES1.iter().combinations(8);
			let mut products1: SmallVec<[u32; 131072]> = comboa.chain(combob).chain(comboc).chain(combod).map(|c| c.iter().map(|&&x| x as u64).product::<u64>()).filter(|&x| x <= u32::MAX as u64).map(|x| x as u32).collect();
			comboa = PRIMES2.iter().combinations(4);
			combob = PRIMES2.iter().combinations(5);
			comboc = PRIMES2.iter().combinations(6);
			let mut products2: SmallVec<[u32; 131072]> = comboa.chain(combob).chain(comboc).map(|c| c.iter().map(|&&x| x as u64).product::<u64>()).filter(|&x| x <= u32::MAX as u64).map(|x| x as u32).collect();
			let mut elapsed1: SmallVec<[f64; 8]> = SmallVec::new();
			let mut elapsed2: SmallVec<[f64; 8]> = SmallVec::new();
			let mut icount = 0;
			for _ in 0..8 {
				let mut inst: Instant = Instant::now();
				for p in &products1 {
					let vec = divisors::get_divisors(*p);
					if bprint { println!("get_divisors({}) = {:?}", p, vec); }
				}
				elapsed1.push((100.0 * inst.elapsed().as_secs_f64()).round() / 100.0);
				inst = Instant::now();
				icount = 0;
				while icount < products1.len() - products2.len() {
					icount += products2.len();
					for p in &products2 {
						let vec = divisors::get_divisors(*p);
						if bprint { println!("get_divisors({}) = {:?}", p, vec); }
					}
				}
				elapsed2.push((100.0 * inst.elapsed().as_secs_f64()).round() / 100.0);				
			}
			
			let min1 = elapsed1.clone().into_iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
			let max1 = elapsed1.clone().into_iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
			let mean1 = elapsed1.iter().sum::<f64>() / elapsed1.len() as f64;
			let var1 = elapsed1.iter().map(|x| (mean1 - x)*(mean1 - x)).sum::<f64>() / (elapsed1.len() as f64 - 1.0);
			
			let min2 = elapsed2.clone().into_iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
			let max2 = elapsed2.clone().into_iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
			let mean2 = elapsed2.iter().sum::<f64>() / elapsed2.len() as f64;
			let var2 = elapsed2.iter().map(|x| (mean2 - x)*(mean2 - x)).sum::<f64>() / (elapsed2.len() as f64 - 1.0);
			
			println!("get_divisors() for combination length {}", products1.len());
			println!("{:.2}-{:.2} +/- {:.2}", min1, max1, (100.0*var1.sqrt()).round()/100.0);
			println!("get_divisors() for combination length {} (icount = {})", products2.len(), icount);
			println!("{:.2}-{:.2} +/- {:.2}", min2, max2, (100.0*var2.sqrt()).round()/100.0);
			let mut test: Vec<(u32, SmallVec<[u32; 512]>)> = Vec::new();
			// sympy.divisors(2*3*3*3*5*19*23*113*233)[1:-1]
			test.push((3877083665, smallvec![5, 7, 13, 17, 19, 23, 31, 35, 37, 65, 85, 91, 95, 115, 119, 133, 155, 161, 185, 217, 221, 247, 259, 299, 323, 391, 403, 437, 455, 481, 527, 589, 595, 629, 665, 703, 713, 805, 851, 1085, 1105, 1147, 1235, 1295, 1495, 1547, 1615, 1729, 1955, 2015, 2093, 2185, 2261, 2405, 2635, 2737, 2821, 2945, 3059, 3145, 3367, 3515, 3565, 3689, 4123, 4199, 4255, 4403, 4921, 4991, 5083, 5681, 5735, 5957, 6851, 7429, 7657, 7735, 8029, 8177, 8645, 9139, 9269, 10013, 10465, 11063, 11305, 11951, 12121, 13547, 13685, 14105, 14467, 14911, 15295, 16169, 16835, 18445, 19499, 20615, 20995, 21793, 22015, 24605, 24955, 25415, 26381, 28405, 29393, 29785, 34255, 35581, 37145, 38285, 39767, 40145, 40885, 45695, 46345, 47957, 50065, 52003, 53599, 55315, 57239, 59755, 60605, 63973, 64883, 67735, 70091, 72335, 74555, 77441, 80845, 83657, 84847, 94829, 96577, 97495, 101269, 104377, 108965, 113183, 130169, 131905, 136493, 146965, 152551, 155363, 157573, 176111, 177905, 184667, 188071, 198835, 210197, 230299, 239785, 253487, 260015, 267995, 274873, 283309, 286195, 319865, 324415, 342953, 350455, 370481, 387205, 418285, 424235, 448477, 474145, 482885, 501239, 506345, 521885, 565915, 650845, 676039, 682465, 762755, 776815, 787865, 880555, 911183, 923335, 940355, 1050985, 1087541, 1103011, 1151495, 1232777, 1267435, 1316497, 1374365, 1416545, 1471379, 1612093, 1714765, 1774409, 1852405, 1924111, 1983163, 2242385, 2400671, 2506195, 2593367, 2993887, 3139339, 3380195, 3508673, 3573349, 4555915, 4816253, 5437705, 5515055, 5830201, 6163885, 6516107, 6582485, 7356895, 8060465, 8521063, 8872045, 9620555, 9915815, 12003355, 12966835, 14969435, 15696695, 17543365, 17866745, 20957209, 24081265, 25013443, 29151005, 32580535, 33713771, 40811407, 42605315, 45612749, 59647441, 104786045, 110773819, 125067215, 168568855, 204057035, 228063745, 298237205, 553869095, 775416733]));
			test.push((2*2*5*19*23*233*1753, smallvec![2, 4, 5, 10, 19, 20, 23, 38, 46, 76, 92, 95, 115, 190, 230, 233, 380, 437, 460, 466, 874, 932, 1165, 1748, 1753, 2185, 2330, 3506, 4370, 4427, 4660, 5359, 7012, 8740, 8765, 8854, 10718, 17530, 17708, 21436, 22135, 26795, 33307, 35060, 40319, 44270, 53590, 66614, 80638, 88540, 101821, 107180, 133228, 161276, 166535, 201595, 203642, 333070, 403190, 407284, 408449, 509105, 666140, 766061, 806380, 816898, 1018210, 1532122, 1633796, 2036420, 2042245, 3064244, 3830305, 4084490, 7660610, 7760531, 8168980, 9394327, 15321220, 15521062, 18788654, 31042124, 37577308, 38802655, 46971635, 77605310, 93943270, 155210620, 178492213, 187886540, 356984426, 713968852, 892461065, 1784922130]));
			test.push((2*3*5*19*23*91*1753, smallvec![2, 3, 5, 6, 7, 10, 13, 14, 15, 19, 21, 23, 26, 30, 35, 38, 39, 42, 46, 57, 65, 69, 70, 78, 91, 95, 105, 114, 115, 130, 133, 138, 161, 182, 190, 195, 210, 230, 247, 266, 273, 285, 299, 322, 345, 390, 399, 437, 455, 483, 494, 546, 570, 598, 665, 690, 741, 798, 805, 874, 897, 910, 966, 1235, 1311, 1330, 1365, 1482, 1495, 1610, 1729, 1753, 1794, 1995, 2093, 2185, 2415, 2470, 2622, 2730, 2990, 3059, 3458, 3506, 3705, 3990, 4186, 4370, 4485, 4830, 5187, 5259, 5681, 6118, 6279, 6555, 7410, 8645, 8765, 8970, 9177, 10374, 10465, 10518, 11362, 12271, 12558, 13110, 15295, 17043, 17290, 17530, 18354, 20930, 22789, 24542, 25935, 26295, 28405, 30590, 31395, 33307, 34086, 36813, 39767, 40319, 45578, 45885, 51870, 52590, 56810, 61355, 62790, 66614, 68367, 73626, 79534, 80638, 85215, 91770, 99921, 113945, 119301, 120957, 122710, 136734, 159523, 166535, 170430, 184065, 198835, 199842, 201595, 227890, 233149, 238602, 241914, 282233, 319046, 333070, 341835, 368130, 397670, 403190, 432991, 466298, 478569, 499605, 524147, 564466, 596505, 604785, 683670, 699447, 766061, 797615, 846699, 865982, 957138, 999210, 1048294, 1165745, 1193010, 1209570, 1298973, 1398894, 1411165, 1532122, 1572441, 1595230, 1693398, 2164955, 2298183, 2331490, 2392845, 2597946, 2620735, 2822330, 3030937, 3144882, 3497235, 3669029, 3830305, 4233495, 4329910, 4596366, 4785690, 5241470, 5362427, 6061874, 6494865, 6994470, 7338058, 7660610, 7862205, 8466990, 9092811, 9958793, 10724854, 11007087, 11490915, 12989730, 15154685, 15724410, 16087281, 18185622, 18345145, 19917586, 22014174, 22981830, 26812135, 29876379, 30309370, 32174562, 36690290, 45464055, 49793965, 53624270, 55035435, 59752758, 69711551, 80436405, 90928110, 99587930, 110070870, 139423102, 149381895, 160872810, 209134653, 298763790, 348557755, 418269306, 697115510, 1045673265]));
			test.push((2*3*3*3*5*19*23*113*233, smallvec![2, 3, 5, 6, 9, 10, 15, 18, 19, 23, 27, 30, 38, 45, 46, 54, 57, 69, 90, 95, 113, 114, 115, 135, 138, 171, 190, 207, 226, 230, 233, 270, 285, 339, 342, 345, 414, 437, 466, 513, 565, 570, 621, 678, 690, 699, 855, 874, 1017, 1026, 1035, 1130, 1165, 1242, 1311, 1398, 1695, 1710, 2034, 2070, 2097, 2147, 2185, 2330, 2565, 2599, 2622, 3051, 3105, 3390, 3495, 3933, 4194, 4294, 4370, 4427, 5085, 5130, 5198, 5359, 6102, 6210, 6291, 6441, 6555, 6990, 7797, 7866, 8854, 10170, 10485, 10718, 10735, 11799, 12582, 12882, 12995, 13110, 13281, 15255, 15594, 16077, 19323, 19665, 20970, 21470, 22135, 23391, 23598, 25990, 26329, 26562, 26795, 30510, 31455, 32154, 32205, 38646, 38985, 39330, 39843, 44270, 46782, 48231, 49381, 52658, 53590, 57969, 58995, 62910, 64410, 66405, 70173, 77970, 78987, 79686, 80385, 96462, 96615, 98762, 101821, 115938, 116955, 117990, 119529, 131645, 132810, 140346, 144693, 148143, 157974, 160770, 193230, 199215, 203642, 233910, 236961, 239058, 241155, 246905, 263290, 289386, 289845, 296286, 305463, 350865, 394935, 398430, 444429, 473922, 482310, 493810, 500251, 509105, 579690, 597645, 605567, 610926, 701730, 710883, 723465, 740715, 789870, 888858, 916389, 1000502, 1018210, 1184805, 1195290, 1211134, 1333287, 1421766, 1446930, 1481430, 1500753, 1527315, 1816701, 1832778, 2222145, 2369610, 2501255, 2666574, 2749167, 3001506, 3027835, 3054630, 3554415, 3633402, 4444290, 4502259, 4581945, 5002510, 5450103, 5498334, 6055670, 6666435, 7108830, 7503765, 9004518, 9083505, 9163890, 10900206, 11505773, 13332870, 13506777, 13745835, 15007530, 16350309, 18167010, 22511295, 23011546, 27013554, 27250515, 27491670, 32700618, 34517319, 45022590, 54501030, 57528865, 67533885, 69034638, 81751545, 103551957, 115057730, 135067770, 163503090, 172586595, 207103914, 310655871, 345173190, 517759785, 621311742, 1035519570, 1553279355]));
			for (n, div) in test {
				for (i, d) in divisors::get_divisors(n).iter().enumerate() {
					if div[i] != *d {
						println!("Failed test for get_divisors({})", n);
						break;
					}
				}
			}
		}).unwrap();
		handle_test.join().unwrap();
    }
	
	if false {
		if line_numbers { println!("{}() line {}", function_name!(), line!()); }
		// target\debug\sequence_rust.exe 1 "[(1,2)]" 2 65536 ratio vec --stacksize 2097152
		// target\release\sequence_rust.exe 1 "[(1,2)]" 2 134217728 ratio smallvec --stacksize 33554432
		let builder = thread::Builder::new().stack_size(args.stacksize + args.finish as usize);
		let handle_main = builder.spawn(move || {
			let mut seq1: Sequence24 = Sequence24::new(args.finish as usize, false, false, args.datatype, args.method);
			for n in 2..512 {
				for this_combination in seq1.factor_combinations_vec(n) {
					//let tvec: TinyVec<[i32; 24]> = this_combination.iter().map(|&x| x as i32).collect();
					let this_density: Ratio<i32> = seq1.calc_density_xor(n as usize, &this_combination);
					if *this_density.denom() <= 5 {
						println!("{}\t{}\t{:?}", this_density, n, this_combination);
					}
				}
			}
			println!("{}() line {}", function_name!(), line!()); 
			for n in [108126720, 111767040, 131155200] {
				for this_combination in seq1.factor_combinations_smallvec(n) {
					let this_density: Ratio<i32> = seq1.calc_density_or(n as usize, &this_combination);
					if *this_density.denom() <= 5 {
						println!("{}\t{}\t{:?}", this_density, n, this_combination);
					}
				}
			}
		}).unwrap();
		handle_main.join().unwrap();
		//return;
	}
			
	/*
	if false {
		min_factors_len = 2;
		for n in [720, 840, 6720] {
			for this_combination in seq1.factor_combinations_vec(n as u32) {
				let this_density: Ratio<i32> = seq1.calc_density(&array_vec);                    
				//if let Some(this_ratio) = vecratios1.iter().find(|&x| *x == this_density) {
				if *this_density.denom() <= 3 {
					println!("{} {} {:?}", this_density, n, this_combination);
				}
			}
		}
		min_factors_len = 4;
	}
	*/	
	
    let mut vec_types: TinyVec<[DataType; 4]> = TinyVec::new();
    //for dt in [DataType::ARRAY, DataType::VEC, DataType::TINYVEC, DataType::ARRAYVEC] {
	for dt in [DataType::VEC, DataType::TINYVEC, DataType::ARRAYVEC] {
        if args.datatype.is_set(dt) {
            vec_types.push(dt);
        }
    }
    
    let (tx1, rx1) = mpsc::channel::<Option<Message>>();
    let outmap2 = if args.numthreads == 1 { Arc::clone(&outmap1) } else { outmap1.clone() };    
    for ratio in &vecratios1 {
        if *ratio.denom() > 4 && *ratio.denom() >= args.start as i32 && *ratio.numer() == 1 {
            let n = *ratio.denom();
            let ratiovec: RatioVec = RatioVec{ n: n, ratio: *ratio, slice: [n, 0, 0, 0, 0, 0, 0, 0, 0, 0] };
            let msg: String = ratiovec.to_string();
            outmap2.lock().unwrap().get_mut(&ratio).unwrap().push(n as u32);
            println!("{}", msg);
            for vt in 0..vec_types.len() {
                tx1.send(Some(Message { threadid: 1, datatype: vec_types[vt], calcdensity: args.method, n: n as u32, ratio: *ratio, msg: msg.clone() }));
            }
        }
    }
	
	
	let builder_main = thread::Builder::new().stack_size(istacksize);
    
	if args.numthreads == 1 {
        
		let handle_main = builder_main.spawn(move || {
			let mut seq1: Sequence24 = Sequence24::new(std::cmp::max(args.finish as usize, 8192), false, false, args.datatype, args.method);
            let seq1_bln_debug = false;
            seq1.bln_debug = seq1_bln_debug;
            seq1.bln_perf = args.perf;
            seq1.bln_divisor_gen = false;
            seq1.bln_exhaustive_search = false;
			seq1.bln_factors = true;
			seq1.bln_divisors = false;
			seq1.bln_gt_half = bln_gt_half;
			seq1.min_factors_len = min_factors_len;
			seq1.set_primes(&vecprimes1);
			
			let mut m = Main { 
				debug: args.debug,
                logging: args.logging,
				ithread: 0,
				t1: t1, 
				anumthreads: AtomicU32::new(args.numthreads as u32), 
				astart: AtomicU32::new(args.start), 
				afinish: AtomicU32::new(args.finish), 
				calcdensity: args.method,
				datatype: args.datatype,
				matches: vecratios1,
				predefined: predefined1,
				outmap: outmap2,
				tx: if args.file { Some(tx1) } else { None },
			};
			_ = print_memory(function_name!(), &pid);
			let (mut max_stack, mut max_combinations): (Vec<usize>, Vec<usize>) = m.do_work(args.start, args.finish, 1, seq1);
			if args.logging {
				println!("max_stack = {:?}", max_stack);
				println!("max_combinations = {:?}", max_combinations);
			}
            if seq1_bln_debug {
                let len = max_combinations.len();
                let avg = max_combinations.iter().map(|&x| x as f64).sum::<f64>() / len as f64;
                let mdn90 = max_combinations[90 * len / 100];
                let mdn95 = max_combinations[95 * len / 100];
                let mdn98 = max_combinations[98 * len / 100];
                let mdn99 = max_combinations[99 * len / 100];
				println!("max_combinations, {}, len = {}, avg = {:.2}, mdn90 = {}, mdn95 = {}, mdn98 = {}, mdn99 = {}", args.finish, len, avg, mdn90, mdn95, mdn98, mdn99);
			}
		}).unwrap();
		
		//if true { println!("{}() line {}, args.file = {}", function_name!(), line!(), args.file); }
		if args.file {
			let builder_file = thread::Builder::new();
			let handle_file = builder_file.spawn(move || {
				let mut messages: Vec<Message> = Vec::new();
				/*
				if fs::metadata(args.filepath.clone().unwrap()).is_ok() {
					println!("File exists {}", args.filepath.clone().unwrap());
				}
				*/
				let mut file = OpenOptions::new().create(true).append(true).open(args.filepath.clone().unwrap()).unwrap();
				//if true { println!("{}() line {}", function_name!(), line!()); }
				for msg in rx1.iter() {
					if msg.is_none() {
						//println!("{}() line {}, msg.is_none()", function_name!(), line!());
						break;
					} else {
						messages.push(msg.unwrap());
						//println!("messages.len() = {}", messages.len());
						if messages.len() >= 8 {
							messages.sort_by_key(|msg| (msg.n, *msg.ratio.denom()));
							let messages_drain: Vec<Message> = messages.drain(..8).collect();
							for msg in messages_drain {
								//println!("{}", msg.msg);
								if !msg.msg.starts_with("thread") {
									if let Err(e) = writeln!(file, "{}", msg.msg) {
										println!("Failed to write to file: {}", e);
										file = OpenOptions::new().create(true).append(true).open(args.filepath.clone().unwrap()).unwrap();
										writeln!(file, "{}", msg.msg);
									}
								}
							}
						}
					}
				}
				if messages.len() > 0 {
					messages.sort_by_key(|msg| (msg.n, *msg.ratio.denom()));
					for msg in messages {
						//println!("{}", msg.msg);
						if !msg.msg.starts_with("thread") {
							writeln!(file, "{}", msg.msg);
						}
					}
				}
			}).unwrap();
			handle_file.join().unwrap();
		}
		
		handle_main.join().unwrap();
				
		
	} else {
	
		let handle_main = builder_main.spawn(move || {
			let mut seq1: Sequence24 = Sequence24::new(std::cmp::max(args.finish as usize, 8192), false, false, args.datatype, args.method);
            seq1.bln_debug = false;
            seq1.bln_perf = args.perf;
            seq1.bln_divisor_gen = false;
            seq1.bln_exhaustive_search = false;
			seq1.bln_factors = true;
			seq1.bln_divisors = false;
			seq1.bln_gt_half = bln_gt_half;
			seq1.min_factors_len = min_factors_len;
			seq1.set_primes(&vecprimes1);
			
			//let mut mem1 = Arc::new(LockedBool::new(false));
			let mut mem1 = LockedBool::new(false);
			let (tx1, rx1) = mpsc::channel::<Option<Message>>();
			thread::scope(|scp| 
			{
				let mut threads = vec![];
				for ith in 0..(args.numthreads as usize)
				{					
					let mut seq2: Sequence24 = Sequence24::new(seq1.capacity, seq1.global, seq1.resize, seq1.datatype, seq1.calcdensity);
                    let seq2_bln_debug = seq1.bln_debug;
                    seq2.bln_debug = seq2_bln_debug;
                    seq2.bln_perf = seq1.bln_perf;
                    seq2.bln_divisor_gen = seq1.bln_divisor_gen;
                    seq2.bln_exhaustive_search = seq1.bln_exhaustive_search;
					seq2.min_factors_len = min_factors_len;
					seq2.bln_factors = seq1.bln_factors;
					seq2.bln_divisors = seq1.bln_divisors;
					seq2.bitprimes = seq1.bitprimes.clone();
					seq2.factors = Arc::clone(&seq1.factors);
					seq2.factor_slices = Arc::clone(&seq1.factor_slices);
					seq2.divisors = Arc::clone(&seq1.divisors);
					//let vecprimes2: Arc<Vec<u32>> = Arc::clone(&vecprimes1);
					let vecratios2: Vec<Ratio<i32>> = vecratios1.clone();
					let predefined2: HashMap<u32, Vec<RatioVec>, RandomState> = predefined1.clone();
					let outmap3 = Arc::clone(&outmap2);
					let tx2 = tx1.clone();
					//println!("{}() line {}", function_name!(), line!());
					threads.push(scp.spawn(move || {
						let mut m = Main { 
							debug: args.debug,
                            logging: args.logging,
							ithread: ith as u32,
							t1: t1, 
							anumthreads: AtomicU32::new(args.numthreads as u32), 
							astart: AtomicU32::new(args.start), 
							afinish: AtomicU32::new(args.finish), 
							calcdensity: args.method,
							datatype: args.datatype,
							matches: vecratios2,
							predefined: predefined2,
							outmap: outmap3,
							tx: Some(tx2),
						};
						let (mut max_stack, mut max_combinations): (Vec<usize>, Vec<usize>) = m.do_work(args.start, args.finish, args.numthreads as u32, seq2);
						println!("maxstack = {:?}", max_stack);
						if args.logging {
							println!("maxcombinations = {:?}", max_combinations);
							//println!("{}() line {} thread {}", function_name!(), line!(), ith);
						}
                        if seq2_bln_debug {
                            let len = max_combinations.len();
                            let avg = max_combinations.iter().map(|&x| x as f64).sum::<f64>() / len as f64;
                            let mdn90 = max_combinations[90 * len / 100];
                            let mdn95 = max_combinations[95 * len / 100];
                            let mdn98 = max_combinations[98 * len / 100];
                            let mdn99 = max_combinations[99 * len / 100];
                            println!("max_combinations, {}, len = {}, avg = {:.2}, mdn90 = {}, mdn95 = {}, mdn98 = {}, mdn99 = {}", args.finish, len, avg, mdn90, mdn95, mdn98, mdn99);
                        }
					}));
				}
				
				if !mem1.lock_and_load() {
					mem1.store(print_memory(function_name!(), &pid));
				}
				mem1.unlock();

				//let rx2 = rx1.clone();
				//let rx2 = Arc::new(Mutex::new(rx1));
				threads.push(scp.spawn(move || {
					let mut messages: Vec<Message> = Vec::new();
					let mut file: Option<std::fs::File> = if args.file { Some(OpenOptions::new().create(true).append(true).open(args.filepath.clone().unwrap()).unwrap()) } else { None };
					let mut ithread = 0;
                    let mut vecmax: Vec<u32> = Vec::new();
                    for _ in 0..args.numthreads {
                        vecmax.push(0);
                    }
					//println!("{}() line {}", function_name!(), line!());
					for msg in rx1.iter() {
						//println!("{}() line {}", function_name!(), line!());
						if msg.is_none() {
							ithread += 1;
							if ithread >= args.numthreads {
								println!("ithread = {}, inumthreads = {}", ithread, args.numthreads);
								break;
							}
						} else {
							messages.push(msg.unwrap());
                            let n = messages[messages.len() - 1].n;
                            let idx = (messages[messages.len() - 1].threadid - 1) as usize;
                            if n > vecmax[idx] { vecmax[idx] = n; }
                            let minmax = vecmax.iter().min().unwrap();
                            
                            let mut messages_extract: Vec<Message> = messages.extract_if(.., |msg| msg.n <= *minmax).collect();                            
							//println!("{}() messages.len() = {}", function_name!(), messages.len());
                            //println!("{}() messages_extract.len() = {}", function_name!(), messages_extract.len());
							if messages_extract.len() > 0 {
								messages_extract.sort_by_key(|msg| (msg.n, *msg.ratio.denom()));
								for msg in messages_extract {
                                    if !msg.msg.starts_with("thread") || args.logging {
                                        println!("{}", msg.msg);
                                    }
									if args.file && !msg.msg.starts_with("thread") && let Some(ref mut f1) = file {
										if let Err(e) = writeln!(f1, "{}", msg.msg) {
											println!("Failed to write to file: {}", e);
											file = Some(OpenOptions::new().create(true).append(true).open(args.filepath.clone().unwrap()).unwrap());
											if let Some(ref mut f2) = file {
												writeln!(f2, "{}", msg.msg);
											}
										}
									}
								}
							}
						}
					}
					if messages.len() > 0 {
						messages.sort_by_key(|msg| (msg.n, *msg.ratio.denom()));
						for msg in messages {
                            if !msg.msg.starts_with("thread") || args.logging {
                                println!("{}", msg.msg);
                            }
							if args.file && !msg.msg.starts_with("thread") && let Some(ref mut f) = file {
								writeln!(f, "{}", msg.msg);
							}
						}
					}
				}));
								
				// 
				// https://stackoverflow.com/questions/68966949/unable-to-join-threads-from-joinhandles-stored-in-a-vector-rust
				// 
				for th in threads 
				{
					let _ = th.join();
				}
				
			});
			
		}).unwrap();
		
		handle_main.join().unwrap();
	}
	
	for (key, val) in outmap1.lock().unwrap().iter_mut() {
		val.sort();
		println!("{}/{}\t{:?}", key.numer(), key.denom(), val);
	}
    // 1 thread ... 0.70 minutes ... 10.05 minutes
    // 2 threads .. 0.41 minutes .... 6.77 minutes
    // 4 threads .. 0.38 minutes
	print_time(function_name!(), "end_time");
	let total_sec = t1.elapsed().as_secs_f64();
    print_elapsed(total_sec, args.start, args.finish, args.numthreads);
	
	/*
	
	target\release\sequence_rust.exe 1 "[(1,2)]" 2 1048576 ratio tinyvec --perf
	
    1048576 release, bln_divisor_gen = false (87-96 seconds)
                   get_divisors ..... 13.4-17.9 (15.3-18.6%)
    factor_combinations_tinyvec ..... 75.6-83.4 (85.8-87.3%)
                        do_work ..... 86.0-94.4 (98.1-99.3%)
                       mult_ary ...... 1.7- 2.0 ( 1.9- 2.1%)
             calc_density_ratio ...... 6.9- 8.5 ( 7.6- 8.8%)
                         output .......0.1- 0.2 ( 0.1- 0.2%)
	
	1048576 release, bln_divisor_gen = true (101-102 seconds)
  				    divisor_gen ..... 19.0 (18.5-18.8%)
				   get_divisors ..... 10.4 (10.1-10.3%)
	factor_combinations_tinyvec ..... 88.5 (86.5-87.8%)
						do_work .... 100.4 (98.1-99.3%)
					   mult_ary ...... 1.8 (      1.8%)
			 calc_density_ratio ...... 7.8 ( 7.6- 7.8%)
						 output ...... 0.1 (      0.1%)
	
	target\release\sequence_rust.exe 1 "[(1,2)]" 2 1048576 ratio smallvec --perf --stacksize 1000000
	
	1048576 release, bln_divisor_gen = false (78-87 seconds) i7-1165G7 @ 2.80GHz
	
				    get_divisors ..... 14.8-17.3 (19.8%)
	factor_combinations_smallvec ..... 67.7-77.2 (88.3%)
					  	 do_work ..... 76.1-86.7 (99.1%)
					    mult_ary ...... 1.8- 2.0 ( 2.3%)
			  calc_density_ratio ...... 7.4- 8.3 ( 9.5%)
						  output ......      0.1 ( 0.2%)
	
	target\release\sequence_rust.exe 1 "[(1,5)]" 5 1048576 ratio smallvec --perf

	1048576 release, bln_divisor_gen = false (48 seconds) i7-1360P

	factor_combinations_smallvec ..... 39.5 (82.5%)
						 do_work ..... 47.0 (98.1%)
				    get_divisors ...... 5.7 (12.0%)
					    mult_ary ...... 1.5 ( 3.0%)
			  calc_density_ratio ...... 6.1 (12.7%)
			 			  output ...... 0.6 ( 1.3%)
					 
	   array release from 2 to 1048576 with 1 threads in 109.76 seconds ( 1.83 minutes) RATIO
	   array release from 2 to 4194304 with 1 threads in 888.47 seconds (14.81 minutes) RATIO
	arrayvec release from 2 to 4194304 with 1 threads in 913.55 seconds (15.23 minutes) RATIO
	arrayvec release from 2 to 4194304 with 1 threads in 876.43 seconds (14.61 minutes) RATIO
     tinyvec release from 2 to 1048576 with 4 threads in 79   seconds ( 1.31 minutes) RATIO
	 
	 tinyvec __debug from 2 to 1048576 with 1 threads in 3867 seconds (64.46 minutes) RATIO ~ 16,268 per min
     tinyvec __debug from 2 to 1048576 with 1 threads in 1099 seconds (18.32 minutes) RATIO ~ 57,244 per min
	smallvec release from 2 to 8388608 with 1 threads in 4405 seconds (73.42 minutes) RATIO ~ 114,250 per min
	
	target\release\sequence_rust.exe 1 "[(1,2)]" 549120 591360 ratio smallvec
 
	   array from 549120 to 591360 with 1 threads in  2.83-12.15-34.41 seconds
	 tinyvec from 549120 to 591360 with 1 threads in  3.21-3.63 seconds
	arrayvec from 549120 to 591360 with 1 threads in  2.67-2.84 seconds
	smallvec from 549120 to 591360 with 1 threads in  3.14-4.28 seconds
	     vec from 549120 to 591360 with 1 threads in  4.87-71.92-97.05-135.97 seconds
	
	i7-1360P @ 2.20GHz
	python.exe "E:\Python\Sequence\sequence_th.py" 1 [(1,5)] 118918800 119024640
	2.24-2.48 min ~ 42,651-47,155 per min
	# 0.11 total minutes ( 4.67%) factorCombinations()
	# 0.15 total minutes ( 6.44%) factorizations_outer()
	# 1.95 total minutes (81.42%) calc_density()

	i7-1360P @ 2.20 GHz
	target\release\sequence_rust.exe 1 "[(1,5)]" 118918800 119024640 ratio smallvec
	7.06 min ~ 14,993 per min ... smallvec
	7.19 min ~ 14,713 per min ... tinyvec
	                get_divisors ...  13.9 ( 3.2%)
	factor_combinations_smallvec ... 416.2 (96.1%)
                         do_work ... 421.0 (97.2%)
                        mult_ary ...   1.1 ( 0.3%)
              calc_density_ratio ...   4.4 ( 1.0%)
	
	33,554,432
	max_stack = maxstack = [2, 3, 4, 6, 10, 14, 16, 18, 22, 25, 26, 28, 30, 34, 38, 46, 58, 62, 70, 78, 82, 88, 94, 98, 106, 118, 126, 142, 158, 166, 178, 190, 198, 214, 222, 238, 254, 286, 318, 334, 358, 382, 398, 430, 446, 478, 502, 510, 574, 598]
	
	max_combinations = [457, 641, 819, 916] // 916*24 = 21984
	max_combinations = [141, 169, 218, 264, 273, 391, 491, 721, 806, 819, 1182, 1291, 1551, 1851, 2150, 2574, 2822, 4101, 4113, 5058]
    max_combinations, 1048576, len =  90889, avg = 25.67, mdn90 = 49, mdn95 =  91, mdn98 = 169, mdn99 = 264
    max_combinations, 4194304, len = 441940, avg = 36.58, mdn90 = 88, mdn95 = 151, mdn98 = 273, mdn99 = 391
	max_combinations, 8388608, len = 514694, avg = 49.15, mdn90 = 91, mdn95 = 169, mdn98 = 391, mdn99 = 689
	
    divisors, 1048576, average = 6.04, median90 = 10, median95 = 14, median98 = 22, median99 = 30
	divisors, 8388608, average = 7.39, median90 = 14, median95 = 18, median98 = 22, median99 = 30
	
	*/
    if args.perf {
        let graph = time_graph::get_full_graph();
		println!("{}", graph.as_dot());
		for timed_span in graph.spans() {
			let secs = timed_span.elapsed.as_secs_f64();
            let dot = format!("... {:.1}", (10.0*secs).round()/10.0);
            let pct = format!("{:.1}", (10.0*100.0*secs/total_sec).round()/10.0);
			println!("{:>27} {:.>10} ({:>4}%)", timed_span.callsite.name(), dot, pct);
		}
        let (avg, mdn90, mdn95, mdn98, mdn99): (f64, u16, u16, u16, u16) = divisors::statistics();
        println!("divisors, {}, average = {:.2}, median90 = {}, median95 = {}, median98 = {}, median99 = {}", args.finish, (100.0*avg).round()/100.0, mdn90, mdn95, mdn98, mdn99);
    }
}



