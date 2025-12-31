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

mod divisors;
mod sequence24;
mod sequence2;

use ahash::{AHasher, AHashMap, AHashSet, HashSetExt, RandomState};
//use bit_vec::BitVec;
use chrono::{Local, Timelike};
use clap::{Parser, Subcommand, ValueEnum};
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
use sequence24::{CalcDensityType, DataType, Sequence24};
//use shared_memory::*;
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
use std::fs::OpenOptions;
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

#[derive(Copy, Clone, PartialEq)]
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
    // 
    // linux debug factor_combinations()
    //  (50,000)  2.03- 2.13 minutes ~ 23,437-24,590 per min
    // (100,000)  6.08- 6.27 minutes ~ 15,957-16,438 per min
    // (150,000) 12.45-12.82 minutes ~ 11,704-12,048 per min
    // (200,000) 21.73-21.73 minutes ~  9,202- 9,202 per min
    // (250,000) 32.47-33.12 minutes ~  7,549-7,700 per min
    // (300,000) 45.48 minutes ~  6,596 per min
    // (350,000) 57.43 minutes ~  6,094 per min
    // (400,000) 70.95 minutes ~  5,638 per min
    // 
    // 
    // windows release factor_combinations()
	// (1,000,000)    1.55 mins ~ 645,161 per min
	// (2,000,000)    4.50 mins ~ 444,444 per min
	// (3,000,000)    8.60 mins ~ 348,837 per min
    // (10,000,000)  54.00 mins ~ 185,189 per min
    // (20,000,000) 130.65 mins ~ 153,084 per min
    // (30,000,000) 227.60 mins ~ 131,813 per min
    // 
    // factor_combinations2()
    // 
    // 
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
pub fn do_work(&mut self, mut nstart: u32, nfinish: u32, inumthreads: u32, mut seq: Sequence24) -> Vec<usize>
{
    /*
    let mut seq: Sequence24 = Sequence24::new(2, nfinish as usize, false);
    seq.bln_factors = true;
    seq.bln_divisors = false;
    seq.set_primes(&primes);
    */

    let nstepby: u32 = if self.matches.len() == 1 && *self.matches[0].denom() == 2 { 2 } else { 1 };
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
	
    let bln_array = self.datatype.is_set(DataType::ARRAY);
	let bln_vec = self.datatype.is_set(DataType::VEC);
	let bln_tinyvec = self.datatype.is_set(DataType::TINYVEC);
	let bln_arrayvec = self.datatype.is_set(DataType::ARRAYVEC);
    let mut vec_types: TinyVec<[DataType; 3]> = TinyVec::new();
    for dt in [DataType::VEC, DataType::TINYVEC, DataType::ARRAYVEC] {
        if self.datatype.is_set(dt) {
            vec_types.push(dt);
        }
    }
    if vec_types.len() == 1 {
        seq.datatype = vec_types[0];
    }
    
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
                        if bln_array {
                            if let Some(tx) = &self.tx {
                                tx.send(Some(Message { threadid: self.ithread + 1, datatype: DataType::ARRAY, calcdensity: seq.calcdensity, n: n, ratio: ratiovec.ratio, msg: msg.clone() }));
                            }
                        }
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
	return seq.max_combinations;
}
}



fn init(maxprime: u32) -> Vec<u32>
{
    let mut primes: Vec<u32> = Vec::<u32>::new();
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


/*

12, 168, 240, 13,440, 14,880, 65,280, 549,120, 591,360, 591,360, 833,280, 954,240, 
1,145,760, 1,317,120, 1,666,560, 1,908,480, 2,143,680, 2,204,160, 3,655,680, 3,886,080, 
4,408,320, 6,990,720, 8,094,720, 17,149,440, 25,259,520, 39,443,712, 39,621,120

1/2    3, 4    12
1/2    3, 7, 8    168
1/2    3, 5, 16    240
1/2    3, 7, 10, 32    6,720
1/2    3, 5, 28, 32    13,440
1/2    3, 5, 31, 32    14,880
1/2    3, 5, 17, 256    65,280
1/2    3, 10, 11, 13, 128    549,120
1/2    3, 7, 11, 32, 80    591,360
1/2    3, 7, 11, 40, 64    591,360
1/2    3, 7, 10, 62, 64    833,280
1/2    3, 7, 10, 64, 71    954,240
1/2    3, 7, 11, 32, 155    1,145,760
1/2    3, 5, 32, 49, 56    1,317,120
1/2    3, 5, 31, 56, 64    1,666,560
1/2    3, 5, 28, 64, 71    1,908,480
1/2    3, 7, 11, 29, 320    2,143,680
1/2    3, 7, 10, 41, 256    2,204,160
1/2    3, 7, 10, 34, 512    3,655,680
1/2    3, 5, 23, 64, 176    3,886,080
1/2    3, 5, 28, 41, 256    4,408,320
1/2    3, 5, 22, 64, 331    6,990,720
1/2    3, 5, 31, 34, 512    8,094,720

*/


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
	#[arg(long, aliases = ["stack", "stacksize", "stack_size"], default_value_t = 64*32*1024)]
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
ary = []
with open(f"{dir}\\sequence (1,2).txt") as f:
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
 * target\release\sequence_rust.exe 1 "[(1,2)]" 2 1048576 RATIO array --file
 * target\release\sequence_rust.exe 1 "[(1,2)]" 2 4194304 OR array --file --stacksize 2097152
 * target\release\sequence_rust.exe 1 "[(1,2)]" 2 8388608 OR arrayvec
 * 33554432
 * 134217728
 * target\release\sequence_rust.exe 1 "[(1,2)]" 4194304 8388608 --file --array --stacksize 1048576
 * target\release\sequence_rust.exe 4 "[(1,2), (1,3), (1,4)]" 2 8388608 ratio tinyvec --file
 * target\release\sequence_rust.exe 1 "[(1,2)]" 2 8388608 RATIO tinyvec --file --logging
 * 
 * target\debug\sequence_rust.exe 1 "[(1,2)]" 549120 591360 ratio array|vec --perf
 * target\debug\sequence_rust.exe 1 "[(1,2)]" 3655680 3886080 ratio vec|arrayvec --perf
 * target\release\sequence_rust.exe 1 "[(1,2)]" 549120 591360 ratio array|vec --perf
 * target\release\sequence_rust.exe 1 "[(1,2)]" 3655680 3886080 ratio array|tinyvec --perf
 * 
 *                     [(1,3)] from 2 to 4194304 with 4 threads in 4299.50 minutes (71.66 hours)
 * i7-1165G7 @ 2.80GHz [(1,3)] from 2 to 4194304 with 1 thread  in   20.50 minutes ( 0.34 hours)
 * i7-1165G7 @ 2.80GHz [(1,2)] from 2 to 1048576 with 1 thread  in    1.83 minutes (RATIO array   )
 * i7-1165G7 @ 2.80GHz [(1,2)] from 2 to 1048576 with 1 thread  in    2.26 minutes (RATIO arrayvec)
 * i7-1165G7 @ 2.80GHz [(1,2)] from 2 to 1048576 with 1 thread  in   17.82 minutes (   OR arrayvec) 17.82/2.26 = 7.88
 * i7-1165G7 @ 2.80GHz [(1,2)] from 2 to 1048576 with 1 thread  in    2.45 minutes (RATIO tinyvec )
 * i7-1165G7 @ 2.80GHz [(1,2)] from 2 to 1048576 with 1 thread  in   21.39 minutes (   OR tinyvec ) 21.39/2.45 = 8.73
 * i7-1165G7 @ 2.80GHz [(1,2)] from 2 to 1048576 with 1 thread  in    2.83 minutes (RATIO array   )
 * i7-1165G7 @ 2.80GHz [(1,2)] from 2 to 1048576 with 1 thread  in    3.36 minutes (RATIO vec     )
 * 
 *                     [(1,2)] (1,000,000) 28.5 mins ~ 35,108 per min
 *                     [(1,2)] from 2 to 1048576 with 1 thread  in   30.7 minutes ( 0.51 hours)
 *                     [(1,2)] from 2 to 1048576 with 4 threads in   27.6 minutes ( 0.46 hours)
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

#[function_name::named]
fn main() 
{
	let line_numbers: bool = false;
	let mut bln_gt_half: bool = false;
	let mut min_factors_len: usize = 4;
	
    let output = Command::new("rustc").arg("--version").output().unwrap();
    println!("{}() rust_version=\"{}\"", function_name!(), String::from_utf8_lossy(&output.stdout).trim_end());
    let cpuid = CpuId::new();
    if let Some(brand) = cpuid.get_processor_brand_string() {
        println!("{}() processor=\"{}\"", function_name!(), brand.as_str().trim());
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
        println!("{:#?}", args);
    }
	if line_numbers { println!("{}() line {}", function_name!(), line!()); }
    if args.perf {
        time_graph::enable_data_collection(true);
    }
    
	let t1: Instant = Instant::now();
	let pid = Pid::from_u32(process::id());
    //let inumthreads: u8 = vecargs[1].parse::<u8>().ok().unwrap();
    //let mut strratios1: String = vecargs[2].clone();
	args.ratios = args.ratios.replace("[", "").replace("]", "").replace(" ", "");
    //let istart: u32 = vecargs[3].parse::<u32>().ok().unwrap();
    //let ifinish: u32 = vecargs[4].parse::<u32>().ok().unwrap();
	
	if line_numbers { println!("{}() line {}", function_name!(), line!()); }
    // let mut setprimes1: Arc<BTreeSet<i64>> = Arc::new(init(args.finish.ilog2() + 1));
    let mut vecprimes1: Arc<Vec<u32>> = Arc::new(init(args.finish as u32));
    // new(i: u32, capacity: usize, global: bool, resize: bool)

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
	
    /*
    let mut seq = Sequence24::new(1049520, 1049520, true);
    seq.set_primes(vecprimes1.clone());
    let vecvec1 = seq.factor_combinations(1049520);
    println!("vecvec1 = {:?}", vecvec1);
    return;
    */
    //test_combinations(vecprimes1.clone(), true, true);
    //test_combinations(vecprimes1.clone(), false, true);
    //test_combinations(vecprimes1.clone(), true, false);
    //test_divisors(vecprimes1.clone());
    //return;
    
	
	if false {
		if line_numbers { println!("{}() line {}", function_name!(), line!()); }
		// target\debug\sequence_rust.exe 1 "[(1,2)]" 2 65536 --vec --stacksize 2097152
		let builder = thread::Builder::new().stack_size(args.stacksize + args.finish as usize);
		let outer_handle = builder.spawn(move || {
			let mut seq1: Sequence24 = Sequence24::new(65536, false, false, args.datatype, args.method);
			for n in 2..512 {
				for this_combination in seq1.factor_combinations_vec(n) {
					//let tvec: TinyVec<[i32; 24]> = this_combination.iter().map(|&x| x as i32).collect();
					let this_density: Ratio<i32> = seq1.calc_density_xor(n as usize, &this_combination);
					if *this_density.denom() <= 3 {
						println!("{} {} {:?}", this_density, n, this_combination);
					}
				}
			}
		}).unwrap();
		outer_handle.join().unwrap();
		return;
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
	
	if line_numbers { println!("{}() line {}", function_name!(), line!()); }
    println!("{}() num_threads={}, vec_ratios=[{}], start={}, finish={}, method={}, stacksize={}", function_name!(), args.numthreads, args.ratios.replace(" ", ""), args.start.separate_with_commas(), args.finish.separate_with_commas(), args.method.to_string(), args.stacksize.separate_with_commas());
    println!("{}() vec={}, array={}, tinyvec={}, arrayvec={}", function_name!(), args.datatype.is_set(DataType::VEC), args.datatype.is_set(DataType::ARRAY), args.datatype.is_set(DataType::TINYVEC), args.datatype.is_set(DataType::ARRAYVEC));
	println!("{}() logging={}, debug={}, perf={}, file={}{}", function_name!(), args.logging, args.debug, args.perf, args.file, if args.file && let Some(ref fp) = args.filepath { format!(", file_path=\"{}\"", fp) } else { "".to_string() });
    
    let mut vec_types: TinyVec<[DataType; 4]> = TinyVec::new();
    for dt in [DataType::ARRAY, DataType::VEC, DataType::TINYVEC, DataType::ARRAYVEC] {
        if args.datatype.is_set(dt) {
            vec_types.push(dt);
        }
    }
    
    let (tx1, rx1) = mpsc::channel::<Option<Message>>();
    let outmap2 = if args.numthreads == 1 { Arc::clone(&outmap1) } else { outmap1.clone() };    
    for ratio in &vecratios1 {
        if *ratio.denom() > 4 && *ratio.numer() == 1 {
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
    
    let mut istacksize = args.numthreads as usize * args.stacksize;
    if args.numthreads == 1 {
        istacksize += args.finish as usize;
    }
    let builder = thread::Builder::new().stack_size(istacksize);
    
	if args.numthreads == 1 {
        
		let outer_handle = builder.spawn(move || {
			let mut seq1: Sequence24 = Sequence24::new(std::cmp::max(args.finish as usize, 8192), false, false, args.datatype, args.method);
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
			let max_combinations: Vec<usize> = m.do_work(args.start, args.finish, 1, seq1);
			if args.debug {
				println!("maxcombinations = {:?}", max_combinations);
			}
			
			if args.file {
				let mut messages: Vec<Message> = Vec::new();
				let mut file = OpenOptions::new().create(true).append(true).open(args.filepath.unwrap()).unwrap();
				for msg in rx1.iter() {
					if msg.is_none() {
						break;
					} else {
						messages.push(msg.unwrap());
						if messages.len() >= 8 {
							messages.sort_by_key(|msg| (msg.n, *msg.ratio.denom()));
                            let messages_drain: Vec<Message> = messages.drain(..4).collect();
							for msg in messages_drain {
								//println!("{}", msg.msg);
                                if !msg.msg.starts_with("thread") {
                                    writeln!(file, "{}", msg.msg);
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
			}
		}).unwrap();
		
		outer_handle.join().unwrap();
		
	} else {
	
		let outer_handle = builder.spawn(move || {
			let mut seq1: Sequence24 = Sequence24::new(std::cmp::max(args.finish as usize, 8192), false, false, args.datatype, args.method);
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
						let max_combinations: Vec<usize> = m.do_work(args.start, args.finish, args.numthreads as u32, seq2);
						if args.debug {
							println!("maxcombinations = {:?}", max_combinations);
							//println!("{}() line {} thread {}", function_name!(), line!(), ith);
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
					let mut file: Option<std::fs::File> = if args.file { Some(OpenOptions::new().create(true).append(true).open(args.filepath.unwrap()).unwrap()) } else { None };
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
									if args.file && !msg.msg.starts_with("thread") && let Some(ref mut f) = file {
										writeln!(f, "{}", msg.msg);
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
		
		outer_handle.join().unwrap();
	}
	
	for (key, val) in outmap1.lock().unwrap().iter_mut() {
		val.sort();
		println!("{}/{}\t{:?}", key.numer(), key.denom(), val);
	}
    let total_sec = t1.elapsed().as_secs_f64();
    let total_min = total_sec/60.0;
    let total_hrs = total_min/60.0;
    let count = (args.finish - args.start) as f64;
    
    // 1 thread ... 0.70 minutes ... 10.05 minutes
    // 2 threads .. 0.41 minutes .... 6.77 minutes
    // 4 threads .. 0.38 minutes
	print_time(function_name!(), "end_time");
    if total_sec < 60.0 {
        println!("finished from {} to {} with {} threads in {:.2} seconds ({:.2} minutes) ({:.2} hours) ~ {} per min", args.start, args.finish, args.numthreads, total_sec, total_min, total_hrs, (count/total_min).round().separate_with_commas());
    } else {
        println!("finished from {} to {} with {} threads in {} seconds ({:.2} minutes) ({:.2} hours) ~ {} per min", args.start, args.finish, args.numthreads, total_sec.round() as i64, total_min, total_hrs, (count/total_min).round().separate_with_commas());
    }
    
	/*
	divisor_gen ... 36.97 (19.19%)
	factor_combinations ... 211.34 (109.68%)
	mult_ary ... 0.12 (0.06%)
	calc_density ... 1.08 (0.56%)
	
	factor_combinations_ary ... 1672.29 (7.75%)
	factor_combinations_tinyvec ... 19907.35 (92.25%)
	
	   array from 2 to 1048576 with 1 threads in 109.76 seconds ( 1.83 minutes) RATIO
	   array from 2 to 4194304 with 1 threads in 888.47 seconds (14.81 minutes) RATIO
	arrayvec from 2 to 4194304 with 1 threads in 913.55 seconds (15.23 minutes) RATIO
	arrayvec from 2 to 4194304 with 1 threads in 876.43 seconds (14.61 minutes) RATIO
	   
	   array from 549120 to 591360 with 1 threads in  2.83-12.15-34.41 seconds
	 tinyvec from 549120 to 591360 with 1 threads in  3.21-3.63 seconds
	arrayvec from 549120 to 591360 with 1 threads in  2.67-2.84 seconds
	     vec from 549120 to 591360 with 1 threads in  4.87-71.92-97.05-135.97 seconds
	
	max_combinations = [457, 641, 819, 916] // 916*24 = 21984
	max_combinations = [141, 169, 218, 264, 273, 391, 491, 721, 806, 819, 1182, 1291, 1551, 1851, 2150, 2574, 2822, 4101, 4113, 5058]
	*/
    if args.perf {
        let graph = time_graph::get_full_graph();
		println!("{}", graph.as_dot());
		for timed_span in graph.spans() {
			let secs = timed_span.elapsed.as_secs_f64();
			println!("{} ... {:.2} ({:.2}%)", timed_span.callsite.name(), secs, 100.0*secs/total_sec);
		}
    }
}


