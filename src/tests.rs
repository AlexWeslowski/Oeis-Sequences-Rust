use super::*;

const TESTS: &str = include_str!("tests.txt");
	
// 
// cargo test -- --nocapture
// 

fn test_thread(vecprimes1: &mut PrimesType, ifinish: u32, datatype: DataType, method: CalcDensityType, expected: Vec<u32>) {
	if vecprimes1.len() < ifinish as usize {
		*vecprimes1 = init(ifinish);
	}
	let vecprimes2: Arc<PrimesType> = Arc::new(vecprimes1.clone());
	let bln_gt_half: bool = false;
	let mut min_factors_len: usize = 4;
	let t1: Instant = Instant::now();
	let mut predefined1 = predefined();
	let mut matches1: Vec<Ratio<i32>> = Vec::new();
	let mut outmap1: Arc<Mutex<AHashMap<Ratio<i32>, ArrayVec<[u32; 4096]>>>> = Arc::new(Mutex::new(AHashMap::new()));
	for ratio in [Ratio::<i32>::new(1, 2)] {
		matches1.push(ratio);
		outmap1.lock().unwrap().insert(ratio, ArrayVec::<[u32; 4096]>::new());
		if *ratio.denom() > 4 || *ratio.numer() > 1 {
			min_factors_len = 2;
		}
	}
	
	let builder_main = thread::Builder::new().stack_size(ifinish as usize + 33554432);
	let handle_main = builder_main.spawn(move || {
		let mut seq1: Sequence24 = Sequence24::new(ifinish as usize, false, false, datatype, method);
		seq1.bln_debug = false;
		seq1.bln_perf = false;
		//seq1.bln_divisor_gen = false;
		//seq1.bln_exhaustive_search = false;
		seq1.bln_factors = true;
		seq1.bln_divisors = false;
		seq1.bln_gt_half = bln_gt_half;
		seq1.min_factors_len = min_factors_len;
		seq1.set_primes(&vecprimes2);
		
		let mut m = Main { 
			debug: false,
			logging: false,
			ithread: 0,
			t1: t1, 
			anumthreads: AtomicU32::new(1), 
			astart: AtomicU32::new(2), 
			afinish: AtomicU32::new(ifinish), 
			calcdensity: method,
			datatype: datatype,
			matches: matches1,
			predefined: predefined1,
			outmap: outmap1,
			tx: None,
		};
		let (mut max_stack, mut max_combinations): (Vec<usize>, Vec<usize>) = m.do_work(2, ifinish, 1, seq1);
		
		for (key, val) in m.outmap.lock().unwrap().iter_mut() {
			val.sort();
			//println!("{}/{}\t{:?}", key.numer(), key.denom(), val);
			if *key == m.matches[0] {
				assert_eq!(val.as_slice(), expected.as_slice());
			}
		}
	}).unwrap();
	
	handle_main.join().unwrap();
	print_elapsed(t1.elapsed().as_secs_f64(), 2, ifinish, 1);
}

#[test]
fn test_main() {
	let mut vecprimes1: PrimesType = PrimesType::new();
	for line in TESTS.lines() {
		if line.len() <= 1 {
			continue;
		}
		//println!("{}", line);
		let mut vars: Vec<&str> = line.split(";").collect();
		let ifinish: u32 = vars[0].trim().parse::<u32>().ok().unwrap();
		let datatype: DataType = DataType::from_name(vars[1].trim()).unwrap();
		let method: CalcDensityType = CalcDensityType::from_name(vars[2].trim()).unwrap();
		let vec: Vec<u32> = serde_json::from_str(vars[3]).unwrap();
		//[2, 12, 168, 240, 6720, 13440, 14880, 65280, 549120, 591360, 591360, 833280, 954240]
		test_thread(&mut vecprimes1, ifinish, datatype, method, vec);
	}
}
