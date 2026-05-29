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
#[function_name::named]
fn test_main() {
	println!("{}() line {}", function_name!(), line!()); 
	let mut vecprimes1: PrimesType = PrimesType::new();
	for line in TESTS.lines() {
		if line.len() <= 1 {
			continue;
		}
		println!("{}", line);
		let mut vars: Vec<&str> = line.split(";").collect();
		let ifinish: u32 = vars[0].trim().parse::<u32>().ok().unwrap();
		let datatype: DataType = DataType::from_name(vars[1].trim()).unwrap();
		let method: CalcDensityType = CalcDensityType::from_name(vars[2].trim()).unwrap();
		let vec: Vec<u32> = serde_json::from_str(vars[3]).unwrap();
		//[2, 12, 168, 240, 6720, 13440, 14880, 65280, 549120, 591360, 591360, 833280, 954240]
		test_thread(&mut vecprimes1, ifinish, datatype, method, vec);
	}
}

#[function_name::named]
fn test_divisors() {
	println!("{}() line {}", function_name!(), line!()); 
	let mut args = Args::parse();
	let builder_test = thread::Builder::new().stack_size(args.stacksize + args.finish as usize);
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

#[function_name::named]
fn test_sequence24_thread() {
	println!("{}() line {}", function_name!(), line!()); 
	let mut args = Args::parse();
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
		//println!("{}() line {}", function_name!(), line!()); 
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
}

#[function_name::named]		
fn test_sequence24() {
	println!("{}() line {}", function_name!(), line!()); 
	let mut args = Args::parse();
	let mut seq1: Sequence24 = Sequence24::new(args.finish as usize, false, false, args.datatype, args.method);
	for n in [720, 840, 6720] {
		for this_combination in seq1.factor_combinations_vec(n as i32) {
			let this_density: Ratio<i32> = seq1.calc_density_ratio(n as usize, &this_combination);                    
			//if let Some(this_ratio) = vecratios1.iter().find(|&x| *x == this_density) {
			if *this_density.denom() <= 3 {
				println!("{} {} {:?}", this_density, n, this_combination);
			}
		}
	}
}
	
