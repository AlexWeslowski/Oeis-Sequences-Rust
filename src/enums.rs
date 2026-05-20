
use function_name::named;
use std::error::Error;
use std::fmt;
use std::ops::{BitOr, BitOrAssign};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CalcDensityType(u8);

impl CalcDensityType {
    pub const NONE: CalcDensityType = CalcDensityType(0);
    pub const RATIO: CalcDensityType = CalcDensityType(1 << 0);
    pub const FRAC: CalcDensityType = CalcDensityType(1 << 0);
    pub const FRACTION: CalcDensityType = CalcDensityType(1 << 0);
    pub const OR: CalcDensityType = CalcDensityType(1 << 1);
    pub const BITOR: CalcDensityType = CalcDensityType(1 << 1);
    pub const BIT_OR: CalcDensityType = CalcDensityType(1 << 1);
    pub const XOR: CalcDensityType = CalcDensityType(1 << 2);
    pub const BITXOR: CalcDensityType = CalcDensityType(1 << 2);
    pub const BIT_XOR: CalcDensityType = CalcDensityType(1 << 2);
	
	#[function_name::named]
	pub fn to_string(self) -> String {
		let debug = false;
		let mut s1 = String::new();
		if debug { println!("{} line {}", function_name!(), line!()); }
		if self.is_set(Self::RATIO) {
			s1.push_str("RATIO|");
		}
		if debug { println!("{} line {}", function_name!(), line!()); }
		if self.is_set(Self::OR) {
			s1.push_str("OR|");
		}
		if debug { println!("{} line {}", function_name!(), line!()); }
		if self.is_set(Self::XOR) {
			s1.push_str("XOR|");
		}
		let ilen = s1.len();
		let s2 = if ilen <= 1 { "" } else { &s1[..ilen - 1] };
		s2.to_string()
	}
	
	#[function_name::named]
	pub fn from_name(name: &str) -> Option<Self> {
		let debug = false;
		if debug { println!("{} line {}", function_name!(), line!()); }
        match name.to_uppercase().trim() {
            "0" => Some(Self::NONE),
            "NONE" => Some(Self::NONE),
            "1" => Some(Self::RATIO),
            "RATIO" => Some(Self::RATIO),
            "FRAC" => Some(Self::FRAC),
            "FRACTION" => Some(Self::FRACTION),
            "2" => Some(Self::OR),
            "OR" => Some(Self::OR),
            "BITOR" => Some(Self::BITOR),
            "BIT_OR" => Some(Self::BIT_OR),
            "4" => Some(Self::XOR),
            "XOR" => Some(Self::XOR),
            "BITXOR" => Some(Self::BITXOR),
            "BIT_XOR" => Some(Self::BIT_XOR),
            _ => None,
        }
    }
	
	#[function_name::named]
    pub fn is_set(&self, flag: CalcDensityType) -> bool {
        (self.0 & flag.0) != 0
    }
}

impl From<u8> for CalcDensityType {
    fn from(u: u8) -> Self {
		let mut cdt = CalcDensityType::NONE;
		if (u | 1) != 0 {
			cdt |= CalcDensityType::RATIO;
		}
		if (u | 2) != 0 {
			cdt |= CalcDensityType::OR;
		}
		if (u | 4) != 0 {
			cdt |= CalcDensityType::XOR;
		}
		cdt
    }
}

impl BitOr for CalcDensityType {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        CalcDensityType(self.0 | rhs.0)
    }
}

impl BitOrAssign for CalcDensityType {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[derive(Debug)]
pub enum CalcDensityParseError {
    EmptyInput(String),
    InvalidFlag(String),
    InvalidFormat(String),
}

impl fmt::Display for CalcDensityParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CalcDensityParseError::EmptyInput(s) => write!(f, "Empty input"),
            CalcDensityParseError::InvalidFlag(s) => write!(f, "Invalid flag: {}", s),
            CalcDensityParseError::InvalidFormat(s) => write!(f, "Invalid format: {}", s),
        }
    }
}

impl Error for CalcDensityParseError {
}

impl fmt::Display for CalcDensityType {
	#[function_name::named]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		let debug = false;
		if debug { println!("{} line {}", function_name!(), line!()); }
        write!(f, "{}", self.to_string())
    }
}

impl FromStr for CalcDensityType {
    type Err = CalcDensityParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.trim().is_empty() {
            return Err(CalcDensityParseError::EmptyInput("".to_string()));
        }
        let mut combined_type = CalcDensityType::NONE;
        for part in s.split('|') {
            match CalcDensityType::from_name(part.trim()) {
                Some(flag) => {
                    combined_type = combined_type | flag;
                }
                None => {
                    return Err(CalcDensityParseError::InvalidFlag(part.to_string()));
                }
            }
        }        
        Ok(combined_type)
    }
}




#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DataType(u8);

impl DataType {
    pub const NONE: DataType = DataType(0);
	//pub const ARY: DataType = DataType(1 << 0);
    //pub const ARRAY: DataType = DataType(1 << 0);
    pub const TINYVEC: DataType = DataType(1 << 1);
    pub const ARRAYVEC: DataType = DataType(1 << 2);
	pub const SMALLVEC: DataType = DataType(1 << 3);
    pub const VEC: DataType = DataType(1 << 4);
	pub const VECTOR: DataType = DataType(1 << 4);
	
	#[function_name::named]
	pub fn to_string(self) -> String {
		let mut s1 = String::new();
		//if self.is_set(Self::ARRAY) { s1.push_str("ARRAY|"); }
		if self.is_set(Self::SMALLVEC) { s1.push_str("SMALLVEC|"); }
		if self.is_set(Self::TINYVEC) { s1.push_str("TINYVEC|"); }
		if self.is_set(Self::ARRAYVEC) { s1.push_str("ARRAYVEC|"); }
		if self.is_set(Self::VEC) { s1.push_str("VEC|"); }
		let ilen = s1.len();
		let s2 = if ilen <= 1 { "" } else { &s1[..ilen - 1] };
		s2.to_string()
	}
	
	#[function_name::named]
	pub fn from_name(name: &str) -> Option<Self> {
		let debug = false;
		if debug { println!("{} line {}", function_name!(), line!()); }
        match name.to_uppercase().trim() {
            "0" => Some(Self::NONE),
            "NONE" => Some(Self::NONE),
            //"1" => Some(Self::ARRAY),
			//"ARY" => Some(Self::ARY),
            //"ARRAY" => Some(Self::ARRAY),
            "2" => Some(Self::TINYVEC),
            "TINYVEC" => Some(Self::TINYVEC),
            "4" => Some(Self::ARRAYVEC),
            "ARRAYVEC" => Some(Self::ARRAYVEC),
			"8" => Some(Self::SMALLVEC),
            "SMALLVEC" => Some(Self::SMALLVEC),
            "16" => Some(Self::VEC),
            "VEC" => Some(Self::VEC),
            "VECTOR" => Some(Self::VECTOR),
            _ => None,
        }
    }
	
	#[function_name::named]
    pub fn is_set(&self, flag: DataType) -> bool {
        (self.0 & flag.0) != 0
    }
}

impl From<u8> for DataType {
    fn from(u: u8) -> Self {
		let mut cdt = DataType::NONE;
		//if (u | 1) != 0 { cdt |= DataType::ARRAY; }
		if (u | 2) != 0 { cdt |= DataType::TINYVEC; }
		if (u | 4) != 0 { cdt |= DataType::ARRAYVEC; }
		if (u | 8) != 0 { cdt |= DataType::SMALLVEC; }
		if (u | 16) != 0 { cdt |= DataType::VEC; }
		cdt
    }
}

impl BitOr for DataType {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        DataType(self.0 | rhs.0)
    }
}

impl BitOrAssign for DataType {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[derive(Debug)]
pub enum DataTypeParseError {
    EmptyInput(String),
    InvalidFlag(String),
    InvalidFormat(String),
}

impl fmt::Display for DataTypeParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataTypeParseError::EmptyInput(s) => write!(f, "Empty input"),
            DataTypeParseError::InvalidFlag(s) => write!(f, "Invalid flag: {}", s),
            DataTypeParseError::InvalidFormat(s) => write!(f, "Invalid format: {}", s),
        }
    }
}

impl Error for DataTypeParseError {
}

impl fmt::Display for DataType {
	#[function_name::named]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		let debug = false;
		if debug { println!("{} line {}", function_name!(), line!()); }
        write!(f, "{}", self.to_string())
    }
}

impl FromStr for DataType {
    type Err = DataTypeParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.trim().is_empty() { return Err(DataTypeParseError::EmptyInput("".to_string())); }
        let mut combined_type = DataType::NONE;
        for part in s.split('|') {
            match DataType::from_name(part.trim()) {
                Some(flag) => { combined_type = combined_type | flag; }
                None => { return Err(DataTypeParseError::InvalidFlag(part.to_string())); }
            }
        }        
        Ok(combined_type)
    }
}

#[derive(Debug, PartialEq)]
pub enum Backtrack {
    Stack,
    Recurse,
}