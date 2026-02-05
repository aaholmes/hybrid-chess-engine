//! Unit tests for bits module (bit iteration and manipulation)

use kingfisher::bits::{bits, btest, ibclr, ibset, parity, popcnt};

// --- bits() iterator tests ---

#[test]
fn test_bits_empty() {
    let n = 0u64;
    let result: Vec<usize> = bits(&n).collect();
    assert!(result.is_empty(), "Empty bitboard should yield no bits");
}

#[test]
fn test_bits_single() {
    let n = 1u64 << 42;
    let result: Vec<usize> = bits(&n).collect();
    assert_eq!(result, vec![42]);
}

#[test]
fn test_bits_full_rank() {
    // First rank: bits 0-7
    let n = 0xFFu64;
    let result: Vec<usize> = bits(&n).collect();
    assert_eq!(result, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn test_bits_scattered() {
    let n = 0b1010_0000_0000_0101u64; // bits 0, 2, 13, 15
    let result: Vec<usize> = bits(&n).collect();
    assert_eq!(result, vec![0, 2, 13, 15]);
}

// --- ibset / ibclr / btest tests ---

#[test]
fn test_ibset_and_ibclr() {
    let n: u128 = 0;
    let with_bit = ibset(n, 10);
    assert!(btest(with_bit, 10), "Bit 10 should be set");
    assert!(!btest(with_bit, 11), "Bit 11 should not be set");

    let cleared = ibclr(with_bit, 10);
    assert!(!btest(cleared, 10), "Bit 10 should be cleared");
    assert_eq!(cleared, 0);
}

#[test]
fn test_btest_true_and_false() {
    let n: u128 = 0b1010;
    assert!(btest(n, 1), "Bit 1 should be set");
    assert!(btest(n, 3), "Bit 3 should be set");
    assert!(!btest(n, 0), "Bit 0 should not be set");
    assert!(!btest(n, 2), "Bit 2 should not be set");
}

// --- popcnt tests ---

#[test]
fn test_popcnt() {
    assert_eq!(popcnt(0), 0);
    assert_eq!(popcnt(1), 1);
    assert_eq!(popcnt(0xFF), 8);
    assert_eq!(popcnt(u64::MAX), 64);
    assert_eq!(popcnt(0b1010_1010), 4);
}

// --- parity tests ---

#[test]
fn test_parity() {
    // Even number of set bits -> parity returns 1
    assert_eq!(parity(0b0000), 1, "0 set bits is even");
    assert_eq!(parity(0b1010), 1, "2 set bits is even");
    assert_eq!(parity(0b1111), 1, "4 set bits is even");

    // Odd number of set bits -> parity returns -1
    assert_eq!(parity(0b0001), -1, "1 set bit is odd");
    assert_eq!(parity(0b0111), -1, "3 set bits is odd");
    assert_eq!(parity(0b1011), -1, "3 set bits is odd");
}
