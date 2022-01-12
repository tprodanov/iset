use alloc::vec::Vec;

/// i >> IX_SHIFT is the same as i / 8.
const IX_SHIFT: usize = 3;
/// i & IX_MASK is the same as i % 8.
const IX_MASK: usize = 7;
/// Starting capacity (in bytes).
const START_CAPACITY: usize = 8;

#[inline]
fn get_byte_len(bit_len: usize) -> usize {
    (bit_len >> IX_SHIFT) + (bit_len & IX_MASK > 0) as usize
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub(crate) struct BitVec {
    len: usize,
    data: Vec<u8>,
}

impl BitVec {
    pub fn new() -> Self {
        Self {
            len: 0,
            data: Vec::with_capacity(START_CAPACITY),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            len: 0,
            data: Vec::with_capacity(get_byte_len(capacity)),
        }
    }

    pub fn from_elem(len: usize, elem: bool) -> Self {
        let fill = if elem { u8::MAX } else { 0 };
        Self {
            len,
            data: vec![fill; get_byte_len(len)],
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit()
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.len = 0;
    }

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        debug_assert!(i < self.len);
        let j = i >> IX_SHIFT;
        let k = i & IX_MASK;
        self.data[j] >> k & 1 == 1
    }

    #[inline]
    pub fn get_end(&self, i: usize) -> bool {
        self.get(self.len - 1 - i)
    }

    #[inline]
    pub fn set0(&mut self, i: usize) {
        debug_assert!(i < self.len);
        let j = i >> IX_SHIFT;
        let k = i & IX_MASK;
        self.data[j] &= !(1 << k);
    }

    #[inline]
    pub fn set1(&mut self, i: usize) {
        debug_assert!(i < self.len);
        let j = i >> IX_SHIFT;
        let k = i & IX_MASK;
        self.data[j] |= 1 << k;
    }

    #[inline]
    pub fn set(&mut self, i: usize, value: bool) {
        if value {
            self.set1(i)
        } else {
            self.set0(i);
        }
    }

    pub fn push(&mut self, value: bool) {
        if self.len & IX_MASK == 0 {
            self.data.push(0);
        }
        self.len += 1;
        self.set(self.len - 1, value);
    }

    pub fn pop(&mut self) -> bool {
        debug_assert!(self.len > 0);
        self.len -= 1;
        let j = self.len >> IX_SHIFT;
        let k = self.len & IX_MASK;
        let value = self.data[j] >> k & 1 == 1;
        if k == 0 {
            self.data.pop();
        }
        value
    }

    pub fn swap_remove(&mut self, i: usize) -> bool {
        debug_assert!(i < self.len);
        let old_val = self.get(i);
        if i + 1 != self.len {
            self.set(i, self.get(self.len - 1));
        }
        self.pop();
        old_val
    }
}
