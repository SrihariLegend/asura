use std::{mem::MaybeUninit, ops::Index};

#[derive(Clone)]
pub struct ArrayVec<T: Copy, const N: usize> {
    data: [MaybeUninit<T>; N],
    len: usize,
}

impl<T: Copy, const N: usize> ArrayVec<T, N> {
    pub const fn new() -> Self {
        Self { data: [const { MaybeUninit::uninit() }; N], len: 0 }
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get(&self, index: usize) -> &T {
        debug_assert!(index < self.len);
        unsafe { self.data.get_unchecked(index).assume_init_ref() }
    }

    pub fn push(&mut self, value: T) {
        debug_assert!(self.len < N);
        unsafe { self.data.get_unchecked_mut(self.len).write(value) };
        self.len += 1;
    }

    pub fn maybe_push(&mut self, mask: bool, value: T) {
        debug_assert!(self.len < N);
        unsafe { self.data.get_unchecked_mut(self.len).write(value) };
        self.len += mask as usize;
    }

    pub const fn clear(&mut self) {
        self.len = 0;
    }

    pub fn swap_remove(&mut self, index: usize) -> T {
        unsafe {
            let value = self.data.get_unchecked(index).assume_init();
            self.len -= 1;
            std::ptr::copy(
                self.data.get_unchecked(self.len).as_ptr(),
                self.data.get_unchecked_mut(index).as_mut_ptr(),
                1,
            );
            value
        }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr().cast(), self.len) }.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr().cast(), self.len) }.iter_mut()
    }
}

impl<T: Copy, const N: usize> Default for ArrayVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, const N: usize> Index<usize> for ArrayVec<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index)
    }
}
