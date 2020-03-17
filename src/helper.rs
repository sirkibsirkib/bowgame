// invariant: self.0.next_index - 1 is within bounds of self.0.v
pub struct Entry<'vec, 'entry, T> {
    draining: &'entry mut Draining<'vec, T>,
}

impl<'vec, 'entry, T> Entry<'vec, 'entry, T> {
    pub fn take(self) -> T {
        self.draining.next_index -= 1; // breaks invariant. no longer off-by-one
        self.draining.vec.remove(self.draining.next_index)
    }
    pub fn get_mut(&mut self) -> &mut T {
        // invariant: off by one
        unsafe { self.draining.vec.get_unchecked_mut(self.draining.next_index - 1) }
    }
}

// two implicit states:
// 1. not borrowed by Entry: self.next index is the index of the next element
// 2. borrowed by Entry: self.next_index is 1 greater than the index of next element
//
// conceptually, could be alternatively achieved by either:
// 1. always being in state 2, where Entry gets the old copy of next_index (so its no longer off-by-one for Entry)
// 2. incrementing next_index whenever Entry is DROPPED. (drops are not guaranteed => unsafe!)
pub(crate) struct Draining<'vec, T> {
    next_index: usize, // always off by one while Draining is borrowed by an Entry
    vec: &'vec mut Vec<T>,
}
impl<'vec, T> Draining<'vec, T> {
    pub fn new(vec: &'vec mut Vec<T>) -> Self {
        Self { vec, next_index: 0 }
    }
    pub fn next<'entry>(&'entry mut self) -> Option<Entry<'vec, 'entry, T>> {
        // checks invariant
        if self.next_index < self.vec.len() {
            self.next_index += 1;
            Some(Entry { draining: self })
        } else {
            None
        }
    }
}

pub(crate) trait Squarable: core::ops::Mul + Copy {
    fn sqr(self) -> <Self as core::ops::Mul>::Output {
        self * self
    }
}
impl<T: core::ops::Mul + Copy> Squarable for T {}
