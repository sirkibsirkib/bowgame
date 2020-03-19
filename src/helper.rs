// invariant: self.0.next_index - 1 is within bounds of self.0.v
use super::*;

pub struct Entry<'vec, 'entry, T> {
    draining: &'entry mut Draining<'vec, T>,
}

impl<'vec, 'entry, T> Entry<'vec, 'entry, T> {
    pub fn take(self) -> (usize, T) {
        self.draining.next_index -= 1; // breaks invariant. no longer off-by-one
        (self.draining.next_index, self.draining.vec.remove(self.draining.next_index))
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

impl TryInto<UiConfig> for UiConfigSerde {
    type Error = ();
    fn try_into(self) -> Result<UiConfig, Self::Error> {
        fn keycode(k: &str) -> Result<KeyCode, <UiConfigSerde as TryInto<UiConfig>>::Error> {
            Ok(match k {
                "A" => KeyCode::A,
                "B" => KeyCode::B,
                "C" => KeyCode::C,
                "D" => KeyCode::D,
                "E" => KeyCode::E,
                "F" => KeyCode::F,
                "G" => KeyCode::G,
                "H" => KeyCode::H,
                "I" => KeyCode::I,
                "J" => KeyCode::J,
                "K" => KeyCode::K,
                "L" => KeyCode::L,
                "M" => KeyCode::M,
                "N" => KeyCode::N,
                "O" => KeyCode::O,
                "P" => KeyCode::P,
                "Q" => KeyCode::Q,
                "R" => KeyCode::R,
                "S" => KeyCode::S,
                "T" => KeyCode::T,
                "U" => KeyCode::U,
                "V" => KeyCode::V,
                "W" => KeyCode::W,
                "X" => KeyCode::X,
                "Y" => KeyCode::Y,
                "Z" => KeyCode::Z,
                "Space" => KeyCode::Space,
                "Escape" => KeyCode::Escape,
                "Key1" => KeyCode::Key1,
                "Key2" => KeyCode::Key2,
                "Key3" => KeyCode::Key3,
                "Key4" => KeyCode::Key4,
                "Key5" => KeyCode::Key5,
                "Key6" => KeyCode::Key6,
                "Key7" => KeyCode::Key7,
                "Key8" => KeyCode::Key8,
                "Key9" => KeyCode::Key9,
                "Key0" => KeyCode::Key0,
                "F1" => KeyCode::F1,
                "F2" => KeyCode::F2,
                "F3" => KeyCode::F3,
                "F4" => KeyCode::F4,
                "F5" => KeyCode::F5,
                "F6" => KeyCode::F6,
                "F7" => KeyCode::F7,
                "F8" => KeyCode::F8,
                "F9" => KeyCode::F9,
                "F10" => KeyCode::F10,
                "Insert" => KeyCode::Insert,
                "Home" => KeyCode::Home,
                "Delete" => KeyCode::Delete,
                "End" => KeyCode::End,
                "PageUp" => KeyCode::PageUp,
                "PageDown" => KeyCode::PageDown,
                "Up" => KeyCode::Up,
                "Right" => KeyCode::Right,
                "Left" => KeyCode::Left,
                "Down" => KeyCode::Down,
                "Back" => KeyCode::Back,
                "Return" => KeyCode::Return,
                _ => return Err(()),
            })
        }
        Ok(UiConfig {
            up: keycode(&self.up)?,
            down: keycode(&self.down)?,
            left: keycode(&self.left)?,
            right: keycode(&self.right)?,
            clockwise: keycode(&self.clockwise)?,
            anticlockwise: keycode(&self.anticlockwise)?,
            aim_assist: keycode(&self.aim_assist)?,
            quit: keycode(&self.quit)?,
            net_mode: self.net_mode.parse()?,
            addr: self.addr.parse().map_err(drop)?,
        })
    }
}
