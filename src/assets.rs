use super::*;

pub const BROWN: Color = Color { r: 0.5, g: 0.2, b: 0.2, a: 1. };

pub const RED: Color = Color { r: 1., g: 0., b: 0., a: 1. };
pub const GREEN: Color = Color { r: 0.1, g: 0.4, b: 0.1, a: 1. };
pub const BLUE: Color = Color { r: 0., g: 0., b: 1., a: 1. };
pub const WALK_SPEED: f32 = 4.0;

// impl core::ops::Index<PullLevel> for AudioAssets {
//     type Output = Source;
//     fn index(&self, t: PullLevel) -> &Self::Output {
//         // &self.taut[t.level()]
//         &self.taut[0]
//     }
// }
// impl core::ops::IndexMut<PullLevel> for AudioAssets {
//     fn index_mut(&mut self, t: PullLevel) -> &mut Self::Output {
//         // &mut self.taut[t.level()]
//         &mut self.taut[0]
//     }
// }
fn linear(mut image: Image) -> Image {
    image.set_filter(FilterMode::Nearest);
    image
}
pub(crate) fn starting_doodads(rng: &mut SmallRng) -> Vec<Doodad> {
    impl Distribution<DoodadKind> for Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> DoodadKind {
            match rng.gen_range(0, 3) {
                0 => DoodadKind::Shrub,
                1 => DoodadKind::Bush,
                _ => DoodadKind::Pebbles,
            }
        }
    }
    let num = rng.gen_range(100, 130);
    let f = move || Doodad {
        kind: rng.gen(),
        pos: Pt3::new(rng.gen_range(-1500., 1500.), rng.gen_range(-1500., 1500.), 0.),
    };
    std::iter::repeat_with(f).take(num).collect()
}
impl Assets {
    pub fn new(ctx: &mut Context) -> Self {
        let tex = TexAssets {
            cross: MeshBuilder::new()
                .line(&[[-1., -1.], [1., 1.]], 0.19, graphics::WHITE)
                .unwrap()
                .line(&[[-1., 1.], [1., -1.]], 0.19, graphics::WHITE)
                .unwrap()
                .build(ctx)
                .unwrap(),
            doodads: SpriteBatch::new(linear(Image::new(ctx, "/doodads.png").unwrap())),
            archer_back: linear(Image::new(ctx, "/archer_back.png").unwrap()),
            archer: linear(Image::new(ctx, "/archer.png").unwrap()),
            archer_front: linear(Image::new(ctx, "/archer_front.png").unwrap()),
            unit_line: MeshBuilder::new()
                .line(&[[0., 0.], [1., 0.]], 1., graphics::WHITE)
                .unwrap()
                .build(ctx)
                .unwrap(),
            arrow_batch: SpriteBatch::new(Image::new(ctx, "/arrow.png").unwrap()),
        };
        let audio = AudioAssets {
            taut: [
                ggez::audio::Source::new(ctx, "/taut1.wav").unwrap(),
                ggez::audio::Source::new(ctx, "/taut2.wav").unwrap(),
                ggez::audio::Source::new(ctx, "/taut3.wav").unwrap(),
                ggez::audio::Source::new(ctx, "/taut4.wav").unwrap(),
            ],
            twang: [
                ggez::audio::Source::new(ctx, "/twang1.wav").unwrap(),
                ggez::audio::Source::new(ctx, "/twang2.wav").unwrap(),
                ggez::audio::Source::new(ctx, "/twang3.wav").unwrap(),
            ],
            loose: [
                //
                ggez::audio::Source::new(ctx, "/loose1.wav").unwrap(),
            ],
            thud: [
                //
                ggez::audio::Source::new(ctx, "/thud1.wav").unwrap(),
            ],
        };
        Assets { audio, tex }
    }
}
