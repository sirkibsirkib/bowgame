use ggez::audio::SoundSource;
use ggez::audio::Source;
use ggez::event::EventHandler;
use ggez::event::MouseButton;
use ggez::{
    graphics::{
        self, spritebatch::SpriteBatch, Color, DrawMode, DrawParam, Image, Mesh, MeshBuilder, Rect,
        BLACK, WHITE,
    },
    nalgebra::{self as na, geometry::Rotation2},
    *,
};

const BROWN: Color = Color { r: 0.5, g: 0.2, b: 0.2, a: 1. };
const RED: Color = Color { r: 1., g: 0., b: 0., a: 1. };
const FLETCH_THICKNESS: f32 = 0.07;
const FLETCH_LENGTH: f32 = 0.2;
const FLETCH_INDENT: f32 = 0.04;
const HEAD_THICKNESS: f32 = 0.05;
const HEAD_LENGTH: f32 = 0.13;
const SHAFT_THICKNESS: f32 = 0.017;
fn main() {
    // Make a Context.
    let (mut ctx, mut event_loop) = ContextBuilder::new("my_game", "Cool Game Author")
        .build()
        .expect("aieee, could not create ggez context!");
    let arrow_png = Image::new(&mut ctx, "/arrow.png").unwrap();
    let mut my_game = MyGame {
        dude: [100., 90.].into(),
        arrow_batch: SpriteBatch::new(arrow_png),
        twang1: ggez::audio::Source::new(&mut ctx, "/twang1.wav").unwrap(),
        twang2: ggez::audio::Source::new(&mut ctx, "/twang2.wav").unwrap(),
        twang3: ggez::audio::Source::new(&mut ctx, "/twang3.wav").unwrap(),
        time: 0.,
        arrows: vec![],
        stuck_arrows: vec![],
        mouse_click_at: None,
        arrowhead: MeshBuilder::new()
            .triangles(
                &[[0., -HEAD_THICKNESS], [HEAD_LENGTH, 0.], [0., HEAD_THICKNESS]],
                graphics::WHITE,
            )
            .unwrap()
            .build(&mut ctx)
            .unwrap(),
        unit_line: MeshBuilder::new()
            .line(&[[0., 0.], [1., 0.]], 1., graphics::WHITE)
            .unwrap()
            .build(&mut ctx)
            .unwrap(),
        arrowshaft: {
            let mut q = MeshBuilder::new();
            q.polygon(
                DrawMode::fill(),
                &[
                    [-0.9, -FLETCH_THICKNESS],
                    [-0.9 + FLETCH_LENGTH, -FLETCH_THICKNESS],
                    [-0.9 + FLETCH_LENGTH + FLETCH_INDENT, -0.],
                    [-0.9 + FLETCH_LENGTH, FLETCH_THICKNESS],
                    [-0.9, FLETCH_THICKNESS],
                    [-0.9 + FLETCH_INDENT, 0.],
                ],
                RED,
            )
            .unwrap();
            q.rectangle(
                DrawMode::fill(),
                Rect { w: -0.9, h: SHAFT_THICKNESS * 2., x: 0., y: -SHAFT_THICKNESS },
                BROWN,
            );
            q.build(&mut ctx).unwrap()
        },
    };
    match event::run(&mut ctx, &mut event_loop, &mut my_game) {
        Ok(_) => println!("Exited cleanly."),
        Err(e) => println!("Error occured: {}", e),
    }
}
type Pt2 = na::Point2<f32>;

struct MyGame {
    arrow_batch: SpriteBatch,
    dude: Pt2,
    twang1: Source,
    twang2: Source,
    twang3: Source,
    time: f32,
    arrowhead: Mesh,
    arrowshaft: Mesh,
    unit_line: Mesh,
    arrows: Vec<Arrow>,
    stuck_arrows: Vec<StuckArrow>,
    mouse_click_at: Option<Pt2>,
}
struct Arrow {
    pos: Pt2,
    vel: Pt2,
    rot: f32,
    age: u8,
}
struct StuckArrow {
    pos: na::Point2<f32>,
    rot: f32,
    vibration_amplitude: f32,
}
impl Arrow {
    fn vel_to_rot(vel: Pt2) -> f32 {
        Rotation2::rotation_between(&Pt2::new(1., 0.).coords, &vel.coords).angle()
    }
    fn stick(self) -> StuckArrow {
        let Self { pos, rot, .. } = self;
        StuckArrow { pos, rot, vibration_amplitude: 0.5 }
    }
}
// invariant: self.0.next_index - 1 is within bounds of self.0.v
struct Entry<'vec, 'entry, T> {
    draining: &'entry mut Draining<'vec, T>,
}

impl<'vec, 'entry, T> Entry<'vec, 'entry, T> {
    fn take(self) -> T {
        self.draining.next_index -= 1; // breaks invariant. no longer off-by-one
        self.draining.vec.remove(self.draining.next_index)
    }
    fn get_mut(&mut self) -> &mut T {
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
struct Draining<'vec, T> {
    next_index: usize, // always off by one while Draining is borrowed by an Entry
    vec: &'vec mut Vec<T>,
}
impl<'vec, T> Draining<'vec, T> {
    fn new(vec: &'vec mut Vec<T>) -> Self {
        Self { vec, next_index: 0 }
    }
    fn next<'entry>(&'entry mut self) -> Option<Entry<'vec, 'entry, T>> {
        // checks invariant
        if self.next_index < self.vec.len() {
            self.next_index += 1;
            Some(Entry { draining: self })
        } else {
            None
        }
    }
}

impl EventHandler for MyGame {
    fn update(&mut self, _ctx: &mut Context) -> GameResult<()> {
        self.time += 1.0;
        let mut draining = Draining::new(&mut self.arrows);
        while let Some(mut entry) = draining.next() {
            let arrow: &mut Arrow = entry.get_mut();
            arrow.pos += arrow.vel.coords;
            if arrow.age > 17 {
                match unsafe { std::mem::transmute::<_, f32>(self.time) } as usize % 3 {
                    0 => &mut self.twang1,
                    1 => &mut self.twang2,
                    _ => &mut self.twang3,
                }
                .play()
                .unwrap();
                self.stuck_arrows.push(entry.take().stick())
            } else {
                arrow.age += 1;
            }
        }
        for stuck_arrow in &mut self.stuck_arrows {
            stuck_arrow.vibration_amplitude *= 0.93
        }
        Ok(())
    }

    fn mouse_button_down_event(&mut self, _ctx: &mut Context, button: MouseButton, x: f32, y: f32) {
        if let MouseButton::Left = button {
            self.mouse_click_at = Some([x, y].into());
        }
    }

    fn mouse_button_up_event(&mut self, _ctx: &mut Context, button: MouseButton, x: f32, y: f32) {
        if let MouseButton::Left = button {
            if let Some(first) = self.mouse_click_at.take() {
                let second: Pt2 = [x, y].into();
                let vel = (first - second.coords) * 0.214;
                self.arrows.push(Arrow {
                    age: 0,
                    pos: self.dude,
                    vel,
                    rot: Arrow::vel_to_rot(vel),
                });
            }
        } else {
            self.stuck_arrows.extend(self.arrows.drain(..).map(Arrow::stick));
        }
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        graphics::clear(ctx, BLACK);
        for arrow in &self.arrows {
            let dest = arrow.pos.into();
            let rotation = arrow.rot;
            let p = DrawParam {
                src: Rect { x: 0., y: 0., h: 0.1, w: 1. },
                dest,
                color: WHITE,
                scale: [40. / 230.; 2].into(),
                rotation,
                offset: [0.95, 0.5].into(),
                ..Default::default()
            };
            self.arrow_batch.add(p);
            // graphics::draw(ctx, &self.arrowhead, p)?;
        }
        for stuck_arrow in &mut self.stuck_arrows {
            let dest = stuck_arrow.pos.into();
            let rotation =
                stuck_arrow.rot + (self.time * 3.0).sin() * stuck_arrow.vibration_amplitude;
            let p = DrawParam {
                src: Rect { x: 0., y: 0., h: 0.1, w: 1. },
                dest,
                color: WHITE,
                scale: [40. / 230.; 2].into(),
                rotation,
                offset: [0.95, 0.5].into(),
                ..Default::default()
            };
            // graphics::draw(ctx, &self.arrowshaft, p)?;

            self.arrow_batch.add(p);
        }
        graphics::draw(ctx, &self.arrow_batch, DrawParam::default())?;
        self.arrow_batch.clear();
        if let Some(start) = self.mouse_click_at {
            let end: Pt2 = ggez::input::mouse::position(ctx).into();
            let dif = end - start.coords;
            let rotation =
                Rotation2::rotation_between(&Pt2::new(1., 0.).coords, &dif.coords).angle();
            graphics::draw(
                ctx,
                &self.unit_line,
                DrawParam {
                    dest: start.into(),
                    color: WHITE,
                    scale: [dif.coords.norm(), 1.0].into(),
                    rotation,
                    ..Default::default()
                },
            )?;
            graphics::draw(
                ctx,
                &self.unit_line,
                DrawParam {
                    dest: self.dude.into(),
                    color: RED,
                    scale: [dif.coords.norm() * 4.0, 1.0].into(),
                    rotation: rotation + core::f32::consts::PI,
                    ..Default::default()
                },
            )?;
        }
        graphics::present(ctx)
    }
}
