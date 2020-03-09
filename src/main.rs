use ggez::event::EventHandler;
use ggez::event::MouseButton;
use ggez::{
    graphics::{self, Color, DrawMode, DrawParam, Mesh, MeshBuilder, Rect, BLACK, WHITE},
    nalgebra as na, *,
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
    let mut my_game = MyGame {
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

struct MyGame {
    arrowhead: Mesh,
    arrowshaft: Mesh,
    arrows: Vec<Arrow>,
    stuck_arrows: Vec<StuckArrow>,
    mouse_click_at: Option<na::Point2<f32>>,
}
struct Arrow {
    pos: na::Point2<f32>,
    vel: na::Point2<f32>,
    rot: f32,
    age: u8,
}
struct StuckArrow {
    pos: na::Point2<f32>,
    rot: f32,
}
impl Arrow {
    fn vel_to_rot(vel: na::Point2<f32>) -> f32 {
        nalgebra::geometry::Rotation2::rotation_between(
            &na::Point2::new(1., 0.).coords,
            &vel.coords,
        )
        .angle()
    }
    fn stick(self) -> StuckArrow {
        let Self { pos, rot, .. } = self;
        StuckArrow { pos, rot }
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
        let mut draining = Draining::new(&mut self.arrows);
        while let Some(mut entry) = draining.next() {
            let arrow: &mut Arrow = entry.get_mut();
            arrow.pos += arrow.vel.coords;
            arrow.vel = nalgebra::geometry::Rotation2::new(0.01).transform_point(&arrow.vel);
            arrow.rot = Arrow::vel_to_rot(arrow.vel);
            if arrow.age > 17 {
                self.stuck_arrows.push(entry.take().stick())
            } else {
                arrow.age += 1;
            }
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
                let second: na::Point2<f32> = [x, y].into();
                let vel = (second - first.coords) * 0.3;
                self.arrows.push(Arrow { age: 0, pos: second, vel, rot: Arrow::vel_to_rot(vel) });
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
            graphics::draw(
                ctx,
                &self.arrowhead,
                DrawParam {
                    dest,
                    color: WHITE,
                    scale: [20.; 2].into(),
                    rotation,
                    ..Default::default()
                },
            )?;
            graphics::draw(
                ctx,
                &self.arrowshaft,
                DrawParam {
                    dest,
                    color: WHITE,
                    scale: [20.; 2].into(),
                    rotation,
                    ..Default::default()
                },
            )?;
        }
        for stuck_arrow in &mut self.stuck_arrows {
            let dest = stuck_arrow.pos.into();
            let rotation = stuck_arrow.rot;
            graphics::draw(
                ctx,
                &self.arrowshaft,
                DrawParam {
                    dest,
                    color: WHITE,
                    scale: [20.; 2].into(),
                    rotation,
                    ..Default::default()
                },
            )?;
        }
        graphics::present(ctx)
    }
}
