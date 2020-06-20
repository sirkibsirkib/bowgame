use core::convert::TryInto;
use core::f32::consts::PI;
use core::str::FromStr;
use ggez::{
    audio::{SoundSource, Source},
    conf::{FullscreenType, WindowMode},
    event::{EventHandler, KeyCode, KeyMods, MouseButton},
    graphics::{
        self, spritebatch::SpriteBatch, Color, DrawParam, FilterMode, Image, Mesh, MeshBuilder,
        Rect, BLACK, WHITE,
    },
    *,
};
use nalgebra::{Rotation2, Rotation3, Transform3, Translation3};
use rand::{
    distributions::{Distribution, Standard},
    rngs::SmallRng,
    Rng, SeedableRng,
};
use serde::{de::DeserializeOwned, Serialize};
use serde_derive::{Deserialize, Serialize};
use std::borrow::Cow;
use std::io::Read;
use std::net::SocketAddr;
use std::net::TcpListener;
use std::net::TcpStream;

#[derive(Default)]
struct Pressing {
    right: Option<bool>,
    down: Option<bool>,
    clockwise: Option<bool>,
}

const MIN_VEL: f32 = 9.;
const MAX_VEL: f32 = 35.;
const ROOT_OF_2: f32 = 1.41421356;
const WIN_DIMS: [f32; 2] = [800., 600.];
const BADDIE_SPEED: f32 = 0.5;
const NUM_BADDIES: usize = 4;

const SCREEN_TRANS_V2: [f32; 2] = [
    //
    WIN_DIMS[0] * 0.3,
    WIN_DIMS[1] * 0.5,
];
const SCREEN_TRANS_V3: [f32; 3] = [
    //
    SCREEN_TRANS_V2[0],
    SCREEN_TRANS_V2[1] * ROOT_OF_2,
    0.,
];
const TO_ARMS: [f32; 3] = [0., 0., -32.];

mod assets;
use assets::*;
mod helper;
use helper::*;
mod net;
use net::*;

fn main() {
    let config: UiConfig = {
        // create a handle to config file
        let s = std::fs::read_to_string("game_config.toml").unwrap_or_else(|_| {
            panic!(
                "Couldn't find `config.toml` at current directory: {:?}",
                std::env::current_dir().ok()
            )
        });
        // parse config file
        let mut x: UiConfigSerde = toml::from_str(&s).expect("Failed to parse config toml!");
        println!("{:?}", &x);
        // use command line args 1 and 2 to overwrite `addr` and `net_mode` config fields if they are provided
        let mut args = std::env::args();
        args.next(); // skip first arg
        if let Some(s) = args.next() {
            x.addr = s;
        }
        if let Some(s) = args.next() {
            x.net_mode = s;
        }
        x.try_into().expect("Failed to parse config toml!")
    };
    println!("addr: {:?} net_mode {:?}", &config.addr, &config.net_mode);
    // build the GGEZ context object. manages the windows, event loop, etc.
    let (mut ctx, mut event_loop) = ContextBuilder::new("bowdraw", "Christopher Esterhuyse")
        .window_mode(WindowMode { width: WIN_DIMS[0], height: WIN_DIMS[1], ..Default::default() })
        .build()
        .unwrap();
    // grab the cursor if config says to do so
    ggez::input::mouse::set_cursor_grabbed(&mut ctx, config.grab_cursor).unwrap();
    // create master game state object
    let mut my_game = MyGame::new(&mut ctx, config);
    // invoke GGEZ game loop. the magic happens at `impl EventHandler for MyGame`
    event::run(&mut ctx, &mut event_loop, &mut my_game).expect("Game Err");
}

// define a load of types for storing game data

// points and vectors in screen (2D) and world (3D) spaces.
type Pt2 = nalgebra::Point2<f32>;
type Pt3 = nalgebra::Point3<f32>;
type Vec2 = nalgebra::Vector2<f32>;
type Vec3 = nalgebra::Vector3<f32>;

struct Assets {
    audio: AudioAssets,
    tex: TexAssets,
}
struct TexAssets {
    cross: Mesh,
    arrow_batch: SpriteBatch,
    doodads: SpriteBatch,
    unit_line: Mesh,
    archer_back: Image,
    archer: Image,
    archer_front: Image,
}
struct AudioAssets {
    taut: [Source; 4],
    loose: [Source; 1],
    twang: [Source; 3],
    thud: [Source; 1],
}
enum DoodadKind {
    Bush,
    Rock,
    Pebbles,
    Shrub,
}
struct Doodad {
    kind: DoodadKind,
    pos: Pt3,
}
struct Camera {
    world_pos: Pt3,
    world_rot: f32,
}
struct Ui {
    camera: Camera,
    lclick_state: Option<LclickState>,
    rclick_state: Option<RclickState>,
    pressing: Pressing,
    aim_assist: u8,
    last_mouse_at: Pt2,
    controlling: usize,
    config: UiConfig,
    current_fullscreen: FullscreenType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Archer {
    entity: Entity,
    shot_vel: Option<Vec3>,
}

// user-facing Ui type: (1) fields are strings to be human-readable, and (2) serializable
#[derive(Debug, Deserialize)]
struct UiConfigSerde {
    up: String,
    down: String,
    left: String,
    right: String,
    clockwise: String,
    anticlockwise: String,
    aim_assist: String,
    quit: String,
    cycle_fullscreen: String,
    net_mode: String,
    addr: String,
    grab_cursor: bool,
}
#[derive(Copy, Clone, Debug)]
enum NetMode {
    Server,
    Client,
    Solo,
}
// game-facing Ui type
struct UiConfig {
    up: KeyCode,
    down: KeyCode,
    left: KeyCode,
    right: KeyCode,
    clockwise: KeyCode,
    anticlockwise: KeyCode,
    aim_assist: KeyCode,
    quit: KeyCode,
    cycle_fullscreen: KeyCode,
    net_mode: NetMode,
    addr: SocketAddr,
    grab_cursor: bool,
}
struct MyGame {
    net_core: NetCore,
    ticks: usize, // ever-counting wrapping integer. used as a super basic "random" seed for animations etc.
    baddies: Vec<Baddie>, // list of bad guys (rocks)
    archers: Vec<Archer>, // list of archers
    arrows: Vec<Entity>, // all arrows currently in flight (have velocity etc.)
    stuck_arrows: Vec<Entity>, // all arrows in the ground
    assets: Assets, // store for the game's assets. Immutable after setup
    doodads: Vec<Doodad>, // doodad instances. Immutable after setup.
    rng: SmallRng, // random number generator for placing enemies etc. NOT synced between players
    ui: Ui,       // player input/output state. Including button presses and camera.
}
struct RclickState {
    anchor_angle: f32,
}
struct LclickState {
    start: Pt2,
    last_pull_level: PullLevel,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Baddie {
    target_archer_index: usize,
    entity: Entity,
    stuck_arrows: Vec<Entity>,
    health: f32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Entity {
    pos: Pt3,
    vel: Vec3,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum PullLevel {
    TooLittle,
    Low,
    Med,
    High,
    Max,
    TooMuch,
}
/////////////////// IMPL IMPL IMPL IMPL IMPL

impl FromStr for NetMode {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "Server" => NetMode::Server,
            "Client" => NetMode::Client,
            "Solo" => NetMode::Solo,
            _ => return Err(()),
        })
    }
}
impl DoodadKind {
    // compute the rectangle for texture region of this doodad
    fn rect(&self) -> Rect {
        let x = match self {
            Self::Bush => 0.0,
            Self::Rock => 0.2,
            Self::Pebbles => 0.4,
            Self::Shrub => 0.6,
        };
        Rect { x, y: 0., w: 0.2, h: 0.2 }
    }
}
impl Entity {
    #[inline]
    // given a world-space vector, apply gravity and air resistance to it
    // (used for arrows only)
    fn grav_and_air_resist(mut v: Vec3) -> Vec3 {
        // gravity
        v += Pt3::new(0., 0., 0.5).coords;
        // air resistance
        let vquad = v.norm_squared().sqr() + 1.;
        v * vquad.powf(0.996) / vquad
    }
}
impl Default for Camera {
    fn default() -> Self {
        Self { world_pos: [0.; 3].into(), world_rot: 0. }
    }
}
impl Camera {
    // query: "is the given archer facing RIGHT in screen space?"
    // (used for drawing the archer facing left/right)
    fn archer_facing_right(&self, archer: &Archer) -> bool {
        archer
            .shot_vel
            .map(|v| self.vec_3_to_2(v)[0] >= 0.)
            .unwrap_or(self.vec_3_to_2(archer.entity.vel)[0] >= 0.)
    }
    fn vec_3_to_2(&self, v: Vec3) -> Vec2 {
        let v: Vec3 = Rotation3::from_axis_angle(&Vec3::z_axis(), self.world_rot) * v;
        [v[0], v[1] / ROOT_OF_2 + v[2] / ROOT_OF_2].into()
    }
    fn vec_2_to_3(&self, v: Vec2) -> Vec3 {
        let v: Vec3 = [v[0], v[1] * ROOT_OF_2, 0.].into();
        Rotation3::from_axis_angle(&Vec3::z_axis(), -self.world_rot) * v
    }
    fn pt_2_to_3(&self, p: Pt2) -> Pt3 {
        // 1. extrapolate [x,y] into flat [x,y*sqrt(2),0]
        let p: Pt3 = [p[0], p[1] * ROOT_OF_2, 0.].into();
        let m = {
            let mut m = Transform3::identity();

            // 4. translate relative to camera origin
            m *= Translation3::from(self.world_pos.coords);
            // 3. rotate about the z axis
            m *= Rotation3::from_axis_angle(&Vec3::z_axis(), -self.world_rot);
            // 2. translate from center to tl corner
            m *= Translation3::from(-Vec3::from(SCREEN_TRANS_V3));
            m
        };
        m * p
    }
    fn pt_3_to_2(&self, p: Pt3) -> Pt2 {
        let m = {
            let mut m = Transform3::identity();
            // 3. translate from tl corner to center
            m *= Translation3::from(Vec3::from(SCREEN_TRANS_V3));
            // 2. rotate about the z axis
            m *= Rotation3::from_axis_angle(&Vec3::z_axis(), self.world_rot);
            // 1. translate relative to camera origin
            m *= Translation3::from(-self.world_pos.coords);
            m
        };
        let p: Pt3 = m * p;

        // 4. fuse y and z axes
        [p[0], p[1] / ROOT_OF_2 + p[2] / ROOT_OF_2].into()
    }
    fn rot_of_xy(v: Vec2) -> f32 {
        Rotation2::rotation_between(&Pt2::new(1., 0.).coords, &v).angle()
    }
    fn vel_to_rot_len(&self, mut vel_xyz: Vec3) -> [f32; 2] {
        vel_xyz = vel_xyz.normalize();
        let xy = self.vec_3_to_2(vel_xyz);
        [Self::rot_of_xy(xy), xy.norm()]
    }
    fn vel_to_rot_len_shadow(&self, mut vel_xyz: Vec3) -> [f32; 2] {
        vel_xyz = vel_xyz.normalize();
        vel_xyz[2] = 0.;
        let xy = self.vec_3_to_2(vel_xyz);
        [Self::rot_of_xy(xy), xy.norm()]
    }
}

impl PullLevel {
    fn can_shoot(self) -> bool {
        match self {
            Self::TooLittle | Self::TooMuch => false,
            _ => true,
        }
    }
    // given the initial velocity of a shot arrow in world-space,
    // categorize the "pull level" of the bow shooting it.
    // used to draw the bow and determine if draw power is too low/high.
    fn from_shot_vel(shot_vel: Vec3) -> Self {
        const DIFF: f32 = MAX_VEL - MIN_VEL;
        use PullLevel::*;
        match shot_vel.norm_squared() {
            x if x < MIN_VEL.sqr() => TooLittle,
            x if x < (MIN_VEL + DIFF * 0.25).sqr() => Low,
            x if x < (MIN_VEL + DIFF * 0.50).sqr() => Med,
            x if x < (MIN_VEL + DIFF * 0.75).sqr() => High,
            x if x < (MIN_VEL + DIFF * 1.00).sqr() => Max,
            _ => TooMuch,
        }
    }
}

impl MyGame {
    fn new(ctx: &mut Context, config: UiConfig) -> Self {
        let mut rng = rand::rngs::SmallRng::from_seed([2; 16]);

        match config.net_mode {
            NetMode::Client => {
                let mut e = {
                    let x = TcpStream::connect(config.addr).unwrap();
                    Endpoint::new(x)
                };
                let (archers, baddies, arrows) = match e.recv::<Clientward>().unwrap().unwrap() {
                    Clientward::Welcome { archers, baddies, arrows } => {
                        (archers.into_owned(), baddies.into_owned(), arrows.into_owned())
                    }
                    _ => unreachable!("Always get a WELCOME first"),
                };
                e.stream.set_nonblocking(true).unwrap();
                let net_core = NetCore::Client(e);

                let ui = Ui {
                    current_fullscreen: FullscreenType::Windowed,
                    controlling: archers.len() - 1,
                    config,
                    camera: Camera::default(),
                    last_mouse_at: [0.; 2].into(),
                    rclick_state: None,
                    lclick_state: None,
                    pressing: Default::default(),
                    aim_assist: 0,
                };
                MyGame {
                    net_core,
                    baddies,
                    ticks: 0,
                    archers,
                    arrows,
                    stuck_arrows: vec![],
                    assets: Assets::new(ctx),
                    doodads: starting_doodads(&mut rng),
                    rng,
                    ui,
                }
            }
            NetMode::Server | NetMode::Solo => {
                let net_core = match config.net_mode {
                    NetMode::Server => NetCore::Server {
                        listener: {
                            let x = TcpListener::bind(config.addr).unwrap();
                            x.set_nonblocking(true).unwrap();
                            x
                        },
                        clients: Clients { endpoints: vec![] },
                    },
                    _ => NetCore::Solo,
                };
                let archers = vec![Archer {
                    shot_vel: None,
                    entity: Entity { pos: [0., 0., 0.].into(), vel: [0.; 3].into() },
                }];
                let baddies = (0..NUM_BADDIES)
                    .map(|_| {
                        let mut b = Baddie {
                            stuck_arrows: vec![],
                            health: 1.0,
                            entity: Entity { pos: [0.; 3].into(), vel: [0.; 3].into() },
                            target_archer_index: 0,
                        };
                        Self::reset_baddie(&mut rng, &archers, &mut b);
                        b
                    })
                    .collect();

                let ui = Ui {
                    current_fullscreen: FullscreenType::Windowed,
                    controlling: 0,
                    config,
                    camera: Camera::default(),
                    last_mouse_at: [0.; 2].into(),
                    rclick_state: None,
                    lclick_state: None,
                    pressing: Default::default(),
                    aim_assist: 0,
                };
                MyGame {
                    net_core,
                    baddies,
                    ticks: 0,
                    archers,
                    arrows: vec![],
                    stuck_arrows: vec![],
                    assets: Assets::new(ctx),
                    doodads: starting_doodads(&mut rng),
                    rng,
                    ui,
                }
            }
        }
    }

    // given the set of archers, and `distance`, compute a point on the ground that is
    // that is exactly `distance` away from a random archer, and at least `distance` away from ALL archers.
    fn boundary_point(rng: &mut SmallRng, distance: f32, archers: &[Archer]) -> (Pt3, usize) {
        assert!(!archers.is_empty());
        let angle = rng.gen_range(0., PI * 2.);
        let pos_offset =
            Rotation3::from_axis_angle(&Vec3::z_axis(), angle) * Vec3::new(distance, 0., 0.);
        let index_offset = rng.gen_range(0, archers.len());
        'archer_loop: for i1 in 0..archers.len() {
            let target_archer_index = (i1 + index_offset) % archers.len();
            let try_pt = archers[target_archer_index].entity.pos + pos_offset;
            for (i2, archer) in archers.iter().enumerate() {
                if i2 != target_archer_index && (archer.entity.pos - try_pt).norm() < distance {
                    // this spawn point is too close to `archer`! try another.
                    continue 'archer_loop;
                }
            }
            return (try_pt, target_archer_index);
        }
        unreachable!("any offset is outside of all archers' range for SOME archer as origin")
    }

    fn reset_baddie(rng: &mut SmallRng, archers: &[Archer], b: &mut Baddie) {
        const SPAWN_DISTANCE: f32 = 900.;
        let (try_pt, target_archer_index) = Self::boundary_point(rng, SPAWN_DISTANCE, archers);
        b.health = 1.;
        b.stuck_arrows.clear();
        b.target_archer_index = target_archer_index;
        b.entity.pos = try_pt;
    }

    // recalculate the velocity of the archer this player controls
    fn recalculate_controlled_vel(&mut self) {
        let index = self.ui.controlling;
        let archer = &mut self.archers[index];
        // compute velocity in SCREEN SPACE
        let vel2 = Vec2::new(
            match self.ui.pressing.right {
                Some(true) => 1.,
                Some(false) => -1.,
                _ => 0.,
            },
            match self.ui.pressing.down {
                Some(true) => 1. / ROOT_OF_2,
                Some(false) => -1. / ROOT_OF_2,
                _ => 0.,
            },
        );
        // "unit speed" before being scaled
        let vel3 = self.ui.camera.vec_2_to_3(vel2);

        // slow x,y walk speeds if BOTH x,y are nonzero (walking diagonally)
        let mut speed = if self.ui.pressing.down.is_some() && self.ui.pressing.right.is_some() {
            WALK_SPEED / ROOT_OF_2
        } else {
            WALK_SPEED
        };
        if let Some(shot_vel) = archer.shot_vel {
            // slow speed further if aiming an arrow
            speed *= 0.6;
            let backpedaling = shot_vel.dot(&vel3) < 0.;
            if backpedaling {
                // slow speed further if walking backwards while aiming forwards
                speed *= 0.4;
            }
        }
        // overwrite controlled archer's speed with scaled "unit speed".
        archer.entity.vel = vel3 * speed;
        match &mut self.net_core {
            NetCore::Solo => (),
            NetCore::Server { clients, .. } => {
                let c =
                    Clientward::ArcherEntityResync { index, entity: Cow::Borrowed(&archer.entity) };
                clients.broadcast(&c).unwrap();
            }
            NetCore::Client(endpoint) => {
                endpoint
                    .send(&Serverward::ArcherEntityResync(Cow::Borrowed(&archer.entity)))
                    .unwrap();
            }
        }
    }
}
impl EventHandler for MyGame {
    fn gamepad_button_down_event(
        &mut self,
        _ctx: &mut Context,
        btn: ggez::event::Button,
        id: ggez::event::GamepadId,
    ) {
        dbg!(btn, id);
    }
    fn gamepad_axis_event(
        &mut self,
        _ctx: &mut Context,
        axis: ggez::event::Axis,
        value: f32,
        id: ggez::event::GamepadId,
    ) {
        dbg!(axis, value, id);
    }
    fn key_down_event(
        &mut self,
        ctx: &mut Context,
        keycode: KeyCode,
        _keymods: KeyMods,
        _repeat: bool,
    ) {
        let Ui { pressing, config, aim_assist, current_fullscreen, .. } = &mut self.ui;
        match keycode {
            x if x == config.up => pressing.down = Some(false),
            x if x == config.left => pressing.right = Some(false),
            x if x == config.anticlockwise => pressing.clockwise = Some(false),
            //
            x if x == config.down => pressing.down = Some(true),
            x if x == config.right => pressing.right = Some(true),
            x if x == config.clockwise => pressing.clockwise = Some(true),
            //
            x if x == config.quit => ggez::event::quit(ctx),
            x if x == config.aim_assist => *aim_assist = (*aim_assist + 1) % 3,
            x if x == config.cycle_fullscreen => {
                use FullscreenType::*;
                let n = match current_fullscreen {
                    Windowed => True,
                    True => Desktop,
                    Desktop => Windowed,
                };
                *current_fullscreen = n;
                ggez::graphics::set_fullscreen(ctx, n).unwrap();
            }
            _ => return,
        }
        self.recalculate_controlled_vel();
    }
    fn key_up_event(&mut self, _ctx: &mut Context, keycode: KeyCode, _keymods: KeyMods) {
        let Ui { pressing, config, .. } = &mut self.ui;
        match keycode {
            x if x == config.up && pressing.down == Some(false) => pressing.down = None,
            x if x == config.left && pressing.right == Some(false) => pressing.right = None,
            x if x == config.anticlockwise && pressing.clockwise == Some(false) => {
                pressing.clockwise = None
            }
            //
            x if x == config.down && pressing.down == Some(true) => pressing.down = None,
            x if x == config.right && pressing.right == Some(true) => pressing.right = None,
            x if x == config.clockwise && pressing.clockwise == Some(true) => {
                pressing.clockwise = None
            }
            _ => return,
        }
        self.recalculate_controlled_vel();
    }

    fn update(&mut self, _ctx: &mut Context) -> GameResult<()> {
        // rotate camera
        self.ui.camera.world_rot += match self.ui.pressing.clockwise {
            None => 0.,
            Some(true) => -0.05,
            Some(false) => 0.05,
        };
        // gravity and air resistance on arrows
        for arrow in self.arrows.iter_mut() {
            arrow.vel = Entity::grav_and_air_resist(arrow.vel);
        }
        // move entities in accordance with their velocity
        for e in self
            .baddies
            .iter_mut()
            .map(|b| &mut b.entity)
            .chain(self.arrows.iter_mut())
            .chain(self.archers.iter_mut().map(|b| &mut b.entity))
        {
            e.pos += e.vel;
        }
        // update camera position
        self.ui.camera.world_pos = self.archers[self.ui.controlling].entity.pos;
        self.ticks = self.ticks.wrapping_add(1);

        if let Some(endpoint) = self.net_core.get_endpoint() {
            // {Client} not {Solo, Server}
            while let Some(c) = endpoint.recv::<Clientward>().unwrap() {
                use Clientward::*;
                match c {
                    Welcome { .. } => panic!("already been welcomed!"),
                    AddArcher(archer) => self.archers.push(archer.into_owned()),
                    ArrowHitGround { index, arrow } => {
                        self.assets.audio.twang[self.ticks % 3].play().unwrap();
                        self.arrows.remove(index);
                        self.stuck_arrows.push(arrow.into_owned());
                    }
                    ArrowHitBaddie { arrow_index, baddie_index, baddie } => {
                        self.assets.audio.thud[0].play().unwrap();
                        self.arrows.remove(arrow_index);
                        self.baddies[baddie_index] = baddie.into_owned(); // overwrite
                    }
                    BaddieResync { index, entity } => {
                        self.baddies[index].entity = entity.into_owned();
                    }
                    ArcherEntityResync { index, entity } => {
                        self.archers[index].entity = entity.into_owned()
                    }
                    ArcherShotVelResync { index, shot_vel } => {
                        self.archers[index].shot_vel = shot_vel.into_owned()
                    }
                    ArcherShootArrow { index, entity } => {
                        self.archers[index].shot_vel = None;
                        self.arrows.push(entity.into_owned());
                        self.assets.audio.loose[0].play().unwrap();
                    }
                }
            }
        } else {
            // {Solo, Server} not {Client}
            if let Some((listener, clients)) = self.net_core.get_listener_and_clients() {
                // {Server} not {Solo, Client}
                // handle and forward all client requests
                while let Some((client_index, msg)) = clients.recv_any::<Serverward>().unwrap() {
                    let archer_index = client_index + 1;
                    match msg {
                        Serverward::ArcherEntityResync(entity) => {
                            let c = &Clientward::ArcherEntityResync {
                                index: archer_index,
                                entity: entity.clone(),
                            };
                            clients.broadcast_excepting(&c, client_index).unwrap();
                            self.archers[archer_index].entity = entity.into_owned()
                        }
                        Serverward::ArcherShotVelResync(shot_vel) => {
                            let c = &Clientward::ArcherShotVelResync {
                                index: archer_index,
                                shot_vel: shot_vel.clone(),
                            };
                            clients.broadcast_excepting(&c, client_index).unwrap();
                            self.archers[archer_index].shot_vel = shot_vel.into_owned()
                        }
                        Serverward::ArcherShootArrow(arrow) => {
                            self.archers[archer_index].shot_vel = None;
                            let c = Clientward::ArcherShootArrow {
                                index: archer_index,
                                entity: arrow.clone(),
                            };
                            // even the original archer doesn't shoot the arrow yet
                            clients.broadcast(&c).unwrap();
                            self.arrows.push(arrow.into_owned());
                        }
                    }
                }
                // accept all waiting clients
                while let Ok((stream, _addr)) = listener.accept() {
                    let new_archer = Archer {
                        entity: Entity {
                            pos: Self::boundary_point(&mut self.rng, 80., &self.archers).0,
                            vel: Vec3::new(0., 0., 0.),
                        },
                        shot_vel: None,
                    };

                    // notify existing clients that there is a new SHERIFF IN TOWNNN
                    let c = Clientward::AddArcher(Cow::Borrowed(&new_archer));
                    clients.broadcast(&c).unwrap();

                    // add this archer to list, welcome and add the new client
                    self.archers.push(new_archer);
                    let mut e = Endpoint::new(stream);
                    let c = Clientward::Welcome {
                        archers: Cow::Borrowed(&self.archers),
                        baddies: Cow::Borrowed(&self.baddies),
                        arrows: Cow::Borrowed(&self.arrows),
                    };
                    e.send(&c).unwrap();
                    clients.endpoints.push(e);
                }
            }
            // // arrow-arrow collisions
            // let mut i = IterPairs::new(&mut self.arrows);
            // while let Some([a, b]) = i.next() {
            //     if (a.pos - b.pos).norm() < 50. {
            //         let v = (a.vel + b.vel) * 0.5;
            //         a.vel = v;
            //         b.vel = v;
            //     }
            // }
            // collision events
            let mut draining = Draining::new(&mut self.arrows);
            'entry_loop: while let Some(mut entry) = draining.next() {
                let arrow: &mut Entity = entry.get_mut();
                if arrow.pos[2] >= 0. {
                    // arrow hit the ground!
                    self.assets.audio.twang[self.ticks % 3].play().unwrap();
                    arrow.pos[2] = arrow.pos[2].min(0.);
                    let (index, arrow) = entry.take();
                    if let Some(clients) = self.net_core.get_clients() {
                        // {Server} not {Solo, Client}
                        let c = Clientward::ArrowHitGround { index, arrow: Cow::Borrowed(&arrow) };
                        clients.broadcast(&c).unwrap();
                    }
                    self.stuck_arrows.push(arrow);
                    // stuck in the ground
                    continue 'entry_loop;
                }
                for (baddie_index, baddie) in self.baddies.iter_mut().enumerate() {
                    let mut baddie_dist = arrow.pos.coords - baddie.entity.pos.coords;
                    baddie_dist[2] *= 0.25;
                    if baddie_dist.norm() < 32. {
                        baddie.entity.vel = [0.; 3].into();
                        baddie.health += arrow.pos[2] * 0.01;
                        let (arrow_index, mut arrow) = entry.take();
                        arrow.pos -= baddie.entity.pos.coords;
                        if baddie.health <= 0. {
                            Self::reset_baddie(&mut self.rng, &self.archers, baddie);
                        } else {
                            baddie.stuck_arrows.push(arrow);
                        }
                        if let Some(clients) = self.net_core.get_clients() {
                            // {Server} not {Solo, Client}
                            let c = Clientward::ArrowHitBaddie {
                                arrow_index,
                                baddie_index,
                                baddie: Cow::Borrowed(baddie),
                            };
                            clients.broadcast(&c).unwrap();
                        }
                        self.assets.audio.thud[0].play().unwrap();
                        // stuck in a baddie
                        continue 'entry_loop;
                    }
                }
            }
            const TICKS_PER_UPDATE: usize = 64;
            if self.ticks % TICKS_PER_UPDATE == 0 {
                // recompute baddie trajectory
                for (index, b) in self.baddies.iter_mut().enumerate() {
                    let archerward =
                        self.archers[b.target_archer_index].entity.pos.coords - b.entity.pos.coords;
                    let n = archerward.norm();
                    b.entity.vel = archerward * BADDIE_SPEED / n;
                    if let Some(clients) = self.net_core.get_clients() {
                        // {Server} not {Solo, Client}
                        let c =
                            Clientward::BaddieResync { index, entity: Cow::Borrowed(&b.entity) };
                        clients.broadcast(&c).unwrap();
                    }
                    // if n < 800. && self.rng.gen_bool(0.3) {
                    //     // this baddie shoots an arrow
                    //     let mut vel = archerward * 0.055;
                    //     vel[2] = -n * 0.02;
                    //     for i in 0..3 {
                    //         vel[i] += self.rng.gen_range(0., 2.0);
                    //     }
                    //     self.arrows
                    //         .push(Entity { pos: b.entity.pos + 4. * Vec3::from(TO_ARMS), vel });
                    // }
                }
            }
        }
        Ok(())
    }

    fn mouse_button_down_event(&mut self, _ctx: &mut Context, button: MouseButton, x: f32, y: f32) {
        match button {
            MouseButton::Left => {
                let start = [x, y].into();
                self.ui.lclick_state =
                    Some(LclickState { start, last_pull_level: PullLevel::TooLittle });
                self.archers[self.ui.controlling].shot_vel = Some(Vec3::new(0., 0., 0.));
            }
            MouseButton::Right => {
                let anchor_pt: Pt2 = [x, y].into();
                let anchor_diff: Pt2 = anchor_pt
                    - self.ui.camera.pt_3_to_2(self.archers[self.ui.controlling].entity.pos).coords;
                self.ui.rclick_state =
                    Some(RclickState { anchor_angle: Camera::rot_of_xy(anchor_diff.coords) })
            }
            _ => {}
        }
    }

    fn mouse_button_up_event(&mut self, _ctx: &mut Context, button: MouseButton, _x: f32, _y: f32) {
        match button {
            MouseButton::Left => {
                if let Some(arrow) =
                    self.archers[self.ui.controlling].shot_vel.take().and_then(|shot_vel| {
                        // compute the velocity vector of the arrow shot in current state
                        self.ui.lclick_state = None;
                        match shot_vel.norm_squared() {
                            x if x < MIN_VEL.sqr() => None,
                            x if x > MAX_VEL.sqr() => None,
                            _ => Some(Entity {
                                pos: self.archers[self.ui.controlling].entity.pos
                                    + Vec3::from(TO_ARMS),
                                vel: shot_vel,
                            }),
                        }
                    })
                {
                    // yes! we shoot an arrow with entity data `arrow`
                    match &mut self.net_core {
                        NetCore::Client(endpoint) => endpoint
                            .send(&Serverward::ArcherShootArrow(Cow::Borrowed(&arrow)))
                            .unwrap(),
                        NetCore::Solo => self.arrows.push(arrow),
                        NetCore::Server { clients, .. } => {
                            let c = Clientward::ArcherShootArrow {
                                index: self.ui.controlling,
                                entity: Cow::Borrowed(&arrow),
                            };
                            self.assets.audio.loose[0].play().unwrap();
                            for t in &mut self.assets.audio.taut {
                                t.stop();
                            }
                            clients.broadcast(&c).unwrap();
                            self.arrows.push(arrow);
                        }
                    }
                }
            }
            MouseButton::Right => self.ui.rclick_state = None,
            _ => {}
        }
    }

    fn mouse_motion_event(&mut self, _ctx: &mut Context, x: f32, y: f32, _dx: f32, _dy: f32) {
        let mouse_at: Pt2 = [x, y].into();
        if self.ui.last_mouse_at == mouse_at {
            // optimization and avoid spamming bow taut noises
            return;
        }
        self.ui.last_mouse_at = mouse_at;
        let archer = &mut self.archers[self.ui.controlling];
        if let (Some(LclickState { start, last_pull_level }), (entity, Some(shot_vel))) =
            (&mut self.ui.lclick_state, {
                let Archer { entity, shot_vel } = archer;
                (entity as &Entity, shot_vel)
            })
        {
            // destructured and matched:
            // 1. some left-clicking state, and
            // 2. the state of the controlled archer
            let line_v = start.coords - mouse_at.coords;
            let pull_v = line_v * 0.11;
            let duderel = self.ui.camera.pt_3_to_2(entity.pos).coords - mouse_at.coords;
            let proj_scalar = pull_v.dot(&duderel) / pull_v.dot(&pull_v);
            let proj_v = pull_v * proj_scalar;
            let perp_v = proj_v - duderel;
            let z = perp_v.norm() * -0.07;
            // update recomputed shot_vel: the velocity vector an arrow released in this state gets
            // ... first in 2d space (along the floor)
            *shot_vel = self.ui.camera.vec_2_to_3(pull_v);
            // ... and then overwriting the vertical velocity
            shot_vel[2] = z;

            let new_pull_level = PullLevel::from_shot_vel(*shot_vel);
            if new_pull_level != *last_pull_level {
                fn pl_to_audio_index(pl: PullLevel) -> Option<usize> {
                    Some(match pl {
                        PullLevel::TooMuch | PullLevel::TooLittle => return None,
                        PullLevel::Low => 0,
                        PullLevel::Med => 1,
                        PullLevel::High => 2,
                        PullLevel::Max => 3,
                    })
                }
                // stop previous taut noise, play the new one
                if let Some(index) = pl_to_audio_index(*last_pull_level) {
                    self.assets.audio.taut[index].stop();
                }
                if let Some(index) = pl_to_audio_index(new_pull_level) {
                    self.assets.audio.taut[index].play().unwrap();
                }
                *last_pull_level = new_pull_level;
            }
            match &mut self.net_core {
                NetCore::Solo => {}
                NetCore::Client(endpoint) => {
                    endpoint
                        .send(&Serverward::ArcherShotVelResync(Cow::Borrowed(&archer.shot_vel)))
                        .unwrap();
                }
                NetCore::Server { clients, .. } => {
                    clients
                        .broadcast(&Clientward::ArcherShotVelResync {
                            index: self.ui.controlling,
                            shot_vel: Cow::Borrowed(&archer.shot_vel),
                        })
                        .unwrap();
                }
            }
        }
        if let Some(RclickState { anchor_angle }) = &mut self.ui.rclick_state {
            let diff_is: Pt2 = mouse_at
                - self.ui.camera.pt_3_to_2(self.archers[self.ui.controlling].entity.pos).coords;
            let angle_is = Camera::rot_of_xy(diff_is.coords);
            let angle_diff = angle_is - *anchor_angle;
            self.ui.camera.world_rot += angle_diff;
            *anchor_angle = angle_is;
        }
    }

    ///////////////////////////////////////////////DRAW DRAW DRAW DRAW DRAW DRAW

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        const DUDE_SCALE: [f32; 2] = [2.; 2];
        const ARROW_SCALE: f32 = 1.7;
        const PULL_LIMP_ANGLE: f32 = -0.4;
        const FRAME_TICKS: usize = 6;

        // reset the screen's pixels to green
        graphics::clear(ctx, GREEN);
        // compute 'x' field of texture rectangle for archers.
        // as a function of "time" (self.ticks) this results in a looping animation.
        let archer_walk_x = match self.ticks % (FRAME_TICKS * 6) {
            x if x < FRAME_TICKS * 1 => 0.0,
            x if x < FRAME_TICKS * 2 => 0.2,
            x if x < FRAME_TICKS * 3 => 0.4,
            x if x < FRAME_TICKS * 4 => 0.6,
            x if x < FRAME_TICKS * 5 => 0.4,
            ________________________ => 0.2,
        };
        for (archer_index, a) in self.archers.iter().enumerate() {
            let right_facing = self.ui.camera.archer_facing_right(a);
            // incremetally compute the DrawParam for drawing this archer's texture.
            let facing_dude_scale = [
                //
                DUDE_SCALE[0] * if right_facing { 1. } else { -1. },
                DUDE_SCALE[1],
            ];
            let dude_arm_pos = self.ui.camera.pt_3_to_2(a.entity.pos + Vec3::from(TO_ARMS));
            let dude_param = DrawParam {
                src: Rect { x: 0., y: 0., h: 0.2, w: 0.2 },
                dest: dude_arm_pos.into(),
                color: WHITE,
                scale: facing_dude_scale.into(),
                offset: [0.5, 0.5].into(),
                ..Default::default()
            };
            let main_src = {
                let x = if a.entity.vel != Vec3::new(0., 0., 0.) { archer_walk_x } else { 0.4 };
                Rect { x, y: 0., h: 0.2, w: 0.2 }
            };
            let (arm_src, arm_angle, arrow_angle) = if let Some(shot_vel) = a.shot_vel {
                // this archer is drawing their bow
                let pull_level = PullLevel::from_shot_vel(shot_vel);
                // compute texture rectangle (i.e. which sprite in the sheet)
                // ALL archers' arms are drawn in according to the angle they are aiming
                let arm_src = {
                    use PullLevel::*;
                    let x = match pull_level {
                        TooLittle | TooMuch => 0.,
                        Low => 0.2,
                        Med => 0.4,
                        High => 0.6,
                        Max => 0.8,
                    };
                    Rect { x, y: 0., h: 0.2, w: 0.2 }
                };
                let (arm_angle, arrow_angle) = if pull_level.can_shoot() {
                    let x = Camera::rot_of_xy(self.ui.camera.vec_3_to_2(shot_vel));
                    (if right_facing { x } else { x + PI }, Some(x))
                } else if right_facing {
                    // "PULL_LIMP_ANGLE" is the angle you hold your bow limply (not aiming)
                    // used when player has a BAD pull (too little or too much)
                    (PULL_LIMP_ANGLE, Some(-PULL_LIMP_ANGLE))
                } else {
                    (-PULL_LIMP_ANGLE, Some(PULL_LIMP_ANGLE + PI))
                };

                if archer_index == self.ui.controlling {
                    // the player controls this archer! draw extra UI stuff
                    let lclick_state = self.ui.lclick_state.as_ref().unwrap();
                    let end: Pt2 = ggez::input::mouse::position(ctx).into();
                    let diff = end - lclick_state.start;
                    let difflen = diff.norm();
                    let diffang = Camera::rot_of_xy(diff);
                    let color = if lclick_state.last_pull_level.can_shoot() { WHITE } else { RED };
                    // draw line that player uses to "pull" the bow. White if they COULD loose an arrow
                    graphics::draw(
                        ctx,
                        &self.assets.tex.unit_line,
                        DrawParam {
                            color,
                            dest: lclick_state.start.into(),
                            rotation: diffang,
                            scale: [difflen, 1.].into(),
                            ..Default::default()
                        },
                    )?;
                    if self.ui.aim_assist >= 1 {
                        // minimal aim assist is toggled.
                        // draw the little angle and pitch line overlay
                        let aim_e_v3 = a.entity.pos + shot_vel * 7.;
                        let aim_e_v2 = self.ui.camera.pt_3_to_2(aim_e_v3);
                        let rel_v2 = aim_e_v2 - Vec2::from(SCREEN_TRANS_V2);
                        graphics::draw(
                            ctx,
                            &self.assets.tex.unit_line,
                            DrawParam {
                                scale: [rel_v2.coords.norm(), 1.].into(),
                                rotation: Camera::rot_of_xy(rel_v2.coords),
                                offset: [0., 0.5].into(),
                                color: BLUE,
                                ..dude_param
                            },
                        )?;
                        let aim_e_v3 = a.entity.pos + Vec3::new(shot_vel[0], shot_vel[1], 0.) * 7.;
                        let aim_e_v2 = self.ui.camera.pt_3_to_2(aim_e_v3);
                        let rel_v2 = aim_e_v2 - Vec2::from(SCREEN_TRANS_V2);
                        let dude_feet_pos = self.ui.camera.pt_3_to_2(a.entity.pos);
                        graphics::draw(
                            ctx,
                            &self.assets.tex.unit_line,
                            DrawParam {
                                dest: dude_feet_pos.into(),
                                scale: [rel_v2.coords.norm(), 1.].into(),
                                rotation: Camera::rot_of_xy(rel_v2.coords),
                                offset: [0., 0.5].into(),
                                color: BLACK,
                                ..dude_param
                            },
                        )?;
                        if self.ui.aim_assist >= 2 {
                            // aim assist is set to max. predict the arrow's impact location.
                            // simple solution: create an arrow entity and simulate its flight WHILE its not hit the ground.
                            let mut arrow = Entity {
                                vel: shot_vel,
                                pos: a.entity.pos - Vec3::new(0., 0., 32.),
                            };
                            while arrow.pos[2] < 0. {
                                arrow.vel = Entity::grav_and_air_resist(arrow.vel);
                                arrow.pos += arrow.vel;
                            }
                            arrow.pos[2] = 0.;
                            // ... and draw a cross on the floor where it impacted
                            graphics::draw(
                                ctx,
                                &self.assets.tex.cross,
                                DrawParam {
                                    dest: self.ui.camera.pt_3_to_2(arrow.pos).into(),
                                    scale: [8., 8. / ROOT_OF_2].into(),
                                    ..Default::default()
                                },
                            )?;
                        }
                    }
                }
                (arm_src, arm_angle, arrow_angle)
            } else {
                let arm_src = Rect { x: 0., y: 0., h: 0.2, w: 0.2 };
                let arm_angle = if right_facing { PULL_LIMP_ANGLE } else { -PULL_LIMP_ANGLE };
                (arm_src, arm_angle, None)
            };
            // draw archer layer 1/4 (back arm)
            graphics::draw(
                ctx,
                &self.assets.tex.archer_back,
                DrawParam { src: arm_src, rotation: arm_angle, ..dude_param },
            )?;
            // draw archer layer 2/4 (body)
            graphics::draw(
                ctx,
                &self.assets.tex.archer,
                DrawParam { src: main_src, ..dude_param },
            )?;
            // draw archer layer 3/4 (front arm + bow)
            graphics::draw(
                ctx,
                &self.assets.tex.archer_front,
                DrawParam { src: arm_src, rotation: arm_angle, ..dude_param },
            )?;
            if let Some(arrow_angle) = arrow_angle {
                // draw archer layer 4/4 (nocked arrow)
                let p = DrawParam {
                    dest: (dude_arm_pos + Vec2::new(0., -5.0)).into(),
                    src: Rect { x: 0., y: 0., w: 1., h: 0.25 },
                    scale: [ARROW_SCALE, ARROW_SCALE].into(),
                    rotation: arrow_angle,
                    offset: [0., 0.5].into(),
                    ..dude_param
                };
                self.assets.tex.arrow_batch.add(p);
            }
        }
        fn flatten(p: Pt3) -> Pt3 {
            [p[0], p[1], 0.].into()
        }
        // draw grasses rocks etc. using INSTANCED RENDERING
        for doodad in self.doodads.iter() {
            self.assets.tex.doodads.add(DrawParam {
                src: doodad.kind.rect(),
                dest: self.ui.camera.pt_3_to_2(doodad.pos).into(),
                offset: [0.5, 0.5].into(),
                ..Default::default()
            });
        }
        // define a closure for adding an arrow instance
        // to the arrow batch (for instanced rendering on GPU)
        let arrow_draw =
            |y: f32, arrow: &Entity, arrow_batch: &mut SpriteBatch, camera: &Camera| {
                // arrow's shadow on the ground
                let src = Rect { x: 0., y, w: 1., h: 0.25 };
                let dest = camera.pt_3_to_2(flatten(arrow.pos));
                let [rotation, len] = camera.vel_to_rot_len_shadow(arrow.vel);
                let p = DrawParam {
                    src,
                    dest: dest.into(),
                    scale: [len * ARROW_SCALE, ARROW_SCALE].into(),
                    rotation,
                    color: BLACK,
                    offset: [0.95, 0.5].into(),
                    ..Default::default()
                };
                // arrow in the air
                arrow_batch.add(p);
                let dest = camera.pt_3_to_2(arrow.pos);
                let [rotation, len] = camera.vel_to_rot_len(arrow.vel);
                let p = DrawParam {
                    src,
                    dest: dest.into(),
                    scale: [len * ARROW_SCALE, ARROW_SCALE].into(),
                    rotation,
                    offset: [0.95, 0.5].into(),
                    ..Default::default()
                };
                arrow_batch.add(p);
            };
        // draw baddies in the world
        for b in &self.baddies {
            self.assets.tex.doodads.add(DrawParam {
                src: DoodadKind::Rock.rect(),
                dest: self.ui.camera.pt_3_to_2(b.entity.pos).into(),
                color: Color { r: 1.0, g: b.health, b: b.health, a: 1. },
                scale: [3.1, 5.0].into(),
                offset: [0.5, 0.75].into(),
                ..Default::default()
            });
            for arrow in &b.stuck_arrows {
                let mut a: Entity = Entity::clone(arrow);
                a.pos += b.entity.pos.coords;
                arrow_draw(0.25, &a, &mut self.assets.tex.arrow_batch, &self.ui.camera);
            }
        }
        // invoke the `arrow_draw` closure for all arrows in flight
        for arrow in self.arrows.iter() {
            arrow_draw(0.00, arrow, &mut self.assets.tex.arrow_batch, &self.ui.camera);
        }
        // invoke the `arrow_draw` closure for all arrows on thr ground
        // use a different texture rectangle (headless arrows because stuck in the target)
        for arrow in self.stuck_arrows.iter() {
            arrow_draw(0.25, arrow, &mut self.assets.tex.arrow_batch, &self.ui.camera);
        }
        // fire off instanced rendering on GPU (synchronous)
        graphics::draw(ctx, &self.assets.tex.arrow_batch, DrawParam::default())?;
        graphics::draw(ctx, &self.assets.tex.doodads, DrawParam::default())?;
        // ... and clear the batch again for use in the next frame
        self.assets.tex.arrow_batch.clear();
        self.assets.tex.doodads.clear();
        // finalize.
        graphics::present(ctx)
    }
}
