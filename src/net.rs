use super::*;

#[derive(Deserialize, Serialize, Clone, Debug)]
pub(crate) enum Serverward<'a> {
    ArcherEntityResync(Cow<'a, Entity>),
    ArcherShotVelResync(Cow<'a, Option<Vec3>>),
    ArcherShootArrow(Cow<'a, Entity>),
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub(crate) enum Clientward<'a> {
    Welcome {
        archers: Cow<'a, Vec<Archer>>,
        baddies: Cow<'a, Vec<Baddie>>,
        arrows: Cow<'a, Vec<Entity>>,
    },
    AddArcher(Cow<'a, Archer>),
    ArrowHitGround {
        index: usize,
        arrow: Cow<'a, Entity>,
    },
    ArrowHitBaddie {
        arrow_index: usize,
        baddie_index: usize,
        baddie: Cow<'a, Baddie>,
    },
    BaddieResync {
        index: usize,
        entity: Cow<'a, Entity>,
    },
    ArcherEntityResync {
        index: usize,
        entity: Cow<'a, Entity>,
    },
    ArcherShotVelResync {
        index: usize,
        shot_vel: Cow<'a, Option<Vec3>>,
    },
    ArcherShootArrow {
        index: usize,
        entity: Cow<'a, Entity>,
    },
}

pub(crate) struct Endpoint {
    got: usize,
    inbuf: Vec<u8>,
    pub stream: TcpStream,
}
impl Endpoint {
    const BUFCAP: usize = 2048;
    pub fn new(stream: TcpStream) -> Self {
        let mut inbuf = Vec::with_capacity(Self::BUFCAP);
        unsafe { inbuf.set_len(Self::BUFCAP) };
        Self { inbuf, stream, got: 0 }
    }
    pub fn send<M: Serialize>(&mut self, msg: &M) -> Result<(), ()> {
        bincode::serialize_into(&mut self.stream, msg).map_err(drop)
    }
    pub fn recv<M: DeserializeOwned>(&mut self) -> Result<Option<M>, ()> {
        loop {
            let mut monitored = MonitoredReader::from(&self.inbuf[..self.got]);
            match bincode::deserialize_from::<_, M>(&mut monitored) {
                Ok(msg) => {
                    let read_bytes = monitored.bytes_read();
                    self.inbuf.drain(0..read_bytes as usize);
                    self.inbuf.resize(Self::BUFCAP, 0u8);
                    self.got -= read_bytes;
                    return Ok(Some(msg));
                }
                Err(e) => match *e {
                    bincode::ErrorKind::Io(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                        // try read from stream
                        let count = match self.stream.read(&mut self.inbuf[self.got..]) {
                            Ok(0) => return Ok(None),
                            Ok(count) => count,
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                                return Ok(None);
                            }
                            Err(e) => {
                                println!("READ ERR {:?}", e);
                                return Err(());
                            }
                        };
                        if count == 0 {
                            return Ok(None);
                        }
                        self.got += count;
                    }
                    e => {
                        println!("ENDPOINT err! {:?}", e);
                        return Err(());
                    }
                },
            }
        }
    }
}
pub struct MonitoredReader<R: Read> {
    bytes: usize,
    r: R,
}
impl<R: Read> From<R> for MonitoredReader<R> {
    fn from(r: R) -> Self {
        Self { r, bytes: 0 }
    }
}
impl<R: Read> MonitoredReader<R> {
    pub fn bytes_read(&self) -> usize {
        self.bytes
    }
}
impl<R: Read> Read for MonitoredReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, std::io::Error> {
        let n = self.r.read(buf)?;
        self.bytes += n;
        Ok(n)
    }
}

pub(crate) enum NetCore {
    Server { listener: TcpListener, clients: Clients },
    Client(Endpoint),
    Solo,
}
pub(crate) struct Clients {
    pub endpoints: Vec<Endpoint>,
}
impl Clients {
    pub fn broadcast<M: Serialize>(&mut self, c: &M) -> Result<(), ()> {
        for e in self.endpoints.iter_mut() {
            e.send(c)?;
        }
        Ok(())
    }
    pub fn broadcast_excepting<M: Serialize>(&mut self, c: &M, index: usize) -> Result<(), ()> {
        for (i, e) in self.endpoints.iter_mut().enumerate() {
            if i != index {
                e.send(c)?;
            }
        }
        Ok(())
    }
    pub fn recv_any<M: DeserializeOwned>(&mut self) -> Result<Option<(usize, M)>, ()> {
        for (index, e) in self.endpoints.iter_mut().enumerate() {
            match e.recv() {
                Ok(Some(msg)) => return Ok(Some((index, msg))),
                Err(()) => return Err(()),
                Ok(None) => continue,
            }
        }
        Ok(None)
    }
}
