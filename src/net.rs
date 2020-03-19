use super::*;

#[derive(Deserialize, Serialize)]
pub(crate) enum Clientward {
    Welcome { archers: Vec<Archer>, baddies: Vec<Baddie>, arrows: Vec<Entity> },
}

pub(crate) struct Endpoint {
    inbuf: Vec<u8>,
    pub stream: TcpStream,
}
impl Endpoint {
    pub fn new(stream: TcpStream) -> Self {
        Self { inbuf: vec![], stream }
    }
    pub fn send_clientward(&mut self, c: &Clientward) -> Result<(), ()> {
        bincode::serialize_into(&mut self.stream, c).map_err(drop)
    }
    pub fn recv_clientward(&mut self) -> Result<Option<Clientward>, ()> {
        loop {
            match bincode::deserialize::<Clientward>(&self.inbuf) {
                Ok(msg) => {
                    let read_bytes = bincode::serialized_size(&msg).unwrap();
                    self.inbuf.drain(0..read_bytes as usize);
                    return Ok(Some(msg));
                }
                Err(e) => match *e {
                    bincode::ErrorKind::Io(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                        // try read from stream
                        let count = self.stream.read_to_end(&mut self.inbuf).map_err(drop)?;
                        if count == 0 {
                            return Ok(None);
                        }
                        // continue
                    }
                    e => return Err(println!("err! {:?}", e)),
                },
            }
        }
    }
}

pub(crate) enum NetCore {
    Server { listener: TcpListener, clients: Vec<Endpoint> },
    Client(Endpoint),
    Solo,
}
