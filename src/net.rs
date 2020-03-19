use super::*;

#[derive(Deserialize, Serialize)]
pub(crate) enum Clientward {
    Welcome { archers: Vec<Archer>, baddies: Vec<Baddie>, arrows: Vec<Entity> },
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
    pub fn send_clientward(&mut self, c: &Clientward) -> Result<(), ()> {
        let b = bincode::serialize(c).map_err(drop)?;
        println!("LEN={:?}", b.len());
        use std::io::Write;
        self.stream.write_all(&b).map_err(drop)
    }
    pub fn recv_clientward(&mut self) -> Result<Option<Clientward>, ()> {
        loop {
            println!("loop");
            match bincode::deserialize::<Clientward>(&self.inbuf[..self.got]) {
                Ok(msg) => {
                    println!("ok");
                    let read_bytes = bincode::serialized_size(&msg).unwrap();
                    self.inbuf.drain(0..read_bytes as usize);
                    unsafe { self.inbuf.set_len(Self::BUFCAP) };
                    return Ok(Some(msg));
                }
                Err(e) => match *e {
                    bincode::ErrorKind::Io(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                        println!("ok not enough to build a message. read!");
                        // try read from stream
                        let count = self.stream.read(&mut self.inbuf[self.got..]).map_err(drop)?;
                        println!("count == {}", count);
                        if count == 0 {
                            return Ok(None);
                        }
                        self.got += count;
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
