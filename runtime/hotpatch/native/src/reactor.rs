//! Native `mio::Poll` ownership and zero-loss reactor migration.
//!
//! A reloaded PyO3 module must not take the application's sockets with it.
//! The replacement reactor creates a fresh `mio::Poll` (its own epoll or
//! kqueue descriptor) and adopts every `(fd, token, interest)` registration
//! from the old reactor.  The sockets themselves are untouched: their kernel
//! receive queues, connection state, and blocking mode survive, so no datagram
//! or accepted connection can be lost by the swap.
//!
//! Adoption is transactional at the registration level: if any descriptor
//! cannot be re-armed, the descriptors already moved into the new poller are
//! removed again before the error is returned.

use std::collections::BTreeMap;
use std::io;
use std::os::fd::{AsRawFd, RawFd};
use std::time::Duration;

use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Token};

#[derive(Clone, Copy, Debug)]
pub struct Registration {
    pub fd: RawFd,
    pub token: u64,
    pub readable: bool,
    pub writable: bool,
}

impl Registration {
    fn interest(self) -> Interest {
        match (self.readable, self.writable) {
            (true, true) => Interest::READABLE | Interest::WRITABLE,
            (true, false) => Interest::READABLE,
            (false, true) => Interest::WRITABLE,
            (false, false) => Interest::READABLE,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Readiness {
    pub token: u64,
    pub readable: bool,
    pub writable: bool,
}

/// A poller plus the interest registry it owns.
pub struct Reactor {
    poll: Option<Poll>,
    registrations: BTreeMap<RawFd, Registration>,
    events: Events,
}

impl Reactor {
    pub fn new() -> io::Result<Self> {
        Ok(Self {
            poll: Some(Poll::new()?),
            registrations: BTreeMap::new(),
            events: Events::with_capacity(128),
        })
    }

    pub fn poll_fd(&self) -> Option<RawFd> {
        self.poll.as_ref().map(AsRawFd::as_raw_fd)
    }

    pub fn is_closed(&self) -> bool {
        self.poll.is_none()
    }

    pub fn register(
        &mut self,
        fd: RawFd,
        token: u64,
        readable: bool,
        writable: bool,
    ) -> io::Result<()> {
        if fd < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot register a negative descriptor",
            ));
        }
        if !readable && !writable {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "register at least one of readable/writable",
            ));
        }
        let poll = self
            .poll
            .as_mut()
            .ok_or_else(|| io::Error::other("reactor is closed"))?;
        let registration = Registration {
            fd,
            token,
            readable,
            writable,
        };
        let mut source = SourceFd(&fd);
        poll.registry()
            .register(&mut source, Token(token as usize), registration.interest())?;
        self.registrations.insert(fd, registration);
        Ok(())
    }

    pub fn deregister(&mut self, fd: RawFd) -> io::Result<()> {
        let poll = self
            .poll
            .as_mut()
            .ok_or_else(|| io::Error::other("reactor is closed"))?;
        let mut source = SourceFd(&fd);
        poll.registry().deregister(&mut source)?;
        self.registrations.remove(&fd);
        Ok(())
    }

    pub fn snapshot(&self) -> Vec<Registration> {
        self.registrations.values().copied().collect()
    }

    /// Adopt every registration from `other` without closing any socket.
    pub fn adopt_from(&mut self, other: &Reactor) -> io::Result<usize> {
        if let (Some(mine), Some(theirs)) = (self.poll_fd(), other.poll_fd()) {
            if mine == theirs {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "source and destination share the live poll descriptor",
                ));
            }
        }
        let mut adopted = Vec::new();
        for registration in other.snapshot() {
            match self.register(
                registration.fd,
                registration.token,
                registration.readable,
                registration.writable,
            ) {
                Ok(()) => adopted.push(registration.fd),
                Err(error) => {
                    for fd in adopted {
                        let _ = self.deregister(fd);
                    }
                    return Err(error);
                }
            }
        }
        Ok(adopted.len())
    }

    /// Poll for readiness.  `timeout = None` blocks indefinitely.
    pub fn wait(&mut self, timeout: Option<Duration>) -> io::Result<Vec<Readiness>> {
        let poll = self
            .poll
            .as_mut()
            .ok_or_else(|| io::Error::other("reactor is closed"))?;
        poll.poll(&mut self.events, timeout)?;
        Ok(self
            .events
            .iter()
            .map(|event| Readiness {
                token: event.token().0 as u64,
                readable: event.is_readable(),
                writable: event.is_writable(),
            })
            .collect())
    }

    /// Close the poll descriptor.  Registered sockets stay open.
    pub fn close(&mut self) {
        self.poll.take();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::UdpSocket;

    fn datagram_pair() -> (UdpSocket, UdpSocket) {
        let server = UdpSocket::bind("127.0.0.1:0").expect("server bind");
        let client = UdpSocket::bind("127.0.0.1:0").expect("client bind");
        server.set_nonblocking(true).unwrap();
        client.set_nonblocking(true).unwrap();
        (server, client)
    }

    #[test]
    fn migration_preserves_queued_datagrams() {
        let (server, client) = datagram_pair();
        let mut old = Reactor::new().unwrap();
        old.register(server.as_raw_fd(), 7, true, false).unwrap();
        client
            .send_to(b"queued-before-migration", server.local_addr().unwrap())
            .unwrap();

        let mut new = Reactor::new().unwrap();
        assert_ne!(old.poll_fd(), new.poll_fd());
        assert_eq!(new.adopt_from(&old).unwrap(), 1);
        old.close();
        assert!(old.is_closed());

        let readiness = new.wait(Some(Duration::from_secs(1))).unwrap();
        assert!(readiness
            .iter()
            .any(|event| event.token == 7 && event.readable));
        let mut buffer = [0u8; 64];
        let (length, _) = server.recv_from(&mut buffer).unwrap();
        assert_eq!(&buffer[..length], b"queued-before-migration");
        assert_eq!(new.poll_fd().is_some(), true);
    }

    #[test]
    fn adoption_of_an_empty_registry_is_a_noop() {
        let other = Reactor::new().unwrap();
        let mut reactor = Reactor::new().unwrap();
        assert_eq!(reactor.adopt_from(&other).unwrap(), 0);
    }

    #[test]
    fn register_requires_an_interest_and_a_real_fd() {
        let mut reactor = Reactor::new().unwrap();
        assert!(reactor.register(-1, 1, true, false).is_err());
        assert!(reactor.register(3, 1, false, false).is_err());
    }

    #[test]
    fn registration_snapshot_round_trips() {
        let (server, _client) = datagram_pair();
        let mut reactor = Reactor::new().unwrap();
        reactor.register(server.as_raw_fd(), 3, true, true).unwrap();
        let snapshot = reactor.snapshot();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].token, 3);
        assert!(snapshot[0].readable && snapshot[0].writable);
        reactor.deregister(server.as_raw_fd()).unwrap();
        assert!(reactor.snapshot().is_empty());
    }
}
