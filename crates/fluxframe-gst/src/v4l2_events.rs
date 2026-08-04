//! `V4L2_EVENT_PRI_CLIENT_USAGE` subscription for v4l2loopback nodes.
//!
//! ## Why this exists
//!
//! Telling "someone is consuming our virtual camera" from "nobody is"
//! used to be inferred from `/proc/*/fd` plus an inotify open/close
//! balance. That inference is structurally unable to produce an
//! authoritative *negative* on a normal desktop: an unprivileged process
//! cannot read `/proc/<pid>/fd` for root-owned or sandboxed processes, so
//! "nobody holds the device" and "I am not allowed to see who holds the
//! device" are indistinguishable. Any drift in the open/close balance
//! then latches "present" for the rest of the process lifetime, and the
//! camera is never released.
//!
//! v4l2loopback ≥ 0.13 exposes the answer directly. The producer
//! subscribes to `V4L2_EVENT_PRI_CLIENT_USAGE` on the fd it already
//! holds and the driver reports capture-side usage as an absolute value
//! — no deltas to accumulate, no `/proc` access, and sandboxed consumers
//! are as visible as any other.
//!
//! ## Semantics, as implemented by v4l2loopback 0.15.3
//!
//! * The payload is `struct v4l2_event_client_usage { __u32 count; }`,
//!   but the driver fills it with `!has_capture_token(stream_tokens)` —
//!   so in practice it is a 0/1 flag meaning "at least one capture
//!   client is streaming", not a client count. [`ClientUsage::count`]
//!   keeps the driver's wording; treat any non-zero value as "attached".
//! * Events are queued from `VIDIOC_STREAMON` / `VIDIOC_STREAMOFF`, not
//!   from `open` / `close`. This is strictly better than counting opens:
//!   an application that merely enumerates devices (open, query, close)
//!   is no longer mistaken for a consumer.
//! * The driver installs `replace` and `merge` ops, so a full event
//!   queue collapses to the newest value instead of dropping it. We
//!   still drain to `EAGAIN` in [`ConsumerWatch::poll`] — that is
//!   belt-and-braces, not a correctness requirement.
//! * Subscribing with `V4L2_EVENT_SUB_FL_SEND_INITIAL` makes the driver
//!   queue the current value immediately, so a fresh subscription knows
//!   the state without waiting for a change. A driver that ignores the
//!   flag simply yields `None` until the first real transition; callers
//!   must not read "no event yet" as "no consumer".
//!
//! ## Unsafe
//!
//! This is the only module in the workspace allowed to use `unsafe`
//! (see the `[lints]` block in this crate's `Cargo.toml`). It is
//! confined to two ioctls whose argument types are checked by
//! compile-time size assertions against the bindgen-generated structs.

use std::io;
use std::os::fd::{AsRawFd, OwnedFd, RawFd};

use v4l::v4l_sys::{
    V4L2_EVENT_PRIVATE_START, V4L2_EVENT_SUB_FL_SEND_INITIAL, v4l2_event, v4l2_event_subscription,
};

/// `V4L2_EVENT_PRI_CLIENT_USAGE`, as defined by v4l2loopback:
/// `V4L2_EVENT_PRIVATE_START + 0x08E0_0000 + 1`.
///
/// The offset is the driver's own namespace marker; it is not part of
/// any kernel UAPI header, which is why it is spelled out here.
const V4L2_EVENT_PRI_CLIENT_USAGE: u32 = V4L2_EVENT_PRIVATE_START + 0x08E0_0000 + 1;

// ---------------------------------------------------------------------------
// ioctl request encoding
// ---------------------------------------------------------------------------
//
// `v4l` keeps its `_IOR!` / `_IOW!` macros private, so the asm-generic
// encoding is reproduced here. It is shared by every architecture Linux
// supports except alpha/mips/powerpc/sparc, none of which this crate
// targets (the whole crate is `compile_error!`-gated to Linux and the
// daemon ships on x86_64/aarch64).
//
// The size of the payload struct is part of the request number, so a
// mismatched struct definition does not corrupt memory — it produces a
// different request number and the kernel answers `ENOTTY`. The
// assertions below turn that silent-degradation failure mode into a
// build error instead.

const IOC_NRSHIFT: u32 = 0;
const IOC_TYPESHIFT: u32 = IOC_NRSHIFT + 8;
const IOC_SIZESHIFT: u32 = IOC_TYPESHIFT + 8;
const IOC_DIRSHIFT: u32 = IOC_SIZESHIFT + 14;
const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;

const fn ioc(dir: u32, ty: u32, nr: u32, size: usize) -> u64 {
    ((dir as u64) << IOC_DIRSHIFT)
        | ((ty as u64) << IOC_TYPESHIFT)
        | ((nr as u64) << IOC_NRSHIFT)
        | ((size as u64) << IOC_SIZESHIFT)
}

/// `VIDIOC_DQEVENT` — `_IOR('V', 89, struct v4l2_event)`.
const VIDIOC_DQEVENT: u64 = ioc(IOC_READ, b'V' as u32, 89, size_of::<v4l2_event>());
/// `VIDIOC_SUBSCRIBE_EVENT` — `_IOW('V', 90, struct v4l2_event_subscription)`.
const VIDIOC_SUBSCRIBE_EVENT: u64 = ioc(
    IOC_WRITE,
    b'V' as u32,
    90,
    size_of::<v4l2_event_subscription>(),
);

// The kernel encodes these sizes into the request number; if bindgen
// ever produces a differently-sized struct the ioctls would silently
// start returning ENOTTY at runtime. Fail the build instead.
const _: () = assert!(size_of::<v4l2_event>() == 136);
const _: () = assert!(size_of::<v4l2_event_subscription>() == 32);

/// A capture-usage reading taken from the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClientUsage {
    /// The driver's `count` field. v4l2loopback reports 0 or 1; treat
    /// any non-zero value as "a capture client is streaming".
    pub count: u32,
}

impl ClientUsage {
    /// Is at least one capture client streaming from the node?
    #[must_use]
    pub fn attached(self) -> bool {
        self.count > 0
    }
}

/// An armed `V4L2_EVENT_PRI_CLIENT_USAGE` subscription.
///
/// Owns a `dup(2)` of the producer's fd rather than borrowing it. The
/// original lives inside the output pipeline and is closed when that
/// pipeline drops; a borrowed raw fd would let this watch outlive it and
/// end up polling a descriptor the kernel has since handed to something
/// else. `dup` also shares the underlying open file description, so it
/// does not consume one of v4l2loopback's `max_openers` slots.
#[derive(Debug)]
pub struct ConsumerWatch {
    fd: OwnedFd,
}

impl ConsumerWatch {
    /// Subscribe to capture-usage events on a duplicate of `fd`.
    ///
    /// Requests `V4L2_EVENT_SUB_FL_SEND_INITIAL` so the driver reports
    /// the current state immediately instead of only on the next change.
    ///
    /// # Errors
    ///
    /// Returns the OS error if `dup` or `VIDIOC_SUBSCRIBE_EVENT` fails.
    /// `ENOTTY` / `EINVAL` mean the node does not implement the event
    /// (a non-loopback sink, or v4l2loopback older than 0.13) and the
    /// caller should fall back to a heuristic detector.
    /// # Safety contract
    ///
    /// `fd` must be an open V4L2 descriptor for the duration of this
    /// call. It is duplicated immediately, so the caller may close it
    /// afterwards — the watch does not borrow it.
    pub fn subscribe(fd: RawFd) -> io::Result<Self> {
        // SAFETY: `fd` is open for the duration of the call per the
        // contract above, and the borrow ends with `try_clone_to_owned`.
        let borrowed = unsafe { std::os::fd::BorrowedFd::borrow_raw(fd) };
        let owned = borrowed.try_clone_to_owned()?;

        let mut sub = v4l2_event_subscription {
            type_: V4L2_EVENT_PRI_CLIENT_USAGE,
            id: 0,
            flags: V4L2_EVENT_SUB_FL_SEND_INITIAL,
            reserved: [0; 5],
        };
        // SAFETY: `sub` is a live, fully-initialised
        // `v4l2_event_subscription`; `VIDIOC_SUBSCRIBE_EVENT` is
        // `_IOW`, so the kernel only reads from it. The request number
        // encodes `size_of::<v4l2_event_subscription>()`, asserted above
        // to match the kernel's definition.
        unsafe {
            v4l::v4l2::ioctl(
                owned.as_raw_fd(),
                VIDIOC_SUBSCRIBE_EVENT,
                std::ptr::from_mut(&mut sub).cast(),
            )?;
        }
        Ok(Self { fd: owned })
    }

    /// Wait up to `timeout_ms` for a usage event and return the newest
    /// reading, or `None` if the wait timed out.
    ///
    /// Always drains the whole queue and returns the last value: a stale
    /// reading is worse than no reading, and the caller acts on the
    /// result immediately.
    ///
    /// `timeout_ms` must be bounded by the caller's shutdown
    /// granularity — this blocks, and the detector thread has to stay
    /// responsive to its stop flag.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor reports `POLLERR` / `POLLHUP`
    /// (device removed, module unloaded) or if `VIDIOC_DQEVENT` fails
    /// for a reason other than "queue empty". Either way the
    /// subscription is dead and the caller must fall back; retrying
    /// would spin, because `poll` returns immediately on a errored fd.
    pub fn poll(&self, timeout_ms: i32) -> io::Result<Option<ClientUsage>> {
        let mut pfd = libc::pollfd {
            fd: self.fd.as_raw_fd(),
            events: libc::POLLPRI,
            revents: 0,
        };
        // SAFETY: single-element array of a live `pollfd` whose `fd` is
        // owned by `self`; `poll` writes only `revents`.
        let ret = unsafe { libc::poll(std::ptr::from_mut(&mut pfd), 1, timeout_ms) };
        if ret < 0 {
            let err = io::Error::last_os_error();
            // A signal during the wait is not a failure of the watch.
            if err.kind() == io::ErrorKind::Interrupted {
                return Ok(None);
            }
            return Err(err);
        }
        if ret == 0 {
            return Ok(None);
        }
        if pfd.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
            return Err(io::Error::other(format!(
                "loopback event fd is no longer usable (revents={:#x})",
                pfd.revents
            )));
        }

        let mut latest = None;
        loop {
            match self.dequeue()? {
                Dequeued::Usage(usage) => latest = Some(usage),
                // Keep draining: stopping on a foreign event would leave
                // the queue non-empty and `POLLPRI` permanently raised,
                // turning the caller's poll loop into a busy loop.
                Dequeued::Other => {}
                Dequeued::Empty => return Ok(latest),
            }
        }
    }

    /// Dequeue exactly one event from the driver's queue.
    fn dequeue(&self) -> io::Result<Dequeued> {
        // Zeroed rather than uninitialised: `VIDIOC_DQEVENT` is `_IOR`,
        // so the kernel fills the struct, but zeroing keeps the value
        // valid Rust even on the error path.
        let mut ev: v4l2_event = unsafe { std::mem::zeroed() };
        // SAFETY: `ev` is a live, fully-initialised `v4l2_event`, and
        // `VIDIOC_DQEVENT` writes exactly `size_of::<v4l2_event>()`
        // bytes into it — the same size encoded in the request number.
        let res = unsafe {
            v4l::v4l2::ioctl(
                self.fd.as_raw_fd(),
                VIDIOC_DQEVENT,
                std::ptr::from_mut(&mut ev).cast(),
            )
        };
        if let Err(e) = res {
            // `EAGAIN`/`ENOENT` both mean "nothing queued" depending on
            // kernel version; neither is a failure.
            return match e.raw_os_error() {
                Some(libc::EAGAIN | libc::ENOENT) => Ok(Dequeued::Empty),
                _ => Err(e),
            };
        }
        if ev.type_ != V4L2_EVENT_PRI_CLIENT_USAGE {
            // We never subscribe to anything else, but the event queue
            // belongs to the file handle and is shared in principle.
            return Ok(Dequeued::Other);
        }
        // Read the payload out of the union's raw byte view rather than
        // naming a union field: `v4l2_event_client_usage` is private to
        // v4l2loopback and absent from the kernel headers bindgen sees.
        // SAFETY: `u` is a 64-byte union; `data` is its byte-array view,
        // valid for any bit pattern.
        let bytes = unsafe { ev.u.data };
        let count = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        Ok(Dequeued::Usage(ClientUsage { count }))
    }
}

/// Outcome of a single `VIDIOC_DQEVENT`.
enum Dequeued {
    /// Nothing left in the queue.
    Empty,
    /// An event we did not subscribe to; drain past it.
    Other,
    /// A capture-usage reading.
    Usage(ClientUsage),
}

#[cfg(test)]
mod tests {
    use super::{
        IOC_READ, IOC_WRITE, V4L2_EVENT_PRI_CLIENT_USAGE, VIDIOC_DQEVENT, VIDIOC_SUBSCRIBE_EVENT,
        ioc,
    };

    #[test]
    fn client_usage_event_number_matches_v4l2loopback() {
        // V4L2_EVENT_PRIVATE_START (0x0800_0000) + 0x08E0_0000 + 1.
        assert_eq!(V4L2_EVENT_PRI_CLIENT_USAGE, 0x10E0_0001);
    }

    #[test]
    fn ioctl_request_numbers_match_videodev2_h() {
        // _IOR('V', 89, struct v4l2_event) with sizeof == 136, and
        // _IOW('V', 90, struct v4l2_event_subscription) with sizeof == 32.
        // Hard-coded so a change in either struct is caught here as well
        // as by the size assertions.
        assert_eq!(VIDIOC_DQEVENT, 0x8088_5659);
        assert_eq!(VIDIOC_SUBSCRIBE_EVENT, 0x4020_565A);
    }

    #[test]
    fn ioc_encoding_matches_asm_generic_layout() {
        // dir in bits 30..32, size in 16..30, type in 8..16, nr in 0..8.
        assert_eq!(ioc(IOC_READ, 0, 0, 0), 0x8000_0000);
        assert_eq!(ioc(IOC_WRITE, 0, 0, 0), 0x4000_0000);
        assert_eq!(ioc(0, 0, 0xAB, 0), 0xAB);
        assert_eq!(ioc(0, 0xCD, 0, 0), 0xCD00);
        assert_eq!(ioc(0, 0, 0, 4), 0x0004_0000);
    }
}
