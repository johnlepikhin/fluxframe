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
//! This module is the reason the crate opts out of the workspace's
//! `unsafe_code = "forbid"` (see the `[lints]` block in `Cargo.toml`,
//! and the `#![deny(unsafe_code)]` in `lib.rs` that keeps the
//! exemption scoped to this module). There are six `unsafe` blocks,
//! all of them single libc/ioctl calls:
//!
//! 1. `fcntl(fd, F_GETFL)` in [`ConsumerWatch::subscribe`], to verify
//!    the descriptor is `O_NONBLOCK`.
//! 2. `VIDIOC_SUBSCRIBE_EVENT` in [`ConsumerWatch::subscribe`].
//! 3. `poll(2)` in [`ConsumerWatch::poll`].
//! 4. `mem::zeroed::<v4l2_event>()` in `ConsumerWatch::dequeue`.
//! 5. `VIDIOC_DQEVENT` in `ConsumerWatch::dequeue`.
//! 6. Reading the event union through its byte view (`ev.u.data`) in
//!    `ConsumerWatch::dequeue`.
//!
//! The ioctl argument types are pinned by compile-time size assertions
//! against the bindgen-generated structs, so a layout drift is a build
//! error rather than a silent `ENOTTY`.

use std::fs::OpenOptions;
use std::io;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

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
// daemon ships on x86_64/aarch64). The architectures that use a
// different layout are rejected explicitly below rather than silently
// building code that would issue the wrong ioctl.
//
// The size of the payload struct is part of the request number, so a
// mismatched struct definition does not corrupt memory — it produces a
// different request number and the kernel answers `ENOTTY`. The
// assertions below turn that silent-degradation failure mode into a
// build error instead.

#[cfg(any(
    target_arch = "mips",
    target_arch = "mips64",
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "sparc",
    target_arch = "sparc64"
))]
compile_error!(
    "this module hard-codes the asm-generic ioctl encoding; mips/powerpc/sparc use a \
     different direction-bit layout and would issue the wrong request number"
);

const IOC_NRSHIFT: u32 = 0;
const IOC_TYPESHIFT: u32 = IOC_NRSHIFT + 8;
const IOC_SIZESHIFT: u32 = IOC_TYPESHIFT + 8;
const IOC_DIRSHIFT: u32 = IOC_SIZESHIFT + IOC_SIZEBITS;
const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;

/// Number of bits the size field occupies in the request number.
const IOC_SIZEBITS: u32 = 14;

/// # Panics
///
/// Panics if `size` does not fit in the 14-bit size field: an oversized
/// payload would overflow into the direction bits and silently produce a
/// different — and wrong — request number. Every call site is a `const`,
/// so this is a build error, not a runtime one.
const fn ioc(dir: u32, ty: u32, nr: u32, size: usize) -> u64 {
    assert!(
        size < (1 << IOC_SIZEBITS),
        "ioctl payload does not fit the 14-bit size field"
    );
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
//
// `v4l2_event` embeds a `timespec`, so its size is pointer-width
// dependent; the literal below is the 64-bit layout. `ioc()` derives
// the request number from `size_of` either way, so a 32-bit build is
// still correct — it just cannot be checked against this constant.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(
    size_of::<v4l2_event>() == 136,
    "unexpected v4l2_event layout: VIDIOC_DQEVENT would encode the wrong size"
);
const _: () = assert!(
    size_of::<v4l2_event_subscription>() == 32,
    "unexpected v4l2_event_subscription layout: VIDIOC_SUBSCRIBE_EVENT would encode the wrong size"
);

/// Upper bound on `VIDIOC_DQEVENT` calls per [`ConsumerWatch::poll`].
///
/// The queue is normally at most one event deep (the driver collapses it
/// via its `replace`/`merge` ops), so this only caps the pathological
/// case where something else queues events on the shared file
/// description faster than we drain them.
const MAX_DRAIN: usize = 64;

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
/// Owns a `dup(2)` of the producer's fd rather than borrowing it: a
/// borrowed raw fd would let this watch outlive the pipeline and end up
/// polling a descriptor the kernel has since handed to something else.
///
/// `dup` shares the underlying *open file description*, so the watch
/// costs no extra `max_openers` slot — but it also **prolongs the life
/// of that description**. As long as a `ConsumerWatch` exists,
/// `v4l2_loopback_close()` does not run and the OUTPUT token is not
/// released, so rebuilding the output pipeline would fail with `EBUSY`.
/// A `ConsumerWatch` must therefore be dropped *before* the
/// `OutputPipeline` it was subscribed on.
///
/// Sharing the file status flags is also why `O_NONBLOCK` cannot be set
/// on the duplicate independently — see [`ConsumerWatch::subscribe`].
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
    /// The descriptor is duplicated immediately, so the borrow ends when
    /// this returns — the watch does not keep `fd` alive, it keeps its
    /// own duplicate.
    ///
    /// # Requirements
    ///
    /// **`fd` must already be open with `O_NONBLOCK`.** The kernel's
    /// `v4l2_event_dequeue()` blocks in `wait_event_interruptible()`
    /// instead of returning `ENOENT` when the file is not non-blocking,
    /// which would wedge the drain loop in [`poll`](Self::poll)
    /// forever — and, because the detector thread is joined on drop,
    /// deadlock shutdown. `dup(2)` shares file status flags with the
    /// original, so this cannot be fixed up locally; it is checked and
    /// rejected instead. (`v4l::Device::with_path` does open with
    /// `O_NONBLOCK`, but that is an implementation detail of another
    /// crate, hence the explicit check.)
    ///
    /// # Errors
    ///
    /// Returns the OS error if `dup`, `fcntl(F_GETFL)` or
    /// `VIDIOC_SUBSCRIBE_EVENT` fails. `EINVAL` with
    /// [`io::ErrorKind::InvalidInput`] means `fd` is a blocking
    /// descriptor (see above). `ENOTTY` / `EINVAL` from the ioctl mean
    /// the node does not implement the event (a non-loopback sink, or
    /// v4l2loopback older than 0.13) and the caller should fall back to
    /// a heuristic detector.
    pub fn subscribe(fd: BorrowedFd<'_>) -> io::Result<Self> {
        let owned = fd.try_clone_to_owned()?;

        // SAFETY: `owned` is a live descriptor; `F_GETFL` takes no
        // argument and only reads the file status flags.
        let flags = unsafe { libc::fcntl(owned.as_raw_fd(), libc::F_GETFL) };
        if flags < 0 {
            return Err(io::Error::last_os_error());
        }
        if flags & libc::O_NONBLOCK == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "loopback fd is not O_NONBLOCK; VIDIOC_DQEVENT would block forever \
                 instead of reporting an empty queue",
            ));
        }

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
    /// would spin, because `poll` returns immediately on an errored fd.
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
            // `ENODEV` so the caller can tell "the node went away" from
            // any other I/O failure via `raw_os_error()`; the raw
            // `revents` stays in the message for diagnostics.
            let err = io::Error::from_raw_os_error(libc::ENODEV);
            tracing::debug!(
                revents = format_args!("{:#x}", pfd.revents),
                "loopback event fd is no longer usable"
            );
            return Err(err);
        }

        let mut latest = None;
        // Bounded drain (see `MAX_DRAIN`). The driver's `replace`/`merge` ops collapse the
        // queue to a single usage event, so one or two iterations is the
        // realistic case; the cap only exists so that a foreign producer
        // queueing events on the shared file description faster than we
        // dequeue cannot pin this call — and therefore the detector
        // thread's join in `Drop` — indefinitely. Whatever we have read
        // by then is returned; the rest is picked up by the next `poll`.
        for _ in 0..MAX_DRAIN {
            match self.dequeue()? {
                Dequeued::Usage(usage) => latest = Some(usage),
                // Keep draining: stopping on a foreign event would leave
                // the queue non-empty and `POLLPRI` permanently raised,
                // turning the caller's poll loop into a busy loop.
                Dequeued::Other => {}
                Dequeued::Empty => return Ok(latest),
            }
        }
        tracing::debug!(
            max_drain = MAX_DRAIN,
            "event queue still non-empty after the drain cap; resuming on the next poll"
        );
        Ok(latest)
    }

    /// Dequeue exactly one event from the driver's queue.
    ///
    /// Relies on the descriptor being `O_NONBLOCK` (enforced in
    /// [`subscribe`](Self::subscribe)): the kernel's
    /// `v4l2_event_dequeue()` only reports the empty queue as
    /// `EAGAIN`/`ENOENT` for non-blocking files, and otherwise sleeps in
    /// `wait_event_interruptible()` — which would turn the terminating
    /// drain call in [`poll`](Self::poll) into an unbounded block.
    fn dequeue(&self) -> io::Result<Dequeued> {
        // Zeroed rather than uninitialised: `VIDIOC_DQEVENT` is `_IOR`,
        // so the kernel fills the struct, but zeroing keeps the value
        // valid Rust even on the error path.
        // SAFETY: `v4l2_event` is a `repr(C)` POD whose union variants
        // are all integer/array types, so the all-zero bit pattern is a
        // valid value. Zeroed rather than uninitialised because
        // `VIDIOC_DQEVENT` is `_IOR` — the kernel fills the struct, but
        // zeroing keeps the value valid Rust on the error path too.
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
            // belongs to the file handle and is shared in principle —
            // log it, because draining it here means whoever did
            // subscribe will never see it.
            tracing::debug!(
                event_type = format_args!("{:#x}", ev.type_),
                event_id = ev.id,
                "discarding a foreign V4L2 event while draining the usage queue"
            );
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

/// Take a one-shot, authoritative reading of capture usage from `device`.
///
/// Opens the node, subscribes with `V4L2_EVENT_SUB_FL_SEND_INITIAL`,
/// dequeues the value the driver queues in response, and closes
/// everything again. Nothing is retained.
///
/// ## Why a fresh descriptor
///
/// A [`ConsumerWatch`] that is already subscribed cannot be asked for
/// the current value: the kernel's `v4l2_event_subscribe()` finds the
/// existing subscription for the `(type, id)` pair on that file handle,
/// returns success and — crucially — does **not** call the driver's
/// `add` op, which is what `SEND_INITIAL` is implemented by. Re-arming
/// in place is therefore a no-op; only a new file handle produces a
/// fresh initial value.
///
/// ## The probe is visible to other subscribers
///
/// v4l2loopback answers the subscription by queueing the event through
/// `v4l2_event_queue()`, which fans out to **every** subscribed file
/// handle on the node — including any long-lived [`ConsumerWatch`] the
/// caller holds. That echo is useful (it repairs a stale verdict on the
/// watch by itself) but it means a caller must not treat "an event
/// arrived shortly after probing" as evidence of consumer activity, and
/// must drain it before measuring how long the node has been quiet.
///
/// ## Cost and safety
///
/// One `open` + one `ioctl` + one `close`, plus one `max_openers` slot
/// for the duration of the call. No `S_FMT`, `REQBUFS` or `STREAMON`, so
/// the single capture slot v4l2loopback hands out is never claimed and a
/// live consumer cannot be displaced. The node is opened read-only:
/// `VIDIOC_SUBSCRIBE_EVENT` / `VIDIOC_DQEVENT` do not need write access.
///
/// `timeout_ms` must stay within the caller's shutdown budget — the same
/// contract as [`ConsumerWatch::poll`], and for the same reason. The
/// driver answers `SEND_INITIAL` synchronously, so a value in the tens
/// of milliseconds is generous. Note the `open(2)` itself is not covered
/// by it: the driver takes an interruptible mutex there, so a pathologically
/// contended node can block for longer than `timeout_ms`.
///
/// # Errors
///
/// * [`io::ErrorKind::TimedOut`] — the driver queued nothing within
///   `timeout_ms`. Deliberately an error rather than a `None`: "the
///   kernel did not answer" must never be mistaken for "nobody is
///   streaming", which would put the daemon to sleep under a live
///   client.
/// * `EBUSY` — no free opener slot on the node.
/// * `ENOTTY` / `EINVAL` — the driver does not implement the event
///   (not a v4l2loopback node, or older than 0.13). Permanent: retrying
///   cannot fix it.
/// * Anything else `open(2)` or the ioctls can raise.
pub fn probe_client_usage(device: &Path, timeout_ms: i32) -> io::Result<ClientUsage> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(device)?;
    let watch = ConsumerWatch::subscribe(file.as_fd())?;
    watch.poll(timeout_ms)?.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::TimedOut,
            "driver queued no client-usage event for a fresh SEND_INITIAL subscription",
        )
    })
}

/// How a caller should react to a [`probe_client_usage`] failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProbeFailure {
    /// Retrying cannot help: the node does not implement the event
    /// (`ENOTTY`/`EINVAL` — not a v4l2loopback node, or older than
    /// 0.13) or we are not allowed to open it (`EACCES`/`EPERM`).
    /// Callers should stop probing rather than log the same failure
    /// every interval.
    Permanent,
    /// `EBUSY`: the node is at `max_openers`. Transient, but the caller
    /// should back off — this is precisely when one more opener hurts,
    /// since a real consumer opening right now gets the same error.
    Busy,
    /// Anything else: the node vanished, a signal interrupted the wait,
    /// the driver stayed silent. Worth retrying on the next tick.
    Transient,
}

/// Classify an error from [`probe_client_usage`].
///
/// Lives here rather than at the call site because it encodes which
/// errnos this specific driver raises — the same knowledge the ioctl
/// wrappers above already own.
#[must_use]
pub fn classify_probe_error(e: &io::Error) -> ProbeFailure {
    match e.raw_os_error() {
        Some(libc::ENOTTY | libc::EINVAL | libc::EACCES | libc::EPERM) => ProbeFailure::Permanent,
        Some(libc::EBUSY) => ProbeFailure::Busy,
        _ => ProbeFailure::Transient,
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
    #[cfg(target_pointer_width = "64")]
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
