//! Widget builders for the main window.
//!
//! Components never depend on the root component's message type: each
//! one reports user intent through its own event type via an [`Emit`]
//! callback, and the root maps those events into its messages.

pub mod chain_page;
pub mod dialogs;
pub mod param_row;
pub mod preset_bar;
pub mod preview;
pub mod status_page;

use std::rc::Rc;

/// Event sink a component calls to report user intent upward. Cheap to
/// clone into signal handlers; lives on the GTK main thread.
pub(crate) type Emit<E> = Rc<dyn Fn(E)>;
