//! Windows on ARM64 is sufficiently different that it gets its own file. The
//! code here is heavily based off the generic aarch64 version (for the register
//! state switching) and the x86_64 Windows version (for the Thread Environment
//! Block management), so you should read those and fully understand them first.
//!
//! On aarch64, Windows differs from other platforms in a few ways:
//! - Information about the current stack is stored in the Thread Environment
//!   Block (TEB). On ARM64 the TEB is reached through the platform register
//!   `x18` instead of the `gs` segment used on x86_64. These TEB fields must be
//!   swapped when switching stacks. Some of these fields are mutable since
//!   Windows grows stacks dynamically. `x18` is reserved by the Windows ABI so
//!   we can safely use it to address the TEB without saving/restoring it.
//! - Windows uses Structured Exception Handling for unwinding instead of DWARF,
//!   which has a different (and more restrictive) set of unwind opcodes. The
//!   SEH directives used in the trampolines mirror those already used by the
//!   generic aarch64 code on UEFI, extended to account for the extra TEB fields
//!   saved in each stack frame.
//!
//! ## Stack layout
//!
//! Here is what the layout of the stack looks like when a coroutine is
//! suspended.
//!
//! ```text
//! +--------------------------+  <- Stack base
//! | TEB.StackLimit           |  <- These are set by switch_and_reset from the
//! +--------------------------+   | values in the TEB so that they can be
//! | TEB.GuaranteedStackBytes |  <- updated in the `Stack` if it is later
//! +--------------------------+     reused for another coroutine.
//! | Initial func             |
//! +--------------------------+
//! | Parent link              |  <- PARENT_LINK_OFFSET (32) below the base
//! +--------------------------+
//! |                          |
//! ~     ...                  ~
//! |                          |
//! +--------------------------+
//! | TEB.GuaranteedStackBytes |  <-
//! +--------------------------+   | These are fields in the Thread Environment
//! | TEB.DeallocationStack    |  <- Block (TEB) which need to be saved and
//! +--------------------------+   | restored when switching stacks.
//! | TEB.StackLimit           |  <-
//! +--------------------------+   |
//! | TEB.StackBase            |  <-
//! +--------------------------+
//! | TEB.ExceptionList        |  <- Legacy SEH chain, unused on Win64 but used
//! +--------------------------+     by Wine.
//! | Saved PC                 |
//! +--------------------------+
//! | Saved X29                |
//! +--------------------------+
//! | Saved X19                |
//! +--------------------------+
//! ```
//!
//! And this is the layout of the parent stack when a coroutine is running:
//!
//! ```text
//! |                          |
//! ~     ...                  ~
//! |                          |
//! +--------------------------+
//! | TEB.GuaranteedStackBytes |
//! +--------------------------+
//! | TEB.DeallocationStack    |
//! +--------------------------+
//! | TEB.StackLimit           |
//! +--------------------------+
//! | TEB.StackBase            |
//! +--------------------------+
//! | TEB.ExceptionList        |
//! +--------------------------+
//! | Saved X19                |
//! +--------------------------+
//! | Saved PC (LR)            |
//! +--------------------------+
//! | Saved X29                |
//! +--------------------------+
//! ```
//!
//! And finally, this is the stack layout of a coroutine that has just been
//! initialized:
//!
//! ```text
//! +--------------------------+  <- Stack base
//! | TEB.StackLimit           |  <- Uninitialized at this point.
//! +--------------------------+
//! | TEB.GuaranteedStackBytes |  <- Uninitialized at this point.
//! +--------------------------+
//! | Initial func             |
//! +--------------------------+
//! | Parent link              |  <- Uninitialized at this point.
//! +--------------------------+
//! |                          |
//! ~ Initial obj              ~  <- Rounded up to `STACK_ALIGNMENT`.
//! |                          |
//! +--------------------------+
//! | TEB.GuaranteedStackBytes |  <-
//! +--------------------------+   |
//! | TEB.DeallocationStack    |  <- Initial values for the TEB fields on the
//! +--------------------------+   | new stack.
//! | TEB.StackLimit           |  <-
//! +--------------------------+   |
//! | TEB.StackBase            |  <-
//! +--------------------------+
//! | Padding (ExceptionList)  |
//! +--------------------------+
//! | Initial PC               |  <- Points to stack_init_trampoline
//! +--------------------------+
//! | Padding (X29)            |
//! +--------------------------+
//! | Padding (X19)            |  <- Initial stack pointer
//! +--------------------------+
//! ```

use core::arch::{asm, global_asm};

use super::{allocate_obj_on_stack, push};
use crate::coroutine::adjusted_stack_base;
use crate::stack::{Stack, StackPointer, StackTebFields};
use crate::unwind::{
    asm_may_unwind_root, asm_may_unwind_yield, InitialFunc, StackCallFunc, TrapHandler,
};
use crate::util::EncodedValue;

pub const STACK_ALIGNMENT: usize = 16;
pub const PARENT_STACK_OFFSET: usize = 0;
pub const PARENT_LINK_OFFSET: usize = 32;
pub type StackWord = u64;

// Offsets of the relevant fields in the TEB, which is addressed through the
// `x18` platform register on Windows ARM64. The 64-bit TEB layout is identical
// across architectures so these match the offsets used by x86_64.
//
// - 0x0000: NtTib.ExceptionList
// - 0x0008: NtTib.StackBase
// - 0x0010: NtTib.StackLimit
// - 0x1478: DeallocationStack
// - 0x1748: GuaranteedStackBytes

global_asm!(
    ".balign 4",
    asm_function_begin!("stack_init_trampoline"),
    ".seh_proc stack_init_trampoline",
    // At this point our register state contains the following:
    // - SP points to the top of the parent stack.
    // - LR contains the return address in the parent context.
    // - X19 and X29 contain their value from the parent context.
    // - X18 points to the (parent) TEB.
    // - X2 points to the top of the coroutine stack.
    // - X1 points to the base of our stack.
    // - X0 contains the argument passed from switch_and_link.
    //
    // Prologue: reserve the parent frame and save the parent context's X29, LR
    // and X19 into it. The remaining slots hold the TEB fields and are filled
    // in below. Windows requires the SEH directives to describe exactly the
    // prologue instructions, so only the register saves live here; everything
    // else happens after `.seh_endprologue`.
    //
    // Because the stack pointer is switched to the coroutine stack in the body,
    // the SEH unwind information for this trampoline only describes its own
    // frame and cannot follow the parent link across the stack switch. This is
    // only used for computing backtraces (panics are caught at the coroutine
    // root and never unwind through here), so an incomplete backtrace across
    // the coroutine boundary is the only consequence.
    "stp x29, x30, [sp, #-64]!",
    ".seh_save_fplr_x 64",
    "str x19, [sp, #16]",
    ".seh_save_reg x19, 16",
    ".seh_endprologue",
    // Read the parent TEB fields and save them into the parent frame.
    "ldr x4, [x18, #0x00]",   // ExceptionList
    "ldr x5, [x18, #0x08]",   // StackBase
    "ldr x6, [x18, #0x10]",   // StackLimit
    "ldr x7, [x18, #0x1478]", // DeallocationStack
    "ldr x8, [x18, #0x1748]", // GuaranteedStackBytes
    "str x4, [sp, #24]",
    "stp x5, x6, [sp, #32]",
    "stp x7, x8, [sp, #48]",
    // Write the parent stack pointer to the parent link and adjust X1 to point
    // to the parent link.
    "mov x3, sp",
    "str x3, [x1, #-32]!",
    // Load the TEB fields for our new stack. ExceptionList is reset to the
    // end-of-chain value (!0) which is what Win64 expects (and Wine uses).
    "movn x4, #0",
    "str x4, [x18, #0x00]", // ExceptionList
    "ldr x5, [x2, #32]",
    "str x5, [x18, #0x08]", // StackBase
    "ldr x6, [x2, #40]",
    "str x6, [x18, #0x10]", // StackLimit
    "ldr x7, [x2, #48]",
    "str x7, [x18, #0x1478]", // DeallocationStack
    "ldr x8, [x2, #56]",
    "str x8, [x18, #0x1748]", // GuaranteedStackBytes
    // Switch to the coroutine stack, popping the saved registers and TEB fields.
    "add sp, x2, #64",
    // Set up the 3rd argument to the initial function to point to the object
    // that init_stack() set up on the stack.
    "mov x2, sp",
    // As in the original aarch64 code, hand-write the call operation so that it
    // doesn't push an entry into the CPU's return prediction stack.
    "adr x30, 0f",
    "ldr x3, [x1, #8]",
    "br x3",
    // Use a local label because stack_init_trampoline_return is a global
    // symbol, which can cause issues with relocations.
    "0:",
    asm_function_alt_entry!("stack_init_trampoline_return"),
    // The SEH unwinder scans forward from a return target looking for an epilog
    // code sequence. The BRK here stops that scan from running off the end of
    // the function since it is not part of any valid epilog.
    "brk #0",
    ".seh_endproc",
    asm_function_end!("stack_init_trampoline"),
);

global_asm!(
    // See stack_init_trampoline for an explanation of the assembler directives
    // used here.
    ".balign 4",
    asm_function_begin!("stack_call_trampoline"),
    ".seh_proc stack_call_trampoline",
    // At this point our register state contains the following:
    // - SP points to the top of the parent stack.
    // - X29 holds its value from the parent context.
    // - X18 points to the (parent) TEB.
    // - X2 is the function that should be called.
    // - X1 points to the top of our stack.
    // - X0 contains the argument to be passed to the function.
    // - X3-X6 contain the TEB fields for the new stack.
    //
    // Prologue: reserve the frame and save X29/LR. We deliberately do NOT emit
    // a `.seh_set_fp` (frame pointer) directive even though we use X29 as a
    // frame pointer at runtime below. The SEH unwind info is therefore frameless
    // and dead-ends on this (switched) stack rather than following the frame
    // pointer back to the original stack.
    //
    // This matters for exception dispatch: a panic raised on the switched stack
    // is caught by `catch_unwind_at_root` in the wrapper running on this stack.
    // When the OS unwinder evaluates that catch it also processes this frame
    // (the wrapper's caller). If our unwind info crossed back to the original
    // stack, the resulting stack pointer would be outside the switched stack's
    // TEB StackBase/StackLimit bounds (the unwinder does not update the TEB when
    // switching stacks) and the dispatch would fail, leaving the exception
    // unhandled. By dead-ending here, the catch on the switched stack succeeds.
    // The cost is that linked backtraces do not cross this boundary.
    "stp x29, x30, [sp, #-64]!",
    ".seh_save_fplr_x 64",
    ".seh_endprologue",
    // Set up the frame pointer for the runtime stack switch below (not described
    // by the SEH unwind info, on purpose; see above).
    "mov x29, sp",
    // Save the parent context's TEB fields into the frame.
    "ldr x7, [x18, #0x00]",
    "str x7, [sp, #16]",     // ExceptionList
    "ldr x9, [x18, #0x08]",  // StackBase
    "ldr x10, [x18, #0x10]", // StackLimit
    "stp x9, x10, [sp, #24]",
    "ldr x11, [x18, #0x1478]", // DeallocationStack
    "ldr x12, [x18, #0x1748]", // GuaranteedStackBytes
    "stp x11, x12, [sp, #40]",
    // Save the new stack base (X1, == adjusted_stack_base) in the spare frame
    // slot so the mutable TEB fields can be written back to it after the call.
    // It cannot be recomputed from TEB.StackBase: for `on_parent_stack` those
    // differ (TEB.StackBase is the real thread base, X1 is a point on it).
    "str x1, [sp, #56]",
    // Switch to the new stack.
    "mov sp, x1",
    // Reserve headroom below StackBase before running the function. The called
    // function therefore runs strictly below StackBase, leaving room above it.
    //
    // This is required for exception dispatch. A panic raised on this stack is
    // caught by catch_unwind_at_root in the wrapper running here. While the OS
    // unwinder processes that catch it also unwinds this frame (the wrapper's
    // caller); the frameless SEH info adds the 64-byte frame size to SP. Without
    // headroom the wrapper would run at the very top (SP == StackBase) and that
    // unwound SP would land above StackBase, outside the TEB stack bounds, which
    // makes the dispatch fail and the panic escape unhandled. With headroom the
    // unwound SP stays in bounds. The coroutine path gets this for free because
    // init_stack reserves a region at the top of the stack.
    "sub sp, sp, #128",
    // Set the TEB fields for the new stack, resetting ExceptionList to !0.
    "movn x7, #0",
    "str x7, [x18, #0x00]",   // ExceptionList
    "str x3, [x18, #0x08]",   // StackBase
    "str x4, [x18, #0x10]",   // StackLimit
    "str x5, [x18, #0x1478]", // DeallocationStack
    "str x6, [x18, #0x1748]", // GuaranteedStackBytes
    // Call the function pointer. The argument is already in X0.
    "blr x2",
    // Save the two mutable TEB fields just below the new stack base (the value
    // passed in X1, saved in the frame above) so they can be retrieved later by
    // update_stack_teb_fields(). SP is in the reserved-headroom region, not at
    // the base, so reload the base rather than using SP.
    "ldr x9, [x29, #56]",   // adjusted_stack_base
    "ldr x4, [x18, #0x10]", // StackLimit
    "str x4, [x9, #-8]",
    "ldr x4, [x18, #0x1748]", // GuaranteedStackBytes
    "str x4, [x9, #-16]",
    // Switch back to the original stack using the frame pointer.
    "mov sp, x29",
    // Restore the parent TEB fields and registers, then return.
    "ldr x7, [sp, #16]",
    "str x7, [x18, #0x00]", // ExceptionList
    "ldp x9, x10, [sp, #24]",
    "str x9, [x18, #0x08]",  // StackBase
    "str x10, [x18, #0x10]", // StackLimit
    "ldp x11, x12, [sp, #40]",
    "str x11, [x18, #0x1478]", // DeallocationStack
    "str x12, [x18, #0x1748]", // GuaranteedStackBytes
    "ldp x29, x30, [sp], #64",
    "ret",
    ".seh_endproc",
    asm_function_end!("stack_call_trampoline"),
);

// These trampolines use a custom calling convention and should only be called
// with inline assembly.
extern "C" {
    fn stack_init_trampoline(arg: EncodedValue, stack_base: StackPointer, stack_ptr: StackPointer);
    static stack_init_trampoline_return: [u8; 0];
    #[allow(dead_code)]
    fn stack_call_trampoline(arg: *mut u8, sp: StackPointer, f: StackCallFunc);
}

#[inline]
pub unsafe fn init_stack<T>(stack: &impl Stack, func: InitialFunc<T>, obj: T) -> StackPointer {
    let mut sp = adjusted_stack_base(stack).get();

    // Placeholders for returning TEB.StackLimit and TEB.GuaranteedStackBytes.
    push(&mut sp, None);
    push(&mut sp, None);

    // Initial function.
    push(&mut sp, Some(func as StackWord));

    // Placeholder for parent link.
    push(&mut sp, None);

    // Allocate space on the stack for the initial object, rounding to
    // STACK_ALIGNMENT.
    allocate_obj_on_stack(&mut sp, 32, obj);

    // Write the TEB fields for the target stack.
    let teb = stack.teb_fields();
    push(&mut sp, Some(teb.GuaranteedStackBytes as StackWord));
    push(&mut sp, Some(teb.DeallocationStack as StackWord));
    push(&mut sp, Some(teb.StackLimit as StackWord));
    push(&mut sp, Some(teb.StackBase as StackWord));

    // Padding for the ExceptionList slot in the suspended frame.
    push(&mut sp, None);

    // Entry point called by switch_and_link(). switch_and_link() looks for the
    // target PC 16 bytes above the stack pointer.
    push(
        &mut sp,
        Some(stack_init_trampoline as *const () as StackWord),
    );

    // Padding for the saved X29 and X19 slots.
    push(&mut sp, None);
    push(&mut sp, None);

    // The stack is aligned to STACK_ALIGNMENT at this point.
    debug_assert_eq!(sp % STACK_ALIGNMENT, 0);

    StackPointer::new_unchecked(sp)
}

#[inline]
pub unsafe fn switch_and_link(
    arg: EncodedValue,
    sp: StackPointer,
    stack_base: StackPointer,
) -> (EncodedValue, Option<StackPointer>) {
    let (ret_val, ret_sp);

    asm_may_unwind_root!(
        // Read the saved PC from the coroutine stack and call it.
        "ldr x3, [x2, #16]",
        "blr x3",

        // Upon returning, our register state contains the following:
        // - X2: The parent stack pointer.
        // - X1: The top of the coroutine stack, or 0 if coming from
        //       switch_and_reset.
        // - X0: The argument passed from the coroutine.

        // Restore the parent TEB fields from the parent frame.
        "ldr x4, [x2, #24]", "str x4, [x18, #0x00]",   // ExceptionList
        "ldr x5, [x2, #32]", "str x5, [x18, #0x08]",   // StackBase
        "ldr x6, [x2, #40]", "str x6, [x18, #0x10]",   // StackLimit
        "ldr x7, [x2, #48]", "str x7, [x18, #0x1478]", // DeallocationStack
        "ldr x8, [x2, #56]", "str x8, [x18, #0x1748]", // GuaranteedStackBytes

        // Switch back to our stack and free the parent frame.
        "add sp, x2, #64",

        // Pass the argument in X0.
        inlateout("x0") arg => ret_val,

        // We get the coroutine stack pointer back in X1.
        lateout("x1") ret_sp,

        // We pass the stack base in X1.
        in("x1") stack_base.get() as u64,

        // We pass the target stack pointer in X2.
        in("x2") sp.get() as u64,

        // Mark all registers as clobbered. The clobber_abi() will automatically
        // mark X18 as clobbered if it is not reserved by the platform; on
        // Windows it is reserved so it is preserved.
        lateout("x20") _, lateout("x21") _, lateout("x22") _, lateout("x23") _,
        lateout("x24") _, lateout("x25") _, lateout("x26") _, lateout("x27") _,
        lateout("x28") _,
        clobber_abi("C"),
    );

    (ret_val, StackPointer::new(ret_sp))
}

#[inline(always)]
pub unsafe fn switch_yield(arg: EncodedValue, parent_link: *mut StackPointer) -> EncodedValue {
    let ret_val;

    asm_may_unwind_yield!(
        // Read the coroutine's TEB fields so they can be saved on its stack.
        "ldr x3, [x18, #0x00]",   // ExceptionList
        "ldr x4, [x18, #0x08]",   // StackBase
        "ldr x5, [x18, #0x10]",   // StackLimit
        "ldr x6, [x18, #0x1478]", // DeallocationStack
        "ldr x7, [x18, #0x1748]", // GuaranteedStackBytes

        // Save X19, X29 and a slot for our PC, followed by the TEB fields.
        "stp x19, x29, [sp, #-64]!",
        "adr x30, 0f",
        "str x30, [sp, #16]",
        "str x3, [sp, #24]",
        "stp x4, x5, [sp, #32]",
        "stp x6, x7, [sp, #48]",

        // Get the parent stack pointer from the parent link.
        "ldr x2, [x2]",

        // Save our stack pointer to X1.
        "mov x1, sp",

        // Restore X19, X29 and LR from the parent frame.
        "ldr x19, [x2, #16]",
        "ldp x29, x30, [x2]",

        // Return into the parent context. The parent TEB fields are restored by
        // switch_and_link() after this returns.
        "ret",

        // This gets called by switch_and_link(). At this point our register
        // state contains the following:
        // - SP points to the top of the parent stack.
        // - LR contains the return address in the parent context.
        // - X19 and X29 contain their value from the parent context.
        // - X18 points to the (parent) TEB.
        // - X2 points to the top of the coroutine stack.
        // - X1 points to the base of our stack.
        // - X0 contains the argument passed from switch_and_link.
        "0:",

        // Read the parent TEB fields so they can be saved on the parent stack.
        "ldr x3, [x18, #0x00]",   // ExceptionList
        "ldr x4, [x18, #0x08]",   // StackBase
        "ldr x5, [x18, #0x10]",   // StackLimit
        "ldr x6, [x18, #0x1478]", // DeallocationStack
        "ldr x7, [x18, #0x1748]", // GuaranteedStackBytes

        // Push the parent context's X29, LR, X19 and TEB fields onto the parent
        // stack to form the parent frame.
        "stp x29, x30, [sp, #-64]!",
        "str x19, [sp, #16]",
        "str x3, [sp, #24]",
        "stp x4, x5, [sp, #32]",
        "stp x6, x7, [sp, #48]",

        // Write the parent stack pointer to the parent link.
        "mov x3, sp",
        "str x3, [x1, #-32]",

        // Load our X19 and X29 values from the coroutine stack.
        "ldp x19, x29, [x2]",

        // Load the coroutine's TEB fields, resetting ExceptionList to !0.
        "movn x3, #0",
        "str x3, [x18, #0x00]",                       // ExceptionList
        "ldr x4, [x2, #32]", "str x4, [x18, #0x08]",   // StackBase
        "ldr x5, [x2, #40]", "str x5, [x18, #0x10]",   // StackLimit
        "ldr x6, [x2, #48]", "str x6, [x18, #0x1478]", // DeallocationStack
        "ldr x7, [x2, #56]", "str x7, [x18, #0x1748]", // GuaranteedStackBytes

        // Switch to the coroutine stack while popping the saved registers and
        // TEB fields.
        "add sp, x2, #64",

        // Pass the argument in X0.
        inlateout("x0") arg => ret_val,

        // The parent link can be in any register, X2 is arbitrarily chosen
        // here.
        in("x2") parent_link as u64,

        // See switch_and_link() for an explanation of the clobbers.
        lateout("x20") _, lateout("x21") _, lateout("x22") _, lateout("x23") _,
        lateout("x24") _, lateout("x25") _, lateout("x26") _, lateout("x27") _,
        lateout("x28") _,
        clobber_abi("C"),
    );

    ret_val
}

#[inline(always)]
pub unsafe fn switch_and_reset(arg: EncodedValue, parent_link: *mut StackPointer) -> ! {
    // Most of this code is identical to switch_yield(), refer to the comments
    // there. Only the differences are commented.
    asm!(
        // Save the 2 mutable TEB fields to the base of the stack so they can be
        // recovered by update_stack_teb_fields() if the stack is reused.
        "ldr x4, [x18, #0x10]",   // StackLimit
        "str x4, [{parent_link}, #24]",
        "ldr x4, [x18, #0x1748]", // GuaranteedStackBytes
        "str x4, [{parent_link}, #16]",

        // Load the parent context's stack pointer.
        "ldr x2, [{parent_link}]",

        // Restore X19, X29 and LR from the parent frame. The parent TEB fields
        // are restored by switch_and_link() once it regains control.
        "ldr x19, [x2, #16]",
        "ldp x29, x30, [x2]",

        // Return into the parent context.
        "ret",

        parent_link = in(reg) parent_link as u64,

        in("x0") arg,

        // Hard-code the returned stack pointer value to 0 to indicate that this
        // coroutine is done.
        in("x1") 0,

        options(noreturn),
    );
}

#[inline]
#[cfg(feature = "asm-unwind")]
pub unsafe fn switch_and_throw(
    forced_unwind: crate::unwind::ForcedUnwind,
    sp: StackPointer,
    stack_base: StackPointer,
) -> (EncodedValue, Option<StackPointer>) {
    extern "C-unwind" fn throw(forced_unwind: crate::unwind::ForcedUnwind) -> ! {
        extern crate std;
        use std::boxed::Box;
        std::panic::resume_unwind(Box::new(forced_unwind));
    }

    let (ret_val, ret_sp);

    asm_may_unwind_root!(
        // Set up a return address.
        "adr x30, 0f",

        // Read the parent TEB fields so they can be saved on the parent stack.
        "ldr x4, [x18, #0x00]",   // ExceptionList
        "ldr x5, [x18, #0x08]",   // StackBase
        "ldr x6, [x18, #0x10]",   // StackLimit
        "ldr x7, [x18, #0x1478]", // DeallocationStack
        "ldr x8, [x18, #0x1748]", // GuaranteedStackBytes

        // Save the registers and TEB fields of the parent context.
        "stp x29, x30, [sp, #-64]!",
        "str x19, [sp, #16]",
        "str x4, [sp, #24]",
        "stp x5, x6, [sp, #32]",
        "stp x7, x8, [sp, #48]",

        // Update the parent link near the base of the coroutine stack.
        "mov x3, sp",
        "str x3, [x1, #-32]",

        // Load the coroutine registers, with the saved PC into LR.
        "ldr x30, [x2, #16]",
        "ldp x19, x29, [x2]",

        // Load the coroutine's TEB fields, resetting ExceptionList to !0.
        "movn x4, #0",
        "str x4, [x18, #0x00]",                       // ExceptionList
        "ldr x5, [x2, #32]", "str x5, [x18, #0x08]",   // StackBase
        "ldr x6, [x2, #40]", "str x6, [x18, #0x10]",   // StackLimit
        "ldr x7, [x2, #48]", "str x7, [x18, #0x1478]", // DeallocationStack
        "ldr x8, [x2, #56]", "str x8, [x18, #0x1748]", // GuaranteedStackBytes

        // Switch to the coroutine stack while popping the saved registers and
        // TEB fields.
        "add sp, x2, #64",

        // Simulate a call with an artificial return address so that the throw
        // function will unwind straight into the switch_yield() call with the
        // register state expected outside the asm! block.
        "b {throw}",

        // Upon returning, our register state is just like a normal return into
        // switch_and_link().
        "0:",

        // Restore the parent TEB fields from the parent frame.
        "ldr x4, [x2, #24]", "str x4, [x18, #0x00]",   // ExceptionList
        "ldr x5, [x2, #32]", "str x5, [x18, #0x08]",   // StackBase
        "ldr x6, [x2, #40]", "str x6, [x18, #0x10]",   // StackLimit
        "ldr x7, [x2, #48]", "str x7, [x18, #0x1478]", // DeallocationStack
        "ldr x8, [x2, #56]", "str x8, [x18, #0x1748]", // GuaranteedStackBytes

        // Switch back to our stack and free the parent frame.
        "add sp, x2, #64",

        // Helper function to trigger stack unwinding.
        throw = sym throw,

        // Argument to pass to the throw function.
        in("x0") forced_unwind.0.get(),

        // Same output registers as switch_and_link().
        lateout("x0") ret_val,
        lateout("x1") ret_sp,

        // Stack pointer and stack base inputs for stack switching.
        in("x1") stack_base.get() as u64,
        in("x2") sp.get() as u64,

        // See switch_and_link() for an explanation of the clobbers.
        lateout("x20") _, lateout("x21") _, lateout("x22") _, lateout("x23") _,
        lateout("x24") _, lateout("x25") _, lateout("x26") _, lateout("x27") _,
        lateout("x28") _,
        clobber_abi("C"),
    );

    (ret_val, StackPointer::new(ret_sp))
}

#[inline]
pub unsafe fn drop_initial_obj(
    stack_base: StackPointer,
    stack_ptr: StackPointer,
    drop_fn: unsafe fn(ptr: *mut u8),
) {
    let ptr = (stack_ptr.get() as *mut u8).add(64);
    drop_fn(ptr);

    // Also copy the TEB fields to the base of the stack so that they can be
    // retrieved by update_stack_teb_fields().
    let base = stack_base.get() as *mut StackWord;
    let stack = stack_ptr.get() as *const StackWord;
    *base.sub(1) = *stack.add(5); // StackLimit
    *base.sub(2) = *stack.add(7); // GuaranteedStackBytes
}

/// This function is called by `force_reset` to update the mutable TEB fields at
/// the bottom of the parent stack with the ones from the suspended state.
///
/// The coroutine must be in a suspended state and *not* in the initial state.
#[inline]
pub unsafe fn reset_teb_fields_from_suspended(stack_base: StackPointer, stack_ptr: StackPointer) {
    let base = stack_base.get() as *mut StackWord;
    let stack = stack_ptr.get() as *const StackWord;
    *base.sub(1) = *stack.add(5); // StackLimit
    *base.sub(2) = *stack.add(7); // GuaranteedStackBytes
}

/// This function must be called after a stack has finished running a coroutine
/// so that the `StackLimit` and `GuaranteedStackBytes` fields from the TEB can
/// be updated in the stack. This is necessary if the stack is reused for
/// another coroutine.
#[inline]
pub unsafe fn update_stack_teb_fields(stack: &mut impl Stack) {
    let base = adjusted_stack_base(stack).get() as *const StackWord;
    let stack_limit = *base.sub(1) as usize;
    let guaranteed_stack_bytes = *base.sub(2) as usize;
    stack.update_teb_fields(stack_limit, guaranteed_stack_bytes);
}

/// This function is called by `on_parent_stack` to read the saved TEB fields
/// saved on the parent stack.
#[inline]
pub unsafe fn read_parent_stack_teb_fields(stack_ptr: StackPointer) -> StackTebFields {
    let stack_ptr = stack_ptr.get() as *const StackWord;
    StackTebFields {
        StackBase: *stack_ptr.add(4) as usize,
        StackLimit: *stack_ptr.add(5) as usize,
        DeallocationStack: *stack_ptr.add(6) as usize,
        GuaranteedStackBytes: *stack_ptr.add(7) as usize,
    }
}

/// This function is called by `on_parent_stack` to update the saved TEB fields
/// saved on the parent stack at the end of execution.
#[inline]
pub unsafe fn update_parent_stack_teb_fields(
    stack_ptr: StackPointer,
    stack_limit: usize,
    guaranteed_stack_bytes: usize,
) {
    let stack_ptr = stack_ptr.get() as *mut StackWord;
    *stack_ptr.add(5) = stack_limit as StackWord;
    *stack_ptr.add(7) = guaranteed_stack_bytes as StackWord;
}

/// Registers which must be updated upon return from a trap handler.
///
/// The exact set of registers that need to be updated varies depending on the
/// target. Note that *all* registers must be updated to the specified values,
/// otherwise behavior is undefined.
///
/// To catch any issues at compilation time, it is recommended to use Rust's
/// pattern matching syntax to extract the individual registers from this
/// struct.
///
/// ```
/// # use corosensei::trap::TrapHandlerRegs;
/// # let regs = TrapHandlerRegs { pc: 0, sp: 0, x0: 0, x1: 0, x29: 0, lr: 0 };
/// let TrapHandlerRegs { pc, sp, x0, x1, x29, lr } = regs;
/// ```
#[allow(missing_docs)]
#[derive(Clone, Copy, Debug)]
pub struct TrapHandlerRegs {
    pub pc: u64,
    pub sp: u64,
    pub x0: u64,
    pub x1: u64,
    pub x29: u64,
    pub lr: u64,
}

pub unsafe fn setup_trap_trampoline<T>(
    stack_base: StackPointer,
    val: T,
    handler: TrapHandler<T>,
) -> TrapHandlerRegs {
    // Preserve the top 32 bytes of the stack since they contain the parent
    // link. The top 16 bytes are filled by switch_and_reset() with TEB fields.
    let parent_link = stack_base.get() - PARENT_LINK_OFFSET;

    // Everything below this can be overwritten. Write the object to the stack.
    let mut sp = parent_link;
    allocate_obj_on_stack(&mut sp, 32, val);
    let val_ptr = sp;

    // Set up registers for entry into the function.
    TrapHandlerRegs {
        pc: handler as u64,
        sp: sp as u64,
        x0: val_ptr as u64,
        x1: parent_link as u64,
        x29: parent_link as u64,
        lr: stack_init_trampoline_return.as_ptr() as u64,
    }
}

/// This function executes a function on the given stack. The argument is passed
/// through to the called function.
#[inline]
pub unsafe fn on_stack<S: Stack>(arg: *mut u8, stack: S, f: StackCallFunc) {
    let stack = scopeguard::guard(stack, |mut stack| {
        let base = adjusted_stack_base(&stack).get() as *const u64;
        let stack_limit = *base.sub(1) as usize;
        let guaranteed_stack_bytes = *base.sub(2) as usize;
        stack.update_teb_fields(stack_limit, guaranteed_stack_bytes);
    });

    let teb_fields = stack.teb_fields();

    asm_may_unwind_root!(
        concat!("bl ", asm_mangle!("stack_call_trampoline")),
        in("x0") arg,
        in("x1") adjusted_stack_base(&*stack).get(),
        in("x2") f,
        in("x3") teb_fields.StackBase,
        in("x4") teb_fields.StackLimit,
        in("x5") teb_fields.DeallocationStack,
        in("x6") teb_fields.GuaranteedStackBytes,
        clobber_abi("C"),
    );
}
