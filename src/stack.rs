use std::ptr;

pub fn remaining_stack() -> Option<usize> {
    let current_sp = &0 as *const i32 as usize;
    let stack_limit = get_stack_limit()?;
    if current_sp > stack_limit {
        Some(current_sp - stack_limit)
    } else {
        None
    }
}

#[cfg(windows)]
fn get_stack_limit() -> Option<usize> {
    use std::mem;
	unsafe extern "system" {
		fn VirtualQuery(
			lpAddress: *const std::ffi::c_void,
			lpBuffer: *mut MEMORY_BASIC_INFORMATION,
			dwLength: usize,
		) -> usize;
	}

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct MEMORY_BASIC_INFORMATION {
        base_address: *mut std::ffi::c_void,
        allocation_base: *mut std::ffi::c_void,
        allocation_protect: u32,
        #[cfg(target_pointer_width = "64")]
        partition_id: u16,
        region_size: usize,
        state: u32,
        protect: u32,
        type_: u32,
    }

    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { mem::zeroed() };
    let local_var = 0;
    
    let result = unsafe {
        VirtualQuery(
            &local_var as *const i32 as *const std::ffi::c_void,
            &mut mbi,
            mem::size_of::<MEMORY_BASIC_INFORMATION>(),
        )
    };

    if result != 0 {
        // allocation_base points to the lowest valid address of the stack region
        Some(mbi.allocation_base as usize)
    } else {
        None
    }
}

// --- Linux Implementation ---
#[cfg(target_os = "linux")]
fn get_stack_limit() -> Option<usize> {
    unsafe {
        let self_thread = libc::pthread_self();
        let mut attr: libc::pthread_attr_t = std::mem::zeroed();
        
        if libc::pthread_getattr_np(self_thread, &mut attr) == 0 {
            let mut stack_addr: *mut std::ffi::c_void = ptr::null_mut();
            let mut stack_size: usize = 0;
            
            if libc::pthread_attr_getstack(&attr, &mut stack_addr, &mut stack_size) == 0 {
                libc::pthread_attr_destroy(&mut attr);
                return Some(stack_addr as usize);
            }
            libc::pthread_attr_destroy(&mut attr);
        }
    }
    None
}

// --- macOS Implementation ---
#[cfg(target_os = "macos")]
fn get_stack_limit() -> Option<usize> {
    unsafe {
        let self_thread = libc::pthread_self();
        let stack_top = libc::pthread_get_stackaddr_np(self_thread) as usize;
        let stack_size = libc::pthread_get_stacksize_np(self_thread);
        
        if stack_top > stack_size {
            Some(stack_top - stack_size)
        } else {
            None
        }
    }
}

// --- Fallback for unsupported OS ---
#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
fn get_stack_limit() -> Option<usize> {
    None
}

