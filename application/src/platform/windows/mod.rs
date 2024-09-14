use core::iter::once;
use std::{ffi::OsStr, fmt, marker::Send, mem, num::NonZeroIsize, os::windows::ffi::OsStrExt, ptr};

use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle,
    RawWindowHandle, Win32WindowHandle, WindowHandle, WindowsDisplayHandle,
};

use windows_sys::Win32::{
    Foundation::*, Graphics::Gdi::*, System::SystemServices::IMAGE_DOS_HEADER,
    UI::WindowsAndMessaging::*,
};

#[derive(Clone, Copy)]
pub(crate) struct Window {
    handle: HWND,
}

struct WindowData {}

struct InitData {
    window: Option<Window>,
}

impl InitData {
    pub fn handle_create(&mut self, window: HWND) -> Option<isize> {
        let window = Window { handle: window };
        let window_data = WindowData {};

        let userdata = Box::into_raw(Box::new(window_data));
        self.window = Some(window);

        Some(userdata as _)
    }
}

pub(crate) struct WindowOptions {
    pub name: String,
}

impl Default for WindowOptions {
    fn default() -> Self {
        Self {
            name: String::from("My Application"),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum WindowError {
    CreationError(&'static str),
}

impl fmt::Display for WindowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowError::CreationError(err) => write!(f, "Window creation error: {}", err),
        }
    }
}

impl Window {
    pub fn create(options: WindowOptions) -> Result<Self, WindowError> {
        create_window(options)
    }

    pub fn run(&self) -> Result<(), WindowError> {
        unsafe {
            let mut message = mem::zeroed();

            loop {
                let status = GetMessageW(&mut message, self.handle, 0, 0);
                print!("");

                if status == 0 {
                    return Ok(());
                }

                if status == -1 {
                    return Err(WindowError::CreationError("Window error"));
                }

                TranslateMessage(&message);
                DispatchMessageW(&message);
            }
        }
    }
}

unsafe impl Sync for Window {}
unsafe impl Send for Window {}

impl HasWindowHandle for Window {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        let mut window_handle =
            Win32WindowHandle::new(unsafe { NonZeroIsize::new_unchecked(self.handle as isize) });

        let hinstance = unsafe { GetWindowLongPtrW(self.handle, GWLP_HINSTANCE) };
        window_handle.hinstance = NonZeroIsize::new(hinstance);

        let raw_window_handle = RawWindowHandle::Win32(window_handle);
        let window_handle = unsafe { WindowHandle::borrow_raw(raw_window_handle) };

        Ok(window_handle)
    }
}

impl HasDisplayHandle for Window {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        let raw_handle = RawDisplayHandle::Windows(WindowsDisplayHandle::new());
        let display_handle = unsafe { DisplayHandle::borrow_raw(raw_handle) };

        Ok(display_handle)
    }
}

fn encode_wide(string: impl AsRef<OsStr>) -> Vec<u16> {
    string.as_ref().encode_wide().chain(once(0)).collect()
}

fn get_module_handle() -> HMODULE {
    extern "C" {
        static __ImageBase: IMAGE_DOS_HEADER;
    }

    unsafe { &__ImageBase as *const _ as _ }
}

pub(super) extern "system" fn public_window_callback(
    window: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    let userdata_ptr = unsafe { GetWindowLongPtrW(window, GWL_USERDATA) };

    let _userdata = match (userdata_ptr, msg) {
        (0, WM_NCCREATE) => unsafe {
            let createstruct = &mut *(lparam as *mut CREATESTRUCTW);
            let initdata = &mut *(createstruct.lpCreateParams as *mut InitData);

            let result = match initdata.handle_create(window) {
                None => -1,
                Some(data) => {
                    SetWindowLongPtrW(window, GWL_USERDATA, data as _);
                    DefWindowProcW(window, msg, wparam, lparam)
                }
            };

            return result;
        },
        (0, WM_CREATE) => {
            return -1;
        }
        (0, _) => unsafe {
            return DefWindowProcW(window, msg, wparam, lparam);
        },
        (_, WM_CREATE) => unsafe {
            return DefWindowProcW(window, msg, wparam, lparam);
        },
        _ => userdata_ptr as *mut WindowData,
    };

    match msg {
        WM_PAINT => unsafe {
            ValidateRect(window, ptr::null());
            DefWindowProcW(window, msg, wparam, lparam)
        },
        WM_DESTROY => unsafe {
            println!("WM_DESTROY called!");
            PostQuitMessage(0);
            0
        },
        _ => unsafe { DefWindowProcW(window, msg, wparam, lparam) },
    }
}

fn register_window_class(class_name: &[u16]) {
    let wc = WNDCLASSEXW {
        lpszMenuName: ptr::null(),
        lpszClassName: class_name.as_ptr(),
        lpfnWndProc: Some(public_window_callback),

        cbClsExtra: 0,
        cbWndExtra: 0,
        cbSize: mem::size_of::<WNDCLASSEXW>() as u32,

        hIcon: ptr::null_mut(),
        hIconSm: ptr::null_mut(),
        hCursor: ptr::null_mut(),
        hInstance: get_module_handle(),
        hbrBackground: ptr::null_mut(),

        style: CS_HREDRAW | CS_VREDRAW,
    };

    unsafe { RegisterClassExW(&wc) };
}

fn create_window(options: WindowOptions) -> Result<Window, WindowError> {
    let title = encode_wide(&options.name);

    let window_class = encode_wide("Blackbird window");
    register_window_class(&window_class);

    let mut initdata = InitData { window: None };

    let handle = unsafe {
        CreateWindowExW(
            WINDOW_EX_STYLE::default(),
            window_class.as_ptr(),
            title.as_ptr(),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            get_module_handle(),
            &mut initdata as *mut _ as *mut _,
        )
    };

    if handle.is_null() {
        return Err(WindowError::CreationError("Null window handle"));
    }

    let window = initdata.window.unwrap();
    Ok(window)
}
