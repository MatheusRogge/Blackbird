mod application;

use crate::application::Application;

fn main() {
    let application = Application::initialize();    
    application.main_loop();
}