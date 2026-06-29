fn main() {
  // tauri_build only re-runs the build when tauri.conf.json changes, so editing
  // the icon PNGs alone never re-embeds the app icon — the dev window icon is
  // baked in from icons/32x32.png at compile time via generate_context!(), and a
  // stale binary silently keeps the old logo. Track the icon files so a logo
  // change triggers a rebuild instead.
  for icon in [
    "icons/32x32.png",
    "icons/128x128.png",
    "icons/128x128@2x.png",
    "icons/icon.icns",
    "icons/icon.ico",
    "icons/icon.png",
  ] {
    println!("cargo:rerun-if-changed={icon}");
  }
  tauri_build::build()
}
