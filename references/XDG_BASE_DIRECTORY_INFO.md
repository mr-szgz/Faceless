Below is the converted Markdown version of the provided wiki content. All wiki-specific syntax, templates, and styling have been removed or normalized to standard Markdown.

---

# XDG Base Directory

This article summarizes the XDG Base Directory specification and tracks software support.

## Specification

Read the full specification:
[https://specifications.freedesktop.org/basedir-spec/latest/](https://specifications.freedesktop.org/basedir-spec/latest/)

Only `XDG_RUNTIME_DIR` is typically set by default (via `pam_systemd`). Other variables must be defined by the user.

Changing runtime behavior may affect tools such as PipeWire or Chromium screen sharing.

### User directories

* **XDG_CONFIG_HOME**
  User-specific configuration files
  Default: `$HOME/.config`

* **XDG_CACHE_HOME**
  Non-essential cached data
  Default: `$HOME/.cache`

* **XDG_DATA_HOME**
  User-specific data files
  Default: `$HOME/.local/share`

* **XDG_STATE_HOME**
  Persistent state data
  Default: `$HOME/.local/state`

* **XDG_RUNTIME_DIR**

  * Runtime data (sockets, pipes, etc.)
  * Must be owned by user with `0700` permissions
  * Local filesystem only
  * Temporary (lifetime of login session)
  * May be cleaned periodically
  * Often `/run/user/$UID`

### System directories

* **XDG_DATA_DIRS**
  Colon-separated list
  Default: `/usr/local/share:/usr/share`

* **XDG_CONFIG_DIRS**
  Colon-separated list
  Default: `/etc/xdg`

---

## Support

This section catalogs software supporting the XDG Base Directory specification.

### Categories

* Supported
* Partial
* Hardcoded

### Supported (examples)

| Application   | Legacy Path               | Notes                                |
| ------------- | ------------------------- | ------------------------------------ |
| act           | `~/.actrc`                | Uses `XDG_CONFIG_HOME/act/actrc`     |
| ALSA          | `~/.asoundrc`             | Uses `XDG_CONFIG_HOME/alsa/asoundrc` |
| aria2         | `~/.aria2`                | Uses config + cache dirs             |
| audacity      | `~/.audacity-data`        | Uses XDG if legacy absent            |
| binwalk       | `~/.binwalk`              | Uses `XDG_CONFIG_HOME/binwalk`       |
| bitwarden-cli | `~/.config/Bitwarden CLI` | Can override via env                 |
| blender       | `~/.blender`              | Supported                            |
| ccache        | `~/.ccache`               | Uses cache + config dirs             |
| curl          | `~/.curlrc`               | Uses `XDG_CONFIG_HOME/curlrc`        |
| emacs         | `~/.emacs`                | Uses `XDG_CONFIG_HOME/emacs/init.el` |
| firefox       | `~/.mozilla`              | Partial issues with XDG              |
| git           | `~/.gitconfig`            | Uses `XDG_CONFIG_HOME/git/...`       |
| htop          | `~/.htoprc`               | Uses `XDG_CONFIG_HOME/htop/htoprc`   |
| neovim        | `~/.nvim`                 | Full support                         |
| tmux          | `~/.tmux.conf`            | Uses XDG config dir                  |
| vim           | `~/.vim`                  | Partial/full support evolving        |

*(Large table truncated; original includes extensive list of applications and migration behavior.)*

---

### Partial Support (examples)

| Application | Legacy Path         | Workaround                   |
| ----------- | ------------------- | ---------------------------- |
| ansible     | `~/.ansible`        | Use environment variables    |
| aws-cli     | `~/.aws`            | Set `AWS_CONFIG_FILE`        |
| docker      | `~/.docker`         | Set `DOCKER_CONFIG`          |
| npm         | `~/.npm`            | Configure via env and config |
| python      | `~/.python_history` | Use `PYTHON_HISTORY`         |
| go          | `~/go`              | Set `GOPATH` and cache dirs  |
| rust/cargo  | `~/.cargo`          | Set `CARGO_HOME`             |

---

### Hardcoded (examples)

| Application          | Legacy Path    | Notes                   |
| -------------------- | -------------- | ----------------------- |
| bash                 | `~/.bashrc`    | Not XDG compliant       |
| ssh                  | `~/.ssh`       | Widely assumed location |
| docker (older parts) | `~/.docker`    | Partial support only    |
| flatpak              | `~/.var`       | Not compliant           |
| steam                | `~/.steam`     | Hardcoded usage         |
| minecraft            | `~/.minecraft` | Will not fix            |
| eclipse              | `~/.eclipse`   | Requires config hacks   |

---

## Tools

* **xdg-ninja**
  Detects files in `$HOME` that should be moved to XDG locations
  [https://github.com/b3nj5m1n/xdg-ninja](https://github.com/b3nj5m1n/xdg-ninja)

* **boxxy**
  Wraps applications to redirect file locations

* **ephemeral**
  Moves Chromium/Electron cache to proper cache dirs

---

## Libraries

### C

* libXDGdirs
* libxdg-basedir

### C++

* xdg-utils-cxx
* xdgpp

### Go

* adrg/xdg
* configdir

### Python

* pyxdg
* platformdirs

### Rust

* directories-rs
* rust-xdg

### JVM

* directories-jvm

### Other languages

* Haskell, Perl, Ruby, Swift, Vala all have implementations

---

## Tips and tricks

### Hide unwanted directories

You can hide directories (e.g., `.config`, `.cache`) in file managers:

```bash
echo path >> ~/.hidden
```

---

## See also

* GNOME XDG initiative
  [https://wiki.gnome.org/Initiatives/GnomeGoals/XDGConfigFolders](https://wiki.gnome.org/Initiatives/GnomeGoals/XDGConfigFolders)

* Rob Pike on dotfiles
  [https://web.archive.org/web/20180827160401/plus.google.com/+RobPikeTheHuman/posts/R58WgWwN9jp](https://web.archive.org/web/20180827160401/plus.google.com/+RobPikeTheHuman/posts/R58WgWwN9jp)

* systemd-path

* file-hierarchy

* Dotfiles notes
  [https://github.com/grawity/dotfiles](https://github.com/grawity/dotfiles)

* Article: Modify apps to use XDG
  [https://ploum.net/207-modify-your-application-to-use-xdg-folders/](https://ploum.net/207-modify-your-application-to-use-xdg-folders/)

---

Source: 
