# Human Review Sample

Generated: 2026-01-21 15:59:13.475660644 UTC
Sample size: 10

## Entry 1 / 10

**ID:** `e5db3568-ac0f-4bf6-8931-11f55e3c3623`

**Source:** `eza-community/eza` @ `58b98cf` (line 41)

**Category:** example

**Quality Score:** 1.000

### Input (Signature)

```rust
const fn is_none(&self) -> bool {}
```

### Output (Documentation)

```rust
/// Returns `true` if `None`
///
/// # Examples
///
/// ```
/// use eza::fs::recursive_size::RecursiveSize;
///
/// let x = RecursiveSize::None;
/// assert_eq!(x.is_none(), true);
///
/// let x = RecursiveSize::Unknown;
/// assert_eq!(x.is_none(), false);
///
/// let x = RecursiveSize::Some(0, 0);
/// assert_eq!(x.is_none(), false);
/// ```
```

### Review

- [ ] Accurate
- [ ] Helpful
- [ ] Idiomatic
- [ ] Complete

**Notes:**

---

## Entry 2 / 10

**ID:** `902a7571-46b9-476a-b875-0fed9d43ebd2`

**Source:** `BurntSushi/ripgrep` @ `0a88ccc` (line 139)

**Category:** function

**Quality Score:** 0.900

### Input (Signature)

```rust
fn parse_low_raw(rawargs: impl IntoIterator < Item = impl Into < OsString > >) -> anyhow :: Result < LowArgs > {}
```

### Output (Documentation)

```rust
/// Parse the sequence of CLI arguments given a low level typed set of
/// arguments.
///
/// This is exposed for testing that the correct low-level arguments are parsed
/// from a CLI. It just runs the parser once over the CLI arguments. It doesn't
/// setup logging or read from a config file.
///
/// This assumes the iterator given does *not* begin with the binary name.
```

### Review

- [ ] Accurate
- [ ] Helpful
- [ ] Idiomatic
- [ ] Complete

**Notes:**

---

## Entry 3 / 10

**ID:** `b54e6af1-bd12-4260-b6ae-86b5d2890065`

**Source:** `BurntSushi/ripgrep` @ `0a88ccc` (line 505)

**Category:** function

**Quality Score:** 0.900

### Input (Signature)

```rust
fn new_no_color(wtr: W) -> Standard < NoColor < W > > {}
```

### Output (Documentation)

```rust
/// Return a standard printer with a default configuration that writes
/// matches to the given writer.
///
/// The writer can be any implementation of `io::Write`. With this
/// constructor, the printer will never emit colors.
```

### Review

- [ ] Accurate
- [ ] Helpful
- [ ] Idiomatic
- [ ] Complete

**Notes:**

---

## Entry 4 / 10

**ID:** `3118234d-1a08-42f6-8dd5-625c64772582`

**Source:** `sharkdp/fd` @ `bf81fb9` (line 2585)

**Category:** error

**Quality Score:** 0.800

### Input (Signature)

```rust
fn test_error_if_hidden_not_set_and_pattern_starts_with_dot() {}
```

### Output (Documentation)

```rust
/// Print error if search pattern starts with a dot and --hidden is not set
/// (Unix only, hidden files on Windows work differently)
```

### Review

- [ ] Accurate
- [ ] Helpful
- [ ] Idiomatic
- [ ] Complete

**Notes:**

---

## Entry 5 / 10

**ID:** `80adc3ae-8261-49b5-98da-db2021d6ff7f`

**Source:** `sharkdp/bat` @ `cd06fe4` (line 345)

**Category:** error

**Quality Score:** 0.800

### Input (Signature)

```rust
fn run() -> Result < bool > {}
```

### Output (Documentation)

```rust
/// Returns `Err(..)` upon fatal errors. Otherwise, returns `Ok(true)` on full success and
/// `Ok(false)` if any intermediate errors occurred (were printed).
```

### Review

- [ ] Accurate
- [ ] Helpful
- [ ] Idiomatic
- [ ] Complete

**Notes:**

---

## Entry 6 / 10

**ID:** `32e58ee9-09c3-4046-98c6-5b40a9c90162`

**Source:** `clap-rs/clap` @ `b8be10b` (line 88)

**Category:** argument

**Quality Score:** 0.900

### Input (Signature)

```rust
fn raw(kind: ErrorKind, message: impl Display) -> Self {}
```

### Output (Documentation)

```rust
/// Create an unformatted error
///
/// This is for you need to pass the error up to
/// a place that has access to the `Command` at which point you can call [`Error::format`].
///
/// Prefer [`Command::error`] for generating errors.
///
/// [`Command::error`]: crate::Command::error
```

### Review

- [ ] Accurate
- [ ] Helpful
- [ ] Idiomatic
- [ ] Complete

**Notes:**

---

## Entry 7 / 10

**ID:** `bfadad70-431d-499e-9f76-6dece6ec03db`

**Source:** `eza-community/eza` @ `58b98cf` (line 152)

**Category:** argument

**Quality Score:** 0.900

### Input (Signature)

```rust
fn search(&self, index: & Path, prefix_lookup: bool) -> f :: Git {}
```

### Output (Documentation)

```rust
/// Searches through this repository for a path (to a file or directory,
/// depending on the prefix-lookup flag) and returns its Git status.
///
/// Actually querying the `git2` repository for the mapping of paths to
/// Git statuses is only done once, and gets cached so we don’t need to
/// re-query the entire repository the times after that.
///
/// The temporary `Processing` enum variant is used after the `git2`
/// repository is moved out, but before the results have been moved in!
/// See <https://stackoverflow.com/q/45985827/3484614>
```

### Review

- [ ] Accurate
- [ ] Helpful
- [ ] Idiomatic
- [ ] Complete

**Notes:**

---

## Entry 8 / 10

**ID:** `85f0d1d2-f838-4d80-87e4-546514607ff2`

**Source:** `clap-rs/clap` @ `b8be10b` (line 1)

**Category:** module

**Quality Score:** 0.750

### Input (Signature)

```rust
mod cargo_example_derive {}
```

### Output (Documentation)

```rust
//! # Example: cargo subcommand (Derive API)
//!
//! ```rust
//! ```
//!
```

### Review

- [ ] Accurate
- [ ] Helpful
- [ ] Idiomatic
- [ ] Complete

**Notes:**

---

## Entry 9 / 10

**ID:** `b3d5cbcc-01c4-4726-857c-542e710e86f2`

**Source:** `BurntSushi/ripgrep` @ `0a88ccc` (line 404)

**Category:** function

**Quality Score:** 0.900

### Input (Signature)

```rust
fn suggest(unrecognized: & str) -> Option < String > {}
```

### Output (Documentation)

```rust
/// Possibly return a message suggesting flags similar in the name to the one
/// given.
///
/// The one given should be a flag given by the user (without the leading
/// dashes) that was unrecognized. This attempts to find existing flags that
/// are similar to the one given.
```

### Review

- [ ] Accurate
- [ ] Helpful
- [ ] Idiomatic
- [ ] Complete

**Notes:**

---

## Entry 10 / 10

**ID:** `e6c1d9e9-1c94-49b6-b08e-a3a89c5bfb9b`

**Source:** `BurntSushi/ripgrep` @ `0a88ccc` (line 10)

**Category:** function

**Quality Score:** 0.900

### Input (Signature)

```rust
fn default_color_specs() -> Vec < UserColorSpec > {}
```

### Output (Documentation)

```rust
/// Returns a default set of color specifications.
///
/// This may change over time, but the color choices are meant to be fairly
/// conservative that work across terminal themes.
///
/// Additional color specifications can be added to the list returned. More
/// recently added specifications override previously added specifications.
```

### Review

- [ ] Accurate
- [ ] Helpful
- [ ] Idiomatic
- [ ] Complete

**Notes:**

---

