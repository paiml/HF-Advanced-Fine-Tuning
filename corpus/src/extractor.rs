//! Documentation extraction from Rust source files.

use crate::{Category, CorpusEntry, CorpusError, Result};
use std::path::Path;
use std::process::Command;

/// Configuration for extraction.
#[derive(Debug, Clone)]
pub struct ExtractorConfig {
    /// Minimum quality score threshold.
    pub min_quality: f32,
    /// Maximum entries per repository.
    pub max_per_repo: usize,
    /// Clone directory for repositories.
    pub clone_dir: std::path::PathBuf,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            min_quality: 0.5,
            max_per_repo: 500,
            clone_dir: std::env::temp_dir().join("corpus-clones"),
        }
    }
}

/// Source repository specification.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RepoSpec {
    /// Repository URL (e.g., "https://github.com/clap-rs/clap")
    pub url: String,
    /// Specific commit SHA (7+ chars)
    pub commit: Option<String>,
    /// Repository name (e.g., "clap-rs/clap")
    pub name: String,
}

impl RepoSpec {
    /// Creates a new repo spec.
    #[must_use]
    pub fn new(url: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            commit: None,
            name: name.into(),
        }
    }

    /// Sets a specific commit.
    #[must_use]
    pub fn with_commit(mut self, commit: impl Into<String>) -> Self {
        self.commit = Some(commit.into());
        self
    }
}

/// Documentation extractor for Rust source files.
#[derive(Debug)]
pub struct DocExtractor {
    config: ExtractorConfig,
}

impl DocExtractor {
    /// Creates a new extractor with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ExtractorConfig::default(),
        }
    }

    /// Creates a new extractor with custom configuration.
    #[must_use]
    pub fn with_config(config: ExtractorConfig) -> Self {
        Self { config }
    }

    /// Clones a repository to the clone directory.
    ///
    /// # Errors
    ///
    /// Returns an error if git clone fails.
    pub fn clone_repo(&self, spec: &RepoSpec) -> Result<std::path::PathBuf> {
        let repo_dir = self.config.clone_dir.join(spec.name.replace('/', "_"));

        if repo_dir.exists() {
            std::fs::remove_dir_all(&repo_dir)?;
        }

        std::fs::create_dir_all(&repo_dir)?;

        let status = Command::new("git")
            .args(["clone", "--depth", "1", &spec.url])
            .arg(&repo_dir)
            .status()
            .map_err(|e| CorpusError::git(format!("git clone failed: {e}")))?;

        if !status.success() {
            return Err(CorpusError::git(format!(
                "git clone failed with status: {}",
                status
            )));
        }

        // Checkout specific commit if specified
        if let Some(ref commit) = spec.commit {
            let status = Command::new("git")
                .args(["checkout", commit])
                .current_dir(&repo_dir)
                .status()
                .map_err(|e| CorpusError::git(format!("git checkout failed: {e}")))?;

            if !status.success() {
                return Err(CorpusError::git(format!(
                    "git checkout {} failed",
                    commit
                )));
            }
        }

        Ok(repo_dir)
    }

    /// Gets the current commit SHA for a repository.
    ///
    /// # Errors
    ///
    /// Returns an error if git rev-parse fails.
    pub fn get_commit_sha(&self, repo_dir: &Path) -> Result<String> {
        let output = Command::new("git")
            .args(["rev-parse", "--short=7", "HEAD"])
            .current_dir(repo_dir)
            .output()
            .map_err(|e| CorpusError::git(format!("git rev-parse failed: {e}")))?;

        if !output.status.success() {
            return Err(CorpusError::git("git rev-parse failed"));
        }

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// Extracts documentation pairs from a repository.
    ///
    /// # Errors
    ///
    /// Returns an error if extraction fails.
    pub fn extract_from_repo(&self, spec: &RepoSpec) -> Result<Vec<CorpusEntry>> {
        let repo_dir = self.clone_repo(spec)?;
        let commit = self.get_commit_sha(&repo_dir)?;

        let mut entries = Vec::new();

        // Find all Rust files
        for entry in walkdir::WalkDir::new(&repo_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
        {
            let path = entry.path();
            let relative_path = path
                .strip_prefix(&repo_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();

            match self.extract_from_file(path, &spec.name, &commit, &relative_path) {
                Ok(file_entries) => entries.extend(file_entries),
                Err(e) => {
                    eprintln!("Warning: Failed to extract from {}: {}", relative_path, e);
                }
            }

            if entries.len() >= self.config.max_per_repo {
                break;
            }
        }

        // Filter by quality
        entries.retain(|e| e.quality_score >= self.config.min_quality);

        // Truncate to max
        entries.truncate(self.config.max_per_repo);

        Ok(entries)
    }

    /// Extracts documentation pairs from a single file.
    fn extract_from_file(
        &self,
        path: &Path,
        repo_name: &str,
        commit: &str,
        relative_path: &str,
    ) -> Result<Vec<CorpusEntry>> {
        let content = std::fs::read_to_string(path)?;
        let mut entries = Vec::new();

        // Parse the file with syn
        let file = match syn::parse_file(&content) {
            Ok(f) => f,
            Err(_) => return Ok(entries), // Skip unparseable files
        };

        // Extract module-level docs
        if let Some(doc) = extract_module_doc(&file) {
            // Extract module name from path (e.g., "src/lib.rs" -> "lib")
            // For "mod.rs" files, use the parent directory name instead
            let path = std::path::Path::new(&relative_path);
            let file_stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("module");
            let mod_name = if file_stem == "mod" || file_stem == "lib" || file_stem == "main" {
                // Use parent directory name for special module files
                path.parent()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .unwrap_or("module")
            } else {
                file_stem
            };
            let mut entry = CorpusEntry::new(
                format!("mod {} {{}}", mod_name),
                doc,
                Category::Module,
            );
            entry.source_repo = repo_name.to_string();
            entry.source_commit = commit.to_string();
            entry.source_file = relative_path.to_string();
            entry.source_line = 1;
            entry.quality_score = self.score_entry(&entry);
            entries.push(entry);
        }

        // Extract item-level docs
        for item in &file.items {
            if let Some((input, output, category, line)) = extract_item_doc(item) {
                let mut entry = CorpusEntry::new(input, output, category);
                entry.source_repo = repo_name.to_string();
                entry.source_commit = commit.to_string();
                entry.source_file = relative_path.to_string();
                entry.source_line = line;
                entry.quality_score = self.score_entry(&entry);
                entries.push(entry);
            }
        }

        Ok(entries)
    }

    /// Scores an entry based on quality heuristics.
    fn score_entry(&self, entry: &CorpusEntry) -> f32 {
        let mut score = 0.6f32; // Higher base score

        // Bonus for longer documentation
        let doc_len = entry.output.len();
        if doc_len > 50 {
            score += 0.05;
        }
        if doc_len > 100 {
            score += 0.1;
        }
        if doc_len > 200 {
            score += 0.1;
        }

        // Bonus for examples
        if entry.output.contains("# Example") || entry.output.contains("```") {
            score += 0.1;
        }

        // Bonus for arguments section
        if entry.output.contains("# Arguments") || entry.output.contains("# Parameters") {
            score += 0.05;
        }

        // Bonus for errors section
        if entry.output.contains("# Errors") || entry.output.contains("# Panics") {
            score += 0.05;
        }

        // Bonus for good formatting
        if entry.output.contains("///") {
            score += 0.05;
        }

        // Penalty for very short docs
        if doc_len < 20 {
            score -= 0.2;
        }

        score.clamp(0.0, 1.0)
    }
}

impl Default for DocExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Extracts module-level documentation from a file.
fn extract_module_doc(file: &syn::File) -> Option<String> {
    let docs: Vec<String> = file
        .attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("doc") {
                if let syn::Meta::NameValue(nv) = &attr.meta {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(s),
                        ..
                    }) = &nv.value
                    {
                        return Some(format!("//!{}", s.value()));
                    }
                }
            }
            None
        })
        .collect();

    if docs.is_empty() {
        None
    } else {
        Some(docs.join("\n"))
    }
}

/// Extracts documentation from an item (fn, struct, etc.).
fn extract_item_doc(item: &syn::Item) -> Option<(String, String, Category, u32)> {
    match item {
        syn::Item::Fn(f) => {
            let docs = extract_attrs_doc(&f.attrs)?;
            // Generate clean signature from syn::Signature
            let sig = format_fn_signature(&f.sig);
            let line = f.sig.fn_token.span.start().line as u32;
            let category = categorize_doc(&docs);
            Some((sig, docs, category, line))
        }
        syn::Item::Struct(s) => {
            let docs = extract_attrs_doc(&s.attrs)?;
            let sig = format!("struct {} {{}}", s.ident);
            let line = s.struct_token.span.start().line as u32;
            Some((sig, docs, Category::Function, line))
        }
        syn::Item::Enum(e) => {
            let docs = extract_attrs_doc(&e.attrs)?;
            let sig = format!("enum {} {{}}", e.ident);
            let line = e.enum_token.span.start().line as u32;
            Some((sig, docs, Category::Function, line))
        }
        syn::Item::Impl(i) => {
            // Extract docs from impl items
            for impl_item in &i.items {
                if let syn::ImplItem::Fn(f) = impl_item {
                    if let Some(docs) = extract_attrs_doc(&f.attrs) {
                        let sig = format_fn_signature(&f.sig);
                        let line = f.sig.fn_token.span.start().line as u32;
                        let category = categorize_doc(&docs);
                        return Some((sig, docs, category, line));
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Formats a function signature cleanly.
fn format_fn_signature(sig: &syn::Signature) -> String {
    let mut result = String::new();

    // Add async/const/unsafe
    if sig.asyncness.is_some() {
        result.push_str("async ");
    }
    if sig.constness.is_some() {
        result.push_str("const ");
    }
    if sig.unsafety.is_some() {
        result.push_str("unsafe ");
    }

    result.push_str("fn ");
    result.push_str(&sig.ident.to_string());

    // Generics
    if !sig.generics.params.is_empty() {
        result.push('<');
        let params: Vec<String> = sig.generics.params.iter().map(|p| {
            match p {
                syn::GenericParam::Type(t) => t.ident.to_string(),
                syn::GenericParam::Lifetime(l) => format!("'{}", l.lifetime.ident),
                syn::GenericParam::Const(c) => format!("const {}", c.ident),
            }
        }).collect();
        result.push_str(&params.join(", "));
        result.push('>');
    }

    // Parameters
    result.push('(');
    let params: Vec<String> = sig.inputs.iter().map(|arg| {
        match arg {
            syn::FnArg::Receiver(r) => {
                if r.reference.is_some() {
                    if r.mutability.is_some() { "&mut self".to_string() }
                    else { "&self".to_string() }
                } else {
                    "self".to_string()
                }
            }
            syn::FnArg::Typed(t) => {
                let pat = &t.pat;
                let ty = &t.ty;
                let pat_str = quote::quote!(#pat).to_string().replace(' ', "");
                let ty_str = quote::quote!(#ty).to_string();
                format!("{}: {}", pat_str, ty_str)
            }
        }
    }).collect();
    result.push_str(&params.join(", "));
    result.push(')');

    // Return type
    if let syn::ReturnType::Type(_, ty) = &sig.output {
        let ty_str = quote::quote!(#ty).to_string();
        result.push_str(" -> ");
        result.push_str(&ty_str);
    }

    // Add empty body
    result.push_str(" {}");

    result
}

/// Extracts documentation from attributes.
fn extract_attrs_doc(attrs: &[syn::Attribute]) -> Option<String> {
    let docs: Vec<String> = attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("doc") {
                if let syn::Meta::NameValue(nv) = &attr.meta {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(s),
                        ..
                    }) = &nv.value
                    {
                        return Some(format!("///{}", s.value()));
                    }
                }
            }
            None
        })
        .collect();

    if docs.is_empty() {
        None
    } else {
        Some(docs.join("\n"))
    }
}

/// Categorizes documentation based on content.
/// More aggressive detection to improve category balance.
fn categorize_doc(doc: &str) -> Category {
    let doc_lower = doc.to_lowercase();

    // Example detection - expanded for v1.1.0 to capture more patterns
    if doc_lower.contains("# example")
        || doc_lower.contains("```rust")
        || doc_lower.contains("```no_run")
        || doc_lower.contains("```ignore")
        || doc_lower.contains("```compile_fail")
        || doc_lower.contains("```should_panic")
        || doc_lower.contains("for example")
        || doc_lower.contains("usage:")
        || doc_lower.contains("basic usage")
        || (doc.contains("```") && (doc_lower.contains("use ") || doc_lower.contains("let ")))
        || (doc.contains("```") && doc_lower.contains("assert"))
        || (doc.contains("```") && doc_lower.contains("fn main"))
    {
        return Category::Example;
    }

    // Argument detection - look for parameter documentation patterns
    if doc_lower.contains("# argument")
        || doc_lower.contains("# parameter")
        || doc_lower.contains("* `")  // Bullet point param docs
        || doc_lower.contains("- `")  // Dash param docs
        || doc_lower.contains("takes a")
        || doc_lower.contains("accepts a")
        || doc_lower.contains("the `")
        || (doc_lower.contains("param") && !doc_lower.contains("example"))
    {
        return Category::Argument;
    }

    // Error detection - look for error-related documentation
    if doc_lower.contains("# error")
        || doc_lower.contains("# panic")
        || doc_lower.contains("returns an error")
        || doc_lower.contains("returns `err")
        || doc_lower.contains("will panic")
        || doc_lower.contains("may fail")
        || doc_lower.contains("can fail")
        || doc_lower.contains("fails if")
        || doc_lower.contains("error if")
        || doc_lower.contains("panics if")
    {
        return Category::Error;
    }

    // Default to Function
    Category::Function
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repo_spec_new() {
        let spec = RepoSpec::new("https://github.com/test/repo", "test/repo");
        assert_eq!(spec.url, "https://github.com/test/repo");
        assert_eq!(spec.name, "test/repo");
        assert!(spec.commit.is_none());
    }

    #[test]
    fn test_repo_spec_with_commit() {
        let spec = RepoSpec::new("https://github.com/test/repo", "test/repo").with_commit("abc1234");
        assert_eq!(spec.commit, Some("abc1234".to_string()));
    }

    #[test]
    fn test_extractor_config_default() {
        let config = ExtractorConfig::default();
        assert_eq!(config.min_quality, 0.5);
        assert_eq!(config.max_per_repo, 500);
    }

    #[test]
    fn test_categorize_doc() {
        assert_eq!(categorize_doc("/// # Examples\n/// ```rust"), Category::Example);
        assert_eq!(categorize_doc("/// # Arguments\n/// * `foo` - the foo"), Category::Argument);
        assert_eq!(categorize_doc("/// # Errors\n/// Returns an error if..."), Category::Error);
        assert_eq!(categorize_doc("/// Simple doc"), Category::Function);
        // Test new patterns
        assert_eq!(categorize_doc("/// Takes a `String`"), Category::Argument);
        assert_eq!(categorize_doc("/// Returns an error if the file is not found"), Category::Error);
        assert_eq!(categorize_doc("/// ```no_run\n/// let x = 1;\n/// ```"), Category::Example);
    }

    #[test]
    fn test_score_entry() {
        let extractor = DocExtractor::new();

        let mut entry = CorpusEntry::new(
            "fn test() {}".to_string(),
            "/// Short".to_string(),
            Category::Function,
        );
        let short_score = extractor.score_entry(&entry);

        entry.output = "/// This is a much longer documentation that provides more context and detail about the function's purpose and behavior.".to_string();
        let long_score = extractor.score_entry(&entry);

        assert!(long_score > short_score);
    }
}
