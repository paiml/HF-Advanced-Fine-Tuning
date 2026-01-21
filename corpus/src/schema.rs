//! Arrow schema definition for corpus data.

use arrow::datatypes::{DataType, Field, Schema};

/// Returns the Arrow schema for corpus data.
#[must_use]
pub fn arrow_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("input", DataType::Utf8, false),
        Field::new("output", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("source_repo", DataType::Utf8, false),
        Field::new("source_commit", DataType::Utf8, false),
        Field::new("source_file", DataType::Utf8, false),
        Field::new("source_line", DataType::Int32, false),
        Field::new("tokens_input", DataType::Int32, false),
        Field::new("tokens_output", DataType::Int32, false),
        Field::new("quality_score", DataType::Float32, false),
    ])
}

/// Field names in the corpus schema.
pub const FIELD_NAMES: &[&str] = &[
    "id",
    "input",
    "output",
    "category",
    "source_repo",
    "source_commit",
    "source_file",
    "source_line",
    "tokens_input",
    "tokens_output",
    "quality_score",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_has_all_fields() {
        let schema = arrow_schema();
        for field_name in FIELD_NAMES {
            assert!(
                schema.field_with_name(field_name).is_ok(),
                "Schema missing field: {}",
                field_name
            );
        }
    }

    #[test]
    fn test_schema_field_count() {
        let schema = arrow_schema();
        assert_eq!(schema.fields().len(), FIELD_NAMES.len());
    }
}
