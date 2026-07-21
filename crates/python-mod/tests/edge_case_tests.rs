// Edge case and boundary condition tests
use std::path::Path;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_python_files() {
        // Test handling of empty Python files
        let empty_file_path = "src/empty_module.py";
        assert!(Path::new(empty_file_path).exists());
        
        let content = std::fs::read_to_string(empty_file_path)
            .expect("Failed to read empty module");
        
        // Should contain at least a comment
        assert!(content.len() > 0);
    }

    #[test] 
    fn test_python_file_extensions() {
        // Test that all our test files have correct extensions
        let python_files = vec![
            "src/simple_math.py",
            "src/string_ops.py", 
            "src/empty_module.py",
            "src/minimal_test.py",
            "src/simple_math_with_preamble.py",
        ];
        
        for file in python_files {
            assert!(file.ends_with(".py"), "File {} should end with .py", file);
            if Path::new(file).exists() {
                let content = std::fs::read_to_string(file)
                    .expect(&format!("Failed to read {}", file));
                // Basic validation that it's not empty (unless it's the empty module)
                if !file.contains("empty") {
                    assert!(content.len() > 10, "File {} seems too short", file);
                }
            }
        }
    }

    #[test]
    fn test_package_init_files() {
        // Test package-style modules have proper __init__.py files
        let package_init = "src/package_test/__init__.py";
        if Path::new(package_init).exists() {
            let content = std::fs::read_to_string(package_init)
                .expect("Failed to read package __init__.py");
            
            assert!(content.len() > 0);
            assert!(content.contains("def ") || content.contains("# "));
        }
    }

    #[test]
    fn test_module_name_validation() {
        // Test that module names follow Python naming conventions
        let valid_names = vec![
            "simple_math",
            "string_ops",
            "package_test", 
            "empty_module",
            "minimal_test",
        ];
        
        for name in valid_names {
            // Should start with letter or underscore
            assert!(name.chars().next().unwrap().is_alphabetic() || name.starts_with('_'));
            
            // Should only contain alphanumeric and underscore
            assert!(name.chars().all(|c| c.is_alphanumeric() || c == '_'));
            
            // Should not be empty
            assert!(!name.is_empty());
            
            // Should not contain spaces
            assert!(!name.contains(' '));
        }
    }

    #[test]
    fn test_file_system_edge_cases() {
        // Test various file system scenarios
        
        // Test that src directory exists
        assert!(Path::new("src").exists());
        assert!(Path::new("src").is_dir());
        
        // Test that tests directory exists
        assert!(Path::new("tests").exists());
        assert!(Path::new("tests").is_dir());
    }

    #[test]
    fn test_python_syntax_validation() {
        // Basic validation of Python files we created
        let files_to_check = vec![
            "src/simple_math.py",
            "src/string_ops.py",
            "src/minimal_test.py",
        ];
        
        for file in files_to_check {
            if Path::new(file).exists() {
                let content = std::fs::read_to_string(file)
                    .expect(&format!("Failed to read {}", file));
                
                // Basic Python syntax checks
                if content.contains("def ") {
                    // Functions should have colons
                    assert!(content.contains(":"), "Python function missing colon in {}", file);
                }
                
                if content.contains("return ") {
                    // Return statements should be indented
                    assert!(content.contains("    return"), "Return statement not indented in {}", file);
                }
            }
        }
    }

    #[test]
    fn test_comment_preservation() {
        // Test that Python comments are preserved in our test files
        let files_with_comments = vec![
            ("src/simple_math.py", "# Simple math operations"),
            ("src/string_ops.py", "# String operations"),
            ("src/empty_module.py", "# Empty module"),
        ];
        
        for (file, expected_comment) in files_with_comments {
            if Path::new(file).exists() {
                let content = std::fs::read_to_string(file)
                    .expect(&format!("Failed to read {}", file));
                assert!(content.contains(expected_comment), 
                    "Expected comment '{}' not found in {}", expected_comment, file);
            }
        }
    }

    #[test]
    fn test_directory_structure() {
        // Test the expected directory structure
        let expected_dirs = vec![
            "src",
            "tests", 
            "src/package_test",
        ];
        
        for dir in expected_dirs {
            if Path::new(dir).exists() {
                assert!(Path::new(dir).is_dir(), "{} should be a directory", dir);
            }
        }
        
        let expected_files = vec![
            "Cargo.toml",
            "README.md",
            "CLAUDE.md",
            "src/lib.rs",
        ];
        
        for file in expected_files {
            assert!(Path::new(file).exists(), "{} should exist", file);
            assert!(Path::new(file).is_file(), "{} should be a file", file);
        }
    }
}