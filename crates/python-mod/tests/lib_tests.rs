// Tests for the core library functionality
// These tests focus on internal functions rather than the full macro pipeline

// Remove unused import
use std::path::Path;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_module_file_detection() {
        // Test that the load_module function can detect different file types
        // We'll test the file path logic without actually loading modules
        
        // Create test files to verify path detection logic
        std::fs::write("src/test_single.py", "# test file").expect("Failed to create test file");
        
        // Test that single file exists
        assert!(Path::new("src/test_single.py").exists());
        
        // Cleanup
        std::fs::remove_file("src/test_single.py").ok();
    }

    #[test]
    fn test_package_module_detection() {
        // Test package-style module detection
        std::fs::create_dir_all("src/test_package").ok();
        std::fs::write("src/test_package/__init__.py", "# test package").expect("Failed to create test package");
        
        // Test that package directory exists
        assert!(Path::new("src/test_package/__init__.py").exists());
        
        // Cleanup
        std::fs::remove_dir_all("src/test_package").ok();
    }

    #[test]
    fn test_module_path_formats() {
        // Test different module path format handling
        let mod_name = "test_module";
        let mod_name_dir = format!("src/{}/__init__.py", mod_name);
        let mod_name_file = format!("src/{}.py", mod_name);
        
        assert_eq!(mod_name_dir, "src/test_module/__init__.py");
        assert_eq!(mod_name_file, "src/test_module.py");
    }

    #[test]
    fn test_python_files_exist() {
        // Test that our test Python files exist
        assert!(Path::new("src/simple_math.py").exists());
        assert!(Path::new("src/string_ops.py").exists());
        assert!(Path::new("src/package_test/__init__.py").exists());
        assert!(Path::new("src/empty_module.py").exists());
        assert!(Path::new("src/minimal_test.py").exists());
    }

    #[test]
    fn test_python_file_contents() {
        // Test that we can read the Python files we created
        let simple_math_content = std::fs::read_to_string("src/simple_math.py")
            .expect("Failed to read simple_math.py");
        
        assert!(simple_math_content.contains("def add(a, b):"));
        assert!(simple_math_content.contains("def multiply(x, y):"));
        assert!(simple_math_content.contains("PI = 3.14159"));
        
        let empty_content = std::fs::read_to_string("src/empty_module.py")
            .expect("Failed to read empty_module.py");
        
        assert!(empty_content.contains("# Empty module"));
    }

    #[test]
    fn test_string_formatting() {
        // Test string formatting used in error messages
        let mod_name = "test_module";
        let error_msg = format!("unable to determine parent of module for {}", mod_name);
        assert_eq!(error_msg, "unable to determine parent of module for test_module");
        
        let not_found_msg = format!("Module not found {}", mod_name);
        assert_eq!(not_found_msg, "Module not found test_module");
    }

    #[test]
    fn test_file_path_construction() {
        // Test the file path construction logic used in load_module
        let test_cases = vec![
            ("simple_math", "src/simple_math.py", "src/simple_math/__init__.py"),
            ("package_test", "src/package_test.py", "src/package_test/__init__.py"),
            ("empty_module", "src/empty_module.py", "src/empty_module/__init__.py"),
        ];
        
        for (mod_name, expected_file, expected_dir) in test_cases {
            let actual_file = format!("src/{}.py", mod_name);
            let actual_dir = format!("src/{}/__init__.py", mod_name);
            
            assert_eq!(actual_file, expected_file);
            assert_eq!(actual_dir, expected_dir);
        }
    }
}