#!/usr/bin/env python3
"""Check if code coverage meets the required threshold."""

import sys
import xml.etree.ElementTree as ET


def check_coverage(xml_file: str, threshold: float = 100.0) -> bool:
    """
    Check if coverage meets threshold.
    
    Args:
        xml_file: Path to coverage.xml
        threshold: Required coverage percentage
        
    Returns:
        True if coverage >= threshold
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    coverage = float(root.attrib['line-rate']) * 100
    
    print(f"Current coverage: {coverage:.2f}%")
    print(f"Required coverage: {threshold:.2f}%")
    
    if coverage >= threshold:
        print("✓ Coverage check passed!")
        return True
    else:
        print(f"✗ Coverage check failed! Missing {threshold - coverage:.2f}%")
        return False


if __name__ == "__main__":
    xml_file = sys.argv[1] if len(sys.argv) > 1 else "coverage.xml"
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    
    success = check_coverage(xml_file, threshold)
    sys.exit(0 if success else 1)