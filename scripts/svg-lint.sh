#!/bin/bash
# SVG Linting Script
# Checks for common SVG issues including text overlaps

set -e

SVG_DIR="${1:-docs/images}"
ERRORS=0

echo "SVG Lint: Checking $SVG_DIR"
echo "==========================="

# Find all SVG files
SVG_FILES=$(find "$SVG_DIR" -name "*.svg" 2>/dev/null)

if [ -z "$SVG_FILES" ]; then
    echo "No SVG files found in $SVG_DIR"
    exit 0
fi

for svg in $SVG_FILES; do
    echo ""
    echo "Checking: $svg"

    # 1. Check for valid XML
    if ! xmllint --noout "$svg" 2>/dev/null; then
        echo "  ERROR: Invalid XML structure"
        ERRORS=$((ERRORS + 1))
        continue
    fi
    echo "  OK: Valid XML"

    # 2. Check for text elements with same y and close x (potential overlap)
    # Extract text elements with their x and y positions
    TEXT_POSITIONS=$(grep -oP '<text[^>]*x="(\d+)"[^>]*y="(\d+)"' "$svg" 2>/dev/null | \
        sed 's/<text[^>]*x="\([0-9]*\)"[^>]*y="\([0-9]*\)"/\1 \2/' || true)

    if [ -n "$TEXT_POSITIONS" ]; then
        # Group by y position and check for close x values
        OVERLAP_FOUND=0
        while IFS= read -r line; do
            y=$(echo "$line" | cut -d' ' -f2)
            x=$(echo "$line" | cut -d' ' -f1)

            # Check if there's another text at same y within 150px
            SAME_Y=$(echo "$TEXT_POSITIONS" | awk -v y="$y" -v x="$x" '
                $2 == y && $1 != x {
                    diff = ($1 > x) ? $1 - x : x - $1
                    if (diff < 150 && diff > 0) print $1, $2, diff
                }
            ')

            if [ -n "$SAME_Y" ]; then
                OVERLAP_FOUND=1
            fi
        done <<< "$TEXT_POSITIONS"

        if [ "$OVERLAP_FOUND" -eq 1 ]; then
            echo "  WARN: Potential text overlap detected (texts within 150px on same line)"
        else
            echo "  OK: No obvious text overlaps"
        fi
    fi

    # 3. Check viewBox is set
    if ! grep -q 'viewBox=' "$svg"; then
        echo "  WARN: No viewBox attribute found"
    else
        echo "  OK: viewBox present"
    fi

    # 4. Check for reasonable file size (< 500KB)
    SIZE=$(stat -f%z "$svg" 2>/dev/null || stat -c%s "$svg" 2>/dev/null)
    if [ "$SIZE" -gt 512000 ]; then
        echo "  WARN: Large file size (${SIZE} bytes > 500KB)"
    else
        echo "  OK: File size (${SIZE} bytes)"
    fi

    # 5. Check for embedded raster images (potential issue)
    if grep -q 'data:image/png\|data:image/jpeg' "$svg"; then
        echo "  WARN: Contains embedded raster images"
    fi
done

echo ""
echo "==========================="
if [ "$ERRORS" -gt 0 ]; then
    echo "FAILED: $ERRORS error(s) found"
    exit 1
else
    echo "PASSED: All SVG checks completed"
    exit 0
fi
