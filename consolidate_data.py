#!/usr/bin/env python3
"""
Consolidate Data.xlsx by grouping B/N/T variants into single entries.
Example: D0001B, D0001N, D0001T → D0001 (all 3 files map to same label)
"""

import openpyxl
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "Data.xlsx"
BACKUP_FILE = BASE_DIR / "Data_backup.xlsx"

def consolidate_data(input_xlsx, output_xlsx, backup=True):
    """
    Consolidate data by removing B/N/T suffixes.
    - Keeps only the base video name (e.g., D0001 instead of D0001B/D0001N/D0001T)
    - Verifies all variants have the same label
    - Creates backup before overwriting
    """
    
    # Create backup
    if backup and output_xlsx.exists():
        backup_path = output_xlsx.parent / f"{output_xlsx.stem}_backup.xlsx"
        if not backup_path.exists():
            import shutil
            shutil.copy(output_xlsx, backup_path)
            print(f"✓ Backup created: {backup_path}")
    
    # Load and parse
    wb = openpyxl.load_workbook(input_xlsx)
    ws = wb.active
    
    # Group by base name
    data_map = defaultdict(list)  # base_name -> [(filename, label), ...]
    
    print("\n📋 Reading Data.xlsx...")
    for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        if not row[0] or not row[1]:
            continue
        
        filename = str(row[0]).strip()
        label = str(row[1]).strip()
        
        # Extract base name (remove B/N/T suffix if present)
        if filename.endswith('.webm') or filename.endswith('.mp4'):
            base = filename.rsplit('.', 1)[0]
            # Remove B/N/T suffix
            if base[-1] in 'BNT':
                base_no_suffix = base[:-1]
            else:
                base_no_suffix = base
        else:
            base_no_suffix = filename
        
        data_map[base_no_suffix].append((filename, label))
    
    # Verify and consolidate
    print("\n🔍 Verifying variants...")
    consolidated = []  # [(base_name, label, variants_list), ...]
    issues = []
    
    for base_name in sorted(data_map.keys()):
        entries = data_map[base_name]
        labels = set(label for _, label in entries)
        
        if len(labels) > 1:
            # Conflict: same base name but different labels
            issue = f"❌ {base_name}: Multiple labels: {labels}"
            issues.append(issue)
            print(issue)
        else:
            label = labels.pop()
            variants = [fn for fn, _ in entries]
            consolidated.append((base_name, label, variants))
            print(f"✓ {base_name}: {label} ({len(variants)} variant{'s' if len(variants) > 1 else ''})")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} conflict(s). Please review above.")
        return False
    
    # Write consolidated data
    print(f"\n💾 Writing consolidated data ({len(consolidated)} entries)...")
    
    # Clear existing rows
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.value = None
    
    # Write new rows (keep all variants in naming but single label entry)
    for idx, (base_name, label, variants) in enumerate(consolidated, start=1):
        ws[f'A{idx+1}'] = variants[0]  # Write first variant as representative
        ws[f'B{idx+1}'] = label
        
        # Optional: write comment with all variants
        if len(variants) > 1:
            comment_text = f"Variants: {', '.join(variants)}"
            # Note: openpyxl comment support is complex, skip for now
    
    # Adjust column widths
    ws.column_dimensions['A'].width = 25
    ws.column_dimensions['B'].width = 40
    
    # Save
    wb.save(output_xlsx)
    wb.close()
    
    print(f"✓ Saved to: {output_xlsx}")
    print(f"  Original: {len(data_map)} base names, {sum(len(v) for v in data_map.values())} total files")
    print(f"  Consolidated: {len(consolidated)} entries")
    
    return True

def generate_variant_mapping(consolidated_data):
    """Generate a mapping of which variants belong to which base class."""
    variant_map = {}
    for base_name, label, variants in consolidated_data:
        for variant_file in variants:
            variant_map[variant_file.replace('.webm', '.mp4')] = (base_name, label)
    return variant_map

if __name__ == "__main__":
    print("=" * 60)
    print("Data.xlsx Consolidation Tool")
    print("=" * 60)
    
    if not DATA_FILE.exists():
        print(f"❌ Error: {DATA_FILE} not found!")
        exit(1)
    
    # Run consolidation
    success = consolidate_data(DATA_FILE, DATA_FILE, backup=True)
    
    if success:
        print("\n✅ Consolidation complete!")
        print("\nNext steps:")
        print("1. Verify the consolidated Data.xlsx in Excel")
        print("2. If OK, you can delete Data_backup.xlsx")
        print("3. Rerun: python backend/train_gpu.py")
    else:
        print("\n⚠️  Consolidation failed due to conflicts.")
        print("   Restore from backup if needed: Data_backup.xlsx")
