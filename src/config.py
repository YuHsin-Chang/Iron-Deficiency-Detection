# ================================
# Path settings
# ================================
import os

# Automatically detect project root based on config.py location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Path to cache and intermediate files
cache_dir = os.path.join(project_root, 'data', 'cache')
feature_selection_dir = os.path.join(project_root, 'data', 'feature_selection')
data_dir = os.path.join(project_root, 'data', 'processed')
# Make sure directories exist (optional)

os.makedirs(data_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(feature_selection_dir, exist_ok=True)


# ================================
# Variable assigning
# ================================
# Basic demographic features
Basic_columns = ["Age", "Gender"]

# CBC: White blood cell count
CBC_WBC_columns = ["WBC"]


# CBC: Differential Count
CBC_DC_columns = ["NE",
    "NE#",
    "MO",
    "MO#",
    "LY",
    "LY#",
    "BA",
    "BA#",
    "EO",
    "EO#",
    "NLR",
    "PLR",
    "MDW",
]

# CBC: Red blood cell-related features
CBC_RBC_columns = ["HGB", "RBC", "NRBC", "MCV", "MCH", "MCHC", "HCT", "RDW"]

# CBC: Platelet-related features
CBC_PLT_columns = ["PLT", "@PDW"]

# CPD: RBC-related CPD parameters
CPD_RBC_columns = ["@MAF", "@LHD"]

# CPD: Platelet-related CPD parameters
CPD_PLT_columns = ["@PCT"]

# CPD: White blood cell CPD parameters
CPD_WBC_columns = [
    "@MN-AL2-EO",
    "@MN-AL2-LY",
    "@MN-AL2-MO",
    "@MN-AL2-NE",
    "@MN-C-EO",
    "@MN-C-LY",
    "@MN-C-MO",
    "@MN-C-NE",
    "@MN-LALS-EO",
    "@MN-LALS-LY",
    "@MN-LALS-MO",
    "@MN-LALS-NE",
    "@MN-LMALS-EO",
    "@MN-LMALS-LY",
    "@MN-LMALS-MO",
    "@MN-LMALS-NE",
    "@MN-MALS-EO",
    "@MN-MALS-LY",
    "@MN-MALS-MO",
    "@MN-MALS-NE",
    "@MN-UMALS-EO",
    "@MN-UMALS-LY",
    "@MN-UMALS-MO",
    "@MN-UMALS-NE",
    "@MN-V-EO",
    "@MN-V-LY",
    "@MN-V-MO",
    "@MN-V-NE",
    "@SD-AL2-EO",
    "@SD-AL2-LY",
    "@SD-AL2-MO",
    "@SD-AL2-NE",
    "@SD-C-EO",
    "@SD-C-LY",
    "@SD-C-MO",
    "@SD-C-NE",
    "@SD-LALS-EO",
    "@SD-LALS-LY",
    "@SD-LALS-MO",
    "@SD-LALS-NE",
    "@SD-LMALS-EO",
    "@SD-LMALS-LY",
    "@SD-LMALS-MO",
    "@SD-LMALS-NE",
    "@SD-MALS-EO",
    "@SD-MALS-LY",
    "@SD-MALS-MO",
    "@SD-MALS-NE",
    "@SD-UMALS-EO",
    "@SD-UMALS-LY",
    "@SD-UMALS-MO",
    "@SD-UMALS-NE",
    "@SD-V-EO",
    "@SD-V-LY",
    "@SD-V-MO",
    "@SD-V-NE",
    "@WDOP",
    "@WNOP",
]
