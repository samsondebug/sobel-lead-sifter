#!/usr/bin/env python3
import pandas as pd
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def sift_sobel_leads(input_csv, output_csv='sobel_worthwhile_leads.csv'):
    """
    Process leads exported from Pebble CRM for Sobel Property Investments.
    Filters leads to identify high-priority ones worth pursuing based on:
    - Valid contact info (Email or Phone)
    - Lead_Status not 'Cold' or 'Rejected'
    - Property_Size > 5 acres
    - Estimated_Value > $50,000
    - Prioritizes leads from Texas or other key states

    Saves filtered leads to a CSV file for follow-up.

    Args:
        input_csv (str): Path to the input CSV file from Pebble CRM
        output_csv (str): Path to save filtered leads (default: sobel_worthwhile_leads.csv)

    Returns:
        None
    """
    try:
        # Read the CSV file
        logging.info(f"Reading input CSV: {input_csv}")
        df = pd.read_csv(input_csv)
        
        # Log available columns for debugging
        logging.info(f"Available columns in CSV: {df.columns.tolist()}")
        
        # Define high-priority states (customizable)
        priority_states = ['TX']  # Texas, where Sobel Properties is based
        
        # Define filtering criteria
        worthwhile = df[
            (df['Email'].notna() | df['Phone'].notna()) &  # Has contact info
            (~df['Lead_Status'].isin(['Cold', 'Rejected'])) &  # Not cold or rejected
            (df['Property_Size'] > 5) &  # Larger than 5 acres
            (df['Estimated_Value'] > 50000)  # Value over $50,000
        ]
        
        # Add a priority flag for leads in high-priority states
        worthwhile['Priority'] = worthwhile['State'].apply(
            lambda x: 'High' if x in priority_states else 'Standard'
        )
        
        # Sort by Priority (High first) and Estimated_Value (descending)
        worthwhile = worthwhile.sort_values(
            by=['Priority', 'Estimated_Value'],
            ascending=[False, False]
        )
        
        # Select relevant columns for output
        output_columns = [
            'Name', 'Email', 'Phone', 'Lead_Status', 'Property_Size',
            'Estimated_Value', 'State', 'Lead_Source', 'Priority'
        ]
        worthwhile = worthwhile[output_columns]
        
        # Save to output CSV
        logging.info(f"Saving {len(worthwhile)} worthwhile leads to {output_csv}")
        worthwhile.to_csv(output_csv, index=False)
        
        # Log summary
        logging.info("Summary:")
        logging.info(f"Total leads processed: {len(df)}")
        logging.info(f"High-priority leads (TX): {len(worthwhile[worthwhile['Priority'] == 'High'])}")
        logging.info(f"Standard-priority leads: {len(worthwhile[worthwhile['Priority'] == 'Standard'])}")
    
    except KeyError as e:
        logging.error(f"Missing column {e}. Ensure CSV includes: Name, Email, Phone, "
                     "Lead_Status, Property_Size, Estimated_Value, State, Lead_Source")
        sys.exit(1)
    except FileNotFoundError:
        logging.error(f"Input file {input_csv} not found.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error("Usage: python sift_sobel_leads.py path_to_exported_leads.csv")
        sys.exit(1)
    sift_sobel_leads(sys.argv[1])