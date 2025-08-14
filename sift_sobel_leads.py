#!/usr/bin/env python3
import pandas as pd
import sys
import logging
import argparse

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
        
        # Ensure we work on a copy to avoid SettingWithCopyWarning
        worthwhile = worthwhile.copy()
        
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

def compute_value_per_acre(row):
    try:
        size = float(row['Property_Size'])
        if size <= 0:
            return 0.0
        return float(row['Estimated_Value']) / size
    except Exception:
        return 0.0


def compute_score(row, priority_states, status_map, thresh_value, thresh_vpa,
                  w_status, w_contact, w_value, w_vpa, w_state, w_source):
    has_email = pd.notna(row.get('Email'))
    has_phone = pd.notna(row.get('Phone'))
    contact_score = int(has_email) + int(has_phone)  # 0..2
    contact_norm = contact_score / 2 * 100

    status_norm = (status_map.get(row.get('Lead_Status'), 0) / 3) * 100

    try:
        value_norm = min(float(row.get('Estimated_Value', 0)) / max(thresh_value, 1), 1.0) * 100
    except Exception:
        value_norm = 0.0

    vpa = row.get('Value_Per_Acre', 0) or 0
    try:
        vpa_norm = min(float(vpa) / max(thresh_vpa, 1), 1.0) * 100
    except Exception:
        vpa_norm = 0.0

    state_norm = 100.0 if row.get('State') in priority_states else 0.0

    # Simple source defaults
    source_map = {
        'Referral': 100,
        'Website': 80,
        'Partner': 70,
        'Direct Mail': 60,
        'Email Campaign': 40,
    }
    source_norm = min(max(source_map.get(row.get('Lead_Source'), 50), 0), 100)

    total_w = sum([w_status, w_contact, w_value, w_vpa, w_state, w_source]) or 1.0
    score = (
        status_norm * w_status + contact_norm * w_contact + value_norm * w_value +
        vpa_norm * w_vpa + state_norm * w_state + source_norm * w_source
    ) / total_w
    return score


def parse_args():
    parser = argparse.ArgumentParser(description='Sift and score Sobel leads')
    parser.add_argument('input_csv', help='Path to the input CSV file')
    parser.add_argument('--output', default='sobel_worthwhile_leads.csv', help='Output CSV path')
    parser.add_argument('--min-acres', type=float, default=5.0)
    parser.add_argument('--min-value', type=float, default=50000.0)
    parser.add_argument('--score-threshold', type=float, default=60.0)
    parser.add_argument('--thresh-value', type=float, default=200000.0)
    parser.add_argument('--thresh-vpa', type=float, default=20000.0)
    parser.add_argument('--priority-states', nargs='*', default=['TX'])
    parser.add_argument('--exclude-statuses', nargs='*', default=['Cold', 'Rejected'])
    parser.add_argument('--weights', nargs=6, type=float, metavar=('W_STATUS','W_CONTACT','W_VALUE','W_VPA','W_STATE','W_SOURCE'),
                        default=[30.0, 20.0, 25.0, 10.0, 10.0, 5.0],
                        help='Weights for [status, contact, value, value-per-acre, state, source]')
    return parser.parse_args()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error("Usage: python sift_sobel_leads.py path_to_exported_leads.csv")
        sys.exit(1)
    # Backward-compatible: if user passes only CSV, run original behavior
    # Otherwise, allow advanced scoring via flags
    try:
        args = parse_args()
        input_csv = args.input_csv
        output_csv = args.output
        priority_states = args.priority_states
        exclude_statuses = args.exclude_statuses
        (w_status, w_contact, w_value, w_vpa, w_state, w_source) = args.weights

        logging.info(f"Reading input CSV: {input_csv}")
        df = pd.read_csv(input_csv)
        logging.info(f"Available columns in CSV: {df.columns.tolist()}")

        # Baseline worthwhile filter
        worthwhile = df[
            (df['Email'].notna() | df['Phone'].notna()) &
            (~df['Lead_Status'].isin(exclude_statuses)) &
            (df['Property_Size'] > args.min_acres) &
            (df['Estimated_Value'] > args.min_value)
        ].copy()

        worthwhile['Value_Per_Acre'] = worthwhile.apply(compute_value_per_acre, axis=1)
        worthwhile['Priority'] = worthwhile['State'].apply(lambda x: 'High' if x in priority_states else 'Standard')

        status_map = { 'New': 1, 'Warm': 2, 'Hot': 3 }
        worthwhile['Score'] = worthwhile.apply(
            compute_score,
            axis=1,
            priority_states=priority_states,
            status_map=status_map,
            thresh_value=args.thresh_value,
            thresh_vpa=args.thresh_vpa,
            w_status=w_status,
            w_contact=w_contact,
            w_value=w_value,
            w_vpa=w_vpa,
            w_state=w_state,
            w_source=w_source,
        )

        worthwhile = worthwhile[worthwhile['Score'] >= args.score_threshold]
        worthwhile = worthwhile.sort_values(by=['Score', 'Estimated_Value'], ascending=[False, False])

        output_columns = [
            'Name', 'Email', 'Phone', 'Lead_Status', 'Property_Size',
            'Estimated_Value', 'State', 'Lead_Source', 'Value_Per_Acre', 'Score', 'Priority'
        ]
        worthwhile = worthwhile[[c for c in output_columns if c in worthwhile.columns]]

        logging.info(f"Saving {len(worthwhile)} worthwhile leads to {output_csv}")
        worthwhile.to_csv(output_csv, index=False)

        logging.info("Summary:")
        logging.info(f"Total leads processed: {len(df)}")
        logging.info(f"Worthwhile by score ≥ {args.score_threshold}: {len(worthwhile)}")
        logging.info(f"High-priority leads (priority states): {len(worthwhile[worthwhile['Priority'] == 'High'])}")
        logging.info(f"Standard-priority leads: {len(worthwhile[worthwhile['Priority'] == 'Standard'])}")
    except SystemExit:
        # argparse already printed error/usage; fall back to original simple run if only CSV provided
        sift_sobel_leads(sys.argv[1])