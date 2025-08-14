import sys
import io
import pandas as pd
import streamlit as st
from typing import List, Tuple


def filter_worthwhile_leads(
    leads_df: pd.DataFrame,
    min_acres: float = 5.0,
    min_value_usd: float = 50_000.0,
    excluded_statuses: List[str] = None,
    priority_states: List[str] = None,
) -> Tuple[pd.DataFrame, int, int, int]:
    """
    Apply the same business rules as sift_sobel_leads.py to a DataFrame and
    return the filtered DataFrame along with summary counts.
    """
    if excluded_statuses is None:
        excluded_statuses = ["Cold", "Rejected"]
    if priority_states is None:
        priority_states = ["TX"]

    required_columns = [
        "Name", "Email", "Phone", "Lead_Status", "Property_Size",
        "Estimated_Value", "State", "Lead_Source"
    ]
    missing = [c for c in required_columns if c not in leads_df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    original_count = len(leads_df)

    worthwhile_df = leads_df[
        (leads_df["Email"].notna() | leads_df["Phone"].notna())
        & (~leads_df["Lead_Status"].isin(excluded_statuses))
        & (leads_df["Property_Size"] > float(min_acres))
        & (leads_df["Estimated_Value"] > float(min_value_usd))
    ].copy()

    worthwhile_df["Priority"] = worthwhile_df["State"].apply(
        lambda s: "High" if s in priority_states else "Standard"
    )

    worthwhile_df = worthwhile_df.sort_values(
        by=["Priority", "Estimated_Value"], ascending=[False, False]
    )

    output_columns = [
        "Name", "Email", "Phone", "Lead_Status", "Property_Size",
        "Estimated_Value", "State", "Lead_Source", "Priority"
    ]
    worthwhile_df = worthwhile_df[output_columns]

    high_count = int((worthwhile_df["Priority"] == "High").sum())
    standard_count = int((worthwhile_df["Priority"] == "Standard").sum())

    return worthwhile_df, original_count, high_count, standard_count


st.set_page_config(page_title="Sobel Lead Sifter", layout="wide")
st.title("Sobel Lead Sifter")
st.write(
    "Upload your Pebble CRM export CSV and apply lead prioritization."
)

with st.sidebar:
    st.header("Filters")
    min_acres = st.number_input("Minimum Property Size (acres)", value=5.0, step=0.5)
    min_value_usd = st.number_input("Minimum Estimated Value (USD)", value=50_000.0, step=5_000.0)
    excluded_statuses = st.multiselect(
        "Exclude Lead Status",
        options=["New", "Warm", "Hot", "Cold", "Rejected"],
        default=["Cold", "Rejected"],
    )
    priority_states = st.multiselect(
        "Priority States (High)",
        options=[
            "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
        ],
        default=["TX"],
    )

uploaded_file = st.file_uploader("Upload leads.csv", type=["csv"])

if uploaded_file is not None:
    try:
        leads_df = pd.read_csv(uploaded_file)
        st.success("CSV loaded.")
        with st.expander("Columns detected", expanded=False):
            st.write(list(leads_df.columns))

        worthwhile_df, original_count, high_count, standard_count = filter_worthwhile_leads(
            leads_df=leads_df,
            min_acres=min_acres,
            min_value_usd=min_value_usd,
            excluded_statuses=excluded_statuses,
            priority_states=priority_states,
        )

        st.subheader("Summary")
        st.write(
            f"Total leads processed: {original_count} | High-priority (TX or selected): {high_count} | Standard-priority: {standard_count}"
        )

        st.subheader("Filtered Leads")
        st.dataframe(worthwhile_df, use_container_width=True)

        csv_buffer = io.StringIO()
        worthwhile_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download filtered CSV",
            data=csv_buffer.getvalue().encode("utf-8"),
            file_name="sobel_worthwhile_leads.csv",
            mime="text/csv",
        )

        save_to_repo = st.checkbox("Also save sobel_worthwhile_leads.csv into project folder")
        if save_to_repo:
            try:
                worthwhile_df.to_csv("sobel_worthwhile_leads.csv", index=False)
                st.info("Saved sobel_worthwhile_leads.csv to project folder.")
            except Exception as save_err:
                st.warning(f"Could not save file locally: {save_err}")

    except KeyError as ke:
        st.error(str(ke))
    except Exception as e:
        st.exception(e)
else:
    st.info("Upload a CSV to begin.")
