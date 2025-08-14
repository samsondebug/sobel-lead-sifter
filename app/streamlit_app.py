import sys
import io
import smtplib
from email.message import EmailMessage
import pandas as pd
import streamlit as st
from typing import List, Tuple, Dict


def filter_worthwhile_leads(
    leads_df: pd.DataFrame,
    min_acres: float = 5.0,
    min_value_usd: float = 50_000.0,
    excluded_statuses: List[str] = None,
    priority_states: List[str] = None,
) -> Tuple[pd.DataFrame, int, int, int]:
    """
    Apply the same baseline rules as sift_sobel_leads.py to a DataFrame and
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


def compute_scores(
    df: pd.DataFrame,
    priority_states: List[str],
    status_map: Dict[str, int],
    source_map: Dict[str, int],
    thresh_value: float,
    thresh_value_per_acre: float,
    w_status: float,
    w_contact: float,
    w_value: float,
    w_vpa: float,
    w_state: float,
    w_source: float,
) -> pd.DataFrame:
    """
    Compute granular metrics and a composite score (0-100) for each lead.
    """
    scored = df.copy()

    # Derived metrics
    scored["Has_Email"] = scored["Email"].notna()
    scored["Has_Phone"] = scored["Phone"].notna()
    scored["Contact_Score"] = scored["Has_Email"].astype(int) + scored["Has_Phone"].astype(int)
    scored["Status_Score"] = scored["Lead_Status"].map(status_map).fillna(0).astype(int)
    scored["Source_Score"] = scored["Lead_Source"].map(source_map).fillna(50).astype(int)

    # Avoid division by zero
    scored["Value_Per_Acre"] = scored.apply(
        lambda r: float(r["Estimated_Value"]) / float(r["Property_Size"]) if float(r["Property_Size"]) > 0 else 0.0,
        axis=1,
    )

    # Normalize subscores to 0..100
    contact_norm = scored["Contact_Score"].clip(lower=0, upper=2) / 2 * 100
    status_norm = scored["Status_Score"].clip(lower=0, upper=3) / 3 * 100
    value_norm = (scored["Estimated_Value"].astype(float) / max(thresh_value, 1)).clip(upper=1.0) * 100
    vpa_norm = (scored["Value_Per_Acre"].astype(float) / max(thresh_value_per_acre, 1)).clip(upper=1.0) * 100
    state_norm = scored["State"].apply(lambda s: 100.0 if s in priority_states else 0.0)
    source_norm = scored["Source_Score"].clip(lower=0, upper=100)

    weights = [w_status, w_contact, w_value, w_vpa, w_state, w_source]
    total_w = sum(weights) if sum(weights) > 0 else 1.0

    scored["Score"] = (
        status_norm * w_status
        + contact_norm * w_contact
        + value_norm * w_value
        + vpa_norm * w_vpa
        + state_norm * w_state
        + source_norm * w_source
    ) / total_w

    return scored


st.set_page_config(page_title="Sobel Lead Sifter", layout="wide")
st.title("Sobel Lead Sifter")
st.write(
    "Upload your Pebble CRM export CSV, tune prioritization, and optionally send outreach."
)

with st.sidebar:
    st.header("Baseline Filters")
    apply_baseline = st.checkbox("Apply baseline filters (recommended)", value=True)
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

    st.header("Scoring Weights")
    w_status = st.slider("Weight: Status", 0.0, 100.0, 30.0, 1.0)
    w_contact = st.slider("Weight: Contactability", 0.0, 100.0, 20.0, 1.0)
    w_value = st.slider("Weight: Estimated Value", 0.0, 100.0, 25.0, 1.0)
    w_vpa = st.slider("Weight: Value per Acre", 0.0, 100.0, 10.0, 1.0)
    w_state = st.slider("Weight: State Priority", 0.0, 100.0, 10.0, 1.0)
    w_source = st.slider("Weight: Lead Source Quality", 0.0, 100.0, 5.0, 1.0)

    st.header("Scoring Thresholds")
    thresh_value = st.number_input("Top-out at Estimated Value (USD)", value=200_000.0, step=10_000.0)
    thresh_vpa = st.number_input("Top-out at Value per Acre (USD/acre)", value=20_000.0, step=1_000.0)
    score_threshold = st.slider("Overall Score threshold (0-100)", 0.0, 100.0, 60.0, 1.0)

uploaded_file = st.file_uploader("Upload leads.csv", type=["csv"])

# Default mappings
status_map_default = {"New": 1, "Warm": 2, "Hot": 3}
source_map_default = {
    "Referral": 100,
    "Website": 80,
    "Partner": 70,
    "Direct Mail": 60,
    "Email Campaign": 40,
}

with st.expander("Advanced: Source quality scores", expanded=False):
    st.write("Adjust perceived quality (0-100) per source.")
    source_map = {}
    for src, default in source_map_default.items():
        source_map[src] = st.slider(f"{src}", 0, 100, default, 5)

if uploaded_file is not None:
    try:
        leads_df = pd.read_csv(uploaded_file)
        st.success("CSV loaded.")
        with st.expander("Columns detected", expanded=False):
            st.write(list(leads_df.columns))

        if apply_baseline:
            baseline_df, original_count, high_count, standard_count = filter_worthwhile_leads(
                leads_df=leads_df,
                min_acres=min_acres,
                min_value_usd=min_value_usd,
                excluded_statuses=excluded_statuses,
                priority_states=priority_states,
            )
        else:
            # No baseline filter; keep all rows
            baseline_df = leads_df.copy()
            original_count = len(leads_df)
            high_count = 0
            standard_count = 0

        # Compute granular metrics and score
        scored_df = compute_scores(
            df=baseline_df,
            priority_states=priority_states,
            status_map=status_map_default,
            source_map=source_map,
            thresh_value=thresh_value,
            thresh_value_per_acre=thresh_vpa,
            w_status=w_status,
            w_contact=w_contact,
            w_value=w_value,
            w_vpa=w_vpa,
            w_state=w_state,
            w_source=w_source,
        )

        worthwhile_df = scored_df[scored_df["Score"] >= score_threshold].copy()
        worthwhile_df = worthwhile_df.sort_values(by=["Score", "Estimated_Value"], ascending=[False, False])

        st.subheader("Summary")
        st.write(
            f"Total leads processed: {original_count} | Worthwhile (score ≥ {int(score_threshold)}): {len(worthwhile_df)}"
        )

        st.subheader("Filtered Leads (with scores)")
        show_cols = [
            "Name", "Email", "Phone", "Lead_Status", "Property_Size", "Estimated_Value",
            "State", "Lead_Source", "Value_Per_Acre", "Score"
        ]
        # Add missing safely
        show_cols = [c for c in show_cols if c in worthwhile_df.columns]
        st.dataframe(worthwhile_df[show_cols], use_container_width=True)

        csv_buffer = io.StringIO()
        worthwhile_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download filtered CSV (with scores)",
            data=csv_buffer.getvalue().encode("utf-8"),
            file_name="sobel_worthwhile_leads.csv",
            mime="text/csv",
        )

        st.divider()
        st.subheader("AutoReach: Email Outreach")
        st.caption("Configure SMTP to send personalized emails to leads with an email address.")
        col1, col2 = st.columns(2)
        with col1:
            smtp_host = st.text_input("SMTP host", value="smtp.gmail.com")
            smtp_port = st.number_input("SMTP port", value=587, step=1)
            use_tls = st.checkbox("Use STARTTLS", value=True)
            smtp_user = st.text_input("SMTP username (email)")
            smtp_pass = st.text_input("SMTP password/app password", type="password")
            from_email = st.text_input("From email", value="")
        with col2:
            email_subject = st.text_input("Subject", value="Sobel Properties | Interest in your land in {State}")
            email_body = st.text_area(
                "Body (use placeholders like {Name}, {State}, {Estimated_Value}, {Property_Size})",
                height=200,
                value=(
                    "Hi {Name},\n\n"
                    "I’m reaching out from Sobel Property Investments. Based on public records, "
                    "your property in {State} might be a good fit for a quick, fair cash offer.\n\n"
                    "Estimated value: ${Estimated_Value} | Size: {Property_Size} acres.\n\n"
                    "If you’re open to a conversation, reply here and we can share a no-obligation offer.\n\n"
                    "Best,\nSobel Property Investments"
                ),
            )
            preview_count = st.slider("Preview first N emails", 1, 20, 5, 1)

        leads_with_email = worthwhile_df[worthwhile_df["Email"].notna()].copy()
        st.write(f"Leads with email addresses: {len(leads_with_email)}")

        # Preview table of rendered messages
        def render_template(row: pd.Series) -> Dict[str, str]:
            try:
                subject = email_subject.format(**row.to_dict())
            except Exception:
                subject = email_subject
            try:
                body = email_body.format(**row.to_dict())
            except Exception:
                body = email_body
            return {"To": row.get("Email", ""), "Subject": subject, "Body": body}

        if len(leads_with_email) > 0:
            preview_rows = [render_template(r) for _, r in leads_with_email.head(preview_count).iterrows()]
            st.write("Preview:")
            st.dataframe(pd.DataFrame(preview_rows))

            drafts_csv = io.StringIO()
            pd.DataFrame([render_template(r) for _, r in leads_with_email.iterrows()]).to_csv(drafts_csv, index=False)
            st.download_button(
                label="Download outreach drafts (CSV)",
                data=drafts_csv.getvalue().encode("utf-8"),
                file_name="outreach_emails.csv",
                mime="text/csv",
            )

            do_send = st.checkbox("I confirm I want to send these emails now", value=False)
            send_btn = st.button("Send emails via SMTP")

            if send_btn:
                if not do_send:
                    st.warning("Please confirm sending by checking the box.")
                elif not (smtp_host and smtp_port and smtp_user and smtp_pass and from_email):
                    st.error("SMTP settings are incomplete.")
                else:
                    try:
                        if use_tls:
                            smtp = smtplib.SMTP(smtp_host, int(smtp_port))
                            smtp.ehlo()
                            smtp.starttls()
                            smtp.ehlo()
                        else:
                            smtp = smtplib.SMTP(smtp_host, int(smtp_port))
                        smtp.login(smtp_user, smtp_pass)

                        sent = 0
                        failures = []
                        for _, row in leads_with_email.iterrows():
                            rendered = render_template(row)
                            to_addr = rendered["To"]
                            if not to_addr:
                                continue
                            msg = EmailMessage()
                            msg["Subject"] = rendered["Subject"]
                            msg["From"] = from_email
                            msg["To"] = to_addr
                            msg.set_content(rendered["Body"]) 
                            try:
                                smtp.send_message(msg)
                                sent += 1
                            except Exception as send_err:
                                failures.append((to_addr, str(send_err)))
                        try:
                            smtp.quit()
                        except Exception:
                            pass

                        st.success(f"Sent {sent} emails. Failures: {len(failures)}")
                        if failures:
                            with st.expander("View failures"):
                                st.write(pd.DataFrame(failures, columns=["Email", "Error"]))
                    except Exception as e:
                        st.exception(e)
        else:
            st.info("No leads with email addresses to contact.")

    except KeyError as ke:
        st.error(str(ke))
    except Exception as e:
        st.exception(e)
else:
    st.info("Upload a CSV to begin.")
