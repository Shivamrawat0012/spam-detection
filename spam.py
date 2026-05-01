import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import base64
import os


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Spam Detector", page_icon="🛡️", layout="wide")

# ─────────────────────────────────────────────
# GOOGLE OAUTH SETTINGS
# (Yahan apna Client ID aur Secret daalo)
# ─────────────────────────────────────────────


CLIENT_ID     = st.secrets["GOOGLE_CLIENT_ID"]
CLIENT_SECRET = st.secrets["GOOGLE_CLIENT_SECRET"]
REDIRECT_URI  = "https://spam-detection-shivamrawat.streamlit.app/"
SCOPES        = ["https://www.googleapis.com/auth/gmail.readonly"]

# ─────────────────────────────────────────────
# TRAIN MODEL (sirf ek baar hoga — cached)
# ─────────────────────────────────────────────
@st.cache_resource
def train_model():
    data = pd.read_csv("spam.csv", encoding="latin-1")
    data.drop_duplicates(inplace=True)
    data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

    mess = data['Message']
    cat  = data['Category']

    mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2, random_state=42)

    cv = CountVectorizer(stop_words='english')
    features = cv.fit_transform(mess_train)

    model = MultinomialNB()
    model.fit(features, cat_train)

    return cv, model

cv, model = train_model()

# ─────────────────────────────────────────────
# SPAM PREDICT FUNCTION (tera original function)
# ─────────────────────────────────────────────
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result[0]  # "Spam" ya "Not Spam"

# ─────────────────────────────────────────────
# GMAIL OAUTH — LOGIN URL BANAO
# ─────────────────────────────────────────────
def get_auth_url():
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [REDIRECT_URI],
            }
        },
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    st.session_state["oauth_state"] = state
    st.session_state["flow"] = flow
    return auth_url

# ─────────────────────────────────────────────
# GMAIL OAUTH — TOKEN EXCHANGE (callback)
# ─────────────────────────────────────────────
def exchange_code_for_token(code):
    flow = st.session_state.get("flow")
    if not flow:
        return None
    flow.fetch_token(code=code)
    creds = flow.credentials
    return creds

# ─────────────────────────────────────────────
# GMAIL — EMAILS FETCH KARO
# ─────────────────────────────────────────────
def fetch_emails(creds, max_results=10):
    service = build("gmail", "v1", credentials=creds)

    results = service.users().messages().list(
        userId="me", maxResults=max_results, labelIds=["INBOX"]
    ).execute()

    messages = results.get("messages", [])
    emails = []

    for msg in messages:
        msg_detail = service.users().messages().get(
            userId="me", id=msg["id"], format="metadata",
            metadataHeaders=["Subject", "From"]
        ).execute()

        headers = msg_detail.get("payload", {}).get("headers", [])
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(No Subject)")
        sender  = next((h["value"] for h in headers if h["name"] == "From"), "Unknown")
        snippet = msg_detail.get("snippet", "")

        emails.append({
            "id":      msg["id"],
            "subject": subject,
            "from":    sender,
            "snippet": snippet,
        })

    return emails

# ─────────────────────────────────────────────
# UI — STREAMLIT APP
# ─────────────────────────────────────────────
st.title("🛡️ Spam Detection")

# ── Tabs: Manual | Gmail ──
tab1, tab2 = st.tabs(["✍️ Manual Check", "📬 Gmail Check"])

# ═══════════════════════════════════════
# TAB 1 — MANUAL (tera original feature)
# ═══════════════════════════════════════
with tab1:
    st.subheader("Enter a message manually")
    input_mess = st.text_input("Enter the message")
    if st.button("Validate"):
        if input_mess.strip():
            output = predict(input_mess)
            if output == "Spam":
                st.error(f"🚨 Result: **{output}**")
            else:
                st.success(f"✅ Result: **{output}**")
        else:
            st.warning("Pehle koi message likhna bhai!")

# ═══════════════════════════════════════
# TAB 2 — GMAIL
# ═══════════════════════════════════════
with tab2:
    st.subheader("Connect your Gmail account")

    # ── Step 1: URL se auth code pakdo ──
    query_params = st.query_params
    auth_code = query_params.get("code", None)

    # ── Step 2: Agar code aaya toh token lo ──
    if auth_code and "credentials" not in st.session_state:
        with st.spinner("Authenticating with Google..."):
            try:
                creds = exchange_code_for_token(auth_code)
                if creds:
                    st.session_state["credentials"] = {
                        "token":         creds.token,
                        "refresh_token": creds.refresh_token,
                        "token_uri":     creds.token_uri,
                        "client_id":     creds.client_id,
                        "client_secret": creds.client_secret,
                        "scopes":        creds.scopes,
                    }
                    st.query_params.clear()
                    st.success("✅ Gmail connected successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {e}")

    # ── Agar logged in hai ──
    if "credentials" in st.session_state:
        creds = Credentials(**st.session_state["credentials"])

        col1, col2 = st.columns([3, 1])
        with col1:
            num_emails = st.slider("Kitni emails check karni hain?", 5, 30, 10)
        with col2:
            if st.button("🔓 Disconnect Gmail"):
                del st.session_state["credentials"]
                st.rerun()

        if st.button("🔍 Fetch & Scan Emails"):
            with st.spinner("Gmail se emails la raha hu..."):
                try:
                    emails = fetch_emails(creds, max_results=num_emails)
                    st.session_state["emails"] = emails
                except Exception as e:
                    st.error(f"Gmail fetch failed: {e}")

        # ── Results dikhao ──
        if "emails" in st.session_state:
            emails = st.session_state["emails"]
            spam_count     = 0
            not_spam_count = 0

            st.markdown("---")
            st.subheader(f"📋 {len(emails)} Emails Scanned")

            for email in emails:
                text_to_check = f"{email['subject']} {email['snippet']}"
                result = predict(text_to_check)

                if result == "Spam":
                    spam_count += 1
                    with st.expander(f"🚨 SPAM — {email['subject'][:60]}"):
                        st.write(f"**From:** {email['from']}")
                        st.write(f"**Preview:** {email['snippet'][:200]}")
                        st.error("Result: **Spam**")
                else:
                    not_spam_count += 1
                    with st.expander(f"✅ Safe — {email['subject'][:60]}"):
                        st.write(f"**From:** {email['from']}")
                        st.write(f"**Preview:** {email['snippet'][:200]}")
                        st.success("Result: **Not Spam**")

            # ── Summary ──
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Emails", len(emails))
            c2.metric("🚨 Spam",     spam_count)
            c3.metric("✅ Not Spam", not_spam_count)

    # ── Agar logged out hai ──
    else:
        st.info("Apna Gmail connect karo taaki emails scan ho sakein.")
        if st.button("🔗 Connect Gmail"):
            auth_url = get_auth_url()
            st.markdown(f"[👉 Click here to connect Gmail]({auth_url})")
            st.caption("Link pe click karo → Google account select karo → Permission do → Wapas aa jaoge")
