import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
from google.oauth2.credentials import Credentials
import requests

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Spam Detector", page_icon="🛡️", layout="wide")

# ─────────────────────────────────────────────
# GOOGLE OAUTH SETTINGS
# ─────────────────────────────────────────────
CLIENT_ID     = st.secrets["GOOGLE_CLIENT_ID"]
CLIENT_SECRET = st.secrets["GOOGLE_CLIENT_SECRET"]
REDIRECT_URI  = "https://spam-detection-shivamrawat.streamlit.app/"
SCOPES        = "https://www.googleapis.com/auth/gmail.readonly"

# ─────────────────────────────────────────────
# TRAIN MODEL
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
# SPAM PREDICT FUNCTION
# ─────────────────────────────────────────────
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result[0]

# ─────────────────────────────────────────────
# OAUTH — LOGIN URL
# ─────────────────────────────────────────────
def get_auth_url():
    url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={SCOPES}"
        f"&access_type=offline"
        f"&prompt=consent"
    )
    return url

# ─────────────────────────────────────────────
# OAUTH — CODE SE TOKEN LO
# ─────────────────────────────────────────────
def exchange_code_for_token(code):
    response = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code":          code,
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "redirect_uri":  REDIRECT_URI,
            "grant_type":    "authorization_code",
        }
    )
    return response.json()

# ─────────────────────────────────────────────
# GMAIL — EMAILS FETCH
# ─────────────────────────────────────────────
def fetch_emails(access_token, max_results=10):
    headers = {"Authorization": f"Bearer {access_token}"}

    list_res = requests.get(
        f"https://gmail.googleapis.com/gmail/v1/users/me/messages?maxResults={max_results}&labelIds=INBOX",
        headers=headers
    ).json()

    messages = list_res.get("messages", [])
    emails = []

    for msg in messages:
        detail = requests.get(
            f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg['id']}?format=metadata&metadataHeaders=Subject&metadataHeaders=From",
            headers=headers
        ).json()

        hdrs    = detail.get("payload", {}).get("headers", [])
        subject = next((h["value"] for h in hdrs if h["name"] == "Subject"), "(No Subject)")
        sender  = next((h["value"] for h in hdrs if h["name"] == "From"), "Unknown")
        snippet = detail.get("snippet", "")

        emails.append({
            "subject": subject,
            "from":    sender,
            "snippet": snippet,
        })

    return emails

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🛡️ Spam Detection")

# ══════════════════════════════════════════════
# OAUTH CALLBACK — TABS SE PEHLE HANDLE KARO
# (Yahi tha asli problem — tab ke andar tha)
# ══════════════════════════════════════════════
auth_code = st.query_params.get("code", None)

if auth_code and "access_token" not in st.session_state:
    with st.spinner("Gmail se connect ho raha hu..."):
        token_data = exchange_code_for_token(auth_code)
        if "access_token" in token_data:
            st.session_state["access_token"] = token_data["access_token"]
            st.query_params.clear()
            st.rerun()
        else:
            st.error(f"Token error: {token_data}")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["✍️ Manual Check", "📬 Gmail Check"])

# ══════════════════════════════════════════════
# TAB 1 — MANUAL
# ══════════════════════════════════════════════
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

# ══════════════════════════════════════════════
# TAB 2 — GMAIL
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Connect your Gmail account")

    # Logged in hai
    if "access_token" in st.session_state:
        col1, col2 = st.columns([3, 1])
        with col1:
            num_emails = st.slider("Kitni emails scan karni hain?", 5, 30, 10)
        with col2:
            if st.button("🔓 Disconnect Gmail"):
                del st.session_state["access_token"]
                if "emails" in st.session_state:
                    del st.session_state["emails"]
                st.rerun()

        if st.button("🔍 Fetch & Scan Emails"):
            with st.spinner("Emails la raha hu..."):
                try:
                    emails = fetch_emails(
                        st.session_state["access_token"],
                        max_results=num_emails
                    )
                    st.session_state["emails"] = emails
                except Exception as e:
                    st.error(f"Gmail fetch failed: {e}")

        # Results dikhao
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

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Emails", len(emails))
            c2.metric("🚨 Spam", spam_count)
            c3.metric("✅ Not Spam", not_spam_count)

    # Logged out hai
    else:
        st.info("Apna Gmail connect karo taaki emails scan ho sakein.")
        if st.button("🔗 Connect Gmail"):
            auth_url = get_auth_url()
            st.markdown(f"### [👉 Click here to connect Gmail]({auth_url})")
            st.caption("Link pe click karo → Google account select karo → Permission do → Wapas aa jaoge")
