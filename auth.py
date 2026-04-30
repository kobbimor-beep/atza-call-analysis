"""
Authentication module — bcrypt passwords stored in auth_config.yaml.
"""
import os
import streamlit as st

AUTH_CONFIG = os.path.join(os.path.dirname(__file__), "auth_config.yaml")
_SESSION_KEY = "atza_auth_v1"


def _load_users() -> dict:
    # On Streamlit Cloud — read flat secrets: AUTH_USER_kobi_password, etc.
    try:
        secrets = st.secrets
        # Support flat format: AUTH_USER_<username>_password / name / role
        users = {}
        for key in secrets:
            if key.startswith("AUTH_USER_"):
                parts = key.split("_", 3)  # AUTH, USER, <username>, <field>
                if len(parts) == 4:
                    _, _, username, field = parts
                    if username not in users:
                        users[username] = {}
                    users[username][field] = secrets[key]
        if users:
            return users
        # Legacy nested format: [users.kobi]
        if "users" in secrets:
            result = {}
            for uname in secrets["users"]:
                result[uname] = dict(secrets["users"][uname])
            return result
    except Exception:
        pass
    # Local dev — read from auth_config.yaml
    if not os.path.exists(AUTH_CONFIG):
        return {}
    import yaml
    with open(AUTH_CONFIG, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("users", {})


def _check_password(username: str, password: str) -> bool:
    import bcrypt
    users = _load_users()
    user  = users.get(username)
    if not user:
        return False
    try:
        return bcrypt.checkpw(password.encode(), user["password"].encode())
    except Exception:
        return False


def require_login() -> bool:
    """
    Show login screen if not authenticated.
    Returns True if user is logged in and app can proceed.
    """
    if st.session_state.get(_SESSION_KEY):
        return True

    if not _load_users():
        st.error("⚠️ לא נמצאו משתמשים. הגדר [users] ב-Secrets או הרץ `python setup_auth.py` מקומית.")
        st.stop()

    st.markdown("""
    <style>
    .stApp { background: #F5F5F5; }
    .login-wrap {
        max-width: 400px; margin: 5rem auto; background: #fff;
        padding: 2.5rem 2rem; border-radius: 16px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.10);
        border-top: 5px solid #E31C3D;
        direction: rtl; text-align: right;
        font-family: 'Heebo', sans-serif;
    }
    .login-title { font-size: 2rem; font-weight: 900; color: #E31C3D; margin-bottom: 0.2rem; }
    .login-sub   { font-size: 0.95rem; color: #666; margin-bottom: 1.5rem; }
    [data-testid="InputInstructions"] { display: none !important; }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Heebo:wght@400;700;900&display=swap" rel="stylesheet">
    <div class="login-wrap">
        <div class="login-title">ATZA</div>
        <div class="login-sub">מערכת ניתוח שיחות — כניסה למנהלים</div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([3, 2, 3])
    with col:
        with st.form("login_form"):
            username = st.text_input("שם משתמש")
            password = st.text_input("סיסמה", type="password")
            submitted = st.form_submit_button("כניסה", use_container_width=True)

    if submitted:
        if _check_password(username, password):
            users = _load_users()
            st.session_state[_SESSION_KEY]   = True
            st.session_state["auth_user"]    = username
            st.session_state["auth_name"]    = users.get(username, {}).get("name", username)
            st.session_state["auth_role"]    = users.get(username, {}).get("role", "viewer")
            st.rerun()
        else:
            st.error("שם משתמש או סיסמה שגויים")

    return False


def logout():
    for k in (_SESSION_KEY, "auth_user", "auth_name", "auth_role"):
        st.session_state.pop(k, None)
    st.rerun()


def current_user_name() -> str:
    return st.session_state.get("auth_name", "")


def current_user_role() -> str:
    return st.session_state.get("auth_role", "viewer")
