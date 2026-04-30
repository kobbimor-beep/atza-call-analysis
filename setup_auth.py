"""
Run once to create auth_config.yaml with hashed manager passwords.
Usage: python setup_auth.py
"""
import os
import getpass
import bcrypt
import yaml

OUTPUT = os.path.join(os.path.dirname(__file__), "auth_config.yaml")


def main():
    existing_users = {}
    if os.path.exists(OUTPUT):
        with open(OUTPUT, encoding="utf-8") as f:
            existing_users = (yaml.safe_load(f) or {}).get("users", {})
        print(f"נמצאו {len(existing_users)} משתמשים קיימים.\n")

    print("=== הגדרת משתמשים — אצה ניתוח שיחות ===")
    print("לחץ Enter ריק לסיום הוספת משתמשים.\n")

    while True:
        username = input("שם משתמש (אנגלית/מספרים): ").strip()
        if not username:
            break
        full_name = input("שם מלא (יוצג במערכת): ").strip()
        password  = getpass.getpass("סיסמה: ")
        confirm   = getpass.getpass("אמת סיסמה: ")
        if password != confirm:
            print("❌ הסיסמאות אינן תואמות. נסה שוב.\n")
            continue
        role = input("תפקיד [admin / manager / viewer] (ברירת מחדל: manager): ").strip() or "manager"
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        existing_users[username] = {"name": full_name, "password": hashed, "role": role}
        print(f"✅ משתמש '{username}' ({full_name}) נוסף.\n")

    if not existing_users:
        print("לא נוספו משתמשים. יוצא.")
        return

    config = {"users": existing_users}
    with open(OUTPUT, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print(f"\n✅ {OUTPUT} נשמר עם {len(existing_users)} משתמשים.")
    print("הפעל את המערכת: python -m streamlit run app.py")


if __name__ == "__main__":
    main()
