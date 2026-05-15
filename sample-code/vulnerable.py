"""
vulnerable.py — Intentionally vulnerable code for security demo.
DO NOT use in production. These are examples of what "vibe coding" produces
when non-developers use AI to generate code without security review.

Each vulnerability is labeled with a comment for reference.
"""

from flask import Flask, request, jsonify
import sqlite3
import os

app = Flask(__name__)

# VULNERABILITY 1: Hardcoded API key
# (Values below are intentionally fake placeholders shaped like the real format,
# kept inline so static scanners like Semgrep still fire on the *pattern*.)
STRIPE_SECRET_KEY = "sk_live_" + "FAKE_DEMO_KEY_NEVER_USE_IN_PRODUCTION_xxx"
SENDGRID_KEY = "SG." + "FAKE_DEMO_KEY_NEVER_USE_IN_PRODUCTION"
DATABASE_URL = "postgresql://admin:" + "FAKE_DEMO_PASSWORD" + "@prod-db.internal:5432/ridgeline"


# VULNERABILITY 2: SQL Injection
@app.route("/api/customers", methods=["GET"])
def get_customer():
    customer_id = request.args.get("id")
    conn = sqlite3.connect("customers.db")
    cursor = conn.cursor()
    # Direct string interpolation — classic SQL injection
    cursor.execute(f"SELECT * FROM customers WHERE id = '{customer_id}'")
    result = cursor.fetchone()
    conn.close()
    return jsonify(result)


# VULNERABILITY 3: Auth bypass (IDOR) — the exact pattern from PPT Slide 13
@app.route("/api/refunds", methods=["POST"])
def process_refund():
    data = request.json
    # PROBLEM: trusts user_id from request body instead of session
    user_id = data["user_id"]
    order_id = data["order_id"]
    amount = data["amount"]

    conn = sqlite3.connect("orders.db")
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE orders SET refunded = 1, refund_amount = ? WHERE id = ? AND user_id = ?",
        (amount, order_id, user_id)
    )
    conn.commit()
    conn.close()
    return jsonify({"status": "refunded", "amount": amount})


# VULNERABILITY 4: XSS — unsanitized user input in HTML
@app.route("/welcome")
def welcome():
    username = request.args.get("name", "Guest")
    # Direct injection of user input into HTML
    return f"""
    <html>
    <body>
        <h1>Welcome, {username}!</h1>
        <p>Your dashboard is ready.</p>
    </body>
    </html>
    """


# VULNERABILITY 5: Insecure Direct Object Reference (no ownership check)
@app.route("/api/tickets/<ticket_id>", methods=["GET"])
def get_ticket(ticket_id):
    conn = sqlite3.connect("tickets.db")
    cursor = conn.cursor()
    # No check that the requesting user owns this ticket
    cursor.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
    ticket = cursor.fetchone()
    conn.close()
    if ticket:
        return jsonify({"ticket": ticket})
    return jsonify({"error": "not found"}), 404


if __name__ == "__main__":
    # VULNERABILITY 6: Debug mode in production
    app.run(host="0.0.0.0", port=5000, debug=True)
