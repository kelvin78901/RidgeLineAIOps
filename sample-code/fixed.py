"""
fixed.py — Secure version of vulnerable.py.
Each fix is labeled with the corresponding vulnerability number.
"""

from flask import Flask, request, jsonify, session, escape
import sqlite3
import os

app = Flask(__name__)
app.secret_key = os.urandom(32)

# FIX 1: Environment variables instead of hardcoded keys.
# The vulnerable version had fake-but-Stripe-shaped placeholders inline; here
# they are loaded from the environment so they never live in source control.
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY")
SENDGRID_KEY = os.environ.get("SENDGRID_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")


# FIX 2: Parameterized queries prevent SQL injection
@app.route("/api/customers", methods=["GET"])
def get_customer():
    customer_id = request.args.get("id")
    conn = sqlite3.connect("customers.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM customers WHERE id = ?", (customer_id,))
    result = cursor.fetchone()
    conn.close()
    return jsonify(result)


# FIX 3: Pull user_id from session, not request body
@app.route("/api/refunds", methods=["POST"])
def process_refund():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "unauthorized"}), 401

    data = request.json
    order_id = data["order_id"]
    amount = data["amount"]

    conn = sqlite3.connect("orders.db")
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE orders SET refunded = 1, refund_amount = ? WHERE id = ? AND user_id = ?",
        (amount, order_id, user_id)
    )
    if cursor.rowcount == 0:
        conn.close()
        return jsonify({"error": "order not found or not yours"}), 404
    conn.commit()
    conn.close()
    return jsonify({"status": "refunded", "amount": amount})


# FIX 4: Escape user input to prevent XSS
@app.route("/welcome")
def welcome():
    username = escape(request.args.get("name", "Guest"))
    return f"""
    <html>
    <body>
        <h1>Welcome, {username}!</h1>
        <p>Your dashboard is ready.</p>
    </body>
    </html>
    """


# FIX 5: Ownership check on ticket access
@app.route("/api/tickets/<ticket_id>", methods=["GET"])
def get_ticket(ticket_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "unauthorized"}), 401

    conn = sqlite3.connect("tickets.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM tickets WHERE id = ? AND assigned_to = ?",
        (ticket_id, user_id)
    )
    ticket = cursor.fetchone()
    conn.close()
    if ticket:
        return jsonify({"ticket": ticket})
    return jsonify({"error": "not found"}), 404


if __name__ == "__main__":
    # FIX 6: Debug mode off in production
    app.run(host="0.0.0.0", port=5000, debug=False)
