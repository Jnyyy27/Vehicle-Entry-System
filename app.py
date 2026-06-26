import os
import pymysql
from dotenv import load_dotenv
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
from flask import Flask, request, redirect, render_template

load_dotenv()

app = Flask(__name__)

COGNITO_DOMAIN = os.getenv("COGNITO_DOMAIN")
CLIENT_ID = os.getenv("CLIENT_ID")

@app.route("/")
def home():
    return '<a href="/login">Login with Cognito</a>'

@app.route("/login")
def login():
    return redirect(
        f"{COGNITO_DOMAIN}/login?response_type=code"
        f"&client_id={CLIENT_ID}"
        f"&redirect_uri=http://localhost:5000/callback"
        f"&scope=email+openid+phone"
    )

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/vehicles", methods=["GET", "POST"])
def vehicles():

    if request.method == "POST":

        student_name = request.form["student_name"]
        student_id = request.form["student_id"]
        plate_number = request.form["plate_number"]
        vehicle_type = request.form["vehicle_type"]

        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )

        cursor = conn.cursor()

        sql = """
        INSERT INTO vehicles
        (student_name, student_id, plate_number, vehicle_type)
        VALUES (%s,%s,%s,%s)
        """

        cursor.execute(sql, (
            student_name,
            student_id,
            plate_number,
            vehicle_type
        ))

        conn.commit()

        cursor.close()
        conn.close()

        return "Vehicle Registered Successfully!"

    return render_template("vehicles.html")

@app.route("/callback")
def callback():
    code = request.args.get("code")

    print("Authorization Code:", code)

    return redirect("/dashboard")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)