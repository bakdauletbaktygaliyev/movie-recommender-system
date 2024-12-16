
from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
import recommender

app = Flask(__name__, template_folder='./templates')
app.secret_key = 'admin'

user_credentials = "./data/user_credentials.csv"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def registration():
    error_message = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm-password']
        
        validation_result = validate_password(password, confirm_password)
        
        if validation_result:
            session['username'] = username
            user_id = generate_unique_user_id()
            new_user = {
                "user_id": user_id,
                "username": username,
                "password": password
            }
            add_user_to_csv(new_user)
            return redirect(url_for('dashboard'))
        else:
            error_message = "Passwords must match. Please try again."
            
    return render_template('register.html', error_message=error_message)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error_message = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        validation_result = validate_username_password(username, password)
        
        if validation_result == 1:
            session['username'] = username
            return redirect(url_for('dashboard'))
        elif validation_result == -1:
            error_message = "Invalid password. Please try again."
        elif validation_result == 0:
            error_message = "Username not found. Please try again."
            
    return render_template('login.html', error_message=error_message)
    
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'username' not in session:
        return redirect(url_for("login"))
    username = session['username']
    return render_template("dashboard.html", username=username)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        # Extract form data (movie_name should be passed from the form input)
        movie_name = request.form.get('movie_name')
        try:
            # Call the recommendation function
            recommendations_json = recommender.item_based_collaborative_filtering(movie_name)
            recommendations = pd.read_json(recommendations_json, orient='records')
            
            return render_template(
                'recommendations.html',
                movie_name=movie_name,
                recommendations=recommendations.to_dict(orient='records')
            )
        except ValueError as e:
            print(e)
            return render_template('recommendations.html', error=str(e))
        except Exception as e:
            print(e)
            return render_template('recommendations.html', error="An unexpected error occurred.")
    else:
        return render_template('recommendations.html')


@app.route('/recommend/preferences', methods=['GET', 'POST'])
def recommend_preferences():
    if request.method == 'POST':
        user_id = get_user_id_by_username(session['username'])
        try:
            # Call the recommendation function
            recommendations_json = recommender.recommend_movies(user_id)
            recommendations = pd.read_json(recommendations_json, orient='records')
            
            return render_template(
                'recommendations-preferences.html',
                username=session['username'],
                recommendations=recommendations.to_dict(orient='records')
            )
        except ValueError as e:
            print(e)
            return render_template('recommendations-preferences.html', error=str(e))
        except Exception as e:
            print(e)
            return render_template('recommendations-preferences.html', error="An unexpected error occurred.")
    else:
        return render_template('recommendations-preferences.html')


def add_user_to_csv(new_user):
    try:
        with open(user_credentials, 'a') as f:
            f.write(f"{new_user['user_id']},{new_user['username']},{new_user['password']}\n")
        return True
    except Exception as e:
        return False
    
def generate_unique_user_id():
    try:
        user_data = pd.read_csv(user_credentials)
        existing_ids = set(user_data['user_id'])
    except FileNotFoundError:
        existing_ids = set()

    # Generate a new ID
    new_id = 1
    while new_id in existing_ids:
        new_id += 1
    
    return new_id

def validate_password(password, confirm_password):
    if password == confirm_password:
        return True
    return False

def validate_username_password(username, password): 
    df = pd.read_csv(user_credentials)
    
    # Create a dictionary of usernames and their corresponding passwords for easy lookup
    user_data = dict(zip(df['username'], df['password']))
    if username in user_data:
        if user_data[username] == password:
            return 1
        else:
            return -1
    else:
        return 0

def get_user_id_by_username(username):
    df = pd.read_csv(user_credentials)
    
    if 'username' not in df.columns or 'user_id' not in df.columns:
        print("Required columns not found in the CSV file.")
        return None
    
    # Find the row where the username matches
    user_row = df[df['username'] == username]
    
    # If the username is found, return the user_id
    if not user_row.empty:
        return user_row['user_id'].iloc[0]
    else:
        return None
    

if __name__ == '__main__':
    app.run(debug=True)



