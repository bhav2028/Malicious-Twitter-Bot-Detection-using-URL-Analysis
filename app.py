from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib
from urllib.parse import urlparse

from werkzeug.security import generate_password_hash

app = Flask(__name__, template_folder='Templates', static_url_path='/static')
app.secret_key = 'your_secret_key'

df = pd.read_csv('malicious_bot_twitter_data.csv')

bot_model = joblib.load('bot_detection_model_3.pkl')


def is_malicious_url(url):
    # Extract features from the URL
    parsed_url = urlparse(url)
    print(parsed_url)

    # Method 2: Check if the domain is a numerical IP address
    if parsed_url.netloc.replace('.', '').isnumeric():
        return "URL is malicious. Not safe to use"

    # Method 3: Check if the URL path is shorter than the full URL
    # if len(parsed_url.path) < len(url):
    #     return "URL is malicious. Not safe to use"

    # Method 4: Check for known malicious patterns in the URL path
    malicious_path_keywords = ['phishing', 'malicious', 'scam', 'bot']
    if any(keyword in parsed_url.path.lower() for keyword in malicious_path_keywords):
        return "URL is malicious. Not safe to use"

    # If none of the methods detect malicious patterns, consider the URL safe
    return "URL is safe to use"


@app.route('/')
def home():
    if 'user' in session:
        return render_template('home.html')
    else:
        return redirect(url_for('login'))


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Validate login credentials (replace with your actual authentication logic)
        if request.form['email'] == 'user@example.com' and request.form['password'] == 'password123':
            session['user'] = 'user@example.com'
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error='Invalid login credentials. Please try again.')

    # Handle GET request (display the login form)
    return render_template('login.html', error=None)


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    if request.method == 'POST':
        # Remove the 'user' key from the session
        session.pop('user', None)
        # Redirect to the thank-you page
        return redirect(url_for('thankyou'))

    # Handle GET request if needed (e.g., redirect to the home page)
    return redirect(url_for('home'))


@app.route('/thankyou')
def thankyou():
    return render_template('thankyou.html')


users = []


def is_username_taken(username):
    return any(user['username'] == username for user in users)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get user input from the registration form
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirmPassword']

        # Check if username is taken
        if is_username_taken(username):
            return render_template('register.html', error='Username is already taken. Please choose another.')

        # Check if passwords match
        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match. Please check and try again.')

        # Hash the password for security
        hashed_password = generate_password_hash(password, method='sha256')

        # Save user data (you may want to store this in a database)
        user_data = {'username': username, 'email': email, 'password': hashed_password}
        users.append(user_data)

        # Redirect to login page after successful registration
        return redirect(url_for('login'))

    # Handle GET request (display the registration form)
    return render_template('register.html', error=None)


@app.route('/analyze', methods=['POST'])
def analyze():
    tweet_link = request.form['tweetLinkInput']
    post_url = request.form['tweetPostUrl']

    # Bot detection for the account
    account_info = df[df['Tweet Link'] == tweet_link]
    bot_predict_account = \
        bot_model.predict(
            account_info[['Retweet Count', 'Mention Count', 'Follower Count', 'Sentiment']])[
            0]

    print('Analysis result of bot: ', bot_predict_account)

    malicious_url_post = is_malicious_url(post_url)

    return render_template('results.html',
                           tweet_link=tweet_link,
                           post_url=post_url,
                           bot_predict_account=bot_predict_account,
                           malicious_url_post=malicious_url_post)


if __name__ == '__main__':
    app.run(debug=True)
