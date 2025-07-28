from app import create_app

# Create the application instance using the factory function
app = create_app()

if __name__ == '__main__':
    # The debug=True argument is great for development.
    # For a real deployment, you would use a production-ready web server
    # like Gunicorn or uWSGI instead of Flask's built-in server.
    app.run(debug=True)