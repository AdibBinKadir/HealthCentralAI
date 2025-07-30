from app import create_app
import os

# Create the application instance using the factory function
app = create_app()

if __name__ == '__main__':
    # Use PORT environment variable that GCP provides
    port = int(os.environ.get('PORT', 8080))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    
    print(f"Starting HealthCentral AI on {host}:{port}")
    print(f"Debug mode: {debug}")
    print(f"Environment: {os.environ.get('FLASK_ENV', 'production')}")
    
    app.run(host=host, port=port, debug=debug)